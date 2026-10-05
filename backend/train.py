import json
import math
import os
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import SegawaDataset
from model import SegawaModel, device

# =====================================================================
#  STAGE 1: train on Cornell / DailyDialog / Dolly / PersonaChat
#  STAGE 2: finish with the Personality dataset (low LR + replay of
#           general data so the model doesn't forget how to chat)
#
#  MODE
#   "memorize"   : big model, no dropout, stage 1 runs until TRAIN perplexity
#                  <= TARGET_PPL and TRAIN accuracy >= TARGET_ACC.
#   "generalize" : regularized model, early stopping on VALIDATION loss.
# =====================================================================
MODE = "memorize"

TARGET_PPL = 10.0
TARGET_ACC = 80.0

PRESETS = {
    "memorize": dict(embed_size=512, num_layers=8, heads=8, dropout=0.0,
                     epochs=150, patience=None, lr=5e-4, weight_decay=0.01),
    "generalize": dict(embed_size=384, num_layers=6, heads=6, dropout=0.2,
                       epochs=40, patience=6, lr=5e-4, weight_decay=0.05),
}
CFG = PRESETS[MODE]

# ---------------- Stage 2 (personality) ----------------
PERSONALITY_EPOCHS = 50
PERSONALITY_LR = 1.5e-4
PERSONALITY_REPLAY = 3     # general samples mixed in per personality sample

# ---------------- Paths (same ones the server uses) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = None  # set below by find_data_dir()
# On Colab, point this at Google Drive so checkpoints survive disconnects:
#   os.environ["SEGAWA_CKPT_DIR"] = "/content/drive/MyDrive/segawa_ckpt"
CKPT_DIR = os.environ.get("SEGAWA_CKPT_DIR") or os.path.join(BASE_DIR, "checkpoints")
RESUME_PATH = os.path.join(CKPT_DIR, "segawa_resume.pt")
RESUME = True
SAVE_EVERY_STEPS = 1500   # mid-epoch resume save (lose at most this many steps)

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 64
WARMUP_EPOCHS = 2
MIN_LR_FRAC = 0.05
GRAD_CLIP = 1.0
MAX_LENGTH = 96           # must match server.py
VOCAB_SIZE = 15000        # must match server.py
TRAIN_EVAL_SAMPLES = 20000
SEED = 42


def find_data_dir():
    """Datasets folder: $SEGAWA_DATA_DIR, else ./datasets next to this file, else ../datasets."""
    env = os.environ.get("SEGAWA_DATA_DIR")
    if env:
        return env
    local = os.path.join(BASE_DIR, "datasets")
    if os.path.isdir(local):
        return local
    return os.path.abspath(os.path.join(BASE_DIR, "..", "datasets"))


def atomic_save(obj, path):
    """Write to a temp file, then rename. A disconnect mid-write can't corrupt the old file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


DATA_DIR = find_data_dir()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": CFG["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=CFG["lr"], betas=(0.9, 0.98), eps=1e-9)


def build_scheduler(optimizer, total_steps, warmup_steps, min_frac):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_frac + (1 - min_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, pad_idx, use_amp, dev_type, desc="Eval"):
    """Token-weighted loss / accuracy / perplexity (answer tokens only)."""
    model.eval()
    sum_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss, total_tokens, correct = 0.0, 0, 0

    for inputs, targets in tqdm(loader, leave=False, desc=desc):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
        outputs = outputs.float().reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)

        mask = targets != pad_idx
        total_loss += sum_loss(outputs, targets).item()
        total_tokens += mask.sum().item()
        correct += ((outputs.argmax(dim=1) == targets) & mask).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    acc = 100.0 * correct / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, acc, ppl


def run_epoch(model, loader, optimizer, scheduler, scaler, criterion, use_amp, dev_type, desc,
              step_callback=None):
    model.train()
    total = 0.0
    loop = tqdm(loader, leave=False)
    for step, (inputs, targets) in enumerate(loop, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
        outputs = outputs.float().reshape(-1, outputs.shape[-1])
        loss = criterion(outputs, targets.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()
        loop.set_description(desc)
        loop.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        if step_callback is not None:
            step_callback(step)
    return total / max(1, len(loader))


def personality_stage(model, dataset, train_idx, val_loader, use_amp, dev_type):
    """Final stage: fine-tune on personality data mixed with replayed general data."""
    pers_idx = dataset.personality_indices()
    if not pers_idx:
        print("\nNo personality data found (datasets/Personality/personality.json) - skipping stage 2.")
        return

    print(f"\n=== STAGE 2: Personality fine-tuning ({len(pers_idx)} personality pairs, "
          f"{PERSONALITY_REPLAY}x general replay) ===")
    rng = random.Random(SEED + 1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PERSONALITY_LR,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    n_replay = min(len(train_idx), PERSONALITY_REPLAY * len(pers_idx))
    steps_per_epoch = math.ceil((len(pers_idx) + n_replay) / BATCH_SIZE)
    total_steps = PERSONALITY_EPOCHS * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, max(1, steps_per_epoch * 2), 0.1)

    pers_eval_loader = DataLoader(Subset(dataset, pers_idx), batch_size=BATCH_SIZE * 2, shuffle=False)

    for epoch in range(PERSONALITY_EPOCHS):
        replay = rng.sample(train_idx, n_replay)  # fresh general samples every epoch
        loader = DataLoader(Subset(dataset, pers_idx + replay), batch_size=BATCH_SIZE, shuffle=True)
        run_epoch(model, loader, optimizer, scheduler, scaler, criterion, use_amp, dev_type,
                  f"Personality [{epoch + 1}/{PERSONALITY_EPOCHS}]")

        if (epoch + 1) % 5 == 0 or epoch + 1 == PERSONALITY_EPOCHS:
            _, p_acc, p_ppl = evaluate(model, pers_eval_loader, dataset.PAD_IDX, use_amp, dev_type, "PersEval")
            _, v_acc, v_ppl = evaluate(model, val_loader, dataset.PAD_IDX, use_amp, dev_type, "Val")
            print(f"  Epoch {epoch + 1}: Personality Acc {p_acc:.1f}% PPL {p_ppl:.2f} | "
                  f"General VAL Acc {v_acc:.2f}% PPL {v_ppl:.2f}")

    atomic_save(model.state_dict(), os.path.join(CKPT_DIR, "segawa_final.pth"))
    print("Saved checkpoints/segawa_final.pth  <- the server uses this one")


def train():
    print(f"Initializing Segawa Training Pipeline  [MODE = {MODE}]")
    set_seed(SEED)

    # 1. Data
    print(f"Data dir: {DATA_DIR}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    vocab_backup = os.path.join(CKPT_DIR, "vocab.json")
    vocab_live = os.path.join(DATA_DIR, "vocab.json")
    if os.path.exists(vocab_backup) and not os.path.exists(vocab_live):
        shutil.copy(vocab_backup, vocab_live)
        print("Restored vocab.json from checkpoint folder.")
    dataset = SegawaDataset(DATA_DIR, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE, load_all=True)

    if len(dataset) == 0:
        found = os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else "FOLDER DOES NOT EXIST"
        raise SystemExit(
            f"\nNo training data found in: {DATA_DIR}\n"
            f"Contents: {found}\n"
            "Fix: put your datasets folder there (movie_lines.txt, movie_conversations.txt, "
            "DailyDialogue/, Dolly/, PersonaChat/, Personality/personality.json), "
            "or set os.environ['SEGAWA_DATA_DIR'] to the right folder before running."
        )

    if os.path.exists(vocab_live):
        shutil.copy(vocab_live, vocab_backup)  # keep vocab next to the weights

    train_idx, val_idx = dataset.split_by_conversation(val_frac=0.05, seed=SEED)
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
    train_size, val_size = len(train_ds), len(val_ds)

    rng = random.Random(SEED)
    sub_idx = rng.sample(train_idx, min(TRAIN_EVAL_SAMPLES, len(train_idx)))
    train_eval_loader = DataLoader(Subset(dataset, sub_idx), batch_size=BATCH_SIZE * 2, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    # 2. Model
    arch = dict(embed_size=CFG["embed_size"], num_layers=CFG["num_layers"],
                heads=CFG["heads"], dropout=CFG["dropout"], max_length=MAX_LENGTH)
    print(f"Loading Model to {device}...")
    model = SegawaModel(vocab_size=VOCAB_SIZE, **arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(os.path.join(CKPT_DIR, "model_config.json"), "w") as f:
        json.dump(arch, f)

    # 3. Optimizer / scheduler / loss
    dev_type = torch.device(device).type
    use_amp = dev_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = build_optimizer(model)
    steps_per_epoch = len(train_loader)
    epochs = CFG["epochs"]
    scheduler = build_scheduler(optimizer, epochs * steps_per_epoch,
                                max(1, WARMUP_EPOCHS * steps_per_epoch), MIN_LR_FRAC)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    best_val, bad_epochs, start_epoch, stage1_done = float("inf"), 0, 0, False
    start_step = 0

    # 4. Resume
    if RESUME and os.path.exists(RESUME_PATH):
        ck = torch.load(RESUME_PATH, map_location=device)
        if ck.get("mode") == MODE and ck.get("arch") == arch:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            best_val, bad_epochs, start_epoch = ck["best_val"], ck["bad_epochs"], ck["epoch"]
            start_step = ck.get("step_in_epoch", 0)
            stage1_done = ck.get("stage1_done", False)
            print(f"Resumed from epoch {start_epoch}" + (f", step {start_step}" if start_step else "") + (" (stage 1 already finished)" if stage1_done else ""))
        else:
            print("Resume file is from a different MODE/architecture - starting fresh.")

    def save_resume(epoch, done, step=0):
        atomic_save({
            "mode": MODE, "arch": arch, "epoch": epoch, "stage1_done": done,
            "step_in_epoch": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_val": best_val, "bad_epochs": bad_epochs,
        }, RESUME_PATH)

    # ================= STAGE 1 =================
    epoch_done = start_epoch
    if not stage1_done:
        print(f"\n=== STAGE 1: general datasets (Train: {train_size}, Val: {val_size}) ===")
        print(f"Targets: PPL <= {TARGET_PPL} and accuracy >= {TARGET_ACC}% "
              f"({'on TRAIN' if MODE == 'memorize' else 'reported only'})\n")

        for epoch in range(start_epoch, epochs):
            # Seeded shuffle per epoch, so a mid-epoch resume continues exactly where it stopped
            g = torch.Generator()
            g.manual_seed(SEED + epoch)
            perm = torch.randperm(len(train_idx), generator=g).tolist()
            skip = start_step if epoch == start_epoch else 0
            order = [train_idx[i] for i in perm][skip * BATCH_SIZE:]
            epoch_loader = DataLoader(Subset(dataset, order), batch_size=BATCH_SIZE,
                                      shuffle=False, drop_last=True)

            def periodic_save(n, epoch=epoch, skip=skip):
                if n % SAVE_EVERY_STEPS == 0:
                    save_resume(epoch, False, skip + n)

            run_epoch(model, epoch_loader, optimizer, scheduler, scaler, criterion,
                      use_amp, dev_type, f"Epoch [{epoch + 1}/{epochs}] Train",
                      step_callback=periodic_save)
            epoch_done = epoch + 1

            tr_loss, tr_acc, tr_ppl = evaluate(model, train_eval_loader, dataset.PAD_IDX,
                                               use_amp, dev_type, desc="TrainEval")
            val_loss, val_acc, val_ppl = evaluate(model, val_loader, dataset.PAD_IDX,
                                                  use_amp, dev_type, desc="Val")

            print(f"Epoch {epoch + 1} finished!")
            print(f"  TRAIN (eval mode): Loss {tr_loss:.4f} | Acc {tr_acc:.2f}% | PPL {tr_ppl:.2f}")
            print(f"  VAL   (unseen)   : Loss {val_loss:.4f} | Acc {val_acc:.2f}% | PPL {val_ppl:.2f}")

            atomic_save(model.state_dict(), os.path.join(CKPT_DIR, "segawa_last.pth"))
            if val_loss < best_val - 1e-4:
                best_val, bad_epochs = val_loss, 0
                atomic_save(model.state_dict(), os.path.join(CKPT_DIR, "segawa_best.pth"))
                print("  New best-validation model saved (segawa_best.pth)")
            else:
                bad_epochs += 1

            save_resume(epoch_done, False)

            if MODE == "memorize":
                if tr_ppl <= TARGET_PPL and tr_acc >= TARGET_ACC:
                    print(f"\nTARGETS REACHED on training data at epoch {epoch + 1}: "
                          f"PPL {tr_ppl:.2f}, Acc {tr_acc:.2f}%")
                    print(f"Validation at this point: PPL {val_ppl:.2f}, Acc {val_acc:.2f}%")
                    break
            else:
                if CFG["patience"] is not None and bad_epochs >= CFG["patience"]:
                    print("\nEarly stopping: validation loss stopped improving.")
                    break
                print(f"  (no-improve counter {bad_epochs}/{CFG['patience']})")
            print()

        save_resume(epoch_done, True)  # stage 1 finished; a rerun goes straight to stage 2
        print(f"Stage 1 done. Best val loss: {best_val:.4f} (PPL {math.exp(best_val):.2f})")

    # In generalize mode, personality starts from the best-validation weights
    best_path = os.path.join(CKPT_DIR, "segawa_best.pth")
    if MODE == "generalize" and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # ================= STAGE 2 =================
    personality_stage(model, dataset, train_idx, val_loader, use_amp, dev_type)
    print("\nAll done!")


if __name__ == "__main__":
    train()
