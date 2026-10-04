import ast
import json
import os
import random
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

# Words (keeping contractions like don't / i'm) OR single punctuation marks
TOKEN_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]")

# ---- Data-quality knobs ----
MAX_SAME_ANSWER = 10     # max copies of an identical answer
MAX_SAME_PREFIX = 400    # max answers sharing the same first 3 words ("i don't know ...")
DOLLY_MAX_ANSWER = 40    # Dolly answers longer than this (in tokens) are skipped
DOLLY_MAX_QUESTION = 40


def tokenize(text):
    text = re.sub(r"<[^>]+>", " ", str(text).lower())  # strip <u>, <i> tags etc.
    return TOKEN_RE.findall(text)


def parse_dialog(s):
    """DailyDialog CSVs sometimes store lists as "['hi' 'how are you']" (no commas),
    which ast.literal_eval can't parse. Try it first, then fall back to a regex."""
    try:
        d = ast.literal_eval(s)
        if isinstance(d, (list, tuple)):
            return [str(x) for x in d]
    except Exception:
        pass
    found = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", s)
    return [a or b for a, b in found]


QUESTION_STARTERS = {"what", "what's", "how", "how's", "who", "why", "where", "when", "which",
                     "do", "does", "did", "are", "is", "can", "could", "will", "would",
                     "have", "should", "tell", "huh", "really"}


def load_personality(path):
    """personality.json = [{"q": [question phrasings], "a": [answers]}, ...]
    Every phrasing is paired with every answer, and each phrasing also gets
    punctuation variants ("?" / "." / "!") because the tokenizer treats
    punctuation as separate tokens and real users type it."""
    with open(path, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    pairs = []
    for gi, g in enumerate(groups):
        for q_text in g["q"]:
            first = q_text.split()[0].lower() if q_text.split() else ""
            suffixes = ["", "?"] if first in QUESTION_STARTERS else ["", ".", "!"]
            for suf in suffixes:
                q = tokenize(q_text + suf)
                for a_text in g["a"]:
                    a = tokenize(a_text)
                    if q and a:
                        pairs.append((q, a, f"pers{gi}"))
    return pairs


class SegawaDataset(Dataset):
    def __init__(self, data_dir, max_length=60, vocab_size=10000, answer_only_loss=True,
                 load_all=True, vocab_only=False):
        """vocab_only=True: if vocab.json already exists, skip loading the corpora entirely
        (use this in the server, which only needs the vocabulary + helpers)."""
        self.max_length = max_length
        self.answer_only_loss = answer_only_loss

        vocab_path = os.path.join(data_dir, "vocab.json")
        raw_pairs = []  # (q_tokens, a_tokens, conversation_id)

        if vocab_only and os.path.exists(vocab_path):
            self.encoded, self.conv_ids, self.is_pers = [], [], []
            self._load_vocab(vocab_path)
            print("Dataset in vocab-only mode (no training pairs loaded).")
            return

        def report(name, n_before):
            print(f"  -> {name}: {len(raw_pairs) - n_before} pairs")

        # ---------- 1. Cornell Movie Dialogs ----------
        lines_path = os.path.join(data_dir, "movie_lines.txt")
        conv_path = os.path.join(data_dir, "movie_conversations.txt")
        if os.path.exists(lines_path) and os.path.exists(conv_path):
            print("Loading Cornell Movie Dialogs...")
            n0 = len(raw_pairs)
            sep = r' \+\+\+\$\+\+\+ '
            lines_df = pd.read_csv(lines_path, sep=sep, engine='python', encoding='latin1',
                                   on_bad_lines='skip',
                                   names=['lineID', 'characterID', 'movieID', 'character', 'text'])
            conv_df = pd.read_csv(conv_path, sep=sep, engine='python', encoding='latin1',
                                  on_bad_lines='skip',
                                  names=['char1ID', 'char2ID', 'movieID', 'lineIDs'])
            line_dict = dict(zip(lines_df['lineID'], lines_df['text'].fillna('')))
            for conv_idx, line_ids_str in enumerate(conv_df['lineIDs']):
                try:
                    line_ids = ast.literal_eval(line_ids_str)
                except Exception:
                    continue
                for i in range(len(line_ids) - 1):
                    q = tokenize(line_dict.get(line_ids[i], ""))
                    a = tokenize(line_dict.get(line_ids[i + 1], ""))
                    if q and a:
                        raw_pairs.append((q, a, f"c{conv_idx}"))
            report("Cornell", n0)

        # ---------- 2. DailyDialogue ----------
        dd_dir = os.path.join(data_dir, "DailyDialogue")
        if load_all and os.path.exists(dd_dir):
            print("Loading DailyDialogue...")
            n0 = len(raw_pairs)
            for split in ["train.csv", "validation.csv", "test.csv"]:
                path = os.path.join(dd_dir, split)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    for idx, row in df.iterrows():
                        dialog = parse_dialog(str(row['dialog']))
                        for i in range(len(dialog) - 1):
                            q, a = tokenize(dialog[i]), tokenize(dialog[i + 1])
                            if q and a:
                                raw_pairs.append((q, a, f"dd_{split}_{idx}"))
            report("DailyDialogue", n0)

        # ---------- 3. Dolly (only short Q/A pairs - long ones don't fit anyway) ----------
        dolly_path = os.path.join(data_dir, "Dolly", "train.csv")
        if load_all and os.path.exists(dolly_path):
            print("Loading Dolly...")
            n0 = len(raw_pairs)
            df = pd.read_csv(dolly_path)
            qs = (df['instruction'].fillna('') + " " + df['context'].fillna('')).tolist()
            ans = df['response'].fillna('').tolist()
            for i, (q_text, a_text) in enumerate(zip(qs, ans)):
                if not isinstance(q_text, str) or not isinstance(a_text, str):
                    continue
                q, a = tokenize(q_text), tokenize(a_text)
                if q and a and len(a) <= DOLLY_MAX_ANSWER and len(q) <= DOLLY_MAX_QUESTION:
                    raw_pairs.append((q, a, f"dolly{i}"))
            report("Dolly", n0)

        # ---------- 4. PersonaChat ----------
        persona_path = os.path.join(data_dir, "PersonaChat", "personachat_self_original.json")
        if load_all and os.path.exists(persona_path):
            print("Loading PersonaChat (this takes a moment)...")
            n0 = len(raw_pairs)
            with open(persona_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            for split in ['train', 'valid']:
                for idx, item in enumerate(d.get(split, [])):
                    if 'utterances' in item and item['utterances']:
                        # Original format: last utterance holds the full history;
                        # the last candidate is the gold reply.
                        last = item['utterances'][-1]
                        dialog = list(last.get('history', []))
                        if last.get('candidates'):
                            dialog.append(last['candidates'][-1])
                    else:
                        dialog = item.get('history', [])
                    for i in range(len(dialog) - 1):
                        q, a = tokenize(dialog[i]), tokenize(dialog[i + 1])
                        if q and a:
                            raw_pairs.append((q, a, f"pc_{split}_{idx}"))
            report("PersonaChat", n0)

        # ---------- Generic-answer cap (fights "i don't know" collapse) ----------
        exact, prefix, filtered = Counter(), Counter(), []
        random.Random(0).shuffle(raw_pairs)
        for q, a, cid in raw_pairs:
            k1, k2 = tuple(a), tuple(a[:3])
            if exact[k1] >= MAX_SAME_ANSWER or prefix[k2] >= MAX_SAME_PREFIX:
                continue
            exact[k1] += 1
            prefix[k2] += 1
            filtered.append((q, a, cid))
        print(f"Generic-answer filter: {len(raw_pairs)} -> {len(filtered)} pairs")
        raw_pairs = filtered

        # ---------- 5. Personality (trained LAST; exempt from the generic-answer cap) ----------
        pers_pairs = []
        pers_path = os.path.join(data_dir, "Personality", "personality.json")
        if os.path.exists(pers_path):
            print("Loading Personality...")
            pers_pairs = load_personality(pers_path)
            print(f"  -> Personality: {len(pers_pairs)} pairs")
        else:
            print(f"  (no personality file found at {pers_path})")

        # ---------- Vocabulary ----------
        if os.path.exists(vocab_path):
            self._load_vocab(vocab_path)
        else:
            print("Building vocabulary...")
            counter = Counter()
            for q, a, _ in raw_pairs + pers_pairs:
                counter.update(q)
                counter.update(a)

            self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
            # Sort by count (descending), then alphabetically for a deterministic order
            sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            self.vocab += [w for w, _ in sorted_items[:vocab_size - 4]]
            try:
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.vocab, f)
            except Exception as e:
                print(f"Warning: could not save vocab.json: {e}")
            self._build_maps()

        # ---------- Pre-encode, truncate safely ----------
        budget = max_length - 2
        max_q = budget // 2
        self.encoded, self.conv_ids, self.is_pers = [], [], []
        dropped = 0
        for flag, pairs in ((False, raw_pairs), (True, pers_pairs)):
            for q, a, cid in pairs:
                q_ids = self.encode_tokens(q)[-max_q:]
                a_ids = self.encode_tokens(a)
                if len(a_ids) > budget - len(q_ids):
                    dropped += 1
                    continue
                self.encoded.append((q_ids, a_ids))
                self.conv_ids.append(cid)
                self.is_pers.append(flag)

        print(f"Dataset ready! Pairs kept: {len(self.encoded)} | dropped (answer too long): {dropped}")
        print(f"UNK rate: {self._unk_rate():.2f}% of tokens")

    # ---------- vocab helpers ----------
    def _load_vocab(self, vocab_path):
        print("Loading existing vocabulary...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self._build_maps()

    def _build_maps(self):
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.PAD_IDX = self.word2idx['<PAD>']
        self.UNK_IDX = self.word2idx.get('<UNK>', 1)
        self.BOS_IDX = self.word2idx.get('<BOS>', 2)
        self.EOS_IDX = self.word2idx.get('<EOS>', 3)

    # ---------- helpers (reused by the server) ----------
    def encode_tokens(self, tokens):
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def encode_prompt(self, text):
        """Build the model prompt for generation: BOS + question + EOS."""
        budget = self.max_length - 2
        q = self.encode_tokens(tokenize(text))[-(budget // 2):]
        return [self.BOS_IDX] + q + [self.EOS_IDX]

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (self.PAD_IDX, self.BOS_IDX):
                continue
            if i == self.EOS_IDX:
                break
            out.append(self.idx2word.get(int(i), '<UNK>'))
        text = " ".join(out)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\s+('\w+)", r"\1", text)  # stray " 's" -> "'s"
        return text

    def _unk_rate(self):
        total = unk = 0
        for q, a in self.encoded:
            for x in q + a:
                total += 1
                unk += (x == self.UNK_IDX)
        return 100.0 * unk / max(1, total)

    def split_by_conversation(self, val_frac=0.05, seed=42):
        """Split so no conversation appears in both train and val."""
        ids = sorted({c for c, p in zip(self.conv_ids, self.is_pers) if not p})
        random.Random(seed).shuffle(ids)
        val_ids = set(ids[:max(1, int(len(ids) * val_frac))])
        train_idx = [i for i, (c, p) in enumerate(zip(self.conv_ids, self.is_pers))
                     if not p and c not in val_ids]
        val_idx = [i for i, (c, p) in enumerate(zip(self.conv_ids, self.is_pers))
                   if not p and c in val_ids]
        return train_idx, val_idx

    def personality_indices(self):
        """Indices of personality pairs (kept out of the main train/val split)."""
        return [i for i, p in enumerate(self.is_pers) if p]

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        q, a = self.encoded[idx]

        seq = [self.BOS_IDX] + q + [self.EOS_IDX] + a + [self.EOS_IDX]
        seq += [self.PAD_IDX] * (self.max_length + 1 - len(seq))
        inputs = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(seq[1:], dtype=torch.long)

        if self.answer_only_loss:
            targets[:len(q) + 1] = self.PAD_IDX

        return inputs, targets
