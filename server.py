import datetime
import json
import os
import re
import secrets
import smtplib
from email.mime.text import MIMEText

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from dataset import SegawaDataset
from model import SegawaModel, device

app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

# ---------------- Configuration (must match train.py) ----------------
MAX_LENGTH = 96
VOCAB_SIZE = 15000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets"))
CKPT_NAME = "segawa_final.pth"  # written by train.py after the personality stage
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", CKPT_NAME)
CONFIG_PATH = os.path.join(BASE_DIR, "checkpoints", "model_config.json")
FRONTEND_DIR = r'C:\Users\Immanuel\Carly\frontend'

# ---------------- Generation settings (tweak these!) ----------------
TEMPERATURE = 0.8       # lower = safer/more boring, higher = more creative
TOP_K = 40
TOP_P = 0.9
REP_PENALTY = 1.3       # >1 discourages repeating words
MIN_LEN = 3             # forbid ending the answer before this many tokens
SENTENCE_PER_LINE = False  # True = put every sentence on its own line

print("Starting Segawa API Server...")

# 1. Load vocabulary only (fast - doesn't re-read the corpora if vocab.json exists)
print(f"Loading vocabulary from {DATA_DIR} ...")
dataset = SegawaDataset(DATA_DIR, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE,
                        load_all=True, vocab_only=True)
assert len(dataset.vocab) <= VOCAB_SIZE, "vocab.json is larger than VOCAB_SIZE"

# 2. Initialize Model
print("Loading Segawa Brain...")
arch = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        arch = json.load(f)
arch["max_length"] = MAX_LENGTH
model = SegawaModel(vocab_size=VOCAB_SIZE, **arch).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Successfully loaded {CHECKPOINT_PATH}")
else:
    print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with untrained brain!")

model.eval()


# ---------------- Text formatting ----------------
PROPER_NOUNS = ["Segawa", "English", "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
def format_response(text):
    """Capitalize 'I' and the first letter of every sentence."""
    text = re.sub(r"\bi\b", "I", text)  # i -> I, i'm -> I'm, i'll -> I'll
    for noun in PROPER_NOUNS:           # segawa -> Segawa, etc.
        text = re.sub(rf"\b{noun.lower()}\b", noun, text)
    text = re.sub(r"(^|[.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    text = text[:1].upper() + text[1:]
    if SENTENCE_PER_LINE:
        text = re.sub(r"([.!?])\s+", r"\1\n", text)
    return text


# ---------------- Sampling ----------------
def sample_next(logits, generated):
    logits = logits.clone().float()

    # repetition penalty
    for t in set(generated):
        logits[t] = logits[t] / REP_PENALTY if logits[t] > 0 else logits[t] * REP_PENALTY

    logits = logits / TEMPERATURE

    # never emit these
    logits[dataset.PAD_IDX] = -float("inf")
    logits[dataset.BOS_IDX] = -float("inf")
    logits[dataset.UNK_IDX] = -float("inf")

    # top-k
    if TOP_K:
        kth = torch.topk(logits, min(TOP_K, logits.size(-1))).values[-1]
        logits[logits < kth] = -float("inf")

    # top-p (nucleus)
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    sorted_p[cum - sorted_p > TOP_P] = 0
    sorted_p = sorted_p / sorted_p.sum()
    return sorted_i[torch.multinomial(sorted_p, 1)].item()


def generate_response(user_input):
    tokens = dataset.encode_prompt(user_input)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    generated_ids = []

    with torch.no_grad():
        # Never exceed MAX_LENGTH total, so BOS + question stay in the window
        for step in range(MAX_LENGTH - len(tokens)):
            logits = model(input_tensor)[0, -1, :]

            if step < MIN_LEN:  # forbid ultra-short answers
                logits[dataset.EOS_IDX] = -float("inf")

            nxt = sample_next(logits, generated_ids)
            if nxt == dataset.EOS_IDX:
                break

            generated_ids.append(nxt)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1
            )

    return format_response(dataset.decode(generated_ids))


# ---------------- Routes ----------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = generate_response(user_message)
        if not response.strip():
            response = "... (Segawa is struggling to find the words)"
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Email verification ----------------
load_dotenv(os.path.expanduser('~/.env'))

# In-memory store: email -> {"code": "123456", "expires": datetime}
verification_codes = {}


@app.route("/send_verification", methods=["POST"])
def send_verification():
    data = request.json or {}
    recipient_email = data.get("email")

    if not recipient_email:
        return jsonify({"error": "No email provided"}), 400

    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        return jsonify({"error": "Server email credentials not configured in ~/.env"}), 500

    # 6-digit code from a cryptographically secure source
    code = str(100000 + secrets.randbelow(900000))
    verification_codes[recipient_email] = {
        "code": code,
        "expires": datetime.datetime.now() + datetime.timedelta(minutes=10),
    }

    msg = MIMEText(f"Your Segawa login code is: {code}\n\nThis code will expire in 10 minutes.")
    msg['Subject'] = 'Segawa Verification Code'
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return jsonify({"message": "Verification code sent"})
    except Exception as e:
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500


@app.route("/verify_code", methods=["POST"])
def verify_code():
    data = request.json or {}
    email = data.get("email")
    code = data.get("code")

    if not email or not code:
        return jsonify({"error": "Email and code required"}), 400

    record = verification_codes.get(email)

    if not record:
        return jsonify({"error": "No code requested for this email"}), 400

    if datetime.datetime.now() > record["expires"]:
        return jsonify({"error": "Code has expired"}), 400

    if not secrets.compare_digest(record["code"], str(code)):
        return jsonify({"error": "Invalid code"}), 400

    del verification_codes[email]
    return jsonify({"message": "Verified"})


if __name__ == "__main__":
    print("Segawa Backend is running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
