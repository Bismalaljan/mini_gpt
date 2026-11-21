"""
Next-word predictor: runs my own decoder-only Transformer on Vercel.

The model was trained in PyTorch (see the notebook) and exported to model/model.npz.
Here the same forward pass is written in plain NumPy, so the app is small and fast
(PyTorch is too large for a Vercel function).
"""
import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"

# ------------------------------------------------------------------ load model
MODEL_READY = (MODEL_DIR / "model.npz").exists() and (MODEL_DIR / "vocab.json").exists()

if MODEL_READY:
    meta = json.loads((MODEL_DIR / "vocab.json").read_text(encoding="utf-8"))
    itos = meta["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    cfg = meta["config"]
    ASCII_ONLY = meta.get("ascii_only", True)
    BLOCK, EMB, N_HEAD, N_LAYER = cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"]
    UNK = 0

    raw = np.load(MODEL_DIR / "model.npz")
    W = {k: raw[k].astype(np.float32) for k in raw.files}     # float16 on disk -> float32 for maths

    # join the per-head Q, K, V weights of each layer into single matrices
    LAYERS = []
    for i in range(N_LAYER):
        p = f"blocks.{i}."
        cat = lambda name: np.concatenate(
            [W[f"{p}sa.heads.{h}.{name}.weight"] for h in range(N_HEAD)], axis=0)
        LAYERS.append(dict(
            ln1_w=W[p + "ln1.weight"], ln1_b=W[p + "ln1.bias"],
            wq=cat("query"), wk=cat("key"), wv=cat("value"),
            proj_w=W[p + "sa.proj.weight"], proj_b=W[p + "sa.proj.bias"],
            ln2_w=W[p + "ln2.weight"], ln2_b=W[p + "ln2.bias"],
            ff1_w=W[p + "ffwd.net.0.weight"], ff1_b=W[p + "ffwd.net.0.bias"],
            ff2_w=W[p + "ffwd.net.2.weight"], ff2_b=W[p + "ffwd.net.2.bias"],
        ))
    TOK, POS = W["tok_emb.weight"], W["pos_emb.weight"]
    LNF_W, LNF_B = W["ln_f.weight"], W["ln_f.bias"]


# ------------------------------------------------------------------ tokenizer (same as the notebook)
def tokenize(text):
    if ASCII_ONLY:
        text = text.encode("ascii", "ignore").decode()
    text = text.lower()
    text = text.replace("<unk>", " ").replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("= ")]
    return re.findall(r"[a-z0-9]+|'[a-z]+|[^\sa-z0-9]", " ".join(lines))


def detokenize(tokens):
    s = " ".join(tokens)
    s = re.sub(r" ([.,;:!?)%\]])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    s = re.sub(r" ('s|'t|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    return s


# ------------------------------------------------------------------ the transformer in NumPy
def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def erf(x):  # Abramowitz-Stegun approximation, max error 1.5e-7
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-x * x))


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def forward_logits(ids):
    """ids: list of token ids (at most BLOCK). Returns logits for the NEXT token."""
    ids = ids[-BLOCK:]
    T = len(ids)
    hs = EMB // N_HEAD
    x = TOK[ids] + POS[:T]                                         # token + position embeddings
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)   # CAUSAL MASK
    for L in LAYERS:
        h = layer_norm(x, L["ln1_w"], L["ln1_b"])
        q = (h @ L["wq"].T).reshape(T, N_HEAD, hs).transpose(1, 0, 2)   # Q: what I look for
        k = (h @ L["wk"].T).reshape(T, N_HEAD, hs).transpose(1, 0, 2)   # K: what I offer
        v = (h @ L["wv"].T).reshape(T, N_HEAD, hs).transpose(1, 0, 2)   # V: information I pass on
        att = softmax(q @ k.transpose(0, 2, 1) * (hs ** -0.5) + mask)   # softmax(QK^T / sqrt(d))
        out = (att @ v).transpose(1, 0, 2).reshape(T, EMB)              # ... times V, heads joined
        x = x + out @ L["proj_w"].T + L["proj_b"]                       # residual connection
        h = layer_norm(x, L["ln2_w"], L["ln2_b"])
        x = x + gelu(h @ L["ff1_w"].T + L["ff1_b"]) @ L["ff2_w"].T + L["ff2_b"]   # feed-forward + residual
    x = layer_norm(x[-1], LNF_W, LNF_B)
    return x @ TOK.T                                                # tied output layer


def generate(ids, n_words, temperature=0.8, top_k=40, repetition_penalty=1.15, seed=None):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    for _ in range(n_words):
        logits = forward_logits(ids) / max(temperature, 1e-5)
        logits[UNK] = -np.inf
        seen = np.unique(ids[-BLOCK:])
        l = logits[seen]
        logits[seen] = np.where(l > 0, l / repetition_penalty, l * repetition_penalty)
        if top_k:
            kth = np.sort(logits)[-min(top_k, len(logits))]
            logits[logits < kth] = -np.inf
        probs = softmax(logits)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return ids


# ------------------------------------------------------------------ web app
app = FastAPI(title="Next-word predictor")


class Query(BaseModel):
    prompt: str
    n_words: int = 30
    temperature: float = 0.8


@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "public" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": MODEL_READY}


@app.post("/api/predict")
def predict(q: Query):
    if not MODEL_READY:
        return JSONResponse({"error": "Model files are missing. Add model/model.npz and model/vocab.json."}, status_code=503)
    prompt = q.prompt[:500]
    toks = tokenize(prompt)
    if not toks:
        return JSONResponse({"error": "Please type some English words."}, status_code=400)
    ids = [stoi.get(w, UNK) for w in toks]
    unknown = sorted({w for w in toks if w not in stoi})

    logits = forward_logits(ids)
    logits[UNK] = -np.inf
    probs = softmax(logits)
    top = np.argsort(-probs)[:5]

    n_words = max(0, min(int(q.n_words), 60))
    temperature = max(0.1, min(float(q.temperature), 2.0))
    out_ids = generate(ids, n_words, temperature=temperature) if n_words else ids
    return {
        "words_seen": toks[-BLOCK:],
        "unknown": unknown,
        "top": [{"word": itos[int(i)], "prob": float(probs[i])} for i in top],
        "continuation": detokenize([itos[i] for i in out_ids]),
    }
