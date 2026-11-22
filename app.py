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

MODEL_ERROR = None
ASCII_ONLY, BLOCK, UNK, itos, stoi = True, 32, 0, [], {}
try:
    if not MODEL_READY:
        raise FileNotFoundError("model/model.npz or model/vocab.json not found")
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
except Exception as e:
    MODEL_READY = False
    MODEL_ERROR = f"{type(e).__name__}: {e}"


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

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Next-Word Predictor</title>
<style>
  :root { --bg:#f6f7fb; --card:#ffffff; --ink:#1b1f2a; --muted:#6b7280; --accent:#4f46e5; --bar:#c7d2fe; --line:#e5e7eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1115; --card:#181b22; --ink:#e8eaf0; --muted:#9aa3b2; --accent:#818cf8; --bar:#3730a3; --line:#2a2f3a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  main { max-width:720px; margin:0 auto; padding:24px 16px 48px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; margin-bottom:16px; }
  label { display:block; font-weight:600; margin-bottom:6px; }
  input[type=text] { width:100%; padding:12px; font-size:1rem; border:1px solid var(--line); border-radius:10px; background:transparent; color:var(--ink); }
  .row { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; align-items:end; }
  .row > div { flex:1 1 140px; }
  input[type=range] { width:100%; }
  button { background:var(--accent); color:#fff; border:0; border-radius:10px; padding:12px 18px; font-size:1rem; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  .chip { background:transparent; color:var(--accent); border:1px solid var(--line); border-radius:999px; padding:6px 12px; font-size:.9rem; font-weight:500; }
  .bar { display:flex; align-items:center; gap:10px; margin:8px 0; }
  .bar .w { width:110px; font-weight:600; overflow:hidden; text-overflow:ellipsis; }
  .bar .t { flex:1; background:var(--line); border-radius:6px; height:14px; overflow:hidden; }
  .bar .f { background:var(--bar); height:100%; }
  .bar .p { width:56px; text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
  .out { white-space:pre-wrap; }
  .muted { color:var(--muted); font-size:.9rem; }
  .err { color:#dc2626; }
</style>
</head>
<body>
<main>
  <h1>Next-Word Predictor</h1>
  <p class="sub">A decoder-only Transformer built from scratch. Type a word or a sentence, and it predicts the next word.</p>

  <div class="card">
    <label for="prompt">Your text</label>
    <input id="prompt" type="text" value="ai is a" autocomplete="off" placeholder="e.g. machine learning is">
    <div class="chips" id="chips"></div>
    <div class="row">
      <div>
        <label for="temp">Creativity: <span id="tv">0.8</span></label>
        <input id="temp" type="range" min="0.2" max="1.5" step="0.1" value="0.8">
      </div>
      <div>
        <label for="n">Words to continue: <span id="nv">30</span></label>
        <input id="n" type="range" min="0" max="60" step="5" value="30">
      </div>
      <div><button id="go">Predict</button></div>
    </div>
  </div>

  <div class="card" id="result" hidden>
    <strong>Top 5 predictions for the next word</strong>
    <div id="bars"></div>
    <p class="muted" id="note"></p>
    <strong>Continuation</strong>
    <p class="out" id="cont"></p>
  </div>

  <p class="muted">Tip: stop your sentence where the next word is natural, for example "islamabad is the capital of". The model only knows what it was trained on, so unfamiliar words show as unknown.</p>
</main>

<script>
const $ = id => document.getElementById(id);
const examples = ["ai is a", "machine learning is", "a transformer is a", "gpt uses only the", "python is a", "islamabad is the capital of"];
examples.forEach(t => {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = t;
  b.onclick = () => { $("prompt").value = t; run(); };
  $("chips").appendChild(b);
});
$("temp").oninput = e => $("tv").textContent = e.target.value;
$("n").oninput = e => $("nv").textContent = e.target.value;
$("go").onclick = run;
$("prompt").addEventListener("keydown", e => { if (e.key === "Enter") run(); });

async function run() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  $("go").disabled = true; $("go").textContent = "Thinking...";
  try {
    const r = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt, n_words: +$("n").value, temperature: +$("temp").value})
    });
    const d = await r.json();
    $("result").hidden = false;
    $("bars").textContent = "";
    if (!r.ok) {
      $("note").textContent = d.error || "Something went wrong.";
      $("note").className = "err"; $("cont").textContent = "";
      return;
    }
    $("note").className = "muted";
    const max = Math.max(...d.top.map(x => x.prob), 0.0001);
    d.top.forEach(x => {
      const row = document.createElement("div"); row.className = "bar";
      const w = document.createElement("span"); w.className = "w"; w.textContent = x.word;
      const t = document.createElement("span"); t.className = "t";
      const f = document.createElement("span"); f.className = "f"; f.style.display = "block"; f.style.width = (x.prob / max * 100) + "%";
      t.appendChild(f);
      const p = document.createElement("span"); p.className = "p"; p.textContent = (x.prob * 100).toFixed(1) + "%";
      row.append(w, t, p); $("bars").appendChild(row);
    });
    $("note").textContent = d.unknown.length ? "Unknown to the model: " + d.unknown.join(", ") : "";
    $("cont").textContent = d.continuation;
  } catch (e) {
    $("result").hidden = false;
    $("note").textContent = "Could not reach the server. Please try again."; $("note").className = "err";
  } finally {
    $("go").disabled = false; $("go").textContent = "Predict";
  }
}
run();
</script>
</body>
</html>
"""


class Query(BaseModel):
    prompt: str
    n_words: int = 30
    temperature: float = 0.8


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML          # the page is stored inside this file, so no extra file is needed


@app.get("/api/health")
def health():
    files = sorted(p.name for p in MODEL_DIR.iterdir()) if MODEL_DIR.exists() else "model folder not found"
    return {"ok": True, "model_loaded": MODEL_READY, "model_error": MODEL_ERROR, "files_in_model_folder": files}


@app.post("/api/predict")
def predict(q: Query):
    try:
        return _predict(q)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


def _predict(q: Query):
    if not MODEL_READY:
        return JSONResponse({"error": "Model not loaded. " + (MODEL_ERROR or "Add model/model.npz and model/vocab.json.")}, status_code=503)
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
