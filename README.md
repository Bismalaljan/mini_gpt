# Next-Word Predictor (my own Transformer)

A decoder-only Transformer built from scratch in PyTorch (token + position embeddings,
masked multi-head self-attention with Q/K/V, feed-forward layers, residual connections, LayerNorm).
It predicts the next word. This web app runs the trained model on Vercel.

- `app.py`: FastAPI app. The Transformer forward pass is written in NumPy (PyTorch is too big for Vercel).
- `public/index.html`: the web page.
- `model/`: `model.npz` and `vocab.json`, exported from the training notebook.
- Training notebook: `mini_gpt_words_final.ipynb`

## Run locally
```
pip install fastapi uvicorn numpy
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Deploy
Push this folder to GitHub, then import the repository on vercel.com.

## API
`POST /api/predict` with `{"prompt": "ai is a", "n_words": 30, "temperature": 0.8}`
returns the top-5 next words and a continuation.
