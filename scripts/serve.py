"""Tiny local web demo for the neural search engine.

Loads the trained encoder + the prebuilt book embeddings once at startup, then
serves a single search page on http://localhost:8000. Each query is a single
forward pass (only the query is encoded; the 870 book chunks are pre-embedded),
so results come back instantly and it runs fine on CPU.

No web framework — just the Python standard library, so it runs in the same
environment as everything else with no extra installs.

Usage:
    python scripts/serve.py                 # serve on http://localhost:8000
    python scripts/serve.py --port 9000     # pick a different port
    python scripts/serve.py --ckpt models/encoder.pt
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.search import find_checkpoint
from scripts.encode_and_eval import load_trained_encoder
from src.inference import encode_texts
from src.trainer import pick_device

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Search Engine</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 820px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem;
    line-height: 1.5; color: #1a1a1a; background: #f4f5f7;
  }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  .sub { color: #6b7280; margin: 0 0 1.5rem; font-size: .95rem; }
  form { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input[type=text] {
    flex: 1; padding: .7rem .9rem; font-size: 1rem; border: 1px solid #cfd3d8;
    border-radius: 8px; outline: none; background: #fff; color: #1a1a1a;
  }
  input[type=text]:focus { border-color: #4f6df5; }
  button {
    padding: .7rem 1.1rem; font-size: 1rem; border: 0; border-radius: 8px;
    background: #4f6df5; color: #fff; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
  .controls { display: flex; align-items: center; gap: .5rem; margin-bottom: 1.5rem;
    font-size: .9rem; color: #6b7280; }
  .controls input { width: 4rem; padding: .3rem .4rem; border-radius: 6px;
    border: 1px solid #cfd3d8; background: #fff; color: #1a1a1a; }
  .hit { border: 1px solid #e2e5e9; background: #fff; color: #1a1a1a;
    border-radius: 10px; padding: .9rem 1rem; margin-bottom: .75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .hit .head { display: flex; justify-content: space-between; gap: 1rem;
    margin-bottom: .35rem; }
  .meta { color: #6b7280; font-size: .82rem; font-variant-numeric: tabular-nums; }
  .rank { font-weight: 600; color: #111; }
  .snippet { font-size: .95rem; color: #1f2937; }
  .status { color: #5b6370; min-height: 1.2rem; }
</style>
</head>
<body>
  <h1>Neural Search Engine</h1>
  <p class="sub">Semantic search over <em>Speech and Language Processing</em>
     (Jurafsky &amp; Martin) — a transformer bi-encoder trained from scratch.</p>

  <form id="f">
    <input type="text" id="q" placeholder="Ask a question…" autofocus autocomplete="off">
    <button type="submit" id="go">Search</button>
  </form>

  <div class="controls">
    <label>results: <input type="number" id="k" value="5" min="1" max="20"></label>
  </div>

  <div class="status" id="status"></div>
  <div id="results"></div>

<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const k = document.getElementById('k');
const go = document.getElementById('go');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');

function esc(s) {
  return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

async function search(query) {
  if (!query.trim()) return;
  go.disabled = true;
  statusEl.textContent = 'Searching…';
  results.innerHTML = '';
  try {
    const url = '/search?q=' + encodeURIComponent(query) + '&k=' + encodeURIComponent(k.value);
    const r = await fetch(url);
    const data = await r.json();
    statusEl.textContent = data.hits.length + ' results for "' + query + '"';
    results.innerHTML = data.hits.map(h => `
      <div class="hit">
        <div class="head">
          <span class="rank">#${h.rank}</span>
          <span class="meta">${esc(h.id)} · sim ${h.sim.toFixed(3)}</span>
        </div>
        <div class="snippet">${esc(h.snippet)}</div>
      </div>`).join('');
  } catch (e) {
    statusEl.textContent = 'Error: ' + e;
  } finally {
    go.disabled = false;
  }
}

f.addEventListener('submit', e => { e.preventDefault(); search(q.value); });
</script>
</body>
</html>
"""


def build_handler(model, tokenizer, device, jm_ids, jm_texts, jm_emb):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet the default per-request logging
            pass

        def _send(self, code, body, content_type):
            payload = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
                return
            if parsed.path == "/search":
                params = parse_qs(parsed.query)
                query = (params.get("q") or [""])[0]
                try:
                    k = max(1, min(20, int((params.get("k") or ["5"])[0])))
                except ValueError:
                    k = 5
                hits = self._search(query, k) if query.strip() else []
                self._send(200, json.dumps({"hits": hits}), "application/json")
                return
            self._send(404, "not found", "text/plain")

        def _search(self, query, k):
            q_emb = encode_texts(model, tokenizer, [query], device, max_len=64)
            sims = (q_emb @ jm_emb.T)[0]
            top = np.argsort(-sims)[:k]
            return [
                {
                    "rank": rank,
                    "id": str(jm_ids[i]),
                    "sim": float(sims[i]),
                    "snippet": " ".join(jm_texts[i].split())[:600],
                }
                for rank, i in enumerate(top, 1)
            ]

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=None, help="path to the trained checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="host/interface to bind")
    args = parser.parse_args()

    ckpt = find_checkpoint(args.ckpt)
    device = pick_device()
    print(f"[serve] loading {ckpt} on {device} …")
    model, tokenizer = load_trained_encoder(ckpt, device)

    jm = json.load(open(ROOT / "data/processed/jm_corpus.json"))
    jm_ids, jm_texts = list(jm), list(jm.values())
    jm_emb = np.load(ROOT / "models/embeddings/jm_embeddings.npy")
    print(f"[serve] {len(jm_ids)} book passages indexed")

    handler = build_handler(model, tokenizer, device, jm_ids, jm_texts, jm_emb)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[serve] ready → open {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
        server.shutdown()


if __name__ == "__main__":
    main()
