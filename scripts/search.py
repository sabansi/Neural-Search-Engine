"""Interactive semantic search over the Jurafsky & Martin book.

Loads the trained encoder + the prebuilt book embeddings (models/embeddings/)
and answers natural-language queries instantly — only the query is encoded at
runtime, the 870 book chunks are already embedded, so each search is a single
forward pass (fast, runs fine on CPU).

Put the trained checkpoint at one of:
    runs/contrastive/ckpt_best.pt     (default)
    models/encoder.pt

Usage:
    python scripts/search.py                       # interactive prompt
    python scripts/search.py "how does beam search work?"   # one-shot
    python scripts/search.py --k 5 "what is attention?"
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.encode_and_eval import load_trained_encoder
from src.inference import encode_texts
from src.trainer import pick_device

CKPT_CANDIDATES = [ROOT / "runs/contrastive/ckpt_best.pt", ROOT / "models/encoder.pt"]


def find_checkpoint(explicit: Path | None) -> Path:
    if explicit:
        if explicit.exists():
            return explicit
        sys.exit(f"[search] checkpoint not found: {explicit}")
    for p in CKPT_CANDIDATES:
        if p.exists():
            return p
    sys.exit(
        "[search] no checkpoint found. Put the trained model at one of:\n"
        + "\n".join(f"    {p.relative_to(ROOT)}" for p in CKPT_CANDIDATES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="query text; omit for an interactive prompt")
    parser.add_argument("--ckpt", type=Path, default=None, help="path to the trained checkpoint")
    parser.add_argument("--k", type=int, default=5, help="number of passages to return")
    parser.add_argument("--chars", type=int, default=280, help="snippet length to print")
    args = parser.parse_args()

    ckpt = find_checkpoint(args.ckpt)
    device = pick_device()
    model, tokenizer = load_trained_encoder(ckpt, device)

    jm = json.load(open(ROOT / "data/processed/jm_corpus.json"))
    jm_ids, jm_texts = list(jm), list(jm.values())
    emb_path = ROOT / "models/embeddings/jm_embeddings.npy"
    if emb_path.exists():
        jm_emb = np.load(emb_path)
    else:
        # No prebuilt embeddings (e.g. fresh clone): encode the 870 book
        # passages once with this checkpoint — a few seconds, then cached.
        print("[search] prebuilt book embeddings not found — encoding the book once...")
        jm_emb = encode_texts(model, tokenizer, jm_texts, device, show_progress=True)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, jm_emb)
    print(f"[search] {ckpt.relative_to(ROOT)} on {device} | {len(jm_ids)} book passages indexed\n")

    def run(query: str) -> None:
        q_emb = encode_texts(model, tokenizer, [query], device, max_len=64)
        sims = (q_emb @ jm_emb.T)[0]
        for rank, i in enumerate(np.argsort(-sims)[: args.k], 1):
            snippet = " ".join(jm_texts[i].split())[: args.chars]
            print(f"  {rank}. [{jm_ids[i]}] (sim {sims[i]:.3f}) {snippet}...")
        print()

    if args.query:
        query = " ".join(args.query)
        print(f"Q: {query}")
        run(query)
        return

    print("Type a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        run(query)


if __name__ == "__main__":
    main()
