"""Stage 4: encode corpora with the trained encoder, evaluate, and demo.

  1. Encodes the SQuAD corpus and the J&M book chunks → models/embeddings/*.npy
  2. Evaluates full-corpus retrieval on the test split (Recall@1/5/10, MRR)
     and prints a comparison against the stored TF-IDF / BM25 baselines
  3. Builds a FAISS inner-product index over the J&M chunks (optional dep)
  4. Runs a few demo queries against the book

Usage:
    python scripts/encode_and_eval.py [--ckpt runs/contrastive/ckpt_best.pt] [--demo "your query"]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import load_corpus, load_jsonl
from src.inference import encode_texts, retrieval_metrics
from src.model import EncoderConfig, Encoder
from src.tokenizer import TextTokenizer
from src.trainer import pick_device

DEMO_QUERIES = [
    "How does beam search decoding work?",
    "What is the difference between stemming and lemmatization?",
    "How are word embeddings learned with skip-gram?",
    "What problem does attention solve in sequence to sequence models?",
]


def load_trained_encoder(ckpt_path: Path, device) -> tuple[Encoder, TextTokenizer]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**(payload.get("config") or payload["encoder_config"]))
    encoder = Encoder(cfg)
    state = payload["model_state"]
    if any(k.startswith("encoder.") for k in state):  # MLM-stage checkpoint
        state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    # The checkpoint records the tokenizer path from the training machine; fall
    # back to the in-repo tokenizer when that absolute path is not present here.
    tok_path = Path(payload.get("tokenizer_path", ""))
    if not tok_path.exists():
        tok_path = ROOT / "models" / "tokenizer"
    tokenizer = TextTokenizer.load(tok_path)
    return encoder, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "contrastive" / "ckpt_best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "embeddings")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-eval", action="store_true", help="skip test-set evaluation")
    parser.add_argument("--demo", nargs="*", default=None, help="demo queries (default: built-in examples)")
    parser.add_argument("--limit", type=int, default=None,
                        help="debug: only encode/evaluate this many docs and queries (wiring check)")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer = load_trained_encoder(args.ckpt, device)
    if args.limit:  # debug artifacts must not clobber the real embeddings
        args.out = args.out / "limit_debug"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] loaded {args.ckpt} on {device}")

    # ---- encode both corpora -------------------------------------------------
    corpus = load_corpus(ROOT / "data/processed/corpus.json")
    jm = load_corpus(ROOT / "data/processed/jm_corpus.json")
    doc_ids, doc_texts = list(corpus.keys()), list(corpus.values())
    jm_ids, jm_texts = list(jm.keys()), list(jm.values())
    if args.limit:
        doc_ids, doc_texts = doc_ids[: args.limit], doc_texts[: args.limit]
        jm_ids, jm_texts = jm_ids[: args.limit], jm_texts[: args.limit]

    corpus_emb = encode_texts(model, tokenizer, doc_texts, device, args.batch_size, show_progress=True)
    jm_emb = encode_texts(model, tokenizer, jm_texts, device, args.batch_size, show_progress=True)
    np.save(args.out / "corpus_embeddings.npy", corpus_emb)
    np.save(args.out / "jm_embeddings.npy", jm_emb)
    (args.out / "doc_ids.json").write_text(json.dumps(doc_ids))
    (args.out / "jm_ids.json").write_text(json.dumps(jm_ids))
    print(f"[eval] saved embeddings to {args.out} (corpus {corpus_emb.shape}, jm {jm_emb.shape})")

    # ---- FAISS index over the book (optional dependency) ---------------------
    try:
        import faiss

        index = faiss.IndexFlatIP(jm_emb.shape[1])
        index.add(jm_emb)
        faiss.write_index(index, str(args.out / "jm_faiss.index"))
        print("[eval] FAISS index written")
    except ImportError:
        print("[eval] faiss not installed — skipping index build (numpy search works fine)")

    # ---- full-corpus test evaluation -----------------------------------------
    test = None
    if not args.no_eval:
        test = load_jsonl(ROOT / "data/processed/test.jsonl")
        doc_index = {d: i for i, d in enumerate(doc_ids)}
        if args.limit:  # keep only queries whose gold doc survived the limit
            test = [t for t in test if t["positive_id"] in doc_index][: args.limit]
            if not test:
                print("[eval] --limit left no evaluable queries; skipping eval")
    if not args.no_eval and test:
        gold = np.array([doc_index[t["positive_id"]] for t in test])
        query_emb = encode_texts(
            model, tokenizer, [t["query"] for t in test], device, args.batch_size, max_len=64, show_progress=True
        )
        metrics = retrieval_metrics(query_emb, corpus_emb, gold)
        results = {"Bi-Encoder": {
            "Recall@1": metrics["recall@1"], "Recall@5": metrics["recall@5"],
            "Recall@10": metrics["recall@10"], "MRR": metrics["mrr"],
        }}

        baseline_path = ROOT / "evaluation" / "baseline_results.json"
        if baseline_path.exists():
            results = {**json.loads(baseline_path.read_text()), **results}

        print(f"\n[eval] test set: {len(test):,} queries vs {len(doc_ids):,} passages")
        header = f"{'Model':<22}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
        print(header + "\n" + "-" * len(header))
        for name, m in results.items():
            print(f"{name:<22}{m['Recall@1']:>8.4f}{m['Recall@5']:>8.4f}{m['Recall@10']:>8.4f}{m['MRR']:>8.4f}")
        if args.limit:
            print("[eval] --limit set: numbers are meaningless, NOT writing results.json")
        else:
            out_path = ROOT / "evaluation" / "results.json"
            out_path.write_text(json.dumps(results, indent=2))
            print(f"[eval] written to {out_path}")

    # ---- demo against the book ------------------------------------------------
    queries = args.demo if args.demo else DEMO_QUERIES
    print("\n[demo] searching Jurafsky & Martin:")
    q_emb = encode_texts(model, tokenizer, queries, device, max_len=64)
    sims = q_emb @ jm_emb.T
    for qi, query in enumerate(queries):
        print(f"\n  Q: {query}")
        for rank, di in enumerate(np.argsort(-sims[qi])[:3], 1):
            snippet = " ".join(jm_texts[di].split())[:180]
            print(f"   {rank}. [{jm_ids[di]}] (sim {sims[qi, di]:.3f}) {snippet}...")


if __name__ == "__main__":
    main()
