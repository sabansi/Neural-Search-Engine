"""Stage 1: train our WordPiece tokenizer on our own corpora.

Training text = SQuAD passages + training queries + the Jurafsky & Martin book
chunks. Only *train*-split queries are used so no test data influences the
vocabulary.

Usage:
    python scripts/train_tokenizer.py [--vocab-size 30000]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_corpus, load_jsonl
from src.tokenizer import train_wordpiece


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "tokenizer")
    args = parser.parse_args()

    if (args.out / "tokenizer.json").exists():
        print(f"[tokenizer] {args.out / 'tokenizer.json'} already exists — skipping (delete it to retrain)")
        return

    corpus = load_corpus(ROOT / "data/processed/corpus.json")
    jm = load_corpus(ROOT / "data/processed/jm_corpus.json")
    train_queries = [t["query"] for t in load_jsonl(ROOT / "data/processed/train.jsonl")]
    texts = list(corpus.values()) + list(jm.values()) + train_queries
    print(f"[tokenizer] training WordPiece (vocab={args.vocab_size}) on {len(texts):,} texts ...")

    tokenizer = train_wordpiece(texts, vocab_size=args.vocab_size, save_dir=args.out)
    print(f"[tokenizer] saved to {args.out} (vocab size {tokenizer.vocab_size:,})")

    sample = "How does beam search decoding work in neural machine translation?"
    ids, mask = tokenizer.encode_batch([sample], max_len=64)
    tokens = [tokenizer.tok.id_to_token(i) for i in ids[0].tolist()]
    print(f"[tokenizer] sample: {sample!r}\n            -> {tokens}")


if __name__ == "__main__":
    main()
