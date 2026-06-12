"""Datasets and collate functions.

Tokenization happens inside the collate functions (the Rust tokenizer is fast
and batch-level padding-to-longest wastes far fewer tokens than padding every
example to the global maximum). DataLoaders are used with num_workers=0 so the
same code runs identically on macOS, Linux and Windows.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import TextTokenizer

# Queries are short questions (~9 words); documents are Wikipedia paragraphs
# (~120 words) or J&M book chunks (200-300 words).
QUERY_MAX_LEN = 64
DOC_MAX_LEN = 288


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: str | Path) -> dict[str, str]:
    """doc_id -> passage text."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class PairDataset(Dataset):
    """(query, positive doc id) pairs for InfoNCE with in-batch negatives.

    The triplets in train.jsonl also carry a random negative_id, but InfoNCE
    gets 255 in-batch negatives per query at batch size 256, which strictly
    dominates one extra random negative — so only the positive pair is used.
    """

    def __init__(self, triplets: list[dict]):
        self.items = [(t["query"], t["positive_id"]) for t in triplets]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.items[idx]


def make_contrastive_collate(
    tokenizer: TextTokenizer,
    corpus: dict[str, str],
    query_max_len: int = QUERY_MAX_LEN,
    doc_max_len: int = DOC_MAX_LEN,
):
    def collate(batch: list[tuple[str, str]]) -> dict:
        queries = [q for q, _ in batch]
        doc_ids = [d for _, d in batch]
        q_ids, q_mask = tokenizer.encode_batch(queries, max_len=query_max_len)
        d_ids, d_mask = tokenizer.encode_batch([corpus[d] for d in doc_ids], max_len=doc_max_len)
        return {
            "q_ids": q_ids,
            "q_mask": q_mask,
            "d_ids": d_ids,
            "d_mask": d_mask,
            "doc_ids": doc_ids,
        }

    return collate


class TextDataset(Dataset):
    """Plain list of texts for MLM pre-training."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def make_mlm_collate(tokenizer: TextTokenizer, max_len: int = DOC_MAX_LEN, mask_prob: float = 0.15):
    """Dynamic masking (re-sampled every epoch, as in RoBERTa): of the 15%
    selected positions, 80% become [MASK], 10% a random token, 10% unchanged."""

    special_ids = torch.tensor(sorted(tokenizer.special_ids))

    def collate(texts: list[str]) -> dict:
        input_ids, attention_mask = tokenizer.encode_batch(texts, max_len=max_len)
        labels = input_ids.clone()

        maskable = ~torch.isin(input_ids, special_ids)  # also excludes [PAD]
        selected = (torch.rand_like(input_ids, dtype=torch.float) < mask_prob) & maskable
        labels[~selected] = -100

        roll = torch.rand_like(input_ids, dtype=torch.float)
        to_mask = selected & (roll < 0.8)
        to_random = selected & (roll >= 0.8) & (roll < 0.9)

        input_ids = input_ids.clone()
        input_ids[to_mask] = tokenizer.mask_id
        if to_random.any():
            input_ids[to_random] = torch.randint(
                0, tokenizer.vocab_size, (int(to_random.sum()),), dtype=torch.long
            )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def train_val_split(items: list, val_frac: float = 0.02, seed: int = 42) -> tuple[list, list]:
    """Deterministic shuffle-and-slice split (used for the MLM text corpus)."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]
