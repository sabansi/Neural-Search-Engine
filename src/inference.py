"""Inference helpers: batch text encoding and dense retrieval evaluation."""

import numpy as np
import torch

from .data import DOC_MAX_LEN
from .model import ScratchEncoder
from .tokenizer import TextTokenizer


@torch.no_grad()
def encode_texts(
    model: ScratchEncoder,
    tokenizer: TextTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 256,
    max_len: int = DOC_MAX_LEN,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts into unit-norm float32 embeddings, (N, proj_dim)."""
    model.eval()
    chunks = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="encoding", total=(len(texts) + batch_size - 1) // batch_size)
    for start in iterator:
        batch = texts[start : start + batch_size]
        input_ids, attention_mask = tokenizer.encode_batch(batch, max_len=max_len)
        emb = model(input_ids.to(device), attention_mask.to(device))
        chunks.append(emb.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def retrieval_metrics(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    gold_indices: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    chunk: int = 1000,
) -> dict[str, float]:
    """Full-corpus Recall@k and (exact) MRR via brute-force cosine — dot
    product on unit vectors. Queries are processed in chunks to bound memory.

    The gold document's exact rank is 1 + the number of documents scoring
    strictly higher, so MRR is computed over the full ranking (comparable to
    the BM25/TF-IDF baseline numbers), not truncated at max(ks).
    """
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    n = len(query_emb)
    for start in range(0, n, chunk):
        sims = query_emb[start : start + chunk] @ corpus_emb.T  # (c, N_docs)
        gold = gold_indices[start : start + chunk][:, None]
        gold_sims = np.take_along_axis(sims, gold, axis=1)  # (c, 1)
        ranks = (sims > gold_sims).sum(axis=1) + 1
        for k in ks:
            hits[k] += int((ranks <= k).sum())
        rr_sum += float((1.0 / ranks).sum())

    metrics = {f"recall@{k}": hits[k] / n for k in ks}
    metrics["mrr"] = rr_sum / n
    return metrics
