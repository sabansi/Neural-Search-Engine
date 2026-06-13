"""Contrastive (symmetric InfoNCE) and MLM losses, implemented in plain PyTorch."""

import torch
import torch.nn.functional as F


def info_nce_loss(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    temperature: float = 0.07,
    doc_ids: list | None = None,
) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE (NT-Xent) over in-batch negatives.

    `q_emb` and `d_emb` are (B, D) L2-normalised embeddings where row i of
    `d_emb` is the positive document for query i; the other B-1 rows act as
    negatives. The symmetric form also trains the document→query direction.

    `doc_ids`: SQuAD has ~4.6 questions per passage, so two queries in one
    batch can share the same positive document. Treating the duplicate as a
    negative would punish a correct match (a false negative), so those logits
    are masked out.

    Returns (loss, in-batch retrieval accuracy at rank 1).
    """
    sim = (q_emb @ d_emb.t()) / temperature  # (B, B)
    B = sim.size(0)
    targets = torch.arange(B, device=sim.device)

    if doc_ids is not None:
        codes: dict = {}
        ids = torch.tensor(
            [codes.setdefault(d, len(codes)) for d in doc_ids], device=sim.device
        )
        duplicate = (ids[:, None] == ids[None, :]) & ~torch.eye(
            B, dtype=torch.bool, device=sim.device
        )
        sim = sim.masked_fill(duplicate, torch.finfo(sim.dtype).min)

    loss = 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))
    acc = (sim.argmax(dim=1) == targets).float().mean().item()
    return loss, acc


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions only (labels are -100 elsewhere)."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
