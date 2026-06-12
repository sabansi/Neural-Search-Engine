"""Stage 2: self-supervised MLM warm-up of the from-scratch encoder.

Masked-language-model pre-training on OUR OWN corpora (SQuAD passages +
training queries + the J&M book), starting from random initialisation. This is
not "using a pretrained model" — we are the ones training it, from scratch —
it just gives the encoder basic language statistics before the contrastive
stage, which matters a lot for a 19M-parameter model with no prior knowledge.

Usage:
    python scripts/pretrain_mlm.py [--epochs 40] [--resume] [--smoke] [--time-budget-hours 5]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import TextDataset, load_corpus, load_jsonl, make_mlm_collate, train_val_split
from src.model import EncoderConfig, MLMHead, ScratchEncoder
from src.tokenizer import TextTokenizer
from src.trainer import Trainer, pick_device, set_seed, to_device, warmup_cosine_schedule


class MLMModel(nn.Module):
    """Encoder + MLM head bundled so the Trainer checkpoints both together."""

    def __init__(self, encoder: ScratchEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = MLMHead(encoder)

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder.hidden_states(input_ids, attention_mask))


def compute_loss(model, batch):
    hidden = model.encoder.hidden_states(batch["input_ids"], batch["attention_mask"])
    masked = batch["labels"] != -100
    if not masked.any():  # vanishingly rare, but keeps the step well-defined
        return hidden.sum() * 0.0, {"mlm_acc": 0.0}
    # Project ONLY the masked positions (~15%) onto the 30k vocabulary — the
    # loss ignores everything else anyway, and the full (B, T, vocab) logits
    # tensor is by far the most expensive part of the step.
    labels = batch["labels"][masked]
    logits = model.head(hidden[masked])
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(-1) == labels).float().mean().item()
    return loss, {"mlm_acc": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-len", type=int, default=288)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "tokenizer_scratch")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "mlm")
    parser.add_argument("--resume", action="store_true", help="continue from ckpt_last.pt")
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs — wiring check only")
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = TextTokenizer.load(args.tokenizer)
    corpus = load_corpus(ROOT / "data/processed/corpus.json")
    jm = load_corpus(ROOT / "data/processed/jm_corpus.json")
    queries = [t["query"] for t in load_jsonl(ROOT / "data/processed/train.jsonl")]
    texts = list(corpus.values()) + list(jm.values()) + queries

    if args.smoke:
        texts = texts[:2000]
        args.epochs = 2
        args.batch_size = min(args.batch_size, 32)
        args.run_dir = ROOT / "runs" / "mlm_smoke"

    train_texts, val_texts = train_val_split(texts, val_frac=0.02, seed=args.seed)
    collate = make_mlm_collate(tokenizer, max_len=args.max_len)
    train_loader = DataLoader(
        TextDataset(train_texts), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(TextDataset(val_texts), batch_size=args.batch_size, collate_fn=collate)

    device = pick_device()
    cfg = EncoderConfig(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = MLMModel(ScratchEncoder(cfg))
    print(f"[mlm] device={device}, encoder params={model.encoder.num_parameters() / 1e6:.1f}M, "
          f"train={len(train_texts):,} val={len(val_texts):,} texts, {len(train_loader)} steps/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = warmup_cosine_schedule(optimizer, total_steps=len(train_loader) * args.epochs)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="neural-search-scratch", name="mlm", config=vars(args))

    def validate(model):
        total, n = 0.0, 0
        for batch in val_loader:
            batch = to_device(batch, device)
            loss, _ = compute_loss(model, batch)
            total += loss.item()
            n += 1
        return {"val_loss": total / max(1, n)}

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        run_dir=args.run_dir,
        device=device,
        time_budget_hours=args.time_budget_hours,
        wandb_run=wandb_run,
        extra_state={"encoder_config": cfg.to_dict(), "tokenizer_path": str(args.tokenizer)},
    )
    trainer.fit(
        train_loader,
        compute_loss,
        epochs=args.epochs,
        validate=validate,
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
