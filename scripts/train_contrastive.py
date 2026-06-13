"""Stage 3: contrastive training of the bi-encoder with symmetric InfoNCE.

Initialises from the MLM warm-up checkpoint when one exists (pass
--no-init to train from random initialisation instead). Validation each epoch
reports both in-batch metrics and TRUE full-corpus retrieval (Recall@k over
all 18,891 passages for a sample of validation queries) — the in-batch number
saturates quickly and is not a good early-stopping signal.

Usage:
    python scripts/train_contrastive.py [--epochs 40] [--resume] [--smoke]
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import PairDataset, load_corpus, load_jsonl, make_contrastive_collate
from src.inference import encode_texts, retrieval_metrics
from src.losses import info_nce_loss
from src.model import EncoderConfig, Encoder
from src.tokenizer import TextTokenizer
from src.trainer import Trainer, pick_device, set_seed, to_device, warmup_cosine_schedule


def load_mlm_encoder_weights(encoder: Encoder, ckpt_path: Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    print(f"[contrastive] initialised encoder from MLM checkpoint {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--init-from", type=Path, default=ROOT / "runs" / "mlm" / "ckpt_best.pt")
    parser.add_argument("--no-init", action="store_true", help="skip MLM init, start from random weights")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models" / "tokenizer")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "contrastive")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs — wiring check only")
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-queries", type=int, default=1000, help="val queries for full-corpus retrieval")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = TextTokenizer.load(args.tokenizer)
    corpus = load_corpus(ROOT / "data/processed/corpus.json")
    train_triplets = load_jsonl(ROOT / "data/processed/train.jsonl")
    val_triplets = load_jsonl(ROOT / "data/processed/val.jsonl")

    if args.smoke:
        train_triplets = train_triplets[:1000]
        val_triplets = val_triplets[:200]
        args.epochs = 2
        args.batch_size = min(args.batch_size, 32)
        args.val_queries = 50
        args.run_dir = ROOT / "runs" / "contrastive_smoke"

    collate = make_contrastive_collate(tokenizer, corpus)
    train_loader = DataLoader(
        PairDataset(train_triplets), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(PairDataset(val_triplets), batch_size=args.batch_size, collate_fn=collate)

    device = pick_device()
    cfg = EncoderConfig(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = Encoder(cfg)
    if not args.no_init and Path(args.init_from).exists():
        load_mlm_encoder_weights(model, args.init_from)
    else:
        print("[contrastive] training from RANDOM initialisation (no MLM warm-up)")
    print(f"[contrastive] device={device}, params={model.num_parameters() / 1e6:.1f}M, "
          f"{len(train_loader)} steps/epoch, batch={args.batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = warmup_cosine_schedule(optimizer, total_steps=len(train_loader) * args.epochs)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project="neural-search", name="contrastive", config=vars(args))

    def compute_loss(model, batch):
        q_emb = model(batch["q_ids"], batch["q_mask"])
        d_emb = model(batch["d_ids"], batch["d_mask"])
        loss, acc = info_nce_loss(q_emb, d_emb, args.temperature, doc_ids=batch["doc_ids"])
        return loss, {"in_batch_acc": acc}

    # Fixed retrieval-validation setup: a deterministic sample of val queries
    # scored against the FULL corpus (or a gold-inclusive subset in smoke mode).
    rng = random.Random(args.seed)
    val_sample = rng.sample(val_triplets, min(args.val_queries, len(val_triplets)))
    if args.smoke:
        doc_ids = sorted({t["positive_id"] for t in val_sample} | set(rng.sample(sorted(corpus), 2000)))
    else:
        doc_ids = sorted(corpus)
    doc_texts = [corpus[d] for d in doc_ids]
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    gold = np.array([doc_index[t["positive_id"]] for t in val_sample])
    val_queries = [t["query"] for t in val_sample]

    def validate(model):
        total, total_acc, n = 0.0, 0.0, 0
        for batch in val_loader:
            batch = to_device(batch, device)
            loss, acc = compute_loss(model, batch)
            total += loss.item()
            total_acc += acc["in_batch_acc"]
            n += 1
        corpus_emb = encode_texts(model, tokenizer, doc_texts, device, batch_size=args.batch_size)
        query_emb = encode_texts(model, tokenizer, val_queries, device, batch_size=args.batch_size, max_len=64)
        retrieval = retrieval_metrics(query_emb, corpus_emb, gold)
        return {
            "val_loss": total / max(1, n),
            "val_in_batch_acc": total_acc / max(1, n),
            "val_r1_full": retrieval["recall@1"],
            "val_r10_full": retrieval["recall@10"],
            "val_mrr_full": retrieval["mrr"],
        }

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        run_dir=args.run_dir,
        device=device,
        time_budget_hours=args.time_budget_hours,
        wandb_run=wandb_run,
        extra_state={"config": cfg.to_dict(), "tokenizer_path": str(args.tokenizer),
                     "temperature": args.temperature},
    )
    trainer.fit(
        train_loader,
        compute_loss,
        epochs=args.epochs,
        validate=validate,
        monitor="val_r10_full",
        mode="max",
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
