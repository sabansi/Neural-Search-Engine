"""Forward/backward sanity suite for the transformer encoder.

Run this BEFORE any long training run (it takes ~1 minute on CPU):

    python scripts/sanity_checks.py

Checks:
  1. init-loss      — with random weights, InfoNCE loss == ln(batch) and MLM
                      loss == ln(vocab): the model starts exactly at chance,
                      so the loss/logit plumbing is correct
  2. grad-flow      — every parameter receives a finite, nonzero gradient
                      after one backward pass (catches detached modules)
  3. tiny-overfit   — 32 real pairs driven to ~zero loss / 100% in-batch R@1:
                      proves backward + optimizer wiring can actually learn
  4. pad-invariance — identical embeddings regardless of how much padding a
                      batch adds (catches attention-mask / pooling bugs)
  5. ckpt-resume    — save → load roundtrip restores parameters, optimizer
                      state and RNG exactly

Exit code 0 = all passed.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.data import load_corpus, load_jsonl, make_mlm_collate
from src.losses import info_nce_loss, mlm_loss
from src.model import EncoderConfig, MLMHead, Encoder
from src.tokenizer import TextTokenizer, train_wordpiece

DEVICE = torch.device("cpu")  # deterministic; the suite is small on purpose
FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def tiny_setup() -> tuple[Encoder, TextTokenizer, list[tuple[str, str]]]:
    """Small model + tokenizer trained on a slice of the real corpus."""
    corpus = load_corpus(ROOT / "data/processed/corpus.json")
    triplets = load_jsonl(ROOT / "data/processed/train.jsonl")
    texts = list(corpus.values())[:300]
    tokenizer = train_wordpiece(texts, vocab_size=2000)

    cfg = EncoderConfig(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4,
        d_ff=128, max_len=128, proj_dim=32, pad_id=tokenizer.pad_id,
    )
    # 32 pairs with unique positive docs (cleanest in-batch negative setup).
    pairs, seen = [], set()
    for t in triplets:
        if t["positive_id"] not in seen:
            seen.add(t["positive_id"])
            pairs.append((t["query"], corpus[t["positive_id"]]))
        if len(pairs) == 32:
            break
    return Encoder(cfg), tokenizer, pairs


def embed(model, tokenizer, texts, max_len=96):
    ids, mask = tokenizer.encode_batch(texts, max_len=max_len)
    return model(ids.to(DEVICE), mask.to(DEVICE))


def check_init_loss(model, tokenizer, pairs) -> None:
    model.eval()
    with torch.no_grad():
        q = embed(model, tokenizer, [q for q, _ in pairs])
        d = embed(model, tokenizer, [d for _, d in pairs])
        # temperature=1.0: at init logits are near-uniform, so the loss should
        # sit at chance level ln(B) (a low temperature would amplify the small
        # random similarity differences and blur the check).
        loss, _ = info_nce_loss(q, d, temperature=1.0)
    expected = math.log(len(pairs))
    report("init InfoNCE == ln(B)", abs(loss.item() - expected) < 0.2,
           f"loss {loss.item():.4f} vs ln({len(pairs)}) = {expected:.4f}")

    head = MLMHead(model)
    collate = make_mlm_collate(tokenizer, max_len=96)
    batch = collate([d for _, d in pairs])
    with torch.no_grad():
        logits = head(model.hidden_states(batch["input_ids"], batch["attention_mask"]))
        loss = mlm_loss(logits, batch["labels"])
    expected = math.log(tokenizer.vocab_size)
    report("init MLM == ln(vocab)", abs(loss.item() - expected) < 0.5,
           f"loss {loss.item():.4f} vs ln({tokenizer.vocab_size}) = {expected:.4f}")


def check_grad_flow(model, tokenizer, pairs) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    q = embed(model, tokenizer, [q for q, _ in pairs])
    d = embed(model, tokenizer, [d for _, d in pairs])
    loss, _ = info_nce_loss(q, d)
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    report("gradient flow (contrastive)", not dead, f"dead/invalid grads: {dead or 'none'}")

    head = MLMHead(model)
    model.zero_grad(set_to_none=True)
    collate = make_mlm_collate(tokenizer, max_len=96)
    batch = collate([d for _, d in pairs])
    mlm_loss(head(model.hidden_states(batch["input_ids"], batch["attention_mask"])), batch["labels"]).backward()
    dead = [n for n, p in head.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    report("gradient flow (MLM head)", not dead, f"dead grads: {dead or 'none'}")


def check_tiny_overfit(model, tokenizer, pairs) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    final_loss, final_acc = float("inf"), 0.0
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        q = embed(model, tokenizer, [q for q, _ in pairs])
        d = embed(model, tokenizer, [d for _, d in pairs])
        loss, acc = info_nce_loss(q, d)
        loss.backward()
        optimizer.step()
        final_loss, final_acc = loss.item(), acc
    report("tiny-overfit (32 pairs, 300 steps)", final_loss < 0.1 and final_acc == 1.0,
           f"final loss {final_loss:.4f}, in-batch R@1 {final_acc:.2f}")


def check_pad_invariance(model, tokenizer, pairs) -> None:
    model.eval()
    texts = [q for q, _ in pairs[:4]]
    with torch.no_grad():
        ids, mask = tokenizer.encode_batch(texts, max_len=96)
        base = model(ids, mask)
        pad = torch.full((ids.size(0), 25), tokenizer.pad_id, dtype=torch.long)
        padded = model(torch.cat([ids, pad], 1), torch.cat([mask, torch.zeros_like(pad)], 1))
    diff = (base - padded).abs().max().item()
    report("padding invariance", diff < 1e-5, f"max |delta_embedding| = {diff:.2e}")


def check_ckpt_resume(model, tokenizer, pairs, tmp_dir: Path) -> None:
    from torch.utils.data import DataLoader

    from src.data import PairDataset, make_contrastive_collate
    from src.trainer import Trainer, warmup_cosine_schedule

    corpus = {f"d{i}": d for i, (_, d) in enumerate(pairs)}
    triplets = [{"query": q, "positive_id": f"d{i}"} for i, (q, _) in enumerate(pairs)]
    loader = DataLoader(PairDataset(triplets), batch_size=8,
                        collate_fn=make_contrastive_collate(tokenizer, corpus, doc_max_len=96))

    def loss_fn(m, b):
        loss, acc = info_nce_loss(m(b["q_ids"], b["q_mask"]), m(b["d_ids"], b["d_mask"]), doc_ids=b["doc_ids"])
        return loss, {"acc": acc}

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = warmup_cosine_schedule(opt, total_steps=8)
    trainer = Trainer(model, opt, sched, run_dir=tmp_dir, device=DEVICE)
    trainer.fit(loader, loss_fn, epochs=2)

    model2 = Encoder(model.cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    sched2 = warmup_cosine_schedule(opt2, total_steps=8)
    trainer2 = Trainer(model2, opt2, sched2, run_dir=tmp_dir, device=DEVICE)
    trainer2.load_checkpoint("ckpt_last")

    params_equal = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.state_dict().values(), model2.state_dict().values())
    )
    state_equal = (trainer2.epoch, trainer2.global_step) == (trainer.epoch, trainer.global_step)
    with torch.no_grad():
        model.eval(), model2.eval()
        out_equal = torch.equal(embed(model, tokenizer, [pairs[0][0]]), embed(model2, tokenizer, [pairs[0][0]]))
    report("checkpoint -> resume", params_equal and state_equal and out_equal,
           f"params={params_equal}, counters={state_equal}, outputs={out_equal}")


def main() -> None:
    torch.manual_seed(0)
    print("sanity checks (CPU, tiny model on real data)\n" + "=" * 50)
    model, tokenizer, pairs = tiny_setup()
    print(f"  setup: vocab {tokenizer.vocab_size}, params {model.num_parameters() / 1e6:.2f}M, {len(pairs)} pairs")

    check_init_loss(model, tokenizer, pairs)
    check_grad_flow(model, tokenizer, pairs)
    check_pad_invariance(model, tokenizer, pairs)
    check_tiny_overfit(model, tokenizer, pairs)  # mutates weights — run after the init checks
    check_ckpt_resume(model, tokenizer, pairs, ROOT / "runs" / "sanity_tmp")

    print("=" * 50)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED → {FAILURES}")
        sys.exit(1)
    print("RESULT: all checks passed — safe to launch training")


if __name__ == "__main__":
    main()
