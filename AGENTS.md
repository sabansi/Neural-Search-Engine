# AGENTS.md — project briefing for AI coding assistants

This file briefs AI assistants (Cursor, Codex, Claude, etc.) helping on this
repository. Read it fully before making changes.

## What this project is

University NLP course group project (deadline June 13): a neural search engine
that retrieves 200-300-word passages from the Jurafsky & Martin *Speech and
Language Processing* book given a natural-language query. Pipeline:

    query → text encoder → query embedding → vector search → top-k chunks

The encoder is a **from-scratch transformer bi-encoder** trained with
self-implemented symmetric InfoNCE on SQuAD v1.1 question-passage pairs, with
an optional self-supervised MLM warm-up stage on our own corpora.

## HARD RULES (course requirements — never violate)

1. **No pretrained models.** No pretrained weights, no HuggingFace model
   classes (`transformers.AutoModel` etc.), no `sentence-transformers`. The
   professor explicitly requires the model architecture to be implemented by
   the students. Everything in `src/model.py` is hand-written PyTorch — keep
   it that way.
2. The HF `tokenizers` **library** is allowed only as a utility to train OUR
   OWN WordPiece vocabulary from our own data. Never load a pretrained
   tokenizer/vocabulary.
3. The MLM warm-up (`scripts/pretrain_mlm.py`) is legal: it trains from random
   init on our own corpora. It is not "using a pretrained model".
4. Never modify anything in `data/processed/` or
   `evaluation/baseline_results.json` (frozen data and baseline results).
5. Never delete files in `runs/` (checkpoints and training logs).
6. The old DistilBERT-based notebooks on the `main` branch are the deprecated
   previous approach — do not port code from them or re-introduce
   `transformers` imports.

## Repo layout (branch: scratch-encoder)

    src/tokenizer.py        WordPiece training + TextTokenizer wrapper
    src/model.py            EncoderConfig, MultiHeadSelfAttention,
                            TransformerBlock, ScratchEncoder (~19M params), MLMHead
    src/data.py             datasets + collates (contrastive pairs, dynamic-mask MLM)
    src/losses.py           symmetric InfoNCE (with duplicate-positive masking), MLM CE
    src/trainer.py          training loop: checkpoint/resume, early stop, time budget,
                            JSONL metrics, curve plotting
    src/inference.py        batch encoding + full-corpus Recall@k / MRR
    scripts/sanity_checks.py      5-check forward/backward verification (run before training)
    scripts/train_tokenizer.py    stage 1
    scripts/pretrain_mlm.py       stage 2 (MLM warm-up)
    scripts/train_contrastive.py  stage 3 (main training)
    scripts/encode_and_eval.py    stage 4 (embeddings, eval vs baselines, demo)
    scripts/run_overnight.py      orchestrates stages 1-4 unattended with retries
    data/processed/         train/val/test.jsonl (query, positive_id, negative_id),
                            corpus.json (18,891 SQuAD passages), jm_corpus.json (870 book chunks)
    runs/<stage>/           metrics.jsonl, history.json, curves.png, ckpt_last.pt,
                            ckpt_best.pt, DONE marker

## Environment (Windows, RTX 5070)

Blackwell GPU (sm_120) → torch must be >= 2.7 from the cu128 wheel index:

    pip install --index-url https://download.pytorch.org/whl/cu128 torch
    pip install -r requirements-train.txt

Run all commands from the repo root. DataLoaders use num_workers=0 on purpose
(Windows compatibility) — do not "optimize" this.

## How to run / success criteria

| Command | Success looks like |
|---|---|
| `python scripts/sanity_checks.py` | all 5 checks PASS, exit code 0 |
| `python scripts/pretrain_mlm.py --smoke` | reaches "[trainer] finished", val_loss < init (~10.3) |
| `python scripts/train_contrastive.py --smoke` | reaches "[trainer] finished" |
| `python scripts/run_overnight.py` (12h budget default) | overnight.log ends with results table + "overnight run finished" |

Rerun semantics of `run_overnight.py`: a fully completed previous run (both
`runs/contrastive/DONE` and `evaluation/scratch_results.json` present) is
auto-archived to `runs/archive/<timestamp>/` and a fresh round starts; a
partial/crashed run resumes from checkpoints. Never delete `runs/archive/`.

During the real run: MLM val_loss should fall well below its ln(30000) ≈ 10.3
starting point (roughly 3-5 by the end); contrastive `val_r10_full` (true
full-corpus Recall@10 on 1000 val queries) should climb steadily — that is the
early-stopping metric. In-batch accuracy saturating near 1.0 is normal and not
informative.

## Known failure modes

- "no kernel image is available" → wrong torch build, reinstall from cu128 index.
- CUDA OOM → re-run the stage with `--batch-size 128`.
- Non-finite loss → trainer aborts intentionally; delete that stage's run dir
  and re-run with `--lr 1e-4`.
- A crashed/interrupted pipeline → re-run `run_overnight.py` with the same
  args; completed stages are skipped (DONE markers), the rest resumes from
  checkpoints (`--resume` is automatic on retry).

## Style

- Plain PyTorch + numpy only in `src/`. No new heavyweight dependencies.
- Keep code documented — clean, well-commented source files are explicitly
  part of the course grade.
- Metrics live in `runs/<stage>/metrics.jsonl` (one JSON per line, `type`:
  "step" or "epoch"); read those instead of re-running training to answer
  "how did training go" questions.
