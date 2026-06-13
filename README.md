# Neural Search Engine

A semantic search engine for retrieving passages from *Speech and Language
Processing* (Jurafsky & Martin) by natural-language query. A transformer
bi-encoder is trained from random initialisation — its own WordPiece tokenizer,
a hand-implemented transformer, a self-supervised MLM warm-up, and contrastive
InfoNCE training — on SQuAD v1.1 question–passage pairs.

The full write-up (in Georgian) is in [report.md](report.md).

## Results

Full-corpus test set (8,761 queries vs 18,891 passages):

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|-------|---------:|---------:|----------:|----:|
| TF-IDF | 0.461 | 0.677 | **0.753** | 0.553 |
| BM25 | **0.505** | 0.671 | 0.724 | **0.576** |
| Bi-Encoder | 0.382 | 0.627 | 0.711 | 0.495 |

## Repository layout

```
src/                     model + training library (plain PyTorch)
  tokenizer.py           WordPiece tokenizer (trained on our corpora)
  model.py               transformer bi-encoder, multi-head attention, MLM head
  losses.py              symmetric InfoNCE + MLM loss
  data.py                datasets and collate functions
  trainer.py             training loop: checkpoint/resume, early stop, logging
  inference.py           batch encoding + Recall@k / MRR
scripts/                 runnable pipeline stages
  train_tokenizer.py     1. train the WordPiece tokenizer
  pretrain_mlm.py        2. self-supervised MLM warm-up
  train_contrastive.py   3. contrastive InfoNCE training
  encode_and_eval.py     4. encode corpora, evaluate, demo search
  run_overnight.py       runs stages 1–4 unattended (resume + time budget)
  sanity_checks.py       forward/backward verification suite
notebooks/
  01_data_preparation.ipynb     SQuAD download, dedup, splits
  02_baselines.ipynb            TF-IDF + BM25, Recall@k / MRR
  03_model_training.ipynb       architecture + training curves
  04_evaluation_and_search.ipynb  results vs baselines + semantic-search demo
data/processed/          corpus.json, jm_corpus.json, train/val/test.jsonl
models/tokenizer/        trained WordPiece tokenizer
models/embeddings/       corpus + book embeddings, FAISS index
evaluation/              results.json, baseline_results.json, demo_results.json
runs/{mlm,contrastive}/  training history, metrics, loss curves
```

## Setup

The training GPU is an NVIDIA Blackwell card (RTX 50-series, sm_120), so install
PyTorch from the CUDA 12.8 wheel index **first**, then the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt
```

(On CPU/other GPUs a normal `pip install torch` is fine.)

## Reproducing the pipeline

```bash
python scripts/sanity_checks.py        # verify the build (forward/backward)
python scripts/train_tokenizer.py      # 1. tokenizer  → models/tokenizer/
python scripts/pretrain_mlm.py         # 2. MLM warm-up → runs/mlm/
python scripts/train_contrastive.py    # 3. contrastive → runs/contrastive/
python scripts/encode_and_eval.py      # 4. embeddings + eval + demo
```

Or run all four stages unattended (with checkpoint/resume and a time budget):

```bash
python scripts/run_overnight.py --time-budget-hours 12
```

The trained encoder checkpoint (`*.pt`) is not stored in the repository.
`encode_and_eval.py` and notebook 04 use it if present at
`runs/contrastive/ckpt_best.pt` or `models/encoder.pt`; otherwise notebook 04
displays the recorded search results from `evaluation/demo_results.json`.

## Demo — semantic search over the book

Interactive search of *Speech and Language Processing* by natural-language query.
Three things are needed: this repo, the Python dependencies, and the trained
checkpoint (distributed separately — it is ~216 MB and not in the repo).

```bash
# 1. dependencies (see Setup above)
pip install -r requirements.txt

# 2. place the trained checkpoint here:
#    runs/contrastive/ckpt_best.pt        (download link: <ADD DRIVE LINK>)

# 3. run the demo
python scripts/search.py                       # interactive prompt
python scripts/search.py "how does beam search decoding work?"   # one-shot
python scripts/search.py --k 5 "what is perplexity?"
```

Each search encodes only the query (one forward pass), so it is instant and runs
fine on CPU. The book passages are already embedded in `models/embeddings/`; if
that file is absent (e.g. a partial clone) the script re-encodes the book once
from the checkpoint. Queries that work well: *how are word2vec embeddings
trained?*, *what is a hidden Markov model?*, *how does the Viterbi algorithm
work?*, *how does naive Bayes classify text?*

## Data

**SQuAD v1.1** (`rajpurkar/squad`): ~87,600 (question, Wikipedia paragraph)
pairs; 18,891 unique passages; 80/10/10 train/val/test split with no query
overlap. Query = natural-language question; document = the passage containing
the answer.
