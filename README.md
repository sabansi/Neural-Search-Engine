# Neural Search Engine

A semantic search system built with contrastive learning (InfoNCE) over a shared bi-encoder architecture, trained on SQuAD v1.1.

## Project Structure

```
Neural-Search-Engine/
├── data/
│   ├── raw/                  # Downloaded dataset cache
│   └── processed/
│       ├── corpus.json       # 18,891 unique Wikipedia passages
│       ├── train.jsonl       # 70,079 (query, positive, negative) triplets
│       ├── val.jsonl         # 8,759 triplets
│       └── test.jsonl        # 8,761 triplets
├── notebooks/
│   ├── phase1_data_collection.ipynb   # SQuAD download, dedup, triplet creation
│   ├── phase2_baseline.ipynb          # BM25 + TF-IDF baselines, Recall@k / MRR
│   └── phase3_model_training.ipynb    # Bi-encoder + InfoNCE training (Phase 3 & 4)
├── models/                   # Saved checkpoints and corpus embeddings
├── evaluation/               # Results JSON, plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Phases

| Phase | Notebook | Status |
|-------|----------|--------|
| 1 — Data Collection | `notebooks/phase1_data_collection.ipynb` | ✅ Done |
| 2 — Baseline (BM25 + TF-IDF) | `notebooks/phase2_baseline.ipynb` | ✅ Done |
| 3 & 4 — Bi-Encoder + InfoNCE Training | `notebooks/phase3_model_training.ipynb` | ✅ Done |
| 5 — Vector Search over Jurafsky & Martin | _coming_ | 🔲 Pending |
| 6 — Evaluation & Report | _coming_ | 🔲 Pending |

## Dataset

**SQuAD v1.1** (`rajpurkar/squad`)
- ~87,600 (question, Wikipedia paragraph) pairs
- 18,891 unique passages in corpus
- 80 / 10 / 10 train / val / test split, no query overlap between splits
- Query: natural language question; Document: Wikipedia paragraph containing the answer

## Baseline Results (Phase 2)

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|-------|----------|----------|-----------|-----|
| TF-IDF | 0.461 | 0.677 | 0.753 | 0.553 |
| BM25 | 0.505 | 0.671 | 0.724 | 0.576 |

## Model (Phase 3 & 4)

- **Architecture**: DistilBERT-base-uncased → mean pooling → Linear(768→256) → L2-norm
- **Loss**: Symmetric InfoNCE with in-batch negatives (batch size 64 → 63 negatives/query)
- **Optimizer**: AdamW, lr=2e-5, weight decay=0.01, 10% linear warmup
- **Epochs**: 5
