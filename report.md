# Neural Search Engine — Project Report

**Course:** Natural Language Processing  
**Dataset:** SQuAD v1.1 (Stanford Question Answering Dataset)  
**Model:** Bi-Encoder with InfoNCE Contrastive Loss  

---

## 1. Introduction

A neural search engine accepts a natural language query and returns the most semantically relevant text passages from a corpus. Unlike traditional keyword-based search, neural retrieval encodes both the query and each document as dense vectors in a shared embedding space, enabling retrieval based on meaning rather than exact word overlap.

The standard pipeline is:

```
query → text encoder → query embedding → cosine similarity → top-k passages
```

This is especially important for cases where the user's wording differs from the passage's wording — for example, asking *"What label was Queen Victoria given during the Irish famine?"* when the passage says *"Victoria was labelled 'The Famine Queen'"*. Keyword-based systems fail here; a semantic model succeeds.

The project builds a complete retrieval pipeline: data preparation, non-neural baselines, a trained bi-encoder, and quantitative and qualitative evaluation.

---

## 2. Data

### 2.1 Dataset Choice — SQuAD v1.1

We use **SQuAD v1.1** (`rajpurkar/squad` on HuggingFace), a reading comprehension dataset of 87,599 (question, Wikipedia paragraph) pairs created by crowd workers.

| Property | Value |
|----------|-------|
| Source | Wikipedia articles |
| Query type | Natural language questions written by humans |
| Document type | Wikipedia paragraphs (100–250 words) |
| Total pairs | 87,599 |
| Unique passages | 18,891 |

**Why SQuAD is suitable:**  
Questions are written by humans about real Wikipedia text, not synthetically generated. There is low lexical overlap between the question and the passage (the question rephrases the information rather than copying it), which forces the model to learn semantic similarity rather than surface-level matching. The paragraphs are naturally sized at 100–250 words — no manual chunking is needed.

### 2.2 Query and Document Definition

- **Query:** A natural language question (e.g. *"To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"*)
- **Document:** The Wikipedia paragraph that contains the answer (e.g. the paragraph about Notre Dame)

### 2.3 Triplet Construction

Each training example is a triplet **(query, positive passage, negative passage)**:

- **Positive:** the Wikipedia paragraph containing the answer to the question
- **Negative:** a randomly sampled passage from the corpus that is not the positive

During training, we use **in-batch negatives** via InfoNCE loss — every other document in the batch serves as an implicit negative for each query, providing up to 63 negatives per query at batch size 64.

### 2.4 Train / Validation / Test Split

| Split | Triplets | Fraction |
|-------|----------|----------|
| Train | 70,079 | 80% |
| Validation | 8,759 | 10% |
| Test | 8,761 | 10% |

Splitting was done by shuffling and slicing. There is no query overlap between train and test sets.

### 2.5 Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Average query length | ~9 words |
| Average passage length | ~120 words |
| Total training triplets | 70,079 |
| Unique passages in corpus | 18,891 |

---

## 3. Baseline

We implemented two non-neural baselines to establish retrieval performance without any learned semantic representations.

### 3.1 TF-IDF Cosine Similarity

TF-IDF (Term Frequency–Inverse Document Frequency) represents each document as a sparse vector where each dimension corresponds to a vocabulary term. The weight for term $t$ in document $d$ is:

$$\text{TF-IDF}(t, d) = \log(1 + \text{tf}(t,d)) \cdot \log\frac{N}{\text{df}(t)}$$

At query time, the query is transformed into the same vector space and cosine similarity is computed against all document vectors. We use bigrams (`ngram_range=(1,2)`) and a vocabulary of 50,000 terms.

### 3.2 BM25 (Okapi BM25)

BM25 is a probabilistic ranking function that improves over TF-IDF in two key ways:
1. **Term frequency saturation:** adding the same term repeatedly has diminishing returns (controlled by parameter $k_1$)
2. **Document length normalisation:** shorter documents are not disadvantaged (controlled by parameter $b$)

We use the standard Okapi BM25 variant via `rank-bm25`.

### 3.3 Baseline Results

Evaluated on the full test set of 8,761 queries against 18,891 passages:

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|-------|----------|----------|-----------|-----|
| TF-IDF | 0.4594 | 0.6770 | 0.7535 | 0.5521 |
| BM25 | **0.5046** | 0.6708 | 0.7245 | **0.5757** |

BM25 achieves better Recall@1 and MRR than TF-IDF, confirming its known advantage for exact-match retrieval. TF-IDF achieves slightly higher Recall@10, likely due to the broader vocabulary coverage from bigrams.

**Limitation of both baselines:** Both rely entirely on exact vocabulary overlap. If the query uses different words than the passage, both methods fail — this is the core motivation for neural retrieval.

---

## 4. Model Architecture

### 4.1 Bi-Encoder

We implement a **shared-weight bi-encoder**: the same neural network encodes both the query and the document, mapping them into a common 256-dimensional embedding space.

```
Input text
    ↓
DistilBERT-base-uncased (66M params)
    ↓  last_hidden_state: (B, T, 768)
Mean Pooling over non-padding tokens
    ↓  pooled: (B, 768)
LayerNorm + Linear(768 → 256, no bias)
    ↓  projected: (B, 256)
L2 Normalisation
    ↓  embedding: (B, 256), unit norm
```

### 4.2 Component Choices

**DistilBERT-base-uncased** was chosen as the backbone because:
- 40% fewer parameters than BERT-base (66M vs 110M) with 97% of BERT's performance
- Faster training and inference, important for 5 epochs on ~70k examples
- Uncased version handles queries with varied capitalisation

**Mean pooling** over all token outputs (rather than using only the `[CLS]` token) produces more stable sentence representations that naturally downweight padding tokens via the attention mask.

**Projection layer** (768→256) serves two purposes: it reduces the embedding dimensionality for faster dot-product computation at search time, and it acts as a learned bottleneck that encourages the model to compress semantics rather than surface features.

**L2 normalisation** ensures that cosine similarity equals the dot product, making the embeddings compatible with FAISS inner-product indices.

### 4.3 Why Shared Weights?

Using the same encoder for queries and documents (rather than two separate encoders) reduces the parameter count by half, prevents the query encoder and document encoder from drifting into incompatible spaces, and works well in practice — this is the same approach used by DPR and SimCSE.

---

## 5. Contrastive Training

### 5.1 Loss Function — InfoNCE

We train with the **symmetric InfoNCE loss** (also called NT-Xent). Given a batch of $B$ (query, positive document) pairs:

1. Encode all queries → matrix $\mathbf{Q} \in \mathbb{R}^{B \times D}$
2. Encode all documents → matrix $\mathbf{D} \in \mathbb{R}^{B \times D}$
3. Compute similarity matrix $\mathbf{S} = \mathbf{Q}\mathbf{D}^T / \tau \in \mathbb{R}^{B \times B}$
4. Correct pairs are on the diagonal → targets = $[0, 1, 2, \ldots, B-1]$

$$\mathcal{L} = \frac{1}{2}\left[\text{CE}(\mathbf{S}, \text{targets}) + \text{CE}(\mathbf{S}^T, \text{targets})\right]$$

The symmetric formulation trains both directions: each query finds its document, and each document finds its query.

**Why InfoNCE?**  
With batch size 64, each query gets 63 free negatives (every other document in the batch). This is far more efficient than Triplet Loss, which provides exactly 1 negative per query. InfoNCE is also numerically stable — it never saturates like the hinge loss in Triplet Loss. It is the loss used by DPR, SimCSE, and CLIP.

**Temperature $\tau = 0.07$:**  
Lower temperature sharpens the similarity distribution, making the model focus harder on the hardest negatives. The value 0.07 is standard from MoCo and SimCLR and worked well in practice.

### 5.2 Training Configuration

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Base model | DistilBERT-base-uncased | Efficient, strong pretrained representations |
| Projection dim | 256 | Balance between expressiveness and speed |
| Temperature τ | 0.07 | Standard contrastive learning value |
| Batch size | 64 | Maximises in-batch negatives within GPU memory |
| Learning rate | 2e-5 | Standard transformer fine-tuning rate |
| Weight decay | 0.01 | Prevents overfitting |
| Warmup | 10% of steps | Prevents early training instability |
| LR schedule | Linear decay after warmup | Standard practice |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Epochs | 5 | Best checkpoint at epoch 3 |
| Optimizer | AdamW | Better than Adam for transformers |

### 5.3 Training Results

| Epoch | Train Loss | Val Loss | Val R@10 (in-batch) | Time |
|-------|-----------|----------|---------------------|------|
| 1 | 0.4301 | 0.2009 | 0.9982 | 7.6 min |
| 2 | 0.1613 | 0.1761 | 0.9983 | 7.6 min |
| **3** ✓ | **0.1057** | **0.1685** | **0.9991** | **7.5 min** |
| 4 | 0.0804 | 0.1649 | 0.9986 | 7.5 min |
| 5 | 0.0682 | 0.1641 | 0.9986 | 7.5 min |

The training loss decreases steadily across all 5 epochs. The validation loss also decreases but begins to plateau after epoch 3, which is where the best checkpoint was saved. The in-batch Recall@10 reaches 99.9% — note that this is an in-batch metric (64 candidates per query), not the full-corpus metric.

---

## 6. Evaluation

### 6.1 Metrics

- **Recall@k** — fraction of queries where the correct passage appears in the top-k retrieved results. Measures coverage at different retrieval depths.
- **MRR (Mean Reciprocal Rank)** — average of $1/\text{rank}$ of the first correct result. Captures ranking quality: finding the correct answer at rank 1 scores 1.0, at rank 2 scores 0.5, etc.

### 6.2 Full Corpus Results

Evaluated on 8,761 test queries against the full corpus of 18,891 passages:

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|-------|----------|----------|-----------|-----|
| TF-IDF | 0.4594 | 0.6770 | 0.7535 | 0.5521 |
| BM25 | 0.5046 | 0.6708 | 0.7245 | 0.5757 |
| **BiEncoder (ours)** | **0.4904** | **0.7422** | **0.8211** | **0.5983** |

### 6.3 Improvement Over Best Baseline

| Metric | Best Baseline | BiEncoder | Change |
|--------|--------------|-----------|--------|
| Recall@1 | 0.5046 (BM25) | 0.4904 | −2.8% |
| Recall@5 | 0.6770 (TF-IDF) | 0.7422 | **+9.6%** |
| Recall@10 | 0.7535 (TF-IDF) | 0.8211 | **+9.0%** |
| MRR | 0.5757 (BM25) | 0.5983 | **+3.9%** |

The bi-encoder outperforms both baselines on Recall@5, Recall@10, and MRR. The only metric where a baseline wins is Recall@1, where BM25 scores 0.5046 versus our 0.4904. This reflects a known pattern: BM25 is very strong at exact-match retrieval when the query words appear verbatim in the passage. However, for any retrieval depth beyond 1, the semantic model consistently outperforms both sparse methods.

### 6.4 Qualitative Analysis

#### Cases where BiEncoder succeeds but BM25 fails

**Example 1 — Acronym resolution**
> Query: *"What is the USAF?"*  
> Passage: *"The United States Air Force (USAF) is the aerial warfare service branch..."*  
> Neural rank: 1 | BM25 rank: not in top-5  
> *BM25 sees only the acronym "USAF" in the query but cannot expand it to "United States Air Force". The neural model maps both to similar embeddings.*

**Example 2 — Paraphrase**
> Query: *"What was the label given to Queen Victoria during the Great Famine?"*  
> Passage: *"...Victoria was labelled 'The Famine Queen'. She personally donated £2,000 to famine relief..."*  
> Neural rank: 1 | BM25 rank: not in top-5  
> *The query asks for a "label" while the passage says "labelled" — trivial for the neural model, but the word mismatch confuses BM25.*

**Example 3 — Implicit knowledge**
> Query: *"There are two hockey teams located in NYC. What are they?"*  
> Passage: *"The New York Islanders and the New York Rangers represent the city in the National Hockey League..."*  
> Neural rank: 1 | BM25 rank: not in top-5  
> *The query says "NYC" while the passage says "New York". The neural model understands these refer to the same place.*

**Example 4 — Date lookup**
> Query: *"When was the Treaty of Nanking signed?"*  
> Passage: *"Under the terms of the Treaty of Nanking, signed in 1843, Ningbo became one of the five Chinese treaty ports..."*  
> Neural rank: 1 | BM25 rank: not in top-5

#### Cases where BM25 succeeds but BiEncoder fails

**Example 1 — Technical jargon with exact match**
> Query: *"What was the name of the first song used to develop the MP3?"*  
> Passage: *"The song 'Tom's Diner' by Suzanne Vega was the first song used by Karlheinz Brandenburg to develop the MP3..."*  
> BM25 rank: 1 | Neural rank: not in top-5  
> *The query contains rare proper nouns ("MP3") that BM25 matches exactly. The neural model may not have seen enough examples with this vocabulary.*

**Example 2 — Rare terminology**
> Query: *"Saqaliba served as what?"*  
> Passage: *"Saqaliba refers to the Slavic mercenaries and slaves in the medieval Arab world..."*  
> BM25 rank: 1 | Neural rank: not in top-5  
> *"Saqaliba" is a rare word that appears verbatim in the passage — BM25's exact match is perfect here.*

#### Hard cases where all systems fail (5.6%)

Out of 3,000 test queries, 168 (5.6%) were not retrieved correctly by any system in the top 10. Examples include:
- Queries with typos (*"tranlation"* instead of *"translation"*)
- Queries referencing things by their foreign-language names (*"in Portuguese"*)
- Highly ambiguous short queries

### 6.5 Error Analysis Summary

| Category | Example | Root Cause |
|----------|---------|-----------|
| Typo in query | *"tranlation of parthenos"* | Neither model handles spelling errors |
| Foreign language reference | *"Where does Brazil's president live, in Portuguese?"* | Query asks for Portuguese name; passage uses Portuguese name but context is lost |
| Rare proper noun | *"Saqaliba"* | Neural model underrepresented in training |

---

## 7. Conclusion

### What worked

- The bi-encoder significantly outperforms both sparse baselines at Recall@5, Recall@10, and MRR — the most important metrics for real-world retrieval where presenting multiple candidates is acceptable.
- InfoNCE with in-batch negatives is efficient and stable: 63 free negatives per query at batch size 64, no explicit negative mining required.
- Mean pooling + projection layer is a lightweight but effective architecture on top of DistilBERT.
- Training converged in 3 epochs (~22 minutes on GPU).

### What didn't work / trade-offs

- BM25 remains better at Recall@1 for queries that contain exact terminology from the passage. For high-precision applications (only returning 1 result), a hybrid system (neural re-ranking of BM25 candidates) would be preferable.
- In-batch negatives are easy negatives — the model has not seen truly hard negatives (semantically similar but incorrect passages). Adding hard negative mining would likely improve Recall@1.
- The model was fine-tuned on SQuAD (Wikipedia QA) and used to search the Jurafsky & Martin textbook — a domain shift that may reduce performance.

### What we would do with more time

1. **Hard negative mining**: use BM25 to find near-miss passages and add them to the training batch for harder contrastive signal
2. **Cross-encoder re-ranker**: use a slower but more accurate cross-encoder to re-rank the top-10 bi-encoder results
3. **Domain adaptation**: fine-tune on NLP textbook QA pairs before searching the J&M book
4. **Larger model**: use `bert-base-uncased` instead of DistilBERT for potentially higher accuracy

---

## Appendix — System Summary

| Component | Choice |
|-----------|--------|
| Training data | SQuAD v1.1 (87,599 question-passage pairs) |
| Corpus | 18,891 unique Wikipedia paragraphs |
| Backbone | DistilBERT-base-uncased |
| Pooling | Mean pooling over all non-padding tokens |
| Projection | Linear(768→256) + LayerNorm |
| Normalisation | L2 (unit norm) |
| Loss | Symmetric InfoNCE, τ=0.07 |
| Optimizer | AdamW, lr=2e-5, wd=0.01 |
| Training | 5 epochs, best at epoch 3 |
| Index | Cosine similarity (numpy) / FAISS IndexFlatIP |
| Demo corpus | Jurafsky & Martin — Speech and Language Processing (3rd ed.) |
