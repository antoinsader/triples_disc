
# Wiki BRASK

A PyTorch implementation of the [BRASK](https://www.sciencedirect.com/science/article/abs/pii/S0957417423004062) algorithm — **Bidirectional Relation-guided Attention network with Semantics and Knowledge** — applied to the Wikidata5m dataset for relational triple extraction.

BRASK is used to extract structured knowledge triples `(head entity, relation, tail entity)` directly from natural language descriptions. It does this by learning to locate entity spans in text guided by relation embeddings, running both a forward pass (head → relation → tail) and a backward pass (tail → relation → head) jointly.

This implementation uses PyTorch with DDP support for distributed training and follows a staged circular training approach: entity extraction, relation-guided extraction, backward extraction, and full fine-tuning.

---

## Download Dataset

```bash
curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz

curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar

curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar
```

Place the extracted files in the path configured in `utils/settings.py` under `RAW_FILES`.

---

## Pipeline

### 1. Pre-processing — `prepare.py`

```bash
python prepare.py
```

A single interactive script that walks through all data preparation stages. At each stage the script asks whether to run it or skip it, so you can resume from any point without rerunning earlier steps. All stages operate on the **minimized** dataset (created in stage 1).

**Stages:**

1. **Minimization** — prompts for a fraction (0–1) of training triples to keep, then sampling the dataset files accordingly using those training triples. output would be minimized version of training triples, aliases, relations, and descriptions.
2. **Normalization** — cleans descriptions (remove non-English characters, Unicode NFKC, collapse spaces) and aliases (same + lowercasing + deduplication), then overwrites the minimized files in place.
3. **Relation embeddings** — computes one 768-dim BERT embedding per relation by mean-pooling across all its aliases (average of last two hidden layers, attention-mask pooled, batched with mixed precision).  `set R in paper`
4. **Description embeddings** — encodes every description with BERT, producing both per-token embeddings `(N, L, H)` and mean-pooled sentence embeddings `(N, H)`.

**Output:**

| File | Content |
|---|---|
| `minimized/triples_train.pkl` | Flat list of `(head_id, relation_id, tail_id)` tuples |
| `minimized/aliases.pkl` | Dictionary of `{entity_id: [alias_str, ...]}` |
| `minimized/relations.pkl` | Dictionary `{relation_id: [alias_str, ...]}` |
| `minimized/descriptions.pkl` | Dictionary `{entity_id: description_text}` |
| `minimized/relation_embeddings.npz` | Tensor `(n_relations, H)` |
| `minimized/description_embeddings_all.pt` | Tensor `(B, L, H)` — per-token |
| `minimized/description_embeddings_mean.pt` | Tensor `(B, H)` — mean-pooled |
| `minimized/description_embeddings_ids.pkl` | List of entity IDs matching the embedding rows |

---

### 2. Train TransE — `train_transe.py`

```bash
python train_transe.py
# or with DDP:
torchrun --nproc_per_node=<N_GPUS> train_transe.py
```

Trains the TransE knowledge graph embedding model (Bordes et al. 2013) on the minimized triples. For each training triple `(h, r, t)`, a negative triple is generated on-the-fly by corrupting either the head or tail with a random entity. The model minimises the margin-based loss `mean(MARGIN + ||h+r-t||₁ − ||h'+r−t'||₁)` with Adam + CosineAnnealingLR. Embedding tables are L1-normalised after every batch. Supports multi-GPU via PyTorch DDP (detected automatically from the `LOCAL_RANK` environment variable set by `torchrun`).

**Output:** `minimized/transe_rel_embs.npz` — relation embedding matrix of shape `(n_relations, TRANSE_EMB_DIM)`.

---

### 3. Build Gold Labels — `prepare_gold_labels.py`

```bash
python prepare_gold_labels.py
```

Generates token-level binary span labels for training the BRASK model. Unlike heuristic silver spans, gold labels are anchored to individual triples: for every triple `(h, r, t)` associated with a description, the script locates the head entity `h` and tail entity `t` in the description text using all known aliases, then records the exact token start and end positions using `tokenizer.char_to_token()`. Processing is parallelised across description chunks via `multiprocessing.Pool`.

This triple-anchored approach is necessary for computing the loss correctly: the model's head and tail predictions are evaluated per-triple, so the labels must reflect which token spans correspond to a specific `(head, relation, tail)` rather than to the entity in general.

**Output:** gold label files containing per-triple head and tail span positions, mapped to token indices, saved to the minimized data directory.

---

### 4. Train BRASK — `train.py`

> **Work in progress.** The model architecture and forward pass are implemented; the loss function and training loop are still being developed.

```bash
python train.py
```

Implements the full BRASK model as a PyTorch `nn.Module` with the following components:

- **`EntityExtractor`** — two linear heads predicting token-level start and end probabilities for entity spans, returning both probabilities and raw logits for BCE loss.
- **`RelationAttention`** — attention mechanism that produces relation-conditioned sentence representations; used twice in parallel: once with BERT semantic relation embeddings, once with TransE relation embeddings.
- **`FuseExtractor`** — fuses subject representations `s_k`, relation context `c_j`, and token embeddings `x_i` into `h_ijk` for object span prediction.
- **`BraskModel`** — full forward + backward pipeline: predict head spans → extract subject representations → compute relation attention → fuse → predict tail spans; symmetric backward path for tail-first extraction.
- **`entity_extractor_loss`** — masked BCE loss for staged training of the entity extractor alone (in `models/EntityExtractor.py`).

**Planned / not yet implemented:**
- Full `brask_loss` combining forward and backward terms with subject-slot masking.
- Span post-processing (overlapping spans, invalid spans where end < start).
- Gating fusion alternative in `FuseExtractor`.
- Relation smart pruning (top-k by cosine similarity pre-computed per description).
- Post-prediction alias filtering to improve precision.
- Circular staged training loop.

---

## Archive

The first version of the training code is preserved in [`/archive`](archive/). It was a monolithic pipeline that ran end-to-end but required large GPU RAM and did not scale well. The current codebase is a full refactor of that work with cleaner separation of concerns, DDP support, and a corrected labeling strategy. [changelog_mar2026](docs/changelog_mar2026) includes what I am refactoring now.


