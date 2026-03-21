# Development Changelog — March 2026

> Changes from Mar 16 – Mar 20, 2026.
> Scope: model architecture, data preparation pipeline, and codebase restructuring.
> This was writtien by ```claude``` asking to read my git stats history and file changes since Mar 16.


---

## Mar 16 — TransE Output Path & Negative Sampling Fix
**Commit:** `670d23d`

- Fixed TransE output path to use `settings.MINIMIZED_FILES.TRANSE_MODEL_RESULTS` / `settings.PREPROCESSED_FILES.TRANSE_MODEL_RESULTS` instead of a hardcoded directory join, so results land in the correct folder for both minimized and full datasets.
- Replaced NumPy-based negative triple generation (`rng.integers`) with PyTorch `torch.randint`, keeping the random corruption of head or tail consistent with the rest of the training stack.
- Added `TRANSE_MODEL_RESULTS` to both `_PreprocessedFiles` and `_MinimizedFiles` settings dataclasses.

---

## Mar 17 — Silver Spans Creation & `train.py` Scaffold
**Commits:** `d380f00`, `685e3f0`

- Created `prepare_silver_spans.py`: discovers token-level head and tail entity spans inside descriptions using alias regex matching; processes descriptions in parallel chunks via `multiprocessing.Pool`; produces binary label tensors `(N, L)` for `head_start`, `head_end`, `tail_start`, `tail_end`.
- Created initial `train.py` with `BraskDataset`, an empty `BraskModel` skeleton, and `check_training_files()` to validate required input files before training.
- Created `prepare_embeddings.py` for computing BERT description embeddings (per-token and mean-pooled).
- Extended `utils/settings.py` with new file paths: `SILVER_SPANS`, `DESCRIPTION_EMBEDDINGS_ALL`, `DESCRIPTION_EMBEDDINGS_MEAN`, `DESCRIPTION_EMBEDDINGS_IDS`.

---

## Mar 18 — Model Components Implemented (Demo 1)
**Commits:** `37913d6`, `e88b496`, `d0882f1`

### Architecture notes added (future plans)
Planning comments were written into `BraskModel.__init__` — these are **not yet implemented**:

- **Span reconstruction issues:** the model predicts start and end independently, so multiple possible spans, overlapping spans, and invalid spans (end < start) need post-processing logic.
- **Loss weighting:** consider a single `alpha` parameter instead of three separate weights (entity loss, forward loss, backward loss).
- **Gating fusion alternative:** instead of additive attention fusion, optionally use `g = sigmoid(W[x_i, r]); h_i = g * x_i + (1-g) * r` (a `gating = False` placeholder exists in `FuseExtractor`).
- **Relation smart pruning:** pre-compute cosine similarity between descriptions and relations (possibly with sentence transformers) to select top-k candidate relations per description, augmented by ground-truth positive relations.
- **Post-prediction alias filtering:** after span prediction, keep only candidate spans matching known aliases (exact or fuzzy) to improve precision — the model should not be forced to predict aliases during training.
- **Circular (staged) training curriculum:**
  1. Train entity extractor alone.
  2. Freeze encoder; train relation attention + forward extractor (isolates relation conditioning).
  3. Add backward extractor.
  4. Full fine-tuning.

### Implemented components
- **`EntityExtractor(nn.Module)`** — two linear heads (`start_linear`, `end_linear`) applied to token embeddings; returns `start_probs, end_probs, start_logits, end_logits` each of shape `(..., L, 1)`.
- **`RelationAttention(nn.Module)`** — attention mechanism guided by relation embeddings and global sentence context: computes `z = tanh(W_x·xi + W_r·rj + W_g·hg)`, then softmax attention weights `a` and context vectors `c` of shape `(B, R, H)`. Used twice in `BraskModel`: once with BERT-based semantic relation embeddings, once with TransE relation embeddings.
- **`FuseExtractor(nn.Module)`** — fuses subject representation `s_k`, relation context `c_j`, and token embedding `x_i` into a joint representation `h_ijk = h_ij + h_ik` where `h_ij = c_j + x_i` and `h_ik = W_s·s_k + W_x·x_i`; output shape `(B, R, S, L, H)`.
- **`extract_sk()`** — given predicted start/end probabilities and thresholds, extracts entity span representations as `avg(x[start] + x[end]) / 2`, then pads to a fixed `max_num_subjects` dimension with a boolean mask.
- **`BraskModel.forward()`** — full forward pass: head/tail prediction → dual relation attention → sk extraction → fuse → tail/head prediction; returns a dict of logits and masks for both forward and backward paths.
- **`BraskDataset`** simplified: items are now `(description_embedding, description_mean_embedding, description_id)` — silver spans are loaded and matched externally, not bundled per dataset item.

---

## Mar 19 — Gold Labels Approach & Loss Drafts
**Commits:** `ced1fdc`, `d2bbbe8`

### Open training questions documented in comments
- Whether to use silver spans or live model predictions as input to `extract_sk` during training (gradient flow concern: using silver spans means gradients do not propagate back through `forward_head_start/end` to the encoder).
- Whether to flatten `(B, R, S, L, H)` before passing to the tail predictor vs keeping the 5-D structure.
- Loss masking: padded subject slots (from `extract_sk` padding) must be zeroed out when computing object prediction loss; `forward_sk_mask` and `backward_sk_mask` are now included in the model output dict.

### Loss function drafts (future implementation — currently stubs)
- `entity_extractor_loss()` stub: BCE explanation and `pos_weight` rationale written; not yet implemented.
- `loss_compute()` stub: placeholder for full Brask loss combining forward and backward paths.
- Commented-out `entity_loss` and `brask_loss` draft in `train.py` showing the intended structure: per-token BCE with masking, separate subject and object losses, symmetric forward and backward terms.

### Gold labels replacing silver spans (critical change)
> **`prepare_gold_labels.py`** was introduced as the primary labeling approach, superseding `prepare_silver_spans.py`.

**Why:** Silver spans used description-level heuristics — they searched for any alias match in the description regardless of which specific triple it belonged to. This misaligns labels with the loss function, which operates per-triple. Gold labels are anchored to individual triples `(h, r, t)`: for each triple, the head and tail entity spans are located in the description using all known aliases, producing per-triple token-level span labels. This makes the loss calculation correct since labels reflect which token spans correspond to a specific relation triple rather than to the entity in general.

**Mechanics:** for each description, iterates over its associated triples; for head entity `h` and tail entity `t`, searches all known aliases using regex (`char_to_token` mapping); records `(token_start, token_end)` per head and per tail per triple; stores as sets to avoid duplicates.

- `EntityExtractor.forward()` updated to also return raw logits alongside probabilities, required for `F.binary_cross_entropy_with_logits` in the loss function.

---

## Mar 20 — Codebase Consolidation & Architecture Finalization
**Commit:** `48e9461`

### File structure refactor
Multiple scattered scripts were merged into a clean, modular structure:

| Deleted | Replaced by |
|---|---|
| `perform_transe.py` | `TransE.py` + `train_transe.py` |
| `embed_relations.py` | `operations/embedding.py` |
| `normalize.py` | `operations/normalizer.py` |
| `minimize.py` | absorbed into `prepare.py` |
| `prepare_embeddings.py` | absorbed into `prepare.py` |
| `prepare_silver_spans.py` | moved to `archive/prepare_silver_spans.py` |

**New files:**
- **`prepare.py`** — unified entry point for all data preparation: minimization (filtering triples/aliases/relations/descriptions), normalization, relation embedding, and description embedding. Merges the former `minimize.py`, `normalize.py`, and `prepare_embeddings.py`.
- **`train_transe.py`** — TransE training loop (extracted from `perform_transe.py`).
- **`TransE.py`** — `TransEDataset` and `TransEModel` classes (L1 margin-based scoring, Xavier-initialized embeddings, per-sample online negative corruption).
- **`models/EntityExtractor.py`** — `EntityExtractor` module extracted from `train.py` into its own file, enabling independent unit testing and staged training. Includes `compute_loss()`: computes masked BCE loss (`F.binary_cross_entropy_with_logits`) over forward head and backward tail predictions per triple. This is the loss for staged training of the entity extractor alone.
- **`operations/embedding.py`** — `get_rel_embs()` (BERT mean-pool per relation over all aliases) and `get_descriptions_embedding()` (per-token and mean-pooled BERT embeddings for descriptions).
- **`operations/normalizer.py`** — `Normalizer` class: regex-based strange character removal, word-level filtering, Unicode NFKC normalization, optional lowercasing.
- **`utils/helpers.py`** — shared utility functions.

### Triples data structure change
Triples are now stored as a flat `list[(h, r, t)]` instead of a `dict{h: [(h, r, t)]}`. This is required because the training pipeline (gold labels, TransE dataset, loss computation) needs to iterate over individual triples rather than entity-grouped collections.

### `prepare_gold_labels.py` updated
Chunk-based multiprocessing using `Pool`; now processes triples directly from the flat list structure; uses `tokenizer.char_to_token()` for precise token-span discovery per triple.

### `train.py` updated
Imports `EntityExtractor` from `models.EntityExtractor`; imports `NUM_WORKERS` from `train_transe`; now outputs raw logits in the return dict for use in `binary_cross_entropy_with_logits`.

---

## Summary of Active vs Future

| Feature | Status |
|---|---|
| TransE training (margin-based, L1, online negative sampling) | ✅ Implemented |
| Silver span labeling | ✅ Archived (superseded) |
| Gold label creation per-triple | ✅ Implemented |
| `EntityExtractor` with BCE loss (staged training) | ✅ Implemented |
| `RelationAttention` (dual: BERT + TransE) | ✅ Implemented |
| `FuseExtractor` (additive fusion `h_ijk`) | ✅ Implemented |
| `BraskModel` full forward pass | ✅ Implemented |
| Full `brask_loss` combining all terms | ⏳ Stub / draft |
| Span reconstruction post-processing | ⏳ Future plan |
| Gating fusion alternative | ⏳ Future plan (`gating = False`) |
| Relation smart pruning (cosine sim, top-k) | ⏳ Future plan |
| Alias-filtered post-prediction | ⏳ Future plan |
| Circular staged training curriculum | ⏳ Future plan |
| PAD token masking in logits | ⏳ Open question |
| `extract_sk` during training (gradient flow) | ⏳ Open question |
