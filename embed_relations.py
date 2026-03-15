import os
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from utils.files import save_tensor
from utils.pre_processed_data import data_loader
from utils.settings import settings


device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
BATCH_SIZE = 8192 if device_str == "cuda" else 32


def _get_rel_embs(relations_dict, bert_tokenizer, bert_model):
    """Compute one averaged BERT embedding per relation.

    All aliases for every relation are flattened into a single list and
    processed in chunks of ``BATCH_SIZE``.  Per-alias embeddings are
    accumulated into per-relation sums via ``scatter_add_``, then divided
    by their alias counts to produce the final averages.

    Each alias embedding is the attention-mask mean-pool of the element-wise
    average of BERT's last two hidden layers.

    Args:
        relations_dict (dict[str, list[str]]): Mapping from relation ID to its
            list of text aliases.  Relations with no aliases fall back to the
            relation ID itself.
        bert_tokenizer (BertTokenizerFast): Pre-loaded BERT tokenizer.
        bert_model (BertModel): Pre-loaded BERT model (already on ``device``).

    Returns:
        torch.Tensor: Float32 tensor of shape ``(n_relations, 768)`` where each
            row is the averaged embedding for the corresponding relation.
    """
    rel_ids = list(relations_dict.keys())
    n_rels = len(rel_ids)

    all_aliases = []
    rel_indices = []
    for i, rel_id in enumerate(rel_ids):
        aliases = relations_dict[rel_id]
        if not aliases:
            aliases = [rel_id]
        for alias in aliases:
            all_aliases.append(alias)
            rel_indices.append(i)

    rel_idx_tensor = torch.tensor(rel_indices, dtype=torch.int64, device=device)

    sums = torch.zeros(n_rels, 768, dtype=torch.float32, device=device)
    counts = torch.zeros(n_rels, dtype=torch.float32, device=device)

    if device_str == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu")

    bert_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(all_aliases), BATCH_SIZE), desc="embedding relations"):
            chunk = all_aliases[start:start + BATCH_SIZE]
            rel_idx_chunk = rel_idx_tensor[start:start + BATCH_SIZE]

            encoded = bert_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with autocast_ctx:
                out = bert_model(**encoded, output_hidden_states=True)

            # average last 2 hidden layers
            hidden = torch.stack(out.hidden_states[-2:], dim=0).mean(dim=0)  # (chunk, seq, 768)

            # attention-mask mean-pool over sequence dimension
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # (chunk, seq, 1)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (chunk, 768)
            emb = emb.float()

            sums.scatter_add_(0, rel_idx_chunk.unsqueeze(1).expand_as(emb), emb)
            ones = torch.ones(len(chunk), dtype=torch.float32, device=device)
            counts.scatter_add_(0, rel_idx_chunk, ones)

    return sums / counts.clamp(min=1).unsqueeze(1)


def _check_source_files(minimized: bool) -> bool:
    """Check that the required relations source file exists."""
    path = settings.MINIMIZED_FILES.RELATIONS if minimized else settings.PREPROCESSED_FILES.RELATIONS
    if not os.path.isfile(path):
        label = "minimized" if minimized else "preprocessed"
        print(f"{label.capitalize()} relations file not found: {path}")
        if minimized:
            print("Run minimize.py first.")
        else:
            print("Run the raw data parser first (data_loader.get_relations()).")
        return False
    return True


def embed_relations():
    """Prompts the user to choose between the minimized and full dataset, warns
    about the output file being overwritten, then:
    1. Loads ``bert-base-cased`` (deferred — not at import time).
    2. Loads the relations dict via ``data_loader.get_relations()``.
    3. Calls ``_get_rel_embs()`` to produce a ``(n_relations, 768)`` tensor.
    4. Saves the result with ``save_tensor()`` to the appropriate path
       (``settings.MINIMIZED_FILES.RELATIONS_EMBEDDINGS`` or
       ``settings.PREPROCESSED_FILES.RELATIONS_EMBEDDINGS``).
    """
    answer = input("Embed minimized relations? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'

    if not _check_source_files(use_minimized):
        return

    if use_minimized:
        out_path = settings.MINIMIZED_FILES.RELATIONS_EMBEDDINGS
    else:
        out_path = settings.PREPROCESSED_FILES.RELATIONS_EMBEDDINGS

    print(f"\nWarning: the following file will be overwritten:")
    print(f"  {out_path}")
    confirm = input("Proceed? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    print(f"Loading BERT on {device_str}...")
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    bert_model = BertModel.from_pretrained("bert-base-cased")
    if device_str == "cuda":
        bert_model = bert_model.half()
    bert_model = bert_model.to(device)

    relations = data_loader.get_relations(minimized=use_minimized)
    print(f"Loaded {len(relations):,} relations.")

    rel_embs = _get_rel_embs(relations, bert_tokenizer, bert_model)

    save_tensor(rel_embs, out_path)
    print(f"Shape: {tuple(rel_embs.shape)}  →  {out_path}")


if __name__ == "__main__":
    embed_relations()
