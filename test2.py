from Data import get_min_descriptionsNorm_triples_relations, LOGGER_FILES, PKLS_FILES
from utils.utils import read_cached_array, cache_array, get_logger
from transformers import BertTokenizerFast
import torch

DESCRIPTION_MAX_LENGTH = 128

def build_alias_id_tensors(aliases, tokenizer, device):
    """
    Flatten your alias dict into a mapping from (entity, alias_string) 
    to a 1D torch tensor of token-IDs on the GPU.
    """
    alias_id_tensors = {}
    for ent, alias_list in aliases.items():
        for alias in alias_list:
            toks = tokenizer.tokenize(alias)
            if not toks:
                continue
            ids = tokenizer.convert_tokens_to_ids(toks)
            alias_id_tensors[(ent, alias)] = torch.tensor(ids, device=device, dtype=torch.long)
    return alias_id_tensors

def get_entity_spans_batch(sent_ids, alias_id_tensors):
    """
    sent_ids: [batch_size, seq_len] of token IDs on GPU
    alias_id_tensors: dict mapping (entity,alias) -> [m] ID tensor on GPU
    Returns: dict mapping (batch_idx, entity, alias) -> list of start_idxs
    """
    batch_size, seq_len = sent_ids.shape
    spans = {}  # (b, entity, alias) -> [start positions]

    # Pre‐compute all unfolded windows for this batch only once per possible window‐size
    # We'll group aliases by length to avoid repeated unfold calls.
    by_length = {}
    for (ent, alias), id_tensor in alias_id_tensors.items():
        m = id_tensor.size(0)
        if m > seq_len:
            continue
        by_length.setdefault(m, []).append(((ent, alias), id_tensor))

    for m, alias_entries in by_length.items():
        # unfold: [batch, seq_len-m+1, m]
        windows = sent_ids.unfold(1, size=m, step=1)  
        # now for each alias of length m, compare all windows in one shot
        for (ent, alias), alias_ids in alias_entries:
            # broadcast compare: [batch, seq_len-m+1, m] == [m] -> [batch, seq_len-m+1, m]
            eq = windows == alias_ids.view(1,1,m)
            # reduce over the m‐dimension: True where all tokens match
            match_mask = eq.all(dim=2)  # [batch, seq_len-m+1]
            # find all indices
            for b in range(batch_size):
                idxs = torch.nonzero(match_mask[b], as_tuple=False).view(-1).cpu().tolist()
                if idxs:
                    spans.setdefault((b, ent, alias), []).extend(idxs)

    return spans

def create_silver_spans(descs, triples, relations, aliases,
                        buid_logger, golden_triples_file, silver_spans_file,
                        bert_tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = buid_logger

    # 1) tokenize all sentences once, send them to the GPU
    sentences = list(descs.values())
    encoded = bert_tokenizer(
        sentences,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=DESCRIPTION_MAX_LENGTH,
    )
    sent_ids = encoded["input_ids"].to(device)  # [batch, seq_len]
    batch_size, seq_len = sent_ids.shape

    # 2) build alias‐ID tensors for heads and tails
    logger.info("Building alias ID tensors on %s", device)
    head_alias_tensors = build_alias_id_tensors(aliases, bert_tokenizer, device)
    tail_alias_tensors = build_alias_id_tensors(aliases, bert_tokenizer, device)

    # 3) run batch‐span‐extraction
    head_spans = get_entity_spans_batch(sent_ids, head_alias_tensors)
    tail_spans = get_entity_spans_batch(sent_ids, tail_alias_tensors)

    # 4) allocate GPU tensors for silver spans
    silver_sub_s = torch.zeros(batch_size, seq_len, device=device)
    silver_sub_e = torch.zeros(batch_size, seq_len, device=device)
    silver_obj_s = torch.zeros(batch_size, seq_len, device=device)
    silver_obj_e = torch.zeros(batch_size, seq_len, device=device)

    golden_triples = {}
    logger.info(f"Processing {batch_size} sentences for golden triples")
    for b, sen_id in enumerate(descs.keys()):
        sen_trps = []
        for h, r, t in triples[sen_id]:
            # gather all alias matches for this head entity
            for alias in aliases[h]:
                for start in head_spans.get((b, h, alias), []):
                    end = start + head_alias_tensors[(h, alias)].size(0) - 1
                    # now tails
                    for tail_alias in aliases[t]:
                        for tstart in tail_spans.get((b, t, tail_alias), []):
                            tend = tstart + tail_alias_tensors[(t, tail_alias)].size(0) - 1

                            # mark silver spans
                            silver_sub_s[b, start] = 1
                            silver_sub_e[b, end]   = 1
                            silver_obj_s[b, tstart] = 1
                            silver_obj_e[b, tend]    = 1

                            sen_trps.append(((start, end), (tstart, tend)))
                            logger.info(
                                "Found TRIPLE: head='%s' [%d,%d], tail='%s' [%d,%d], rel_aliases=%s",
                                alias, start, end, tail_alias, tstart, tend, relations[r]
                            )

        golden_triples[sen_id] = {
            "sentence_tokens": bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][b]),
            "triples": sen_trps
        }

    # move silver spans back to CPU before caching
    silver_spans = {
        "head_start": silver_sub_s.cpu(),
        "head_end":   silver_sub_e.cpu(),
        "tail_start": silver_obj_s.cpu(),
        "tail_end":   silver_obj_e.cpu(),
    }

    cache_array(silver_spans, "./draft_silver_spans_new.pkl")
    cache_array(golden_triples, "./draft_golden_triples_new.pkl")
    
    return silver_spans




if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    k = 100
    descs, triples, relations, aliases = get_min_descriptionsNorm_triples_relations(k)
    golden_triples_file = PKLS_FILES["golden_triples"][k]
    silver_spans_file = PKLS_FILES["silver_spans"][k]
    buid_logger = get_logger("build_golden_triples", LOGGER_FILES["build_golden_triples"])
    
    create_silver_spans(
                        descs = descs, triples = triples, relations= relations, aliases= aliases,
                        buid_logger=buid_logger, golden_triples_file=golden_triples_file, silver_spans_file = silver_spans_file,
                        bert_tokenizer = tokenizer
                        
                        )
    

import pickle
def read_cached_array(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

silver_spans_file = PKLS_FILES["silver_spans"]["full"]
silver_spans = read_cached_array(silver_spans_file)
silver_spans.shape


