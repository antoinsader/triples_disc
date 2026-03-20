# so my goal here is to generate gold labels for 
# (forward_head_start, forward_head_end, backward_tail_start, backward_tail_end) each is a tensor with shape (B, L, 1) having 0,1 values
# (forward_tail_start, forward_tail_end, backward_head_start, backward_tail_end) each is a tensor with shape (B, R, S, L, 1) where S is maximum number of heads for forward, maximum number of tails for backward
import torch
import re
from transformers import BertTokenizerFast
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm



from utils.files import cache_array
from utils.settings import settings
from utils.chunking import chunk_dict, chunk_list
from utils.pre_processed_data import data_loader

# For multiprocessing
_TOKENIZER =None
_DESCS_HEADS_ALIASES = None
_DESCS_TAILS_ALIASES = None
_ALIASES_PATTERNS_MAP = None
_ALIASES_DICT = None
_DESCRIPTIONS_DICT = None

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
NUM_WORKERS = 4 if use_cuda else 0



def init_worker_discover_labels(tokenizer: BertTokenizerFast, descriptions_heads_aliases, descriptions_tails_aliases, aliases_patterns_map):
    global _TOKENIZER, _DESCS_HEADS_ALIASES, _DESCS_TAILS_ALIASES, _ALIASES_PATTERNS_MAP
    _TOKENIZER = tokenizer
    _DESCS_HEADS_ALIASES = descriptions_heads_aliases
    _DESCS_TAILS_ALIASES = descriptions_tails_aliases
    _ALIASES_PATTERNS_MAP = aliases_patterns_map

def init_worker_discover_aliases(tokenizer: BertTokenizerFast, aliases_patterns_map, aliases_dict: dict):
    global _TOKENIZER,  _ALIASES_PATTERNS_MAP, _ALIASES_DICT
    _TOKENIZER = tokenizer
    _ALIASES_PATTERNS_MAP = aliases_patterns_map
    _ALIASES_DICT = aliases_dict


def create_aliases_patterns_map(aliases : dict) -> dict[str, re.Pattern]:
    """Creates a map of alias strings to regex patterns that match them."""
    patterns_map = {}
    for als_lst in tqdm(aliases.values(), desc="creating aliases patterns map"):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r'\ ', r'\s+')
            pattern = rf"(?<!\w){flexible}(?!\w)"
            patterns_map[als_str] = re.compile(pattern, flags=re.IGNORECASE)
    del aliases
    return patterns_map

def chunk_description_discover_aliases_spans(process_idx: int,triples_chunk: list,descriptions_chunk: dict, max_descriptions_length: int):
    """Discover head_aliases, tail_aliases spans for each description 
    Parameters:
    ----------
    triples_chunk:  list[(head_entity_id, relation_id, tail_entity_id), ..]

    Returns:
    ----------
    head_spans: dict {entity_id: [(head_start, head_end)... for each head]}
    tail_spans: dict {entity_id: [(tail_start, tail_end)... for each tail]}
    """

    triples_head_ids = set([t[0] for t in triples_chunk ])
    descriptions = {k:v for k,v in descriptions_chunk.items() if k in triples_head_ids}
    descriptions_ids = list(descriptions.keys())
    descriptions_texts = list(descriptions.values())


    print(f"[PROCESS_{process_idx}]: Tokenizing")
    enc = _TOKENIZER(
        descriptions_texts,
        return_offsets_mapping=False,
        add_special_tokens = False,
        return_attention_mask=False,
        return_token_type_ids = False,
        padding="max_length",
        truncation=True,
        max_length = max_descriptions_length
    )
    triple_spans = defaultdict(list)  # h -> [(head_spans, rel_idx, tail_spans), ...]
    print(f"[PROCESS_{process_idx}]: extracting")

    for h,r,t in tqdm(triples_chunk, desc=f"[PROCESS_{process_idx}] Extracting aliases from triples"):
        head_spans_found = set()
        tail_spans_found = set()


        description_idx = descriptions_ids.index(h)
        description_text=descriptions_texts[description_idx]

        head_aliases_patterns = [_ALIASES_PATTERNS_MAP[als_str]  for als_str  in  _ALIASES_DICT[h] ]
        tail_alises_patterns = [_ALIASES_PATTERNS_MAP[als_str]  for als_str  in  _ALIASES_DICT[t] ]
        for head_alias_pattern in head_aliases_patterns:
            for match in head_alias_pattern.finditer(description_text):
                char_start, char_end = match.span()
                actual_char_end = char_end - 1
                token_start_idx = enc.char_to_token(description_idx, char_start)
                token_end_idx = enc.char_to_token(description_idx, actual_char_end)
                if token_start_idx is not None and token_end_idx is not None:
                    head_spans_found.add((token_start_idx, token_end_idx))

        for tail_pattern in tail_alises_patterns:
            for match in tail_pattern.finditer(description_text):
                char_start, char_end = match.span()
                actual_char_end = char_end - 1
                token_start_idx = enc.char_to_token(description_idx, char_start)
                token_end_idx = enc.char_to_token(description_idx, actual_char_end)
                if token_start_idx is not None and token_end_idx is not None:
                    tail_spans_found.add((token_start_idx, token_end_idx))

        for h_spans in head_spans_found:
            for t_spans in tail_spans_found:
                triple_spans[h].append((h_spans, r, t_spans))


    return triple_spans


def chunk_description_discover_golden_spans(description_chunk: dict, max_descriptions_length: int):
    L = max_descriptions_length
    chunk_size =  len(descriptions_ids)

    descriptions_ids = list(description_chunk.keys())
    descriptions_texts = list(description_chunk.values())


    enc = _TOKENIZER(
        descriptions_texts,
        return_offsets_mapping=False,
        add_special_tokens = False,
        return_attention_mask=False,
        return_token_type_ids = False,
        padding="max_length",
        truncation=True,
        max_length = L
    )


    tokens = [encoding.tokens for encoding in enc.encodings]

def main(use_minimized):
    max_descriptions_length = 128
    chunks_n = 16 if use_cuda else 4
    print("loadiung dictionaries...")
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)
    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    triples = data_loader.get_triples_train(minimized=use_minimized)

    print("chunking...")
    triples_chunks = chunk_list(triples, chunks_n=chunks_n)
    descriptions_chunks = []
    for t_chunk in triples_chunks:
        t_chunk_ids = [t[0] for t in t_chunk]
        desc_chunk = {k:v for k, v in descriptions.items if k in t_chunk_ids}
        descriptions_chunks.append(desc_chunk)

    del descriptions , triples

    tokenizer= BertTokenizerFast.from_pretrained('bert-base-cased')

    print(f"Distributing to {chunks_n} processes")
    with Pool(
        processes=chunks_n,  
        initializer=init_worker_discover_aliases, 
        initargs=(tokenizer, aliases_pattern_map, aliases_dict, descriptions)) as pool:

        args = [(idx, t_chunk, d_chunk, max_descriptions_length) for idx, (t_chunk,d_chunk) in enumerate(zip(triples_chunks, descriptions_chunks))]
        results_chunks = pool.starmap(chunk_description_discover_aliases_spans, args)
        results_all  = defaultdict(list)
        for res_chunk in results_chunks:
            for h_id, gold_triple in res_chunk.items():
                results_all[h_id].append(gold_triple)

        cache_array(results_all, settings.MINIMIZED_FILES.GOLD_TRIPLES)

if __name__=="__main__":
    main(True)