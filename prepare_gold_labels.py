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
from utils.chunking import chunk_dict
from utils.pre_processed_data import data_loader

# For multiprocessing
_TOKENIZER =None
_DESCS_HEADS_ALIASES = None
_DESCS_TAILS_ALIASES = None
_ALIASES_PATTERNS_MAP = None
_ALIASES_DICT = None

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

def chunk_description_discover_aliases_spans(description_chunk: dict, triples_chunk: dict, max_descriptions_length: int):
    """Discover head_aliases, tail_aliases spans for each description 
    Parameters:
    ----------
    description_chunk: dict {entity_id: entity_text}
    triples_chunk: dict {entity_id: list[(head_entity_id, relation_id, tail_entity_id), ..] }

    Returns:
    ----------
    head_spans: dict {entity_id: [(head_start, head_end)... for each head]}
    tail_spans: dict {entity_id: [(tail_start, tail_end)... for each tail]}
    """


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
        max_length = max_descriptions_length
    )
    head_spans = defaultdict(set)
    tail_spans = defaultdict(set)

    for desc_idx, desc_id in tqdm(enumerate(descriptions_ids), total=len(descriptions_ids), desc="Extracting spans"):
        description_text = descriptions_texts[desc_idx]
        if not desc_id in triples_chunk:
            continue
        description_triples = triples_chunk[desc_id]

        for h, _, t in description_triples:
            head_aliases_patterns = [_ALIASES_PATTERNS_MAP[als_str]  for als_str  in  _ALIASES_DICT[h] ]
            tail_alises_patterns = [_ALIASES_PATTERNS_MAP[als_str]  for als_str  in  _ALIASES_DICT[t] ]


            for head_alias_pattern in head_aliases_patterns:
                for match in head_alias_pattern.finditer(description_text):
                    char_start, char_end = match.span()
                    actual_char_end = char_end - 1
                    token_start_idx = enc.char_to_token(desc_idx, char_start)
                    token_end_idx = enc.char_to_token(desc_idx, actual_char_end)
                    if token_start_idx is not None and token_end_idx is not None:
                        head_spans[desc_id].add((token_start_idx, token_end_idx))
            for tail_pattern in tail_alises_patterns:
                for match in tail_pattern.finditer(description_text):
                    char_start, char_end = match.span()
                    actual_char_end = char_end - 1
                    token_start_idx = enc.char_to_token(desc_idx, char_start)
                    token_end_idx = enc.char_to_token(desc_idx, actual_char_end)
                    if token_start_idx is not None and token_end_idx is not None:
                        tail_spans[desc_id].add((token_start_idx, token_end_idx))
    return head_spans, tail_spans


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
    chunks_n = 8 if use_minimized else 16 
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)
    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    triples_dict = data_loader.get_triples_train(minimized=use_minimized)
    descriptions_chunks = chunk_dict(descriptions, chunks_n=NUM_WORKERS)
    del descriptions
    triples_chunks = {}
    

    tokenizer= BertTokenizerFast.from_pretrained('bert-base-cased')


    with Pool(processes=NUM_WORKERS,  initializer=init_worker_discover_aliases, initargs=(tokenizer, aliases_pattern_map, aliases_dict)) as pool:
        args = [(chunk, max_descriptions_length) for chunk in descriptions_chunks]
