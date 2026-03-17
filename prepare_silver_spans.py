


from collections import defaultdict
from multiprocessing import Pool

import torch
from tqdm import tqdm
import re
from transformers import BertTokenizerFast



from perform_transe import NUM_WORKERS
from utils.files import cache_array
from utils.settings import settings
from utils.chunking import chunk_dict
from utils.pre_processed_data import data_loader


# For multiprocessing
_TOKENIZER =None
_DESCS_HEADS_ALIASES = None
_DESCS_TAILS_ALIASES = None
_ALIASES_PATTERNS_MAP = None


def init_worker(tokenizer: BertTokenizerFast, descriptions_heads_aliases, descriptions_tails_aliases, aliases_patterns_map):
    global _TOKENIZER, _DESCS_HEADS_ALIASES, _DESCS_TAILS_ALIASES, _ALIASES_PATTERNS_MAP
    _TOKENIZER = tokenizer
    _DESCS_HEADS_ALIASES = descriptions_heads_aliases
    _DESCS_TAILS_ALIASES = descriptions_tails_aliases
    _ALIASES_PATTERNS_MAP = aliases_patterns_map


def process_descriptions_chunk(description_chunk: dict, max_descriptions_length: int):
    """Extract head and tail start and end spans for a chunk of descriptions, 
    returns a tuple of (silver_spans_head_start, silver_spans_head_end, silver_spans_tail_start, silver_spans_tail_end, all_sentences_tokens, descriptions_ids)"""

    L = max_descriptions_length

    descriptions_ids = list(description_chunk.keys())
    descriptions_texts = list(description_chunk.values())
    
    chunk_size =  len(descriptions_ids)
    
    silver_spans_head_start = torch.zeros(chunk_size, L)
    silver_spans_head_end = torch.zeros(chunk_size, L)
    silver_spans_tail_start = torch.zeros(chunk_size, L)
    silver_spans_tail_end = torch.zeros(chunk_size, L)

    desc_texts = list(description_chunk.values())
    enc = _TOKENIZER(
        desc_texts,
        return_offsets_mapping=True,
        add_special_tokens = False,
        return_attention_mask=False,
        return_token_type_ids = False,
        padding="max_length",
        truncation=True,
        max_length = L
    )

    offsets = enc["offset_mapping"]
    tokens = [encoding.tokens for encoding in enc.encodings]

    for desc_idx, desc_id in tqdm(enumerate(descriptions_ids), total=len(descriptions_ids), desc="Extracting silver"):
        description = descriptions_texts[desc_idx]
        description_tokens_offsets = offsets[desc_idx]

        description_heads_aliases = _DESCS_HEADS_ALIASES[desc_id]
        description_tails_aliases = _DESCS_TAILS_ALIASES[desc_id]

        for als_str in  description_heads_aliases:
            pattern = _ALIASES_PATTERNS_MAP[als_str]
            m = pattern.search(description)
            if not m: continue
            start_char, end_char = m.span()
            token_indices = [
                i
                for i, (s,e) in enumerate(description_tokens_offsets)
                if (s < end_char and e > start_char)
            ]
            if len(token_indices) > 0:
                head_start, head_end = token_indices[0], token_indices[-1]
                silver_spans_head_start[desc_idx, head_start] = 1
                silver_spans_head_end[desc_idx, head_end] = 1
                break

        for als_str in description_tails_aliases:
            pattern = _ALIASES_PATTERNS_MAP[als_str]
            m = pattern.search(description)
            if not m: continue
            start_char, end_char = m.span()
            token_indices = [
                i
                for i, (s,e) in enumerate(description_tokens_offsets)
                if (s < end_char and e > start_char)
            ]
            if len(token_indices) > 0:
                tail_start, tail_end = token_indices[0], token_indices[-1]
                if silver_spans_tail_start[desc_idx, tail_start] == 0 and silver_spans_tail_end[desc_idx, tail_end] == 0:
                    silver_spans_tail_start[desc_idx, tail_start] = 1
                    silver_spans_tail_end[desc_idx, tail_end] = 1
                break
    return (silver_spans_head_start, silver_spans_head_end, silver_spans_tail_start, silver_spans_tail_end, tokens, descriptions_ids)


def create_aliases_patterns_map(aliases : dict) -> dict[str, re.Pattern]:
    """Creates a map of alias strings to regex patterns that match them."""
    patterns_map = {}
    for als_lst in tqdm(aliases.values(), desc="creating aliases patterns map"):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r'\ ', r'\s*')
            pattern = rf"\b{flexible}\b"
            patterns_map[als_str] = re.compile(pattern, flags=re.IGNORECASE)
    del aliases
    return patterns_map

def create_description_heads_tails_map_aliases(descriptions, triples, aliases):

    heads_to_aliases_map = defaultdict(list)
    tails_to_aliases_map = defaultdict(list)

    for d_id in tqdm(descriptions.keys(), desc="extracting heaads from descriptions"):
        if d_id in triples and d_id in aliases:
            for als in aliases[d_id]:
                heads_to_aliases_map[d_id].append(als)
            for _, _, t in triples[d_id]:
                if t in aliases:
                    for als in aliases[t]:
                        tails_to_aliases_map[d_id].append(als)

    return heads_to_aliases_map, tails_to_aliases_map





def main(use_minimized: bool):
    NUM_WORKERS = 4
    max_descriptions_length = 128
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)


    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    triples_dict = data_loader.get_triples_train(minimized=use_minimized)
    descriptions_heads_aliases, descriptions_tails_aliases = create_description_heads_tails_map_aliases(descriptions, triples_dict, aliases_dict)

    del triples_dict, aliases_dict

    descriptions_chunks = chunk_dict(descriptions, chunks_n=NUM_WORKERS)

    tokenizer :BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-cased')

    print("Starting multiprocessing pool for silver spans extraction...")
    with Pool(processes=NUM_WORKERS, 
              initializer=init_worker, 
              initargs=(tokenizer, descriptions_heads_aliases, descriptions_tails_aliases, aliases_pattern_map) 
              ) as pool:
        # Use starmap to pass multiple arguments to process_descriptions_chunk
        args = [(chunk, max_descriptions_length) for chunk in descriptions_chunks]
        results_chunks = pool.starmap(process_descriptions_chunk, args)
        silver_spans_head_start_ar = []
        silver_spans_head_end_ar = []
        silver_spans_tail_start_ar = []
        silver_spans_tail_end_ar = []
        sentences_tokens = []
        desc_ids = []
        for batch in results_chunks:
            b_ss_h_s, b_ss_h_e, b_ss_t_s, b_ss_t_e, b_tokens, b_desc_ids = batch
            silver_spans_head_start_ar.extend(b_ss_h_s)
            silver_spans_head_end_ar.extend(b_ss_h_e)
            silver_spans_tail_start_ar.extend(b_ss_t_s)
            silver_spans_tail_end_ar.extend(b_ss_t_e)
            sentences_tokens.extend(b_tokens)
            desc_ids.extend(b_desc_ids)

        silver_spans_result = {
            "silver_spans_head_start": torch.stack(silver_spans_head_start_ar, dim=0),
            "silver_spans_head_end": torch.stack(silver_spans_head_end_ar, dim=0),
            "silver_spans_tail_start": torch.stack(silver_spans_tail_start_ar, dim=0),
            "silver_spans_tail_end": torch.stack(silver_spans_tail_end_ar, dim=0),
            "sentences_tokens": sentences_tokens,
            "desc_ids": desc_ids
        }

        out_path = settings.MINIMIZED_FILES.SILVER_SPANS if use_minimized else settings.PREPROCESSED_FILES.SILVER_SPANS
        print(f"Caching silver spans result to {out_path}...")
        cache_array(silver_spans_result, out_path)



if __name__ == "__main__":
    answer = input("Extract silver spans from minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)

