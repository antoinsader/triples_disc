
from Data import get_min_descriptionsNorm_triples_relations,  PKLS_FILES
from utils.utils import read_cached_array, cache_array

import torch
from transformers import BertTokenizerFast

from math import ceil
from tqdm import tqdm
import re 

def extract_silver_spans(descs, triples, aliases):
    CHUNK_SIZE= int(1000/20)
    L = DESCRIPTION_MAX_LENGTH = 128
    BATCH_SIZE = len(descs)
    print("\t preparing for extraction")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    sentences_ids = list(descs.keys())
    sentences_texts = list(descs.values())
    
    sentences_triples_heads_aliases = [
        [aliases[t[0]] for t in triples[s]] 
        for s in sentences_ids
    ]
    sentences_triples_tails_aliases = [
        [aliases[t[2]] for t in triples[s]] 
        for s in sentences_ids
    ]

    print("\t doing compiled patterns for aliases ")
        
    alias_pattern_map = {} 
    for lst in tqdm(aliases.values(), total=len(aliases), desc="creating compiled patterns "):
        for alias in lst:
            escaped = re.escape(alias)
            flexible  = escaped.replace(r"\ ", r"\s*")
            pattern   = rf"\b{flexible}\b"
            alias_pattern_map[alias] = re.compile(pattern, re.IGNORECASE)


    print("\t creating empty tensors")
    
    silver_span_head_s = torch.zeros(BATCH_SIZE, L )
    silver_span_head_e = torch.zeros(BATCH_SIZE, L )
    silver_span_tail_s = torch.zeros(BATCH_SIZE, L )
    silver_span_tail_e = torch.zeros(BATCH_SIZE, L )

    all_sentences_tokens = []
    all_sentences_offsets = []

    print("\t starting creating")
    total_batches = ceil(len(sentences_texts) / CHUNK_SIZE)
    for i in tqdm(
        range(0, len(sentences_texts), CHUNK_SIZE),
        total=total_batches,
        desc="Tokenizing batches",
        unit="batch"
    ):

        batch = sentences_texts[i : i + CHUNK_SIZE]
        enc = tokenizer(
            batch, 
            return_offsets_mapping=True,
            add_special_tokens = False,
            padding="max_length", 
            truncation=True,
            max_length=DESCRIPTION_MAX_LENGTH
            
        )
        all_sentences_offsets.extend(enc.offset_mapping)

        for sen_idx, enc_obj in enumerate(enc.encodings):
            all_sentences_tokens.append(enc_obj.tokens)

            sentence_idx_in_batch = i + sen_idx
            current_description = sentences_texts[sentence_idx_in_batch]
            sentence_heads_aliases = sentences_triples_heads_aliases[sentence_idx_in_batch]
            sentence_tails_aliases = sentences_triples_tails_aliases[sentence_idx_in_batch]
            sentence_tokens_offset = all_sentences_offsets[sentence_idx_in_batch]

            for one_als_list in sentence_heads_aliases:
                for als_str in one_als_list:
                    pattern = alias_pattern_map[als_str]
                    m = pattern.search(current_description)
                    if not m: continue 
                    start_char, end_char = m.span()
                    token_indices = [
                        i for i, (s, e) in enumerate(sentence_tokens_offset)
                        if (s < end_char) and (e > start_char)
                    ]
                    if len(token_indices) > 0:
                        head_start, head_end = token_indices[0], token_indices[-1]
                        silver_span_head_s[sentence_idx_in_batch, head_start] = 1
                        silver_span_head_e[sentence_idx_in_batch, head_end] = 1
                        break

            for one_als_list in sentence_tails_aliases:
                for als_str in one_als_list:
                    pattern =  alias_pattern_map[als_str]
                    
                    m = pattern.search(current_description)
                    if not m: continue 
                    start_char, end_char = m.span()
                    token_indices = [
                        i for i, (s, e) in enumerate(sentence_tokens_offset)
                        if (s < end_char) and (e > start_char)
                    ]
                    if len(token_indices) > 0 :
                        tail_start, tail_end = token_indices[0], token_indices[-1]
                        silver_span_tail_s[sentence_idx_in_batch, tail_start] = 1
                        silver_span_tail_e[sentence_idx_in_batch, tail_end] = 1
                        break
    print("\t finsihed ")
    return  silver_span_head_s, silver_span_head_e,  silver_span_tail_s,silver_span_tail_e, all_sentences_tokens



def main_cpu():
    k = 1000

    descs, triples, _, aliases = get_min_descriptionsNorm_triples_relations(k)
    silver_span_head_s, silver_span_head_e,  silver_span_tail_s,silver_span_tail_e, _  = extract_silver_spans(descs, triples, aliases)
    silver_spans = {
        "head_start":silver_span_head_s ,
        "head_end": silver_span_head_e,
        "tail_start": silver_span_tail_s,
        "tail_end":silver_span_tail_e ,
    }
    cache_array(silver_spans, PKLS_FILES["silver_spans"][k])

def test_extract_silver_spans():
    aliases = {
        "q100": ["football", "football soccer"],
        "q500": ["number 9", "num 0"],
        "q700": ["Roland crystal"],
        "q800": ["ML", "machine learning"],
        "q900": ["italy", "italia"],
        "q901": ["europe", "european union"],
        "q1000": ["soccer football player"]
    }
    triples= {
        "q2": [ ("q700", "r2", "q800")],
        "q1": [ ("q100" , "r1", "q500"), ("q100", "r2", "q1000")],
        "q3": [("q900", "r3", "q901")]
    }

    descriptions = {
        "q1": "Raymond Neifel is an indian football soccer player with number 9",
        "q2": "Roland Crystal is the greatest machine learning engineer",
        "q3": "Italia is a country in the european union"
    }
    BATCH_SIZE = len(descriptions)
    sentences_texts = list(descriptions.values())
    silver_span_head_s, silver_span_head_e,  silver_span_tail_s,silver_span_tail_e, all_sentences_tokens  = extract_silver_spans(descriptions, triples, aliases)
    all_heads = []
    all_tails = []
    for sen_idx in range(BATCH_SIZE):
        head_starts = silver_span_head_s[sen_idx]
        head_ends = silver_span_head_e[sen_idx]
        sen_tokens = all_sentences_tokens[sen_idx]
        
        head_starts_idxs = torch.nonzero(head_starts == 1, as_tuple=True)[0]
        head_ends_idxs = torch.nonzero(head_ends == 1, as_tuple=True)[0]
        
        print(f"sentence: " , sentences_texts[sen_idx])
        heads_one = []
        for h_s, h_e in zip(head_starts_idxs, head_ends_idxs):
            heads_one.append(sen_tokens[h_s: h_e + 1])
            print(f"\t HEAD: {sen_tokens[h_s: h_e + 1]}")
            
        all_heads.append(heads_one)            

    for sen_idx in range(BATCH_SIZE):
        tail_starts = silver_span_tail_s[sen_idx]
        tail_ends = silver_span_tail_e[sen_idx]
        sen_tokens = all_sentences_tokens[sen_idx]
        
        tail_starts_idxs = torch.nonzero(tail_starts == 1, as_tuple=True)[0]
        tail_ends_idxs = torch.nonzero(tail_ends == 1, as_tuple=True)[0]
        
        print(f"sentence: " , sentences_texts[sen_idx])
        tails_one = []
        
        for t_s, t_e in zip(tail_starts_idxs, tail_ends_idxs):
            tails_one.append( sen_tokens[t_s: t_e + 1])
            print(f"\t TAIL: {sen_tokens[t_s: t_e + 1]}")
        all_tails.append(tails_one)
    assert [[['football']], [['Roland', 'Crystal']], [['Italia']]] == all_heads
    assert [[['number', '9']], [['machine', 'learning']], [['euro', '##pe', '##an', 'union']]] == all_tails
    print("all good and great ")
    
    
if __name__ == "__main__":
    main_cpu()
    # test_extract_silver_spans()