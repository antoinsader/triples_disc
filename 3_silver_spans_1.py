
NUM_WORKERS = 100
L = desc_max_len = 128
USE_CACHE=  False

import pickle
import os
import re
from tqdm import tqdm
import numpy as np 
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import torch
from transformers import BertTokenizerFast





root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
TEMP_FOLDER = f"{root}/data/temp"

DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"


RESULT_FILES  = {
    "descriptions_unormalized": f"{RESULTS_FOLDER}/descriptions_unormalized.pkl",
    "descriptions": f"{RESULTS_FOLDER}/descriptions.pkl",
    "aliases": f"{RESULTS_FOLDER}/aliases.pkl",
    "alias_patterns": f"{RESULTS_FOLDER}/aliases_patterns.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl",
    "silver_spans": {
        "head_start": f"{RESULTS_FOLDER}/ss_head_start.npz",
        "head_end": f"{RESULTS_FOLDER}/ss_head_end.npz",
        "tail_start": f"{RESULTS_FOLDER}/ss_tail_start.npz",
        "tail_end": f"{RESULTS_FOLDER}/ss_tail_end.npz",
        "sentence_tokens": f"{RESULTS_FOLDER}/ss_sentence_tokens.pkl",
        "desc_ids": f"{RESULTS_FOLDER}/desc_ids.pkl",

    }
}
TEMP_FILES = {
    "heads_aliases": f"{TEMP_FOLDER}/heads_aliases.pkl",
    "tails_aliases": f"{TEMP_FOLDER}/tails_aliases.pkl",
    "aliases_patterns": f"{TEMP_FOLDER}/aliases_patterns.pkl",
    "sentences_tokens": f"{TEMP_FOLDER}/sentences_tokens.pkl",
    "results_spans": f"{TEMP_FOLDER}/results_spans.pkl",
}


def save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    np.savez_compressed(path, arr=tensor.cpu().numpy())
    print(f"Tensor chached in file {path}")



def read_tensor(path):
    print(f"reading from path {path}")
    loaded = np.load(path)
    return torch.from_numpy(loaded["arr"])


def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")

def split_list(data, num_chunks):
    chunk_size = (len(data) + num_chunks - 1) // num_chunks  # ceiling division
    return [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]




_shared_heads_aliases = None
_shared_tails_aliases = None
_shared_tokenizer = None
_shared_aliases_pattern_map = None

def init_globals(aliases_pattern_map_, heads_aliases_, tails_aliases_):
    print("intiiating globals")
    global _shared_tokenizer,_shared_aliases_pattern_map,  _shared_heads_aliases, _shared_tails_aliases
    _shared_heads_aliases = heads_aliases_
    _shared_tails_aliases = tails_aliases_
    _shared_aliases_pattern_map = aliases_pattern_map_
    _shared_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    print("finsihed initiating")


def worker(descs_chunk):
    descs_ids_chunk = list(descs_chunk.keys())
    print(f"new cpu worker proccessing {len(descs_ids_chunk)}")



    BATCH_SIZE = len(descs_chunk)
    silver_spans_head_s = torch.zeros(BATCH_SIZE, L)
    silver_spans_head_e = torch.zeros(BATCH_SIZE, L)
    silver_spans_tail_s = torch.zeros(BATCH_SIZE, L)
    silver_spans_tail_e = torch.zeros(BATCH_SIZE, L)

    descs_texts = list(descs_chunk.values())

    enc = _shared_tokenizer(
            descs_texts,
            return_offsets_mapping=True,
            add_special_tokens = False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=L
        )

    all_sentences_offsets = enc.offset_mapping
    all_sentences_tokens = [enc_obj.tokens for enc_obj in enc.encodings ]

    for sen_idx, sen_id in tqdm(enumerate(descs_ids_chunk)) :
        description_text = descs_texts[sen_idx]
        description_tokens_offset = all_sentences_offsets[sen_idx]

        description_heads_aliases = _shared_heads_aliases[sen_id]
        description_tails_aliases = _shared_tails_aliases[sen_id]


        for head_als_list in description_heads_aliases:
            for als_str in head_als_list:
                pattern = _shared_aliases_pattern_map[als_str]
                m = pattern.search(description_text)
                if not m: continue
                start_char, end_char = m.span()
                token_indices = [
                    i for i, (s,e) in enumerate(description_tokens_offset)
                    if (s < end_char) and (e > start_char)
                ]
                if len(token_indices) > 0:
                    head_start, head_end = token_indices[0], token_indices[-1]
                    silver_spans_head_s[sen_idx, head_start] = 1
                    silver_spans_head_e[sen_idx, head_end] = 1
                    break

        for tail_als_list in description_tails_aliases:
            for als_str in tail_als_list:
                pattern =  _shared_aliases_pattern_map[als_str]
                m = pattern.search(description_text)
                if not m: continue
                start_char, end_char = m.span()
                token_indices = [
                    i for i, (s,e) in enumerate(description_tokens_offset)
                    if (s < end_char) and (e > start_char)
                ]
                if len(token_indices) > 0:
                    tail_start, tail_end = token_indices[0], token_indices[-1]
                    if silver_spans_tail_s[sen_idx, tail_start] == 0 and silver_spans_tail_s[sen_idx, tail_end] == 0:
                        silver_spans_tail_s[sen_idx, tail_start] = 1
                        silver_spans_tail_e[sen_idx, tail_end] = 1
                    break
    print("finished")
    return  silver_spans_head_s, silver_spans_head_e, silver_spans_tail_s, silver_spans_tail_e,all_sentences_tokens, descs_ids_chunk

def get_tailsHeadsAliases_aliasesPatternMap():
    if USE_CACHE and  os.path.exists(TEMP_FILES["results_spans"]):
        results = read_cached_array(TEMP_FILES["results_spans"])
    else:
        print("reading dicts")
        full_descs_dict = read_cached_array(RESULT_FILES["descriptions"])
        aliases_patterns_map = read_cached_array(RESULT_FILES["alias_patterns"])
        heads_aliases = read_cached_array(TEMP_FILES["heads_aliases"])
        tails_aliases = read_cached_array(TEMP_FILES["tails_aliases"])

        print("splitting chunks")
        full_descs_ids = list(full_descs_dict.keys())
        chunks = split_list(full_descs_ids, NUM_WORKERS)
        print("chunks have been splitted")
        descriptions_chunks = [{k: full_descs_dict[k] for k in chunk} for chunk in chunks]
        print("splitting into processes")
        with Pool(processes=NUM_WORKERS, initializer=init_globals, initargs=( aliases_patterns_map, heads_aliases, tails_aliases)) as pool:
            results = pool.map(worker, descriptions_chunks)
        cache_array(results, TEMP_FILES["results_spans"])

    silver_spans_head_start_ar = []
    silver_spans_head_end_ar = []
    silver_spans_tail_start_ar = []
    silver_spans_tail_end_ar = []
    sentences_tokens = []
    desc_ids = []
    for g in results:
        silver_spans_head_start_ar.extend(g[0])
        silver_spans_head_end_ar.extend(g[1])
        silver_spans_tail_start_ar.extend(g[2])
        silver_spans_tail_end_ar.extend(g[3])
        sentences_tokens.extend(g[4])
        desc_ids.extend(g[5])
    silver_spans_head_start = torch.stack(silver_spans_head_start_ar, dim=0)
    silver_spans_head_end = torch.stack(silver_spans_head_end_ar, dim=0)
    silver_spans_tail_start = torch.stack(silver_spans_tail_start_ar, dim=0)
    silver_spans_tail_end = torch.stack(silver_spans_tail_end_ar, dim=0)

    save_tensor(silver_spans_head_start, RESULT_FILES["silver_spans"]["head_start"])
    save_tensor(silver_spans_head_end, RESULT_FILES["silver_spans"]["head_end"])
    save_tensor(silver_spans_tail_start, RESULT_FILES["silver_spans"]["tail_start"])
    save_tensor(silver_spans_tail_end, RESULT_FILES["silver_spans"]["tail_end"])
    cache_array(sentences_tokens, RESULT_FILES["silver_spans"]["sentence_tokens"])
    cache_array(desc_ids, RESULT_FILES["silver_spans"]["desc_ids"])
    return True

if __name__ == "__main__":
    get_tailsHeadsAliases_aliasesPatternMap()








    # aliases = {
    #     "q1": ["que 1", "qqqq 1"],
    #     "q2": ["q two", "que 2", "qqqq 2"],
    #     "q3": ["que 3", "qqqq 3"],
    #     "q4": [ "qqqq 4"],
    #     "q5": ["que 5", "qqqq 5"],
    #     "q6": ["q six", "qqqq 6"],
    #     "q7": ["q svn", "que 7", ],
    #     "q8": ["q right", "que 8", ],
    #     "q9": ["q nine", "que 9", ],
    #     "q10": ["q ten", "que 10", ],
    # }

    # triples  = {
    #     "q1": [("q1", "r1", "q2"), ("q1", "r2", "q5")],
    #     "q2": [("q2", "r1", "q3"), ("q2", "r2", "q6")],
    #     "q3": [("q3", "r1", "q4"), ("q3", "r2", "q7")],
    #     "q4": [("q4", "r1", "q5"), ("q4", "r2", "q8")],
    #     "q5": [("q5", "r1", "q6"), ("q5", "r2", "q9")],
    #     "q6": [("q6", "r1", "q7"), ("q6", "r2", "q10")],
    #     "q7": [("q7", "r1", "q8"), ],
    #     "q8": [("q8", "r1", "q9"), ],
    #     "q9": [("q9", "r1", "q10"), ],
    #     "q10": [("q10", "r1", "q1"),],
    # }


    # full_descs_dict = {
    #     "q1": "I am q1",
    #     "q2": "I am q2",
    #     "q3": "I am q3",
    #     "q4": "I am q4",
    #     "q5": "I am q5",
    #     "q6": "I am q6",
    #     "q7": "I am q7",
    #     "q8": "I am q8",
    #     "q9": "I am q9",
    #     "q10": "I am q10",
    # }


def test_silver_span(desc_id="Q854"):
    print(f"desc_id: {desc_id}")
    
    triples  = read_cached_array(RESULT_FILES["triples"])
    aliases  = read_cached_array(RESULT_FILES["aliases"])
    descriptions  = read_cached_array(RESULT_FILES["descriptions"])
    # alias_pattern_map = read_cached_array(RESULT_FILES["alias_patterns"])
    # relations  = read_cached_array(RESULT_FILES["relations"])
    silver_spans_head_start = read_tensor( RESULT_FILES["silver_spans"]["head_start"])
    silver_spans_head_end = read_tensor( RESULT_FILES["silver_spans"]["head_end"])
    silver_spans_tail_start = read_tensor( RESULT_FILES["silver_spans"]["tail_start"])
    silver_spans_tail_end = read_tensor( RESULT_FILES["silver_spans"]["tail_end"])
    sentences_tokens = read_cached_array( RESULT_FILES["silver_spans"]["sentence_tokens"])
    desc_ids = read_cached_array( RESULT_FILES["silver_spans"]["desc_ids"])
    sen_idx = desc_ids.index(desc_id)
    print(f"Sentence: {descriptions[desc_id]}")

    sentence = sentences_tokens[sen_idx]
    print(f"tokens: {sentence}")


    sen_triples = triples[desc_id]
    print("TRIPLES: ")
    for h,_,t in sen_triples:
        print("\t triple: ")
        print(f"\t\t HEAD: {aliases[h]}")
        print(f"\t\t TAIL: {aliases[t]}")


    hs = silver_spans_head_start[sen_idx]
    h_s_idxs = torch.nonzero(hs > 0, as_tuple=True)[0]

    he = silver_spans_head_end[sen_idx]
    h_e_idxs = torch.nonzero(he > 0, as_tuple=True)[0]

    print(f"h_s_idxs:  {h_s_idxs}")
    print(f"h_e_idxs:  {h_e_idxs}")
    heads = []
    used_ends = set()
    for h_s_id in h_s_idxs:
        head_start_idx = h_s_id.item()
        h_e_id = h_e_idxs[0].item()
        e_idx = 0
        while e_idx >= len(h_e_idxs) or h_e_id < head_start_idx or h_e_id in used_ends: 
            e_idx += 1
            h_e_id = h_e_idxs[e_idx]
        used_ends.add(h_e_id)
        head = (head_start_idx, h_e_id + 1)
        print(f"HEAD: {sentence[head[0] : head[1]]}")
        heads.append(head )

    ts = silver_spans_tail_start[sen_idx]
    t_s_idxs = torch.nonzero(ts > 0, as_tuple=True)[0]

    te = silver_spans_tail_end[sen_idx]
    t_e_idxs = torch.nonzero(te > 0, as_tuple=True)[0]
    print(t_s_idxs)
    tails = []
    used_ends = set()
    for t_s_id in t_s_idxs:
        tail_start_idx = t_s_id.item()
        t_e_id = t_e_idxs[0].item()
        e_idx = 0
        while e_idx >= len(t_e_idxs) or t_e_id < tail_start_idx or t_e_id in used_ends: 
            e_idx += 1
            t_e_id = t_e_idxs[e_idx]
        used_ends.add(t_e_id)
        tail = (tail_start_idx, t_e_id + 1)
        print(f"Tail: {sentence[tail[0] : tail[1]]}")
        tails.append(tail )

