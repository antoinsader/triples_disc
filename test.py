
from math import ceil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import numpy as np 
from tqdm import tqdm
import pickle
import torch.multiprocessing as mp
import random
import os
from multiprocessing import Pool, cpu_count



from collections import defaultdict
import itertools
import random 
import logging

from torch.amp import autocast, GradScaler
import re 
from joblib import Parallel, delayed
from joblib import dump, load

import concurrent.futures
import multiprocessing
from functools import partial
import unicodedata

import nltk
from nltk.corpus import words

import torch 
import os 
from transformers import BertTokenizerFast


root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
HELPERS_FOLDER = f"{root}/data/helpers"
CHECKPOINT_FOLDER = f"{root}/data/checkpoints"

RESULT_FILES  = {
    "descriptions_unormalized": f"{RESULTS_FOLDER}/descriptions_unormalized.pkl",
    "descriptions": f"{RESULTS_FOLDER}/descriptions.pkl",
    "aliases": f"{RESULTS_FOLDER}/aliases.pkl",
    "alias_patterns": f"{RESULTS_FOLDER}/aliases_patterns.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl"
}

CHECKPOINTS_FILES = {
    "transe_triples": f"{CHECKPOINT_FOLDER}/transe_triples.pkl",
    "transe_model": f"{CHECKPOINT_FOLDER}/transe_model.pth",
}


def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")

def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def split_list(data, num_chunks):
    chunk_size = (len(data) + num_chunks - 1) // num_chunks  # ceiling division
    return [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]




def create_alias_patterns_map(aliases_all_dict):
    alias_patterns_map = {}
    for als_lst in tqdm(aliases_all_dict.values()):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r"\ ", r"\s*")
            pattern = rf"\b{flexible}\b"
            alias_patterns_map[als_str] = re.compile(pattern, re.IGNORECASE)
    return alias_patterns_map

als_lists = {
    "q1": ["tony", "sader", "rami"],
    "q2": ["samer", "daad", "shakib"],
    "q3": ["george", "gaya", "sisi"],
    "q4": ["widad", "ilias"],
    "q5": ["bassel", "ramzi"],
    "q6": ["louisa", "lena"],
    "q7": ["mirna", "riad"],
    "q8": ["kamel", "joelle"],
}
triples = {
    "q1": [("q1", "r1", "q5"), ("q1", "r1", "q6"), ],
    "q2": [("q2", "r1", "q4"), ("q2", "r1", "q7"), ],
    "q3": [("q3", "r1", "q2"), ("q3", "r1", "q6"), ],
    "q4": [("q4", "r1", "q8"), ("q4", "r1", "q3"), ],
    "q5": [("q5", "r1", "q1"), ("q5", "r1", "q4"), ],
    "q6": [("q6", "r1", "q1"), ("q6", "r1", "q2"), ],
    "q7": [("q7", "r1", "q3"), ("q7", "r1", "q5"), ],
    "q8": [("q8", "r1", "q8"), ("q8", "r1", "q2"), ],
    
}

descs_lists = {
        "q1": "I bassel am 1 lena",
        "q2": "I ilias am riad 2",
        "q3": "I  daad am 3 louisa",
        "q4": "joelle I am 4 gaya",

        "q5": "widad I am 5 tony",
        "q6": "I samer am sader 6 ",
        "q7": "I sisi am ramzi 7",
        "q8": "I kamel am shakib 8",

}

alias_pattern_map = create_alias_patterns_map(als_lists)



_shared_aliases = None
_shared_triples = None
_shared_tokenizer = None
_shared_aliases_pattern_map = None

def init_globals(aliases_, triples_,aliases_pattern_map_):
    print("intiiating globals")
    global _shared_aliases, _shared_triples,_shared_tokenizer,_shared_aliases_pattern_map
    _shared_aliases = aliases_
    _shared_aliases_pattern_map = aliases_pattern_map_
    _shared_triples = triples_
    _shared_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    print("finsihed initiating")

def worker(descs_chunk):
    descs_ids_chunk = list(descs_chunk.keys())
    print(f"new cpu worker proccessing {len(descs_ids_chunk)}")
    triples_keys = list(triples.keys())

    tails_aliases = {}
    heads_aliases = {}

    for d_id in tqdm(descs_ids_chunk, total=len(descs_ids_chunk), desc="processing descs ids"):
        heads_aliases.setdefault(d_id, []).append(als_lists[d_id])
        if d_id in triples_keys:
            for _, _, t in triples[d_id]:
                tails_aliases.setdefault(d_id, []).append(als_lists[t])

    BATCH_SIZE = len(descs_chunk)
    L = 10
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

    for sen_idx, sen_id in tqdm(enumerate(descs_ids_chunk)):
        description_text = descs_texts[sen_idx]
        description_tokens_offset = all_sentences_offsets[sen_idx]

        description_heads_aliases = heads_aliases[sen_id]
        description_tails_aliases = tails_aliases[sen_id]


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
                    silver_spans_head_s[sen_idx, tail_start] = 1
                    silver_spans_head_e[sen_idx, tail_end] = 1
                    break
    print("finished")
    return  silver_spans_head_s, silver_spans_head_e, silver_spans_tail_s, silver_spans_tail_e,all_sentences_tokens

if __name__ == "__main__":
    full_descs_ids = list(descs_lists.keys())
    chunks = split_list(full_descs_ids, 2)
    descriptions_chunks = [{k: descs_lists[k] for k in chunk} for chunk in chunks]
    
    with Pool(processes=2, initializer=init_globals, initargs=(als_lists, triples, alias_pattern_map)) as pool:
        results = pool.map(worker, descriptions_chunks)

    cache_array(results, "./data/temp/zz.pkl")
    