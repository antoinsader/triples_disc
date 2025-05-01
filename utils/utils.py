import pickle
from tqdm import tqdm
from collections import defaultdict
import random
import itertools
import numpy as np

import logging

import torch
import os
import numpy as np 


#region pickles
def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")
    
    
def read_cached_array(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    


def save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path  , arr= tensor.cpu().numpy() )
    print(f"Saved tensor → {path}")


def load_tensor(path, device=None):
    data = np.load(path)["arr"]
    tensor = torch.from_numpy(data)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
#endregion

#region loaders 
def load_descriptions_dict_from_text(fp):
    di = {}
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating descriptions"):
            try:
                entity_id,  description, *rest = line.strip().split("\t")
                di[entity_id] =  description
            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di

def load_aliases_dict_from_text(fp):
    di = {}
    rev_dict = defaultdict(list)
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating aliases"):
            try:
                split_line = line.strip().split("\t")
                entity_id = str(split_line[0])
                entity_name = split_line[1]
                aliases = split_line[2:]
                di[entity_name] = entity_id
                rev_dict[entity_id].append(entity_name)
                for al in aliases:
                    di[al] = entity_id
                    rev_dict[entity_id].append(al)

            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di,rev_dict
def load_relations(file_path):
    relations= {}
    with open(file_path, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating relationships dict"):
            line_parts = line.strip().split("\t")
            relations[line_parts[0]] = line_parts[1:]
    return relations

def load_triples(file_path):
    triples_lookup = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating knowledge graph (triples)"):
            head, relation, tail = line.strip().split('\t')
            triples_lookup[head].append((head, relation, tail))
            
    return triples_lookup
#endregion



#region minimizing and batching 
def minimize_dict(dic, n):
    shuffled_items = list(dic.items())  
    random.shuffle(shuffled_items)
    return dict(itertools.islice(shuffled_items, n))

def batch_dict(dictionary, batch_size):
    items = list(dictionary.items())
    return [
        dict(items[i : i + batch_size])
        for i in range(0, len(items), batch_size)
    ]
    
    

#endregion


#region normalization helpers

def replace_special_chars(text, compiled_patterns):
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text





#endregion


import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def load_descriptions_dict_from_text(fp):
    di = {}
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating descriptions"):
            try:
                entity_id,  description, *rest = line.strip().split("\t")
                di[entity_id] =  description
            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di


def load_triples(file_path):
    triples_lookup = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating knowledge graph (triples)"):
            head, relation, tail = line.strip().split('\t')
            triples_lookup[head].append((head, relation, tail))
            
    return triples_lookup


def load_relations(file_path):
    relations= {}
    with open(file_path, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating relationships dict"):
            line_parts = line.strip().split("\t")
            relations[line_parts[0]] = line_parts[1:]
    return relations


def load_aliases_dict_from_text(fp):
    di = {}
    rev_dict = defaultdict(list)
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating aliases"):
            try:
                split_line = line.strip().split("\t")
                entity_id = str(split_line[0])
                entity_name = split_line[1]
                aliases = split_line[2:]
                di[entity_name] = entity_id
                rev_dict[entity_id].append(entity_name)
                for al in aliases:
                    di[al] = entity_id
                    rev_dict[entity_id].append(al)

            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di,rev_dict





def get_logger(log_name, log_file, append=False):
    logger = logging.getLogger(log_name)  # Unique logger per block
    logger.setLevel(logging.INFO)  # Set logging level

    if logger.hasHandlers():
        logger.handlers.clear()
    
    base, ext = os.path.splitext(log_file)
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if not os.path.exists(candidate):
            chosen_file = candidate
            break
        idx += 1
    mode = 'a' if append else 'w'
    
    handler = logging.FileHandler(chosen_file, mode=mode, encoding='utf-8')
    fmt = logging.Formatter("%(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    
    logger.info(f"Logging started in {chosen_file!r}")
    
    return logger  

