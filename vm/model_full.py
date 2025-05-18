



# pip install numpy  tqdm pandas
# pip install torch transformers
# pip install scikit-learn  nltk datasets psutil




from math import ceil
import torch
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
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import torch.multiprocessing as mp
import random
import os


from collections import defaultdict
import itertools

import logging

from torch.amp import autocast, GradScaler
import re 
from joblib import Parallel, delayed


import concurrent.futures
import multiprocessing
from functools import partial
import unicodedata

import nltk
from nltk.corpus import words
# nltk.download('words') 









#region utils
def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")
    
    
def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

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
    
    




def replace_special_chars(text, compiled_patterns):
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text







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



import torch
from torch.nn.utils.rnn import pad_sequence

def get_embs(data, model, batch_size, temp_save_dir="./temp_embs"):
    """
    Consider this as part 1 from get_embs, we take string data and we extract last hidden state to get embeddings
    """
    print("getting embs")
    os.makedirs(temp_save_dir, exist_ok=True)
    
    
    
    
    embs = torch.empty((batch_size, DESCRIPTION_MAX_LENGTH, 768), dtype=torch.float16)
    masks = torch.empty((batch_size, DESCRIPTION_MAX_LENGTH), dtype=torch.bool)

    idx = 0
    file_idx = 0
    
    

    with torch.inference_mode():
        for batch_ids, batch_mask in tqdm(data,total=len(data), unit="batch", desc="embeding batches", dynamic_ncols=True):
            batch_ids   = batch_ids.to(device, non_blocking=True)
            batch_mask  = batch_mask.to(device, non_blocking=True)
            out = model(input_ids=batch_ids, attention_mask=batch_mask)
            last_hidden = out.last_hidden_state.cpu()          # (bs, seq_len, hidden)
            bs = last_hidden.size(0)
            # print(f"bs: {bs}")
            embs[idx:idx+bs] = last_hidden
            masks[idx:idx+bs] = batch_mask.cpu().to(torch.bool)
            idx += bs
            
            
            # free GPU memory from this batch
            
            
            del batch_ids, batch_mask, out, last_hidden
            torch.cuda.empty_cache()
    
    
    
    return embs, masks
    
    

    

def get_h_gs(sentences, tokenizer, model, max_length, device):
    print(f"\ttokenizing ")
    # dataset = HFDataset.from_dict({"text": sentences})
    # def tokenize_function(batch):
    #     return tokenizer(
    #         batch["text"],
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #         max_length=max_length
    #     )
    # encoded = dataset.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=NUM_WORKERS
    # )
    # print("finished the tokenize function")
    # cache_array(encoded, PKLS_FILES["encoded_tokenization"])


    encoded = read_cached_array( PKLS_FILES["encoded_tokenization"])  # liast len = batch_size, len(0) = seq_len
    print(f"\tafter tokenizing ")

    encoded.set_format(type="torch", columns=["input_ids", "attention_mask"])
    ds = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    loader = TorchDataLoader(ds, 
                             batch_size= 512, 
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                            pin_memory=True,           
                            persistent_workers=True, 
                             )
    print("finished loading")

    model = model.to(device).half()
    model.eval()
    model = torch.compile(model)
    
    # get token_embs
    embs, masks = get_embs(loader, model, len(sentences) ) #tensor with shape (B, seq_len, hidden_size) # so for every token, in every sentence, hidden_size vector; masks shape is (batch_size, seq_len)
    torch.save((embs, masks), PKLS_FILES["h_g_s_all_embs"])
    
    print("saved embs, now creating h_gs")    
    
    
    #get h_gs
    #split across 4 gpus to get the result
    devices = ["cuda:0", "cuda:1","cuda:3", "cuda:4" ]
    chunks = len(devices)
    emb_chunks = torch.chunk(embs, chunks)
    mask_chunks = torch.chunk(masks, chunks)
    print("chunk finished")
    mean_embed_parts = []
    for i, (e_chunk, m_chunk) in tqdm(enumerate(zip(emb_chunks, mask_chunks)), total=len(devices)):
        e_chunk = e_chunk.to(devices[i])
        m_chunk = m_chunk.to(devices[i])
        
        m_exp = m_chunk.unsqueeze(-1).float()
        summed = (e_chunk * m_exp).sum(dim=1)
        counts = m_exp.sum(dim=1).clamp(min=1)
        means = summed / counts

        mean_embed_parts.append(means.cpu())
    
    mean_embeds = torch.cat(mean_embed_parts, dim=0)
    
    torch.save(mean_embeds, PKLS_FILES["h_g_s_all_means"])

    return mean_embeds, embs



def extract_first_embeddings(token_embs, start_probs, end_probs, max_len, threshold=.6):
    """
        args:
            token_embs: sentence embeddings with shape (batch_size, seq_length, hidden_size)
            start_probs: probabilities that an entity is a start (start of subject for forward, start of object for backward) with shape (batch_size, seq_len)
            end_probs: probabilities that an entity is an end (end of subject for forward, end of object for backward) with shape (batch_size, seq_len)
            threshold: threshold that if the probability > threshold, then it is considered start or end of starting entity
        returns:
            padded_embs: these are padded embeddings of all subjects or objects with shape (B, max_ents, H)
            mask_embs: mask to show the padded embs (0 for padding)
            start_idxs:  list (len=batch_size) where each item is a list of tuples, each tuple (start_idx, end_idx) of the entities
    """
    batch_size, seq_len, hidden_size = token_embs.shape

    start_mask = start_probs > threshold
    end_mask = end_probs > threshold

    start_idxs = []
    all_ents_embs = []
    all_masks = []

    for sent_idx in range(batch_size):
        start_indices = torch.nonzero(start_mask[sent_idx], as_tuple=False).squeeze(-1)
        end_indices = torch.nonzero(end_mask[sent_idx], as_tuple=False).squeeze(-1)

        ents_embs = []
        idxs_sentence = []
        used_ends = set()

        for start_idx in start_indices.tolist():
            candidates = [end_idx for end_idx in end_indices.tolist() if end_idx >= start_idx and end_idx not in used_ends]
            if candidates:
                end_idx = candidates[0]
                used_ends.add(end_idx)
                idxs_sentence.append((start_idx, end_idx))

                # Compute average embedding
                ent_emb = (token_embs[sent_idx, start_idx] + token_embs[sent_idx, end_idx]) / 2.0
                ents_embs.append(ent_emb)

        start_idxs.append(idxs_sentence)

        if ents_embs:
            ent_tensor = torch.stack(ents_embs)
            all_ents_embs.append(ent_tensor)
            all_masks.append(torch.ones(ent_tensor.size(0), dtype=torch.bool, device=token_embs.device))
        else:
            all_ents_embs.append(torch.empty(0, hidden_size, device=token_embs.device))
            all_masks.append(torch.zeros(0, dtype=torch.bool, device=token_embs.device))

    # Pad sequences
    padded_embs = pad_sequence(all_ents_embs, batch_first=True, padding_value=0.0)
    mask_embs = pad_sequence(all_masks, batch_first=True, padding_value=False)

    if max_len is not None:
        # Pad or truncate embeddings
        curr_len = padded_embs.size(1)
        if curr_len < max_len:
            pad_size = max_len - curr_len
            padding = torch.zeros(padded_embs.size(0), pad_size, padded_embs.size(2), device=padded_embs.device)
            padded_embs = torch.cat([padded_embs, padding], dim=1)

            mask_padding = torch.zeros(mask_embs.size(0), pad_size, dtype=torch.bool, device=mask_embs.device)
            mask_embs = torch.cat([mask_embs, mask_padding], dim=1)

        elif curr_len > max_len:
            padded_embs = padded_embs[:, :max_len, :]
            mask_embs = mask_embs[:, :max_len]

    return padded_embs, mask_embs, start_idxs

def extract_last_idxs(start_probs, end_probs, threshold):
    """
      args:
        start_probs, end_probs are vectors with length of L, those are for one sentence 
      returns:
        idxs: a list where each item contains (start_idx, end_idx) of entities in the sentence
    """
    start_idxs = torch.nonzero(start_probs > threshold).squeeze()
    end_idxs = torch.nonzero(end_probs > threshold).squeeze()
    
    if start_idxs.ndim == 0:
        start_idxs = start_idxs.unsqueeze(0)
    if end_idxs.ndim == 0:
        end_idxs = end_idxs.unsqueeze(0)

    idxs = []
    used_ends = set()
    
    for start_idx in start_idxs.tolist():
        for end_idx in end_idxs.tolist():
            if end_idx >= start_idx and end_idx not in used_ends:
                idxs.append((start_idx, end_idx))
                used_ends.add(end_idx)
                break
            
    return idxs 
  


def extract_triples(first_idxs, start_probs, end_probs,is_forward, threshold=.6):
    """
        args:
            first_idxs list with len = B, each one is list of indexes for the starting entities. List sentence-1[(start_ent_start_idx, start_ent_end_idx ), ...]
            start_probs, end_probs with shape (B, L, R)
        does:
          loop through sentences in the batch
          for each relationship: find objects that has high probability to make (rel, obj)
          assign those to each subject to have (sub, rel, obj)
          (I am saying sub,rel,obj but in backward it would be obj,rel,sub)
        
        returns:
          batch_triples: which is a list with size of B and each item is a list having the triples
          

    """
    batch_size, seq_len, num_rels = start_probs.shape
    batch_triples  = []
    
    for b in range(batch_size):
        sentence_triples = []
        sentence_first_idxs = first_idxs[b]
        
        for rel in range(num_rels):
            last_start_vec = start_probs[b, :, rel]
            last_end_vec = end_probs[b, :, rel]
            last_idxs = extract_last_idxs(last_start_vec, last_end_vec, threshold)
            
            for first_idx in sentence_first_idxs:
                for last_idx in last_idxs:
                    # if is forward:
                    #     subject_end_idx should be < obj_start_idx 
                    # if backward:
                    #     obj_start_idx > subject_end_idx
                    if (is_forward and first_idx[1] < last_idx[0]) or (not is_forward and first_idx[0] > last_idx[1]):
                      sentence_triples.append((first_idx,rel ,last_idx))
        batch_triples.append(sentence_triples)
    
    return batch_triples    
        


def merge_triples(forward_triples, backward_triples):
  """
    args:
      forwarD_triples: List (len=batch_size ? ) of items, each item is a triple (head_idx, rel_idx, obj_idx)  
      backward_triples: List (len=batch_size ? ) of items, each item is a triple (head_idx, rel_idx, obj_idx)  
    returns:
      list of triples (length=batch_size) and each item is  a triple (h,r,t)  
    
  """
  final_triples = []
  for f_triples, b_triples in zip(forward_triples, backward_triples):
    final = list(set(f_triples).intersection(set(b_triples)))
    final_triples.append(final)
  return final_triples





#endregion





#region load_data


def create_descriptions_dict():
    """"
        does:
            - Create description dict from raw text file and save resuls in pickle file 
            - the dict  {document_id: description_text}
    """
    
    desc_raw_f = RAW_TXT_FILES["descriptions"]
    descriptions_dict_f = PKLS_FILES["descriptions"]["full"]
    print(f"Full description is being created and will be saved in {descriptions_dict_f}...")
    descriptions_dict = load_descriptions_dict_from_text(desc_raw_f)
    cache_array(descriptions_dict, descriptions_dict_f)
    
    return descriptions_dict

# 
#    aliases_dict would have {alias_name: document_id}
#    aliases_dict_rev would have {document_id: [alias_name1, alias_name2, ...]}
def create_aliases_dicts():
    """
        does:
            - will create two aliases dicts from the raw text file of aliases and save them in pickle files
            - aliases_dict would have {alias_text: document_id}
            - aliases_dict_rev would have {document_id: [alias_1, alias_2, ...]}
    """
    raw_f = RAW_TXT_FILES["aliases"]
    dict_f = PKLS_FILES["aliases_dict"]
    dict_rev_f = PKLS_FILES["aliases_rev"]
    
    
    print(f"Aliases are being created and will be saved in {dict_f} and {dict_rev_f}...")
    aliases_dict, aliases_rev = load_aliases_dict_from_text(raw_f)
    cache_array(aliases_dict, dict_f)
    cache_array(aliases_rev, dict_rev_f)
    return aliases_rev

def create_triples_dict():
    """
        does:
        - triples_dict would have {head: [triple1, triple2, ...]} and each triple is a triple (head, relation, tail)
        - head, relation and tail are all ids. head, tail id refering to document_id (same as keys in deescription and aliases_rev) and relation id refering to relation_id (key in relations_dict)
    """
    raw_f = RAW_TXT_FILES["triples"]
    dict_f = PKLS_FILES["triples"]["full"]
    triples_dict = load_triples(raw_f)
    print(f"Saving triples in {dict_f}:")
    cache_array(triples_dict, dict_f)
    
    return triples_dict

def create_relations_dict():
    """
        does:
            - relations_dict would have {relation_id: [relation_name1, relation_name2, ...]}
    """
    raw_f = RAW_TXT_FILES["relations"]
    dict_f = PKLS_FILES["relations"]["full"]
    relations = load_relations(raw_f)
    print(f"Saving relations in {dict_f}:")
    cache_array(relations, dict_f)
    return relations



def get_keys_in_desc_notin_als():
    """
        does:
            - Return list  containing description keys that are not in the aliases
    """
    print("reading dicts...")
    aliases_all_dict = read_cached_array(PKLS_FILES["aliases_rev"])
    desc_all_dict = read_cached_array(PKLS_FILES["descriptions"]["full"])

    desc_keys = desc_all_dict.keys()
    als_keys_set = set(aliases_all_dict.keys())
    print("starting..")
    not_in_als = [d_id for d_id in desc_keys if d_id not in als_keys_set]
    

    print(f"{len(not_in_als)}/{len(desc_keys)} of desc keys not are in  aliases")
    return not_in_als


def save_strange_chars_dict():
    """
        does:
            - save file of compiled strange chars with their subtitution, used for normalization after
    """
    compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in {
        r"[€“©]": "",
        r"[áăắặằẳẵǎâấậầẩẫäǟȧǡạȁàảȃāąåǻḁãǽǣ]": "a",
        r"[ḃḅḇ]": "b",
        r"[ćčçḉĉċ]": "c",
        r"[ďḑḓḋḍḏ]": "d",
        r"[éĕěȩḝêếệềểễḙëėẹȅèẻȇēḗḕęẽḛé]": "e",
        r"[ḟ]": "f",
        r"[ǵğǧģĝġḡ]": "g",
        r"[ḫȟḩĥḧḣḥẖ]": "h",
        r"[íĭǐîïḯi̇ịȉìỉȋīįĩḭı]": "i",
        r"[ǰĵ]": "j",
        r"[ḱǩķḳḵ]": "k",
        r"[ĺľļḽḷḹḻ]": "l",
        r"[ḿṁṃ]": "m",
        r"[ńňņṋṅṇǹṉñ]": "n",
        r"[óŏǒôốộồổỗöȫȯȱọőȍòỏơớợờởỡȏōṓṑǫǭõṍṏȭǿøɔ]": "o",
        r"[ṕṗ]": "p",
        r"[ŕřŗṙṛṝȑȓṟ]": "r",
        r"[śṥšṧşŝșṡẛṣṩ]": "s",
        r"[ťţṱțẗṫṭṯ]": "t",
        r"[úŭǔûṷüǘǚǜǖṳụűȕùủưứựừửữȗūṻųůũṹṵ]": "u",
        r"[ṿṽ]": "v",
        r"[ẃŵẅẇẉẁẘ]": "w",
        r"[ẍẋ]": "x",
        r"[ýŷÿẏỵỳỷȳẙỹy]": "y",
        r"[źžẑżẓẕʐ]": "z",
        r"[&]": "and"
    }.items()]
    cache_array(compiled_patterns, HELPER_FILES['strange_chars'])


#endregion

#region data.py


root_path = os.path.dirname(os.path.abspath(__file__))
root = "."

RAW_FOLDER = f"{root}/data/raw"
HELPERS_FOLDER = f"{root}/data/helpers"
TRANSE_CHECKPOINT_FOLDER = f"{root}/data/transe/checkpoint"

TEMP_FOLDER = f"{root}/data/temp"
TRANSE_FOLDER = f"{root}/data/transe"
CHECKPOINTS_FOLDER = f"{root}/data/checkpoints"
DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
DESCRIPTIONS_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
ALIASES_FOLDER = f"{DICTIONARIES_FOLDER}/aliases"
RELATIONS_FOLDER = f"{DICTIONARIES_FOLDER}/relations"
GOLDEN_TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/golden_triples"
SILVER_SPANS_FOLDER = f"{DICTIONARIES_FOLDER}/silver_spans"

H_Gs_folder = f"{DICTIONARIES_FOLDER}/h_gs"
LOG_FOLDER = f"{root}/logs"
LOG_FOLDER_BUILD_GOLDEN = f"{root}/logs/golden_truth_build"

folders_to_check = [H_Gs_folder , CHECKPOINTS_FOLDER, SILVER_SPANS_FOLDER, GOLDEN_TRIPLES_FOLDER, LOG_FOLDER_BUILD_GOLDEN, LOG_FOLDER, ALIASES_FOLDER, TRANSE_CHECKPOINT_FOLDER, TEMP_FOLDER, TRANSE_FOLDER, DESCRIPTIONS_NORMALIZED_FOLDER, HELPERS_FOLDER, TRIPLES_FOLDER, RELATIONS_FOLDER,DICTIONARIES_FOLDER, DESCRIPTIONS_FOLDER]
for fo in folders_to_check:
    if not os.path.exists(fo):
        os.makedirs(fo)


RAW_TXT_FILES ={
    "descriptions": f"{RAW_FOLDER}/wikidata5m_text.txt",    
    "aliases": f"{RAW_FOLDER}/wikidata5m_entity.txt", 
    "relations": f"{RAW_FOLDER}/wikidata5m_relation.txt",
    "triples": f"{RAW_FOLDER}/wikidata5m_transductive_train.txt",
}

THEMODEL_PATH = "BRASK_MODEL.pth"

PKLS_FILES = {
    "descriptions": {
        "full": f"{DESCRIPTIONS_FOLDER}/descriptions_full.pkl",
        10: f"{DESCRIPTIONS_FOLDER}/descriptions_min_10.pkl",
        100: f"{DESCRIPTIONS_FOLDER}/descriptions_min_100.pkl",
        1_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_1k.pkl",
        10_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_10k.pkl",
        1_000_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_1m.pkl"
    },
    "descriptions_normalized": {
        "full": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full.pkl",
        10: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_10.pkl",
        100: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_100.pkl",
        1_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1k.pkl",
        10_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_10k.pkl",
        1_000_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1m.pkl",
    },
    "descriptions_tokenized": {
        "full_input_ids": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_input_ids.pkl",
        "full_attention_mask": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_attention_masks.pkl",
        "sentence_tokens": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_sentence_tokens.pkl",
        "aliases_tokenized": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/aliases_tokenized.pkl"
        
        
    },
    "aliases_dict": f"{DICTIONARIES_FOLDER}/aliases_dict.pkl",
    "aliases_rev": f"{DICTIONARIES_FOLDER}/aliases_rev.pkl",
    "aliases_rev_norm": f"{DICTIONARIES_FOLDER}/aliases_rev_norm.pkl",
    #This is aliases with keys as query_id and value is a list (it should be named aliases_dict)
    "aliases":{
        "full": f"{ALIASES_FOLDER}/aliases_full.pkl",
        10: f"{ALIASES_FOLDER}/aliases_min_10.pkl",
        100: f"{ALIASES_FOLDER}/aliases_min_100.pkl",
        1_000: f"{ALIASES_FOLDER}/aliases_min_1k.pkl",
        10_000: f"{ALIASES_FOLDER}/aliases_min_10k.pkl",
        1_000_000: f"{ALIASES_FOLDER}/aliases_min_1m.pkl"
        
    },
    "triples": {
        "full": f"{TRIPLES_FOLDER}/triples_full.pkl",
        10: f"{TRIPLES_FOLDER}/triples_min_10.pkl",
        100: f"{TRIPLES_FOLDER}/triples_min_100.pkl",
        1_000: f"{TRIPLES_FOLDER}/triples_min_1k.pkl",
        10_000: f"{TRIPLES_FOLDER}/triples_min_10k.pkl",
        1_000_000: f"{TRIPLES_FOLDER}/triples_min_1m.pkl"
    },
    "relations": {
        "full": f"{RELATIONS_FOLDER}/relations_full.pkl",
        10: f"{RELATIONS_FOLDER}/relations_min_10.pkl",
        100: f"{RELATIONS_FOLDER}/relations_min_100.pkl",
        1_000: f"{RELATIONS_FOLDER}/relations_min_1k.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_min_1m.pkl"
    },
    "relations_embs": {
        "full": f"{RELATIONS_FOLDER}/relations_embs_full.pkl",
        10: f"{RELATIONS_FOLDER}/relations_embs_min_10.pkl",
        100: f"{RELATIONS_FOLDER}/relations_embs_min_100.pkl",
        1_000: f"{RELATIONS_FOLDER}/relations_embs_min_1k.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_embs_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_embs_min_1m.pkl"
    },
    
    "golden_triples": {
        "full": f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_full.pkl",
        10: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_10.pkl",
        100: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_100.pkl",
        1_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_1k.pkl",
        10_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_10k.pkl",
        1_000_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_1m.pkl"
    },
    
    "silver_spans": {
        "full": f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_full.pkl",
        10: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_10.pkl",
        100: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_100.pkl",
        1_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_1k.pkl",
        10_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_10k.pkl",
        1_000_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_1m.pkl"
    },
    "transE_relation_embeddings": f"{TRANSE_FOLDER}/relation_embs.pkl" ,
    "transE_entity_embeddings": f"{TRANSE_FOLDER}/entity_embs.pkl" ,
    "ss_triples_head_aliases": f"{GOLDEN_TRIPLES_FOLDER}/head_aliases.pkl",
    "ss_triples_tail_aliases": f"{GOLDEN_TRIPLES_FOLDER}/tail_aliases.pkl",
    "alias_pattern_map": f"{GOLDEN_TRIPLES_FOLDER}/alias_patterns_map.pkl",
    "encoded_tokenization": f"{DICTIONARIES_FOLDER}/encoded_tokenization.pkl",
      "h_g_s_all_means": f"{DICTIONARIES_FOLDER}/h_g_s_all_means.pt",
    "h_g_s_all_embs": f"{DICTIONARIES_FOLDER}/h_g_s_all_embs.pt",
    


}


TEMP_FILES = {
    "dataset":{
        "full": f"{TEMP_FOLDER}/dataset_full.pkl",
        10: f"{TEMP_FOLDER}/dataset_min_10.pkl",
        100: f"{TEMP_FOLDER}/dataset_min_100.pkl",
        1_000: f"{TEMP_FOLDER}/dataset_min_1k.pkl",
        10_000: f"{TEMP_FOLDER}/dataset_min_10k.pkl",
        1_000_000: f"{TEMP_FOLDER}/dataset_min_1m.pkl"
    } 
}

HELPER_FILES = {
    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl",
    "keys_not_in_als": f"{HELPERS_FOLDER}/keys_not_in_als.pkl"
}

CHECKPOINT_FILES = {
    "transe": f"{CHECKPOINTS_FOLDER}/transE_checkpoint.pth",
    "brask": f"{CHECKPOINTS_FOLDER}/BRASK_checkpoint.pth"
}

LOGGER_FILES = {
    "build_golden_triples": f"{LOG_FOLDER_BUILD_GOLDEN}/build_golden_triples.log"
}

def get_min_descriptionsNorm_triples_relations(k):
    min_desc_norm_dict_f = PKLS_FILES["descriptions_normalized"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    min_aliases_dict_f = PKLS_FILES["aliases"][k]
    
    descs = read_cached_array(min_desc_norm_dict_f)
    triples = read_cached_array(min_triples_dict_f)
    relations = read_cached_array(min_relations_dict_f)
    aliases = read_cached_array(min_aliases_dict_f)
    
    return descs, triples, relations, aliases

def get_compiled_strange_chars():
    f = HELPER_FILES["strange_chars"]
    return  read_cached_array(f)








#endregion





#region normalization



def normalize_description_BRASK(sentence):
    """
        args:
            - sentence: need to be normalized 
        does:
            - use nfkc and remove multiple white spaces
        return:
            - normalized sentence
    """
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # sentence = re.sub(r'\([^)]*\)', '', sentence)
    return sentence


def normalize_desc_batch_BRASK(descs_batch):
    """
        args:
            descs_batch: a batch of description dictionary 
        does:
            - Replace strange chars in the batch texts
            -   Remove non-english words
            - Execute normalize_description_BRASK on every sentence in the dictionary
        return: 
            normalized dict 
    """
    compiled_strange_patterns = get_compiled_strange_chars()
    descs_batch = {k:  replace_special_chars(v, compiled_strange_patterns) for k,v in descs_batch.items()}
    
    
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    
    descs_batch = {k: keep_only_english_chars(v) for k,v in descs_batch.items()}
    
    new_dict=  {}
    for desc_id, desc in tqdm(descs_batch.items(), total=len(descs_batch.keys()), desc="Normalizing batch description"):
        new_dict[desc_id]= normalize_description_BRASK(desc)
  
    return new_dict

    
    
def normalize_desc_parallel(descs_all, num_workers = 8):
    """
        args:
            descs_all: dictionary containing all descriptions need to be normalized
            num_workers: the num of workers where parallel processing would be distributed
        does:
            - create batches from the description (for improvements I can use dataset dataloaders)
            - call the function normalize_desc_batch_BRASK by parallel processing 
            - merge the results and return them 
        returns: 
            - normalized description dictionary 
            
    """
    batch_size = max(1, len(descs_all) // num_workers)
    desc_batches = batch_dict(descs_all, batch_size)
    print(f"normalizing on numworkers: {num_workers}, batch_size is {batch_size} ")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_desc_batch_BRASK)
        results = list(tqdm(
            executor.map(normalize_func, desc_batches),
            total=len(desc_batches),
            desc="Processing batches"
        ))
        
    normalized_descs = {k: v for batch in results for k, v in batch.items()}
    return normalized_descs



#endregion




#region extract_relations_embs

def extract_relations(triples):
    """
        args:
            triples: is a dictionary in shape {head: [list]} each item in the list is a tuple (h,r,t)
        returns: 
            set of r which is the ids of the relations
    """
    rels = set()
    for _, tr_list in triples.items():
        for (_,r, _) in tr_list:
            rels.add(r)

    return list(rels)



def get_rel_embs(relations, relations_full_dict, bert_tokenizer, bert_model):
    """
        args:
            relations: a list of relation ids we want to embed 
            relations_full_dict: a dictionary containing all relations in shape: {rel_id: [list of relation aliases]}

        do: 
            loop through each relation id
            get the list of strings 
            get embedding of each string by getting last two embeding layers of bert 
            average all embedings for the relation 
        return: 
            list of all_rels_embs which contains for each relation in relations, a vector representing it
    """
    tokenizer = bert_tokenizer
    model = bert_model
    model.eval()
    all_rels_embs = []

    for rel_id in relations:
        if rel_id not in relations_full_dict.keys():
            print(f"rel_id was not found in all relations")
            continue
        rel_embs = [] # for each relation string append the embedding
        for rel_str in relations_full_dict[rel_id]:
            inputs = tokenizer(rel_str, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_layer = hidden_states[-1].squeeze(0)
            before_last_layer = hidden_states[-2].squeeze(0)
            average_layers = (last_layer + before_last_layer) / 2.0
            r_j = average_layers.mean(dim=0)
            rel_embs.append(r_j)
        
        #average the rel_embs to represent one relation 
        rel_emb = torch.mean(torch.stack(rel_embs), dim=0)
        all_rels_embs.append(rel_emb)
    return all_rels_embs

def get_rel_embs_opt_2(rel_loader,  bert_model, relations, device):
    emb_dict = {}

    with torch.no_grad():
        for rel_ids, inputs in rel_loader:
            inputs = {k: v.to(device) for k,v in inputs.items()}
            with torch.cuda.amp.autocast():
                outs = bert_model(**inputs, output_hidden_states=True).hidden_states
            avg_layers = (outs[-1] + outs[-2]) / 2
            mask = inputs['attention_mask'].unsqueeze(-1)
            emb = (avg_layers * mask).sum(1) / mask.sum(1)
            for rel_id, vec in zip(rel_ids, emb):
                emb_dict.setdefault(rel_id, []).append(vec.cpu())

    # Final per-relation embeddings
    all_rels_embs = [torch.stack(emb_dict[rel_id]).mean(0)
                    for rel_id in relations]

    return all_rels_embs


#endregion


#region prepare ground truth


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
    
    cache_array(sentences_triples_heads_aliases,PKLS_FILES["ss_triples_head_aliases"] )
    cache_array(sentences_triples_tails_aliases,PKLS_FILES["ss_triples_tail_aliases"] )
    
    
    # sentences_triples_heads_aliases = read_cached_array(PKLS_FILES["ss_triples_head_aliases"] )
    # sentences_triples_tails_aliases = read_cached_array(PKLS_FILES["ss_triples_tail_aliases"] )
    
    
    print("\t doing compiled patterns for aliases ")
        
    alias_pattern_map = {} 
    for lst in tqdm(aliases.values(), total=len(aliases), desc="creating compiled patterns "):
        for alias in lst:
            escaped = re.escape(alias)
            flexible  = escaped.replace(r"\ ", r"\s*")
            pattern   = rf"\b{flexible}\b"
            alias_pattern_map[alias] = re.compile(pattern, re.IGNORECASE)

    cache_array(alias_pattern_map,PKLS_FILES["alias_pattern_map"] )
    
    # alias_pattern_map = read_cached_array(PKLS_FILES["alias_pattern_map"] )
    
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


#endregion



seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def clean_descriptions_dict(descriptions_dict, silver_spans):
    print(f"\t cleaning descriptions_dict with size {len(descriptions_dict)} ")
    silver_span_head_s = silver_spans["head_start"]
    silver_span_head_e = silver_spans["head_end"]
    silver_span_tail_s = silver_spans["tail_start"]
    silver_span_tail_e = silver_spans["tail_end"]
    print(f"\t and silver spans with shape {silver_span_tail_e.shape}")
    
    mask = (
        silver_span_head_s .any(dim=1) &
        silver_span_head_e .any(dim=1) &
        silver_span_tail_s .any(dim=1) &
        silver_span_tail_e .any(dim=1)
    
    )
    
    #wow! I did not know that I can do this
    filtered_dict = {
        key: value
        for (key, value), keep in zip(descriptions_dict.items(), mask)
        if keep.item()
    }
    idx = mask.nonzero(as_tuple=True)[0] 
    return (
        filtered_dict,
        silver_span_head_s[idx],
        silver_span_head_e[idx],
        silver_span_tail_s[idx],
        silver_span_tail_e[idx],
    )







class BRASKDataSet(Dataset):
    def __init__(self, descriptions_dict, silver_spans , device, desc_max_length=128):
        #silver_spans should be dictionary having keys head_start, head_end, tail_start, tail_end, each one is tensor with shape (B, seq_len) 
        print("\tInitiating dataset.. ")
        cleaned_descriptions, silver_span_head_s, silver_span_head_e, silver_span_tail_s, silver_span_tail_e = clean_descriptions_dict(descriptions_dict, silver_spans)
        print(f"\tfinished clean with new descriptions {len(cleaned_descriptions)} and silver spans with shape {silver_span_tail_e.shape}")
        valid = (len(cleaned_descriptions), desc_max_length) == silver_span_head_s.shape == silver_span_tail_s.shape == silver_span_tail_e.shape==silver_span_head_e.shape 
        assert valid 
        if valid:
            print("\tvalid")
            tokenizer =BertTokenizerFast.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')
            print("\tcreating clean descriptions")
            
            print(f"\twe have {len(cleaned_descriptions)} descriptions")
            
            sentences = list(cleaned_descriptions.values())
            print("\tcreating h_gs")
            
            self.h_gs, self.embs = get_h_gs(sentences, tokenizer, model, max_length=desc_max_length, device=device  )  #  h_gs (batch_size, hidden_size), embs (batch_size, seq_len, hidden_size)
            print(f"\tself embs shape: {self.embs.shape} should be ({len(cleaned_descriptions)}, {desc_max_length},hidden_size )")
            assert self.embs.shape[0] == len(cleaned_descriptions)
            assert self.embs.shape[1] == desc_max_length
            self.labels_head_start, self.labels_head_end, self.labels_tail_start, self.labels_tail_end =  silver_span_head_s, silver_span_head_e, silver_span_tail_s, silver_span_tail_e 

    def __getitem__(self,idx):
        return  {
            "h_gs": self.h_gs[idx], 
            "embs": self.embs[idx],
            "labels_head_start": self.labels_head_start[idx] ,
            "labels_head_end": self.labels_head_end[idx],
            "labels_tail_start": self.labels_tail_start[idx],
            "labels_tail_end": self.labels_tail_end[idx],

        }


    def __len__(self):
        return self.h_gs.shape[0]
    
    def save(self, path):
        di = {
            "h_gs": self.h_gs.cpu(),
            "embs": self.embs.cpu(),
            "labels_head_start": self.labels_head_start.cpu() ,
            "labels_head_end": self.labels_head_end.cpu(),
            "labels_tail_start":  self.labels_tail_start.cpu(),
            "labels_tail_end":  self.labels_tail_end.cpu(),
            
        }
        cache_array(di, path)
    @classmethod
    def load(cls, path):
        print("loadding dataset from cache.. ")
        data = read_cached_array(path)

        dataset = cls.__new__(cls)
        dataset.h_gs = data["h_gs"]
        print(f"\t dataset.h_gs.shape: {dataset.h_gs.shape}")
        dataset.embs = data["embs"]
        print(f"\t dataset.embs.shape: {dataset.embs.shape}")
        dataset.labels_head_start = data["labels_head_start"]
        dataset.labels_head_end = data["labels_head_end"]
        dataset.labels_tail_start = data["labels_tail_start"]
        dataset.labels_tail_end = data["labels_tail_end"]
        
        return dataset
    
class BRASKModel(nn.Module):
    def __init__(self, rel_embs, rel_transe_embs, hidden_size=768, transE_emb_dim=100, thresholds=[.5,.5,.5,.5], device="cpu"):
        super(BRASKModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.rel_embs = rel_embs.to(device)
        self.rel_transe_embs = rel_transe_embs.to(device)
        
        self.transE_emb_dim = transE_emb_dim

        #f_subj, b_obj, f_obj, b_subj
        self.f_subj_threshold = thresholds[0]
        self.b_obj_threshold = thresholds[1]
        self.f_obj_threshold = thresholds[2]
        self.b_subj_threshold = thresholds[3]

        #forward
        self.f_start_sub_fc = nn.Linear(hidden_size, 1)
        self.f_end_sub_fc = nn.Linear(hidden_size, 1)

        self.f_start_obj_fc = nn.Linear(hidden_size, 1)
        self.f_end_obj_fc = nn.Linear(hidden_size, 1)
        

        #backward
        self.b_start_obj_fc = nn.Linear(hidden_size, 1)
        self.b_end_obj_fc = nn.Linear(hidden_size, 1)

        self.b_start_sub_fc = nn.Linear(hidden_size, 1)
        self.b_end_sub_fc = nn.Linear(hidden_size, 1)





        # for s_k in forward or o_k in backward 
        self.f_W_s = nn.Linear(hidden_size, hidden_size)
        self.b_W_s = nn.Linear(hidden_size, hidden_size)

        self.r_proj = nn.Linear(transE_emb_dim, hidden_size)


        #forward:
        self.f_W_r = nn.Linear(hidden_size, hidden_size)
        self.f_W_g = nn.Linear(hidden_size, hidden_size)
        self.f_W_x = nn.Linear(hidden_size, hidden_size)

        #backward:
        self.b_W_r = nn.Linear(hidden_size, hidden_size)
        self.b_W_g = nn.Linear(hidden_size, hidden_size)
        self.b_W_x = nn.Linear(hidden_size, hidden_size)

        # e = V^T tanh()... , V is the same for forward and backward as in the paper 
        self.V   = nn.Linear(hidden_size, 1)

        self.f_Wx2 = nn.Linear(hidden_size, hidden_size)
        self.b_Wx2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, ds_batch):
        """ 
            From now on when we say in the shape: 
                b: batch_size 
                l: seq_len
                r: relation_nums
                h: hidden_size (786)
            The parameter names if it starts with
                f: forward (subject, relation, object)
                b: backward (object, relation, subject)
                 
        """
        device = ds_batch["h_gs"].device


        h_gs = ds_batch["h_gs"].to(device) # (b, h)
        token_embs = ds_batch["embs"].to(device) #(b, l, h)
        f_rel_embs =  self.rel_embs.to(device) # (r, h)
        b_rel_transe_embs = self.rel_transe_embs #(r, h_transE)

        # project transE embeddings into bert space
        b_rel_embs = self.r_proj(b_rel_transe_embs) #(r, hidden_size)
        

        batch_size, seq_len, hidden_size = token_embs.shape 
        num_relations = f_rel_embs.shape[0]
        
        

        #forward
        f_sub_s_logits = self.f_start_sub_fc(token_embs)
        f_sub_e_logits = self.f_end_sub_fc(token_embs)
        f_sub_start_probs = self.sigmoid(f_sub_s_logits).squeeze(-1) # (b, l)
        f_sub_end_probs = self.sigmoid(f_sub_e_logits).squeeze(-1) 
        #backward
        
        b_obj_s_logits = self.b_start_obj_fc(token_embs)
        b_obj_e_logits = self.b_end_obj_fc(token_embs)
        b_obj_start_probs = self.sigmoid(b_obj_s_logits).squeeze(-1)
        b_obj_end_probs = self.sigmoid(b_obj_e_logits).squeeze(-1) # (b,l)
        

        # f_padded_subj_embs has shape (B, L, H)
        # f_mask_subj_embs shape (B, L) each item is 1 if it is padding, 0 otherwize
        # f_subj_idxs is a list where each item representing one sentence from the batch and inside each item is a list of tuples (start_subj_idx, end_subj_idx)

        f_padded_subj_embs, f_mask_subj_embs, f_subj_idxs = extract_first_embeddings(
            token_embs, f_sub_start_probs, f_sub_end_probs, seq_len, threshold=self.f_subj_threshold) # , 



        b_padded_obj_embs, b_mask_obj_embs, b_obj_idxs = extract_first_embeddings(
            token_embs, b_obj_start_probs, b_obj_end_probs, seq_len, threshold=self.b_obj_threshold) # b_padded_obj_embs has shape (B, L, H)



        # f_s_w is Wsf(S_k) for forwad (all subject embeddings weighted), 
        # b_o_w is Wsb(O_k) for backward (all objet embeddings weighted), 
        f_s_w = self.f_W_s(f_padded_subj_embs) # (B, max_subjs, H)
        b_o_w = self.b_W_s(b_padded_obj_embs) # (B, max_objs, H)

        #zero out the padded entries
        f_s_w = f_s_w * f_mask_subj_embs.unsqueeze(-1).float()  # (B, max_subjs, H)
        b_o_w = b_o_w * b_mask_obj_embs.unsqueeze(-1).float() # (B, max_objs, H)


        token_embs_exp = token_embs.unsqueeze(2) #(B, L, 1, H)        
        h_g_exp = h_gs.unsqueeze(1).unsqueeze(2) #(B, 1,1 , H)

      
        #FORWARD
        f_rel_embs_exp = f_rel_embs.unsqueeze(0).unsqueeze(1)  #(1, 1, R, H)
        f_rel_embs_exp = f_rel_embs_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations, hidden_size)

        #backward
        b_rel_embs_exp = b_rel_embs.unsqueeze(0).unsqueeze(1)  #(1, 1, R, H)
        b_rel_embs_exp = b_rel_embs_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations, hidden_size)


        
        
        # attention scores:
        # we make tanh for adding non-linearity
        # v_e has shape (B,L,R,1) for every token a score, this score is kind of how much the entity is related to the relationship

        #Forward:
        f_e = torch.tanh(
            self.f_W_r(f_rel_embs_exp) + self.f_W_g(h_g_exp) + self.f_W_x(token_embs_exp)
        ) # (B, L, R, H)
        f_v_e = self.V(f_e).squeeze(-1) # (B, L, R)


        #Backward:
        b_e = torch.tanh(
            self.b_W_r(b_rel_embs_exp) + self.b_W_g(h_g_exp) + self.b_W_x(token_embs_exp)
        ) # (B, L, R, H)
        b_v_e = self.V(b_e).squeeze(-1) # (B, L, R)


        #normalize attention score
        #on dim = 1 because on the sentences dimension because we want to distribute attention on tokens, like we want to get probability distribution for all tokens if they are relevant to the relation
        f_A = F.softmax(f_v_e, dim=1).unsqueeze(-1)  # (B, L, R, 1) 
        b_A = F.softmax(b_v_e, dim=1).unsqueeze(-1)  # (B, L, R, 1) 

        f_C = torch.sum(f_A * token_embs_exp, dim=1) # (B ,R, H)
        b_C = torch.sum(b_A * token_embs_exp, dim=1) # (B ,R, H)

        

        




        f_Hik =  f_s_w + self.f_Wx2(token_embs) #  (B, L, H) 
        b_Hik =  b_o_w + self.b_Wx2(token_embs)

        f_Hij = f_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)
        b_Hij = b_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)

        f_Hijk = f_Hik.unsqueeze(2) + f_Hij # (B, L, 1, H) +   (B, L, R, H) =  (B, L, R, H)
        b_Hijk = b_Hik.unsqueeze(2) + b_Hij
        




        
        
        #forward
        f_obj_s_logits = self.f_start_obj_fc(f_Hijk)
        f_obj_e_logits = self.f_end_obj_fc(f_Hijk)
        f_obj_start_probs = self.sigmoid(f_obj_s_logits).squeeze(-1) # (b, l , r)
        f_obj_end_probs = self.sigmoid(f_obj_e_logits).squeeze(-1)  # (b, l , r)

        #backward
        b_sub_start_logits = self.b_start_sub_fc(b_Hijk)
        b_sub_end_logits = self.b_end_sub_fc(b_Hijk)
        b_sub_start_probs = self.sigmoid(b_sub_start_logits).squeeze(-1) # (b, l , r)
        b_sub_end_probs = self.sigmoid(b_sub_end_logits).squeeze(-1) # (b, l , r)




        # forward_triples  = extract_triples(f_subj_idxs, f_obj_start_probs, f_obj_end_probs,  True , threshold=self.f_obj_threshold )
        # backward_triples = extract_triples(b_obj_idxs, b_sub_start_probs, b_sub_end_probs,  False, threshold=self.b_subj_threshold)
        
        # predicted_triples = merge_triples(forward_triples, backward_triples)
        

        return {
            "forward": {
                "sub_s": f_sub_s_logits.squeeze(-1), 
                "sub_e": f_sub_e_logits.squeeze(-1), 
                "obj_s": f_obj_s_logits.squeeze(-1) , 
                "obj_e": f_obj_e_logits.squeeze(-1)  , 
            },
            "backward": {
                
                "obj_s": b_obj_s_logits.squeeze(-1), 
                "obj_e": b_obj_e_logits.squeeze(-1), 
                "sub_s": b_sub_start_logits.squeeze(-1), 
                "sub_e": b_sub_end_logits.squeeze(-1), 
            },
            # "predicted_triples": predicted_triples
        }

def get_pos_weights(dataset, batch_size=256, num_workers=4):
    label_keys = [
        'labels_head_start', 'labels_head_end',
        'labels_tail_start','labels_tail_end'
    ]
    pos_counts = {k: 0 for k in label_keys}
    neg_counts = {k: 0 for k in label_keys}
    
    loader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False)
    
    for batch in loader:
        for k in label_keys:
            lbl = batch[k]
            p = lbl.sum().item() 
            n = lbl.numel() - p 
            pos_counts[k] += p
            neg_counts[k] += n
    return {
        k: (neg_counts[k] / pos_counts[k]) if pos_counts[k] > 0 else 1.0
        for k in label_keys
        
    }

    


def save_checkpoint(model, optimizer, epoch, last_total_loss, filename="checkpoint.pth"):
    checkpoint = {
       "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_total_loss": last_total_loss
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        last_total_loss = checkpoint["last_total_loss"]
        print(f"Resumed from epoch {start_epoch} with loss {last_total_loss:.4f}")
        return start_epoch, last_total_loss
    else:
        return 0, 0.0
    
    

def compute_loss(out, batch, bce_losses):
    L = 0.0

    L += bce_losses['f_sub_s'](out['forward']['sub_s'].squeeze(-1),
                               batch['labels_head_start'])
    L += bce_losses['f_sub_e'](out['forward']['sub_e'].squeeze(-1),
                               batch['labels_head_end'])
    L += bce_losses['f_obj_s'](out['forward']['obj_s'].squeeze(-1),
                               batch['labels_tail_start'])
    L += bce_losses['f_obj_e'](out['forward']['obj_e'].squeeze(-1),
                               batch['labels_tail_end'])

    L += bce_losses['b_obj_s'](out['backward']['obj_s'].squeeze(-1),
                               batch['labels_tail_start'])
    L += bce_losses['b_obj_e'](out['backward']['obj_e'].squeeze(-1),
                               batch['labels_tail_end'])
    L += bce_losses['b_sub_s'](out['backward']['sub_s'].squeeze(-1),
                               batch['labels_head_start'])
    L += bce_losses['b_sub_e'](out['backward']['sub_e'].squeeze(-1),
                               batch['labels_head_end'])
    return L / 8.0
def build_bce_loss_dict(pos_weights):
    pw = {k: torch.tensor(v, dtype=torch.float32).to(device)
      for k, v in pos_weights.items()}
    bce_losses = {
        # forward
        'f_sub_s': nn.BCEWithLogitsLoss(pos_weight=pw['labels_head_start']),
        'f_sub_e': nn.BCEWithLogitsLoss(pos_weight=pw['labels_head_end']),
        'f_obj_s': nn.BCEWithLogitsLoss(pos_weight=pw['labels_tail_start']),
        'f_obj_e': nn.BCEWithLogitsLoss(pos_weight=pw['labels_tail_end']),
    }
    # backward re-uses the same four weights:
    bce_losses.update({
        'b_obj_s': bce_losses['f_obj_s'],
        'b_obj_e': bce_losses['f_obj_e'],
        'b_sub_s': bce_losses['f_sub_s'],
        'b_sub_e': bce_losses['f_sub_e'],
    })
    return bce_losses


def train_model(dataset,k,thresholds,checkpoint_path, transE_emb_dim=100, batch_size=128, num_workers=48, learning_rate=1e-5, num_epochs=1000, device="cpu", ):
    print("loading dataloader..")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True , persistent_workers=True) 
    
    relations_embs = read_cached_array(PKLS_FILES["relations_embs"][k])
    relations_embs_transE = read_cached_array(PKLS_FILES["transE_relation_embeddings"])

    
    pos_weights = get_pos_weights(dataset,batch_size, num_workers )
    bce_losses = build_bce_loss_dict(pos_weights)



    ###### REMEMBER TO MAKE IT torch.compile  in production
    print("creating model..")
    model = BRASKModel(relations_embs, relations_embs_transE, transE_emb_dim=transE_emb_dim, thresholds=thresholds,device=device)
    model.to(device)
    
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = torch.compile(model)

    model.train()
    optimizer = optim.Adam(model.parameters(),  lr=learning_rate )
    start_epoch, last_total_loss = load_checkpoint(model, optimizer, filename=checkpoint_path)
    scaler = GradScaler(device_type="cuda")
    print("starting epochs loop")
    with tqdm(range(start_epoch - 1, num_epochs), desc="Training Epochs", total=(num_epochs-start_epoch - 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            total_loss = last_total_loss 
            last_loss = 0.0
            batch_count = 0
            for ds_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                ds_batch = {key: value.to(device, non_blocking=True) for key, value in ds_batch.items()}
                with autocast(device_type="cuda", dtype=torch.float16):
                    out  = model(ds_batch)
                    loss = compute_loss(out, ds_batch, bce_losses)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # loss.backward()
                # optimizer.step()
                
                last_loss = loss.item()
                total_loss += last_loss
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            pbar_epoch.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss:.4f}",last_loss=last_loss )
            save_checkpoint(model, optimizer, epoch, total_loss, filename=checkpoint_path)
            
            
    return model



class RelAliasDataset(Dataset):
    def __init__(self, relations_full_dict):
        self.pairs = [(rel, alias) 
                      for rel, aliases in relations_full_dict.items() 
                      for alias in aliases]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

DESCRIPTION_MAX_LENGTH = 128
NUM_WORKERS = 96
import os 

if __name__ == "__main__":
    k = "full" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # bert_model = BertModel.from_pretrained('bert-base-cased').half().to(device).eval()
    # span_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    
    
    d = torch.randn(2786974)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    
    
    get_h_gs( d , bert_tokenizer, model, max_length=128, device=device  )  #  h_gs (batch_size, hidden_size), embs (batch_size, seq_len, hidden_size)
    
    
    
    
    print("create data files...")
    # save_strange_chars_dict()
    # descriptions_unormalized = create_descriptions_dict()
    # aliases = create_aliases_dicts()
    # triples = create_triples_dict()
    # relations = relations_full_dict = create_relations_dict()
    
    # relations = relations_full_dict = read_cached_array(PKLS_FILES["relations"]["full"])
    # aliases = read_cached_array(PKLS_FILES["aliases_rev"])
    # triples = read_cached_array(PKLS_FILES["triples"]["full"])
    
    # descriptions_unormalized = read_cached_array(PKLS_FILES["descriptions"]["full"])
    
    print("finished creating data files")
    
    
    print("starting normalization...")
    
    # descs = normalized_descs = descriptions_dict = normalize_desc_parallel(descriptions_unormalized, NUM_WORKERS)
    # cache_array(normalized_descs, PKLS_FILES["descriptions_normalized"][k])
    
 
    # descs = normalized_descs = descriptions_dict = read_cached_array(PKLS_FILES["descriptions_normalized"][k])
    
    print("normalization finished...")



    print("extracting relations emb...")
    

    # def collate_fn(batch):
    #     rel_ids, aliases = zip(*batch)
    #     enc = bert_tokenizer(list(aliases), padding=True, truncation=True, return_tensors='pt')
    #     return rel_ids, enc


    # rel_loader = DataLoader(
    #     RelAliasDataset(relations_full_dict),
    #     batch_size=32,
    #     collate_fn=collate_fn,
    #     num_workers=NUM_WORKERS,  # parallel tokenization
    #     drop_last=False
    # )
    
    # rel_embs = get_rel_embs_opt_2(rel_loader, bert_model,  relations, device )
    # cache_array(rel_embs, PKLS_FILES["relations_embs"][k] )


    
    # rel_embs = read_cached_array(PKLS_FILES["relations_embs"][k])
    
    
    # print("finished extracting relations emb len: " , len(rel_embs), " first shape:  " , rel_embs[0].shape)
    
    
    

    print("starting preparing silver spans...")
    
    # silver_span_head_s, silver_span_head_e,  silver_span_tail_s,silver_span_tail_e, _  = extract_silver_spans(descs, triples, aliases)
    # silver_spans = {
    #     "head_start":silver_span_head_s ,
    #     "head_end": silver_span_head_e,
    #     "tail_start": silver_span_tail_s,
    #     "tail_end":silver_span_tail_e ,
    # }
    # cache_array(silver_spans, PKLS_FILES["silver_spans"][k])
    
    # silver_spans = read_cached_array( PKLS_FILES["silver_spans"][k])
    # print("finished preparing silver spans...")
    




    
    # print("start the training model")    
    
    # transE_emb_dim = 100
    # batch_size = 20_048
    # learning_rate = 0.001
    # num_epochs = 128
    # #f_subj, b_obj, f_obj, b_subj
    # thresholds = [.5,.5,.5,.5]
    
    # checkpoint_file = CHECKPOINT_FILES["brask"]
    
    # print("creating ds")
    # dataset = BRASKDataSet(descriptions_dict,silver_spans, device )
    # os.remove(PKLS_FILES["h_g_s_all_embs"])
    # dataset.save(TEMP_FILES["dataset"][k])
    
    # # print("reading dataset from cache")
    # # dataset = BRASKDataSet.load(TEMP_FILES["dataset"][k])
    
    # model = train_model(
    #         dataset, k, transE_emb_dim=transE_emb_dim, batch_size=batch_size,
    #                         num_workers=NUM_WORKERS, learning_rate=learning_rate, 
    #                         num_epochs=num_epochs, thresholds=thresholds, 
    #                         device=device, checkpoint_path=checkpoint_file)
    
    
    
    # torch.save(model.state_dict(), THEMODEL_PATH)