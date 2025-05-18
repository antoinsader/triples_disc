import torch


device_str = 'cpu'  
if torch.cuda.is_available():
  device_str = "cuda"

device = torch.device(device_str)


NUM_WORKERS = 0
RELATIONS_BATCH_SIZE = 32
DESCRIPTIONS_MAX_LENGTH = 128





# curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
# curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
# curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar


# pip install numpy  tqdm pandas
# pip install torch transformers
# pip install scikit-learn  nltk datasets psutil
# pip install joblib

import lz4.frame
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





# prepare_main:
    # get dedscriptions has triples 
    # take half of them 
    # get tail ids and add them to description ids 
    # get triplees of description ids 
    # get aliases
    
#normalize descriptions:
    # replace special chars with their propper subtitute 
    # keep only english letters 
    # remove multiples spaces 
    # save normalized descriptions




def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")


def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)





def save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    np.savez_compressed(path, arr=tensor.cpu().numpy())
    print(f"Tensor chached in file {path}")


def read_tensor(path):
    loaded = np.load(path)
    return torch.from_numpy(loaded("data"))



root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
HELPERS_FOLDER = f"{root}/data/helpers"
CHECKPOINT_FOLDER = f"{root}/data/checkpoints"

RAW_TXT_FILES ={
    "descriptions": f"{RAW_FOLDER}/wikidata5m_text.txt",    
    "aliases": f"{RAW_FOLDER}/wikidata5m_entity.txt", 
    "relations": f"{RAW_FOLDER}/wikidata5m_relation.txt",
    "triples": f"{RAW_FOLDER}/wikidata5m_transductive_train.txt",
}

RESULT_FILES  = {
    "descriptions_unormalized": f"{RESULTS_FOLDER}/descriptions_unormalized.pkl",
    "descriptions": f"{RESULTS_FOLDER}/descriptions.pkl",
    "aliases": f"{RESULTS_FOLDER}/aliases.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl",
    "rel_embs_tensor": f"{RESULTS_FOLDER}/rel_embs_tensor.npz"
}

HELPER_FILES = {
    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl",
    "keys_not_in_als": f"{HELPERS_FOLDER}/keys_not_in_als.pkl",
    
    "descs_all": f"{HELPERS_FOLDER}/descs_all.pkl",
    "aliases_all": f"{HELPERS_FOLDER}/aliases_all.pkl",
    "triples_all": f"{HELPERS_FOLDER}/triples_all.pkl",
    "relations_all": f"{HELPERS_FOLDER}/relations_all.pkl",
    "descs_keys_min": f"{HELPERS_FOLDER}/descs_keys_min.pkl",
}
CHECKPOINTS_FILES = {
    "transe_triples": f"{CHECKPOINT_FOLDER}/transe_triples.pkl",
    "transe_model": f"{CHECKPOINT_FOLDER}/transe_model.pth",
}


folders_to_check = [RESULTS_FOLDER, HELPERS_FOLDER]

for fo in folders_to_check:
    if not os.path.exists(fo):
        os.makedirs(fo)

#region preparation

def get_triples(fp):
    triples_dict = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc="reading triples"):
            head, relation, tail = line.strip().split('\t')
            triples_dict[head].append((head, relation, tail))
    return triples_dict

def get_descriptions(fp):
    dic = {}
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc="reading descs"):
            try: 
                entity_id, description, *rest = line.strip().split("\t")
                dic[entity_id] = description
            except ValueError as e:
                print(f"line has not enough arguments: {line.strip()}")
                break 
    return dic 

def get_aliases(fp):
    dic = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc="reading aliases"):
            try:
                split_line = line.strip().split("\t")
                entity_id = str(split_line[0])
                entity_name = str(split_line[1])
                aliases = split_line[2:]
                dic[entity_id].append(entity_name)
                for al in aliases:
                    dic[entity_id].append(str(al))
        
            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
        return dic
def get_relations(fp):
    dic = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc="reading relations"):
            try:
                split_line = line.strip().split("\t")
                relation_id = str(split_line[0])
                relation_names = split_line[1:]
                for r_n in relation_names:
                    dic[relation_id].append(str(r_n))

            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return dic

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

def prepare_main_cpu_1(percentage=.5):
    triples_fp = RAW_TXT_FILES["triples"]
    triples_dict = get_triples(triples_fp)
    descs_raw_f  = RAW_TXT_FILES["descriptions"]
    desc_dict_all = get_descriptions(descs_raw_f)
    desc_dict = {k:v for k,v in desc_dict_all.items() if k in triples_dict}
    print(f"we have {len(desc_dict)} descriptions that has triples ")
    half_size = ceil(len(desc_dict) * percentage) 
    desc_half_ids = random.sample(list(desc_dict.keys()), half_size)
    print(f"we have now {len(desc_half_ids)} descriptions")
    my_triples = {k:v for k,v in triples_dict.items()  if k in desc_half_ids}
    my_tails = [t for  v in my_triples.values() for _, _, t in v  ]
    final_desc_dict_keys = list(set(desc_half_ids + my_tails))
    final_desc_dict = {k: v for k,v in desc_dict_all.items() if k in final_desc_dict_keys  }
    cache_array(final_desc_dict, RESULT_FILES["descriptions_unormalized"])
    final_triples = {k: v for k,v in triples_dict.items() if k in final_desc_dict_keys}
    cache_array(final_triples, RESULT_FILES["triples"])


def prepare_main_cpu_2():
    relations_raw_fp = RAW_TXT_FILES["relations"]
    aliases_raw_fp = RAW_TXT_FILES["aliases"]

    aliases_dict_all = get_aliases(aliases_raw_fp)
    relations_dict_all = get_relations(relations_raw_fp)

    final_desc_dict = read_cached_array( RESULT_FILES["descriptions_unormalized"])
    final_triples = read_cached_array(RESULT_FILES["triples"])



    final_desc_dict_keys = final_desc_dict.keys()
    final_aliases_dict = {k:v for k,v in aliases_dict_all.items() if k in final_desc_dict_keys}
    cache_array(final_aliases_dict, RESULT_FILES["aliases"])


    my_relations_keys = list(set([r for tr_lst in list(final_triples.values()) for _,r, _ in tr_lst]))
    final_relations_dict = {k:v for k,v in relations_dict_all.items() if k in my_relations_keys}
    cache_array(final_relations_dict, RESULT_FILES["relations"])


def prepare_main(percentage=.5):
    triples_fp = RAW_TXT_FILES["triples"]
    descs_raw_f  = RAW_TXT_FILES["descriptions"]
    aliases_raw_fp = RAW_TXT_FILES["aliases"]
    relations_raw_fp = RAW_TXT_FILES["relations"]

    if os.path.exists(HELPER_FILES["triples_all"]):
        print("reading triples from cache")
        triples_dict = read_cached_array(HELPER_FILES["triples_all"])
    else:
        triples_dict = get_triples(triples_fp)
        cache_array(triples_dict, HELPER_FILES["triples_all"])
    print(f"We have {len(triples_dict)} triple heads")



    if os.path.exists(HELPER_FILES["descs_all"]):
        desc_dict_all = read_cached_array( HELPER_FILES["descs_all"])
    else:
        desc_dict_all = get_descriptions(descs_raw_f)
        cache_array(desc_dict_all, HELPER_FILES["descs_all"])
    print(f"we have {len(desc_dict_all)} full descriptions ")


    if os.path.exists(HELPER_FILES["aliases_all"]):
        aliases_dict_all = read_cached_array( HELPER_FILES["aliases_all"])
    else:
        aliases_dict_all = get_aliases(aliases_raw_fp)
        cache_array(aliases_dict_all, HELPER_FILES["aliases_all"])
    print(f"we have: aliases_dict_all length: {len(aliases_dict_all)}")

    if os.path.exists(HELPER_FILES["relations_all"]):
        relations_dict_all = read_cached_array( HELPER_FILES["relations_all"])
    else:
        relations_dict_all = get_relations(relations_raw_fp)
        cache_array(relations_dict_all, HELPER_FILES["relations_all"])
    print(f"we have: relations_dict_all length: {len(relations_dict_all)}")



    desc_dict = {k:v for k,v in desc_dict_all.items() if k in triples_dict}
    print(f"we have {len(desc_dict)} descriptions that has triples ")


    #take half of them 
    half_size = ceil(len(desc_dict) * percentage) 
    desc_half_ids = random.sample(list(desc_dict.keys()), half_size)
    print(f"we have now {len(desc_half_ids)} descriptions")

    # get tail ids and add them to description ids 

    if os.path.exists(HELPER_FILES["descs_keys_min"]):
        final_desc_dict_keys = read_cached_array( HELPER_FILES["descs_keys_min"])
    else:
        my_triples = {k:v for k,v in triples_dict.items()  if k in desc_half_ids}
        my_tails = [t for  v in my_triples.values() for _, _, t in v  ]
        my_relations_keys = [r for  v in my_triples.values() for _, r, _ in v  ]
        final_desc_dict_keys = list(set(desc_half_ids + my_tails))
        cache_array(final_desc_dict_keys, HELPER_FILES["descs_keys_min"])


    if os.path.exists(HELPER_FILES["descriptions_unormalized"]):
        final_desc_dict = read_cached_array( RESULT_FILES["descriptions_unormalized"])
    else:
        print(f"Final desc len is :{len(final_desc_dict_keys)}")
        final_desc_dict = {k: v for k,v in desc_dict_all.items() if k in final_desc_dict_keys  }
        cache_array(final_desc_dict, RESULT_FILES["descriptions_unormalized"])

    if os.path.exists(HELPER_FILES["triples"]):
        final_triples = read_cached_array(RESULT_FILES["triples"])
    else:
        print("making triples")
        final_triples = {k: v for k,v in triples_dict.items() if k in final_desc_dict_keys}
        cache_array(final_triples, RESULT_FILES["triples"])


    print(f"aliases dict all has: {len(aliases_dict_all)} and first key is : {list(aliases_dict_all.keys())[0]} and first value is : {list(aliases_dict_all.values())[0]}")
    print(f" triples has {len(final_triples)} heads.  making aliases...")
    final_aliases_dict = {k:v for k,v in aliases_dict_all.items() if k in final_desc_dict_keys}
    cache_array(final_aliases_dict, RESULT_FILES["aliases"])


    print(f"aliases dict has len: {len(final_aliases_dict)}.  making relations...")
    final_relations_dict = {k:v for k,v in relations_dict_all.items() if k in my_relations_keys}
    print(f"relations dict len is: {len(final_relations_dict)}.  saving...") 
    cache_array(final_relations_dict, RESULT_FILES["relations"])

#endregion


#region normalize_descriptions

def replace_special_chars(text, compiled_patterns):
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text


def normalize_als_batch(als_batch_dict):
    compiled_strange_chars = read_cached_array(HELPER_FILES["strange_chars"])
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    new_dict = defaultdict(list)
    for k, val in als_batch_dict.items():
        local_set = set()
        for als in val:
            aa  = replace_special_chars(als, compiled_strange_chars)
            aa = keep_only_english_chars(aa)
            aa = unicodedata.normalize('NFKC', aa)
            aa = re.sub(r'\s+', ' ', aa).strip()
            aa = aa.lower()

            local_set.add(aa)

        new_dict[k]  = list(local_set)
    return new_dict


def normalize_desc_batch(descs_batch_dict):
    compiled_strange_chars = read_cached_array(HELPER_FILES["strange_chars"])
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    new_dict = {}
    for k, val in descs_batch_dict.items():
        val = replace_special_chars(val, compiled_strange_chars)
        val = keep_only_english_chars(val)
        val = unicodedata.normalize('NFKC', val)
        val = re.sub(r'\s+', ' ', val).strip()
        new_dict[k]  = val

    return new_dict

    
def normalize_descriptions():
    descs = read_cached_array(RESULT_FILES["descriptions_unormalized"])

    batch_size =  len(descs) // NUM_WORKERS
    items = list(descs.items())
    
    batches = [
        dict(items[i : i+batch_size])
        for i in range(0, len(items), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_desc_batch)
        results = list(tqdm(
            executor.map(normalize_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    dic_norm = {k: v for batch in results for k,v in batch.items()}
    cache_array(dic_norm, RESULT_FILES["descriptions"])


def normalize_aliases():
    aliases = read_cached_array(RESULT_FILES["aliases"])

    batch_size =  len(aliases) // NUM_WORKERS
    items = list(aliases.items())

    batches = [
        dict(items[i : i+batch_size])
        for i in range(0, len(items), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_als_batch)
        results = list(tqdm(
            executor.map(normalize_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    dic_norm = {k: v for batch in results for k,v in batch.items()}
    cache_array(dic_norm, RESULT_FILES["aliases"])



#endregion


#region prepare_relations_bert

def collate_fn(batch, bert_tokenizer):
    rel_ids, aliases = zip(*batch)
    return rel_ids, bert_tokenizer(list(aliases), padding=True, truncation=True, return_tensors="pt") 

def get_rel_embs(loader, bert_model, rel_keys):
    emb_dict = {}

    with torch.no_grad():
        for rel_ids, inputs in tqdm(loader, total=len(loader), desc="extracting relations embs" ):
            #move to device
            inputs = {k: v.to(device) for k,v in inputs.items()}
            with torch.autocast(device_type=device_str):
                outs = bert_model(**inputs, output_hidden_states=True).hidden_states
            avg_layers = (outs[-1] + outs[-2]) / 2.0
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (avg_layers * mask).sum(1) / mask.sum(1)
            for rel_id, vec in zip(rel_ids, emb):
                emb_dict.setdefault(rel_id, []).append(vec.cpu())
    all_rels_embs = [torch.stack(emb_dict[rel_id]).mean(0) for rel_id in rel_keys]
    return torch.stack(all_rels_embs)

class RelAliasDataset(Dataset):
    def __init__(self, relations_full_dict): 
        self.pairs = [
            (rel, alias)
            for rel, aliases in relations_full_dict.items()
            for alias in aliases
        ]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]



def prep_relations_main():
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained("bert-base-cased").half().to(device)
    bert_model = torch.compile(bert_model)
    relations_full_dict = read_cached_array(RESULT_FILES["relations"])
    print("creating dataloader")
    rel_loader = DataLoader(
        RelAliasDataset(relations_full_dict),
        batch_size= RELATIONS_BATCH_SIZE,
        collate_fn=lambda batch: collate_fn(batch, bert_tokenizer),
        num_workers=NUM_WORKERS,
        drop_last=False
    )
    print("create rel embs")
    rel_embs = get_rel_embs(rel_loader, bert_model, list(relations_full_dict.keys()))
    print("rel embs done, saving..")
    save_tensor(rel_embs, RESULT_FILES["rel_embs_tensor"])
#endregion 

#region extract_silver_spans
def extract_silver_spans_main():
    descs_dict = read_cached_array(RESULT_FILES["descriptions"])
    triples_dict = read_cached_array(RESULT_FILES["triples"])
    aliases_dict =read_cached_array(RESULT_FILES["aliases"])

    batch_size = len(descs_dict)
    chunk_size = batch_size // 20
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    sentences_ids = list(descs_dict.keys())
    sentences_texts = list(descs_dict.values())

    sentences_heads = []
    sentences_tails = []
    for s in tqdm(sentences_ids, total=len(sentences_ids), desc="setting sentences heads and tails"):
        for h, _, t in triples_dict[s]:
            sentences_heads += [aliases_dict[h]] 
            sentences_tails += [aliases_dict[t]]
    print("\t doing compiled patterns for aliases ")

    alias_pattern_map = {}
    for lst in tqdm(aliases_dict.values(), total=len(aliases_dict), desc="creating compiled patterns"):
        for als in lst:
            escaped = re.escape(als)
            flexible = escaped.replace(r"\ ", r"\s*")
            pattern = rf"\b{flexible}\b"
            alias_pattern_map[als] = re.comple(pattern, re.IGNORECASE)

    
    
#endregion

if __name__ == "__main__":
    # if device_str == "cpu":
        # prepare_main_cpu_1(.00005)
        # prepare_main_cpu_2()
    # else:
        # prepare_main(0.5)

    # save_strange_chars_dict()
    # normalize_descriptions()
    # normalize_aliases()
    prep_relations_main()    
    # extract_silver_spans_main()
