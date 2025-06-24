import torch


device_str = 'cpu'  
if torch.cuda.is_available():
  device_str = "cuda"

device = torch.device(device_str)


device_str = 'cpu'  
if torch.cuda.is_available():
  device_str = "cuda"

device = torch.device(device_str)

if device_str == 'cpu':
    NUM_WORKERS = 4
    RELATIONS_BATCH_SIZE = 32
    DESCRIPTIONS_MAX_LENGTH = 128
    ON_UNIX = False
    

else:
    NUM_WORKERS = 48
    RELATIONS_BATCH_SIZE = 8192
    DESCRIPTIONS_MAX_LENGTH = 128
    ON_UNIX = True



# curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
# curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
# curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar


# pip install numpy  tqdm pandas
# pip install torch transformers
# pip install scikit-learn  nltk datasets psutil
# pip install joblib

from math import ceil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast
import numpy as np 
from tqdm import tqdm
import pickle
import random
import os


from collections import defaultdict
import random 

import re 

import concurrent.futures
from functools import partial
import unicodedata

import nltk
from nltk.corpus import stopwords




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
    print(f"reading from path {path}")
    loaded = np.load(path)
    return torch.from_numpy(loaded["arr"])



root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
HELPERS_FOLDER = f"{root}/data/helpers"
CHECKPOINT_FOLDER = f"{root}/data/checkpoints"
TEMP_FOLDER = f"{root}/data/temp"


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
    "alias_patterns": f"{RESULTS_FOLDER}/aliases_patterns.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl",
    "rel_embs_tensor": f"{RESULTS_FOLDER}/rel_embs_tensor.npz",
    "silver_spans": {
        "head_start": f"{RESULTS_FOLDER}/ss_head_start.npz",
        "head_end": f"{RESULTS_FOLDER}/ss_head_end.npz",
        "tail_start": f"{RESULTS_FOLDER}/ss_tail_start.npz",
        "tail_end": f"{RESULTS_FOLDER}/ss_tail_end.npz",
        "sentence_tokens": f"{RESULTS_FOLDER}/ss_sentence_tokens.pkl",
        "desc_ids": f"{RESULTS_FOLDER}/desc_ids.pkl",

    },
}

HELPER_FILES = {
    "desc_half_ids": f"{HELPERS_FOLDER}/desc_half_ids.pkl",
    "my_triples": f"{HELPERS_FOLDER}/my_triples.pkl",
    "my_relations_keys": f"{HELPERS_FOLDER}/my_relations_keys.pkl",
    "new_desc_dict": f"{HELPERS_FOLDER}/new_desc_dict.pkl",


    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl",
    "stop_words": f"{HELPERS_FOLDER}/stop_words.pkl",
    "keys_not_in_als": f"{HELPERS_FOLDER}/keys_not_in_als.pkl",

    "descs_all": f"{HELPERS_FOLDER}/descs_all.pkl",
    "aliases_all": f"{HELPERS_FOLDER}/aliases_all.pkl",
    "triples_all": f"{HELPERS_FOLDER}/triples_all.pkl",
    "relations_all": f"{HELPERS_FOLDER}/relations_all.pkl",
    "descs_keys_min": f"{HELPERS_FOLDER}/descs_keys_min.pkl",

    "transe_prepared": f"{HELPERS_FOLDER}/transe_prepared.pkl"
}

TEMP_FILES = {
    "heads_aliases": f"{TEMP_FOLDER}/heads_aliases.pkl",
    "tails_aliases": f"{TEMP_FOLDER}/tails_aliases.pkl",
    "aliases_patterns": f"{TEMP_FOLDER}/aliases_patterns.pkl",
    "sentences_tokens": f"{TEMP_FOLDER}/sentences_tokens.pkl",
    "results_spans": f"{TEMP_FOLDER}/results_spans.pkl",
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

def save_stop_words():
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    cache_array(stop_words, HELPER_FILES["stop_words"])

def prep_raw():
    triples_fp = RAW_TXT_FILES["triples"]
    triples_dict = get_triples(triples_fp)
    descs_raw_f  = RAW_TXT_FILES["descriptions"]
    desc_dict_all = get_descriptions(descs_raw_f)
    relations_raw_fp = RAW_TXT_FILES["relations"]
    aliases_raw_fp = RAW_TXT_FILES["aliases"]

    aliases_dict_all = get_aliases(aliases_raw_fp)
    relations_dict_all = get_relations(relations_raw_fp)



def prepare_main_cpu_1(percentage=.5):
    """
        Get all triples, all descriptions 
        Make desc_keys as random sample from description keys 
        make my_triples which is triples having keys inside desc_keys 
        make my_tails which are keys of the tails of my_triples 
        make final description keys which are keys of desc_keys + keys of my_tails
        save final triples, final descriptions
    """
    return
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
    return
    final_desc_dict = read_cached_array( RESULT_FILES["descriptions_unormalized"])
    final_triples = read_cached_array(RESULT_FILES["triples"])

    my_tails  = [t for v in final_triples.values() for _,_,t in v]

    final_desc_dict_keys = final_desc_dict.keys()
    final_aliases_dict = {k:v for k,v in tqdm(aliases_dict_all.items()) if k in final_desc_dict_keys or k in my_tails}
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




    if os.path.exists(HELPER_FILES["desc_half_ids"]):
        desc_half_ids = read_cached_array(HELPER_FILES["desc_half_ids"])
        desc_half_ids_set = set(desc_half_ids)
    else:
        desc_dict = {k:v for k,v in desc_dict_all.items() if k in triples_dict}
        print(f"we have {len(desc_dict)} descriptions that has triples ")
        half_size = ceil(len(desc_dict) * percentage) 
        desc_half_ids = random.sample(list(desc_dict.keys()), half_size)
        desc_half_ids_set = set(desc_half_ids)
        cache_array(desc_half_ids,HELPER_FILES["desc_half_ids"] )
        print(f"we have now {len(desc_half_ids)} descriptions")

    if os.path.exists(HELPER_FILES["my_triples"]):
        my_triples = read_cached_array(HELPER_FILES["my_triples"])
    else:
        print("creating my triples")
        my_triples = {k: triples_dict[k] for k in desc_half_ids_set if k in triples_dict }
        cache_array(my_triples, HELPER_FILES["my_triples"])
        
    if os.path.exists(HELPER_FILES["my_relations_keys"]):
        my_relations_keys = read_cached_array(HELPER_FILES["my_relations_keys"])
        new_desc_dict = read_cached_array(HELPER_FILES["new_desc_dict"])
        new_desc_dict_set = set(new_desc_dict)
    else:
        print("creating my relations")
        my_tails = set()
        my_relations_keys = set()

        for trples_lst in my_triples.values():
            for _, r, t in trples_lst:
                my_relations_keys.add(r)
                my_tails.add(t)
        new_desc_dict_set = set(desc_half_ids_set | my_tails)
        new_desc_dict = list(new_desc_dict_set)
        cache_array(my_relations_keys, HELPER_FILES["my_relations_keys"])
        cache_array(new_desc_dict, HELPER_FILES["new_desc_dict"])
        
    


    if not os.path.exists(RESULT_FILES["descriptions_unormalized"]):
        descriptions_dict_unormalized = {k: desc_dict_all[k] for k in new_desc_dict_set if k in desc_dict_all}
        print(f"Description dict unormalized len: {len(descriptions_dict_unormalized)}")
        cache_array(descriptions_dict_unormalized, RESULT_FILES["descriptions_unormalized"])


    if  os.path.exists(RESULT_FILES["triples"]):
        final_triples = read_cached_array(RESULT_FILES["triples"])
    else:
        final_triples = {k: triples_dict[k] for k in  new_desc_dict_set if k in triples_dict   }
        print(f"Final triples len: {len(final_triples)}")
        cache_array(final_triples, RESULT_FILES["triples"])

    if not os.path.exists(RESULT_FILES["aliases"]):
        last_tails = [t for v in final_triples.values() for _,_,t in v]
        final_aliases = {k: aliases_dict_all[k] for k in set( new_desc_dict_set | set(last_tails) ) if k in aliases_dict_all}
        print(f"aliases dict has len: {len(final_aliases)}.")
        cache_array(final_aliases, RESULT_FILES["aliases"])

    if not os.path.exists(RESULT_FILES["relations"]):
        final_relations_dict = {k: relations_dict_all[k] for k in  my_relations_keys if k in relations_dict_all}
        print(f"relations dict len is: {len(final_relations_dict)}.") 
        cache_array(final_relations_dict, RESULT_FILES["relations"])

#endregion


#region normalize_descriptions

def replace_special_chars(text, compiled_patterns):
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text


def normalize_als_batch(als_batch_dict):
    stop_words = read_cached_array(HELPER_FILES["stop_words"])
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
            if aa != '' and als not in stop_words:
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
    print(len(descs))
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
    id2idx = {rel: i for i, rel in enumerate(rel_keys)}
    n_rels = len(rel_keys)
    hidden_size = bert_model.config.hidden_size
    sums   = torch.zeros(n_rels, hidden_size, device=device, dtype=torch.float32)
    counts = torch.zeros(n_rels,        device=device, dtype=torch.long)

    with torch.no_grad():
        for rel_ids, inputs in tqdm(loader, total=len(loader), desc="extracting relations embs" ):
            #move to device
            inputs = {k: v.to(device) for k,v in inputs.items()}
            if device_str == "cpu":
                with torch.autocast(device_type=device_str):
                    outs = bert_model(**inputs, output_hidden_states=True).hidden_states
            else:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outs = bert_model(**inputs, output_hidden_states=True).hidden_states

            avg_layers = (outs[-1] + outs[-2]) / 2.0
            mask = inputs["attention_mask"].unsqueeze(-1).to(avg_layers.dtype)
            emb = (avg_layers * mask).sum(1) / (mask.sum(1))
            
            for rel_id, vec in zip(rel_ids, emb):
                idx = id2idx[rel_id]
                sums[idx]   += vec
                counts[idx] += 1

    final  = (sums / counts.unsqueeze(-1)).cpu()
    return final


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
    # bert_model = torch.compile(bert_model)
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

#region alias_patterns_map
def create_alias_patterns_map():
    aliases_all_dict = read_cached_array(RESULT_FILES["aliases"])
    alias_patterns_map = {}
    for als_lst in tqdm(aliases_all_dict.values()):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r"\ ", r"\s*")
            pattern = rf"\b{flexible}\b"
            alias_patterns_map[als_str] = re.compile(pattern, re.IGNORECASE)
    cache_array(alias_patterns_map, RESULT_FILES["alias_patterns"])

#endregion

#region prepare_triples_for_transe

def do_transe_triples():
    print("preparing transe_triples")
    triples_dict = read_cached_array(RESULT_FILES["triples"])
    all_triples = [trpl for triples in triples_dict.values() for trpl in triples]
    entities = set()
    relations = set()

    for h, r, t in tqdm(all_triples):
        relations.add(r)
        entities.update([h,t])
    ent2id = {ent: idx for idx, ent in tqdm(enumerate(sorted(entities)))}
    rel2id = {rel: idx for idx, rel in enumerate(sorted(relations))}
    triples = torch.tensor([(ent2id[h], rel2id[r], ent2id[t]) for (h,r,t) in tqdm(all_triples, total=len(all_triples))   ] )
    
    neg_triples = triples.clone()
    n = triples.size(0)
    
    #false to corrupt head, true to corrupt tail
    mask = torch.randint(0,2, (n,), dtype=torch.bool)
    #one random ent per sample to corrupt with 
    random_ents = torch.randint(0, len(ent2id), (n,))

    # sample neg
    neg_triples[~mask, 0] = random_ents[~mask]
    neg_triples[mask, 2] = random_ents[mask]
    
    helper_dict = {
        "triples": triples,
        "neg_triples": neg_triples,
        "n_rels": len(rel2id),
        "n_ents": len(ent2id),
    }
    cache_array(helper_dict, HELPER_FILES["transe_prepared"])
#endregion

#region  prepare_head_tails_aliases

_ALIASES = read_cached_array(RESULT_FILES["aliases"])
_TRIPLES = read_cached_array(RESULT_FILES["triples"])



def create_heads_tails_aliases_batch(descs_batch_dict):
    descs_ids_chunk = list(descs_batch_dict.keys())
    print(f"new cpu worker proccessing {len(descs_ids_chunk)}")
    triples_keys = list(_TRIPLES.keys())
    tails_aliases = {}
    heads_aliases = {}
    for d_id in tqdm(descs_ids_chunk, total=len(descs_ids_chunk), desc="processing descs ids"):
        heads_aliases[d_id] = []
        tails_aliases[d_id] = []
        if d_id in _ALIASES:
            heads_aliases[d_id].append(_ALIASES[d_id])
        if d_id in triples_keys:
            for _, _, t in _TRIPLES[d_id]:
                if t in _ALIASES:
                    tails_aliases[d_id].append(_ALIASES[t])

    return tails_aliases, heads_aliases

def prep_heads_tails_aliases():
    print("reading dicts")
    descs = read_cached_array(RESULT_FILES["descriptions"])
    print(f"descs to process: {len(descs)}")
    batch_size =  len(descs) // NUM_WORKERS
    items = list(descs.items())
    batches = [
        dict(items[i : i+batch_size])
        for i in range(0, len(items), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        my_func = partial(create_heads_tails_aliases_batch)
        results = list(tqdm(
            executor.map(my_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    all_heads_aliases = {}
    all_tails_aliases = {}
    for tails, heads in results:
        all_tails_aliases.update(tails)
        all_heads_aliases.update(heads)

    cache_array(all_heads_aliases, TEMP_FILES["heads_aliases"])
    cache_array(all_tails_aliases, TEMP_FILES["tails_aliases"])

#endregion

if __name__ == "__main__":
    
    if device_str == "cpu":
        prepare_main_cpu_1(.00005)
        prepare_main_cpu_2()
    else:
        print("**********************CUDA*********")
        prepare_main(0.005)

    save_strange_chars_dict()
    save_stop_words()
    normalize_descriptions()
    prep_relations_main()

    normalize_aliases()
    create_alias_patterns_map()
    do_transe_triples()
    prep_heads_tails_aliases()