
# pip install numpy  tqdm
# pip install torch 
# pip install scikit-learn 
# pip install pykeen

#install raw files: 

# mkdir data
# cd data
# mkdir raw 
# cd raw 
# curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
# curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
# curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar
# cd ..
# cd ..

# python -m venv myenv 
# source myenv/bin/activate 
# pip install numpy tqdm scikit-learn 
# pip install pykeen


# export TORCH_CUDA_ARCH_LIST="12.0"
# pip install torch
# pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118


import os

import pickle
from tqdm import tqdm
from collections import defaultdict
import random
import itertools
import numpy as np
import torch
import re 
import concurrent.futures
from functools import partial
import unicodedata
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader






root_path = os.path.dirname(os.path.abspath(__file__))
root = "."

RAW_FOLDER = f"{root}/data/raw"
HELPERS_FOLDER = f"{root}/data/helpers"
TRANSE_FOLDER = f"{root}/data/transe"
TRANSE_CHECKPOINT_FOLDER = f"{root}/data/transe/checkpoint"
DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
DESCRIPTIONS_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
RELATIONS_FOLDER = f"{DICTIONARIES_FOLDER}/relations"

folders_to_check = [TRANSE_CHECKPOINT_FOLDER, TRANSE_FOLDER, DESCRIPTIONS_NORMALIZED_FOLDER, HELPERS_FOLDER, TRIPLES_FOLDER, RELATIONS_FOLDER,DICTIONARIES_FOLDER, DESCRIPTIONS_FOLDER]
for fo in folders_to_check:
    if not os.path.exists(fo):
        os.makedirs(fo)


RAW_TXT_FILES ={
    "descriptions": f"{RAW_FOLDER}/wikidata5m_text.txt",    
    "aliases": f"{RAW_FOLDER}/wikidata5m_entity.txt", 
    "relations": f"{RAW_FOLDER}/wikidata5m_relation.txt",
    "triples": f"{RAW_FOLDER}/wikidata5m_transductive_train.txt",
}



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
        1_000_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1m.pkl"
    },
    "aliases_dict": f"{DICTIONARIES_FOLDER}/aliases_dict.pkl",
    "aliases_rev": f"{DICTIONARIES_FOLDER}/aliases_rev.pkl",
    
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
    "transE_relation_embeddings": f"{TRANSE_FOLDER}/relation_embs.pkl" ,
    "transE_entity_embeddings": f"{TRANSE_FOLDER}/entity_embs.pkl" ,
}



HELPER_FILES = {
    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl"
}

def get_min_descriptionsNorm_triples_relations(k):
    min_desc_norm_dict_f = PKLS_FILES["descriptions_normalized"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    
    descs = read_cached_array(min_desc_norm_dict_f)
    triples = read_cached_array(min_triples_dict_f)
    relations = read_cached_array(min_relations_dict_f)
    
    return descs, triples, relations

def get_compiled_strange_chars():
    f = HELPER_FILES["strange_chars"]
    return  read_cached_array(f)




#region pickles
def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")
    
    
def read_cached_array(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
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









# will create descriptions dict from the raw text file and save the result in pickle file
#    descriptions_dict would have {document_id: description}
def create_descriptions_dict():
    desc_raw_f = RAW_TXT_FILES["descriptions"]
    descriptions_dict_f = PKLS_FILES["descriptions"]["full"]
    print(f"Full description is being created and will be saved in {descriptions_dict_f}...")
    descriptions_dict = load_descriptions_dict_from_text(desc_raw_f)
    cache_array(descriptions_dict, descriptions_dict_f)


# will create two aliases dicts from the raw text file of aliases and save them in pickle files
#    aliases_dict would have {alias_name: document_id}
#    aliases_dict_rev would have {document_id: [alias_name1, alias_name2, ...]}
def create_aliases_dicts():
    raw_f = RAW_TXT_FILES["aliases"]
    dict_f = PKLS_FILES["aliases_dict"]
    dict_rev_f = PKLS_FILES["aliases_rev"]
    
    
    print(f"Aliases are being created and will be saved in {dict_f} and {dict_rev_f}...")
    aliases_dict, aliases_rev = load_aliases_dict_from_text(raw_f)
    cache_array(aliases_dict, dict_f)
    cache_array(aliases_rev, dict_rev_f)

#   triples_dict would have {head: [triple1, triple2, ...]} and each triple is a triple (head, relation, tail)
#    head, relation and tail are all ids. head, tail id refering to document_id and relation id refering to relation_id
def create_triples_dict():
    raw_f = RAW_TXT_FILES["triples"]
    dict_f = PKLS_FILES["triples"]["full"]
    triples_dict = load_triples(raw_f)
    print(f"Saving triples in {dict_f}:")
    cache_array(triples_dict, dict_f)
    

#  relations_dict would have {relation_id: [relation_name1, relation_name2, ...]}
def create_relations_dict():
    raw_f = RAW_TXT_FILES["relations"]
    dict_f = PKLS_FILES["relations"]["full"]
    relations = load_relations(raw_f)
    print(f"Saving relations in {dict_f}:")
    cache_array(relations, dict_f)
    


# taking k from [10, 100, 1k, 10k, 1m] -> will not be exactly the number because we will add other documents 
#    will create desc_min_dict which is random k values from the full descriptions 

def create_min(k):
    
    print("read dictionaries..")
    desc_all_dict = read_cached_array(PKLS_FILES["descriptions"]["full"])
    triples_all_dict = read_cached_array(PKLS_FILES["triples"]["full"])
    relations_all_dict = read_cached_array(PKLS_FILES["relations"]["full"])
    
    print("minimizing descriptions..")
    desc_min_dict = minimize_dict(desc_all_dict, k)
    
    
    desc_ids_to_add = set(desc_min_dict.keys())    
    relation_ids_to_add = set()
    
    min_triples = defaultdict(list)
    for d_id in tqdm(desc_min_dict.keys(), total = len(desc_min_dict), desc="getting triples from descriptions" ):
        triples_lst = triples_all_dict[d_id]
        for _, r, t in triples_lst:
            relation_ids_to_add.add(r)
            desc_ids_to_add.add(t)
        min_triples[d_id] = triples_lst
    
    print("Creating full descriptions and full relations")
    last_min_desc_dict = {}
    for d_id in list(desc_ids_to_add):
        last_min_desc_dict[d_id] = desc_all_dict[d_id]
    
    last_min_relations_dict = {}
    for r_id in list(relation_ids_to_add):
        if r_id in relations_all_dict:
            last_min_relations_dict[r_id] = relations_all_dict[r_id]
    
    print(f"We have {len(last_min_desc_dict)} descriptions")
    print(f"We have {len(min_triples)} head triples")
    print(f"We have {len(last_min_relations_dict)} relations")
    
    min_desc_dict_f = PKLS_FILES["descriptions"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    
    
    cache_array(last_min_desc_dict, min_desc_dict_f)
    cache_array(min_triples, min_triples_dict_f)
    cache_array(last_min_relations_dict, min_relations_dict_f)
    



def save_strange_chars_dict():
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



def normalize_description_BRASK(sentence):
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def normalize_desc_batch_BRASK(descs_batch):
    compiled_strange_patterns = get_compiled_strange_chars()
    descs_batch = {k:  replace_special_chars(v, compiled_strange_patterns) for k,v in descs_batch.items()}
    
    
    new_dict=  {}
    for desc_id, desc in tqdm(descs_batch.items(), total=len(descs_batch.keys()), desc="Normalizing batch description"):
        new_dict[desc_id]= normalize_description_BRASK(desc)
  
    return new_dict

    
    
def normalize_desc_parallel(descs_all, num_workers = 8):
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




seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class TripleDataset(Dataset):
    
    def __init__(self, triples):
        
        self.triples = triples 
        
    def __len__(self):
        return self.triples.shape[0]
    
    def __getitem__(self, idx):
        return self.triples[idx]



def negative_sampling(pos_triples, n_entities):
    neg_triples = pos_triples.clone()
    batch_size = pos_triples.shape[0]
    
    mask = torch.randint(0, 2, (batch_size,), device=pos_triples.device, dtype=torch.bool)
    random_entities = torch.randint(0, n_entities, (batch_size,), device=pos_triples.device)
    neg_triples[~mask, 0] = random_entities[~mask]
    neg_triples[mask, 2] = random_entities[mask]
    
    return neg_triples




def save_checkpoint(model, optimizer, epoch, last_total_loss, filename="./checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_total_loss": last_total_loss
    }
    torch.save(checkpoint, filename)


def save_trps_prep(triples, n_ents, n_rels, filename="./trps.pkl"):
    trps_prep = {
        "triples": triples,
        "n_ents": n_ents,
        "n_rels": n_rels,
    }
    cache_array(trps_prep, filename)

def load_trps_prep(filename="./trps.pkl"):
    if os.path.exists(filename):
        print(f"\t Loading file {filename}")
        checkpoint = read_cached_array(filename)
        triples = checkpoint["triples"] 
        n_ents = checkpoint["n_ents"] 
        n_rels = checkpoint["n_rels"] 
        return triples, n_ents, n_rels
    else:
        return [], 0 , 0
    
def load_checkpoint(model, optimizer, filename="./checkpoint.pth"):
    if os.path.exists(filename):
        print(f"\t Loading file {filename}")
        checkpoint = torch.load(filename)
        print(f"\t loading state model")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\t loading state optimizer")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        last_total_loss = checkpoint["last_total_loss"]
        print(f"\t Resumed from epoch {start_epoch} with last_total_loss {last_total_loss:.4f}")
        return start_epoch, last_total_loss
    else:
        return 0, 0.0
    
#transe model: 
# we have entity e which is embeded by a vector e 
# we have relation r which is embeded by a vector r 
# We have a score for triple which is || h + r - t || 
# The goal is to make distiance small for positive triples and big for negative sample by at least a margin 

class TransEModel(nn.Module):
    def __init__(self, n_entities, n_relations, emb_dim=100, margin=1.0, p_norm=1):
        super(TransEModel, self).__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = emb_dim
        self.margin = margin
        self.p_norm = p_norm

        self.ent_embs = nn.Embedding(n_entities, emb_dim)
        self.rel_embs = nn.Embedding(n_relations, emb_dim)
        
        # init embeds with xavier distribution 
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)
        
    def forward(self, pos_triples, neg_triples):
        # pos_triples, neg_triples have shape (batch_size, 3), each row (head, relation, tail)
        # returns distance of positive triples, distance of negative triples 
        
        #positive: 
        
        pos_heads, pos_rels, pos_tails = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]

        #negative: 
        neg_heads, neg_rels, neg_tails = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]

       
        #emb lookup
        pos_head_entity = self.ent_embs(pos_heads)
        pos_rel_entity = self.rel_embs(pos_rels)
        pos_tail_entity = self.ent_embs(pos_tails)
        
        neg_head_entity = self.ent_embs(neg_heads)
        neg_rel_entity = self.rel_embs(neg_rels)
        neg_tail_entity = self.ent_embs(neg_tails)
        
        # compuite neg and pos distance  || h + r - t|| _{p_norm}
        pos_dist = torch.norm(pos_head_entity + pos_rel_entity - pos_tail_entity, dim=1)
        neg_dist = torch.norm(neg_head_entity + neg_rel_entity - neg_tail_entity, dim=1)
        
        return pos_dist, neg_dist



def calc_loss(pos_dist, neg_dist, margin):
    #loss = mean (max(0, margin + pos_dist - neg_dist))
    return torch.clamp(margin+pos_dist - neg_dist, min =0).mean()
import os

def train_model(triples_dict, n_ents, n_rels, emb_dim=100, lr=0.001, num_epochs=1000, batch_size=1024, num_workers=8, checkpoint_path="./checkpoint.pth"):
    margin = 1.0 
    p_norm = 1
    
    dataset = TripleDataset(triples_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    
    model = torch.compile(TransEModel(n_ents, n_rels, emb_dim=emb_dim, margin=margin, p_norm=p_norm))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr )
    
    start_epoch, last_total_loss = load_checkpoint(model, optimizer, filename=checkpoint_path)
    
    with tqdm(range(start_epoch - 1, num_epochs), desc="Training Epochs", total=(num_epochs-start_epoch - 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            total_loss = last_total_loss
            batch_count = 0
            for pos_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                pos_batch = pos_batch.long().to(device)
                
                
                neg_batch = negative_sampling(pos_batch, n_entities=n_ents)
                
                #forward pass:
                pos_dist, neg_dist = model(pos_batch, neg_batch)
                loss = calc_loss(pos_dist=pos_dist, neg_dist=neg_dist, margin=margin)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                
                #normalize embs to unit length
                model.ent_embs.weight.data = F.normalize(model.ent_embs.weight.data, p=2, dim=1)
                model.rel_embs.weight.data = F.normalize(model.rel_embs.weight.data , p=2, dim=1)
                
                total_loss += loss.item()
                batch_count += 1
                
                
                
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            pbar_epoch.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, total_loss, filename=checkpoint_path)
    return  model.rel_embs.weight.data
def prep_triples(triples_dict):
    ents = set()
    rels = set()
    triples_ar = []    

    for _, tr_lst in tqdm(triples_dict.items(), total=len(triples_dict), desc="preparing triples.."):
        triples_ar.extend(tr_lst)
        for (h,r,t) in tr_lst:
            ents.add(h)
            ents.add(t)
            rels.add(r)
    #converting to torch tensor: 
    ents_ids = {ent: idx for idx, ent in enumerate(ents)}
    rel_ids = {rel: idx for idx, rel in enumerate(rels)}
    n_ar = [ (ents_ids[h], rel_ids[r], ents_ids[t])   for (h,r,t) in triples_ar  ]


    triples_ar = torch.tensor(n_ar)
    return len(ents), len(rels), triples_ar
            


if __name__ == '__main__':
    k = "full"
    num_workers = 56
    
    # print("creating descs")
    # save_strange_chars_dict()
    # create_descriptions_dict()
    # print("creating als")
    # create_aliases_dicts()
    # print("creating triples")
    # create_triples_dict()
    # print("creating relations")
    # create_relations_dict()
    
    # print("reading.. ")
    # descriptions = read_cached_array(PKLS_FILES["descriptions"][k])
    # print("normalizing.. ")
    # normalized_descs = normalize_desc_parallel(descriptions, num_workers)
    # cache_array(normalized_descs, PKLS_FILES["descriptions_normalized"][k])
    
    
    
    triples, n_ents, n_rels = load_trps_prep()
    if n_rels == 0:
        print("reading trreiples")
        triples_dict = read_cached_array(PKLS_FILES["triples"][k])
        print("preparing triples..")
        n_ents, n_rels, triples =   prep_triples(triples_dict)
        save_trps_prep(triples, n_ents, n_rels)
    
    
    print(f"We have {n_ents} entities, {n_rels} relationships and {triples.shape} triples")
    rel_embeddings = train_model(triples, n_ents, n_rels, emb_dim=100, lr=0.001, num_epochs=140, batch_size=52224, num_workers=num_workers)
    print(f"relation embeddings shape: {rel_embeddings.shape}")
    cache_array(rel_embeddings,  PKLS_FILES["transE_relation_embeddings"])
    