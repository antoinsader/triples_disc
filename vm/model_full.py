
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



#region pickles
def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")
    
    
def read_cached_array(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
#endregion










seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






def save_checkpoint(model, optimizer, epoch, last_total_loss, filename="./checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_total_loss": last_total_loss
    }
    torch.save(checkpoint, filename)


def save_prep(n_ents, n_rels, filename="./preps.pkl"):
    prep = {
        "n_ents": n_ents,
        "n_rels": n_rels,
    }
    cache_array(prep, filename)

def load_preps(filename="./preps.pkl"):
    if os.path.exists(filename):
        print(f"\t Loading file {filename}")
        checkpoint = read_cached_array(filename)
        n_ents = checkpoint["n_ents"] 
        n_rels = checkpoint["n_rels"] 
        return  n_ents, n_rels
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
        # last_total_loss = checkpoint["last_total_loss"]
        last_total_loss = 0.0
        print(f"\t Resumed from epoch {start_epoch} with last_total_loss {last_total_loss:.4f}")
        return start_epoch, last_total_loss
    else:
        return 0, 0.0



class BraskModel(nn.Module):
    def __init__(self, ):
        super(BraskModel, self).__init__()
        pass
    def forward(self, ):
        pass 
        return 



def calc_loss(pos_dist, neg_dist, margin):
    pass

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
                
                
                
                pos_dist, neg_dist = model(pos_batch, neg_batch)
                loss = calc_loss(pos_dist=pos_dist, neg_dist=neg_dist, margin=margin)
                


                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                

                total_loss += loss.item()



                batch_count += 1
                
                
                
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            pbar_epoch.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, total_loss, filename=checkpoint_path)
    return None 

            


if __name__ == '__main__':
    k = "full"
    