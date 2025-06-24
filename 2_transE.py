

import torch


device_str = 'cpu'
if torch.cuda.is_available():
  device_str = "cuda"

device = torch.device(device_str)



if device_str == 'cpu':
    TRANSE_BATCH_SIZE = 32
    TRANSE_NUM_EPOCHS = 124
    TRANSE_EMB_DIM = 100
    NUM_WORKERS = 1


else:
    TRANSE_BATCH_SIZE = 52224
    TRANSE_NUM_EPOCHS = 140
    TRANSE_EMB_DIM = 100
    NUM_WORKERS = 48



from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler  
from torch.nn.parallel import DistributedDataParallel as DDP

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


import concurrent.futures
import multiprocessing
from functools import partial
import unicodedata

import nltk
from nltk.corpus import words




def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")

def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
HELPERS_FOLDER = f"{root}/data/helpers"
CHECKPOINT_FOLDER = f"{root}/data/checkpoints"

RESULT_FILES  = {
    "descriptions_unormalized": f"{RESULTS_FOLDER}/descriptions_unormalized.pkl",
    "descriptions": f"{RESULTS_FOLDER}/descriptions.pkl",
    "aliases": f"{RESULTS_FOLDER}/aliases.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl"
}

CHECKPOINTS_FILES = {
    "transe_triples": f"{CHECKPOINT_FOLDER}/transe_triples.pkl",
    "transe_model": f"{CHECKPOINT_FOLDER}/transe_model.pth",
}

#region transe

def do_transe_triples(triples_dict):
    all_triples = [trpl for triples in triples_dict.values() for trpl in triples]
    entities = set()
    relations = set()

    rel2id = {rel: idx  for _,r,_ in all_triples for idx, rel in  enumerate(r)}
    ls = []

    for h,r,t in all_triples:
        entities.update([h,t])
        relations.add(r)

    ent2id = {ent: idx  for idx, ent in enumerate(sorted(entities)) }
    rel2id = {rel: idx  for idx, rel in enumerate(sorted(relations)) }
    triples_tensor  = torch.tensor([(ent2id[h], rel2id[r], ent2id[t]) for (h, r, t) in all_triples])
    return len(entities), len(relations), triples_tensor 

def prepare_triples():
    checkpoint_f = CHECKPOINTS_FILES["transe_triples"]
    if os.path.exists(checkpoint_f):
        print("reading transe triples from cache checkpoint ")
        checkpoint = read_cached_array(checkpoint_f)
        return  checkpoint["triples"], checkpoint["n_ents"], checkpoint["n_rels"]
    else:
        print("creating transe triples.. ")
        triples_dict = read_cached_array(RESULT_FILES["triples"])
        print(f"We have {len(triples_dict)} triples")
        n_ents, n_rels, triples = do_transe_triples(triples_dict)
        chckpoint = {
            "triples": triples,
            "n_ents": n_ents, 
            "n_rels": n_rels
        }
        cache_array(chckpoint, checkpoint_f)
        return triples, n_ents, n_rels 


def save_transe_checkpoint(model, optimizer, epoch, last_total_loss, filename):
    checkpoint = {
        "epoch": epoch,
        "last_total_loss": last_total_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_transe_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        print("\tloading transe checkopint from history")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        last_total_loss = checkpoint["last_total_loss"]
        print(f"\tResuming from epoch {start_epoch} with last total loss {last_total_loss} ")
        return start_epoch, last_total_loss
    else:
        return 0, 0.0




class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return self.triples.shape[0]
    def __getitem__(self, idx):
        return self.triples[idx]


class TransEModel(nn.Module):
    def __init__(self, n_ents, n_rels, margin):
        super(TransEModel, self).__init__()

        self.p_norm = 1

        self.n_entities = n_ents
        self.n_relations = n_rels 
        self.embedding_dim = TRANSE_EMB_DIM 

        self.ent_embs = nn.Embedding(n_ents, TRANSE_EMB_DIM)
        self.rel_embs = nn.Embedding(n_rels, TRANSE_EMB_DIM)

        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)


    def forward(self, pos_triples, neg_triples):
        """
            pos_triples, neg_triples has shape (batch_size, 3), 3 because (head, relation, tail)
            forward() will return distance of positive triples, distance of negative triples 
        """
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
    return torch.clamp(margin + pos_dist - neg_dist, min=0).mean()

def train_transE_model_gpu(triples, n_ents, n_rels,lr,local_rank ):
    num_epochs, batch_size, num_workers, emb_dim = TRANSE_NUM_EPOCHS, TRANSE_BATCH_SIZE, NUM_WORKERS, TRANSE_EMB_DIM
    margin = 1.0 

    dataset = TripleDataset(triples)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True)


    model = DDP(torch.compile(TransEModel(n_ents, n_rels, margin).cuda(local_rank)),
            device_ids=[local_rank])
    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model_checkpoint_path = CHECKPOINTS_FILES["transe_model"].replace(".pth", f"_{local_rank}.pth")

    start_epoch, last_total_loss = 0, 0.0
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRANSE_NUM_EPOCHS)

    # start_epoch, last_total_loss = load_transe_checkpoint(model, optimizer, model_checkpoint_path)
    with tqdm(range(start_epoch, num_epochs), total=(num_epochs - start_epoch) , desc="training epochs") as epochs_bar:
        for epoch in epochs_bar:
            sampler.set_epoch(epoch)
            model.train()
            total_loss = last_total_loss


            batch_count = 0
            for pos_batch in tqdm(dataloader, desc=f"epoch {epoch+1}", leave=False):
                pos_batch = pos_batch.cuda(local_rank, non_blocking=True)

                neg_triples = pos_batch.clone()
                batch_size = pos_batch.size(0)

                mask = torch.randint(0,2 , (batch_size,), device = pos_batch.device, dtype=torch.bool)
                random_ents = torch.randint(0, n_ents, (batch_size,), device=pos_batch.device)
                neg_triples[~mask, 0] = random_ents[~mask]
                neg_triples[mask, 2] = random_ents[mask]
                with autocast():
                    #forward:         
                    pos_dist, neg_dist = model(pos_batch, neg_triples)
                    loss = calc_loss(pos_dist=pos_dist, neg_dist=neg_dist, margin=margin)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                model.ent_embs.weight.data = F.normalize(model.ent_embs.weight.data, p=1, dim=1)
                model.rel_embs.weight.data = F.normalize(model.rel_embs.weight.data, p=1, dim=1)

                total_loss += loss.item() 
                batch_count += 1

            scheduler.step()
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0 
            epochs_bar.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss: .4f}")
            # save_transe_checkpoint(model, optimizer, epoch, total_loss, filename=model_checkpoint_path)
    dist.destroy_process_group()
    return  model.rel_embs.weight.data


def train_transE_model_cpu(triples, n_ents, n_rels,lr ):
    num_epochs, batch_size, num_workers, emb_dim = TRANSE_NUM_EPOCHS, TRANSE_BATCH_SIZE, NUM_WORKERS, TRANSE_EMB_DIM
    margin = 1.0 

    dataset = TripleDataset(triples)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0)


    
    model =  TransEModel(n_ents, n_rels, margin)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch, last_total_loss = 0, 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRANSE_NUM_EPOCHS)

    with tqdm(range(start_epoch, num_epochs), total=(num_epochs - start_epoch) , desc="training epochs") as epochs_bar:
        for epoch in epochs_bar:
            model.train()
            total_loss = last_total_loss

            print("************************ EPOCH STARTED")
            batch_count = 0
            for pos_batch in tqdm(dataloader, desc=f"epoch {epoch+1}", leave=False):
                pos_batch = pos_batch.to(device)

                neg_triples = pos_batch.clone()
                batch_size = pos_batch.size(0)
                mask = torch.randint(0,2 , (batch_size,), device = pos_batch.device, dtype=torch.bool)
                random_ents = torch.randint(0, n_ents, (batch_size,), device=pos_batch.device)
                neg_triples[~mask, 0] = random_ents[~mask]
                neg_triples[mask, 2] = random_ents[mask]

                pos_dist, neg_dist = model(pos_batch, neg_triples)
                loss = calc_loss(pos_dist=pos_dist, neg_dist=neg_dist, margin=margin)
                loss.backward()
                optimizer.step()




                model.ent_embs.weight.data = F.normalize(model.ent_embs.weight.data, p=1, dim=1)
                model.rel_embs.weight.data = F.normalize(model.rel_embs.weight.data, p=1, dim=1)

                total_loss += loss.item() 
                batch_count += 1
            print("************************ EPOCH finsihed")

            scheduler.step()
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0 
            epochs_bar.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss: .4f}")
            # save_transe_checkpoint(model, optimizer, epoch, total_loss, filename=model_checkpoint_path)
    return  model.rel_embs.weight.data


def transe_main():
    triples, n_ents, n_rels = prepare_triples()
    print(f"We have {n_ents} entities, {n_rels} relationships and {triples.shape} triples")
    if device_str == 'cpu':
        rel_embeddings = train_transE_model_cpu(triples, n_ents, n_rels, 0.001)
        print(f"relation embeddings shape: {rel_embeddings.shape}")
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # dist.init_process_group(backend='nccl')
        rel_embeddings = train_transE_model_gpu(triples, n_ents, n_rels, 0.001, local_rank)
        print(f"relation embeddings shape: {rel_embeddings.shape}")

    cache_array(rel_embeddings,  RESULT_FILES["transE_relation_embeddings"])



#endregion
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn', force=True)
    transe_main()
