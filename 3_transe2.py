from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Data import PKLS_FILES
from utils.utils import cache_array, read_cached_array
import os

import torch
import random
import numpy as np

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
    
    mask = torch.randint(high=2, size=(batch_size,))
    for i in range(batch_size):
        if mask[i] == 0:
            neg_triples[i, 0] = torch.randint(high=n_entities, size=(1,))
        else:
            neg_triples[i, 2] = torch.randint(high=n_entities, size=(1,))
    return neg_triples

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
    
def calc_loss(pos_dist, neg_dist, margin):
    #loss = mean (max(0, margin + pos_dist - neg_dist))
    return torch.clamp(margin+pos_dist - neg_dist, min =0).mean()

def train_model(triples_dict, n_ents, n_rels, emb_dim=100, lr=0.001, num_epochs=1000, batch_size=1024, num_workers=8, checkpoint_path="checkpoint.pth"):
    margin = 1.0 
    p_norm = 1
    
    dataset = TripleDataset(triples_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    model = TransEModel(n_ents, n_rels, emb_dim=emb_dim, margin=margin, p_norm=p_norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=lr )
    start_epoch, last_total_loss = load_checkpoint(model, optimizer, filename=checkpoint_path)
    
    with tqdm(range(start_epoch - 1, num_epochs), desc="Training Epochs", total=(num_epochs-start_epoch - 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            total_loss = last_total_loss
            batch_count = 0
            for pos_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                pos_batch = pos_batch.long()
                
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

    for t_head, tr_lst in triples_dict.items():
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
            
if __name__=="__main__":
    k = 100
    
    triples_dict = read_cached_array(PKLS_FILES["triples"][k])
    
    n_ents, n_rels, triples =   prep_triples(triples_dict)
    print(f"We have {n_ents} entities, {n_rels} relationships and {triples.shape} triples")
    #I did for k=1_000 num_epochs = 140 and it was nice, for full I did batch_size: 52224, but I did some modifications to laverage big gpu
    rel_embeddings = train_model(triples, n_ents, n_rels, emb_dim=100, lr=0.001, num_epochs=140, batch_size=24)
    print(f"relation embeddings shape: {rel_embeddings.shape}")
    cache_array(rel_embeddings,  PKLS_FILES["transE_relation_embeddings"])
    

