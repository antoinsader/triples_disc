

import os
import pickle
from math import ceil

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
import torch.nn.functional as F


from tqdm import tqdm 
import numpy as np 
import psutil


# for 589 rels --> batch_size = 256
BATCH_SIZE = 256
NUM_WORKERS =  75
GPU_NUM = 4
PIN_MEMORY = True
DATA_FILE = f"./data/helpers/transe_prepared.pkl"
OUT_FILE = f"./data/results/transE_rel_embs.npz"

MARGIN = 1.0 
TRANSE_EMB_DIM = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 120

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


def print_available_ram_gb():
    path="/"
    mem = psutil.virtual_memory()
    gb_mem = mem.available / (1024 ** 3)
    disk = psutil.disk_usage(path)
    disk_free_gb = disk.free / (1024 ** 3)
    print(f"Available system RAM before cat: {gb_mem:.2f} GB, disk: {disk_free_gb:.2f} GB")


class TransEDataset(Dataset):
    def __init__(self):
        dic = read_cached_array(DATA_FILE)
        self.triples, self.neg_triples, self.n_rels, self.n_ents = dic["triples"], dic["neg_triples"], dic["n_rels"], dic["n_ents"]
    def __len__(self):
        return self.triples.shape[0]
    def __getitem__(self,idx):
        return self.triples[idx], self.neg_triples[idx]


class TransEModel(nn.Module):
    def __init__(self, n_ents, n_rels):
        super(TransEModel, self).__init__()
        self.p_norm = 1
        self.emb_dim = TRANSE_EMB_DIM

        self.ent_embs = nn.Embedding(n_ents, TRANSE_EMB_DIM)
        self.rel_embs = nn.Embedding(n_rels, TRANSE_EMB_DIM)

        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward(self, pos_triples, neg_triples):
        #pos
        pos_heads, pos_rels, pos_tails = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        pos_head_entity = self.ent_embs(pos_heads)
        pos_rel = self.rel_embs(pos_rels)
        pos_tail_entity = self.ent_embs(pos_tails)

        #neg
        neg_heads, neg_rels, neg_tails = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        neg_head_entity = self.ent_embs(neg_heads)
        neg_rel = self.rel_embs(neg_rels)
        neg_tail_entity = self.ent_embs(neg_tails)

        #compute neg and pos distance || h + r - t|| _{p_norm}
        pos_dist = torch.norm(pos_head_entity + pos_rel - pos_tail_entity, dim =1)
        neg_dist = torch.norm(neg_head_entity + neg_rel - neg_tail_entity, dim =1)

        return pos_dist, neg_dist

def main():
    print_available_ram_gb()

    #init DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    print(f"LOCAL RANK: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")


    #Load dataloader
    dataset = TransEDataset()
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)



    #LOAD MODEL
    model = TransEModel(dataset.n_ents, dataset.n_rels).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #Opt and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    #TRAIN
    for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS, desc=f"Epochs, [GPU {local_rank}]"):
        model.train()
        total_loss = 0.00
        for pos_batch, neg_batch in tqdm(loader, desc=f"epoch {epoch + 1}, [GPU {local_rank}]"):
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=True):
                pos_dist, neg_dist = model(pos_batch, neg_batch)
                loss = torch.clamp(MARGIN + pos_dist - neg_dist, min=0).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.module.ent_embs.weight.data = F.normalize(model.module.ent_embs.weight.data, p=1, dim=1)
            model.module.rel_embs.weight.data = F.normalize(model.module.rel_embs.weight.data, p=1, dim=1)

            total_loss += loss.item()
        print(f"[GPU {local_rank}] Epoch {epoch+1} Loss: {total_loss:.4f}")
        scheduler.step()

    dist.barrier()
    dist.destroy_process_group()
    if local_rank == 0:
        save_tensor(model.module.rel_embs.weight.data.cpu(),OUT_FILE)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
