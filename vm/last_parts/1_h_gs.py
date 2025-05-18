# GPU-0: start: 0, end: 174186
# GPU-1: start: 174186, end: 348372
# GPU-2: start: 348372, end: 522558
# GPU-3: start: 522558, end: 696744



import os
import pickle
from math import ceil

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from transformers import BertModel
from torch.amp import autocast
from tqdm import tqdm 
import numpy as np 
import psutil


def print_available_ram_gb():
    path="/"
    mem = psutil.virtual_memory()
    gb_mem = mem.available / (1024 ** 3)
    disk = psutil.disk_usage(path)
    disk_free_gb = disk.free / (1024 ** 3)
    
    print(f"Available system RAM before cat: {gb_mem:.2f} GB, disk: {disk_free_gb:.2f} GB")


def save_tensor(tensor, path):
    try:
        print(f"Saving tensor to path: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, arr=tensor.cpu().numpy())
        print(f"Saved tensor → {path}")
    except Exception as e:
        print(f"[Error] Failed to save tensor at {path}. Reason: {e}")


# ─── Config ───────────────────────────────────────────────────────────────────
PART = 0

PARTS_NUM = 4 
GPU_NUM = 4

DESCRIPTION_MAX_LENGTH = 128
NUM_WORKERS           = 90
BATCH_SIZE            = 128
DATA_ROOT             = "./data"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print_available_ram_gb()
    # —— 1) INIT DDP & DEVICE ───────────────────────────────────────────────────
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # uses LOCAL_RANK & WORLD_SIZE internally :contentReference[oaicite:0]{index=0}
    print(f"local_rank: {local_rank} " )
    device = torch.device(f"cuda:{local_rank}")

    # —— 2) LOAD MODEL ──────────────────────────────────────────────────────────
    print(f"rank-{local_rank}: load model ")
    model = BertModel.from_pretrained("bert-base-cased").half().to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # wrap in DDP :contentReference[oaicite:1]{index=1}

    # —— 3) LOAD & SLICE YOUR TOKENIZED INPUTS ────────────────────────────────
    print(f"rank-{local_rank}: load slice ")
    enc_path = f"{DATA_ROOT}/dictionaries/encoded_tokenization.pkl"
    with open(enc_path, "rb") as f:
        encoded = pickle.load(f)

    
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    total = len(encoded["input_ids"])
    
    part_size = ceil(total / (GPU_NUM * PARTS_NUM) )
    start = part_size * (4 * PART + local_rank)
    end = min(start + part_size , total)
    
    print(f"GPU-{local_rank}: start: {start}, end: {end}")
    ids = encoded["input_ids"][start:end].pin_memory()
    masks = encoded["attention_mask"][start:end].pin_memory()
    
    # —— 4) DATASET + DISTRIBUTED SAMPLER + DATALOADER ────────────────────────
    print(f"rank-{local_rank}: load dataset ")
    dataset = TensorDataset(ids, masks)
    
    print(f"\trank-{local_rank}: loader")
    loader  = DataLoader(dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         pin_memory=True)
    print("finished preparing")
    # —— 5) INFERENCE LOOP ─────────────────────────────────────────────────────
    embs_parts = []
    mean_embed_parts = []

    
    embs_all = torch.empty((end-start, DESCRIPTION_MAX_LENGTH , 768), dtype=torch.float16)
    mean_embs_all = torch.empty((end-start, 768), dtype=torch.float16)
    
    
    start_idx = 0
    with torch.no_grad():
        for batch_ids, batch_masks in tqdm(loader,  total =len(loader)):
            batch_size = batch_ids.size(0)
            
            
            batch_ids   = batch_ids.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            
            
            outputs = model(batch_ids, batch_masks)
            emb = outputs.last_hidden_state
            
            
            #calc average
            m_exp = batch_masks.unsqueeze(-1).float()
            summed = (emb * m_exp).sum(dim=1)
            counts = m_exp.sum(dim=1).clamp(min=1)
            means = summed / counts
            
            end_idx = start_idx + batch_size
            embs_all[start_idx:end_idx] = emb.cpu()
            mean_embs_all[start_idx:end_idx] = means.cpu()
            start_idx = end_idx
            
            

    
    print_available_ram_gb()        
    print("torch catting")
    
    # —— 6) SAVE YOUR PARTITIONED OUTPUT ──────────────────────────────────────
    print("saving")
    save_tensor(embs_all , f"{DATA_ROOT}/temp/draft_embs_part_{PART}_{local_rank}.npz")
    save_tensor(mean_embs_all , f"{DATA_ROOT}/temp/draft_meanembs_part_{PART}_{local_rank}.npz")
    print_available_ram_gb()
    

    # —— CLEAN UP ──────────────────────────────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
