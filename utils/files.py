import pickle
import os
import numpy as np
import torch
from tqdm import tqdm



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

def scan_text_file_lines(fp, scan_head_ids= False):
    total = 0
    head_ids = set()
    with open(fp, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"scanning {fp} lines"):
            if scan_head_ids:
                head_ids.add(line.split("\t", 1)[0])
            total +=1
    return total, head_ids