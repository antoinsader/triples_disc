import torch

MODEL_PATH = "./data/BRASK_MODEL.pth"

device_str = 'cpu'
if torch.cuda.is_available():
  device_str = "cuda"
  

device = torch.device(device_str)

THRESHOLD = .5
if device_str == 'cpu':
    NUM_WORKERS = 0
    BATCH_SIZE = 24
    NUM_SAMPLES = 100
    PIN_MEMORY = False
    TENSOR_DTYPE = torch.float32 

else:
    NUM_WORKERS = 86
    BATCH_SIZE = 1024
    NUM_SAMPLES = 1000
    PIN_MEMORY = True
    TENSOR_DTYPE = torch.float16 



import os
import numpy as np
import pickle
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim

from datasets import Dataset as HFDataset
from transformers import BertModel, BertTokenizerFast


root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"



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
    "embs": {
        "hgs": f"{RESULTS_FOLDER}/descriptions_hgs.npz",
        "embs": f"{RESULTS_FOLDER}/descriptions_embs.npz",
    }
}


def save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    np.savez_compressed(path, arr=tensor.cpu().numpy())
    print(f"Tensor chached in file {path}")



def read_tensor(path):
    print(f"reading from path {path}")
    loaded = np.load(path)
    return torch.from_numpy(loaded["arr"])


def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")






class BRASKDataset(Dataset):
    def __init__(self):
        self.descriptions_all = read_cached_array(RESULT_FILES["descriptions"])
        self.h_s = read_tensor(RESULT_FILES["silver_spans"]["head_start"])
        self.h_e = read_tensor(RESULT_FILES["silver_spans"]["head_end"])
        self.t_s = read_tensor(RESULT_FILES["silver_spans"]["tail_start"]) # (B, L)
        self.t_e = read_tensor(RESULT_FILES["silver_spans"]["tail_end"])
        self.descs_ids = read_cached_array(RESULT_FILES["silver_spans"]["desc_ids"])

        self.h_gs = read_tensor(RESULT_FILES["embs"]["hgs"])
        self.embs = read_tensor(RESULT_FILES["embs"]["embs"])
    def __getitem__(self,idx):
        desc_id = self.descs_ids[idx]
        return {
            "id": desc_id,
            "sentence": self.descriptions_all[desc_id],
            "embs": self.embs[idx],
            "h_gs": self.h_gs[idx],
            "h_s": self.h_s[idx],
            "h_e": self.h_e[idx],
            "t_s": self.t_s[idx],
            "t_e": self.t_e[idx],
        }
    def __len__(self):
        return self.h_gs.shape[0]

class BRASKModel(nn.Module):
    def __init__(self):
        super(BRASKModel, self).__init__()

        rel_transe_embs = read_cached_array(RESULT_FILES["transE_relation_embeddings"])
        self.rel_transe_embs = rel_transe_embs.to(device)

        rel_embs = read_tensor(RESULT_FILES["rel_embs_tensor"])
        self.rel_embs = rel_embs.to(device)

        self.r_proj = nn.Linear(TRANSE_EMB_DIM, HIDDEN_SIZE)


        self.transE_emb_dim = TRANSE_EMB_DIM
        self.f_head_threshold = THRESHOLDS[0]
        self.b_tail_threshold = THRESHOLDS[1]
        self.f_tail_threshold = THRESHOLDS[2]
        self.b_head_threshold = THRESHOLDS[3]


        self.sigmoid = nn.Sigmoid()

        #forward 
        self.f_start_head_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.f_end_head_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.f_start_tail_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.f_end_tail_fc = nn.Linear(HIDDEN_SIZE, 1)


        #backward 
        self.b_start_tail_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.b_end_tail_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.b_start_head_fc = nn.Linear(HIDDEN_SIZE, 1)
        self.b_end_head_fc = nn.Linear(HIDDEN_SIZE, 1)

        #for s_k in forward, o_k in backward
        self.f_W_h = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.b_W_t = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)


        #Forward
        self.f_W_r = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f_W_g = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f_W_x = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        #Forward
        self.b_W_r = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.b_W_g = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.b_W_x = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        # e = V^T tanh()... , V is the same for forward and backward as in the paper 
        self.V = nn.Linear(HIDDEN_SIZE, 1)

        self.f_Wx2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.b_Wx2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, ds_batch):
        """
            B: batch_size
            L: max sequence length
            R: relation_num
            H: hidden size (786 BERT)

            f: forward (head, relation, tail)
            b: backward (tail, relation, head)
        """
        device = ds_batch["h_gs"].device

        h_gs = ds_batch["h_gs"] #(B, H)
        token_embs = ds_batch["embs"].to(device) #(B, L, H)

        f_rel_embs = self.rel_embs.to(device) # (R, H)

        b_rel_embs_TRANSE = self.rel_transe_embs.to(device) # (R, TRANSE_EMB_DIM)
        b_rel_embs = self.r_proj(b_rel_embs_TRANSE) # (R, H)

        B, L, H = token_embs.shape
        R = f_rel_embs.shape[0]

        #forward
        f_head_start_logits = self.f_start_head_fc(token_embs) #(B, L, 1)
        f_head_end_logits = self.f_end_head_fc(token_embs) #(B, L, 1)
        f_head_start_probs = self.sigmoid(f_head_start_logits).squeeze(-1) #(B, L) 
        f_head_end_probs = self.sigmoid(f_head_end_logits).squeeze(-1) #(B, L) 

        #backward
        b_tail_start_logits = self.b_start_tail_fc(token_embs) #(B, L, 1)
        b_tail_end_logits = self.b_end_tail_fc(token_embs) #(B, L, 1)
        b_tail_start_probs = self.sigmoid(b_tail_start_logits).squeeze(-1) #(B, L) 
        b_tail_end_probs = self.sigmoid(b_tail_end_logits).squeeze(-1) #(B, L) 

        # f_padded_subj_embs has shape (B, L, H)
        # f_mask_subj_embs shape (B, L) each item is 1 if it is padding, 0 otherwize
        # f_subj_idxs is a list where each item representing one sentence from the batch and inside each item is a list of tuples (start_subj_idx, end_subj_idx)
        f_padded_head_embs, f_mask_head_embs, f_head_idxs = extract_first_embeddings(
            token_embs, f_head_start_probs,f_head_end_probs, L, self.f_head_threshold
        )

        b_padded_tail_embs, b_mask_tail_embs, b_tail_idxs = extract_first_embeddings(
            token_embs, b_tail_start_probs,b_tail_end_probs, L, self.b_tail_threshold
        )

        # f_s_w is weighted head embeddings for forward extraction (in paper s_k weighted for subjects)
        f_h_w = self.f_W_h(f_padded_head_embs) #(B, L, H)
        b_t_w = self.b_W_t(b_padded_tail_embs) #(B, L, H)

        #zero the padded entries
        f_h_w = f_h_w * f_mask_head_embs.unsqueeze(-1).to(device=device, dtype=TENSOR_DTYPE)
        b_t_w = b_t_w * b_mask_tail_embs.unsqueeze(-1).to(device=device, dtype=TENSOR_DTYPE)

        token_embs_exp = token_embs.unsqueeze(2).expand(B,L,R,H).to(device=device, dtype=TENSOR_DTYPE) #(B, L, 1, H )
        h_g_exp =  h_gs.unsqueeze(1).unsqueeze(1).expand(B,L,R,H).to(device=device, dtype=TENSOR_DTYPE) #(B, 1, 1, H )

        #forward:
        f_rel_embs_exp = f_rel_embs.unsqueeze(0).unsqueeze(1).expand(B,L,R,H).to(device=device, dtype=TENSOR_DTYPE) #(1, 1, R, H)

        #backward:
        b_rel_embs_exp = b_rel_embs.unsqueeze(0).unsqueeze(1).expand(B,L,R,H).to(device=device, dtype=TENSOR_DTYPE) #(1, 1, R, H)

        # attention scores:
        # we make tanh for adding non-linearity
        # v_e has shape (B,L,R,1) for every token  a score, this score is how much the entity is related to the relationship
        
        f_e = torch.tanh(
            self.f_W_r(f_rel_embs_exp) + self.f_W_g(h_g_exp) + self.f_W_x(token_embs_exp)
        ) # (B, L, R, H)
        f_v_e = self.V(f_e).squeeze(-1) #(B, L, R)

        b_e = torch.tanh(
            self.b_W_r(b_rel_embs_exp) + self.b_W_g(h_g_exp) + self.b_W_x(token_embs_exp)
        ) # (B, L, R, H)
        b_v_e = self.V(b_e).squeeze(-1) #(B, L, R)

        #normalize attention score
        # like we want to distribute attention on tokens, get prob distribution for all tokens if they are relevant to relation
        f_A = F.softmax(f_v_e, dim=1).unsqueeze(-1) #(B, L, R, 1)
        b_A = F.softmax(b_v_e, dim=1).unsqueeze(-1) #(B, L, R, 1)

        f_C = torch.sum(f_A * token_embs_exp, dim=1) #(B, R, H)
        b_C = torch.sum(b_A * token_embs_exp, dim=1) #(B, R, H)


        f_Hik =  f_h_w + self.f_Wx2(token_embs) #  (B, L, H) 
        b_Hik =  b_t_w + self.b_Wx2(token_embs)

        f_Hij = f_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)
        b_Hij = b_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)

        f_Hijk = f_Hik.unsqueeze(2) + f_Hij # (B, L, 1, H) +   (B, L, R, H) =  (B, L, R, H)
        b_Hijk = b_Hik.unsqueeze(2) + b_Hij

        f_tail_s_logits = self.f_start_tail_fc(f_Hijk) #(B, L, R , 1)
        f_tail_e_logits = self.f_end_tail_fc(f_Hijk) # (B<L,R, 1)

        b_head_s_logits = self.b_start_head_fc(b_Hijk) # (B<L,R, 1)
        b_head_e_logits = self.b_end_head_fc(b_Hijk)# (B<L,R, 1)

        return {
            "forward": {
                "head_s": f_head_start_logits,   #(B, L, 1)
                "head_e": f_head_end_logits,  #(B, L, 1)
                "tail_s": f_tail_s_logits.squeeze(-1) , # (B<L,R)
                "tail_e": f_tail_e_logits.squeeze(-1)  , # (B<L,R)
            },
            "backward": {
                "tail_s": b_tail_start_logits, #(B, L, 1) 
                "tail_e": b_tail_end_logits, #(B, L, 1)
                "head_s": b_head_s_logits.squeeze(-1), # (B<L,R)
                "head_e": b_head_e_logits.squeeze(-1), # (B<L,R)
            },
        }


def sample_and_evaluate(model, dataset):
    idxs = np.random.choice(len(dataset), size=NUM_SAMPLES, replace=False)
    sampler = SubsetRandomSampler(idxs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model.eval()
    all_preds = {
        k: [] for k in [
            "f_head_s", "f_head_e", "f_tail_s", "f_tail_e",
            "b_tail_s", "b_tail_e", "b_head_s", "b_head_e", 
        ]
    }
    all_labels = {k: [] for k in all_preds}
    with torch.no_grad():
        for batch in loader:
            out = model(batch)

            f_hs = out["forward"]["head_s"].squeeze(-1).sigmoid #(B, L)
            f_he = out["forward"]["head_e"].squeeze(-1).sigmoid #(B, L)

            f_ts = out["forward"]["tail_s"].sigmoid().max(dim=2).values
            f_te = out["forward"]["tail_e"].sigmoid().max(dim=2).values

            b_ts = out["backward"]["tail_s"].squeeze(-1).sigmoid #(B, L)
            b_te = out["backward"]["tail_e"].squeeze(-1).sigmoid #(B, L)

            b_hs = out["backward"]["head_s"].sigmoid().max(dim=2).values
            b_he = out["backward"]["head_e"].sigmoid().max(dim=2).values

            preds =  {
                "f_head_s": (f_hs > THRESHOLD).cpu().numpy().flatten(),
                "f_head_e": (f_he > THRESHOLD).cpu().numpy().flatten(),
                "f_tail_s": (f_ts > THRESHOLD).cpu().numpy().flatten(),
                "f_tail_e": (f_te > THRESHOLD).cpu().numpy().flatten(),
                "b_tail_s": (b_ts > THRESHOLD).cpu().numpy().flatten(),
                "b_tail_e": (b_te > THRESHOLD).cpu().numpy().flatten(),
                "b_head_s": (b_hs > THRESHOLD).cpu().numpy().flatten(),
                "b_head_e": (b_he > THRESHOLD).cpu().numpy().flatten(),
            }

            labels = {
                "f_head_s": batch["h_s"].cpu().numpy().flatten(),
                "f_head_e": batch["h_e"].cpu().numpy().flatten(),
                "f_tail_s": batch["t_s"].cpu().numpy().flatten(),
                "f_tail_e": batch["t_e"].cpu().numpy().flatten(),
                "b_tail_s": batch["t_s"].cpu().numpy().flatten(),
                "b_tail_e": batch["t_e"].cpu().numpy().flatten(),
                "b_head_s": batch["h_s"].cpu().numpy().flatten(),
                "b_head_e": batch["h_e"].cpu().numpy().flatten(),
                
            }
            
            for key in all_preds:
                all_preds[key].append(preds[key])
                all_labels[key].append(labels[key])
    
    print("=== Token-level accuracy ===")
    for key in all_preds:
        y_pred = np.concatenate(all_preds[key])
        y_true = np.concatenate(all_labels[key])
        acc = accuracy_score(y_true, y_pred)
        print(f"{key:12s}: {acc*100:5.2f}%")
    print("\n=== Full classification reports ===")
    for key in all_preds:
        print(f"\n-- {key} --")
        print(classification_report(
            np.concatenate(all_labels[key]),
            np.concatenate(all_preds[key]),
            digits=4,
            target_names=['no','yes']
        ))

if __name__ == "__main__":
    dataset = BRASKDataset()
    model  = BRASKModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    sample_and_evaluate(model, dataset)

