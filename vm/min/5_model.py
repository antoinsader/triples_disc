import torch


CHECKPOINT_FILE = "./data/checkpoints/BRASK_chkpoint.pth"
MODEL_PATH = "./data/BRASK_MODEL.pth"

L = MAX_LENGTH = 128
HIDDEN_SIZE = 768
TRANSE_EMB_DIM = 100
THRESHOLDS = [.6,.6,.6,.6]

USE_CACHE=  False

device_str = 'cpu'
if torch.cuda.is_available():
  device_str = "cuda"
  

device = torch.device(device_str)

if device_str == 'cpu':
    NUM_WORKERS = 1
    BATCH_SIZE = 48
    PIN_MEMORY = False
    LEARNING_RATE = 1e-4
    NUM_EPOCHS= 100
    TENSOR_DTYPE = torch.float32 

else:
    NUM_WORKERS = 86
    BATCH_SIZE = 1024
    PIN_MEMORY = True
    LEARNING_RATE = 1e-5
    NUM_EPOCHS= 1000
    TENSOR_DTYPE = torch.float16 


import os
import numpy as np
import pickle
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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



def extract_first_embeddings(token_embs, start_probs, end_probs, max_len, threshold=.6):
    """
        args:
            token_embs: sentence embeddings with shape (batch_size, seq_length, hidden_size)
            start_probs: probabilities that an entity is a start (start of subject for forward, start of object for backward) with shape (batch_size, seq_len)
            end_probs: probabilities that an entity is an end (end of subject for forward, end of object for backward) with shape (batch_size, seq_len)
            threshold: threshold that if the probability > threshold, then it is considered start or end of starting entity
        returns:
            padded_embs: these are padded embeddings of all subjects or objects with shape (B, max_ents, H)
            mask_embs: mask to show the padded embs (0 for padding)
            head_idxs: list (len=batch_size) where each item is a list of tuples, each tuple (start_idx, end_idx) of the entities
    """
    batch_size, seq_len, hidden_size = token_embs.shape

    start_mask = start_probs > threshold
    end_mask = end_probs > threshold

    head_idxs = []
    all_ents_embs = []
    all_masks = []

    for sent_idx in range(batch_size):
        start_indices = torch.where(start_mask[sent_idx])[0]
        end_indices = torch.where(end_mask[sent_idx])[0]

        ents_embs = []
        idxs_sentence = []
        used_ends = set()

        for start_idx in start_indices.tolist():
            end_ptr = 0 
            while end_ptr < len(end_indices) and (  end_indices[end_ptr].item() < start_idx or end_indices[end_ptr].item() in used_ends) :
                end_ptr += 1
            if end_ptr < len(end_indices):
                end = end_indices[end_ptr].item()
                used_ends.add(end)
                idxs_sentence.append((start_idx, end))

                #compute average embedding (in paper shows ((emb of start + emb of end) / 2), I will do between them )
                sum = token_embs[sent_idx , start_idx : end + 1].sum(dim=0)
                dominator = end + 1 - start_idx
                ent_emb = sum / dominator
                ents_embs.append(ent_emb)

        head_idxs.append(idxs_sentence)

        if ents_embs:
            ent_tensor = torch.stack(ents_embs)
            all_ents_embs.append(ent_tensor)
            all_masks.append(torch.ones(ent_tensor.size(0), dtype=torch.bool, device=token_embs.device))
        else:
            all_ents_embs.append(torch.empty(0, hidden_size, device=token_embs.device))
            all_masks.append(torch.zeros(0, dtype=torch.bool, device=token_embs.device))

    # Pad sequences
    padded_embs = pad_sequence(all_ents_embs, batch_first=True, padding_value=0.0)
    mask_embs = pad_sequence(all_masks, batch_first=True, padding_value=False)

    if max_len is not None:
        # Pad or truncate embeddings
        curr_len = padded_embs.size(1)
        if curr_len < max_len:
            pad_size = max_len - curr_len
            padding = torch.zeros(padded_embs.size(0), pad_size, padded_embs.size(2), device=padded_embs.device)
            padded_embs = torch.cat([padded_embs, padding], dim=1)

            mask_padding = torch.zeros(mask_embs.size(0), pad_size, dtype=torch.bool, device=mask_embs.device)
            mask_embs = torch.cat([mask_embs, mask_padding], dim=1)

        elif curr_len > max_len:
            padded_embs = padded_embs[:, :max_len, :]
            mask_embs = mask_embs[:, :max_len]

    return padded_embs, mask_embs, head_idxs


def build_bce_loss_dict(pos_weights):
    pw = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in pos_weights.items()}
    bce_losses = {
        #forward
        "f_head_s": nn.BCEWithLogitsLoss(pos_weight=pw["h_s"]),
        "f_head_e": nn.BCEWithLogitsLoss(pos_weight=pw["h_e"]),
        "f_tail_s": nn.BCEWithLogitsLoss(pos_weight=pw["t_s"]),
        "f_tail_e": nn.BCEWithLogitsLoss(pos_weight=pw["t_e"]),
    }
    #backward
    bce_losses.update({
        "b_tail_s": bce_losses["f_tail_s"],
        "b_tail_s": bce_losses["f_tail_s"],
        "b_head_s": bce_losses["f_head_s"],
        "b_head_e": bce_losses["f_head_e"],
    })
    return bce_losses

class BRASKDataset(Dataset):
    def __init__(self):
        self.descriptions_all = read_cached_array(RESULT_FILES["descriptions"])
        self.h_s = read_tensor(RESULT_FILES["silver_spans"]["head_start"])
        self.h_e = read_tensor(RESULT_FILES["silver_spans"]["head_end"])
        self.t_s = read_tensor(RESULT_FILES["silver_spans"]["tail_start"])
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

        f_tail_s_logits = self.f_start_tail_fc(f_Hijk)
        f_tail_e_logits = self.f_end_tail_fc(f_Hijk)

        b_head_s_logits = self.b_start_head_fc(b_Hijk)
        b_head_e_logits = self.b_end_head_fc(b_Hijk)

        return {
            "forward": {
                "head_s": f_head_start_logits.squeeze(-1), 
                "head_e": f_head_end_logits.squeeze(-1), 
                "tail_s": f_tail_s_logits.squeeze(-1) , 
                "tail_e": f_tail_e_logits.squeeze(-1)  , 
            },
            "backward": {
                "tail_s": b_tail_start_logits.squeeze(-1), 
                "tail_e": b_tail_end_logits.squeeze(-1), 
                "head_s": b_head_s_logits.squeeze(-1), 
                "head_e": b_head_e_logits.squeeze(-1), 
            },
        }

def compute_loss(out, batch, bce_losses):
    L = 0.0

    L += bce_losses['f_sub_s'](out['forward']['head_s'].squeeze(-1),
                               batch['h_s'])
    L += bce_losses['f_sub_e'](out['forward']['head_e'].squeeze(-1),
                               batch['h_e'])
    L += bce_losses['f_obj_s'](out['forward']['tail_s'].squeeze(-1),
                               batch['t_s'])
    L += bce_losses['f_obj_e'](out['forward']['tail_e'].squeeze(-1),
                               batch['t_e'])

    L += bce_losses['b_obj_s'](out['backward']['tail_s'].squeeze(-1),
                               batch['t_s'])
    L += bce_losses['b_obj_e'](out['backward']['tail_e'].squeeze(-1),
                               batch['t_e'])
    L += bce_losses['b_sub_s'](out['backward']['head_s'].squeeze(-1),
                               batch['h_s'])
    L += bce_losses['b_sub_e'](out['backward']['head_e'].squeeze(-1),
                               batch['h_e'])
    return L / 8.0

def get_pos_weights(dataloader):
    label_keys = [
        "h_s", "h_e", "t_s", "t_e"
    ]
    
    pos_conuts = {k: 0 for k in label_keys}
    neg_counts = {k: 0 for k in label_keys}

    for batch in dataloader:
        for k in label_keys:
            lbl = batch[k]
            p = lbl.sum().item()
            n = lbl.numel() - p
            pos_conuts[k] += p
            neg_counts[k] += n
    return {
        k: (neg_counts[k] / pos_conuts[k]) if pos_conuts[k] > 0 else 1.0 
        for k in label_keys
    }


def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        last_total_loss = checkpoint["last_total_loss"]
        print(f"resumed from epoch {start_epoch} with last total loss: {last_total_loss:.4f}")
        return start_epoch, last_total_loss
    else:
        return 0, 0.00

def save_checkpoint(model, optimizer, epoch, last_total_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "last_total_loss": last_total_loss
    }
    torch.save(checkpoint, CHECKPOINT_FILE)

def train_model(dataset, ):
    print("loading data loader..")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print("Getting pos weights")
    pos_weights = get_pos_weights(dataloader)
    print("Building bce loss dict")
    
    bce_losses = build_bce_loss_dict(pos_weights)
    print("init model..")
    # in prod (should do torch.compile)
    model = BRASKModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch, last_total_loss = load_checkpoint(model, optimizer)

    print("starting epoch loops")
    with tqdm(range(start_epoch - 1, NUM_EPOCHS), desc="Training epochs", total=(NUM_EPOCHS - start_epoch - 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            total_loss = last_total_loss
            last_loss = 0.0
            batch_count = 0
            for ds_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                # ds_batch = {k: v.to(device) for k, v in ds_batch.items()}
                out = model(ds_batch)
                loss = compute_loss(out, ds_batch, bce_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                last_loss =loss.item()
                total_loss += last_loss
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            pbar_epoch.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss:.4f}", last_loss=last_loss)
            save_checkpoint(model, optimizer, epoch, total_loss, ) 

    return model
if __name__ == "__main__":
    
    print("creating dataset..")
    dataset = BRASKDataset()
    print("training..")
    model = train_model(dataset)
    torch.save(model.state_dict(), )