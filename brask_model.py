import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
import numpy as np 
from tqdm import tqdm


from Data import CHECKPOINT_FILES, PKLS_FILES, TEMP_FILES, THEMODEL_PATH
from utils.utils import cache_array, read_cached_array
from utils.model_helpers import get_h_gs, extract_first_embeddings,extract_last_idxs, extract_triples,  merge_triples
import random
import os

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)







class BRASKDataSet(Dataset):
    def __init__(self, descriptions_dict, silver_spans , desc_max_length=128):
        #silver_spans should be dictionary having keys head_start, head_end, tail_start, tail_end, each one is tensor with shape (B, seq_len) 
        valid = (len(descriptions_dict), desc_max_length) == silver_spans["head_start"].shape == silver_spans["head_end"].shape == silver_spans["tail_start"].shape==silver_spans["tail_end"].shape 
        
        if valid:
            
            tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')

            sentences = list(descriptions_dict.values())
            print("creating h_gs")
            self.h_gs, self.embs = get_h_gs(sentences, tokenizer, model, max_length=desc_max_length  )  #  h_gs (batch_size, hidden_size), embs (batch_size, seq_len, hidden_size)
            print("")
            print(f"self embs shape: {self.embs.shape}")
            self.labels_head_start, self.labels_head_end, self.labels_tail_start, self.labels_tail_end =  silver_spans["head_start"], silver_spans["head_end"], silver_spans["tail_start"], silver_spans["tail_end"] 

    def __getitem__(self,idx):
        return  {
            "h_gs": self.h_gs[idx], 
            "embs": self.embs[idx],
            "labels_head_start": self.labels_head_start[idx] ,
            "labels_head_end": self.labels_head_end[idx],
            "labels_tail_start": self.labels_tail_start[idx],
            "labels_tail_end": self.labels_tail_end[idx],

        }


    def __len__(self):
        return self.h_gs.shape[0]
    
    def save(self, path):
        di = {
            "h_gs": self.h_gs.cpu(),
            "embs": self.embs.cpu(),
            "labels_head_start": self.labels_head_start.cpu() ,
            "labels_head_end": self.labels_head_end.cpu(),
            "labels_tail_start":  self.labels_tail_start.cpu(),
            "labels_tail_end":  self.labels_tail_end.cpu(),
            
        }
        cache_array(di, path)
    @classmethod
    def load(cls, path):
        print("loadding dataset from cache.. ")
        data = read_cached_array(path)

        dataset = cls.__new__(cls)
        dataset.h_gs = data["h_gs"]
        print(f"\t dataset.h_gs.shape: {dataset.h_gs.shape}")
        dataset.embs = data["embs"]
        print(f"\t dataset.embs.shape: {dataset.embs.shape}")
        dataset.labels_head_start = data["labels_head_start"]
        dataset.labels_head_end = data["labels_head_end"]
        dataset.labels_tail_start = data["labels_tail_start"]
        dataset.labels_tail_end = data["labels_tail_end"]
        
        return dataset
class BRASKModel(nn.Module):
    def __init__(self, rel_embs, rel_transe_embs, hidden_size=768, transE_emb_dim=100, thresholds=[.5,.5,.5,.5], device="cpu"):
        super(BRASKModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.rel_embs = rel_embs.to(device)
        self.rel_transe_embs = rel_transe_embs.to(device)
        
        self.transE_emb_dim = transE_emb_dim

        #f_subj, b_obj, f_obj, b_subj
        self.f_subj_threshold = thresholds[0]
        self.b_obj_threshold = thresholds[1]
        self.f_obj_threshold = thresholds[2]
        self.b_subj_threshold = thresholds[3]

        #forward
        self.f_start_sub_fc = nn.Linear(hidden_size, 1)
        self.f_end_sub_fc = nn.Linear(hidden_size, 1)

        self.f_start_obj_fc = nn.Linear(hidden_size, 1)
        self.f_end_obj_fc = nn.Linear(hidden_size, 1)
        

        #backward
        self.b_start_obj_fc = nn.Linear(hidden_size, 1)
        self.b_end_obj_fc = nn.Linear(hidden_size, 1)

        self.b_start_sub_fc = nn.Linear(hidden_size, 1)
        self.b_end_sub_fc = nn.Linear(hidden_size, 1)





        # for s_k in forward or o_k in backward 
        self.f_W_s = nn.Linear(hidden_size, hidden_size)
        self.b_W_s = nn.Linear(hidden_size, hidden_size)

        self.r_proj = nn.Linear(transE_emb_dim, hidden_size)


        #forward:
        self.f_W_r = nn.Linear(hidden_size, hidden_size)
        self.f_W_g = nn.Linear(hidden_size, hidden_size)
        self.f_W_x = nn.Linear(hidden_size, hidden_size)

        #backward:
        self.b_W_r = nn.Linear(hidden_size, hidden_size)
        self.b_W_g = nn.Linear(hidden_size, hidden_size)
        self.b_W_x = nn.Linear(hidden_size, hidden_size)

        # e = V^T tanh()... , V is the same for forward and backward as in the paper 
        self.V   = nn.Linear(hidden_size, 1)

        self.f_Wx2 = nn.Linear(hidden_size, hidden_size)
        self.b_Wx2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, ds_batch):
        """ 
            From now on when we say in the shape: 
                b: batch_size 
                l: seq_len
                r: relation_nums
                h: hidden_size (786)
            The parameter names if it starts with
                f: forward (subject, relation, object)
                b: backward (object, relation, subject)
                 
        """
        device = ds_batch["h_gs"].device


        h_gs = ds_batch["h_gs"].to(device) # (b, h)
        token_embs = ds_batch["embs"].to(device) #(b, l, h)
        f_rel_embs =  self.rel_embs.to(device) # (r, h)
        b_rel_transe_embs = self.rel_transe_embs #(r, h_transE)

        # project transE embeddings into bert space
        b_rel_embs = self.r_proj(b_rel_transe_embs) #(r, hidden_size)
        

        batch_size, seq_len, hidden_size = token_embs.shape 
        num_relations = f_rel_embs.shape[0]

        #forward
        f_sub_start_probs = self.sigmoid(self.f_start_sub_fc(token_embs)).squeeze(-1) # (b, l)
        f_sub_end_probs = self.sigmoid(self.f_end_sub_fc(token_embs)).squeeze(-1) 
        #backward
        b_obj_start_probs = self.sigmoid(self.b_start_obj_fc(token_embs)).squeeze(-1)
        b_obj_end_probs = self.sigmoid(self.b_end_obj_fc(token_embs)).squeeze(-1) # (b,l)
        

        # f_padded_subj_embs has shape (B, L, H)
        # f_mask_subj_embs shape (B, L) each item is 1 if it is padding, 0 otherwize
        # f_subj_idxs is a list where each item representing one sentence from the batch and inside each item is a list of tuples (start_subj_idx, end_subj_idx)

        f_padded_subj_embs, f_mask_subj_embs, f_subj_idxs = extract_first_embeddings(
            token_embs, f_sub_start_probs, f_sub_end_probs, seq_len, threshold=self.f_subj_threshold) # , 



        b_padded_obj_embs, b_mask_obj_embs, b_obj_idxs = extract_first_embeddings(
            token_embs, b_obj_start_probs, b_obj_end_probs, seq_len, threshold=self.b_obj_threshold) # b_padded_obj_embs has shape (B, L, H)



        # f_s_w is Wsf(S_k) for forwad (all subject embeddings weighted), 
        # b_o_w is Wsb(O_k) for backward (all objet embeddings weighted), 
        f_s_w = self.f_W_s(f_padded_subj_embs) # (B, max_subjs, H)
        b_o_w = self.b_W_s(b_padded_obj_embs) # (B, max_objs, H)

        #zero out the padded entries
        f_s_w = f_s_w * f_mask_subj_embs.unsqueeze(-1).float()  # (B, max_subjs, H)
        b_o_w = b_o_w * b_mask_obj_embs.unsqueeze(-1).float() # (B, max_objs, H)


        token_embs_exp = token_embs.unsqueeze(2) #(B, L, 1, H)        
        h_g_exp = h_gs.unsqueeze(1).unsqueeze(2) #(B, 1,1 , H)

      
        #FORWARD
        f_rel_embs_exp = f_rel_embs.unsqueeze(0).unsqueeze(1)  #(1, 1, R, H)
        f_rel_embs_exp = f_rel_embs_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations, hidden_size)

        #backward
        b_rel_embs_exp = b_rel_embs.unsqueeze(0).unsqueeze(1)  #(1, 1, R, H)
        b_rel_embs_exp = b_rel_embs_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations, hidden_size)


        
        
        # attention scores:
        # we make tanh for adding non-linearity
        # v_e has shape (B,L,R,1) for every token a score, this score is kind of how much the entity is related to the relationship

        #Forward:
        f_e = torch.tanh(
            self.f_W_r(f_rel_embs_exp) + self.f_W_g(h_g_exp) + self.f_W_x(token_embs_exp)
        ) # (B, L, R, H)
        f_v_e = self.V(f_e).squeeze(-1) # (B, L, R)


        #Backward:
        b_e = torch.tanh(
            self.b_W_r(b_rel_embs_exp) + self.b_W_g(h_g_exp) + self.b_W_x(token_embs_exp)
        ) # (B, L, R, H)
        b_v_e = self.V(b_e).squeeze(-1) # (B, L, R)


        #normalize attention score
        #on dim = 1 because on the sentences dimension because we want to distribute attention on tokens, like we want to get probability distribution for all tokens if they are relevant to the relation
        f_A = F.softmax(f_v_e, dim=1).unsqueeze(-1)  # (B, L, R, 1) 
        b_A = F.softmax(b_v_e, dim=1).unsqueeze(-1)  # (B, L, R, 1) 

        f_C = torch.sum(f_A * token_embs_exp, dim=1) # (B ,R, H)
        b_C = torch.sum(b_A * token_embs_exp, dim=1) # (B ,R, H)

        

        




        f_Hik =  f_s_w + self.f_Wx2(token_embs) #  (B, L, H) 
        b_Hik =  b_o_w + self.b_Wx2(token_embs)

        f_Hij = f_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)
        b_Hij = b_C.unsqueeze(1) + token_embs_exp #  (B, 1 ,R, H) + (B, L, 1, H)  = (B, L, R, H)

        f_Hijk = f_Hik.unsqueeze(2) + f_Hij # (B, L, 1, H) +   (B, L, R, H) =  (B, L, R, H)
        b_Hijk = b_Hik.unsqueeze(2) + b_Hij
        




        
        
        #forward
        f_obj_start_probs = self.sigmoid(self.f_start_obj_fc(f_Hijk)).squeeze(-1) # (b, l , r)
        f_obj_end_probs = self.sigmoid(self.f_end_obj_fc(f_Hijk)).squeeze(-1)  # (b, l , r)

        #backward
        b_sub_start_probs = self.sigmoid(self.b_start_sub_fc(b_Hijk)).squeeze(-1) # (b, l , r)
        b_sub_end_probs = self.sigmoid(self.b_end_sub_fc(b_Hijk)).squeeze(-1) # (b, l , r)




        forward_triples  = extract_triples(f_subj_idxs, f_obj_start_probs, f_obj_end_probs,  True , threshold=self.f_obj_threshold )
        backward_triples = extract_triples(b_obj_idxs, b_sub_start_probs, b_sub_end_probs,  False, threshold=self.b_subj_threshold)
        
        # predicted_triples = merge_triples(forward_triples, backward_triples)
        
        

        return {
            "forward": {
                "sub_s": f_sub_start_probs, 
                "sub_e": f_sub_end_probs, 
                "obj_s": f_obj_start_probs, 
                "obj_e": f_obj_end_probs, 
            },
            "backward": {
                "obj_s": b_obj_start_probs, 
                "obj_e": b_obj_end_probs, 
                "sub_s": b_sub_start_probs, 
                "sub_e": b_sub_end_probs, 
            },
            # "predicted_triples": predicted_triples
        }

def compute_loss(out, batch):
    bce = nn.BCEWithLogitsLoss()
    loss = 0
    loss += bce(out["forward"]["sub_s"], batch["labels_head_start"])
    loss += bce(out["forward"]["sub_e"], batch["labels_head_end"])
    loss += bce(out["forward"]["obj_s"], batch["labels_tail_start"])
    loss += bce(out["forward"]["obj_e"], batch["labels_tail_end"])

    loss += bce(out["backward"]["obj_s"], batch["labels_tail_start"])
    loss += bce(out["backward"]["obj_e"], batch["labels_tail_end"])
    loss += bce(out["backward"]["sub_s"], batch["labels_head_start"])
    loss += bce(out["backward"]["sub_e"], batch["labels_head_end"])
    return loss


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
    
    

def train_model(dataset,k,thresholds,checkpoint_path, transE_emb_dim=100, batch_size=128, num_workers=8, learning_rate=1e-5, num_epochs=1000, device="cpu", ):
    print("loading dataloader..")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    
    relations_embs = read_cached_array(PKLS_FILES["relations_embs"][k])
    relations_embs_transE = read_cached_array(PKLS_FILES["transE_relation_embeddings"])

    

    
    ###### REMEMBER TO MAKE IT torch.compile  in production
    print("creating model..")
    model = torch.compile(BRASKModel(relations_embs, relations_embs_transE, transE_emb_dim=transE_emb_dim, thresholds=thresholds,device=device))
    model.to(device)
    optimizer = optim.Adam(model.parameters(),  lr=learning_rate )
    start_epoch, last_total_loss = load_checkpoint(model, optimizer, filename=checkpoint_path)
    
    print("starting epochs loop")
    with tqdm(range(start_epoch - 1, num_epochs), desc="Training Epochs", total=(num_epochs-start_epoch - 1)) as pbar_epoch:
        for epoch in pbar_epoch:
            total_loss = last_total_loss 
            last_loss = 0.0
            batch_count = 0
            for ds_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                ds_batch = {key: value.to(device) for key, value in ds_batch.items()}
                out = model(ds_batch)
                loss = compute_loss(out, ds_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                last_loss = loss.item()
                total_loss += last_loss
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            pbar_epoch.set_postfix(epoch=epoch+1, avg_loss=f"{avg_loss:.4f}",last_loss=last_loss )
            save_checkpoint(model, optimizer, epoch, total_loss, filename=checkpoint_path)
            
            
    return model

if __name__ == "__main__":
    k = 1000 
    transE_emb_dim = 100
    batch_size = 256
    num_workers = 8
    learning_rate = 0.001
    num_epochs = 1
    #f_subj, b_obj, f_obj, b_subj
    thresholds = [.5,.5,.5,.5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = CHECKPOINT_FILES["brask"]
    

    use_cached_ds = False

    if use_cached_ds:
        print("reading cached ds")
        dataset = BRASKDataSet.load(TEMP_FILES["dataset"][k])
        
    else:
        print("reading dictionaries...")
        descriptions_dict = read_cached_array(PKLS_FILES["descriptions_normalized"][k])
        silver_spans  = read_cached_array(PKLS_FILES["silver_spans"][k])
        print("creating ds")
        dataset = BRASKDataSet(descriptions_dict,silver_spans )
        dataset.save(TEMP_FILES["dataset"][k])

    model = train_model(
            dataset, k, transE_emb_dim=transE_emb_dim, batch_size=batch_size,
                            num_workers=num_workers, learning_rate=learning_rate, 
                            num_epochs=num_epochs, thresholds=thresholds, 
                            device=device, checkpoint_path=checkpoint_file)
    torch.save(model.state_dict(), THEMODEL_PATH)