import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
import numpy as np 
import math


from Data import PKLS_FILES, TEMP_FILES
from utils.utils import cache_array, read_cached_array
import random


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# If using CUDA:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)







def get_h_gs(sentences, tokenizer, model, max_length):
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    with torch.no_grad():
        bert_output = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state # (batch_size, seq_len ,hidden_size)

    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())

    masked_embeddings = embeddings * mask_expanded

    sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden_size]
    token_counts = mask_expanded.sum(dim=1)  # [batch, 1]
    token_counts = token_counts.clamp(min=1)
    return sum_embeddings / token_counts, embeddings



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
            start_idxs:  list (len=batch_size) where each item is a list of tuples, each tuple (start_idx, end_idx) of the entities
    """
    batch_size, seq_len, hidden_size = token_embs.shape

    start_mask = start_probs > threshold
    end_mask = end_probs > threshold

    start_idxs = []
    all_ents_embs = []
    all_masks = []

    for sent_idx in range(batch_size):
        start_indices = torch.nonzero(start_mask[sent_idx], as_tuple=False).squeeze(-1)
        end_indices = torch.nonzero(end_mask[sent_idx], as_tuple=False).squeeze(-1)

        ents_embs = []
        idxs_sentence = []
        used_ends = set()

        for start_idx in start_indices.tolist():
            candidates = [end_idx for end_idx in end_indices.tolist() if end_idx >= start_idx and end_idx not in used_ends]
            if candidates:
                end_idx = candidates[0]
                used_ends.add(end_idx)
                idxs_sentence.append((start_idx, end_idx))

                # Compute average embedding
                ent_emb = (token_embs[sent_idx, start_idx] + token_embs[sent_idx, end_idx]) / 2.0
                ents_embs.append(ent_emb)

        start_idxs.append(idxs_sentence)

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

    return padded_embs, mask_embs, start_idxs

def extract_last_idxs(start_probs, end_probs, threshold):
    """
      args:
        start_probs, end_probs are vectors with length of L, those are for one sentence 
      returns:
        idxs: a list where each item contains (start_idx, end_idx) of entities in the sentence
    """
    start_idxs = torch.nonzero(start_probs > threshold).squeeze()
    end_idxs = torch.nonzero(end_probs > threshold).squeeze()
    
    if start_idxs.ndim == 0:
        start_idxs = start_idxs.unsqueeze(0)
    if end_idxs.ndim == 0:
        end_idxs = end_idxs.unsqueeze(0)

    idxs = []
    used_ends = set()
    
    for start_idx in start_idxs.tolist():
        for end_idx in end_idxs.tolist():
            if end_idx >= start_idx and end_idx not in used_ends:
                idxs.append((start_idx, end_idx))
                used_ends.add(end_idx)
                break
            
    return idxs 
  


def extract_triples(first_idxs, start_probs, end_probs,is_forward, threshold=.6):
    """
        args:
            first_idxs list with len = B, each one is list of indexes for the starting entities. List sentence-1[(start_ent_start_idx, start_ent_end_idx ), ...]
            start_probs, end_probs with shape (B, L, R)
        does:
          loop through sentences in the batch
          for each relationship: find objects that has high probability to make (rel, obj)
          assign those to each subject to have (sub, rel, obj)
          (I am saying sub,rel,obj but in backward it would be obj,rel,sub)
        
        returns:
          batch_triples: which is a list with size of B and each item is a list having the triples
          

    """
    batch_size, seq_len, num_rels = start_probs.shape
    batch_triples  = []
    
    for b in range(batch_size):
        sentence_triples = []
        sentence_first_idxs = first_idxs[b]
        
        for rel in range(num_rels):
            last_start_vec = start_probs[b, :, rel]
            last_end_vec = end_probs[b, :, rel]
            last_idxs = extract_last_idxs(last_start_vec, last_end_vec, threshold)
            
            for first_idx in sentence_first_idxs:
                for last_idx in last_idxs:
                    # if is forward:
                    #     subject_end_idx should be < obj_start_idx 
                    # if backward:
                    #     obj_start_idx > subject_end_idx
                    if (is_forward and first_idx[1] < last_idx[0]) or (not is_forward and first_idx[0] > last_idx[1]):
                      sentence_triples.append((first_idx,rel ,last_idx))
        batch_triples.append(sentence_triples)
    
    return batch_triples    
        


def merge_triples(forward_triples, backward_triples):
  """
    args:
      forwarD_triples: List (len=batch_size ? ) of items, each item is a triple (head_idx, rel_idx, obj_idx)  
      backward_triples: List (len=batch_size ? ) of items, each item is a triple (head_idx, rel_idx, obj_idx)  
    returns:
      list of triples (length=batch_size) and each item is  a triple (h,r,t)  
    
  """
  final_triples = []
  for f_triples, b_triples in zip(forward_triples, backward_triples):
    final = list(set(f_triples).intersection(set(b_triples)))
    final_triples.append(final)
  return final_triples




class BRASKDataSet(Dataset):
    def __init__(self, descriptions_dict , desc_max_length=128):
        if descriptions_dict is not None:
            
            tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
            model = BertModel.from_pretrained('bert-base-cased')

            sentences = list(descriptions_dict.values())
            self.h_gs, self.embs = get_h_gs(sentences, tokenizer, model, max_length=desc_max_length  )  #  h_gs (batch_size, hidden_size), embs (batch_size, seq_len, hidden_size)
            
            
            print(f"self embs shape: {self.embs.shape}")
            
    def __getitem__(self,idx):
        return  {"h_gs": self.h_gs[idx], "embs": self.embs[idx] }
    def __len__(self):
        return self.h_gs.shape[0]
    
    def save(self, path):
        di = {
            "h_gs": self.h_gs.cpu(),
            "embs": self.embs.cpu(),
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
        
        print(f"len forward triples:  {len(forward_triples)}, first item in forward triple len: {len(forward_triples[0])} and first item: {forward_triples[0]}")
        print(f"len backward triples:  {len(backward_triples)}, first item in backward  triple len: {len(backward_triples[0])} and first item: {backward_triples[0]}")


        
        print("****************")

        return forward_triples, backward_triples

def train_model(dataset,k,thresholds, transE_emb_dim=100, batch_size=128, num_workers=8, learning_rate=0.001, num_epochs=1000, device="cpu"):
    print("loading dataloader..")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    
    relations_embs = read_cached_array(PKLS_FILES["relations_embs"][k])
    relations_embs_transE = read_cached_array(PKLS_FILES["transE_relation_embeddings"])


    
    ###### REMEMBER TO MAKE IT torch.compile  in production
    print("creating model..")
    model = torch.compile(BRASKModel(relations_embs, relations_embs_transE, transE_emb_dim=transE_emb_dim, thresholds=thresholds,device=device))
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # start_epoch, last_total_loss = load_checkpoint(model, optimizer, filename=checkpoint_path)
    
    print("starting epochs loop")
    for epoch in range(num_epochs):
        total_loss = 0 
        for ds_batch in dataloader:
            ds_batch = {key: value.to(device) for key, value in ds_batch.items()}
            forward_triples, backward_triples = model(ds_batch)
            break

        break
    return [], []



if __name__ == "__main__":
    k = 100 
    transE_emb_dim = 100
    batch_size = 256
    num_workers = 8
    learning_rate = 0.001
    num_epochs = 1
    #f_subj, b_obj, f_obj, b_subj
    thresholds = [.5,.5,.5,.5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    use_cached_ds = False

    if use_cached_ds:
        print("reading cached ds")
        dataset = BRASKDataSet.load(TEMP_FILES["dataset"][k])
        
    else:
        print("reading dictionaries...")
        descriptions_dict = read_cached_array(PKLS_FILES["descriptions_normalized"][k])
        y_triples = read_cached_array(PKLS_FILES["triples"][k])
        
        print("creating ds")
        dataset = BRASKDataSet(descriptions_dict)
        dataset.save(TEMP_FILES["dataset"][k])

    # forward_triples, backward_triples = train_model(dataset, k, transE_emb_dim=transE_emb_dim, batch_size=batch_size,
    #                                                 num_workers=num_workers, learning_rate=learning_rate, 
    #                                                 num_epochs=num_epochs, thresholds=thresholds, 
    #                                                 device=device)