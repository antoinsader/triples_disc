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




def get_relation_embedings(relations):
    tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    model.eval()
    relation_embedings = []
    print(f"\t \t create relation embeddings from bert from : {len(relations)} relations")
    for label in relations:
      print(f"other label: {label}")
      inputs = tokenizer(label, return_tensors="pt")
      with torch.no_grad():
          outputs = model(**inputs, output_hidden_states=True)
      
      hidden_states = outputs.hidden_states 
      
      last_layer = hidden_states[-1].squeeze(0) # (seq_len, hidden_size)
      before_last_layer = hidden_states[-2].squeeze(0)
      
      average_layers = (last_layer + before_last_layer) / 2.0

      r_j = average_layers.mean(dim=0)
      print(f"\t\t r_j: {r_j.shape}")
      
      relation_embedings.append(r_j)
      print("embed has been appended")
    
    print(f"\t \t relation embeddings shape: {relation_embedings.shape}")
    return relation_embedings


class SentencesDS(Dataset):
    def __init__(self, descriptions, max_length=128):
        self.descriptions_dict = descriptions
        self.ids = list(descriptions.keys())
        self.sentences = list(descriptions.values())
        tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased')
    
        encoded = tokenizer(self.sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        with torch.no_grad():
            bert_output = model(input_ids=input_ids, attention_mask=attention_mask)
            self.embeddings = bert_output.last_hidden_state # (batch_size, seq_len ,hidden_size)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(self.embeddings.size())

        masked_embeddings = self.embeddings * mask_expanded

        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden_size]
        token_counts = mask_expanded.sum(dim=1)  # [batch, 1]
        token_counts = token_counts.clamp(min=1)

        self.hg  =  sum_embeddings / token_counts # [batch, hidden_size]
        
        
    
    def __getitem__(self,idx):
        return  {"string": self.sentences[idx], "hg": self.hg[idx], "embedding": self.embeddings[idx], "id": self.ids[idx] }

    def getitem_byid(self,desc_id):
        idx = self.ids.index(desc_id)
        return self.__getitem__(idx)
    

    def __len__(self):
        return len(self.ids) 


class RelationsDS(Dataset):
    def __init__(self, relations_dict, relations_transe_embs):
        self.relations_dict = relations_dict 

        self.ids = list(relations_dict.keys())
        self.relations_lst = list(relations_dict.values())
        print(f"\t relations has {len(self.ids)} ids ")
        embeddings_list = get_relation_embedings(self.relations_lst)
        print(f"embs list is: {embeddings_list}")

        self.embeddings = torch.stack(embeddings_list, dim=0)  #(num_relations, hidden_size)
        print(f"\t embeddings has shape {self.embeddings.shape} ")
        self.transe_embeddings = relations_transe_embs
        print(f"\t embeddings transe has shape {self.transe_embeddings.shape} ")
        
        
    
    def __getitem__(self,idx):
        return {"string": self.relations_lst[idx],   "id":self.ids[idx], "embedding": self.embeddings[idx]}
     

    def getitem_byid(self,relation_id):
        idx = self.ids.index(relation_id)
        return self.__getitem__(idx)
    

    def __len__(self):
        return len(self.ids) 

def extract_first_embeddings(token_embeddings, start_probs, end_probs, threshold=.5 ):
    batch_size, seq_len, hidden_size = token_embeddings.shape

    subj_obj_idxs = [] # (batch_size, subjects_len, 2)
    s_k = [] #batch_size, subjects, hidden_size
    padded_s_k_list = []
    mask_s_k_list = []

    for b in range(batch_size):
      # print(f"sentece: {b}")
      subj_obj_idxs_sentence = []
      used_ends = set()
      for tk_idx in range(seq_len):
        # print(f"\t tk idx: {tk_idx}")
        if start_probs[b][tk_idx] > threshold:
          for j in range(tk_idx, seq_len):
            if end_probs[b][j] > threshold and j not in used_ends:
              subj_obj_idxs_sentence.append((tk_idx, j))
              used_ends.add(j)
              break
      subj_obj_idxs.append(subj_obj_idxs_sentence)

      sentence_subjects_objects = []

      for sub_start_idx, sub_end_idx in subj_obj_idxs_sentence:
        sub_emb = (token_embeddings[b, sub_start_idx] + token_embeddings[b, sub_end_idx])  /2.0
        sentence_subjects_objects.append(sub_emb)
      s_k.append(sentence_subjects_objects)


    for subj_obj_embs in s_k:
      if len(subj_obj_embs) > 0:
        subj_tensor = torch.stack(subj_obj_embs, dim=0)
        padded_s_k_list.append(subj_tensor)
        mask_s_k_list.append(torch.ones(subj_tensor.shape[0], dtype=torch.bool, device=token_embeddings.device))
      else:
        padded_s_k_list.append(torch.empty(0, hidden_size, device = token_embeddings.device))
        mask_s_k_list.append(torch.empty(0, dtype=torch.bool, device=token_embeddings.device))

    padded_s_k = pad_sequence(padded_s_k_list, batch_first=True, padding_value=0.0)
    mask_s_k = pad_sequence(mask_s_k_list, batch_first=True, padding_value=False)




    return padded_s_k, mask_s_k , subj_obj_idxs

def extract_obj_spans_relation(obj_start_probs, obj_end_probs, threshold):

  seq_len = obj_start_probs.shape[0]
  obj_idxs = []
  used_ends = set()
  for tk_idx in range(seq_len):
    if obj_start_probs[tk_idx] > threshold:
      for j in range(tk_idx, seq_len):
        if obj_end_probs[j] > threshold and j not in used_ends:
          obj_idxs.append((tk_idx, j))
          used_ends.add(j)
          break
  return obj_idxs


def extract_triples(token_embs, idxs, start_probs, end_probs, threshold=.5):
  batch_size, seq_len, max_n_subs, num_rels = start_probs.shape
  batch_triples = []
  for sentence_idx in range(batch_size):
    sentence_triples = []
    subj_obj_spans = idxs[sentence_idx] #[(s_o__start, s_o__end), ...]
    n_valid_subj_obj = len(subj_obj_spans)

    for subj_idx in range(n_valid_subj_obj):
      for rel_idx in range(num_rels):
        obj_start_vec = start_probs[sentence_idx, :, subj_idx, rel_idx] #probabilities of this sentence with this subject and relation of all tokens
        obj_end_vec = end_probs[sentence_idx, :, subj_idx, rel_idx]
        obj_idxs = extract_obj_spans_relation(obj_start_vec, obj_end_vec, threshold)
        for obj_idx in obj_idxs:
          #We should do the obj as head ??!! 
          sentence_triples.append((obj_idx, rel_idx,subj_obj_spans[subj_idx] ))
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
class MyBRASKModel(nn.Module):
    def __init__(self,  hidden_size=768, transE_emb_size=80):
        super(MyBRASKModel, self).__init__()
        
        bert_model = BertModel.from_pretrained('bert-base-cased')
    
        self.bert = bert_model

        #forward
        self.start_subject_fc = nn.Linear(hidden_size, 1)
        self.end_subject_fc = nn.Linear(hidden_size, 1)
        self.start_object_fc = nn.Linear(hidden_size, 1)
        self.end_object_fc = nn.Linear(hidden_size, 1)

        #backward
        self.b_start_obj_fc = nn.Linear(hidden_size, 1)
        self.b_end_obj_fc = nn.Linear(hidden_size, 1)
        self.b_start_subj_fc = nn.Linear(hidden_size, 1)
        self.b_end_subj_fc = nn.Linear(hidden_size, 1)




        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_g = nn.Linear(hidden_size, hidden_size)
        self.W_x = nn.Linear(hidden_size, hidden_size)
        
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_g = nn.Linear(hidden_size, hidden_size)
        self.W_x = nn.Linear(hidden_size, hidden_size)
        self.r_proj = nn.Linear(transE_emb_size, hidden_size)
        
        
        self.V   = nn.Linear(hidden_size, 1)


        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_x2 = nn.Linear(hidden_size, hidden_size)



        self.sigmoid = nn.Sigmoid()

    def forward(self, descriptions_dataset, relation_dataset):

        batch_size = descriptions_dataset.embeddings.shape[0]
        token_embs = descriptions_dataset.embeddings # (batch_size, seq_len, hidden_size)


        #apply linear + sigmoid to each token
        #forward:
        sub_start_probs = self.sigmoid(self.start_subject_fc(token_embs )).squeeze(-1) # (batch_size, seq_len)
        sub_end_probs = self.sigmoid(self.end_subject_fc(token_embs)).squeeze(-1)


        #backward
        b_obj_start_probs = self.sigmoid(self.b_start_obj_fc(token_embs )).squeeze(-1) # (batch_size, seq_len)
        b_obj_end_probs = self.sigmoid(self.b_end_obj_fc(token_embs)).squeeze(-1)



        print(f"sub start probs: {sub_start_probs}")
        #forward
        padded_s_k, mask_s_k , subj_idxs    = extract_first_embeddings(
            token_embs, sub_start_probs, sub_end_probs, threshold=0.5
        ) # padded s_k (batch_size, max_subjects, hidden_size)
        s_k_w = self.W_s(padded_s_k)  # (batch_size, max_subjects, hidden_size)
        # Zero-out the padded entries.
        s_k_w = s_k_w * mask_s_k.unsqueeze(-1).float()  # (batch_size, max_subjects, hidden_size)

        #backward
        b_padded_s_k, b_mask_s_k , b_obj_idxs    = extract_first_embeddings(
            token_embs, b_obj_start_probs, b_obj_end_probs, threshold=0.5
        ) # b_padded_s_k s_k (batch_size, max_objs, hidden_size)
        b_s_k_w = self.W_s(b_padded_s_k)  # (batch_size, max_objs, hidden_size)
        # Zero-out the padded entries.
        b_s_k_w = b_s_k_w * b_mask_s_k.unsqueeze(-1).float()  # (batch_size, max_objs, hidden_size)



        h_g = descriptions_dataset.hg # (batch_size, hidden_size)


        relations_embeddings = relation_dataset.embeddings #(num_relations, hidden_size)
        #backward relation embeddings
        b_relations_embeddings = relation_dataset.transe_embeddings #(num_relations, 80)
        b_relations_embeddings = self.r_proj(b_relations_embeddings) # (num_relations, hidden_size)
                
        
        
        

        token_embs_exp = token_embs.unsqueeze(2) #(batch_size, seq_len, 1, hidden_size) 
        h_g_exp = h_g.unsqueeze(1).unsqueeze(2) #(batch_size, 1,1 , hidden_size)
        
        #forward
        relation_exp = relations_embeddings.unsqueeze(0).unsqueeze(1)  #(1, 1, num_relations, hidden_size)
        relation_exp = relation_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations, hidden_size) #repeat through the batch

        #backward
        
        b_relation_exp = b_relations_embeddings.unsqueeze(0).unsqueeze(1)  #(1, 1, num_relations_b, hidden_size)
        b_relation_exp = b_relation_exp.expand(batch_size, -1, -1, -1) #(batch_size, 1, num_relations_b, hidden_size) #repeat through the batch
        
        
        #attention scores
        #tanh to add nonlinearity
        #forward
        e = torch.tanh(
            self.W_r(relation_exp) + self.W_g(h_g_exp) + self.W_x(token_embs_exp)
        ) # (batch_size, seq_len, num_relations, hidden_size)
        
        
        #backward
        b_e = torch.tanh(
            self.W_r_b(b_relation_exp) + self.W_g_b(h_g_exp) + self.W_x_b(token_embs_exp)
        ) # (batch_size, seq_len, num_relations, hidden_size)
        
        
        #self.V is linear (hidden, 1) because we want a scalar value for each token (to calculate the relevance between the token and the relation)
        v_e = self.V(e).squeeze(-1)  # shape: (batch_size, seq_len, num_relations)
        b_v_e = self.V(b_e).squeeze(-1)  # shape: (batch_size, seq_len, num_relations_b)
        
        #normalize attention score
        #on dim = 1 because on the sentences dimension because we want to distribute attention on tokens, like we want to get probability distribution for all tokens if they are relevant to the relation
        A = F.softmax(v_e, dim=1)  # (batch_size, seq_len, num_relations) 
        A_exp = A.unsqueeze(-1) # (batch_size, seq_len, num_relations, 1)
        C = torch.sum(A_exp * token_embs_exp, dim=1)  # (batch_size ,num_relations, hidden_size)
        
        #backward:
        b_A = F.softmax(b_v_e, dim=1)  # (batch_size, seq_len, num_relations_B) 
        b_A_exp = b_A.unsqueeze(-1) # (batch_size, seq_len, num_relations_B, 1)
        b_C = torch.sum(b_A_exp * token_embs_exp, dim=1)  # (batch_size ,num_relations_b, hidden_size)
        
        
        
        W_x_xi = self.W_x2(token_embs) #(batch_size, seq_len,  hidden_size)
        h_i_k = s_k_w + W_x_xi #(batch_size, seq_len,  hidden_size)
        h_i_k_exp = h_i_k.unsqueeze(2) #(batch_size, seq_len, 1,  hidden_size)
        
        
        C_exp = C.unsqueeze(1) # (batch_size, 1 ,num_relations, hidden_size)
        b_C_exp = b_C.unsqueeze(1) # (batch_size, 1 ,num_relations_b, hidden_size)
        # token_embs_exp (batch_size, seq_len, 1, hidden_size) 
        h_i_j = C_exp + token_embs_exp  # (batch_size, seq_len, num_relations, hidden_size) 
        b_h_i_j = b_C_exp + token_embs_exp  # (batch_size, seq_len, num_relations, hidden_size) 
        
        
        H_i_j_k = h_i_j + h_i_k_exp
        b_H_i_j_k = b_h_i_j + h_i_k_exp
        
        
        #For each sentence, for each token, for each relation, what is the probability that this token is the start (or end) of the object?
        obj_start_probs = self.sigmoid(self.start_object_fc(H_i_j_k )).squeeze(-1) # (batch_size, seq_len, num_relations)
        obj_end_probs = self.sigmoid(self.end_object_fc(H_i_j_k)).squeeze(-1)  # (batch_size, seq_len, num_relations)

        b_sub_start_probs = self.sigmoid(self.b_start_subj_fc(b_H_i_j_k )).squeeze(-1) # (batch_size, seq_len, num_relations)
        b_sub_end_probs = self.sigmoid(self.b_end_subj_fc(b_H_i_j_k)).squeeze(-1)  # (batch_size, seq_len, num_relations)
        
        forward_triples  = extract_triples(token_embs, subj_idxs, obj_start_probs, obj_end_probs)
        backward_triples  = extract_triples(token_embs, b_obj_idxs, b_sub_start_probs, b_sub_end_probs)
        
        merge_triples(forward_triples, backward_triples )

        
        return forward_triples, backward_triples
        
def train_model(descriptions_dataset,relations_dataset, transE_emb_size=100):
  
  

  model = MyBRASKModel(transE_emb_size=transE_emb_size)
  forward_triples, backward_triples = model(descriptions_dataset, relations_dataset)
  return forward_triples, backward_triples
  
if __name__ == '__main__':
    k  = 1_000
    transE_emb_size = 100

    use_cached_ds = False

    if use_cached_ds: 
      print("reading cached desc ds")
      descriptions_dataset = read_cached_array(TEMP_FILES["descriptions_ds"])
      print("reading cached rels ds")
      relations_dataset = read_cached_array( TEMP_FILES["relations_ds"])
      
    else:
      # print("reading descriptions.. ")
      # descriptions = read_cached_array(PKLS_FILES["descriptions_normalized"][k])
      # print("reading relations")
      relations = read_cached_array(PKLS_FILES["relations"][k])
      print("reading relations transe ")
      relations_transE = read_cached_array(PKLS_FILES["transE_relation_embeddings"])
      
      print("creating desc ds ")
      # descriptions_dataset = SentencesDS(descriptions)
      # cache_array(descriptions_dataset, TEMP_FILES["descriptions_ds"])
      print("create relations ds")
      relations_dataset = RelationsDS(relations, relations_transE)
      cache_array(relations_dataset, TEMP_FILES["relations_ds"])


    forward_triples, backward_triples = train_model(k, transE_emb_size=transE_emb_size, )
    
    forward_triples_f = PKLS_FILES["forward_triples"][k]
    backward_triples_f = PKLS_FILES["backward_triples"][k]
    
    cache_array(forward_triples, forward_triples_f)
    cache_array(backward_triples, backward_triples_f)
    
    
    
