
import torch
from torch.nn.utils.rnn import pad_sequence

#greattttttttt approach for optimization
def get_h_gs_opt(sentences, tokenizer, model, max_length, device, num_workers):
    print(f"tokenizing ")
    dataset = HFDataset.from_dict({"text": sentences})
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )
    encoded = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers
    )
    input_ids = torch.tensor(encoded['input_ids']).to(device)
    attention_mask = torch.tensor(encoded['attention_mask']).to(device)

    
    print(f"after tokenizing ")
    with torch.no_grad():
        bert_output = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state # (batch_size, seq_len ,hidden_size)


    attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Sum only valid token embeddings
    sum_embeddings = (embeddings * attention_mask).sum(dim=1)
    token_counts = attention_mask.sum(dim=1).clamp(min=1)

    mean_embeddings = sum_embeddings / token_counts

    return mean_embeddings, embeddings


def get_h_gs(sentences, tokenizer, model, max_length):
    encoded = tokenizer(sentences, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
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



