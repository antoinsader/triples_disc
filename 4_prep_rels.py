from transformers import BertModel, BertTokenizer
import torch 
from Data import PKLS_FILES
from utils.utils import read_cached_array, cache_array

def extract_relations(triples):
    """
        args:
            triples: is a dictionary in shape {head: [list]} each item in the list is a tuple (h,r,t)
        returns: 
            set of r which is the ids of the relations
    """
    rels = set()
    for _, tr_list in triples.items():
        for (_,r, _) in tr_list:
            rels.add(r)

    return list(rels)



def get_rel_embs(relations, relations_full_dict):
    """
        args:
            relations: a list of relation ids we want to embed 
            relations_full_dict: a dictionary containing all relations in shape: {rel_id: [list of relation aliases]}

        do: 
            loop through each relation id
            get the list of strings 
            get embedding of each string by getting last two embeding layers of bert 
            average all embedings for the relation 
        return: 
            list of all_rels_embs which contains for each relation in relations, a vector representing it
    """
    tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    model.eval()
    all_rels_embs = []

    for rel_id in relations:
        if rel_id not in relations_full_dict.keys():
            print(f"rel_id was not found in all relations")
            continue
        rel_embs = [] # for each relation string append the embedding
        for rel_str in relations_full_dict[rel_id]:
            inputs = tokenizer(rel_str, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_layer = hidden_states[-1].squeeze(0)
            before_last_layer = hidden_states[-2].squeeze(0)
            average_layers = (last_layer + before_last_layer) / 2.0
            r_j = average_layers.mean(dim=0)
            rel_embs.append(r_j)
        
        #average the rel_embs to represent one relation 
        rel_emb = torch.mean(torch.stack(rel_embs), dim=0)
        all_rels_embs.append(rel_emb)
    return all_rels_embs


def get_rel_embs_opt(relations, relations_full_dict):
    """
        args:
            relations: a list of relation ids we want to embed 
            relations_full_dict: a dictionary containing all relations in shape: {rel_id: [list of relation aliases]}

        do: 
            Optimized version of the function get_rel_embs
        return: 
            list of all_rels_embs which contains for each relation in relations, a vector representing it
    """
    tokenizer =BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    model.eval()
    all_rels_embs = []

    for rel_id in relations:
        if rel_id not in relations_full_dict.keys():
            print(f"rel_id was not found in all relations")
            continue
        
        aliases = relations_full_dict[rel_id]
        inputs = tokenizer(aliases, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states 

        #get last two layers
        last_layer = hidden_states[-1] #(batch_size, seq_len, hidden_size)
        second_last_layer  = hidden_states[-2] #(batch_size, seq_len, hidden_size)
        average_layers = (last_layer + second_last_layer) / 2.0

        #use mask to average only over real tokens (avoid padded)
        mask = inputs['attention_mask'].unsqueeze(-1).float()  # shape: (batch_size, seq_len, 1)
        sum_embeddings = torch.sum(average_layers * mask, dim=1)
        lengths = torch.sum(mask, dim=1)
        sentence_embs = sum_embeddings / lengths #shape: (batch_size, hidden_size, )

        #average all sentences embeddings for this relation
        rel_emb = sentence_embs.mean(dim=0)
        all_rels_embs.append(rel_emb)
    return all_rels_embs


if __name__ == "__main__": 
    k = 100
    print("reading dictionaries..")
    triples = read_cached_array(PKLS_FILES["triples"][k])
    relations_full_dict  = read_cached_array(PKLS_FILES["relations"]["full"])
    relations = extract_relations(triples) # [...num_relations] each one is id

    print("processing..")
    rel_embs_opt = get_rel_embs_opt(relations, relations_full_dict)

    print(f" len(rel_embs_opt): {len(rel_embs_opt)}, shape[0]: {rel_embs_opt[0].shape}")
    rel_embs = torch.stack(rel_embs_opt, dim=0)
    cache_array(rel_embs, PKLS_FILES["relations_embs"][k] )
    
