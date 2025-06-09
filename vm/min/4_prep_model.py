import torch

L = MAX_LENGTH = 128
USE_CACHE=  False

device_str = 'cpu'
if torch.cuda.is_available():
  device_str = "cuda"

device = torch.device(device_str)

if device_str == 'cpu':
    NUM_WORKERS = 1

else:
    NUM_WORKERS = 86



import os
import numpy as np
import pickle
from tqdm import tqdm


from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset

from transformers import BertModel, BertTokenizerFast



root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"
TEMP_FOLDER = f"{root}/data/temp"

DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"


RESULT_FILES  = {
    "descriptions_unormalized": f"{RESULTS_FOLDER}/descriptions_unormalized.pkl",
    "descriptions": f"{RESULTS_FOLDER}/descriptions.pkl",
    "aliases": f"{RESULTS_FOLDER}/aliases.pkl",
    "alias_patterns": f"{RESULTS_FOLDER}/aliases_patterns.pkl",
    "relations": f"{RESULTS_FOLDER}/relations.pkl",
    "triples": f"{RESULTS_FOLDER}/triples.pkl",
    "transE_relation_embeddings": f"{RESULTS_FOLDER}/transE_rel_embs.pkl",
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



def keep_descriptions_having_silver_spans(descriptions_dict):
    print("reading tensors")
    ss_h_s = read_tensor(RESULT_FILES["silver_spans"]["head_start"])
    ss_h_e = read_tensor(RESULT_FILES["silver_spans"]["head_end"])
    ss_t_s = read_tensor(RESULT_FILES["silver_spans"]["tail_start"])
    ss_t_e = read_tensor(RESULT_FILES["silver_spans"]["tail_end"])
    descs_ids = read_cached_array(RESULT_FILES["silver_spans"]["desc_ids"])
    print("creating mask")
    mask = (
        ss_h_s.any(dim=1) &
        ss_h_e.any(dim=1) &
        ss_t_s.any(dim=1) &
        ss_t_e.any(dim=1)
    )
    print("cleaning")
    cleaned_desc_dict = {
        k: descriptions_dict[k]
        for k, keep in tqdm(
            zip(descs_ids, mask), 
            total=len(descs_ids), 
            desc="filtering descriptions")
        if keep.item()
    }
    print(f"we were having {len(descs_ids)} descriptions, now: {len(cleaned_desc_dict)}, saving..")
    cache_array(cleaned_desc_dict, RESULT_FILES["descriptions"])
    print("saving silver spans")
    save_tensor(ss_h_s[mask], RESULT_FILES["silver_spans"]["head_start"])
    save_tensor(ss_h_e[mask], RESULT_FILES["silver_spans"]["head_end"])
    save_tensor(ss_t_s[mask], RESULT_FILES["silver_spans"]["tail_start"])
    save_tensor(ss_t_e[mask], RESULT_FILES["silver_spans"]["tail_end"])
    filtered_desc_ids = [k for k,keep in zip(descs_ids, mask) if keep.item()]
    cache_array(filtered_desc_ids, RESULT_FILES["silver_spans"]["desc_ids"])
    return True


def get_hgs(sentences, tokenizer, model):
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH
        )

    dataset = HFDataset.from_dict({"text": sentences})
    print("tokenizing")
    encoded = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_WORKERS
    )
    input_ids = torch.tensor(encoded["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).to(device)
    print("getting embs")
    with torch.no_grad():
        bert_out = model(input_ids=input_ids, attention_mask=attention_mask)
        embs = bert_out.last_hidden_state # (batch_size, L, hidden_size)

    print("multiplying attention masks")
    attention_mask = attention_mask.unsqueeze(-1) #(batch_size, L, 1)
    sum_embs = (embs * attention_mask).sum(dim=1)
    token_counts = attention_mask.sum(dim=1).clamp(min=1)
    mean_embs = sum_embs / token_counts
    return mean_embs, embs


if __name__ == "__main__":
    descriptions_all = read_cached_array(RESULT_FILES["descriptions"])
    # keep_descriptions_having_silver_spans(descriptions_all)

    tokenizer =BertTokenizerFast.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    sentences = list(descriptions_all.values())
    h_gs, embs = get_hgs(sentences, tokenizer, model)
    assert embs.shape[0] == len(descriptions_all)
    assert embs.shape[1] == MAX_LENGTH
    save_tensor(h_gs, RESULT_FILES["embs"]["hgs"])
    save_tensor(embs, RESULT_FILES["embs"]["embs"])