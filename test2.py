import pickle 
import re 
import concurrent.futures
from functools import partial
from tqdm import tqdm 
import unicodedata

from collections import defaultdict


aliases = {
    "q1": ["q one", "q 1", "q-1", "Q 1"],
    "q2": ["q two", "q 2", "q-2", "q 2 يب"],
    "q3": ["q three", "q 3", "q-3", "àù2"],
    "q4": ["q four", "q 4", "q-4"],
    "q5": ["q five", "q 5", "q-5"],
    "q6": ["q six", "q 6", "q-6"],
    "q7": ["q seven", "q 7", "q-7"],
    "q8": ["q eiight", "q 8", "q-8"],
    "q9": ["q nine", "q 9", "q-9"],
    "q10": ["q ten", "q 10", "q-10"],
    "q11": ["q eleven", "q 11", "q-11"],
    "q12": ["q twelve", "q 12", "q-12"],
}

desc_dict_all = {
    "q1": "I àm     q1 يسش  text", 
    "q2": "I am q2 text", 
    "q3": "I am q3 text", 
    "q4": "I am q4 text", 
    "q5": "I am q5 text", 
    "q6": "I am q6 text", 
    "q7": "I am q7 text", 
    "q8": "I am q8 text", 
    "q9": "I am q9 text", 
    "q10": "I am q10 text", 
    "q11": "I am q11 text", 
    "q12": "I am q12 text", 
    "q13": "I am q13 text", 
    "q14": "I am q14 text", 
    "q15": "I am q15 text", 
    "q16": "I am q16 text", 
}

def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)



def replace_special_chars(text, compiled_patterns):
    for pattern, replacement in compiled_patterns:
        text = pattern.sub(replacement, text)
    return text


def normalize_als_batch(als_batch_dict):
    compiled_strange_chars = read_cached_array(f"./data/helpers/strange_chars.pkl")
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    new_dict = defaultdict(list)
    for k, val in als_batch_dict.items():
        local_set = set()
        for als in val:
            aa  = replace_special_chars(als, compiled_strange_chars)
            aa = keep_only_english_chars(aa)
            aa = unicodedata.normalize('NFKC', aa)
            aa = re.sub(r'\s+', ' ', aa).strip()
            aa = aa.lower()
            local_set.add(aa)

        new_dict[k]  = list(local_set)
    return new_dict


def normalize_desc_batch(descs_batch_dict):
    compiled_strange_chars = read_cached_array(f"./data/helpers/strange_chars.pkl")
    
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    new_dict = {}
    for k, val in descs_batch_dict.items():
        val = replace_special_chars(val, compiled_strange_chars)
        val = keep_only_english_chars(val)
        val = unicodedata.normalize('NFKC', val)
        val = re.sub(r'\s+', ' ', val).strip()
        new_dict[k]  = val

    return new_dict

    
def normalize_descriptions():
    descs = desc_dict_all
    batch_size =  len(descs) // 4
    items = list(descs.items())
    
    batches = [
        dict(items[i : i+batch_size])
        for i in range(0, len(items), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_desc_batch)
        results = list(tqdm(
            executor.map(normalize_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    dic_norm = {k: v for batch in results for k,v in batch.items()}
    print(dic_norm)

def normalize_aliases():

    batch_size =  len(aliases) // 4
    items = list(aliases.items())

    batches = [
        dict(items[i : i+batch_size])
        for i in range(0, len(items), batch_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_als_batch)
        results = list(tqdm(
            executor.map(normalize_func, batches),
            total=len(batches),
            desc="Processing batches"
        ))

    dic_norm = {k: v for batch in results for k,v in batch.items()}
    print(dic_norm)

    # return dic_norm
if __name__ == "__main__":
    normalize_aliases()
    normalize_descriptions()