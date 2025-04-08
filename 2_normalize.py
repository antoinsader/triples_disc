from collections import defaultdict
import re 
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from functools import partial
import unicodedata

from utils.utils import batch_dict, read_cached_array, replace_special_chars, cache_array
from Data import get_compiled_strange_chars, PKLS_FILES


def normalize_description_BRASK(sentence):
    """
        args:
            - sentence: need to be normalized 
        does:
            - use nfkc and remove multiple white spaces
        return:
            - normalized sentence
    """
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    # sentence = re.sub(r'\([^)]*\)', '', sentence)
    return sentence


def normalize_desc_batch_BRASK(descs_batch):
    """
        args:
            descs_batch: a batch of description dictionary 
        does:
            - Replace strange chars in the batch texts
            - Execute normalize_description_BRASK on every sentence in the dictionary
        return: 
            normalized dict 
    """
    compiled_strange_patterns = get_compiled_strange_chars()
    descs_batch = {k:  replace_special_chars(v, compiled_strange_patterns) for k,v in descs_batch.items()}
    
    
    new_dict=  {}
    for desc_id, desc in tqdm(descs_batch.items(), total=len(descs_batch.keys()), desc="Normalizing batch description"):
        new_dict[desc_id]= normalize_description_BRASK(desc)
  
    return new_dict

    
    
def normalize_desc_parallel(descs_all, num_workers = 8):
    """
        args:
            descs_all: dictionary containing all descriptions need to be normalized
            num_workers: the num of workers where parallel processing would be distributed
        does:
            - create batches from the description (for improvements I can use dataset dataloaders)
            - call the function normalize_desc_batch_BRASK by parallel processing 
            - merge the results and return them 
        returns: 
            - normalized description dictionary 
            
    """
    batch_size = max(1, len(descs_all) // num_workers)
    desc_batches = batch_dict(descs_all, batch_size)
    print(f"normalizing on numworkers: {num_workers}, batch_size is {batch_size} ")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        normalize_func = partial(normalize_desc_batch_BRASK)
        results = list(tqdm(
            executor.map(normalize_func, desc_batches),
            total=len(desc_batches),
            desc="Processing batches"
        ))
        
    normalized_descs = {k: v for batch in results for k, v in batch.items()}
    return normalized_descs


if __name__ == '__main__':
    k = "full"
    num_workers = 4
    descriptions = read_cached_array(PKLS_FILES["descriptions"][k])
    normalized_descs = normalize_desc_parallel(descriptions, num_workers)
    cache_array(normalized_descs, PKLS_FILES["descriptions_normalized"][k])
    
    
    # aliases = read_cached_array(PKLS_FILES["aliases_dict"])
    # normalized_als = normalize_desc_parallel(aliases, num_workers)
    # als_rev = defaultdict(list)
    # for als, als_id in normalized_als.items():
    #     als_rev[als_id].append(als)
    # cache_array(als_rev, PKLS_FILES["aliases_rev_norm"])
    
    
