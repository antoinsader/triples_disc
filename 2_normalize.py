from collections import defaultdict
import re 
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from functools import partial
import unicodedata

from utils.utils import batch_dict, read_cached_array, replace_special_chars, cache_array
from Data import get_compiled_strange_chars, PKLS_FILES
import nltk
from nltk.corpus import words
# nltk.download('words') 


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
            -   Remove non-english words
            - Execute normalize_description_BRASK on every sentence in the dictionary
        return: 
            normalized dict 
    """
    compiled_strange_patterns = get_compiled_strange_chars()
    descs_batch = {k:  replace_special_chars(v, compiled_strange_patterns) for k,v in descs_batch.items()}
    
    
    valid_word_pattern = re.compile(r"^[A-Za-z0-9.,!?;:'\"()\-]+$")
    def keep_only_english_chars(text):
        return ' '.join(word for word in text.split() if valid_word_pattern.match(word))
    
    descs_batch = {k: keep_only_english_chars(v) for k,v in descs_batch.items()}
    
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
    k = 1000
    num_workers = 1
  
    # descriptions = read_cached_array(PKLS_FILES["descriptions"][k])
    # normalized_descs = normalize_desc_parallel(descriptions, num_workers)
    # cache_array(normalized_descs, PKLS_FILES["descriptions_normalized"][k])
    
    aliases = read_cached_array(PKLS_FILES["aliases"][k])
    normalized_aliases =  {k:  list(set(   [al.lower()  for al in v]   )) for k, v in tqdm(aliases.items(), total=len(aliases))}
    cache_array(normalized_aliases, PKLS_FILES["aliases"][k])
    
    
  