# pip install numpy  tqdm
# pip install torch 
# pip install scikit-learn 
# pip install pykeen

#install raw files: 
# curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
# curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
# curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar



from collections import defaultdict
from tqdm import tqdm
import re 
from joblib import Parallel, delayed


from utils.utils import read_cached_array, cache_array,load_descriptions_dict_from_text, load_aliases_dict_from_text, load_relations, load_triples, minimize_dict
from Data import RAW_TXT_FILES, PKLS_FILES, HELPER_FILES


# will create descriptions dict from the raw text file and save the result in pickle file
#    descriptions_dict would have {document_id: description}
def create_descriptions_dict():
    desc_raw_f = RAW_TXT_FILES["descriptions"]
    descriptions_dict_f = PKLS_FILES["descriptions"]["full"]
    print(f"Full description is being created and will be saved in {descriptions_dict_f}...")
    descriptions_dict = load_descriptions_dict_from_text(desc_raw_f)
    cache_array(descriptions_dict, descriptions_dict_f)


# will create two aliases dicts from the raw text file of aliases and save them in pickle files
#    aliases_dict would have {alias_name: document_id}
#    aliases_dict_rev would have {document_id: [alias_name1, alias_name2, ...]}
def create_aliases_dicts():
    raw_f = RAW_TXT_FILES["aliases"]
    dict_f = PKLS_FILES["aliases_dict"]
    dict_rev_f = PKLS_FILES["aliases_rev"]
    
    
    print(f"Aliases are being created and will be saved in {dict_f} and {dict_rev_f}...")
    aliases_dict, aliases_rev = load_aliases_dict_from_text(raw_f)
    cache_array(aliases_dict, dict_f)
    cache_array(aliases_rev, dict_rev_f)

#   triples_dict would have {head: [triple1, triple2, ...]} and each triple is a triple (head, relation, tail)
#    head, relation and tail are all ids. head, tail id refering to document_id and relation id refering to relation_id
def create_triples_dict():
    raw_f = RAW_TXT_FILES["triples"]
    dict_f = PKLS_FILES["triples"]["full"]
    triples_dict = load_triples(raw_f)
    print(f"Saving triples in {dict_f}:")
    cache_array(triples_dict, dict_f)
    

#  relations_dict would have {relation_id: [relation_name1, relation_name2, ...]}
def create_relations_dict():
    raw_f = RAW_TXT_FILES["relations"]
    dict_f = PKLS_FILES["relations"]["full"]
    relations = load_relations(raw_f)
    print(f"Saving relations in {dict_f}:")
    cache_array(relations, dict_f)
    


# taking k from [10, 100, 1k, 10k, 1m] -> will not be exactly the number because we will add other documents 
#    will create desc_min_dict which is random k values from the full descriptions 

def create_min(k):
    keys_not_in_als = read_cached_array(HELPER_FILES["keys_not_in_als"])
    print("read dictionaries..")
    desc_all_dict = read_cached_array(PKLS_FILES["descriptions_normalized"]["full"])
    triples_all_dict = read_cached_array(PKLS_FILES["triples"]["full"])
    relations_all_dict = read_cached_array(PKLS_FILES["relations"]["full"])
    aliases_all_dict = read_cached_array(PKLS_FILES["aliases_rev_norm"])
    
    print("minimizing descriptions..")
    desc_min_dict = minimize_dict(desc_all_dict, k)
    
    
    desc_ids_to_add = set(desc_min_dict.keys())    
    relation_ids_to_add = set()
    
    
    min_triples = defaultdict(list)
    for d_id in tqdm(desc_min_dict.keys(), total = len(desc_min_dict), desc="getting triples from descriptions" ):
        triples_lst = triples_all_dict[d_id]
        for h, r, t in triples_lst:
            if h in keys_not_in_als or t in keys_not_in_als:
                continue
            relation_ids_to_add.add(r)
            desc_ids_to_add.add(h)
            desc_ids_to_add.add(t)
            min_triples[d_id].append((h,r,t))
    
    print("Creating full descriptions and full relations")
    last_min_desc_dict = {}
    for d_id in list(desc_ids_to_add):
        last_min_desc_dict[d_id] = desc_all_dict[d_id]
    
    last_min_relations_dict = {}
    for r_id in list(relation_ids_to_add):
        if r_id in relations_all_dict:
            last_min_relations_dict[r_id] = relations_all_dict[r_id]
    
    aliases_dict = {}
    for d_id in last_min_desc_dict.keys():
        if d_id in aliases_all_dict:
            aliases_dict[d_id] = aliases_all_dict[d_id]
        
        
    
    print(f"We have {len(last_min_desc_dict)} descriptions")
    print(f"We have {len(aliases_dict)} aliases heads")
    print(f"We have {len(min_triples)} head triples")
    print(f"We have {len(last_min_relations_dict)} relations")
    
    min_desc_dict_f = PKLS_FILES["descriptions_normalized"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    min_aliases_dict_f = PKLS_FILES["aliases"][k]
    
    
    
    cache_array(last_min_desc_dict, min_desc_dict_f)
    cache_array(min_triples, min_triples_dict_f)
    cache_array(last_min_relations_dict, min_relations_dict_f)
    cache_array(aliases_dict, min_aliases_dict_f)
    


def save_strange_chars_dict():
    compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in {
        r"[€“©]": "",
        r"[áăắặằẳẵǎâấậầẩẫäǟȧǡạȁàảȃāąåǻḁãǽǣ]": "a",
        r"[ḃḅḇ]": "b",
        r"[ćčçḉĉċ]": "c",
        r"[ďḑḓḋḍḏ]": "d",
        r"[éĕěȩḝêếệềểễḙëėẹȅèẻȇēḗḕęẽḛé]": "e",
        r"[ḟ]": "f",
        r"[ǵğǧģĝġḡ]": "g",
        r"[ḫȟḩĥḧḣḥẖ]": "h",
        r"[íĭǐîïḯi̇ịȉìỉȋīįĩḭı]": "i",
        r"[ǰĵ]": "j",
        r"[ḱǩķḳḵ]": "k",
        r"[ĺľļḽḷḹḻ]": "l",
        r"[ḿṁṃ]": "m",
        r"[ńňņṋṅṇǹṉñ]": "n",
        r"[óŏǒôốộồổỗöȫȯȱọőȍòỏơớợờởỡȏōṓṑǫǭõṍṏȭǿøɔ]": "o",
        r"[ṕṗ]": "p",
        r"[ŕřŗṙṛṝȑȓṟ]": "r",
        r"[śṥšṧşŝșṡẛṣṩ]": "s",
        r"[ťţṱțẗṫṭṯ]": "t",
        r"[úŭǔûṷüǘǚǜǖṳụűȕùủưứựừửữȗūṻųůũṹṵ]": "u",
        r"[ṿṽ]": "v",
        r"[ẃŵẅẇẉẁẘ]": "w",
        r"[ẍẋ]": "x",
        r"[ýŷÿẏỵỳỷȳẙỹy]": "y",
        r"[źžẑżẓẕʐ]": "z",
        r"[&]": "and"
    }.items()]
    cache_array(compiled_patterns, HELPER_FILES['strange_chars'])

def save_desc_keys_not_in_als():
    print("reading dicts...")
    aliases_all_dict = read_cached_array(PKLS_FILES["aliases_rev"])
    desc_all_dict = read_cached_array(PKLS_FILES["descriptions"]["full"])

    desc_keys = desc_all_dict.keys()
    als_keys_set = set(aliases_all_dict.keys())
    print("starting..")
    not_in_als = [d_id for d_id in desc_keys if d_id not in als_keys_set]
    

    print(f"{len(not_in_als)}/{len(desc_keys)} of desc keys not are in  aliases")
    cache_array(not_in_als, HELPER_FILES["keys_not_in_als"])


    
if __name__ == '__main__':
    save_strange_chars_dict()
    create_descriptions_dict()
    create_aliases_dicts()
    create_triples_dict()
    create_relations_dict()
    
    save_desc_keys_not_in_als()
    
    
    
    #do create min only after normalization
    # create_min(10)