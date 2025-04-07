
import os

from utils.utils import read_cached_array


root_path = os.path.dirname(os.path.abspath(__file__))
root = "."

RAW_FOLDER = f"{root}/data/raw"
HELPERS_FOLDER = f"{root}/data/helpers"
TRANSE_CHECKPOINT_FOLDER = f"{root}/data/transe/checkpoint"

TEMP_FOLDER = f"{root}/data/temp"
TRANSE_FOLDER = f"{root}/data/transe"
DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
DESCRIPTIONS_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
RELATIONS_FOLDER = f"{DICTIONARIES_FOLDER}/relations"

folders_to_check = [TRANSE_CHECKPOINT_FOLDER, TEMP_FOLDER, TRANSE_FOLDER, DESCRIPTIONS_NORMALIZED_FOLDER, HELPERS_FOLDER, TRIPLES_FOLDER, RELATIONS_FOLDER,DICTIONARIES_FOLDER, DESCRIPTIONS_FOLDER]
for fo in folders_to_check:
    if not os.path.exists(fo):
        os.makedirs(fo)


RAW_TXT_FILES ={
    "descriptions": f"{RAW_FOLDER}/wikidata5m_text.txt",    
    "aliases": f"{RAW_FOLDER}/wikidata5m_entity.txt", 
    "relations": f"{RAW_FOLDER}/wikidata5m_relation.txt",
    "triples": f"{RAW_FOLDER}/wikidata5m_transductive_train.txt",
}



PKLS_FILES = {
    "descriptions": {
        "full": f"{DESCRIPTIONS_FOLDER}/descriptions_full.pkl",
        10: f"{DESCRIPTIONS_FOLDER}/descriptions_min_10.pkl",
        100: f"{DESCRIPTIONS_FOLDER}/descriptions_min_100.pkl",
        1_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_1k.pkl",
        10_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_10k.pkl",
        1_000_000: f"{DESCRIPTIONS_FOLDER}/descriptions_min_1m.pkl"
    },
    "descriptions_normalized": {
        "full": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full.pkl",
        10: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_10.pkl",
        100: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_100.pkl",
        1_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1k.pkl",
        10_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_10k.pkl",
        1_000_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1m.pkl"
    },
    "aliases_dict": f"{DICTIONARIES_FOLDER}/aliases_dict.pkl",
    "aliases_rev": f"{DICTIONARIES_FOLDER}/aliases_rev.pkl",
    
    "triples": {
        "full": f"{TRIPLES_FOLDER}/triples_full.pkl",
        10: f"{TRIPLES_FOLDER}/triples_min_10.pkl",
        100: f"{TRIPLES_FOLDER}/triples_min_100.pkl",
        1_000: f"{TRIPLES_FOLDER}/triples_min_1k.pkl",
        10_000: f"{TRIPLES_FOLDER}/triples_min_10k.pkl",
        1_000_000: f"{TRIPLES_FOLDER}/triples_min_1m.pkl"
    },
    "relations": {
        "full": f"{RELATIONS_FOLDER}/relations_full.pkl",
        10: f"{RELATIONS_FOLDER}/relations_min_10.pkl",
        100: f"{RELATIONS_FOLDER}/relations_min_100.pkl",
        1_000: f"{RELATIONS_FOLDER}/relations_min_1k.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_min_1m.pkl"
    },
    "relations_embs": {
        "full": f"{RELATIONS_FOLDER}/relations_embs_full.pkl",
        10: f"{RELATIONS_FOLDER}/relations_embs_min_10.pkl",
        100: f"{RELATIONS_FOLDER}/relations_embs_min_100.pkl",
        1_000: f"{RELATIONS_FOLDER}/relations_embs_min_1k.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_embs_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_embs_min_1m.pkl"
    },
    "forward_triples": {
        "full": f"{RELATIONS_FOLDER}/relations_full.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_min_1m.pkl"
    },
    "backward_triples": {
        "full": f"{RELATIONS_FOLDER}/relations_full.pkl",
        10_000: f"{RELATIONS_FOLDER}/relations_min_10k.pkl",
        1_000_000: f"{RELATIONS_FOLDER}/relations_min_1m.pkl"
    },
    "transE_relation_embeddings": f"{TRANSE_FOLDER}/relation_embs.pkl" ,
    "transE_entity_embeddings": f"{TRANSE_FOLDER}/entity_embs.pkl" ,


}


TEMP_FILES = {
    "dataset":{
         "full": f"{TEMP_FOLDER}/dataset_full.pkl",
        10: f"{TEMP_FOLDER}/dataset_min_10.pkl",
        100: f"{TEMP_FOLDER}/dataset_min_100.pkl",
        1_000: f"{TEMP_FOLDER}/dataset_min_1k.pkl",
        10_000: f"{TEMP_FOLDER}/dataset_min_10k.pkl",
        1_000_000: f"{TEMP_FOLDER}/dataset_min_1m.pkl"
    } 
}

HELPER_FILES = {
    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl"
}

def get_min_descriptionsNorm_triples_relations(k):
    min_desc_norm_dict_f = PKLS_FILES["descriptions_normalized"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    
    descs = read_cached_array(min_desc_norm_dict_f)
    triples = read_cached_array(min_triples_dict_f)
    relations = read_cached_array(min_relations_dict_f)
    
    return descs, triples, relations

def get_compiled_strange_chars():
    f = HELPER_FILES["strange_chars"]
    return  read_cached_array(f)

