
import os

from utils.utils import read_cached_array


root_path = os.path.dirname(os.path.abspath(__file__))
root = "."

RAW_FOLDER = f"{root}/data/raw"
HELPERS_FOLDER = f"{root}/data/helpers"
TRANSE_CHECKPOINT_FOLDER = f"{root}/data/transe/checkpoint"

TEMP_FOLDER = f"{root}/data/temp"
TRANSE_FOLDER = f"{root}/data/transe"
CHECKPOINTS_FOLDER = f"{root}/data/checkpoints"
DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
DESCRIPTIONS_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
ALIASES_FOLDER = f"{DICTIONARIES_FOLDER}/aliases"
RELATIONS_FOLDER = f"{DICTIONARIES_FOLDER}/relations"
GOLDEN_TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/golden_triples"
SILVER_SPANS_FOLDER = f"{DICTIONARIES_FOLDER}/silver_spans"


LOG_FOLDER = f"{root}/logs"
LOG_FOLDER_BUILD_GOLDEN = f"{root}/logs/golden_truth_build"

folders_to_check = [CHECKPOINTS_FOLDER, SILVER_SPANS_FOLDER, GOLDEN_TRIPLES_FOLDER, LOG_FOLDER_BUILD_GOLDEN, LOG_FOLDER, ALIASES_FOLDER, TRANSE_CHECKPOINT_FOLDER, TEMP_FOLDER, TRANSE_FOLDER, DESCRIPTIONS_NORMALIZED_FOLDER, HELPERS_FOLDER, TRIPLES_FOLDER, RELATIONS_FOLDER,DICTIONARIES_FOLDER, DESCRIPTIONS_FOLDER]
for fo in folders_to_check:
    if not os.path.exists(fo):
        os.makedirs(fo)


RAW_TXT_FILES ={
    "descriptions": f"{RAW_FOLDER}/wikidata5m_text.txt",    
    "aliases": f"{RAW_FOLDER}/wikidata5m_entity.txt", 
    "relations": f"{RAW_FOLDER}/wikidata5m_relation.txt",
    "triples": f"{RAW_FOLDER}/wikidata5m_transductive_train.txt",
}

THEMODEL_PATH = "BRASK_MODEL.pth"

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
        1_000_000: f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_1m.pkl",
    },
    "descriptions_tokenized": {
        "full_input_ids": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_input_ids.pkl",
        "full_attention_mask": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_attention_masks.pkl",
        "sentence_tokens": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_full_sentence_tokens.pkl",
        "aliases_tokenized": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/aliases_tokenized.pkl"
        
        
    },
    "aliases_dict": f"{DICTIONARIES_FOLDER}/aliases_dict.pkl",
    "aliases_rev": f"{DICTIONARIES_FOLDER}/aliases_rev.pkl",
    "aliases_rev_norm": f"{DICTIONARIES_FOLDER}/aliases_rev_norm.pkl",
    #This is aliases with keys as query_id and value is a list (it should be named aliases_dict)
    "aliases":{
        "full": f"{ALIASES_FOLDER}/aliases_full.pkl",
        10: f"{ALIASES_FOLDER}/aliases_min_10.pkl",
        100: f"{ALIASES_FOLDER}/aliases_min_100.pkl",
        1_000: f"{ALIASES_FOLDER}/aliases_min_1k.pkl",
        10_000: f"{ALIASES_FOLDER}/aliases_min_10k.pkl",
        1_000_000: f"{ALIASES_FOLDER}/aliases_min_1m.pkl"
        
    },
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
    
    "golden_triples": {
        "full": f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_full.pkl",
        10: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_10.pkl",
        100: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_100.pkl",
        1_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_1k.pkl",
        10_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_10k.pkl",
        1_000_000: f"{GOLDEN_TRIPLES_FOLDER}/golden_triples_min_1m.pkl"
    },
    
    "silver_spans": {
        "full": f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_full.pkl",
        10: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_10.pkl",
        100: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_100.pkl",
        1_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_1k.pkl",
        10_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_10k.pkl",
        1_000_000: f"{GOLDEN_TRIPLES_FOLDER}/silver_spans_min_1m.pkl"
    },
    "transE_relation_embeddings": f"{TRANSE_FOLDER}/relation_embs.pkl" ,
    "transE_entity_embeddings": f"{TRANSE_FOLDER}/entity_embs.pkl" ,
    "ss_triples_head_aliases": f"{GOLDEN_TRIPLES_FOLDER}/head_aliases.pkl",
    "ss_triples_tail_aliases": f"{GOLDEN_TRIPLES_FOLDER}/tail_aliases.pkl",
    "alias_pattern_map": f"{GOLDEN_TRIPLES_FOLDER}/alias_patterns_map.pkl",
    "encoded_tokenization": f"{DICTIONARIES_FOLDER}/encoded_tokenization.pkl",
    "h_g_s_all_means": f"{DICTIONARIES_FOLDER}/h_g_s_all_means.pt",
    "h_g_s_all_embs": f"{DICTIONARIES_FOLDER}/h_g_s_all_embs.pt",
    
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
    "strange_chars": f"{HELPERS_FOLDER}/strange_chars.pkl",
    "keys_not_in_als": f"{HELPERS_FOLDER}/keys_not_in_als.pkl"
}

CHECKPOINT_FILES = {
    "transe": f"{CHECKPOINTS_FOLDER}/transE_checkpoint.pth",
    "brask": f"{CHECKPOINTS_FOLDER}/BRASK_checkpoint.pth"
}

LOGGER_FILES = {
    "build_golden_triples": f"{LOG_FOLDER_BUILD_GOLDEN}/build_golden_triples.log"
}

def get_min_descriptionsNorm_triples_relations(k):
    min_desc_norm_dict_f = PKLS_FILES["descriptions_normalized"][k]
    min_triples_dict_f = PKLS_FILES["triples"][k]
    min_relations_dict_f = PKLS_FILES["relations"][k]
    min_aliases_dict_f = PKLS_FILES["aliases"][k]
    
    descs = read_cached_array(min_desc_norm_dict_f)
    triples = read_cached_array(min_triples_dict_f)
    relations = read_cached_array(min_relations_dict_f)
    aliases = read_cached_array(min_aliases_dict_f)
    
    return descs, triples, relations, aliases

def get_compiled_strange_chars():
    f = HELPER_FILES["strange_chars"]
    return  read_cached_array(f)

