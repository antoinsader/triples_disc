
NUM_WORKERS = 5


import pickle

import re
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count
from collections import defaultdict




root = "."
RAW_FOLDER = f"{root}/data/raw"
RESULTS_FOLDER = f"{root}/data/results"

DICTIONARIES_FOLDER = f"{root}/data/dictionaries"
TRIPLES_FOLDER = f"{DICTIONARIES_FOLDER}/triples"
DESCRIPTIONS_NORMALIZED_FOLDER = f"{DICTIONARIES_FOLDER}/descriptions_normalized"

RESULT_FILES  = {
    "descriptions": f"{DESCRIPTIONS_NORMALIZED_FOLDER}/descriptions_min_100.pkl",
    "aliases": f"{DICTIONARIES_FOLDER}/aliases_dict.pkl",
    "triples": f"{TRIPLES_FOLDER}/triples_min_100.pkl",
}



def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")

def split_list(data, num_chunks):
    chunk_size = (len(data) + num_chunks - 1) // num_chunks  # ceiling division
    return [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]




_shared_aliases = None
_shared_triples = None

def init_globals(aliases_, triples_):
    print("intiiating globals")
    global _shared_aliases, _shared_triples
    _shared_aliases = aliases_
    _shared_triples = triples_
    print("finsihed initiating")

# aliases = {
#     "q1": ["que 1", "qqqq 1"],
#     "q2": ["q two", "que 2", "qqqq 2"],
#     "q3": ["que 3", "qqqq 3"],
#     "q4": [ "qqqq 4"],
#     "q5": ["que 5", "qqqq 5"],
#     "q6": ["q six", "qqqq 6"],
#     "q7": ["q svn", "que 7", ],
#     "q8": ["q right", "que 8", ],
#     "q9": ["q nine", "que 9", ],
#     "q10": ["q ten", "que 10", ],
# }

# triples  = {
#     "q1": [("q1", "r1", "q2"), ("q1", "r2", "q5")],
#     "q2": [("q2", "r1", "q3"), ("q2", "r2", "q6")],
#     "q3": [("q3", "r1", "q4"), ("q3", "r2", "q7")],
#     "q4": [("q4", "r1", "q5"), ("q4", "r2", "q8")],
#     "q5": [("q5", "r1", "q6"), ("q5", "r2", "q9")],
#     "q6": [("q6", "r1", "q7"), ("q6", "r2", "q10")],
#     "q7": [("q7", "r1", "q8"), ],
#     "q8": [("q8", "r1", "q9"), ],
#     "q9": [("q9", "r1", "q10"), ],
#     "q10": [("q10", "r1", "q1"),],
# }


# full_descs_dict = {
#     "q1": "I am q1",
#     "q2": "I am q2",
#     "q3": "I am q3",
#     "q4": "I am q4",
#     "q5": "I am q5",
#     "q6": "I am q6",
#     "q7": "I am q7",
#     "q8": "I am q8",
#     "q9": "I am q9",
#     "q10": "I am q10",
# }


def worker(descs_ids_chunk):
    print(f"new cpu worker proccessing {len(descs_ids_chunk)}")
    tails_aliases = {}
    heads_aliases = {}
    alias_pattern_map = {}
    for d_id in tqdm(descs_ids_chunk, total=len(descs_ids_chunk), desc="processing descs ids"):
        heads_aliases.setdefault(d_id, []).extend(_shared_aliases[d_id])
        for _, _, t in _shared_triples[d_id]:
            tails_aliases.setdefault(d_id, []).extend(_shared_aliases[t])

        for als in _shared_aliases[d_id]:
            escaped = re.escape(als)
            flexible = escaped.replace(r"\ ", r"\s*")
            pattern = rf"\b{flexible}\b"
            alias_pattern_map[als] = re.compile(pattern, re.IGNORECASE)

    return  heads_aliases, tails_aliases, alias_pattern_map



def get_tailsHeadsAliases_aliasesPatternMap():

    print("reading dicts")
    full_descs_dict = read_cached_array(RESULT_FILES["descriptions"])
    aliases = read_cached_array(RESULT_FILES["aliases"])
    triples = read_cached_array(RESULT_FILES["triples"])

    full_descs_ids = list(full_descs_dict.keys())
    chunks = split_list(full_descs_ids, NUM_WORKERS)
    print("splitting into processes")
    with Pool(processes=NUM_WORKERS, initializer=init_globals, initargs=(aliases, triples)) as pool:
        results = pool.map(worker, chunks)

    tails_aliases_all  = {}
    heads_aliases_all = {}
    aliases_patterns_all = {}
    for  h_a, t_a, a_p_m in results:
        heads_aliases_all.update(h_a)
        tails_aliases_all.update(t_a)
        aliases_patterns_all.update(a_p_m)
    print(f"processesed dictionaries")
    return heads_aliases_all, tails_aliases_all, aliases_patterns_all

if __name__ == "__main__":
    heads_aliases_all, tails_aliases_all, aliases_patterns_all = get_tailsHeadsAliases_aliasesPatternMap()
    print(f"heads aliases: {heads_aliases_all}")
    print(f"tails_aliases_all: {tails_aliases_all}")
    print(f"aliases_patterns_all: {aliases_patterns_all}")
    cache_array(heads_aliases_all, "./data/temp/heads_aliases_all.pkl")
    cache_array(tails_aliases_all, "./data/temp/tails_aliases_all.pkl")
    cache_array(aliases_patterns_all, "./data/temp/aliases_patterns_all.pkl")