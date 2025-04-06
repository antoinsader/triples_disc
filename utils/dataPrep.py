
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def load_descriptions_dict_from_text(fp):
    di = {}
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating descriptions"):
            try:
                entity_id,  description, *rest = line.strip().split("\t")
                di[entity_id] =  description
            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di


def load_triples(file_path):
    triples_lookup = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating knowledge graph (triples)"):
            head, relation, tail = line.strip().split('\t')
            triples_lookup[head].append((head, relation, tail))
            
    return triples_lookup


def load_relations(file_path):
    relations= {}
    with open(file_path, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating relationships dict"):
            line_parts = line.strip().split("\t")
            relations[line_parts[0]] = line_parts[1:]
    return relations


def load_aliases_dict_from_text(fp):
    di = {}
    rev_dict = defaultdict(list)
    with open(fp, 'r', encoding="utf-8") as f:
        l_ns = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=l_ns, desc=f"Creating aliases"):
            try:
                split_line = line.strip().split("\t")
                entity_id = str(split_line[0])
                entity_name = split_line[1]
                aliases = split_line[2:]
                di[entity_name] = entity_id
                rev_dict[entity_id].append(entity_name)
                for al in aliases:
                    di[al] = entity_id
                    rev_dict[entity_id].append(al)

            except ValueError as e:
                print(f"The line has not enough arguments: {line.strip()}")
                break
    return di,rev_dict

