from multiprocessing import Pool

from numpy import c_
import torch

from utils.chunking import chunk_dict


descriptions = {
    "Q1": "I am John Doe  from Honolulu, and I am living in Italy", 
    "Q2": "Rome is the capital of Italy",
    "Q3": "Milan is a city in Italy, and Milan likes Rome",
    "Q4": "Honolulu is a city in Utopia",
    "Q5": "Italy is a good country "
}

aliases = {
    "Q1": ["John Doe", "JD"],
    "Q2": ["Rome", "Roma"],
    "Q3": ["Milan", "Milano"],
    "Q4": ["Honolulu", "Honhon city"],
}

relations = {
    "R1": ["is from", "born in", "comes from"],
    "R2": ["is living in", "resides in", "lives in"],
    "R3": ["is the capital of", "capital of"],
    "R4": ["is a city in", "city in"],
    "R5": ["likes", "loves", "admires"]
}

triples = {
    "Q1": [("Q1", "R1", "Q4"), ("Q1", "R2", "Q5")],
    "Q2": [("Q2", "R3", "Q5")],
    "Q3": [("Q3", "R4", "Q5"), ("Q3", "R5", "Q2")]
}


def test_chunk_dict():
    print(descriptions.items())
    print(f"chunking into 2 chunks...")
    c = chunk_dict(descriptions, chunks_n=2)
    print(f"chunks: {c}")
    c_flat = dict([item for chunk in c for item in chunk.items()])
    print(f"c_flat: {c_flat}")
    assert c_flat == descriptions, "chunk_dict did not yield the original dictionary when recombined"
# test_chunk_dict()


def mock_worker(desc_chunk:dict, max_length: int):
    results = {}
    for k, v in desc_chunk.items():
        tokens = v.split(" ")
        results[k] = " ".join(tokens[:max_length])
    return results, list(desc_chunk.keys()), [1,2,3,4,]

def test_parallel():
    max_length = 2
    chunks = chunk_dict(descriptions, chunks_n=2)
    print(f"descriptions: {descriptions}")
    with Pool(
        processes=2,
    )as pool:
        args = [(chunk, max_length) for chunk in chunks]
        results = pool.starmap(mock_worker, args)

    res_chunks = {}
    res_keys= []
    res_nums= []
    for (res_chunk, keys, nums) in results:
        res_chunks.update(res_chunk)
        res_keys.extend(keys)
        res_nums.extend(nums)

    print(f"res_chunks: {res_chunks}")
    print(f"res_keys: {res_keys}")
    print(f"res_nums: {res_nums}")


if __name__ == "__main__":
    # test_parallel()
    from transformers import BertTokenizerFast
    from utils.chunking import chunk_dict
    from prepare_silver_spans import create_aliases_patterns_map, init_worker, process_descriptions_chunk, create_description_heads_tails_map_aliases
    max_length = 3
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    aliases_pattern_map = create_aliases_patterns_map(aliases)
    descriptions_heads_aliases, descriptions_tails_aliases = create_description_heads_tails_map_aliases(descriptions, triples, aliases)
    print(f"descriptions_heads_aliases: {descriptions_heads_aliases}")
    print(f"descriptions_tails_aliases: {descriptions_tails_aliases}")
    desc_chunks = chunk_dict(descriptions, chunks_n=2)
    
    
    with Pool(processes=2, initializer=init_worker, initargs=(tokenizer, descriptions_heads_aliases, descriptions_tails_aliases, aliases_pattern_map)) as pool:
        args = [(chunk, 10) for chunk in desc_chunks]
        results = pool.starmap(process_descriptions_chunk, args)


    silver_spans_head_start_ar = []
    silver_spans_head_end_ar = []
    silver_spans_tail_start_ar = []
    silver_spans_tail_end_ar = []
    sentences_tokens = []
    desc_ids = []
    for batch in results:
        b_ss_h_s, b_ss_h_e, b_ss_t_s, b_ss_t_e, b_tokens, b_desc_ids = batch
        silver_spans_head_start_ar.extend(b_ss_h_s)
        silver_spans_head_end_ar.extend(b_ss_h_e)
        silver_spans_tail_start_ar.extend(b_ss_t_s)
        silver_spans_tail_end_ar.extend(b_ss_t_e)
        sentences_tokens.extend(b_tokens)
        desc_ids.extend(b_desc_ids)
    silver_spans_head_start = torch.stack(silver_spans_head_start_ar, dim=0)
    silver_spans_head_end = torch.stack(silver_spans_head_end_ar, dim=0)
    silver_spans_tail_start = torch.stack(silver_spans_tail_start_ar, dim=0)
    silver_spans_tail_end = torch.stack(silver_spans_tail_end_ar, dim=0)


    for d_idx, d_id in enumerate(desc_ids):
        print(f"description: {descriptions[d_id]}" )

        print("description triples: ")
        if d_id in triples:
            for t_idx, (h,_,t) in enumerate(triples[d_id]):
                print(f"\t triple number: {t_idx + 1}")
                print(f"\t\t head aliases: {aliases[h] if h in aliases else 'None'}")
                print(f"\t\t tail aliases: {aliases[t] if t in aliases else 'None'}")
        print("description tokens: ", sentences_tokens[d_idx])
        print(f"head starts: {silver_spans_head_start_ar[d_idx]}")
        print(f"head ends: {silver_spans_head_end_ar[d_idx]}")
        print(f"tail starts: {silver_spans_tail_start_ar[d_idx]}")
        print(f"tail ends: {silver_spans_tail_end_ar[d_idx]}")
        


