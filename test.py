from multiprocessing import Pool

from numpy import c_
from regex import T
import torch

from utils.chunking import chunk_dict


descriptions = {
    "Q1": "I am John Doe  from Honolulu, and I am living in Italy", 
    "Q4": "Honolulu is a city in Utopia",
    "Q2": "Rome is the capital of Italy",
    "Q3": "Milan is a city in Italy, and Milan likes Rome",
    "Q5": "Italy is a good country "
}

aliases = {
    "Q1": ["John Doe", "JD"],
    "Q4": ["Honolulu", "Honhon city"],
    "Q2": ["Rome", "Roma"],
    "Q3": ["Milan", "Milano"],
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


def test_silver_spans():
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
        

    mask = (
        silver_spans_head_start.any(dim=1) |
        silver_spans_head_end.any(dim=1) |
        silver_spans_tail_start.any(dim=1) |
        silver_spans_tail_end.any(dim=1)
    )
    new_descriptions = {
        k: descriptions[k]
        for k, keep in zip(desc_ids, mask)
        if keep.item()
    }

    silver_spans_head_start = silver_spans_head_start[mask]
    print(f"new descriptions: {new_descriptions}")
    print(f"new silver spans head start: {silver_spans_head_start}")
    print(f"we cleared  {len(descriptions) - len(new_descriptions)}/{len(descriptions)} descriptions, now we have {len(new_descriptions)} descriptions")


def test_extract_sk():
    # 2 SENTENCES, 5 TOKENS EACH. EMBEDDING SIZE 4
    hidden_dim = 4
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 7.0, 1.0],
            [7.0, 1.0, 2.0, 1.0],
             [1.0, 3.0, 1.0, 1.0], 
             [2.0, 1.0, 3.0, 1.0],
        ],
        [
            [5.0, 4.0, 3.0, 4.0],
            [2.0, 2.0, 5.0, 7.0],
            [1.0, 2.0, 5.0, 4.0], [6.0, 3.0, 3.0, 7.0], [8.0, 1.0, 2.0, 1.0],
    ],]
    )
    print(f"x shape: {x.shape} ")

    head_start = torch.tensor([
        [[0.6], [0.4], [0.7], [0.9], [0.1]],
        [[0.8], [0.3], [0.1], [0.8], [0.4]],
    ])


    head_end = torch.tensor([
        [[0.5], [0.3], [0.6], [0.1], [0.2]],
        [[0.1], [0.4], [0.2], [0.3], [0.9]],
    ])


    correct_spans = [
        [(0, 0), (2,2), ],
        [(3,4)]
    ]



    threshold_head_start = 0.5
    threshold_head_end = 0.5
    max_span_len = 3

    B = x.size(0)

    forward_s_k = []
    all_spans =[]
    for b in range(B):
        x_emb = x[b] #(5, 128)
        start_idx  = (head_start[b].squeeze(-1) >= threshold_head_start).nonzero(as_tuple=False).squeeze(-1)
        start_idx = torch.sort(start_idx).values
        end_idx  = (head_end[b].squeeze(-1) >= threshold_head_end).nonzero(as_tuple=False).squeeze(-1)
        consumed_ends = set()
        spans = []
        for s in start_idx:
            # print("********")
            # print(f"    start idx: {s}")
            end_mask = ( (end_idx >= s) & (end_idx < s+max_span_len))
            valid_ends = end_idx[end_mask]
            # print(f"    valid_ends: {valid_ends}")
            valid_ends = [e.item() for e in valid_ends if e.item() not in consumed_ends]
            # print(f"    valid_ends: {valid_ends}")
            if len(valid_ends) == 0:
                continue
            e = min(valid_ends)
            spans.append((s.item(), e))
            consumed_ends.add(e)
        all_spans.append(spans)
        s_k_list = []
        for (s, e)  in spans:
            print(f" start emb: {x_emb[s]}, end emb: {x_emb[e]}")
            s_k = (x_emb[s] + x_emb[e]) / 2
            s_k_list.append(s_k)
        if s_k_list:
            s_k_list = torch.stack(s_k_list, dim=0)
        forward_s_k.append(s_k_list)
    assert all_spans == correct_spans, f"extracted spans {all_spans} do not match the correct spans {correct_spans}"

    print(f"forward_s_k: {forward_s_k}")
    return forward_s_k

def test_padding_sk(forward_s_k):
    #forward s_k is a list where each element is tensor (num_subjects, H)
    H = forward_s_k[0].shape[1]
    max_num_subjects = max([s_k.shape[0] for s_k in forward_s_k])
    padded_sk = []
    mask = []
    for s in forward_s_k:
        K = s.shape[0]
        if K < max_num_subjects:
            pad = torch.zeros(max_num_subjects - K, H)
            s_padded  = torch.cat([s, pad], dim=0)
            m = torch.cat([torch.ones(K), torch.zeros(max_num_subjects - K)], dim=0)
        else:
            s_padded = s
            m=torch.ones(K)

        padded_sk.append(s_padded)
        mask.append(m)
    forward_s_k = torch.stack(padded_sk, dim=0) #(B, max_num_subjects, H)
    forward_s_k_mask = torch.stack(mask, dim=0) #(B, max_num_subjects)
    print(f"forward sk shape {forward_s_k.shape} : {forward_s_k}")
    print(f"forward sk mask shape {forward_s_k_mask.shape} : {forward_s_k_mask}")
    return forward_s_k, forward_s_k_mask

def test_fuse_extractor(forward_s_k, forward_s_k_mask):
    B, max_num_subjects, H = forward_s_k.shape
    l = torch.nn.Linear(H, H)
    new_sk = l(forward_s_k) #(B, max_num_subjects, H)
    new_sk = forward_s_k_mask.unsqueeze(-1) * new_sk #(B, max_num_subjects, H) with padded subjects zeroed out
    print(f"new sk shape {new_sk.shape} : {new_sk}")


if __name__ == "__main__":
    # test_parallel()
    # test_silver_spans()
    forward_s_k = test_extract_sk()
    s_k, s_k_mask =  test_padding_sk(forward_s_k)
    test_fuse_extractor(s_k, s_k_mask)