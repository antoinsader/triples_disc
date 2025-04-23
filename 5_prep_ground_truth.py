from Data import get_min_descriptionsNorm_triples_relations, LOGGER_FILES, PKLS_FILES
from utils.utils import read_cached_array, cache_array, get_logger
from transformers import BertTokenizerFast
from flashtext import KeywordProcessor
from transformers import BertModel
import torch


DESCRIPTION_MAX_LENGTH = 128



def get_entity_start_end_idxs(entity_description_tokens, entity_to_find_aliases, tokenizer):
    """
        I try to find using flashtext the entity to find in the description, if found return the idxs 
    """ 
    spans = []
    alias_token_lists = []
    for alias in entity_to_find_aliases:
        toks = tokenizer.tokenize(alias)
        alias_token_lists.append((toks, alias))
    
    L = len(entity_description_tokens)
    for toks, alias in alias_token_lists:
        m = len(toks)
        if m == 0 or m > L:
            continue
        for i in range(0, L - m + 1):
            if entity_description_tokens[i : i + m] == toks:
                spans.append((i, i + m - 1, alias))
    return spans

def create_silver_spans(descs, triples, relations, aliases, buid_logger,golden_triples_file, silver_spans_file):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


    sentences_ids = list(descs.keys())
    sentences = list(descs.values())
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=DESCRIPTION_MAX_LENGTH)
    sentences_tokens = [tokenizer.convert_ids_to_tokens(sen) for sen in encoded["input_ids"]]
    seq_len = len(sentences_tokens[0])
    batch_size = len(sentences)

    silver_span_sub_s = torch.zeros(batch_size, seq_len)
    silver_span_sub_e = torch.zeros(batch_size, seq_len)
    silver_span_obj_s = torch.zeros(batch_size, seq_len)
    silver_span_obj_e = torch.zeros(batch_size, seq_len)

    golden_triples = {}
    buid_logger.info(f"Building golden triples for {len(descs)}: \n We will log every description sentence and then the aliases for the head, tail of the description triples and log the extracted golden triples.. By triples I mean double of (head, tail)")

    for sentence_idx, (sen_tokens, sen_id) in enumerate(zip(sentences_tokens, sentences_ids)):
        buid_logger.info(f"\n\nSentence : {descs[sen_id]}")
        buid_logger.info(f"Sentence tokens: {sen_tokens}")
        #I should allow duplications because the network might predict duplicated 
        sen_trps = []
        trpls = triples[sen_id]
        triples_found_logs = []
        for h, r, t in trpls:
            buid_logger.info(f"\n\t next triple: ")
            h_aliases = aliases[h]
            buid_logger.info(f"\thead aliases: {h_aliases}")
            trp_h = get_entity_start_end_idxs(sen_tokens, h_aliases, tokenizer)
            if len(trp_h) == 0:
                buid_logger.info(f"\t \t no head aliases matched with the sentence")
                continue
            buid_logger.info(f"\t \thead aliases matched {trp_h}")

            t_aliases = aliases[t]
            buid_logger.info(f"\ttail aliases: {t_aliases}")
            trp_t = get_entity_start_end_idxs(sen_tokens, t_aliases, tokenizer)

            if len(trp_t) == 0:
                buid_logger.info(f"\t \tno tail aliases matched with the sentence")
                continue 
            buid_logger.info(f"\t \ttail aliases matched {trp_t}")

            relation_aliases = relations[r]
            for matched_head in trp_h:
                for matched_tail in trp_t:
                    
                    silver_span_sub_s[sentence_idx, matched_head[0]] = 1.0 
                    silver_span_sub_e[sentence_idx, matched_head[1]] = 1.0 
                    silver_span_obj_s[sentence_idx, matched_tail[0]] = 1.0 
                    silver_span_obj_e[sentence_idx, matched_tail[1]] = 1.0 

                    sen_trps.append(( (matched_head[0], matched_head[1]) ,  (matched_tail[0], matched_tail[1])  ))
                    triples_found_logs.append(f"\n\n\t******** TRIPLE FOUND:  (HEAD, TAIL): ({matched_head[2]}, {matched_tail[2]}), (head_idxs, tail_idxs): ({(matched_head[0], matched_head[1]) ,  (matched_tail[0], matched_tail[1])}) \n\tRelation aliases: {relation_aliases}"   )

        for l in triples_found_logs:
            buid_logger.info(l)

        golden_triples[sen_id] = {"sentence_tokens": sen_tokens, "triples": sen_trps}

    silver_spans = {
        "head_start":silver_span_sub_s ,
        "head_end": silver_span_sub_e,
        "tail_start": silver_span_obj_s,
        "tail_end":silver_span_obj_e ,
    }


    cache_array(silver_spans, silver_spans_file)
    cache_array(golden_triples, golden_triples_file)

        


if __name__ == "__main__":
        
    k = 1000
    descs, triples, relations, aliases = get_min_descriptionsNorm_triples_relations(k)
    golden_triples_file = PKLS_FILES["golden_triples"][k]
    silver_spans_file = PKLS_FILES["silver_spans"][k]
    buid_logger = get_logger("build_golden_triples", LOGGER_FILES["build_golden_triples"])
    
    create_silver_spans(descs,triples,relations, aliases , buid_logger,golden_triples_file, silver_spans_file )