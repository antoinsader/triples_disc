from Data import PKLS_FILES
from utils.utils import read_cached_array

enT_embs = read_cached_array(PKLS_FILES["transE_entity_embeddings"] )
rel_embs = read_cached_array(PKLS_FILES["transE_relation_embeddings"] )
triples = read_cached_array(PKLS_FILES["triples"][10_000])
print(rel_embs.shape)

rels = set()
ents = set()
for tr in triples.values():
    for (h, rel, t) in tr:
        rels.add(rel)
        ents.add(h)
        ents.add(t)
print(len(list(rels)))

print(len(list(ents)))
