## Triples discovery old code

### Files overview

#### Prep.py

Prepare datasets in pickle files and doing minimization 

**prepare_main(percentage) function**:

Each of the files descriptions, aliases, relations, triples are converted to dict and saved in pickle file to use after, the dicts are formatted in the following way:
- descriptions: dict where entity_id is the dictionary key and description (str text describing the entity) is the value. 
- aliases: dict where key is entity_id and value is a list of aliases representing the entity
- relations: dict have relation_id as a key and list of strings representing the relation texts
- triples: dict having as a key the head and as value (head, relation, tail) where head and tail are entity_id from aliases and relation is a relation_id from relations


After reading the files (from pickle files) or convert them into dicts, we start the minimizing process:

- we extract entity_ids from descriptions  where entity_id exists in triples heads (the key of triples dictionary), creating minimized_description_dict and minimized_entity_ids
- calculate the minimization number, and create a minimized_entity_ids (set) having values chosen randomly from description_ids that exists in triples
- we create another triples dict having only heads from the minimized entity_ids we have (minimized_triples)
- we extract relation_id contained in minimized_triples and create new relations dictionary (minimized_relations)
- we extract from triples the tail_id to include them with their descriptions in the minimized_description_dict



**normalize_descriptions() function**:
Reads the descriptions dict and perform normalization which performing for each description value:
- Replace special chars (e.g replace ŏǒôốộồổỗöȫȯȱọőȍòỏơớợờởỡȏōṓṑǫǭõṍṏȭǿøɔ to o)
- Keep only enlglish looking words using r"^[A-Za-z0-9.,!?;:'\"()\-]+$"
-  Unicode NFKC normalization 
- Collapse multiple spaces, strip 
- **Does NOT lowercase**
- **Does NOT remove stop words**


**normalize_aliases() function**:
Reads the aliases dict and for each alias in the aliases list do:

- Replace special chars
- Keep only english-looking words
- NFKC normalization
- Collapse multiple spaces
- Lowercase
- Remove stop words
- Remove duplicates from list of aliases of one entity_id

**do_transe_triples() function**:
Prepare data for transe algorithm which will be performed in `2_transe.py`

Perpared data will have:
- triples: list of tuples (head_entity_id, relation_id, tail_entity_id)
- neg_triples: doing the same size of triples but messing up the head_id and tail_id
- n_rels: number of relations
- n_ents: number of entities

