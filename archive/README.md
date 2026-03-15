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



