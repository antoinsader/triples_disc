# Dataset download:

The project is using **wikidata5m** dataset which their files can be downloaded by:

```bash
curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar
```

---


## Files description: 

#### Data.py: 
- all the paths to the pickle files I am using during the application

#### 1_Prep.py: 
- Creating dictionaries (description, aliases, triples, relations) from the raw files and  save them in pickle files.
- Create min (10,100,1k,100k,1m) depending on your choice of k for the dictionaries with function create_min(k)


#### 2_normalize.py: 
- Basic normalization for the descriptions dictionary or aliases dictionary using parallel processing
- NFKC normalization
- Removing multiple spaces and replacing strange chars with their equivalents
- Did not apply other normalization techniques because I want to save the semantics meaning of the descriptions to use it after in BERT

#### 3_transe2.py
- Applying (Translating Embeddings for Modeling Multi-relational Data) transE for building  knowledge graph from the triples
- Pytorch
- Paper: https://papers.nips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

#### 4_prep_rels:
- Preparing BERT embedings for all the relationships
- using BERT and pool averaging last two layers of each  relationship emb
- I am not sure if I should use here other model for better semantic representation, the paper likes BERT


#### 5_prep_ground.py:
- This file is not complete and I am still trying to figure out in which way is the best to get the ground truth or gold binary labels from the triples I have
- The problem is that from the training I am getting triples containing of ((subject_start_idx, subject_end_idx), relationship, (object_start_idx, object_end_idx)) -ASSUMING FORWARD EXTRACTION-, now I have to have gold tagging binary representing the ground truth which I don't
- I have in the ground truth triples (subject_entity_id, relationship_id, object_entity_id) and for each id a list of aliases for the entity or relationship
- The problem is that the subject_entity_id representing a description and list of aliases for this description but the aliases texts might not be visible in the text and this is the problem


#### brask_model.py:
- Here is the model we are developing but still missing the error calculation

### Project start:

To start the project: 
- DOWNLOAD THE RAW FILES in data/raw/
- Create python venv
- Pip install numpy  tqdm torch scikit-learn  
- Start executing starting from 1_prep.py

### BRASK paper: 
 Bidirectional relation-guided attention network.pdf



### GPU Code: 
 1- vm/min/1_prep.py:
        
        -  prepare_main(percentage):
                specify the percentage of descriptions to be used, full descriptions are 5M
                let's say you choose 0.0002, then you will have around 1k descriptions

                Function steps:
                    - random.sample depending on percentage for descriptions 
                    - Get triples of the selected descriptions 
                    - Save the tails of those triples in my_tails 
                    - Add the tails keys into list of descriptions keys (that's why the descriptions length would not be exactly as the percentage)
                    - get relations from triples 
                    - get aliases for selected description keys
                    - get descriptions texts for all 
                    - Save descriptions_unormalized, aliases, relations, triples dicts in pkl files
        - normalize_descriptions(): 
            Function steps:
                - read unormalized dictionaries
                - split the descriptions into batches depending on NUM_WORKERS (specified based on cpu cores )
                - use parallelization on num_workers to execute normalize_desc_batch
                - normalize desc_batch do the following for each description:
                        - replace special chars (like à into a)
                        - keep only english letters (remove arabic chineese,.. ) letters
                        - remove multiple consequent spaces
                - collect results from workers and save the result of normalized descriptions in pkl file

        - normalize aliases():
            - Same as normalize_descriptions but here with making aliases lower case 

