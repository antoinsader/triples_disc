
  

# Triples extraction using BRASK

  

Train neural network to extract triples (head, relation, tail) from wikidata5m dataset using BRASK

  

## Download dataset:


    curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz
    
    curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
    
    curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.tar

  

## Create env:

    python -m venv myenv

  

## Install requirements:

    pip install numpy tqdm pandas
    
    pip install torch transformers
    
    pip install scikit-learn nltk datasets psutil
    
    pip install joblib

  

## Code:


**1_prep.py:**

at main(), you have:

 - **prepare_main_cpu_1(percentage) and prepare_main_cpu_2()** if you're
   using cpu
   
 - **prepare_main(percentage)**: if you're using gpu, it will convert the raw
   files into dictionaries and save them in pickle files to use them.
   (descriptions, aliases, relationships, triples)

   
 - **save_strange_chars_dict(), save_stop_words()**: creates helper files to use after.
 - **normalize_descriptions()**:  normalize the description dictionary (keep only english letters, replace strange chars, remove multiple spaces)
 - **prep_relations_main()**: create embeddings space for the relationships using BERT
 -  **normalize_aliases()**
 - **create_alias_patterns_map()**: create for each alias a regex pattern that we will use after when creating silver spans for searching
 - **do_transe_triples()**: prepare rel2id, ent2id and create triples based on those ids, also neg_triples, we will use in the transe creation
 - **create_heads_tails_aliases_batch():** create 2 dictionaries (tails_aliases, heads_aliases) that would include the aliases of heads and tails for each description, would be used after to extract silver spans


**2_transE_2.py:**
Using GPU parallelization, create transe embeddings for the relationships we have. 
TRANSE_EMB_DIM = 100 by default which will create 100d for each relationship, you can modify it.
To execute this file:

    torchrun --nproc_per_node=<NUM GPUs> 2_transE_2.py

If you want to use CPU instead, you can execute 2_transE.py

**3_silver_spans_1.py:**
Create the ground truth for our model.
Try to find triples aliases inside the descriptions texts.

**4_prep_model.py:**

