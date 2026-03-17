
  

# Triples extraction using BRASK
Train neural network to extract triples (head, relation, tail) from wikidata5m dataset using [BRASK](docs/Bidirectional%20relation-guided%20attention%20network.pdf) algorithm.
The project report is available at [REPORT](docs/report.pdf)


## Training graph:

![Training_graph](docs/graph.png)

## Download dataset:
```
    curl -L -o wikidata5m_text.txt.gz "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1" && gunzip wikidata5m_text.txt.gz    
    curl -L -o wikidata5m_alias.tar.gz "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1" && gunzip wikidata5m_alias.tar.gz && tar -xvf wikidata5m_alias.tar
    curl -L -o wikidata5m_transductive.tar.gz "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1" && gunzip wikidata5m_transductive.tar.gz && tar -xvf wikidata5m_transductive.
```

## Codebase:
The first version of the code exists inside **/archive**, now I am refactoring the steps in the root file. The new code for the training pipeline is not totally ready, the old code of archive worked but only using big GPU RAM and using pytorch ddp, that's the main reason of the refactoring process.

## Training steps:

### **1- Minimize**:

#### Objective and run:

Wikidata5m is a huge dataset, so if you are running on low RAM, you need to minimize it so you can test the algorithm, that's why you can run minimize.py by: 

```
python minimize.py
```

The script will ask you the minimize factor (between 0 and 1), then it will calculate approximate numbers of the size of descriptions after minimization and you can proceed by typing y.
After it runs 4 streaming passes (triples, descriptions, relations, aliases) and save the minimized files.

After when you execute other steps, the terminal will ask you if you want to perform the operations on the minimized version or the full dataset.

#### Output:
Parsed full dictionaries files if not cached for desecriptions, triples, relations, aliases
Minimized files for descriptions, triples, relations, aliases



### **2- Normalize**:

#### Objective and run:

Normalization cleans descriptions and aliases so they are ready for downstream NLP tasks. Run it with:

```
python normalize.py
```

The script will ask whether to normalize the minimized dataset or the full dataset (default: minimized). If the minimized files are missing, it will prompt you to run `minimize.py` first.

It then warns you which files will be overwritten and asks for confirmation before proceeding.

#### Details:
Normalization steps applied to each value in **descriptions** dictionary:
- Replace special characters with their equivalent (e.g. `á→a`, `é→e`, `&→and`)
- Remove words that contain non-English characters
- Apply Unicode NFKC normalization
- Collapse multiple spaces
- **Does NOT lowercase**
- **Does NOT remove stop words**


Normalization steps applied to each list of **aliases**:
- Replace special characters with their equivalent (e.g. `á→a`, `é→e`, `&→and`)
- Skip aliases that are English stop words
- Apply Unicode NFKC normalization
- Collapse multiple spaces
- Lowercase the result
- Deduplicate aliases per entity


#### Output:
Overwrite aliases and descriptions files with normalized content.


### **3- Embed relations**:

#### Objective and run:

This step produces one 768-dimensional BERT embedding per relation, averaged across all its text aliases, in order to use those embeddings in perform_transe file where we are performing transE algorithm on the relations. Run it with:

```
python embed_relations.py
```

The script will ask whether to embed the minimized or the full dataset (default: minimized). If the source `relations.pkl` file is missing, it will prompt you to run the appropriate prior step first.
It then warns you which file will be overwritten and asks for confirmation before proceeding.


#### Details:

**Embedding approach:**
- Loads `bert-base-cased` at runtime (not at import time) and moves it to GPU if available
- Flattens all aliases across all relations into a single list and processes them in batches (`BATCH_SIZE = 32` on CPU, `8192` on CUDA)
- Each alias embedding is the attention-mask mean-pool of the average of BERT's last two hidden layers
- Per-alias embeddings are accumulated per relation with `scatter_add_` and divided by alias count → one averaged embedding per relation
- Uses `torch.autocast` for mixed-precision (float16 on CUDA, bfloat16 on CPU)
- Relations with no aliases fall back to using the relation ID itself as input text

#### Output:

a compressed `.npz` file of shape `(n_relations, 768)` saved to `relation_embeddings.npz`



### **4- Perform TransE algorithm**

#### Objective and run:
TransE knowledge graph embedding model (Bordes et al. 2013).
Learns entity and relation embeding by minimizing the scoring function of ```|| h + r - t ||``` to learn the relations of (head, relation, tail).

```
python perform_transe.py
```

#### Details:

- The training is detecting wheter LOCAL_RANK environment variable exists, which will be when using torchrun, to use **Torch Distributed Data Parallel - torch DDP**, and falls back to single-gpu or cpu when LOCAL_RANK does not exist

- **Create the dataset**: We are creating **TransEDataset**, which will generate ent2idx, rel2idx (mapping from entities and relations to their idxs), n_ents, n_rels, triples, neg_triples (corrupted triples where head or tail is corrupted). the __getitem__ of the dataset will return tuple[torch tensor from triples, torch tensor from negative triples]. The dataset is inheriting from torch.utils.Dataset
- Build the ```TransEModel```, with Adam as an ```optimizer``` and ```CosineAnnealingLR``` as scheduler. 
- Training the model, where forward pass is computing L1 distances for positive and negative triple batches. and using with loss as  the mean of (MARGIN + pos_distance - neg_distance)


#### Output
Save the model results in the file ```transe_rel_embs.npz``` with shape (n_relations, TRANSE_EMB_DIM)


### **5- Prepare silver spans**:

#### Objective and run:

The objective of this step is to extract silver spans to be used in the training after.
The script can be used also to filter descriptions to only those having at least one value in the silver spans.

### Details:

The scripts at first prompts the user if to use minimized dataset or the normal one.
After the script is finished from extracting and saving silver spans, it will prompt if to filter the descriptions

Silver spans are 4 types each is a torch tensor with shape (DESCRIPTIONS, MAX_LENGTH):
- Head start silver spans: For each description there is an array containing as a value 1 if the token position is a **start of a head**, 0 otherwise
- Head end silver spans: For each description there is an array containing as a value 1 if the token position is a **end of a head**, 0 otherwise
- Tail start silver spans: For each description there is an array containing as a value 1 if the token position is a **start of a tail**, 0 otherwise
- Tail end silver spans: For each description there is an array containing as a value 1 if the token position is a **end of a tail**, 0 otherwise

Also in the results we have:
- sentences_tokens: the tokens of each description
- desc_ids: to map the indexes

We are preparing the results by:
- Create aliases regex pattern map: For each alias, a regex compiled pattern. To be used after for finding aliases inside descriptions texts.
- Create descriptions_heads_aliases, descriptions_tails_aliases: where for each description, we extract the aliases of their heads and aliases of their tails
- **Parallel processing** for description chunks to extract the spans using the previously prepared maps.


#### Output:

- The result is saved in ```silver_spans.pkl``` file containing a dict with 6 keys

```
{
    "head_start": tensor(N,L), 
    "head_end": tensor(N,L), 
    "tail_start": tensor(N,L), 
    "tail_end": tensor(N,L), 
    "sentences_tokens": list[list[str]], 
    "desc_ids": list[str]
}
```
    Where N is the number of descriptions, L is the maximum_length for the tokenizer

- Overwrite desceriptions file if approved by user.



