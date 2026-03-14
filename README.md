
  

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

## File overview:

### 1_prep.py
Preparing files for training

### 2_transe.py
Perform TransE algorithm on the data

### 3_silver_spans_1.py
Create silver spans for the training

### 4_prep_model.py
Preparing the model for training

### 5_model.py
Running the algorithm and train the model

### 6_eval.py
For evaluation

