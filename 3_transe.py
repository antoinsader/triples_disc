
from sympy import numer
from tqdm import tqdm 
import os 
import sys
import pickle
import numpy as np
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from Data import PKLS_FILES, TRANSE_FOLDER
from utils.utils import read_cached_array, cache_array

def train_transE(triples,modelSaveFolder ,training_fraction=0.8, batch_size=1024, device="cuda" ):

    tf = TriplesFactory.from_labeled_triples(triples)
    

    train_tf, test_tf = tf.split([training_fraction, 1 - training_fraction])
    print(f"Train size: {train_tf.num_triples}, Test size: {test_tf.num_triples}")
    result = pipeline(
        model='TransE',
        training=train_tf,
        testing=test_tf,
        model_kwargs={
            'embedding_dim': 100,
        },
        optimizer='Adam',
        optimizer_kwargs=dict(lr=0.001),
        
        training_kwargs={
            'num_epochs': 100,
            'batch_size': batch_size,
        },

        negative_sampler='basic', 

        random_seed=42,        
        device=device
    )
    print("Training completed!")
    print(f"Best model MRR on test set: {result.metric_results.get_metric('MRR')}")
    result.save_to_directory(modelSaveFolder)
    best_model = result.model
    best_model.save_state(path=f"{modelSaveFolder}/model_state.pt")
    print(f"Saved TransE model to {modelSaveFolder} and file {modelSaveFolder}/model_state.pt")
    
    print("Mean Reciprocal Rank (MRR):", result.get_metric('mean_reciprocal_rank'))
    entity_embeddings = result.model.entity_representations[0]()
    relation_embeddings = result.model.relation_representations[0]()

    print("Entity embeddings shape:", entity_embeddings.shape)
    print("Relation embeddings shape:", relation_embeddings.shape)
    return entity_embeddings, relation_embeddings
if __name__ == "__main__":
    k = 1_000 
    
    
    triples = read_cached_array(PKLS_FILES["triples"][k])
    triples_ar = []
    for tr_head, trpls in triples.items():
        triples_ar.extend(trpls)
    triples_ar = np.array(triples_ar)    

    
    transe_model_folder = TRANSE_FOLDER
    entity_embeddings, relation_embeddings = train_transE(triples_ar, transe_model_folder, batch_size=128, device="cpu")
    
    cache_array(relation_embeddings,  PKLS_FILES["transE_relation_embeddings"])
    cache_array(entity_embeddings , PKLS_FILES["transE_entity_embeddings"] )
    
    # result = load_pipeline_result(transe_model_folder)
    # mrr = result.metric_results.get_metric('MRR')
    # print("MRR on test set:", mrr)
    # entity_embeddings = result.model.entity_representations[0]()
    # relation_embeddings = result.model.relation_representations[0]()
    
    # print("Entity embeddings shape:", entity_embeddings.shape)
    # print("Relation embeddings shape:", relation_embeddings.shape)