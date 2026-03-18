
import torch
import os
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, DistributedSampler


from perform_transe import NUM_WORKERS
from utils.files import read_cached_array, read_tensor
from utils.settings import settings
from utils.pre_processed_data import check_preprocessed_files, data_loader, check_minimized_files



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




def check_training_files(use_minimized=False):
    if use_minimized and not check_minimized_files():
        return False

    if not use_minimized and not check_preprocessed_files():
        return False

    files_paths = settings.MINIMIZED_FILES if use_minimized else settings.PREPROCESSED_FILES
    important_files = [files_paths.SILVER_SPANS, files_paths.TRANSE_MODEL_RESULTS, files_paths.DESCRIPTION_EMBEDDINGS_ALL, files_paths.DESCRIPTION_EMBEDDINGS_MEAN, files_paths.DESCRIPTION_EMBEDDINGS_IDS]
    missing = [p for p in important_files if not os.path.isfile(p)]
    print(f"missing: {missing}")
    if missing:
        print("The following important files are missing. Run the appropriate scripts to generate them before training.")
        for p in missing:
            print(f"  Missing: {p}")
        return False

    return True


class BraskDataset(Dataset):
    """Dataset for training the Brask model. 
    Each item is a tuple of (description_embedding: Tensor[L,D], description_mean_embeddings: Tensor[D], silver_span_head_start: Tensor[L], silver_span_head_end: Tensor[L], silver_span_tail_start: Tensor[L], silver_span_tail_end: Tensor[L])"""


    def __init__(self, 
                 silver_spans_head_start: torch.Tensor, 
                 silver_spans_head_end: torch.Tensor, 
                 silver_spans_tail_start: torch.Tensor, 
                 silver_spans_tail_end: torch.Tensor, 
                 silver_spans_desc_ids: list[str], 
                 description_embeddings: torch.Tensor,
                 description_mean_embeddings: torch.Tensor,
                 description_embeddings_ids: list[str],):

        self.N = len(silver_spans_desc_ids)
        assert self.N == len(description_embeddings_ids)
        assert self.N == silver_spans_head_start.shape[0] == silver_spans_head_end.shape[0] == silver_spans_tail_start.shape[0] == silver_spans_tail_end.shape[0] == description_embeddings.shape[0]
        max_length = description_embeddings.shape[1]
        assert silver_spans_head_start.shape[1] == silver_spans_head_end.shape[1] == silver_spans_tail_start.shape[1] == silver_spans_tail_end.shape[1] == max_length
        self.silver_spans_head_start = silver_spans_head_start
        self.silver_spans_head_end = silver_spans_head_end
        self.silver_spans_tail_start = silver_spans_tail_start
        self.silver_spans_tail_end = silver_spans_tail_end
        self.description_embeddings = description_embeddings
        self.description_mean_embeddings = description_mean_embeddings
        self.description_embeddings_ids = description_embeddings_ids
        self.silver_spans_desc_ids = silver_spans_desc_ids
    def __getitem__(self, idx):
        description_id = self.silver_spans_desc_ids[idx]
        description_idx = self.description_embeddings_ids.index(description_id)


        return (
            self.description_embeddings[description_idx],
            self.description_mean_embeddings[description_idx],
            self.silver_spans_head_start[idx],
            self.silver_spans_head_end[idx],
            self.silver_spans_tail_start[idx],
            self.silver_spans_tail_end[idx],
        )

    def __len__(self):
        return self.N

class EntityExtractor(nn.Module):
    """Which tokens could be the start/end of entities, regardless of relation?"""

    def __init__(self, hidden_dim):
        """Entity extraction module, predits head start end and tail start end"""
        super().__init__()
        self.forward_head_start = nn.Linear(hidden_dim, 1)
        self.forward_head_end = nn.Linear(hidden_dim, 1)
        self.backward_tail_start = nn.Linear(hidden_dim, 1)
        self.backward_tail_end = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        """Sigmoid activations for head_start, head_end, tail_start, tail_end Returns 4 tensors of shape (B, L) with values between 0 and 1"""
        head_start = torch.sigmoid(self.forward_head_start(X))
        head_end = torch.sigmoid(self.forward_head_end(X))
        tail_start = torch.sigmoid(self.backward_tail_start(X))
        tail_end = torch.sigmoid(self.backward_tail_end(X))
        return head_start, head_end, tail_start, tail_end

class SemanticRelationAttention(nn.Module):
    """In paper 3.3.2. Semantic relation guidance: returning fine-grained sentence representatio
    We introduce MLP, 
    We introduce attention_emb_dim
    
    """



    def __init__(self, hidden_dim, rel_dim, attention_dim=256):
        """rel_dim is the dim of relation embeddings (might be 768 if bert, or other for transe)"""
        super().__init__()
        self.w_r = nn.Linear(rel_dim, attention_dim)
        self.w_g = nn.Linear(hidden_dim, attention_dim)
        self.w_x = nn.Linear(hidden_dim , attention_dim)

        self.V  = nn.Linear(attention_dim, 1)


    def forward(self, X, semantic_relation_embedding, tokens_mean_embedding):
        """
            X: (B, L, H) token embeddings
            semantic_relation_embedding: (R, H)
            tokens_mean_embedding: (B, H)
        """
        wx_xi = self.w_x(X) #(B, L, attention_dim)
        wr_rj = self.w_r(semantic_relation_embedding) #(R, attention_dim)
        wg_hg = self.w_g(tokens_mean_embedding) #(B, attention_dim)

        x_exp = wx_xi.unsqueeze(1) #(B, 1, L, attention_dim)
        r_exp = wr_rj.unsqueeze(0).unsqueeze(2) #(1, R, 1, attention_dim)
        g_exp = wg_hg.unsqueeze(1).unsqueeze(2) #(B, 1, 1, attention_dim)


        z = torch.tanh(x_exp + r_exp + g_exp) #(B, R, L, attention_dim)
        e = self.V(z).squeeze(-1) #(B, R, L)

        pass


class TransERelationAttention(torch.nn.Module):

    pass


class BraskModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(BraskModel, self).__init__()


        # ! I should be careful about the span reconstructions because (the model predicts start and end independently):
        # ! Multiple possible spans
        # ! Overlapping spans
        # ! Invalid spans (end < start)

        # ! for loss I can do with 3 parameters instead of (Entity loss, forward loss, backward loss) do alpha with starting as 1 and tune after.

        # ! in the algorithm , I have to do fusion (token emb, relation emb, entity emb (subject, object)):
        # ! thie way the paper do the confusion is attention fusion (score =f(x_i, r, global_context)), we might want in the future do other fusion like gating which would be g = sigmoid(W[x_i, r]); h_i = g * x_i + (1-g) * r


        # ! I can do relations smart pruning which is doing pre-processed cosine similarity between description and the relations, maybe not bert but something else (I am thinking sentemce transformers but not neccessarily, )
        # ! After I can use it to decide which relations (top-k) to consider .
        # ! +  positive relations from ground truth triples !!!.



        # After prediction:
        #     Generate candidate spans
        #     Keep only spans that:
        #     match aliases (exact or fuzzy)
        #     This improves precision a lot
        # DO NOT: force model to only predict aliases

        # Circular traning:
        #    1- train entity extractor first
        #    2- frozen encoder - train relation extraction and forward extractor (goal : learn relation conditioning without noise )
        # 3-  add backward extractor
        # 4- final fine tuning 


        

        # for predicted head and predicted tail:
        # entity representation = avg(X[start] + X[end]) / 2

        self.starting_entity = EntityExtractor(hidden_dim)


    def __forward__(self, batch):


        # B: batch size, L: sequence length, H: hidden dimension

        description_embeddings = batch[0] # shape (B, L, H)
        description_mean_embeddings = batch[1] # shape (B, H)

        # shape (B, L)
        silver_spans_head_start, silver_spans_head_end, silver_spans_tail_start, silver_spans_tail_end = batch[2], batch[3], batch[4], batch[5] 

        forward_head_start, forward_head_end, backward_tail_start, backward_tail_end = self.starting_entity(description_embeddings)



def main(use_minimized: bool):
    
    if not check_training_files(use_minimized):
        return


    BATCH_SIZE = 64 if use_cuda else 16
    NUM_WORKERS = 4 if use_cuda else 0


    print("Loading data")
    files_paths = settings.MINIMIZED_FILES if use_minimized else settings.PREPROCESSED_FILES
    silver_spans_fp = files_paths.SILVER_SPANS
    description_embeddings_all_fp = files_paths.DESCRIPTION_EMBEDDINGS_ALL
    description_embeddings_mean_fp = files_paths.DESCRIPTION_EMBEDDINGS_MEAN

    silver_spans = read_cached_array(silver_spans_fp)
    description_embeddings_all = read_tensor(description_embeddings_all_fp)
    description_embeddings_mean = read_tensor(description_embeddings_mean_fp)
    description_embeddings_ids = read_cached_array(files_paths.DESCRIPTION_EMBEDDINGS_IDS)


    silver_spans_head_start = silver_spans["head_start"]
    silver_spans_head_end = silver_spans["head_end"]
    silver_spans_tail_start = silver_spans["tail_start"]
    silver_spans_tail_end = silver_spans["tail_end"]
    silver_spans_desc_ids = silver_spans["desc_ids"]

    dataset = BraskDataset(
        silver_spans_head_start=silver_spans_head_start,
        silver_spans_head_end=silver_spans_head_end,
        silver_spans_tail_start=silver_spans_tail_start,
        silver_spans_tail_end=silver_spans_tail_end,
        silver_spans_desc_ids=silver_spans_desc_ids,
        description_embeddings=description_embeddings_all,
        description_mean_embeddings=description_embeddings_mean,
        description_embeddings_ids=description_embeddings_ids
    )
    sampler = DistributedSampler(dataset)
    print("creating data loader")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler = sampler,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
    )



    for ds_batch in dataloader:
        print(ds_batch)
        break


    pass


if __name__ == "__main__":
    answer = input("Train on minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)