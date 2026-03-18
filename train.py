
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
    Each item is a tuple of (description_embedding: Tensor[L,D], description_mean_embeddings: Tensor[D], description_id: str)"""


    def __init__(self, 
                 description_embeddings: torch.Tensor,
                 description_mean_embeddings: torch.Tensor,
                 description_embeddings_ids: list[str],):

        self.N = description_embeddings.shape[0]
        self.description_embeddings = description_embeddings
        self.description_mean_embeddings = description_mean_embeddings
        self.description_embeddings_ids = description_embeddings_ids

    def __getitem__(self, idx):
        return (
            self.description_embeddings[idx],
            self.description_mean_embeddings[idx],
            self.description_embeddings_ids[idx],
        )

    def __len__(self):
        return self.N

class EntityExtractor(nn.Module):
    """Which tokens could be the start/end of entities, regardless of relation"""

    def __init__(self, hidden_dim):
        """Entity extraction module, predits start and end"""
        super().__init__()
        self.start_linear = nn.Linear(hidden_dim, 1)
        self.end_linear = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sigmoid activations for start, end Returns 2 tensors of shape (B, L) with values between 0 and 1"""
        start = torch.sigmoid(self.start_linear(X))
        end = torch.sigmoid(self.end_linear(X))
        return start, end

class RelationAttention(nn.Module):
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


    def forward(self, X, relation_embedding, tokens_mean_embedding):
        """
        args:
            X: (B, L, H) token embeddings
            relation_embedding: (R, H) H can be hidden_dim or transe_rel_dim
            tokens_mean_embedding: (B, H)
        returns:
            c: (B, R, H) context-aware token representations for each relation
            a: (B, R, L) attention weights for each token and relation
        """
        wx_xi = self.w_x(X) #(B, L, attention_dim)
        wr_rj = self.w_r(relation_embedding) #(R, attention_dim)
        wg_hg = self.w_g(tokens_mean_embedding) #(B, attention_dim)

        x_exp = wx_xi.unsqueeze(1) #(B, 1, L, attention_dim)
        r_exp = wr_rj.unsqueeze(0).unsqueeze(2) #(1, R, 1, attention_dim)
        g_exp = wg_hg.unsqueeze(1).unsqueeze(2) #(B, 1, 1, attention_dim)


        z = torch.tanh(x_exp + r_exp + g_exp) #(B, R, L, attention_dim)
        e = self.V(z).squeeze(-1) #(B, R, L)

        a = torch.softmax(e, dim=-1) #(B, R, L)
        a_exp = a.unsqueeze(-1) #(B, R, L, 1)

        x_exp = X.unsqueeze(1) #(B, 1, L, H)
        c = (a_exp * x_exp).sum(dim=2) #(B, R, H)

        return c,a

class FuseExtractor(nn.Module):
    """3.3.3. Extraction of objects, 3.4. Backward triple extraction"""

    """
    fuse subject representations  and fine-grained sentence expression into ith token representation
    Hik = Ws Sk + Wx xi
    Hij = cj + xi
    Hijk = Hij + Hik

    we feed the special representation into fully connected neural network to obtain start and end probabilities
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.w_s = nn.Linear(hidden_dim , hidden_dim )
        self.w_x = nn.Linear(hidden_dim, hidden_dim)



    def forward(self, X: torch.Tensor, c: torch.Tensor, sk: torch.Tensor, sk_mask: torch.Tensor) -> torch.Tensor:
        """
        args:
            X Tensor for token embeddings (B, L, H)
            c Tensor Relation context (B, R, H)
            sk tensor with shape: (B, max_num_subjects, H)
            sk_mask tensor with shape: (B, max_num_subjects) with 1 for valid subject representations and 0 for padded ones

        Returns:
            h_ijk tensor with shape (B, R, max_num_subjects, L, H)
        """

        R  = c.shape[1]

        X_exp = X.unsqueeze(1) #(B, 1, L, H)
        X_exp = X_exp.expand(-1, R, -1, -1) # (B, R, L, H)

        c_exp = c.unsqueeze(2) # (B, R, 1, H)

        w_x  = self.w_x(X) #(B, L, H)
        w_sk = self.w_s(sk) #(B, max_num_subjects, H)
        w_sk = sk_mask.unsqueeze(-1) * w_sk #(B, max_num_subjects, H) with padded subjects zeroed out

        h_ik = w_sk.unsqueeze(2) + w_x.unsqueeze(1) #(B, max_num_subjects, 1, H) + (B, 1, L, H) -> (B, max_num_subjects, L, H)
        h_ij = c_exp + X_exp # (B, R, 1, H) + (B, R, L, H) -> (B, R, L, H)


        # I need (B, R, L, H)
        h_ijk = h_ik.unsqueeze(1) + h_ij.unsqueeze(2) # (B, 1, SUBJECT, L, H) + (B, R, 1, L, H ) -> (B, R, SUBJECT, L, H)


        return h_ijk


def extract_sk(description_embeddings: torch.Tensor, 
               start_probs: torch.Tensor, 
               end_probs: torch.Tensor, 
               start_threshold: float, 
               end_threshold: float, 
               max_span_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract subject representations s_k and padding it.
    args:
        description_embeddings: (B, L, H)
        start_probs: (B, L) with values between 0 and 1
        end_probs: (B, L) with values between 0 and 1
        start_threshold: scalar between 0 and 1
        end_threshold: scalar between 0 and 1
        max_span_length: maximum length of subject spans
    returns:
        forward_s_k: (B, max_num_subjects, H) with padded subject representations zeroed out
        mask: (B, max_num_subjects) with 1 for valid subject representations and 0 for padded ones
    """
    

    #! Here I should do the idea of weighting the subjects using aliases dictionary
    B = description_embeddings.shape[0]
    H = description_embeddings.shape[2]
    s_k = []
    for b in range(B):
        x_emb = description_embeddings[b]
        start_idx = (start_probs[b].squeeze(-1) >= start_threshold).nonzero(as_tuple=False).squeeze(-1)
        start_idx = torch.sort(start_idx).values
        end_idx  = (end_probs[b].squeeze(-1) >= end_threshold).nonzero(as_tuple=False).squeeze(-1)
        consumed_ends = set()
        spans = []
        for s in start_idx:
            end_mask = ( (end_idx >= s) & (end_idx < s+max_span_length))
            valid_ends = end_idx[end_mask]
            valid_ends = [e.item() for e in valid_ends if e.item() not in consumed_ends]
            if len(valid_ends) == 0:
                continue
            e = min(valid_ends)
            spans.append((s.item(), e))
            consumed_ends.add(e)
        s_k_list = []
        for (s, e)  in spans:
            s_k = (x_emb[s] + x_emb[e]) / 2
            s_k_list.append(s_k)
        if s_k_list:
            s_k_list = torch.stack(s_k_list, dim=0)
        if len(s_k_list) == 0:
            s_k_list = torch.zeros(1, H, device=x_emb.device)
        s_k.append(s_k_list)
    max_num_subjects = max([s.shape[0] for s in s_k])
    padded_sk = []
    mask = []
    for s in s_k:
        K = s.shape[0]
        if K < max_num_subjects:
            pad = torch.zeros(max_num_subjects - K, s.shape[1], device=s.device)
            s_padded  = torch.cat([s, pad], dim=0)
            m = torch.cat([torch.ones(K, device=s.device), torch.zeros(max_num_subjects - K, device=s.device)], dim=0)
        else:
            s_padded = s
            m=torch.ones(K, device=s.device)

        padded_sk.append(s_padded)
        mask.append(m)
    padded_sk = torch.stack(padded_sk, dim=0) #(B, max_num_subjects, H)
    mask = torch.stack(mask, dim=0) #(B, max_num_subjects)
    return padded_sk, mask


class BraskModel(torch.nn.Module):
    def __init__(self, hidden_dim, trane_rel_dim):
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

        self.forward_head_predict = EntityExtractor(hidden_dim)
        self.backward_tail_predict = EntityExtractor(hidden_dim)
        self.semantic_relation_attention = RelationAttention(hidden_dim, rel_dim=hidden_dim)
        self.trane_relation_attention = RelationAttention(hidden_dim, rel_dim=trane_rel_dim)

        self.fuse_extractor_forward = FuseExtractor(hidden_dim)
        self.fuse_extractor_backward = FuseExtractor(hidden_dim)

        self.forward_tail_predict = EntityExtractor(hidden_dim)
        self.backward_head_predict = EntityExtractor(hidden_dim)


        self.threshold_head_start = 0.5
        self.threshold_head_end = 0.5
        self.threshold_tail_start = 0.5
        self.threshold_tail_end = 0.5
        self.max_span_len = 10

    def forward(self, batch, semantic_relation_embeddings, transe_relation_embeddings):


        # B: batch size, L: sequence length, H: hidden dimension

        description_embeddings = batch[0] # shape (B, L, H)
        description_mean_embeddings = batch[1] # shape (B, H)
        description_ids = batch[2] # shape (B,)

        B, L, H = description_embeddings.shape




        forward_head_start, forward_head_end = self.forward_head_predict(description_embeddings)
        backward_tail_start, backward_tail_end = self.backward_tail_predict(description_embeddings)

        forward_c, forward_a = self.semantic_relation_attention(description_embeddings, semantic_relation_embeddings, description_mean_embeddings)
        backward_c, backward_a = self.trane_relation_attention(description_embeddings, transe_relation_embeddings, description_mean_embeddings)


        # Extract sk
        forward_sk, forward_sk_mask = extract_sk(
            description_embeddings=description_embeddings,
            start_probs=forward_head_start,
            end_probs=forward_head_end,
            start_threshold=self.threshold_head_start,
            end_threshold=self.threshold_head_end,
            max_span_length=self.max_span_len
            )
        backward_sk, backward_sk_mask = extract_sk(
            description_embeddings=description_embeddings,
            start_probs=backward_tail_start,
            end_probs=backward_tail_end,
            start_threshold=self.threshold_tail_start,
            end_threshold=self.threshold_tail_end,
            max_span_length=self.max_span_len
        )




        forward_hijk = self.fuse_extractor_forward(description_embeddings, forward_c, forward_sk, forward_sk_mask) # (B, R, max_num_subjects, L, H)
        backward_hijk = self.fuse_extractor_backward(description_embeddings, backward_c, backward_sk, backward_sk_mask) # (B, R, max_num_subjects, L, H)

        forward_tail_start, forward_tail_end = self.forward_tail_predict(forward_hijk)
        backward_head_start, backward_head_end = self.backward_head_predict(backward_hijk)


        return {
            "description_ids": description_ids,
            "froward": {
                "head_start": forward_head_start,
                "head_end": forward_head_end,
                "tail_start": forward_tail_start,
                "tail_end": forward_tail_end,
            },
            "backward": {
                "tail_start": backward_tail_start,
                "tail_end": backward_tail_end,
                "head_start": backward_head_start,
                "head_end": backward_head_end,
            }
        }


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


    dataset = BraskDataset(
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



    silver_spans_head_start = silver_spans["head_start"]
    silver_spans_head_end = silver_spans["head_end"]
    silver_spans_tail_start = silver_spans["tail_start"]
    silver_spans_tail_end = silver_spans["tail_end"]
    silver_spans_desc_ids = silver_spans["desc_ids"]

    for ds_batch in dataloader:
        print(ds_batch)
        break


    pass


if __name__ == "__main__":
    answer = input("Train on minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)