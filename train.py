
import torch
import os

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

class BraskModel(torch.nn.Module):
    def __init__(self):
        super(BraskModel, self).__init__()


        # 4 classifiers for head start, head end, tail start, tail end
        # each classifier is linear (H -> 1) + sigmoid


        # for predicted head and predicted tail:
        # entity representation = avg(X[start] + X[end]) / 2
        
        
        pass



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