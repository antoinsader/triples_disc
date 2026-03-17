
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoModel


from utils.files import cache_array, save_tensor
from utils.pre_processed_data import data_loader,  check_minimized_files
from utils.settings import settings

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def get_embeddings(sentences: list[str], max_length: int=128) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed each sentence using BERT, returns (mean_embs: Tensor[N, D], all_embs: Tensor[N, L, D] )"""
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased')
    model  = model.to(device)
    model.eval()
    batch_size = 128 if use_cuda else 16

    all_means = []
    all_embs = []
    for start in tqdm(range(0, len(sentences), batch_size) , desc="Embedding senteneces"):
        end = min(start + batch_size, len(sentences))
        chunk = sentences[start: end]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            embs = out.last_hidden_state #(B, L, H)
        attention_mask = attention_mask.unsqueeze(-1) #(B, L, 1)
        sum_embs = (embs * attention_mask).sum(dim=1) #(B, H)
        token_counts = attention_mask.sum(dim=1).clamp(min=1) #(B, 1)
        mean_embs = sum_embs / token_counts #(B, H)

        all_means.append(mean_embs.cpu())
        all_embs.append(embs.cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final_mean_embs = torch.cat(all_means, dim = 0)
    final_all_embs = torch.cat(all_embs, dim = 0)
    print(f"mean_embs shape: {final_mean_embs.shape} should be (N, H) (num_descs, 768)")
    print(f"mean_embs shape: {final_all_embs.shape} should be (N, L, H) (num_descs, {max_length},  768)")
    return final_mean_embs, final_all_embs



def main(use_minimized):
    if use_minimized and not check_minimized_files():
        return
    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    sentences = list(descriptions.values())
    max_length = 128
    # mean_embs shape: (num_descs, 768)
    # all_embs shape: (num_descs, 128, 768)
    mean_embs, all_embs = get_embeddings(sentences, max_length=max_length)

    out_mean_embs = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN if use_minimized else settings.PREPROCESSED_FILES.DESCRIPTION_EMBEDDINGS_MEAN
    out_all_embs = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL if use_minimized else settings.PREPROCESSED_FILES.DESCRIPTION_EMBEDDINGS_ALL
    out_ids = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_IDS if use_minimized else settings.PREPROCESSED_FILES.DESCRIPTION_EMBEDDINGS_IDS
    print(f"\nSaving mean embeddings to {out_mean_embs} and all embeddings to {out_all_embs}...")
    save_tensor(mean_embs, out_mean_embs)
    save_tensor(all_embs, out_all_embs)
    cache_array(list(descriptions.keys()), out_ids)




if __name__ == "__main__":
    answer = input("Prepare description embeddings on minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)
