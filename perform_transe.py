import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils.files import save_tensor
from utils.pre_processed_data import data_loader
from utils.settings import settings


MINIMIZED = True
BATCH_SIZE = 256
NUM_WORKERS = 4
MARGIN = 1.0
TRANSE_EMB_DIM = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 120


class TransEDataset(Dataset):
    """PyTorch Dataset for TransE training.

    Loads training triples via ``data_loader``, builds integer index maps for
    entities and relations, and pre-generates one negative triple per positive
    triple by corrupting either the head or the tail entity.

    Attributes:
        ent2idx (dict[str, int]): Maps entity ID string to integer index.
        rel2idx (dict[str, int]): Maps relation ID string to integer index.
        n_ents (int): Total number of unique entities.
        n_rels (int): Total number of unique relations.
        triples (np.ndarray): Positive triples as integer indices, shape (N, 3).
        neg_triples (np.ndarray): Corrupted negative triples, shape (N, 3).
    """

    def __init__(self, minimized: bool):
        """Builds index maps and pre-generates negative triples.

        Args:
            minimized (bool): If True, loads the minimized subset of training
                triples; otherwise loads the full training set.
        """
        triples_dict = data_loader.get_triples_train(minimized=minimized)
        flat_triples = [t for ts in triples_dict.values() for t in ts]

        entity_ids = sorted({h for h, _, _ in flat_triples} | {t for _, _, t in flat_triples})
        relation_ids = sorted({r for _, r, _ in flat_triples})
        self.ent2idx = {e: i for i, e in enumerate(entity_ids)}
        self.rel2idx = {r: i for i, r in enumerate(relation_ids)}
        self.n_ents = len(entity_ids)
        self.n_rels = len(relation_ids)

        self.triples = np.array(
            [(self.ent2idx[h], self.rel2idx[r], self.ent2idx[t]) for h, r, t in flat_triples],
            dtype=np.int64,
        )
        # Pre-generate negative triples (corrupt head or tail randomly)
        
        #false to corrupt head, true to corrupt tail
        mask = torch.randint(0,2, size=(len(self.triples),), dtype=torch.bool)
        random_ents = torch.randint(0, self.n_ents, size=(len(self.triples), ), dtype=torch.int64)

        self.neg_triples = self.triples.copy()
        self.neg_triples[~mask, 0] = random_ents[~mask]
        self.neg_triples[mask, 2] = random_ents[mask]

        # rng = np.random.default_rng(42)
        # corrupt_head = rng.integers(0, 2, size=len(self.triples), dtype=bool)
        # rand_ents = rng.integers(0, self.n_ents, size=len(self.triples), dtype=np.int64)
        # self.neg_triples = self.triples.copy()
        # self.neg_triples[corrupt_head, 0] = rand_ents[corrupt_head]
        # self.neg_triples[~corrupt_head, 2] = rand_ents[~corrupt_head]

    def __len__(self):
        """Returns the number of training triples."""
        return len(self.triples)

    def __getitem__(self, idx):
        """Returns the positive and negative triple at index ``idx``.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of int64 tensors of shape
                ``(3,)`` — the positive triple and its corrupted negative counterpart.
        """
        return (
            torch.from_numpy(self.triples[idx]),
            torch.from_numpy(self.neg_triples[idx]),
        )


class TransEModel(nn.Module):
    """TransE knowledge graph embedding model (Bordes et al. 2013).

    Learns entity and relation embeddings such that for a valid triple
    ``(h, r, t)``, the scoring function ``||h + r - t||_1`` is minimized.
    Both embedding tables are Xavier-initialized.

    Attributes:
        p_norm (int): Norm order used for distance computation (1 = L1).
        emb_dim (int): Dimensionality of entity and relation embeddings.
        ent_embs (nn.Embedding): Entity embedding table of shape
            ``(n_ents, TRANSE_EMB_DIM)``.
        rel_embs (nn.Embedding): Relation embedding table of shape
            ``(n_rels, TRANSE_EMB_DIM)``.
    """

    def __init__(self, n_ents, n_rels):
        """Initialises embedding tables with Xavier uniform weights.

        Args:
            n_ents (int): Number of unique entities in the knowledge graph.
            n_rels (int): Number of unique relations in the knowledge graph.
        """
        super(TransEModel, self).__init__()
        self.p_norm = 1
        self.emb_dim = TRANSE_EMB_DIM

        self.ent_embs = nn.Embedding(n_ents, TRANSE_EMB_DIM)
        self.rel_embs = nn.Embedding(n_rels, TRANSE_EMB_DIM)

        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward(self, pos_triples, neg_triples):
        """Computes L1 distances for positive and negative triple batches.

        Args:
            pos_triples (torch.Tensor): Positive triple indices of shape
                ``(B, 3)`` with columns ``[head_idx, rel_idx, tail_idx]``.
            neg_triples (torch.Tensor): Corrupted negative triple indices,
                same shape as ``pos_triples``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of 1-D float tensors of
                shape ``(B,)`` — ``pos_dist`` and ``neg_dist`` — representing
                ``||h + r - t||_1`` for positive and negative triples
                respectively.
        """
        pos_heads, pos_rels, pos_tails = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]
        pos_head_entity = self.ent_embs(pos_heads)
        pos_rel = self.rel_embs(pos_rels)
        pos_tail_entity = self.ent_embs(pos_tails)

        neg_heads, neg_rels, neg_tails = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]
        neg_head_entity = self.ent_embs(neg_heads)
        neg_rel = self.rel_embs(neg_rels)
        neg_tail_entity = self.ent_embs(neg_tails)

        pos_dist = torch.norm(pos_head_entity + pos_rel - pos_tail_entity, dim=1)
        neg_dist = torch.norm(neg_head_entity + neg_rel - neg_tail_entity, dim=1)

        return pos_dist, neg_dist


def main():
    """Trains TransE and saves the resulting relation embeddings.

    Detects whether to use Distributed Data Parallel by checking for the
    ``LOCAL_RANK`` environment variable (set automatically by ``torchrun``).
    Falls back to single-GPU or CPU training when ``LOCAL_RANK`` is absent.

    Training procedure:
        1. Builds ``TransEDataset`` from the training triples.
        2. Initialises ``TransEModel`` and optionally wraps it with DDP.
        3. Runs ``NUM_EPOCHS`` epochs with Adam + CosineAnnealingLR.
        4. After each batch, L1-normalises entity and relation embeddings.
        5. On completion (rank 0 only), saves the relation embedding matrix
           to ``transe_rel_embs.npz`` inside the minimized or preprocessed
           data directory depending on ``MINIMIZED``.

    Output file shape: ``(n_rels, TRANSE_EMB_DIM)``, accessible via
    ``np.load(path)["arr"]``.
    """
    local_rank_str = os.environ.get("LOCAL_RANK")
    use_ddp = local_rank_str is not None

    if use_ddp:
        local_rank = int(local_rank_str)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        print(f"LOCAL RANK: {local_rank}")
    else:
        local_rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Running on {device} (no DDP)")

    dataset = TransEDataset(minimized=MINIMIZED)
    print(f"Dataset: {len(dataset):,} triples | {dataset.n_ents:,} entities | {dataset.n_rels:,} relations")

    if use_ddp:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    model = TransEModel(dataset.n_ents, dataset.n_rels).to(device)
    if use_ddp:
        model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    if device.type == "cuda":
        autocast_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = lambda: autocast(device_type="cpu", dtype=torch.bfloat16)

    def get_core_model():
        return model.module if use_ddp else model

    for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS, desc=f"Epochs [rank {local_rank}]"):
        model.train()
        if use_ddp:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        for pos_batch, neg_batch in tqdm(loader, desc=f"Epoch {epoch + 1} [rank {local_rank}]", leave=False):
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)

            optimizer.zero_grad()
            with autocast_ctx():
                pos_dist, neg_dist = model(pos_batch, neg_batch)
                loss = torch.clamp(MARGIN + pos_dist - neg_dist, min=0).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            core = get_core_model()
            core.ent_embs.weight.data = F.normalize(core.ent_embs.weight.data, p=1, dim=1)
            core.rel_embs.weight.data = F.normalize(core.rel_embs.weight.data, p=1, dim=1)

            total_loss += loss.item()
        print(f"[rank {local_rank}] Epoch {epoch + 1} Loss: {total_loss:.4f}")
        scheduler.step()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if local_rank == 0:
        out_path = settings.MINIMIZED_FILES.TRANSE_MODEL_RESULTS if MINIMIZED else settings.PREPROCESSED_FILES.TRANSE_MODEL_RESULTS
        save_tensor(get_core_model().rel_embs.weight.data, out_path)
        print(f"Shape: {tuple(get_core_model().rel_embs.weight.data.shape)}  →  {out_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
