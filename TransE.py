import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np




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

    def __init__(self, triples):
        """Builds index maps and pre-generates negative triples.

        Args:
            minimized (bool): If True, loads the minimized subset of training
                triples; otherwise loads the full training set.
        """
        
        flat_triples =triples

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

    def __len__(self):
        """Returns the number of training triples."""
        return len(self.triples)

    def __getitem__(self, idx):
        """Returns the positive and a freshly-sampled negative triple at index ``idx``.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of int64 tensors of shape
                ``(3,)`` — the positive triple and its corrupted negative counterpart.
        """
        pos = self.triples[idx]
        neg = pos.copy()
        if torch.randint(0, 2, (1,)).item():  # corrupt tail
            neg[2] = torch.randint(0, self.n_ents, (1,)).item()
        else:                                  # corrupt head
            neg[0] = torch.randint(0, self.n_ents, (1,)).item()
        return torch.from_numpy(pos), torch.from_numpy(neg)


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

    def __init__(self, n_ents, n_rels, emb_dim):
        """Initialises embedding tables with Xavier uniform weights.

        Args:
            n_ents (int): Number of unique entities in the knowledge graph.
            n_rels (int): Number of unique relations in the knowledge graph.
        """
        super(TransEModel, self).__init__()
        self.p_norm = 1
        self.emb_dim = emb_dim

        self.ent_embs = nn.Embedding(n_ents, emb_dim)
        self.rel_embs = nn.Embedding(n_rels, emb_dim)

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

        pos_dist = torch.norm(pos_head_entity + pos_rel - pos_tail_entity, p=self.p_norm, dim=1)
        neg_dist = torch.norm(neg_head_entity + neg_rel - neg_tail_entity, p=self.p_norm, dim=1)

        return pos_dist, neg_dist
