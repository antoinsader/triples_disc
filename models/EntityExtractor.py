

import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityExtractor(nn.Module):
    """Which tokens could be the start/end of entities, regardless of relation"""

    def __init__(self, hidden_dim):
        """Entity extraction module, predits start and end"""
        super().__init__()
        self.start_linear = nn.Linear(hidden_dim, 1)
        self.end_linear = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            X can be any shape (..., L, H).
            Returns start_probs, end_probs, start_logits, end_logits each of shape (..., L, 1) — trailing H collapsed to scalar.
        """
        start_logits = self.start_linear(X)
        end_logits = self.end_linear(X)
        start = torch.sigmoid(start_logits)
        end = torch.sigmoid(end_logits)
        return start, end, start_logits, end_logits


# ? This for just staged and we will not use this loss for the whole BraskModel 
def compute_loss(
    fwd_head_start_logits: torch.Tensor, #(B, L)
    fwd_head_end_logits: torch.Tensor, #(B,L)
    bwd_tail_start_logits: torch.Tensor, #(B,L)
    bwd_tail_end_logits: torch.Tensor, #(B,L)
    gold_triples_per_sentence: list, # [b] -> [(hs, he), r, (ts, te)]
    token_mask: torch.Tensor, # (B, L) 0 if pad
    max_length: int # L
)-> torch.Tensor:

    device = fwd_head_start_logits.device
    fwd_loss_head = torch.tensor(0.0, device=device)
    bwd_loss_tail = torch.tensor(0.0, device=device)
    n_triples = 0

    for b, triples in enumerate(gold_triples_per_sentence):
        mask = token_mask[b]
        def masked_bce(logits, gold):
            loss = F.binary_cross_entropy_with_logits(
                logits, gold, reduction="None"
            )
            return (loss * mask).sum()  / (mask.sum() + 1e-8)

        for (head_start, head_end), r, (tail_start, tail_end) in triples:
            gold_h_start = torch.zeros(max_length, device=device)
            gold_h_end = torch.zeros(max_length, device=device)
            gold_h_start[head_start] = 1.0
            gold_h_end[head_end] = 1.0

            gold_t_start = torch.zeros(max_length, device=device)
            gold_t_end   = torch.zeros(max_length, device=device)
            gold_t_start[tail_start] = 1.0
            gold_t_end[tail_end]     = 1.0

            fwd_loss_head = fwd_loss_head +  masked_bce(fwd_head_start_logits[b], gold_h_start) +  masked_bce(fwd_head_end_logits[b], gold_h_end)
            bwd_loss_tail = bwd_loss_tail + masked_bce(bwd_tail_start_logits[b], gold_h_start) +  masked_bce(bwd_tail_end_logits[b], gold_h_end)

            n_triples += 1

            #! here usually I include for the object which I will not do here

    return (fwd_loss_head + bwd_loss_tail) / max(n_triples, 1)
def test_train_alone():
    pass