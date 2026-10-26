"""Paper-inspired Attention1D oracle with optional image-feature fusion.

Inputs are per-residue ESM-2 states, frozen ResNet18 image features, and
temperature/pH/duration. No Ridge fallback and no randomly initialized serving.
"""
import torch
from torch import nn


class AttentionOracle(nn.Module):
    def __init__(self, sequence_dim=480, image_dim=512, hidden=64):
        super().__init__()
        self.sequence = nn.Sequential(nn.Linear(sequence_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.attention = nn.Linear(hidden, 1)
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.ReLU())
        self.environment = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 3 + 1, hidden), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(hidden, 1))

    def forward(self, sequence, mask, image, image_present, environment):
        if not mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one residue.")
        residues = self.sequence(sequence)
        weights = self.attention(residues).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        pooled = (residues * weights.unsqueeze(-1)).sum(1)
        visual = self.image(image) * image_present.reshape(-1, 1)
        fused = torch.cat([pooled, visual, self.environment(environment),
                           image_present.reshape(-1, 1)], dim=-1)
        return self.head(fused).squeeze(-1), weights


def batch_features(items, device="cpu"):
    length = max(len(x["sequence"]) for x in items)
    dim = items[0]["sequence"].shape[-1]
    sequence = torch.zeros(len(items), length, dim, device=device)
    mask = torch.zeros(len(items), length, dtype=torch.bool, device=device)
    for i, item in enumerate(items):
        n = len(item["sequence"])
        sequence[i, :n] = torch.as_tensor(item["sequence"], device=device)
        mask[i, :n] = True
    return sequence, mask, torch.stack([torch.as_tensor(x["image"], device=device) for x in items])
