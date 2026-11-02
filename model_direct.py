"""Direct neural circle regression without a geometric solver in the forward path."""

import torch
from torch import nn
from torch.nn import functional as F

from repro_model import CircleNetRepro


class CircleNetDirect(nn.Module):
    """Predict center, radius, and normal directly from point features.

    The boundary score controls a permutation-invariant attention pool, so the
    parameter losses can update the point classifier as well as the heads.
    """

    def __init__(self):
        super().__init__()
        self.backbone = CircleNetRepro()
        self.normal_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.center_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.radius_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, query, context, teacher_mask=None, teacher_ratio=0.0):
        logits, weight_logits, features = self.backbone(query, context, return_features=True)
        probability = logits.softmax(dim=-1)[..., 1].square()
        if teacher_mask is not None and teacher_ratio > 0:
            probability = (1 - teacher_ratio) * probability + teacher_ratio * teacher_mask.float()
        attention = probability * (0.05 + 0.95 * weight_logits.sigmoid()) + 1e-4
        attention = attention / attention.sum(dim=1, keepdim=True)
        pooled = (attention[..., None] * features).sum(dim=1)
        origin = (attention[..., None] * query).sum(dim=1)
        scale = query.std(dim=1).norm(dim=-1).clamp_min(1e-3)
        center = origin + 2 * scale[:, None] * self.center_head(pooled).tanh()
        radius = scale * self.radius_head(pooled).squeeze(-1).clamp(-6, 3).exp()
        normal = F.normalize(self.normal_head(pooled), dim=-1)
        return logits, weight_logits, normal, center, radius
