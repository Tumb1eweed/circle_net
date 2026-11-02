"""Circle-Net v2: boundary prediction, learned weights, and a learned plane."""

import torch
from torch import nn
from torch.nn import functional as F

from repro_model import CircleNetRepro, weighted_circle_fit


class CircleNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CircleNetRepro()
        self.normal_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, query, context):
        logits, weight_logits, features = self.backbone(query, context, return_features=True)
        normal = F.normalize(self.normal_head(features.mean(dim=1)), dim=-1)
        return logits, weight_logits, normal

    @staticmethod
    def fit(query, logits, weight_logits, normal, teacher_mask=None, teacher_ratio=0.0):
        boundary_probability = logits.softmax(dim=-1)[..., 1]
        predicted_mask = boundary_probability.square()
        if teacher_mask is not None and teacher_ratio > 0:
            predicted_mask = (1 - teacher_ratio) * predicted_mask + teacher_ratio * teacher_mask.float()
        weights = predicted_mask * (0.05 + 0.95 * weight_logits.sigmoid())
        center, radius = weighted_circle_fit(query, weights, normal, ridge=1e-4)
        return center, radius, weights
