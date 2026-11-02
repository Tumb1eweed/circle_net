"""PTv3-inspired point-to-point attention for the 500 query points.

This is a compact ablation, not the official PTv3 architecture or pretrained
Sonata weights. It uses Morton-ordered local windows and a final global block.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model_v2 import CircleNetV2


def morton_order(points):
    """Sort each cloud by a 10-bit-per-axis Z-order code."""
    low = points.amin(dim=1, keepdim=True)
    span = (points.amax(dim=1, keepdim=True) - low).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    grid = (((points - low) / span) * 1023).long().clamp(0, 1023)
    code = torch.zeros(points.shape[:2], dtype=torch.long, device=points.device)
    for bit in range(10):
        for axis in range(3):
            code |= ((grid[..., axis] >> bit) & 1) << (3 * bit + axis)
    return code.argsort(dim=1, stable=True)


class PointAttentionBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, tokens, padding_mask=None):
        normalized = self.norm1(tokens)
        update, _ = self.attn(normalized, normalized, normalized,
                              key_padding_mask=padding_mask, need_weights=False)
        tokens = tokens + update
        return tokens + self.mlp(self.norm2(tokens))


class CircleNetPointAttention(CircleNetV2):
    def __init__(self, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.position = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128))
        self.local_blocks = nn.ModuleList(PointAttentionBlock() for _ in range(2))
        self.global_block = PointAttentionBlock()
        self.output_norm = nn.LayerNorm(128)

    def _window_attention(self, tokens, order, block, shift):
        batch, count, width = tokens.shape
        sorted_tokens = torch.gather(tokens, 1, order[..., None].expand(-1, -1, width))
        if shift:
            sorted_tokens = sorted_tokens.roll(-shift, dims=1)
        pad = (-count) % self.window_size
        if pad:
            sorted_tokens = F.pad(sorted_tokens, (0, 0, 0, pad))
        windows = sorted_tokens.reshape(batch, -1, self.window_size, width)
        flat = windows.reshape(-1, self.window_size, width)
        mask = None
        if pad:
            mask = torch.arange(count + pad, device=tokens.device) >= count
            mask = mask.reshape(1, -1, self.window_size).expand(batch, -1, -1)
            mask = mask.reshape(-1, self.window_size)
        flat = block(flat, mask)
        sorted_tokens = flat.reshape(batch, count + pad, width)[:, :count]
        if shift:
            sorted_tokens = sorted_tokens.roll(shift, dims=1)
        inverse = order.argsort(dim=1)
        return torch.gather(sorted_tokens, 1, inverse[..., None].expand(-1, -1, width))

    def forward(self, query, context):
        _, _, features = self.backbone(query, context, return_features=True)
        origin = query.mean(dim=1, keepdim=True)
        scale = (query - origin).norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
        features = features + self.position((query - origin) / scale[..., None])
        order = morton_order(query)
        for index, block in enumerate(self.local_blocks):
            shift = self.window_size // 2 if index % 2 else 0
            features = self._window_attention(features, order, block, shift)
        features = self.output_norm(self.global_block(features))
        logits = self.backbone.classifier(features)
        weight_logits = self.backbone.weight_head(features).squeeze(-1)
        normal = F.normalize(self.normal_head(features.mean(dim=1)), dim=-1)
        return logits, weight_logits, normal
