"""Compact Circle-Net variant with a differentiable weighted circle fit."""

import torch
from torch import nn
from torch.nn import functional as F


class NeighborhoodEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, width, 1), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
        )

    def forward(self, neighborhoods):
        batch, points, neighbors, _ = neighborhoods.shape
        x = neighborhoods.reshape(batch * points, neighbors, 3).transpose(1, 2)
        return self.layers(x).amax(dim=-1).reshape(batch, points, -1)


class CircleNetRepro(nn.Module):
    """Return per-point class logits and per-point fitting-weight logits."""

    def __init__(self, width=128, local_k=16, global_k=128):
        super().__init__()
        self.local_k = local_k
        self.global_k = global_k
        self.local_encoder = NeighborhoodEncoder(width)
        self.global_encoder = NeighborhoodEncoder(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 2))
        self.weight_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, query, context, return_features=False):
        if context.shape[1] < self.global_k:
            raise ValueError("The context must contain at least global_k points")
        origin = context.mean(dim=1, keepdim=True)
        scale = (context - origin).norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
        query = (query - origin) / scale.unsqueeze(-1)
        context = (context - origin) / scale.unsqueeze(-1)

        indices = torch.cdist(query, context).topk(self.global_k, largest=False).indices
        expanded = context[:, None].expand(-1, query.shape[1], -1, -1)
        neighbors = torch.gather(expanded, 2, indices[..., None].expand(-1, -1, -1, 3))
        relative = neighbors - query[:, :, None]
        radius = relative.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1e-6)
        relative = relative / radius[..., None]

        local = self.local_encoder(relative[:, :, :self.local_k])
        global_feature = self.global_encoder(relative)
        tokens = torch.stack((local, global_feature), dim=2)
        batch, points, _, width = tokens.shape
        feature = self.fusion(tokens.reshape(batch * points, 2, width)).mean(dim=1)
        feature = feature.reshape(batch, points, width)
        logits = self.classifier(feature)
        weight_logits = self.weight_head(feature).squeeze(-1)
        if return_features:
            return logits, weight_logits, feature
        return logits, weight_logits


def weighted_circle_fit(points, weights, normals, ridge=1e-5):
    """Fit one 3D circle per batch item; gradients reach the point weights.

    The supplied normal fixes the circle plane, while the center and radius are
    solved in that plane. The mainline model predicts this normal per patch.
    """
    eps = 1e-8
    fallback = torch.tensor([0., 1., 0.], device=points.device, dtype=points.dtype)
    normals = torch.where(normals.norm(dim=-1, keepdim=True) > eps, normals, fallback)
    normal = F.normalize(normals, dim=-1)
    z_axis = torch.tensor([0., 0., 1.], device=points.device, dtype=points.dtype)
    x_axis = torch.tensor([1., 0., 0.], device=points.device, dtype=points.dtype)
    reference = torch.where(normal[:, 2:3].abs() < 0.9, z_axis, x_axis)
    u = F.normalize(torch.linalg.cross(normal, reference), dim=-1)
    v = torch.linalg.cross(normal, u)

    weights = weights.clamp_min(0)
    normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    origin = (normalized_weights[..., None] * points).sum(dim=1)
    offsets = points - origin[:, None]
    xy = torch.stack(((offsets * u[:, None]).sum(-1),
                      (offsets * v[:, None]).sum(-1)), dim=-1)
    spread = ((normalized_weights[..., None] * xy.square()).sum(dim=(1, 2)))
    scale = spread.sqrt().clamp_min(1e-5)
    xy = xy / scale[:, None, None]

    design = torch.cat((2 * xy, torch.ones_like(xy[..., :1])), dim=-1)
    squared_distance = xy.square().sum(-1, keepdim=True)
    weighted_design = design * normalized_weights[..., None]
    matrix = design.transpose(1, 2) @ weighted_design
    eye = torch.eye(3, device=points.device, dtype=points.dtype)[None]
    rhs = design.transpose(1, 2) @ (normalized_weights[..., None] * squared_distance)
    solution = torch.linalg.solve(matrix + ridge * eye, rhs).squeeze(-1)
    center = origin + scale[:, None] * (solution[:, :1] * u + solution[:, 1:2] * v)
    radius_squared = solution[:, 2] + solution[:, :2].square().sum(-1)
    radius = scale * radius_squared.clamp_min(eps).sqrt()
    return center, radius
