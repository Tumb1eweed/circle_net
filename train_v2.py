"""Train Circle-Net v2 with group-held-out data and predicted-mask fitting."""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model_v2 import CircleNetV2
from model_direct import CircleNetDirect
from model_point_attention import CircleNetPointAttention
from repro_model import weighted_circle_fit
from train_repro import RADII, ROOT


GROUPS = {"train": (0, 1, 3, 4, 6, 7, 9, 10), "val": (2, 5), "test": (8, 11)}
BALANCED_GROUPS = {"train": (0, 1, 2, 3, 4, 5, 6, 7), "val": (8, 9), "test": (10, 11)}
SPLIT_PROTOCOLS = {"stress": GROUPS, "balanced": BALANCED_GROUPS}


class CirclePatchesV2(Dataset):
    """Use whole small patches, including one negative patch per scan group."""

    def __init__(self, split, h5_path=ROOT / "data/circle_data/data.hdf5", groups=GROUPS):
        self.split = split
        self.ids = [81 * group + hole for group in groups[split] for hole in range(81)]
        with h5py.File(h5_path, "r") as h5:
            small = h5["data_l0"][self.ids]
            large = h5["data_l1"][self.ids]
            normals = h5["data_l2"][self.ids]
        self.query = torch.from_numpy(small[..., :3].astype(np.float32))
        self.context = torch.from_numpy(large[..., :3].astype(np.float32))
        self.labels = torch.from_numpy(small[..., 3].astype(np.int64))
        self.radius_codes = torch.from_numpy(small[..., 4].max(axis=1).astype(np.int64))
        self.normals = torch.from_numpy(normals.astype(np.float32))
        positive = (self.labels == 1).any(dim=1)
        if not torch.equal(positive, self.radius_codes >= 0):
            raise ValueError("Radius code and positive labels disagree")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (self.query[index], self.context[index], self.labels[index],
                self.radius_codes[index], self.normals[index])


def losses_and_predictions(model, batch, positive_weight, teacher_ratio=0.0):
    query, context, labels, radius_codes, normal_gt = batch
    if isinstance(model, CircleNetDirect):
        logits, weight_logits, normal, center, radius = model(
            query, context, teacher_mask=labels, teacher_ratio=teacher_ratio)
    else:
        logits, weight_logits, normal = model(query, context)
    probability = logits.softmax(dim=-1)[..., 1]
    weight = torch.tensor([1.0, positive_weight], device=query.device)
    ce = F.cross_entropy(logits.transpose(1, 2), labels, weight=weight)
    target = labels.float()
    dice = 1 - (2 * (probability * target).sum(dim=1) + 1) / (
        probability.sum(dim=1) + target.sum(dim=1) + 1)
    class_loss = ce + 0.25 * dice.mean()

    positive = radius_codes >= 0
    if not isinstance(model, CircleNetDirect):
        center = query.new_zeros((len(query), 3))
        radius = query.new_zeros(len(query))
    if positive.any():
        if not isinstance(model, CircleNetDirect):
            positive_indices = positive.nonzero(as_tuple=False).flatten()
            fitted_center, fitted_radius, _ = model.fit(
                query[positive], logits[positive], weight_logits[positive], normal[positive],
                teacher_mask=labels[positive], teacher_ratio=teacher_ratio)
            center = center.index_copy(0, positive_indices, fitted_center)
            radius = radius.index_copy(0, positive_indices, fitted_radius)
        target_radius = RADII.to(query.device)[radius_codes[positive]]
        log_radius_error = torch.log(radius[positive].clamp_min(1e-6) / target_radius)
        radius_loss = F.smooth_l1_loss(log_radius_error, torch.zeros_like(log_radius_error), beta=0.2)

        gt_normal = F.normalize(normal_gt[positive], dim=-1)
        normal_loss = (1 - (normal[positive] * gt_normal).sum(dim=-1).abs()).mean()
        offsets = query[positive] - center[positive, None]
        plane_distance = (offsets * normal[positive, None]).sum(dim=-1)
        radial_distance = (offsets.square().sum(dim=-1) - plane_distance.square()).clamp_min(1e-10).sqrt()
        radial_error = (radial_distance - radius[positive, None]) / target_radius[:, None]
        boundary = labels[positive] == 1
        residual_loss = F.smooth_l1_loss(radial_error[boundary],
                                         torch.zeros_like(radial_error[boundary]), beta=0.2)
    else:
        zero = logits.sum() * 0
        radius_loss = normal_loss = residual_loss = zero
    total_loss = class_loss + 0.5 * radius_loss + 0.1 * normal_loss + 0.05 * residual_loss
    parts = {"loss": total_loss, "class_loss": class_loss, "radius_loss": radius_loss,
             "normal_loss": normal_loss, "residual_loss": residual_loss}
    return parts, (probability, weight_logits, normal, center, radius)


def move_batch(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def train_epoch(model, loader, optimizer, positive_weight, device, teacher_ratio):
    model.train()
    sums = {name: 0.0 for name in ("loss", "class_loss", "radius_loss", "normal_loss", "residual_loss")}
    total = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        losses, _ = losses_and_predictions(model, batch, positive_weight, teacher_ratio)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for name, value in losses.items():
            sums[name] += value.item() * len(batch[0])
        total += len(batch[0])
    return {name: value / total for name, value in sums.items()}


@torch.no_grad()
def evaluate(model, loader, positive_weight, device):
    model.eval()
    sums = {name: 0.0 for name in ("loss", "class_loss", "radius_loss", "normal_loss", "residual_loss")}
    total = tp = fp = fn = circle_fp = negatives = 0
    soft_errors, hard_errors, oracle_errors, proxy_center_errors = [], [], [], []
    hard_valid = 0
    positive_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        query, _, labels, radius_codes, normal_gt = batch
        losses, (probability, weight_logits, normal, center, radius) = losses_and_predictions(
            model, batch, positive_weight)
        for name, value in losses.items():
            sums[name] += value.item() * len(query)
        total += len(query)
        predicted = probability >= 0.5
        tp += (predicted & (labels == 1)).sum().item()
        fp += (predicted & (labels == 0)).sum().item()
        fn += ((~predicted) & (labels == 1)).sum().item()
        positive = radius_codes >= 0
        positive_count += positive.sum().item()
        negatives += (~positive).sum().item()
        circle_fp += ((~positive) & (predicted.sum(dim=1) >= 10)).sum().item()
        if positive.any():
            target_radius = RADII.to(device)[radius_codes[positive]]
            soft_errors.extend((radius[positive] - target_radius).abs().cpu().tolist())
            confidence = probability.square() * (0.05 + 0.95 * weight_logits.sigmoid())
            _, hard_radius = weighted_circle_fit(
                query[positive], (confidence * predicted)[positive], normal[positive], ridge=1e-4)
            valid = predicted[positive].sum(dim=1) >= 10
            hard_valid += valid.sum().item()
            if valid.any():
                hard_errors.extend((hard_radius[valid] - target_radius[valid]).abs().cpu().tolist())
            oracle_center, oracle_radius = weighted_circle_fit(
                query[positive], (labels[positive] == 1).float(), normal_gt[positive], ridge=1e-4)
            oracle_errors.extend((oracle_radius - target_radius).abs().cpu().tolist())
            proxy_center_errors.extend((center[positive] - oracle_center).norm(dim=-1).cpu().tolist())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {name: value / total for name, value in sums.items()}
    result.update({"precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                   "soft_radius_mae": float(np.mean(soft_errors)),
                   "soft_radius_mse": float(np.mean(np.square(soft_errors))),
                   "hard_radius_mae": float(np.mean(hard_errors)) if hard_errors else None,
                   "hard_radius_mse": float(np.mean(np.square(hard_errors))) if hard_errors else None,
                   "oracle_radius_mae": float(np.mean(oracle_errors)),
                   "oracle_radius_mse": float(np.mean(np.square(oracle_errors))),
                   "proxy_center_distance_mae": float(np.mean(proxy_center_errors)),
                   "hard_fit_coverage": hard_valid / max(positive_count, 1),
                   "negative_patch_false_circle_rate": circle_fp / max(negatives, 1)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-protocol", choices=SPLIT_PROTOCOLS, default="balanced")
    parser.add_argument("--model", choices=("fit", "direct", "point_attention"),
                        default="point_attention", help="model variant; point_attention is the mainline default")
    parser.add_argument("--out-dir", type=Path, help="output directory (default: log/<model>)")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = ROOT / "log" / args.model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = SPLIT_PROTOCOLS[args.split_protocol]
    training = CirclePatchesV2("train", groups=groups)
    validation = CirclePatchesV2("val", groups=groups)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              pin_memory=device.type == "cuda", generator=shuffle_generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            pin_memory=device.type == "cuda")
    positives = int((training.labels == 1).sum())
    negatives = int((training.labels == 0).sum())
    positive_weight = (negatives / positives) ** 0.5
    model_classes = {"fit": CircleNetV2, "direct": CircleNetDirect,
                     "point_attention": CircleNetPointAttention}
    model = model_classes[args.model]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    metadata = {"args": vars(args) | {"out_dir": str(args.out_dir)},
                "torch": torch.__version__, "device": str(device),
                "groups": groups, "train_patches": len(training), "val_patches": len(validation),
                "test_patches": len(CirclePatchesV2("test", groups=groups)),
                "positive_weight": positive_weight,
                "model_type": args.model,
                "fit_input": ("direct neural center/radius prediction from predicted attention"
                              if args.model == "direct" else
                              "predicted soft mask after teacher warmup; no GT mask in final fit"),
                "normal_input": "learned patch normal; HDF5 normal only for auxiliary training loss"}
    (args.out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "train": len(training), "val": len(validation),
                      "positive_weight": positive_weight}), flush=True)
    for epoch in range(args.epochs):
        teacher_ratio = max(0.0, 1.0 - epoch / max(args.warmup_epochs, 1))
        train_metrics = train_epoch(model, train_loader, optimizer, positive_weight,
                                    device, teacher_ratio)
        val_metrics = evaluate(model, val_loader, positive_weight, device)
        record = {"epoch": epoch + 1, "teacher_ratio": teacher_ratio,
                  "train": train_metrics, "val": val_metrics}
        with (args.out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": metadata},
                   args.out_dir / "last_model.pth")


if __name__ == "__main__":
    main()
