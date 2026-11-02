"""Train the differentiable Circle-Net reproduction on the bundled HDF5 data.

Example:
  python train_repro.py --epochs 30 --out-dir log/repro

The HDF5 file contains boundary labels and radius codes, but no CAD center GT.
The fitting loss therefore supervises radius only. The supplied data_l2 vector
is used as the circle-plane normal for fitting, not as a network input.
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from repro_model import CircleNetRepro, weighted_circle_fit


RADII = torch.tensor([0.06, 0.05, 0.04, 0.03, 0.02, 0.0125, 0.0075, 0.005])
ROOT = Path(__file__).resolve().parent


class CirclePatches(Dataset):
    """Hold out two hole positions per radius across all 12 scan groups."""

    def __init__(self, split, h5_path=ROOT / "data/circle_data/data.hdf5"):
        if split not in {"train", "val"}:
            raise ValueError(split)
        all_ids = [81 * scan + hole for scan in range(12) for hole in range(80)
                   if ((hole % 10 < 8) == (split == "train"))]
        with h5py.File(h5_path, "r") as h5:
            small = h5["data_l0"][all_ids]
            large = h5["data_l1"][all_ids]
            normals = h5["data_l2"][all_ids]
        self.query = torch.from_numpy(small[:, :, :3].astype(np.float32))
        self.context = torch.from_numpy(large[:, :, :3].astype(np.float32))
        self.labels = torch.from_numpy(small[:, :, 3].astype(np.int64))
        self.radius_codes = torch.from_numpy(small[:, :, 4].max(axis=1).astype(np.int64))
        self.normals = torch.from_numpy(normals.astype(np.float32))
        if (self.radius_codes < 0).any() or (self.radius_codes >= len(RADII)).any():
            raise ValueError("A selected patch has no valid radius code")

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        return (self.query[index], self.context[index], self.labels[index],
                self.radius_codes[index], self.normals[index])


def move_batch(batch, device):
    return tuple(value.to(device, non_blocking=True) for value in batch)


def train_epoch(model, loader, optimizer, class_weights, fit_weight, device):
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for raw_points, context, labels, radius_codes, normals in loader:
        raw_points, context, labels, radius_codes, normals = move_batch(
            (raw_points, context, labels, radius_codes, normals), device)
        logits, weight_logits = model(raw_points, context)
        class_loss = F.cross_entropy(logits.transpose(1, 2), labels, weight=class_weights)

        boundary_probability = logits.softmax(dim=-1)[..., 1]
        fitting_confidence = 0.2 + 0.8 * boundary_probability * weight_logits.sigmoid()
        fitting_weights = (labels == 1).float() * fitting_confidence
        _, radius = weighted_circle_fit(raw_points, fitting_weights, normals)
        ground_truth_radius = RADII.to(device)[radius_codes]
        relative_error = (radius - ground_truth_radius) / ground_truth_radius
        fit_loss = F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), beta=0.2)
        total_loss = class_loss + fit_weight * fit_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        totals += np.array([total_loss.item(), class_loss.item(), fit_loss.item()]) * len(labels)
        count += len(labels)
    return {"loss": totals[0] / count, "class_loss": totals[1] / count,
            "fit_loss": totals[2] / count}


@torch.no_grad()
def validate(model, loader, device, threshold=0.5, class_weights=None, fit_weight=1.0):
    model.eval()
    tp = fp = fn = 0
    oracle_errors = []
    predicted_errors = []
    predicted_fit_count = 0
    total = 0
    losses = np.zeros(3, dtype=np.float64)
    for raw_points, context, labels, radius_codes, normals in loader:
        raw_points, context, labels, radius_codes, normals = move_batch(
            (raw_points, context, labels, radius_codes, normals), device)
        logits, weight_logits = model(raw_points, context)
        probability = logits.softmax(dim=-1)[..., 1]
        predicted = probability >= threshold
        tp += ((predicted) & (labels == 1)).sum().item()
        fp += ((predicted) & (labels == 0)).sum().item()
        fn += ((~predicted) & (labels == 1)).sum().item()
        ground_truth_radius = RADII.to(device)[radius_codes]

        confidence = 0.2 + 0.8 * probability * weight_logits.sigmoid()
        _, oracle_radius = weighted_circle_fit(raw_points, confidence * (labels == 1), normals)
        oracle_errors.extend((oracle_radius - ground_truth_radius).abs().cpu().tolist())
        if class_weights is not None:
            class_loss = F.cross_entropy(logits.transpose(1, 2), labels, weight=class_weights)
            relative_error = (oracle_radius - ground_truth_radius) / ground_truth_radius
            fit_loss = F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), beta=0.2)
            losses += np.array([(class_loss + fit_weight * fit_loss).item(),
                                class_loss.item(), fit_loss.item()]) * len(labels)

        _, predicted_radius = weighted_circle_fit(raw_points, confidence * predicted, normals)
        valid = predicted.sum(dim=1) >= 10
        predicted_fit_count += valid.sum().item()
        predicted_errors.extend((predicted_radius[valid] - ground_truth_radius[valid]).abs().cpu().tolist())
        total += len(labels)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "oracle_radius_mae": float(np.mean(oracle_errors)),
            "oracle_radius_mse": float(np.mean(np.square(oracle_errors))),
            "predicted_radius_mae": float(np.mean(predicted_errors)) if predicted_errors else None,
            "predicted_radius_mse": float(np.mean(np.square(predicted_errors))) if predicted_errors else None,
            "predicted_fit_coverage": predicted_fit_count / total}
    if class_weights is not None:
        result.update({"loss": losses[0] / total, "class_loss": losses[1] / total,
                       "fit_loss": losses[2] / total})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fit-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "log/repro")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.out_dir.mkdir(parents=True, exist_ok=True)

    training = CirclePatches("train")
    validation = CirclePatches("val")
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    negatives = (training.labels == 0).sum().item()
    positives = (training.labels == 1).sum().item()
    class_weights = torch.tensor([1.0, negatives / positives], device=device)
    model = CircleNetRepro().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    metadata = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(device), "train_samples": len(training),
                "val_samples": len(validation), "positive_points": positives,
                "negative_points": negatives, "radius_gt": RADII.tolist(),
                "fit_gt": "radius only; no CAD center GT in bundled HDF5",
                "split": "for each scan group and radius, hole index mod 10 < 8 trains; >= 8 validates",
                "args": vars(args) | {"out_dir": str(args.out_dir),
                                      "resume": str(args.resume) if args.resume else None}}
    (args.out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("torch", "device", "train_samples", "val_samples")}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, class_weights,
                                    args.fit_weight, device)
        val_metrics = validate(model, val_loader, device, class_weights=class_weights,
                               fit_weight=args.fit_weight)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        with (args.out_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, args.out_dir / "last_model.pth")


if __name__ == "__main__":
    main()
