"""Evaluate Circle-Net v2 on a held-out scan-group split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_v2 import CircleNetV2
from model_direct import CircleNetDirect
from model_point_attention import CircleNetPointAttention
from train_v2 import CirclePatchesV2, RADII, evaluate, losses_and_predictions, move_batch


@torch.no_grad()
def evaluate_by_radius(model, loader, positive_weight, device):
    model.eval()
    totals = [{"patches": 0, "tp": 0, "fp": 0, "fn": 0,
               "radius_abs_error": 0.0, "radius_sq_error": 0.0,
               "covered": 0} for _ in range(len(RADII))]
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        _, _, labels, radius_codes, _ = batch
        _, (probability, _, _, _, radius) = losses_and_predictions(
            model, batch, positive_weight)
        predicted = probability >= 0.5
        for code, item in enumerate(totals):
            selected = radius_codes == code
            if not selected.any():
                continue
            target = labels[selected] == 1
            guess = predicted[selected]
            item["patches"] += selected.sum().item()
            item["tp"] += (guess & target).sum().item()
            item["fp"] += (guess & ~target).sum().item()
            item["fn"] += (~guess & target).sum().item()
            item["radius_abs_error"] += (radius[selected] - RADII[code].to(device)).abs().sum().item()
            item["radius_sq_error"] += (radius[selected] - RADII[code].to(device)).square().sum().item()
            item["covered"] += (guess.sum(dim=1) >= 10).sum().item()
    result = {}
    for code, item in enumerate(totals):
        p = item["tp"] / max(item["tp"] + item["fp"], 1)
        r = item["tp"] / max(item["tp"] + item["fn"], 1)
        result[str(code)] = {"target_radius": float(RADII[code]), "patches": item["patches"],
                             "precision": p, "recall": r,
                             "f1": 2 * p * r / max(p + r, 1e-12),
                             "radius_mae": item["radius_abs_error"] / max(item["patches"], 1),
                             "radius_mse": item["radius_sq_error"] / max(item["patches"], 1),
                             "fit_coverage": item["covered"] / max(item["patches"], 1)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "log/point_attention/last_model.pth")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--by-radius", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint["config"].get("model_type", "fit")
    model_classes = {"fit": CircleNetV2, "direct": CircleNetDirect,
                     "point_attention": CircleNetPointAttention}
    model = model_classes[model_type]().to(device)
    model.load_state_dict(checkpoint["model"])
    groups = checkpoint["config"]["groups"]
    dataset = CirclePatchesV2(args.split, groups=groups)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "epoch": checkpoint["epoch"] + 1,
              "split": args.split,
              "scan_groups": groups[args.split],
              "patches": len(dataset),
              "metrics": evaluate(model, loader, checkpoint["config"]["positive_weight"], device)}
    if args.by_radius:
        result["by_radius"] = evaluate_by_radius(
            model, loader, checkpoint["config"]["positive_weight"], device)
    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
