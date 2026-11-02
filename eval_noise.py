"""Evaluate differentiable-fit CircleNet under controlled Gaussian XYZ noise.

Noise sigma is a percentage of each CLEAN l0 patch's bounding-box diagonal.
Both l0 and l1 receive independent isotropic Gaussian noise with that sigma.
The HDF5 file has no CAD center ground truth, so center error is explicitly a
proxy against a clean GT-boundary fit, not the paper's true AD(c).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_point_attention import CircleNetPointAttention
from model_v2 import CircleNetV2
from repro_model import weighted_circle_fit
from train_v2 import CirclePatchesV2, RADII


METRICS = ("precision", "recall", "f1", "ad_c_proxy", "ad_r", "mse_r",
           "fit_coverage", "negative_patch_false_circle_rate")


@torch.inference_mode()
def clean_reference(dataset, batch_size, device):
    """Fit centers from clean GT boundary points and HDF5 normals once."""
    centers = torch.zeros((len(dataset), 3), dtype=dataset.query.dtype)
    radii = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        positive = dataset.radius_codes[start:end] >= 0
        if not positive.any():
            continue
        local = torch.arange(start, end)[positive]
        query = dataset.query[local].to(device)
        weights = (dataset.labels[local] == 1).float().to(device)
        normals = dataset.normals[local].to(device)
        center, radius = weighted_circle_fit(query, weights, normals, ridge=1e-4)
        centers[local] = center.cpu()
        radii.append((radius - RADII.to(device)[dataset.radius_codes[local].to(device)])
                     .abs().cpu())
    return centers, float(torch.cat(radii).mean())


@torch.inference_mode()
def evaluate_one(model, dataset, query, context, reference_centers, batch_size, device):
    tp = fp = fn = covered = positives = negatives = false_circles = 0
    center_errors, radius_errors = [], []
    by_radius = {str(code): {"patches": 0, "tp": 0, "fp": 0, "fn": 0,
                             "center_abs_sum": 0.0, "radius_abs_sum": 0.0,
                             "radius_squared_sum": 0.0, "covered": 0}
                 for code in range(len(RADII))}
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        q = query[start:end].to(device)
        c = context[start:end].to(device)
        labels = dataset.labels[start:end].to(device)
        codes = dataset.radius_codes[start:end].to(device)
        logits, weight_logits, normal = model(q, c)
        probability = logits.softmax(dim=-1)[..., 1]
        predicted = probability >= 0.5
        center, radius, _ = model.fit(q, logits, weight_logits, normal)

        target = labels == 1
        tp += int((predicted & target).sum())
        fp += int((predicted & ~target).sum())
        fn += int((~predicted & target).sum())
        positive = codes >= 0
        positives += int(positive.sum())
        covered += int((predicted[positive].sum(dim=1) >= 10).sum())
        negative = ~positive
        negatives += int(negative.sum())
        false_circles += int((predicted[negative].sum(dim=1) >= 10).sum())
        if positive.any():
            reference = reference_centers[start:end].to(device)[positive]
            c_error = (center[positive] - reference).norm(dim=-1)
            r_error = radius[positive] - RADII.to(device)[codes[positive]]
            center_errors.extend(c_error.cpu().tolist())
            radius_errors.extend(r_error.cpu().tolist())
        for code in range(len(RADII)):
            selected = codes == code
            if not selected.any():
                continue
            item = by_radius[str(code)]
            chosen = predicted[selected]
            truth = target[selected]
            item["patches"] += int(selected.sum())
            item["tp"] += int((chosen & truth).sum())
            item["fp"] += int((chosen & ~truth).sum())
            item["fn"] += int((~chosen & truth).sum())
            item["covered"] += int((chosen.sum(dim=1) >= 10).sum())
            c_error = (center[selected] - reference_centers[start:end].to(device)[selected]).norm(dim=-1)
            r_error = radius[selected] - RADII.to(device)[code]
            item["center_abs_sum"] += float(c_error.sum())
            item["radius_abs_sum"] += float(r_error.abs().sum())
            item["radius_squared_sum"] += float(r_error.square().sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    errors = np.asarray(radius_errors, dtype=np.float64)
    result = {"precision": precision, "recall": recall,
              "f1": 2 * precision * recall / max(precision + recall, 1e-12),
              "ad_c_proxy": float(np.mean(center_errors)),
              "ad_r": float(np.abs(errors).mean()),
              "mse_r": float(np.square(errors).mean()),
              "fit_coverage": covered / max(positives, 1),
              "negative_patch_false_circle_rate": false_circles / max(negatives, 1),
              "tp": tp, "fp": fp, "fn": fn, "positive_patches": positives,
              "negative_patches": negatives}
    result["by_radius"] = {}
    for code, item in by_radius.items():
        n = item["patches"]
        p = item["tp"] / max(item["tp"] + item["fp"], 1)
        r = item["tp"] / max(item["tp"] + item["fn"], 1)
        result["by_radius"][code] = {
            "target_radius": float(RADII[int(code)]), "patches": n,
            "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "ad_c_proxy": item["center_abs_sum"] / max(n, 1),
            "ad_r": item["radius_abs_sum"] / max(n, 1),
            "mse_r": item["radius_squared_sum"] / max(n, 1),
            "fit_coverage": item["covered"] / max(n, 1)}
    return result


def summarize(runs):
    result = {}
    for metric in METRICS:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()),
                          "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    result["by_radius"] = {}
    for code in runs[0]["by_radius"]:
        result["by_radius"][code] = {}
        for metric in ("precision", "recall", "f1", "ad_c_proxy", "ad_r",
                       "mse_r", "fit_coverage"):
            values = np.asarray([run["by_radius"][code][metric] for run in runs])
            result["by_radius"][code][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-percent", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--out", type=Path, default=Path("log/point_attention_balanced/noise_test.json"))
    args = parser.parse_args()
    if any(p <= 0 for p in args.noise_percent) or len(set(args.noise_percent)) != len(args.noise_percent):
        parser.error("noise percentages must be positive and unique")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint["config"].get("model_type", "fit")
    model_classes = {"point_attention": CircleNetPointAttention, "fit": CircleNetV2}
    if model_type not in model_classes:
        parser.error("this experiment expects a point_attention or fit checkpoint")
    model = model_classes[model_type]().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    groups = checkpoint["config"]["groups"]
    dataset = CirclePatchesV2(args.split, groups=groups)
    reference_centers, oracle_radius_mae = clean_reference(dataset, args.batch_size, device)
    diagonal = (dataset.query.amax(dim=1) - dataset.query.amin(dim=1)).norm(dim=-1)
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "model_type": model_type,
        "split": args.split, "scan_groups": groups[args.split],
        "patches": len(dataset), "seeds": args.seeds,
        "noise_definition": "independent isotropic N(0,sigma^2) on l0 and l1 XYZ; sigma = percentage/100 * clean l0 bounding-box diagonal, per patch",
        "noise_labels": "original point identities and boundary labels kept fixed after perturbing XYZ",
        "center_reference": "clean GT-boundary points fitted with HDF5 normal; a proxy, not CAD center ground truth",
        "distance_unit": "HDF5 coordinate unit; no confirmed millimeter conversion",
        "clean_l0_diagonal_median": float(diagonal.median()),
        "clean_oracle_radius_mae": oracle_radius_mae,
        "clean": evaluate_one(model, dataset, dataset.query, dataset.context,
                              reference_centers, args.batch_size, device),
        "noisy": {},
    }
    print(json.dumps({"condition": "clean", "metrics": {k: output["clean"][k] for k in METRICS}}), flush=True)
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        query_noise = torch.randn(dataset.query.shape, generator=generator)
        context_noise = torch.randn(dataset.context.shape, generator=generator)
        for percent in args.noise_percent:
            key = str(percent)
            if key not in output["noisy"]:
                output["noisy"][key] = {"noise_percent": percent,
                                         "sigma_median": float((percent / 100 * diagonal).median()),
                                         "runs": []}
            sigma = (percent / 100 * diagonal)[:, None, None]
            query = dataset.query + sigma * query_noise
            context = dataset.context + sigma * context_noise
            run = evaluate_one(model, dataset, query, context, reference_centers,
                               args.batch_size, device)
            output["noisy"][key]["runs"].append({"seed": seed, **run})
            print(json.dumps({"condition": key, "seed": seed,
                              "metrics": {k: run[k] for k in METRICS}}), flush=True)
    for value in output["noisy"].values():
        value["summary"] = summarize(value["runs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
