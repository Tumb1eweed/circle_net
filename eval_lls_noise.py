"""Compare ordinary least-squares circle fitting with the learned fit.

Both fits use the same point-attention boundary predictions, HDF5 patches and
Gaussian perturbations as eval_noise.py. LLS estimates a plane with PCA and
solves x^2+y^2 = 2*a*x + 2*b*y + c on hard predicted boundary points.
This is a local baseline, not a rerun of the paper's full-cloud benchmark.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_noise import clean_reference
from model_point_attention import CircleNetPointAttention
from train_v2 import BALANCED_GROUPS, CirclePatchesV2, RADII


def fit_lls_circle(points):
    """Return the 3D circle center and radius, or None for a degenerate set."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 10:
        return None
    origin = points.mean(axis=0)
    centered = points - origin
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[1] < 1e-10:
        return None
    xy = centered @ axes[:2].T
    design = np.column_stack((2 * xy, np.ones(len(xy))))
    solution, _, rank, _ = np.linalg.lstsq(design, np.square(xy).sum(axis=1), rcond=None)
    if rank < 3:
        return None
    radius_squared = solution[2] + np.square(solution[:2]).sum()
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        return None
    center = origin + solution[0] * axes[0] + solution[1] * axes[1]
    return center, float(np.sqrt(radius_squared))


def metric_summary(center_errors, radius_errors, true_radii):
    center_errors = np.asarray(center_errors, dtype=np.float64)
    radius_errors = np.asarray(radius_errors, dtype=np.float64)
    true_radii = np.asarray(true_radii, dtype=np.float64)
    return {
        "circles": len(radius_errors),
        "ad_c_proxy": float(center_errors.mean()),
        "ad_r": float(np.abs(radius_errors).mean()),
        "mse_r": float(np.square(radius_errors).mean()),
        "relative_radius_mae": float((np.abs(radius_errors) / true_radii).mean()),
    }


@torch.inference_mode()
def evaluate(model, dataset, query, context, reference_centers, batch_size, device):
    lls_center, lls_radius, learned_center, learned_radius, true_radii = [], [], [], [], []
    positive_count = 0
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        q = query[start:end].to(device)
        c = context[start:end].to(device)
        logits, weight_logits, normal = model(q, c)
        predicted = logits.softmax(dim=-1)[..., 1] >= 0.5
        fitted_center, fitted_radius, _ = model.fit(q, logits, weight_logits, normal)
        q_np = q.cpu().numpy()
        mask_np = predicted.cpu().numpy()
        fitted_center = fitted_center.cpu().numpy()
        fitted_radius = fitted_radius.cpu().numpy()
        for local in range(end - start):
            global_index = start + local
            code = int(dataset.radius_codes[global_index])
            if code < 0:
                continue
            positive_count += 1
            lls = fit_lls_circle(q_np[local, mask_np[local]])
            if lls is None:
                continue
            target_radius = float(RADII[code])
            reference = reference_centers[global_index].numpy()
            lls_center.append(float(np.linalg.norm(lls[0] - reference)))
            lls_radius.append(lls[1] - target_radius)
            learned_center.append(float(np.linalg.norm(fitted_center[local] - reference)))
            learned_radius.append(float(fitted_radius[local] - target_radius))
            true_radii.append(target_radius)
    if not true_radii:
        raise RuntimeError("No circles had at least 10 predicted boundary points")
    return {
        "positive_circles": positive_count,
        "common_valid_circles": len(true_radii),
        "coverage": len(true_radii) / positive_count,
        "lls": metric_summary(lls_center, lls_radius, true_radii),
        "learned_fit": metric_summary(learned_center, learned_radius, true_radii),
    }


def summarize(runs):
    result = {"coverage": {}, "lls": {}, "learned_fit": {}}
    for name in ("coverage",):
        values = np.asarray([run[name] for run in runs])
        result[name] = {"mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    for method in ("lls", "learned_fit"):
        for metric in ("ad_c_proxy", "ad_r", "mse_r", "relative_radius_mae"):
            values = np.asarray([run[method][metric] for run in runs])
            result[method][metric] = {"mean": float(values.mean()),
                                      "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-percent", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--out", type=Path, default=Path("log/point_attention_balanced_20261102/lls_noise_test.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["config"].get("model_type") != "point_attention":
        parser.error("expected a point_attention checkpoint")
    model = CircleNetPointAttention().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    groups = checkpoint["config"].get("groups", BALANCED_GROUPS)
    dataset = CirclePatchesV2("test", groups=groups)
    references, _ = clean_reference(dataset, args.batch_size, device)
    diagonal = (dataset.query.amax(dim=1) - dataset.query.amin(dim=1)).norm(dim=-1)

    result = {
        "checkpoint": str(args.checkpoint),
        "split": "test", "scan_groups": groups["test"],
        "seeds": args.seeds, "noise_percent": args.noise_percent,
        "distance_unit": "HDF5 coordinate unit; physical scale is unverified",
        "lls_definition": "PCA plane + unweighted algebraic least squares on >=10 hard predicted boundary points",
        "comparison": "both methods scored on the same LLS-valid positive patches",
        "center_reference": "clean GT-boundary fit proxy, not CAD center",
        "noise_definition": "sigma = noise_percent/100 * clean l0 bounding-box diagonal, added independently to l0/l1 XYZ",
        "clean": evaluate(model, dataset, dataset.query, dataset.context,
                          references, args.batch_size, device),
        "noisy": {},
    }
    print(json.dumps({"condition": "clean", "result": result["clean"]}), flush=True)
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        query_noise = torch.randn(dataset.query.shape, generator=generator)
        context_noise = torch.randn(dataset.context.shape, generator=generator)
        for percent in args.noise_percent:
            key = str(percent)
            sigma = (percent / 100 * diagonal)[:, None, None]
            run = evaluate(model, dataset,
                           dataset.query + sigma * query_noise,
                           dataset.context + sigma * context_noise,
                           references, args.batch_size, device)
            result["noisy"].setdefault(key, {"runs": []})["runs"].append({"seed": seed, **run})
            print(json.dumps({"condition": key, "seed": seed, "result": run}), flush=True)
    for value in result["noisy"].values():
        value["summary"] = summarize(value["runs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
