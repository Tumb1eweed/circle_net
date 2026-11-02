"""Compare local circle fits on one HDF5 test split and identical noise draws.

The four classical methods share the point-attention model's hard boundary
predictions. Neural fits use their own learned soft masks. This is a local
comparison, not a numerical reproduction of the paper's CAD benchmark.
"""

import argparse
import json
from pathlib import Path

import circle_fit
import numpy as np
import torch

from eval_noise import clean_reference
from model_point_attention import CircleNetPointAttention
from model_v2 import CircleNetV2
from train_v2 import CirclePatchesV2, RADII


METHODS = ("Hyper", "LLS", "RANSAC", "RLTS-approx", "Ours-local", "Ours+Attn")
METRICS = ("ad_c_proxy", "ad_r", "mse_r")


def project(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 10 or not np.isfinite(points).all():
        return None
    origin = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - origin, full_matrices=False)
    if singular[1] < 1e-10:
        return None
    return origin, axes[:2], (points - origin) @ axes[:2].T


def lls(xy):
    a = np.column_stack((2 * xy, np.ones(len(xy))))
    z, _, rank, _ = np.linalg.lstsq(a, np.square(xy).sum(axis=1), rcond=None)
    radius_squared = float(z[2] + z[:2] @ z[:2])
    if rank < 3 or radius_squared <= 0 or not np.isfinite(radius_squared):
        return None
    return z[:2], float(np.sqrt(radius_squared))


def hyper(xy):
    try:
        cx, cy, radius, _ = circle_fit.hyperSVD(xy)
        if not np.isfinite((cx, cy, radius)).all() or radius <= 0:
            return None
        return np.array((cx, cy)), float(radius)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


def circle_from_triplets(xy, rng, trials):
    """Vectorized random minimal circle hypotheses in a fixed PCA plane."""
    ids = np.argsort(rng.random((trials, len(xy))), axis=1)[:, :3]
    a, b, c = (xy[ids[:, j]] for j in range(3))
    u, v = b - a, c - a
    det = 2 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    valid = np.abs(det) > 1e-12
    centers = np.full((trials, 2), np.nan)
    uu, vv = np.square(u).sum(axis=1), np.square(v).sum(axis=1)
    centers[valid, 0] = a[valid, 0] + (uu[valid] * v[valid, 1] - vv[valid] * u[valid, 1]) / det[valid]
    centers[valid, 1] = a[valid, 1] + (vv[valid] * u[valid, 0] - uu[valid] * v[valid, 0]) / det[valid]
    radii = np.linalg.norm(centers - a, axis=1)
    valid &= np.isfinite(radii) & (radii > 0)
    return centers[valid], radii[valid]


def ransac(xy, rng, threshold, trials=256):
    centers, radii = circle_from_triplets(xy, rng, trials)
    if not len(radii):
        return None
    residual = np.abs(np.linalg.norm(xy[None] - centers[:, None], axis=-1) - radii[:, None])
    inliers = residual <= threshold
    counts = inliers.sum(axis=1)
    # Inlier count first; clipped residual breaks ties without using the GT.
    rank = counts - np.minimum(residual, threshold).mean(axis=1) / (threshold * 10)
    best = int(rank.argmax())
    if counts[best] < 3:
        return None
    return lls(xy[inliers[best]])


def rlts_approx(xy, rng, starts=16, keep_fraction=0.7, rounds=6):
    """Repeated trimmed Hyper fit; not the authors' exact RLTS implementation."""
    # Three almost collinear points can define a huge spurious circle whose
    # residual on a short arc is tiny. Reject it using only observed XY extent.
    max_radius = 2.0 * np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))
    initial = [hyper(xy), lls(xy)]
    centers, radii = circle_from_triplets(xy, rng, starts)
    initial += [(center, radius) for center, radius in zip(centers, radii)]
    count = max(4, int(np.ceil(len(xy) * keep_fraction)))
    best_fit, best_cost = None, np.inf
    for fit in initial:
        if fit is None or fit[1] > max_radius:
            continue
        for _ in range(rounds):
            residual = np.abs(np.linalg.norm(xy - fit[0], axis=1) - fit[1])
            selected = np.argpartition(residual, count - 1)[:count]
            updated = hyper(xy[selected])
            if updated is None or updated[1] > max_radius:
                break
            fit = updated
        residual = np.abs(np.linalg.norm(xy - fit[0], axis=1) - fit[1])
        cost = float(np.mean(np.partition(residual, count - 1)[:count] ** 2))
        if cost < best_cost:
            best_fit, best_cost = fit, cost
    return best_fit


def lift(fit, origin, axes):
    if fit is None:
        return None
    center = origin + fit[0] @ axes
    return center, fit[1]


def summarize_errors(errors):
    if not errors:
        return None
    a = np.asarray(errors, dtype=np.float64)
    return {"ad_c_proxy": float(a[:, 0].mean()),
            "ad_r": float(np.abs(a[:, 1]).mean()),
            "mse_r": float(np.square(a[:, 1]).mean())}


def load_model(path, cls, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = cls().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(dataset, query, context, reference, attn, base, sigma, seed, batch_size, device):
    by_method = {name: {} for name in METHODS}
    coverage = {name: 0 for name in METHODS}
    positive = int((dataset.radius_codes >= 0).sum())
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        q, c = query[start:end].to(device), context[start:end].to(device)
        attn_logits, attn_weights, attn_normal = attn(q, c)
        base_logits, base_weights, base_normal = base(q, c)
        attn_center, attn_radius, _ = attn.fit(q, attn_logits, attn_weights, attn_normal)
        base_center, base_radius, _ = base.fit(q, base_logits, base_weights, base_normal)
        masks = (attn_logits.softmax(-1)[..., 1] >= 0.5).cpu().numpy()
        xyz = q.cpu().numpy()
        ac, ar = attn_center.cpu().numpy(), attn_radius.cpu().numpy()
        bc, br = base_center.cpu().numpy(), base_radius.cpu().numpy()
        for j in range(end - start):
            idx = start + j
            code = int(dataset.radius_codes[idx])
            if code < 0:
                continue
            ref = reference[idx].numpy()
            gt_radius = float(RADII[code])
            fits = {"Ours-local": (bc[j], br[j]), "Ours+Attn": (ac[j], ar[j])}
            projection = project(xyz[j, masks[j]])
            if projection is not None:
                origin, axes, xy = projection
                rng = np.random.default_rng(seed * 1000003 + idx)
                fits.update({
                    "Hyper": lift(hyper(xy), origin, axes),
                    "LLS": lift(lls(xy), origin, axes),
                    "RANSAC": lift(ransac(xy, rng, max(2.0 * float(sigma[idx]), 1e-6)), origin, axes),
                    "RLTS-approx": lift(rlts_approx(xy, rng), origin, axes),
                })
            for method, fit in fits.items():
                if fit is None or not np.isfinite(fit[0]).all() or not np.isfinite(fit[1]):
                    continue
                coverage[method] += 1
                by_method[method][idx] = (float(np.linalg.norm(fit[0] - ref)),
                                          float(fit[1] - gt_radius))
    common_ids = sorted(set.intersection(*(set(errors) for errors in by_method.values())))
    return {
        "positive_patches": positive,
        "common_valid_patches": len(common_ids),
        "coverage": coverage,
        "metrics": {name: summarize_errors([errors[idx] for idx in common_ids])
                    for name, errors in by_method.items()},
        "per_method_metrics": {name: summarize_errors(list(errors.values()))
                               for name, errors in by_method.items()},
    }


def aggregate(runs):
    return {
        "common_valid_patches_mean": float(np.mean([run["common_valid_patches"] for run in runs])),
        "coverage_mean": {name: float(np.mean([run["coverage"][name] for run in runs]))
                          for name in METHODS},
        "metrics": {name: {metric: {
            "mean": float(np.mean([run["metrics"][name][metric] for run in runs])),
            "sd": float(np.std([run["metrics"][name][metric] for run in runs], ddof=1))
        } for metric in METRICS} for name in METHODS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-checkpoint", type=Path, default=Path("log/point_attention_balanced_20261102/last_model.pth"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("log/v2_balanced_20261102/last_model.pth"))
    parser.add_argument("--out", type=Path, default=Path("results/all_fits_noise_test.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--noise-percent", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn, attn_ckpt = load_model(args.attn_checkpoint, CircleNetPointAttention, device)
    base, base_ckpt = load_model(args.base_checkpoint, CircleNetV2, device)
    groups = attn_ckpt["config"]["groups"]
    if groups != base_ckpt["config"]["groups"]:
        parser.error("models must use the same scan-group split")
    dataset = CirclePatchesV2("test", groups=groups)
    references, _ = clean_reference(dataset, args.batch_size, device)
    diagonal = (dataset.query.amax(dim=1) - dataset.query.amin(dim=1)).norm(dim=-1)
    output = {
        "attn_checkpoint": str(args.attn_checkpoint.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "test_scan_groups": groups["test"], "patches": len(dataset),
        "positive_patches": int((dataset.radius_codes >= 0).sum()),
        "seeds": args.seeds, "noise_percent": args.noise_percent,
        "distance_unit": "raw HDF5 coordinate; physical conversion unknown",
        "center_reference": "clean GT boundary weighted fit; AD(c) proxy, not CAD center",
        "noise_definition": "independent Gaussian XYZ noise on l0 and l1, sigma = percent/100 * clean l0 bounding-box diagonal",
        "classical_input": "point-attention model predicted boundary points with probability >=0.5, minimum 10 points",
        "geometry_plane": "unweighted PCA on predicted boundary points",
        "Hyper": "circle-fit 0.2.1 hyperSVD",
        "LLS": "linear algebraic least-squares",
        "RANSAC": "256 random 3-point samples; threshold max(2*sigma, 1e-6); LLS inlier refit",
        "RLTS-approx": "16 random 3-point starts plus Hyper/LLS; 70% trimmed residual, 6 Hyper iterations, reject radius above twice projected point bbox diagonal; approximation, not author code",
        "PIE-NET": "unavailable: paper retrained modified PIE-NET; no compatible checkpoint for this HDF5 protocol",
        "Ours-local": "local no-attention CircleNetV2 checkpoint; not authors' original Ours",
        "Ours+Attn": "local point-attention CircleNetPointAttention checkpoint",
        "scoring": "all reported table metrics use each run's common valid positive patches across six measured methods",
        "noisy": {},
    }
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        q_noise = torch.randn(dataset.query.shape, generator=generator)
        c_noise = torch.randn(dataset.context.shape, generator=generator)
        for percent in args.noise_percent:
            key = str(percent)
            sigma = percent / 100 * diagonal
            run = evaluate(dataset, dataset.query + sigma[:, None, None] * q_noise,
                           dataset.context + sigma[:, None, None] * c_noise,
                           references, attn, base, sigma, seed, args.batch_size, device)
            output["noisy"].setdefault(key, {"runs": [], "sigma_median": float(sigma.median())})["runs"].append({"seed": seed, **run})
            print(json.dumps({"noise_percent": percent, "seed": seed,
                              "common": run["common_valid_patches"],
                              "metrics": run["metrics"]}), flush=True)
    for value in output["noisy"].values():
        value["summary"] = aggregate(value["runs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
