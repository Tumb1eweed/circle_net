"""Compare the archived paper checkpoint with point-attention checkpoints.

This is a diagnostic checkpoint comparison. The paper checkpoint's original
random split overlaps the balanced group-held-out test split; results are
reported separately for paper-trained and paper-unseen patches.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy import spatial

from eval_noise import clean_reference
from model_point_attention import CircleNetPointAttention
from repro_model import weighted_circle_fit
from train_v2 import CirclePatchesV2, BALANCED_GROUPS, RADII


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paper_split_ids():
    ids = np.asarray([81 * group + hole for group in range(12) for hole in range(50)])
    order = np.arange(len(ids))
    np.random.seed(1111)
    np.random.shuffle(order)
    train_end = int(len(ids) * 0.9 + 1)
    return set(ids[order[:train_end]].tolist()), set(ids[order[train_end:]].tolist())


def paper_input(query, context, patch_id, paper_data, paper_train):
    query, context = paper_data.pc_normalize(query[None], context[None])
    query, context = query[0], context[0]
    distances, indices = spatial.cKDTree(context).query(query, k=500)
    rng = np.random.default_rng(10_000 + patch_id)
    subset = np.concatenate(([0], rng.permutation(np.arange(1, 500))[:127]))
    scale = distances[:, -1].reshape(-1, 1, 1).clip(min=1e-8)
    local = context[indices[:, :16]] / scale
    broad = context[indices[:, subset]] / scale
    broad, local = paper_train.pca_normalize(broad, local)
    return (torch.from_numpy(local.astype(np.float32)).transpose(1, 2),
            torch.from_numpy(broad.astype(np.float32)).transpose(1, 2))


def evaluate_paper(dataset, paper_root, device, limit=None):
    paper_model = load_module("paper_seg_model2", paper_root / "models/seg_model2.py")
    paper_data = load_module("paper_circle_loader", paper_root / "data_utils/CircleDataLoader.py")
    paper_train = load_module("paper_train_partseg2", paper_root / "train_partseg2.py")
    checkpoint = torch.load(paper_root / "log/part_seg/seg_model2/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)
    model = paper_model.get_model(2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = {}
    count = min(len(dataset), limit) if limit else len(dataset)
    for index in range(count):
        patch_id = dataset.ids[index]
        query = dataset.query[index].numpy()
        context = dataset.context[index].numpy()
        local, broad = paper_input(query, context, patch_id, paper_data, paper_train)
        with torch.inference_mode():
            log_probability, weight_logits = model(local.to(device), broad.to(device))
            predicted = log_probability.argmax(dim=-1).cpu().numpy().astype(bool)
            weight_logits = weight_logits.squeeze(-1)
        fit = None
        if predicted.sum() >= 10 and int(dataset.radius_codes[index]) >= 0:
            weights = torch.softmax(weight_logits[predicted], dim=0)
            _, center, radius = paper_model.circle_segmentation(query[predicted], weights)
            center = np.asarray(center, dtype=np.float64)
            radius = float(radius)
            if np.isfinite(center).all() and np.isfinite(radius):
                fit = {"center": center.tolist(), "radius": radius}
        predictions[str(patch_id)] = {"predicted": predicted.astype(int).tolist(), "fit": fit}
        if (index + 1) % 20 == 0 or index + 1 == count:
            print(f"paper checkpoint: {index + 1}/{count}", flush=True)
    return predictions, checkpoint["epoch"]


def evaluate_attention(dataset, checkpoint_path, device, limit=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"]["groups"] != BALANCED_GROUPS:
        raise ValueError("Point-attention checkpoint uses a different split")
    model = CircleNetPointAttention().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = {}
    count = min(len(dataset), limit) if limit else len(dataset)
    with torch.inference_mode():
        for start in range(0, count, 8):
            end = min(start + 8, count)
            query = dataset.query[start:end].to(device)
            context = dataset.context[start:end].to(device)
            logits, weights, normal = model(query, context)
            center, radius, _ = model.fit(query, logits, weights, normal)
            probability = logits.softmax(dim=-1)[..., 1]
            predicted_mask = logits.argmax(dim=-1) == 1
            confidence = probability.square() * (0.05 + 0.95 * weights.sigmoid())
            hard_center, hard_radius = weighted_circle_fit(
                query, confidence * predicted_mask, normal, ridge=1e-4)
            predicted = predicted_mask.cpu().numpy()
            for j, index in enumerate(range(start, end)):
                predictions[str(dataset.ids[index])] = {
                    "predicted": predicted[j].astype(int).tolist(),
                    "fit": {"center": center[j].cpu().tolist(),
                            "radius": float(radius[j])},
                    "hard_fit": ({"center": hard_center[j].cpu().tolist(),
                                  "radius": float(hard_radius[j])}
                                 if predicted[j].sum() >= 10 else None),
                }
    return predictions


def metrics(dataset, predictions, references, ids, fit_key="fit"):
    tp = fp = fn = covered = positive_count = 0
    center_errors, radius_errors = [], []
    for index, patch_id in enumerate(dataset.ids):
        if patch_id not in ids or str(patch_id) not in predictions:
            continue
        output = predictions[str(patch_id)]
        guess = np.asarray(output["predicted"], dtype=bool)
        truth = dataset.labels[index].numpy() == 1
        tp += int((guess & truth).sum())
        fp += int((guess & ~truth).sum())
        fn += int((~guess & truth).sum())
        code = int(dataset.radius_codes[index])
        if code < 0:
            continue
        positive_count += 1
        fit = output[fit_key]
        if fit is None:
            continue
        covered += 1
        center_errors.append(float(np.linalg.norm(np.asarray(fit["center"]) - references[index])))
        radius_errors.append(float(fit["radius"] - RADII[code]))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"patches": len(ids & set(dataset.ids)), "positive_patches": positive_count,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "fit_count": covered, "fit_coverage": covered / max(positive_count, 1),
            "ad_c_proxy": float(np.mean(center_errors)) if center_errors else None,
            "ad_r": float(np.abs(radius_errors).mean()) if radius_errors else None,
            "mse_r": float(np.square(radius_errors).mean()) if radius_errors else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "circle_net_paper_baseline")
    parser.add_argument("--attention-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CirclePatchesV2("test", groups=BALANCED_GROUPS)
    references, _ = clean_reference(dataset, 8, device)
    references = references.numpy()
    paper_train_ids, paper_heldout_ids = paper_split_ids()
    groups = {
        "all_balanced_test": set(dataset.ids),
        "paper_train_overlap": set(dataset.ids) & paper_train_ids,
        "paper_original_holdout": set(dataset.ids) & paper_heldout_ids,
        "paper_never_selected_holes_50_80": {i for i in dataset.ids if i % 81 >= 50},
        "paper_unseen_union": set(dataset.ids) - paper_train_ids,
    }
    paper_predictions, paper_epoch = evaluate_paper(dataset, args.paper_root, device, args.limit)
    attention_predictions = {
        str(path): evaluate_attention(dataset, path, device, args.limit)
        for path in args.attention_checkpoints}
    available = set(map(int, paper_predictions))
    output = {
        "paper_checkpoint": str((args.paper_root / "log/part_seg/seg_model2/checkpoints/best_model.pth").resolve()),
        "paper_checkpoint_epoch_zero_based": paper_epoch,
        "paper_split": "random 90/10 of holes 0..49 in all 12 scan groups",
        "attention_split": BALANCED_GROUPS,
        "preprocessing": "paper: original centering/scaling, KNN 16/128 and PCA; deterministic 128-neighbor sample per patch; all 500 query points without original stochastic resampling",
        "center_reference": "clean GT-boundary fit proxy, not CAD center",
        "distance_unit": "raw HDF5 coordinate",
        "groups": {},
    }
    for name, ids in groups.items():
        available_ids = ids & available
        common_fit_ids = {patch_id for patch_id in available_ids
                          if paper_predictions[str(patch_id)]["fit"] is not None}
        group = output["groups"][name] = {
            "ids": sorted(available_ids),
            "paper": metrics(dataset, paper_predictions, references, available_ids),
            "point_attention": {},
        }
        for path, predictions in attention_predictions.items():
            group["point_attention"][path] = {
                "soft_all": metrics(dataset, predictions, references, available_ids),
                "hard_all": metrics(dataset, predictions, references, available_ids,
                                    fit_key="hard_fit"),
                "soft_on_paper_valid": metrics(dataset, predictions, references, common_fit_ids),
                "hard_on_paper_valid": metrics(dataset, predictions, references, common_fit_ids,
                                               fit_key="hard_fit"),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
