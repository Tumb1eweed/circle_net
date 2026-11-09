"""Paired added-noise comparison: archived paper checkpoint versus point attention.

The paper model is evaluated on patches it never trained on, with its original
11 validation patches reported separately. All models see identical perturbed
coordinates. Radius error is conditional on a valid paper-style hard fit.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from eval_noise import clean_reference
from eval_paper_baseline import load_module, metrics, paper_input, paper_split_ids
from model_point_attention import CircleNetPointAttention
from repro_model import weighted_circle_fit
from train_v2 import BALANCED_GROUPS, CirclePatchesV2


def paper_predict(dataset, model, paper_model, paper_data, paper_train, device):
    predictions = {}
    for index, patch_id in enumerate(dataset.ids):
        query = dataset.query[index].numpy()
        context = dataset.context[index].numpy()
        local, broad = paper_input(query, context, patch_id, paper_data, paper_train)
        with torch.inference_mode():
            scores, logits = model(local.to(device), broad.to(device))
            predicted = scores.argmax(dim=-1).cpu().numpy().astype(bool)
        fit = None
        if predicted.sum() >= 10 and int(dataset.radius_codes[index]) >= 0:
            try:
                weights = torch.softmax(logits.squeeze(-1)[predicted], dim=0)
                _, center, radius = paper_model.circle_segmentation(query[predicted], weights)
                center = np.asarray(center, dtype=np.float64)
                radius = float(radius)
                if np.isfinite(center).all() and np.isfinite(radius):
                    fit = {"center": center.tolist(), "radius": radius}
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass
        predictions[str(patch_id)] = {"predicted": predicted.astype(int).tolist(),
                                      "fit": fit}
    return predictions


@torch.inference_mode()
def attention_predict(dataset, model, device):
    predictions = {}
    for start in range(0, len(dataset), 8):
        end = min(start + 8, len(dataset))
        query = dataset.query[start:end].to(device)
        context = dataset.context[start:end].to(device)
        logits, weights, normal = model(query, context)
        centers, radii, _ = model.fit(query, logits, weights, normal)
        probability = logits.softmax(dim=-1)[..., 1]
        predicted = probability >= 0.5
        confidence = probability.square() * (0.05 + 0.95 * weights.sigmoid())
        hard_centers, hard_radii = weighted_circle_fit(
            query, confidence * predicted, normal, ridge=1e-4)
        predicted = predicted.cpu().numpy()
        for j, index in enumerate(range(start, end)):
            predictions[str(dataset.ids[index])] = {
                "predicted": predicted[j].astype(int).tolist(),
                "fit": {"center": centers[j].cpu().tolist(),
                        "radius": float(radii[j])},
                "hard_fit": ({"center": hard_centers[j].cpu().tolist(),
                              "radius": float(hard_radii[j])}
                             if predicted[j].sum() >= 10 else None),
            }
    return predictions


def select_dataset(dataset, indices, query, context):
    selected = copy.copy(dataset)
    selected.ids = [dataset.ids[i] for i in indices]
    selected.query = query[indices]
    selected.context = context[indices]
    selected.labels = dataset.labels[indices]
    selected.radius_codes = dataset.radius_codes[indices]
    selected.normals = dataset.normals[indices]
    return selected


def score_condition(dataset, references, paper, models, paper_modules, device, groups):
    paper_outputs = paper_predict(dataset, paper, *paper_modules, device)
    attention_outputs = {key: attention_predict(dataset, model, device)
                         for key, model in models.items()}
    scored = {}
    for name, ids in groups.items():
        scored[name] = {"paper": metrics(dataset, paper_outputs, references, ids),
                        "attention": {}}
        for key, outputs in attention_outputs.items():
            scored[name]["attention"][key] = {
                "soft": metrics(dataset, outputs, references, ids),
                "hard": metrics(dataset, outputs, references, ids, fit_key="hard_fit"),
            }
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "circle_net_paper_baseline")
    parser.add_argument("--attention-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--noise-percent", nargs="+", type=float, default=(0.1, 0.5, 1.0))
    parser.add_argument("--seeds", nargs="+", type=int, default=(101, 102, 103, 104, 105))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = CirclePatchesV2("test", groups=BALANCED_GROUPS)
    full_references, _ = clean_reference(original, 8, device)
    full_references = full_references.numpy()
    paper_train_ids, paper_holdout_ids = paper_split_ids()
    indices = [i for i, patch_id in enumerate(original.ids) if patch_id not in paper_train_ids]
    references = full_references[indices]
    ids = {original.ids[i] for i in indices}
    groups = {"paper_unseen": ids,
              "paper_original_holdout": ids & paper_holdout_ids,
              "never_selected_holes_50_80": {i for i in ids if i % 81 >= 50}}

    paper_model = load_module("paper_seg_model_noise", args.paper_root / "models/seg_model2.py")
    paper_data = load_module("paper_loader_noise", args.paper_root / "data_utils/CircleDataLoader.py")
    paper_train = load_module("paper_train_noise", args.paper_root / "train_partseg2.py")
    checkpoint = torch.load(args.paper_root / "log/part_seg/seg_model2/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)
    paper = paper_model.get_model(2).to(device)
    paper.load_state_dict(checkpoint["model_state_dict"])
    paper.eval()
    models = {}
    for path in args.attention_checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config"]["groups"] != BALANCED_GROUPS:
            raise ValueError(f"checkpoint has different group split: {path}")
        model = CircleNetPointAttention().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        models[str(path.resolve())] = model

    diagonal = (original.query.amax(dim=1) - original.query.amin(dim=1)).norm(dim=-1)
    output = {"paper_checkpoint": str(args.paper_root / "log/part_seg/seg_model2/checkpoints/best_model.pth"),
              "attention_checkpoints": list(models), "paper_checkpoint_epoch_zero_based": checkpoint["epoch"],
              "noise_percent": args.noise_percent, "seeds": args.seeds,
              "noise_definition": "independent Gaussian XYZ noise on l0 and l1; sigma = percent/100 times clean l0 bounding-box diagonal",
              "point_identity_and_labels": "unchanged after perturbation",
              "paper_input": "original normalization, KNN and PCA; deterministic 128-neighbor sampling; original 500 query points",
              "center_reference": "clean GT-boundary fit proxy, not CAD center",
              "distance_unit": "raw HDF5 coordinate", "groups": {k: sorted(v) for k, v in groups.items()},
              "conditions": []}

    def run(percent, seed, query, context):
        subset = select_dataset(original, indices, query, context)
        scored = score_condition(subset, references, paper, models,
                                 (paper_model, paper_data, paper_train), device, groups)
        output["conditions"].append({"noise_percent": percent, "seed": seed,
                                     "groups": scored})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        holdout = scored["paper_original_holdout"]
        print(json.dumps({"noise_percent": percent, "seed": seed,
                          "paper_holdout_f1": holdout["paper"]["f1"],
                          "paper_holdout_fit_count": holdout["paper"]["fit_count"],
                          "conditions_done": len(output["conditions"])}), flush=True)

    run(0.0, None, original.query, original.context)
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        q_noise = torch.randn(original.query.shape, generator=generator)
        c_noise = torch.randn(original.context.shape, generator=generator)
        for percent in args.noise_percent:
            sigma = (percent / 100 * diagonal)[:, None, None]
            run(percent, seed, original.query + sigma * q_noise,
                original.context + sigma * c_noise)


if __name__ == "__main__":
    main()
