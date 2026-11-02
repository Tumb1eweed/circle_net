"""Run the reproduced model on a raw scan and export point and circle results."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from repro_model import CircleNetRepro, weighted_circle_fit


ROOT = Path(__file__).resolve().parent


def connected_components(points, radius):
    tree = cKDTree(points)
    parent = np.arange(len(points))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in tree.query_pairs(radius):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for index in range(len(points)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def write_ply(path, xyz, selected):
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write("element vertex %d\n" % len(xyz))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, keep in zip(xyz, selected):
            rgb = (240, 45, 45) if keep else (155, 160, 165)
            stream.write("%.7f %.7f %.7f %d %d %d\n" % (*point, *rgb))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/circle_data/test_data/real10_2_0.8_norm.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "log/repro_inference")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cluster-radius", type=float, default=0.06)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--min-radius-ratio", type=float, default=0.01,
                        help="Minimum contour radius as a fraction of the scan bounding-box diagonal")
    parser.add_argument("--max-radius-ratio", type=float, default=0.15,
                        help="Maximum contour radius as a fraction of the scan bounding-box diagonal")
    parser.add_argument("--concentric-center-factor", type=float, default=1.0,
                        help="Merge contour centers closer than this factor times the larger radius")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan = np.loadtxt(args.input, dtype=np.float32)[:, :3]
    context = torch.from_numpy(scan)[None].to(device)
    model = CircleNetRepro().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    scores, weights = [], []
    with torch.no_grad():
        for start in range(0, len(scan), args.batch_size):
            query = context[:, start:start + args.batch_size]
            logits, weight_logits = model(query, context)
            scores.append(logits.softmax(-1)[0, :, 1].cpu().numpy())
            weights.append(weight_logits.sigmoid()[0].cpu().numpy())
    scores = np.concatenate(scores)
    weights = np.concatenate(weights)
    selected = scores >= args.threshold
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_dir / "point_predictions.txt",
               np.column_stack((scan, scores, weights, selected.astype(np.int32))),
               fmt="%.7f", header="x y z boundary_probability fitting_weight selected")
    write_ply(args.out_dir / "boundary_predictions.ply", scan, selected)

    selected_indices = np.flatnonzero(selected)
    contours = []
    for cluster in connected_components(scan[selected], args.cluster_radius):
        if len(cluster) < args.min_points:
            continue
        indices = selected_indices[cluster]
        subset = scan[indices]
        _, _, right = np.linalg.svd(subset - subset.mean(axis=0), full_matrices=False)
        normal = torch.from_numpy(right[-1:].copy()).to(device)
        point_tensor = torch.from_numpy(subset[None]).to(device)
        weight_tensor = torch.from_numpy((scores[indices] * weights[indices])[None]).to(device)
        with torch.no_grad():
            center, radius = weighted_circle_fit(point_tensor, weight_tensor, normal)
        contours.append({"center": center[0].cpu().tolist(), "radius": radius.item(),
                         "point_count": len(indices), "point_indices": indices.tolist(),
                         "mean_probability": float(scores[indices].mean())})
    contours.sort(key=lambda item: item["point_count"], reverse=True)
    (args.out_dir / "contours.json").write_text(json.dumps(contours, indent=2), encoding="utf-8")

    # A hole can produce two concentric contours at different depths. Group
    # centers in the main surface plane, while retaining both measured radii.
    extent = float(np.linalg.norm(np.ptp(scan, axis=0)))
    candidates = [contour for contour in contours
                  if args.min_radius_ratio * extent <= contour["radius"] <= args.max_radius_ratio * extent]
    hole_selected = np.zeros(len(scan), dtype=bool)
    for contour in candidates:
        hole_selected[contour["point_indices"]] = True
    write_ply(args.out_dir / "hole_boundary_candidates.ply", scan, hole_selected)
    _, _, axes = np.linalg.svd(scan - scan.mean(axis=0), full_matrices=False)
    normal = axes[-1]
    projected = []
    for contour in candidates:
        center = np.asarray(contour["center"])
        projected.append(center - np.dot(center, normal) * normal)
    parent = list(range(len(candidates)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            max_radius = max(candidates[a]["radius"], candidates[b]["radius"])
            if np.linalg.norm(projected[a] - projected[b]) < args.concentric_center_factor * max_radius:
                parent[find(b)] = find(a)
    groups = {}
    for index, contour in enumerate(candidates):
        groups.setdefault(find(index), []).append(contour)
    holes = []
    for group in groups.values():
        group.sort(key=lambda item: item["radius"], reverse=True)
        holes.append({"outer_contour": group[0], "other_contours": group[1:]})
    holes.sort(key=lambda item: item["outer_contour"]["point_count"], reverse=True)
    (args.out_dir / "holes.json").write_text(json.dumps(holes, indent=2), encoding="utf-8")
    print(json.dumps({"scan_points": len(scan), "selected_points": int(selected.sum()),
                      "contour_candidates": len(contours), "hole_candidates": len(holes),
                      "output": str(args.out_dir)}))


if __name__ == "__main__":
    main()
