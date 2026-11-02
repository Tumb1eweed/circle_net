"""Export matching data_l0/data_l1 HDF5 samples as colored PLY point clouds."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "data/circle_data/data.hdf5"
OUT_DIR = ROOT / "data/sample"

# Gray is background (-1). Boundary colors distinguish the eight radius codes.
RADIUS_COLORS = np.array([
    [230, 40, 40], [240, 120, 30], [220, 190, 30], [45, 170, 65],
    [30, 185, 195], [45, 95, 230], [145, 70, 200], [220, 65, 165],
], dtype=np.uint8)


def write_ply(path, points):
    """Write XYZ, visual colors, and the two original per-point annotations."""
    label = points[:, 3].astype(np.int8)
    radius_code = points[:, 4].astype(np.int8)
    colors = np.full((len(points), 3), 175, dtype=np.uint8)
    positive = label == 1
    colors[positive] = RADIUS_COLORS[radius_code[positive]]
    with path.open("w", encoding="ascii") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write("comment original XYZ coordinates; no normalization\n")
        out.write("comment gray=background; colored=boundary; see README.txt\n")
        out.write(f"element vertex {len(points)}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        out.write("property uchar boundary_label\nproperty char radius_code\nend_header\n")
        for xyz, color, boundary, code in zip(points[:, :3], colors, label, radius_code):
            out.write(f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                      f"{color[0]} {color[1]} {color[2]} {int(boundary)} {int(code)}\n")


def codes(points):
    values = np.unique(points[points[:, 3] == 1, 4].astype(np.int8))
    return ";".join(map(str, values.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=H5_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as h5:
        total = len(h5["data_l0"])
        if not 1 <= args.count <= total:
            parser.error(f"--count must be between 1 and {total}")
        ids = np.random.default_rng(args.seed).choice(total, size=args.count, replace=False)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample_id in ids.tolist():
            small = h5["data_l0"][sample_id]
            large = h5["data_l1"][sample_id]
            normal = h5["data_l2"][sample_id]
            stem = f"sample_{sample_id:04d}"
            write_ply(args.out_dir / f"{stem}_l0.ply", small)
            write_ply(args.out_dir / f"{stem}_l1.ply", large)
            records.append({
                "sample_id": sample_id,
                "scan_group": sample_id // 81,
                "index_in_group": sample_id % 81,
                "l0_boundary_points": int((small[:, 3] == 1).sum()),
                "l1_boundary_points": int((large[:, 3] == 1).sum()),
                "l0_radius_codes": codes(small),
                "l1_radius_codes": codes(large),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
            })
    with (args.out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    legend = ["Matching l0/l1 files share a sample ID.",
              "Each PLY stores original XYZ, RGB, boundary_label, and radius_code.",
              "Gray (175,175,175): non-boundary point, radius_code=-1.",
              "Boundary colors by radius code (HDF5 coordinate units):"]
    radii = [0.06, 0.05, 0.04, 0.03, 0.02, 0.0125, 0.0075, 0.005]
    legend.extend(f"  {i}: radius {radius:g}, RGB {tuple(map(int, color))}"
                  for i, (radius, color) in enumerate(zip(radii, RADIUS_COLORS)))
    legend.append(f"Random seed: {args.seed}; selected IDs appear in samples.csv.")
    (args.out_dir / "README.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")
    print(f"Exported {len(ids)} matching sample pairs ({len(ids) * 2} PLY files) to {args.out_dir}")
    print("Sample IDs:", ", ".join(map(str, ids.tolist())))


if __name__ == "__main__":
    main()
