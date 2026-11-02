"""Inspect the bundled CircleNet data without loading the neural network.

Run with a Python environment containing numpy, h5py, and matplotlib:
    python inspect_data.py
    python inspect_data.py --sample 10 --output my_preview.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "data/circle_data/data.hdf5"
SCAN_PATH = ROOT / "data/circle_data/test_data/real10_2_0.8_norm.txt"
RESULT_DIR = ROOT / "log/part_seg/seg_model2/results"
RESULT_STEM = SCAN_PATH.stem


def show_values(name, values):
    unique, counts = np.unique(values, return_counts=True)
    pairs = [f"{value:g}: {count}" for value, count in zip(unique, counts)]
    print(f"  {name}: " + ", ".join(pairs))


def pca_projection(points, center=None, axes=None):
    if center is None:
        center = points.mean(axis=0)
    if axes is None:
        _, _, vt = np.linalg.svd(points - center, full_matrices=False)
        axes = vt[:2].T
    return (points - center) @ axes, center, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="HDF5 sample index")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data_preview.png",
        help="Where to save the visualization",
    )
    args = parser.parse_args()

    print(f"Training data: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as h5:
        for name, dataset in h5.items():
            print(f"  {name}: shape={dataset.shape}, dtype={dataset.dtype}")
        small = h5["data_l0"]
        large = h5["data_l1"]
        if not 0 <= args.sample < len(small):
            parser.error(f"--sample must be between 0 and {len(small) - 1}")
        patch = small[args.sample]
        context = large[args.sample]
        extra = h5["data_l2"][args.sample]

    print(f"\nSample {args.sample}: small patch {patch.shape}, context {context.shape}")
    print("  columns 0:3 = x, y, z; column 3 = boundary label; column 4 = radius code")
    print("  first five rows of the small patch:")
    print(np.array2string(patch[:5], precision=5, suppress_small=True))
    show_values("small-patch labels", patch[:, 3])
    show_values("small-patch radius codes", patch[:, 4])
    show_values("context labels", context[:, 3])
    print(f"  data_l2 sample (unused by the current loader): {extra}")

    scan = np.loadtxt(SCAN_PATH, dtype=np.float32)
    print(f"\nTest scan: {SCAN_PATH}")
    print(f"  shape={scan.shape}; first five rows:")
    print(np.array2string(scan[:5], precision=5, suppress_small=True))
    print(f"  xyz min={scan[:, :3].min(axis=0)}")
    print(f"  xyz max={scan[:, :3].max(axis=0)}")
    if scan.shape[1] == 3:
        print("  This test file has XYZ only; it contains no ground-truth labels.")

    output_path = RESULT_DIR / f"{RESULT_STEM}_output.txt"
    weight_path = RESULT_DIR / f"{RESULT_STEM}_weight.txt"
    predicted = np.loadtxt(output_path, dtype=np.float32) if output_path.exists() else None
    weighted = np.loadtxt(weight_path, dtype=np.float32) if weight_path.exists() else None
    if predicted is not None:
        show_values("saved predicted labels", predicted[:, 3])
    if weighted is not None:
        print(
            "  saved raw weight output: "
            f"min={weighted[:, 3].min():.4f}, "
            f"mean={weighted[:, 3].mean():.4f}, "
            f"max={weighted[:, 3].max():.4f}"
        )

    # PCA gives a readable view even if the point cloud is rotated in 3D.
    patch_xy, center, axes = pca_projection(patch[:, :3])
    context_xy, _, _ = pca_projection(context[:, :3], center, axes)
    scan_xy, _, _ = pca_projection(scan[:, :3])

    fig, panels = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ax = panels[0, 0]
    ax.scatter(patch_xy[:, 0], patch_xy[:, 1], c="lightgray", s=5)
    edge = patch[:, 3] == 1
    ax.scatter(patch_xy[edge, 0], patch_xy[edge, 1], c="crimson", s=8)
    ax.set_title(f"data_l0[{args.sample}]: red = boundary label 1")

    ax = panels[0, 1]
    ax.scatter(context_xy[:, 0], context_xy[:, 1], c="lightgray", s=3)
    ax.scatter(patch_xy[:, 0], patch_xy[:, 1], c="royalblue", s=5)
    ax.set_title("data_l1 context (gray) + data_l0 patch (blue)")

    ax = panels[1, 0]
    ax.scatter(scan_xy[:, 0], scan_xy[:, 1], c="lightgray", s=3)
    if predicted is not None and len(predicted) == len(scan):
        edge = predicted[:, 3] == 1
        ax.scatter(scan_xy[edge, 0], scan_xy[edge, 1], c="crimson", s=7)
        ax.set_title("Real scan: red = saved prediction 1")
    else:
        ax.set_title("Real scan: XYZ points")

    ax = panels[1, 1]
    if weighted is not None and len(weighted) == len(scan):
        artist = ax.scatter(
            scan_xy[:, 0], scan_xy[:, 1], c=weighted[:, 3], s=4, cmap="viridis"
        )
        fig.colorbar(artist, ax=ax, label="raw model output")
        ax.set_title("Real scan: saved weight-head output")
    else:
        ax.scatter(scan_xy[:, 0], scan_xy[:, 1], c="steelblue", s=3)
        ax.set_title("Real scan: XYZ points")

    for ax in panels.flat:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("PCA axis 1")
        ax.set_ylabel("PCA axis 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
