"""Plot training and validation diagnostics from train_v2.py metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line]
    epoch = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for split in ("train", "val"):
        for key, ax in (("loss", axes[0, 0]), ("class_loss", axes[0, 1])):
            ax.plot(epoch, [row[split][key] for row in rows], label=split)
    axes[0, 0].set_title("Total loss")
    axes[0, 1].set_title("Classification loss")
    axes[1, 0].plot(epoch, [row["val"]["f1"] for row in rows], color="tab:green")
    axes[1, 0].set_title("Validation point F1")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(epoch, [row["val"]["soft_radius_mae"] for row in rows],
                    label="Predicted soft mask")
    axes[1, 1].plot(epoch, [row["val"]["oracle_radius_mae"] for row in rows],
                    label="GT boundary (reference)")
    axes[1, 1].set_title("Validation radius MAE (coordinate units)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if ax.lines and len(ax.lines) > 1:
            ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
