"""Plot train/validation loss and validation F1 from metrics.jsonl."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not records or any("loss" not in record["val"] for record in records):
        raise ValueError("The metrics file is empty or does not contain validation losses")
    epochs = np.array([record["epoch"] for record in records])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat[:3],
                                ("loss", "class_loss", "fit_loss"),
                                ("Total loss", "Classification loss", "Circle fitting loss")):
        train = np.array([record["train"][key] for record in records])
        val = np.array([record["val"][key] for record in records])
        axis.plot(epochs, train, label="Train", lw=1.5, color="#1976d2")
        axis.plot(epochs, val, label="Validation", lw=1.5, color="#d95f02")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    f1 = [record["val"]["f1"] for record in records]
    axis = axes.flat[3]
    axis.plot(epochs, f1, lw=1.5, color="#388e3c")
    axis.set(title="Validation point F1 (threshold 0.5)", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    axis.grid(True, alpha=0.25)
    fig.suptitle("Circle-Net reproduction: training history", fontsize=14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
