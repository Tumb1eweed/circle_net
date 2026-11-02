"""Plot the controlled-noise evaluation from two eval_noise.py JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def series(payload, metric):
    levels = [0.0] + sorted(float(key) for key in payload["noisy"])
    values = [payload["clean"][metric]]
    deviations = [0.0]
    for level in levels[1:]:
        item = payload["noisy"][str(level)]["summary"][metric]
        values.append(item["mean"])
        deviations.append(item["sd"])
    return np.asarray(levels), np.asarray(values), np.asarray(deviations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-attention", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=Path("figures/noise_robustness"))
    args = parser.parse_args()
    with args.point_attention.open() as stream:
        point = json.load(stream)
    with args.baseline.open() as stream:
        baseline = json.load(stream)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), constrained_layout=True)
    specs = (("f1", "Boundary F1 (%)", 100),
             ("ad_r", "AD(r) (HDF5 units)", 1),
             ("ad_c_proxy", "Center proxy error (HDF5 units)", 1))
    models = ((baseline, "Two-token fit baseline", "#b45d16", "s"),
              (point, "Point attention + fit", "#275d95", "o"))
    for ax, (metric, ylabel, factor) in zip(axes, specs):
        for payload, label, color, marker in models:
            _, y, err = series(payload, metric)
            ax.errorbar(np.arange(len(y)), factor * y, yerr=factor * err, color=color,
                        marker=marker, markersize=4.2, linewidth=1.5,
                        capsize=2.0, label=label)
        ax.set_xticks(range(4), ["Original", "0.1%", "0.5%", "1.0%"])
        ax.set_xlabel("Gaussian noise level (% of l0 bbox diagonal)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d9e1e8", linewidth=0.55)
        ax.set_xlim(-0.15, 3.15)
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.suptitle("Noise robustness on held-out scan groups (5 noise seeds)", fontsize=10)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.out_prefix.with_suffix("." + suffix), dpi=240,
                    facecolor="white")
    svg_path = args.out_prefix.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
