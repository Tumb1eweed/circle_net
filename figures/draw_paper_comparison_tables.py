"""Recreate the two paper tables with the local point-attention results at right.

The published numbers are transcribed from the two tables in weizeyong.pdf.
The new column is read from eval_noise.py's JSON output. The second table is
transposed so that the added method occupies a column in both figures.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PAPER_CLASSIFICATION = [
    ["Precision", "39.04", "40.38", "33.93", "32.60", "77.95", "86.06"],
    ["Recall", "70.10", "22.15", "69.81", "26.82", "69.79", "77.62"],
    ["F1", "49.21", "28.61", "45.66", "29.43", "73.64", "81.58"],
]

PAPER_FITTING = {
    "0.1": {
        "AD(c)": ["0.465328", "0.466897", "0.438734", "0.469162", "0.649781", "0.421035"],
        "AD(r)": ["0.296654", "0.296752", "0.309176", "0.278478", "0.312905", "0.103621"],
        "MSE(r)": ["0.089543", "0.089596", "0.103949", "0.080783", "0.121466", "0.016823"],
    },
    "0.5": {
        "AD(c)": ["0.38116", "0.385131", "0.454232", "0.50883", "0.67379", "0.398301"],
        "AD(r)": ["0.31696", "0.321495", "0.377379", "0.239431", "0.312294", "0.082654"],
        "MSE(r)": ["0.103115", "0.105989", "0.172154", "0.092588", "0.155506", "0.01207"],
    },
    "1.0": {
        "AD(c)": ["0.667738", "0.675546", "0.809487", "0.707097", "0.783924", "0.671365"],
        "AD(r)": ["0.271244", "0.279059", "0.281098", "0.252474", "0.268966", "0.016359"],
        "MSE(r)": ["0.076074", "0.08018", "0.108321", "0.075499", "0.207283", "0.006351"],
    },
}


def draw_table(headers, rows, widths, out_prefix, *, fontsize=9.0, row_height=0.34):
    """Draw a restrained, publication-style table with a tinted last column."""
    fig_width = 12.2
    fig_height = 0.88 + row_height * len(rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.07, top=0.94)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = [0.0]
    for width in widths:
        x.append(x[-1] + width)
    top, header_bottom, bottom = 0.93, 0.72, 0.07
    ax.add_patch(Rectangle((x[-2], bottom), widths[-1], top - bottom,
                           facecolor="#eaf2f8", edgecolor="none"))
    ax.plot([0, 1], [top, top], color="#242424", lw=1.2)
    ax.plot([0, 1], [header_bottom, header_bottom], color="#242424", lw=0.8)
    ax.plot([0, 1], [bottom, bottom], color="#242424", lw=1.2)
    ax.plot([x[-2], x[-2]], [bottom, top], color="#a9becd", lw=0.8)

    for i, header in enumerate(headers):
        ax.text((x[i] + x[i + 1]) / 2, (top + header_bottom) / 2,
                header, ha="center", va="center", fontsize=fontsize,
                fontweight="semibold" if i == len(headers) - 1 else "normal")

    row_h = (header_bottom - bottom) / len(rows)
    for row_index, row in enumerate(rows):
        center_y = header_bottom - (row_index + 0.5) * row_h
        for col_index, value in enumerate(row):
            ax.text((x[col_index] + x[col_index + 1]) / 2, center_y,
                    value, ha="center", va="center", fontsize=fontsize)
        if len(rows) > 3 and (row_index + 1) % 3 == 0 and row_index < len(rows) - 1:
            y = header_bottom - (row_index + 1) * row_h
            ax.plot([0, 1], [y, y], color="#b3b3b3", lw=0.55)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out_prefix.with_suffix("." + extension), dpi=250,
                    facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Point-attention eval_noise.py JSON output")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    with args.results.open() as stream:
        result = json.load(stream)

    clean = result["clean"]
    cls_rows = [row + [f"{clean[key] * 100:.2f}"]
                for row, key in zip(PAPER_CLASSIFICATION,
                                    ("precision", "recall", "f1"))]
    draw_table(
        ["Metric", "Normal-based", "Clustering-based", "Geometry-based",
         "EC-NET", "PIE-NET", "Ours", "Point attention\n+ fit (local)"],
        cls_rows,
        [0.09, 0.12, 0.15, 0.15, 0.09, 0.09, 0.09, 0.22],
        args.output_dir / "classification_table_with_module",
        fontsize=9.1,
    )

    fit_rows = []
    metric_keys = (("AD(c)", "ad_c_proxy"), ("AD(r)", "ad_r"),
                   ("MSE(r)", "mse_r"))
    for noise in ("0.1", "0.5", "1.0"):
        for metric, key in metric_keys:
            local = result["noisy"][noise]["summary"][key]["mean"]
            fit_rows.append([f"{noise}%", metric]
                            + PAPER_FITTING[noise][metric] + [f"{local:.6f}"])
    draw_table(
        ["Noise", "Metric", "Hyper", "LLS", "RANSAC", "RLTS",
         "PIE-NET", "Ours", "Point attention\n+ fit (local)"],
        fit_rows,
        [0.07, 0.085, 0.105, 0.105, 0.115, 0.105, 0.105, 0.105, 0.205],
        args.output_dir / "noise_fitting_table_with_module",
        fontsize=8.6,
        row_height=0.33,
    )


if __name__ == "__main__":
    main()
