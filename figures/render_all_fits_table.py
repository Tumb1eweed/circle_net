"""Render a single-protocol LaTeX table from eval_all_fits_noise.py results."""

import argparse
import json
from pathlib import Path


ROWS = (
    ("Hyper", "Hyper"),
    ("LLS", "LLS"),
    ("RANSAC", "RANSAC"),
    ("RLTS (approx.)", "RLTS-approx"),
    ("PIE-NET", None),
    ("Ours (local)", "Ours-local"),
    ("Ours+Attn", "Ours+Attn"),
)
METRICS = ("ad_c_proxy", "ad_r", "mse_r")


def render(result):
    levels = result["noise_percent"]
    assert len(levels) == 3, "The paper layout has three noise groups"
    summaries = [result["noisy"][str(level)]["summary"]["metrics"] for level in levels]
    best = {(group, metric): min(summaries[group][key][metric]["mean"]
                                 for _, key in ROWS if key is not None)
            for group in range(3) for metric in METRICS}
    lines = [
        r"\documentclass[border=3pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{tgtermes}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        r"\begingroup",
        r"\fontsize{10.5}{12}\selectfont",
        r"\setlength{\tabcolsep}{6.4pt}",
        r"\setlength{\heavyrulewidth}{0.55pt}",
        r"\setlength{\lightrulewidth}{0.45pt}",
        r"\setlength{\cmidrulewidth}{0.45pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{@{}cccccccccc@{}}",
        r"% All numeric cells are local HDF5 test results in raw coordinate units.",
        r"% AD(c) is a clean-GT-boundary center proxy; PIE-NET was not rerun.",
        r"\toprule",
        r"Noise level & \multicolumn{3}{c}{0.1\% noise} & \multicolumn{3}{c}{0.5\% noise} & \multicolumn{3}{c}{1\% noise} \\",
        r"\cmidrule(l{1pt}r{1pt}){2-4}\cmidrule(l{1pt}r{1pt}){5-7}\cmidrule(l{1pt}r{1pt}){8-10}",
        r"Method & $AD(\mathbf{c})$ & $AD(r)$ & $MSE(r)$ & $AD(\mathbf{c})$ & $AD(r)$ & $MSE(r)$ & $AD(\mathbf{c})$ & $AD(r)$ & $MSE(r)$ \\",
        r"\midrule",
    ]
    for label, key in ROWS:
        cells = []
        for group in range(3):
            for metric in METRICS:
                if key is None:
                    cells.append(r"--")
                    continue
                value = summaries[group][key][metric]["mean"]
                cell = f"{value:.6f}"
                if abs(value - best[group, metric]) < 1e-15:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{document}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/all_fits_noise_test.json"))
    parser.add_argument("--out", type=Path, default=Path("figures/noise_fitting_table_with_module.tex"))
    args = parser.parse_args()
    args.out.write_text(render(json.loads(args.results.read_text(encoding="utf-8"))), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
