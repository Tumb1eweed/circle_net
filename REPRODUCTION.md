# Circle-Net reproduction with a differentiable fit

This is an improved reproduction using the bundled `data/circle_data/data.hdf5`.
It trains a compact local/global point encoder, per-point boundary classifier,
and fitting-weight head. A batched weighted algebraic circle fit runs entirely
in PyTorch, so fitting gradients reach both heads and the shared encoder.

## Environment

```bash
conda create -n circlenet-repro python=3.10 pip -y
conda run -n circlenet-repro python -m pip install -r requirements_repro.txt
```

The tested setup uses PyTorch 2.6.0 with CUDA 12.6 on an RTX 4090.

## Train and inspect

From the `CircleNet` directory:

```bash
conda run -n circlenet-repro python train_repro.py --epochs 400 --batch-size 8 \
  --lr 1e-5 --out-dir log/repro_paperlr_400_20261102
conda run -n circlenet-repro python eval_repro.py \
  --checkpoint log/repro_paperlr_400_20261102/last_model.pth \
  --out log/repro_paperlr_400_20261102/final_eval.json
conda run -n circlenet-repro python plot_training.py \
  --metrics log/repro_paperlr_400_20261102/metrics.jsonl \
  --out log/repro_paperlr_400_20261102/training_validation_loss.png
conda run -n circlenet-repro python infer_repro.py \
  --checkpoint log/repro_paperlr_400_20261102/last_model.pth \
  --out-dir log/repro_paperlr_400_20261102/real_scan
```

Training starts from random weights unless `--resume` is explicitly supplied.
The script saves only `last_model.pth`, and evaluation uses the final epoch.
`metrics.jsonl` records train and validation total/classification/fitting
losses, point precision, recall and F1, plus radius error. The plotting script
shows the three losses and validation F1 across all epochs.
`oracle_radius_mae` uses ground-truth boundary point membership and isolates
the quality of learned fitting weights. `predicted_radius_mae` uses predicted
boundary points and is the more realistic metric; `predicted_fit_coverage`
reports how many samples yielded at least 10 predicted points.
The corresponding `*_mse` fields are mean squared radius errors.
The fitting loss uses relative smooth L1 radius error and a positive weight
floor on ground-truth boundary points. This keeps rare near-degenerate small
circles from dominating training while preserving fitting gradients to the
classifier, fitting-weight head, and shared encoder.

The 400-epoch run started from random weights, with AdamW and learning rate
1e-5. At the fixed 0.5 threshold, epoch 400 achieved precision 68.32%, recall
93.15%, and F1 78.82% on the 192 held-out patches. Train total loss fell to
0.1956, while validation total loss reached 0.4741. The lowest logged
validation loss was 0.3067 at epoch 102; the later rise is mainly in the
classification component. The last checkpoint is retained regardless of this
trend. Predicted-boundary radius MAE was 0.01343 in HDF5 coordinate units.
The learning rate and epoch count match the paper's stated training settings;
the optimizer does not: the paper reports Adam, while this modified code uses
AdamW with weight decay 1e-4.

The earlier 50-epoch run in `log/repro_final_20261102` used learning rate 3e-4,
seed 42, a
batch size of 8, and PyTorch 2.6.0+cu126. On its 192 held-out patches, the
final checkpoint at a fixed 0.5 classification threshold achieved precision
66.38%, recall 95.03%, and F1 78.16%. Predicted-boundary radius MAE was
0.013333 and MSE was 0.0004743 in the HDF5 coordinate units; fitting coverage
was 91.15%. The bundled unlabeled real scan produced 5 heuristic hole
candidates, with no accuracy claim. A diagnostic threshold sweep on the same
validation patches gave F1 84.94% at threshold 0.9, but this is tuned on the
validation set and must not be treated as an independent test result.

For context, the paper's Table 2 reports precision 86.06%, recall 77.62%, and
F1 81.58% on virtually scanned full point clouds. Its Table 3 reports radius
AD(r) and MSE(r) separately at 0.1%, 0.5%, and 1% noise, in mm and mm².

The split holds out two hole positions per radius in all 12 scan groups:
768 training patches and 192 validation patches, covering all eight radii.
The 81st patch in each group contains no labeled hole and is excluded.
This split differs from the original
paper's evaluation protocol, so its numbers must not be compared directly with
the paper's tables. The HDF5 file provides per-point boundary labels, radius
codes, and a direction vector (`data_l2`); it does **not** contain CAD circle
center ground truth. The fitting loss therefore supervises radius, while the
direction vector supplies the fitting plane. It is not a network input.
This compact PyTorch model is a modified reproduction, not an exact re-run of
the paper's original architecture or evaluation protocol. The HDF5 coordinate
units do not establish a mm conversion, so the radius errors above cannot be
numerically compared with Table 3. Center error cannot be computed without
center ground truth.

`infer_repro.py` also processes the bundled unlabeled real scan. It exports
per-point scores, a colored PLY, geometric contour candidates, and grouped
hole candidates. Because this scan has no ground-truth labels, the grouped
hole count is a qualitative result, not a measured detection accuracy.
The clustering radius and contour-size filters are heuristic defaults for the
bundled scan; tune them for other scan scales and hole layouts.
