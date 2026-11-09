# CircleNet: circle extraction from 3D point clouds

The mainline model is `CircleNetPointAttention`: a PyTorch 2.6 CircleNet variant with differentiable circle fitting and point-to-point attention. It adds Morton-ordered 32-point windows, a shifted window, and 500-point global attention before the fitting heads. It is inspired by PTv3; it is not the official PTv3 architecture and does not use Sonata weights.

## Data and environment

The HDF5 dataset and experiment checkpoints are **not included** in this code repository. Place the CircleNet dataset at `data/circle_data/data.hdf5` before training. The current training code reads `data_l0`, `data_l1`, and `data_l2` from that file.

```bash
conda create -n circlenet-repro python=3.10 pip -y
conda run -n circlenet-repro python -m pip install -r requirements_repro.txt
```

The recorded experiments used PyTorch 2.6.0+cu126. Adjust the PyTorch wheel to match your CUDA setup if necessary.

## Train and evaluate the mainline model

Run these commands from this repository directory:

```bash
conda run -n circlenet-repro python train_v2.py

conda run -n circlenet-repro python eval_v2.py \
  --split test --by-radius \
  --out log/point_attention/test_by_radius.json

conda run -n circlenet-repro python plot_v2.py \
  --metrics log/point_attention/metrics.jsonl \
  --out log/point_attention/training_curves.png
```

`train_v2.py` now defaults to `point_attention`, 60 epochs, batch size 8, learning rate `3e-4`, the `balanced` split, and output directory `log/point_attention/`. `eval_v2.py` reads that checkpoint by default. The `--model fit` and `--model direct` options remain available for ablation runs; their default output directories are `log/fit/` and `log/direct/`. The balanced split uses scan groups 0–7 for training, 8–9 for validation, and 10–11 for testing. Each `data_l0` patch has 500 query points and represents at most one target circle; `data_l1` supplies 2000 context points. The model predicts boundary logits, point-weight logits, and a plane normal. A differentiable weighted algebraic fit then produces center and radius. The first 20 training epochs gradually replace the ground-truth boundary mask with predicted probabilities inside the fit; validation and testing use predictions throughout.

## Architecture compared with the paper source

Both versions build local and wider neighborhoods for each query point and fuse their two feature tokens. In the [paper baseline](https://github.com/Tumb1eweed/circle_net/tree/paper-baseline), its four-layer Transformer attends only between those two tokens **within one query point**. The mainline model uses a smaller 128-channel neighborhood encoder and one token-fusion layer, then adds communication **between different query points**:

1. Encode the 16 nearest and 128 wider context neighbors for each of 500 query points; fuse the two tokens into a 128-channel feature per point.
2. Add normalized XYZ position features, order points by a Morton code, and apply attention in 32-point windows followed by windows shifted by 16 points.
3. Apply one global attention block over all 500 query points; predict boundary class, fitting weight, and a patch-level plane normal.
4. Fit one circle per positive patch with predicted soft boundary probabilities and weights using a differentiable weighted algebraic solver. The paper source instead selects hard boundary points and fits with a PCA plane and weighted least squares.

The paper source has 36,198,563 parameters; the mainline model has 615,366. The gain is not attributable to attention alone because the backbone, fitting method, and training objective also changed. The internal `fit` ablation keeps the modern backbone and solver but omits point-to-point attention; it has 209,094 parameters. See the [architecture figure](figures/point_attention_architecture.svg) and [paired ablation report](ABLATION_REPORT_20261109.md).

See [EXPERIMENTS_V2.md](EXPERIMENTS_V2.md) for the recorded results, [NOISE_EXPERIMENT.md](NOISE_EXPERIMENT.md) for controlled Gaussian-noise tests, [ALL_FITS_NOISE_COMPARISON.md](ALL_FITS_NOISE_COMPARISON.md) for the same-data fitting table, [LLS_NOISE_COMPARISON.md](LLS_NOISE_COMPARISON.md) for a focused LLS comparison and unit audit, [REPRODUCTION.md](REPRODUCTION.md) for the earlier reproduction, and [the architecture figure](figures/point_attention_architecture.svg) for tensor dimensions. Source files for the current pipeline are `model_point_attention.py`, `model_v2.py`, `repro_model.py`, `train_v2.py`, and `eval_v2.py`.

## Original implementation

The original implementation remains in `models/seg_model2.py`, `train_partseg2.py`, and `test_partseg2.py`, and is isolated in the [`paper-baseline` branch](https://github.com/Tumb1eweed/circle_net/tree/paper-baseline). Its legacy dependency specification is `env.yml`. The mainline point-attention model uses the environment specification above.
