# CircleNet: circle extraction from 3D point clouds

This repository contains the original CircleNet implementation and a PyTorch 2.6 reproduction with differentiable circle fitting. The current point-attention experiment adds Morton-ordered 32-point windows, a shifted window, and 500-point global attention before the fitting heads. It is inspired by PTv3; it is not the official PTv3 architecture and does not use Sonata weights.

## Data and environment

The HDF5 dataset and experiment checkpoints are **not included** in this code repository. Place the CircleNet dataset at `data/circle_data/data.hdf5` before training. The current training code reads `data_l0`, `data_l1`, and `data_l2` from that file.

```bash
conda create -n circlenet-repro python=3.10 pip -y
conda run -n circlenet-repro python -m pip install -r requirements_repro.txt
```

The recorded experiments used PyTorch 2.6.0+cu126. Adjust the PyTorch wheel to match your CUDA setup if necessary.

## Train and evaluate the current model

Run these commands from this repository directory:

```bash
conda run -n circlenet-repro python train_v2.py \
  --epochs 60 --batch-size 8 --lr 3e-4 \
  --split-protocol balanced --model point_attention \
  --out-dir log/point_attention_balanced

conda run -n circlenet-repro python eval_v2.py \
  --checkpoint log/point_attention_balanced/last_model.pth \
  --split test --by-radius \
  --out log/point_attention_balanced/test_by_radius.json

conda run -n circlenet-repro python plot_v2.py \
  --metrics log/point_attention_balanced/metrics.jsonl \
  --out log/point_attention_balanced/training_curves.png
```

The balanced split uses scan groups 0–7 for training, 8–9 for validation, and 10–11 for testing. Each `data_l0` patch has 500 query points and represents at most one target circle; `data_l1` supplies 2000 context points. The model predicts boundary logits, point-weight logits, and a plane normal. A differentiable weighted algebraic fit then produces center and radius. The first 20 training epochs gradually replace the ground-truth boundary mask with predicted probabilities inside the fit; validation and testing use predictions throughout.

See [EXPERIMENTS_V2.md](EXPERIMENTS_V2.md) for the recorded results, [REPRODUCTION.md](REPRODUCTION.md) for the earlier reproduction, and [the architecture figure](figures/point_attention_architecture.svg) for tensor dimensions. Source files for the current pipeline are `model_point_attention.py`, `model_v2.py`, `repro_model.py`, `train_v2.py`, and `eval_v2.py`.

## Original implementation

The original implementation remains in `models/seg_model2.py`, `train_partseg2.py`, and `test_partseg2.py`. Its legacy dependency specification is `env.yml`. The updated reproduction and point-attention experiment have their own training scripts and environment specification above.
