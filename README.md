# Deep Algebraic Fitting for Multiple Circle Primitives Extraction from Raw Point Clouds
This is our implementation of CircleNet

## Environment
* Python 3.5+
* PyTorch 1.3.1+
* CUDA and CuDNN (CUDA 10.1+ & CuDNN 7.5+)
* TensorboardX (2.0) if logging training info.

## Installation

You can install via conda environment .yaml file

```bash
conda env create -f env.yml
conda activate circlenet
```

## Train
Use the script `train.py` to train a model in the our dataset (the trained model will be saved at `./log/part_seg/seg_model2/checkpoints/best_model.pth`):
``` bash
cd CircleNet
python train.py
```
## Test (The filtered results will be saved at `./log/part_seg/seg_model2/results/`)
``` bash
cd CircleNet
python test.py
```

## Citation
