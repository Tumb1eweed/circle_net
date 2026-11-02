"""Evaluate a saved final-epoch checkpoint on the held-out HDF5 patches."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from repro_model import CircleNetRepro
from train_repro import CirclePatches, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CircleNetRepro().to(device)
    model.load_state_dict(checkpoint["model"])
    training = CirclePatches("train")
    negatives = (training.labels == 0).sum().item()
    positives = (training.labels == 1).sum().item()
    class_weights = torch.tensor([1.0, negatives / positives], device=device)
    dataset = CirclePatches("val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "epoch": checkpoint["epoch"] + 1,
              "validation_patches": len(dataset),
              "threshold": args.threshold,
              "metrics": validate(model, loader, device, args.threshold,
                                  class_weights=class_weights)}
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
