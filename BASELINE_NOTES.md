# Paper source baseline

This branch contains the original CircleNet source from the user-provided
`CircleNet(1).zip` archive (SHA-256
`21ca74051cdb66e1112c78d5cd0cae5305aa5db1cdf6e4bc24a0833804f35676`).
It starts from repository commit `897c03d`, then replaces the newer reproduction
and point-attention files with the paper implementation at the repository root.

## Intentional changes to the archive

- Removed `.detach()` from the radius loss in `models/seg_model2.py`. The same
  expression in the archived training-log copy was updated locally as well.
  This lets `loss2` backpropagate through the weighted least-squares circle fit
  on a modern PyTorch version. Boundary point selection still uses an argmax,
  so this does not make the classification decision differentiable.
- Changed the training script's radius logging from NumPy conversion of CUDA
  tensors to `torch.cat(...).mean().item()`. Once `loss2` has gradients, the
  original NumPy conversion would fail. This change affects logging only.
- Kept the archive's original `README.md`. Its `train.py` and `test.py` command
  names are stale; the actual entry points are `train_partseg2.py` and
  `test_partseg2.py`. Trailing whitespace and excess blank lines were cleaned
  in the changed files.

The archive's dataset (`data/circle_data/data.hdf5`), example test data, logs,
and 434 MB `best_model.pth` checkpoint are available in this local worktree but
are ignored by Git. Generated logs, bytecode, and duplicate source snapshots
inside `log/` are not part of the pushed branch. The original backup model file
has a platform-encoded name in the ZIP; the identical file already tracked as
`models/seg_model2 (#U590d#U4ef6).py` keeps its repository spelling.

## Validation

Using the local `circlenet-repro` environment (PyTorch 2.6.0+cu126), a real
positive patch from the archived HDF5 dataset produced a radius loss with
`requires_grad=True`; backward yielded finite, nonzero gradients on all 49
selected boundary-point weights. This is a focused gradient smoke test, not a
full retraining of the original paper model. The archived `env.yml` pins the
older original environment and is preserved for provenance.
