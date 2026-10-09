# RSDehamba-Lite Project Handoff

Created on 2026-10-09.

## Ready

- Project scaffold for a reduced-VDB RSDehamba-style variant.
- Default model schedule: `[1, 2, 2]`.
- Paper-inspired U-Net, VDB, SSM branch, and four-direction DSM structure.
- Pure PyTorch scan fallback for quick setup.
- Optional `mamba_ssm` path if installed.
- Dataset inspection script for SateHaze1K-style paired folders.
- Training and evaluation scripts.
- Shape, gradient, and parameter-budget tests.
- README, access checks, and project specification.

## Local Verification

- Syntax compilation passed for all Python files.
- Environment check ran successfully.
- Full model tests were not run because this desktop environment has no PyTorch, Pillow, NumPy, pytest, Kaggle CLI, or Mamba package installed.

## Next Essential Step

Use Kaggle or a Python 3.10-3.12 environment, add the SateHaze1K dataset, then run:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/inspect_dataset.py --data-root /kaggle/input/satehaze1k
pytest -q
python scripts/train.py --config configs/rsdehamba_lite.json --dry-run
```

If the dataset inspector finds zero pairs, the only essential user choice is to share the actual Kaggle folder tree or decide the exact hazy/clear folder names.

