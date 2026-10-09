# Kaggle Workflow: RSDehamba-Lite `[1,2,2]`

Use this after creating a Kaggle notebook and attaching the SateHaze1K dataset.

## 1. Confirm GPU

```python
import torch, os
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("inputs:", os.listdir("/kaggle/input"))
```

If CUDA is false, enable GPU in Kaggle notebook settings before training.

## 2. Upload This Project

In Kaggle, upload the project folder as a dataset or use the notebook file browser to upload the files. Then set `PROJECT_DIR` to that folder.

```python
from pathlib import Path

PROJECT_DIR = Path("/kaggle/input/rsdehamba-lite-project")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path("/kaggle/working/rsdehamba-lite-project")
print(PROJECT_DIR)
```

## 3. Install Project

```python
%cd {PROJECT_DIR}
!pip install -q -r requirements.txt
!pip install -q -e .
```

## 4. Find The Dataset Root

```python
from pathlib import Path

for root in Path("/kaggle/input").iterdir():
    print(root)
    for child in list(root.iterdir())[:10]:
        print("  ", child.name)
```

Set the actual dataset root:

```python
DATA_ROOT = "/kaggle/input/satehaze1k"
```

## 5. Inspect Dataset Layout

```python
!python scripts/inspect_dataset.py --data-root "$DATA_ROOT"
```

If this reports `pairs: 0`, copy the printed Kaggle folder tree and share it so the loader aliases can be adjusted.

## 6. Run Tests And Dry Run

```python
!pytest -q
!python scripts/train.py --config configs/rsdehamba_lite_smoke.json --data-root "$DATA_ROOT" --dry-run
```

## 7. First Tiny Training Stage

```python
!python scripts/train.py \
  --config configs/rsdehamba_lite_smoke.json \
  --data-root "$DATA_ROOT" \
  --output-dir /kaggle/working/rsdehamba_lite_outputs
```

Success means the loop runs, loss prints each epoch, and `/kaggle/working/rsdehamba_lite_outputs/checkpoints/best.pt` exists.

## 8. Evaluate The Tiny Checkpoint

```python
!python scripts/evaluate.py \
  --config configs/rsdehamba_lite_smoke.json \
  --data-root "$DATA_ROOT" \
  --checkpoint /kaggle/working/rsdehamba_lite_outputs/checkpoints/best.pt
```

Do not use this tiny run as final research performance. It only proves the pipeline.

## 9. Full Training Run

After the smoke test works:

```python
!python scripts/train.py \
  --config configs/rsdehamba_lite.json \
  --data-root "$DATA_ROOT" \
  --output-dir /kaggle/working/rsdehamba_lite_full \
  --epochs 50 \
  --batch-size 4 \
  --image-size 256
```

Then evaluate:

```python
!python scripts/evaluate.py \
  --config configs/rsdehamba_lite.json \
  --data-root "$DATA_ROOT" \
  --checkpoint /kaggle/working/rsdehamba_lite_full/checkpoints/best.pt
```

## What To Record For The Report

- Dataset folder layout and number of discovered pairs.
- Parameter count printed by `train.py`.
- Smoke-test result: tests pass, dry-run shape, tiny training loss.
- Full run metrics by haze level: Thin, Moderate, Thick.
- Caveat: this is an RSDehamba-inspired reduced-VDB implementation, not an exact official reproduction.

