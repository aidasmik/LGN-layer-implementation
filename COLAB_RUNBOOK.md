# LGN_Nano Colab Runbook

Purpose: run integrity tests and optionally launch heavier LGN_Nano experiments on a Colab GPU.

## 0. Pick A Runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then run:

```python
!nvidia-smi
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Colab GPU availability changes over time. Free Colab often gives T4-class GPUs;
paid tiers may expose stronger GPUs such as L4/A100 when available. Always trust
`nvidia-smi` for the actual runtime you got.

## 1. Get The Project Into Colab

### Option A: Clone From GitHub

```python
!git clone <YOUR_REPO_URL> /content/Project_Logic
%cd /content/Project_Logic
```

### Option B: Copy From Google Drive

Upload/copy the project folder to Drive first, then:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/Project_Logic" /content/Project_Logic
%cd /content/Project_Logic
```

## 2. Install Test Dependencies

```python
!pip -q install pytest datasets
```

## 3. Run Integrity Tests

```python
%cd /content/Project_Logic/LGN_Nano
!python -m pytest tests/test_lgn_integrity.py -q
```

Expected result:

```text
......                                                                   [100%]
6 passed
```

These tests check:

- thermometer STE has unit total gradient;
- hybrid attention and `ln_1` are frozen;
- hybrid frozen sublayers stay in eval mode after `.train()`;
- the `ste` toggle is reset after fine-tune;
- token shift layout is channel-aligned and soft/hard paths match;
- scaling imitation can use live-prefix inputs with teacher target layer;
- aggressive `width_mult` no-op behavior is explicitly documented by test.

## 4. Optional: Run A Small LGN_Nano Experiment

If you have a baseline checkpoint in Drive, use it. This avoids retraining the
baseline every Colab session.

```python
%cd /content/Project_Logic/LGN_Nano
!python run.py heatmap \
  --checkpoint /content/drive/MyDrive/Project_Logic/LGN_Nano/results/baseline.pt \
  --results_dir /content/drive/MyDrive/lgn_colab_results/smoke_heatmap \
  --layers 0 5 11 \
  --imitation_steps 200 \
  --finetune_steps 200
```

For a real run, raise steps back toward defaults:

```python
%cd /content/Project_Logic/LGN_Nano
!python run.py heatmap \
  --checkpoint /content/drive/MyDrive/Project_Logic/LGN_Nano/results/baseline.pt \
  --results_dir /content/drive/MyDrive/lgn_colab_results/heatmap_full \
  --imitation_steps 1000 \
  --finetune_steps 1000
```

Scaling run:

```python
%cd /content/Project_Logic/LGN_Nano
!python run.py scale \
  --checkpoint /content/drive/MyDrive/Project_Logic/LGN_Nano/results/baseline.pt \
  --heatmap /content/drive/MyDrive/lgn_colab_results/heatmap_full/heatmap.json \
  --results_dir /content/drive/MyDrive/lgn_colab_results/scale_full
```

## 5. Quick Local-vs-Colab Benchmark Cell

Use this on both machines to get an apples-ish runtime snapshot:

```python
import time, torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

torch.manual_seed(0)
x = torch.randn(4096, 4096, device=device)
torch.cuda.synchronize() if device == "cuda" else None
t0 = time.time()
for _ in range(25):
    y = x @ x
torch.cuda.synchronize() if device == "cuda" else None
print("matmul sec:", round(time.time() - t0, 3))
```

This is not a perfect LGN benchmark, but it tells you quickly whether the Colab
runtime you received is worth using for the heavier experiments.
