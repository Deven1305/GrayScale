# RUNNING WITHOUT AN NVIDIA GPU
## For teammates on laptops with integrated graphics, AMD, or Apple silicon

Only one machine on the team has an NVIDIA GPU. This guide covers everyone
else. **You can do almost everything without one** — the only thing that is
genuinely impractical on CPU is training from scratch, and Colab solves that
for free.

---

## What works where

| Task | CPU laptop | Free Colab (T4) | Notes |
|---|---|---|---|
| **Run inference** on the test set | ✅ ~4 min | ✅ ~25 s | 400 images |
| Run inference on a few images | ✅ seconds | ✅ | |
| View the data (`v01_view.py`) | ✅ | ✅ | pure numpy/matplotlib |
| Run the test suite | ✅ ~2 min | ✅ | 34 tests, no GPU needed |
| All Phase 0 forensics (`f01`–`f25`) | ✅ | ✅ | CPU-only by design |
| Compute baselines (bicubic, BM3D) | ✅ ~5 min | ✅ | |
| Evaluate a checkpoint | ✅ slow | ✅ | |
| **Train the model** | ❌ days | ✅ ~2 h | use Colab |
| Reproduce the throughput numbers | ❌ | ⚠️ different GPU | numbers won't match ours |

---

## OPTION A — CPU only, on your own laptop

Everything except training. This is enough to review the work, look at the
data, verify the outputs and run the tests.

### 1. Install (no CUDA involved)

```bash
conda create -y -n klasr python=3.11 pip
conda activate klasr

# CPU build of PyTorch — small, ~200 MB instead of ~2.5 GB
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-inference.txt   # for inference only
# or, for everything:
pip install "numpy<2.4" scipy opencv-python-headless scikit-image scikit-learn \
            matplotlib pandas pyyaml tqdm tensorboard pytest \
            torchmetrics lpips bm3d
```

### 2. Verify

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

`cuda: False` is **expected and fine** here.

### 3. Run inference

```bash
python inference.py --input_dir sample_test --output_dir out_sample
```

The script auto-detects the absence of a GPU and runs on CPU. Nothing to
configure. To be explicit:

```bash
python inference.py --input_dir sample_test --output_dir out_sample --device cpu
```

Full test set on CPU takes roughly 4 minutes:

```bash
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs --device cpu
```

### 4. Run the tests

```bash
python -m pytest tests/ -q          # expect 34 passed
```

The inference tests already force `--device cpu`, so they pass identically on
every machine.

### 5. Look at the data

```bash
python scripts/v01_view.py pair --index 2204
python scripts/v01_view.py export --split test --limit 40
```

---

## OPTION B — Google Colab (free T4 GPU)

Use this to train, or just to get GPU speed without owning one.

### Quick start

1. Go to **https://colab.research.google.com**
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**
3. Paste the cells below.

### Cell 1 — clone and install

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>

# Colab already ships a CUDA build of PyTorch; do NOT reinstall torch
!pip install -q torchmetrics lpips bm3d pyyaml opencv-python-headless

import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

⚠️ **Do not `pip install torch` on Colab.** It already has a matching CUDA
build; replacing it usually breaks the runtime.

### Cell 2 — inference on the shipped sample (no dataset needed)

```python
!python inference.py --input_dir sample_test --output_dir out_sample
```

Weights are in the repo via Git LFS. If `weights/model_fp16.pt` comes down as a
small text pointer instead of ~30 MB, run:

```python
!git lfs install && git lfs pull
```

### Cell 3 — look at a result

```python
import numpy as np, matplotlib.pyplot as plt, glob
inp = sorted(glob.glob('sample_test/*.npy'))[0]
out = sorted(glob.glob('out_sample/*.npy'))[0]
a, b = np.load(inp), np.load(out)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(a, cmap='gray', vmin=0, vmax=1); ax[0].set_title(f'input {a.shape}')
ax[1].imshow(b, cmap='gray', vmin=0, vmax=1); ax[1].set_title(f'restored {b.shape}')
for x in ax: x.axis('off')
plt.show()
```

### Cell 4 — mount the dataset (only if you want to train)

The dataset is ~2 GB and is **not** in the repo. Put it in your own Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data
!ln -s "/content/drive/MyDrive/kla_data/train" data/train
!ln -s "/content/drive/MyDrive/kla_data/Test_NoisyLR" data/Test_NoisyLR
!ls data/train/train/GT | head -3
```

### Cell 5 — train

```python
!python scripts/precrop_patches.py          # one-off, ~1 min
!python train.py --config configs/nafnet_w32.yaml --epochs 20
```

**Colab-specific advice, learned the hard way:**

* A **free T4 has 16 GB VRAM** — more than the 8 GB the shipped config assumes,
  so `configs/nafnet_w48.yaml` fits comfortably. But free Colab **disconnects
  after ~90 minutes idle and caps sessions around 12 h**, so keep runs short.
* **Checkpoint to Drive**, or you will lose the run:
  ```python
  !python train.py --config configs/nafnet_w32.yaml \
      --tag colab --epochs 20
  !cp -r experiments/runs/nafnet_w32_colab /content/drive/MyDrive/
  ```
* A T4 is roughly **2–3× slower** than the GPU our numbers came from, and has no
  bf16 — the trainer falls back to fp16 with a GradScaler automatically.
* **Do not compare throughput numbers measured on Colab with ours.** Different
  GPU, shared machine, virtualised I/O. Quality metrics are comparable; speed
  is not.

### Cell 6 — evaluate

```python
!python evaluate.py --ckpt experiments/runs/nafnet_w32_colab/best.pt
```

---

## OPTION C — Kaggle Notebooks

Similar to Colab, with a longer runtime (9 h) and a P100 or T4×2. Same
instructions; enable the GPU under *Settings → Accelerator*, and note that
Kaggle's internet access is off by default, which blocks
`build_external_data.py`.

---

## What each teammate can usefully do

| You have | Do this |
|---|---|
| Any laptop | Run the tests, run inference on `sample_test/`, read the docs, look at the data with `v01_view.py` |
| Any laptop + the dataset | Run the full test-set inference on CPU, run the Phase 0 forensics, compute baselines |
| Colab account | Train a variant, run an ablation row, evaluate a checkpoint |
| The NVIDIA machine | Anything requiring reproducible timing — the throughput numbers must all come from one machine |

---

## Frequently hit problems

| Symptom | Cause | Fix |
|---|---|---|
| `weights/model_fp16.pt` is ~130 bytes | Git LFS pointer, not the file | `git lfs install && git lfs pull` |
| `CUDA error: no kernel image` on Colab | You reinstalled torch | Restart runtime; don't reinstall torch |
| `Killed` during training on Colab | RAM exhausted | Lower `data.batch_size` and `epoch_len` |
| Session dies mid-training | Colab idle timeout | Checkpoint to Drive; shorten epochs |
| Inference very slow on CPU | Expected | ~0.6 s/image on CPU vs ~0.02 s on GPU |
| `bm3d` fails to install | No wheel for your Python | Only needed for baselines; skip it |

---

## The honest summary

**You do not need a GPU to contribute.** Reviewing the code, checking the
outputs, running the tests, doing forensics and reading the analysis all work on
any laptop. Training needs one, and free Colab provides it.

The one thing that must stay on a single machine is **throughput
benchmarking** — those numbers are only comparable when measured on identical
hardware, which is why every timing figure in this repository comes from one
GPU and says so.
