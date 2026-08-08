# SETUP — environment, installation, verification

> This is the **development** setup (training + analysis). If you only want to
> run the model on images, you need three packages — see §3.
>
> `inference.py` deliberately imports far less than this environment provides,
> because script startup is a scored quantity.

---

## 1. What you need

| | Requirement |
|---|---|
| Python | **3.11** (3.12 works; 3.13 has gaps in `bm3d` / `pyiqa`) |
| GPU | Any CUDA GPU with ≥ 6 GB VRAM. We trained on **8 GB** and peaked at 2.1 GB |
| CUDA | A PyTorch build matching your GPU — see §2, this is the one thing people get wrong |
| Disk | ~3 GB for the dataset, ~1 GB for the memory-mapped training bundle |
| CPU-only | Everything runs, just slowly. Training is impractical; inference is fine |

---

## 2. ⚠️ The one thing that goes wrong: the CUDA build

`pip install torch` gives you a **CPU-only** wheel. It imports fine, reports no
error, and silently runs ~50× slower. Worse, a CUDA build compiled for older
architectures will **refuse to run at all** on a newer GPU.

### Step 1 — find out what your GPU needs

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

That prints your GPU's **compute capability**, e.g. `8.6`, `9.0`, `12.0`.

### Step 2 — install a PyTorch build that supports it

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`cu128` (CUDA 12.8) covers everything from compute capability 7.5 through 12.0,
including the H100 (9.0) that KLA benchmarks on. Older `cu121` / `cu124` wheels
do **not** cover the newest architectures.

### Step 3 — verify, don't assume

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

Expected shape of the answer:

```
2.11.0+cu128 True
['sm_75', 'sm_80', 'sm_86', 'sm_90', ...]
```

**Your GPU's compute capability must appear in that list**, with the dot removed
— capability `7.5` becomes `sm_75`, `8.6` becomes `sm_86`, `9.0` becomes
`sm_90`, and so on.

If `cuda.is_available()` is `False`, or your `sm_XX` is missing, stop here.
Everything downstream will silently run on the CPU and every timing number will
be meaningless.

### Step 4 — prove it actually computes

```bash
python -c "
import torch
a = torch.randn(2048, 2048, device='cuda')
print('matmul ok:', float((a @ a).mean()))
x = torch.randn(4, 3, 64, 64, device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    y = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()(x)
print('bf16 conv ok:', tuple(y.shape))
print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
"
```

A wrong-architecture build fails **here**, not at import.

---

## 3. Installation

### Option A — inference only (3 packages, ~10 s)

```bash
pip install -r requirements-inference.txt
```

`torch`, `numpy`, `opencv-python-headless`. Enough to run `inference.py`.

### Option B — full development environment

```bash
conda create -y -n klasr python=3.11 pip
conda activate klasr

# PyTorch FIRST, from the CUDA index (see §2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# everything else
pip install "numpy<2.4" scipy opencv-python-headless scikit-image scikit-learn \
            matplotlib pandas pyyaml tqdm tensorboard pytest \
            torchmetrics lpips bm3d pyiqa python-pptx pymupdf
```

Or `conda env create -f environment.yml`, then the PyTorch line separately —
the YAML deliberately omits torch so conda cannot substitute a CPU build.

### Option C — Docker

```bash
docker build -t kla-ps01 .
docker run --gpus all -v /host/in:/data/in -v /host/out:/data/out \
    kla-ps01 --input_dir /data/in --output_dir /data/out
```

---

## 4. Versions verified working together

| package | version | used by |
|---|---|---|
| python | 3.11.15 | — |
| **torch** | **2.11.0+cu128** | everything |
| torchvision | 0.26.0+cu128 | training only — **never** in `inference.py` |
| numpy | 2.3.5 | everything |
| scipy | 1.17.1 | forensics (`convolve2d`, `uniform_filter`) |
| opencv-python-headless | 5.0.0.93 | image I/O, forensics |
| scikit-image | 0.26.0 | forensics |
| scikit-learn | 1.9.0 | source clustering |
| torchmetrics | 1.9.0 | PSNR / SSIM / MS-SSIM |
| lpips | 0.1.4 | LPIPS (AlexNet) |
| pyiqa | 0.1.16 | NIQE / BRISQUE / PIQE |
| bm3d | 4.0.3 | classical baseline |
| tensorboard | 2.21.0 | training curves |
| pytest | 9.1.1 | tests |
| python-pptx / pymupdf | 1.0.2 / — | deck build and inspection |

Complete pin list: `requirements.txt` (119 packages, real `pip freeze`).

`pyiqa` pulls in `timm` transitively. Fine for training; `timm` must **never**
be imported by `inference.py`.

---

## 5. Get the data in place

```
data/
├── train/train/
│   ├── GT/          000000.npy … 003199.npy   3200 files · 256×256 float32
│   └── NoisyLR/     000000.npy … 003199.npy   3200 files · 128×128 float32
└── Test_NoisyLR/
    └── NoisyLR/     000000.npy … 000399.npy    400 files · 128×128 float32
```

Then build the memory-mapped training bundle (one-off, ~11 s):

```bash
python scripts/precrop_patches.py
```

This writes `data/processed/{gt,lr}.npy` (~1 GB). It is not optional for
training speed — without it the GPU sits idle ~84 % of the time waiting on file
opens.

`data/` and `brief/` are gitignored. `brief/` in particular holds a
KLA-Confidential-marked deck and must never be published.

---

## 6. Smoke test — is everything working?

```bash
# 1. all dependencies import
python -c "
import importlib
for m in ['torch','numpy','scipy','cv2','skimage','sklearn','torchmetrics','lpips','bm3d','pyiqa']:
    importlib.import_module(m); print(' ok', m)
"

# 2. the test suite (20 tests, ~60 s)
python -m pytest tests/ -q

# 3. inference on the 6 shipped sample images
python inference.py --input_dir sample_test --output_dir /tmp/out
```

If all three pass, the environment is correct.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | CPU-only wheel | Reinstall from the cu128 index (§2) |
| `no kernel image is available for execution` | Wheel predates your GPU | Same — you need a newer CUDA build |
| `CondaError: Run 'conda init'` | Not in an Anaconda shell | Use Anaconda Prompt, not plain cmd |
| `ModuleNotFoundError` | Wrong environment active | `conda activate klasr` |
| Training crawls at <20 img/s | Bundle not built | `python scripts/precrop_patches.py` |
| `CUDA out of memory` | Batch too large for your VRAM | Lower `data.batch_size` in the config; raise `accum_steps` to keep the effective batch |
| Windows: `Can't pickle local object` | Old code path | Fixed; ensure you are on the current commit |

---

## 8. Reproducibility notes

* Seed is `1337`, set in the config and written into every checkpoint.
* Determinism and `cudnn.benchmark` are mutually exclusive. We choose
  **deterministic for training** and **autotune-free for inference** — the
  reasoning is recorded in `06_TECHNICAL_ARCHITECTURE.md` §7.
* Every checkpoint stores the resolved config, the git commit SHA, the seed, the
  torch version and the validation metrics, so any published number can be
  traced back to the exact code that produced it.
