# AI-Based Restoration of Degraded Images
### SEMICON India Hackathon 2026 · KLA Problem Statement 01

Restores grayscale images degraded by **Gamma multi-look speckle + additive
Gaussian noise + 2× downsampling**, at exactly 2× the input resolution.

---

## Results

Validation split held out **by source image**, 80 of 800 sources / 320 samples,
never seen during training. Metrics at `data_range=1.0`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 *(floor)* | 23.067 | 0.5129 | 0.4425 |
| BM3D + bicubic ×2 | 25.956 | 0.6527 | 0.5576 |
| NAFNet-w48, v1 loss | 26.415 | 0.7333 | 0.2455 |
| **NAFNet-w48, HF-weighted loss (shipped)** | **26.616** | **0.7344** | **0.2244** |

Ahead of both baselines on all three scored metrics. Note BM3D's LPIPS (0.558)
is *worse than plain bicubic* (0.443): it buys PSNR by over-smoothing, which
PSNR rewards and LPIPS punishes — the exact failure the problem statement warns
against, quantified.

**Out-of-distribution** (ground truth manufactured with our own reconstructed
degradation, since KLA withholds the test GT):

| Family | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Urban100 — buildings, the OOD case KLA named | 24.587 | 0.7792 | 0.2147 |
| BSD100 — natural scenes | 26.510 | 0.7508 | 0.2543 |
| Set14 | 26.795 | 0.7671 | 0.2071 |

Urban100 drops **2.03 dB** against in-distribution — a genuine generalisation
penalty, reported rather than hidden. BSD100 LPIPS is **2.0 % worse** than the
v1 loss; every other split improved. Both are stated rather than dropped.

---

## ⚡ How to run — pick your situation

Three paths. **They do not overlap — follow exactly one.**
Every block below is copy-paste as-is. No file needs editing, ever.

---

### ▶ A. EVALUATORS — you have an NVIDIA GPU

This is the scored path. Four commands, start to finish.

```bash
git clone https://github.com/Deven1305/GrayScale.git
cd GrayScale
git lfs install && git lfs pull
pip install -r requirements-inference.txt
```

Then run it on your test directory:

```bash
python inference.py --input_dir <YOUR_TEST_DIR> --output_dir <YOUR_OUTPUT_DIR>
```

That is the whole contract. **Zero manual edits.** The GPU is detected
automatically; weights load from a path relative to the script.

Expected output:

```
[info] 400 input images
[info] arch=nafnet_w48 device=cuda startup=1.4s
[done] wrote 400/400 images to outputs
[time] startup 1.4s | warmup 0.03s | total 9.0s | 44.4 img/s
```

**Verify it works before pointing it at real data** — six images ship in the
repo, no dataset download needed:

```bash
python inference.py --input_dir sample_test --output_dir out_sample
```

Should print `wrote 6/6 images` in a few seconds.

> ⚠️ **`git lfs pull` is not optional.** The weights are stored in Git LFS.
> Without it, `weights/model_fp16.pt` arrives as a ~130-byte text pointer and
> the run fails. Check with `ls -la weights/` — it must be **~30 MB**.

---

### ▶ B. CPU ONLY — no NVIDIA GPU

Identical commands; the script detects the absence of a GPU and switches to CPU
by itself. Only the install differs.

```bash
git clone https://github.com/Deven1305/GrayScale.git
cd GrayScale
git lfs install && git lfs pull
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-inference.txt
```

```bash
python inference.py --input_dir sample_test --output_dir out_sample
```

To force CPU explicitly:

```bash
python inference.py --input_dir sample_test --output_dir out_sample --device cpu
```

Roughly **0.6 s per image** on CPU versus 0.02 s on GPU — the full 400-image
test set takes about 4 minutes. Results are numerically equivalent.

---

### ▶ C. GOOGLE COLAB — free T4 GPU

First: **Runtime → Change runtime type → T4 GPU → Save.**
Then paste these five cells in order.

**Cell 1 — confirm the GPU is attached**

```python
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Expect something like `2.11.0+cu128 True Tesla T4`. If it says `False`, the
runtime type was not changed.

**Cell 2 — clone**

```python
!git clone https://github.com/Deven1305/GrayScale.git
%cd GrayScale
```

**Cell 3 — fetch the real weights**

```python
!git lfs install && git lfs pull
!ls -la weights/
```

`model_fp16.pt` must be **~30 MB**. If it is ~130 bytes, LFS did not run.

**Cell 4 — install**

```python
!pip install -q -r requirements-inference.txt
```

> ⚠️ **Do NOT `pip install torch` on Colab.** It already ships a matching CUDA
> build; replacing it breaks the runtime. A `numba`/`numpy` version warning here
> is harmless — nothing in this pipeline uses numba.

**Cell 5 — run, and view the result**

```python
!python inference.py --input_dir sample_test --output_dir out_npy --num_workers 2

import numpy as np, matplotlib.pyplot as plt, glob
a = np.load(sorted(glob.glob('sample_test/*.npy'))[0])
b = np.load(sorted(glob.glob('out_npy/*.npy'))[0])
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(a, cmap='gray', vmin=0, vmax=1); ax[0].set_title(f'input {a.shape}')
ax[1].imshow(b, cmap='gray', vmin=0, vmax=1); ax[1].set_title(f'restored {b.shape}')
for x in ax: x.axis('off')
plt.show()
```

`--num_workers 2` matters: Colab gives 2 CPU cores, and the default of 4 spawns
workers that each re-import torch.

---

## Input and output

| | |
|---|---|
| Input formats | `.npy` float32 (what KLA ships), `.png`, `.tif` |
| Input sizes | **128×128 and 256×256** — both handled, bucketed and warmed separately |
| Output | same base filename, **exactly 2× the input**, clamped to [0,1] |
| Output format | matches the input by default; add `--output_format png` for viewable images |

To get browsable PNGs instead of `.npy`:

```bash
python inference.py --input_dir sample_test --output_dir out_png --output_format png
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `weights/model_fp16.pt` is ~130 bytes | Git LFS pointer, not the file | `git lfs install && git lfs pull` |
| `ModuleNotFoundError: torchmetrics` | Wrong environment active | Only tests need it: `pip install -r requirements.txt` |
| `torch.cuda.is_available()` is `False` on a GPU box | CPU-only wheel installed | Reinstall from the CUDA index (§ Training below) |
| `CUDA error: no kernel image` on Colab | torch was reinstalled | Restart runtime; do not `pip install torch` |
| Very slow on CPU | Expected | ~0.6 s/image vs ~0.02 s on GPU |

---

## Training environment (only if you want to retrain)

Inference needs none of this.

```bash
conda create -y -n klasr python=3.11 pip && conda activate klasr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

⚠️ Install PyTorch from a CUDA index matching your GPU. A plain
`pip install torch` silently gives a CPU-only wheel. See `docs/01_SETUP.md` §2.

**Docker:**

```bash
docker build -t kla-ps01 .
docker run --gpus all -v /host/in:/data/in -v /host/out:/data/out \
    kla-ps01 --input_dir /data/in --output_dir /data/out
```

---

## Method in one page

### The degradation was reconstructed, not assumed

No metadata ships with the dataset, so we reverse-engineered it from the paired
images before writing any model code. Full derivation: `docs/05_FORENSICS_REPORT.md`.

```
y = D_a( x · n + g )      n ~ Gamma(L, 1/L),  g ~ N(0, σ²),  decimation LAST
```

| Finding | Evidence |
|---|---|
| Speckle is **Gamma multi-look**, not Gaussian-multiplicative | negatives enriched **21×** at dark pixels; **100 %** of samples positively skewed |
| Downsampler is a **4-tap cubic convolution** (a ≈ −0.6) | kernel regressed directly from `E[y\|x]`; box/area/bilinear/Lanczos/strided excluded |
| Noise applied **before** decimation, in a **fixed** order | flat-region noise autocorrelation −0.059 (zero if applied after); unimodal, not bimodal |
| **L median 17.7**, σ ∈ [1e-3, 4e-2] | variance regression, calibrated on synthetic ground truth |
| Dataset is **800 sources × 4 crops** | similarity vanishes exactly at index gap 3 |

**The finding that drove the design:** speckle std ≈ 0.24 against additive σ of
0.001–0.04 — multiplicative noise dominates by **6× to 240×**. This is a
despeckling problem that also has additive noise and downsampling, not three
equally weighted problems.

Our replica reproduces the real data to **0.982 histogram overlap**;
`tests/test_degradation.py` is the acceptance gate.

### Architecture — NAFNet with an SR head, 15.24 M params

```
degraded input (never clipped)
   ├─ bicubic ×2 ─────────────────────────┐  residual anchor
   ├─ [x, log x] 2-channel stem            │  speckle → additive
   ↓                                       │
 NAFNet encoder / middle / decoder @ LR    │
   ↓                                       │
 conv → PixelShuffle(2), ICNR init         │
   ↓                                       ↓
 output = anchor + residual ───────────────
```

Processes at LR throughout with a single PixelShuffle at the end (~4× cheaper
than upsampling first); residual on a bicubic anchor; LayerNorm rather than
BatchNorm for OOD stability; fully convolutional so both input sizes work.

### Loss

```
1.0·Charbonnier + 0.6·FFT(high-frequency weighted) + 0.5·high-pass
              + 0.15·(1 − MS-SSIM) + 0.05·gradient + 0.05·VGG
```

Those weights are measured, not guessed. `scripts/f26_loss_spectrum.py`
backpropagates each term alone and reports what fraction of its gradient lands
in the high half of the spectrum. The **original** recipe scored **49.11 %** —
*less* than plain Charbonnier at 52.82 %, meaning the two terms added for
sharpness were making the output softer. The Sobel `gradient` term measured
22.58 %, thirty points below the term it was meant to sharpen.

Rebalancing to **61.31 %** bought **+0.20 dB PSNR and −8.6 % LPIPS at identical
inference cost** — same architecture, same 15.24 M parameters, same 4.6 ms per
image. Full derivation: `docs/13_RESOLUTION_IMPROVEMENT.md`.

**No adversarial loss** — the spec forbids "artificial patterns or ringing",
and GAN training costs PSNR and SSIM, two of the three scored metrics. Ablation
evidence for each term: `docs/09_ABLATION_RESULTS.md`.

---

## Reproducing

```bash
python scripts/precrop_patches.py                    # pack the data
python scripts/build_external_data.py                # DIV2K + OOD families
python scripts/run_baselines.py --limit 120          # the floor
python train.py --config configs/nafnet_w48_sharp.yaml   # ~2.5 h on an 8 GB GPU
python scripts/export_weights.py --ckpt experiments/runs/nafnet_w48_sharp/best.pt \
                                 --out weights/model_fp16.pt
python evaluate.py --ckpt experiments/runs/nafnet_w48_sharp/best.pt --proxy-ood \
       --ood data/external/ood/Urban100 data/external/ood/BSD100
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

Seed `1337`, fixed and logged. Every checkpoint stores the resolved config, the
git commit, the seed, the torch version and the validation metrics.

Full experiment history: `experiments/EXPERIMENT_LOG.md`.

---

## Throughput

Measured runtime includes **script startup and model initialisation**, so
`inference.py` is written for total wall clock:

* minimal imports; OpenCV imported lazily and only for PNG input
* model defined **inline** — no package tree imported to fetch one class
* **no `torch.compile`** (a 30–120 s compile would dominate)
* `weights_only=True, mmap=True`, fp16 weights
* reader → GPU → writer overlap, shape bucketing, per-shape warmup
* GPU-side clamp before transfer back

| Configuration | 400 images | img/s |
|---|---|---|
| 4 DataLoader workers — the usual advice | 35.2 s | 11.4 |
| **shipped defaults** | **9.0 s** | **44.4** |

Two standard recommendations were measured and **reversed**: cudnn autotuning
costs more than it saves at this scale, and DataLoader workers are slower where
they spawn. Both are platform-aware flags rather than hardcoded, since a Linux
H100 may flip the answer back.

---

## Repository layout

```
inference.py          ⚠️ THE SCORED FILE — standalone, dir → dir
train.py              config-driven training
evaluate.py           offline metrics vs baselines and OOD families
configs/              every hyperparameter; nothing hardcoded in src/
src/                  models · data · losses · metrics · engine · utils
tests/                34 tests — metrics, splits, degradation, inference, integrity
weights/              shipped fp16 weights (Git LFS)
outputs/              400 restored test images (Git LFS)
sample_test/          6 images so inference runs without the dataset
docs/                 setup, how-to-run, forensics, analysis, OOD, ablations
experiments/          EXPERIMENT_LOG.md + all result JSON
```

---

## Tests

```bash
python -m pytest tests/ -q      # 34 passed
```

| File | Asserts |
|---|---|
| `test_metrics.py` | `PSNR(x,x) == inf` — the tripwire for a wrong `data_range` |
| `test_splits.py` | no source in both splits; the leakage detector fires |
| `test_degradation.py` | replica matches real histogram, tails, moments |
| `test_inference.py` | dir→dir, exactly 2×, **both 128 and 256 inputs**, survives a corrupt file, no absolute paths, no banned imports |
| `test_repo_integrity.py` | every source file is tracked; KLA data is not |

---

## Known limitations

Stated plainly rather than buried:

* **Over-smooths high-frequency texture.** On dense foliage the model gains
  3.4 dB PSNR while SSIM stays ~0.38 — it removes speckle and fine texture
  together.
* **Fine periodic patterns remain marginally below bicubic** (checkerboard
  15.29 vs 15.64 dB). Adding procedurally generated gratings to training
  narrowed the gap but did not close it. This matters because semiconductor
  structures are periodic.
* **Trained only at 128→256.** The 256→512 path is verified correct and gains
  +1.96 dB over bicubic on Urban100, but the model was never trained at that
  scale.
* **Timing numbers come from one 8 GB consumer GPU.** They will not transfer to
  an H100.

---

## References

1. T. Kumar et al. *Image Data Augmentation Approaches: A Comprehensive Survey.* IEEE Access 12, 2024.
2. L. Zhai et al. *A Comprehensive Review of Deep Learning-Based Real-World Image Restoration.* IEEE Access 11, 2023.
3. J. Terven et al. *A Comprehensive Survey of Loss Functions and Metrics in Deep Learning.* AI Review 58, 2025.
4. V. Monga, Y. Li, Y. C. Eldar. *Algorithm Unrolling.* IEEE SPM 38(2), 2021.
5. L. Chen et al. *Simple Baselines for Image Restoration.* ECCV 2022. (NAFNet)
6. K. Zhang et al. *Designing a Practical Degradation Model for Deep Blind SR.* ICCV 2021. (BSRGAN)
7. L. Sun et al. *Spatially-Adaptive Feature Modulation for Efficient SR.* ICCV 2023. (SAFMN)
8. K. Dabov et al. *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering.* IEEE TIP 2007. (BM3D)
9. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018. (LPIPS)
10. E. Agustsson, R. Timofte. *NTIRE 2017 Challenge on Single Image Super-Resolution.* CVPRW 2017. (DIV2K)

References 1–4 are the four cited by KLA in the problem-statement materials.

---

## Team

| | |
|---|---|
| Team name | *‹ fill in ›* |
| Members | *‹ fill in ›* |
| College | *‹ fill in ›* |
| Contact | *‹ fill in ›* |
