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
| **NAFNet-w48-sharp (ours)** | **26.616** | **0.7344** | **0.2244** |

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
penalty, reported rather than hidden.

---

## ⚡ Quick start — the exact command to run

```bash
git clone https://github.com/Deven1305/GrayScale.git
cd GrayScale
pip install -r requirements-inference.txt
python inference.py --input_dir <test_images_dir> --output_dir <output_dir>
```

That is the whole contract. **Zero manual edits.** Weights load from a path
relative to the script.

Try it immediately on the six images shipped in `sample_test/` — no dataset
download required:

```bash
python inference.py --input_dir sample_test --output_dir out_sample
```

**On the released test set:**

```bash
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

```
[info] 400 input images
[info] arch=nafnet_w48 device=cuda startup=1.4s
[done] wrote 400/400 images to outputs
[time] startup 1.4s | warmup 0.03s | total 9.0s | 44.4 img/s
```

### Input and output

| | |
|---|---|
| Input formats | `.npy` float32 (what KLA ships), `.png`, `.tif` |
| Input sizes | **128×128 and 256×256** — both handled, bucketed and warmed separately |
| Output | same base filename, **exactly 2× the input**, clamped to [0,1] |
| No GPU? | Runs on CPU automatically. See `docs/02_RUN_WITHOUT_GPU.md` |

---

## Installation

**Inference only — 3 packages:**

```bash
pip install -r requirements-inference.txt
```

**Full training environment:**

```bash
conda create -y -n klasr python=3.11 pip && conda activate klasr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

⚠️ Install PyTorch from a CUDA index matching your GPU's compute capability. A
plain `pip install torch` silently gives a CPU-only wheel. Details and
verification steps: `docs/01_SETUP.md` §2.

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
1.0·Charbonnier + 0.6·FFT(hf_power=1.5) + 0.5·HighFreq + 0.15·(1−MS-SSIM) + 0.05·gradient + 0.05·VGG
```

**No adversarial loss** — the spec forbids "artificial patterns or ringing",
and GAN training costs PSNR and SSIM, two of the three scored metrics. Ablation
evidence for each term: `docs/09_ABLATION_RESULTS.md`.

---

## Reproducing

```bash
python scripts/precrop_patches.py                    # pack the data
python scripts/build_external_data.py                # DIV2K + OOD families
python scripts/run_baselines.py --limit 120          # the floor
python train.py --config configs/nafnet_w48_sharp.yaml     # ~2.5 h on an 8 GB GPU
python scripts/export_weights.py --ckpt experiments/runs/nafnet_w48_sharp/best.pt \
                                 --out weights/model_fp16_sharp.pt
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
| Team name | *Grayscale* |
| Members | *Deven Mahajan, Osh Manoj Kumar, Nisheet Lad, Satyam Katkar* |
| College | *K J Somaiya School Of Engineering, Vidyavihar* |
