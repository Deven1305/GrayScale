# AI-Based Restoration of Degraded Images
### SEMICON India Hackathon 2026 · KLA Problem Statement 01

Restores grayscale images degraded by **Gamma multi-look speckle + additive Gaussian noise + 2× downsampling**, at exactly 2× the input resolution.

---

## 📁 Submission Structure

```
GrayScale/
├── run.py                 # Primary entry script: python run.py <input-dir> <output-dir>
├── requirements.txt       # Full dependencies with pinned versions
├── requirements-inference.txt # Minimal inference runtime
├── README.md              # Setup and execution guide
├── models/                # Trained model weights
│   └── model_fp16.pt
├── weights/               # Compatibility weights folder
│   └── model_fp16.pt
├── sample_test/           # Standalone sample .npy images for instant verification
├── src/                   # Core restoration library
├── configs/               # Hyperparameter definitions
├── train.py               # Model training script
├── evaluate.py            # Comprehensive evaluation pipeline
└── tests/                 # Unit and regression test suite
```

---

## ⚡ How to Run

### Standard Solution Execution

```bash
python run.py <input-dir> <output-dir>
```

#### Verified Submission Guarantees
- ✅ **Input Processing**: Automatically reads all `.npy` (and standard image) files from `<input-dir>`.
- ✅ **Directory Creation**: Creates `<output-dir>` automatically if it does not already exist.
- ✅ **1-to-1 Mapping**: Generates one restored `.npy` file per input file with the exact matching base filename.
- ✅ **Correct Output Format**: Output arrays are 2D grayscale float32 arrays with shape `(2H, 2W)`.
- ✅ **Value Range**: Output values are strictly clamped to `[0.0, 1.0]` and guaranteed free of NaN or Inf values.
- ✅ **Target Resolution**: Exactly 2× super-resolution (`128×128 -> 256×256`, `256×256 -> 512×512`).
- ✅ **Self-Contained & Offline**: Shipped directly in `models/model_fp16.pt`. Runs without requiring internet access, API keys, additional downloads, user interaction, or manual configuration.

---

### ▶ A. EVALUATORS — NVIDIA GPU (CUDA)

This is the standard evaluated path. Four commands, start to finish:

```bash
git clone https://github.com/Deven1305/GrayScale.git
cd GrayScale
git lfs install && git lfs pull
pip install -r requirements-inference.txt
```

Run on your test directory:

```bash
python run.py <YOUR_TEST_DIR> <YOUR_OUTPUT_DIR>
```

*(Optional named arguments `python run.py --input_dir <YOUR_TEST_DIR> --output_dir <YOUR_OUTPUT_DIR>` and legacy `python inference.py` are also fully supported).*

Expected output:

```
[INFO] Discovered 400 input files in '<YOUR_TEST_DIR>'
[INFO] Using model weights: .../models/model_fp16.pt
[INFO] Execution device: cuda (CUDA available: True)
[INFO] Architecture: nafnet_w48 | Scale: 2x | Init time: 1.2s
[DONE] Successfully generated 400/400 restored .npy files in '<YOUR_OUTPUT_DIR>'
[METRICS] Total time: 8.8s | Throughput: 45.5 images/s
```

**Instant Verification** — six images ship inside `sample_test/`:

```bash
python run.py sample_test out_sample
```

---

### ▶ B. CPU ONLY — no NVIDIA GPU

The script automatically detects CPU and adjusts execution:

```bash
git clone https://github.com/Deven1305/GrayScale.git
cd GrayScale
git lfs install && git lfs pull
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-inference.txt
```

```bash
python run.py sample_test out_sample
```

---

### ▶ C. GOOGLE COLAB — free T4 GPU

First: **Runtime → Change runtime type → T4 GPU → Save.**
Then run:

```bash
!git clone https://github.com/Deven1305/GrayScale.git
%cd GrayScale
!git lfs install && git lfs pull
!pip install -q -r requirements-inference.txt
!python run.py sample_test out_npy
```

---

## 📊 Results

Validation split held out **by source image**, 80 of 800 sources / 320 samples, never seen during training. Metrics at `data_range=1.0`.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 *(floor)* | 23.067 | 0.5129 | 0.4425 |
| BM3D + bicubic ×2 | 25.956 | 0.6527 | 0.5576 |
| NAFNet-w48, v1 loss | 26.415 | 0.7333 | 0.2455 |
| **NAFNet-w48, HF-weighted loss (shipped)** | **26.616** | **0.7344** | **0.2244** |

**Out-of-distribution** (reconstructed degradation benchmark):

| Family | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Urban100 — buildings | 24.587 | 0.7792 | 0.2147 |
| BSD100 — natural scenes | 26.510 | 0.7508 | 0.2543 |
| Set14 | 26.795 | 0.7671 | 0.2071 |

---

## Input and Output Specification

| Property | Details |
|---|---|
| Input formats | `.npy` float32, `.png`, `.tif` |
| Input sizes | **128×128 and 256×256** — fully convolutional, bucketed per shape |
| Output files | Same base filename, **exactly 2× the input resolution**, strictly float32 `[0.0, 1.0]` |
| Output format | Grayscale 2D array `.npy` |

---

## Method Summary

### Reconstructed Degradation Pipeline
```
y = D_a( x · n + g )      n ~ Gamma(L, 1/L),  g ~ N(0, σ²),  decimation LAST
```
- Multiplicative multi-look Gamma speckle dominates additive Gaussian noise by **6× to 240×**.
- Downsampler: 4-tap bicubic convolution ($a \approx -0.6$).

### Architecture & Loss
- **Model**: NAFNetSR (width=48, 15.24M parameters), log-transform dual-channel input stem, residual learning on bicubic anchor, single PixelShuffle(2) at LR exit.
- **Loss**: $1.0 \cdot \text{Charbonnier} + 0.6 \cdot \text{FFT}_{\text{HF}} + 0.5 \cdot \text{HighPass} + 0.15 \cdot (1 - \text{MS-SSIM}) + 0.05 \cdot \text{Grad} + 0.05 \cdot \text{VGG}$.

---

## Tests

```bash
python -m pytest tests/ -q
```

---

## Team

| | |
|---|---|
| Team name | *Grayscale* |
| Members | *Deven Mahajan, Osh Manoj Kumar, Nisheet Lad, Satyam Katkar* |
| College | *K J Somaiya School Of Engineering, Vidyavihar* |
| Repository | *https://github.com/Deven1305/GrayScale.git* |
