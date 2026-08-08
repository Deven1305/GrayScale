# TECHNICAL ARCHITECTURE
## KLA PS01 — AI-Based Restoration of Degraded Images

How the system is built, why each piece is the way it is, and what runs when.

---

## 1. THE PROBLEM, STATED PRECISELY

Given a degraded image `y` at 128×128, recover the clean image `x` at 256×256.

Reconstructed from the paired data in Phase 0 (`05_FORENSICS_REPORT.md`):

```
y = D_a( x · n + g )

  x    clean image, float32, strictly [0,1]
  n    speckle       ~ Gamma(L, 1/L),  E[n]=1, Var[n]=1/L,  L median 17.7
  g    Gaussian      ~ N(0, σ²),       σ ∈ [1e-3, 4e-2]
  D_a  2× decimation, separable 4-tap cubic convolution, a ≈ −0.6, applied LAST
```

Three properties of this forward model drive every architectural decision:

| Property | Consequence for the design |
|---|---|
| Speckle is **multiplicative** and dominates σ by 6–240× | Log channel; despeckling is the primary job, not one of three equal jobs |
| Noise is injected **before** decimation | The LR noise is spatially **correlated**, so a pixel-independent denoiser is mis-specified |
| Decimation is **cubic with negative lobes** | Some out-of-range values are kernel undershoot, not noise — another reason never to clip |

---

## 2. SYSTEM MAP

```
                        ┌──────────────── TRAINING ────────────────┐
data/train/train/       │                                          │
├── GT/      (256²)     │  KLAPairs ──┐                            │
└── NoisyLR/ (128²)     │             ├─ MixedDataset ─ DataLoader │
                        │  Synthetic ─┘        │                   │
data/processed/         │  Pairs (replica)     │                   │
└── gt.npy, lr.npy ─────┤                      ▼                   │
    (memmap bundle)     │              NAFNetSR (4.95 M)           │
                        │                      │                   │
                        │              CompositeLoss               │
                        │              Charb + MS-SSIM + FFT + grad│
                        │                      │                   │
                        │              AdamW + cosine + EMA        │
                        │                      │                   │
                        │       experiments/runs/*/best.pt         │
                        └──────────────────────┬───────────────────┘
                                               │ export_weights.py
                                               ▼
                                      weights/model_fp16.pt (10 MB)
                                               │
                        ┌──────────────── INFERENCE ───────────────┐
                        │  inference.py  (standalone, inline model)│
                        │  reader ─queue→ GPU ─queue→ writer       │
                        └──────────────────────┬───────────────────┘
                                               ▼
                                          outputs/*.npy (256²)
```

---

## 3. MODEL — `src/models/nafnet.py`

### 3.1 Forward path

```
input (B,1,128,128) float32, NOT clipped, may exceed [0,1]
  │
  ├──────────────────────────────────────────────► bicubic ×2 ──┐  RESIDUAL ANCHOR
  │                                                             │
  ▼ stem: concat[ x , log(clamp_min(x,1e-3)) ]  → (B,2,128,128) │
  │        speckle is multiplicative; log makes it additive     │
  ▼ intro conv 3×3, 2→32                                        │
  │                                                             │
  ├─ ENCODER ────────────────────────────────────────┐          │
  │    NAFBlock ×2  @ 32ch  128×128  ──── skip ──┐   │          │
  │    down conv 2×2 stride 2                    │   │          │
  │    NAFBlock ×2  @ 64ch   64×64   ──── skip ──┼┐  │          │
  │    down conv 2×2 stride 2                    ││  │          │
  │    NAFBlock ×4  @128ch   32×32   ──── skip ──┼┼┐ │          │
  │    down conv 2×2 stride 2                    │││ │          │
  ├─ MIDDLE: NAFBlock ×8 @ 256ch  16×16 ─────────┼┼┤ │          │
  ├─ DECODER (add skips) ────────────────────────┼┼┘ │          │
  │    up(PixelShuffle) + NAFBlock ×2 @128ch ◄───┼┘  │          │
  │    up(PixelShuffle) + NAFBlock ×2 @ 64ch ◄───┘   │          │
  │    up(PixelShuffle) + NAFBlock ×2 @ 32ch ────────┘          │
  │                                                             │
  ▼ SR HEAD: conv 3×3, 32 → 1·2² = 4   (ICNR init)              │
  ▼ PixelShuffle(2)                  → (B,1,256,256)            │
  │                                                             │
  └──────────────────── + ◄─────────────────────────────────────┘
                        │
                        ▼
              output (B,1,256,256)   clamped ONLY at save time
```

### 3.2 Why each choice

| Choice | Reason |
|---|---|
| **Process at LR, one PixelShuffle at the end** | Upsampling first and working at 256² costs ~4× the FLOPs for no quality gain. Largest single throughput decision in the model |
| **Residual on a bicubic anchor** | The network learns only the correction. Converges much faster and transfers better OOD, because the anchor already supplies the low frequencies |
| **Log channel** | `log(x·n) = log x + log n` — multiplicative speckle becomes additive, which is the whole basis of homomorphic despeckling. Costs one extra input channel |
| **LayerNorm, not BatchNorm** | LayerNorm never couples to training-batch statistics. This matters when the test distribution differs from training |
| **SimpleGate instead of activations** | `chunk(2) → a*b`. Cheaper than GELU and empirically better (NAFNet, ECCV 2022) |
| **SCA (pooled 1×1 attention)** | Channel attention at negligible cost |
| **ICNR init on the sub-pixel conv** | PixelShuffle produces checkerboard artefacts at init unless the `scale²` output channels start identical |
| **Reflect-pad to a multiple of 8** | The 3-level encoder needs divisible dimensions; padding is cropped back at 2× |
| **Fully convolutional** | 128→256 and 256→512 both work with no code change |

### 3.3 NAFBlock internals

```
x ─→ LayerNorm ─→ conv1×1 (c→2c) ─→ dwconv3×3 ─→ SimpleGate ─→ ·SCA ─→ conv1×1 ─→ ⊕ ─→
                                                                              β↑    │
  ┌───────────────────────────────────────────────────────────────────────────────┘
  └─→ LayerNorm ─→ conv1×1 (c→2c) ─→ SimpleGate ─→ conv1×1 ─→ ⊕ ─→ out
                                                            γ↑
```

`β` and `γ` are learnable per-channel scalars initialised to **zero**, so every
block starts as an identity map and the network begins life as exactly the
bicubic anchor. This is why training is stable from step one.

### 3.4 Size

| | params | val PSNR | val SSIM | notes |
|---|---|---|---|---|
| **NAFNetSR w32** | **4.95 M** | 25.867 | 0.7050 | shipped |
| NAFNetSR w16 | ~1.3 M | — | — | implemented, not trained |
| SAFMNSR | ~0.24 M | — | — | implemented, not trained |

---

## 4. DATA PIPELINE — `src/data/`

### 4.1 `degradation.py` — the replica

Encodes the Phase 0 findings. The **decoupled** form is the shipped default:

```python
y = D(x, a1) + D(x·(n−1), a2) + D(g, a2)
```

* `a1 ~ U(−0.75, −0.45)` — the **signal** kernel, from regressing `E[y|x]`
* `a2 ~ U(−0.40, −0.15)` — the **noise** kernel, from the flat-region noise
  autocorrelation
* Setting `a1 = a2` collapses to the single-kernel model, so this is a strict
  generalisation, not a different model

The signal and noise paths genuinely fit **different** kernels. Both estimators
were bias-checked on synthetic ground truth, and the diagonal identity
`ρ(1,1) = ρ(0,1)·ρ(1,0)` rules out contamination. We could not identify the
generator step that produces it; it is modelled rather than hidden.

**Order is fixed**, not randomised — the brief says randomised, the data says
otherwise (per-sample autocorrelation is unimodal; randomised controls are
bimodal). `randomize_order=True` exists as an explicit OOD augmentation switch,
off by default.

**Acceptance gate** (`tests/test_degradation.py`), measured against real data:

| statistic | real | replica |
|---|---|---|
| histogram overlap | — | **0.982** |
| fraction > 1 | 0.0288 | 0.0287 |
| fraction < 0 | 0.00282 | 0.00249 |
| flat-region noise variance | 0.0288 | 0.0286 |
| noise lag-1 autocorrelation | −0.0588 | −0.0574 |

### 4.2 `splits.py` — the leakage guard

```python
source_id = sample_index // 4        # 800 sources × 4 crops
```

Every split function asserts disjointness and raises. A random per-sample split
would put overlapping crops of the same photograph on both sides and make the
validation score fiction. `tests/test_splits.py` also asserts the detector
itself fires on a deliberately bad split.

### 4.3 `dataset.py` + the memmap bundle

`KLAPairs` reads from `data/processed/{gt,lr}.npy` (memory-mapped) when
present, falling back to per-file `np.load`. The bundle exists because two file
opens per sample left the GPU idle ~84 % of the time.

`MixedDataset` interleaves real KLA pairs with synthetic pairs at a configurable
ratio (`external_ratio`, currently 1.0 = KLA only).

---

## 5. LOSS — `src/losses/composite.py`

```
L = 1.0·Charbonnier + 0.2·(1 − MS-SSIM) + 0.1·FFT + 0.05·gradient
```

| term | purpose |
|---|---|
| Charbonnier | smooth L1. Pure L2 over-smooths, and the spec forbids blurring to denoise |
| MS-SSIM | directly optimises one of the three scored metrics |
| FFT | L1 on the complex spectrum — restores the band decimation destroyed, without ringing |
| gradient | L1 on Sobel magnitude — keeps edges sharp |
| VGG | implemented, weight 0. Proxy for LPIPS; off because it pulls torchvision into training for a small gain |

**No adversarial term.** The spec forbids "artificial patterns or ringing", and
GAN training costs PSNR and SSIM — two of three scored metrics. The BM3D
baseline demonstrates the opposite failure quantitatively: +2.9 dB PSNR over
bicubic while its LPIPS gets *worse*.

⚠️ `forward()` returns **detached tensors**, not floats. Calling `float()` per
term per step forces a GPU sync and cost a measured 15× throughput
(125 → 8 img/s). The trainer accumulates on-device and syncs once per epoch.

---

## 6. TRAINING ENGINE — `src/engine/trainer.py`

| | |
|---|---|
| Precision | **bf16 autocast** — recent NVIDIA architectures; needs no GradScaler and cannot overflow |
| Optimiser | AdamW, lr 2e-4, betas (0.9, 0.9), wd 1e-4 |
| Schedule | cosine with 2-epoch linear warmup |
| Grad clip | 1.0 — restoration runs spike on outlier crops |
| EMA | decay 0.999 with warmup ramp; **evaluated with EMA** |
| Selection | **best val SSIM**, rule fixed before training started |
| Checkpoint | model, EMA, optimiser, scheduler, epoch, full config, **git SHA**, seed, torch version, metrics |

Determinism vs speed is resolved explicitly: **deterministic for training**
(`cudnn.deterministic=True`, `benchmark=False`), **autotune-free for inference**
(see §7). The two are mutually exclusive; the choice is documented rather than
left implicit.

---

## 7. INFERENCE — `inference.py` (the scored file)

Measured runtime includes **script startup and model initialisation**, so the
file is optimised for total wall clock, not forward-pass FLOPs.

```
files in --input_dir
   │
   ├─ header-only shape scan → buckets {128×128: [...], 256×256: [...]}
   │
   ├─ warmup: one pass per bucket AT THE REAL BATCH SIZE
   │
   └─ per bucket:
         DataLoader workers ──queue──►  GPU main loop  ──queue──►  writer threads
         read .npy (no clip)            H2D → forward → clamp      np.save / cv2
                                        → cast → D2H               (PNG level 1)
```

### 7.1 Startup discipline

| Decision | Saving |
|---|---|
| Imports limited to argparse/glob/os/queue/sys/threading/time/numpy/torch | 1–4 s |
| cv2 imported **lazily**, only if a PNG is actually encountered | ~0.3 s |
| Model class defined **inline**, not imported from `src/` | 1–3 s |
| `torch.load(weights_only=True, mmap=True)` | 0.1–1 s |
| fp16 weight file (10 MB vs 80 MB checkpoint) | 0.1–1 s |
| **No `torch.compile`** | 30–120 s |

Measured startup: **0.4 s**.

### 7.2 Two measured reversals of the plan

`brief/02` recommends `cudnn.benchmark=True` and DataLoader workers. Both are
**losses** on this workload:

| configuration | end-to-end (400 images) | img/s |
|---|---|---|
| benchmark ON, 4 workers | 37.7 s | 10.6 |
| **benchmark OFF, 0 workers** | **17.7 s** | **22.6** |

* Autotuning costs **11.7 s** of warmup for an ~8 % main-loop gain — a 2.4×
  end-to-end loss. It only amortises above roughly 15 000 images.
* On Windows, DataLoader workers spawn and **re-import torch per worker**; 400
  small images never repay that.

Both are flags (`--cudnn_benchmark`, `--num_workers`), not hardcoded, because an
H100 with a larger test set may flip the answer.

### 7.3 Per-image stage cost

| stage | ms/img | share |
|---|---|---|
| compute | 2.58 | 68 % |
| write | 0.90 | 24 % |
| read | 0.21 | 6 % |
| D2H | 0.08 | 2 % |
| H2D | 0.03 | 1 % |

### 7.4 Robustness contract

* output filenames match input basenames
* output resolution is exactly 2× the input
* output clamped to [0,1]; **input never clipped**
* one malformed file is logged and skipped, never fatal
* zero hardcoded absolute paths
* weights resolved relative to the script's own directory

---

## 8. WHAT RUNS WHEN

| Command | Reads | Writes |
|---|---|---|
| `scripts/precrop_patches.py` | `data/train/train/` | `data/processed/*.npy` |
| `scripts/run_baselines.py` | val split | `experiments/baselines.json` |
| `train.py --config configs/nafnet_w32.yaml` | bundle + config | `experiments/runs/*/best.pt` |
| `scripts/export_weights.py` | `best.pt` | `weights/model_fp16.pt` |
| `evaluate.py --ckpt ... --proxy-ood` | `best.pt`, val | `experiments/eval_results.json` |
| `inference.py --input_dir --output_dir` | `weights/`, images | restored images |
| `scripts/benchmark_throughput.py` | `weights/`, test | `experiments/throughput.json` |
| `scripts/consistency_check.py` | `best.pt`, test | `experiments/consistency.json` |
| `scripts/r01_visual_report.py` | `best.pt`, all splits | `docs/figures/report_*.png` |

---

## 9. REPOSITORY LAYOUT

```
inference.py          ⚠️ scored — standalone, inline model, minimal imports
train.py              config-driven entrypoint
evaluate.py           offline metrics vs baselines + proxy-OOD
configs/*.yaml        every hyperparameter; none hardcoded in src/
src/
├── models/           nafnet.py · safmn.py · blocks.py · registry.py
├── data/             degradation.py · dataset.py · splits.py
├── losses/           composite.py
├── metrics/          full_reference.py (data_range=1.0) · no_reference.py
├── engine/           trainer.py · ema.py
└── utils/            seed.py (determinism + git hash)
tests/                metrics · splits · degradation · inference e2e
scripts/              f* forensics · v* viewer · d* deck · r* report · utilities
docs/                 forensics_report · DATASET · SETUP · TECHNICAL_ARCHITECTURE
                      · IMPROVEMENTS · figures/
experiments/          EXPERIMENT_LOG.md · baselines · eval · throughput · consistency
weights/              model_fp16.pt (Git LFS)
outputs/              400 restored test images (Git LFS)
```

---

## 10. KNOWN ARCHITECTURAL LIMITATIONS

Measured, not speculated — see `10_IMPROVEMENTS.md` for the evidence and fixes.

1. **Over-smooths high-frequency texture.** The dominant failure mode.
2. **Fine periodic patterns are worse than bicubic** (checkerboard: 15.20 vs
   15.64 dB). Directly relevant, since semiconductor structures are periodic.
3. **Trained at one scale only** (128→256); fully convolutional but untested at
   256→512.
4. **No external training content** — no Urban100/DTD/DIV2K mixing yet.
5. **Log channel, decoupled replica and residual anchor are all unablated** — we
   cannot yet attribute the gain to any one of them.
