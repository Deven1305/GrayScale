# END-TO-END ANALYSIS
## Input data → trained model → benchmark comparison → where it lacks

Four stages, in the order they happened. Every number is measured and
reproducible; nothing here is estimated.

```
  STAGE 1              STAGE 2              STAGE 3              STAGE 4
  ┌────────┐          ┌────────┐          ┌────────┐          ┌────────┐
  │  EDA   │─────────▶│ TRAIN  │─────────▶│ BENCH  │─────────▶│  GAPS  │
  │ what   │          │ can we │          │ vs     │          │ where  │
  │ is the │          │ undo   │          │ classi-│          │ it     │
  │ data?  │          │ it?    │          │ cal +  │          │ fails  │
  └────────┘          └────────┘          │ generic│          └────────┘
   §1                  §2                 └────────┘           §5
                                           §3, §4
```

---

# STAGE 1 — INPUT DATA ANALYSIS (EDA)

## 1.1 What we were given

| | count | shape | dtype |
|---|---|---|---|
| Training targets (`GT`) | 3200 | 256×256 | float32 |
| Training inputs (`NoisyLR`) | 3200 | 128×128 | float32 |
| Test inputs | 400 | 128×128 | float32 |
| **Metadata / labels** | **0** | — | **none shipped** |

No CSV, no JSON, no manifest. The degradation had to be reverse-engineered
before any model code was written.

## 1.2 Value distributions — the first surprise

| statistic | GT | NoisyLR | Test |
|---|---|---|---|
| min | **0.000000** | **−0.278563** | −0.224881 |
| max | **1.000000** | **2.158005** | 2.158016 |
| mean | 0.4335 | 0.4335 | 0.4427 |
| std | 0.2726 | 0.2848 | 0.2843 |
| fraction < 0 | 0 | 0.28 % | 0.66 % |
| fraction > 1 | 0 | **3.11 %** | 3.08 % |

GT is *exactly* [0, 1] across all 209.7 M pixels. The degraded images are not —
and that is not corruption. Speckle is multiplicative, so a bright pixel times a
noise value above 1 exceeds 1.

> **Design consequence #1: never clip the input.** 3.11 % of pixels carry
> out-of-range values that encode local noise strength. Clipping deletes the
> single most useful cue a denoiser has.

The test split is *not* identically distributed: slightly brighter (0.443 vs
0.434) with **2.3× more negative pixels**. A small but real distribution shift.

## 1.3 Which noise model? — the diagnostic that needed sharpening

The brief proposed a simple test: `(NoisyLR < 0).mean()` ≈ 0 means Gamma,
"a few percent" means Gaussian-multiplicative.

**Measured: 0.285 %.** Neither. And the global minimum of −0.279 is ~31σ under
the expected σ. The test was inconclusive, so we asked a sharper question:
*where* do the negatives live?

| local GT brightness | at negative pixels | everywhere |
|---|---|---|
| mean | **0.0208** | 0.4336 |
| median | 0.0054 | 0.4129 |

**Negatives are enriched 21× at dark pixels.** Under Gaussian-multiplicative
(`y = x(1+n)`), a negative requires `n < −1`, which has a fixed probability
*independent of brightness* — so the two distributions would match. They do not,
and the observed rate is 150× higher than that model predicts.

Confirmed positively, two more ways:

| test | observed | Gamma predicts | Gaussian predicts |
|---|---|---|---|
| samples with **positive** skew | **100.0 %** | ~100 % | ~50 % |
| excess-kurtosis / (6·var) | **1.061** | 1.00 | 0 |

> **Conclusion: Gamma multi-look speckle**, `n ~ Gamma(L, 1/L)`.

## 1.4 The downsampling kernel — recovered, not guessed

Because the noise is conditionally zero-mean, `E[y|x] = D(x)` exactly. So the
kernel can be **regressed** rather than selected from a list:

```
recovered 1-D taps :  [-0.016  0.024 -0.083  0.581  0.568 -0.080  0.016 -0.009]
cubic conv a=-0.75 :  [ 0      0     -0.094  0.594  0.594 -0.094  0      0    ]
```

| candidate | rms tap error |
|---|---|
| **cubic convolution, a ≈ −0.6** | **0.017** |
| Lanczos-4 | 0.052 |
| box / area / bilinear | 0.057 |
| PIL bicubic (antialiased) | 0.170 |
| strided `x[::2,::2]` | 0.253 |

The recovered kernel has **negative side lobes**, which alone excludes box,
area, bilinear and strided decimation — all non-negative.

> **Design consequence #2:** some out-of-range values are kernel *undershoot*,
> not noise. Another reason not to clip.

## 1.5 Operation order — the brief was wrong

`brief/00` cites the webinar: *"they may appear in any order."* We measured the
flat-region noise autocorrelation, which is zero if noise is added *after*
decimation and negative if added *before*:

| | median lag-1 autocorr | distribution shape |
|---|---|---|
| **real data** | **−0.062** | tight, **unimodal**, IQR [−0.071, −0.050] |
| synthetic, randomised order | −0.091 | clearly **bimodal** |
| synthetic, noise after | ~0.000 | unimodal at zero |

90.5 % of real samples sit in a narrow central band. A randomised order would
split the corpus into two separated populations — the control shows exactly
that; the data does not.

> **Conclusion: a single fixed pipeline — noise at HR, decimation last.**
> Order randomisation is kept as an OOD *augmentation* flag, off by default.

## 1.6 Noise magnitude — the finding that drives the whole design

| component | magnitude |
|---|---|
| speckle std = 1/√L | **≈ 0.238** (L median 17.7) |
| Gaussian σ | 0.001 – 0.04 |
| **ratio** | **6× to 240×** |

> **Design consequence #3:** the problem statement lists three degradations as
> if equal. They are not. This is a **despeckling** problem that also has
> additive noise and downsampling.

## 1.7 Dataset structure — 800 photographs, not 3200 images

Similarity between samples by index gap:

| gap | 1 | 2 | **3** | 4 | 8 | 20 |
|---|---|---|---|---|---|---|
| excess similarity | +0.290 | +0.144 | **−0.002** | −0.007 | −0.003 | −0.006 |

Correlation vanishes **exactly at gap 3**. Resolving by position mod 4:

| pair | thumbnail | intensity hist |
|---|---|---|
| (4k, 4k+1) | −0.011 | +0.111 |
| (4k+1, 4k+2) | **+0.602** | **+0.843** |
| (4k+2, 4k+3) | **+0.572** | **+0.839** |
| (4k+3, 4k+4) | **−0.006** | **−0.008** ← source boundary |

> **Design consequence #4:** `source_id = index // 4`, **800 sources**.
> A random split puts overlapping crops of the same photograph on both sides
> and turns the validation score into fiction.

## 1.8 What the images actually contain

**Ordinary photographs** — buildings, foliage, water, brick, fabric, people,
signage. One crop even carries a photographer's watermark. 800 sources is
exactly the size of DIV2K.

They are **not** semiconductor images, and **not** the dendrite/microscopy
pictures in KLA's slides — those come from a *different release* (512→256).

Clustering the 800 sources produces **tonal strata, not content families**:
13.2 % are dark (mean < 0.25), 3.1 % are flat (std < 0.10). There is no
"texture vs dendrite vs photo" taxonomy to hold out.

## 1.9 The replica — EDA's deliverable

```python
y = D_a( x · n + g )    n ~ Gamma(L,1/L), g ~ N(0,σ²), decimation LAST
```

Validated against the real data:

| statistic | real | replica |
|---|---|---|
| **intensity histogram overlap** | — | **0.982** |
| fraction > 1 | 0.0288 | 0.0287 |
| fraction < 0 | 0.00282 | 0.00249 |
| flat-region noise variance | 0.0288 | 0.0286 |
| noise lag-1 autocorrelation | −0.0588 | −0.0574 |

This is what makes everything downstream possible: **we can now manufacture
ground truth for any image**, which is exactly what KLA withheld.

---

# STAGE 2 — TRAINING ANALYSIS

## 2.1 Configuration

| | |
|---|---|
| Model | NAFNet + SR head, **4.95 M** params |
| Input | 64×64 LR crops → 128×128 HR, 8-fold dihedral |
| Loss | `1.0 Charbonnier + 0.2 (1−MS-SSIM) + 0.1 FFT + 0.05 gradient` |
| Optimiser | AdamW, lr 2e-4, cosine + 2-epoch warmup, clip 1.0 |
| Precision | bf16 autocast |
| EMA | 0.999, evaluated with EMA |
| Split | **by source**, 720 train / 80 val |
| Selection | **best val SSIM**, rule fixed before training |
| Epochs | 40 (selected epoch 36) |
| Peak VRAM | **2.1 GB of 8 GB** |

## 2.2 The learning curve

| epoch | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 2 | 22.980 | 0.5071 | 0.4371 |
| 4 | 24.684 | 0.5939 | 0.3883 |
| 8 | 25.260 | 0.6376 | 0.3639 |
| 12 | 25.642 | 0.6726 | 0.3365 |
| 18 | 25.788 | 0.6919 | 0.3216 |
| 26 | 25.850 | 0.7013 | 0.3092 |
| 30 | 25.867 | 0.7039 | 0.3043 |
| **36** | **25.867** | **0.7050** | **0.3006** |

It passed bicubic by **epoch 4**. PSNR plateaued around epoch 26; **SSIM and
LPIPS were still improving when training stopped.**

> **Read that again:** we stopped an improving model at 40 epochs while using a
> quarter of the available VRAM. This is the cheapest available gain.

## 2.3 Two throughput bugs found during training

The first run trained at **8 img/s** against **125 img/s** of pure GPU compute —
the GPU was idle ~94 % of the time.

| cause | fix | result |
|---|---|---|
| `CompositeLoss` called `float()` on every term every step → 5 GPU syncs/iteration | return detached tensors, sync once per epoch | 8 → **20 img/s** |
| two `np.load` calls per sample at random offsets | pack into a memory-mapped bundle | 20 → **136 img/s** |

**17× total, with no change to the model.** Worth recording because the same
reasoning drove the inference pipeline.

---

# STAGE 3 — BENCHMARK COMPARISON

## 3.1 Against classical baselines

320 held-out validation images, split by source, `data_range=1.0`:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MS-SSIM ↑ |
|---|---|---|---|---|
| Bicubic ×2 *(floor)* | 23.067 | 0.5129 | 0.4425 | 0.7770 |
| BM3D + bicubic ×2 | **25.956** | 0.6527 | 0.5576 | 0.8451 |
| **NAFNet-w32 (ours)** | 25.867 | **0.7050** | **0.3006** | **0.8976** |

**vs bicubic:** +2.80 dB, +0.192 SSIM, −0.142 LPIPS — better on all three.

**vs BM3D:** +0.052 SSIM, **−0.257 LPIPS**, but **−0.09 dB PSNR**.

## 3.2 Why BM3D wins PSNR and loses everything else

This is the most instructive row in the table.

BM3D gains **+2.9 dB over bicubic while its LPIPS gets 0.11 WORSE** — worse than
doing nothing but upsampling. That is the exact signature of over-smoothing:

* **PSNR** is mean-squared error in disguise. Blurring reduces squared error
  because it moves every pixel toward the local mean. PSNR *rewards* blur.
* **LPIPS** compares deep features. Blur destroys the texture those features
  respond to, so LPIPS *punishes* blur.

The problem statement says: *"Do not blur the image to remove noise — that
destroys useful information."* The BM3D row is the quantitative proof of why,
and it is why we did not optimise PSNR alone.

> **A model that maximised PSNR would be the wrong model for this task.**

## 3.3 Speed

| Configuration | End-to-end, 400 images | img/s |
|---|---|---|
| `cudnn.benchmark=True`, 4 workers — the standard advice | 37.7 s | 10.6 |
| **`benchmark=False`, 0 workers — measured optimum** | **17.7 s** | **22.6** |

Per-image: compute 2.58 ms · write 0.90 ms · read 0.21 ms · D2H 0.08 ms ·
H2D 0.03 ms. Startup 0.4 s.

Two recommendations from the brief, measured and reversed:

* **cudnn autotuning costs 11.7 s** of warmup for an ~8 % main-loop gain — a
  **2.4× end-to-end loss** at this scale. It only amortises above ~15 000 images.
* **More DataLoader workers made it slower** — on Windows, spawn re-imports
  torch per worker.

Both are now flags, not hardcoded, because an H100 with a larger test set may
flip the answer back.

## 3.4 Against a general-purpose generative restorer

> ⚠️ **Not measured here.** We did not run a diffusion or GAN restorer. The
> comparison below is from published literature and from the task constraints —
> it is reasoning, not a benchmark row, and is labelled as such.

The obvious question is: *why not just use a big generative model that
"restores" images?* Three reasons, in order of decisiveness.

**1. The specification forbids it.** *"without introducing artificial patterns
or ringing."* Generative restorers work by *inventing* plausible detail. On a
metrology image, invented detail is a fabricated measurement.

**2. It would score worse on two of the three metrics.** The perception–
distortion tradeoff (Blau & Michaeli, CVPR 2018) is a proved bound: you cannot
improve perceptual quality and distortion simultaneously past a limit.
GAN/diffusion restorers typically trade **1–3 dB of PSNR** for better LPIPS.
Since scoring is *SSIM + pSNR + LPIPS combined*, giving up PSNR and SSIM to win
LPIPS is a losing trade here.

**3. It would lose the throughput axis outright.** A diffusion restorer runs
20–1000 denoising steps per image. Our model runs **one forward pass, 2.58 ms**.
Even at 20 steps a comparable network is ~50 ms/image before I/O — and script
startup for a large generative stack is seconds, which is *also* scored.

| | our specialist | typical generative restorer |
|---|---|---|
| detail source | recovered from the input | **partly invented** |
| PSNR / SSIM | optimised directly | typically 1–3 dB worse |
| LPIPS | good (0.301) | usually better |
| forward passes | **1** | 20–1000 |
| spec-compliant | ✅ | ❌ "artificial patterns" |

**The honest framing:** a generative model would probably produce images that
*look* nicer to a casual viewer, and would score worse on this task while
possibly fabricating structure. For restoration feeding a *measurement*, our
choice is the correct one — but it is a deliberate trade, not a free win, and
the over-smoothing in §5.1 is the price we pay for it.

---

# STAGE 4 — OUT-OF-DISTRIBUTION ANALYSIS

Content that appears **nowhere** in the training corpus of 800 photographs.
Ground truth exists because we generate it and degrade it with our validated
replica.

| Content | bicubic PSNR | ours PSNR | bicubic SSIM | ours SSIM | verdict |
|---|---|---|---|---|---|
| smooth gradient | 25.20 | **38.28** | 0.385 | **0.952** | ✅ +13 dB |
| text-like strokes | 17.70 | **19.57** | 0.342 | **0.694** | ✅ strong |
| geometric shapes | 25.00 | **26.05** | 0.839 | **0.936** | ✅ good |
| **fine periodic grid** | **15.64** | 15.20 | **0.807** | 0.792 | ❌ **worse** |

Plus a proxy-OOD split (held-out tonal extremes, 824 images): PSNR 27.835,
SSIM 0.7586, LPIPS 0.2645 — **higher than in-distribution**, which means the
split is *easier*, not harder. It is therefore weak evidence and we do not
present it as an OOD win.

---

# STAGE 5 — WHERE THE MODEL LACKS

## 5.1 🔴 Over-smoothing of high-frequency texture — the dominant failure

`report_landscape.png`, sample `val 002949`:

| | PSNR | SSIM |
|---|---|---|
| bicubic | 20.05 | 0.301 |
| ours | 23.41 | **0.382** |

**+3.4 dB of PSNR while SSIM stays at 0.382.** The ground truth is dense
foliage; our output is a smooth grey field. The model removed the speckle *and*
the texture, because at 128×128 it genuinely cannot tell them apart — both are
high-frequency, low-amplitude, spatially irregular.

Contrast `val 000253` (smooth sky): 20.60 → **28.31 dB**, SSIM 0.228 →
**0.787**. Same model, opposite content, opposite outcome.

> **The model has learned "when uncertain, smooth."** Under
> Charbonnier + MS-SSIM that is the loss-minimising policy. It is also precisely
> what the spec warns against.

**Root causes, ranked:**
1. Loss weighting — Charbonnier dominates at 1.0; FFT (0.1) and gradient (0.05)
   are the only terms defending detail.
2. Capacity and training length — 4.95 M params, 40 epochs, SSIM still rising.
3. No perceptual term — VGG weight is 0.
4. Training content — photographs are texture-rich but the model was never
   *forced* to preserve structure it could not resolve.

## 5.2 🔴 Fine periodic patterns — worse than bicubic

Checkerboard, 8-px period: **15.20 vs 15.64 dB**, SSIM **0.792 vs 0.807**.

The output shows irregular blotching where the truth is a perfectly regular
grid — the model is *inventing* structure rather than reconstructing it.

**Why this matters more than a synthetic curiosity suggests:** semiconductor
structures *are* periodic — DRAM arrays, FinFET gates, gratings. The 800
photographs contain almost no strong periodic content, so the model never had to
learn it. If the scored test set contains such imagery, this is where we lose.

**This is also the one case with a cheap fix:** procedurally generate gratings,
grids and rings and mix ~5 % into training. No download required.

## 5.3 🟠 Evidence gaps

| Claim | Evidence |
|---|---|
| Beats bicubic on all three scored metrics | **Strong** — 320 source-disjoint images |
| Removes speckle effectively | **Strong** — +13 dB on smooth content |
| Beats BM3D perceptually | **Strong** — LPIPS 0.301 vs 0.558 |
| Beats BM3D overall | **No** — trails 0.09 dB PSNR |
| Generalises to unseen content | **Weak** — proxy-OOD is easier than in-distribution; no external datasets tested |
| Handles 256→512 | **Untested** — architecture supports it, nothing measured |
| Preserves fine periodic structure | **Refuted** |

## 5.4 🟠 Unablated design choices

Three decisions are currently **unattributed**: the log channel, the decoupled
replica (`a1 ≠ a2`), and the bicubic residual anchor. Each is a one-line config
flip. Without ablations we cannot claim any of them helped.

---

## SUMMARY

**What worked:** reconstructing the degradation before writing model code. It
produced the split rule that makes validation honest, the replica that
manufactures ground truth, and the insight that this is a despeckling problem
first. The model beats the floor on all three scored metrics and is 2.1× faster
end-to-end than the naive pipeline.

**What did not:** detail reconstruction. The model despeckles far better than it
reconstructs, and on fine periodic structure it is worse than doing nothing.

**The single highest-value next step:** more training and more diverse content.
SSIM was still rising at epoch 36 of 40, on a quarter of the available VRAM,
with `external_ratio = 1.0` (no external data at all). See
`11_FUTURE_SCOPE.md`.
