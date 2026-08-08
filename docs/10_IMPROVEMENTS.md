# WHERE THIS MODEL CAN IMPROVE
## Evidence-based analysis, and how to validate without ground truth

Written after looking at the actual inputs and outputs, not from theory.
Reproduce every figure with `python scripts/r01_visual_report.py`.

---

## PART 1 — WHAT THE IMAGES ACTUALLY SHOW

### The headline number hides the real story

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Bicubic ×2 | 23.067 | 0.5129 | 0.4425 |
| BM3D + bicubic | **25.956** | 0.6527 | 0.5576 |
| **Ours** | 25.867 | **0.7050** | **0.3006** |

We beat the bicubic floor on all three. We beat BM3D on SSIM and LPIPS but
**lose PSNR by 0.09 dB**. Looking at the images explains why, and points
directly at the fix.

### 🔴 Failure mode 1 — over-smoothing of high-frequency texture

**This is the dominant problem.** `report_landscape.png`, sample `val 002949`:

| | PSNR | SSIM |
|---|---|---|
| bicubic | 20.05 | 0.301 |
| ours | 23.41 | **0.382** |

PSNR improves by 3.4 dB while SSIM stays terrible at 0.382. The ground truth is
dense foliage; our output is a smooth grey blur that has *removed the speckle
and the texture together*. The model cannot distinguish "noise" from "fine
detail that looks like noise", so it deletes both.

Contrast with `val 000253` (smooth sky): 20.60 → **28.31 dB**, SSIM 0.228 →
**0.787**. Where the content is genuinely smooth, the model is excellent.

> **The model has learned "when uncertain, smooth."** That is the safe policy
> under Charbonnier + MS-SSIM, and it is exactly the behaviour the problem
> statement warns against: *"Do not blur the image to remove noise — that
> destroys useful information."*

### 🔴 Failure mode 2 — fine periodic patterns are WORSE than bicubic

`report_synthetic_ood.png`, row 2 — a checkerboard with an 8-px period:

| | PSNR | SSIM |
|---|---|---|
| bicubic | **15.64** | **0.807** |
| ours | 15.20 | 0.792 |

**We lose to bicubic on both.** The output shows irregular blotching where the
ground truth is a perfectly regular grid — the model is *inventing* structure
where it cannot resolve the true one.

This matters more than a synthetic curiosity suggests. **Semiconductor
structures are periodic** — DRAM arrays, FinFET gates, gratings. If the scored
test set contains any such content, this is where we lose. The training corpus
is 800 natural photographs and contains almost no strong periodic structure, so
the model has never had to learn it.

### ✅ Where the model is genuinely strong

| Content | bicubic → ours (PSNR) | bicubic → ours (SSIM) |
|---|---|---|
| smooth gradient | 25.20 → **38.28** | 0.385 → **0.952** |
| text-like strokes | 17.70 → **19.57** | 0.342 → **0.694** |
| geometric shapes | 25.00 → **26.05** | 0.839 → **0.936** |
| fine periodic grid | 15.64 → 15.20 ❌ | 0.807 → 0.792 ❌ |

Speckle removal on piecewise-smooth content is excellent — a +13 dB gain on the
gradient. The despeckler works. The reconstruction of high-frequency detail
does not.

---

## PART 2 — IMPROVEMENTS, RANKED BY EXPECTED VALUE

### 🥇 1. Train longer and larger — the cheapest real gain

The run was **40 epochs on 4.95 M parameters**, and the validation curve was
still climbing when it stopped:

```
ep 26  PSNR 25.850  SSIM 0.7013
ep 28  PSNR 25.870  SSIM 0.7032
ep 30  PSNR 25.867  SSIM 0.7039     ← still rising on SSIM
ep 36  PSNR 25.867  SSIM 0.7050     ← selected
```

PSNR plateaued but SSIM had not. Peak VRAM was **2.1 GB of 8 GB** — we used
a quarter of the GPU.

**Do:** raise `width` 32→48 and `middle_blk_num` 8→12 (≈15 M params, still
fits), `epochs` 40→150, `lr_patch` 64→96. Expected **+0.5 to +1.0 dB** and the
PSNR gap to BM3D closes on its own.

### 🥇 2. Add external training content — the largest OOD lever

`external_ratio` is currently **1.0 (KLA pairs only)**. The whole synthetic-OOD
machinery is built and unused because nothing has been downloaded.

**Do:**
```bash
# DIV2K (800 imgs), Urban100 (buildings — the OOD case KLA named),
# DTD (textures), BSD500 → data/external/
python train.py --config configs/nafnet_w32.yaml   # set external_ratio: 0.5
```
`SyntheticPairs` + the validated replica already handle it. Urban100
specifically supplies the long straight edges and periodic window grids the
photograph corpus lacks — which attacks failure mode 2 directly.

### 🥈 3. Add synthetic periodic content — attacks the worst failure directly

We do not need a download for this. `scripts/r01_visual_report.py::make_synthetic`
already generates gratings, checkerboards and stripes. Mixing a few percent of
procedurally generated periodic patterns into training would teach the model
that regular structure exists and must be preserved rather than smoothed.

**Do:** add a `SyntheticPatternDataset` (gratings at random period/phase/angle,
line grids, concentric rings) at ~5 % of the mix. Cheap, and it targets the one
case where we currently lose to bicubic.

### 🥈 4. Re-weight the loss toward high frequencies

Current: `1.0 Charb + 0.2 MS-SSIM + 0.1 FFT + 0.05 grad`. The evidence says the
model under-weights detail.

**Do, one variable per run** (these are ablations 002–007, already scaffolded in
`../experiments/EXPERIMENT_LOG.md`):

| run | change | hypothesis |
|---|---|---|
| 004b | FFT 0.1 → 0.3 | frequency-domain L1 penalises the missing high band |
| 005b | gradient 0.05 → 0.15 | Sobel L1 penalises softened edges |
| 006 | VGG 0.0 → 0.01 | perceptual term proxies LPIPS, resists smoothing |
| 007 | L2 instead of Charbonnier | control: should over-smooth *more*, confirming the mechanism |

### 🥉 5. Self-ensemble (×8 TTA) — free quality, 8× latency

Average the model over the 8 dihedral transforms. Reliably worth **+0.1–0.3 dB**
with no retraining. It costs 8× compute, which at 2.58 ms/img is still only
~21 ms/img — likely affordable, but it must be **measured** against the
throughput axis, not assumed.

### 🥉 6. Run the ablations that attribute the gain

Three design choices are currently unvalidated: the **log channel**, the
**decoupled replica** (`a1 ≠ a2`), and the **bicubic residual anchor**. Each is
a one-line config flip. Without these we cannot claim any of them helped — and
the training-hygiene axis rewards exactly this kind of attribution.

### 🥉 7. Train SAFMN and build the Pareto curve

Implemented, never trained. Needed for the operating-point argument
("quality primary, speed tiebreak") which currently has one point on it.

### 8. The innovation track — log-domain unrolled network

Phase 6, untouched. KLA's own unpresented appendix slide cites Monga et al. on
algorithm unrolling. We already have the log transform in the stem and the
degradation model in closed form, which is most of the machinery.

---

## PART 3 — HOW TO VALIDATE WHEN THERE IS NO GROUND TRUTH

The 400 test images have **no ground truth** — KLA holds it. You cannot compute
PSNR/SSIM/LPIPS on them. Here is the full toolkit, strongest first.

### ⭐⭐⭐ Method 1 — Synthetic ground truth (the primary answer)

**This is why Phase 0 mattered.** Because we reconstructed the degradation and
validated it to 0.982 histogram overlap, we can take *any* clean image, degrade
it ourselves, and manufacture a labelled pair.

```bash
python evaluate.py --ckpt experiments/runs/nafnet_w32/best.pt \
                   --ood data/external/Urban100 data/external/DTD
```

We get PSNR/SSIM/LPIPS on content KLA never gave us labels for. This already
works on synthetic patterns (Part 1 table) — it just needs real external
datasets downloaded.

### ⭐⭐ Method 2 — Degradation-consistency (works on the real test set)

```
x̂ = model(y)              restore
ŷ = degradation(x̂)        re-degrade with our replica
error = ‖ŷ − y‖           should be small
```

If the restoration, re-degraded, does not reproduce the observed input, that
image is suspect. Already implemented and run:

```bash
python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w32/best.pt
```

| | RMSE |
|---|---|
| median | 0.0924 |
| p95 | 0.1467 |
| max | 0.1957 |

The worst 15 are listed in `experiments/consistency.json`. **Look at those
images** — they are where the model is least confident.

⚠️ Caveat: consistency is necessary, not sufficient. An over-smoothed output can
still re-degrade to something close to the input, because the degradation
destroys high frequencies anyway. Use it to catch gross failure, not to rank
good models.

### ⭐⭐ Method 3 — No-reference IQA

NIQE / BRISQUE / PIQE need no ground truth. `src/metrics/no_reference.py` is
implemented (pyiqa) but **not yet run** on the outputs.

**Do:** score `outputs/` and the bicubic baseline on the same images and compare
*relatively*.

⚠️ Caveat: NR-IQA frequently **rewards hallucinated texture**, which the spec
forbids. Use as a sanity check, never as an optimisation target.

### ⭐ Method 4 — Ensemble disagreement as an OOD detector

Train a second model with a different seed. Where the two disagree strongly, the
input is likely OOD or hard. This gives a per-image difficulty estimate on
unlabelled data. Not implemented.

### ⭐ Method 5 — Distribution-gap measurement

Compare the test set's statistics against training. We already know from Phase 0
that the test split is slightly brighter (mean 0.443 vs 0.434) and has **2.3×
more negative pixels** — so it is *not* identically distributed. Worth
quantifying further with feature statistics or radial power spectra.

### ⭐ Method 6 — Look at the images

Unglamorous and irreplaceable. `report_test_no_gt.png` puts input, bicubic and
our output side by side at 4× zoom. Ringing, checkerboard artefacts,
over-smoothing and invented texture are obvious to the eye and invisible in
aggregate metrics.

---

## PART 4 — THE HONEST SCORECARD

| Claim | Evidence strength |
|---|---|
| Beats bicubic on all three scored metrics | **Strong** — 320 held-out images, source-disjoint split |
| Removes speckle effectively | **Strong** — +13 dB on smooth content, visually obvious |
| Beats BM3D perceptually | **Strong** — LPIPS 0.301 vs 0.558 |
| Beats BM3D overall | **No** — trails 0.09 dB PSNR |
| Generalises out of distribution | **Weak** — the only OOD number is a tonal split that is *easier* than in-distribution; no external content tested |
| Handles 256→512 | **Untested** — architecture supports it, nothing measured |
| Preserves fine periodic structure | **Refuted** — worse than bicubic on a checkerboard |

---

## PART 5 — SUGGESTED ORDER OF WORK

1. Download DIV2K + Urban100 + DTD → real OOD numbers, and the largest expected
   quality gain. *(unblocks the weakest claim above)*
2. Retrain at width 48 / 150 epochs / patch 96. *(closes the PSNR gap)*
3. Add procedural periodic patterns to the mix. *(fixes the one case we lose)*
4. Run loss ablations 002–007 → `09_ABLATION_RESULTS.md`.
5. Train SAFMN → Pareto curve → justified operating point.
6. Measure ×8 self-ensemble on both axes.
7. Log-domain unrolled network (Phase 6).
