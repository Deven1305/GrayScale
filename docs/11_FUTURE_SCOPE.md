# FUTURE SCOPE
## What is done, what is missing, and what to do next — ranked

Cross-checked against `PROMPT.md` (dev repo) (the 10-phase plan), `brief/00`–`brief/05`
(the research), and `brief/reference/` (the official KLA material).

---

## PART 0 — STATUS AT A GLANCE

```
PHASE 0  Forensics ..................... ✅ DONE   degradation reconstructed, 0.982 overlap
PHASE 1  Metrics + baselines ........... ✅ DONE   bicubic 23.07 · BM3D 25.96
PHASE 2  Replica + data pipeline ....... ✅ DONE   DIV2K 800 crops + patterns + memmap bundle
PHASE 3  Baseline model ................ ✅ DONE   NAFNet-w32, 25.87 / 0.705 / 0.301
PHASE 3+ Scaled model .................. 🔄 TRAINING nafnet_w48, 15.2 M, 7.1 GB of 8.5
PHASE 4  Loss ablations ................ 🟡 CODE READY  run_ablations.py, 8 runs
PHASE 5  OOD robustness ................ 🟡 CODE READY  214 OOD images downloaded
PHASE 6  Log-domain unrolled network ... 🟡 BUILT    model done; log error MEASURED
PHASE 7  Throughput engineering ........ ✅ DONE   2.1× measured; Pareto script ready
PHASE 8  Packaging + fresh-machine test  ✅ DONE   all 6 items, clone-tested
PHASE 9  Deck ......................... 🟡 PARTIAL 9 slides, real numbers; template missing
```

Everything is now **implemented**; what remains is GPU time. The exact command
sequence to finish is in `03_HOW_TO_RUN.md` §5b.

### 🔴 A measured correction to `brief/04` Part 6

The brief argues the log-domain approach is safe because *"σ ≈ 0.001–0.009
against speckle std ≈ 0.24, so the approximation error is small"*, and asks for
it to be quantified. We did (`scripts/measure_log_error.py`):

| σ | signal x | RMS error | as % of log-signal spread |
|---|---|---|---|
| 0.0086 (the deck's value) | 0.4 | 0.024 nats | **9.8 %** |
| **0.04 (measured maximum)** | 0.4 | 0.112 nats | **46.4 %** |
| **0.04** | **0.05 (dark)** | 1.505 nats | **623 %** — 11.9 % of samples hit the log floor |

**The claim holds only at the deck's σ and at mid brightness.** At the σ range
Phase 0 actually measured, and in dark regions, the log-additivity assumption
breaks down badly. Since 13.2 % of this corpus is dark, the unrolled model
should be expected to help *least* exactly where the data is hardest.

That does not kill the approach — the network can learn to compensate — but the
clean theoretical story does not survive measurement, and saying so is worth
more than repeating it.

---

## PART 1 — CROSS-CHECK AGAINST `brief/`

### 1.1 What the brief got right, and we confirmed

| brief claim | status |
|---|---|
| Speckle is Gamma multi-look, `n ~ Gamma(L,1/L)` (`brief/04` §2.1) | ✅ **confirmed** three independent ways |
| Never clip the input (`brief/04`, `brief/02` §7) | ✅ **confirmed** — 3.11 % of pixels exceed 1 |
| Split by source, never randomly (`brief/04` Part 3) | ✅ **confirmed**, and the exact rule recovered |
| Speckle dominates the Gaussian (`brief/04` §2.2) | ✅ **confirmed** — 6–240× |
| Timing includes startup (`brief/04` Part 5) | ✅ **confirmed** — startup is 0.4 s of 17.7 s |
| No GAN / diffusion (`brief/05` Part 1) | ✅ followed, and BM3D quantifies why |
| L ≈ 16.9–18.1 | ✅ **consistent** — measured median 17.7 |

### 1.2 What the brief got wrong, and we corrected

| brief claim | measured reality |
|---|---|
| Both 512→256 and 256→128 pairs exist | **100 % is 256→128** in this release |
| ~2.7 samples per source | **exactly 4**, `source_id = index // 4`, 800 sources |
| Degradations applied in **random order** | **fixed order** — unimodal vs bimodal controls |
| σ ∈ [0.001, 0.009] | consistent with **log-uniform up to ~0.04** |
| Origins: texture / dendrite / natural photo | **one homogeneous corpus of photographs** |
| Validate the estimator against KLA's published L/σ | **impossible** — those figures describe a *different release* |
| `cudnn.benchmark = True` gives 1.1–1.3× (`brief/02` §4) | **2.4× end-to-end LOSS** at 400 images |
| DataLoader workers speed up the reader stage | **slower** on Windows (spawn re-imports torch) |

> **Takeaway for the team:** the brief is excellent research but it is
> *inference from slides and a transcript*. Eight of its load-bearing claims did
> not survive contact with the data. Anything in `brief/` that has not been
> measured should be treated as a hypothesis.

### 1.3 `brief/reference/` — is it correctly implemented?

| file | status |
|---|---|
| `submission_criteria.txt` | ✅ all 6 repo items delivered; deck follows the 9-slide structure exactly |
| `descripton.txt` / `background.txt` | ✅ requirements reflected (no blur, no ringing, 2× output, grayscale) |
| `Problem Statement 01_KLA.pptx` | ✅ read; figures extracted; **correctly gitignored** (KLA-Confidential footer) |
| `7b675083-….pptx` (webinar deck) | ✅ read, including the unpresented appendix slide on algorithm unrolling |
| **`Idea_Submission_Template.pptx`** | ❌ **MISSING** — `PROMPT.md` (dev repo) Phase 9 calls it mandatory |

**Two live compliance risks:**

1. 🔴 **The template is missing.** The criteria say *"Use the provided Hackathon
   Idea Submission Template."* Our deck maps 1:1 onto the required slide
   structure, so content transfers cleanly — but download the template from the
   portal and re-lay the content onto it.
2. 🔴 **The repo is private.** *"Your GitHub repository must be public."* A
   private repo cannot be benchmarked, and unscored submissions cannot win.

Also outstanding: rename `TEAMNAME_KLA_PS01.*` to your real team name, and fill
slide 1.

---

## PART 2 — RANKED ROADMAP

Ordered by **expected score gain ÷ effort**. Each item states *why*, grounded in
a measurement from `07_ANALYSIS.md`.

### 🥇 1. Ingest external training data — biggest single lever

**Evidence:** `external_ratio` is **1.0** — the model has seen only 800
photographs. The entire synthetic-OOD machinery is built and unused.

**Why it works:** we have a replica validated to 0.982 histogram overlap, so any
clean image becomes a training pair. `SyntheticPairs` already handles it.

**Do:**
```bash
# → data/external/{DIV2K,Flickr2K,Urban100,BSD500,DTD}/
# then set external_ratio: 0.5 in configs/nafnet_w32.yaml
python train.py --config configs/nafnet_w32.yaml
python evaluate.py --ckpt ... --ood data/external/Urban100 data/external/DTD
```

**Urban100 specifically** supplies the long straight edges and periodic window
grids the photograph corpus lacks — it attacks failure mode 5.2 directly. It is
also the exact OOD case KLA named in the webinar.

**Expected:** first *real* OOD numbers (currently the weakest claim), plus
+0.2–0.5 dB in-distribution from content diversity.
**Effort:** a few hours, mostly download.

### 🥇 2. Train longer and larger

**Evidence:** SSIM was **still rising at epoch 36 of 40**, and peak VRAM was
**2.1 GB of 8 GB**. We stopped an improving model using a quarter of the GPU.

**Do:** `width` 32→48, `middle_blk_num` 8→12 (~15 M params), `epochs` 40→150,
`lr_patch` 64→96, `batch_size` 32→16 with `accum_steps` 2.

**Expected:** +0.5 to +1.0 dB. The 0.09 dB PSNR gap to BM3D closes on its own.
**Effort:** one overnight run, no new code.

### 🥇 3. Add procedural periodic patterns

**Evidence:** the checkerboard case is the **only** one where we lose to bicubic
(15.20 vs 15.64 dB). Semiconductor structures are periodic.

**Do:** a `SyntheticPatternDataset` — gratings at random period/phase/angle,
line grids, concentric rings, checkerboards — mixed in at ~5 %. The generator
already exists in `scripts/r01_visual_report.py::make_synthetic`.

**Expected:** turns a loss into a win on the content type most likely to appear
in a semiconductor-flavoured OOD test set.
**Effort:** ~50 lines. **No download needed.**

### 🥈 4. Loss ablations (Phase 4)

**Evidence:** the model over-smooths, and we cannot currently attribute that to
any specific loss term.

| run | change | hypothesis |
|---|---|---|
| 002 | Charbonnier only | ablation baseline |
| 003 | + 0.2 MS-SSIM | |
| 004 | + 0.1 FFT | |
| 004b | **FFT 0.1 → 0.3** | frequency-domain L1 defends the high band |
| 005b | **gradient 0.05 → 0.15** | Sobel L1 defends edges |
| 006 | + 0.01 VGG | perceptual term proxies LPIPS, resists smoothing |
| 007 | L2 instead of Charbonnier | **control** — should over-smooth *more* |

Run 007 matters as much as the others: if L2 visibly over-smooths, it confirms
the mechanism behind our dominant failure mode.

**Expected:** +0.1–0.3 dB, and a table that demonstrates method rather than
luck — directly what the training-hygiene axis rewards.
**Effort:** 7 runs × ~35 min, fully automatable.

### 🥈 5. Ablate the three unattributed design choices

The **log channel**, the **decoupled replica** (`a1 ≠ a2`), and the **bicubic
residual anchor** are all currently unvalidated. Each is a one-line config flip.
Without these we cannot honestly claim any of them helped.

### 🥉 6. Self-ensemble (×8 TTA)

Average over the 8 dihedral transforms. Reliably **+0.1–0.3 dB** with no
retraining, at 8× compute — 2.58 ms → ~21 ms/image, likely still affordable.
**Must be measured on both axes**, not assumed.

### 🥉 7. SAFMN + the Pareto curve (Phase 7 completion)

Implemented, never trained. The operating-point argument ("quality primary,
speed tiebreak") currently has **one point on the curve**.

---

## PART 3 — THE INNOVATION TRACK (Phase 6)

### Why this is worth real marks

`brief/04` Part 6 documents an appendix slide in KLA's webinar deck, positioned
*after* "Thank You" and **never presented**:

> *"In advanced forms, it combines analytical iterative methods with learnt AI
> priors!"* — citing **V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling",
> IEEE Signal Processing Magazine 38(2), 2021.**

That is KLA telling you what they consider a standout solution, in a slide
almost nobody will read. Plain image-to-image regression — which is what we
built — is the baseline they *expect*.

### What to build

```
1. Estimate L and σ from the input
   (small head, or reuse the Phase 0 variance regression — already written)
2. Log-transform:  log(x·n) = log x + log n
   multiplicative speckle becomes ADDITIVE → a standard linear inverse problem
3. K unrolled steps (start K=4), each alternating:
     data-fidelity step — analytical, uses the KNOWN 2× cubic decimation
                          and the estimated noise level
     prior step         — a small trained CNN denoiser
4. Exponentiate back
5. PixelShuffle to full resolution
```

We already have: the log transform (in the stem), the degradation in closed form
(`src/data/degradation.py`), and the noise estimator (`scripts/f21_L_sigma.py`).
**Most of the machinery exists.**

### Reference points

| method | relevance |
|---|---|
| **USRNet** (Zhang et al., CVPR 2020) | deep unfolding **super-resolution**; takes LR + kernel + noise level, alternates closed-form data-fidelity with a CNN prior. Closest published match |
| **DPIR / DRUNet** (Zhang et al., TPAMI 2021) | plug-and-play denoiser prior inside HQS iterations |
| **ISTA-Net⁺** | unrolled ISTA, simplest to implement |
| SAR deep-unfolding despeckling | unrolling applied specifically to multiplicative noise |

### The honest caveat to state, not hide

`log(a+b) ≠ log(a) + log(b)`, so the additive Gaussian breaks the log
transform's exactness. **Quantify it.** With σ up to 0.04 against speckle std
0.238, the error is small but *not* negligible at the top of the σ range —
measure it across the range, not at one value. Reporting that number is a
stronger move than omitting it.

**Also measure latency:** K unrolled steps cost ~K× a forward pass. With K=4 and
a small prior this stays competitive, but it must be measured against NAFNet,
not assumed.

**If it wins, it leads the innovation slide. If it loses, it still goes in the
ablation table** as a technique implemented from KLA's own reference and
measured honestly — which reads far better than not having tried.

---

## PART 4 — RESEARCH DIRECTIONS BEYOND THE HACKATHON

Grounded in the specific failure modes measured in `07_ANALYSIS.md` §5.

### 4.1 Noise-conditional restoration (FFDNet-style)

**Problem it solves:** the model applies one fixed smoothing policy regardless
of local noise strength, which is why it over-smooths clean texture.

**Approach:** feed an estimated noise map as an extra channel. We can *already*
estimate L and σ per image (`f21_L_sigma.py`), and the out-of-range overshoot is
a direct local noise-strength cue. This lets the network smooth hard where the
noise is strong and preserve detail where it is not.

*Zhang, Zuo, Zhang — "FFDNet: Toward a Fast and Flexible Solution for
CNN-based Image Denoising", IEEE TIP 2018.*

### 4.2 Frequency-aware architectures

**Problem it solves:** the fine-periodic-grid failure is a *frequency* failure —
structure near Nyquist is destroyed.

**Approach:** explicit frequency-domain branches (FFC, wavelet decomposition,
Fourier units) treat high-frequency bands as first-class rather than something
convolutions must infer. Our FFT loss touches this; the *architecture* does not.

### 4.3 Speckle-specific classical priors as learned modules

Lee, Kuan, Frost and SRAD filters were designed for multiplicative noise. As
differentiable, learnable modules inside the network they would inject the right
inductive bias — a middle path between our black-box CNN and full unrolling.

### 4.4 Degradation-aware / blind restoration

**Problem it solves:** the test set is *not* identically distributed (2.3× more
negative pixels). Real-world parameters drift.

**Approach:** BSRGAN/Real-ESRGAN-style wide degradation randomisation, plus a
degradation encoder that predicts the parameters and conditions the restorer.
Our replica already supports randomising L, σ, both kernels and the operation
order — the augmentation half is built; the conditioning half is not.

### 4.5 Uncertainty estimation

**Problem it solves:** on the real test set we have no ground truth and no
per-pixel confidence.

**Approach:** predict a variance map alongside the restoration (heteroscedastic
loss), or use ensemble disagreement. In a metrology setting *"this region is
unreliable"* is arguably more valuable than a slightly better mean, because it
tells the operator which measurements to distrust.

### 4.6 The real 512→256 case

Every released pair is 256→128. KLA's own slides show 512→256. Our model is
fully convolutional but **has never been trained or evaluated at that scale**.
Generate 512→256 pairs with the replica and check whether performance holds.

---

## PART 5 — IF YOU ONLY HAVE ONE WEEKEND

| Day | Do | Why |
|---|---|---|
| Sat AM | Download DIV2K + Urban100 + DTD | Unblocks the weakest claim |
| Sat PM | Kick off the bigger/longer training run | Runs overnight, free gain |
| Sat PM | Write the procedural pattern dataset | 50 lines, fixes the one loss |
| Sun AM | Evaluate on real OOD families | First honest generalisation number |
| Sun PM | Loss ablations 002–007 | The table for the innovation slide |
| Sun PM | **Make the repo public**, get the template, fill slide 1 | Compliance — worth more than any metric |

The last row is not optional. A model that cannot be benchmarked scores zero
regardless of how good it is.
