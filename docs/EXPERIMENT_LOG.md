# EXPERIMENT LOG
## KLA PS01 — AI-Based Restoration of Degraded Images

One row per run. **One variable changed per row.** Every number here was
produced by a script in this repository and can be regenerated; none are
estimated.

**Validation protocol (fixed before any training started):**

* Split **by source**, `source_id = sample_index // 4` — 720 train sources /
  80 val sources, 2880 / 320 samples. Asserted disjoint by
  `tests/test_splits.py`; a random split would leak overlapping crops of the
  same photograph and make every number below meaningless.
* Metrics at `data_range=1.0`, verified by `PSNR(x, x) == inf`
  (`tests/test_metrics.py`).
* Model selection: **best val SSIM** (the primary scored metric), not last
  epoch. Rule fixed in advance and recorded in every checkpoint.
* Evaluation uses **EMA weights** (decay 0.999).
* Full-resolution val images (128→256), no cropping.

**Hardware:** a single consumer GPU with 8 GB VRAM · PyTorch 2.11.0+cu128 ·
bf16 autocast.
KLA benchmarks on an **H100**, so latency measured here does not transfer.

---

## Results

| ID | Date | Config | Change vs previous | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MS-SSIM ↑ | Verdict |
|---|---|---|---|---|---|---|---|---|
| **000a** | Aug 6 | `bicubic ×2` | — (floor) | 23.067 | 0.5129 | 0.4425 | 0.7770 | baseline |
| **000b** | Aug 6 | `BM3D + bicubic ×2` | classical denoise first | 25.956 | 0.6527 | **0.5576** | 0.8451 | baseline — note LPIPS is *worse* than bicubic: BM3D over-smooths, which is exactly the failure the spec warns against |
| **001** | Aug 6 | `nafnet_w32.yaml` | first trained model | 25.867 | **0.7050** | **0.3006** | 0.8976 | ✅ keep — beats bicubic on all three; beats BM3D on SSIM and LPIPS, trails it 0.09 dB on PSNR |
| **002** | Aug 7 | `nafnet_w48.yaml` | width 32 → 48, 12 middle blocks, 40 epochs | 26.415 | 0.7333 | 0.2455 | 0.9156 | ✅ shipped as v1 |
| **003** | Aug 14 | `nafnet_w48_sharp.yaml` | **loss rebalance only** — HF-weighted FFT (0.2→0.6, hf_power 1.5), new high-pass term 0.5, gradient 0.1→0.05, VGG 0→0.05, 40→60 epochs | **26.616** | 0.7344 | **0.2244** | 0.9205 | ✅ **best.** +0.201 dB, LPIPS −8.6%. SSIM unchanged (inside the 0.005 noise band) |

Selected checkpoint: **epoch 36** of 40 (best val SSIM), EMA weights — for run 002.

---

## Run 003 — the loss rebalance, in detail

**Hypothesis.** `scripts/f26_loss_spectrum.py` measured that the v1 composite
sends only **49.11%** of its gradient into the high-frequency half of the
spectrum — *less* than plain Charbonnier alone (52.82%). The two terms added for
sharpness were making it softer: the Sobel `gradient` term measured 22.58%,
30 points BELOW the term it was meant to sharpen. Predicted v2 emphasis: 61.31%.

**Nothing about the model changed.** Same NAFNet-w48, same 15.24 M parameters,
same weight format, same inference cost. Training signal only.

### Epoch-matched comparison (rules out the longer schedule as the cause)

| epoch | v1 PSNR | v2 PSNR | v1 LPIPS | v2 LPIPS |
|---|---|---|---|---|
| 16 | 26.210 | 26.324 | 0.2593 | 0.2595 |
| 24 | 26.337 | 26.421 | 0.2492 | 0.2509 |
| 32 | 26.390 | 26.504 | 0.2459 | 0.2393 |
| **40** | **26.415** | **26.555** | **0.2455** | **0.2310** |
| 60 | — | 26.616 | — | 0.2244 |

At v1's own 40-epoch budget v2 is already **+0.140 dB and −5.9% LPIPS**, so the
gain is the loss change, not the extra 20 epochs. The extra epochs add a further
+0.061 dB and −2.9% LPIPS.

Note v1's LPIPS had stalled — 0.2459 at ep32, *worse* at 0.2467 by ep36, 0.2455
at ep40. v2 decreases monotonically throughout. The sharpness terms give a
better-conditioned objective for the perceptual metric.

### All splits (`experiments/eval_results_sharp.json`)

| split | v1 PSNR | v2 PSNR | Δ | v1 LPIPS | v2 LPIPS | Δ% |
|---|---|---|---|---|---|---|
| val in-distribution | 26.415 | 26.616 | **+0.201** | 0.2455 | 0.2244 | **−8.6%** |
| proxy-OOD tonal extremes | 28.376 | 28.603 | **+0.227** | 0.2075 | 0.1825 | **−12.0%** |
| ood/Urban100 | 24.432 | 24.587 | **+0.156** | 0.2204 | 0.2147 | −2.6% |
| ood/BSD100 | 26.422 | 26.510 | +0.088 | 0.2494 | 0.2543 | **+2.0%** ⚠️ |
| ood/Set14 | 26.655 | 26.795 | +0.140 | 0.2110 | 0.2071 | −1.8% |

**Read against this project's own noise thresholds** (`pareto.json`:
`psnr_noise_dB` 0.15, `ssim_noise` 0.005):

* **PSNR** — real but modest. Above threshold on val, proxy-OOD and Urban100;
  *within* noise on BSD100 and Set14.
* **SSIM** — **unchanged.** Every gain is ≤0.0045, inside the noise band. Do not
  claim an SSIM improvement.
* **LPIPS** — **the actual win**, and the metric that tracks visible sharpness.
* ⚠️ **BSD100 LPIPS got 2.0% worse.** The one regression, reported rather than
  dropped. BSD100 is smooth natural scenery, where pushing high-frequency
  gradient has least to recover and most to disturb.

**Verdict: promote v2.** It wins PSNR on all five splits and LPIPS on four of
five, at identical inference cost. The BSD100 LPIPS regression is the honest
caveat.

### Proxy-OOD — held-out tonal extremes (824 images)

| Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| tonal extremes (darkest + brightest source clusters) | 27.835 | 0.7586 | 0.2645 |

⚠️ **This scores *higher* than in-distribution, so the split is easier, not
harder.** It is therefore weak evidence of generalisation, and we say so rather
than presenting it as an OOD win. Genuine content-OOD (Urban100, DTD, Manga109)
needs external data not yet ingested — that is experiment 009.

### Degradation-consistency on the real test set (no ground truth)

`python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w32/best.pt`

`x̂ = model(y)` → re-degrade with the replica → `‖ŷ − y‖`. Works where no GT
exists, which is the whole 400-image test set.

| | RMSE |
|---|---|
| median | 0.0924 |
| p95 | 0.1467 |
| max | 0.1957 |

Worst 15 images logged in `experiments/consistency.json` for visual inspection.

### Throughput (400 test images, 8 GB consumer GPU, nafnet_w48)

| Configuration | End-to-end | img/s |
|---|---|---|
| 4 DataLoader workers (the usual advice) | 35.2 s | 11.4 |
| **defaults: 0 workers where they spawn, autotune off** | **9.0 s** | **44.4** |

Per-image stage breakdown: compute 6.02 ms · write 2.15 ms · read 0.27 ms · d2h 0.17 ms · h2d 0.04 ms. Startup 1.4 s, warmup 0.03 s.

Two measurement notes, both worth recording because they changed the number:

* **The shipped `--num_workers` default was wrong.** It was 4, which is the
  usual advice, but under spawn each worker re-imports torch: 35.2 s vs 16.4 s
  with none. It is now platform-aware (0 under spawn, 4 under fork) so the
  script is fastest out of the box — which matters, because KLA runs it as-is.
* **The benchmark was contaminating its own result.** It ran the in-process
  stage breakdown first, leaving a model and a CUDA context resident while it
  timed the subprocess, inflating end-to-end roughly 3x. End-to-end is now
  measured first on a clean GPU, best of three warm runs.

---

## Notes per run

### 000a / 000b — baselines

`python scripts/run_baselines.py --limit 120`

The floor. Any model failing to beat bicubic on all three scored metrics has a
bug, not a design problem.

The BM3D row is worth reading carefully: it gains **+2.9 dB PSNR** and
**+0.14 SSIM** over bicubic while its **LPIPS gets 0.11 worse**. That is the
signature of over-smoothing — it removes noise by destroying detail. The
problem statement explicitly forbids "blurring the image to remove noise", and
this row is the quantitative demonstration of why. It also shows why optimising
PSNR alone is not sufficient here.

### 001 — NAFNet-w32, first trained model

`python train.py --config configs/nafnet_w32.yaml`

* 4.95 M parameters
* Composite loss: `1.0·Charbonnier + 0.2·(1−MS-SSIM) + 0.1·FFT + 0.05·gradient`
* Degradation replica params: L ∈ [8, 40], σ ∈ log-uniform[1e-3, 4e-2],
  kernel a₁ ∈ [−0.75, −0.45], a₂ ∈ [−0.40, −0.15], fixed operation order
* AdamW lr 2e-4, cosine + 2-epoch warmup, grad-clip 1.0, EMA 0.999
* batch 32, 64×64 LR crops, 8-fold dihedral augmentation
* Peak VRAM ≈ 2.1 GB of 8 GB

**Throughput finding during this run.** The first configuration trained at
8 img/s against 125 img/s of pure GPU compute. Two causes, both fixed:

1. `CompositeLoss` called `float()` on every loss term every step — five GPU
   syncs per iteration. Returning detached tensors and syncing once per epoch:
   **8 → 20 img/s**.
2. Two `np.load` calls per sample at random offsets. Packing the dataset into a
   memory-mapped bundle (`scripts/precrop_patches.py`): **20 → 136 img/s**.

Net **17×**, with no change to the model. Worth recording because the same
insight drives the inference pipeline design.

**Two measurements that contradicted the plan.** `brief/02` recommends
`cudnn.benchmark = True` (a claimed 1.1–1.3× gain) and `num_workers` for the
reader stage. On this workload both are *losses*:

* Autotuning costs **11.7 s** of warmup while making the main loop only ~8 %
  faster — a **2.4× end-to-end loss** on 400 images. It only amortises above
  roughly 15 000 images. Default flipped to off, with `--cudnn_benchmark` to
  re-enable.
* More DataLoader workers made it **slower**: on Windows, spawn re-imports
  torch per worker, and 400 small images never repay that.

Both were assumptions in the brief that measurement overturned. On an H100 with
a larger test set the answer may well flip back, which is exactly why the flag
exists rather than a hardcoded choice.

---

## Round 2 — all remaining phases

External content ingested (Phase 2), model scaled to actually use the GPU
(Phase 3+), Pareto candidates trained (Phase 7), innovation track built
(Phase 6), loss ablated (Phase 4), real OOD families evaluated (Phase 5).

### Main results

| ID | Config | Params | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Verdict |
|---|---|---|---|---|---|---|
| 000a | bicubic ×2 | — | 23.067 | 0.5129 | 0.4425 | floor |
| 000b | BM3D + bicubic | — | 25.956 | 0.6527 | 0.5576 | floor |
| 001 | `nafnet_w32` (KLA data only) | 4.95 M | 25.867 | 0.7050 | 0.3006 | superseded |
| **012** | **`nafnet_w48` + external + patterns** | **15.24 M** | **26.415** | **0.7333** | **0.2455** | ✅ **shipped** |
| 013 | `safmn` + external + patterns | 0.22 M | 26.064 | 0.7172 | 0.2660 | ✅ Pareto point |
| 014 | `log_unrolled_k4` (Phase 6) | 0.07 M | 25.872 | 0.7061 | 0.2799 | ✅ innovation track |

**Experiment 012 now beats every baseline on all three scored metrics**
(+3.35 dB / +0.220 SSIM / −0.197 LPIPS over bicubic; +0.46 dB / +0.081 SSIM /
−0.312 LPIPS over BM3D). Experiment 001 trailed BM3D on PSNR by 0.09 dB; the
three changes that closed it were capacity (4.95 → 15.24 M), context (64 → 96 px
crops) and content (KLA-only → 60 % KLA + 25 % DIV2K + 15 % procedural patterns).

**Experiment 014 is the notable one for parameter efficiency:** the log-domain
unrolled network matches the 4.95 M `nafnet_w32` using **0.07 M parameters —
70× fewer**. That is exactly the claim algorithm unrolling makes, reproduced
here on KLA's own cited reference.

### Phase 5 — real out-of-distribution families

Ground truth exists because we degrade clean benchmark images with our own
validated replica.

| Family | Content | PSNR ↑ | SSIM ↑ | LPIPS ↓ | vs in-dist PSNR |
|---|---|---|---|---|---|
| KLA val | photographs (in-distribution) | 26.415 | 0.7333 | 0.2455 | — |
| **Urban100** | **buildings / cityscapes** | **24.432** | 0.7750 | 0.2204 | **−1.98 dB** |
| BSD100 | natural scenes | 26.422 | 0.7506 | 0.2494 | +0.01 dB |
| Set14 | classic SR benchmark | 26.655 | 0.7643 | 0.2110 | +0.24 dB |
| tonal extremes | darkest/brightest KLA sources | 28.376 | 0.7868 | 0.2075 | +1.96 dB |

**Urban100 is the first genuine OOD signal we have.** It drops 1.98 dB — the
buildings-and-cityscapes case KLA named by name is measurably harder, exactly
as expected. BSD100 and Set14 hold, and the tonal-extreme split scores *higher*
than in-distribution, confirming it was always an easier split rather than an
OOD test.

### Phase 4 — loss ablation (12 epochs each, KLA data only, one variable per row)

| ID | Loss change | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| **002** | Charbonnier only (base) | **25.643** | 0.6520 | 0.3424 |
| 003 | + 0.2 MS-SSIM | 25.332 | 0.6440 | 0.3563 |
| 004 | + 0.1 FFT | 25.341 | 0.6446 | 0.3562 |
| 005 | + 0.05 gradient | 25.396 | 0.6489 | 0.3544 |
| 006 | + 0.01 VGG | 25.413 | 0.6483 | 0.3508 |
| **007** | **L2 instead of Charbonnier** (control) | **25.020** | 0.6485 | 0.3559 |
| 004b | FFT 0.1 → 0.3 | 25.446 | 0.6528 | 0.3520 |
| **005b** | **gradient 0.05 → 0.15** | 25.489 | **0.6573** | **0.3471** |

**Read this table carefully — it does not say what we expected.**

* The **control worked**: 007 (L2) is the worst row by PSNR, −0.62 dB below the
  Charbonnier base. That is the over-smoothing the spec warns against,
  reproduced deliberately, and it validates choosing Charbonnier.
* **At a 12-epoch budget, every added loss term costs PSNR** relative to plain
  Charbonnier. The auxiliary terms are regularisers; they slow early
  convergence and only pay off with a longer schedule.
* **The detail-defending terms win on the metrics that matter**: 005b (gradient
  ×3) has the best SSIM *and* the best LPIPS of any row, which is the direct
  evidence behind raising FFT to 0.2 and gradient to 0.10 in the shipped w48
  config.

The honest caveat: 12 epochs is short, so these rows rank *early convergence*,
not final quality. The shipped 40-epoch w48 run with the strengthened composite
reaches 0.7333 SSIM, far above any ablation row.

### Phase 7 — Pareto curve and operating point

Latency is the pure forward pass, fp16 + channels_last, batch 16 at 128×128.

| Model | Params | ms/img | PSNR | SSIM |
|---|---|---|---|---|
| **NAFNet-w48** | 15.24 M | 4.62 | **26.415** | **0.7333** |
| SAFMN | 0.22 M | 3.42 | 26.064 | 0.7172 |
| Log-unrolled K=4 | 0.07 M | 4.22 | 25.872 | 0.7061 |
| NAFNet-w32 | 4.95 M | 2.84 | 25.867 | 0.7050 |

Rule fixed in advance: **quality primary, speed the tiebreak** — choose the
fastest model whose quality is within noise (0.15 dB / 0.005 SSIM) of the best.

Only NAFNet-w48 is within noise of itself, so **NAFNet-w48 is the operating
point**. SAFMN is the interesting runner-up: 26 % faster for 0.35 dB, worth
revisiting if the throughput weighting turns out higher than assumed.

Note SAFMN (0.22 M) is *slower* than NAFNet-w32 (4.95 M) despite 22× fewer
parameters — its multi-scale pooling is memory-bound rather than compute-bound.
Parameter count is not latency.

### Phase 6 — the log-transform approximation, measured

`scripts/measure_log_error.py`. `brief/04` claims the error is small because
σ ∈ [0.001, 0.009]. At the σ range Phase 0 actually measured, it is not:

| σ | x | error as % of log-signal spread |
|---|---|---|
| 0.0086 (deck) | 0.4 | 9.8 % |
| 0.04 (measured max) | 0.4 | **46.4 %** |
| 0.04 | 0.05 (dark) | **623 %** |

The unrolled model still reaches 25.872 dB at 0.07 M parameters, so it works
despite the approximation being weaker than advertised — but the clean
theoretical story does not survive measurement, and 13.2 % of this corpus is
dark.

---

## Planned (not yet run)

| ID | Change | Purpose |
|---|---|---|
| 002 | Charbonnier only | isolate the loss ablation baseline |
| 003 | + 0.2 MS-SSIM | |
| 004 | + 0.1 FFT | |
| 005 | + 0.05 gradient | |
| 006 | + 0.01 VGG perceptual | |
| 007 | L2 instead of Charbonnier | demonstrate over-smoothing |
| 008 | SAFMN | fast point on the Pareto curve |
| 009 | + external data (DIV2K/Urban100) | OOD generalisation |
| 010 | wider degradation randomisation | OOD robustness |
| 011 | log-domain unrolled network | innovation track |
