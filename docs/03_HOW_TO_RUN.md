# HOW TO RUN — complete end-to-end guide

Everything from a fresh clone to restored images, with nothing assumed.
If you only read one section, read **§2**.

---

## 0. THE 30-SECOND VERSION

```bash
conda activate klasr
cd <path-to-repo>
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

400 degraded 128×128 images in → 400 restored 256×256 images out, ~9 seconds.

---

## 1. THE WHOLE PIPELINE AT A GLANCE

```
                          ┌─────────────────────────────────┐
                          │  RAW DATA (KLA-supplied, 2 GB)  │
                          │  3200 train pairs + 400 test    │
                          └───────────────┬─────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         │                                │                                │
         ▼ (A) UNDERSTAND                 ▼ (B) BUILD                      ▼ (C) USE
  ┌──────────────┐               ┌─────────────────┐              ┌────────────────┐
  │ f01…f25      │               │ precrop_patches │              │ inference.py   │
  │ forensics    │               │ → memmap bundle │              │ dir → dir      │
  │ v01_view     │               └────────┬────────┘              └───────┬────────┘
  │ image viewer │                        │                               │
  └──────┬───────┘                        ▼                               ▼
         │                       ┌─────────────────┐              ┌────────────────┐
         ▼                       │ run_baselines   │              │ outputs/*.npy  │
  05_FORENSICS_REPORT.md       │ → the FLOOR     │              │ 400 restored   │
  04_DATASET.md                └────────┬────────┘              └────────────────┘
                                          │
                                          ▼
                                 ┌─────────────────┐
                                 │ train.py        │  40 epochs, ~1.6 h
                                 │ + configs/*.yaml│
                                 └────────┬────────┘
                                          │
                                          ▼
                              experiments/runs/*/best.pt  (245 MB)
                                          │
                                          ▼ export_weights.py
                                 weights/model_fp16.pt   (30 MB)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             evaluate.py          benchmark_throughput   consistency_check
             quality vs GT        speed                  quality WITHOUT GT
```

**(A)** answers *what is the data?*
**(B)** answers *can we learn to undo it?*
**(C)** answers *produce the deliverable.*

---

## 2. RUNNING INFERENCE — the part that is scored

### The contract

```bash
python inference.py --input_dir <folder of degraded images> \
                    --output_dir <folder for restored images>
```

| | |
|---|---|
| Input | any folder of `.npy` (float32) or `.png` / `.tif` images |
| Output | same base filenames, **exactly 2× the input resolution** |
| Weights | `weights/model_fp16.pt`, found relative to the script — no config needed |
| Manual edits required | **none** |

### Try it on the shipped sample first

```bash
python inference.py --input_dir sample_test --output_dir out_sample
```

`sample_test/` holds 6 images (385 KB) so you can verify the pipeline without
the 2 GB dataset. Expected output:

```
[info] 6 input images
[info] arch=nafnet_w48 device=cuda startup=1.4s
[info] shape buckets: {'128x128': 6}
[info] warmup 0.03s
[done] wrote 6/6 images to out_sample
[time] startup 1.4s | warmup 0.05s | total 2.6s | 2.3 img/s
```

### The full test set

```bash
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

```
[info] 400 input images
[done] wrote 400/400 images to outputs
[time] startup 1.4s | warmup 0.03s | total 9.0s | 44.4 img/s
```

### Useful flags

| Flag | Default | When to change it |
|---|---|---|
| `--batch_size` | 16 | Measured optimum. Larger was *slower* here |
| `--num_workers` | auto | 0 where workers spawn (Windows/macOS), 4 where they fork (Linux). Measured: 4 workers 35.2 s vs 0 workers 16.4 s under spawn |
| `--output_format` | `same` | Force `npy` or `png` |
| `--cudnn_benchmark` | off | Turn **on** only for very large runs (>15k images) |
| `--device` | auto | Force `cpu` to test without a GPU |
| `--weights` | `weights/model_fp16.pt` | Point at a different checkpoint |

### ❓ Does inference run on the test data only?

**No — it runs on whatever folder you point it at.** It has no notion of
"train" or "test"; it reads a directory and writes a directory.

We *ran* it on the test set because that is submission item #5. You can equally
point it at training images, your own photos, or anything else:

```bash
python inference.py --input_dir data/train/train/NoisyLR --output_dir out_train
```

---

## 3. LOOKING AT THE DATA — `.npy` explained

### What `.npy` is, and why it is not a picture file

`.npy` is **NumPy's raw array format**: a ~128-byte header (dtype, shape, byte
order) followed by raw binary numbers. It is a *matrix on disk*, not an image.
Windows Photos, Paint, Chrome and Explorer thumbnails cannot open it, because
there is no image format inside — no PNG chunks, no JPEG markers, no compression.

### Why KLA chose it instead of PNG

Because **PNG cannot store the data**. An 8-bit PNG holds integers 0–255,
mapping to [0, 1]. The degraded images contain:

| | value |
|---|---|
| maximum | **2.158** |
| minimum | **−0.279** |
| pixels above 1.0 | **3.11 %** |
| pixels below 0 | 0.28 % |

Speckle is *multiplicative*: a bright pixel times a noise value above 1 lands
above 1. Save that as PNG and 3.11 % of the pixels get silently clipped to
white — and those out-of-range values are precisely the local noise-strength
signal the model needs. The float32 `.npy` choice is a deliberate hint.

### How to actually see them

```bash
# 1. one pair, side by side, with a zoomed crop  ← START HERE
python scripts/v01_view.py pair --index 2204

# 2. convert to PNG so you can browse in your file manager
python scripts/v01_view.py export --split train_gt --limit 40
python scripts/v01_view.py export --split train_lr --limit 40
python scripts/v01_view.py export --split test     --limit 40

# 3. a contact sheet of many images at once
python scripts/v01_view.py sheet --split train_gt --start 0 --count 60

# 4. the numbers for one sample
python scripts/v01_view.py stats --index 0
```

Everything lands in `docs/preview/`.

`pair` gives six panels: GT, NoisyLR, bicubic upscale, two zoomed crops where
the speckle is visible, and the histogram showing the out-of-range tail. Move
the zoom box with `--cx --cy --crop`.

### In your own code

```python
import numpy as np

gt = np.load("data/train/train/GT/000000.npy")        # (256, 256) float32
lr = np.load("data/train/train/NoisyLR/000000.npy")   # (128, 128) float32

print(lr.min(), lr.max())        # e.g. -0.003  1.326
# ❌ NEVER do this before feeding a model:
# lr = np.clip(lr, 0, 1)
```

Clip only when saving a picture for a human to look at.

---

## 4. WHERE EVERY FILE LIVES

| What | Path | Notes |
|---|---|---|
| Training inputs (degraded) | `data/train/train/NoisyLR/` | 3200 × 128² |
| Training targets (clean) | `data/train/train/GT/` | 3200 × 256² |
| Test inputs | `data/Test_NoisyLR/NoisyLR/` | 400 × 128², **no ground truth** |
| **Restored test outputs** | **`outputs/`** | **400 × 256² — submission item 5** |
| Sample inputs for reviewers | `sample_test/` | 6 images, 385 KB |
| Shipped weights | `weights/model_fp16.pt` | 30 MB |
| Training checkpoints | `experiments/runs/*/` | gitignored, 80–245 MB each |
| Memmap bundle | `data/processed/` | gitignored, ~1 GB, rebuildable |
| Human-viewable PNGs | `docs/preview/` | gitignored, rebuildable |
| Figures for the deck/report | `docs/figures/` | tracked |

---

## 5. REPRODUCING THE WHOLE THING

Run in this order. Times are for one consumer GPU with 8 GB VRAM.

```bash
# ── 0. environment (once) ───────────────────────────────────────────────
conda activate klasr

# ── 1. pack the dataset for fast loading (~11 s, once) ──────────────────
python scripts/precrop_patches.py

# ── 2. measure the floor (~3 min) ───────────────────────────────────────
python scripts/run_baselines.py --limit 120
#    → experiments/baselines.json
#    bicubic  PSNR 23.07  SSIM 0.5129  LPIPS 0.4425
#    BM3D     PSNR 25.96  SSIM 0.6527  LPIPS 0.5576

# ── 3. train (~35 min, 40 epochs) ───────────────────────────────────────
python train.py --config configs/nafnet_w32.yaml
#    → experiments/runs/nafnet_w32/best.pt  (see 5b for the shipped w48 run)

# ── 4. slim the weights for inference (~2 s) ────────────────────────────
python scripts/export_weights.py \
    --ckpt experiments/runs/nafnet_w32/best.pt \
    --out weights/model_fp16.pt
#    80 MB → 10 MB, and self-verifies it loads under weights_only=True

# ── 5. score it against ground truth (~2 min) ───────────────────────────
python evaluate.py --ckpt experiments/runs/nafnet_w32/best.pt --proxy-ood
#    → experiments/eval_results.json
#    val  PSNR 25.867  SSIM 0.7050  LPIPS 0.3006

# ── 6. produce the deliverable (~18 s) ──────────────────────────────────
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs

# ── 7. optional: speed, no-GT quality, figures ──────────────────────────
python scripts/benchmark_throughput.py
python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w48/best.pt
python scripts/r01_visual_report.py
python scripts/make_results_table.py      # rewrites the README table
```

## 5b. FINISHING THE REMAINING PHASES

All of this is implemented; it just needs GPU time. Run in order — each step
writes a JSON artefact the later steps and the report generators consume.

```bash
# ── external content (once, ~15 min, needs network) ─────────────────────
python scripts/build_external_data.py
#    DIV2K  -> data/external/train/    800 crops, training diversity
#    Urban100 / BSD100 / Set14 -> data/external/ood/   214 images
python scripts/precrop_patches.py         # rebundles KLA + external

# ── the scaled primary (~1.5 h, 40 epochs) ──────────────────────────────
python train.py --config configs/nafnet_w48.yaml
#    15.2 M params, 96 px crops, 7.1 GB of 8.5 — uses the whole GPU,
#    where the w32 run used a quarter of it

# ── the fast Pareto point and the innovation track (~40 min each) ───────
python train.py --config configs/safmn.yaml
python train.py --config configs/log_unrolled.yaml

# ── loss ablations, 8 single-variable runs (~1.5 h) ─────────────────────
python scripts/run_ablations.py --epochs 12

# ── evaluate everything ─────────────────────────────────────────────────
python scripts/export_weights.py --ckpt experiments/runs/nafnet_w48/best.pt \
                                 --out weights/model_fp16.pt
python evaluate.py --ckpt experiments/runs/nafnet_w48/best.pt --proxy-ood \
       --ood data/external/ood/Urban100 data/external/ood/BSD100 \
             data/external/ood/Set14
python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w48/best.pt
python scripts/build_pareto.py
python scripts/measure_log_error.py

# ── regenerate every report and figure from the JSON ────────────────────
python scripts/make_reports.py        # ablation_results.md + ood_report.md
python scripts/make_results_table.py  # README results table
python scripts/r01_visual_report.py   # before/after figures
python scripts/d01_deck_figures.py && python scripts/d02_build_deck.py

# ── produce the deliverable ─────────────────────────────────────────────
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

**Total GPU time ≈ 4–5 hours.** Nothing needs supervision; each stage is
resumable because completed runs are skipped.

---

### Watching training

```bash
tensorboard --logdir experiments/runs
```

Or read `experiments/runs/nafnet_w48/history.json` directly:

```bash
python -c "
import json
h = json.load(open('experiments/runs/nafnet_w48/history.json'))
for r in h:
    if 'val/psnr' in r:
        print(f\"ep{r['epoch']:3d}  PSNR {r['val/psnr']:.3f}  SSIM {r['val/ssim']:.4f}\")
"
```

---

## 5c. BUILDING THE SUBMISSION FOLDER

The working repository contains development material — forensics scripts, deck
tooling, the brief — that a reviewer was not asked to look at. `submission/`
holds only what `brief/reference/submission_criteria.txt` requires.

```bash
python scripts/build_submission.py
```

It copies the six required items plus the docs and evidence, writes a
submission-specific README and .gitignore, then verifies all six criteria are
satisfied and refuses to claim success if any is missing.

Push it as its own repository:

```bash
cd submission
git init -b main
git lfs install && git lfs track "weights/*.pt" "outputs/*.npy"
git add -A && git commit -m "KLA PS01 submission"
git remote add origin <NEW_REPO_URL>
git push -u origin main
```

⚠️ `submission/` is gitignored in this repo on purpose — committing it here
would duplicate 400 LFS outputs and the weights.

---

## 5d. NO NVIDIA GPU?

Teammates on CPU-only laptops can run inference, the tests, the forensics and
the viewer; training needs Colab. Full instructions:
**`02_RUN_WITHOUT_GPU.md`**.

---

## 6. IS IT WORKING? — how to tell

Four independent checks, in increasing order of effort.

### Check 1 — the test suite (60 s)

```bash
python -m pytest tests/ -q
```

Expect **33 passed**. These assert:

| Test | Catches |
|---|---|
| `PSNR(x,x) == inf` | a wrong `data_range`, which would silently invalidate every metric |
| no shared source between splits | validation leakage |
| replica histogram overlap > 0.95 | the degradation model drifting from reality |
| output is exactly 2×, in [0,1] | a broken inference contract |
| survives a corrupt input file | a crash on KLA's machine |
| no absolute paths / banned imports | an unrunnable submission |

### Check 2 — beats the floor

```bash
python evaluate.py --ckpt experiments/runs/nafnet_w48/best.pt
```

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic (floor) | 23.067 | 0.5129 | 0.4425 |
| BM3D + bicubic | 25.956 | 0.6527 | 0.5576 |
| **ours** | **26.415** | **0.7333** | **0.2455** |

Better than **both** on all three. **If a model does not beat bicubic, it has a bug, not a
design problem.**

### Check 3 — look at the pictures

```bash
python scripts/r01_visual_report.py
```

Writes four figures to `docs/figures/`:

| Figure | Data | Ground truth? |
|---|---|---|
| `report_val_with_gt.png` | held-out validation | ✅ |
| `report_landscape.png` | landscape/scenery | ✅ |
| `report_test_no_gt.png` | the real 400 test images | ❌ |
| `report_synthetic_ood.png` | content absent from training | ✅ (we generate it) |

### Check 4 — quality where there is no ground truth

The 400 test images have no clean version — KLA holds it, so PSNR/SSIM/LPIPS
are impossible there. Instead:

```bash
python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w48/best.pt
```

```
x̂ = model(y)              restore
ŷ = degradation(x̂)        re-degrade with our replica
error = ‖ŷ − y‖           should be small
```

| | RMSE |
|---|---|
| median | 0.0924 |
| p95 | 0.1467 |
| max | 0.1957 |

The worst 15 images are listed in `experiments/consistency.json` — look at them.

⚠️ Consistency is **necessary, not sufficient**: an over-smoothed output can
still re-degrade convincingly, because the degradation destroys high
frequencies anyway. Use it to catch gross failure, not to rank good models.

**The full no-ground-truth toolkit is in `10_IMPROVEMENTS.md` Part 3.**

---

## 7. WHAT THE MODEL DOES WELL AND BADLY

Measured, from `scripts/r01_visual_report.py`.

| Content | bicubic → ours (PSNR) | bicubic → ours (SSIM) | verdict |
|---|---|---|---|
| smooth gradient | 25.24 → **42.25** | 0.390 → **0.983** | ✅ excellent |
| text-like strokes | 17.71 → **19.61** | 0.342 → **0.785** | ✅ strong |
| geometric shapes | 25.02 → **29.94** | 0.840 → **0.898** | ✅ good |
| dense foliage | 20.05 → 23.4 | 0.301 → 0.38 | ⚠️ over-smoothed |
| fine periodic grid | 15.64 → 15.29 | 0.808 → 0.800 | ⚠️ **still just below bicubic** (patterns narrowed it) |

**Summary:** the despeckler works very well. Detail *reconstruction* does not
keep up — the model has learned "when uncertain, smooth", which is safe under
the loss but is exactly what the spec warns against. Fine periodic structure is
the one case where it loses outright, and that matters because semiconductor
patterns are periodic.

---

## 8. COMMON PROBLEMS

| Symptom | Fix |
|---|---|
| `'conda' is not recognized` | Use **Anaconda Prompt**, not plain cmd |
| `torch.cuda.is_available()` → False | Wrong PyTorch build — see `01_SETUP.md` §2 |
| Training crawls below 20 img/s | Run `python scripts/precrop_patches.py` |
| `CUDA out of memory` | Lower `data.batch_size`, raise `accum_steps` |
| Inference is slow | Leave `--num_workers` and `--cudnn_benchmark` at their defaults — both are already the measured optimum |
| Outputs look flat/grey | You are viewing raw `.npy` in a tool that assumes 0–255. Use `v01_view.py` |
| Windows path errors | Quote paths, and use `cd /d` to change drive |
