# THE DATASET — what it is, how to look at it, what to do with it

> Companion to `05_FORENSICS_REPORT.md` (which proves the claims here).
> This file is the plain-language version.

---

## 1. THE ONE-PARAGRAPH VERSION

KLA gave us **3200 training pairs**. Each pair is one **clean 256×256 image** (`GT`)
and the **degraded 128×128 version** of it (`NoisyLR`). The job is to learn the
inverse: given the 128×128 degraded image, reproduce the 256×256 clean one.
They also gave **400 degraded test images with no clean version** — those are for
us to restore and hand back. Everything is stored as **float32 NumPy arrays**
(`.npy`), *not* as PNG or JPEG, which is why you cannot double-click them.

---

## 2. WHERE THE FILES ARE

```
data/
├── train/train/
│   ├── GT/          000000.npy … 003199.npy    3200 files · 256×256 · float32
│   └── NoisyLR/     000000.npy … 003199.npy    3200 files · 128×128 · float32
└── Test_NoisyLR/
    └── NoisyLR/     000000.npy … 000399.npy     400 files · 128×128 · float32
```

**Pairing is by filename.** `GT/001234.npy` is the clean version of
`NoisyLR/001234.npy`. Nothing else links them — there is no manifest, no CSV,
no metadata file of any kind. We checked; §1 of the forensics report lists every
non-image file in `data/` (they are all macOS zip artefacts).

Total on disk: **2.0 GB**. It is gitignored and must never be committed — it is
KLA's data, not ours to redistribute.

---

## 3. WHAT "GT" AND "NoisyLR" MEAN

| | GT | NoisyLR |
|---|---|---|
| Stands for | Ground Truth | Noisy, Low-Resolution |
| Role | the **target** — what the model should output | the **input** — what the model receives |
| Size | 256×256 | 128×128 |
| Value range | **exactly [0, 1]** | **−0.279 … 2.158** |
| Pixels below 0 | 0 | 0.28 % |
| Pixels above 1 | 0 | **3.11 %** |

### ⚠️ The single most important property

**NoisyLR values go outside [0, 1], and you must never clip them.**

3.11 % of input pixels are above 1.0 and 0.28 % are below 0. This is not
corruption — speckle is *multiplicative*, so a bright pixel multiplied by a
noise value above 1 lands above 1. Those out-of-range values carry information
about how strong the noise was locally, which is exactly what a denoiser needs.

Clip the input and you throw that away. Clip only at the very end, when saving a
final image for human viewing.

This is also *why* the data ships as float32 `.npy` rather than 8-bit PNG: an
8-bit PNG **cannot represent** a value of 1.4 or −0.05. The format choice is a
deliberate signal.

---

## 4. HOW THE DATASET IS STRUCTURED — 800 photographs, 4 crops each

This is not 3200 independent images. It is **800 source photographs**, each cut
into **4 crops**, laid out in consecutive blocks:

```
source 0   ->  samples 000000, 000001, 000002, 000003
source 1   ->  samples 000004, 000005, 000006, 000007
source 2   ->  samples 000008, 000009, 000010, 000011
...
source 799 ->  samples 003196, 003197, 003198, 003199
```

So:

```python
source_id = sample_index // 4
```

We proved this rather than assumed it: visual similarity between neighbouring
samples is high inside a block and drops to **exactly zero** at the boundary
between sample `4k+3` and `4k+4`. Three independent descriptors agree.
(Run `python scripts/f20_period4.py` to see it.)

### 🔴 Why this matters more than it sounds

Samples 000001, 000002 and 000003 are **overlapping crops of the same
photograph**. If you shuffle all 3200 samples and split 80/20 at random, crops of
the same photo end up on both sides. Your validation score then measures how well
the model *memorised that photograph*, not how well it generalises — and it will
look fantastic while being meaningless.

**Always split on `source_id`, never on sample index.**

---

## 5. WHAT THE IMAGES ACTUALLY SHOW

**Ordinary photographs.** Buildings, foliage, water, brick walls, fabric, people,
animals, street signs, books. One crop even carries a photographer's watermark
("© Magnus …").

They are **not** semiconductor images, and they are **not** the
dendrite/microscopy pictures shown in KLA's slides — those come from a different
release of the dataset (512→256 instead of 256→128). See §7 of the forensics
report.

800 sources = exactly the size of **DIV2K**, a standard public super-resolution
dataset of Flickr photographs, and the content matches. Strong hypothesis,
not yet verified by download.

There are **no distinct "categories"** in the data — clustering the 800 sources
produces bright/dark/high-texture groupings, not different kinds of imagery.
13.2 % of sources are dark (mean < 0.25).

---

## 6. WHAT WAS DONE TO THE IMAGES (the degradation)

Reconstructed from the paired data in Phase 0:

```python
y = downsample_cubic_2x( x * n + g )

  x = clean 256×256 image
  n ~ Gamma(L, 1/L)        speckle: MULTIPLICATIVE, mean 1, variance 1/L
  g ~ Normal(0, sigma²)    additive Gaussian noise
  downsample: 4-tap cubic convolution, 2×, applied LAST
```

Measured parameters:

| | value |
|---|---|
| `L` (speckle looks) | median **17.7**, middle half 14.3 – 21.8 |
| speckle std = 1/√L | **≈ 0.24** |
| `sigma` (Gaussian) | 0.001 – 0.04 |
| **ratio** | speckle dominates by **6× to 240×** |

### The one thing to take away

The problem statement lists three degradations as if they were equals. **They are
not.** Speckle carries almost all the noise energy. Build a *despeckler* that
also handles the rest — do not split your effort in three.

Two further findings that contradict the original brief:

* The three operations are applied in a **fixed order** (noise first, downsample
  last), not a random one.
* Only **256→128** exists in this release. KLA's slides show 512→256, so keep the
  model fully convolutional — but there is nothing to train on at that size here.

---

## 7. HOW TO LOOK AT THE IMAGES

`.npy` is a raw array file. Windows Photos, Paint and browsers cannot open it.
Use the viewer:

```bat
cd /d "<path-to-repo>"
conda activate klasr
```

### a) One pair, side by side, with a zoomed crop — **start here**

```bat
python scripts\v01_view.py pair --index 2204
```

Writes `docs\preview\pair_002204.png`. Six panels: GT, NoisyLR, NoisyLR upscaled
bicubic, two zoomed crops so you can actually see the speckle, and the histogram
showing the out-of-range tail.

Move the zoom box with `--cx`, `--cy`, `--crop`:

```bat
python scripts\v01_view.py pair --index 2204 --cx 128 --cy 32 --crop 64
```

### b) Convert to PNG and browse in File Explorer

```bat
python scripts\v01_view.py export --split train_gt --limit 40
python scripts\v01_view.py export --split train_lr --limit 40
python scripts\v01_view.py export --split test     --limit 40
```

Writes to `docs\preview\<split>\`. Open the folder and use arrow keys.
*(Values outside [0,1] are clipped for display only — the source data is untouched.)*

### c) Contact sheet — many images at once

```bat
python scripts\v01_view.py sheet --split train_gt --start 0 --count 60
```

### d) Numbers for one sample

```bat
python scripts\v01_view.py stats --index 0
```

```
=== sample 000000 ===
  source image id : 0   (crop 0 of 4 from that photograph)
  GT        shape=(256, 256)  min=+0.00000  max=+1.00000  mean=0.21819
  NoisyLR   shape=(128, 128)  min=-0.00264  max=+1.32578  mean=0.21844
            below 0:  0.031%   above 1:  0.537%
  siblings (same source): ['000000', '000001', '000002', '000003']
```

### e) In your own code

```python
import numpy as np
gt = np.load("data/train/train/GT/000000.npy")        # (256,256) float32
lr = np.load("data/train/train/NoisyLR/000000.npy")   # (128,128) float32
# do NOT np.clip(lr, 0, 1)
```

---

## 8. THE TEST SET — and an open question

`data/Test_NoisyLR/NoisyLR/` holds **400 degraded images with no ground truth.**
You cannot compute PSNR/SSIM/LPIPS on them yourself; KLA holds the clean
versions and scores you against them.

Its statistics are close to training but not identical — slightly brighter
(mean 0.443 vs 0.434) and **2.3× more negative pixels**. Worth remembering.

> ⚠️ Note the wording in the official materials: *"test set will be shared after
> the training phase."* The 400 images we have may be a practice set rather than
> the scored one. Assume the scored set arrives later and may include
> out-of-distribution content — the dendrite/microscopy imagery from KLA's slides
> is the obvious candidate.

**Open question for the organisers:** the input is `.npy`, but nothing states
what format the *output* should be. `.npy`? PNG? uint8 or float? This affects the
inference script directly and is worth asking.

---

## 9. WHAT IS *NOT* IN THE DATASET

| Missing | Consequence |
|---|---|
| Any metadata file (`L`, `sigma`, source ID per sample) | We reconstructed the degradation ourselves — Phase 0 |
| Ground truth for the test set | Cannot self-score; needs synthetic OOD validation instead |
| 512→256 pairs | Only 256→128 exists, despite the brief |
| Source photograph filenames | Recovered structurally as `index // 4` |
| Any semiconductor imagery | It is natural photography |

---

## 10. RULES OF THUMB

✅ **Do**

- Load with `np.load`, keep `float32`
- Split by `source_id = index // 4`
- Use `data_range=1.0` in PSNR/SSIM (check `PSNR(x, x) == inf`)
- Treat speckle as the dominant problem
- Keep the model fully convolutional

❌ **Don't**

- Clip the input to [0, 1]
- Split randomly by sample index
- Convert to uint8 anywhere in the training path
- Assume the three degradations matter equally
- Commit `data/` to git

---

## 11. WHERE TO GO DEEPER

| Question | File |
|---|---|
| How do you *know* all this? | `05_FORENSICS_REPORT.md` |
| How do I set up / run things? | `01_SETUP.md` |
| What happens next? | `PROMPT.md` (dev repo) (Phase 1 onward) |
| The rules that must not drift | `CLAUDE.md` (dev repo) |
