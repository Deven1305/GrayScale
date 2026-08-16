# FIXING THE SOFTNESS — why the output is blurry, and what to change

## The complaint, stated precisely

Look at `docs/preview/compare_NoisyLR_0.png`. The three rows do not fail equally:

| Row | Content | Noise removed? | Detail preserved? |
|---|---|---|---|
| 1 | Crocodile head — strong structural edges | ✅ yes | ✅ good |
| 2 | **Dense rock / foliage texture** | ✅ yes | ❌ **mush** |
| 3 | Diagonal metal slats — strong periodic edges | ✅ yes | ✅ good |

**The model is not uniformly blurry. It is blurry exactly on stochastic
texture, and sharp on structural edges.** That split is the diagnostic clue,
and it points at the loss function rather than at the architecture.

This document measures why, fixes it, and shows what to run.

---

## 1. The root cause, measured

### 1.1 Why texture specifically

A regression model trained with an L1-type loss predicts the **conditional
median** of the target given the input. For a strong edge there is only one
plausible answer, so the median *is* the edge and it comes out sharp. For
stochastic texture there are thousands of plausible answers — the exact grain
of a rock face is unrecoverable after 2× decimation plus speckle at std 0.24 —
and the median of thousands of plausible textures is a **flat grey field**.

Blur is not a bug in that setting. It is the loss function getting exactly what
it asked for. So the fix is to ask for something else.

### 1.2 The v1 recipe was asking for the wrong thing

`configs/nafnet_w48.yaml` already contained two terms added specifically to
prevent this:

```yaml
fft: 0.2          # "restores the high-frequency band that decimation destroyed"
gradient: 0.1     # "keeps edges sharp"
```

`scripts/f26_loss_spectrum.py` tests whether they do that. Method: take 64 real
GT images, build the exact failure mode (a prediction with 60% of its high-pass
detail stripped), backpropagate each loss term on its own, and measure what
fraction of the resulting gradient energy lands in the **high half of the
spectrum** — the band that makes an image read as sharp.

```
64 GT images 256x256   high band = |f| >= 0.5 x f_max
  spectral ENERGY in high band :  25.57%
  spectral BINS   in high band :  60.75%

share of each term's gradient landing in the high band:

  charbonnier             52.82%
  msssim                  32.12%   -20.70% vs charbonnier   <-- below
  fft_hf_power_0_v1       60.81%    +7.99% vs charbonnier
  gradient_sobel          22.58%   -30.24% vs charbonnier   <-- below
```

Read the last line again. **The Sobel "keep edges sharp" term is the most
low-frequency-biased term in the entire composite** — 30 points *below* the
plain Charbonnier it was supposed to be sharpening. And the FFT term, at
+8.0 points, is very nearly redundant with Charbonnier.

Weight-average the whole recipe and the result is worse than doing nothing:

```
projected high-band emphasis of a whole recipe:

  v1 shipped (nafnet_w48.yaml)          49.11%   -3.71% vs plain L1
  charbonnier alone (reference)         52.82%
```

> **The v1 composite loss targets high frequencies WORSE than plain L1 with no
> sharpness terms at all.** The two terms added to keep the output sharp were,
> together, making it softer.

That is the root cause, it is measured on real data, and it is reproducible:

```bash
python scripts/f26_loss_spectrum.py
```

### 1.3 A hypothesis I had to throw away

My first explanation was that natural images have a 1/f spectrum, so an
unweighted spectral L1 would be swamped by the huge low-frequency
coefficients. **That was wrong, and measuring it is what showed the wrongness.**

The gradient of `|z|` has unit magnitude regardless of how large `z` is, so an
L1 in the frequency domain gives every bin equal gradient weight no matter its
amplitude — and since 60.75% of the bins are in the high band, the unweighted
FFT loss already sends 60.81% of its gradient there. The 1/f argument would
only hold for an L2 spectral loss.

The real problem was never that the FFT term was low-frequency biased. It was
that the FFT term was **too weak to matter (0.2)** and that the gradient term
was **actively pulling the other way**.

---

## 2. The fix — a loss change, not an architecture change

This matters for your constraint: **nothing here changes the model, the
parameter count, the weight file, or the inference cost.** Same 15.24 M
NAFNet-w48, same 4.6 ms/image. Only the training signal moves.

### 2.1 What changed

`configs/nafnet_w48_sharp.yaml`:

| Term | v1 | v2 | Why |
|---|---|---|---|
| `charbonnier` | 1.0 | 1.0 | unchanged — this is the PSNR anchor |
| `fft` | 0.2 | **0.6** | tripled, so it can actually outvote Charbonnier |
| `fft_hf_power` | *(none)* | **1.5** | **new** — radial weighting, 60.8% → 81.2% |
| `highfreq` | *(none)* | **0.5** | **new** — Charbonnier on the high-pass residual |
| `gradient` | 0.1 | **0.05** | halved — measured as the worst offender |
| `msssim` | 0.2 | 0.15 | LF-biased, but kept: SSIM is scored |
| `vgg` | 0.0 | **0.05** | **on** — LPIPS is scored; this is its trainable proxy |
| `epochs` | 40 | 60 | sharpness terms converge later than plain L1 |

Result, same measurement:

```
  v1 shipped (nafnet_w48.yaml)          49.11%
  v2 sharp   (nafnet_w48_sharp.yaml)    61.31%    +12.2 points
```

### 2.2 The two new loss terms

**Radially-weighted FFT** (`src/losses/composite.py`, `FFTLoss`). Each spectral
bin is weighted by `(|f| / f_max) ** hf_power`, so the high band gets
proportionally more gradient. `hf_power=0` reproduces the v1 loss exactly, so
the change is backwards compatible and the old behaviour stays reachable.

| `hf_power` | HF gradient share |
|---|---|
| 0.0 *(v1)* | 60.81% |
| 1.0 | 76.01% |
| **1.5** *(chosen)* | **81.19%** |
| 2.0 | 84.97% |

**High-pass Charbonnier** (`HighFrequencyLoss`). The spectral term is global and
phase-sensitive; this one is local and blunt: *wherever the target has fine
detail, the prediction must have detail in the same place at the same
amplitude.* Implemented as `x - gaussian_blur(x)`, two depthwise convs, no
learnable parameters. Measured HF share: 67.05%.

### 2.3 What to run

```bash
python train.py --config configs/nafnet_w48_sharp.yaml
```

~2.5 h on an 8 GB GPU (60 epochs vs v1's 40). Then compare honestly:

```bash
python evaluate.py --ckpt experiments/runs/nafnet_w48_sharp/best.pt
python scripts/v01_view.py compare --input_dir data/Test_NoisyLR/NoisyLR \
       --output_dir outputs --count 3
```

### 2.4 Expect a trade, and check for it

Pushing gradient into the high band should improve **SSIM**, **LPIPS** and
visible resolution. It may cost a little **PSNR**, because PSNR rewards exactly
the hedging-toward-the-mean that causes the blur. That is the
perception–distortion frontier (`12_WHY_NOT_AN_LLM.md` §8), and moving along it
is the whole point of this change.

**Ship whichever wins on the balance of all three metrics.** Do not assume v2
wins — the config is a hypothesis with a measured mechanism behind it, not a
result. Log both rows in `../experiments/EXPERIMENT_LOG.md`.

Two failure modes to watch for:

* **Ringing / halos at strong edges** — `fft` or `fft_hf_power` too high. Drop
  `fft_hf_power` to 1.0. The spec forbids "artificial patterns or ringing".
* **Noise amplified back into flat regions** — the high band contains speckle
  as well as detail. If flat areas get grainy, lower `highfreq` to 0.3.

---

## 3. If the loss change is not enough

Ranked by gain per millisecond of inference cost. The first four are **free at
inference**.

| # | Change | Inference cost | Notes |
|---|---|---|---|
| 1 | **The loss rebalance above** | **0 ms** | Start here. Mechanism is measured. |
| 2 | **Pretrained init** from NAFNet-SIDD | **0 ms** | We train from random init today — see `12_WHY_NOT_AN_LLM.md` §10.1 |
| 3 | **Distillation** from a large offline teacher | **0 ms** | The correct way to buy big-model quality under a time budget (§10.3) |
| 4 | Train longer / more external data | **0 ms** | |
| 5 | Widen to `nafnet_w64` | ~×1.8 | Returns already flat: SAFMN at 0.22 M scores within 0.35 dB of w48 |
| 6 | ×8 self-ensemble | **×8** | +0.2–0.4 dB but blows the time budget |
| 7 | GAN / diffusion | ×100–1000 | ❌ Forbidden by the spec and by Commandment 4 |

**Do not reach for a bigger model first.** The Pareto data
(`experiments/pareto.json`) shows a 0.22 M model within 0.35 dB of the 15.24 M
one — capacity is not the binding constraint here. The training signal is.

---

## 4. The file you asked about: `inference.py`

**`inference.py` is the one that does load → preprocess → model → predict →
store.** It is the scored file, it is standalone, and it imports nothing from
`src/` (the model is defined inline so no package tree gets pulled in).

```bash
python inference.py --input_dir data/Test_NoisyLR/NoisyLR --output_dir outputs
```

### The flow, with line numbers

```
  main()                                                inference.py:371
    │
 1. parse args                                                     :403
 2. glob the input dir  →  sorted file list                        :429
    │
 3. ┌─ START background header scan ─────────────────┐             :445
    │    scan_shapes()  →  thread pool reads only     │            :277
    │    the .npy HEADER of all 400 files, buckets    │
    │    them by shape. Never decodes pixels.         │
    │                                                 │  overlapped
 4. │  pick device (cuda / cpu)                       │            :447
 5. │  LOAD MODEL                                     │            :453
    │    torch.load(weights_only=True, mmap=True)     │
    │    → build arch → load_state_dict → .eval()     │            :458
    │    → .half() → channels_last                    │            :459
 6. └─ JOIN the scan; buckets are ready ─────────────┘             :466
    │
 7. warm up once per shape (only if --cudnn_benchmark)             :476
 8. start WRITER threads                                           :515
    │
 9. per shape bucket:                                              :525
    │   threaded_batches()  ── reads N files in parallel           :313
    │        ↓  torch.from_numpy(stack)                    PREPROCESS
    │   x = batch.to(device, non_blocking=True)                    :534
    │   x = x.half().to(channels_last)                             :537
    │        ↓
    │   y = model(x)                                       PREDICT :544
    │        ↓  (bicubic anchor + learned residual, PixelShuffle 2x)
    │   y.clamp_(0, 1)   ← clamp on the GPU                        :546
    │   out = y.cpu().numpy()                                      :547
    │        ↓
    │   wq.put(...)  → writer thread → np.save             STORE   :561
    └─
```

The critical preprocessing detail: **the input is never clipped.** 3.11% of
pixels sit above 1.0 and 0.28% below 0.0, and those out-of-range values encode
the local noise strength. Only the **output** is clamped, at line 546. See
`read_image()` at line 236.

### Other entry points, so the map is complete

| File | Does |
|---|---|
| **`inference.py`** | **dir → dir restoration. The scored file.** |
| `train.py` | config-driven training |
| `evaluate.py` | offline metrics vs baselines and OOD families |
| `scripts/v01_view.py` | look at `.npy` inputs/outputs as images |
| `scripts/f26_loss_spectrum.py` | the diagnostic in §1 |

---

## 5. Making inference faster

### 5.1 What changed

**Two structural problems, both fixed.**

**(a) The header scan blocked the pipeline.** Bucketing needs every input's
shape, which meant 400 serial file opens before the first batch could start —
pure latency at the front. It now runs on a **background thread that starts
before the model load** (line 445, joined at 466), so it hides behind the ~1.4 s
of `import torch` + weight loading instead of adding to it.

**(b) Reads and GPU compute were fully serialised.** The shipped default was
`--num_workers 0`, chosen because DataLoader workers *spawn* on Windows and each
re-imports torch (measured: 35.2 s vs 16.4 s over 400 images). But
`num_workers=0` reads **inline in the main loop** — so despite the three-stage
overlap in the docstring, reads and compute never actually overlapped.

The new `threaded_batches()` (line 313) uses a **thread pool** instead. `np.load`
releases the GIL during the file read, so reader threads genuinely run while the
GPU works on the previous batch — the overlap DataLoader was supposed to give,
without the spawn cost.

```
--reader auto        resolves to dataloader -- the benchmarked default (see 5.3)
--reader threads     force the thread pool
--reader dataloader  force the DataLoader path
--read_threads 8     pool size when --reader threads
```

§5.3 explains why `auto` stayed on `dataloader`: the change turned out not to be
worth the risk to the scored file.

### 5.2 Correctness first

Both readers were verified **bit-identical** across all 400 test images, against
each other and against the shipped `outputs/`:

```
threads vs dataloader   max abs difference: 0.000e+00   files differing: 0
threads vs outputs/     max abs difference: 0.000e+00   files differing: 0
```

Test coverage extended to both backends: `tests/test_inference.py` now
parametrises over `(dataloader, 0)`, `(dataloader, 2)`, `(threads, 0)`,
`(auto, 0)`. **34 tests pass.**

### 5.3 The result: no measurable speedup, and here is why

Measured, best of 3 over 400 images:

| Reader | Best | Spread |
|---|---|---|
| `dataloader` *(shipped behaviour)* | **12.07 s** | 12.07 – 14.41 |
| `threads` *(new)* | 12.31 s | 12.31 – 12.99 |

**Within noise.** The threaded reader is not faster.

The reason was already sitting in `experiments/throughput.json` and should have
been checked before writing any code. Scaling the per-stage timings to 400
images:

| Stage | Time | Share |
|---|---|---|
| **read** | **0.11 s** | **1.2%** |
| h2d | 0.02 s | 0.2% |
| compute | 2.41 s | 26.7% |
| d2h | 0.07 s | 0.8% |
| write | 0.86 s | 9.5% |
| startup | 1.40 s | 15.5% |
| **other (framework / Python)** | **4.15 s** | **46.0%** |

> **Reading is 1.2% of the runtime.** Overlapping reads perfectly could not save
> more than 0.11 s of a 9.01 s run. The optimisation targeted a slice that was
> never the bottleneck.

So `--reader auto` now resolves to `dataloader` — the previously benchmarked
behaviour. Changing the default of the scored file for no measured gain is pure
risk. `--reader threads` stays available and tested, and would pay on storage
where reads genuinely are slow: network mounts, cold cache, spinning disks.

The header-scan overlap (§5.1a) is kept regardless: it costs nothing, removes
serial latency from the front of the pipeline, and is measured inside the same
noise band.

⚠️ **A note on these numbers.** The baseline is 9.01 s from
`experiments/throughput.json`, taken on mains power. The 12 s figures above were
taken while the laptop was charging from 10%, so still power-limited — `startup`
alone was 2.2 s against the baseline's 1.4 s. They are valid for comparing the
two readers *against each other* (both were measured in the same state) but are
**not** a new absolute baseline. Do not put 12 s in the deck.

An earlier run did show 7.65 s, but it was not reproducible: the machine was
then discharging at 11% battery and every subsequent run — both readers and the
untouched baseline — landed near 17 s. Timings from a throttled laptop are
worthless in both directions.

### 5.4 Where the real headroom is

The table above says it plainly: **46% of the runtime is framework and Python
overhead**, not compute, not I/O. That is the only target worth attacking.

| Idea | Est. saving | Risk |
|---|---|---|
| Cut per-image Python work in the write loop (400 iterations of path building + array slicing + queue put) | part of the 4.15 s | Low |
| One `DataLoader` for all buckets instead of one per bucket | part of the 4.15 s | Low |
| Larger `--batch_size` (fewer `.cpu()` syncs) | small | Free to try |
| Write fp16 instead of fp32 | ~0.4 s (halves 102 MB) | Grader may expect fp32. **Ask KLA first.** |
| Overlap D2H with a pinned staging buffer | ~0.07 s | Small reward |
| `torch.compile` | **negative** | 30–120 s compile. Forbidden by Commandment 7. |

Startup (1.40 s, 15.5%) is dominated by `import torch` and is essentially
irreducible without a different runtime. Compute (2.41 s, 26.7%) is already
fp16 + channels_last on a 15 M model.

**Before optimising anything further, re-run `scripts/benchmark_throughput.py`
on mains power at full charge** so there is a trustworthy baseline to measure
against.

---

## 6. Summary

| Question | Answer |
|---|---|
| Why is it blurry? | L1-type losses predict the conditional median; on stochastic texture that is a flat field |
| Why didn't the sharpness terms help? | **Measured:** the v1 composite (49.11%) targets high frequencies *worse* than plain L1 (52.82%). The Sobel term is 30 points below Charbonnier |
| Architecture change needed? | **No.** Capacity is not the constraint — a 0.22 M model scores within 0.35 dB of the 15.24 M one |
| What to change? | Loss weights + two new terms → `configs/nafnet_w48_sharp.yaml`, projected 61.31% |
| Inference cost of the fix? | **Zero.** Same model, same weights format, same 4.6 ms/image |
| Which file runs load→predict→store? | **`inference.py`** — flow mapped with line numbers in §4 |
| Is it faster now? | **No — and measurement says it cannot be.** Reads are 1.2% of runtime; 46% is framework overhead (§5.3). Outputs are bit-identical; the default keeps the benchmarked path |
| Next best free wins | Pretrained init, then distillation — both 0 ms (`12_WHY_NOT_AN_LLM.md` §10) |

---

*Reproduce §1 with `python scripts/f26_loss_spectrum.py` → `experiments/loss_spectrum.json`.*
