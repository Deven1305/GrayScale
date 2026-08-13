# WHY NOT AN LLM? — the architecture decision, defended with numbers

## Or: "shouldn't we just fine-tune an open-weight SLM/VLM on this data instead?"

**Short answer:** No — but the instinct behind the question is *half right*, and
this document says exactly which half, because that half is worth acting on.

A large language model, a small language model, or a vision-language model
fine-tuned on our 3,200 image pairs would be **slower by 23× to 650×**, would
hit a **quality ceiling below the score we already have**, would **not fit on
the hardware we have**, and would optimise **one of the three scored metrics at
the cost of the other two**. Each of those four claims is quantified below.

The half that is right: *pretrained open weights are permitted and we should be
using them.* Just not from a language model. Section 10 says which weights, and
names the one thing we genuinely left on the table.

---

## 0. How to read the numbers in this document

The other documents in `docs/` carry a promise: nothing in them is estimated.
**This document breaks that promise deliberately**, because comparing against a
system we did not build requires estimating it. So everything is labelled:

| Tag | Meaning |
|---|---|
| 📏 **Measured** | Produced by a script in this repository. Traceable to `experiments/*.json`. |
| 📐 **Derived** | Arithmetic on measured numbers. Reproduce it yourself from the figures given. |
| 📚 **Published** | From the literature, cited. Approximate, order-of-magnitude. |
| 🔮 **Estimated** | My projection for a system we did not build. Treat with suspicion; §12 says how to falsify it. |

Nothing in the argument below depends on a 🔮 figure being exactly right. The
gaps are one to three orders of magnitude wide. They survive being wrong by 2×.

---

## 1. The question, made precise

"Use an open-source LLM/SLM" turns out to be three quite different proposals
that people run together. They fail for different reasons, so they need
separating.

| # | Proposal | What it concretely means | Verdict |
|---|---|---|---|
| **A** | Fine-tune a text LLM (Qwen2.5-0.5B, SmolLM2-1.7B, Phi-3-mini) on the image pairs | Serialise pixels into tokens, train next-token prediction | ❌ Fails hardest. §4, §6 |
| **B** | Fine-tune an open **vision-language** model (Qwen2-VL-2B, LLaVA, Phi-3.5-vision) | Feed the degraded image to a vision encoder, decode the restored image as tokens | ❌ Fails on all six axes. §4–§9 |
| **C** | Fine-tune an open **image foundation model** (Stable Diffusion, StableSR, DiffBIR, SUPIR, SwinIR, Restormer) | Use a pretrained *visual* prior, not a linguistic one | ⚠️ **The right family.** Diffusion variants lose on time (§10.2); the deterministic ones we **should** have used (§10.1) |

Most people asking the question mean **B** and are reaching for the intuition
behind **C**. That intuition is sound. The vehicle is wrong.

---

## 2. Verdict table

Everything in this table is defended in the section named.

| Axis | Our NAFNet-w48 | Fine-tuned 2B VLM | § |
|---|---|---|---|
| Parameters | 📏 **15.24 M** | 🔮 ~2.2 B (**144×**) | §9 |
| Time, 400 images | 📏 **9.0 s** | 🔮 205 s – 98 min (**23–650×**) | §6 |
| Forward passes per image | 📏 **1** | 📐 ~1,024 sequential | §6 |
| Quality ceiling | 📏 26.415 dB *achieved* | 📚 ~20–24 dB *tokenizer round-trip alone* | §4 |
| Handles input values > 1.0 | 📏 Yes, by design | 🔮 **No — clipped by the preprocessor** | §5.3 |
| Optimises which scored metrics | 📏 All three | 📐 LPIPS only; loses PSNR + SSIM **by theorem** | §8 |
| Fits in 8 GB to train | 📏 Yes — peak **2.1 GB** | 🔮 No — ~10–16 GB for LoRA | §9 |
| Training data needed | 📏 3,200 pairs sufficed | 📚 10⁵–10⁹ pairs | §7 |
| Hallucinates structure | 📏 No (deterministic) | 📐 Yes, by construction | §8.3 |

---

## 3. The constraint that decides everything

You stated the requirement: **execution time must stay low.** That is not a soft
preference here — it is a scored axis. From `CLAUDE.md` (dev repo):

> Timing includes **script startup + model init**, disk read, inference, disk write.

Our current measurement, 📏 from `experiments/throughput.json`:

```
400 images · 8 GB consumer GPU · best of 3 warm runs

  startup (python + torch + weights)   1.4  s
  per-shape warmup                     0.03 s
  ─────────────────────────────────────────
  TOTAL                                9.0  s        44.4 images/s
                                                     22.5 ms/image end-to-end
                                                      4.6 ms/image model compute
```

Note what dominates: **1.4 s of the 9.0 s is `import torch` and loading
weights** — 16% of the entire budget, before a single image is read. That is why
`inference.py` defines its model inline and imports nothing it does not need.

Now hold that number in mind. **9.0 seconds, everything included.** Every
alternative below has to beat it, or beat our quality by enough to justify
losing it.

---

## 4. Reason 1 — the output is 65,536 calibrated floats, not a sentence

This is the deepest reason and the one that cannot be engineered around.

An LLM emits a **discrete token from a fixed vocabulary**, one at a time. Our
task emits a **256×256 grid of continuous values**, all at once, each accurate to
about two parts in 256. These are different kinds of object.

### 4.1 The precision we actually need

📐 PSNR converts to per-pixel error. At `data_range = 1.0`:

```
RMSE = 10^(−PSNR/20)

  our current 26.415 dB  →  RMSE = 0.0478   (12.2 grey levels out of 256)
  a hypothetical 28 dB   →  RMSE = 0.0398   (10.2 grey levels)
                            ─────────────────
                  the gain =  0.0080         ( 2.0 grey levels)
```

**To gain 1.6 dB, you must tighten the error on all 65,536 pixels by two grey
levels out of 256 — one part in 125 of the dynamic range.** That is the scale of
precision this task operates at. Keep that "one part in 125" in mind for the
next subsection.

### 4.2 The tokenizer ceiling — the argument that ends the discussion

To emit an image autoregressively you must first *discretise* it. Every
autoregressive image model does this with a VQ tokenizer (VQGAN, VQ-VAE,
MoVQ, or a modern successor). Take the standard configuration:

```
📐 VQGAN, downsample factor f = 8, codebook size 16,384

   output image          256 × 256  = 65,536 pixels
   token grid            32 × 32    =  1,024 tokens
   bits per token        log2(16384) =    14 bits
   ────────────────────────────────────────────────
   total budget          1,024 × 14  = 14,336 bits
   per pixel             14,336 / 65,536 = 0.219 bits/pixel
```

Compare that to what we emit: **16 bits/pixel** (fp16). Even against an ordinary
8-bit PNG it is a **36× information bottleneck**. And §4.1 established that the
improvement we are chasing lives at one part in 125 — roughly 7 bits of
precision — in a channel that carries 0.22.

The consequence is measurable and has been measured, by other people:

> 📚 Published f=8 VQ tokenizers reconstruct **clean, undegraded** natural
> images at roughly **20–24 dB PSNR**. The continuous (non-quantised) KL
> autoencoder used by Stable Diffusion does better, around 25–26 dB, and it is
> not usable for autoregressive decoding precisely *because* it is continuous.

Read that against our result:

```
   VQ tokenizer round-trip of a CLEAN image     📚 ~20–24 dB
   our model, restoring a DEGRADED image        📏  26.415 dB
```

**The tokenizer's ceiling is below our floor.** A perfect language model, making
zero errors, predicting exactly the right token every time, decoding a *clean*
image it was handed for free, would still land below where we already are — not
because the model is bad, but because the representation it is forced to speak
in cannot carry the precision. Everything the LLM contributes happens *inside*
that ceiling.

This is not a tuning problem. It is the container being smaller than the
contents.

### 4.3 The alternative — and why it is just our model again

You could avoid the tokenizer by having the model emit continuous values from a
regression head. But at that point you have deleted the vocabulary, deleted the
softmax, deleted autoregressive decoding, and deleted the entire reason the
pretrained weights were useful. What remains is a very large, very slow
transformer doing dense regression — which is a *worse* SwinIR, and SwinIR is
option **C** in §1, discussed in §10.

---

## 5. Reason 2 — a vision encoder deletes exactly the signal we need

Suppose we sidestep §4 by only *reading* with the VLM. The encoder is still
wrong for this task, for three separate reasons.

### 5.1 Patch embeddings are semantic summaries, not pixels

A CLIP ViT-L/14 or SigLIP encoder cuts the image into **14×14 patches** and maps
each to a ~1024-dimensional vector **trained to align with text**. That training
objective rewards "this patch contains fur", "this is sky", "there is an edge
here". It has no incentive whatever to preserve the exact amplitude of a
single pixel — and pretraining actively discards it, because two images
differing only in noise realisation should map to the *same* embedding for
retrieval to work.

Our signal is 📏 speckle with **std ≈ 0.24**, living at the individual-pixel
level, on top of additive noise with σ between 0.001 and 0.04. The whole task is
separating per-pixel noise from per-pixel structure.

**The vision encoder is trained to be invariant to precisely the thing we must
estimate.** Feeding a despeckling problem through a semantic encoder is like
transcribing a symphony by reading the concert review.

### 5.2 The resolution arithmetic

📐 A ViT-L/14 at its native 224×224 produces 256 patch tokens. Our input is
128×128 — smaller than the encoder's native size, so it gets *upsampled* to fit,
producing 81 tokens of a 14× interpolated image, each summarising a region
**larger than the noise correlation length we spent all of Phase 0 measuring**.

The 4-tap cubic decimation kernel we recovered (`a ≈ −0.6`) has a support of
4 pixels. A 14×14 patch token averages over 196. The kernel is invisible at that
granularity — and reconstructing its inverse is the core of the task.

### 5.3 The preprocessor clips, and clipping is forbidden

This one is specific to *this* dataset and it is fatal on its own.

📏 From the Phase 0 forensics, verified over every pixel of the release:

```
   degraded input range        exceeds [0, 1]
     3.11 % of pixels          above 1.0   (observed max 1.54)
     0.28 % of pixels          below 0.0   (observed min −0.02)
```

Commandment 1 of this project exists because of that measurement: **never clip
the degraded input.** The out-of-range values are not corruption — they are
where the multiplicative noise ran hot, and they are the model's best local
estimate of noise strength. Delete them and you delete the σ signal.

Now: **every open VLM image pipeline clips.** The standard path is
`PIL.Image.open → uint8 → normalize → tensor`. It is `uint8` in the middle.
Values at 1.54 saturate to 255; values at −0.02 saturate to 0. You would be
silently discarding 3.4% of every input image, and specifically the 3.4% that
carries the most information about the degradation.

You could rewrite the preprocessor. But once you have replaced the tokenizer
(§4.3) and the preprocessor (here) and the output head, you are not fine-tuning a
VLM any more — you are using its transformer body as an expensive random
initialisation.

---

## 6. Reason 3 — autoregression multiplies runtime by the token count

Our model runs **one forward pass per image**. An autoregressive decoder runs
**one forward pass per output token**, in strict sequence, because token *n+1*
depends on token *n*. This is not a parallelism problem to be solved; it is the
definition of autoregression.

📐 The parameter-passes per image:

| | Passes | × params | Parameter-passes |
|---|---|---|---|
| **NAFNet-w48 (ours)** | 1 | 15.24 M | **1.5 × 10⁷** |
| Qwen2-VL-2B, 1024 tokens | 1,024 | 2.2 B | **2.3 × 10¹²** |

That is a factor of **~148,000**. Not 148,000× the wall clock — batching, KV
caching and memory-bandwidth limits claw a lot of it back — but that is the size
of the hole the engineering has to dig out of.

🔮 Wall-clock estimate, being deliberately generous to the alternative:

| Scenario | Throughput assumption | 400 images | vs our 9.0 s |
|---|---|---|---|
| Naïve, batch 1, consumer GPU | ~70 tok/s (bandwidth-bound) | **98 min** | **650× slower** |
| Optimistic — vLLM, continuous batching | ~2,000 tok/s aggregate | **3.4 min** | **23× slower** |
| **Ours** 📏 | — | **9.0 s** | 1× |

Both rows assume the tokenizer of §4.2 — so both rows are also capped at ~24 dB.
You would be paying 23× to 650× the time for *less* quality.

### 6.1 "But KLA benchmarks on an H100"

True, and it does not rescue this. Autoregressive decoding at batch 1 is
**memory-bandwidth bound**, not compute bound — each token requires streaming the
entire weight matrix from HBM. An H100's ~3.35 TB/s helps the decoder, but it
helps our single dense forward pass at least as much, and our pass was never
bandwidth-starved to begin with (15 M params is 30 MB in fp16; it lives in
cache).

Faster hardware scales both sides. It does not change a ratio of 148,000
parameter-passes into a ratio of 1.

### 6.2 The startup problem, which is worse than it looks

Remember 📏 1.4 s of our 9.0 s budget is startup. Now price the alternative's
startup: `import transformers` alone is 📚 typically 3–8 s, before you load
2.2 B parameters (4.4 GB in bf16) from disk and initialise a KV cache.

🔮 A realistic cold start for a 2B VLM is **15–40 s** — meaning the alternative
loses the entire race, by 2–4×, *before it looks at the first image.*

---

## 7. Reason 4 — there is no relevant pretrained prior to transfer

This is the conceptual heart of it, and the place where the intuition behind the
question goes wrong.

### 7.1 Fine-tuning *elicits* capability; it does not *install* modality

Fine-tuning works spectacularly when the base model already contains the
capability and you are teaching it **format, preference, or emphasis**. It works
poorly to hopelessly when you are teaching a **new input or output modality**.

Two worked examples, same model, same data budget, opposite outcomes:

> **Works.** Fine-tune a 7B LLM to write SQL from 3,000 examples. Excellent
> results. *Why:* pretraining already ingested millions of SQL statements, table
> schemas, and query semantics. The 3,000 examples teach it *your* dialect, your
> conventions, when to use a CTE. You are eliciting.

> **Fails.** Fine-tune the same 7B LLM to emit 65,536 calibrated floats
> describing a despeckled image, from 3,200 examples. *Why:* nothing in text
> pretraining encodes `Gamma(17.7, 1/17.7)` multi-look speckle statistics, or a
> 4-tap cubic decimation kernel with `a = −0.6`, or how multiplicative noise
> behaves in dark regions. You are installing a sensory modality. Different
> problem, different data scale by four to six orders of magnitude.

### 7.2 What the scale actually is, from teams who did it

📚 The data budgets required to teach a language model to handle images at all:

| System | What it learned | Data |
|---|---|---|
| LLaVA-1.5 | To **describe** images (read only, never emit) | 558K alignment pairs + 665K instruction pairs |
| Autoregressive image generators (Parti, Chameleon, Emu3) | To **emit** images as tokens | **Billions** of image–text pairs |
| **This project** 📏 | To emit calibrated restored images | **3,200 pairs** (800 sources × 4 crops) |

LLaVA needed 1.2 M pairs just to teach an LLM to *look*. Emitting is
categorically harder and costs three more orders of magnitude. We have 3,200.

We are not short by a factor that more epochs fix. We are short by a factor
of 10⁵.

### 7.3 And our 3,200 were enough — because the architecture matched

📏 The counterpoint that makes the argument concrete: NAFNet-w48 reached
26.415 dB from those same 3,200 pairs (plus DIV2K augmentation and procedural
patterns), in 40 epochs, on one 8 GB GPU, in about 1.6 hours.

A convolutional restoration network has the right inductive biases baked into
its *structure* — locality, translation equivariance, multi-scale processing.
It does not need to learn from data that neighbouring pixels are related; the
convolution asserts it. That is why 3,200 pairs suffice here and would not
suffice for a transformer with no such prior.

**The architecture is the prior.** Choosing NAFNet was choosing to get for free
what an LLM would have to learn from data we do not have.

---

## 8. Reason 5 — generative models optimise 1 of the 3 scored metrics, and lose the other 2 by theorem

### 8.1 The theorem

📚 Blau & Michaeli, *The Perception–Distortion Tradeoff*, CVPR 2018, proves
something stronger than an empirical observation: for **any** distortion measure
and **any** estimator, improving perceptual quality — how close the output
distribution is to the distribution of real images — beyond a certain point
**necessarily** increases distortion. There is a frontier, and you cannot be on
both ends of it.

- **MMSE / deterministic regressors** (ours) sit at the distortion-optimal end.
  They predict `E[x|y]` — the average over everything the degraded image could
  have come from.
- **Generative models** (GANs, diffusion, autoregressive LLMs) sit at the
  perception-optimal end by construction. They *sample* from `p(x|y)` — they
  commit to one plausible answer rather than averaging.

Now map that onto the scoring:

| Scored metric | Type | Which end wins |
|---|---|---|
| PSNR | Distortion | Ours |
| SSIM | Distortion | Ours |
| LPIPS | Perceptual | Generative |

**The scoring function is weighted 2:1 toward distortion.** A generative approach
optimises the minority metric and gives up the majority — not through bad
implementation, but as a mathematical consequence of what it is.

### 8.2 We have measured this tradeoff on this exact dataset

📏 The tradeoff is not theoretical here. From `experiments/baselines.json`:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| bicubic ×2 | 23.067 | 0.5129 | 0.4425 |
| BM3D + bicubic ×2 | **25.956** (+2.889 dB) | 0.6527 | **0.5576** (26% *worse*) |
| **NAFNet-w48 (ours)** | **26.415** | **0.7333** | **0.2455** |

BM3D buys +2.889 dB of PSNR and **loses 26% of its LPIPS relative to plain
bicubic interpolation** — it becomes perceptually worse than doing nothing
clever. That is the frontier, visible in our own numbers, on our own data.

📏 Our model beats both baselines on all three metrics simultaneously. That is
the point of a well-tuned deterministic restorer with a composite loss: you can
find a point that *dominates* the classical options. A generative model would
trade a chunk of that dominance for LPIPS it does not need — LPIPS is already
our strongest metric relative to the field (0.2455 against bicubic's 0.4425, a
45% improvement).

**We are winning the metric a generative model would buy, and we are winning it
without paying for it.**

### 8.3 The domain argument — the one that should matter most to KLA

This is the argument I would lead with in the room.

The problem statement forbids "artificial patterns or ringing". A generative
model does not merely risk violating that — **hallucinating plausible detail is
its objective function.** That is what sampling from `p(x|y)` means.

Consider what KLA builds: **semiconductor inspection and metrology equipment.**
A restored image is not the product; it is an *input to a defect-detection
decision.* So:

> A generative model looks at a noisy line edge, cannot resolve whether there is
> a notch, and — correctly, by its own objective — **invents the most plausible
> edge**. If it invents a notch that is not there, you have manufactured a false
> defect and scrapped a good die. If it smooths away a notch that *is* there,
> you have shipped a bad one.

A deterministic MMSE estimator handles the same ambiguity by producing a
**blurred, visibly uncertain** edge. That output is *honest*: it says "the
evidence does not resolve this." Downstream, blur is a signal to be cautious.
Hallucinated sharpness is a lie that looks like data.

📏 Our own known limitation is exactly this, and it is the correct failure to
have: the model over-smooths fine periodic patterns (checkerboard 15.29 dB vs
bicubic's 15.64 dB). We report it in `10_IMPROVEMENTS.md` rather than burying
it. It is a real weakness — and it errs on the side a metrology pipeline should
err on.

**In an inspection context, a model that blurs when uncertain is safe. A model
that invents when uncertain is dangerous.** That is not a metric preference; it
is a requirements decision.

---

## 9. Reason 6 — it does not fit on the hardware

📏 Current training footprint, from `../experiments/EXPERIMENT_LOG.md`:

```
   peak VRAM during training      2.1 GB  of 8 GB   (26% utilised)
   full training run              ~1.6 hours, 40 epochs
   parameters                     15.24 M
```

🔮 What fine-tuning a 2B VLM would need:

| Approach | Weights | + activations, grads, optimiser | Total | Fits in 8 GB? |
|---|---|---|---|---|
| Full fine-tune, bf16 | 4.4 GB | ~30 GB | ~35 GB | ❌ Not close |
| LoRA, bf16 base | 4.4 GB | ~6–11 GB | ~10–16 GB | ❌ No |
| QLoRA, 4-bit base | 1.2 GB | ~5–8 GB | ~6–9 GB | ⚠️ Marginal, batch size 1 |

QLoRA at batch size 1 is the only row that even approaches feasibility, and it
means: 4-bit quantised weights (adding quantisation error on top of the
tokenizer error from §4.2), batch size 1 (destroying throughput and destabilising
batch statistics), on a task needing 10⁵ more data than we have (§7.2).

📚 Model scale, for context:

| Model | Params | × ours |
|---|---|---|
| **SAFMN (our fast variant)** 📏 | 0.22 M | 0.015× |
| **NAFNet-w48 (ours, shipped)** 📏 | **15.24 M** | **1×** |
| Qwen2.5-0.5B | ~0.49 B | 32× |
| SmolLM2-1.7B | ~1.7 B | 112× |
| Qwen2-VL-2B | ~2.2 B | 144× |
| Phi-3.5-vision | ~4.2 B | 276× |
| LLaVA-1.5-7B | ~7.1 B | 466× |

Note the first row too. 📏 SAFMN at **0.22 M parameters** — 1/69th of our shipped
model — scores 26.064 dB, within 0.35 dB of NAFNet-w48. On this task, the
returns to scale are already flat at *fifteen million* parameters. The idea that
two billion would help is not supported by the local evidence: we measured the
curve and it had already levelled off.

---

## 10. The steelman — the version of this idea that is right

Everything above argues against **language** models. The underlying instinct —
*use pretrained open weights instead of training from scratch* — is correct, it
is explicitly permitted ("pre-trained weights ✅" in the project rules), and it
is where this document stops disagreeing with you.

### 10.1 Pretrained restoration backbones — and what we left on the table

📏 Verified by inspection of `configs/*.yaml` and `src/engine/trainer.py`: there
is no `pretrained` or `init_from` key anywhere. **We trained NAFNet-w48 from
random initialisation.** That is a genuine gap, and it is exactly the gap your
question points at.

The right version of "fine-tune open weights" for this task:

| Open weights | Pretrained on | Why it transfers here |
|---|---|---|
| **NAFNet (SIDD)** | Real camera sensor noise | Same architecture we ship — a drop-in initialisation |
| **NAFNet (GoPro)** | Motion deblurring | Same architecture, complementary low-level prior |
| **Restormer** | Denoising, deraining, deblurring | Transformer restoration backbone, multi-task low-level prior |
| **SwinIR** | Classical + real-world SR | The closest published task to ours |

📚 Typical gain from a matched restoration initialisation at a fixed epoch
budget: faster convergence and roughly **+0.1 to +0.3 dB**.

**And the inference cost is exactly zero.** Same architecture, same 15.24 M
parameters, same 4.6 ms. Only the starting point of the weights changes. It is
free quality under your time constraint — the single best remaining move in this
whole document, and we did not make it. That is on us, not on the reasoning.

### 10.2 Diffusion priors — right family, wrong budget

StableSR, DiffBIR and SUPIR are the genuinely strong open-weight image
restoration models, and on perceptual quality they would likely beat us.

They are still ruled out, for two independent reasons:

1. **Time.** 📚 Diffusion sampling needs 20–200 denoising steps, each a full
   forward pass through a ~1 B parameter UNet. 🔮 That is 2–15 s per image on a
   consumer GPU at this resolution — 400 images in **13 to 100 minutes**,
   against our 9.0 s. Same structural problem as §6: many sequential passes
   through a much larger network.
2. **§8.3.** They are generative. They hallucinate. In a metrology context that
   is a correctness failure, not a stylistic one.

This is already settled policy — Commandment 4 of `CLAUDE.md` (dev repo) reads "NO GAN, NO
diffusion", and §8 is the long form of why.

### 10.3 Distillation — how to actually get big-model quality at small-model speed

If what you want is *"the accuracy of a large pretrained model, at our current
execution time"* — that is a real, well-established technique, and it is neither
of the above:

```
   1. Train the biggest model that fits, offline. Slow. Nobody times it.
      Or ensemble several. Or use a diffusion prior. Time does not matter here.

   2. Run it over the training set to produce refined targets.

   3. Distil into the shipped 15.24 M NAFNet:
         L = α·L(student, ground_truth) + (1−α)·L(student, teacher_output)

   4. Ship the student. 📏 4.6 ms/image. Unchanged.
```

📚 Typical distillation gain in restoration: **+0.2 to +0.5 dB** at **zero**
inference cost. All the expense moves to training time, which is not scored.

**This is the correct answer to the question you actually asked.** You want big-model
benefit under a small-model time budget. Distillation is the mechanism designed
for precisely that trade — not fine-tuning the big model and shipping it.

---

## 11. Ranked improvement paths, by gain per millisecond

Every remaining option, ordered by what it buys against what it costs at
inference. 🔮 Gains are estimates; costs are 📐 derived from measured timings.

| Rank | Path | Est. gain | Inference cost | Verdict |
|---|---|---|---|---|
| **1** | Init from pretrained NAFNet-SIDD (§10.1) | +0.1–0.3 dB | **0 ms** | ✅ **Do it.** Free. |
| **2** | Distil from a large teacher (§10.3) | +0.2–0.5 dB | **0 ms** | ✅ **Do it.** Free. |
| **3** | Longer schedule + more external data | +0.2–0.4 dB | **0 ms** | ✅ Free, just slow to train |
| **4** | Sharper degradation replica (raise 0.982 overlap) | +0.1–0.3 dB | **0 ms** | ✅ Free |
| **5** | Frequency-weighted loss for the periodic-pattern gap | targeted | **0 ms** | ✅ Fixes a known weakness |
| 6 | ×8 self-ensemble (flips/rotations) | +0.2–0.4 dB | ×8 → 180 ms/img | ⚠️ Only if the budget allows |
| 7 | Ensemble w48 + SAFMN | +0.1–0.2 dB | ×2 → 45 ms/img | ⚠️ Poor ratio |
| 8 | Swap in a larger NAFNet (w64, w96) | +0.1–0.3 dB | ×2–4 | ⚠️ §9 says returns are flat |
| 9 | Diffusion prior (StableSR/DiffBIR) | LPIPS ↑, PSNR ↓ | **×100–1000** | ❌ §10.2 |
| 10 | Fine-tuned VLM | ceiling **below** current | **×23–650** | ❌ This entire document |

**Ranks 1 through 5 all cost zero milliseconds.** There is a meaningful amount of
free quality still available without touching the time budget. Nothing about the
LLM route becomes attractive until all five are exhausted — and after they are,
it is still ranked last.

---

## 12. How to prove this document wrong

I would rather be refuted with data than believed on rhetoric. The claims are
falsifiable, cheaply, in this order:

**Test 1 — the tokenizer ceiling (§4.2). Two hours. Kills or confirms the whole thing.**

Skip the LLM entirely. Just take a pretrained VQ tokenizer, encode and decode our
**ground-truth** images, and measure PSNR against the originals.

```bash
# round-trip GT through a VQ tokenizer, measure PSNR
# no training, no fine-tuning, no LLM — just the encode/decode
```

- If it scores **below 26.415 dB**, the ceiling is real and §4 is settled.
  No autoregressive pipeline can beat us. Stop here.
- If it scores **meaningfully above** it, §4.2 is wrong and the rest deserves
  reopening.

📚 I predict 20–24 dB. This is the cheapest decisive experiment available and it
requires no GPU training at all.

**Test 2 — the time floor (§6).** Load any 2B VLM, generate 1,024 tokens, time
it, multiply by 400. Compare to 9.0 s. No fine-tuning needed — if the *inference*
does not fit the budget, training it is moot.

**Test 3 — the free win (§10.1).** Download NAFNet-SIDD weights, initialise from
them, retrain with an otherwise identical config, compare on the same validation
split. This one is not a refutation attempt — it is the improvement, and it
should be run regardless of everything else in this file.

---

## 13. Verdict

**Our current solution is the right one for this problem, and the reasons are
structural rather than incidental.**

The task is **dense continuous regression under a hard latency budget, scored
2:1 toward distortion metrics, in a domain where inventing detail is a
correctness failure.** That description names a small deterministic convolutional
restoration network almost uniquely. It is what we built.

An LLM or VLM is the wrong tool on six independent axes, any *one* of which
would be disqualifying on its own:

1. Its output representation cannot carry the required precision (§4) — 📚 the
   tokenizer's ceiling of ~24 dB sits below our 📏 achieved 26.415 dB.
2. Its input encoder is trained to discard the signal we must estimate, and its
   preprocessor clips the 3.4% of pixels that carry the noise level (§5).
3. It needs 🔮 23–650× our runtime, under an explicit latency constraint (§6).
4. It has no relevant pretrained prior; we would need 📚 10⁵× more data (§7).
5. It optimises 1 of 3 scored metrics and loses the other 2 **by theorem** (§8).
6. It does not fit in 8 GB to train (§9).

**But the instinct was right and it identified a real gap.** Pretrained open
weights *are* permitted, *are* valuable, and we *are* leaving value on the table
by training from random init. The fix is **§10.1** — initialise from pretrained
NAFNet restoration weights — and **§10.3** — distil from a large offline teacher.

Both give you what you actually asked for: **better accuracy, at zero cost to
execution time.**

The disagreement was never "should we use pretrained open weights." It was
"*which* open weights." Not a language model's. A restoration model's.

---

## References

1. A. Blau, T. Michaeli. *The Perception-Distortion Tradeoff.* CVPR 2018. — §8.1, the theorem.
2. L. Chen et al. *Simple Baselines for Image Restoration.* ECCV 2022. — NAFNet; pretrained SIDD/GoPro weights, §10.1.
3. P. Esser, R. Rombach, B. Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR 2021. — VQGAN; the tokenizer of §4.2.
4. R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. — autoencoder reconstruction figures, §4.2.
5. H. Liu et al. *Improved Baselines with Visual Instruction Tuning.* CVPR 2024. — LLaVA-1.5 data scale, §7.2.
6. J. Liang et al. *SwinIR: Image Restoration Using Swin Transformer.* ICCVW 2021. — §10.1.
7. S. Zamir et al. *Restormer: Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022. — §10.1.
8. J. Wang et al. *Exploiting Diffusion Prior for Real-World Image Super-Resolution.* IJCV 2024. — StableSR, §10.2.
9. G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS-W 2015. — §10.3.
10. E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — §9.
11. V. Monga, Y. Li, Y. C. Eldar. *Algorithm Unrolling.* IEEE SPM 38(2), 2021. — cited by KLA; the basis of our log-unrolled variant.

---

*Measured figures trace to `experiments/pareto.json`, `experiments/baselines.json`,
`experiments/eval_results.json`, `experiments/throughput.json` and
`../experiments/EXPERIMENT_LOG.md`. Reproduce them with the commands in
`03_HOW_TO_RUN.md`.*
