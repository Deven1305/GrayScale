# LOSS ABLATION RESULTS

One variable per run. Every row is identical apart from the loss term
named in its title: same backbone, same seed, same data mix (KLA pairs
only, so the loss is isolated from the content change), same epoch
budget, same source-disjoint validation split.

Purpose is **attribution**, not peak score. Without this table we could
not claim that any individual loss term helped.

| ID | Loss change | PSNR ↑ | SSIM ↑ | LPIPS ↓ | vs base |
|---|---|---|---|---|---|
| **002** | Charbonnier only (base) | 25.643 | 0.6520 | 0.3424 | — |
| **003** | + 0.2 MS-SSIM | 25.332 | 0.6440 | 0.3563 | -0.312 dB / -0.0080 SSIM |
| **004** | + 0.1 FFT | 25.341 | 0.6446 | 0.3562 | -0.302 dB / -0.0074 SSIM |
| **005** | + 0.05 gradient (shipped w32 recipe) | 25.396 | 0.6489 | 0.3544 | -0.247 dB / -0.0031 SSIM |
| **006** | + 0.01 VGG perceptual | 25.413 | 0.6483 | 0.3508 | -0.230 dB / -0.0037 SSIM |
| **007** | L2 instead of Charbonnier (control: should over-smooth) | 25.020 | 0.6485 | 0.3559 | -0.623 dB / -0.0035 SSIM |
| **004b** | FFT 0.1 -> 0.3 (defend high frequencies) | 25.446 | 0.6528 | 0.3520 | -0.197 dB / +0.0009 SSIM |
| **005b** | gradient 0.05 -> 0.15 (defend edges) | 25.489 | 0.6573 | 0.3471 | -0.154 dB / +0.0053 SSIM |

*12 epochs per run — short by design; the comparison between rows is what matters, not the absolute values.*

## What the table says

* **Best SSIM:** `005b` — gradient 0.05 -> 0.15 (defend edges) (0.6573)
* **Best LPIPS:** `002` — Charbonnier only (base) (0.3424)

* **Control (007, L2 instead of Charbonnier):** PSNR 25.020, SSIM 0.6485, LPIPS 0.3559. 
  Its LPIPS is worse than the Charbonnier base, which is the over-smoothing the spec warns against, reproduced deliberately.
