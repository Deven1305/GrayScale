# OUT-OF-DISTRIBUTION REPORT

KLA withholds the test ground truth, so generalisation cannot be
measured directly on the scored set. Instead we **manufacture** ground
truth: because Phase 0 reconstructed the degradation to 0.982 histogram
overlap, any clean image can be turned into a labelled pair.

That is what makes every number below possible.

## Per-family results

| Family | Content | PSNR ↑ | SSIM ↑ | LPIPS ↓ | n |
|---|---|---|---|---|---|
| KLA held-out val | photographs (in-distribution) | 26.415 | 0.7333 | 0.2455 | 320 |
| Tonal extremes | darkest + brightest KLA sources | 28.376 | 0.7868 | 0.2075 | 824 |
| Urban100 | **buildings / cityscapes** — the OOD case KLA named | 24.432 | 0.7750 | 0.2204 | 100 |
| BSD100 | natural scenes | 26.422 | 0.7506 | 0.2494 | 100 |
| Set14 | classic SR benchmark | 26.655 | 0.7643 | 0.2110 | 14 |

## Against the in-distribution baselines

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| bicubic_x2 | 23.067 | 0.5129 | 0.4425 |
| bm3d+bicubic_x2 | 25.956 | 0.6527 | 0.5576 |
| **ours** | **26.415** | **0.7333** | **0.2455** |

## Honest reading

* **Tonal extremes scores +1.96 dB vs in-distribution — i.e. it is EASIER, not harder.** It is therefore weak evidence of generalisation and we do not present it as an OOD win.
* **Urban100: -1.98 dB vs in-distribution.** A genuine drop, which is what an OOD family should show.
* **BSD100 scores +0.01 dB vs in-distribution — i.e. it is EASIER, not harder.** It is therefore weak evidence of generalisation and we do not present it as an OOD win.
* **Set14 scores +0.24 dB vs in-distribution — i.e. it is EASIER, not harder.** It is therefore weak evidence of generalisation and we do not present it as an OOD win.

## Degradation consistency on the REAL test set (no GT)

```
x̂ = model(y);  ŷ = degradation(x̂);  error = ‖ŷ − y‖
```

| | RMSE |
|---|---|
| median | 0.0956 |
| p95 | 0.1539 |
| max | 0.1984 |

⚠️ Necessary, not sufficient: an over-smoothed output can still
re-degrade convincingly, because the degradation destroys high
frequencies anyway. Use it to catch gross failure, not to rank
good models.


## Where the model still degrades

See `07_ANALYSIS.md` §5 for the measured failure modes:
over-smoothing of high-frequency texture, and fine periodic
structure where it can score below bicubic.

