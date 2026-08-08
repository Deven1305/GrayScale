"""KLA degradation replica.

Reconstructed in Phase 0 from the paired data; see docs/forensics_report.md.

    y = D_a( x * n + g )        n ~ Gamma(L, 1/L),  g ~ N(0, sigma^2)

  * Gamma multi-look speckle, NOT Gaussian-multiplicative. Proven three ways:
    negatives are enriched 21x at dark pixels, 100% of samples have positive
    skew, and excess-kurtosis/(6*var) = 1.06.
  * The 2x decimation is a 4-tap cubic convolution, a ~ -0.6, applied LAST.
    Recovered by regressing E[y|x]; box/area/bilinear/Lanczos/strided all
    excluded.
  * The operation order is FIXED, not randomised. The brief says randomised;
    the data says otherwise (per-sample noise autocorrelation is unimodal,
    randomised controls are bimodal). Order randomisation is exposed as an
    explicit OOD augmentation flag, OFF by default.
  * Output is float32 and NEVER clipped.

The `decoupled` variant uses a separate kernel for the noise path, which is
the only form that reproduces the measured noise autocorrelation (-0.059).
Setting a2 = a1 recovers the single-kernel model.
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 4-tap cubic convolution at phase 0.5: k = [c, 0.5-c, 0.5-c, c], c = a/8.
# a = 0 is exactly the 2x2 box filter; a = -0.75 is cv2.INTER_CUBIC.
_K0 = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64)
_DD = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)


def cubic_taps(a: float) -> np.ndarray:
    return _K0 + (a / 8.0) * _DD


@dataclass
class DegradationConfig:
    """Measured ranges, widened ~50% for OOD robustness (Phase 0 §8.2)."""
    L_range: Tuple[float, float] = (8.0, 40.0)          # measured median 17.7
    sigma_log10_range: Tuple[float, float] = (-3.0, -1.4)  # 1e-3 .. 0.04
    a1_range: Tuple[float, float] = (-0.75, -0.45)      # signal kernel
    a2_range: Tuple[float, float] = (-0.40, -0.15)      # noise kernel
    decoupled: bool = True
    randomize_order: bool = False   # OFF: the real data uses a fixed order
    scale: int = 2


def _sep_down_torch(x: torch.Tensor, a: float, scale: int = 2) -> torch.Tensor:
    """Separable 4-tap cubic 2x decimation. x: (B,C,H,W)."""
    k = torch.as_tensor(cubic_taps(a), dtype=x.dtype, device=x.device)
    B, C, H, W = x.shape
    xr = F.pad(x, (1, 2, 1, 2), mode="reflect")
    kh = k.view(1, 1, 1, 4).expand(C, 1, 1, 4)
    xr = F.conv2d(xr, kh, groups=C)
    kv = k.view(1, 1, 4, 1).expand(C, 1, 4, 1)
    xr = F.conv2d(xr, kv, groups=C)
    return xr[:, :, ::scale, ::scale][:, :, :H // scale, :W // scale]


def degrade_torch(x: torch.Tensor, cfg: DegradationConfig,
                  gen: torch.Generator = None) -> torch.Tensor:
    """Apply the replica to a batch of clean HR images.

    x: (B, C, H, W) float32 in [0,1]. Returns (B, C, H/2, W/2), NOT clipped.
    Parameters are drawn per-sample.
    """
    B = x.shape[0]
    dev = x.device
    dt = x.dtype

    def u(lo, hi):
        return torch.rand(B, 1, 1, 1, generator=gen, device=dev, dtype=dt) \
            * (hi - lo) + lo

    L = u(*cfg.L_range)
    sigma = torch.pow(10.0, u(*cfg.sigma_log10_range))

    # Gamma(L, 1/L): mean 1, variance 1/L. torch.distributions handles the
    # per-sample concentration; rate = concentration gives mean 1.
    conc = L.expand_as(x)
    gamma = torch._standard_gamma(conc) / conc          # mean 1, var 1/L
    noise = torch.randn(x.shape, generator=gen, device=dev, dtype=dt) * sigma

    if cfg.randomize_order and bool(torch.rand(1, generator=gen,
                                               device=dev).item() < 0.5):
        # OOD augmentation only: decimate first, then apply noise at LR.
        a1 = float(u(*cfg.a1_range)[0, 0, 0, 0])
        z = _sep_down_torch(x, a1, cfg.scale)
        conc_s = L.expand_as(z)
        g2 = torch._standard_gamma(conc_s) / conc_s
        n2 = torch.randn(z.shape, generator=gen, device=dev, dtype=dt) * sigma
        return z * g2 + n2

    a1 = float(u(*cfg.a1_range)[0, 0, 0, 0])
    if not cfg.decoupled:
        return _sep_down_torch(x * gamma + noise, a1, cfg.scale)

    a2 = float(u(*cfg.a2_range)[0, 0, 0, 0])
    sig = _sep_down_torch(x, a1, cfg.scale)
    spk = _sep_down_torch(x * (gamma - 1.0), a2, cfg.scale)
    add = _sep_down_torch(noise, a2, cfg.scale)
    return sig + spk + add


def degrade_numpy(x: np.ndarray, cfg: DegradationConfig,
                  rng: np.random.Generator) -> np.ndarray:
    """Single-image numpy path, used by tests and dataset workers."""
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None]
    g = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    return degrade_torch(t, cfg, g)[0, 0].numpy()
