"""Full-reference metrics: PSNR, SSIM, MS-SSIM, LPIPS.

⚠️ data_range=1.0 EVERYWHERE. Images are [0,1] floats; the torchmetrics
default is not what we want, and a wrong data_range silently invalidates every
number produced downstream. tests/test_metrics.py asserts PSNR(x,x) == inf,
which is the cheapest possible tripwire for this class of bug.

LPIPS needs 3-channel input in [-1,1]: we replicate grayscale x3 and rescale.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    multiscale_structural_similarity_index_measure as _msssim,
)
from torchmetrics.functional import peak_signal_noise_ratio as _psnr
from torchmetrics.functional import structural_similarity_index_measure as _ssim

DATA_RANGE = 1.0
_LPIPS = None


def _lpips_model(device):
    global _LPIPS
    if _LPIPS is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _LPIPS = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=False).to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    return _LPIPS


def psnr(pred, target):
    return _psnr(pred, target, data_range=DATA_RANGE)


def ssim(pred, target):
    return _ssim(pred, target, data_range=DATA_RANGE)


def msssim(pred, target):
    # MS-SSIM needs >= 161px for 5 scales; fall back to SSIM when too small.
    if min(pred.shape[-2:]) < 161:
        return _ssim(pred, target, data_range=DATA_RANGE)
    return _msssim(pred, target, data_range=DATA_RANGE)


@torch.no_grad()
def lpips(pred, target):
    """Grayscale [0,1] -> 3-channel [-1,1], as LPIPS expects."""
    m = _lpips_model(pred.device)
    p = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    t = target.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    return m(p.float(), t.float())


@torch.no_grad()
def evaluate_batch(pred, target, with_lpips: bool = True) -> Dict[str, float]:
    """pred/target: (B,1,H,W) float in roughly [0,1]. Returns plain floats.

    NOTE: pred is clamped to [0,1] for METRIC purposes only, because GT is
    guaranteed to lie in [0,1] and a model overshoot should be scored as an
    error, not rewarded. The model's raw output is never clamped in training.
    """
    p = pred.clamp(0, 1).float()
    t = target.float()
    out = {
        "psnr": float(psnr(p, t)),
        "ssim": float(ssim(p, t)),
        "msssim": float(msssim(p, t)),
        "l1": float(F.l1_loss(p, t)),
    }
    if with_lpips:
        out["lpips"] = float(lpips(p, t))
    return out
