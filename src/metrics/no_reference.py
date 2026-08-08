"""No-reference IQA — the only quality signal available on the real test set,
where KLA withholds the ground truth.

⚠️ Use COMPARATIVELY, never as an optimisation target. NR-IQA correlates
imperfectly with PSNR/SSIM and frequently REWARDS hallucinated texture, which
is exactly what the problem statement forbids. The right use is: compare our
output's NIQE against the bicubic baseline's NIQE on the same images, and look
for gross failure.
"""
from typing import Dict, List

import torch

_CACHE = {}


def _metric(name: str, device):
    key = (name, str(device))
    if key not in _CACHE:
        import pyiqa
        _CACHE[key] = pyiqa.create_metric(name, device=device)
    return _CACHE[key]


@torch.no_grad()
def score(images: torch.Tensor, metrics: List[str] = ("niqe", "brisque"),
          device=None) -> Dict[str, float]:
    """images: (B,1,H,W) in [0,1]. Returns the mean of each metric."""
    device = device or images.device
    x = images.clamp(0, 1).repeat(1, 3, 1, 1).float().to(device)
    out = {}
    for m in metrics:
        try:
            out[m] = float(_metric(m, device)(x).mean())
        except Exception as e:                      # keep going; NR-IQA is advisory
            out[m] = float("nan")
            out[f"{m}_error"] = str(e)[:80]
    return out
