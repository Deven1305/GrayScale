#!/usr/bin/env python
"""f26 — WHERE DOES EACH LOSS TERM'S GRADIENT ACTUALLY GO?

Motivation: the shipped v1 model removes noise well but leaves textured regions
visibly soft. The v1 recipe already contained two terms added specifically to
prevent that (`fft: 0.2`, `gradient: 0.1`), so before changing the architecture
it is worth checking whether those terms do what their names claim.

Method: take real GT images, construct the exact failure mode (a prediction
with 60% of its high-pass detail removed), backpropagate each loss term
individually, and measure what fraction of the resulting gradient energy lands
in the high-frequency half of the spectrum.

This is a diagnostic, not a training run. It takes seconds and needs no GPU.

    python scripts/f26_loss_spectrum.py

Writes experiments/loss_spectrum.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.composite import (  # noqa: E402
    CharbonnierLoss, FFTLoss, GradientLoss, HighFrequencyLoss, MSSSIMLoss,
)

# Split the spectrum at half of Nyquist. Everything at or above this is the
# band that 2x decimation destroyed and that makes an image read as "sharp".
HIGH_BAND_CUTOFF = 0.5


def high_band_mask(h, w, cutoff=HIGH_BAND_CUTOFF):
    fy = torch.fft.fftfreq(h).view(-1, 1)
    fx = torch.fft.rfftfreq(w).view(1, -1)
    r = torch.sqrt(fy * fy + fx * fx)
    return (r / r.max()) >= cutoff


def hf_gradient_share(loss_fn, pred_src, target, mask):
    """Fraction of this loss's input-gradient energy sitting in the high band."""
    p = pred_src.clone().requires_grad_(True)
    loss_fn(p, target).backward()
    g = torch.fft.rfft2(p.grad.float(), norm="ortho").abs()
    return (g[..., mask].sum() / g.sum().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default="data/train/train/GT")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--detail_removed", type=float, default=0.6,
                    help="fraction of high-pass detail stripped to synthesise "
                         "the over-smoothing failure mode")
    ap.add_argument("--out", default="experiments/loss_spectrum.json")
    args = ap.parse_args()

    files = sorted((ROOT / args.gt_dir).glob("*.npy"))[:args.limit]
    if not files:
        print(f"[error] no .npy under {args.gt_dir}", file=sys.stderr)
        return 1
    gt = torch.from_numpy(np.stack([np.load(f) for f in files]))[:, None].float()
    h, w = gt.shape[-2:]
    mask = high_band_mask(h, w)

    spec = torch.fft.rfft2(gt, norm="ortho").abs()
    energy_share = (spec[..., mask].sum() / spec.sum()).item()
    bin_share = mask.float().mean().item()

    # The failure mode, built explicitly: a blurred version of the truth.
    hp = HighFrequencyLoss(sigma=1.0)
    pred = gt - hp._highpass(gt) * args.detail_removed

    terms = {
        "charbonnier":        CharbonnierLoss(),
        "msssim":             MSSSIMLoss(),
        "fft_hf_power_0_v1":  FFTLoss(hf_power=0.0),
        "fft_hf_power_1":     FFTLoss(hf_power=1.0),
        "fft_hf_power_1.5":   FFTLoss(hf_power=1.5),
        "fft_hf_power_2":     FFTLoss(hf_power=2.0),
        "gradient_sobel":     GradientLoss(),
        "highfreq_sigma_1":   HighFrequencyLoss(sigma=1.0),
    }
    shares = {k: hf_gradient_share(v, pred, gt, mask) for k, v in terms.items()}

    print(f"{len(files)} GT images {h}x{w}   high band = |f| >= "
          f"{HIGH_BAND_CUTOFF} x f_max")
    print(f"  spectral ENERGY in high band : {energy_share:7.2%}")
    print(f"  spectral BINS   in high band : {bin_share:7.2%}")
    print(f"\nprediction = GT with {args.detail_removed:.0%} of its high-pass "
          f"detail removed\nshare of each term's gradient landing in the high "
          f"band:\n")
    base = shares["charbonnier"]
    for k, v in shares.items():
        flag = ""
        if k != "charbonnier":
            d = v - base
            flag = (f"   {d:+6.2%} vs charbonnier"
                    + ("  <-- BELOW the term it is meant to sharpen"
                       if d < 0 else ""))
        print(f"  {k:22s} {v:7.2%}{flag}")

    # ---- project whole RECIPES -------------------------------------------
    # A composite's emphasis is the weight-average of its terms' shares. This
    # lets a recipe be tuned in seconds instead of by training runs.
    def project(recipe):
        num = den = 0.0
        for key, (w, term) in recipe.items():
            if w <= 0:
                continue
            num += w * shares[term]
            den += w
        return num / max(den, 1e-12)

    recipes = {
        "v1 shipped (nafnet_w48.yaml)": {
            "charbonnier": (1.0, "charbonnier"), "msssim": (0.2, "msssim"),
            "fft": (0.2, "fft_hf_power_0_v1"), "gradient": (0.1, "gradient_sobel"),
        },
        "charbonnier alone (reference)": {
            "charbonnier": (1.0, "charbonnier"),
        },
        "v2 sharp (nafnet_w48_sharp.yaml)": {
            "charbonnier": (1.0, "charbonnier"), "msssim": (0.15, "msssim"),
            "fft": (0.6, "fft_hf_power_1.5"), "highfreq": (0.5, "highfreq_sigma_1"),
            "gradient": (0.05, "gradient_sobel"),
        },
    }
    projected = {k: project(v) for k, v in recipes.items()}
    print("\nprojected high-band emphasis of a whole recipe "
          "(weight-average of its terms):\n")
    ref = projected["charbonnier alone (reference)"]
    for k, v in projected.items():
        d = v - ref
        note = "" if abs(d) < 1e-9 else f"   {d:+6.2%} vs plain L1"
        print(f"  {k:36s} {v:7.2%}{note}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_images": len(files), "shape": [h, w],
        "high_band_cutoff": HIGH_BAND_CUTOFF,
        "detail_removed": args.detail_removed,
        "high_band_energy_share": energy_share,
        "high_band_bin_share": bin_share,
        "hf_gradient_share": shares,
        "recipe_projection": projected,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
