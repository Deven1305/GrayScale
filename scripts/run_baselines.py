"""Baselines — the floor. Any model that does not beat bicubic on all three
scored metrics has a bug, not a design problem.

  * bicubic x2          — the trivial upsampler
  * BM3D then bicubic   — classical denoise + upsample

Writes experiments/baselines.json and a row for EXPERIMENT_LOG.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import split_by_source           # noqa: E402
from src.metrics.full_reference import evaluate_batch  # noqa: E402

GT = ROOT / "data/train/train/GT"
LR = ROOT / "data/train/train/NoisyLR"


def bicubic(lr, scale=2):
    import torch.nn.functional as F
    t = torch.from_numpy(lr)[None, None]
    return F.interpolate(t, scale_factor=scale, mode="bicubic",
                         align_corners=False)[0, 0].numpy()


def bm3d_then_bicubic(lr, sigma=0.24):
    import bm3d
    # speckle std ~0.24 at the measured median L; BM3D assumes additive
    # Gaussian, so this is a deliberately imperfect but standard baseline.
    den = bm3d.bm3d(np.clip(lr, 0, 1), sigma_psd=sigma)
    return bicubic(den.astype(np.float32))


def main(limit=160, with_bm3d=True):
    _, val_idx = split_by_source(val_frac=0.10, seed=1337)
    val_idx = val_idx[:limit]
    print(f"[baselines] {len(val_idx)} val images (held out by source)")

    results = {}
    for name, fn in (("bicubic_x2", bicubic),
                     ("bm3d+bicubic_x2", bm3d_then_bicubic if with_bm3d else None)):
        if fn is None:
            continue
        agg, n, t0 = {}, 0, time.perf_counter()
        for k, i in enumerate(val_idx):
            gt = np.load(GT / f"{i:06d}.npy").astype(np.float32)
            lr = np.load(LR / f"{i:06d}.npy").astype(np.float32)
            try:
                pred = fn(lr)
            except Exception as e:
                print(f"  {name} failed on {i}: {e}")
                continue
            m = evaluate_batch(torch.from_numpy(pred)[None, None],
                               torch.from_numpy(gt)[None, None])
            n += 1
            for kk, vv in m.items():
                agg[kk] = agg.get(kk, 0.0) + vv
            if (k + 1) % 40 == 0:
                print(f"  {name}: {k+1}/{len(val_idx)}", flush=True)
        res = {kk: vv / max(n, 1) for kk, vv in agg.items()}
        res["seconds_total"] = time.perf_counter() - t0
        res["n_images"] = n
        results[name] = res
        print(f"[{name}] " + "  ".join(f"{kk}={vv:.4f}" for kk, vv in res.items()))

    out = ROOT / "experiments/baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("\n--- EXPERIMENT_LOG rows ---")
    for k, v in results.items():
        print(f"| 000 | baseline | {k} | — | {v['psnr']:.2f} | {v['ssim']:.4f} "
              f"| {v.get('lpips', float('nan')):.4f} | floor |")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--no-bm3d", action="store_true")
    a = ap.parse_args()
    main(a.limit, not a.no_bm3d)
