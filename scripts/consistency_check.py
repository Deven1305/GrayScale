"""Degradation-consistency check — quality signal on the REAL test set, where
there is no ground truth at all.

    x_hat = model(y)                 restore
    y_hat = degradation(x_hat)       re-degrade with the Phase 0 replica
    err   = ||y_hat - y||            should be small

If a restoration, when re-degraded, does not reproduce the observed input,
something is wrong with that image. This is the only per-image diagnostic
available on the scored test set, and it flags the worst cases for inspection.

    python scripts/consistency_check.py --ckpt experiments/runs/nafnet_w32/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.degradation import DegradationConfig, degrade_torch   # noqa: E402
from evaluate import load_model                                     # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_dir", default="data/Test_NoisyLR/NoisyLR")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_draws", type=int, default=4,
                    help="noise realisations to average over")
    ap.add_argument("--out", default="experiments/consistency.json")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model(args.ckpt, dev)
    dcfg = DegradationConfig(**cfg["degradation"])

    files = sorted(Path(args.input_dir).glob("*.npy"))
    print(f"[consistency] {len(files)} images from {args.input_dir}")

    rows = []
    for k, p in enumerate(files):
        y = np.load(p).astype(np.float32)
        yt = torch.from_numpy(y)[None, None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=dev.startswith("cuda")):
            xh = model(yt).float().clamp(0, 1)
        errs = []
        for d in range(args.n_draws):
            g = torch.Generator(device=dev).manual_seed(1000 + d)
            yh = degrade_torch(xh, dcfg, g)
            errs.append(float(torch.sqrt(((yh - yt) ** 2).mean())))
        rows.append({"file": p.name, "rmse": float(np.mean(errs)),
                     "input_std": float(y.std())})
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(files)}", flush=True)

    e = np.array([r["rmse"] for r in rows])
    summary = {
        "n": len(rows),
        "mean_rmse": float(e.mean()),
        "median_rmse": float(np.median(e)),
        "p95_rmse": float(np.percentile(e, 95)),
        "max_rmse": float(e.max()),
    }
    worst = sorted(rows, key=lambda r: -r["rmse"])[:15]
    print("\n--- consistency (lower is better) ---")
    for k, v in summary.items():
        print(f"  {k:>12}: {v}")
    print("\n  worst 15 images (inspect these):")
    for r in worst:
        print(f"    {r['file']}  rmse={r['rmse']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "worst": worst, "all": rows},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
