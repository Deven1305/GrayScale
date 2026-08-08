"""Phase 4 — systematic loss ablation, one variable per run.

Purpose is attribution, not raw score: without this we cannot claim that any
individual loss term helped. That claim is exactly what the training-hygiene
axis rewards, and the table it produces goes on the innovation slide.

Runs are deliberately SHORT and on the smaller backbone — the comparison
between rows is what matters, and every row uses identical settings apart from
the one variable named in its title.

    python scripts/run_ablations.py                 # all
    python scripts/run_ablations.py --only 002 007  # a subset
    python scripts/run_ablations.py --epochs 12     # quicker
"""
import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# id -> (human label, loss overrides). Each differs from 002 by ONE term,
# except 007 which swaps the base term (the deliberate control).
ABLATIONS = {
    "002": ("Charbonnier only (base)",
            dict(charbonnier=1.0, msssim=0.0, fft=0.0, gradient=0.0, vgg=0.0, l2=0.0)),
    "003": ("+ 0.2 MS-SSIM",
            dict(charbonnier=1.0, msssim=0.2, fft=0.0, gradient=0.0, vgg=0.0, l2=0.0)),
    "004": ("+ 0.1 FFT",
            dict(charbonnier=1.0, msssim=0.2, fft=0.1, gradient=0.0, vgg=0.0, l2=0.0)),
    "005": ("+ 0.05 gradient (shipped w32 recipe)",
            dict(charbonnier=1.0, msssim=0.2, fft=0.1, gradient=0.05, vgg=0.0, l2=0.0)),
    "006": ("+ 0.01 VGG perceptual",
            dict(charbonnier=1.0, msssim=0.2, fft=0.1, gradient=0.05, vgg=0.01, l2=0.0)),
    "007": ("L2 instead of Charbonnier (control: should over-smooth)",
            dict(charbonnier=0.0, msssim=0.2, fft=0.1, gradient=0.05, vgg=0.0, l2=1.0)),
    "004b": ("FFT 0.1 -> 0.3 (defend high frequencies)",
             dict(charbonnier=1.0, msssim=0.2, fft=0.3, gradient=0.05, vgg=0.0, l2=0.0)),
    "005b": ("gradient 0.05 -> 0.15 (defend edges)",
             dict(charbonnier=1.0, msssim=0.2, fft=0.1, gradient=0.15, vgg=0.0, l2=0.0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/nafnet_w32.yaml")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/ablations.json")
    args = ap.parse_args()

    base = yaml.safe_load(open(ROOT / args.base, encoding="utf-8"))
    ids = args.only or list(ABLATIONS)
    results = {}
    outp = ROOT / args.out
    if outp.exists():
        results = json.load(open(outp))

    tmp_dir = ROOT / "configs/_ablation"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, aid in enumerate(ids, 1):
        if aid not in ABLATIONS:
            print(f"unknown ablation {aid}")
            continue
        if aid in results:
            print(f"[{aid}] already done, skipping")
            continue
        label, loss = ABLATIONS[aid]
        print(f"\n{'='*72}\n[{aid}] {label}   ({i}/{len(ids)})\n{'='*72}")

        cfg = deepcopy(base)
        cfg["loss"] = loss
        cfg["optim"]["epochs"] = args.epochs
        cfg["eval"]["val_every"] = max(2, args.epochs // 4)
        cfg["out_dir"] = f"experiments/runs/abl_{aid}"
        cfg["name"] = f"abl_{aid}"
        # ablations isolate the LOSS, so hold the data mix fixed at KLA-only
        cfg["data"]["external_ratio"] = 1.0
        cfg["data"]["pattern_frac"] = 0.0
        cfg["data"]["external_glob"] = None

        cpath = tmp_dir / f"{aid}.yaml"
        yaml.safe_dump(cfg, open(cpath, "w", encoding="utf-8"), sort_keys=False)

        r = subprocess.run([sys.executable, str(ROOT / "train.py"),
                            "--config", str(cpath)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            continue

        bm = ROOT / cfg["out_dir"] / "best_metrics.json"
        if bm.exists():
            m = json.load(open(bm))
            results[aid] = {"label": label, "loss": loss, "metrics": m,
                            "epochs": args.epochs}
            print(f"  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  "
                  f"LPIPS {m.get('lpips', float('nan')):.4f}")
            json.dump(results, open(outp, "w"), indent=2)

    # ---------------- report ---------------------------------------------
    if not results:
        return
    print(f"\n{'='*84}\nABLATION RESULTS ({args.epochs} epochs each, "
          f"identical apart from the loss)\n{'='*84}")
    print(f"{'id':>6} {'change':>46} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    base_m = results.get("002", {}).get("metrics")
    for aid in ABLATIONS:
        if aid not in results:
            continue
        m = results[aid]["metrics"]
        d = ""
        if base_m and aid != "002":
            d = f"  ({m['psnr']-base_m['psnr']:+.2f} dB)"
        print(f"{aid:>6} {results[aid]['label'][:46]:>46} {m['psnr']:>8.3f} "
              f"{m['ssim']:>8.4f} {m.get('lpips', float('nan')):>8.4f}{d}")
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
