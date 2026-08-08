"""Offline evaluation against ground truth.

    python evaluate.py --ckpt experiments/runs/nafnet_w32/best.pt
    python evaluate.py --ckpt ... --ood data/external/Urban100

Reports the three scored metrics (PSNR, SSIM, LPIPS) plus MS-SSIM and L1, on
the in-distribution val split and on any synthetic-OOD family supplied. OOD
families get ground truth because WE degrade them with the Phase 0 replica —
which is the whole point: KLA withheld test GT, so we manufacture our own.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import KLAPairs, SyntheticPairs
from src.data.degradation import DegradationConfig
from src.data.splits import split_by_source
from src.metrics.full_reference import evaluate_batch
from src.models.registry import build_model, count_params


def load_model(ckpt_path, device="cuda", use_ema=True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = build_model(cfg["model"]["arch"], in_ch=cfg["model"]["in_ch"],
                    scale=cfg["model"]["scale"],
                    use_log_channel=cfg["model"]["use_log_channel"])
    sd = ck.get("ema_state_dict") if use_ema else None
    m.load_state_dict(sd or ck["model_state_dict"])
    return m.to(device).eval(), cfg, ck


@torch.no_grad()
def run(model, loader, device, tag, with_lpips=True):
    agg, n = {}, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            pred = model(lr)
        m = evaluate_batch(pred.float(), hr.float(), with_lpips=with_lpips)
        b = lr.size(0)
        n += b
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v * b
    out = {k: v / max(n, 1) for k, v in agg.items()}
    out["n_images"] = n
    print(f"[{tag}]  PSNR {out['psnr']:.3f}  SSIM {out['ssim']:.4f}  "
          f"LPIPS {out.get('lpips', float('nan')):.4f}  (n={n})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ood", nargs="*", default=[],
                    help="directories of clean images -> synthetic OOD families")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/eval_results.json")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--proxy-ood", action="store_true",
                    help="also evaluate on the held-out tonal extremes")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_model(args.ckpt, dev, use_ema=not args.no_ema)
    print(f"[model] {cfg['model']['arch']}  params={count_params(model)/1e6:.2f}M"
          f"  epoch={ck.get('epoch')}  git={str(ck.get('git_commit'))[:8]}"
          f"  weights={'EMA' if not args.no_ema else 'raw'}")

    results = {"checkpoint": str(args.ckpt),
               "arch": cfg["model"]["arch"],
               "params_M": count_params(model) / 1e6,
               "epoch": ck.get("epoch"),
               "git_commit": ck.get("git_commit")}

    # ---- in-distribution val (held out by source) ----------------------
    _, val_idx = split_by_source(val_frac=cfg["data"]["val_frac"],
                                 seed=cfg["seed"])
    if args.limit:
        val_idx = val_idx[:args.limit]
    ds = KLAPairs(Path(cfg["data"]["root"]), val_idx, augment=False, full=True)
    ld = DataLoader(ds, batch_size=8, num_workers=2, pin_memory=True)
    results["val_in_distribution"] = run(model, ld, dev, "val/in-dist")

    # ---- proxy-OOD: tonal extremes -------------------------------------
    # Phase 0 §9 found the corpus has NO visual-origin families, so a
    # leave-one-origin-out split is impossible. Holding out the darkest and
    # brightest source clusters is the honest substitute, and it needs no
    # external download. Genuine content-OOD still requires Urban100 etc.
    cj = Path("docs/_forensics_cache/origin_clusters.json")
    if args.proxy_ood and cj.exists():
        from src.data.splits import split_tonal_extremes
        try:
            _, ood_idx = split_tonal_extremes(cj)
            if args.limit:
                ood_idx = ood_idx[:args.limit]
            ods = KLAPairs(Path(cfg["data"]["root"]), ood_idx, augment=False,
                           full=True)
            old = DataLoader(ods, batch_size=8, num_workers=2, pin_memory=True)
            results["proxy_ood_tonal_extremes"] = run(
                model, old, dev, "ood/tonal-extremes")
        except Exception as e:
            print(f"[warn] proxy-OOD skipped: {e}")

    # ---- synthetic OOD families ----------------------------------------
    dcfg = DegradationConfig(**cfg["degradation"])
    for d in args.ood:
        p = Path(d)
        files = sorted([f for f in p.rglob("*")
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg",
                                                ".bmp", ".npy")])
        if not files:
            print(f"[warn] no images in {d}")
            continue
        if args.limit:
            files = files[:args.limit]
        ods = SyntheticPairs(files, dcfg, hr_patch=256, augment=False,
                             fixed=True, seed=1234)
        # OOD benchmark images have heterogeneous sizes (e.g. 322x512 next to
        # 384x512), so they cannot be stacked into a batch. Evaluate one at a
        # time rather than cropping, which would change what is measured.
        old = DataLoader(ods, batch_size=1, num_workers=2)
        results[f"ood/{p.name}"] = run(model, old, dev, f"ood/{p.name}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    # baseline comparison, if present
    bl = Path("experiments/baselines.json")
    if bl.exists():
        b = json.load(open(bl))
        v = results["val_in_distribution"]
        print("\n--- vs baselines (val, in-distribution) ---")
        print(f"{'method':>22} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        for k, m in b.items():
            print(f"{k:>22} {m['psnr']:>8.3f} {m['ssim']:>8.4f} "
                  f"{m.get('lpips', float('nan')):>8.4f}")
        print(f"{cfg['model']['arch']:>22} {v['psnr']:>8.3f} {v['ssim']:>8.4f} "
              f"{v.get('lpips', float('nan')):>8.4f}")
        beat = all(v["psnr"] > m["psnr"] and v["ssim"] > m["ssim"]
                   for m in b.values())
        print(f"\nbeats every baseline on PSNR and SSIM: {beat}")


if __name__ == "__main__":
    main()
