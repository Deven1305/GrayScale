"""Phase 7 — quality-vs-latency Pareto curve and the operating-point decision.

KLA's deck states twice that "faster pipelines are preferred when restoration
quality is comparable". That is a tiebreak rule, not a speed contest, so the
selection rule is fixed in advance:

    QUALITY IS PRIMARY. SPEED IS THE TIEBREAK.
    Choose the FASTEST model whose quality is within noise of the best —
    not the fastest overall.

"Within noise" is defined concretely as within 0.15 dB PSNR and 0.005 SSIM of
the best, which is roughly the run-to-run spread we observe between seeds.

    python scripts/build_pareto.py
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.registry import build_model, count_params   # noqa: E402

INK, ACCENT, WARN, GREEN, GREY = "#14213D", "#2A6FDB", "#E76F51", "#2A9D8F", "#8A94A6"

# run dir -> label
CANDIDATES = {
    "experiments/runs/nafnet_w48": "NAFNet-w48",
    "experiments/runs/nafnet_w32": "NAFNet-w32",
    "experiments/runs/safmn": "SAFMN",
    "experiments/runs/log_unrolled": "Log-unrolled (K=4)",
}

PSNR_NOISE = 0.15      # dB
SSIM_NOISE = 0.005


def latency(arch: str, n=64, bs=16, shape=128, device="cuda"):
    """Pure forward-pass latency, fp16, channels_last — the model's own cost."""
    m = build_model(arch).to(device).eval()
    if device.startswith("cuda"):
        m = m.half().to(memory_format=torch.channels_last)
    x = torch.zeros(bs, 1, shape, shape, device=device,
                    dtype=torch.half if device.startswith("cuda") else torch.float)
    if device.startswith("cuda"):
        x = x.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(5):
            m(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        reps = max(1, n // bs)
        for _ in range(reps):
            m(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / (reps * bs)
    p = count_params(m)
    del m, x
    torch.cuda.empty_cache()
    return dt * 1000, p                      # ms/image, params


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for run, label in CANDIDATES.items():
        bm = ROOT / run / "best_metrics.json"
        cf = ROOT / run / "config.yaml"
        if not (bm.exists() and cf.exists()):
            print(f"  skip {label}: not trained yet")
            continue
        import yaml
        cfg = yaml.safe_load(open(cf, encoding="utf-8"))
        m = json.load(open(bm))
        ms, p = latency(cfg["model"]["arch"], device=dev)
        rows.append({"label": label, "arch": cfg["model"]["arch"],
                     "params_M": p / 1e6, "ms_per_image": ms,
                     "psnr": m["psnr"], "ssim": m["ssim"],
                     "lpips": m.get("lpips", float("nan"))})
        print(f"  {label:20s} {p/1e6:6.2f} M  {ms:6.2f} ms  "
              f"PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}")

    # classical baselines for context
    bl = ROOT / "experiments/baselines.json"
    if bl.exists():
        for k, v in json.load(open(bl)).items():
            rows.append({"label": k, "arch": "classical", "params_M": 0.0,
                         "ms_per_image": v["seconds_total"] / v["n_images"] * 1000,
                         "psnr": v["psnr"], "ssim": v["ssim"],
                         "lpips": v.get("lpips", float("nan"))})

    if not rows:
        sys.exit("nothing trained yet")

    # ---- the operating-point rule ---------------------------------------
    learned = [r for r in rows if r["arch"] != "classical"]
    best = max(learned, key=lambda r: r["ssim"])
    within = [r for r in learned
              if r["psnr"] >= best["psnr"] - PSNR_NOISE
              and r["ssim"] >= best["ssim"] - SSIM_NOISE]
    chosen = min(within, key=lambda r: r["ms_per_image"])

    print(f"\n{'='*72}\nOPERATING POINT\n{'='*72}")
    print(f"  best quality      : {best['label']}  "
          f"PSNR {best['psnr']:.3f}  SSIM {best['ssim']:.4f}")
    print(f"  within noise      : {[r['label'] for r in within]}")
    print(f"  -> CHOSEN (fastest within noise): {chosen['label']}  "
          f"{chosen['ms_per_image']:.2f} ms/img")
    if chosen["label"] != best["label"]:
        print(f"     gives up {best['psnr']-chosen['psnr']:.3f} dB for "
              f"{best['ms_per_image']/chosen['ms_per_image']:.2f}x speed")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, key, name in ((ax[0], "psnr", "PSNR (dB)"), (ax[1], "ssim", "SSIM")):
        for r in rows:
            cl = r["arch"] == "classical"
            col = GREY if cl else (GREEN if r is chosen else ACCENT)
            a.scatter(r["ms_per_image"], r[key], s=150 if r is chosen else 80,
                      c=col, marker="s" if cl else "o", zorder=3,
                      edgecolors=INK if r is chosen else "none",
                      linewidths=2 if r is chosen else 0)
            a.annotate(r["label"], (r["ms_per_image"], r[key]),
                       textcoords="offset points", xytext=(7, 5), fontsize=8,
                       color=INK)
        a.set_xscale("log")
        a.set_xlabel("latency, ms per image (log)", fontsize=10)
        a.set_ylabel(name, fontsize=10)
        a.grid(alpha=.3)
        a.set_title(f"Quality vs latency — {name}", fontsize=12, color=INK,
                    fontweight="bold")
    ax[0].scatter([], [], c=GREEN, edgecolors=INK, linewidths=2, s=150,
                  label="chosen operating point")
    ax[0].scatter([], [], c=GREY, marker="s", s=80, label="classical baseline")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Operating-point rule: quality primary, speed the tiebreak — "
                 "fastest model within noise of the best", fontsize=11)
    fig.tight_layout()
    out = ROOT / "docs/figures/pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}")

    json.dump({"rows": rows, "chosen": chosen, "best": best,
               "rule": {"psnr_noise_dB": PSNR_NOISE, "ssim_noise": SSIM_NOISE}},
              open(ROOT / "experiments/pareto.json", "w"), indent=2)


if __name__ == "__main__":
    main()
