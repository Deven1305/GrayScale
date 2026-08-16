#!/usr/bin/env python
"""r02 — INPUT vs MODEL OUTPUT vs GROUND TRUTH, side by side.

The three-way comparison is only possible on the TRAINING release, because the
released test set ships no ground truth. That has a consequence this script
refuses to hide: most training indices were actually trained on, so their
reconstructions are NOT evidence of generalisation. Every row is therefore
labelled TRAIN or HELD-OUT, decided by the real split
(src/data/splits.py, source_id = index // 4, seed 1337).

Output is produced by running the shipped inference.py as a subprocess, so the
figure shows exactly what the scored pipeline produces -- not a reimplementation
of it.

    python scripts/r02_gt_triptych.py
    python scripts/r02_gt_triptych.py --ids 000044 000991 --out docs/figures/x.png
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The four examples, with what they actually show.
DEFAULT = [("000044", "bird"), ("000991", "fence"),
           ("000757", "wood grain"), ("002014", "market / onions")]

INK = "#14213D"
ORANGE = "#E76F51"
TEAL = "#2A9D8F"
GREY = "#6C757D"


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def run_inference(ids, lr_dir, weights=None):
    """Run the real inference.py over just these ids. Returns {id: array}."""
    tmp = Path(tempfile.mkdtemp(prefix="triptych_"))
    try:
        src, dst = tmp / "in", tmp / "out"
        src.mkdir()
        for i in ids:
            shutil.copy2(lr_dir / f"{i}.npy", src / f"{i}.npy")
        cmd = [sys.executable, str(ROOT / "inference.py"),
               "--input_dir", str(src), "--output_dir", str(dst)]
        if weights:
            cmd += ["--weights", str(weights)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(r.stdout + r.stderr, file=sys.stderr)
            raise SystemExit("inference.py failed")
        return {i: np.load(dst / f"{i}.npy") for i in ids}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--gt_dir", default="data/train/train/GT")
    ap.add_argument("--lr_dir", default="data/train/train/NoisyLR")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--label", default=None,
                    help="which model produced this, e.g. 'v2 SHARP'. Shown in "
                         "the title so two figures can never be confused.")
    ap.add_argument("--out", default="docs/figures/gt_triptych.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.metrics import structural_similarity as ssim_fn
    from src.data.splits import source_id, split_by_source

    rows = ([(i, "") for i in args.ids] if args.ids else DEFAULT)
    gt_dir, lr_dir = ROOT / args.gt_dir, ROOT / args.lr_dir
    missing = [i for i, _ in rows
               if not (gt_dir / f"{i}.npy").exists()
               or not (lr_dir / f"{i}.npy").exists()]
    if missing:
        raise SystemExit(f"missing .npy for: {missing}")

    _, val = split_by_source(3200, 0.1, 1337)
    val = set(val)

    print(f"running inference.py on {len(rows)} images...")
    outs = run_inference([i for i, _ in rows], lr_dir, args.weights)

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    axes = np.atleast_2d(axes)
    title = "Degraded input  →  model output  →  ground truth"
    if args.label:
        title += f"\nmodel: {args.label}"
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, y=0.997)

    # the model name lives in the suptitle only -- repeating it in the column
    # header makes the three headers collide
    heads = [("DEGRADED INPUT", ORANGE), ("MODEL OUTPUT", TEAL),
             ("GROUND TRUTH", INK)]

    for r, (idx, label) in enumerate(rows):
        lo = np.load(lr_dir / f"{idx}.npy").astype(np.float32)
        hi = outs[idx].astype(np.float32)
        gt = np.load(gt_dir / f"{idx}.npy").astype(np.float32)

        held = int(idx) in val
        p = psnr(np.clip(hi, 0, 1), gt)
        s = ssim_fn(np.clip(hi, 0, 1), gt, data_range=1.0)

        subs = [
            f"{lo.shape[0]}x{lo.shape[1]}   range [{lo.min():.2f}, {lo.max():.2f}]",
            f"{hi.shape[0]}x{hi.shape[1]}   PSNR {p:.2f} dB   SSIM {s:.4f}",
            f"{gt.shape[0]}x{gt.shape[1]}   range [{gt.min():.2f}, {gt.max():.2f}]",
        ]
        for c, (img, (head, col), sub) in enumerate(zip([lo, hi, gt], heads, subs)):
            ax = axes[r, c]
            # nearest everywhere: matplotlib must not smooth the comparison,
            # and the input must visibly be half the resolution
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2)
            if r == 0:
                ax.set_title(head, fontsize=12, fontweight="bold", color=col,
                             pad=10)
            ax.set_xlabel(sub, fontsize=8.5, color=GREY, labelpad=5)

        tag = "HELD OUT" if held else "IN TRAINING SET"
        tag_col = TEAL if held else ORANGE
        name = f"{idx}.npy" + (f"\n{label}" if label else "")
        axes[r, 0].set_ylabel(name, fontsize=11, fontweight="bold", color=INK,
                              labelpad=12)
        axes[r, 0].add_patch(Rectangle((0.0, 0.0), 1.0, 0.085, transform=axes[r, 0].transAxes,
                                       facecolor=tag_col, alpha=0.92, zorder=5))
        axes[r, 0].text(0.5, 0.042, tag, transform=axes[r, 0].transAxes,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white", zorder=6)
        print(f"  {idx}  src {source_id(int(idx)):3d}  "
              f"{'HELD OUT' if held else 'in training set':16s}  "
              f"PSNR {p:6.2f}  SSIM {s:.4f}")

    n_train = sum(1 for i, _ in rows if int(i) not in val)
    fig.text(0.5, 0.004,
             f"{n_train} of {n} rows were IN THE TRAINING SET (orange badge) — "
             f"those reconstructions are not evidence of generalisation.\n"
             f"Split by source image, source_id = index // 4, seed 1337. "
             f"Metrics at data_range=1.0.",
             ha="center", fontsize=8.5, color=GREY)

    fig.tight_layout(rect=[0.0, 0.028, 1.0, 0.985])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
