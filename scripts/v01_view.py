"""Look at the dataset. The images are float32 .npy arrays, so Windows Photos
cannot open them — this converts / displays them.

  # one GT+NoisyLR pair, side by side, with a zoomed crop
  python scripts/v01_view.py pair --index 0

  # export .npy -> .png so you can browse them in File Explorer
  python scripts/v01_view.py export --split train_gt  --limit 40
  python scripts/v01_view.py export --split train_lr  --limit 40
  python scripts/v01_view.py export --split test      --limit 40

  # a contact sheet of many images in one picture
  python scripts/v01_view.py sheet --split train_gt --start 0 --count 60

  # everything about one sample, as numbers
  python scripts/v01_view.py stats --index 0

Outputs go to docs/preview/ (gitignored — regenerate any time).
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train_gt": ROOT / "data/train/train/GT",
    "train_lr": ROOT / "data/train/train/NoisyLR",
    "test": ROOT / "data/Test_NoisyLR/NoisyLR",
}
PREVIEW = ROOT / "docs/preview"


def load(split, index):
    d = SPLITS[split]
    p = d / f"{index:06d}.npy"
    if not p.exists():
        raise SystemExit(f"no such file: {p}")
    return np.load(p)


def to_png_array(a):
    """float32 (possibly outside [0,1]) -> uint8 for display.

    Clipping is ONLY for viewing. Never clip when feeding a model.
    """
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def cmd_export(args):
    import cv2
    d = SPLITS[args.split]
    out = PREVIEW / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(d.glob("*.npy"))[args.start:args.start + args.limit]
    for p in files:
        a = np.load(p)
        cv2.imwrite(str(out / f"{p.stem}.png"), to_png_array(a))
    print(f"wrote {len(files)} PNGs to {out}")
    print("Open that folder in File Explorer and browse with the arrow keys.")
    print("NOTE: values outside [0,1] are clipped for display only.")


def cmd_sheet(args):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = SPLITS[args.split]
    files = sorted(d.glob("*.npy"))[args.start:args.start + args.count]
    nc = 10
    nr = int(np.ceil(len(files) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 1.5, nr * 1.62))
    ax = np.atleast_2d(ax)
    for k, a_ in enumerate(ax.ravel()):
        a_.axis("off")
        if k < len(files):
            a_.imshow(np.load(files[k]), cmap="gray", vmin=0, vmax=1)
            a_.set_title(files[k].stem, fontsize=5, pad=1)
    fig.suptitle(f"{args.split}  —  samples {args.start} to "
                 f"{args.start + len(files) - 1}", fontsize=11)
    fig.tight_layout()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    o = PREVIEW / f"sheet_{args.split}_{args.start}.png"
    fig.savefig(o, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {o}")


def cmd_pair(args):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import cv2

    gt = load("train_gt", args.index)
    lr = load("train_lr", args.index)
    up = cv2.resize(lr.astype(np.float32), (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    cs = args.crop
    cy = min(max(args.cy, 0), gt.shape[0] - cs)
    cx = min(max(args.cx, 0), gt.shape[1] - cs)

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.4))
    ax[0, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].add_patch(Rectangle((cx, cy), cs, cs, ec="#E76F51", fc="none", lw=2))
    ax[0, 0].set_title(f"GT  {gt.shape}  (the target)", fontsize=11)

    ax[0, 1].imshow(lr, cmap="gray", vmin=0, vmax=1)
    ax[0, 1].set_title(f"NoisyLR  {lr.shape}  (the input)", fontsize=11)

    ax[0, 2].imshow(up, cmap="gray", vmin=0, vmax=1)
    ax[0, 2].set_title("NoisyLR upscaled ×2 (bicubic baseline)", fontsize=11)

    ax[1, 0].imshow(gt[cy:cy + cs, cx:cx + cs], cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
    ax[1, 0].set_title("GT — zoomed crop", fontsize=11)
    ax[1, 1].imshow(up[cy:cy + cs, cx:cx + cs], cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
    ax[1, 1].set_title("Degraded — zoomed crop (see the speckle)", fontsize=11,
                       color="#E76F51")
    for a_ in ax[:, :2].ravel():
        a_.set_xticks([])
        a_.set_yticks([])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])

    a_ = ax[1, 2]
    bins = np.linspace(-0.3, 1.8, 140)
    a_.hist(gt.ravel(), bins, density=True, alpha=.6, label="GT", color="#2A6FDB")
    a_.hist(lr.ravel(), bins, density=True, alpha=.6, label="NoisyLR",
            color="#E76F51")
    a_.axvline(0, color="k", lw=.7, ls=":")
    a_.axvline(1, color="k", lw=.7, ls=":")
    a_.set_yscale("log")
    a_.legend(fontsize=9)
    a_.set_title(f"NoisyLR range [{lr.min():.3f}, {lr.max():.3f}]  —  "
                 f"{(lr > 1).mean()*100:.2f}% above 1", fontsize=10)

    fig.suptitle(f"Sample {args.index:06d}   (source image "
                 f"{args.index // 4}, crop {args.index % 4} of 4)", fontsize=13)
    fig.tight_layout()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    o = PREVIEW / f"pair_{args.index:06d}.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {o}")


def cmd_stats(args):
    gt = load("train_gt", args.index)
    lr = load("train_lr", args.index)
    print(f"=== sample {args.index:06d} ===")
    print(f"  source image id : {args.index // 4}   (crop "
          f"{args.index % 4} of 4 from that photograph)")
    for name, a in (("GT     ", gt), ("NoisyLR", lr)):
        print(f"  {name}  shape={a.shape}  dtype={a.dtype}")
        print(f"           min={a.min():+.5f}  max={a.max():+.5f}  "
              f"mean={a.mean():.5f}  std={a.std():.5f}")
        print(f"           below 0: {(a < 0).mean()*100:6.3f}%   "
              f"above 1: {(a > 1).mean()*100:6.3f}%")
    print(f"  siblings (same source): "
          f"{[f'{(args.index//4)*4+k:06d}' for k in range(4)]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="convert .npy to browsable .png")
    p.add_argument("--split", choices=SPLITS, default="train_gt")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("sheet", help="contact sheet of many images")
    p.add_argument("--split", choices=SPLITS, default="train_gt")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=60)
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("pair", help="GT vs NoisyLR for one training sample")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--crop", type=int, default=64)
    p.add_argument("--cx", type=int, default=96)
    p.add_argument("--cy", type=int, default=96)
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("stats", help="numeric summary of one sample")
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
