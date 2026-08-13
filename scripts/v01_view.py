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


def cmd_compare(args):
    """Side-by-side INPUT vs MODEL OUTPUT for any two folders of .npy files.

    Works on the test set, where no ground truth exists, and on any folder you
    have run inference over.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ind, outd = Path(args.input_dir), Path(args.output_dir)
    ins = sorted(ind.glob("*.npy"))[args.start:args.start + args.count]
    if not ins:
        raise SystemExit(f"no .npy files in {ind}")

    rows = []
    for f in ins:
        o = outd / f.name
        if o.exists():
            rows.append((f.stem, np.load(f), np.load(o)))
    if not rows:
        raise SystemExit(f"no matching outputs in {outd}")

    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(12, 3.7 * n))
    ax = np.atleast_2d(ax)
    for r, (name, lo, hi) in enumerate(rows):
        big = np.kron(lo, np.ones((2, 2)))          # nearest x2, same size as out
        cs = min(96, hi.shape[0])
        best, by, bx = -1, 0, 0
        for y in range(0, hi.shape[0] - cs + 1, 24):
            for x in range(0, hi.shape[1] - cs + 1, 24):
                v = hi[y:y + cs, x:x + cs].std()
                if v > best:
                    best, by, bx = v, y, x
        nl = "\n"
        panels = [
            (big, f"INPUT {lo.shape[0]}x{lo.shape[1]}{nl}"
                  f"range [{lo.min():.2f}, {lo.max():.2f}]", "#E76F51"),
            (hi, f"MODEL OUTPUT {hi.shape[0]}x{hi.shape[1]}{nl}"
                 f"range [{hi.min():.2f}, {hi.max():.2f}]", "#2A9D8F"),
            (hi[by:by + cs, bx:bx + cs], "OUTPUT, zoomed", "#14213D"),
        ]
        for c, (img, title, col) in enumerate(panels):
            ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1,
                            interpolation="nearest")
            ax[r, c].set_title(title, fontsize=10, color=col, fontweight="bold")
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(f"{ind}  ->  {outd}", fontsize=12)
    fig.tight_layout()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    o = PREVIEW / f"compare_{ind.name}_{args.start}.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {o}")
    print("Open that PNG to see input vs model output side by side.")


def cmd_topng(args):
    """Convert ANY folder of .npy to browsable PNG. Works on inputs, outputs,
    anything."""
    import cv2
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npy"))[:args.limit] if args.limit else sorted(src.glob("*.npy"))
    for f in files:
        cv2.imwrite(str(dst / f"{f.stem}.png"), to_png_array(np.load(f)))
    print(f"wrote {len(files)} PNGs to {dst}")
    print("Open that folder in File Explorer and browse with the arrow keys.")
    print("NOTE: values outside [0,1] are clipped for display only.")


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

    p = sub.add_parser("compare", help="INPUT vs MODEL OUTPUT, side by side")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=4)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("topng", help="convert ANY folder of .npy to PNG")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_topng)

    p = sub.add_parser("stats", help="numeric summary of one sample")
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
