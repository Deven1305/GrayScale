"""Pack the KLA pairs into two memory-mapped .npy bundles.

Per-sample file opens dominate dataloading: the training loop ran at 20 img/s
against 125 img/s of pure GPU compute, i.e. the GPU was idle ~84% of the time.
Two np.load calls per sample at random offsets is the cause.

Bundling into a single contiguous array and reading it with mmap_mode='r'
removes the per-sample open entirely. Size is modest:
    GT  3200 x 256 x 256 x 4B = 838 MB
    LR  3200 x 128 x 128 x 4B = 210 MB

    python scripts/precrop_patches.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/train/train/GT"
LR = ROOT / "data/train/train/NoisyLR"
OUT = ROOT / "data/processed"


def bundle_external(patch=256, crops_per_image=6):
    """Pack external PNGs into one memmap of clean HR patches.

    SyntheticPairs costs 38.6 ms/item against KLAPairs' 2.2 ms, and PNG decode
    is the bulk of it. Bundling removes the decode entirely — the same fix that
    took KLA dataloading from 20 to 136 img/s.
    """
    import cv2
    src = ROOT / "data/external/train"
    files = sorted(src.rglob("*.png")) + sorted(src.rglob("*.npy"))
    if not files:
        print("[external] nothing to bundle (run build_external_data.py first)")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_total = len(files) * crops_per_image
    out = np.lib.format.open_memmap(OUT / "external_hr.npy", mode="w+",
                                    dtype=np.float32,
                                    shape=(n_total, patch, patch))
    k = 0
    for i, f in enumerate(files):
        if f.suffix == ".npy":
            a = np.load(f).astype(np.float32)
        else:
            a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            a = a.astype(np.float32) / 255.0
        h, w = a.shape
        if min(h, w) < patch:
            continue
        for _ in range(crops_per_image):
            y = int(rng.integers(0, h - patch + 1))
            x = int(rng.integers(0, w - patch + 1))
            out[k] = a[y:y + patch, x:x + patch]
            k += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} -> {k} patches", flush=True)
    out.flush()

    # trim to what we actually filled
    if k < n_total:
        trimmed = np.lib.format.open_memmap(
            OUT / "external_hr_tmp.npy", mode="w+", dtype=np.float32,
            shape=(k, patch, patch))
        trimmed[:] = out[:k]
        trimmed.flush()
        del trimmed, out
        (OUT / "external_hr.npy").unlink()
        (OUT / "external_hr_tmp.npy").rename(OUT / "external_hr.npy")
    print(f"[external] {k} clean {patch}x{patch} patches from {len(files)} files"
          f" -> {(OUT/'external_hr.npy').stat().st_size/1e6:.0f} MB")
    return k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(GT.glob("*.npy"))
    n = len(files)
    if n == 0:
        sys.exit("no GT files found")

    g0 = np.load(files[0])
    l0 = np.load(LR / files[0].name)
    print(f"packing {n} pairs: GT{g0.shape} LR{l0.shape}")

    gt_path = OUT / "gt.npy"
    lr_path = OUT / "lr.npy"
    gt = np.lib.format.open_memmap(gt_path, mode="w+", dtype=np.float32,
                                   shape=(n,) + g0.shape)
    lr = np.lib.format.open_memmap(lr_path, mode="w+", dtype=np.float32,
                                   shape=(n,) + l0.shape)

    t0 = time.perf_counter()
    for i, p in enumerate(files):
        gt[i] = np.load(p)
        lr[i] = np.load(LR / p.name)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)
    gt.flush()
    lr.flush()
    dt = time.perf_counter() - t0

    print(f"wrote {gt_path} ({gt_path.stat().st_size/1e6:.0f} MB)")
    print(f"wrote {lr_path} ({lr_path.stat().st_size/1e6:.0f} MB)")
    print(f"took {dt:.1f}s")

    # index i in the bundle == sample index i, so source_id = i // 4 still holds
    a = np.load(gt_path, mmap_mode="r")
    b = np.load(lr_path, mmap_mode="r")
    ok = np.array_equal(a[7], np.load(files[7])) and \
        np.array_equal(b[7], np.load(LR / files[7].name))
    print("verify sample 7 round-trips:", ok)
    if not ok:
        sys.exit("BUNDLE MISMATCH")

    print()
    bundle_external()


if __name__ == "__main__":
    main()
