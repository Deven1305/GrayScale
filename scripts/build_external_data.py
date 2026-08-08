"""Phase 2 — ingest external clean imagery for training and OOD validation.

Why this matters: the KLA corpus is 800 natural photographs and nothing else.
The model consequently over-smooths texture and LOSES to bicubic on fine
periodic structure. External content is the largest single lever we have, and
it is explicitly permitted ("synthetic data generation is explicitly
encouraged", KLA FAQ slide 20).

Downloads clean HR images, converts to grayscale float32 [0,1], centre-crops to
a multiple of 2, and stores as .npy so the training pipeline can degrade them
on the fly with our validated replica.

    python scripts/build_external_data.py                 # all families
    python scripts/build_external_data.py --only Urban100
    python scripts/build_external_data.py --limit 50      # quick smoke test

Families and their role:
    DIV2K     general natural content  -> training diversity
    Urban100  buildings / cityscapes   -> the OOD case KLA named by name
    BSD100    natural scenes           -> OOD validation
    Set14     classic SR benchmark     -> OOD validation
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external"

# HuggingFace repos holding the standard SR benchmark images.
# mode "archive" = one tar/zip of images; mode "files" = individual files.
FAMILIES = {
    "DIV2K":    ("mAiello00/DIV2K",      "files",   "train",
                 "DIV2K_train_HR/"),
    "Urban100": ("eugenesiow/Urban100",  "archive", "ood",  None),
    "BSD100":   ("eugenesiow/BSD100",    "archive", "ood",  None),
    "Set14":    ("eugenesiow/Set14",     "archive", "ood",  None),
}

# DIV2K images are ~2040x1356. Stored as float32 grayscale that is 11 MB each,
# so we save 512x512 grayscale PNG crops instead: same content diversity,
# ~50x less disk. SyntheticPairs reads PNG natively.
CROP = 512
CROPS_PER_IMAGE = 2


def to_gray_f32(path: Path) -> np.ndarray:
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise IOError(f"unreadable: {path}")
    a = a.astype(np.float32) / 255.0
    h, w = a.shape
    return np.ascontiguousarray(a[: h // 2 * 2, : w // 2 * 2])


def fetch(repo: str, split: str, dest: Path, limit: int) -> int:
    """Pull the HR images out of a HuggingFace SR dataset repo."""
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(repo, repo_type="dataset")
    # these repos ship archived HR/LR folders (.tar.gz or .zip); we want HR
    ARCH = (".tar.gz", ".tgz", ".zip", ".tar")
    cands = [f for f in files
             if f.lower().endswith(ARCH) and "_hr" in f.lower()]
    if not cands:
        cands = [f for f in files if f.lower().endswith(ARCH)]
    if not cands:
        print(f"    no archive found in {repo}; files: {files[:8]}")
        return 0

    cands.sort(key=len)                       # prefer the plain HR archive
    zf = cands[0]
    print(f"    downloading {zf}")
    local = hf_hub_download(repo, zf, repo_type="dataset")

    tmp = dest.parent / f"_tmp_{dest.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.unpack_archive(local, tmp)

    imgs = sorted([p for p in tmp.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    if limit:
        imgs = imgs[:limit]
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in imgs:
        try:
            a = to_gray_f32(p)
        except Exception as e:
            print(f"      skip {p.name}: {e}")
            continue
        if min(a.shape) < 128:                # too small to crop 128 HR patches
            continue
        np.save(dest / f"{n:04d}.npy", a)
        n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return n


def fetch_files(repo: str, prefix: str, dest: Path, limit: int) -> int:
    """Repos that store individual images. Saves 512x512 grayscale PNG crops."""
    import cv2
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix)
                   and f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit:
        files = files[:limit]
    if not files:
        print(f"    nothing under {prefix!r} in {repo}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 0
    for i, f in enumerate(files):
        try:
            local = hf_hub_download(repo, f, repo_type="dataset")
            a = to_gray_f32(Path(local))
        except Exception as e:
            print(f"      skip {f}: {type(e).__name__}")
            continue
        h, w = a.shape
        if min(h, w) < CROP:
            continue
        for c in range(CROPS_PER_IMAGE):
            y = int(rng.integers(0, h - CROP + 1))
            x = int(rng.integers(0, w - CROP + 1))
            patch = (a[y:y + CROP, x:x + CROP] * 255.0 + 0.5).astype(np.uint8)
            cv2.imwrite(str(dest / f"{n:05d}.png"), patch)
            n += 1
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(files)} source images -> {n} crops",
                  flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    names = args.only or list(FAMILIES)
    summary = {}
    for name in names:
        if name not in FAMILIES:
            print(f"unknown family {name}; options {list(FAMILIES)}")
            continue
        repo, mode, role, prefix = FAMILIES[name]
        dest = OUT / role / name
        have = len(list(dest.glob("*.npy"))) + len(list(dest.glob("*.png"))) \
            if dest.exists() else 0
        if have and not args.limit:
            print(f"[{name}] already present: {have} images")
            summary[name] = (role, have)
            continue
        print(f"[{name}] role={role} mode={mode}")
        try:
            n = (fetch_files(repo, prefix, dest, args.limit) if mode == "files"
                 else fetch(repo, mode, dest, args.limit))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {str(e)[:160]}")
            n = 0
        print(f"    -> {n} images at {dest.relative_to(ROOT)}")
        summary[name] = (role, n)

    print("\n=== summary ===")
    tr = sum(n for r, n in summary.values() if r == "train")
    od = sum(n for r, n in summary.values() if r == "ood")
    for k, (r, n) in summary.items():
        print(f"  {k:10s} {r:6s} {n:5d}")
    print(f"\n  training pool : {tr}")
    print(f"  OOD families  : {od}")
    if tr:
        print("\nNext: set  external_glob: 'data/external/train/**/*.npy'")
        print("      and  external_ratio: 0.5   in configs/*.yaml")


if __name__ == "__main__":
    main()
