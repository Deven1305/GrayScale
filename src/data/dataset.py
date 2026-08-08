"""Paired datasets.

Two sources of training pairs:
  1. KLAPairs      — the real (GT, NoisyLR) pairs. Anchors us to the true
                     degradation.
  2. SyntheticPairs — clean images + the Phase 0 replica, applied on the fly.
                     Supplies content diversity for OOD robustness.

MixedDataset interleaves them at a configurable ratio (default 50/50).

Inputs are NEVER clipped. Everything stays float32.
"""
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, degrade_torch


def dihedral(x: np.ndarray, y: np.ndarray, k: int):
    """8-fold dihedral augmentation applied identically to both images."""
    if k & 1:
        x, y = x[:, ::-1], y[:, ::-1]
    if k & 2:
        x, y = x[::-1, :], y[::-1, :]
    if k & 4:
        x, y = x.T, y.T
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class KLAPairs(Dataset):
    """Real KLA pairs: NoisyLR (128) -> GT (256).

    Reads from the memory-mapped bundle in data/processed/ when it exists
    (built by scripts/precrop_patches.py), otherwise falls back to per-file
    np.load. The bundle removes the per-sample file open that otherwise leaves
    the GPU idle ~84% of the time.
    """

    def __init__(self, root: Path, indices: Sequence[int], lr_patch: int = 64,
                 augment: bool = True, scale: int = 2, full: bool = False):
        root = Path(root)
        self.gt_dir = root / "train/train/GT"
        self.lr_dir = root / "train/train/NoisyLR"
        self.indices = list(indices)
        self.lr_patch = lr_patch
        self.augment = augment
        self.scale = scale
        self.full = full            # full image, no crop (for validation)

        gt_b, lr_b = root / "processed/gt.npy", root / "processed/lr.npy"
        self.bundled = gt_b.exists() and lr_b.exists()
        self._gt = self._lr = None
        self._gt_path, self._lr_path = gt_b, lr_b

    def _maps(self):
        # opened lazily so each DataLoader worker gets its own handle
        if self._gt is None:
            self._gt = np.load(self._gt_path, mmap_mode="r")
            self._lr = np.load(self._lr_path, mmap_mode="r")
        return self._gt, self._lr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if self.bundled:
            g, l = self._maps()
            gt = np.asarray(g[idx])
            lr = np.asarray(l[idx])
        else:
            gt = np.load(self.gt_dir / f"{idx:06d}.npy")
            lr = np.load(self.lr_dir / f"{idx:06d}.npy")

        if not self.full:
            p = self.lr_patch
            H, W = lr.shape
            if H > p:
                ty = np.random.randint(0, H - p + 1)
                tx = np.random.randint(0, W - p + 1)
                lr = lr[ty:ty + p, tx:tx + p]
                gt = gt[ty * self.scale:(ty + p) * self.scale,
                        tx * self.scale:(tx + p) * self.scale]
            if self.augment:
                lr, gt = dihedral(lr, gt, np.random.randint(0, 8))

        return (torch.from_numpy(lr.astype(np.float32))[None],
                torch.from_numpy(gt.astype(np.float32))[None])


class SyntheticPairs(Dataset):
    """Clean images + the degradation replica, applied on the fly on GPU-free
    CPU workers. Used for external content (DIV2K, Urban100, ...) and for the
    OOD validation families."""

    def __init__(self, files: List[Path], cfg: DegradationConfig,
                 hr_patch: int = 128, augment: bool = True, seed: int = 0,
                 fixed: bool = False):
        self.files = list(files)
        self.cfg = cfg
        self.hr_patch = hr_patch
        self.augment = augment
        self.fixed = fixed          # deterministic degradation (validation)
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def _load(self, p: Path) -> np.ndarray:
        if p.suffix == ".npy":
            a = np.load(p)
        else:
            import cv2
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if a is None:
                raise IOError(f"cannot read {p}")
            a = a.astype(np.float32) / 255.0
        return a.astype(np.float32)

    def __getitem__(self, i):
        hr = self._load(self.files[i])
        p = self.hr_patch
        H, W = hr.shape
        if not self.fixed and (H > p or W > p):
            ty = np.random.randint(0, max(H - p, 0) + 1)
            tx = np.random.randint(0, max(W - p, 0) + 1)
            hr = hr[ty:ty + p, tx:tx + p]
        # crop to an even multiple so the 2x decimation is exact
        H, W = hr.shape
        hr = hr[: H // 2 * 2, : W // 2 * 2]
        if self.augment and not self.fixed:
            hr, _ = dihedral(hr, hr, np.random.randint(0, 8))

        t = torch.from_numpy(np.ascontiguousarray(hr, np.float32))[None, None]
        g = torch.Generator()
        g.manual_seed(self.seed + i if self.fixed else int(
            torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        lr = degrade_torch(t, self.cfg, g)[0]
        return lr, t[0]


class MixedDataset(Dataset):
    """Interleave two datasets at a fixed ratio. Length is set by `epoch_len`
    so the ratio is exact regardless of the underlying sizes."""

    def __init__(self, primary: Dataset, secondary: Optional[Dataset],
                 ratio: float = 0.5, epoch_len: Optional[int] = None):
        self.primary = primary
        self.secondary = secondary
        self.ratio = ratio if secondary is not None else 1.0
        self.epoch_len = epoch_len or len(primary)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        if self.secondary is None or np.random.rand() < self.ratio:
            return self.primary[np.random.randint(len(self.primary))]
        return self.secondary[np.random.randint(len(self.secondary))]


class ExternalHRPairs(Dataset):
    """Clean HR patches from the memory-mapped external bundle, degraded on the
    fly with the Phase 0 replica.

    Reads from data/processed/external_hr.npy (built by precrop_patches.py).
    Avoids the ~25 ms PNG decode that made SyntheticPairs 18x slower than
    KLAPairs.
    """

    def __init__(self, bundle: Path, cfg: DegradationConfig,
                 hr_patch: int = 192, augment: bool = True):
        self.path = Path(bundle)
        self.cfg = cfg
        self.hr_patch = hr_patch
        self.augment = augment
        self._m = None
        self.n = int(np.load(self.path, mmap_mode="r").shape[0])

    def _map(self):
        if self._m is None:
            self._m = np.load(self.path, mmap_mode="r")
        return self._m

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        m = self._map()
        hr = np.asarray(m[i % self.n])
        p = self.hr_patch
        H, W = hr.shape
        if H > p:
            ty = np.random.randint(0, H - p + 1)
            tx = np.random.randint(0, W - p + 1)
            hr = hr[ty:ty + p, tx:tx + p]
        if self.augment:
            hr, _ = dihedral(hr, hr, np.random.randint(0, 8))
        t = torch.from_numpy(np.ascontiguousarray(hr, np.float32))[None, None]
        g = torch.Generator()
        g.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        return degrade_torch(t, self.cfg, g)[0], t[0]
