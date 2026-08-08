"""Procedurally generated structured content.

WHY THIS EXISTS. The KLA corpus is 800 natural photographs and contains almost
no strong periodic structure. Measured consequence: on a fine checkerboard our
model scores 15.20 dB / 0.792 SSIM against bicubic's 15.64 / 0.807 — it is
WORSE than doing nothing, because it invents irregular blotching where the
truth is a regular grid.

That matters beyond a synthetic curiosity: semiconductor structures are
periodic (DRAM arrays, FinFET gates, gratings), and the scored test set is
promised to contain out-of-distribution content.

This module manufactures that missing content — gratings, checkerboards, line
grids, concentric rings, edges and steps — at random period, phase, orientation
and contrast. No download required, and unlimited quantity.

Generated at HR; the training pipeline degrades them with the Phase 0 replica,
so ground truth is exact.
"""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, degrade_torch


def _grating(h, w, rng):
    """Sinusoidal or square-wave grating at a random period and angle."""
    period = rng.uniform(4, 40)
    theta = rng.uniform(0, np.pi)
    phase = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    v = np.sin(2 * np.pi * proj / period + phase)
    if rng.random() < 0.5:                       # square wave half the time
        v = np.sign(v)
    return v


def _checker(h, w, rng):
    p = int(rng.integers(3, 24))
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // p) + (xx // p)) % 2 == 0, 1.0, -1.0).astype(np.float32)


def _linegrid(h, w, rng):
    p = int(rng.integers(6, 40))
    t = max(1, int(p * rng.uniform(0.12, 0.4)))
    a = np.full((h, w), -1.0, np.float32)
    a[::p] = 1.0
    for k in range(1, t):
        a[k::p] = 1.0
    if rng.random() < 0.6:                       # cross-hatch
        a[:, ::p] = 1.0
        for k in range(1, t):
            a[:, k::p] = 1.0
    return a


def _rings(h, w, rng):
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    period = rng.uniform(5, 30)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    v = np.sin(2 * np.pi * r / period)
    return np.sign(v) if rng.random() < 0.5 else v


def _steps(h, w, rng):
    """Piecewise-constant blocks — hard edges at arbitrary orientation."""
    a = np.zeros((h, w), np.float32)
    n = int(rng.integers(2, 7))
    ys = np.sort(rng.choice(np.arange(1, h), n - 1, replace=False))
    xs = np.sort(rng.choice(np.arange(1, w), n - 1, replace=False))
    ys = np.concatenate([[0], ys, [h]])
    xs = np.concatenate([[0], xs, [w]])
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = rng.uniform(-1, 1)
    return a


def _shapes(h, w, rng):
    a = np.full((h, w), rng.uniform(-1, -0.4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(2, 6))):
        val = rng.uniform(-0.2, 1.0)
        if rng.random() < 0.5:
            y0, x0 = rng.integers(0, h - 8), rng.integers(0, w - 8)
            hh = int(rng.integers(8, max(9, h // 2)))
            ww = int(rng.integers(8, max(9, w // 2)))
            a[y0:y0 + hh, x0:x0 + ww] = val
        else:
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            r = rng.uniform(6, min(h, w) / 3)
            a[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = val
    return a


GENERATORS = (_grating, _checker, _linegrid, _rings, _steps, _shapes)
WEIGHTS = np.array([0.28, 0.20, 0.20, 0.12, 0.10, 0.10])


def make_pattern(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """One clean HR pattern in [0,1]."""
    g = GENERATORS[rng.choice(len(GENERATORS), p=WEIGHTS)]
    a = g(h, w, rng)

    # random contrast and offset so the model sees these at every brightness,
    # including the dark end where speckle is hardest to separate from signal
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-8)
    contrast = rng.uniform(0.15, 1.0)
    offset = rng.uniform(0, 1 - contrast)
    a = a * contrast + offset

    if rng.random() < 0.3:                       # occasional smooth background
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (yy / h) * rng.uniform(-0.3, 0.3) + (xx / w) * rng.uniform(-0.3, 0.3)
        a = np.clip(a + ramp, 0, 1)
    return np.ascontiguousarray(a, dtype=np.float32)


class PatternPairs(Dataset):
    """(degraded LR, clean HR) pairs from procedurally generated structure."""

    def __init__(self, cfg: DegradationConfig, hr_patch: int = 128,
                 length: int = 2000, seed: int = 0, fixed: bool = False):
        self.cfg = cfg
        self.hr = hr_patch
        self.length = length
        self.seed = seed
        self.fixed = fixed                        # deterministic (validation)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        s = self.seed + i if self.fixed else int(
            torch.randint(0, 2 ** 31 - 1, (1,)).item())
        rng = np.random.default_rng(s)
        hr = make_pattern(self.hr, self.hr, rng)
        t = torch.from_numpy(hr)[None, None]
        g = torch.Generator().manual_seed(s)
        lr = degrade_torch(t, self.cfg, g)[0]
        return lr, t[0]
