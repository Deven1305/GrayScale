"""THE ACCEPTANCE GATE for the whole strategy.

If our replica does not reproduce the real NoisyLR statistics, then every
synthetic pair we train on is off-distribution and the OOD strategy collapses.
Thresholds come from the measured values in docs/forensics_report.md §8.4.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.degradation import DegradationConfig, degrade_torch

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/train/train/GT"
LR = ROOT / "data/train/train/NoisyLR"
pytestmark = pytest.mark.skipif(not GT.exists(), reason="dataset not present")

N = 120


def _real_and_sim(seed=0):
    rng = np.random.default_rng(seed)
    files = sorted(GT.glob("*.npy"))
    pick = [files[i] for i in rng.choice(len(files), N, replace=False)]
    cfg = DegradationConfig()
    g = torch.Generator().manual_seed(seed)
    real, sim = [], []
    for p in pick:
        hr = np.load(p).astype(np.float32)
        real.append(np.load(LR / f"{p.stem}.npy").astype(np.float32).ravel())
        t = torch.from_numpy(hr)[None, None]
        sim.append(degrade_torch(t, cfg, g)[0, 0].numpy().ravel())
    return np.concatenate(real), np.concatenate(sim)


def test_output_is_float32_and_half_size():
    x = torch.rand(2, 1, 256, 256)
    y = degrade_torch(x, DegradationConfig(), torch.Generator().manual_seed(0))
    assert y.shape == (2, 1, 128, 128)
    assert y.dtype == torch.float32


def test_output_is_not_clipped():
    """Out-of-range values are a feature; a clipped replica is a bug."""
    x = torch.rand(4, 1, 128, 128) * 0.6 + 0.4
    y = degrade_torch(x, DegradationConfig(), torch.Generator().manual_seed(1))
    assert float(y.max()) > 1.0, "replica never exceeds 1.0 — is it clipping?"


def test_speckle_is_multiplicative():
    """Noise magnitude must scale with signal brightness."""
    g = torch.Generator().manual_seed(3)
    cfg = DegradationConfig(sigma_log10_range=(-4, -4))
    dark = torch.full((8, 1, 128, 128), 0.1)
    bright = torch.full((8, 1, 128, 128), 0.8)
    sd = float((degrade_torch(dark, cfg, g) - 0.1).std())
    sb = float((degrade_torch(bright, cfg, g) - 0.8).std())
    assert sb > 4 * sd, f"noise not signal-dependent: {sd:.4f} vs {sb:.4f}"


def test_histogram_overlap_matches_real():
    real, sim = _real_and_sim()
    bins = np.linspace(-0.3, 2.2, 200)
    hr, _ = np.histogram(real, bins, density=True)
    hs, _ = np.histogram(sim, bins, density=True)
    overlap = float(np.minimum(hr, hs).sum() * (bins[1] - bins[0]))
    assert overlap > 0.95, f"histogram overlap {overlap:.4f} < 0.95"


def test_out_of_range_fractions_match_real():
    real, sim = _real_and_sim()
    assert abs((sim > 1).mean() - (real > 1).mean()) < 0.010
    assert abs((sim < 0).mean() - (real < 0).mean()) < 0.004


def test_moments_match_real():
    real, sim = _real_and_sim()
    assert abs(sim.mean() - real.mean()) < 0.02
    assert abs(sim.std() - real.std()) < 0.03
