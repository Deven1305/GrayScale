"""Metric correctness. The PSNR(x,x)==inf assertion is the tripwire for a
wrong data_range, which would silently invalidate every number downstream."""
import torch

from src.metrics.full_reference import evaluate_batch, psnr, ssim


def test_psnr_identical_is_inf():
    x = torch.rand(2, 1, 64, 64)
    assert torch.isinf(psnr(x, x)), "PSNR(x,x) must be inf; data_range is wrong"


def test_ssim_identical_is_one():
    x = torch.rand(2, 1, 64, 64)
    assert abs(float(ssim(x, x)) - 1.0) < 1e-4


def test_psnr_known_value():
    """A constant offset of 0.1 on [0,1] data gives exactly 20*log10(1/0.1)."""
    x = torch.full((1, 1, 32, 32), 0.5)
    y = torch.full((1, 1, 32, 32), 0.6)
    assert abs(float(psnr(y, x)) - 20.0) < 1e-3


def test_psnr_decreases_with_noise():
    x = torch.rand(1, 1, 64, 64)
    a = float(psnr((x + 0.01 * torch.randn_like(x)).clamp(0, 1), x))
    b = float(psnr((x + 0.10 * torch.randn_like(x)).clamp(0, 1), x))
    assert a > b


def test_evaluate_batch_keys():
    x = torch.rand(2, 1, 64, 64)
    m = evaluate_batch(x, x, with_lpips=False)
    for k in ("psnr", "ssim", "msssim", "l1"):
        assert k in m
