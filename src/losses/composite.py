"""Composite restoration loss.

    L = 1.0*Charbonnier + 0.2*(1-MS-SSIM) + 0.1*FFT + 0.05*gradient [+ VGG]

Rationale, tied to what is actually scored (SSIM, pSNR, LPIPS):
  * Charbonnier — a smooth L1. Pure L2/MSE over-smooths, and the spec
    explicitly forbids blurring to remove noise.
  * MS-SSIM     — directly optimises one of the three scored metrics.
  * FFT         — L1 in the frequency domain; restores the high-frequency band
    that 2x decimation destroyed, without the ringing an adversarial loss
    would introduce.
  * gradient    — L1 on Sobel maps; keeps edges sharp.
  * VGG         — optional perceptual term, proxy for LPIPS. Off by default
    because it pulls torchvision into the training path for a small gain.

❌ NO adversarial loss. The spec forbids "artificial patterns or ringing", and
GAN training costs PSNR and SSIM — two of the three scored metrics.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    multiscale_structural_similarity_index_measure as msssim_fn,
)
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class FFTLoss(nn.Module):
    """L1 between the complex spectra, as real/imag stacks."""

    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred.float(), norm="ortho")
        tf = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(torch.stack([pf.real, pf.imag], -1),
                         torch.stack([tf.real, tf.imag], -1))


class GradientLoss(nn.Module):
    """L1 on Sobel gradient magnitude maps."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def _grad(self, x):
        c = x.shape[1]
        kx = self.kx.to(x.dtype).expand(c, 1, 3, 3)
        ky = self.ky.to(x.dtype).expand(c, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))


class MSSSIMLoss(nn.Module):
    def forward(self, pred, target):
        p = pred.clamp(0, 1).float()
        t = target.float()
        if min(p.shape[-2:]) < 161:
            return 1.0 - ssim_fn(p, t, data_range=1.0)
        return 1.0 - msssim_fn(p, t, data_range=1.0)


class VGGPerceptualLoss(nn.Module):
    """Optional. Imports torchvision lazily so it never touches inference."""

    def __init__(self, layer: int = 16):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.net = vgg16(weights=VGG16_Weights.DEFAULT).features[:layer].eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = x.clamp(0, 1).repeat(1, 3, 1, 1).float()
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        return F.l1_loss(self.net(self._prep(pred)), self.net(self._prep(target)))


class CompositeLoss(nn.Module):
    """Weights come from the config; zero disables a term entirely."""

    def __init__(self, charbonnier=1.0, msssim=0.2, fft=0.1, gradient=0.05,
                 vgg=0.0, l2=0.0):
        super().__init__()
        self.w = {"charbonnier": charbonnier, "msssim": msssim, "fft": fft,
                  "gradient": gradient, "vgg": vgg, "l2": l2}
        self.charb = CharbonnierLoss()
        self.ms = MSSSIMLoss() if msssim > 0 else None
        self.fft = FFTLoss() if fft > 0 else None
        self.grad = GradientLoss() if gradient > 0 else None
        self.vgg = VGGPerceptualLoss() if vgg > 0 else None

    def forward(self, pred, target) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """Returns (total, parts).

        `parts` holds DETACHED TENSORS, not floats. Calling float() here would
        force a GPU sync on every term on every step — measured at a 15x
        throughput loss (125 img/s -> 8 img/s). The trainer accumulates these
        on-device and syncs once per epoch instead.
        """
        parts = {}
        total = pred.new_zeros(())
        if self.w["charbonnier"]:
            v = self.charb(pred, target)
            parts["charbonnier"] = v.detach()
            total = total + self.w["charbonnier"] * v
        if self.w["l2"]:
            v = F.mse_loss(pred, target)
            parts["l2"] = v.detach()
            total = total + self.w["l2"] * v
        if self.ms is not None:
            v = self.ms(pred, target)
            parts["msssim"] = v.detach()
            total = total + self.w["msssim"] * v
        if self.fft is not None:
            v = self.fft(pred, target)
            parts["fft"] = v.detach()
            total = total + self.w["fft"] * v
        if self.grad is not None:
            v = self.grad(pred, target)
            parts["gradient"] = v.detach()
            total = total + self.w["gradient"] * v
        if self.vgg is not None:
            v = self.vgg(pred, target)
            parts["vgg"] = v.detach()
            total = total + self.w["vgg"] * v
        parts["total"] = total.detach()
        return total, parts
