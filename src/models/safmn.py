"""SAFMN — the fast point on the Pareto curve.

Spatially-Adaptive Feature Modulation (Sun et al., ICCV 2023), ~0.24M params.
Same SR-head pattern as NAFNetSR: process at LR, one PixelShuffle at the end,
residual on a bicubic anchor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, icnr_


class CCM(nn.Module):
    """Channel-mixing feed-forward."""

    def __init__(self, dim, growth=2.0):
        super().__init__()
        h = int(dim * growth)
        self.net = nn.Sequential(
            nn.Conv2d(dim, h, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SAFM(nn.Module):
    """Multi-scale feature modulation: pool to several scales, process, fuse."""

    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk = dim // n_levels
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk, chunk, 3, padding=1, groups=chunk)
             for _ in range(n_levels)])
        self.aggr = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p = 2 ** i
                s = F.adaptive_max_pool2d(xc[i], (max(h // p, 1), max(w // p, 1)))
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        return self.act(self.aggr(torch.cat(out, dim=1))) * x


class AttBlock(nn.Module):
    def __init__(self, dim, growth=2.0):
        super().__init__()
        self.n1 = LayerNorm2d(dim)
        self.n2 = LayerNorm2d(dim)
        self.safm = SAFM(dim)
        self.ccm = CCM(dim, growth)

    def forward(self, x):
        x = x + self.safm(self.n1(x))
        return x + self.ccm(self.n2(x))


class SAFMNSR(nn.Module):
    def __init__(self, in_ch=1, dim=36, n_blocks=8, scale=2,
                 use_log_channel=True, **_):
        super().__init__()
        self.scale = scale
        self.use_log_channel = use_log_channel
        stem_in = in_ch * (2 if use_log_channel else 1)
        self.to_feat = nn.Conv2d(stem_in, dim, 3, padding=1)
        self.feats = nn.Sequential(*[AttBlock(dim) for _ in range(n_blocks)])
        self.sr_head = nn.Conv2d(dim, in_ch * scale * scale, 3, padding=1)
        icnr_(self.sr_head.weight, scale)
        nn.init.zeros_(self.sr_head.bias)
        self.shuffle = nn.PixelShuffle(scale)
        self.padder_size = 1

    def _stem(self, x):
        if not self.use_log_channel:
            return x
        return torch.cat([x, torch.log(x.clamp_min(1e-3))], dim=1)

    def forward(self, inp):
        anchor = F.interpolate(inp, scale_factor=self.scale, mode="bicubic",
                               align_corners=False)
        x = self.to_feat(self._stem(inp))
        x = self.feats(x)
        return anchor + self.shuffle(self.sr_head(x))


def safmn(**kw):
    return SAFMNSR(dim=36, n_blocks=8, **kw)
