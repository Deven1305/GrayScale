"""NAFNet with a 2x super-resolution head.

Design decisions, all from docs/forensics_report.md and brief/02:

  * The network runs entirely at LR resolution and upsamples ONCE at the end
    via PixelShuffle(2). Upsampling first and processing at HR would cost ~4x
    the compute for no quality gain.
  * The output is a RESIDUAL on top of a bicubic anchor, so the network only
    has to learn the correction. Converges faster and transfers better OOD.
  * Fully convolutional, so 128->256 and 256->512 both work with no change.
  * Optional log channel: speckle is multiplicative, and log() makes it
    additive. Cheap, well-grounded in SAR despeckling, and few teams do it.
  * The input is NEVER clipped anywhere in this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, NAFBlock, icnr_


class NAFNetSR(nn.Module):
    def __init__(self, in_ch=1, width=32, middle_blk_num=8,
                 enc_blk_nums=(2, 2, 4), dec_blk_nums=(2, 2, 2),
                 scale=2, use_log_channel=True, drop_out=0.0,
                 input_transform="log"):
        super().__init__()
        self.scale = scale
        self.use_log_channel = use_log_channel
        self.input_transform = input_transform
        self.out_ch = in_ch
        stem_in = in_ch * (2 if use_log_channel else 1)

        self.intro = nn.Conv2d(stem_in, width, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out=drop_out)
                                                 for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, drop_out=drop_out)
                                      for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out=drop_out)
                                                 for _ in range(n)]))

        # SR head: one conv to scale^2 channels, then a single PixelShuffle
        self.sr_head = nn.Conv2d(width, self.out_ch * scale * scale, 3,
                                 padding=1, bias=True)
        icnr_(self.sr_head.weight, scale)
        nn.init.zeros_(self.sr_head.bias)
        self.shuffle = nn.PixelShuffle(scale)

        self.padder_size = 2 ** len(self.encoders)

    # ------------------------------------------------------------------
    def _stem(self, x):
        if not self.use_log_channel:
            return x
        # speckle is multiplicative -> log makes it additive.
        # clamp_min only guards the log's domain; it does not clip the signal
        # that reaches the residual path.
        if getattr(self, "input_transform", "log") == "asinh":
            return torch.cat([x, torch.asinh(x / 0.1)], dim=1)
        return torch.cat([x, torch.log(x.clamp_min(1e-3))], dim=1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, inp):
        """inp: (B, C, H, W) float, NOT clipped. Returns (B, C, 2H, 2W)."""
        anchor = F.interpolate(inp, scale_factor=self.scale, mode="bicubic",
                               align_corners=False)

        x, h, w = self.check_image_size(inp)
        x = self.intro(self._stem(x))

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.shuffle(self.sr_head(x))
        x = x[..., : h * self.scale, : w * self.scale]
        return anchor + x                       # residual on the bicubic anchor


def nafnet_w32(**kw):
    return NAFNetSR(width=32, middle_blk_num=8, enc_blk_nums=(2, 2, 4),
                    dec_blk_nums=(2, 2, 2), **kw)


def nafnet_w16(**kw):
    return NAFNetSR(width=16, middle_blk_num=4, enc_blk_nums=(1, 1, 2),
                    dec_blk_nums=(1, 1, 1), **kw)


def nafnet_w48(**kw):
    """Scaled primary. ~15 M params — sized to use the available VRAM rather
    than a quarter of it, which is where the w32 run left performance."""
    return NAFNetSR(width=48, middle_blk_num=12, enc_blk_nums=(2, 2, 4),
                    dec_blk_nums=(2, 2, 2), **kw)
