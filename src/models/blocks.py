"""NAFNet building blocks (Chen et al., ECCV 2022).

The design point: no activation functions at all. SimpleGate replaces
GELU/ReLU with an element-wise product of two channel halves, and SCA is
channel attention reduced to global-average-pool -> 1x1 conv -> multiply.
LayerNorm (not BatchNorm) is used throughout, which is a large part of why
NAFNet transfers well out-of-distribution: it never couples to training-batch
statistics.
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply. Replaces every nonlinearity."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2,
                 drop_out: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1, bias=True)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw, bias=True)
        self.conv3 = nn.Conv2d(dw // 2, c, 1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1, bias=True),
        )
        self.sg = SimpleGate()

        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.drop1 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma


def icnr_(tensor: torch.Tensor, scale: int = 2) -> None:
    """ICNR init for the sub-pixel conv (Aitken et al. 2017).

    Initialises the PixelShuffle conv so all `scale**2` output channels start
    identical, which removes the checkerboard artefact PixelShuffle otherwise
    produces at initialisation.
    """
    out_c, in_c, kh, kw = tensor.shape
    sub = out_c // (scale ** 2)
    k = torch.zeros([sub, in_c, kh, kw], device=tensor.device)
    nn.init.kaiming_normal_(k)
    k = k.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        tensor.copy_(k)
