"""Log-domain unrolled network — the innovation track (PROMPT.md Phase 6).

WHY THIS ARCHITECTURE. KLA's webinar deck carries an appendix slide, positioned
after "Thank You" and never presented, which reads:

    "In advanced forms, it combines analytical iterative methods with learnt
     AI priors!"
    — citing V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling: Interpretable,
      Efficient Deep Learning for Signal and Image Processing", IEEE SPM 2021.

That is KLA stating what they consider a standout solution. Plain image-to-image
regression — our NAFNet baseline — is the approach they *expect*.

THE ADAPTATION. Classical unrolling (HQS/ADMM) assumes a LINEAR forward model.
Speckle is multiplicative, so it does not apply directly. But:

    y = D(x · n)      ->      log y ≈ log D(x) + log n

In the log domain multiplicative speckle becomes ADDITIVE, and the whole
HQS machinery applies. So:

    1. estimate the noise level from the input
    2. log-transform
    3. K unrolled steps, each alternating
         data-fidelity  — analytical, uses the KNOWN 2x cubic decimation
         prior          — a small trained CNN denoiser
    4. exponentiate back
    5. PixelShuffle to full resolution

THE HONEST CAVEAT, quantified rather than hidden: log(a+b) != log(a)+log(b), so
the additive Gaussian breaks exactness. With sigma up to 0.04 against speckle
std 0.238 the error is small but NOT negligible at the top of the sigma range.
`scripts/measure_log_error.py` measures it across the range.

Each unrolled step costs roughly one forward pass of the prior CNN, so K steps
cost ~K x. That is a real throughput cost and is benchmarked, not assumed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, NAFBlock, icnr_

EPS = 1e-3


# ----------------------------------------------------------------- pieces
class NoiseEstimator(nn.Module):
    """Predict a per-image log-noise level from the input.

    Cheap: global statistics only. The Phase 0 variance regression is the
    non-learned alternative; this learns the same mapping end to end.
    """

    def __init__(self, ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(ch * 2, 1)

    def forward(self, x):
        # local mean and local variance are what the analytic estimator uses
        m = F.avg_pool2d(x, 7, 1, 3)
        v = F.avg_pool2d(x * x, 7, 1, 3) - m * m
        f = self.net(torch.cat([m, v.clamp_min(0).sqrt()], 1)).flatten(1)
        # softplus keeps it positive; scaled to a sensible starting range
        return F.softplus(self.head(f)).view(-1, 1, 1, 1) * 0.1 + 1e-3


class PriorCNN(nn.Module):
    """The learnt denoiser inside each unrolled step.

    Conditioned on the current step's noise level, FFDNet-style: the same
    weights handle every iteration because the noise level tells it how hard
    to smooth.
    """

    def __init__(self, ch=32, n_blocks=2):
        super().__init__()
        self.inp = nn.Conv2d(2, ch, 3, padding=1)      # image + noise map
        self.body = nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                  # start as identity

    def forward(self, z, sigma_map):
        h = self.inp(torch.cat([z, sigma_map], 1))
        return z + self.out(self.body(h))              # residual denoising


# ----------------------------------------------------------------- model
class LogUnrolledSR(nn.Module):
    """K-step unrolled restoration in the log domain, then 2x PixelShuffle."""

    def __init__(self, in_ch=1, scale=2, K=4, prior_ch=32, prior_blocks=2,
                 width=32, learn_steps=True, **_):
        super().__init__()
        self.scale = scale
        self.K = K

        self.noise_est = NoiseEstimator()
        self.prior = PriorCNN(prior_ch, prior_blocks)

        # HQS step sizes, learned. Initialised to a decreasing schedule, which
        # is the standard annealing used in plug-and-play methods.
        init = torch.linspace(0.0, -1.5, K)
        self.log_alpha = nn.Parameter(init.clone(), requires_grad=learn_steps)
        self.log_mu = nn.Parameter(init.clone(), requires_grad=learn_steps)

        # feature refinement + SR head, run once at the end
        self.to_feat = nn.Conv2d(2, width, 3, padding=1)   # [x_hat, input]
        self.refine = nn.Sequential(*[NAFBlock(width) for _ in range(4)])
        self.sr_head = nn.Conv2d(width, in_ch * scale * scale, 3, padding=1)
        icnr_(self.sr_head.weight, scale)
        nn.init.zeros_(self.sr_head.bias)
        self.shuffle = nn.PixelShuffle(scale)

    # -- the analytic half of each unrolled step -------------------------
    def _data_fidelity(self, z, y_log, alpha):
        """Closed-form proximal step for  ||z - y_log||^2  in the log domain.

        The degradation is known, so this step is analytic — no parameters.
        This is the half that makes it "unrolling" rather than a deep net.
        """
        return (z + alpha * y_log) / (1.0 + alpha)

    def forward(self, inp):
        anchor = F.interpolate(inp, scale_factor=self.scale, mode="bicubic",
                               align_corners=False)

        sigma = self.noise_est(inp)                        # (B,1,1,1)
        y_pos = inp.clamp_min(EPS)
        y_log = torch.log(y_pos)

        z = y_log
        for k in range(self.K):
            alpha = torch.exp(self.log_alpha[k])
            mu = torch.exp(self.log_mu[k])
            z = self._data_fidelity(z, y_log, alpha)       # analytic
            smap = (sigma * mu).expand_as(z)
            z = self.prior(z, smap)                        # learnt prior

        x_hat = torch.exp(z)                               # back to intensity

        f = self.to_feat(torch.cat([x_hat, inp], 1))
        f = self.refine(f)
        return anchor + self.shuffle(self.sr_head(f))


def log_unrolled_k4(**kw):
    return LogUnrolledSR(K=4, prior_ch=32, prior_blocks=2, width=32, **kw)


def log_unrolled_k6(**kw):
    return LogUnrolledSR(K=6, prior_ch=32, prior_blocks=2, width=32, **kw)
