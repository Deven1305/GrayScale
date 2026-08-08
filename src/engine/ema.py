"""Exponential moving average of weights. Reliably +0.1-0.2 dB for free."""
from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # warm up the decay so early steps are not dominated by the init
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()
