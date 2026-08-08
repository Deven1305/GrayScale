"""name -> model factory. Keeps train/eval/inference free of if-else chains."""
from .nafnet import nafnet_w16, nafnet_w32, nafnet_w48
from .safmn import safmn
from .unrolled import log_unrolled_k4, log_unrolled_k6

REGISTRY = {
    "nafnet_w48": nafnet_w48,          # scaled primary
    "nafnet_w32": nafnet_w32,
    "nafnet_w16": nafnet_w16,
    "safmn": safmn,                    # fast Pareto point
    "log_unrolled_k4": log_unrolled_k4,   # innovation track
    "log_unrolled_k6": log_unrolled_k6,
}


def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. options: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
