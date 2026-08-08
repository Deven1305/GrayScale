"""Train/val splitting — by SOURCE, never randomly.

Phase 0 established that the 3200 samples are 800 source photographs x 4
overlapping crops, laid out in contiguous blocks:

    source_id = sample_index // 4

Similarity between samples is high inside a block and drops to exactly zero
at the boundary between 4k+3 and 4k+4 (three independent descriptors agree).
A random per-sample split therefore puts overlapping crops of the SAME
photograph on both sides, and the validation score measures memorisation
instead of generalisation.

Every split function here asserts disjointness and fails loudly.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

CROPS_PER_SOURCE = 4


def source_id(sample_index: int) -> int:
    return sample_index // CROPS_PER_SOURCE


def assert_no_leakage(train_idx, val_idx) -> None:
    ts = {source_id(i) for i in train_idx}
    vs = {source_id(i) for i in val_idx}
    overlap = ts & vs
    if overlap:
        raise AssertionError(
            f"SPLIT LEAKAGE: {len(overlap)} source(s) appear in both splits, "
            f"e.g. {sorted(overlap)[:8]}. Split by source_id, never randomly.")


def split_by_source(n_samples: int = 3200, val_frac: float = 0.1,
                    seed: int = 1337) -> Tuple[List[int], List[int]]:
    """Hold out whole sources (all 4 crops together)."""
    n_src = n_samples // CROPS_PER_SOURCE
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_src)
    n_val = max(1, int(round(n_src * val_frac)))
    val_src = set(perm[:n_val].tolist())

    train_idx, val_idx = [], []
    for i in range(n_samples):
        (val_idx if source_id(i) in val_src else train_idx).append(i)
    assert_no_leakage(train_idx, val_idx)
    return train_idx, val_idx


def split_tonal_extremes(cluster_json: Path, n_samples: int = 3200,
                         hold_out=(1, 7)) -> Tuple[List[int], List[int]]:
    """Proxy-OOD split: hold out the darkest and brightest source clusters.

    Phase 0 §9 found the corpus has NO visual-origin families — clustering
    yields tonal strata. A leave-one-origin-out split is therefore impossible;
    holding out the tonal extremes is the honest substitute.
    """
    labels = json.load(open(cluster_json))["labels"]
    hold = set(hold_out)
    train_idx, val_idx = [], []
    for i in range(n_samples):
        s = source_id(i)
        if s < len(labels) and labels[s] in hold:
            val_idx.append(i)
        else:
            train_idx.append(i)
    assert_no_leakage(train_idx, val_idx)
    return train_idx, val_idx


def summarise(train_idx, val_idx) -> Dict:
    return {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_train_sources": len({source_id(i) for i in train_idx}),
        "n_val_sources": len({source_id(i) for i in val_idx}),
        "leakage": False,
    }


if __name__ == "__main__":
    tr, va = split_by_source()
    print(json.dumps(summarise(tr, va), indent=2))
    print("no leakage — assertion passed")
