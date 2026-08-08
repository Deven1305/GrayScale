"""No train/val leakage. A random split would put overlapping crops of the
same photograph on both sides and make validation meaningless."""
import pytest

from src.data.splits import (assert_no_leakage, source_id, split_by_source,
                             summarise)


def test_source_id_blocks_of_four():
    assert source_id(0) == source_id(1) == source_id(2) == source_id(3) == 0
    assert source_id(4) == 1
    assert source_id(3199) == 799


def test_split_has_no_shared_source():
    tr, va = split_by_source(3200, val_frac=0.1, seed=1337)
    assert_no_leakage(tr, va)                       # raises on failure
    s = summarise(tr, va)
    assert s["n_train"] + s["n_val"] == 3200
    assert s["n_train_sources"] + s["n_val_sources"] == 800


def test_split_keeps_crops_together():
    """All 4 crops of a source must land on the same side."""
    tr, va = split_by_source(3200, val_frac=0.1, seed=7)
    tr_set = set(tr)
    for s in range(800):
        members = [s * 4 + k for k in range(4)]
        inside = [m in tr_set for m in members]
        assert all(inside) or not any(inside), f"source {s} was split"


def test_leakage_detector_actually_fires():
    """A deliberately random split must be rejected."""
    with pytest.raises(AssertionError):
        assert_no_leakage([0, 1, 2], [3, 4])        # 0..3 share source 0
