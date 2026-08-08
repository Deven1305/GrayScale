"""End-to-end smoke test of the SCORED script.

Checks the contract KLA relies on: dir -> dir, output is exactly 2x the input,
filenames are preserved, values are in [0,1], and one malformed file does not
crash the run.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _make_weights(path: Path):
    sys.path.insert(0, str(ROOT))
    from src.models.registry import build_model
    m = build_model("nafnet_w16", in_ch=1, scale=2, use_log_channel=True)
    torch.save({"state_dict": m.state_dict(), "arch": "nafnet_w16",
                "in_ch": 1, "scale": 2, "use_log_channel": True}, path)


import pytest


@pytest.mark.parametrize("workers", [0, 2])
def test_inference_end_to_end(tmp_path, workers):
    """Runs with workers=0 AND workers=2.

    The multi-worker case is not redundant: DataLoader workers spawn on
    Windows/macOS and pickle the dataset class by reference, so a class defined
    inside main() fails only when num_workers > 0. That bug shipped once and
    was caught by a real run, not by the workers=0 test.
    """
    inp = tmp_path / f"in{workers}"
    out = tmp_path / f"out{workers}"
    inp.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        np.save(inp / f"{i:06d}.npy",
                (rng.random((128, 128)) * 1.2 - 0.1).astype(np.float32))
    # a deliberately malformed file: the run must survive it
    (inp / "broken.npy").write_bytes(b"not a numpy file")

    w = tmp_path / "w.pt"
    _make_weights(w)

    r = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"),
         "--input_dir", str(inp), "--output_dir", str(out),
         "--weights", str(w), "--batch_size", "2",
         "--num_workers", str(workers), "--device", "cpu"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"inference.py failed:\n{r.stdout}\n{r.stderr}"

    for i in range(3):
        p = out / f"{i:06d}.npy"
        assert p.exists(), f"missing output {p.name}"
        a = np.load(p)
        assert a.shape == (256, 256), f"expected 2x, got {a.shape}"
        assert a.dtype == np.float32
        assert a.min() >= 0.0 and a.max() <= 1.0, "output must be clamped"


def test_inference_handles_both_required_resolutions(tmp_path):
    """The problem statement names BOTH scales:
        512x512 GT -> 256x256 degraded   and   256x256 GT -> 128x128 degraded

    Only the 128->256 case exists in the released data, so the 256->512 path
    would otherwise ship untested. It is a stated requirement, not an extra.
    """
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    rng = np.random.default_rng(0)
    np.save(inp / "small.npy", rng.random((128, 128)).astype(np.float32))
    np.save(inp / "large.npy", rng.random((256, 256)).astype(np.float32))

    w = tmp_path / "w.pt"
    _make_weights(w)

    r = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"),
         "--input_dir", str(inp), "--output_dir", str(out),
         "--weights", str(w), "--num_workers", "0", "--device", "cpu"],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"failed:\n{r.stdout}\n{r.stderr}"

    assert np.load(out / "small.npy").shape == (256, 256)
    assert np.load(out / "large.npy").shape == (512, 512)
    # mixed sizes must be bucketed, not batched together
    assert "128x128" in r.stdout and "256x256" in r.stdout


def test_inference_has_no_absolute_paths():
    src = (ROOT / "inference.py").read_text(encoding="utf-8")
    for bad in ("D:\\", "C:\\", "/home/", "/Users/"):
        assert bad not in src, f"hardcoded absolute path {bad!r} in inference.py"


def _code_lines(path):
    """Source with comments and docstrings stripped, so prose that merely
    MENTIONS a banned name does not trip the check."""
    import io
    import tokenize
    out = []
    with open(path, "rb") as f:
        toks = list(tokenize.tokenize(f.readline))
    prev = None
    for t in toks:
        if t.type == tokenize.COMMENT:
            continue
        if t.type == tokenize.STRING and (prev is None or prev.type in (
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
                tokenize.NL, tokenize.ENCODING)):
            continue                                  # docstring
        out.append(t.string)
        if t.type not in (tokenize.NL, tokenize.NEWLINE):
            prev = t
    return " ".join(out)


def test_inference_imports_are_minimal():
    """Startup time is scored. These imports must never appear."""
    src = (ROOT / "inference.py").read_text(encoding="utf-8")
    head = "\n".join(l for l in src.splitlines()
                     if l.startswith("import ") or l.startswith("from "))
    for banned in ("torchvision", "matplotlib", "pandas", "timm",
                   "transformers", "sklearn", "scipy", "skimage"):
        assert banned not in head, f"inference.py imports {banned} at top level"


def test_inference_does_not_call_torch_compile():
    """A 30-120 s compile to save a few seconds is a net loss on scored time."""
    code = _code_lines(ROOT / "inference.py")
    assert "torch.compile" not in code.replace(" ", ""), \
        "inference.py calls torch.compile"
