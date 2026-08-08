"""Repository integrity — would a fresh clone actually work?

WHY THIS FILE EXISTS. `.gitignore` contained an unanchored `data/` entry, which
git matches against a directory of that name at ANY depth. It silently excluded
the entire `src/data/` package — degradation.py, splits.py, dataset.py — from
the repository. `train.py` and two test files would have failed immediately on
a fresh clone, and the Phase 0 degradation replica, the centrepiece of the whole
submission, was simply absent.

The existing fresh-clone check missed it because `inference.py` is deliberately
standalone and never imports `src/`.

These tests assert that every file the code imports is actually tracked by git.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True)
    return {line.strip().replace("\\", "/") for line in out.stdout.splitlines()
            if line.strip()}


def _is_git_root():
    """True only when ROOT is itself the top of a git work tree.

    The submission folder is assembled by copying files, so before `git init`
    it has no index at all — and if it sits inside another repo, `git ls-files`
    answers about the PARENT. Either way these checks are meaningless there, so
    they skip rather than fail.
    """
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    top = Path(r.stdout.strip()).resolve()
    return top == ROOT.resolve()


needs_git = pytest.mark.skipif(
    not _is_git_root(),
    reason="not the root of a git work tree (e.g. an assembled submission "
           "folder before git init); repository-integrity checks do not apply")


@needs_git
def test_every_source_file_is_tracked():
    """Any .py under src/ that exists on disk must be in the repository."""
    files = tracked()
    missing = []
    for p in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel not in files:
            missing.append(rel)
    assert not missing, (
        "these source files exist on disk but are NOT in the repository "
        f"(check .gitignore anchoring): {missing}")


@needs_git
def test_entrypoints_and_configs_tracked():
    files = tracked()
    required = ["train.py", "evaluate.py", "inference.py",
                "requirements.txt", "requirements-inference.txt",
                "configs/nafnet_w32.yaml", "src/data/degradation.py",
                "src/data/splits.py", "src/data/dataset.py"]
    missing = [r for r in required if r not in files]
    assert not missing, f"required files not tracked: {missing}"


@needs_git
def test_kla_data_and_brief_are_NOT_tracked():
    """The flip side: KLA's dataset and the confidential-marked deck must
    never be published."""
    files = tracked()
    leaked = [f for f in files
              if f.startswith("data/train/") or f.startswith("data/Test_")
              or f.startswith("brief/")]
    assert not leaked, f"KLA-supplied material is tracked: {leaked[:5]}"


@pytest.mark.parametrize("mod", [
    "src.data.degradation", "src.data.splits", "src.data.dataset",
    "src.data.patterns", "src.models.registry", "src.losses.composite",
    "src.metrics.full_reference", "src.engine.trainer", "src.utils.seed",
])
def test_module_imports(mod):
    """Every module the training path needs must import cleanly."""
    sys.path.insert(0, str(ROOT))
    __import__(mod)


def test_all_registered_models_build_and_run():
    import torch
    sys.path.insert(0, str(ROOT))
    from src.models.registry import REGISTRY, build_model
    x = torch.randn(1, 1, 32, 32)
    for name in REGISTRY:
        m = build_model(name).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, 64, 64), f"{name} gave {tuple(y.shape)}"
