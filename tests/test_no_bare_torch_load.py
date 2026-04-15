"""CI check: ``torch.load(`` is forbidden in ``scripts/`` and
``aivc/training/`` outside of the strict loader module.

Phase 6 established ``aivc/training/ckpt_loader.py`` as the only legal
load path for pretrain checkpoints. A bare ``torch.load`` anywhere in
``scripts/`` or ``aivc/training/`` is a regression: it bypasses the
strict schema validation and re-introduces the "silent skip" failure
mode that invalidates the pretraining ablation.

Any new caller that needs checkpoint data should either:
  * use ``ckpt_loader.load_pretrained_simple_rna_encoder(...)``, or
  * use ``ckpt_loader.peek_pretrain_ckpt_config(...)`` for metadata, or
  * extend ``ckpt_loader`` with a second named function for a new
    artifact type (e.g., optimizer state) rather than doing a bare
    ``torch.load``.

Whitelist policy
----------------
A small set of pre-existing files load artifacts unrelated to pretrain
checkpoints (legacy HPC training checkpoints, zero-shot evaluation
model state). They are explicitly listed in ``ALLOWED_LEGACY_LOADERS``
below. New additions to this list require a reviewer justification in
the PR body.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories under which bare torch.load is policed.
POLICED_DIRS = ("scripts", "aivc/training")

# The only module legally permitted to call bare torch.load.
LOADER_MODULE = "aivc/training/ckpt_loader.py"

# Pre-existing non-pretrain loaders. These load different artifact
# types (legacy training checkpoints, evaluation model state) and
# predate the Phase 6 integration contract. Do not grow this list
# without reviewer justification.
ALLOWED_LEGACY_LOADERS = (
    "scripts/train_hpc.py",
    "scripts/evaluate_zero_shot.py",
)

_TORCH_LOAD_PATTERN = re.compile(r"\btorch\.load\s*\(")


def _iter_py_files(root: Path, subdir: str):
    base = root / subdir
    if not base.exists():
        return
    for dirpath, _, filenames in os.walk(base):
        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def _is_allowed(rel: str) -> bool:
    return rel == LOADER_MODULE or rel in ALLOWED_LEGACY_LOADERS


def test_no_bare_torch_load_outside_ckpt_loader():
    offenders: list[str] = []
    for subdir in POLICED_DIRS:
        for path in _iter_py_files(_REPO_ROOT, subdir):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if _is_allowed(rel):
                continue
            with open(path) as fh:
                src = fh.read()
            for lineno, line in enumerate(src.splitlines(), 1):
                if _TORCH_LOAD_PATTERN.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Bare `torch.load(` calls found outside "
        "aivc/training/ckpt_loader.py:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse "
        "aivc.training.ckpt_loader.load_pretrained_simple_rna_encoder "
        "(or peek_pretrain_ckpt_config for metadata). If you need to "
        "load a different artifact type, extend ckpt_loader with a new "
        "named function rather than re-introducing a bare torch.load."
    )
