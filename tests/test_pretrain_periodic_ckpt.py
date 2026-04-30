"""Tests for PR #48 periodic checkpointing.

Synthetic mock-fallback runs cover:
  - every_n=2 with epochs=4 produces 2 periodic ckpts + 1 final
  - every_n > epochs produces 0 periodic (only final)
  - every_n=0 disables periodic ckpts entirely
  - All ckpts (periodic + final) have valid resume_state
  - --max-steps mid-epoch skips redundant periodic save
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _mock_argv(tmp_path, *, epochs, every_n_epochs, max_steps=None):
    argv = [
        "--no_wandb",
        "--steps", "2",
        "--epochs", str(epochs),
        "--every_n_epochs", str(every_n_epochs),
        "--n_cells", "50",
        "--n_genes", "100",
        "--n_peaks", "200",
        "--checkpoint_dir", str(tmp_path),
        "--seed", "0",
    ]
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    return argv


def _periodic_ckpts(checkpoint_dir):
    return sorted(Path(checkpoint_dir).glob("pretrain_encoders_epoch_*.pt"))


def _final_ckpt(checkpoint_dir):
    p = Path(checkpoint_dir) / "pretrain_encoders.pt"
    return p if p.exists() else None


def test_every_n_2_with_4_epochs_produces_2_periodic_plus_1_final(tmp_path):
    """epochs=4, every_n=2 -> save at end of epochs 2 and 4 + final = 3 total."""
    from pretrain_multiome import main
    main(_mock_argv(tmp_path, epochs=4, every_n_epochs=2))

    periodic = _periodic_ckpts(tmp_path)
    final = _final_ckpt(tmp_path)

    assert final is not None, "End-of-training save missing"
    periodic_names = [p.name for p in periodic]
    assert any("epoch_0002" in n for n in periodic_names), \
        f"Missing epoch 2 periodic save: {periodic_names}"
    assert any("epoch_0004" in n for n in periodic_names), \
        f"Missing epoch 4 periodic save: {periodic_names}"
    assert len(periodic) == 2, f"Expected 2 periodic saves, got {len(periodic)}"


def test_every_n_greater_than_epochs_no_periodic(tmp_path):
    from pretrain_multiome import main
    main(_mock_argv(tmp_path, epochs=3, every_n_epochs=10))

    assert _final_ckpt(tmp_path) is not None
    assert len(_periodic_ckpts(tmp_path)) == 0


def test_every_n_zero_disables_periodic(tmp_path):
    from pretrain_multiome import main
    main(_mock_argv(tmp_path, epochs=4, every_n_epochs=0))

    assert _final_ckpt(tmp_path) is not None
    assert len(_periodic_ckpts(tmp_path)) == 0


def test_periodic_ckpts_have_valid_resume_state(tmp_path):
    from pretrain_multiome import main
    main(_mock_argv(tmp_path, epochs=4, every_n_epochs=2))

    periodic = _periodic_ckpts(tmp_path)
    assert len(periodic) >= 1

    for p in periodic:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        assert "resume_state" in ckpt
        rs = ckpt["resume_state"]
        assert all(k in rs for k in ("optimizer", "scheduler", "epoch", "global_step"))
        assert "rna_encoder" in ckpt
        assert "atac_encoder" in ckpt


def test_max_steps_skip_redundant_periodic_save(tmp_path):
    """--max-steps mid-epoch skips periodic at termination epoch (final handles it)."""
    from pretrain_multiome import main
    main(_mock_argv(tmp_path, epochs=4, every_n_epochs=1, max_steps=3))

    periodic = _periodic_ckpts(tmp_path)
    final = _final_ckpt(tmp_path)

    assert final is not None
    # Some completed epochs may have produced periodic saves before max-steps fired;
    # what matters is no DUPLICATE save at the in-progress epoch where max-steps hit.
    # With --steps 2 and --max-steps 3, max-steps fires partway through epoch 1
    # (after epoch 0 has produced its periodic save). epoch 1 is incomplete and
    # MUST NOT have a periodic save.
    assert len(periodic) <= 1, \
        f"max_steps should skip periodic at termination, got {len(periodic)}"
