"""Tests for PR #44 (logical) resume mechanism — approximate granularity.

Exercises the three module-level helpers in scripts/pretrain_multiome.py:
  - _build_resume_state
  - _validate_resume_config
  - _apply_resume_state
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Import resume helpers from the script. The script's top-level imports
# trigger torch + aivc.* loads but do not call into main(), so this is safe.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pretrain_multiome import (  # noqa: E402
    _RESUME_CRITICAL_FIELDS,
    _apply_resume_state,
    _build_resume_state,
    _validate_resume_config,
)


def _make_dummy_optim_scheduler():
    """Build a tiny optimizer + scheduler for state_dict round-trip tests."""
    p = torch.nn.Linear(4, 2)
    optim = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    return p, optim, sched


def test_build_resume_state_has_required_keys():
    p, optim, sched = _make_dummy_optim_scheduler()
    rs = _build_resume_state(optim, sched, epoch=5, global_step=125)
    assert set(rs.keys()) == {"optimizer", "scheduler", "epoch",
                              "global_step", "saved_at_utc"}
    assert rs["epoch"] == 5
    assert rs["global_step"] == 125
    assert isinstance(rs["optimizer"], dict)
    assert isinstance(rs["scheduler"], dict)


def test_build_resume_state_optimizer_is_loadable():
    """Round-trip: build, then load_state_dict should not raise + restore step counter."""
    p, optim, sched = _make_dummy_optim_scheduler()
    # Take a step so optimizer state is non-trivial
    p.weight.data.fill_(1.0)
    loss = p(torch.randn(3, 4)).sum()
    loss.backward()
    optim.step()
    sched.step()

    rs = _build_resume_state(optim, sched, epoch=2, global_step=50)

    # Build a fresh optim + sched and load
    p2, optim2, sched2 = _make_dummy_optim_scheduler()
    optim2.load_state_dict(rs["optimizer"])
    sched2.load_state_dict(rs["scheduler"])

    # Verify scheduler step counter restored
    assert sched2._step_count == sched._step_count


def test_validate_resume_config_match_passes():
    saved = {f: i for i, f in enumerate(_RESUME_CRITICAL_FIELDS)}
    _validate_resume_config(saved, dict(saved))  # exact match — no raise


def test_validate_resume_config_arm_mismatch_raises():
    saved = {"arm": "lll", "n_genes": 36601, "n_peaks": 323500,
             "n_proteins": 210, "rna_latent": 128, "atac_latent": 64,
             "proj_dim": 128, "protein_latent": 64,
             "n_lysis_categories": 2, "lysis_cov_dim": 8}
    current = dict(saved)
    current["arm"] = "dig"
    with pytest.raises(ValueError, match=r"arm.*saved='lll'.*current='dig'"):
        _validate_resume_config(saved, current)


def test_validate_resume_config_n_lysis_categories_mismatch_raises():
    saved = {f: 0 for f in _RESUME_CRITICAL_FIELDS}
    saved["n_lysis_categories"] = 2
    current = dict(saved)
    current["n_lysis_categories"] = 0
    with pytest.raises(ValueError, match=r"n_lysis_categories"):
        _validate_resume_config(saved, current)


def test_validate_resume_config_lists_all_mismatches():
    """Multi-mismatch error surfaces all bad fields, not just the first."""
    saved = {f: 0 for f in _RESUME_CRITICAL_FIELDS}
    saved["arm"] = "lll"
    saved["n_genes"] = 36601
    saved["n_peaks"] = 323500
    current = dict(saved)
    current["arm"] = "dig"
    current["n_peaks"] = 99999
    with pytest.raises(ValueError) as exc_info:
        _validate_resume_config(saved, current)
    msg = str(exc_info.value)
    assert "arm" in msg and "n_peaks" in msg


def test_validate_resume_config_none_equals_none():
    """Genuinely-absent field on both sides (None==None) is not a mismatch.

    Pre-PR-#43 single-arm ckpts have no n_lysis_categories key; current
    single-arm runs use n_lysis_categories=0. Both should resolve cleanly.
    """
    saved = {f: 0 for f in _RESUME_CRITICAL_FIELDS}
    # Drop one field on both sides — emulates a pre-PR ckpt
    del saved["n_lysis_categories"]
    current = dict(saved)
    _validate_resume_config(saved, current)  # no raise


def test_apply_resume_state_pre_pr44_ckpt_returns_zero():
    """Ckpt without 'resume_state' key (pre-PR-#44) → start fresh."""
    p, optim, sched = _make_dummy_optim_scheduler()
    ckpt = {"schema_version": 1, "rna_encoder": {}, "config": {}}
    start_epoch, gstep = _apply_resume_state(ckpt, optim, sched)
    assert start_epoch == 0
    assert gstep == 0


def test_apply_resume_state_advances_epoch():
    """Saved epoch=5 → resume starts at epoch=6 (next un-run epoch)."""
    p, optim, sched = _make_dummy_optim_scheduler()
    rs = _build_resume_state(optim, sched, epoch=5, global_step=125)

    p2, optim2, sched2 = _make_dummy_optim_scheduler()
    ckpt = {"schema_version": 1, "config": {}, "resume_state": rs}
    start_epoch, gstep = _apply_resume_state(ckpt, optim2, sched2)
    assert start_epoch == 6
    assert gstep == 125
