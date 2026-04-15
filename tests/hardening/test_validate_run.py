"""Tests 6-12: classify_run status matrix per spec §9 + promotion."""
from __future__ import annotations

import math

import pytest

from scripts.build_signature import RunSignature
from scripts.validate_run import RunStatus, classify_run, promote_checkpoint
from tests.hardening.conftest import FakeKang, FakeMeta, FakeSuite


def _best_sig(pearson_r=0.875):
    """Build a minimal best signature stub."""
    return RunSignature.model_validate({
        "schema_version": "aivc/run_signature/v1",
        "run_id": "best_prev",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "git": {"commit": "abc1234", "branch": "main", "dirty": False, "diff_sha": None},
        "config": {"path": "x", "sha256": "y", "snapshot_path": "z"},
        "dataset": {"name": "kang", "version": "v1", "sha256": "s", "dvc_path": None},
        "environment": {"python": "3.11", "cuda": None, "torch": "2.2", "pip_freeze_sha": "x"},
        "model": {"checkpoint_path": "x", "checkpoint_sha256": "y"},
        "metrics": {"pearson_r": pearson_r, "delta_nonzero_pct": 5.0, "ctrl_memorisation_score": 0.1},
        "context_snapshot_sha": "x",
        "status": "SUCCESS",
    })


# Test 6: SUCCESS + promote (pearson > best + margin)
def test_success_with_promotion():
    meta = FakeMeta(pearson_r=0.890, delta_nonzero_pct=5.0)
    suite = FakeSuite(kang=FakeKang(True, 0.890, 5.0))
    res = classify_run(meta, suite, _best_sig(0.875))
    assert res.status == RunStatus.SUCCESS
    assert res.promote_checkpoint is True


# Test 7: SUCCESS no promote (within margin)
def test_success_no_promotion():
    meta = FakeMeta(pearson_r=0.8755, delta_nonzero_pct=5.0)
    suite = FakeSuite(kang=FakeKang(True, 0.8755, 5.0))
    res = classify_run(meta, suite, _best_sig(0.875))
    assert res.status == RunStatus.SUCCESS
    assert res.promote_checkpoint is False


# Test 8: REGRESSION (pearson < best - tolerance)
def test_regression():
    meta = FakeMeta(pearson_r=0.875, delta_nonzero_pct=5.0)
    suite = FakeSuite(kang=FakeKang(True, 0.875, 5.0))
    res = classify_run(meta, suite, _best_sig(0.890))
    assert res.status == RunStatus.REGRESSION
    assert res.failure_reason == "pearson_regression"


# Test 9: FAILURE below floor
def test_failure_below_floor():
    meta = FakeMeta(pearson_r=0.5, delta_nonzero_pct=5.0)
    suite = FakeSuite(kang=FakeKang(True, 0.5, 5.0))
    res = classify_run(meta, suite, None)
    assert res.status == RunStatus.FAILURE
    assert res.failure_reason == "below_floor"


# Test 10: FAILURE delta collapse
def test_failure_delta_collapse():
    meta = FakeMeta(pearson_r=0.880, delta_nonzero_pct=0.5)
    suite = FakeSuite(kang=FakeKang(True, 0.880, 0.5))
    res = classify_run(meta, suite, None)
    assert res.status == RunStatus.FAILURE
    assert res.failure_reason == "delta_collapse"


# Test 11: FAILURE kang guard failed
def test_failure_kang_guard():
    meta = FakeMeta(pearson_r=0.880, delta_nonzero_pct=5.0)
    suite = FakeSuite(kang=FakeKang(False, 0.880, 5.0))
    res = classify_run(meta, suite, None)
    assert res.status == RunStatus.FAILURE
    assert res.failure_reason == "kang_guard_failed"


# Test 12: FAILURE NaN metrics + promote_checkpoint atomic symlink
def test_failure_nan_and_promote(tmp_path):
    meta = FakeMeta(pearson_r=float("nan"), delta_nonzero_pct=5.0)
    suite = FakeSuite()
    res = classify_run(meta, suite, None)
    assert res.status == RunStatus.FAILURE
    assert res.failure_reason == "nan_metrics"

    # promote_checkpoint smoke test
    target = tmp_path / "ckpt.pt"
    target.write_bytes(b"weights")
    import scripts.validate_run as vr
    orig = vr.EXPERIMENTS_DIR
    vr.EXPERIMENTS_DIR = tmp_path / "experiments"
    try:
        promote_checkpoint(str(target))
        link = vr.EXPERIMENTS_DIR / "best" / "best.pt"
        assert link.exists() or link.is_symlink()
    finally:
        vr.EXPERIMENTS_DIR = orig
