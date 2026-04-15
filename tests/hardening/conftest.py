"""Shared fixtures for hardening tests."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


class FakeKang:
    def __init__(self, regression_guard_passed=True, pearson_r=0.88, delta_nonzero_pct=5.0):
        self.regression_guard_passed = regression_guard_passed
        self.pearson_r = pearson_r
        self.delta_nonzero_pct = delta_nonzero_pct
        self.passed = regression_guard_passed


class FakeSuite:
    def __init__(self, kang=None, norman=None):
        self.kang = kang or FakeKang()
        self.norman = norman
        self.overall_passed = self.kang.regression_guard_passed
        self.halt_reason = None

    def model_dump_json(self, **_):
        return json.dumps({
            "kang": {
                "regression_guard_passed": self.kang.regression_guard_passed,
                "pearson_r": self.kang.pearson_r,
                "delta_nonzero_pct": self.kang.delta_nonzero_pct,
            },
            "norman": None,
            "overall_passed": self.overall_passed,
            "halt_reason": None,
        })


class FakeMeta:
    def __init__(self, run_id="test_run", pearson_r=0.88, delta_nonzero_pct=5.0,
                 ctrl_memorisation_score=0.1, dataset="kang2018_pbmc",
                 checkpoint_path="models/v1.1/model_v11_best.pt"):
        self.run_id = run_id
        self.pearson_r = pearson_r
        self.delta_nonzero_pct = delta_nonzero_pct
        self.ctrl_memorisation_score = ctrl_memorisation_score
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path


@pytest.fixture
def fake_meta():
    return FakeMeta()


@pytest.fixture
def fake_suite():
    return FakeSuite()


@pytest.fixture
def isolated_experiments(tmp_path, monkeypatch):
    """Redirect experiments/ + index.jsonl to tmp_path for safety."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "best").mkdir()
    idx = exp_dir / "index.jsonl"
    idx.touch()

    import scripts.build_signature as bs
    import scripts.validate_run as vr
    import scripts.validate_proposal as vp

    monkeypatch.setattr(bs, "EXPERIMENTS_DIR", exp_dir)
    monkeypatch.setattr(bs, "INDEX_PATH", idx)
    monkeypatch.setattr(vr, "EXPERIMENTS_DIR", exp_dir)
    monkeypatch.setattr(vp, "INDEX_PATH", idx)
    return exp_dir
