"""Tests 21-24: validate_proposal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import validate_proposal as vp


def _good_proposal():
    return {
        "hypothesis": "Try smaller w_scale.",
        "config_diff": {"neumann_k": 4, "lambda_l1": 0.02},
        "expected_metric_delta": {"pearson_r": 0.005, "delta_nonzero_pct": 0.5},
    }


# Test 21: schema check rejects empty
def test_proposal_schema_missing(isolated_experiments):
    res = vp.validate_proposal({"hypothesis": "x", "config_diff": {}, "expected_metric_delta": {}})
    assert res.approved is False
    assert "schema_missing" in (res.rejection_reason or "")


# Test 22: duplicate detection against index
def test_proposal_duplicate(isolated_experiments):
    p = _good_proposal()
    sha = vp._config_diff_sha(p["config_diff"])
    idx = isolated_experiments / "index.jsonl"
    idx.write_text(json.dumps({
        "run_id": "prev_run",
        "status": "SUCCESS",
        "config_diff_sha": sha,
    }) + "\n", encoding="utf-8")
    res = vp.validate_proposal(p)
    assert res.approved is False
    assert res.rejection_reason == "duplicate"
    assert res.duplicate_of == "prev_run"


# Test 23: boundary violations
def test_proposal_boundary_violations(isolated_experiments):
    p = _good_proposal()
    p["config_diff"] = {"neumann_k": 100, "w_scale": 5.0}  # both out of bounds
    res = vp.validate_proposal(p)
    assert res.approved is False
    assert res.rejection_reason == "boundary_violations"
    assert len(res.boundary_violations) >= 1


# Test 24: regression loop detection
def test_proposal_regression_loop(isolated_experiments):
    p = _good_proposal()
    idx = isolated_experiments / "index.jsonl"
    lines = []
    for i in range(3):
        lines.append(json.dumps({
            "run_id": f"r{i}",
            "status": "REGRESSION",
            "config_diff_sha": f"diffsha_{i}",
        }))
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = vp.validate_proposal(p)
    assert res.approved is False
    assert res.rejection_reason == "regression_loop_detected"
