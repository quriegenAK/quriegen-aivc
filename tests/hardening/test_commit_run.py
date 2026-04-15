"""Tests 18-20: commit_run."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from scripts import commit_run as cr
from scripts.build_signature import RunSignature


def _sig(status="SUCCESS"):
    return RunSignature.model_validate({
        "schema_version": "aivc/run_signature/v1",
        "run_id": "abc123",
        "timestamp_utc": "2026-04-10T00:00:00+00:00",
        "git": {"commit": "abc1234", "branch": "main", "dirty": False, "diff_sha": None},
        "config": {"path": "configs/c.yaml", "sha256": "ab" * 32, "snapshot_path": "x"},
        "dataset": {"name": "kang2018", "version": "v1", "sha256": "ab" * 32, "dvc_path": None},
        "environment": {"python": "3.11", "cuda": None, "torch": "2.2", "pip_freeze_sha": "x"},
        "model": {"checkpoint_path": "x.pt", "checkpoint_sha256": "y"},
        "metrics": {"pearson_r": 0.880, "delta_nonzero_pct": 5.0, "ctrl_memorisation_score": 0.1},
        "context_snapshot_sha": "ab" * 32,
        "status": status,
    })


# Test 18: commit_run skips on non-SUCCESS
def test_commit_run_skips_failure():
    sig = _sig(status="FAILURE")
    sha = cr.commit_run(sig)
    assert sha is None


# Test 19: commit_run on SUCCESS calls git with correct author identity
def test_commit_run_author_identity(monkeypatch):
    sig = _sig(status="SUCCESS")
    calls = []

    def fake_git(*args, check=True, capture=True):
        calls.append(args)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        # Simulate "something staged" so we don't bail early
        if args and args[0] == "diff" and "--cached" in args and "--name-only" in args:
            result.stdout = "experiments/abc123/signature.json\n"
        if args and args[0] == "rev-parse" and "HEAD" in args:
            result.stdout = "deadbeef\n"
        return result

    monkeypatch.setattr(cr, "_git", fake_git)
    monkeypatch.setattr(cr, "_staged_diff_only_timestamp_bump", lambda: False)
    sha = cr.commit_run(sig)
    # Look for the commit invocation
    saw_commit = any(
        ("commit" in a) and (
            any("user.email=a.khan@quriegen.com" in str(x) for x in a)
        )
        for a in calls
    )
    assert saw_commit, f"calls: {calls}"
    assert sha == "deadbeef"


# Test 20: timestamp-only bump returns None
def test_commit_run_timestamp_only_skip(monkeypatch):
    sig = _sig(status="SUCCESS")

    def fake_git(*args, check=True, capture=True):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(cr, "_git", fake_git)
    monkeypatch.setattr(cr, "_staged_diff_only_timestamp_bump", lambda: True)
    sha = cr.commit_run(sig)
    assert sha is None
