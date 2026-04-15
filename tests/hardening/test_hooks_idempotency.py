"""Test 25: post_train idempotency — double-call writes one index.jsonl entry."""
from __future__ import annotations

import json

import pytest

from scripts import hooks
from tests.hardening.conftest import FakeMeta, FakeSuite


def test_post_train_idempotent(isolated_experiments, monkeypatch):
    # Override hooks' EXPERIMENTS_DIR / INDEX_PATH to match
    monkeypatch.setattr(hooks, "EXPERIMENTS_DIR", isolated_experiments)
    monkeypatch.setattr(hooks, "INDEX_PATH", isolated_experiments / "index.jsonl")
    # Disable side effects we don't want
    monkeypatch.setattr(hooks, "sync_context", lambda **kw: True)
    monkeypatch.setattr("scripts.commit_run.commit_run", lambda sig, branch="experiments/auto": None)

    meta = FakeMeta(run_id="idem_run", pearson_r=0.880, delta_nonzero_pct=5.0)
    suite = FakeSuite()

    hooks.post_train(meta, suite)
    hooks.post_train(meta, suite)

    idx = isolated_experiments / "index.jsonl"
    lines = [l for l in idx.read_text().splitlines() if l.strip()]
    matching = [l for l in lines if json.loads(l).get("run_id") == "idem_run"]
    assert len(matching) == 1, f"Expected 1 idem_run entry, got {len(matching)}"
