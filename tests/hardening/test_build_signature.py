"""Tests 1-5: build_signature."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_signature import (
    GitInfo,
    RunSignature,
    build_signature,
    get_best_signature,
    load_signature,
    save_signature,
)
from tests.hardening.conftest import FakeMeta, FakeSuite


# Test 1: build_signature returns a RunSignature with required fields
def test_build_signature_basic(fake_meta, fake_suite):
    sig = build_signature(fake_meta, fake_suite)
    assert isinstance(sig, RunSignature)
    assert sig.run_id == fake_meta.run_id
    assert sig.metrics.pearson_r == fake_meta.pearson_r
    assert sig.environment.python


# Test 2: save_signature writes signature.json + appends index.jsonl atomically
def test_save_signature_atomic(isolated_experiments, fake_meta, fake_suite):
    sig = build_signature(fake_meta, fake_suite)
    p = save_signature(sig)
    assert p.exists()
    text = p.read_text()
    assert "run_id" in text

    idx = isolated_experiments / "index.jsonl"
    lines = idx.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["run_id"] == fake_meta.run_id


# Test 3: get_best_signature returns highest pearson_r SUCCESS
def test_get_best_signature(isolated_experiments, fake_meta, fake_suite):
    s1 = build_signature(FakeMeta(run_id="r1", pearson_r=0.880), fake_suite)
    s1.status = "SUCCESS"
    save_signature(s1)
    s2 = build_signature(FakeMeta(run_id="r2", pearson_r=0.890), fake_suite)
    s2.status = "SUCCESS"
    save_signature(s2)
    s3 = build_signature(FakeMeta(run_id="r3", pearson_r=0.999), fake_suite)
    s3.status = "FAILURE"
    save_signature(s3)
    best = get_best_signature()
    assert best is not None
    assert best.run_id == "r2"


# Test 4: load_signature returns None for missing
def test_load_signature_missing(isolated_experiments):
    assert load_signature("nonexistent_xyz") is None


# Test 5: missing dataset h5ad → sha256="unknown" but still returns a signature
def test_build_signature_missing_dataset(fake_suite):
    meta = FakeMeta(dataset="nonexistent_dataset_xyz")
    sig = build_signature(meta, fake_suite)
    assert sig.dataset.sha256 == "unknown"
