"""Tests 13-17: sync_context, validate_context_md, freshness."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.sync_context import (
    REQUIRED_SECTIONS,
    sync_context,
    validate_context_md,
    verify_snapshot_fresh,
)


def _good_context(path: Path):
    parts = ["### Project: Test\n", "Last updated: 2026-04-10\n"]
    for s in REQUIRED_SECTIONS:
        if s == "Project":
            continue
        parts.append(f"\n### {s}\nbody\n")
    path.write_text("".join(parts), encoding="utf-8")


# Test 13: validate_context_md happy path
def test_validate_context_ok(tmp_path):
    cp = tmp_path / "context.md"
    _good_context(cp)
    ok, reason = validate_context_md(cp)
    assert ok is True
    assert reason is None


# Test 14: validate_context_md missing section
def test_validate_context_missing_section(tmp_path):
    cp = tmp_path / "context.md"
    cp.write_text("### Project: X\nLast updated: 2026-04-10\n", encoding="utf-8")
    ok, reason = validate_context_md(cp)
    assert ok is False
    assert "missing" in reason


# Test 15: sync_context writes frontmatter atomically
def test_sync_context_writes_frontmatter(tmp_path):
    cp = tmp_path / "context.md"
    sp = tmp_path / "snapshot.md"
    lp = tmp_path / ".context.lock"
    _good_context(cp)
    ok = sync_context(cp, sp, lp)
    assert ok is True
    text = sp.read_text()
    assert text.startswith("---\n")
    assert "snapshot_sha:" in text
    assert "snapshot_ts:" in text
    assert "synced_from_run:" in text


# Test 16: verify_snapshot_fresh detects sha mismatch as stale
def test_freshness_sha_mismatch(tmp_path):
    cp = tmp_path / "context.md"
    sp = tmp_path / "snapshot.md"
    lp = tmp_path / ".context.lock"
    _good_context(cp)
    sync_context(cp, sp, lp)
    assert verify_snapshot_fresh(sp, cp, 24) is True
    # Mutate context
    cp.write_text(cp.read_text() + "\nextra\n", encoding="utf-8")
    assert verify_snapshot_fresh(sp, cp, 24) is False


# Test 17: verify_snapshot_fresh detects expiry
def test_freshness_expiry(tmp_path):
    cp = tmp_path / "context.md"
    sp = tmp_path / "snapshot.md"
    lp = tmp_path / ".context.lock"
    _good_context(cp)
    sync_context(cp, sp, lp)
    # Manually re-write with old timestamp
    text = sp.read_text()
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    import re
    text = re.sub(r"snapshot_ts: .+\n", f"snapshot_ts: {old_ts}\n", text)
    sp.write_text(text, encoding="utf-8")
    assert verify_snapshot_fresh(sp, cp, 24) is False
