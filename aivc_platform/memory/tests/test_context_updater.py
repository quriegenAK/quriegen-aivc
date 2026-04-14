"""Tests for aivc_platform.memory.context_updater."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aivc_platform.memory.context_updater import update_context
from aivc_platform.tracking.schemas import Modality, RunMetadata, RunStatus

FIXTURE = Path(__file__).parent / "fixtures" / "context_sample.md"


def _copy_fixture(tmp_path: Path, current_best: float | None = None) -> Path:
    """Copy the fixture into tmp_path, optionally seeding a best_pearson_r line."""
    dst = tmp_path / "context.md"
    dst.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    if current_best is not None:
        existing = dst.read_text(encoding="utf-8")
        # Insert as a sibling to `Last updated`.
        seeded = existing.replace(
            "Last updated: 2026-04-10",
            f"Last updated: 2026-04-10\nbest pearson_r: {current_best:.4f}",
            1,
        )
        dst.write_text(seeded, encoding="utf-8")
    return dst


def _make_meta(pearson_r: float = 0.90) -> RunMetadata:
    return RunMetadata(
        run_id="ctx_test_001",
        dataset="kang2018_pbmc",
        modality=Modality.IFNB,
        pearson_r=pearson_r,
        delta_nonzero_pct=40.0,
        checkpoint_path="/tmp/ckpt_v11.pt",
        wandb_url="https://wandb.ai/quriegen/aivc_genelink/runs/abc",
        status=RunStatus.SUCCESS,
        finished_at=datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── Test 9 ──────────────────────────────────────────────────────────────
def test_updates_last_updated(tmp_path):
    path = _copy_fixture(tmp_path)
    update_context(_make_meta(), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"Last updated: {today}" in body
    assert "Last updated: 2026-04-10" not in body


# ── Test 10 ─────────────────────────────────────────────────────────────
def test_updates_best_pearson_when_higher(tmp_path):
    path = _copy_fixture(tmp_path, current_best=0.85)
    update_context(_make_meta(pearson_r=0.90), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    assert "best pearson_r: 0.9000" in body
    assert "best pearson_r: 0.8500" not in body


# ── Test 11 ─────────────────────────────────────────────────────────────
def test_does_not_update_best_pearson_when_lower(tmp_path):
    path = _copy_fixture(tmp_path, current_best=0.95)
    update_context(_make_meta(pearson_r=0.90), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    assert "best pearson_r: 0.9500" in body
    assert "best pearson_r: 0.9000" not in body


# ── Test 12 ─────────────────────────────────────────────────────────────
def test_idempotent(tmp_path):
    path = _copy_fixture(tmp_path)
    meta = _make_meta()
    update_context(meta, None, context_path=str(path))
    first = path.read_bytes()
    update_context(meta, None, context_path=str(path))
    second = path.read_bytes()
    assert first == second


# ── Test 13 ─────────────────────────────────────────────────────────────
def test_open_questions_untouched(tmp_path):
    # Inject an updatable string inside ### Open questions.
    path = _copy_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    poisoned = text.replace(
        "### Open questions\n",
        "### Open questions\nLast updated: 1999-01-01\n",
        1,
    )
    path.write_text(poisoned, encoding="utf-8")
    update_context(_make_meta(), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    # The poisoned line inside Open questions must survive intact.
    assert "Last updated: 1999-01-01" in body


# ── Test 14 ─────────────────────────────────────────────────────────────
def test_do_not_touch_untouched(tmp_path):
    path = _copy_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    poisoned = text.replace(
        "### Do not touch\n",
        "### Do not touch\n- **latest checkpoint**: /frozen/baseline.pt\n",
        1,
    )
    path.write_text(poisoned, encoding="utf-8")
    update_context(_make_meta(), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    assert "/frozen/baseline.pt" in body


# ── Test 15 ─────────────────────────────────────────────────────────────
def test_missing_context_raises(tmp_path):
    missing = tmp_path / "nope.md"
    with pytest.raises(FileNotFoundError):
        update_context(_make_meta(), None, context_path=str(missing))


# ── Test 16 ─────────────────────────────────────────────────────────────
def test_dry_run_no_write(tmp_path, capsys):
    path = _copy_fixture(tmp_path)
    before = path.read_bytes()
    update_context(_make_meta(), None, context_path=str(path), dry_run=True)
    after = path.read_bytes()
    assert before == after
    captured = capsys.readouterr().out
    assert "---" in captured or "@@" in captured or "+++" in captured


# ── Test 17 ─────────────────────────────────────────────────────────────
def test_missing_field_appended_under_architecture(tmp_path):
    path = _copy_fixture(tmp_path)
    update_context(_make_meta(), None, context_path=str(path))
    body = path.read_text(encoding="utf-8")
    assert "## Architecture decisions" in body
    assert "latest checkpoint" in body.lower()
    assert "last W&B run" in body or "last W&B" in body
    assert "last run date" in body.lower()
