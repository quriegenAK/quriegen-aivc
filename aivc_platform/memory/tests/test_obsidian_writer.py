"""Tests for aivc_platform.memory.vault + obsidian_writer."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pytest

from aivc_platform.memory.obsidian_writer import (
    write_experiment_note,
    write_failure_note,
)
from aivc_platform.memory.vault import SUBDIRS, ObsidianConfig, init_vault
from aivc_platform.tracking.schemas import (
    Modality,
    PostRunDecision,
    RunMetadata,
    RunStatus,
)
from eval.benchmarks.kang_eval import KangEvalReport
from eval.benchmarks.norman_eval import NormanEvalReport
from eval.eval_runner import EvalSuite


def _make_meta(**overrides) -> RunMetadata:
    base = dict(
        run_id="test_run_001",
        dataset="kang2018_pbmc",
        modality=Modality.IFNB,
        pearson_r=0.88,
        pearson_r_std=0.01,
        delta_nonzero_pct=45.0,
        ctrl_memorisation_score=0.2,
        jakstat_within_3x=10,
        ifit1_pred_fc=3.2,
        checkpoint_path="/tmp/ckpt.pt",
        wandb_url="https://wandb.ai/quriegen/aivc_genelink/runs/xyz",
        status=RunStatus.SUCCESS,
        decision=PostRunDecision.REGISTER,
        training_time_s=123.0,
        finished_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return RunMetadata(**base)


def _make_suite(
    kang_passed: bool = True,
    norman: NormanEvalReport | None = None,
    halt_reason: str | None = None,
) -> EvalSuite:
    kang = KangEvalReport(
        run_id="test_run_001",
        pearson_r=0.88,
        pearson_r_std=0.01,
        pearson_r_ctrl_sub=0.31,
        delta_nonzero_pct=45.0,
        ctrl_memorisation_score=0.2,
        top_k_overlap=0.4,
        regression_guard_passed=kang_passed,
        passed=kang_passed,
        failure_reason=None if kang_passed else "regression guard failed: r=0.70 < 0.873",
    )
    return EvalSuite(
        kang=kang,
        norman=norman,
        replogle=None,
        overall_passed=bool(kang_passed and norman and norman.passed),
        halt_reason=halt_reason,
    )


def _make_norman(
    delta_nonzero_pct: float = 15.0,
    passed: bool = True,
    ctrl_mem: float = 0.3,
) -> NormanEvalReport:
    return NormanEvalReport(
        run_id="test_run_001",
        n_perturbations=17,
        n_pseudobulk_pairs=340,
        pearson_r_ctrl_sub=0.22,
        delta_nonzero_pct=delta_nonzero_pct,
        ctrl_memorisation_score=ctrl_mem,
        top_k_overlap=0.5,
        passed=passed,
    )


# ── Test 1 ──────────────────────────────────────────────────────────────
def test_init_vault_creates_subdirs(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    root = config.resolved_vault()
    for sub in SUBDIRS:
        assert (root / sub).is_dir(), f"missing subdir: {sub}"
    assert (root / "papers" / "kang2018.md").exists()
    assert (root / "papers" / "norman2019.md").exists()


# ── Test 2 ──────────────────────────────────────────────────────────────
def test_init_vault_idempotent(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    # Mutate stub; second init must not clobber.
    kang_path = config.resolved_vault() / "papers" / "kang2018.md"
    kang_path.write_text("USER EDIT", encoding="utf-8")
    init_vault(config)
    assert kang_path.read_text(encoding="utf-8") == "USER EDIT"


# ── Test 3 ──────────────────────────────────────────────────────────────
def test_init_vault_dry_run(tmp_path, capsys):
    vault_dir = tmp_path / "dryvault"
    config = ObsidianConfig(vault_path=str(vault_dir), dry_run=True)
    init_vault(config)
    captured = capsys.readouterr().out
    assert "DRY RUN" in captured
    assert not vault_dir.exists()


# ── Test 4 ──────────────────────────────────────────────────────────────
def test_write_experiment_note_creates_file(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    meta = _make_meta()
    suite = _make_suite(norman=_make_norman())
    out = write_experiment_note(meta, suite, config)
    assert out.exists()
    assert out == config.resolved_vault() / "experiments" / "test_run_001.md"
    body = out.read_text(encoding="utf-8")
    assert "run_id: test_run_001" in body
    assert "n/a (v1.2)" in body  # anndata_hash
    assert "n/a (file store, no registry)" in body


# ── Test 5 ──────────────────────────────────────────────────────────────
def test_write_experiment_note_idempotent(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    suite = _make_suite(norman=_make_norman())
    p1 = write_experiment_note(_make_meta(pearson_r=0.88), suite, config)
    p2 = write_experiment_note(_make_meta(pearson_r=0.91), suite, config)
    assert p1 == p2
    body = p2.read_text(encoding="utf-8")
    # Second write wins (overwrites).
    assert "modality: IFN-b" in body


# ── Test 6 ──────────────────────────────────────────────────────────────
def test_write_experiment_note_dry_run(tmp_path, capsys):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"), dry_run=True)
    meta = _make_meta()
    suite = _make_suite(norman=_make_norman())
    out = write_experiment_note(meta, suite, config)
    captured = capsys.readouterr().out
    assert "DRY RUN" in captured
    assert "run_id: test_run_001" in captured
    assert not out.exists()


# ── Test 7 ──────────────────────────────────────────────────────────────
def test_write_failure_note_delta_collapse(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    meta = _make_meta(
        pearson_r=0.85,
        delta_nonzero_pct=0.0,
        ctrl_memorisation_score=0.98,
        status=RunStatus.CTRL_MEMORISATION,
        decision=PostRunDecision.TRIGGER_TRAINING_AGENT,
    )
    suite = _make_suite(
        kang_passed=True,
        norman=_make_norman(delta_nonzero_pct=0.0, passed=False, ctrl_mem=0.99),
        halt_reason="delta_nonzero == 0: ctrl memorisation",
    )
    out = write_failure_note(meta, PostRunDecision.TRIGGER_TRAINING_AGENT, suite, config)
    body = out.read_text(encoding="utf-8")
    assert "delta_collapse" in body
    assert "w_scale_range too aggressive" in body
    assert "Shrink w_scale_range" in body


# ── Test 8 ──────────────────────────────────────────────────────────────
def test_write_failure_note_regression_guard(tmp_path):
    config = ObsidianConfig(vault_path=str(tmp_path / "vault"))
    init_vault(config)
    meta = _make_meta(
        pearson_r=0.70,
        status=RunStatus.REGRESSION,
        decision=PostRunDecision.TRIGGER_TRAINING_AGENT,
    )
    suite = _make_suite(
        kang_passed=False,
        norman=None,
        halt_reason="regression guard failed: r=0.70 < 0.873",
    )
    out = write_failure_note(meta, PostRunDecision.TRIGGER_TRAINING_AGENT, suite, config)
    body = out.read_text(encoding="utf-8")
    assert "regression_guard" in body
    assert "GAT unfreeze leaked" in body
    assert "frozen_modules" in body
