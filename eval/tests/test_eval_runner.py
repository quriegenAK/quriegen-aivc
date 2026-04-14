"""
Tests for eval/eval_runner.py — suite orchestration.
Uses unittest.mock — no live GPU or file I/O.
"""
import numpy as np
import pytest
from unittest import mock

from eval.benchmarks.kang_eval import KangEvalReport
from eval.benchmarks.norman_eval import NormanEvalReport
from eval.eval_runner import EvalSuite, run_eval_suite, populate_run_metadata


def _kang_fail() -> KangEvalReport:
    return KangEvalReport(
        run_id="test",
        pearson_r=0.85,
        regression_guard_passed=False,
        passed=False,
        failure_reason="Kang regression guard: r=0.8500 < 0.873",
    )


def _kang_pass() -> KangEvalReport:
    return KangEvalReport(
        run_id="test",
        pearson_r=0.89,
        pearson_r_ctrl_sub=0.3,
        delta_nonzero_pct=5.0,
        regression_guard_passed=True,
        passed=True,
    )


def _norman_fail() -> NormanEvalReport:
    return NormanEvalReport(
        run_id="test",
        delta_nonzero_pct=0.5,
        ctrl_memorisation_score=0.99,
        passed=False,
        failure_reason="Norman pass gate: delta_nonzero_pct=0.50%",
    )


def _norman_pass() -> NormanEvalReport:
    return NormanEvalReport(
        run_id="test",
        delta_nonzero_pct=5.0,
        ctrl_memorisation_score=0.8,
        pearson_r_ctrl_sub=0.2,
        passed=True,
    )


# Test 11: Kang fails → norman=None, replogle=None, overall_passed=False
@mock.patch("eval.eval_runner.run_norman_eval")
@mock.patch("eval.eval_runner.run_kang_eval")
def test_kang_fails_halts_suite(mock_kang, mock_norman):
    """Kang failure halts the suite — Norman is never called."""
    mock_kang.return_value = _kang_fail()

    suite = run_eval_suite("/fake.pt", run_id="test")

    assert not suite.overall_passed
    assert suite.norman is None
    assert suite.replogle is None
    assert suite.halt_reason is not None
    assert "0.873" in suite.halt_reason
    mock_norman.assert_not_called()


# Test 12: Kang passes, Norman fails → overall_passed=False, norman populated
@mock.patch("eval.eval_runner.run_norman_eval")
@mock.patch("eval.eval_runner.run_kang_eval")
def test_kang_pass_norman_fail(mock_kang, mock_norman):
    """Kang passes but Norman fails → overall_passed=False, norman report present."""
    mock_kang.return_value = _kang_pass()
    mock_norman.return_value = _norman_fail()

    suite = run_eval_suite("/fake.pt", run_id="test")

    assert not suite.overall_passed
    assert suite.norman is not None
    assert not suite.norman.passed
    assert suite.kang.passed


# Test 13: replogle always None
@mock.patch("eval.eval_runner.run_norman_eval")
@mock.patch("eval.eval_runner.run_kang_eval")
def test_replogle_always_none(mock_kang, mock_norman):
    """Replogle is always None in v1."""
    mock_kang.return_value = _kang_pass()
    mock_norman.return_value = _norman_pass()

    suite = run_eval_suite("/fake.pt", run_id="test")

    assert suite.replogle is None
    assert suite.overall_passed


# Test 14: populate_run_metadata maps fields correctly
def test_populate_run_metadata():
    """populate_run_metadata maps kang.pearson_r and norman fields to meta."""
    from aivc_platform.tracking.schemas import RunMetadata

    meta = RunMetadata(run_id="test")
    suite = EvalSuite(
        kang=_kang_pass(),
        norman=_norman_pass(),
        overall_passed=True,
    )

    result = populate_run_metadata(meta, suite)

    assert result.pearson_r == suite.kang.pearson_r
    assert result.delta_nonzero_pct == suite.norman.delta_nonzero_pct
    assert result.ctrl_memorisation_score == suite.norman.ctrl_memorisation_score
