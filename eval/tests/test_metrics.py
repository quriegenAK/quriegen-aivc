"""
Tests for eval/metrics.py — pure metric functions.
"""
import numpy as np
import pytest

from eval.metrics import (
    _per_cell_pearson,
    ctrl_memorisation_score,
    delta_nonzero_pct,
    pearson_r_ctrl_subtracted,
)


# Test 1: pearson_r_ctrl_subtracted(ctrl, ctrl, ctrl) → nan
def test_pearson_r_ctrl_subtracted_degenerate():
    """When pred==ctrl==truth, truth delta is zero → returns NaN."""
    ctrl = np.random.rand(5, 100)
    result = pearson_r_ctrl_subtracted(ctrl, ctrl, ctrl)
    assert np.isnan(result), f"Expected NaN, got {result}"


# Test 2: delta_nonzero_pct(ctrl, ctrl) → 0.0
def test_delta_nonzero_pct_identical():
    """Identical pred and ctrl → 0% nonzero delta."""
    ctrl = np.random.rand(5, 100)
    result = delta_nonzero_pct(ctrl, ctrl)
    assert result == 0.0, f"Expected 0.0, got {result}"


# Test 3: ctrl_memorisation_score(ctrl, ctrl) → 1.0 ± 1e-12
def test_ctrl_memorisation_score_identical():
    """Identical pred and ctrl → cosine similarity of 1.0."""
    ctrl = np.abs(np.random.rand(5, 100)) + 0.1  # ensure non-negative
    result = ctrl_memorisation_score(ctrl, ctrl)
    assert abs(result - 1.0) < 1e-12, f"Expected 1.0, got {result}"


# Test 4: delta_nonzero_pct with |pred-ctrl| > 1e-6 everywhere → 100.0
def test_delta_nonzero_pct_all_different():
    """When every entry differs by more than epsilon → 100%."""
    ctrl = np.zeros((5, 100))
    pred = np.ones((5, 100))  # all differ by 1.0 >> epsilon
    result = delta_nonzero_pct(pred, ctrl)
    assert result == 100.0, f"Expected 100.0, got {result}"


# Test 5: _per_cell_pearson — row with zero std → 0.0, not NaN
def test_per_cell_pearson_zero_std():
    """A constant row (std=0) should return 0.0, not NaN."""
    a = np.array([[1.0, 1.0, 1.0, 1.0],  # constant row
                   [1.0, 2.0, 3.0, 4.0]])
    b = np.array([[5.0, 6.0, 7.0, 8.0],
                   [1.0, 2.0, 3.0, 4.0]])
    rs = _per_cell_pearson(a, b)
    assert rs[0] == 0.0, f"Expected 0.0 for constant row, got {rs[0]}"
    assert not np.isnan(rs[0]), "Got NaN for constant row"
    assert rs[1] == pytest.approx(1.0, abs=1e-6), f"Expected ~1.0 for identical rows, got {rs[1]}"
