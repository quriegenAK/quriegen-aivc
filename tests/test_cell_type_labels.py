"""Tests for scripts/assign_cell_type_labels.py.

Coverage: CLR normalization (incl. pre-CLR passthrough regression),
per-marker z-score, 6-lineage gating with asymmetric thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from assign_cell_type_labels import (  # noqa: E402
    GATING_RULES,
    assign_labels,
    build_marker_index,
    clr_normalize,
    zscore_per_marker,
)


# Full antibody panel used across gating tests (matches DOGMA panel subset).
PANEL = ["CD3-1", "CD4-1", "CD8", "CD19", "CD56(NCAM)",
         "CD14", "CD16", "CD11c", "HLA-DR"]


def _zvec(panel: list[str], **kwargs) -> np.ndarray:
    """Build a 1-cell z-score vector from named-marker overrides; default 0."""
    z = np.zeros(len(panel), dtype=np.float64)
    for marker, val in kwargs.items():
        # underscores in kwarg → marker name with hyphens etc.
        canonical = {
            "CD3_1": "CD3-1", "CD4_1": "CD4-1", "CD56": "CD56(NCAM)",
            "HLA_DR": "HLA-DR",
        }.get(marker, marker)
        z[panel.index(canonical)] = val
    return z[None, :]


# --- CLR --------------------------------------------------------------------

def test_clr_normalize_rows_sum_to_zero():
    """CLR row sums are ~0 by construction (mean-centering)."""
    rng = np.random.RandomState(0)
    adt = rng.poisson(5.0, size=(10, 20)).astype(np.float64)
    clr = clr_normalize(adt)
    row_sums = clr.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)


def test_clr_normalize_passthrough_when_input_already_clr(capsys):
    """If input has negative values it's assumed pre-CLR; pass through.

    Regression for PR #53 production failure 2026-05-03: DOGMA h5ad
    obsm['protein'] is CLR-normalized at write time. A second log(x+1)
    on x < -1 produces NaN, sending 9415/13763 cells to Unknown via
    NaN propagation through > / <= comparisons.
    """
    pre_clr = np.array(
        [
            [1.5, -2.0, 0.3, -0.5],
            [-1.0, 2.0, -0.3, 0.5],
        ],
        dtype=np.float64,
    )
    out = clr_normalize(pre_clr)
    np.testing.assert_array_equal(out, pre_clr)
    captured = capsys.readouterr()
    assert "Assuming pre-CLR-normalized" in captured.err
    assert "skipping in-script CLR" in captured.err
    assert not np.isnan(out).any()


def test_clr_normalize_no_nan_on_already_normalized():
    """Hard regression: no NaN on pre-CLR input (the production bug)."""
    rng = np.random.RandomState(1)
    pre_clr = rng.normal(loc=0.0, scale=1.5, size=(50, 30)).astype(np.float64)
    assert (pre_clr < -1).any(), "test setup: need values < -1 to be a real regression"
    out = clr_normalize(pre_clr)
    assert not np.isnan(out).any(), "pre-CLR passthrough must not produce NaN"
    assert not np.isinf(out).any()


# --- z-score ----------------------------------------------------------------

def test_zscore_per_marker_columns_have_zero_mean_unit_std():
    """Per-marker z-score: each column has mean ~0 and std ~1."""
    rng = np.random.RandomState(2)
    mat = rng.normal(loc=3.0, scale=2.0, size=(500, 10)).astype(np.float64)
    z = zscore_per_marker(mat)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-3)


def test_zscore_per_marker_handles_constant_column():
    """A constant-valued column shouldn't blow up (eps in denominator)."""
    mat = np.zeros((50, 5), dtype=np.float64)
    mat[:, 1] = 1.0  # constant column
    z = zscore_per_marker(mat)
    assert not np.isnan(z).any()
    assert not np.isinf(z).any()


# --- gating rules (6 lineages) ----------------------------------------------

def test_gating_rules_are_six_lineages():
    """Lock the lineage set to the 6 we labelled (post-2026-05-03 simplification)."""
    expected = {"DC", "Monocyte", "NK", "B", "CD4_T", "CD8_T"}
    assert set(GATING_RULES.keys()) == expected


def test_assign_labels_cd4_t():
    z = _zvec(PANEL, CD3_1=2.0, CD4_1=2.0)  # all others 0 (≤ 1.0 ✓)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "CD4_T"


def test_assign_labels_cd8_t():
    z = _zvec(PANEL, CD3_1=2.0, CD8=2.0)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "CD8_T"


def test_assign_labels_b_cell():
    z = _zvec(PANEL, CD19=2.0)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "B"


def test_assign_labels_nk_cell():
    z = _zvec(PANEL, CD56=2.0)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "NK"


def test_assign_labels_monocyte():
    z = _zvec(PANEL, CD14=2.0)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "Monocyte"


def test_assign_labels_dc():
    z = _zvec(PANEL, CD11c=2.0, HLA_DR=2.0)  # CD14=0 → not strongly +
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "DC"


def test_assign_labels_unknown_when_no_match():
    z = np.zeros((1, len(PANEL)), dtype=np.float64)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "Unknown"


def test_assign_labels_asymmetric_threshold_neg_marker_moderate_signal():
    """A real CD4_T cell can have moderate (non-strong) CD8 signal and still match.

    With threshold=1.0 (asymmetric): pos requires z>1, neg requires z<=1.
    A CD4 T cell with CD3-1=2.0, CD4-1=2.0, CD8=0.7 (moderate non-specific)
    should still get CD4_T label — not Unknown.
    """
    z = _zvec(PANEL, CD3_1=2.0, CD4_1=2.0, CD8=0.7)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "CD4_T", (
        "asymmetric thresholding should tolerate moderate off-target signal"
    )


def test_assign_labels_dc_before_monocyte_when_dc_specific_markers_strong():
    """DC ordered before Monocyte: CD11c+HLA-DR+CD14- → DC, not Unknown."""
    z = _zvec(PANEL, CD11c=2.0, HLA_DR=2.0)
    labels = assign_labels(z, PANEL, threshold=1.0)
    assert labels[0] == "DC"


def test_build_marker_index_returns_correct_positions():
    names = ["CD3-1", "CD4-1", "CD8", "CD56(NCAM)"]
    idx = build_marker_index(names)
    assert idx["CD3-1"] == 0
    assert idx["CD4-1"] == 1
    assert idx["CD8"] == 2
    assert idx["CD56(NCAM)"] == 3


def test_assign_labels_handles_missing_marker_gracefully(capsys):
    """If a rule references a missing marker, that rule is skipped (with stderr
    warning) and other rules still apply."""
    # Subset panel: drop CD11c/HLA-DR/CD14/CD16 → DC, Monocyte must skip
    minimal_panel = ["CD3-1", "CD4-1", "CD8", "CD19", "CD56(NCAM)"]
    z = _zvec(minimal_panel, CD3_1=2.0, CD4_1=2.0)
    labels = assign_labels(z, minimal_panel, threshold=1.0)
    assert labels[0] == "CD4_T"
    captured = capsys.readouterr()
    assert "skip rule" in captured.err
