"""Tests for scripts/assign_cell_type_labels.py."""
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
)


def test_clr_normalize_rows_sum_to_zero():
    """CLR row sums are ~0 by construction (mean-centering)."""
    rng = np.random.RandomState(0)
    adt = rng.poisson(5.0, size=(10, 20)).astype(np.float64)
    clr = clr_normalize(adt)
    row_sums = clr.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)


def test_assign_labels_basic_cd4_t():
    """A cell with high CD3-1 + CD4-1 + low CD8/CD19/CD56(NCAM) -> CD4_T."""
    antibody_names = ["CD3-1", "CD4-1", "CD8", "CD8a", "CD19", "CD56(NCAM)"]
    clr = np.array([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float64)
    labels = assign_labels(clr, antibody_names, threshold=0.5)
    assert labels[0] == "CD4_T"


def test_assign_labels_unknown_when_no_match():
    """A cell that fails all rules gets 'Unknown'."""
    antibody_names = ["CD3-1", "CD4-1", "CD8", "CD8a", "CD19", "CD56(NCAM)",
                      "CD14", "CD16", "CD11c", "HLA-DR", "CD25", "CD127",
                      "CD45RA", "CD45RO"]
    clr = np.zeros((1, len(antibody_names)), dtype=np.float64)
    labels = assign_labels(clr, antibody_names, threshold=0.5)
    assert labels[0] == "Unknown"


def test_assign_labels_first_match_wins():
    """When multiple rules could match, first in dict order wins."""
    antibody_names = ["CD3-1", "CD4-1", "CD8", "CD8a", "CD19", "CD56(NCAM)",
                      "CD45RA", "CD45RO"]
    clr_vals = {
        "CD3-1": 1.0, "CD4-1": 1.0,
        "CD8": -1.0, "CD8a": -1.0, "CD19": -1.0,
        "CD56(NCAM)": -1.0,
        "CD45RA": 1.0, "CD45RO": -1.0,
    }
    clr = np.array([[clr_vals[n] for n in antibody_names]], dtype=np.float64)
    labels = assign_labels(clr, antibody_names, threshold=0.5)
    assert labels[0] == "CD4_T", f"first-match-wins violated: got {labels[0]}"


def test_build_marker_index_returns_correct_positions():
    names = ["CD3-1", "CD4-1", "CD8", "CD56(NCAM)"]
    idx = build_marker_index(names)
    assert idx["CD3-1"] == 0
    assert idx["CD4-1"] == 1
    assert idx["CD8"] == 2
    assert idx["CD56(NCAM)"] == 3


def test_assign_labels_handles_missing_marker_gracefully(capsys):
    """If a rule references a missing marker, that rule is skipped (with stderr warning)
    and other rules still apply."""
    antibody_names = ["CD3-1", "CD4-1", "CD8", "CD19", "CD56(NCAM)"]
    clr = np.array([[1.0, 1.0, -1.0, -1.0, -1.0]], dtype=np.float64)
    labels = assign_labels(clr, antibody_names, threshold=0.5)
    assert labels[0] == "CD4_T"
    captured = capsys.readouterr()
    assert "CD8_T" in captured.err or "skip rule" in captured.err
