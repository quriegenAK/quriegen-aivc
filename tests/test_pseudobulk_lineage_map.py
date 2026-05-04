"""Unit test for the Calderon 25→6 lineage mapping in
scripts/eval_pseudobulk_compatibility.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_pseudobulk_compatibility import map_to_lineage  # noqa: E402


@pytest.mark.parametrize("calderon_label, expected_lineage", [
    # B cell variants
    ("Bulk_B",                     "B"),
    ("Naive_B",                    "B"),
    ("Memory_B",                   "B"),
    ("Plasmablast",                "B"),
    # CD4 T variants
    ("Naive_CD4pos_T",             "CD4_T"),
    ("Memory_CD4pos_T",            "CD4_T"),
    ("Effector_CD4pos_T",          "CD4_T"),
    ("Th1_precursors",             "CD4_T"),
    ("Th2_precursors",             "CD4_T"),
    ("Th17_precursors",            "CD4_T"),
    ("Regulatory_T",               "CD4_T"),
    ("Follicular_T_Helper",        "CD4_T"),
    # CD8 T variants
    ("CD8pos_T",                   "CD8_T"),
    ("Naive_CD8pos_T",             "CD8_T"),
    ("Central_memory_CD8pos_T",    "CD8_T"),
    ("Effector_memory_CD8pos_T",   "CD8_T"),
    # NK variants
    ("Immature_NK",                "NK"),
    ("Mature_NK",                  "NK"),
    ("NK_CD56_dim",                "NK"),
    ("NK_CD56_bright",             "NK"),
    # Monocyte
    ("Monocytes",                  "Monocyte"),
    ("Classical_Monocytes",        "Monocyte"),
    # DC variants
    ("Myeloid_DC",                 "DC"),
    ("pDC",                        "DC"),
    ("Plasmacytoid_dendritic",     "DC"),
])
def test_known_calderon_labels_map_correctly(calderon_label, expected_lineage):
    assert map_to_lineage(calderon_label) == expected_lineage


@pytest.mark.parametrize("calderon_label", [
    "Gamma_delta_T",   # gamma-delta T cells — not in any of our 6 lineages
    "Erythrocyte",     # not in DOGMA panel
    "Unknown",
    "weird_class_42",
])
def test_unmappable_returns_none(calderon_label):
    """Calderon types that don't fit any of the 6 lineage rules are dropped."""
    assert map_to_lineage(calderon_label) is None


def test_case_insensitive():
    """Mapping should not be case-sensitive (defensive against label drift)."""
    assert map_to_lineage("CD4POS_T") == "CD4_T"
    assert map_to_lineage("naive_b") == "B"
    assert map_to_lineage("NK_dim") == "NK"
