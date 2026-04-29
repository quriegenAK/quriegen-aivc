"""Tests for DogmaJointLoader (synthetic; real_data smoke separate)."""
from __future__ import annotations

import numpy as np
import pytest

from aivc.data.multiome_loader import DogmaJointLoader


class _MockArmLoader:
    """Mock single-arm loader exposing the MultiomeLoader interface."""

    def __init__(self, n_cells, n_genes, n_peaks, n_proteins, prefix):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.n_proteins = n_proteins
        self.prefix = prefix

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return {
            "rna": np.ones(self.n_genes, dtype=np.float32) * idx,
            "atac_peaks": np.ones(self.n_peaks, dtype=np.float32) * idx,
            "protein": np.ones(self.n_proteins, dtype=np.float32) * idx,
            "_source": self.prefix,
        }


def test_joint_length_is_sum():
    lll = _MockArmLoader(10, 36, 50, 4, "lll")
    dig = _MockArmLoader(15, 36, 50, 4, "dig")
    joint = DogmaJointLoader(lll, dig)
    assert len(joint) == 25


def test_joint_lll_indices_get_lysis_zero():
    lll = _MockArmLoader(5, 36, 50, 4, "lll")
    dig = _MockArmLoader(7, 36, 50, 4, "dig")
    joint = DogmaJointLoader(lll, dig)
    for i in range(5):
        item = joint[i]
        assert item["lysis_idx"] == 0
        assert item["_source"] == "lll"


def test_joint_dig_indices_get_lysis_one():
    lll = _MockArmLoader(5, 36, 50, 4, "lll")
    dig = _MockArmLoader(7, 36, 50, 4, "dig")
    joint = DogmaJointLoader(lll, dig)
    for i in range(5, 12):
        item = joint[i]
        assert item["lysis_idx"] == 1
        assert item["_source"] == "dig"


def test_joint_dim_mismatch_raises():
    lll = _MockArmLoader(5, 36, 50, 4, "lll")
    dig = _MockArmLoader(7, 36, 99, 4, "dig")  # n_peaks mismatch
    with pytest.raises(ValueError, match=r"n_peaks mismatch"):
        DogmaJointLoader(lll, dig)


def test_joint_n_genes_mismatch_raises():
    lll = _MockArmLoader(5, 36, 50, 4, "lll")
    dig = _MockArmLoader(7, 99, 50, 4, "dig")
    with pytest.raises(ValueError, match=r"n_genes mismatch"):
        DogmaJointLoader(lll, dig)


def test_joint_properties_match_arms():
    lll = _MockArmLoader(5, 36, 50, 4, "lll")
    dig = _MockArmLoader(7, 36, 50, 4, "dig")
    joint = DogmaJointLoader(lll, dig)
    assert joint.n_genes == 36
    assert joint.n_peaks == 50
    assert joint.n_proteins == 4
