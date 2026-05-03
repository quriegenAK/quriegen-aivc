"""Tests for PR #54b1: MultiomeLoader supervised-label loading + collate.

Synthetic-data tests over the loader extension:
  - Backward compat: labels_obs_col=None -> __getitem__ unchanged
  - Manifest-less mode: builds class_to_idx from this h5ad alone
  - Manifest mode: reads canonical mapping from JSON
  - Manifest fingerprint surfacing
  - masked_classes excludes from supcon_eligible
  - min_confidence + confidence_obs_col filters out below-threshold cells
  - Unknown class in h5ad with manifest -> raises (provenance contract)
  - Collate pass-through: cell_type_idx + supcon_eligible_mask stacked
  - Collate heterogeneous-presence raise

Real-data tests deferred to PR #54b2 smoke (joint h5ad on disk).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from aivc.data.collate import dogma_collate
from aivc.data.multiome_loader import MultiomeLoader


# --- Helpers -----------------------------------------------------------------

PEAK_SET_TSV = """chr1\t100\t200\tpeak_0
chr1\t300\t400\tpeak_1
chr1\t500\t600\tpeak_2
chr1\t700\t800\tpeak_3
chr1\t900\t1000\tpeak_4
"""


def _write_synthetic_h5ad(
    tmp: Path,
    n_cells: int = 8,
    n_genes: int = 20,
    n_peaks: int = 5,
    n_proteins: int = 4,
    cell_types: list = None,
    confidences: list = None,
) -> Path:
    """Build a labeled DOGMA-shaped h5ad in tmp dir; return its path."""
    rng = np.random.RandomState(0)
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    atac = rng.poisson(1.0, size=(n_cells, n_peaks)).astype(np.float32)
    protein = rng.poisson(5.0, size=(n_cells, n_proteins)).astype(np.float32)

    obs = pd.DataFrame({
        "lysis_protocol": ["LLL"] * n_cells,
        "donor_id": ["d0"] * n_cells,
    })
    if cell_types is not None:
        assert len(cell_types) == n_cells
        obs["cell_type"] = pd.Categorical(cell_types)
    if confidences is not None:
        assert len(confidences) == n_cells
        obs["cell_type_confidence"] = np.asarray(confidences, dtype=np.float32)

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["atac_peaks"] = atac
    adata.obsm["protein"] = protein

    h5ad_path = tmp / "synth.h5ad"
    adata.write_h5ad(h5ad_path, compression=None)

    # Companion peak set
    peak_path = tmp / "peaks.tsv"
    peak_path.write_text(PEAK_SET_TSV)
    return h5ad_path


def _build_loader(tmp: Path, **kwargs) -> MultiomeLoader:
    # Auto-size n_cells to len(cell_types) when caller passes cell_types but
    # not n_cells explicitly. Removes a class of off-by-N test bugs.
    h5ad_kwargs = {k: v for k, v in kwargs.items()
                   if k in ("n_cells", "n_genes", "n_peaks",
                            "n_proteins", "cell_types", "confidences")}
    if "cell_types" in h5ad_kwargs and "n_cells" not in h5ad_kwargs:
        h5ad_kwargs["n_cells"] = len(h5ad_kwargs["cell_types"])
    h5ad = _write_synthetic_h5ad(tmp, **h5ad_kwargs)
    loader_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("n_cells", "n_genes", "n_peaks",
                                  "n_proteins", "cell_types", "confidences")}
    return MultiomeLoader(
        h5ad_path=str(h5ad),
        schema="obsm_atac",
        peak_set_path=str(tmp / "peaks.tsv"),
        atac_key="atac_peaks",
        protein_obsm_key="protein",
        lysis_protocol="LLL",
        protein_panel_id="totalseq_a_210",
        **loader_kwargs,
    )


# --- Backward compat ---------------------------------------------------------

def test_loader_no_labels_obs_col_no_label_state(tmp_path: Path):
    """labels_obs_col=None: __getitem__ output is unchanged from pre-PR shape."""
    loader = _build_loader(tmp_path)
    item = loader[0]
    assert "cell_type_idx" not in item
    assert "supcon_eligible" not in item
    assert loader.n_classes is None
    assert loader.class_to_idx is None
    assert loader.manifest_fingerprint is None


# --- Manifest-less mode (build from this h5ad) ------------------------------

def test_loader_loads_labels_without_manifest(tmp_path: Path):
    """No manifest path: build class_to_idx from this h5ad alone."""
    cell_types = ["B", "CD4_T", "CD4_T", "CD8_T", "NK", "B", "CD4_T", "DC"]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        labels_obs_col="cell_type",
    )
    assert loader.n_classes == 5  # B, CD4_T, CD8_T, DC, NK -> 5 distinct, sorted
    expected = {"B": 0, "CD4_T": 1, "CD8_T": 2, "DC": 3, "NK": 4}
    assert loader.class_to_idx == expected

    # First item has cell_type "B" -> index 0; eligibility default True
    item0 = loader[0]
    assert item0["cell_type_idx"] == 0
    assert item0["supcon_eligible"] is True

    # Item 4 has cell_type "NK" -> index 4
    item4 = loader[4]
    assert item4["cell_type_idx"] == 4


# --- Manifest mode -----------------------------------------------------------

def test_loader_uses_provided_manifest(tmp_path: Path):
    """class_index_manifest path: read mapping from JSON."""
    cell_types = ["CD4_T", "B"]
    # Manifest says CD4_T=3, B=7 (deliberately non-lexicographic)
    manifest = {
        "class_to_idx": {"CD4_T": 3, "B": 7},
        "n_classes": 2,
        "fingerprint_sha256": "abc123" + "0" * 58,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loader = _build_loader(
        tmp_path,
        cell_types=cell_types + ["CD4_T", "B", "CD4_T", "B", "CD4_T", "B"],
        labels_obs_col="cell_type",
        class_index_manifest=str(manifest_path),
    )
    assert loader.class_to_idx == {"CD4_T": 3, "B": 7}
    assert loader.manifest_fingerprint == "abc123" + "0" * 58
    assert loader[0]["cell_type_idx"] == 3   # CD4_T
    assert loader[1]["cell_type_idx"] == 7   # B


def test_loader_raises_on_unknown_class_when_manifest_provided(tmp_path: Path):
    """h5ad has 'Mystery' class not in manifest -> ValueError."""
    cell_types = ["CD4_T", "Mystery"] + ["CD4_T"] * 6
    manifest = {"class_to_idx": {"CD4_T": 0}, "n_classes": 1}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Mystery"):
        _build_loader(
            tmp_path,
            cell_types=cell_types,
            labels_obs_col="cell_type",
            class_index_manifest=str(manifest_path),
        )


# --- Eligibility masking -----------------------------------------------------

def test_masked_classes_excluded_from_eligible(tmp_path: Path):
    cell_types = ["CD4_T", "other", "CD4_T", "other_T", "CD4_T", "B", "DC", "NK"]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        labels_obs_col="cell_type",
        masked_classes=["other", "other_T"],
    )
    # Items with masked classes have eligible=False
    assert loader[0]["supcon_eligible"] is True   # CD4_T
    assert loader[1]["supcon_eligible"] is False  # other
    assert loader[3]["supcon_eligible"] is False  # other_T
    assert loader[5]["supcon_eligible"] is True   # B


def test_min_confidence_filters_below_threshold(tmp_path: Path):
    cell_types = ["CD4_T"] * 8
    confidences = [0.95, 0.30, 0.62, 0.59, 0.99, 0.50, 0.75, 0.61]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        confidences=confidences,
        labels_obs_col="cell_type",
        confidence_obs_col="cell_type_confidence",
        min_confidence=0.6,
    )
    elig = [loader[i]["supcon_eligible"] for i in range(8)]
    expected = [True, False, True, False, True, False, True, True]
    assert elig == expected


def test_min_confidence_with_missing_confidence_column_falls_back_to_max(tmp_path: Path, capsys):
    """Confidence column requested but not in h5ad -> treat all cells as
    max conf (no filtering on confidence axis).

    This is the LLL-arm path: published Azimuth labels are gold-standard
    so the labeled h5ad has no `cell_type_confidence` column. The joint
    factory passes the same kwargs to both arms, so the loader must
    gracefully fall back rather than hard-fail when the column is absent.
    """
    cell_types = ["CD4_T", "B", "CD4_T", "CD4_T", "DC", "NK", "CD8_T", "CD4_T"]
    # NOTE: confidences=None -> obs column not written
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        labels_obs_col="cell_type",
        confidence_obs_col="cell_type_confidence",  # absent in h5ad
        min_confidence=0.6,
    )
    # All 8 should be eligible (only class mask applies; no class is masked)
    elig = [loader[i]["supcon_eligible"] for i in range(8)]
    assert elig == [True] * 8
    captured = capsys.readouterr()
    assert "not present" in captured.err
    assert "max confidence" in captured.err


def test_masked_classes_and_min_confidence_combined(tmp_path: Path):
    cell_types = ["CD4_T", "other", "CD4_T", "B"]
    confidences = [0.95, 0.99, 0.30, 0.85]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        confidences=confidences,
        labels_obs_col="cell_type",
        confidence_obs_col="cell_type_confidence",
        masked_classes=["other"],
        min_confidence=0.6,
    )
    elig = [loader[i]["supcon_eligible"] for i in range(4)]
    # CD4_T high conf -> True
    # other (masked) -> False (regardless of conf)
    # CD4_T low conf -> False (conf filter)
    # B high conf -> True
    assert elig == [True, False, False, True]


# --- Collate -----------------------------------------------------------------

def test_collate_passes_through_label_keys(tmp_path: Path):
    cell_types = ["CD4_T", "B", "CD4_T", "DC"]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types,
        labels_obs_col="cell_type",
    )
    items = [loader[i] for i in range(4)]
    out = dogma_collate(items)
    assert "cell_type_idx" in out
    assert out["cell_type_idx"].dtype == torch.long
    assert out["cell_type_idx"].shape == (4,)
    # Sorted classes in this h5ad: B=0, CD4_T=1, DC=2 → labels [1, 0, 1, 2]
    assert out["cell_type_idx"].tolist() == [1, 0, 1, 2]
    assert "supcon_eligible_mask" in out
    assert out["supcon_eligible_mask"].dtype == torch.bool
    assert out["supcon_eligible_mask"].tolist() == [True, True, True, True]


def test_collate_heterogeneous_label_presence_raises(tmp_path: Path):
    cell_types = ["CD4_T", "B"]
    loader = _build_loader(
        tmp_path,
        cell_types=cell_types + ["CD4_T"] * 6,
        labels_obs_col="cell_type",
    )
    items = [loader[i] for i in range(4)]
    # Strip label keys from one item
    items[2] = {k: v for k, v in items[2].items()
                if k not in ("cell_type_idx", "supcon_eligible")}
    with pytest.raises(ValueError, match="Heterogeneous cell_type_idx"):
        dogma_collate(items)


def test_collate_no_labels_when_loader_didnt_load_any(tmp_path: Path):
    """labels_obs_col=None: collate output has no cell_type_idx key."""
    loader = _build_loader(tmp_path)
    items = [loader[i] for i in range(4)]
    out = dogma_collate(items)
    assert "cell_type_idx" not in out
    assert "supcon_eligible_mask" not in out
