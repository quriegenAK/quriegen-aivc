"""Tests for aivc/data/modality_mask.py — canonical contract module."""
import torch
import pytest

from aivc.data.modality_mask import (
    ModalityKey,
    TEMPORAL_ORDER,
    build_mask,
    RNA_KEY,
    ATAC_KEY,
    PROTEIN_KEY,
    PHOSPHO_KEY,
    MASK_KEY,
    LYSIS_KEY,
    PROTEIN_PANEL_KEY,
)


def test_temporal_order_matches_fusion():
    """ModalityKey values must match TemporalCrossModalFusion.TEMPORAL_ORDER."""
    from aivc.skills.fusion import TemporalCrossModalFusion
    fusion_order = TemporalCrossModalFusion.TEMPORAL_ORDER
    assert ModalityKey.ATAC.value == fusion_order["atac"] == 0
    assert ModalityKey.PHOSPHO.value == fusion_order["phospho"] == 1
    assert ModalityKey.RNA.value == fusion_order["rna"] == 2
    assert ModalityKey.PROTEIN.value == fusion_order["protein"] == 3


def test_temporal_order_list():
    assert TEMPORAL_ORDER == [
        ModalityKey.ATAC,
        ModalityKey.PHOSPHO,
        ModalityKey.RNA,
        ModalityKey.PROTEIN,
    ]


def test_build_mask_shape():
    m = build_mask({ModalityKey.RNA, ModalityKey.ATAC}, batch_size=5)
    assert m.shape == (5, 4)
    assert m.dtype == torch.float32


def test_build_mask_correct_positions():
    """DOGMA: ATAC + RNA + Protein, no Phospho."""
    m = build_mask(
        {ModalityKey.ATAC, ModalityKey.RNA, ModalityKey.PROTEIN},
        batch_size=3,
    )
    # Column order: [ATAC, Phospho, RNA, Protein] = [1, 0, 1, 1]
    expected = torch.tensor([[1.0, 0.0, 1.0, 1.0]] * 3)
    assert torch.equal(m, expected)


def test_build_mask_all_present():
    m = build_mask(set(ModalityKey), batch_size=2)
    assert torch.equal(m, torch.ones(2, 4))


def test_build_mask_single_modality():
    m = build_mask({ModalityKey.RNA}, batch_size=4)
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]] * 4)
    assert torch.equal(m, expected)


def test_build_mask_empty_set():
    m = build_mask(set(), batch_size=2)
    assert torch.equal(m, torch.zeros(2, 4))


def test_build_mask_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        build_mask({ModalityKey.RNA}, batch_size=0)


def test_key_constants_distinct():
    keys = [RNA_KEY, ATAC_KEY, PROTEIN_KEY, PHOSPHO_KEY, MASK_KEY, LYSIS_KEY, PROTEIN_PANEL_KEY]
    assert len(set(keys)) == len(keys), f"duplicate keys among: {keys}"



# ---------------------------------------------------------------------
# mask_from_obs (Day 3 addition)
# ---------------------------------------------------------------------
def test_mask_from_obs_rna_only():
    """Kang-style dataset: has_rna=True, others False -> mask = [0,0,1,0]."""
    from aivc.data.modality_mask import mask_from_obs
    row = {"has_rna": True, "has_atac": False, "has_protein": False, "has_phospho": False}
    m = mask_from_obs(row)
    assert m.shape == (4,)
    assert m[int(ModalityKey.ATAC)] == 0.0
    assert m[int(ModalityKey.PHOSPHO)] == 0.0
    assert m[int(ModalityKey.RNA)] == 1.0
    assert m[int(ModalityKey.PROTEIN)] == 0.0


def test_mask_from_obs_trimodal_dogma():
    """DOGMA-style: has_rna + has_atac + has_protein all True."""
    from aivc.data.modality_mask import mask_from_obs
    row = {"has_rna": True, "has_atac": True, "has_protein": True, "has_phospho": False}
    m = mask_from_obs(row)
    assert m[int(ModalityKey.ATAC)] == 1.0
    assert m[int(ModalityKey.PHOSPHO)] == 0.0
    assert m[int(ModalityKey.RNA)] == 1.0
    assert m[int(ModalityKey.PROTEIN)] == 1.0


def test_mask_from_obs_missing_columns_default_false():
    """Missing has_* keys default to False (modality absent)."""
    from aivc.data.modality_mask import mask_from_obs
    row = {"has_rna": True}
    m = mask_from_obs(row)
    assert m[int(ModalityKey.RNA)] == 1.0
    assert m[int(ModalityKey.ATAC)] == 0.0
    assert m[int(ModalityKey.PROTEIN)] == 0.0
    assert m[int(ModalityKey.PHOSPHO)] == 0.0


def test_mask_from_obs_with_pandas_series():
    """mask_from_obs works on pandas Series (AnnData.obs row idiom)."""
    import pandas as pd
    from aivc.data.modality_mask import mask_from_obs
    row = pd.Series({
        "has_rna": True, "has_atac": True,
        "has_protein": False, "has_phospho": False,
    })
    m = mask_from_obs(row)
    assert m[int(ModalityKey.RNA)] == 1.0
    assert m[int(ModalityKey.ATAC)] == 1.0
    assert m[int(ModalityKey.PROTEIN)] == 0.0
