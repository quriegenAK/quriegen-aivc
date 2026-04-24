"""Tests for aivc/data/collate.py — canonical DOGMA batch collation."""
import pytest
import torch

from aivc.data.collate import (
    dogma_collate,
    lysis_protocol_to_code,
    LYSIS_PROTOCOL_CODES,
)
from aivc.data.modality_mask import (
    ModalityKey, build_mask,
    RNA_KEY, ATAC_KEY, PROTEIN_KEY, MASK_KEY,
    LYSIS_KEY, PROTEIN_PANEL_KEY,
)


def _mk_item(has_protein=True, lysis="LLL", panel="totalseq_a_210",
             dataset_kind="observational"):
    """Build a single MultiomeLoader-style item."""
    item = {
        RNA_KEY: torch.randn(500),
        ATAC_KEY: torch.randn(1000),
        MASK_KEY: build_mask(
            ({ModalityKey.ATAC, ModalityKey.RNA, ModalityKey.PROTEIN}
             if has_protein else
             {ModalityKey.ATAC, ModalityKey.RNA}),
            batch_size=1,
        ).squeeze(0),
        LYSIS_KEY: lysis,
        PROTEIN_PANEL_KEY: panel,
        "dataset_kind": dataset_kind,
    }
    if has_protein:
        item[PROTEIN_KEY] = torch.randn(210)
    return item


def test_lysis_protocol_to_code_known():
    assert lysis_protocol_to_code("LLL") == 0
    assert lysis_protocol_to_code("DIG") == 1
    assert lysis_protocol_to_code("unknown") == -1


def test_lysis_protocol_to_code_unknown_maps_to_sentinel():
    assert lysis_protocol_to_code("weird_string") == -1
    assert lysis_protocol_to_code("") == -1


def test_collate_trimodal_basic():
    batch = [_mk_item() for _ in range(4)]
    out = dogma_collate(batch)
    assert out[RNA_KEY].shape == (4, 500)
    assert out[ATAC_KEY].shape == (4, 1000)
    assert out[PROTEIN_KEY].shape == (4, 210)
    assert out[MASK_KEY].shape == (4, 4)
    assert out[LYSIS_KEY].shape == (4,)
    assert out[LYSIS_KEY].dtype == torch.long
    assert out[PROTEIN_PANEL_KEY] == "totalseq_a_210"
    assert out["dataset_kind"] == "observational"


def test_collate_bimodal_no_protein():
    batch = [_mk_item(has_protein=False) for _ in range(3)]
    out = dogma_collate(batch)
    assert PROTEIN_KEY not in out
    assert out[RNA_KEY].shape == (3, 500)
    assert (out[MASK_KEY][:, int(ModalityKey.PROTEIN)] == 0).all()


def test_collate_lysis_protocol_int_encoding():
    batch = [
        _mk_item(lysis="LLL"),
        _mk_item(lysis="DIG"),
        _mk_item(lysis="LLL"),
    ]
    out = dogma_collate(batch)
    assert torch.equal(out[LYSIS_KEY], torch.tensor([0, 1, 0], dtype=torch.long))


def test_collate_unknown_lysis_maps_to_neg1():
    batch = [_mk_item(lysis="unknown"), _mk_item(lysis="LLL")]
    out = dogma_collate(batch)
    assert torch.equal(out[LYSIS_KEY], torch.tensor([-1, 0], dtype=torch.long))


def test_collate_empty_batch_raises():
    with pytest.raises(ValueError, match="empty batch"):
        dogma_collate([])


def test_collate_mixed_protein_presence_raises_DD2():
    batch = [_mk_item(has_protein=True), _mk_item(has_protein=False)]
    with pytest.raises(ValueError, match="Heterogeneous Protein presence"):
        dogma_collate(batch)


def test_collate_mixed_panel_id_raises():
    batch = [
        _mk_item(panel="totalseq_a_210"),
        _mk_item(panel="totalseq_a_227"),
    ]
    with pytest.raises(ValueError, match="Heterogeneous protein_panel_id"):
        dogma_collate(batch)


def test_collate_mixed_dataset_kind_raises():
    batch = [
        _mk_item(dataset_kind="observational"),
        _mk_item(dataset_kind="interventional"),
    ]
    with pytest.raises(ValueError, match="Heterogeneous dataset_kind"):
        dogma_collate(batch)


def test_collate_modality_mask_values_preserved():
    """Modality_mask column values survive stacking."""
    batch = [_mk_item(has_protein=True) for _ in range(2)]
    out = dogma_collate(batch)
    for row in out[MASK_KEY]:
        assert row[int(ModalityKey.ATAC)] == 1.0
        assert row[int(ModalityKey.PHOSPHO)] == 0.0
        assert row[int(ModalityKey.RNA)] == 1.0
        assert row[int(ModalityKey.PROTEIN)] == 1.0


def test_collate_constants_align_with_modality_mask_module():
    """LYSIS_PROTOCOL_CODES exposes the 3 canonical values."""
    assert set(LYSIS_PROTOCOL_CODES.keys()) == {"LLL", "DIG", "unknown"}
    assert LYSIS_PROTOCOL_CODES["LLL"] == 0
    assert LYSIS_PROTOCOL_CODES["DIG"] == 1
    assert LYSIS_PROTOCOL_CODES["unknown"] == -1
