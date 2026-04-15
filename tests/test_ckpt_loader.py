"""Unit tests for the strict pretrained-checkpoint loader.

These tests cover:
* a valid checkpoint round-trips through ``load_pretrained_simple_rna_encoder``,
* ``schema_version`` mismatch raises with a clear message,
* missing ``rna_encoder`` state_dict keys raise with a named missing-key list,
* unexpected keys raise (strict loading, no silent skip).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.skills.rna_encoder import SimpleRNAEncoder
from aivc.training.ckpt_loader import (
    CheckpointSchemaError,
    load_pretrained_simple_rna_encoder,
)


N_GENES = 32
HIDDEN = 16
LATENT = 8


def _valid_ckpt_dict(enc: SimpleRNAEncoder) -> dict:
    return {
        "schema_version": 1,
        "rna_encoder": enc.state_dict(),
        "atac_encoder": {},
        "pretrain_head": {},
        "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
        "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
        "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
        "config": {"n_genes": N_GENES, "hidden_dim": HIDDEN, "latent_dim": LATENT},
    }


@pytest.fixture
def enc():
    torch.manual_seed(0)
    return SimpleRNAEncoder(n_genes=N_GENES, hidden_dim=HIDDEN, latent_dim=LATENT)


def test_valid_checkpoint_loads(tmp_path: Path, enc: SimpleRNAEncoder):
    ckpt = _valid_ckpt_dict(enc)
    p = tmp_path / "pretrain_encoders.pt"
    torch.save(ckpt, p)

    loaded = load_pretrained_simple_rna_encoder(p, expected_schema_version=1)

    assert isinstance(loaded, SimpleRNAEncoder)
    assert loaded.n_genes == N_GENES
    assert loaded.hidden_dim == HIDDEN
    assert loaded.latent_dim == LATENT
    for k, v in enc.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_schema_version_mismatch_raises(tmp_path: Path, enc: SimpleRNAEncoder):
    ckpt = _valid_ckpt_dict(enc)
    ckpt["schema_version"] = 2
    p = tmp_path / "bad_version.pt"
    torch.save(ckpt, p)

    with pytest.raises(CheckpointSchemaError, match="schema_version mismatch"):
        load_pretrained_simple_rna_encoder(p, expected_schema_version=1)


def test_missing_state_dict_key_raises(tmp_path: Path, enc: SimpleRNAEncoder):
    ckpt = _valid_ckpt_dict(enc)
    rna = dict(ckpt["rna_encoder"])
    del rna["decoder.bias"]  # drop one key
    ckpt["rna_encoder"] = rna
    p = tmp_path / "missing_key.pt"
    torch.save(ckpt, p)

    with pytest.raises(CheckpointSchemaError) as excinfo:
        load_pretrained_simple_rna_encoder(p, expected_schema_version=1)
    assert "decoder.bias" in str(excinfo.value)


def test_unexpected_state_dict_key_raises(tmp_path: Path, enc: SimpleRNAEncoder):
    ckpt = _valid_ckpt_dict(enc)
    rna = dict(ckpt["rna_encoder"])
    rna["extra.weight"] = torch.zeros(1)
    ckpt["rna_encoder"] = rna
    p = tmp_path / "unexpected_key.pt"
    torch.save(ckpt, p)

    with pytest.raises(CheckpointSchemaError) as excinfo:
        load_pretrained_simple_rna_encoder(p, expected_schema_version=1)
    assert "extra.weight" in str(excinfo.value)


def test_missing_top_level_key_raises(tmp_path: Path, enc: SimpleRNAEncoder):
    ckpt = _valid_ckpt_dict(enc)
    del ckpt["pretrain_head_class"]
    p = tmp_path / "missing_top.pt"
    torch.save(ckpt, p)

    with pytest.raises(CheckpointSchemaError, match="pretrain_head_class"):
        load_pretrained_simple_rna_encoder(p, expected_schema_version=1)
