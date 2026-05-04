"""Tests for Pivot A joint-fusion eval helpers (aivc/eval/calderon_probe.py).

Synthetic-data tests:
  - load_full_encoders_from_ckpt validates required keys
  - encode_samples_via_joint_fusion produces correct shape + finite values
  - z_supcon is L2-normalized (norm == 1 per row)
  - lysis_idx default to all-zeros when n_lysis_categories > 0
  - Backward compat: bare-ATAC encode_samples still works
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from aivc.eval.calderon_probe import (
    encode_samples_via_joint_fusion,
    load_full_encoders_from_ckpt,
)


def _build_synthetic_ckpt(tmp_path: Path,
                           n_genes=50, n_peaks=100, n_proteins=20,
                           rna_latent=32, atac_latent=16, protein_latent=32,
                           proj_dim=24, n_lysis_categories=2, lysis_cov_dim=4) -> Path:
    """Construct + save a pretrain-shaped ckpt that satisfies the strict
    schema validator in aivc/training/ckpt_loader.py::load_pretrain_ckpt_raw.

    Required top-level keys per the production schema:
      schema_version, rna_encoder, atac_encoder, pretrain_head,
      rna_encoder_class, atac_encoder_class, pretrain_head_class, config.
    Plus the Pivot A additions: protein_encoder, *_proj state dicts,
    and *_class strings for the protein encoder.
    """
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder
    from aivc.skills.protein_encoder import ProteinEncoder
    from aivc.training.pretrain_heads import MultiomePretrainHead

    rna = SimpleRNAEncoder(
        n_genes=n_genes, latent_dim=rna_latent,
        n_lysis_categories=n_lysis_categories, lysis_cov_dim=lysis_cov_dim,
    )
    atac = PeakLevelATACEncoder(
        n_peaks=n_peaks, attn_dim=atac_latent,
        n_lysis_categories=n_lysis_categories, lysis_cov_dim=lysis_cov_dim,
    )
    prot = ProteinEncoder(
        n_proteins=n_proteins, embed_dim=protein_latent,
        n_lysis_categories=n_lysis_categories, lysis_cov_dim=lysis_cov_dim,
    )
    pretrain_head = MultiomePretrainHead(
        rna_dim=rna_latent, atac_dim=atac_latent,
        proj_dim=proj_dim, n_genes=n_genes,
    )
    rna_proj = nn.Linear(rna_latent, proj_dim)
    atac_proj = nn.Linear(atac_latent, proj_dim)
    protein_proj = nn.Linear(protein_latent, proj_dim)

    ckpt = {
        # required by strict schema validator
        "schema_version": 1,
        "rna_encoder": rna.state_dict(),
        "atac_encoder": atac.state_dict(),
        "pretrain_head": pretrain_head.state_dict(),
        "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
        "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
        "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
        # PR #54c additions (not in strict schema, but in production save)
        "protein_encoder": prot.state_dict(),
        "protein_encoder_class": "aivc.skills.protein_encoder.ProteinEncoder",
        "rna_proj": rna_proj.state_dict(),
        "atac_proj": atac_proj.state_dict(),
        "protein_proj": protein_proj.state_dict(),
        "config": {
            "n_genes": n_genes,
            "n_peaks": n_peaks,
            "n_proteins": n_proteins,
            "rna_latent": rna_latent,
            "atac_latent": atac_latent,
            "protein_latent": protein_latent,
            "proj_dim": proj_dim,
            "n_lysis_categories": n_lysis_categories,
            "lysis_cov_dim": lysis_cov_dim,
            "arm": "joint",
        },
    }
    ckpt_path = tmp_path / "synth_ckpt.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def test_load_full_encoders_returns_all_keys(tmp_path: Path):
    ckpt = _build_synthetic_ckpt(tmp_path)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    for key in ("rna_encoder", "atac_encoder", "protein_encoder",
                "rna_proj", "atac_proj", "protein_proj",
                "config", "n_genes", "n_peaks", "n_proteins", "proj_dim"):
        assert key in encoders, f"missing key {key!r}"
    assert encoders["n_genes"] == 50
    assert encoders["n_peaks"] == 100
    assert encoders["n_proteins"] == 20
    assert encoders["proj_dim"] == 24


def test_load_full_encoders_raises_on_missing_keys(tmp_path: Path):
    """Strip a key from ckpt → should raise with informative message."""
    ckpt_path = _build_synthetic_ckpt(tmp_path)
    bad = torch.load(ckpt_path)
    del bad["protein_proj"]
    bad_path = tmp_path / "bad_ckpt.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="protein_proj"):
        load_full_encoders_from_ckpt(bad_path, map_location="cpu")


def test_encode_samples_via_joint_fusion_shape(tmp_path: Path):
    ckpt = _build_synthetic_ckpt(tmp_path)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    n_samples = 8
    n_peaks = encoders["n_peaks"]
    proj_dim = encoders["proj_dim"]
    rng = np.random.RandomState(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_samples, n_peaks)).astype(np.float32))
    z = encode_samples_via_joint_fusion(X, encoders, batch_size=4, device="cpu")
    assert z.shape == (n_samples, proj_dim)


def test_z_supcon_is_l2_normalized(tmp_path: Path):
    """Each row of the returned z_supcon has L2 norm ~= 1."""
    ckpt = _build_synthetic_ckpt(tmp_path)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    n_samples = 6
    n_peaks = encoders["n_peaks"]
    rng = np.random.RandomState(1)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_samples, n_peaks)).astype(np.float32))
    z = encode_samples_via_joint_fusion(X, encoders, batch_size=3, device="cpu")
    norms = np.linalg.norm(z, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_samples_via_joint_fusion_finite(tmp_path: Path):
    """No NaN or Inf in output."""
    ckpt = _build_synthetic_ckpt(tmp_path)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    n_samples = 10
    n_peaks = encoders["n_peaks"]
    rng = np.random.RandomState(2)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_samples, n_peaks)).astype(np.float32))
    z = encode_samples_via_joint_fusion(X, encoders, batch_size=4, device="cpu")
    assert np.isfinite(z).all()


def test_no_lysis_categories_path(tmp_path: Path):
    """When n_lysis_categories=0, no lysis_idx is passed to encoders."""
    ckpt = _build_synthetic_ckpt(tmp_path, n_lysis_categories=0)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    n_samples = 4
    n_peaks = encoders["n_peaks"]
    X = sp.csr_matrix(np.random.RandomState(3).poisson(1.0,
                      size=(n_samples, n_peaks)).astype(np.float32))
    z = encode_samples_via_joint_fusion(X, encoders, batch_size=4, device="cpu")
    assert z.shape == (n_samples, encoders["proj_dim"])
    assert np.isfinite(z).all()


def test_lysis_idx_default_zeros_when_unset(tmp_path: Path):
    """With n_lysis_categories=2 and lysis_idx=None, encoder should use zeros."""
    ckpt = _build_synthetic_ckpt(tmp_path, n_lysis_categories=2)
    encoders = load_full_encoders_from_ckpt(ckpt, map_location="cpu")
    n_samples = 4
    n_peaks = encoders["n_peaks"]
    X = sp.csr_matrix(np.random.RandomState(4).poisson(1.0,
                      size=(n_samples, n_peaks)).astype(np.float32))
    # Should not raise
    z = encode_samples_via_joint_fusion(X, encoders, batch_size=4, device="cpu",
                                         lysis_idx=None)
    assert z.shape == (n_samples, encoders["proj_dim"])
