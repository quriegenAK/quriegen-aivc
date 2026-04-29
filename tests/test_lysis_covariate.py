"""Tests for lysis_protocol covariate threading in encoders (logical PR #43).

Adapted from PROMPT 43 spec: ProteinEncoder uses ``embed_dim`` (not
``attn_dim`` as drafted in the prompt) — matches the existing on-main API.
"""
from __future__ import annotations

import torch

from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
from aivc.skills.protein_encoder import ProteinEncoder
from aivc.skills.rna_encoder import SimpleRNAEncoder


# --- RNA encoder ---

def test_rna_encoder_no_covariate_back_compat():
    """n_lysis_categories=0 (default): forward unchanged, no embedding allocated."""
    enc = SimpleRNAEncoder(n_genes=100, latent_dim=32)
    assert enc.lysis_emb is None
    x = torch.rand(4, 100)
    z, recon = enc(x)
    assert z.shape == (4, 32)
    assert recon.shape == (4, 100)


def test_rna_encoder_covariate_changes_output():
    """Same input, different lysis_idx -> different output."""
    enc = SimpleRNAEncoder(n_genes=100, latent_dim=32, n_lysis_categories=2)
    x = torch.rand(4, 100)
    idx_a = torch.zeros(4, dtype=torch.long)
    idx_b = torch.ones(4, dtype=torch.long)
    z_a, _ = enc(x, lysis_idx=idx_a)
    z_b, _ = enc(x, lysis_idx=idx_b)
    assert z_a.shape == z_b.shape == (4, 32)
    # Outputs differ when covariate differs
    assert not torch.allclose(z_a, z_b, atol=1e-4)


def test_rna_encoder_covariate_consistent_with_same_idx():
    """Same input + same lysis_idx (deterministic init) -> same output."""
    torch.manual_seed(0)
    enc = SimpleRNAEncoder(n_genes=50, latent_dim=16, n_lysis_categories=2)
    enc.eval()
    x = torch.rand(3, 50)
    idx = torch.tensor([0, 1, 0], dtype=torch.long)
    with torch.no_grad():
        z1, _ = enc(x, lysis_idx=idx)
        z2, _ = enc(x, lysis_idx=idx)
    torch.testing.assert_close(z1, z2)


# --- ATAC encoder ---

def test_atac_encoder_no_covariate_back_compat():
    enc = PeakLevelATACEncoder(n_peaks=200, attn_dim=16)
    assert enc.lysis_emb is None
    x = torch.rand(4, 200)
    z = enc(x)
    assert z.shape == (4, 16)


def test_atac_encoder_covariate_changes_output():
    enc = PeakLevelATACEncoder(n_peaks=200, attn_dim=16, n_lysis_categories=2)
    x = torch.rand(4, 200)
    z_a = enc(x, lysis_idx=torch.zeros(4, dtype=torch.long))
    z_b = enc(x, lysis_idx=torch.ones(4, dtype=torch.long))
    assert not torch.allclose(z_a, z_b, atol=1e-4)


# --- Protein encoder ---

def test_protein_encoder_no_covariate_back_compat():
    """Existing ProteinEncoder uses embed_dim= (not attn_dim= per prompt draft).

    n_lysis_categories=0 default keeps forward signature backward-compatible
    so existing callers (multimodal_smoke_test.py, 13 prior tests) still work.
    """
    enc = ProteinEncoder(n_proteins=20, embed_dim=8)
    assert enc.lysis_emb is None
    assert enc.lysis_to_embed is None
    x = torch.rand(4, 20)
    z = enc(x)
    assert z.shape == (4, 8)


def test_protein_encoder_covariate_changes_output():
    enc = ProteinEncoder(n_proteins=20, embed_dim=8, n_lysis_categories=2)
    enc.eval()  # disable dropout for deterministic comparison
    x = torch.rand(4, 20)
    with torch.no_grad():
        z_a = enc(x, lysis_idx=torch.zeros(4, dtype=torch.long))
        z_b = enc(x, lysis_idx=torch.ones(4, dtype=torch.long))
    assert not torch.allclose(z_a, z_b, atol=1e-4)


# --- Backward gradient flow ---

def test_rna_covariate_gradients_flow_to_lysis_embedding():
    """Lysis embedding must receive gradients during backward (RNA)."""
    enc = SimpleRNAEncoder(n_genes=50, latent_dim=16, n_lysis_categories=2)
    x = torch.rand(4, 50)
    idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    z, _ = enc(x, lysis_idx=idx)
    z.sum().backward()
    assert enc.lysis_emb.weight.grad is not None
    assert enc.lysis_emb.weight.grad.abs().sum() > 0


def test_atac_covariate_gradients_flow_to_lysis_embedding():
    """ATAC: cov is concat'd into MLP path; embedding must receive grads."""
    enc = PeakLevelATACEncoder(n_peaks=80, attn_dim=8, n_lysis_categories=2)
    x = torch.rand(4, 80)
    idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    z = enc(x, lysis_idx=idx)
    z.sum().backward()
    assert enc.lysis_emb.weight.grad is not None
    assert enc.lysis_emb.weight.grad.abs().sum() > 0


def test_protein_covariate_gradients_flow_to_lysis_embedding():
    """Protein: cov is additive shift; embedding + projection both grad.

    Loss must NOT be ``.sum()`` here: the encoder ends with
    ``LayerNorm(embed_dim)``, and ``LayerNorm(x).sum(dim=-1)`` is
    identically 0 (mean-centering). With ``z.sum()``, gradient w.r.t.
    every upstream parameter is exactly 0 — not a real bug, just a
    quirk of summing across a mean-centered dim. Use ``z.pow(2).sum()``
    (L2) so the loss surface depends non-trivially on the lysis
    contribution.
    """
    enc = ProteinEncoder(n_proteins=20, embed_dim=8, n_lysis_categories=2)
    x = torch.rand(4, 20)
    idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    z = enc(x, lysis_idx=idx)
    z.pow(2).sum().backward()
    assert enc.lysis_emb.weight.grad is not None
    assert enc.lysis_emb.weight.grad.abs().sum() > 0
    assert enc.lysis_to_embed.weight.grad is not None
    assert enc.lysis_to_embed.weight.grad.abs().sum() > 0
