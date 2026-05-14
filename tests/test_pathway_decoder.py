"""Unit tests for aivc.skills.pathway_decoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aivc.skills.pathway_decoder import (
    PathwayAwareRNADecoder,
    PathwayAwareProteinDecoder,
    PathwayAwarePhosphoDecoder,
    DECODER_CKPT_SCHEMA_VERSION,
    save_decoder,
    load_decoder,
)


# ---------- RNA decoder ----------

def test_rna_decoder_no_pathway_pool():
    """Without pool, decoder returns (gene_logits, None)."""
    dec = PathwayAwareRNADecoder(d=64, n_genes=100, gene_to_pathway_pool=None)
    z = torch.randn(8, 64)
    gene_logits, pathway_scores = dec(z)
    assert gene_logits.shape == (8, 100)
    assert pathway_scores is None


def test_rna_decoder_with_pathway_pool():
    """With pool, decoder returns (gene_logits, pathway_scores)."""
    n_pathways, n_genes = 5, 50
    # Build a synthetic row-normalized pool: each pathway covers 10 genes
    pool = torch.zeros(n_pathways, n_genes)
    for p in range(n_pathways):
        pool[p, p*10:(p+1)*10] = 1.0 / 10
    dec = PathwayAwareRNADecoder(d=32, n_genes=n_genes, gene_to_pathway_pool=pool)
    z = torch.randn(4, 32)
    gene_logits, pathway_scores = dec(z)
    assert gene_logits.shape == (4, n_genes)
    assert pathway_scores.shape == (4, n_pathways)
    # Pathway score for pathway p = mean of gene_logits[i, p*10:(p+1)*10]
    expected = torch.stack([gene_logits[:, p*10:(p+1)*10].mean(dim=1) for p in range(n_pathways)], dim=1)
    assert torch.allclose(pathway_scores, expected, atol=1e-5)


def test_rna_decoder_pool_dim_mismatch_raises():
    pool = torch.zeros(5, 99)  # wrong gene count
    with pytest.raises(ValueError, match="gene_to_pathway_pool"):
        PathwayAwareRNADecoder(d=32, n_genes=100, gene_to_pathway_pool=pool)


def test_rna_decoder_with_hidden_mult():
    """hidden_mult > 0 gives 2-layer MLP head."""
    dec = PathwayAwareRNADecoder(d=64, n_genes=100, hidden_mult=2)
    z = torch.randn(2, 64)
    gene_logits, _ = dec(z)
    assert gene_logits.shape == (2, 100)
    # Verify 2-layer (Linear → GELU → Linear) structure
    assert len(list(dec.head.children())) == 3


def test_rna_decoder_gradient_flow():
    pool = torch.eye(50, 100)[:5] * 0.2  # 5 pathways × first 25 genes
    dec = PathwayAwareRNADecoder(d=32, n_genes=100, gene_to_pathway_pool=pool[:, :100])
    z = torch.randn(8, 32, requires_grad=False)
    gene_logits, pathway_scores = dec(z)
    loss = gene_logits.pow(2).sum() + pathway_scores.pow(2).sum()
    loss.backward()
    for pname, p in dec.head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{pname}: no grad"


# ---------- Protein decoder ----------

def test_protein_decoder_default_no_pool():
    """Mimitou's 38 antibodies → pool defaults None → pathway_scores None."""
    dec = PathwayAwareProteinDecoder(d=64, n_proteins=38)
    z = torch.randn(8, 64)
    expr, pathway_scores = dec(z)
    assert expr.shape == (8, 38)
    assert pathway_scores is None


def test_protein_decoder_with_pool():
    pool = torch.zeros(3, 38)
    pool[0, :10] = 0.1; pool[1, 10:20] = 0.1; pool[2, 20:30] = 0.1
    dec = PathwayAwareProteinDecoder(d=32, n_proteins=38, protein_to_pathway_pool=pool)
    z = torch.randn(4, 32)
    expr, pathway_scores = dec(z)
    assert expr.shape == (4, 38)
    assert pathway_scores.shape == (4, 3)


def test_protein_decoder_pool_dim_mismatch():
    pool = torch.zeros(3, 40)
    with pytest.raises(ValueError, match="protein_to_pathway_pool"):
        PathwayAwareProteinDecoder(d=32, n_proteins=38, protein_to_pathway_pool=pool)


# ---------- Phospho decoder ----------

def test_phospho_decoder_stub_returns_none():
    """n_sites=0 → forward returns (None, None) — Phase 2 not wired yet."""
    dec = PathwayAwarePhosphoDecoder(d=64, n_sites=0)
    z = torch.randn(4, 64)
    out_a, out_b = dec(z)
    assert out_a is None and out_b is None


def test_phospho_decoder_active():
    dec = PathwayAwarePhosphoDecoder(d=32, n_sites=10)
    z = torch.randn(4, 32)
    out_a, out_b = dec(z)
    assert out_a.shape == (4, 10)
    assert out_b is None  # no pool supplied


# ---------- Validation ----------

def test_decoder_input_dim_validation():
    dec = PathwayAwareRNADecoder(d=32, n_genes=100)
    with pytest.raises(ValueError, match="last dim"):
        dec(torch.randn(4, 64))  # wrong d


def test_decoder_zero_dim_rejected():
    with pytest.raises(ValueError, match="positive"):
        PathwayAwareRNADecoder(d=0, n_genes=100)


# ---------- Checkpoint round-trip ----------

def test_rna_decoder_save_load(tmp_path):
    pool = torch.zeros(5, 100)
    for p in range(5):
        pool[p, p*20:(p+1)*20] = 0.05
    dec = PathwayAwareRNADecoder(d=32, n_genes=100, gene_to_pathway_pool=pool)
    ckpt = tmp_path / "rna_decoder.pt"
    save_decoder(dec, ckpt, decoder_kind="rna")
    loaded = load_decoder(
        ckpt, PathwayAwareRNADecoder, n_genes=100, gene_to_pathway_pool=pool,
    )
    assert isinstance(loaded, PathwayAwareRNADecoder)
    # Inputs produce matching outputs
    z = torch.randn(4, 32)
    dec.eval(); loaded.eval()
    a_logits, a_pw = dec(z)
    b_logits, b_pw = loaded(z)
    assert torch.allclose(a_logits, b_logits, atol=1e-5)
    assert torch.allclose(a_pw, b_pw, atol=1e-5)


def test_decoder_save_unknown_kind_raises():
    dec = PathwayAwareRNADecoder(d=32, n_genes=10)
    with pytest.raises(ValueError, match="decoder_kind"):
        save_decoder(dec, "/tmp/x.pt", decoder_kind="invalid")
