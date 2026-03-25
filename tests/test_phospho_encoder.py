"""
Tests for PhosphoEncoder (PhosphoNorm + KSGraphAttention).
All mock data, CPU only, under 15 seconds.
"""
import pytest
import torch
import torch.nn as nn

SEED = 42
BATCH = 32
N_SITES = 400


def _make_phospho(batch=BATCH, n_sites=N_SITES):
    torch.manual_seed(SEED)
    return torch.rand(batch, n_sites).abs() * 1e5


def _make_rna_emb(batch=BATCH):
    torch.manual_seed(SEED)
    return torch.randn(batch, 128)


def _make_edges(n_sites=N_SITES, n_edges=500):
    torch.manual_seed(SEED)
    return torch.randint(0, n_sites, (2, n_edges))


class TestPhosphoNorm:

    def test_log_normalization_applied(self):
        """Large values compressed to reasonable range."""
        from aivc.skills.phospho_encoder import PhosphoNorm
        norm = PhosphoNorm()
        # Need batch >= 2 for z-score std computation
        x = torch.tensor([[1e6, 1e3, 1.0], [1e5, 1e2, 0.5]])
        out = norm(x)
        # After log1p + z-score, values should be in reasonable range
        assert not out.isnan().any()
        assert out.abs().max() < 20.0

    def test_z_score_per_site(self):
        """Each site should have mean~0, std~1 across batch."""
        from aivc.skills.phospho_encoder import PhosphoNorm
        norm = PhosphoNorm()
        x = _make_phospho(batch=100)
        out = norm(x)
        site_means = out.mean(dim=0)
        site_stds = out.std(dim=0)
        assert site_means.abs().max() < 0.2, f"Site means not ~0: {site_means.abs().max():.4f}"
        assert (site_stds - 1.0).abs().max() < 0.3, f"Site stds not ~1: max dev {(site_stds-1).abs().max():.4f}"

    def test_handles_zero_intensity_sites(self):
        """All-zero sites handled without NaN."""
        from aivc.skills.phospho_encoder import PhosphoNorm
        norm = PhosphoNorm()
        x = torch.zeros(BATCH, N_SITES)
        out = norm(x)
        assert not out.isnan().any()
        assert not out.isinf().any()


class TestPhosphoEncoder:

    def test_output_shape_without_graph(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        out = enc(_make_phospho(), edge_index=None)
        assert out.shape == (BATCH, 64)

    def test_output_shape_with_ks_graph(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        out = enc(_make_phospho(), edge_index=_make_edges())
        assert out.shape == (BATCH, 64)

    def test_output_differs_with_and_without_graph(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        torch.manual_seed(SEED)
        enc = PhosphoEncoder(N_SITES)
        phospho = _make_phospho()
        enc.eval()
        with torch.no_grad():
            out_with = enc(phospho, edge_index=_make_edges())
            out_without = enc(phospho, edge_index=None)
        assert not torch.allclose(out_with, out_without, atol=1e-3), \
            "Graph attention should change the output"

    def test_no_nan_in_forward_pass(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        out = enc(_make_phospho())
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_no_nan_with_extreme_intensities(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        x = torch.rand(BATCH, N_SITES) * 1e8
        out = enc(x)
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_embed_dim_is_64(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        assert enc.embed_dim == 64

    def test_use_ks_graph_false_skips_graph_layer(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES, use_ks_graph=False)
        assert enc.ks_graph is None
        # With or without edges, output should be identical
        phospho = _make_phospho()
        enc.eval()
        with torch.no_grad():
            out1 = enc(phospho, edge_index=_make_edges())
            out2 = enc(phospho, edge_index=None)
        assert torch.allclose(out1, out2)

    def test_fusion_compatibility(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        from aivc.skills.fusion import TemporalCrossModalFusion
        enc = PhosphoEncoder(N_SITES)
        fusion = TemporalCrossModalFusion()
        rna = _make_rna_emb()
        phos = enc(_make_phospho())
        out = fusion(rna_emb=rna, phospho_emb=phos)
        assert out.shape == (BATCH, 384)

    def test_rna_aligned_embedding_shape(self):
        """get_rna_aligned_embedding returns (batch, 128) when align_to_rna=True."""
        from aivc.skills.phospho_encoder import PhosphoEncoder
        from aivc.skills.contrastive_loss import PairedModalityContrastiveLoss
        enc = PhosphoEncoder(N_SITES, align_to_rna=True)
        phospho = _make_phospho()
        aligned = enc.get_rna_aligned_embedding(phospho)
        assert aligned.shape == (BATCH, 128)
        # Now contrastive loss works (both 128-dim)
        rna = _make_rna_emb()
        cl = PairedModalityContrastiveLoss()
        loss = cl(rna, aligned)
        assert not loss.isnan().any()
        assert loss.item() >= 0


class TestPhosphoEncoderTraining:

    def test_gradients_flow_through_mlp(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES)
        out = enc(_make_phospho())
        loss = out.mean()
        loss.backward()
        for name, p in enc.output_proj.named_parameters():
            assert p.grad is not None, f"No grad for output_proj.{name}"

    def test_gradients_flow_through_ks_graph(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        enc = PhosphoEncoder(N_SITES, use_ks_graph=True)
        out = enc(_make_phospho(), edge_index=_make_edges())
        loss = out.mean()
        loss.backward()
        for name, p in enc.ks_graph.named_parameters():
            assert p.grad is not None, f"No grad for ks_graph.{name}"

    def test_encoder_deterministic_in_eval_mode(self):
        from aivc.skills.phospho_encoder import PhosphoEncoder
        torch.manual_seed(SEED)
        enc = PhosphoEncoder(N_SITES)
        enc.eval()
        phospho = _make_phospho()
        with torch.no_grad():
            out1 = enc(phospho)
            out2 = enc(phospho)
        assert torch.allclose(out1, out2)
