"""
Tests for ProteinEncoder (CLRNorm + cross-attention to RNA).
All mock data, CPU only, under 15 seconds.
"""
import pytest
import torch
import torch.nn as nn

SEED = 42
BATCH = 32
N_PROTEINS = 200


def _make_adt(batch=BATCH, n_proteins=N_PROTEINS):
    torch.manual_seed(SEED)
    return torch.rand(batch, n_proteins).abs() * 100 + 1.0


def _make_rna_emb(batch=BATCH):
    torch.manual_seed(SEED)
    return torch.randn(batch, 128)


class TestCLRNorm:

    def test_clr_output_sums_to_zero_per_cell(self):
        """CLR(x).sum(dim=-1) should be ~0 per cell.

        atol=1e-3: pre-PR #42 was 1e-4 but float32 accumulation drift
        over N_PROTEINS=200 log-values empirically reaches ~1.4e-4 when
        upstream RNG state shifts (passed on prior CI by RNG luck; PR #42
        adds new tests that perturb the order). 1e-3 is still a tight
        property check while no longer rejecting valid float32 output.
        """
        from aivc.skills.protein_encoder import CLRNorm
        clr = CLRNorm()
        x = torch.rand(BATCH, N_PROTEINS).abs() * 100 + 1.0
        out = clr(x)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(BATCH), atol=1e-3), \
            f"CLR row sums should be ~0, got max {row_sums.abs().max():.6f}"

    def test_clr_handles_zero_counts(self):
        """Zero counts use eps floor, no NaN/Inf."""
        from aivc.skills.protein_encoder import CLRNorm
        clr = CLRNorm()
        x = torch.zeros(BATCH, N_PROTEINS)
        out = clr(x)
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_clr_output_shape_preserved(self):
        """Input (32, 200) -> output (32, 200)."""
        from aivc.skills.protein_encoder import CLRNorm
        clr = CLRNorm()
        x = torch.rand(BATCH, N_PROTEINS)
        assert clr(x).shape == (BATCH, N_PROTEINS)


class TestProteinEncoder:

    def test_output_shape_with_rna_alignment(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        out = enc(_make_adt(), rna_emb=_make_rna_emb())
        assert out.shape == (BATCH, 128)

    def test_output_shape_without_rna_alignment(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        out = enc(_make_adt(), rna_emb=None)
        assert out.shape == (BATCH, 128)

    def test_output_differs_with_and_without_rna(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        torch.manual_seed(SEED)
        enc = ProteinEncoder(N_PROTEINS)
        adt = _make_adt()
        rna = _make_rna_emb()
        enc.eval()
        with torch.no_grad():
            out_with = enc(adt, rna_emb=rna)
            out_without = enc(adt, rna_emb=None)
        assert not torch.allclose(out_with, out_without, atol=1e-3), \
            "Cross-attention should change the output"

    def test_no_nan_in_forward_pass(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        out = enc(_make_adt(), rna_emb=_make_rna_emb())
        assert not out.isnan().any()
        assert not out.isinf().any()

    def test_no_nan_with_zero_adt_counts(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        adt = torch.zeros(BATCH, N_PROTEINS)
        out = enc(adt, rna_emb=None)
        assert not out.isnan().any()

    def test_embed_dim_is_128(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        assert enc.embed_dim == 128

    def test_attention_weights_shape(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        adt = _make_adt()
        rna = _make_rna_emb()
        w = enc.get_protein_attention_weights(adt, rna)
        assert w.shape == (BATCH, 1, 1)

    def test_fusion_compatibility(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        from aivc.skills.fusion import TemporalCrossModalFusion
        enc = ProteinEncoder(N_PROTEINS)
        fusion = TemporalCrossModalFusion()
        rna = _make_rna_emb()
        prot = enc(_make_adt(), rna_emb=rna)
        out = fusion(rna_emb=rna, protein_emb=prot)
        assert out.shape == (BATCH, 384)

    def test_contrastive_loss_compatibility(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        from aivc.skills.contrastive_loss import PairedModalityContrastiveLoss
        enc = ProteinEncoder(N_PROTEINS)
        rna = _make_rna_emb()
        prot = enc(_make_adt(), rna_emb=rna)
        cl = PairedModalityContrastiveLoss()
        loss = cl(rna, prot)
        assert not loss.isnan().any()
        assert loss.item() >= 0


class TestProteinEncoderTraining:

    def test_gradients_flow_through_encoder(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        adt = _make_adt()
        out = enc(adt, rna_emb=None)
        loss = out.mean()
        loss.backward()
        for name, p in enc.encoder.named_parameters():
            assert p.grad is not None, f"No grad for encoder.{name}"

    def test_gradients_flow_through_cross_attention(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        enc = ProteinEncoder(N_PROTEINS)
        adt = _make_adt()
        rna = _make_rna_emb()
        out = enc(adt, rna_emb=rna)
        loss = out.mean()
        loss.backward()
        for name, p in enc.cross_attn.named_parameters():
            assert p.grad is not None, f"No grad for cross_attn.{name}"

    def test_encoder_is_deterministic_in_eval_mode(self):
        from aivc.skills.protein_encoder import ProteinEncoder
        torch.manual_seed(SEED)
        enc = ProteinEncoder(N_PROTEINS)
        enc.eval()
        adt = _make_adt()
        with torch.no_grad():
            out1 = enc(adt, rna_emb=None)
            out2 = enc(adt, rna_emb=None)
        assert torch.allclose(out1, out2)
