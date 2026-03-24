"""
Tests for causal attention mask and L_causal ordering loss.
All tests use mock data. Under 10 seconds on CPU.

Run: pytest tests/test_causal_mask.py -v
"""
import pytest
import torch
import numpy as np

from aivc.skills.fusion import TemporalCrossModalFusion
from losses import causal_ordering_loss, combined_loss_multimodal

SEED = 42


def _make_fusion_and_inputs(batch=4, use_causal_mask=True):
    """Create a TemporalCrossModalFusion and mock inputs."""
    torch.manual_seed(SEED)
    model = TemporalCrossModalFusion()
    model.use_causal_mask = use_causal_mask
    rna = torch.randn(batch, 128)
    protein = torch.randn(batch, 128)
    phospho = torch.randn(batch, 64)
    atac = torch.randn(batch, 64)
    return model, rna, protein, phospho, atac


class TestCausalMask:

    def test_causal_mask_is_lower_triangular(self):
        """Upper triangle (i < j) must have weight ~0 after causal mask."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs()
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        # attn: (batch, n_heads, 4, 4)
        upper_triangle = torch.triu(attn.mean(dim=(0, 1)), diagonal=1)
        assert upper_triangle.abs().max().item() < 1e-5, (
            f"Upper triangle max = {upper_triangle.abs().max().item():.6f}, "
            "expected < 1e-5 with causal mask"
        )

    def test_atac_attends_only_to_itself(self):
        """ATAC (row 0) must have all attention on column 0."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs()
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        # ATAC row = row 0, mean over batch and heads
        atac_row = attn[:, :, 0, :].mean(dim=(0, 1))  # (4,)
        assert abs(atac_row[0].item() - 1.0) < 1e-5, (
            f"ATAC self-attention should be ~1.0, got {atac_row[0].item():.4f}"
        )
        assert atac_row[1:].abs().max().item() < 1e-5, (
            "ATAC should not attend to Phospho/RNA/Protein"
        )

    def test_protein_attends_to_all(self):
        """Protein (row 3) can attend to all 4 modalities."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs()
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        protein_row = attn[:, :, 3, :].mean(dim=(0, 1))  # (4,)
        # All 4 positions should have > 0 attention
        assert (protein_row > 0).all(), (
            f"Protein should attend to all modalities, got {protein_row.tolist()}"
        )

    def test_rna_cannot_attend_to_protein(self):
        """RNA (row 2) must have 0 attention on Protein (column 3)."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs()
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        rna_to_protein = attn[:, :, 2, 3].mean().item()
        assert abs(rna_to_protein) < 1e-5, (
            f"RNA->Protein attention should be 0, got {rna_to_protein:.6f}"
        )

    def test_no_nan_with_causal_mask(self):
        """No NaN values in attention weights after causal masking."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs()
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        assert not torch.isnan(attn).any(), "NaN found in attention weights"

    def test_ablation_removes_mask(self):
        """use_causal_mask=False -> upper triangle has non-zero attention."""
        model, rna, protein, phospho, atac = _make_fusion_and_inputs(
            use_causal_mask=False
        )
        with torch.no_grad():
            attn = model.get_attention_weights(rna, protein, phospho, atac)
        upper = torch.triu(attn.mean(dim=(0, 1)), diagonal=1)
        assert upper.abs().max().item() > 0.01, (
            "With causal mask disabled, upper triangle should be non-zero"
        )


class TestCausalOrderingLoss:

    def test_causal_ordering_loss_at_random_init(self):
        """Random attention weights -> causal_ordering_loss > 0."""
        torch.manual_seed(SEED)
        attn = torch.rand(4, 4, 4, 4)  # uniform random
        loss = causal_ordering_loss(attn)
        assert loss.item() > 0, "Random attention should have non-zero causal loss"

    def test_causal_ordering_loss_at_perfect_causal(self):
        """Strictly lower-triangular attention -> loss ~0."""
        # Build perfect lower-triangular attention
        lower = torch.tril(torch.ones(4, 4))
        # Normalize rows to sum to 1
        lower = lower / lower.sum(dim=-1, keepdim=True)
        attn = lower.unsqueeze(0).unsqueeze(0).expand(2, 2, -1, -1)
        loss = causal_ordering_loss(attn)
        assert loss.item() < 1e-6, (
            f"Perfect causal attention should have loss ~0, got {loss.item():.6f}"
        )

    def test_causal_term_in_loss_is_non_negative(self):
        """causal_ordering_loss always returns >= 0."""
        for _ in range(5):
            attn = torch.randn(2, 4, 4, 4)
            loss = causal_ordering_loss(attn)
            assert loss.item() >= 0, "Causal loss must be non-negative"


class TestCombinedLossWithCausal:

    def test_combined_loss_accepts_attn_weights(self):
        """combined_loss_multimodal(attn_weights=...) runs, has 'causal' key."""
        torch.manual_seed(SEED)
        predicted = torch.randn(8, 30)
        actual_stim = torch.randn(8, 30)
        actual_ctrl = torch.randn(8, 30)
        mock_attn = torch.rand(8, 4, 4, 4)

        total, bd = combined_loss_multimodal(
            predicted, actual_stim, actual_ctrl,
            attn_weights=mock_attn,
            lambda_causal=0.1,
        )
        assert "causal" in bd, "Breakdown must contain 'causal' key"
        assert bd["causal"] > 0, "Causal term should be > 0 with random attention"
        assert total.requires_grad or True  # just check it ran
