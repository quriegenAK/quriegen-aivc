"""Unit tests for fusion.py modality_mask support.

Ensures absent modalities (mask=0) do not contribute to or receive
attention, and the output is deterministic regardless of the values
of absent-modality tensors.
"""
import torch
import pytest

from aivc.skills.fusion import TemporalCrossModalFusion


def _make_inputs(batch=2, seed=0):
    torch.manual_seed(seed)
    return {
        "rna_emb":     torch.randn(batch, 128),
        "protein_emb": torch.randn(batch, 128),
        "phospho_emb": torch.randn(batch, 64),
        "atac_emb":    torch.randn(batch, 64),
    }


def test_mask_absent_modality_attn_zero():
    """If a modality is masked out, its attention row AND column are 0."""
    m = TemporalCrossModalFusion()
    m.eval()
    inputs = _make_inputs(batch=2)
    mask = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.bool)
    attn = m.get_attention_weights(**inputs, modality_mask=mask)
    assert torch.allclose(attn[:, :, :, 1], torch.zeros_like(attn[:, :, :, 1])), \
        "Phospho column (key) not fully masked"
    assert torch.allclose(attn[:, :, 1, :], torch.zeros_like(attn[:, :, 1, :])), \
        "Phospho row (query) not fully masked"


def test_output_invariant_to_absent_modality_values():
    """Output unchanged when absent-modality tensor values vary."""
    m = TemporalCrossModalFusion()
    m.eval()
    inputs_a = _make_inputs(batch=2, seed=0)
    inputs_b = {**inputs_a, "phospho_emb": torch.randn(2, 64) * 100}
    mask = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        out_a = m(**inputs_a, modality_mask=mask)
        out_b = m(**inputs_b, modality_mask=mask)
    assert torch.allclose(out_a, out_b, atol=1e-5), \
        "Output depends on masked-out modality values — mask is leaking"


def test_no_mask_backward_compat():
    """Default modality_mask=None preserves pre-hotfix behavior."""
    m = TemporalCrossModalFusion()
    m.eval()
    inputs = _make_inputs(batch=2)
    with torch.no_grad():
        out_no_mask = m(**inputs)
        out_all_present = m(**inputs, modality_mask=torch.ones(2, 4, dtype=torch.bool))
    assert torch.allclose(out_no_mask, out_all_present, atol=1e-6), \
        "modality_mask=None differs from modality_mask=all-ones"


def test_mask_shape_validation():
    """Wrong shape raises ValueError."""
    m = TemporalCrossModalFusion()
    inputs = _make_inputs(batch=2)
    bad = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="modality_mask shape"):
        m(**inputs, modality_mask=bad)


def test_dogma_scenario_phospho_absent():
    """DOGMA case: RNA+ATAC+Protein present, Phospho absent."""
    m = TemporalCrossModalFusion()
    m.eval()
    inputs = _make_inputs(batch=4)
    dogma_mask = torch.tensor([[1, 0, 1, 1]] * 4, dtype=torch.bool)
    with torch.no_grad():
        out = m(**inputs, modality_mask=dogma_mask)
    assert out.shape == (4, 384), f"unexpected output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
