"""Unit tests for aivc.skills.decomposed_readout.

Covers:
  - Forward shape contracts (all 4 head activation combinations)
  - Zero-arm constraint enforcement (zero_arm_loss correctness)
  - Gradient flow per-head (each head receives gradients when active)
  - Time encoding properties
  - Input validation
"""
from __future__ import annotations

import pytest
import torch

from aivc.skills.decomposed_readout import (
    DecomposedReadout,
    sinusoidal_time_encoding,
    zero_arm_loss,
)


# ---------- sinusoidal_time_encoding ----------

def test_time_encoding_shape():
    t = torch.tensor([0.0, 30.0, 60.0, 180.0])  # minutes
    enc = sinusoidal_time_encoding(t, dim=16)
    assert enc.shape == (4, 16)


def test_time_encoding_scalar_promotes():
    t = torch.tensor(0.0)
    enc = sinusoidal_time_encoding(t, dim=16)
    assert enc.shape == (1, 16)


def test_time_encoding_odd_dim_raises():
    with pytest.raises(ValueError, match="even"):
        sinusoidal_time_encoding(torch.tensor([0.0]), dim=15)


def test_time_encoding_at_zero():
    """At t=0, sin → 0 and cos → 1. Encoded vector should be [0...0, 1...1]."""
    enc = sinusoidal_time_encoding(torch.tensor([0.0]), dim=16)
    half = 16 // 2
    assert torch.allclose(enc[0, :half], torch.zeros(half), atol=1e-6)
    assert torch.allclose(enc[0, half:], torch.ones(half), atol=1e-6)


def test_time_encoding_distinguishes_timepoints():
    t = torch.tensor([0.0, 90.0])
    enc = sinusoidal_time_encoding(t, dim=16)
    # Different timepoints must produce different encodings
    cos_sim = torch.nn.functional.cosine_similarity(enc[0:1], enc[1:2], dim=-1)
    assert cos_sim.item() < 0.99


# ---------- DecomposedReadout ----------

@pytest.fixture
def readout():
    torch.manual_seed(0)
    return DecomposedReadout(d=64, pert_dim=16, output_dim=64, t_enc_dim=8)


@pytest.fixture
def sample_inputs():
    torch.manual_seed(0)
    B = 8
    return {
        "z_dyn":    torch.randn(B, 64),
        "z_static": torch.randn(B, 64),
        "t":        torch.tensor([0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]),
        "stim_emb": torch.randn(B, 16),
        "inh_emb":  torch.randn(B, 16),
    }


def test_forward_baseline_only(readout, sample_inputs):
    """No stim, no inh → only h_b active."""
    out = readout(sample_inputs["z_dyn"], sample_inputs["z_static"],
                  sample_inputs["t"])
    assert out.shape == (8, 64)


def test_forward_with_stim_only(readout, sample_inputs):
    out = readout(sample_inputs["z_dyn"], sample_inputs["z_static"],
                  sample_inputs["t"], stim_emb=sample_inputs["stim_emb"])
    assert out.shape == (8, 64)


def test_forward_with_inh_only(readout, sample_inputs):
    out = readout(sample_inputs["z_dyn"], sample_inputs["z_static"],
                  sample_inputs["t"], inh_emb=sample_inputs["inh_emb"])
    assert out.shape == (8, 64)


def test_forward_with_both(readout, sample_inputs):
    """Both stim and inh → all 4 heads active (h_b + Δ_s + Δ_i + Δ_xy)."""
    out = readout(**sample_inputs)
    assert out.shape == (8, 64)


def test_baseline_only_excludes_delta_heads(readout, sample_inputs):
    """Forward with no stim/inh should equal h_b output alone."""
    out_baseline = readout(sample_inputs["z_dyn"], sample_inputs["z_static"],
                           sample_inputs["t"])
    deltas = readout.head_deltas(**sample_inputs)
    h_b_only = deltas["h_b"]
    assert torch.allclose(out_baseline, h_b_only, atol=1e-6)


def test_both_heads_active_sums_correctly(readout, sample_inputs):
    """With both stim+inh, output = h_b + Δ_s + Δ_i + Δ_xy."""
    out_full = readout(**sample_inputs)
    deltas = readout.head_deltas(**sample_inputs)
    expected = deltas["h_b"] + deltas["delta_s"] + deltas["delta_i"] + deltas["delta_xy"]
    assert torch.allclose(out_full, expected, atol=1e-6)


def test_input_shape_validation(readout, sample_inputs):
    """Wrong-shape inputs raise with diagnostic."""
    bad = sample_inputs.copy()
    bad["z_dyn"] = torch.randn(8, 32)   # wrong d
    with pytest.raises(ValueError, match="z_dyn shape"):
        readout(**bad)

    bad = sample_inputs.copy()
    bad["t"] = torch.tensor([[0.0], [1.0]])  # wrong shape
    with pytest.raises(ValueError, match="t shape"):
        readout(**bad)

    bad = sample_inputs.copy()
    bad["stim_emb"] = torch.randn(8, 8)  # wrong pert_dim
    with pytest.raises(ValueError, match="stim_emb"):
        readout(**bad)


# ---------- zero_arm_loss ----------

def test_zero_arm_loss_zero_when_compliant(readout, sample_inputs):
    """If all Δ_s/Δ_i/Δ_xy are zero, loss is zero."""
    deltas = {
        "delta_s":  torch.zeros(8, 64),
        "delta_i":  torch.zeros(8, 64),
        "delta_xy": torch.zeros(8, 64),
    }
    arm_mask = {
        "has_stim": torch.tensor([False, True, False, True] * 2),
        "has_inh":  torch.tensor([False, False, True, True] * 2),
    }
    loss = zero_arm_loss(deltas, arm_mask)
    assert loss.item() == 0.0


def test_zero_arm_loss_penalizes_inactive_heads():
    """NTC cells (no stim, no inh) should incur Δ_s + Δ_i + Δ_xy penalty."""
    deltas = {
        "delta_s":  torch.ones(4, 16),
        "delta_i":  torch.ones(4, 16),
        "delta_xy": torch.ones(4, 16),
    }
    # All 4 cells are NTC (no stim, no inh) → all 3 deltas penalized fully
    arm_mask = {
        "has_stim": torch.tensor([False, False, False, False]),
        "has_inh":  torch.tensor([False, False, False, False]),
    }
    loss = zero_arm_loss(deltas, arm_mask)
    # Each delta contributes ||ones(16)||² = 16, averaged over 4 cells = 16
    # Total: 16 + 16 + 16 = 48
    assert loss.item() == pytest.approx(48.0)


def test_zero_arm_loss_stim_only_unconstrained_on_delta_s():
    """Stim-only cells: Δ_s is allowed (training signal), only Δ_i + Δ_xy penalized."""
    deltas = {
        "delta_s":  torch.ones(4, 16),
        "delta_i":  torch.ones(4, 16),
        "delta_xy": torch.ones(4, 16),
    }
    arm_mask = {
        "has_stim": torch.tensor([True, True, True, True]),
        "has_inh":  torch.tensor([False, False, False, False]),
    }
    loss = zero_arm_loss(deltas, arm_mask)
    # delta_s: not penalized (has_stim=True) → 0
    # delta_i: penalized (has_inh=False) → 16
    # delta_xy: penalized (not has_stim AND has_inh) → 16
    assert loss.item() == pytest.approx(32.0)


def test_zero_arm_loss_double_perturbation_no_constraint():
    """Stim+inh cells: no constraint on any head."""
    deltas = {
        "delta_s":  torch.ones(4, 16),
        "delta_i":  torch.ones(4, 16),
        "delta_xy": torch.ones(4, 16),
    }
    arm_mask = {
        "has_stim": torch.tensor([True, True, True, True]),
        "has_inh":  torch.tensor([True, True, True, True]),
    }
    loss = zero_arm_loss(deltas, arm_mask)
    assert loss.item() == 0.0


def test_zero_arm_loss_gradient_flows():
    """Loss should produce gradients on the delta tensors."""
    deltas = {
        "delta_s":  torch.ones(4, 16, requires_grad=True),
        "delta_i":  torch.ones(4, 16, requires_grad=True),
        "delta_xy": torch.ones(4, 16, requires_grad=True),
    }
    arm_mask = {
        "has_stim": torch.tensor([False, False, False, False]),
        "has_inh":  torch.tensor([False, False, False, False]),
    }
    loss = zero_arm_loss(deltas, arm_mask)
    loss.backward()
    for k in ("delta_s", "delta_i", "delta_xy"):
        assert deltas[k].grad is not None
        assert deltas[k].grad.abs().sum() > 0


# ---------- End-to-end backward pass ----------

def test_full_backward_with_both_active(readout, sample_inputs):
    """All 4 head MLPs should receive gradients when both stim and inh are active."""
    out = readout(**sample_inputs)
    loss = out.pow(2).sum()
    loss.backward()
    head_names = ["h_b", "delta_s", "delta_i", "delta_xy"]
    for head_name in head_names:
        head = getattr(readout, head_name)
        for pname, p in head.named_parameters():
            assert p.grad is not None, f"no gradient for {head_name}.{pname}"
            assert p.grad.abs().sum() > 0, f"zero gradient for {head_name}.{pname}"


def test_backward_baseline_only_skips_delta_heads(readout, sample_inputs):
    """If only h_b is active (no stim/inh), Δ heads should have None gradients."""
    out = readout(sample_inputs["z_dyn"], sample_inputs["z_static"],
                  sample_inputs["t"])  # no stim_emb, no inh_emb
    loss = out.pow(2).sum()
    loss.backward()
    # h_b should have gradients
    for pname, p in readout.h_b.named_parameters():
        assert p.grad is not None, f"no gradient for h_b.{pname}"
    # Δ heads should be untouched (grad is None)
    for head_name in ("delta_s", "delta_i", "delta_xy"):
        head = getattr(readout, head_name)
        for pname, p in head.named_parameters():
            assert p.grad is None, f"unexpected gradient on {head_name}.{pname}"
