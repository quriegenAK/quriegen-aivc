"""Unit tests for aivc.training.perturbation_loss."""
from __future__ import annotations

import pytest
import torch

from aivc.training.perturbation_loss import (
    l_recon,
    l_pathway_consistency,
    l_smooth,
    supcon_loss,
    PerturbationLoss,
)


# ---------- l_recon ----------

def test_l_recon_protein_only():
    """Stage 3a Mimitou: only protein modality present."""
    out = l_recon(
        protein_pred=torch.zeros(4, 38),
        protein_true=torch.ones(4, 38),
    )
    assert "protein" in out and "total" in out
    # MSE(zeros, ones) = 1.0
    assert out["protein"].item() == pytest.approx(1.0)
    assert out["total"].item() == pytest.approx(1.0)
    # Absent modalities not in dict
    assert "rna" not in out
    assert "phospho" not in out


def test_l_recon_all_modalities():
    out = l_recon(
        rna_pred=torch.zeros(4, 10), rna_true=torch.ones(4, 10),
        protein_pred=torch.zeros(4, 5), protein_true=torch.ones(4, 5),
        phospho_pred=torch.zeros(4, 3), phospho_true=torch.ones(4, 3),
    )
    for k in ("rna", "protein", "phospho"):
        assert out[k].item() == pytest.approx(1.0)
    assert out["total"].item() == pytest.approx(3.0)


def test_l_recon_with_weights():
    out = l_recon(
        rna_pred=torch.zeros(4, 10), rna_true=torch.ones(4, 10),
        protein_pred=torch.zeros(4, 5), protein_true=torch.ones(4, 5),
        weights={"rna": 0.5, "protein": 2.0},
    )
    assert out["rna"].item() == pytest.approx(0.5)
    assert out["protein"].item() == pytest.approx(2.0)


def test_l_recon_all_none_raises():
    with pytest.raises(ValueError, match="all modalities are None"):
        l_recon()


# ---------- l_pathway_consistency ----------

def test_pathway_consistency_zero_when_one_modality():
    """Only one modality → no consistency to enforce → returns 0."""
    loss = l_pathway_consistency(rna_pathway=torch.randn(4, 5))
    assert loss.item() == 0.0


def test_pathway_consistency_two_modalities():
    a = torch.zeros(4, 5)
    b = torch.ones(4, 5)
    loss = l_pathway_consistency(rna_pathway=a, protein_pathway=b)
    # MSE(zeros, ones) = 1.0, single pair → mean=1.0
    assert loss.item() == pytest.approx(1.0)


def test_pathway_consistency_three_modalities():
    """3 pathway scores → 3 pairs averaged."""
    a = torch.zeros(4, 5)
    b = torch.zeros(4, 5)
    c = torch.zeros(4, 5)
    loss = l_pathway_consistency(rna_pathway=a, protein_pathway=b, phospho_pathway=c)
    assert loss.item() == 0.0


# ---------- l_smooth ----------

def test_l_smooth_zero_on_constant_trajectory():
    """A trajectory that doesn't change has zero smoothness penalty."""
    B, T, D = 4, 5, 8
    y_traj = torch.ones(B, T, D)  # identical at every t
    times = torch.linspace(0, 180, T)
    assert l_smooth(y_traj, times).item() == 0.0


def test_l_smooth_positive_on_oscillation():
    B, T, D = 2, 4, 4
    y_traj = torch.tensor([[[1.0, 1, 1, 1], [10, 10, 10, 10], [1, 1, 1, 1], [10, 10, 10, 10]]] * 2)
    times = torch.tensor([0.0, 30.0, 60.0, 90.0])
    loss = l_smooth(y_traj, times)
    assert loss.item() > 0


def test_l_smooth_shape_validation():
    with pytest.raises(ValueError, match="l_smooth expects"):
        l_smooth(torch.randn(4, 5), torch.randn(5))   # y_traj wrong dim
    with pytest.raises(ValueError, match="T dim"):
        l_smooth(torch.randn(4, 5, 3), torch.tensor([0.0]))


# ---------- supcon_loss ----------

def test_supcon_shape_validation():
    with pytest.raises(ValueError, match="z must be"):
        supcon_loss(torch.randn(4), torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="labels must be"):
        supcon_loss(torch.randn(4, 8), torch.randn(4))    # float labels


def test_supcon_temperature_validation():
    with pytest.raises(ValueError, match="positive"):
        supcon_loss(torch.randn(4, 8), torch.tensor([0, 1, 0, 1]), temperature=0)


def test_supcon_with_clear_class_structure():
    """Two well-separated clusters → low loss; mixed → high loss."""
    # Clearly separated
    z_clear = torch.cat([
        torch.randn(4, 8) + 5.0,
        torch.randn(4, 8) - 5.0,
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss_clear = supcon_loss(z_clear, labels).item()

    # Mixed (labels random)
    torch.manual_seed(42)
    z_mixed = torch.randn(8, 8)
    loss_mixed = supcon_loss(z_mixed, labels).item()

    # Separated representation should give lower loss
    assert loss_clear < loss_mixed


def test_supcon_no_positives_returns_zero():
    """If every cell has a unique label, no positives → 0 loss (sampler should fix)."""
    z = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])   # all distinct
    loss = supcon_loss(z, labels)
    assert loss.item() == 0.0


def test_supcon_gradient_flow():
    z = torch.randn(8, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loss = supcon_loss(z, labels)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


# ---------- PerturbationLoss composite ----------

def test_composite_loss_stage3a_defaults():
    """Stage 3a-shaped batch: protein recon + zero-arm only."""
    loss_fn = PerturbationLoss(lambda_recon=1.0, lambda_zero=1.0, lambda_pathway=0.5, lambda_smooth=0.0)
    out = loss_fn(
        protein_pred=torch.zeros(4, 38),
        protein_true=torch.ones(4, 38),
        deltas={
            "delta_s":  torch.zeros(4, 64),
            "delta_i":  torch.zeros(4, 64),
            "delta_xy": torch.zeros(4, 64),
        },
        arm_mask={
            "has_stim": torch.tensor([True, False, True, False]),
            "has_inh":  torch.tensor([False, True, True, False]),
        },
    )
    assert "L_recon" in out and "L_zero_arm" in out and "L_pathway" in out and "L_smooth" in out
    assert "total" in out
    # Protein recon (MSE(zeros, ones))*lambda_recon = 1.0
    assert out["L_recon"].item() == pytest.approx(1.0)
    # Zero-arm with zero deltas = 0
    assert out["L_zero_arm"].item() == 0.0
    # Pathway with no inputs = 0
    assert out["L_pathway"].item() == 0.0
    # Smooth disabled
    assert out["L_smooth"].item() == 0.0
    assert out["total"].item() == pytest.approx(1.0)


def test_composite_loss_negative_lambda_rejected():
    with pytest.raises(ValueError, match="lambda_recon"):
        PerturbationLoss(lambda_recon=-1.0)


def test_composite_loss_zero_arm_only_when_inputs_present():
    """Without deltas + arm_mask, L_zero_arm is 0 (silent-zero)."""
    loss_fn = PerturbationLoss(lambda_zero=1.0)
    out = loss_fn(
        protein_pred=torch.zeros(4, 10),
        protein_true=torch.zeros(4, 10),
    )
    assert out["L_zero_arm"].item() == 0.0


def test_composite_loss_pathway_uses_lambda():
    """Pathway loss is scaled by lambda_pathway."""
    loss_fn = PerturbationLoss(lambda_recon=0.0, lambda_zero=0.0, lambda_pathway=2.0)
    out = loss_fn(
        protein_pred=torch.zeros(4, 10),
        protein_true=torch.zeros(4, 10),
        rna_pathway=torch.zeros(4, 5),
        protein_pathway=torch.ones(4, 5),
    )
    # Raw pathway loss = MSE(zeros, ones) = 1.0; * lambda_pathway=2.0 = 2.0
    assert out["L_pathway"].item() == pytest.approx(2.0)


def test_composite_loss_full_backward():
    """End-to-end backward through all components."""
    loss_fn = PerturbationLoss()
    # Create tensors that require gradients
    prot_pred = torch.zeros(4, 10, requires_grad=True)
    delta_s = torch.ones(4, 8, requires_grad=True)
    out = loss_fn(
        protein_pred=prot_pred,
        protein_true=torch.zeros(4, 10),
        deltas={
            "delta_s":  delta_s,
            "delta_i":  torch.zeros(4, 8),
            "delta_xy": torch.zeros(4, 8),
        },
        arm_mask={
            "has_stim": torch.tensor([False, False, False, False]),
            "has_inh":  torch.tensor([False, False, False, False]),
        },
    )
    out["total"].backward()
    assert delta_s.grad is not None and delta_s.grad.abs().sum() > 0
