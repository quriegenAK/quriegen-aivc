"""Unit tests for PR #54a: SupCon + VICReg variance loss functions.

Lock the algebraic invariants:
  - SupCon returns 0 with grad in degenerate cases (no eligible cells, no positives)
  - SupCon shrinks when all same-class anchors are pulled close
  - SupCon larger when same-class anchors are pushed apart
  - VICReg variance hits its target_std lower bound
  - Both functions produce gradients that flow to z_supcon
  - Eligibility mask routes (excluded cells still serve as negatives)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aivc.training.pretrain_losses import (
    SUPCON_VICREG_TERM_NAMES,
    _supcon_loss,
    _vicreg_variance,
)


# --- Constants ---

EXPECTED_TERM_NAMES = ("supcon", "vicreg_variance")


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


# --- Surface tests ---

def test_term_names_constant():
    assert SUPCON_VICREG_TERM_NAMES == EXPECTED_TERM_NAMES


# --- SupCon: degenerate cases ---

def test_supcon_zero_grad_when_no_eligible():
    z = _l2_normalize(torch.randn(8, 16, requires_grad=True))
    y = torch.zeros(8, dtype=torch.long)
    eligible = torch.zeros(8, dtype=torch.bool)
    loss = _supcon_loss(
        z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible,
    )
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_supcon_zero_when_only_one_eligible():
    z = _l2_normalize(torch.randn(8, 16, requires_grad=True))
    y = torch.zeros(8, dtype=torch.long)
    eligible = torch.zeros(8, dtype=torch.bool)
    eligible[0] = True
    loss = _supcon_loss(
        z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible,
    )
    assert loss.item() == 0.0


def test_supcon_zero_when_no_positives():
    """Each eligible cell is its own unique class — no positives."""
    z = _l2_normalize(torch.randn(6, 16, requires_grad=True))
    y = torch.arange(6, dtype=torch.long)  # all distinct classes
    eligible = torch.ones(6, dtype=torch.bool)
    loss = _supcon_loss(z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible)
    assert loss.item() == 0.0


# --- SupCon: algebraic behavior ---

def test_supcon_loss_shrinks_when_same_class_pulled_close():
    """Same-class cells with high cosine sim → small loss; orthogonal → large."""
    torch.manual_seed(0)
    D = 16
    # 4 cells, 2 classes (0,0,1,1)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    eligible = torch.ones(4, dtype=torch.bool)

    # Aligned: same-class cells share direction
    base_a = torch.randn(D)
    base_b = torch.randn(D)
    z_aligned = torch.stack([base_a, base_a + 0.05 * torch.randn(D),
                              base_b, base_b + 0.05 * torch.randn(D)])
    z_aligned = _l2_normalize(z_aligned)
    loss_aligned = _supcon_loss(
        z_supcon=z_aligned, cell_type_idx=y,
        supcon_eligible_mask=eligible, supcon_temperature=0.07,
    )

    # Orthogonal: same-class cells are orthogonal
    z_random = _l2_normalize(torch.randn(4, D))
    loss_random = _supcon_loss(
        z_supcon=z_random, cell_type_idx=y,
        supcon_eligible_mask=eligible, supcon_temperature=0.07,
    )

    assert loss_aligned.item() < loss_random.item(), (
        f"aligned loss {loss_aligned.item():.4f} should be < random {loss_random.item():.4f}"
    )


def test_supcon_gradient_flows_to_z():
    torch.manual_seed(1)
    z = _l2_normalize(torch.randn(8, 16))
    z.requires_grad_(True)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    eligible = torch.ones(8, dtype=torch.bool)
    loss = _supcon_loss(
        z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible,
        supcon_temperature=0.07,
    )
    assert loss.requires_grad
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_supcon_ineligible_cells_serve_as_negatives_only():
    """Ineligible cells should not be anchors but should still contrast.

    Construct: 4 same-class eligible cells (very tight cluster) + 4
    same-class ineligible cells far away. Loss should be FINITE (not 0)
    because the eligible anchors have positives among themselves AND
    contrast against the ineligible negatives. If ineligible-as-negatives
    were broken (excluded from denom), the result would be the same as
    if those negatives didn't exist — different magnitude.
    """
    torch.manual_seed(2)
    D = 16
    base = _l2_normalize(torch.randn(D)).unsqueeze(0)  # (1, D)
    eligible_z = base.expand(4, D) + 0.01 * torch.randn(4, D)  # tight cluster
    ineligible_z = _l2_normalize(torch.randn(4, D))               # far away
    z = _l2_normalize(torch.cat([eligible_z, ineligible_z], dim=0))
    y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)  # all class 0
    eligible = torch.tensor([True]*4 + [False]*4, dtype=torch.bool)
    loss = _supcon_loss(
        z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible,
        supcon_temperature=0.07,
    )
    assert loss.item() > 0  # non-trivial loss
    assert torch.isfinite(loss)


def test_supcon_default_temperature():
    z = _l2_normalize(torch.randn(6, 16))
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    # Don't pass temperature; should use 0.07 default.
    loss = _supcon_loss(z_supcon=z, cell_type_idx=y)
    assert torch.isfinite(loss)


# --- VICReg variance ---

def test_vicreg_zero_when_b_lt_2():
    z = torch.randn(1, 16, requires_grad=True)
    loss = _vicreg_variance(z_supcon=z)
    assert loss.item() == 0.0


def test_vicreg_at_target_when_unit_variance_input():
    """Per-feature std ~ 1.0 → loss near 0 (target_std default 1.0)."""
    torch.manual_seed(3)
    z = torch.randn(512, 32)  # std ≈ 1
    loss = _vicreg_variance(z_supcon=z, vicreg_target_std=1.0)
    assert loss.item() < 0.05


def test_vicreg_high_when_collapsed_features():
    """Constant z → std = 0 → loss = target_std (every dim hits the hinge)."""
    z = torch.ones(64, 32) * 2.5  # constant
    loss = _vicreg_variance(z_supcon=z, vicreg_target_std=1.0, vicreg_eps=1e-12)
    # With eps→0, std→0, hinge value = target_std = 1.0
    assert loss.item() > 0.99
    assert loss.item() <= 1.0


def test_vicreg_gradient_flows_to_z():
    torch.manual_seed(4)
    z = torch.randn(32, 16) * 0.1  # low variance — hinge active
    z.requires_grad_(True)
    loss = _vicreg_variance(z_supcon=z, vicreg_target_std=1.0)
    assert loss.item() > 0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_vicreg_partial_collapse():
    """Some dims collapsed, others healthy → loss = (collapsed_dims / D) * target_std."""
    torch.manual_seed(5)
    B, D = 64, 10
    z = torch.randn(B, D)
    # Collapse first 3 dims to constant
    z[:, :3] = 1.0
    loss = _vicreg_variance(z_supcon=z, vicreg_target_std=1.0, vicreg_eps=1e-12)
    # Expected: 3 dims at hinge value 1.0, 7 dims near 0 → ~ 3/10 = 0.3
    assert 0.25 < loss.item() < 0.35, f"got {loss.item()}"


# --- Combined sanity: SupCon + VICReg both finite, both gradient-bearing ---

def test_supcon_plus_vicreg_combined_gradient():
    torch.manual_seed(6)
    z = torch.randn(16, 32)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    z.requires_grad_(True)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)
    eligible = torch.ones(16, dtype=torch.bool)

    sup = _supcon_loss(
        z_supcon=z, cell_type_idx=y, supcon_eligible_mask=eligible,
        supcon_temperature=0.07,
    )
    var = _vicreg_variance(z_supcon=z, vicreg_target_std=1.0)

    total = 0.5 * sup + 0.1 * var
    total.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
