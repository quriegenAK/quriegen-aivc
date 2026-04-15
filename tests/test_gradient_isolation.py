"""
Phase 3 gradient-isolation guard.

The causal parameter matrix `NeumannPropagation.W` must NEVER receive
gradient from an observational (unpaired) batch. Observational data flows
through the `"pretrain"` loss stage, which is required to be a strict safe
subset: no causal_ordering penalty, no MSE-against-stim, no LFC, no term
that references interventional targets.

This test proves the guarantee on a synthetic observational batch: we run
the full model forward at stage="pretrain", compute the registry total,
call backward, and assert that W.grad is either None or exactly zero. We
also assert that the causal_ordering term is absent from the returned
component breakdown.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from perturbation_model import PerturbationPredictor
from aivc.skills.neumann_propagation import NeumannPropagation
from aivc.training.loss_registry import LossRegistry, LossTerm
from losses import causal_ordering_loss, log_fold_change_loss
import torch.nn.functional as F


def _build_model(n_genes: int = 8, n_edges: int = 12, seed: int = 0):
    """Full model at observational-pretrain-ready scale (tiny).

    Attaches a real NeumannPropagation with a learnable W so we can
    inspect W.grad after backward.
    """
    torch.manual_seed(seed)
    model = PerturbationPredictor(
        n_genes=n_genes,
        num_perturbations=2,
        feature_dim=8,
        hidden1_dim=8,
        hidden2_dim=4,
        num_head1=2,
        num_head2=2,
        decoder_hidden=16,
    )
    # Synthetic edge set (src, dst) — fixed topology, nonzero init weights.
    src = torch.randint(0, n_genes, (n_edges,))
    dst = torch.randint(0, n_genes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.rand(n_edges) * 500.0  # STRING-scale confidence
    model.neumann = NeumannPropagation(
        n_genes=n_genes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        K=2,
        lambda_l1=0.001,
    )
    return model, edge_index


def _build_registry_mirroring_joint():
    """Registry with the same joint-stage terms losses.combined_loss_multimodal
    registers. None of them are tagged pretrain/joint_safe, so a pretrain
    compute() call MUST return an empty component dict.
    """
    def _mse_fn(predicted, actual_stim, **_):
        return F.mse_loss(predicted, actual_stim)

    def _lfc_fn(predicted, actual_stim, actual_ctrl, **_):
        return log_fold_change_loss(predicted, actual_stim, actual_ctrl)

    def _l1_fn(predicted, neumann_module=None, **_):
        if neumann_module is not None:
            return neumann_module.l1_penalty()
        return torch.tensor(0.0, device=predicted.device)

    def _causal_fn(predicted, attn_weights=None, **_):
        if attn_weights is not None:
            return causal_ordering_loss(attn_weights)
        return torch.tensor(0.0, device=predicted.device)

    reg = LossRegistry()
    reg.register(LossTerm("mse",             _mse_fn,    1.0, "joint"))
    reg.register(LossTerm("lfc",             _lfc_fn,    0.1, "joint"))
    reg.register(LossTerm("l1",              _l1_fn,     1.0, "joint"))
    reg.register(LossTerm("causal_ordering", _causal_fn, 0.1, "joint"))
    return reg


def test_pretrain_stage_leaks_no_gradient_into_neumann_W():
    """Observational batch + stage=pretrain → W.grad is None or all zeros."""
    model, edge_index = _build_model()
    n_genes = model.n_genes

    # Synthetic observational batch: no ctrl/stim pair. We have only an
    # "x" expression vector. Simulate by running ctrl-like expression
    # through forward_batch (the model doesn't know it's observational;
    # the *loss stage* is what gates gradient to W).
    torch.manual_seed(1)
    batch = 3
    x_obs = torch.rand(batch, n_genes)
    pert_id = torch.tensor([0])  # perturbation index doesn't matter at pretrain

    predicted = model.forward_batch(x_obs, edge_index, pert_id)
    assert predicted.requires_grad, "predicted must retain grad for a meaningful test"

    registry = _build_registry_mirroring_joint()
    total, components = registry.compute(
        stage="pretrain",
        predicted=predicted,
        # Provide interventional-only targets to prove they are IGNORED.
        actual_stim=x_obs.detach(),
        actual_ctrl=x_obs.detach(),
        neumann_module=model.neumann,
        attn_weights=torch.softmax(torch.randn(batch, 2, 4, 4), dim=-1),
    )

    # Assertion 1: causal_ordering MUST NOT be in components.
    assert "causal_ordering" not in components, (
        f"causal_ordering leaked into pretrain components: {components}"
    )
    # And neither should other interventional-target terms.
    assert "mse" not in components
    assert "lfc" not in components
    # In Phase 3 no pretrain term is registered yet → components is empty.
    assert components == {}, f"Expected empty pretrain components, got {components}"

    # Assertion 2: total is a grad-bearing scalar so backward is well-defined.
    assert isinstance(total, torch.Tensor)
    assert total.dim() == 0
    assert total.requires_grad, "pretrain placeholder must preserve the autograd graph"

    # Clear any stale grads, then backward.
    model.zero_grad(set_to_none=True)
    total.backward()

    # Assertion 3: NeumannPropagation.W received no signal.
    W_grad = model.neumann.W.grad
    if W_grad is not None:
        assert torch.allclose(W_grad, torch.zeros_like(W_grad)), (
            f"W.grad leaked nonzero signal under pretrain stage: "
            f"max|grad|={W_grad.abs().max().item():.3e}"
        )


def test_joint_stage_DOES_pull_gradient_into_neumann_W():
    """Counterfactual: under stage="joint", W.grad MUST be nonzero.

    Without this test, the pretrain isolation assertion (W.grad is None or
    allclose(0)) could pass for the wrong reason — namely that W never
    participated in the forward graph at all. This test proves the forward
    path does exercise W, so the pretrain None/zero result is meaningful
    gradient-flow evidence rather than a structural artifact.
    """
    model, edge_index = _build_model(seed=2)
    n_genes = model.n_genes

    torch.manual_seed(7)
    batch = 3
    ctrl_b = torch.rand(batch, n_genes)
    stim_b = ctrl_b + torch.randn(batch, n_genes) * 0.3
    pert_id = torch.tensor([1])

    predicted = model.forward_batch(ctrl_b, edge_index, pert_id)
    assert predicted.requires_grad

    registry = _build_registry_mirroring_joint()
    total, components = registry.compute(
        stage="joint",
        predicted=predicted,
        actual_stim=stim_b,
        actual_ctrl=ctrl_b,
        neumann_module=model.neumann,
        attn_weights=None,  # no attention tensor; causal term contributes 0
    )

    # Joint-stage components must include the interventional-target terms
    # that are explicitly excluded from pretrain.
    assert "mse" in components
    assert "lfc" in components

    model.zero_grad(set_to_none=True)
    total.backward()

    W_grad = model.neumann.W.grad
    assert W_grad is not None, (
        "W.grad is None at stage='joint' — forward path does NOT exercise W. "
        "The pretrain isolation test would then pass for the wrong reason; "
        "fix the model wiring before relying on the isolation guarantee."
    )
    assert torch.any(W_grad != 0), (
        f"W.grad is all-zero at stage='joint' — forward path numerically "
        f"bypasses W. max|grad|={W_grad.abs().max().item():.3e}. "
        "The pretrain isolation test is not meaningful until this is fixed."
    )


def test_pretrain_excludes_forbidden_names_even_when_mistagged():
    """Belt-and-braces: if a causal_ordering term is (mis)tagged joint_safe,
    the registry must still refuse to run it at pretrain."""
    def _evil_causal_fn(predicted, **_):
        # Would pull gradient through `predicted` → decoder → genelink → ...
        # We only need a term that would produce nonzero output; the
        # registry should never call it.
        raise AssertionError(
            "forbidden term was invoked at pretrain stage — isolation broken"
        )

    reg = LossRegistry()
    reg.register(LossTerm("causal_ordering", _evil_causal_fn, 1.0, "joint_safe"))

    predicted = torch.randn(2, 4, requires_grad=True)
    total, components = reg.compute(stage="pretrain", predicted=predicted)

    assert components == {}
    assert "causal_ordering" not in components
    # Backward is still safe.
    total.backward()
