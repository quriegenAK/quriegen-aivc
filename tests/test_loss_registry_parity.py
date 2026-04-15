"""
Parity test: refactored combined_loss_multimodal (registry-delegated) vs
an inline reference implementation that replicates the pre-refactor math.

Must match to atol=1e-7 for the total and every component of the breakdown.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Ensure repo root on path so `losses` and `aivc` resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from losses import (
    combined_loss_multimodal,
    log_fold_change_loss,
    cosine_loss,
    causal_ordering_loss,
)


# -- Reference implementation: copy of pre-refactor combined_loss_multimodal.
# Intentionally NOT calling any registry code so the test is an independent
# oracle rather than a tautology.
def _reference_combined_loss_multimodal(
    predicted,
    actual_stim,
    actual_ctrl,
    neumann_module=None,
    emb_pairs=None,
    contrastive_loss_fn=None,
    cross_modal_fn=None,
    attn_weights=None,
    alpha=1.0,
    beta=0.1,
    gamma=0.1,
    lambda_contrast=0.05,
    lambda_cross=0.05,
    lambda_causal=0.1,
):
    mse = F.mse_loss(predicted, actual_stim)
    lfc = log_fold_change_loss(predicted, actual_stim, actual_ctrl)
    cos = cosine_loss(predicted, actual_stim)

    base = alpha * mse + beta * lfc + gamma * cos
    l1_val = 0.0
    if neumann_module is not None:
        l1 = neumann_module.l1_penalty()
        base = base + l1
        l1_val = l1.item()

    contrastive_term = torch.tensor(0.0, device=predicted.device)
    if emb_pairs is not None and contrastive_loss_fn is not None:
        for a, b in emb_pairs:
            contrastive_term = contrastive_term + contrastive_loss_fn(a, b)
        if len(emb_pairs) > 0:
            contrastive_term = contrastive_term / len(emb_pairs)

    cross_modal_term = torch.tensor(0.0, device=predicted.device)
    if emb_pairs is not None and cross_modal_fn is not None:
        for a, b in emb_pairs:
            cross_modal_term = cross_modal_term + cross_modal_fn(a, b)["loss"]
        if len(emb_pairs) > 0:
            cross_modal_term = cross_modal_term / len(emb_pairs)

    causal_term = torch.tensor(0.0, device=predicted.device)
    if attn_weights is not None:
        causal_term = causal_ordering_loss(attn_weights)

    total = (
        base
        + lambda_contrast * contrastive_term
        + lambda_cross * cross_modal_term
        + lambda_causal * causal_term
    )

    return total, {
        "mse": mse.item(),
        "lfc": lfc.item(),
        "cosine": cos.item(),
        "l1": l1_val,
        "contrastive": contrastive_term.item(),
        "cross_modal": cross_modal_term.item(),
        "causal": causal_term.item(),
        "total": total.item(),
    }


# -- Synthetic test helpers --------------------------------------------------

class _StubNeumann:
    def __init__(self, W):
        self.W = W

    def l1_penalty(self):
        return 0.001 * self.W.abs().sum()


class _StubContrastive:
    def __call__(self, a, b):
        return ((a - b) ** 2).mean()


class _StubCrossModal:
    def __call__(self, a, b):
        return {"loss": (a * b).sum(dim=-1).mean()}


def _make_batch(seed=0):
    torch.manual_seed(seed)
    batch, n_genes = 4, 32
    ctrl = torch.rand(batch, n_genes) * 2.0
    stim = ctrl + torch.randn(batch, n_genes) * 0.3
    stim = stim.clamp(min=0.0)
    pred = stim + torch.randn_like(stim) * 0.1
    pred = pred.clamp(min=0.0)

    # Multi-modal extras
    emb_pairs = [
        (torch.randn(batch, 8), torch.randn(batch, 8)),
        (torch.randn(batch, 8), torch.randn(batch, 8)),
    ]
    attn = torch.softmax(torch.randn(batch, 2, 4, 4), dim=-1)
    W = torch.randn(n_genes, n_genes) * 0.1
    return {
        "predicted": pred,
        "actual_stim": stim,
        "actual_ctrl": ctrl,
        "neumann_module": _StubNeumann(W),
        "emb_pairs": emb_pairs,
        "contrastive_loss_fn": _StubContrastive(),
        "cross_modal_fn": _StubCrossModal(),
        "attn_weights": attn,
    }


# -- Parity cases ------------------------------------------------------------

PARITY_CASES = [
    pytest.param("full",        True,  True,  True,  True,  id="full_multimodal"),
    pytest.param("rna_only",    False, False, False, False, id="rna_only_no_extras"),
    pytest.param("neumann_only",True,  False, False, False, id="neumann_l1_only"),
    pytest.param("contrast",    False, True,  False, False, id="contrastive_only"),
    pytest.param("cross",       False, False, True,  False, id="cross_modal_only"),
    pytest.param("causal",      False, False, False, True,  id="causal_only"),
]


@pytest.mark.parametrize("label,use_neumann,use_contrast,use_cross,use_causal", PARITY_CASES)
def test_parity(label, use_neumann, use_contrast, use_cross, use_causal):
    b = _make_batch(seed=123)
    kwargs = dict(
        predicted=b["predicted"],
        actual_stim=b["actual_stim"],
        actual_ctrl=b["actual_ctrl"],
        neumann_module=b["neumann_module"] if use_neumann else None,
        emb_pairs=b["emb_pairs"] if (use_contrast or use_cross) else None,
        contrastive_loss_fn=b["contrastive_loss_fn"] if use_contrast else None,
        cross_modal_fn=b["cross_modal_fn"] if use_cross else None,
        attn_weights=b["attn_weights"] if use_causal else None,
    )

    ref_total, ref_bd = _reference_combined_loss_multimodal(**kwargs)
    new_total, new_bd = combined_loss_multimodal(**kwargs)

    # Same key set
    assert set(ref_bd.keys()) == set(new_bd.keys()), (
        f"[{label}] breakdown keys drifted: "
        f"ref={sorted(ref_bd)} vs new={sorted(new_bd)}"
    )

    # Per-component parity
    for k in ref_bd:
        assert abs(ref_bd[k] - new_bd[k]) < 1e-7, (
            f"[{label}] component '{k}' diverged: "
            f"ref={ref_bd[k]:.10f} vs new={new_bd[k]:.10f}"
        )

    # Total tensor parity (keeps gradient graph)
    assert abs(ref_total.item() - new_total.item()) < 1e-7, (
        f"[{label}] total diverged: {ref_total.item()} vs {new_total.item()}"
    )
