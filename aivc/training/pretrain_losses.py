"""
pretrain_losses.py — register the four Phase 5 pretrain-stage loss terms.

Registered terms (all tagged ``stage="pretrain"``):
    - masked_rna_recon      (weight 1.0)
    - masked_atac_recon     (weight 1.0)
    - cross_modal_infonce   (weight 0.5)
    - peak_to_gene_aux      (weight 0.1)

Invariants
----------
- All four terms pass the Phase 3 allow-list in
  ``loss_registry._is_term_active`` (stage == "pretrain" is active
  under the pretrain request stage).
- All four names are OUTSIDE ``loss_registry._PRETRAIN_FORBIDDEN_NAMES``,
  so they will not be defensively skipped at compute() time.
- Registration is pure dict/list insertion — no module-level tensor
  construction, no RNG calls, no dtype/device changes. The Phase 4
  "torch global state byte-identical pre/post import" invariant is
  preserved (tests/test_phase5_no_import_side_effects.py).

This module does NOT modify loss_registry.py internals. The additional
substring guard in ``_guard_pretrain_name`` is a Phase 5 registration-time
defense against a future PR accidentally registering a causal-adjacent
term under stage="pretrain" (synthetic-leak test).

Defense-in-depth invariant
--------------------------
This module enforces forbidden-term rejection at registration time
(fail-fast). The LossRegistry's in-registry forbidden block (Phase 3)
is retained as a runtime backstop. Both must remain in place. Removing
either layer "because the other one is sufficient" is explicitly
forbidden by the Phase 5 PR contract.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from aivc.training.loss_registry import (
    LossRegistry,
    LossTerm,
    _PRETRAIN_FORBIDDEN_NAMES,
)


# Four names registered by this module. Exposed for tests.
PRETRAIN_TERM_NAMES = (
    "masked_rna_recon",
    "masked_atac_recon",
    "cross_modal_infonce",
    "peak_to_gene_aux",
)

# Substrings that indicate a term probably drags gradient through the
# causal head (NeumannPropagation.W) or an interventional target. We
# reject these at registration time for stage="pretrain" so a leak
# cannot reach compute() and silently contaminate W. This is in ADDITION
# to loss_registry._PRETRAIN_FORBIDDEN_NAMES (exact match at compute()).
_PRETRAIN_NAME_SUBSTRING_BLOCK = (
    "causal",
    "ordering",
    "intervention",
    "leak",
    "lfc",
)


def _guard_pretrain_name(name: str) -> None:
    """Raise ValueError if ``name`` looks causal-adjacent.

    Used at registration time by ``register_pretrain_terms``. A silent
    skip at compute() (as done by loss_registry) is fine for defense
    in depth, but for registration we want a loud failure so a buggy
    PR is caught at PR review rather than in production training.
    """
    lowered = name.lower()
    if name in _PRETRAIN_FORBIDDEN_NAMES:
        raise ValueError(
            f"Refusing to register pretrain term {name!r}: appears in "
            f"loss_registry._PRETRAIN_FORBIDDEN_NAMES."
        )
    for bad in _PRETRAIN_NAME_SUBSTRING_BLOCK:
        if bad in lowered:
            raise ValueError(
                f"Refusing to register pretrain term {name!r}: name "
                f"contains forbidden substring {bad!r}. Causal-adjacent "
                f"terms must not be registered under stage='pretrain'."
            )


# ----------------------------------------------------------------------
# Loss functions.
#
# Each accepts **batch and returns a scalar torch.Tensor. They are
# intentionally minimal — the scripts/pretrain_multiome.py entrypoint
# is responsible for populating the batch dict with the tensors each
# fn needs (e.g. rna_recon, rna_target, atac_recon, atac_target, z_rna,
# z_atac, gene_pred, gene_target).
# ----------------------------------------------------------------------
def _masked_rna_recon(**batch) -> torch.Tensor:
    pred = batch["rna_recon"]
    target = batch["rna_target"]
    mask = batch.get("rna_mask")
    if mask is None:
        return F.mse_loss(pred, target)
    # Only score the masked positions. Divide by mask-count to keep
    # scale comparable across batches with different mask densities.
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def _masked_atac_recon(**batch) -> torch.Tensor:
    pred = batch["atac_recon"]
    target = batch["atac_target"]
    mask = batch.get("atac_mask")
    if mask is None:
        return F.mse_loss(pred, target)
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def _cross_modal_infonce(**batch) -> torch.Tensor:
    """Symmetric InfoNCE between L2-normalized rna/atac projections."""
    z_rna = batch["z_rna"]
    z_atac = batch["z_atac"]
    temperature = float(batch.get("infonce_temperature", 0.1))
    logits = z_rna @ z_atac.t() / temperature
    labels = torch.arange(z_rna.shape[0], device=z_rna.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _peak_to_gene_aux(**batch) -> torch.Tensor:
    return F.mse_loss(batch["gene_pred"], batch["gene_target"])


# ----------------------------------------------------------------------
# Registration helper.
# ----------------------------------------------------------------------
_PRETRAIN_SPEC = (
    ("masked_rna_recon", _masked_rna_recon, 1.0),
    ("masked_atac_recon", _masked_atac_recon, 1.0),
    ("cross_modal_infonce", _cross_modal_infonce, 0.5),
    ("peak_to_gene_aux", _peak_to_gene_aux, 0.1),
)


def register_pretrain_terms(registry: LossRegistry) -> None:
    """Insert the four Phase 5 pretrain terms into ``registry``.

    Pure dict/list insertion — no tensor construction, no RNG use.
    """
    for name, fn, weight in _PRETRAIN_SPEC:
        _guard_pretrain_name(name)
        registry.register(
            LossTerm(name=name, fn=fn, weight=weight, stage="pretrain")
        )


# ----------------------------------------------------------------------
# Phase 6.5e — joint_contrastive_only_e1 stage (weight-mask dispatch).
#
# E1-rev1 is a *weight-change* study over the parent pretrain objective,
# not a new loss. It fine-tunes the existing pretrain ckpt under a
# weight mask {masked_rna_recon: 0, masked_atac_recon: 0,
# cross_modal_infonce: 1.0, peak_to_gene_aux: 0} — i.e. isolates the
# contrastive signal that was previously co-weighted at 0.5 alongside
# two reconstruction terms (1.0 each) and one aux term (0.1).
#
# Design invariants:
#   - Reuses the SAME ``_cross_modal_infonce`` loss defined above. No
#     new loss module, so algebraic parity with the parent objective's
#     contrastive term is byte-identical.
#   - Does NOT remove or rewrite ``register_pretrain_terms``. The
#     existing four pretrain registrations remain intact and unchanged.
#   - The new stage ``joint_contrastive_only_e1`` has exactly ONE
#     active term (the contrastive term with weight 1.0). The other
#     three terms in the weight mask are implicitly zero — they are
#     NOT registered under this stage, so registry.compute() does not
#     invoke their fns. This avoids computing RNA/ATAC reconstructions
#     (and materializing the decoder graph) during a pure-contrastive
#     fine-tune.
#   - The conceptual weight mask {recon:0, recon:0, contrastive:1.0,
#     aux:0} is exposed via ``E1_WEIGHT_MASK`` for tripwire T6.
# ----------------------------------------------------------------------
E1_STAGE = "joint_contrastive_only_e1"

# Conceptual 4-term weight mask for the E1 stage. Terms not registered
# under E1_STAGE have implicit weight 0.0. Exposed for T6 assertion.
E1_WEIGHT_MASK: Dict[str, float] = {
    "masked_rna_recon": 0.0,
    "masked_atac_recon": 0.0,
    "cross_modal_infonce": 1.0,
    "peak_to_gene_aux": 0.0,
}


def register_joint_contrastive_only_e1_terms(registry: LossRegistry) -> None:
    """Register the single active term for the E1-rev1 weight-mask stage.

    Only ``cross_modal_infonce`` (weight 1.0) is registered under
    ``stage="joint_contrastive_only_e1"``. The other three terms in
    E1_WEIGHT_MASK have implicit weight 0.0 and are intentionally NOT
    registered here — their pretrain-stage registrations are untouched.
    """
    registry.register(
        LossTerm(
            name="cross_modal_infonce",
            fn=_cross_modal_infonce,
            weight=E1_WEIGHT_MASK["cross_modal_infonce"],
            stage=E1_STAGE,
        )
    )
