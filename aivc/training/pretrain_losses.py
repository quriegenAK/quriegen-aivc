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
from aivc.data.modality_mask import ModalityKey


# Four names registered by this module. Exposed for tests.
PRETRAIN_TERM_NAMES = (
    "masked_rna_recon",
    "masked_atac_recon",
    "cross_modal_infonce",
    "peak_to_gene_aux",
)

# DOGMA-specific term names (Day 2). Registered via _dogma_pretrain_loss
# in losses.py, NOT via register_pretrain_terms — so they are not in
# _PRETRAIN_SPEC and do not affect existing Phase 5 registration.
DOGMA_PRETRAIN_TERM_NAMES = (
    "masked_rna_recon",
    "masked_atac_recon",
    "masked_protein_recon",
    "cross_modal_infonce_triad",
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


def _masked_protein_recon(**batch) -> torch.Tensor:
    """Masked protein (ADT) reconstruction, gated on per-cell modality presence.

    Gates on modality_mask[:, ModalityKey.PROTEIN]. Per D2 semantics:
    if no cell in the batch has Protein present (mask column all-zero),
    returns 0 contribution silently — enables mixed-corpus training where
    some batches are DOGMA (Protein present) and others are RNA+ATAC-only.
    """
    pred = batch["protein_recon"]
    target = batch["protein_target"]
    token_mask = batch.get("protein_mask")        # token-level mask, optional
    modality_mask = batch.get("modality_mask")    # (B, 4) per-cell presence

    if modality_mask is not None:
        protein_present = modality_mask[:, int(ModalityKey.PROTEIN)]
        if float(protein_present.sum()) < 1.0:
            # D2: silent zero when whole batch has no Protein-present cells
            return torch.zeros((), device=pred.device, requires_grad=True)
        cell_weight = protein_present.unsqueeze(-1)  # (B, 1)
    else:
        cell_weight = torch.ones(pred.shape[0], 1, device=pred.device)

    if token_mask is None:
        diff = (pred - target) ** 2 * cell_weight
        denom = cell_weight.expand_as(pred).sum().clamp_min(1.0)
    else:
        combined = token_mask * cell_weight
        diff = (pred - target) ** 2 * combined
        denom = combined.sum().clamp_min(1.0)
    return diff.sum() / denom


def _cross_modal_infonce(**batch) -> torch.Tensor:
    """Symmetric InfoNCE between L2-normalized rna/atac projections."""
    z_rna = batch["z_rna"]
    z_atac = batch["z_atac"]
    temperature = float(batch.get("infonce_temperature", 0.1))
    logits = z_rna @ z_atac.t() / temperature
    labels = torch.arange(z_rna.shape[0], device=z_rna.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _cross_modal_infonce_triad(**batch) -> torch.Tensor:
    """Pairwise-average 3-way InfoNCE for RNA+ATAC+Protein (D1 decision).

    Computes:
      mean over {NCE(z_rna, z_atac), NCE(z_rna, z_protein), NCE(z_atac, z_protein)}

    Each pairwise term is skipped (contributes 0) if:
      - either projection key (z_rna / z_atac / z_protein) is absent from batch, OR
      - either modality's mask column is all-zero for the batch (D2 silent skip)

    Falls back cleanly to bimodal (2 present terms averaged) when only 2
    modalities present; returns 0 contribution if fewer than 2 modality
    projections are available.

    Reuses the same NCE kernel as the 2-way _cross_modal_infonce. The 2-way
    function is untouched; this is additive.
    """
    modality_mask = batch.get("modality_mask")
    temperature = float(batch.get("infonce_temperature", 0.1))

    def _is_present(mk: ModalityKey) -> bool:
        if modality_mask is None:
            return True  # backward-compat: assume modality present
        return bool((modality_mask[:, int(mk)].sum() > 0).item())

    def _pair_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        logits = z_a @ z_b.t() / temperature
        labels = torch.arange(z_a.shape[0], device=z_a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    pair_defs = (
        (ModalityKey.RNA,  ModalityKey.ATAC,    "z_rna", "z_atac"),
        (ModalityKey.RNA,  ModalityKey.PROTEIN, "z_rna", "z_protein"),
        (ModalityKey.ATAC, ModalityKey.PROTEIN, "z_atac", "z_protein"),
    )
    terms = []
    for mk_a, mk_b, k_a, k_b in pair_defs:
        if not (_is_present(mk_a) and _is_present(mk_b)):
            continue
        if k_a not in batch or k_b not in batch:
            continue
        if batch[k_a] is None or batch[k_b] is None:
            continue
        terms.append(_pair_nce(batch[k_a], batch[k_b]))

    if not terms:
        # D2 silent skip: pick any tensor for device inheritance
        anchor = next(
            (v for v in batch.values() if isinstance(v, torch.Tensor)),
            None,
        )
        device = anchor.device if anchor is not None else torch.device("cpu")
        return torch.zeros((), device=device, requires_grad=True)

    return torch.stack(terms).mean()


def _peak_to_gene_aux(**batch) -> torch.Tensor:
    return F.mse_loss(batch["gene_pred"], batch["gene_target"])


# ----------------------------------------------------------------------
# Phase 6.5g.2 PR #54 — Supervised contrastive (Khosla 2020) +
# VICReg variance regularizer (Bardes 2022).
#
# Operates on z_supcon: a per-cell joint embedding (typically the fusion
# module's output, L2-normalized). Required because the existing triad
# InfoNCE is unsupervised cross-modal — it pulls positive pairs that
# are the same cell across modalities, but does not pull cells of the
# same cell type together. SupCon adds the cell-type-aware positive set.
#
# VICReg's variance term hinge-penalizes per-feature std below the
# target. Prevents the encoder from collapsing some dimensions to
# near-constants under the SupCon pull.
# ----------------------------------------------------------------------
SUPCON_VICREG_TERM_NAMES = ("supcon", "vicreg_variance")


def _supcon_loss(**batch) -> torch.Tensor:
    """Supervised contrastive loss (Khosla 2020, NeurIPS).

    For each eligible anchor i with positives P(i) = {j : y_j = y_i, j != i}:
        L_i = -1/|P(i)| * sum_{p in P(i)} log[ exp(s_ip / tau) / sum_{a != i} exp(s_ia / tau) ]
    Aggregated as mean over anchors that have at least one positive.

    Eligibility (`supcon_eligible_mask`) lets the loader exclude:
      - Cells with masked-out cell types (e.g. 'other', 'other_T' for DOGMA)
      - Cells below a confidence threshold (e.g. transferred labels with
        cell_type_confidence < 0.6 for the DIG arm)
    Excluded cells still contribute to the *negative* set (denominator)
    when other anchors contrast against them, which is the correct
    behavior — they are valid as "definitely not your class" examples.

    Required batch keys:
      z_supcon : (B, D) — L2-normalized per-cell embedding
      cell_type_idx : (B,) LongTensor — integer class IDs
    Optional batch keys:
      supcon_eligible_mask : (B,) bool — True = use as anchor
      supcon_temperature : float (default 0.07; Khosla SupCon original)

    Returns a scalar tensor. Returns 0 (with grad) if fewer than 2
    eligible anchors or no anchor has a positive in the batch.
    """
    z = batch["z_supcon"]
    y = batch["cell_type_idx"].long()
    tau = float(batch.get("supcon_temperature", 0.07))

    eligible = batch.get("supcon_eligible_mask")
    if eligible is None:
        eligible = torch.ones_like(y, dtype=torch.bool)
    eligible = eligible.bool()

    if int(eligible.sum().item()) < 2:
        return torch.zeros((), device=z.device, requires_grad=True)

    # Pairwise sim over the FULL batch — eligible cells are anchors,
    # ineligible cells remain valid negatives.
    sim = (z @ z.t()) / tau  # (B, B)
    B = z.shape[0]

    # Mask self from denominator with a LARGE FINITE NEGATIVE, NOT -inf.
    # Rationale: -inf * 0 = NaN under IEEE 754, and a downstream
    # `(log_prob * pos_mask.float()).sum()` includes the diagonal
    # (pos_mask[i,i] = False = 0.0), which would corrupt the row sum to
    # NaN. -1e9 is far enough below any real cosine-sim/τ value
    # (max ≈ 1/0.07 ≈ 14.3) that exp(-1e9) underflows to 0 in
    # logsumexp, giving the same denominator as true -inf, but
    # (-1e9) * 0 = 0 stays finite. Matches the supcon-pytorch reference
    # implementation pattern.
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)

    # log-softmax over the row (denominator runs over all j != i).
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)

    # Positive set: same class as anchor, not self, and ANCHOR is eligible.
    same_class = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask  # (B, B)
    # We only use ELIGIBLE rows as anchors (their per-anchor loss is
    # computed). Positive partners (columns) can be any cell with the
    # same class, eligible or not. This matches the spec: ineligible
    # cells contribute only as negatives or, if same-class, as positives
    # for an eligible anchor.
    anchor_mask = eligible.unsqueeze(1)  # (B, 1)
    pos_mask = same_class & anchor_mask  # (B, B)

    pos_count = pos_mask.sum(dim=1)  # (B,)
    valid_anchor = pos_count > 0  # (B,)
    valid_anchor = valid_anchor & eligible
    if int(valid_anchor.sum().item()) == 0:
        return torch.zeros((), device=z.device, requires_grad=True)

    # Per-anchor loss: average -log_prob over positive partners. The
    # masked_fill above guarantees no -inf in log_prob, so (log_prob *
    # pos_mask.float()) is finite everywhere.
    log_prob_pos_sum = (log_prob * pos_mask.float()).sum(dim=1)  # (B,)
    per_anchor = -log_prob_pos_sum / pos_count.clamp_min(1).float()  # (B,)
    return per_anchor[valid_anchor].mean()


def _vicreg_variance(**batch) -> torch.Tensor:
    """VICReg variance regularizer (Bardes 2022).

        L = (1/D) * sum_d max(0, gamma - sqrt(Var(z_d) + eps))

    Hinge-penalizes per-feature std below `gamma` (target_std). Encourages
    feature diversity across the batch; works as a collapse guard against
    SupCon's tendency to compress some embedding dimensions.

    Required batch keys:
      z_supcon : (B, D)
    Optional:
      vicreg_target_std : float (default 1.0)
      vicreg_eps : float (default 1e-4)

    Returns a scalar tensor. Defined for B >= 2 (variance needs >=2 samples).
    Returns 0 if B < 2.
    """
    z = batch["z_supcon"]
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device, requires_grad=True)

    target = float(batch.get("vicreg_target_std", 1.0))
    eps = float(batch.get("vicreg_eps", 1e-4))

    # var with unbiased=False to keep the denominator predictable across
    # batch-size variations; 1/N rather than 1/(N-1).
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # (D,)
    return F.relu(target - std).mean()


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
