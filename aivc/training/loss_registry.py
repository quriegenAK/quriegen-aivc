"""
loss_registry.py — Pluggable loss-term registry for AIVC training.

Goals:
    - Decouple individual loss terms from the monolithic combined_loss_*
      functions in losses.py.
    - Allow per-stage ("pretrain" / "causal" / "joint") composition without
      rewriting training loops.
    - Preserve BIT-FOR-BIT parity with the hardcoded combined_loss_multimodal
      so W&B / MLflow scalar keys and values do not drift.

The registry is additive: total = sum_i(weight_i * fn_i(**batch)), computed
in the registration order. Dict keys in the returned components mirror the
legacy keys expected by logging infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import torch


Stage = Literal[
    "pretrain",
    "causal",
    "joint",
    "joint_safe",
    "joint_contrastive_only_e1",
]


# Terms that MUST NEVER contribute to the pretrain total, even if (through
# bug or misconfiguration) they are tagged "pretrain"/"joint_safe". These
# leak gradient into the causal parameter matrix NeumannPropagation.W from
# observational batches, which would violate the Phase 3 gradient-isolation
# guarantee. Match is by LossTerm.name.
_PRETRAIN_FORBIDDEN_NAMES = frozenset({
    "causal_ordering",  # attention-graph causal ordering penalty
    "causal",           # legacy alias used in combined_loss_multimodal
    "mse",              # MSE against stim = interventional target
    "lfc",              # log-fold-change vs ctrl→stim = interventional target
})


def _is_term_active(term_stage: "Stage", requested_stage: "Stage") -> bool:
    """Return True if a term tagged `term_stage` runs under `requested_stage`.

    Pretrain is a *safe subset*: only terms tagged "pretrain" or the
    explicit "joint_safe" allow-list may contribute. Every other stage
    matches exactly (legacy behavior).
    """
    if requested_stage == "pretrain":
        return term_stage in ("pretrain", "joint_safe")
    return term_stage == requested_stage


@dataclass
class LossTerm:
    """A single weighted loss contribution.

    Attributes:
        name:   Key used in the components breakdown dict (e.g. "mse").
        fn:     Callable(**batch) -> scalar torch.Tensor. Must accept **kwargs
                so the registry can pass a superset of batch fields.
        weight: Scalar multiplier applied when summing into total.
        stage:  Training stage in which this term is active.
    """
    name: str
    fn: Callable[..., torch.Tensor]
    weight: float
    stage: Stage


class LossRegistry:
    """Ordered registry of LossTerm objects.

    Preserves insertion order so that floating-point summation in compute()
    is deterministic across calls — important for reproducing legacy
    combined_loss_multimodal values to 1e-7.
    """

    def __init__(self) -> None:
        self._terms: List[LossTerm] = []

    def register(self, term: LossTerm) -> None:
        self._terms.append(term)

    def terms(self) -> List[LossTerm]:
        return list(self._terms)

    def compute(
        self,
        stage: Stage,
        **batch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate all terms registered for `stage`.

        Returns:
            (total, components) where `components[name]` is the UNWEIGHTED
            scalar value of each term (for logging), and `total` is the
            weighted sum as a torch.Tensor preserving the compute graph.
        """
        total: torch.Tensor | None = None
        components: Dict[str, float] = {}
        for term in self._terms:
            if not _is_term_active(term.stage, stage):
                continue
            # Defense-in-depth: interventional-target terms are hard-blocked
            # from the pretrain total regardless of their stage tag. This
            # prevents a misregistered "joint_safe" causal/MSE/LFC term from
            # leaking gradient into NeumannPropagation.W under observational
            # batches (see Phase 3 gradient-isolation guard).
            if stage == "pretrain" and term.name in _PRETRAIN_FORBIDDEN_NAMES:
                continue
            value = term.fn(**batch)
            components[term.name] = (
                value.item() if isinstance(value, torch.Tensor) else float(value)
            )
            contribution = term.weight * value
            total = contribution if total is None else total + contribution
        if total is None:
            # No terms matched. For pretrain this is the expected state in
            # Phase 3 (no pretrain losses registered yet).
            #
            # IMPORTANT — structural W-independence of the placeholder:
            # We MUST NOT return a scalar whose autograd graph includes
            # `predicted`. In the current model, `predicted` is the output
            # of model.forward(...) which passes through NeumannPropagation
            # (see perturbation_model.PerturbationPredictor.forward_batch
            # L381-383). A placeholder like `(predicted * 0.0).sum()` would
            # numerically be zero but would still register a graph edge
            # into W — backward() would then populate W.grad with a zero
            # tensor rather than leaving it None. That would make the
            # gradient-isolation guard pass for the WRONG reason (the
            # allclose-0 branch instead of the None branch), and would
            # mask real leaks if a future pretrain term were accidentally
            # added with a W-touching path.
            #
            # Instead we construct a freshly-allocated leaf zero tensor
            # with requires_grad=True. It has no incoming graph edges at
            # all, so .backward() is a valid no-op w.r.t. every model
            # parameter, and W.grad stays None unless a real pretrain
            # term (registered in a future phase) pulls gradient through W.
            ref = batch.get("predicted")
            device = ref.device if isinstance(ref, torch.Tensor) else "cpu"
            total = torch.zeros((), device=device, requires_grad=True)
        return total, components
