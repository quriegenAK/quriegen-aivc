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


Stage = Literal["pretrain", "causal", "joint"]


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
            if term.stage != stage:
                continue
            value = term.fn(**batch)
            components[term.name] = (
                value.item() if isinstance(value, torch.Tensor) else float(value)
            )
            contribution = term.weight * value
            total = contribution if total is None else total + contribution
        if total is None:
            # No terms matched the stage — return a zero tensor on a best-guess device.
            ref = batch.get("predicted")
            device = ref.device if isinstance(ref, torch.Tensor) else "cpu"
            total = torch.tensor(0.0, device=device)
        return total, components
