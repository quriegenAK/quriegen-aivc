"""aivc_platform/registry/model_registry.py — ModelCard + file-based registry.

Registry layout:
  artifacts/registry/latest_checkpoint.json   — most recently registered card (overwritten)
  artifacts/registry/history.jsonl            — append-only history, one ModelCard per line

register() refuses to write if suite.kang.regression_guard_passed is False.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.eval_runner import EvalSuite

logger = logging.getLogger("aivc.registry.model_registry")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelCard(BaseModel):
    """Canonical model registry entry."""

    version: str
    run_id: str
    pearson_r: float = Field(..., ge=-1.0, le=1.0)
    pearson_r_std: Optional[float] = None
    delta_nonzero_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    ctrl_memorisation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    jakstat_within_3x: Optional[int] = None
    checkpoint_path: str
    datasets_trained: list[str] = Field(default_factory=lambda: ["kang2018_pbmc"])
    frozen_modules: list[str] = Field(default_factory=lambda: ["genelink"])
    wandb_url: Optional[str] = None
    registered_at: datetime = Field(default_factory=_utcnow)
    notes: str = ""
    regression_guard_passed: bool = True


class ModelRegistry:
    """File-based model registry."""

    REGISTRY_DIR = Path("artifacts/registry")
    LATEST_PATH = REGISTRY_DIR / "latest_checkpoint.json"
    HISTORY_PATH = REGISTRY_DIR / "history.jsonl"

    def register(self, meta: "RunMetadata", suite: "EvalSuite") -> ModelCard:
        """Register a checkpoint. Raises ValueError on regression-guard failure."""
        kang = suite.kang
        if not kang.regression_guard_passed:
            raise ValueError(
                "regression_guard_passed=False — refusing to register "
                f"(pearson_r={kang.pearson_r:.4f})"
            )

        norman = suite.norman
        card = ModelCard(
            version=f"v1.1-r{meta.pearson_r:.3f}",
            run_id=meta.run_id,
            pearson_r=meta.pearson_r,
            pearson_r_std=meta.pearson_r_std or None,
            delta_nonzero_pct=(norman.delta_nonzero_pct if norman else meta.delta_nonzero_pct),
            ctrl_memorisation_score=(
                norman.ctrl_memorisation_score if norman else meta.ctrl_memorisation_score
            ),
            jakstat_within_3x=meta.jakstat_within_3x,
            checkpoint_path=meta.checkpoint_path or "",
            datasets_trained=[meta.dataset],
            frozen_modules=list(meta.frozen_modules),
            wandb_url=meta.wandb_url,
            notes="",
            regression_guard_passed=True,
        )

        self.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        payload = card.model_dump_json()
        self.LATEST_PATH.write_text(payload + "\n", encoding="utf-8")
        with self.HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")
        logger.info(f"Registered {card.version} (run_id={card.run_id})")
        return card

    def get_best(self, min_pearson: float = 0.873) -> Optional[ModelCard]:
        """Return highest-pearson ModelCard with pearson_r >= min_pearson."""
        if not self.HISTORY_PATH.exists():
            return None
        best: Optional[ModelCard] = None
        try:
            for line in self.HISTORY_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    card = ModelCard.model_validate_json(line)
                except Exception:
                    continue
                if card.pearson_r < min_pearson:
                    continue
                if best is None or card.pearson_r > best.pearson_r:
                    best = card
        except OSError as e:
            logger.warning(f"get_best read failed: {e}")
            return None
        return best

    def get_latest(self) -> Optional[ModelCard]:
        """Return the most recently registered ModelCard, or None."""
        if not self.LATEST_PATH.exists():
            return None
        try:
            return ModelCard.model_validate_json(self.LATEST_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"get_latest parse failed: {e}")
            return None
