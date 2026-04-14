"""
platform/tracking/schemas.py — Pydantic schemas for AIVC experiment tracking.

Defines the canonical data structures for training run metadata, sweep bounds,
and post-run decision routing. All tracking components consume these schemas.

frozen_modules default: ["genelink"] only.
feature_expander trains freely by design.
genelink (GAT encoder) is frozen to preserve r=0.873 baseline.

Escape hatch: set AIVC_ALLOW_GAT_UNFREEZE=1 to skip the genelink freeze validator.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RunStatus(str, Enum):
    """Terminal status of a training run."""
    SUCCESS = "success"
    FAILURE = "failure"
    CTRL_MEMORISATION = "ctrl_memorisation"
    REGRESSION = "regression"


class Modality(str, Enum):
    """Data modality / perturbation type."""
    IFNB = "IFN-b"
    IFNG = "IFN-g"
    CRISPR = "CRISPR"


class PostRunDecision(str, Enum):
    """Action taken after a training run completes."""
    REGISTER = "register"
    TRIGGER_TRAINING_AGENT = "trigger_training_agent"
    TRIGGER_RESEARCH_AGENT = "trigger_research_agent"
    NOOP = "noop"


class SweepBounds(BaseModel):
    """Bounds for a single sweep hyperparameter."""
    min: float
    max: float

    @model_validator(mode="after")
    def _min_le_max(self) -> SweepBounds:
        if self.min > self.max:
            raise ValueError(
                f"SweepBounds min ({self.min}) must be <= max ({self.max})"
            )
        return self


class RunMetadata(BaseModel):
    """Canonical metadata for a single AIVC training run.

    Consumed by ExperimentLogger, post_run_hook, and AgentDispatcher.
    """
    run_id: str = Field(..., description="Unique run identifier (W&B or UUID).")
    dataset: str = Field(default="kang2018_pbmc", description="Dataset name.")
    modality: Modality = Field(default=Modality.IFNB, description="Perturbation type.")

    # Hyperparameters
    lfc_beta: float = Field(default=0.1, ge=0.0, le=1.0)
    neumann_k: int = Field(default=3, ge=1, le=10)
    lambda_l1: float = Field(default=0.01, ge=0.0)
    w_scale_range: tuple[float, float] = Field(
        default=(0.01, 0.3),
        description="(min, max) scale for Neumann W initialisation.",
    )

    # Results
    pearson_r: float = Field(default=0.0, ge=-1.0, le=1.0)
    pearson_r_std: float = Field(default=0.0, ge=0.0)
    delta_nonzero_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of genes with non-zero predicted delta (stim - ctrl).",
    )
    ctrl_memorisation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between predicted stim and ctrl (1.0 = full memorisation).",
    )
    jakstat_within_3x: int = Field(default=0, ge=0, le=15)
    ifit1_pred_fc: float = Field(default=0.0)

    # Paths
    checkpoint_path: Optional[str] = Field(default=None)
    failure_note_path: Optional[str] = Field(default=None)
    wandb_url: Optional[str] = Field(default=None)

    # Status
    status: RunStatus = Field(default=RunStatus.FAILURE)
    decision: PostRunDecision = Field(default=PostRunDecision.NOOP)

    # Frozen modules
    frozen_modules: list[str] = Field(
        default_factory=lambda: ["genelink"],
        description="Module attribute names on PerturbationPredictor frozen during training.",
    )

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = Field(default=None)
    training_time_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _must_freeze_gat_encoder(self) -> RunMetadata:
        """Enforce genelink is frozen unless escape hatch is set."""
        if os.environ.get("AIVC_ALLOW_GAT_UNFREEZE") == "1":
            return self
        if "genelink" not in self.frozen_modules:
            raise ValueError(
                "frozen_modules must include 'genelink'. "
                "The GAT encoder is frozen to preserve r=0.873 baseline. "
                "Set AIVC_ALLOW_GAT_UNFREEZE=1 to override."
            )
        return self

    @model_validator(mode="after")
    def _w_scale_range_ordered(self) -> RunMetadata:
        lo, hi = self.w_scale_range
        if lo > hi:
            raise ValueError(
                f"w_scale_range[0] ({lo}) must be <= w_scale_range[1] ({hi})"
            )
        return self
