"""
platform/tracking/wandb_config.py — W&B configuration for AIVC experiment tracking.

Provides:
  - init_wandb(): initialise a W&B run with graceful offline fallback
  - build_run_group(): deterministic group name for sweep configurations
  - SWEEP_CONFIG: canonical W&B sweep definition for v1.1 Neumann sweep

Credentials:
  WANDB_ENTITY and WANDB_PROJECT are hardcoded (quriegen / aivc_genelink).
  WANDB_API_KEY is read from environment — if absent, degrades to offline mode.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("aivc.tracking.wandb")

# ── Hardcoded project coordinates ─────────────────────────────────────
WANDB_ENTITY = "quriegen"
WANDB_PROJECT = "aivc_genelink"
WANDB_URL = "https://wandb.ai/quriegen/aivc_genelink"

# ── Sweep configuration (v1.1 Neumann W-scale sweep) ─────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/pearson_r", "goal": "maximize"},
    "parameters": {
        "w_scale_min": {"min": 0.001, "max": 0.1},
        "w_scale_max": {"min": 0.05, "max": 0.5},
        "neumann_K": {"values": [2, 3, 5]},
        "lfc_beta": {"values": [0.05, 0.10, 0.20, 0.50]},
        "lambda_l1": {"values": [0.0001, 0.001, 0.01]},
        "lr": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
    },
}


def init_wandb(
    run_name: str,
    config: dict,
    tags: Optional[list[str]] = None,
    group: Optional[str] = None,
) -> Optional[object]:
    """Initialise a W&B run with graceful offline fallback.

    Args:
        run_name: Human-readable run name.
        config: Hyperparameter dict logged to W&B.
        tags: Optional list of tags for filtering.
        group: Optional group name for sweep runs.

    Returns:
        wandb.Run or None if wandb is not available.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — tracking disabled.")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        logger.warning("WANDB_API_KEY not set — running offline")
        os.environ["WANDB_MODE"] = "offline"

    try:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            tags=tags or [],
            group=group,
            reinit=True,
        )
        logger.info(f"W&B run initialised: {run.url or 'offline'}")
        return run
    except Exception as e:
        logger.warning(f"wandb.init failed — continuing without tracking: {e}")
        return None


def build_run_group(
    lfc_beta: float,
    neumann_k: int,
    lambda_l1: float,
) -> str:
    """Build a deterministic group name for a sweep configuration.

    Returns:
        e.g. "beta0.10_K3_l10.010"
    """
    return f"beta{lfc_beta:.2f}_K{neumann_k}_l1{lambda_l1:.3f}"
