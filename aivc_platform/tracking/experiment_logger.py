"""
platform/tracking/experiment_logger.py — AIVC experiment lifecycle manager.

Contains:
  - ExperimentLogger: wraps MLflow + W&B, drives post-run decisions
  - AgentDispatcher: writes structured prompts and optionally calls Claude API
  - apply_frozen_modules: freezes named submodules on PerturbationPredictor

MLflow is file://mlruns permanently in v1. Model Registry is disabled.
On SUCCESS (pearson_r >= 0.873), checkpoint is logged as artifact and
written to artifacts/registry/latest_checkpoint.json (the v1 "registry").
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import torch

from .schemas import PostRunDecision, RunMetadata, RunStatus
from .wandb_config import WANDB_URL, build_run_group, init_wandb

if TYPE_CHECKING:
    import torch.nn as nn
    from eval.eval_runner import EvalSuite

logger = logging.getLogger("aivc.tracking.experiment_logger")

# ── Thresholds ────────────────────────────────────────────────────────
PEARSON_R_THRESHOLD = 0.873
DELTA_NONZERO_THRESHOLD = 0.0  # == 0 means ctrl memorisation


# =====================================================================
# apply_frozen_modules
# =====================================================================
def apply_frozen_modules(model: nn.Module, frozen_modules: list[str]) -> None:
    """Freeze named submodules on a model.

    Sets requires_grad=False for all parameters of each named submodule.
    Raises AttributeError if a named module does not exist on the model.

    Args:
        model: The model (typically PerturbationPredictor).
        frozen_modules: List of attribute names to freeze (e.g. ["genelink"]).
    """
    for name in frozen_modules:
        submodule = getattr(model, name)  # raises AttributeError if missing
        for param in submodule.parameters():
            param.requires_grad = False

        # Verify freeze took effect
        frozen_params = [p for p in submodule.parameters() if not p.requires_grad]
        total_params = list(submodule.parameters())
        assert len(frozen_params) == len(total_params), (
            f"Partial freeze on {name}: "
            f"{len(frozen_params)}/{len(total_params)} params frozen."
        )
        logger.info(
            f"Frozen {name}: {len(frozen_params)} params (requires_grad=False)"
        )


# =====================================================================
# AgentDispatcher
# =====================================================================
class AgentDispatcher:
    """
    Triggers downstream Claude agents by writing a structured prompt
    to artifacts/agent_queue/ and optionally calling the Claude API.

    v1 behaviour:
      - Always writes the prompt file (observable, debuggable)
      - Calls Claude API if ANTHROPIC_API_KEY is set
      - Silently skips API call if key is absent (file remains)
    """

    AGENT_PROMPTS = {
        "training_agent": """
# Training Agent Task

## Context
Run ID: {run_id}
Dataset: {dataset}
Pearson r: {pearson_r}
Delta non-zero %: {delta_nonzero_pct}
ctrl memorisation score: {ctrl_memorisation_score}
Neumann K: {neumann_k}
Failure note: {failure_note_path}

## Problem
The model collapsed to ctrl memorisation (delta_nonzero_pct == 0).
Predicted perturbation response is indistinguishable from ctrl.

## Your task
1. Diagnose the most likely cause from:
   a. W scale too aggressive (sparsity collapse)
   b. Neumann K too small (insufficient propagation depth)
   c. Learning rate too high (GAT overfits to ctrl distribution)
   d. Data issue (ctrl/stim label swap or contamination)
2. Recommend exact hyperparameter changes for the next run.
3. Output a configs/sweep_w_scale_retry.yaml with corrected bounds.

## Constraints
- genelink must remain frozen (requires_grad=False)
- Do not change n_genes=3010 or edge index
- Write output to Claude Outputs/aivc_genelink/
""",
        "research_agent": """
# Research Agent Task

## Context
Run ID: {run_id}
Dataset: {dataset}
Pearson r: {pearson_r}
Delta non-zero %: {delta_nonzero_pct}
W scale range: {w_scale_range}
Neumann K: {neumann_k}
Checkpoint: {checkpoint_path}
W&B run: {wandb_url}

## Your task
1. Summarise what changed vs the previous best checkpoint (r=0.873).
2. Interpret the delta_nonzero improvement biologically:
   what does a higher % mean for perturbation prediction in PBMCs?
3. Write an Obsidian experiment note to:
   ~/Documents/Obsidian/aivc_genelink/experiments/{run_id}.md
   Use the standard experiment note template.
4. Suggest one concrete next experiment (hypothesis + config change).
5. Update context.md field "best pearson_r" if this run exceeds current best.

## Constraints
- Write only to Claude Outputs/ and Obsidian vault
- Do not modify any source file
""",
    }

    def __init__(
        self,
        queue_dir: str = "artifacts/agent_queue",
        model: str = "claude-sonnet-4-20250514",
        execute: bool = True,
    ):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.execute = execute
        self._api_key = os.getenv("ANTHROPIC_API_KEY")

    def trigger(
        self,
        agent_name: Literal["training_agent", "research_agent"],
        payload: dict,
    ) -> None:
        """Write a prompt file and optionally call the Claude API.

        Args:
            agent_name: Which agent to trigger.
            payload: Dict of format-string values for the prompt template.
        """
        prompt = self.AGENT_PROMPTS[agent_name].format(**payload)
        timestamp = int(time.time())
        run_id = payload.get("run_id", "unknown")

        # Always write prompt file
        prompt_file = self.queue_dir / f"{agent_name}_{run_id}_{timestamp}.md"
        try:
            prompt_file.write_text(prompt)
            logger.info(f"AgentDispatcher: prompt written to {prompt_file}")
        except OSError as e:
            logger.warning(f"AgentDispatcher: could not write prompt: {e}")
            return

        # execute=False → pure-IO mode; agents own Claude calls
        if not self.execute:
            logger.info(
                "AgentDispatcher execute=False — prompt written, no API call"
            )
            return

        # Call Claude API if key is available
        if not self._api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — agent prompt written to queue "
                f"but not executed: {prompt_file}"
            )
            return

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            response_file = (
                self.queue_dir / f"{agent_name}_{run_id}_{timestamp}_response.md"
            )
            response_file.write_text(response_text)
            logger.info(
                f"AgentDispatcher: {agent_name} response written to {response_file}"
            )
        except Exception as e:
            logger.warning(
                f"AgentDispatcher: Claude API call failed for {agent_name}: {e}. "
                f"Prompt remains in queue at {prompt_file}."
            )


# Module-level alias so agents can import the single source of truth.
AGENT_PROMPTS = AgentDispatcher.AGENT_PROMPTS


# =====================================================================
# ExperimentLogger
# =====================================================================
class ExperimentLogger:
    """Orchestrates MLflow + W&B logging and post-run decision routing.

    MLflow: file://mlruns (v1, no DB, registry disabled).
    W&B: quriegen/aivc_genelink (online if API key set, else offline).

    Lifecycle:
      1. start(meta) — open MLflow run + W&B run
      2. log_epoch(...) — per-epoch metrics
      3. finish(meta) — end runs, call post_run_hook
    """

    def __init__(self) -> None:
        self._mlflow_uri = "file://mlruns"
        self._registry_available = False
        logger.info(
            "MLflow: file store (file://mlruns). Model Registry disabled in v1."
        )

        self._mlflow = self._init_mlflow()
        self._wandb_run = None
        self._dispatcher = AgentDispatcher(execute=False)
        self._finished_runs: set[str] = set()

        # Lazy import to avoid circular dependency via aivc_platform.memory →
        # aivc_platform.tracking.schemas → aivc_platform.tracking.__init__.
        from aivc_platform.memory.vault import ObsidianConfig, init_vault

        self._obsidian_config = ObsidianConfig()
        try:
            init_vault(self._obsidian_config)
        except Exception as e:
            logger.warning(f"init_vault failed (non-fatal): {e}")

    def _init_mlflow(self) -> Optional[object]:
        """Initialise MLflow with file backend."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._mlflow_uri)
            return mlflow
        except ImportError:
            logger.warning("mlflow not installed — MLflow tracking disabled.")
            return None

    # ── Run lifecycle ─────────────────────────────────────────────

    def start(self, meta: RunMetadata) -> None:
        """Start a new tracked run.

        Args:
            meta: RunMetadata with run_id, hyperparameters, etc.
        """
        # MLflow
        if self._mlflow is not None:
            try:
                self._mlflow.set_experiment("aivc_v11_neumann_sweep")
                self._mlflow.start_run(run_name=meta.run_id)
                self._mlflow.log_params(
                    {
                        "lfc_beta": meta.lfc_beta,
                        "neumann_K": meta.neumann_k,
                        "lambda_l1": meta.lambda_l1,
                        "dataset": meta.dataset,
                        "frozen_modules": ",".join(meta.frozen_modules),
                        "w_scale_range": str(meta.w_scale_range),
                    }
                )
            except Exception as e:
                logger.warning(f"MLflow start failed: {e}")

        # W&B
        group = build_run_group(meta.lfc_beta, meta.neumann_k, meta.lambda_l1)
        self._wandb_run = init_wandb(
            run_name=meta.run_id,
            config={
                "lfc_beta": meta.lfc_beta,
                "neumann_k": meta.neumann_k,
                "lambda_l1": meta.lambda_l1,
                "dataset": meta.dataset,
                "w_scale_range": meta.w_scale_range,
                "frozen_modules": meta.frozen_modules,
            },
            tags=["v1.1", "neumann", meta.dataset],
            group=group,
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_r: float,
        **extra_metrics: float,
    ) -> None:
        """Log per-epoch metrics to MLflow and W&B.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_r: Validation Pearson r for this epoch.
            **extra_metrics: Additional metrics to log.
        """
        metrics = {"train/loss": train_loss, "val/pearson_r": val_r, **extra_metrics}

        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(metrics, step=epoch)
            except Exception as e:
                logger.debug(f"MLflow epoch log failed: {e}")

        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=epoch)
            except Exception as e:
                logger.debug(f"W&B epoch log failed: {e}")

    def finish(self, meta: RunMetadata, suite: "EvalSuite | None" = None) -> None:
        """End the current run, trigger post_run_hook, close trackers.

        Idempotent: calling finish() twice with the same run_id dispatches once.

        Args:
            meta: Final RunMetadata with results filled in.
            suite: Optional EvalSuite; enables memory layer writes when provided.
        """
        if meta.run_id in self._finished_runs:
            logger.warning(f"Run {meta.run_id} already finished — skipping.")
            return
        self._finished_runs.add(meta.run_id)

        # Log final metrics
        final_metrics = {
            "final/pearson_r": meta.pearson_r,
            "final/pearson_r_std": meta.pearson_r_std,
            "final/delta_nonzero_pct": meta.delta_nonzero_pct,
            "final/jakstat_within_3x": float(meta.jakstat_within_3x),
            "final/ifit1_pred_fc": meta.ifit1_pred_fc,
            "final/training_time_s": meta.training_time_s,
        }

        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(final_metrics)
            except Exception as e:
                logger.debug(f"MLflow final metrics failed: {e}")

        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(final_metrics)
            except Exception as e:
                logger.debug(f"W&B final metrics failed: {e}")

        # Post-run hook
        self.post_run_hook(meta, suite=suite)

        # End MLflow run
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception as e:
                logger.debug(f"MLflow end_run failed: {e}")

        # End W&B run
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as e:
                logger.debug(f"W&B finish failed: {e}")
            self._wandb_run = None

    # ── Post-run decision routing ─────────────────────────────────

    def post_run_hook(
        self,
        meta: RunMetadata,
        suite: "EvalSuite | None" = None,
    ) -> None:
        """Route post-run actions based on results.

        FAILURE path (delta_nonzero_pct == 0):
          - Write failure note to artifacts/failure_notes/
          - Trigger training_agent via AgentDispatcher
          - Do NOT write latest_checkpoint.json

        SUCCESS path (pearson_r >= 0.873 and delta_nonzero_pct > 0):
          - Log checkpoint to MLflow as artifact
          - Write artifacts/registry/latest_checkpoint.json
          - Trigger research_agent via AgentDispatcher

        Model Registry is disabled in v1 — never calls mlflow.register_model.
        """
        # FAILURE: ctrl memorisation
        if meta.delta_nonzero_pct <= DELTA_NONZERO_THRESHOLD:
            self._handle_failure(meta, suite=suite)
            return

        # SUCCESS: meets baseline
        if meta.pearson_r >= PEARSON_R_THRESHOLD:
            self._handle_success(meta, suite=suite)
            return

        # Below threshold but not memorisation — noop
        logger.info(
            f"Run {meta.run_id}: pearson_r={meta.pearson_r:.4f} < {PEARSON_R_THRESHOLD} "
            f"but delta_nonzero_pct={meta.delta_nonzero_pct:.1f}% > 0. No action."
        )

    def _handle_failure(
        self,
        meta: RunMetadata,
        suite: "EvalSuite | None" = None,
    ) -> None:
        """Handle ctrl memorisation failure."""
        failure_dir = Path("artifacts/failure_notes")
        failure_dir.mkdir(parents=True, exist_ok=True)

        note = {
            "run_id": meta.run_id,
            "dataset": meta.dataset,
            "pearson_r": meta.pearson_r,
            "delta_nonzero_pct": meta.delta_nonzero_pct,
            "ctrl_memorisation_score": meta.ctrl_memorisation_score,
            "w_scale_range": list(meta.w_scale_range),
            "neumann_k": meta.neumann_k,
            "lambda_l1": meta.lambda_l1,
            "frozen_modules": meta.frozen_modules,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        failure_path = failure_dir / f"failure_{meta.run_id}.json"
        try:
            failure_path.write_text(json.dumps(note, indent=2))
            logger.info(f"Failure note written: {failure_path}")
        except OSError as e:
            logger.warning(f"Could not write failure note: {e}")

        meta.failure_note_path = str(failure_path)

        # Memory layer (best-effort)
        if suite is not None:
            try:
                from aivc_platform.memory.obsidian_writer import write_failure_note
                write_failure_note(
                    meta,
                    PostRunDecision.TRIGGER_TRAINING_AGENT,
                    suite,
                    self._obsidian_config,
                )
                # DO NOT call update_context on failure
            except Exception as e:
                logger.warning(f"memory write (failure) failed: {e}")

        # Trigger training agent
        self._dispatcher.trigger(
            "training_agent",
            {
                "run_id": meta.run_id,
                "dataset": meta.dataset,
                "pearson_r": meta.pearson_r,
                "delta_nonzero_pct": meta.delta_nonzero_pct,
                "ctrl_memorisation_score": meta.ctrl_memorisation_score,
                "w_scale_range": str(meta.w_scale_range),
                "neumann_k": meta.neumann_k,
                "failure_note_path": str(failure_path),
            },
        )

    def _handle_success(
        self,
        meta: RunMetadata,
        suite: "EvalSuite | None" = None,
    ) -> None:
        """Handle successful run that meets baseline."""
        # Log checkpoint to MLflow as artifact
        if self._mlflow is not None and meta.checkpoint_path:
            try:
                self._mlflow.log_artifact(meta.checkpoint_path)
                logger.info(f"Checkpoint logged to MLflow: {meta.checkpoint_path}")
            except Exception as e:
                logger.warning(f"MLflow log_artifact failed: {e}")

        # Write v1 registry file
        registry_dir = Path("artifacts/registry")
        registry_dir.mkdir(parents=True, exist_ok=True)

        registry_entry = {
            "run_id": meta.run_id,
            "pearson_r": meta.pearson_r,
            "checkpoint_path": meta.checkpoint_path,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "wandb_url": meta.wandb_url,
        }
        registry_path = registry_dir / "latest_checkpoint.json"
        try:
            registry_path.write_text(json.dumps(registry_entry, indent=2))
            logger.info(f"Registry updated: {registry_path}")
        except OSError as e:
            logger.warning(f"Could not write registry: {e}")

        # Memory layer (best-effort)
        if suite is not None:
            try:
                from aivc_platform.memory.obsidian_writer import write_experiment_note
                from aivc_platform.memory.context_updater import update_context
                write_experiment_note(meta, suite, self._obsidian_config)
                update_context(meta, suite)
            except Exception as e:
                logger.warning(f"memory write (success) failed: {e}")

        # Trigger research agent
        self._dispatcher.trigger(
            "research_agent",
            {
                "run_id": meta.run_id,
                "dataset": meta.dataset,
                "pearson_r": meta.pearson_r,
                "delta_nonzero_pct": meta.delta_nonzero_pct,
                "w_scale_range": str(meta.w_scale_range),
                "neumann_k": meta.neumann_k,
                "checkpoint_path": meta.checkpoint_path or "",
                "wandb_url": meta.wandb_url or WANDB_URL,
                "meta_json": meta.model_dump_json(),
                "suite_json": suite.model_dump_json() if suite is not None else None,
            },
        )
