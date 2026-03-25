"""
aivc/memory/experimental.py — Persistent experiment history.

Saved to disk as JSON. Used for: regression detection,
hyperparameter sweep analysis, historical comparison in R&D report.
"""

import json
import os
import time
from typing import Optional


class ExperimentalMemory:
    """
    Persistent experiment history. Saved to disk as JSON.
    """

    STORAGE_PATH = "memory/experiments.json"

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or self.STORAGE_PATH
        self._runs: list[dict] = []
        self._load()
        # ── MLflow backend (additive — no-op if mlflow not installed) ──
        from aivc.memory.mlflow_backend import MLflowBackend
        self._mlflow = MLflowBackend()
        # ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load experiment history from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self._runs = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._runs = []
        else:
            self._runs = []

    def _save(self) -> None:
        """Persist experiment history to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self._runs, f, indent=2, default=str)

    def save_run(self, workflow_name: str, results: dict,
                 context=None) -> None:
        """
        Save a completed workflow run to persistent history.
        Extracts key metrics from results for comparison.
        """
        # Extract metrics from results
        metrics = {}
        for skill_name, result in results.items():
            if hasattr(result, "outputs"):
                for key in ["pearson_r_mean", "pearson_r_std",
                             "best_val_pearson_r", "n_pairs",
                             "jakstat_recovery_score", "mean_plausibility_score"]:
                    if key in result.outputs:
                        metrics[key] = result.outputs[key]
            if hasattr(result, "metadata"):
                for key in ["lfc_beta", "batch_size", "n_epochs",
                             "learning_rate", "split_method"]:
                    if key in result.metadata:
                        metrics[f"hp_{key}"] = result.metadata[key]

        run_record = {
            "workflow_name": workflow_name,
            "timestamp": time.time(),
            "metrics": metrics,
            "skills_executed": list(results.keys()),
            "success": all(
                getattr(r, "success", True) for r in results.values()
            ),
        }

        self._runs.append(run_record)
        self._save()

        # Mirror to MLflow if available (non-blocking — never raises)
        try:
            self._mlflow.log_final_metrics(metrics)
        except Exception:
            pass  # MLflow failure never breaks existing JSON path

    def get_best_run(self, metric: str = "pearson_r_mean") -> Optional[dict]:
        """
        Return the run with the highest value for the given metric.
        Returns None if no runs have this metric.
        """
        best = None
        best_value = float("-inf")
        for run in self._runs:
            value = run.get("metrics", {}).get(metric)
            if value is not None and value > best_value:
                best_value = value
                best = run
        return best

    def detect_regression(self, current_r: float,
                          threshold: float = 0.02) -> Optional[str]:
        """
        Alert if current Pearson r is more than threshold below best historical.
        Threshold of 0.02 is meaningful given our std of 0.064.
        Returns warning message if regression detected, None otherwise.
        """
        best = self.get_best_run("pearson_r_mean")
        if best is None:
            return None
        best_r = best["metrics"].get("pearson_r_mean", 0)
        if current_r < best_r - threshold:
            return (
                f"REGRESSION DETECTED: current Pearson r = {current_r:.4f}, "
                f"best historical = {best_r:.4f}, "
                f"delta = {current_r - best_r:.4f} "
                f"(threshold = {threshold})"
            )
        return None

    def get_hyperparameter_history(self) -> list[dict]:
        """
        Returns all runs with their hyperparameters and results.
        Used to guide lfc_beta sweep in Week 4.
        """
        history = []
        for run in self._runs:
            hp = {k: v for k, v in run.get("metrics", {}).items()
                  if k.startswith("hp_")}
            result_metrics = {k: v for k, v in run.get("metrics", {}).items()
                              if not k.startswith("hp_")}
            history.append({
                "timestamp": run.get("timestamp"),
                "workflow": run.get("workflow_name"),
                "hyperparameters": hp,
                "results": result_metrics,
                "success": run.get("success"),
            })
        return history

    def get_mlflow_summary(self) -> list:
        """
        Query MLflow for all v1.1 sweep runs and return a summary table.
        Returns JSON-serialisable list sorted by test_r descending.
        Falls back to JSON history if MLflow unavailable.
        """
        mlflow_summary = self._mlflow.get_sweep_summary()
        if mlflow_summary:
            return mlflow_summary
        # Fallback: return from JSON history
        return sorted(
            self.get_hyperparameter_history(),
            key=lambda r: r.get("results", {}).get("pearson_r_mean", 0),
            reverse=True,
        )

    def get_all_runs(self) -> list[dict]:
        """Return all stored runs."""
        return list(self._runs)
