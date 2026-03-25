"""
aivc/memory/mlflow_backend.py — MLflow logging backend for AIVC.

Provides structured experiment tracking for train_v11.py sweeps.
Wraps mlflow with graceful fallback when mlflow is not installed.

Design:
  - All public methods are safe to call regardless of mlflow availability.
  - If mlflow is not installed: operations are no-ops with logged warnings.
  - If mlflow server is unreachable: falls back to ./mlruns/ file store.
  - Never raises in production code paths.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("aivc.memory.mlflow")

MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning(
        "mlflow not installed. Experiment tracking disabled. "
        "Install with: pip install mlflow"
    )

# ─── AIVC experiment configuration ────────────────────────────────
EXPERIMENT_NAME = "aivc_v11_neumann_sweep"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FALLBACK_URI = "./mlruns"

# Tags applied to every run
PLATFORM_TAGS = {
    "platform": "AIVC",
    "model": "GeneLink-GAT-Neumann",
    "dataset": "Kang2018-PBMC-IFNb",
    "cell_types": "8-PBMC",
    "n_genes": "3010",
    "ppi_edges": "13878",
    "benchmark": "r=0.873-v1.0-baseline",
}


class MLflowBackend:
    """
    MLflow experiment tracking backend for AIVC training runs.

    Handles:
      - Experiment creation and run lifecycle
      - Hyperparameter logging (params)
      - Metric logging (per-epoch and final)
      - Artifact logging (model checkpoint, training log)
      - Sparsity history logging (W density trajectory)
      - JAK-STAT edge logging (top-20 biological edges)
      - Graceful fallback when MLflow unavailable

    Args:
        tracking_uri:   MLflow tracking server URI.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        tracking_uri: str = TRACKING_URI,
        experiment_name: str = EXPERIMENT_NAME,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active_run = None
        self._run_id: Optional[str] = None
        self._available = MLFLOW_AVAILABLE

        if self._available:
            self._initialise()

    def _initialise(self) -> None:
        """Connect to MLflow server or fall back to local file store."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.search_experiments()
            logger.info(f"MLflow connected: {self.tracking_uri}")
        except Exception:
            logger.warning(
                f"MLflow server unreachable at {self.tracking_uri}. "
                f"Falling back to local file store: {FALLBACK_URI}"
            )
            mlflow.set_tracking_uri(FALLBACK_URI)

        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    tags=PLATFORM_TAGS,
                )
                logger.info(f"MLflow experiment created: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"MLflow experiment setup failed: {e}")
            self._available = False

    # ─── Run lifecycle ─────────────────────────────────────────────

    def start_run(
        self,
        run_name: str,
        params: dict,
        tags: dict = None,
    ) -> Optional[str]:
        """Start a new MLflow run. Returns run_id or None."""
        if not self._available:
            return None
        try:
            all_tags = {**PLATFORM_TAGS, **(tags or {})}
            self._active_run = mlflow.start_run(
                run_name=run_name,
                tags=all_tags,
            )
            self._run_id = self._active_run.info.run_id
            clean_params = {str(k): str(v) for k, v in params.items()}
            mlflow.log_params(clean_params)
            logger.info(f"MLflow run started: {run_name} (id={self._run_id})")
            return self._run_id
        except Exception as e:
            logger.error(f"MLflow start_run failed: {e}")
            return None

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run."""
        if not self._available or self._active_run is None:
            return
        try:
            mlflow.end_run(status=status)
            self._active_run = None
            self._run_id = None
        except Exception as e:
            logger.error(f"MLflow end_run failed: {e}")

    # ─── Metric logging ────────────────────────────────────────────

    def log_epoch_metrics(
        self,
        epoch: int,
        train_losses: dict,
        val_r: float,
        w_density: float,
        w_density_large: float,
        n_pruned: int,
    ) -> None:
        """Log per-epoch training metrics."""
        if not self._available:
            return
        try:
            metrics = {
                "train/loss_mse": train_losses.get("mse", 0.0),
                "train/loss_lfc": train_losses.get("lfc", 0.0),
                "train/loss_cosine": train_losses.get("cosine", 0.0),
                "train/loss_l1": train_losses.get("l1", 0.0),
                "train/loss_total": train_losses.get("total", 0.0),
                "val/pearson_r": val_r,
                "w_matrix/density": w_density,
                "w_matrix/density_large": w_density_large,
                "w_matrix/edges_pruned": float(n_pruned),
            }
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            logger.debug(f"MLflow epoch metric logging failed (non-fatal): {e}")

    def log_final_metrics(self, result: dict) -> None:
        """Log final test-set metrics at end of training run."""
        if not self._available:
            return
        try:
            core = {
                "test/pearson_r": result.get("test_r", 0.0),
                "test/pearson_r_std": result.get("test_std", 0.0),
                "val/best_pearson_r": result.get("best_val_r", 0.0),
                "val/best_epoch": float(result.get("best_epoch", 0)),
            }
            jakstat = {
                "jakstat/recovery_3x": float(result.get("jakstat_within_3x", 0)),
                "jakstat/recovery_10x": float(result.get("jakstat_within_10x", 0)),
                "jakstat/ifit1_pred_fc": result.get("ifit1_pred_fc", 0.0),
                "jakstat/ifit1_actual_fc": result.get("ifit1_actual_fc", 0.0),
                "jakstat/ifit1_fc_ratio": (
                    result.get("ifit1_pred_fc", 0.0)
                    / max(result.get("ifit1_actual_fc", 1.0), 1e-6)
                ),
            }
            ct_r = result.get("ct_r", {})
            cell_type_metrics = {}
            for ct_name, r_val in ct_r.items():
                safe_name = ct_name.replace(" ", "_").replace("+", "pos")
                cell_type_metrics[f"celltype/{safe_name}_r"] = float(r_val)

            w_matrix = {
                "w_matrix/final_density": result.get("w_density", 1.0),
            }
            perf = {
                "perf/training_time_s": result.get("training_time_s", 0.0),
                "perf/training_time_min": result.get("training_time_s", 0.0) / 60.0,
            }
            baseline_r = 0.873
            regression_delta = result.get("test_r", 0.0) - baseline_r
            guard = {
                "guard/r_vs_baseline": regression_delta,
                "guard/regression_detected": float(regression_delta < -0.01),
            }
            all_metrics = {
                **core, **jakstat, **cell_type_metrics,
                **w_matrix, **perf, **guard,
            }
            mlflow.log_metrics(all_metrics)
        except Exception as e:
            logger.error(f"MLflow final metric logging failed: {e}")

    def log_sparsity_history(self, sparsity_history: list) -> None:
        """Log W matrix sparsity trajectory as a time-series."""
        if not self._available or not sparsity_history:
            return
        try:
            for record in sparsity_history:
                epoch = record.get("epoch", 0)
                mlflow.log_metrics(
                    {
                        "w_matrix/density": record.get("density", 1.0),
                        "w_matrix/density_large": record.get("density_large", 1.0),
                        "w_matrix/edges_pruned": float(record.get("n_pruned", 0)),
                        "val/pearson_r_at_sparsity": record.get("val_r", 0.0),
                    },
                    step=epoch,
                )
        except Exception as e:
            logger.debug(f"MLflow sparsity history logging failed (non-fatal): {e}")

    def log_top_edges(self, top_edges: list, n_jakstat_in_top20: int) -> None:
        """Log Neumann W matrix top-20 edges as MLflow artifact."""
        if not self._available:
            return
        try:
            mlflow.log_metric("jakstat/edges_in_top20_w", float(n_jakstat_in_top20))

            lines = [
                "Rank  Source          Dest            Weight      JAK-STAT",
                "-" * 60,
            ]
            JAKSTAT_PAIRS = {
                ("JAK1", "STAT1"), ("JAK1", "STAT2"),
                ("JAK2", "STAT1"), ("STAT1", "IFIT1"),
                ("STAT1", "MX1"), ("STAT2", "IFIT1"),
                ("IRF9", "IFIT1"), ("STAT1", "ISG15"),
            }
            for edge in top_edges:
                src = edge.get("src_name") or str(edge.get("src_idx", "?"))
                dst = edge.get("dst_name") or str(edge.get("dst_idx", "?"))
                w = edge.get("weight", 0.0)
                is_jakstat = (src, dst) in JAKSTAT_PAIRS
                marker = " *" if is_jakstat else ""
                lines.append(
                    f"{edge.get('rank', 0):<5} {src:<15} {dst:<15} "
                    f"{w:>10.6f}{marker}"
                )
            lines.append(f"\nJAK-STAT edges in top-20: {n_jakstat_in_top20}/8")

            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="top_w_edges_", delete=False
            ) as f:
                f.write("\n".join(lines))
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="w_matrix")
            os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"MLflow top edges logging failed (non-fatal): {e}")

    def log_model_checkpoint(self, checkpoint_path: str) -> None:
        """Log model checkpoint as MLflow artifact."""
        if not self._available:
            return
        try:
            if os.path.exists(checkpoint_path):
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        except Exception as e:
            logger.debug(f"MLflow checkpoint logging failed (non-fatal): {e}")

    # ─── Query interface ───────────────────────────────────────────

    def get_best_run(
        self, metric: str = "test/pearson_r", min_r: float = 0.873
    ) -> Optional[dict]:
        """Query MLflow for the best run by metric."""
        if not self._available:
            return None
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=f"metrics.`test/pearson_r` >= {min_r}",
                order_by=[f"metrics.`{metric}` DESC"],
                max_results=1,
            )
            if runs.empty:
                return None
            row = runs.iloc[0]
            return {
                "run_id": row.get("run_id"),
                "params": {
                    k.replace("params.", ""): v
                    for k, v in row.items() if k.startswith("params.")
                },
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in row.items() if k.startswith("metrics.")
                },
            }
        except Exception as e:
            logger.error(f"MLflow query failed: {e}")
            return None

    def get_sweep_summary(self) -> list:
        """Return all v1.1 sweep runs sorted by test Pearson r."""
        if not self._available:
            return []
        try:
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.`test/pearson_r` DESC"],
                max_results=100,
            )
            summary = []
            for _, row in runs.iterrows():
                def get(col, default=0.0):
                    val = row.get(col)
                    return val if val is not None else default

                summary.append({
                    "run_id": row.get("run_id", ""),
                    "lfc_beta": get("params.lfc_beta"),
                    "neumann_K": get("params.neumann_K"),
                    "lambda_l1": get("params.lambda_l1"),
                    "test_r": get("metrics.test/pearson_r"),
                    "jakstat_3x": get("metrics.jakstat/recovery_3x"),
                    "ifit1_pred_fc": get("metrics.jakstat/ifit1_pred_fc"),
                    "cd14_r": get("metrics.celltype/CD14pos_Monocytes_r"),
                    "w_density": get("metrics.w_matrix/final_density"),
                    "training_time_min": get("metrics.perf/training_time_min"),
                    "regression": get("metrics.guard/regression_detected") > 0.5,
                })
            return summary
        except Exception as e:
            logger.error(f"MLflow sweep summary query failed: {e}")
            return []

    def print_sweep_table(self) -> None:
        """Print a formatted comparison table of all sweep runs."""
        summary = self.get_sweep_summary()
        if not summary:
            print("No MLflow runs found. Train first or check tracking URI.")
            return

        print(f"\n{'='*70}")
        print(f"AIVC v1.1 Neumann Sweep — MLflow Summary")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*70}")
        print(
            f"  {'Beta':<6} {'K':<4} {'L1':<8} {'Test r':<8} "
            f"{'JS 3x':<7} {'IFIT1 FC':<10} {'CD14 r':<8} "
            f"{'W dens':<8} {'Time(m)'}"
        )
        print(f"  {'-'*68}")
        for i, run in enumerate(summary):
            marker = " <- BEST" if i == 0 else ""
            reg_marker = " !! REGRESSION" if run["regression"] else ""
            print(
                f"  {run['lfc_beta']:<6} "
                f"{run['neumann_K']:<4} "
                f"{run['lambda_l1']:<8} "
                f"{float(run['test_r']):<8.4f} "
                f"{int(float(run['jakstat_3x']))}/15   "
                f"{float(run['ifit1_pred_fc']):<10.1f} "
                f"{float(run['cd14_r']):<8.4f} "
                f"{float(run['w_density']):<8.3f} "
                f"{float(run['training_time_min']):.1f}"
                f"{marker}{reg_marker}"
            )
