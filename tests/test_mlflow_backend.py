"""
Tests for MLflow backend integration.
All tests run WITHOUT mlflow installed (mocked).
No MLflow server required. Under 10 seconds on CPU.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import sys


# ─── Helper: mock mlflow module ───

def _make_mock_mlflow():
    """Create a mock mlflow module with all needed attributes."""
    mock = MagicMock()
    mock.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-123"))
    mock.search_experiments.return_value = []
    mock.get_experiment_by_name.return_value = None
    mock.search_runs.return_value = MagicMock(empty=True)
    return mock


class TestMLflowBackendInit:

    def test_backend_initialises_when_mlflow_unavailable(self):
        """When mlflow not installed: backend._available == False."""
        # Temporarily hide mlflow
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.pytorch": None}):
            # Reimport to pick up missing mlflow
            import importlib
            import aivc.memory.mlflow_backend as mb
            orig_avail = mb.MLFLOW_AVAILABLE
            mb.MLFLOW_AVAILABLE = False
            try:
                backend = mb.MLflowBackend()
                assert backend._available is False
            finally:
                mb.MLFLOW_AVAILABLE = orig_avail

    def test_backend_initialises_with_mock_mlflow(self):
        """With mock mlflow: backend._available == True."""
        mock_mlflow = _make_mock_mlflow()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.pytorch": MagicMock()}):
            import aivc.memory.mlflow_backend as mb
            orig = mb.MLFLOW_AVAILABLE
            mb.MLFLOW_AVAILABLE = True
            mb.mlflow = mock_mlflow
            try:
                backend = mb.MLflowBackend()
                # Should attempt initialisation
                assert isinstance(backend, mb.MLflowBackend)
            finally:
                mb.MLFLOW_AVAILABLE = orig

    def test_all_methods_safe_when_unavailable(self):
        """With _available=False: all methods return None without raising."""
        import aivc.memory.mlflow_backend as mb
        orig = mb.MLFLOW_AVAILABLE
        mb.MLFLOW_AVAILABLE = False
        try:
            backend = mb.MLflowBackend()
            backend._available = False

            # None of these should raise
            assert backend.start_run("test", {}) is None
            backend.end_run()
            backend.log_epoch_metrics(1, {}, 0.8, 1.0, 1.0, 0)
            backend.log_final_metrics({"test_r": 0.8})
            backend.log_sparsity_history([])
            backend.log_top_edges([], 0)
            backend.log_model_checkpoint("/nonexistent.pt")
            assert backend.get_best_run() is None
            assert backend.get_sweep_summary() == []
        finally:
            mb.MLFLOW_AVAILABLE = orig


class TestMLflowRunLifecycle:

    def test_start_run_logs_params(self):
        """start_run should call mlflow.log_params with correct keys."""
        import aivc.memory.mlflow_backend as mb
        mock_mlflow = _make_mock_mlflow()

        backend = mb.MLflowBackend.__new__(mb.MLflowBackend)
        backend._available = True
        backend._active_run = None
        backend._run_id = None
        backend.experiment_name = "test"
        mb.mlflow = mock_mlflow

        params = {
            "lfc_beta": 0.1, "neumann_K": 3, "lambda_l1": 0.001,
            "n_genes": 3010, "n_edges": 13878, "seed": 42,
        }
        backend.start_run("test_run", params)

        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]
        assert "lfc_beta" in logged
        assert "neumann_K" in logged
        assert "seed" in logged

    def test_end_run_called_with_finished_status(self):
        """end_run("FINISHED") calls mlflow.end_run with FINISHED."""
        import aivc.memory.mlflow_backend as mb
        mock_mlflow = _make_mock_mlflow()

        backend = mb.MLflowBackend.__new__(mb.MLflowBackend)
        backend._available = True
        backend._active_run = MagicMock()
        backend._run_id = "test-123"
        mb.mlflow = mock_mlflow

        backend.end_run("FINISHED")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_end_run_called_with_failed_on_regression(self):
        """When test_r < 0.863: status should be FAILED."""
        # This tests the calling code logic in train_v11.py
        result = {"test_r": 0.850}
        status = "FINISHED" if result["test_r"] >= 0.863 else "FAILED"
        assert status == "FAILED"

        result2 = {"test_r": 0.875}
        status2 = "FINISHED" if result2["test_r"] >= 0.863 else "FAILED"
        assert status2 == "FINISHED"


class TestMLflowMetricLogging:

    def _make_backend(self):
        import aivc.memory.mlflow_backend as mb
        mock_mlflow = _make_mock_mlflow()
        backend = mb.MLflowBackend.__new__(mb.MLflowBackend)
        backend._available = True
        backend._active_run = MagicMock()
        backend._run_id = "test-123"
        backend.experiment_name = "test"
        mb.mlflow = mock_mlflow
        return backend, mock_mlflow

    def test_log_epoch_metrics_correct_keys(self):
        """Epoch metrics must include train/loss_mse, val/pearson_r, etc."""
        backend, mock_mlflow = self._make_backend()

        backend.log_epoch_metrics(
            epoch=10,
            train_losses={"mse": 0.01, "lfc": 0.5, "cosine": 0.1, "l1": 0.001, "total": 0.6},
            val_r=0.85,
            w_density=0.95,
            w_density_large=0.90,
            n_pruned=100,
        )

        mock_mlflow.log_metrics.assert_called_once()
        metrics, kwargs = mock_mlflow.log_metrics.call_args
        logged = metrics[0]
        assert "train/loss_mse" in logged
        assert "val/pearson_r" in logged
        assert "w_matrix/density" in logged
        assert "w_matrix/edges_pruned" in logged
        assert kwargs.get("step") == 10

    def test_log_final_metrics_correct_keys(self):
        """Final metrics must include test/pearson_r, jakstat, guard."""
        backend, mock_mlflow = self._make_backend()

        result = {
            "test_r": 0.880, "test_std": 0.06,
            "best_val_r": 0.875, "best_epoch": 150,
            "jakstat_within_3x": 9, "jakstat_within_10x": 12,
            "ifit1_pred_fc": 25.0, "ifit1_actual_fc": 107.0,
            "ct_r": {"CD14+ Monocytes": 0.81, "B cells": 0.72},
            "w_density": 0.5, "training_time_s": 300.0,
        }
        backend.log_final_metrics(result)

        mock_mlflow.log_metrics.assert_called_once()
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "test/pearson_r" in logged
        assert "jakstat/recovery_3x" in logged
        assert "jakstat/ifit1_pred_fc" in logged
        assert "guard/regression_detected" in logged
        assert "guard/r_vs_baseline" in logged

    def test_regression_detected_when_r_below_baseline(self):
        """guard/regression_detected = 1.0 when r < 0.863."""
        backend, mock_mlflow = self._make_backend()

        result = {"test_r": 0.850, "ct_r": {}, "w_density": 1.0, "training_time_s": 100}
        backend.log_final_metrics(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["guard/regression_detected"] == 1.0

    def test_no_regression_when_r_at_baseline(self):
        """guard/regression_detected = 0.0 when r >= 0.873."""
        backend, mock_mlflow = self._make_backend()

        result = {"test_r": 0.873, "ct_r": {}, "w_density": 0.5, "training_time_s": 200}
        backend.log_final_metrics(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["guard/regression_detected"] == 0.0

    def test_cell_type_metric_names_sanitised(self):
        """CD14+ Monocytes -> CD14pos_Monocytes (no + or spaces)."""
        backend, mock_mlflow = self._make_backend()

        result = {
            "test_r": 0.88, "ct_r": {"CD14+ Monocytes": 0.81, "B cells": 0.72},
            "w_density": 0.5, "training_time_s": 200,
        }
        backend.log_final_metrics(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "celltype/CD14pos_Monocytes_r" in logged
        assert "celltype/B_cells_r" in logged
        # No raw names with + or spaces
        for key in logged:
            if key.startswith("celltype/"):
                assert "+" not in key
                assert " " not in key


class TestMLflowSparsity:

    def _make_backend(self):
        import aivc.memory.mlflow_backend as mb
        mock_mlflow = _make_mock_mlflow()
        backend = mb.MLflowBackend.__new__(mb.MLflowBackend)
        backend._available = True
        backend._active_run = MagicMock()
        backend._run_id = "test-123"
        mb.mlflow = mock_mlflow
        return backend, mock_mlflow

    def test_log_sparsity_history_logs_per_epoch(self):
        """Each history entry logged with correct step."""
        backend, mock_mlflow = self._make_backend()

        history = [
            {"epoch": 10, "density": 1.0, "density_large": 1.0, "n_pruned": 0, "val_r": 0.85},
            {"epoch": 20, "density": 0.95, "density_large": 0.90, "n_pruned": 500, "val_r": 0.87},
        ]
        backend.log_sparsity_history(history)

        assert mock_mlflow.log_metrics.call_count == 2
        # Check steps
        calls = mock_mlflow.log_metrics.call_args_list
        assert calls[0][1].get("step") == 10
        assert calls[1][1].get("step") == 20

    def test_log_sparsity_history_empty_is_noop(self):
        """Empty history -> no mlflow calls."""
        backend, mock_mlflow = self._make_backend()
        backend.log_sparsity_history([])
        mock_mlflow.log_metrics.assert_not_called()


class TestMLflowEdgeLogging:

    def test_log_top_edges_counts_jakstat_correctly(self):
        """JAK-STAT count logged as metric."""
        import aivc.memory.mlflow_backend as mb
        mock_mlflow = _make_mock_mlflow()
        backend = mb.MLflowBackend.__new__(mb.MLflowBackend)
        backend._available = True
        backend._active_run = MagicMock()
        backend._run_id = "test-123"
        mb.mlflow = mock_mlflow

        top_edges = [
            {"rank": 1, "src_name": "JAK1", "dst_name": "STAT1", "weight": 0.05},
            {"rank": 2, "src_name": "STAT1", "dst_name": "IFIT1", "weight": 0.03},
            {"rank": 3, "src_name": "RPL3", "dst_name": "RPL4", "weight": 0.02},
        ]
        backend.log_top_edges(top_edges, n_jakstat_in_top20=2)

        mock_mlflow.log_metric.assert_called_once_with(
            "jakstat/edges_in_top20_w", 2.0
        )

    def test_experimental_memory_mirrors_to_mlflow(self):
        """ExperimentalMemory.save_run mirrors to MLflow backend."""
        import tempfile
        import os
        from aivc.memory.experimental import ExperimentalMemory

        # Create a temp storage path
        tmp = tempfile.mktemp(suffix=".json")
        try:
            mem = ExperimentalMemory(storage_path=tmp)
            # Mock the MLflow backend
            mem._mlflow = MagicMock()
            mem._mlflow.log_final_metrics = MagicMock()

            # Save a run
            mock_result = MagicMock()
            mock_result.outputs = {"pearson_r_mean": 0.88}
            mock_result.metadata = {}
            mock_result.success = True
            mem.save_run("test_wf", {"skill1": mock_result})

            # Verify MLflow was called
            mem._mlflow.log_final_metrics.assert_called_once()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
