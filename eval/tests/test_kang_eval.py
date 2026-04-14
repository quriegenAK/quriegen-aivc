"""
Tests for eval/benchmarks/kang_eval.py.
Uses unittest.mock — no live GPU or file I/O.
"""
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from unittest import mock

from eval.benchmarks.kang_eval import KangEvalReport, run_kang_eval, BASELINE_R, N_GENES


# Test 6: KangEvalReport with n_genes != 3010 raises ValidationError
def test_kang_report_wrong_n_genes():
    """n_genes must be exactly 3010."""
    with pytest.raises(ValidationError, match="3010"):
        KangEvalReport(run_id="test", n_genes=5000)


# Test 7: run_kang_eval on mock checkpoint with r < 0.873
@mock.patch("eval.benchmarks.kang_eval._load_model")
@mock.patch("eval.benchmarks.kang_eval._load_test_data")
@mock.patch("eval.benchmarks.kang_eval._load_gene_order")
def test_kang_regression_guard_fails(mock_genes, mock_data, mock_model):
    """Checkpoint with r < 0.873 → regression_guard_passed=False."""
    n_pairs = 10
    gene_order = [f"gene_{i}" for i in range(N_GENES)]
    mock_genes.return_value = gene_order

    # Create ctrl data (random) and make pred ≈ ctrl (low r with stim)
    np.random.seed(42)
    X_ctrl = np.random.rand(n_pairs, N_GENES).astype(np.float32)
    X_stim = X_ctrl + np.random.rand(n_pairs, N_GENES).astype(np.float32) * 2.0
    mock_data.return_value = (X_ctrl, X_stim, n_pairs)

    # Model that returns values close to ctrl (bad prediction → low r)
    fake_model = mock.MagicMock()
    # Return pred ≈ ctrl + small noise (will have low r with stim)
    pred_tensor = torch.tensor(
        X_ctrl + np.random.rand(n_pairs, N_GENES).astype(np.float32) * 0.01,
        dtype=torch.float32,
    )
    fake_model.forward_batch.return_value = pred_tensor
    fake_model.eval.return_value = None
    mock_model.return_value = fake_model

    report = run_kang_eval(
        "/fake/checkpoint.pt",
        run_id="test-regression",
        device="cpu",
    )

    assert not report.regression_guard_passed, "Should fail regression guard"
    assert not report.passed, "Should not pass overall"
    assert report.failure_reason is not None
    assert f"r=" in report.failure_reason
    assert str(BASELINE_R) in report.failure_reason
