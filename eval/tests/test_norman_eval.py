"""
Tests for eval/benchmarks/norman_eval.py.
Uses unittest.mock — no live GPU or file I/O.
"""
import logging

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from unittest import mock

from eval.benchmarks.norman_eval import (
    NormanEvalReport,
    run_norman_eval,
    N_GENES,
)


# Test 8: NormanEvalReport collapse invariant
def test_norman_collapse_invariant():
    """delta_nonzero_pct=0 with ctrl_memorisation_score < 0.99 → ValidationError."""
    with pytest.raises(ValidationError, match="Collapse invariant"):
        NormanEvalReport(
            run_id="test",
            delta_nonzero_pct=0.0,
            ctrl_memorisation_score=0.5,
        )


# Test 9: NaN pearson_r_ctrl_sub coercion
@mock.patch("eval.benchmarks.norman_eval._load_model")
@mock.patch("eval.benchmarks.norman_eval._load_norman_data")
@mock.patch("eval.benchmarks.norman_eval._load_gene_order")
def test_nan_pearson_coerced(mock_genes, mock_data, mock_model, caplog):
    """NaN pearson_r_ctrl_sub is coerced to 0.0 with warning."""
    gene_order = [f"gene_{i}" for i in range(N_GENES)]
    mock_genes.return_value = gene_order

    n_pairs = 5
    # Make ctrl == pert so truth delta is zero → NaN Pearson
    X_ctrl = np.random.rand(n_pairs, N_GENES).astype(np.float32)
    X_pert = X_ctrl.copy()  # identical → degenerate truth delta
    mock_data.return_value = (X_ctrl, X_pert, 10, n_pairs, 0)

    fake_model = mock.MagicMock()
    # Return something different from ctrl so delta_nonzero > 0
    pred = torch.tensor(
        X_ctrl + np.ones((n_pairs, N_GENES), dtype=np.float32) * 0.5,
        dtype=torch.float32,
    )
    fake_model.forward_batch.return_value = pred
    fake_model.eval.return_value = None
    mock_model.return_value = fake_model

    with caplog.at_level(logging.WARNING, logger="aivc.eval.norman"):
        report = run_norman_eval(
            "/fake/checkpoint.pt",
            run_id="test-nan",
            device="cpu",
        )

    assert report.pearson_r_ctrl_sub == 0.0, "NaN should be coerced to 0.0"
    assert any("NaN" in r.message or "coercing to 0.0" in r.message for r in caplog.records), \
        "Should log warning about NaN coercion"


# Test 10: Norman ctrl mask uses nperts==0
def test_norman_ctrl_mask_uses_nperts():
    """Verify _load_norman_data identifies ctrl cells by nperts==0, not perturbation=='ctrl'."""
    import anndata as ad
    import pandas as pd
    from eval.benchmarks.norman_eval import _load_norman_data

    gene_order = [f"gene_{i}" for i in range(N_GENES)]

    n_ctrl = 50
    n_pert = 50
    n_total = n_ctrl + n_pert

    # Create fake AnnData: ctrl cells have nperts=0 but perturbation="ctrl_label"
    # (NOT "ctrl"). If code checks perturbation=="ctrl", it would find 0 ctrl cells.
    X = np.random.rand(n_total, N_GENES).astype(np.float32)
    obs = pd.DataFrame({
        "nperts": [0] * n_ctrl + [1] * n_pert,
        "perturbation": ["ctrl_label"] * n_ctrl + ["GENE_A"] * n_pert,
        "gemgroup": [1] * n_total,
    })
    var = pd.DataFrame(index=gene_order)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    with mock.patch("anndata.read_h5ad", return_value=adata):
        X_ctrl_pb, X_pert_pb, n_perts, n_pairs, n_dropped = _load_norman_data(
            "/fake/norman.h5ad",
            gene_order,
            min_cells=10,
        )

    # Key assertion: we got 1 pair (one perturbation "GENE_A" × one gemgroup 1).
    # If code used perturbation=="ctrl" instead of nperts==0, ctrl set would be
    # empty and all groups would be dropped.
    assert n_pairs == 1, (
        f"Expected 1 pseudo-bulk pair (nperts==0 ctrl). Got {n_pairs}. "
        "Code may be using perturbation=='ctrl' instead of nperts==0."
    )
    assert X_ctrl_pb.shape == (1, N_GENES)
    assert X_pert_pb.shape == (1, N_GENES)
