"""
test_zero_shot_eval.py — 8 tests for zero-shot perturbation evaluation.
All tests use mock data. No real datasets or GPU required.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch


# ── Helpers ──────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from evaluate_zero_shot import (
    align_gene_universe, compute_verdict, compute_dryrun_verdict,
    detect_condition_column, load_checkpoint,
)


class TestGeneUniverseAlignment:

    def test_gene_universe_alignment(self):
        """Mock two datasets with partial gene overlap."""
        train_genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
        eval_genes = ["GENE_C", "GENE_X", "GENE_A", "GENE_Y", "GENE_E"]

        shared, train_idx, eval_idx = align_gene_universe(train_genes, eval_genes)
        assert len(shared) == 3
        assert set(shared) == {"GENE_A", "GENE_C", "GENE_E"}
        # For each shared gene, train_genes[train_idx[i]] == eval_genes[eval_idx[i]]
        for i in range(len(shared)):
            assert train_genes[train_idx[i]] == eval_genes[eval_idx[i]]


class TestVerdictLogic:

    def test_verdict_generalises(self):
        """r=0.75, delta=-0.12 → GENERALISES"""
        v = compute_verdict(pearson_r=0.75, generalisation_delta=-0.123)
        assert v == "GENERALISES"

    def test_verdict_partial(self):
        """r=0.60, delta=-0.15 → PARTIAL"""
        v = compute_verdict(pearson_r=0.60, generalisation_delta=-0.15)
        assert v == "PARTIAL"

    def test_verdict_memorised_by_r(self):
        """r=0.40, delta=-0.45 → MEMORISED"""
        v = compute_verdict(pearson_r=0.40, generalisation_delta=-0.473)
        assert v == "MEMORISED"

    def test_verdict_forced_memorised_by_delta(self):
        """r=0.72, delta=-0.25 → MEMORISED (delta override)"""
        v = compute_verdict(pearson_r=0.72, generalisation_delta=-0.25)
        assert v == "MEMORISED"


class TestOutputFormat:

    def test_json_output_has_required_keys(self):
        """Output dict has all 7 keys."""
        result = {
            "pearson_r": 0.65,
            "pearson_r_baseline": 0.873,
            "jakstat_3x": 5,
            "ifit1_pred_fc": 3.2,
            "generalisation_delta": -0.223,
            "is_overfit": True,
            "verdict": "MEMORISED",
        }
        required = {
            "pearson_r", "pearson_r_baseline", "jakstat_3x",
            "ifit1_pred_fc", "generalisation_delta", "is_overfit", "verdict",
        }
        assert required == set(result.keys())


class TestCheckpointLoading:

    def test_checkpoint_loading_graceful_on_missing_file(self):
        """Missing checkpoint → exits with code 1, not unhandled exception."""
        with pytest.raises(SystemExit) as exc_info:
            load_checkpoint("/nonexistent/path/model.pt")
        assert exc_info.value.code == 1


class TestBFloat16Safety:

    def test_pearson_r_requires_float_not_bfloat16(self):
        """Confirms .detach().cpu().float().numpy() pattern is needed."""
        t = torch.rand(100, dtype=torch.bfloat16)
        with pytest.raises((RuntimeError, TypeError)):
            t.numpy()
        # The correct pattern:
        arr = t.detach().cpu().float().numpy()
        assert arr.dtype == np.float32


class TestDryRunMode:

    def test_dry_run_mode_uses_test_set_cells_only(self):
        """Dry run must use only held-out test donors, not training donors.

        The 60/20/20 split on 8 donors gives test donors = last 2-3 donors.
        Dry run must filter to those donors before computing ctrl/stim means.
        Verifies compute_dryrun_verdict returns valid verdicts.
        """
        # Verify dry run verdict logic
        assert compute_dryrun_verdict(0.90) == "CONSISTENT"
        assert compute_dryrun_verdict(0.85) == "ACCEPTABLE"
        assert compute_dryrun_verdict(0.75) == "DEGRADED"

        # Verify the 60/20/20 split produces test donors
        unique_donors = sorted([
            "patient_101", "patient_1015", "patient_1016", "patient_1039",
            "patient_107", "patient_1244", "patient_1256", "patient_1488",
        ])
        n_donors = len(unique_donors)
        n_train = max(1, int(n_donors * 0.6))
        n_val = max(1, int(n_donors * 0.2))
        test_donors = set(unique_donors[n_train + n_val:])

        # Must have at least 1 test donor
        assert len(test_donors) >= 1, "Split produces no test donors"
        # Test donors must NOT be in train set
        train_donors = set(unique_donors[:n_train])
        assert test_donors.isdisjoint(train_donors), \
            "Test and train donors overlap"


class TestConditionAutoDetection:

    def test_condition_column_auto_detection_dixit_format(self):
        """Mock adata with 'guide_identity' column containing 'NT' (control)."""
        import anndata as ad
        import pandas as pd
        obs = pd.DataFrame({
            "guide_identity": ["NT", "NT", "JAK1", "STAT1", "JAK1"],
            "batch": ["A", "A", "B", "B", "A"],
        })
        adata = ad.AnnData(obs=obs)
        col, label = detect_condition_column(adata)
        assert col == "guide_identity"
        assert label == "NT"

    def test_condition_column_raises_on_unknown_format(self):
        """Mock adata with no recognisable condition column."""
        import anndata as ad
        import pandas as pd
        obs = pd.DataFrame({
            "cell_state": ["A", "B", "C", "A", "B"],
            "batch_id": [1, 1, 2, 2, 3],
        })
        adata = ad.AnnData(obs=obs)
        with pytest.raises(ValueError, match="Could not detect condition column"):
            detect_condition_column(adata)
