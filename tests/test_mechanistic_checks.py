"""
Tests for mechanistic direction checks in BiologicalCritic.
All tests use mock data. No real files. Under 10 seconds on CPU.

Run: pytest tests/test_mechanistic_checks.py -v
"""
import pytest
import numpy as np
from aivc.critics.biological import BiologicalCritic
from aivc.interfaces import SkillResult


def make_skill_result(**outputs):
    return SkillResult(
        skill_name="test", version="0.0.0", success=True,
        outputs=outputs, metadata={}, warnings=[], errors=[],
    )


def make_gene_array(n_cells=10, n_genes=20, gene_to_idx=None,
                    ctrl_val=1.0, stim_vals=None):
    ctrl = np.full((n_cells, n_genes), ctrl_val, dtype=np.float32)
    pred = ctrl.copy()
    if stim_vals and gene_to_idx:
        for gene, fc in stim_vals.items():
            if gene in gene_to_idx:
                pred[:, gene_to_idx[gene]] = ctrl_val * fc
    return ctrl, pred


GENES = [
    "STAT1", "STAT3", "IFIT1", "OAS2",
    "JAK1", "JAK2", "MX1", "ISG15",
] + [f"GENE_{i}" for i in range(12)]
GENE_TO_IDX = {g: i for i, g in enumerate(GENES)}


class TestStat1OverStat3:

    def test_passes_when_stat1_higher(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX, stim_vals={"STAT1": 5.0, "STAT3": 2.0},
        )
        result = BiologicalCritic()._check_stat1_over_stat3(pred, ctrl, GENE_TO_IDX)
        assert result["passed"] is True
        assert result["stat1_fc"] > result["stat3_fc"]

    def test_fails_when_stat3_higher(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX, stim_vals={"STAT1": 2.0, "STAT3": 5.0},
        )
        result = BiologicalCritic()._check_stat1_over_stat3(pred, ctrl, GENE_TO_IDX)
        assert result["passed"] is False
        assert "IL-6" in result["message"]

    def test_skips_when_gene_missing(self):
        ctrl, pred = make_gene_array()
        result = BiologicalCritic()._check_stat1_over_stat3(pred, ctrl, {"GENEONLY": 0})
        assert result["passed"] is True
        assert result.get("skipped") is True


class TestIfit1OverOas2:

    def test_passes_when_ifit1_higher(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX, stim_vals={"IFIT1": 50.0, "OAS2": 20.0},
        )
        result = BiologicalCritic()._check_ifit1_over_oas2(pred, ctrl, GENE_TO_IDX)
        assert result["passed"] is True

    def test_fails_when_oas2_higher(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX, stim_vals={"IFIT1": 10.0, "OAS2": 30.0},
        )
        result = BiologicalCritic()._check_ifit1_over_oas2(pred, ctrl, GENE_TO_IDX)
        assert result["passed"] is False
        assert "ISG" in result["message"]

    def test_skips_when_gene_missing(self):
        ctrl, pred = make_gene_array()
        result = BiologicalCritic()._check_ifit1_over_oas2(pred, ctrl, {"X": 0})
        assert result["passed"] is True
        assert result.get("skipped") is True


class TestJak1UpstreamIfit1:

    def test_passes_with_positive_weights(self):
        edges = [
            {"src_name": "JAK1", "dst_name": "STAT1", "weight": 0.05},
            {"src_name": "STAT1", "dst_name": "IFIT1", "weight": 0.03},
        ]
        result = BiologicalCritic()._check_jak1_upstream_of_ifit1(edges, GENE_TO_IDX)
        assert result["passed"] is True
        assert result["jak1_stat1_weight"] == pytest.approx(0.05)

    def test_fails_with_negative_jak1_stat1(self):
        edges = [
            {"src_name": "JAK1", "dst_name": "STAT1", "weight": -0.02},
            {"src_name": "STAT1", "dst_name": "IFIT1", "weight": 0.03},
        ]
        result = BiologicalCritic()._check_jak1_upstream_of_ifit1(edges, GENE_TO_IDX)
        assert result["passed"] is False

    def test_fails_with_missing_edge(self):
        edges = [{"src_name": "MX1", "dst_name": "ISG15", "weight": 0.02}]
        result = BiologicalCritic()._check_jak1_upstream_of_ifit1(edges, GENE_TO_IDX)
        assert result["passed"] is False
        assert result["jak1_stat1_weight"] == pytest.approx(0.0)
        assert result["stat1_ifit1_weight"] == pytest.approx(0.0)

    def test_empty_edges_fails_gracefully(self):
        result = BiologicalCritic()._check_jak1_upstream_of_ifit1([], GENE_TO_IDX)
        assert result["passed"] is False

    def test_none_edges_fails_gracefully(self):
        result = BiologicalCritic()._check_jak1_upstream_of_ifit1(None, GENE_TO_IDX)
        assert result["passed"] is False


class TestValidateIntegration:

    def test_direction_checks_in_validate_output(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX,
            stim_vals={"STAT1": 5.0, "STAT3": 2.0, "IFIT1": 40.0, "OAS2": 15.0},
        )
        result = make_skill_result(
            pred_np=pred, ctrl_np=ctrl, gene_to_idx=GENE_TO_IDX,
            jakstat_recovery_score=8,
        )
        report = BiologicalCritic().validate(result)
        all_messages = " ".join(report.checks_passed + report.uncertainty_flags)
        assert "direction" in all_messages.lower() or "STAT1" in all_messages

    def test_direction_failures_are_warnings_not_blocking(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX,
            stim_vals={"STAT1": 1.5, "STAT3": 5.0, "IFIT1": 10.0, "OAS2": 30.0},
        )
        result = make_skill_result(
            pred_np=pred, ctrl_np=ctrl, gene_to_idx=GENE_TO_IDX,
            jakstat_recovery_score=8,
        )
        report = BiologicalCritic().validate(result)
        # Direction failures must NOT be in checks_failed
        direction_in_failed = any(
            "direction" in c.lower() or "STAT" in c
            for c in report.checks_failed
        )
        assert not direction_in_failed
        # They must be in warnings
        has_warning = any(
            "direction" in w.lower() or "STAT" in w
            for w in report.uncertainty_flags
        )
        assert has_warning

    def test_recommendation_set_when_direction_fails(self):
        ctrl, pred = make_gene_array(
            gene_to_idx=GENE_TO_IDX,
            stim_vals={"STAT1": 1.0, "STAT3": 8.0},
        )
        result = make_skill_result(
            pred_np=pred, ctrl_np=ctrl, gene_to_idx=GENE_TO_IDX,
        )
        report = BiologicalCritic().validate(result)
        assert report.recommendation is not None
        assert "STAT1" in report.recommendation or "mechanistic" in report.recommendation.lower()

    def test_no_direction_checks_when_arrays_missing(self):
        result = make_skill_result(jakstat_recovery_score=8)
        report = BiologicalCritic().validate(result)
        all_msg = " ".join(report.checks_passed + report.uncertainty_flags)
        assert "direction" not in all_msg.lower()
        assert "STAT1 FC" not in all_msg
