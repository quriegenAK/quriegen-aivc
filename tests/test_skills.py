"""
tests/test_skills.py — Unit tests for individual skills.

All tests use mock data — no real h5ad files needed.
All tests run on CPU in under 60 seconds total.
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.context import SessionContext


def make_mock_context():
    """Create a mock SessionContext for testing."""
    return SessionContext.create_default(data_dir="data/", device="cpu")


class TestScRNAPreprocessorValidation:
    """Test ScRNAPreprocessor validation logic with mock results."""

    def test_validate_passes_on_good_result(self):
        from aivc.skills.preprocessing import ScRNAPreprocessor
        skill = ScRNAPreprocessor()

        gene_to_idx = {g: i for i, g in enumerate(
            [f"gene_{j}" for j in range(3000)]
            + skill.MUST_INCLUDE
        )}

        result = SkillResult(
            skill_name="scRNA_preprocessor",
            version="1.0.0",
            success=True,
            outputs={
                "gene_to_idx": gene_to_idx,
                "n_genes_final": len(gene_to_idx),
                "n_pathway_genes_included": 15,
                "genes_forced_in": [],
            },
            metadata={"random_seed_used": 42},
            warnings=[],
            errors=[],
        )

        report = skill.validate(result)
        assert report.passed is True
        assert len(report.checks_failed) == 0

    def test_validate_fails_on_low_gene_count(self):
        from aivc.skills.preprocessing import ScRNAPreprocessor
        skill = ScRNAPreprocessor()

        result = SkillResult(
            skill_name="scRNA_preprocessor",
            version="1.0.0",
            success=True,
            outputs={
                "gene_to_idx": {f"gene_{i}": i for i in range(100)},
                "n_genes_final": 100,
                "n_pathway_genes_included": 5,
                "genes_forced_in": [],
            },
            metadata={},
            warnings=[],
            errors=[],
        )

        report = skill.validate(result)
        assert report.passed is False
        assert any("too low" in c.lower() or "100" in c
                    for c in report.checks_failed)


class TestGraphBuilderValidation:
    """Test GraphBuilder validation logic with mock results."""

    def test_validate_passes_on_good_graph(self):
        import torch
        from aivc.skills.graph_builder import GraphBuilder
        skill = GraphBuilder()

        # Create mock edge_index with no self-loops
        edges = []
        for i in range(3000):
            edges.append([i, (i + 1) % 3000])
            edges.append([(i + 1) % 3000, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        result = SkillResult(
            skill_name="graph_builder",
            version="1.0.0",
            success=True,
            outputs={
                "edge_index": edge_index,
                "n_edges": edge_index.shape[1],
                "jak_stat_edges": 10,
                "jak_stat_genes_in_graph": ["JAK1", "STAT1"],
            },
            metadata={"random_seed_used": 42},
            warnings=[],
            errors=[],
        )

        report = skill.validate(result)
        assert report.passed is True

    def test_validate_fails_on_self_loops(self):
        import torch
        from aivc.skills.graph_builder import GraphBuilder
        skill = GraphBuilder()

        # Create edge_index WITH self-loops
        edge_index = torch.tensor([
            [0, 1, 2, 2],
            [1, 0, 2, 0],  # (2,2) is a self-loop
        ], dtype=torch.long)

        result = SkillResult(
            skill_name="graph_builder",
            version="1.0.0",
            success=True,
            outputs={
                "edge_index": edge_index,
                "n_edges": 4,
                "jak_stat_edges": 0,
            },
            metadata={"random_seed_used": 42},
            warnings=[],
            errors=[],
        )

        report = skill.validate(result)
        assert report.passed is False
        assert any("self-loop" in c.lower() for c in report.checks_failed)


class TestBiologicalPlausibilityScorer:
    """Test BiologicalPlausibilityScorer with known JAK-STAT pairs."""

    def test_jakstat_interactions_score_high(self):
        from aivc.skills.plausibility import BiologicalPlausibilityScorer
        skill = BiologicalPlausibilityScorer()
        ctx = make_mock_context()

        # Use known JAK-STAT interactions
        predicted = [
            {"gene_a": "JAK1", "gene_b": "STAT1", "score": 0.9},
            {"gene_a": "STAT1", "gene_b": "IRF9", "score": 0.85},
            {"gene_a": "IRF9", "gene_b": "IFIT1", "score": 0.8},
        ]

        result = skill.execute(
            {
                "predicted_interactions": predicted,
                "gene_to_idx": {"JAK1": 0, "STAT1": 1, "IRF9": 2, "IFIT1": 3},
                "data_dir": "data/",
            },
            ctx,
        )

        assert result.success is True
        scored = result.outputs["scored_interactions"]
        assert len(scored) == 3

        # All JAK-STAT interactions should have high plausibility
        for s in scored:
            assert s["plausibility_score"] >= 0.7
            assert s["in_jakstat"] is True

    def test_novel_interactions_score_low(self):
        from aivc.skills.plausibility import BiologicalPlausibilityScorer
        skill = BiologicalPlausibilityScorer()
        ctx = make_mock_context()

        predicted = [
            {"gene_a": "GENE_X", "gene_b": "GENE_Y", "score": 0.5},
        ]

        result = skill.execute(
            {
                "predicted_interactions": predicted,
                "gene_to_idx": {"GENE_X": 0, "GENE_Y": 1},
                "data_dir": "data/",
            },
            ctx,
        )

        assert result.success is True
        scored = result.outputs["scored_interactions"]
        assert scored[0]["plausibility_score"] == 0.3  # Novel


class TestTwoAudienceRenderer:
    """Test TwoAudienceRenderer with mock evaluation results."""

    def test_renderer_produces_both_reports(self):
        from aivc.skills.reporting import TwoAudienceRenderer
        skill = TwoAudienceRenderer()
        ctx = make_mock_context()

        mock_eval = SkillResult(
            skill_name="benchmark_evaluator", version="1.0.0",
            success=True,
            outputs={
                "pearson_r_mean": 0.873,
                "pearson_r_std": 0.064,
                "demo_ready": True,
                "benchmark_table": {
                    "AIVC_GeneLink": 0.873,
                    "CPA_published": 0.856,
                    "scGEN_published": 0.820,
                },
                "cell_type_pearson_r": {
                    "CD14+ Monocytes": 0.745,
                    "B cells": 0.920,
                },
                "jakstat_lfc_errors": {
                    "IFIT1": {"predicted_fc": 2.05, "actual_fc": 107.0, "lfc_error": 5.7},
                },
            },
            metadata={}, warnings=[], errors=[],
        )

        mock_attn = SkillResult(
            skill_name="attention_extractor", version="1.0.0",
            success=True,
            outputs={
                "top_20_genes": [("IFIT1", 0.5)],
                "jakstat_attention": {"IFIT1": 0.5, "JAK1": 0.3},
                "jakstat_recovery_score": 7,
            },
            metadata={}, warnings=[], errors=[],
        )

        mock_unc = SkillResult(
            skill_name="uncertainty_estimator", version="1.0.0",
            success=True,
            outputs={
                "high_uncertainty_pathways": ["STAT3", "IRF1"],
                "active_learning_recommendations": [],
            },
            metadata={}, warnings=[], errors=[],
        )

        mock_plaus = SkillResult(
            skill_name="plausibility_scorer", version="1.0.0",
            success=True,
            outputs={
                "mean_plausibility_score": 0.65,
                "quarantine_fraction": 0.1,
                "jakstat_recovery_in_predictions": 5,
            },
            metadata={}, warnings=[], errors=[],
        )

        result = skill.execute(
            {
                "evaluation_result": mock_eval,
                "attention_result": mock_attn,
                "uncertainty_result": mock_unc,
                "plausibility_result": mock_plaus,
            },
            ctx,
        )

        assert result.success is True
        assert "rd_report" in result.outputs
        assert "discovery_report" in result.outputs
        assert result.outputs["demo_ready"] is True
        assert "GO" in result.outputs["recommendation"]

        # Clean up generated files
        for path in result.outputs.get("output_paths", []):
            if os.path.exists(path):
                os.remove(path)


class TestSkillCheckInputs:
    """Test _check_inputs validation."""

    def test_check_inputs_passes_with_all_required(self):
        from aivc.skills.preprocessing import ScRNAPreprocessor
        skill = ScRNAPreprocessor()

        # Should not raise
        skill._check_inputs(
            {"adata_path": "/path/to/file.h5ad"},
            ["adata_path"],
        )

    def test_check_inputs_raises_on_missing(self):
        from aivc.skills.preprocessing import ScRNAPreprocessor
        skill = ScRNAPreprocessor()

        with pytest.raises(ValueError, match="missing required inputs"):
            skill._check_inputs({}, ["adata_path"])
