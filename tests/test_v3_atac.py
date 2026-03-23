"""
Tests for AIVC v3.0 — ATACSeqEncoder, Fusion, SCM, PeakGeneEdgeBuilder.
All tests use mock data, no real files required.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


# ─── ATACSeqEncoder tests ───

class TestATACSeqEncoder:

    def test_output_shape_64dim(self):
        """ATAC embedding must be 64-dim."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        encoder = ATACSeqEncoder(n_tfs=32, embed_dim=64)
        x = torch.randn(4, 32)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_input_is_chromvar_not_peaks(self):
        """Input dimension should be n_tfs (~32-900), NOT n_peaks (~100k)."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        # Valid: 800 TFs
        encoder = ATACSeqEncoder(n_tfs=800, embed_dim=64)
        assert encoder.n_tfs == 800

    def test_quality_weights_deemphasis(self):
        """Low atac_quality_weight should reduce embedding magnitude."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        encoder = ATACSeqEncoder(n_tfs=32)
        x = torch.randn(4, 32)

        out_full = encoder(x, atac_quality_weights=torch.ones(4))
        out_half = encoder(x, atac_quality_weights=torch.full((4,), 0.5))

        # Half weight should produce roughly half the norm
        ratio = out_half.norm() / out_full.norm()
        assert 0.3 < ratio.item() < 0.7

    def test_quality_weight_zero_gives_zero(self):
        """Weight=0 should zero out the embedding."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        encoder = ATACSeqEncoder(n_tfs=32)
        x = torch.randn(4, 32)
        out = encoder(x, atac_quality_weights=torch.zeros(4))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_no_quality_weight_ok(self):
        """Should work without quality weights."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        encoder = ATACSeqEncoder(n_tfs=32)
        out = encoder(torch.randn(4, 32))
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        """Gradients must flow through encoder."""
        from aivc.skills.atac_encoder import ATACSeqEncoder
        encoder = ATACSeqEncoder(n_tfs=32)
        x = torch.randn(4, 32, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None


# ─── Fusion tests ───

class TestTemporalCrossModalFusion:

    def test_output_384dim(self):
        """Fused output must be 384-dim."""
        from aivc.skills.fusion import TemporalCrossModalFusion
        fusion = TemporalCrossModalFusion()
        rna = torch.randn(2, 128)
        protein = torch.randn(2, 128)
        phospho = torch.randn(2, 64)
        atac = torch.randn(2, 64)
        out = fusion(rna, protein, phospho, atac, pert_id=torch.tensor([1, 1]))
        assert out.shape == (2, 384)

    def test_rna_only_works(self):
        """Fusion should work with RNA only (other modalities None)."""
        from aivc.skills.fusion import TemporalCrossModalFusion
        fusion = TemporalCrossModalFusion()
        rna = torch.randn(2, 128)
        out = fusion(rna)
        assert out.shape == (2, 384)
        assert not torch.isnan(out).any()

    def test_atac_is_t0(self):
        """ATAC must be temporal position 0."""
        from aivc.skills.fusion import TemporalCrossModalFusion
        assert TemporalCrossModalFusion.TEMPORAL_ORDER["atac"] == 0
        assert TemporalCrossModalFusion.TEMPORAL_ORDER["phospho"] == 1
        assert TemporalCrossModalFusion.TEMPORAL_ORDER["rna"] == 2
        assert TemporalCrossModalFusion.TEMPORAL_ORDER["protein"] == 3

    def test_attention_weights_shape(self):
        """Attention weights should be (batch, n_heads, 4, 4)."""
        from aivc.skills.fusion import TemporalCrossModalFusion
        fusion = TemporalCrossModalFusion(n_heads=4)
        rna = torch.randn(2, 128)
        atac = torch.randn(2, 64)
        attn = fusion.get_attention_weights(rna, atac_emb=atac)
        assert attn.shape == (2, 4, 4, 4)


# ─── SCM tests ───

class TestStructuralCausalModel:

    def test_4_nodes(self):
        """SCM must have exactly 4 nodes."""
        from aivc.orchestration.scm import StructuralCausalModel
        scm = StructuralCausalModel()
        assert len(scm.NODES) == 4
        assert "atac" in scm.NODES
        assert "phospho" in scm.NODES
        assert "rna" in scm.NODES
        assert "protein" in scm.NODES

    def test_atac_is_t0(self):
        """ATAC must be at temporal order 0."""
        from aivc.orchestration.scm import StructuralCausalModel
        assert StructuralCausalModel.NODES["atac"].temporal_order == 0

    def test_causal_ordering(self):
        """Order must be ATAC(0) -> Phospho(1) -> RNA(2) -> Protein(3)."""
        from aivc.orchestration.scm import StructuralCausalModel
        nodes = StructuralCausalModel.NODES
        assert nodes["atac"].temporal_order < nodes["phospho"].temporal_order
        assert nodes["phospho"].temporal_order < nodes["rna"].temporal_order
        assert nodes["rna"].temporal_order < nodes["protein"].temporal_order

    def test_3_causal_edges(self):
        """Must have 3 directed causal edges."""
        from aivc.orchestration.scm import StructuralCausalModel
        assert len(StructuralCausalModel.EDGES) == 3

    def test_causal_consistency_loss(self):
        """Causal consistency loss should be computable and differentiable."""
        from aivc.orchestration.scm import StructuralCausalModel
        scm = StructuralCausalModel()

        atac = torch.randn(4, 64, requires_grad=True)
        phospho = torch.randn(4, 64)
        rna = torch.randn(4, 128)
        protein = torch.randn(4, 128)

        loss = scm.causal_consistency_loss(atac, phospho, rna, protein)
        assert loss.item() >= 0
        loss.backward()
        assert atac.grad is not None

    def test_do_intervention_propagates(self):
        """do(X=x) should propagate downstream."""
        from aivc.orchestration.scm import StructuralCausalModel
        scm = StructuralCausalModel()

        current = {
            "atac": torch.randn(2, 64),
            "phospho": torch.randn(2, 64),
            "rna": torch.randn(2, 128),
            "protein": torch.randn(2, 128),
        }

        # Intervene on ATAC
        new_atac = torch.zeros(2, 64)
        result = scm.do_intervention("atac", new_atac, current)

        # ATAC should be the intervention value
        assert torch.allclose(result["atac"], new_atac)
        # Downstream should be modified (different from original)
        assert not torch.allclose(result["phospho"], current["phospho"])

    def test_do_peak_closed(self):
        """do(peak=closed) should zero out some TF scores."""
        from aivc.orchestration.scm import StructuralCausalModel
        scm = StructuralCausalModel()

        scores = torch.randn(2, 32)
        result = scm.do_peak_closed([0, 1, 2], {}, scores)
        assert "modified_chromvar_scores" in result
        modified = result["modified_chromvar_scores"]
        # Some scores should be zeroed
        assert (modified[:, :3] == 0).all()

    def test_get_causal_graph(self):
        """Causal graph should return nodes and edges."""
        from aivc.orchestration.scm import StructuralCausalModel
        scm = StructuralCausalModel()
        graph = scm.get_causal_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 4
        assert len(graph["edges"]) == 3


# ─── PeakGeneEdgeBuilder tests ───

class TestPeakGeneEdgeBuilder:

    def test_builds_directed_edges(self):
        """Peak->gene edges must be directed."""
        from aivc.skills.peak_gene_edge_builder import build_peak_gene_edges

        links = pd.DataFrame({
            "peak_id": ["chr1-1000-2000", "chr1-3000-4000"],
            "gene_name": ["JAK1", "STAT1"],
            "correlation": [0.5, 0.7],
            "pval": [0.01, 0.001],
            "fdr": [0.02, 0.002],
            "distance_to_tss": [5000, 10000],
        })
        gene_to_idx = {"JAK1": 0, "STAT1": 1, "IFIT1": 2}

        edge_index, edge_weight = build_peak_gene_edges(links, gene_to_idx, n_genes=3)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 2
        # Source indices should be >= n_genes (peak nodes are offset)
        assert (edge_index[0] >= 3).all()
        # Target indices should be < n_genes (gene nodes)
        assert (edge_index[1] < 3).all()

    def test_empty_links_returns_empty(self):
        """Empty DataFrame should produce empty edge tensors."""
        from aivc.skills.peak_gene_edge_builder import build_peak_gene_edges

        edge_index, edge_weight = build_peak_gene_edges(
            pd.DataFrame(), {"A": 0}, n_genes=1
        )
        assert edge_index.shape == (2, 0)
        assert edge_weight.shape == (0,)

    def test_combine_graphs_keeps_types_separate(self):
        """PPI and ATAC edges must stay in separate edge types."""
        from aivc.skills.peak_gene_edge_builder import combine_graphs

        ppi_ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ppi_ew = torch.tensor([0.8, 0.8])
        atac_ei = torch.tensor([[3, 4], [0, 1]], dtype=torch.long)
        atac_ew = torch.tensor([0.5, 0.7])

        graph = combine_graphs(ppi_ei, ppi_ew, atac_ei, atac_ew)

        assert "gene_gene" in graph
        assert "peak_gene" in graph
        assert graph["gene_gene"]["directed"] is False
        assert graph["peak_gene"]["directed"] is True


# ─── BiologicalCritic ATAC extension tests ───

class TestBiologicalCriticATAC:

    def test_atac_jakstat_coverage_pass(self):
        """Coverage >= 10 should pass."""
        from aivc.critics.biological import BiologicalCritic
        from aivc.interfaces import SkillResult

        result = SkillResult(
            skill_name="test", version="3.0", success=True,
            outputs={"atac_jakstat_coverage": 12},
            metadata={}, warnings=[], errors=[],
        )
        report = BiologicalCritic().validate(result)
        assert any("ATAC JAK-STAT coverage: 12/15" in c for c in report.checks_passed)

    def test_atac_jakstat_coverage_fail(self):
        """Coverage < 10 should fail."""
        from aivc.critics.biological import BiologicalCritic
        from aivc.interfaces import SkillResult

        result = SkillResult(
            skill_name="test", version="3.0", success=True,
            outputs={"atac_jakstat_coverage": 5},
            metadata={}, warnings=[], errors=[],
        )
        report = BiologicalCritic().validate(result)
        assert any("ATAC JAK-STAT coverage: 5/15" in c for c in report.checks_failed)

    def test_irf_stat_motif_pass(self):
        """Enriched STAT1 and IRF3 should pass."""
        from aivc.critics.biological import BiologicalCritic
        from aivc.interfaces import SkillResult

        result = SkillResult(
            skill_name="test", version="3.0", success=True,
            outputs={
                "irf_stat_motif_direction": {
                    "STAT1": {"enriched": True},
                    "IRF3": {"enriched": True},
                },
            },
            metadata={}, warnings=[], errors=[],
        )
        report = BiologicalCritic().validate(result)
        assert any("STAT1 motif enriched" in c for c in report.checks_passed)
        assert any("IRF3 motif enriched" in c for c in report.checks_passed)

    def test_irf_stat_motif_fail(self):
        """Non-enriched STAT1 should fail."""
        from aivc.critics.biological import BiologicalCritic
        from aivc.interfaces import SkillResult

        result = SkillResult(
            skill_name="test", version="3.0", success=True,
            outputs={
                "irf_stat_motif_direction": {
                    "STAT1": {"enriched": False},
                    "IRF3": {"enriched": True},
                },
            },
            metadata={}, warnings=[], errors=[],
        )
        report = BiologicalCritic().validate(result)
        assert any("STAT1 motif NOT enriched" in c for c in report.checks_failed)

    def test_causal_ordering_correct(self):
        """ATAC->RNA attention > RNA->ATAC should pass."""
        from aivc.critics.biological import BiologicalCritic
        from aivc.interfaces import SkillResult

        result = SkillResult(
            skill_name="test", version="3.0", success=True,
            outputs={
                "atac_causal_attention": {
                    "atac_to_rna": 0.35,
                    "rna_to_atac": 0.15,
                },
            },
            metadata={}, warnings=[], errors=[],
        )
        report = BiologicalCritic().validate(result)
        assert any("causal ordering OK" in c for c in report.checks_passed)


# ─── Workflow tests ───

class TestWorkflow4:

    def test_workflow4_exists(self):
        """WORKFLOW_4 must be registered."""
        from aivc.orchestration.workflows import WORKFLOWS
        assert "atac_multimodal_v3" in WORKFLOWS

    def test_workflow4_has_10_steps(self):
        """WORKFLOW_4 must have 10 steps."""
        from aivc.orchestration.workflows import WORKFLOWS
        wf = WORKFLOWS["atac_multimodal_v3"]
        assert len(wf.steps) == 10

    def test_workflow4_includes_atac_encoder(self):
        """WORKFLOW_4 must include atac_seq_encoder step."""
        from aivc.orchestration.workflows import WORKFLOWS
        wf = WORKFLOWS["atac_multimodal_v3"]
        skill_names = [s.skill for s in wf.steps]
        assert "atac_seq_encoder" in skill_names

    def test_workflow4_includes_peak_gene_builder(self):
        """WORKFLOW_4 must include peak_gene_edge_builder step."""
        from aivc.orchestration.workflows import WORKFLOWS
        wf = WORKFLOWS["atac_multimodal_v3"]
        skill_names = [s.skill for s in wf.steps]
        assert "peak_gene_edge_builder" in skill_names


# ─── Skill registration tests ───

class TestSkillRegistration:

    def test_atac_encoder_registered(self):
        """atac_seq_encoder skill must be registered."""
        from aivc.registry import registry
        skill = registry.get("atac_seq_encoder")
        assert skill is not None
        assert skill.version == "3.0.0"

    def test_peak_gene_builder_registered(self):
        """peak_gene_edge_builder skill must be registered."""
        from aivc.registry import registry
        skill = registry.get("peak_gene_edge_builder")
        assert skill is not None
        assert skill.version == "3.0.0"
