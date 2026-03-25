"""
Tests for SCM causal intervention engine.
All mock data, CPU only, under 20 seconds.
"""
import pytest
import torch
import torch.nn as nn
from aivc.skills.neumann_propagation import NeumannPropagation
from aivc.skills.scm_engine import SCMEngine, CounterfactualResult

SEED = 42
N_GENES = 100

# Gene names: JAK-STAT panel + generic fill
GENE_NAMES = [
    "JAK1", "JAK2", "TYK2", "STAT1", "STAT2", "STAT3",
    "IRF9", "IFIT1", "IFIT2", "MX1", "MX2", "OAS2", "ISG15",
] + [f"GENE_{i}" for i in range(87)]

GENE_TO_IDX = {g: i for i, g in enumerate(GENE_NAMES)}


@pytest.fixture
def setup():
    """Create mock neumann + decoder + engine with known edges."""
    torch.manual_seed(SEED)

    # Build edges ensuring JAK1->STAT1 (0->3) and STAT1->IFIT1 (3->7)
    known = torch.tensor([
        [0, 3], [1, 3], [3, 7], [3, 9], [3, 12], [4, 6], [6, 7],
    ], dtype=torch.long).t()
    random_edges = torch.randint(0, N_GENES, (2, 200))
    edge_index = torch.cat([known, random_edges], dim=1)

    neumann = NeumannPropagation(N_GENES, edge_index, K=3, lambda_l1=0.001)

    # Set known edges to meaningful weights
    with torch.no_grad():
        for i in range(len(neumann.W)):
            s, d = neumann.edge_src[i].item(), neumann.edge_dst[i].item()
            if s == 0 and d == 3:  # JAK1->STAT1
                neumann.W[i] = 0.05
            elif s == 3 and d == 7:  # STAT1->IFIT1
                neumann.W[i] = 0.03

    decoder = nn.Linear(N_GENES, N_GENES)
    engine = SCMEngine(neumann, decoder, gene_names=GENE_NAMES)
    ctrl_expr = torch.rand(N_GENES)

    return neumann, decoder, engine, ctrl_expr


class TestSCMEngineInit:

    def test_engine_initialises_with_gene_names(self, setup):
        _, _, engine, _ = setup
        assert engine._gene_to_idx["JAK1"] == 0

    def test_engine_initialises_without_gene_names(self, setup):
        neumann, decoder, _, _ = setup
        engine = SCMEngine(neumann, decoder, gene_names=None)
        assert engine._gene_to_idx == {}

    def test_validate_readiness_warns_on_untrained_W(self, setup):
        neumann, decoder, _, _ = setup
        with torch.no_grad():
            neumann.W.data.zero_()
        engine = SCMEngine(neumann, decoder, gene_names=GENE_NAMES)
        warnings = engine._validate_readiness()
        assert any("not yet sufficiently trained" in w for w in warnings)


class TestDoGeneKO:

    def test_do_gene_ko_returns_counterfactual_result(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("JAK1", ctrl)
        assert isinstance(result, CounterfactualResult)

    def test_do_gene_ko_output_shapes(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("JAK1", ctrl)
        assert result.baseline_pred.shape == (N_GENES,)
        assert result.counterfactual_pred.shape == (N_GENES,)
        assert result.delta.shape == (N_GENES,)

    def test_do_gene_ko_delta_is_nonzero(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("JAK1", ctrl)
        assert result.delta.abs().max() > 0

    def test_do_gene_ko_does_not_modify_W(self, setup):
        neumann, _, engine, ctrl = setup
        W_before = neumann.W.data.clone()
        engine.do_gene_ko("JAK1", ctrl)
        W_after = neumann.W.data.clone()
        assert torch.allclose(W_before, W_after)

    def test_do_gene_ko_unknown_gene_returns_invalid(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("FAKE_GENE", ctrl)
        assert result.is_valid is False
        assert len(result.warnings) > 0

    def test_do_gene_ko_no_nan_in_output(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("JAK1", ctrl)
        assert not result.delta.isnan().any()
        assert not result.counterfactual_pred.isnan().any()

    def test_do_gene_ko_zeroes_incoming_edges(self, setup):
        neumann, _, engine, _ = setup
        jak1_idx = engine._gene_to_idx["JAK1"]
        mask = engine._build_intervention_mask([jak1_idx])
        incoming = (neumann.edge_dst == jak1_idx)
        assert mask[incoming].sum() == 0


class TestDoPathwayBlock:

    def test_do_pathway_block_multiple_genes(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_pathway_block(["JAK1", "JAK2", "TYK2"], ctrl)
        assert result.intervention_type == "do_pathway_block"
        assert len(result.intervened_genes) == 3

    def test_do_pathway_block_larger_delta_than_single_ko(self, setup):
        _, _, engine, ctrl = setup
        single = engine.do_gene_ko("JAK1", ctrl)
        multi = engine.do_pathway_block(["JAK1", "JAK2", "TYK2"], ctrl)
        assert multi.delta.abs().sum() >= single.delta.abs().sum()

    def test_do_pathway_block_partial_missing_genes(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_pathway_block(["JAK1", "FAKE1", "FAKE2"], ctrl)
        assert result.is_valid  # JAK1 exists
        assert any("not found" in w for w in result.warnings)


class TestDoGeneOE:

    def test_do_gene_oe_scales_direct_effect(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_oe("STAT1", fold_change=5.0, ctrl_expr=ctrl)
        assert result.intervention_type == "do_gene_oe"
        assert result.delta.abs().max() > 0

    def test_do_gene_oe_invalid_fold_change(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_oe("STAT1", fold_change=0.0, ctrl_expr=ctrl)
        assert result.is_valid is False


class TestDoPeakClosed:

    def test_do_peak_closed_returns_stub(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_peak_closed("chr2:48700000", ctrl)
        assert result.is_valid is False
        assert any("STUB" in w for w in result.warnings)
        assert any("tf_motif_scanner" in w for w in result.warnings)


class TestCounterfactualReport:

    def test_counterfactual_report_runs_all_panel_genes(self, setup):
        _, _, engine, ctrl = setup
        report = engine.counterfactual_report(ctrl)
        assert len(report["interventions"]) == 6

    def test_counterfactual_report_has_ifit1_delta(self, setup):
        _, _, engine, ctrl = setup
        report = engine.counterfactual_report(ctrl)
        for interv in report["interventions"]:
            assert "ifit1_delta" in interv

    def test_counterfactual_result_to_dict(self, setup):
        _, _, engine, ctrl = setup
        result = engine.do_gene_ko("JAK1", ctrl)
        d = result.to_dict()
        required = [
            "intervention_type", "intervened_genes",
            "top_affected_genes", "is_valid", "warnings",
            "delta_nonzero", "max_delta_gene",
        ]
        for k in required:
            assert k in d, f"Missing key: {k}"

    def test_w_never_modified_across_multiple_interventions(self, setup):
        neumann, _, engine, ctrl = setup
        W_original = neumann.W.data.clone()
        engine.do_gene_ko("JAK1", ctrl)
        engine.do_gene_ko("STAT1", ctrl)
        engine.do_pathway_block(["JAK1", "JAK2"], ctrl)
        engine.do_gene_oe("IFIT1", 3.0, ctrl)
        W_after = neumann.W.data.clone()
        assert torch.allclose(W_original, W_after), \
            "W was permanently modified by an intervention!"
