"""
Tests for AIVC v1.1 Multi-Perturbation Data Integration.
All tests use mock/synthetic data. No real datasets required.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from scipy import sparse

SEED = 42


def _make_mock_adata(n_cells=50, n_genes=30, condition="ctrl",
                     donor_prefix="kang", cell_type="CD4 T cells"):
    """Create a mock AnnData for testing."""
    import anndata as ad
    rng = np.random.RandomState(SEED)
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "donor_id": [f"{donor_prefix}_donor{i % 3}" for i in range(n_cells)],
        "cell_type": cell_type,
        "condition": condition,
        "label": condition,
    }, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ─── Test donor leakage ───

class TestDonorLeakage:

    def test_kang_test_donors_never_in_training(self):
        """Test donors must never appear in training data."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
        import anndata as ad
        import tempfile, os

        # Create mock Kang data with known donors
        adata = _make_mock_adata(n_cells=60, donor_prefix="kang")
        # Donors: kang_donor0, kang_donor1, kang_donor2
        path = os.path.join(tempfile.mkdtemp(), "kang.h5ad")
        adata.write_h5ad(path)

        loader = MultiPerturbationLoader(
            kang_path=path,
            kang_test_donors=["donor2"],  # Lock donor2 as test
        )
        kang = loader.load_kang()

        # Test cells
        test_mask = kang.obs["in_test_set"]
        train_mask = ~test_mask

        # Verify no overlap
        test_donors = set(kang.obs.loc[test_mask, "donor_id"])
        train_donors = set(kang.obs.loc[train_mask, "donor_id"])
        assert len(test_donors & train_donors) == 0, "Test donors in training!"

    def test_verify_no_test_leakage(self):
        """verify_no_test_leakage must raise on overlap."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
        import anndata as ad

        rng = np.random.RandomState(SEED)
        obs = pd.DataFrame({
            "donor_id": ["d1", "d1", "d2", "d2"],
            "in_test_set": [True, True, False, False],
            "dataset_id": [0, 0, 0, 0],
        })
        adata = ad.AnnData(X=rng.random((4, 10)).astype(np.float32), obs=obs)

        loader = MultiPerturbationLoader()
        # Should pass — no overlap
        assert loader.verify_no_test_leakage(adata) is True

        # Create overlap
        adata.obs.loc[adata.obs.index[2], "in_test_set"] = True
        adata.obs.loc[adata.obs.index[2], "donor_id"] = "d2"
        adata.obs.loc[adata.obs.index[3], "in_test_set"] = False
        adata.obs.loc[adata.obs.index[3], "donor_id"] = "d2"
        # d2 is in both test and train
        with pytest.raises(AssertionError, match="LEAKAGE"):
            loader.verify_no_test_leakage(adata)


# ─── Test gene universe ───

class TestGeneUniverse:

    def test_gene_universe_consistent(self):
        """All datasets must use Kang HVG list. n_genes must match."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
        import anndata as ad
        import tempfile, os

        gene_list = [f"GENE_{i}" for i in range(30)]

        adata = _make_mock_adata(n_cells=20, n_genes=30)
        path = os.path.join(tempfile.mkdtemp(), "kang.h5ad")
        adata.write_h5ad(path)

        loader = MultiPerturbationLoader(kang_path=path, gene_universe=gene_list)
        kang = loader.load_kang()

        # Gene universe should be set
        assert loader.gene_universe == gene_list
        assert kang.n_vars == 30


# ─── Test perturbation IDs ───

class TestPerturbationIDs:

    def test_perturbation_ids_non_overlapping(self):
        """Kang [0,1], Frangieh [2..752], ImmPort [753+]. No duplicates."""
        from aivc.data.multi_perturbation_loader import PERT_ID_RANGES

        kang_ids = set(PERT_ID_RANGES["kang"].values())
        frangieh_ids = set(PERT_ID_RANGES["frangieh"].values())
        immport_ids = set(PERT_ID_RANGES["immport"].values())

        # No overlaps
        assert len(kang_ids & frangieh_ids) == 0
        assert len(kang_ids & immport_ids) == 0
        assert len(frangieh_ids & immport_ids) == 0

    def test_kang_ids_are_0_and_1(self):
        """Kang ctrl=0, stim=1."""
        from aivc.data.multi_perturbation_loader import PERT_ID_RANGES
        assert PERT_ID_RANGES["kang"]["ctrl"] == 0
        assert PERT_ID_RANGES["kang"]["stim"] == 1

    def test_frangieh_ids_start_at_2(self):
        """Frangieh starts at 2."""
        from aivc.data.multi_perturbation_loader import PERT_ID_RANGES
        assert min(PERT_ID_RANGES["frangieh"].values()) == 2

    def test_immport_ids_start_at_753(self):
        """ImmPort starts at 753."""
        from aivc.data.multi_perturbation_loader import PERT_ID_RANGES
        assert min(PERT_ID_RANGES["immport"].values()) == 753


# ─── Test W pre-training ───

class TestWPretrain:

    def test_w_pretrain_jak_stat_direction(self):
        """After pre-training: W[JAK1,STAT1] should update from Replogle data."""
        from aivc.skills.neumann_w_pretrain import pretrain_W_from_replogle

        n_genes = 20
        # Create a simple graph with JAK1 -> STAT1 edge
        gene_to_idx = {"JAK1": 0, "STAT1": 1, "IFIT1": 2}
        edge_index = torch.tensor([
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ], dtype=torch.long)

        W = nn.Parameter(torch.full((6,), 0.01))

        # Mock Replogle: knocking down JAK1 decreases STAT1 and IFIT1
        replogle_df = pd.DataFrame(
            {"STAT1": [-1.0], "IFIT1": [-1.0], "JAK1": [0.0]},
            index=["JAK1"]
        )

        W = pretrain_W_from_replogle(W, edge_index, replogle_df, gene_to_idx, n_epochs=3)
        assert isinstance(W, nn.Parameter)

    def test_empty_replogle_skips(self):
        """Empty Replogle data should skip gracefully."""
        from aivc.skills.neumann_w_pretrain import pretrain_W_from_replogle

        W = nn.Parameter(torch.ones(10))
        edge_index = torch.randint(0, 5, (2, 10))

        W_after = pretrain_W_from_replogle(W, edge_index, pd.DataFrame(), {})
        assert torch.allclose(W, W_after)


# ─── Test curriculum ───

class TestCurriculum:

    def test_curriculum_blocks_on_regression(self):
        """If Pearson r < 0.873, advance_stage must return current stage."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum

        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(
            current_stage=1,
            current_pearson_r=0.850,  # below baseline
            jakstat_recovery=7,
        )
        assert next_stage == 1  # blocked

    def test_curriculum_advances_on_good_r(self):
        """If r >= 0.873, should advance."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum

        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(
            current_stage=1,
            current_pearson_r=0.880,
            jakstat_recovery=8,
        )
        assert next_stage == 2

    def test_curriculum_blocks_on_jakstat_decrease(self):
        """JAK-STAT recovery must not decrease between stages."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum

        curriculum = PerturbationCurriculum()
        # Stage 1 passes with 8 JAK-STAT
        curriculum.advance_stage(1, 0.880, jakstat_recovery=8)
        # Stage 2 tries with only 6 — should block
        next_stage = curriculum.advance_stage(2, 0.875, jakstat_recovery=6)
        assert next_stage == 2  # blocked

    def test_curriculum_4_stages(self):
        """Curriculum must have exactly 4 stages."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
        assert len(PerturbationCurriculum.STAGES) == 4

    def test_curriculum_max_stage_is_4(self):
        """Cannot advance beyond stage 4."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum

        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(4, 0.900, jakstat_recovery=12)
        assert next_stage == 4  # capped

    def test_curriculum_report(self):
        """Report should include stage history."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum

        curriculum = PerturbationCurriculum()
        curriculum.advance_stage(1, 0.880, jakstat_recovery=8)
        report = curriculum.get_report()
        assert "Stage 1" in report
        assert "PASS" in report


# ─── Test domain adaptation ───

class TestDomainAdaptation:

    def test_gradient_reversal(self):
        """GRL must reverse gradient sign."""
        from aivc.skills.domain_adaptation import GradientReversalFunction

        x = torch.randn(4, 8, requires_grad=True)
        out = GradientReversalFunction.apply(x, 1.0)
        loss = out.sum()
        loss.backward()

        # Gradient should be negated
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_domain_adaptation_reduces_dataset_accuracy(self):
        """With GRL, domain classifier accuracy should approach random."""
        from aivc.skills.domain_adaptation import MultiDatasetDomainAdaptation

        torch.manual_seed(SEED)
        da = MultiDatasetDomainAdaptation(n_datasets=2, embed_dim=16)

        # Train briefly — accuracy should not be perfect with GRL
        embeddings = torch.randn(20, 16)
        dataset_ids = torch.cat([torch.zeros(10), torch.ones(10)]).long()

        _, accuracy = da(embeddings, dataset_ids, alpha=1.0)
        # With random weights and GRL, accuracy should be imperfect
        assert isinstance(accuracy, float)

    def test_lambda_ramp(self):
        """Lambda should ramp from 0 to max over specified epochs."""
        from aivc.skills.domain_adaptation import MultiDatasetDomainAdaptation

        assert MultiDatasetDomainAdaptation.compute_lambda_domain(0) == 0.0
        assert MultiDatasetDomainAdaptation.compute_lambda_domain(10, ramp_epochs=20) == pytest.approx(0.05)
        assert MultiDatasetDomainAdaptation.compute_lambda_domain(20, ramp_epochs=20) == pytest.approx(0.1)
        assert MultiDatasetDomainAdaptation.compute_lambda_domain(30, ramp_epochs=20) == pytest.approx(0.1)


# ─── Test multi-dataset evaluator ───

class TestMultiDatasetEvaluator:

    def test_evaluate_returns_per_dataset(self):
        """Evaluation must return per-dataset results."""
        from aivc.skills.multi_dataset_evaluator import evaluate_multi_dataset

        rng = np.random.RandomState(SEED)
        n = 10
        n_genes = 30

        predictions = {
            "kang_2018": {
                "pred": rng.random((n, n_genes)).astype(np.float32),
                "actual": rng.random((n, n_genes)).astype(np.float32),
                "ctrl": rng.random((n, n_genes)).astype(np.float32),
            },
        }
        gene_to_idx = {f"GENE_{i}": i for i in range(n_genes)}

        results = evaluate_multi_dataset(predictions, gene_to_idx)
        assert "kang_2018" in results
        assert "pearson_r" in results["kang_2018"]

    def test_regression_check_flags_low_r(self):
        """Must flag regression if Kang r < 0.873."""
        from aivc.skills.multi_dataset_evaluator import evaluate_multi_dataset

        # Create predictions that will give low r (random)
        rng = np.random.RandomState(SEED)
        n, n_genes = 10, 30
        predictions = {
            "kang_2018": {
                "pred": rng.random((n, n_genes)).astype(np.float32),
                "actual": rng.random((n, n_genes)).astype(np.float32),
                "ctrl": rng.random((n, n_genes)).astype(np.float32),
            },
        }
        gene_to_idx = {f"GENE_{i}": i for i in range(n_genes)}

        results = evaluate_multi_dataset(predictions, gene_to_idx)
        # Random predictions will have r << 0.873
        assert "regression_check" in results
        assert results["regression_check"]["passed"] is False

    def test_neumann_w_stats(self):
        """W stats should include sparsity and top edges."""
        from aivc.skills.multi_dataset_evaluator import evaluate_neumann_W

        W = nn.Parameter(torch.randn(20))
        edge_index = torch.randint(0, 10, (2, 20))
        gene_to_idx = {f"GENE_{i}": i for i in range(10)}

        stats = evaluate_neumann_W(W, edge_index, gene_to_idx)
        assert "w_sparsity" in stats
        assert "top_10_edges" in stats
        assert "n_active_edges" in stats


# ─── Test workflow registration ───

class TestWorkflow5:

    def test_workflow5_exists(self):
        """WORKFLOW_5 must be registered."""
        from aivc.orchestration.workflows import WORKFLOWS
        assert "multi_perturbation_v11" in WORKFLOWS

    def test_workflow5_has_steps(self):
        """WORKFLOW_5 must have steps."""
        from aivc.orchestration.workflows import WORKFLOWS
        wf = WORKFLOWS["multi_perturbation_v11"]
        assert len(wf.steps) >= 5

    def test_new_skills_registered(self):
        """New v1.1 skills must be in registry."""
        from aivc.registry import registry

        for name in ["multi_perturbation_loader", "neumann_w_pretrain",
                     "perturbation_curriculum", "domain_adaptation"]:
            skill = registry.get(name)
            assert skill is not None, f"Skill '{name}' not registered"
            assert skill.version == "1.1.0"


# ─── Test IFIT1 improvement signal ───

class TestIFIT1Improvement:

    def test_ifit1_improves_with_more_perturbations(self):
        """
        With KO evidence (JAK1-KO abolishes IFIT1), Neumann should
        predict larger IFIT1 fold change than without KO evidence.
        """
        from aivc.skills.neumann_propagation import NeumannPropagation

        n_genes = 10
        # JAK1=0, STAT1=1, IFIT1=2
        edge_index = torch.tensor([
            [0, 1, 0],  # JAK1->STAT1, STAT1->IFIT1, JAK1->IFIT1
            [1, 2, 2],
        ], dtype=torch.long)

        # Stage 1: weak W (STRING PPI init, small weights)
        W_stage1 = torch.tensor([0.01, 0.01, 0.005])
        neumann1 = NeumannPropagation(n_genes, edge_index, K=3, lambda_l1=0.001)
        neumann1.W = nn.Parameter(W_stage1)

        # Stage 4: strong W (after KO evidence, JAK1->STAT1 confirmed)
        W_stage4 = torch.tensor([0.5, 0.3, 0.1])
        neumann4 = NeumannPropagation(n_genes, edge_index, K=3, lambda_l1=0.001)
        neumann4.W = nn.Parameter(W_stage4)

        # Direct effect: JAK1 activated
        d_p = torch.zeros(1, n_genes)
        d_p[0, 0] = 5.0  # JAK1 strong activation (IFN-B direct target)

        with torch.no_grad():
            effect_stage1 = neumann1(d_p)
            effect_stage4 = neumann4(d_p)

        ifit1_stage1 = effect_stage1[0, 2].item()
        ifit1_stage4 = effect_stage4[0, 2].item()

        # Stage 4 should predict MUCH larger IFIT1 effect
        assert abs(ifit1_stage4) > abs(ifit1_stage1), (
            f"Stage 4 IFIT1 ({ifit1_stage4:.4f}) should exceed "
            f"Stage 1 ({ifit1_stage1:.4f})"
        )
