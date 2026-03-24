"""
Tests for ambient RNA decontamination.
All tests use mock AnnData. No real files. All complete in < 15 seconds.

Run: pytest tests/test_ambient_rna.py -v
"""
import pytest
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from aivc.qc.ambient_rna import AmbientRNAEstimator
from aivc.qc.benchmark_integrity import BenchmarkIntegrityChecker

JAKSTAT = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
    "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]


def make_mock_adata(
    n_ctrl=100,
    n_stim=100,
    n_genes=50,
    ifit1_ambient_spike: float = 0.0,
    seed=42,
):
    """
    Create mock AnnData with ctrl and stim cells.
    ifit1_ambient_spike: if > 0, add this fraction of IFIT1 expression
                         to ctrl cells as ambient contamination.
    """
    rng = np.random.RandomState(seed)
    n_cells = n_ctrl + n_stim

    gene_names = JAKSTAT[:15] + [f"GENE_{i}" for i in range(n_genes - 15)]

    X = rng.negative_binomial(2, 0.8, size=(n_cells, n_genes)).astype(np.float32)

    ifit1_idx = gene_names.index("IFIT1")
    stat1_idx = gene_names.index("STAT1")
    mx1_idx = gene_names.index("MX1")

    # Stim cells: high JAK-STAT expression
    X[n_ctrl:, ifit1_idx] = rng.negative_binomial(50, 0.3, size=n_stim)
    X[n_ctrl:, stat1_idx] = rng.negative_binomial(20, 0.4, size=n_stim)
    X[n_ctrl:, mx1_idx] = rng.negative_binomial(30, 0.35, size=n_stim)

    # Ambient contamination in ctrl cells
    if ifit1_ambient_spike > 0:
        ambient_amount = X[n_ctrl:, ifit1_idx].mean() * ifit1_ambient_spike
        X[:n_ctrl, ifit1_idx] += rng.poisson(ambient_amount, size=n_ctrl)

    obs = pd.DataFrame(
        {
            "label": ["ctrl"] * n_ctrl + ["stim"] * n_stim,
            "donor_id": [f"donor_{i % 4}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


class TestAmbientProfile:

    def test_ambient_profile_sums_to_one(self):
        """Ambient profile must be a probability distribution summing to 1."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata, raw_matrix_path=None)
        profile = est.estimate_ambient_profile()
        np.testing.assert_allclose(
            profile.sum(), 1.0, atol=1e-5,
            err_msg="Ambient profile must sum to 1.0",
        )

    def test_ambient_profile_non_negative(self):
        """All ambient profile values must be >= 0."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        profile = est.estimate_ambient_profile()
        assert (profile >= 0).all(), "Ambient profile must be non-negative"

    def test_ambient_profile_length_matches_genes(self):
        """Profile length must equal number of genes."""
        adata = make_mock_adata(n_genes=50)
        est = AmbientRNAEstimator(adata)
        profile = est.estimate_ambient_profile()
        assert len(profile) == 50

    def test_fallback_mode_activated_without_raw(self):
        """When raw_matrix_path is None, mode must be 'fallback'."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata, raw_matrix_path=None)
        est.estimate_ambient_profile()
        assert est.mode == "fallback"

    def test_ifit1_enriched_in_ambient_when_spiked(self):
        """When ctrl cells have ambient spike, IFIT1 should be above average."""
        adata = make_mock_adata(ifit1_ambient_spike=0.3)
        est = AmbientRNAEstimator(adata)
        profile = est.estimate_ambient_profile()
        ifit1_idx = adata.var_names.tolist().index("IFIT1")
        ifit1_ambient_frac = profile[ifit1_idx]
        mean_frac = 1.0 / len(profile)
        assert ifit1_ambient_frac > mean_frac, (
            f"IFIT1 ambient fraction {ifit1_ambient_frac:.4f} not above "
            f"mean {mean_frac:.4f}."
        )


class TestContaminationEstimation:

    def test_contamination_rates_in_valid_range(self):
        """All contamination rates must be in [0, cap]."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata, contamination_cap=0.20)
        est.estimate_ambient_profile()
        rho = est.estimate_contamination_per_cell()
        assert (rho >= 0).all(), "Contamination rates must be >= 0"
        assert (rho <= 0.20).all(), "Contamination rates must be <= cap (0.20)"

    def test_contamination_higher_for_spiked_ctrl(self):
        """Ctrl cells with ambient spike should show higher contamination."""
        adata_clean = make_mock_adata(ifit1_ambient_spike=0.0)
        adata_spiked = make_mock_adata(ifit1_ambient_spike=0.5)

        est_clean = AmbientRNAEstimator(adata_clean)
        est_clean.estimate_ambient_profile()
        rho_clean = est_clean.estimate_contamination_per_cell()

        est_spiked = AmbientRNAEstimator(adata_spiked)
        est_spiked.estimate_ambient_profile()
        rho_spiked = est_spiked.estimate_contamination_per_cell()

        n_ctrl = 100
        mean_clean = rho_clean[:n_ctrl].mean()
        mean_spiked = rho_spiked[:n_ctrl].mean()
        assert mean_spiked >= mean_clean, (
            f"Spiked ctrl contamination ({mean_spiked:.4f}) should be >= "
            f"clean ctrl ({mean_clean:.4f})"
        )


class TestDecontamination:

    def test_decontaminated_counts_non_negative(self):
        """Decontaminated counts must be >= 0 (clipped)."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        X_clean = adata_clean.X
        if sp.issparse(X_clean):
            X_clean = X_clean.toarray()
        assert (X_clean >= 0).all(), "Decontaminated counts must be non-negative"

    def test_decontaminated_counts_leq_original(self):
        """Decontaminated counts must be <= original counts."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        X_orig = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        X_clean = adata_clean.X.toarray() if sp.issparse(adata_clean.X) else adata_clean.X
        assert (X_clean <= X_orig + 1e-6).all(), (
            "Decontaminated counts must be <= original counts"
        )

    def test_ifit1_reduced_in_ctrl_after_decontamination(self):
        """IFIT1 expression in ctrl should be lower after decontamination when spiked."""
        adata = make_mock_adata(ifit1_ambient_spike=0.4, seed=42)
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()

        ifit1_idx = adata.var_names.tolist().index("IFIT1")
        n_ctrl = 100
        X_orig = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        X_clean = adata_clean.X.toarray() if sp.issparse(adata_clean.X) else adata_clean.X

        orig_ctrl_ifit1 = X_orig[:n_ctrl, ifit1_idx].mean()
        clean_ctrl_ifit1 = X_clean[:n_ctrl, ifit1_idx].mean()
        assert clean_ctrl_ifit1 <= orig_ctrl_ifit1 + 1e-6, (
            f"IFIT1 ctrl mean should decrease. "
            f"orig={orig_ctrl_ifit1:.4f} clean={clean_ctrl_ifit1:.4f}"
        )

    def test_decontamination_preserves_cell_count(self):
        """Cell count must not change after decontamination."""
        adata = make_mock_adata(n_ctrl=80, n_stim=80)
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        assert adata_clean.n_obs == adata.n_obs
        assert adata_clean.n_vars == adata.n_vars


class TestGeneContaminationReport:

    def test_report_contains_all_jakstat_genes(self):
        """Report must include rows for all present JAK-STAT genes."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        report = est.gene_contamination_report(adata_clean)
        present = set(report["gene_name"])
        for g in JAKSTAT:
            if g in adata.var_names:
                assert g in present, f"JAK-STAT gene {g} missing from report"

    def test_report_ambient_fraction_range(self):
        """Ambient fraction must be in [0, 1]."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        report = est.gene_contamination_report(adata_clean)
        assert (report["ambient_fraction"] >= 0).all()
        assert (report["ambient_fraction"] <= 1).all()

    def test_report_interpretation_categories(self):
        """Interpretation must be one of CLEAN/MODERATE/CONTAMINATED."""
        adata = make_mock_adata()
        est = AmbientRNAEstimator(adata)
        est.estimate_ambient_profile()
        est.estimate_contamination_per_cell()
        adata_clean = est.decontaminate()
        report = est.gene_contamination_report(adata_clean)
        valid = {"CLEAN", "MODERATE", "CONTAMINATED"}
        assert set(report["interpretation"]).issubset(valid)


class TestBenchmarkIntegrity:

    def test_clean_verdict_when_low_contamination(self):
        """r_delta < 0.005 and ifit1 < 5% -> verdict CLEAN."""
        checker = BenchmarkIntegrityChecker()
        report = pd.DataFrame({
            "gene_name":        ["IFIT1", "MX1", "STAT1"],
            "ctrl_raw_mean":    [0.10, 0.05, 0.30],
            "ctrl_clean_mean":  [0.097, 0.048, 0.292],
            "ambient_mean":     [0.003, 0.002, 0.008],
            "ambient_fraction": [0.03, 0.04, 0.027],
            "is_jakstat":       [True, True, True],
            "interpretation":   ["CLEAN", "CLEAN", "CLEAN"],
        })
        result = checker.run_integrity_check(report, r_raw=0.873, r_clean=0.871)
        assert result["verdict"] == "CLEAN"

    def test_inflated_verdict_when_high_contamination(self):
        """ifit1 > 10% -> verdict INFLATED."""
        checker = BenchmarkIntegrityChecker()
        report = pd.DataFrame({
            "gene_name":        ["IFIT1", "MX1"],
            "ctrl_raw_mean":    [0.50, 0.30],
            "ctrl_clean_mean":  [0.40, 0.25],
            "ambient_mean":     [0.10, 0.05],
            "ambient_fraction": [0.20, 0.167],
            "is_jakstat":       [True, True],
            "interpretation":   ["CONTAMINATED", "CONTAMINATED"],
        })
        result = checker.run_integrity_check(report, r_raw=0.873, r_clean=0.851)
        assert result["verdict"] == "INFLATED"

    def test_r_delta_computed_correctly(self):
        """r_delta must equal r_raw - r_clean."""
        checker = BenchmarkIntegrityChecker()
        report = pd.DataFrame({
            "gene_name": ["IFIT1"],
            "ctrl_raw_mean": [0.1], "ctrl_clean_mean": [0.09],
            "ambient_mean": [0.01], "ambient_fraction": [0.10],
            "is_jakstat": [True], "interpretation": ["MODERATE"],
        })
        result = checker.run_integrity_check(report, r_raw=0.873, r_clean=0.862)
        assert abs(result["r_delta"] - (0.873 - 0.862)) < 1e-6
