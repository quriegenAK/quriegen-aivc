"""
Tests for the ATAC-RNA preprocessing pipeline.

All tests use mock data — no real files required.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# ─────────────────────────────────────────────────────────
# Mock MuData factory
# ─────────────────────────────────────────────────────────

def _make_mock_mudata(
    n_cells=100,
    n_genes=50,
    n_peaks=80,
    cell_types=None,
    conditions=None,
    seed=42,
):
    """Create a mock MuData with RNA and ATAC modalities."""
    import anndata as ad
    import muon as mu

    rng = np.random.RandomState(seed)

    barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    peak_names = [f"chr1-{i*1000}-{i*1000+500}" for i in range(n_peaks)]

    # Insert some JAK-STAT genes
    jakstat = ["JAK1", "STAT1", "STAT2", "IRF1", "IRF9", "MX1", "IFIT1",
               "ISG15", "OAS1", "IFIT3", "IFITM1", "IFITM3", "JAK2",
               "STAT3", "MX2"]
    for i, g in enumerate(jakstat):
        if i < n_genes:
            gene_names[i] = g

    # RNA: log-normal expression
    rna_X = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=rng)
    rna_X.data = np.abs(rng.normal(1.0, 0.5, size=rna_X.data.shape))

    rna_obs = pd.DataFrame(index=barcodes)
    rna_obs["n_genes"] = np.array((rna_X > 0).sum(axis=1)).flatten()
    rna_obs["pct_mt"] = rng.uniform(0, 15, n_cells)
    rna_var = pd.DataFrame(index=gene_names)
    rna_var["mt"] = [g.startswith("MT-") for g in gene_names]

    rna = ad.AnnData(X=rna_X, obs=rna_obs, var=rna_var)

    # ATAC: binary accessibility
    atac_X = sparse.random(n_cells, n_peaks, density=0.15, format="csr", random_state=rng)
    atac_X.data[:] = 1.0

    atac_obs = pd.DataFrame(index=barcodes)
    atac_obs["tss_score"] = rng.uniform(3.0, 10.0, n_cells)
    atac_obs["n_fragments"] = rng.randint(500, 5000, n_cells)
    atac_var = pd.DataFrame(index=peak_names)

    atac = ad.AnnData(X=atac_X, obs=atac_obs, var=atac_var)

    mdata = mu.MuData({"rna": rna, "atac": atac})

    # Add cell types and conditions
    if cell_types is None:
        cell_types = rng.choice(
            ["CD4 T cells", "CD14+ Monocytes", "B cells", "NK cells"],
            size=n_cells,
        )
    mdata.obs["cell_type"] = cell_types
    mdata["rna"].obs["cell_type"] = cell_types
    mdata["rna"].obs["pred_score"] = rng.uniform(0.3, 1.0, n_cells)

    if conditions is None:
        conditions = rng.choice(["ctrl", "stim"], size=n_cells)
    mdata.obs["condition"] = conditions

    mdata.update()
    return mdata


# ─────────────────────────────────────────────────────────
# Pipeline tests
# ─────────────────────────────────────────────────────────

class TestATACRNAPipeline:
    """Tests for ATACRNAPipeline class."""

    def test_step02_filters_low_tss(self):
        """TSS score < 4.0 must be discarded."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=50)
        # Force some cells to have low TSS
        mdata["atac"].obs["tss_score"].iloc[:10] = 2.0
        mdata.update()

        pipeline = ATACRNAPipeline()
        filtered = pipeline.step02_filter_cells(mdata, min_tss_score=4.0)

        # Those 10 cells should be gone
        assert filtered.n_obs < mdata.n_obs
        if "tss_score" in filtered["atac"].obs.columns:
            assert (filtered["atac"].obs["tss_score"] >= 4.0).all()

    def test_step02_filters_high_mt(self):
        """Cells with pct_mt > 20% must be discarded."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=50)
        mdata["rna"].obs["pct_mt"].iloc[:5] = 25.0
        mdata.update()

        pipeline = ATACRNAPipeline()
        filtered = pipeline.step02_filter_cells(mdata, max_mt_pct=20.0)
        assert filtered.n_obs <= mdata.n_obs - 5

    def test_step02_joint_filter(self):
        """Joint filter: cell must pass BOTH RNA and ATAC."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=50)
        # Cell 0: fails RNA (high mt), passes ATAC
        mdata["rna"].obs["pct_mt"].iloc[0] = 30.0
        # Cell 1: passes RNA, fails ATAC (low TSS)
        mdata["atac"].obs["tss_score"].iloc[1] = 1.0
        mdata.update()

        pipeline = ATACRNAPipeline()
        filtered = pipeline.step02_filter_cells(mdata)

        barcodes = set(filtered.obs_names)
        assert "CELL_0000" not in barcodes
        assert "CELL_0001" not in barcodes

    def test_step03_discards_low_pred_score(self):
        """Cells with pred_score < 0.5 must be discarded."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=50)
        # Set some cells to have low prediction scores
        mdata["rna"].obs["pred_score"].iloc[:8] = 0.2
        mdata.update()

        pipeline = ATACRNAPipeline()
        filtered = pipeline.step03_annotate_cell_types(mdata, min_pred_score=0.5)

        assert filtered.n_obs <= mdata.n_obs - 8

    def test_step04_produces_modality_weights(self):
        """WNN integration must produce rna_weight and atac_weight."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=30)
        pipeline = ATACRNAPipeline()
        result = pipeline.step04_integrate_modalities(mdata)

        assert "rna_weight" in result.obs.columns
        assert "atac_weight" in result.obs.columns
        # Weights should sum to ~1.0 per cell
        sums = result.obs["rna_weight"] + result.obs["atac_weight"]
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-5)

    def test_step05_default_window_300kb(self):
        """Default window must be 300kb, NOT 100kb."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline
        import inspect

        pipeline = ATACRNAPipeline()
        sig = inspect.signature(pipeline.step05_link_peaks_to_genes)
        default = sig.parameters["window_kb"].default
        assert default == 300, f"Default window_kb must be 300, got {default}"

    def test_step05_stores_links_in_uns(self):
        """Peak-gene links must be stored in mdata.uns['peak_gene_links']."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=30, n_genes=20, n_peaks=30)
        pipeline = ATACRNAPipeline()
        result = pipeline.step05_link_peaks_to_genes(mdata, window_kb=300)

        assert "peak_gene_links" in result.uns
        links = result.uns["peak_gene_links"]
        assert isinstance(links, pd.DataFrame)
        if len(links) > 0:
            required_cols = ["peak_id", "gene_name", "correlation", "pval", "fdr", "distance_to_tss"]
            for col in required_cols:
                assert col in links.columns, f"Missing column: {col}"

    def test_step06_produces_chromvar_scores(self):
        """Step 06 must produce chromvar_scores in atac.obsm."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=30)
        pipeline = ATACRNAPipeline()
        result = pipeline.step06_compute_tf_motif_enrichment(mdata)

        assert "chromvar_scores" in result["atac"].obsm
        scores = result["atac"].obsm["chromvar_scores"]
        assert scores.shape[0] == result["atac"].n_obs
        assert scores.shape[1] > 0  # at least some TFs
        assert "tf_names" in result["atac"].uns

    def test_step06_chromvar_not_raw_peaks(self):
        """ATACSeqEncoder input must be chromvar_scores, NOT raw peak counts."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=30, n_peaks=100)
        pipeline = ATACRNAPipeline()
        result = pipeline.step06_compute_tf_motif_enrichment(mdata)

        scores = result["atac"].obsm["chromvar_scores"]
        n_peaks = result["atac"].n_vars
        # chromvar_scores should have far fewer columns than peaks
        assert scores.shape[1] < n_peaks, (
            f"chromvar_scores has {scores.shape[1]} columns, "
            f"same as {n_peaks} peaks. Should be TF scores, not raw peaks."
        )

    def test_pipeline_tracks_qc_report(self):
        """Pipeline must track QC stats at every step."""
        from aivc.preprocessing.atac_rna_pipeline import ATACRNAPipeline

        mdata = _make_mock_mudata(n_cells=30)
        pipeline = ATACRNAPipeline()

        pipeline.step01_load_from_mudata(mdata)
        assert "step01" in pipeline.qc_report

        filtered = pipeline.step02_filter_cells(mdata)
        assert "step02" in pipeline.qc_report
        assert "n_before" in pipeline.qc_report["step02"]
        assert "n_after" in pipeline.qc_report["step02"]


# ─────────────────────────────────────────────────────────
# Peak-gene linker tests
# ─────────────────────────────────────────────────────────

class TestPeakGeneLinker:
    """Tests for peak-gene linking."""

    def test_window_warning_below_300(self):
        """Must run without error at any window size including < 300."""
        from aivc.preprocessing.peak_gene_linker import link_peaks_to_genes

        mdata = _make_mock_mudata(n_cells=20, n_genes=10, n_peaks=10)

        # Verify it runs without error at window < 300
        links = link_peaks_to_genes(mdata, window_kb=100)
        assert isinstance(links, pd.DataFrame)

    def test_returns_correct_columns(self):
        """Links DataFrame must have required columns."""
        from aivc.preprocessing.peak_gene_linker import link_peaks_to_genes

        mdata = _make_mock_mudata(n_cells=50, n_genes=20, n_peaks=40)
        links = link_peaks_to_genes(mdata, window_kb=300)

        required_cols = ["peak_id", "gene_name", "correlation", "pval", "fdr", "distance_to_tss"]
        for col in required_cols:
            assert col in links.columns, f"Missing column: {col}"

    def test_empty_input_returns_empty(self):
        """Empty MuData should return empty DataFrame, not crash."""
        import anndata as ad
        import muon as mu

        rna = ad.AnnData(X=sparse.csr_matrix((0, 5)))
        atac = ad.AnnData(X=sparse.csr_matrix((0, 5)))
        mdata = mu.MuData({"rna": rna, "atac": atac})

        from aivc.preprocessing.peak_gene_linker import link_peaks_to_genes
        links = link_peaks_to_genes(mdata, window_kb=300)

        assert isinstance(links, pd.DataFrame)
        assert len(links) == 0


# ─────────────────────────────────────────────────────────
# TF motif scanner tests
# ─────────────────────────────────────────────────────────

class TestTFMotifScanner:
    """Tests for TF motif enrichment scoring."""

    def test_chromvar_output_shape(self):
        """chromVAR scores must be (n_cells, n_TFs)."""
        from aivc.preprocessing.tf_motif_scanner import compute_chromvar_scores
        import anndata as ad

        rng = np.random.RandomState(42)
        n_cells, n_peaks = 50, 100
        X = sparse.random(n_cells, n_peaks, density=0.15, format="csr", random_state=rng)
        X.data[:] = 1.0
        atac = ad.AnnData(
            X=X,
            var=pd.DataFrame(index=[f"chr1-{i*1000}-{i*1000+500}" for i in range(n_peaks)]),
        )

        tf_names = ["STAT1", "IRF3", "NFKB1", "JUN"]
        scores, names = compute_chromvar_scores(atac, tf_names=tf_names)

        assert scores.shape == (n_cells, len(tf_names))
        assert names == tf_names

    def test_chromvar_scores_are_continuous(self):
        """chromVAR scores must be continuous (not binary)."""
        from aivc.preprocessing.tf_motif_scanner import compute_chromvar_scores
        import anndata as ad

        rng = np.random.RandomState(42)
        X = sparse.random(50, 80, density=0.2, format="csr", random_state=rng)
        X.data[:] = 1.0
        atac = ad.AnnData(
            X=X,
            var=pd.DataFrame(index=[f"chr1-{i*1000}-{i*1000+500}" for i in range(80)]),
        )

        scores, _ = compute_chromvar_scores(atac)
        unique_vals = len(np.unique(scores.round(4)))
        assert unique_vals > 2, "Scores should be continuous, not binary"

    def test_ifn_beta_validation(self):
        """Validation function must return per-TF results."""
        from aivc.preprocessing.tf_motif_scanner import validate_ifn_beta_enrichment

        rng = np.random.RandomState(42)
        n_cells = 100
        tf_names = ["STAT1", "IRF3", "NFKB1"]

        # Create scores where STAT1 and IRF3 are higher in stim
        scores = rng.normal(0, 1, (n_cells, 3))
        conditions = np.array(["ctrl"] * 50 + ["stim"] * 50)
        # Boost stim scores for STAT1 and IRF3
        scores[50:, 0] += 2.0  # STAT1
        scores[50:, 1] += 2.0  # IRF3

        results = validate_ifn_beta_enrichment(scores, tf_names, conditions)

        assert "STAT1" in results
        assert results["STAT1"]["enriched"] == True
        assert "IRF3" in results
        assert results["IRF3"]["enriched"] == True


# ─────────────────────────────────────────────────────────
# Cell type annotator tests
# ─────────────────────────────────────────────────────────

class TestCellTypeAnnotator:
    """Tests for cell type annotation."""

    def test_discards_low_scores(self):
        """Cells with pred_score < 0.5 must be discarded."""
        from aivc.preprocessing.cell_type_annotator import transfer_labels
        import anndata as ad

        rng = np.random.RandomState(42)
        n_cells = 50
        X = sparse.random(n_cells, 20, density=0.3, format="csr", random_state=rng)
        adata = ad.AnnData(X=X)
        adata.obs["cell_type"] = rng.choice(["CD4 T cells", "B cells"], n_cells)
        adata.obs["pred_score"] = rng.uniform(0.1, 1.0, n_cells)

        # Count how many should be discarded
        n_low = (adata.obs["pred_score"] < 0.5).sum()

        result, report = transfer_labels(adata, min_pred_score=0.5)
        assert result.n_obs == n_cells - n_low

    def test_discard_report_structure(self):
        """Discard report must have per-cell-type stats."""
        from aivc.preprocessing.cell_type_annotator import transfer_labels
        import anndata as ad

        rng = np.random.RandomState(42)
        X = sparse.random(40, 20, density=0.3, format="csr", random_state=rng)
        adata = ad.AnnData(X=X)
        adata.obs["cell_type"] = ["CD4 T cells"] * 20 + ["B cells"] * 20
        adata.obs["pred_score"] = 0.8

        _, report = transfer_labels(adata, min_pred_score=0.5)

        assert "CD4 T cells" in report
        assert "B cells" in report
        assert "n_total" in report["CD4 T cells"]
        assert "n_discarded" in report["CD4 T cells"]

    def test_no_labels_no_ref_assigns_unknown(self):
        """Without labels or reference, should assign 'unknown'."""
        from aivc.preprocessing.cell_type_annotator import transfer_labels
        import anndata as ad

        X = sparse.random(10, 5, density=0.3, format="csr")
        adata = ad.AnnData(X=X)

        result, report = transfer_labels(adata)
        assert (result.obs["cell_type"] == "unknown").all()


# ─────────────────────────────────────────────────────────
# Validation suite tests
# ─────────────────────────────────────────────────────────

class TestValidation:
    """Tests for validation functions."""

    def test_jakstat_coverage_pass(self):
        """Coverage >= 10/15 should pass."""
        from aivc.preprocessing.validate_atac_pipeline import validate_jakstat_coverage
        from aivc.preprocessing.utils import JAKSTAT_GENES

        links = pd.DataFrame({
            "gene_name": JAKSTAT_GENES[:12],  # 12/15 covered
            "peak_id": [f"peak_{i}" for i in range(12)],
            "correlation": [0.3] * 12,
            "pval": [0.01] * 12,
            "fdr": [0.02] * 12,
            "distance_to_tss": [50000] * 12,
        })

        result = validate_jakstat_coverage(links, min_coverage=10)
        assert result["passed"] is True
        assert result["coverage"] == 12

    def test_jakstat_coverage_fail(self):
        """Coverage < 10/15 should fail."""
        from aivc.preprocessing.validate_atac_pipeline import validate_jakstat_coverage

        links = pd.DataFrame({
            "gene_name": ["JAK1", "STAT1", "IRF1"],
            "peak_id": ["p1", "p2", "p3"],
            "correlation": [0.3] * 3,
            "pval": [0.01] * 3,
            "fdr": [0.02] * 3,
            "distance_to_tss": [50000] * 3,
        })

        result = validate_jakstat_coverage(links, min_coverage=10)
        assert result["passed"] is False
        assert result["coverage"] == 3

    def test_jakstat_coverage_empty(self):
        """Empty links should fail."""
        from aivc.preprocessing.validate_atac_pipeline import validate_jakstat_coverage

        result = validate_jakstat_coverage(pd.DataFrame(), min_coverage=10)
        assert result["passed"] is False
        assert result["coverage"] == 0

    def test_cell_type_retention_pass(self):
        """No cell type losing >20% should pass."""
        from aivc.preprocessing.validate_atac_pipeline import validate_cell_type_retention

        before = pd.DataFrame({"cell_type": ["CD4 T cells"] * 100 + ["B cells"] * 100})
        after = pd.DataFrame({"cell_type": ["CD4 T cells"] * 90 + ["B cells"] * 95})

        result = validate_cell_type_retention(before, after)
        assert result["all_pass"] is True

    def test_cell_type_retention_fail_high_loss(self):
        """Cell type losing >20% should fail."""
        from aivc.preprocessing.validate_atac_pipeline import validate_cell_type_retention

        before = pd.DataFrame({"cell_type": ["CD4 T cells"] * 100 + ["B cells"] * 100})
        after = pd.DataFrame({"cell_type": ["CD4 T cells"] * 70 + ["B cells"] * 95})

        result = validate_cell_type_retention(before, after)
        assert result["all_pass"] is False
        assert result["CD4 T cells"]["passed"] is False

    def test_monocyte_special_threshold(self):
        """CD14+ monocytes have stricter 10% threshold."""
        from aivc.preprocessing.validate_atac_pipeline import validate_cell_type_retention

        before = pd.DataFrame({"cell_type": ["CD14+ Monocytes"] * 100})
        # Lose 15% — passes 20% general but fails 10% monocyte threshold
        after = pd.DataFrame({"cell_type": ["CD14+ Monocytes"] * 85})

        result = validate_cell_type_retention(before, after)
        assert result["CD14+ Monocytes"]["passed"] is False

    def test_motif_enrichment_validation(self):
        """Motif validation must check critical TFs."""
        from aivc.preprocessing.validate_atac_pipeline import validate_motif_enrichment_direction

        rng = np.random.RandomState(42)
        n_cells = 100
        tf_names = ["STAT1", "IRF3", "NFKB1"]
        scores = rng.normal(0, 1, (n_cells, 3))
        conditions = np.array(["ctrl"] * 50 + ["stim"] * 50)
        scores[50:, 0] += 3.0  # STAT1 enriched
        scores[50:, 1] += 3.0  # IRF3 enriched

        result = validate_motif_enrichment_direction(scores, tf_names, conditions)
        assert result["critical_tfs_pass"] is True
