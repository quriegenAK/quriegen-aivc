"""
Tests for dataset composition fixes (Frangieh -> PBMC IFN-G).
All tests use mock AnnData. No real files. Under 15 seconds on CPU.

Run: pytest tests/test_dataset_fixes.py -v
"""
import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

SEED = 42


def _make_mock_kang(n_cells=100, n_genes=50):
    """Create mock Kang 2018 AnnData."""
    rng = np.random.RandomState(SEED)
    gene_names = ["STAT1", "JAK1", "IFIT1", "MX1", "ISG15", "OAS1", "OAS2"] + [
        f"GENE_{i}" for i in range(n_genes - 7)
    ]
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "label": ["ctrl"] * (n_cells // 2) + ["stim"] * (n_cells // 2),
            "donor_id": [f"donor_{i % 4}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    return adata


def _make_mock_frangieh(n_cells=60, n_genes=50):
    """Create mock Frangieh AnnData (melanoma)."""
    rng = np.random.RandomState(SEED)
    gene_names = ["STAT1", "JAK1", "IFIT1", "MX1", "ISG15"] + [
        f"GENE_{i}" for i in range(n_genes - 5)
    ]
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "condition": ["control"] * (n_cells // 2) + ["IFN-gamma"] * (n_cells // 2),
            "cell_type": "melanoma",
        },
        index=[f"frangieh_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


# ─── Curriculum Dataset Fixes ───

class TestCurriculumDatasetFixes:

    def test_stage2_uses_pbmc_ifng_not_frangieh(self):
        """Stage 2 datasets must contain 'pbmc_ifng', not 'frangieh_ifng'."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
        stage2 = PerturbationCurriculum.STAGES[2]
        assert "pbmc_ifng" in stage2["datasets"]
        assert "frangieh_ifng" not in stage2["datasets"]

    def test_frangieh_only_in_w_pretrain(self):
        """Frangieh must only appear in W pretrain, not in any stage datasets."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
        curriculum = PerturbationCurriculum()

        # Frangieh should be in W pretrain datasets
        w_datasets = curriculum.get_w_pretrain_datasets()
        assert "frangieh_jakstat_grn" in w_datasets

        # Frangieh should NOT be in any stage's main datasets
        for stage_id, config in PerturbationCurriculum.STAGES.items():
            for ds in config["datasets"]:
                assert "frangieh" not in ds, (
                    f"Stage {stage_id} contains Frangieh dataset '{ds}' "
                    "in main training. Frangieh must be W-pretrain only."
                )

    def test_stage_advance_blocked_on_regression(self):
        """r=0.850 -> advance_stage must return current_stage."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(1, 0.850, jakstat_recovery=7)
        assert next_stage == 1

    def test_stage_advance_passes_at_baseline(self):
        """r=0.873 -> advance_stage must return next_stage."""
        from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(1, 0.873, jakstat_recovery=7)
        assert next_stage == 2

    def test_response_training_datasets_constant(self):
        """RESPONSE_TRAINING_DATASETS must contain kang and pbmc_ifng."""
        from aivc.data.multi_perturbation_loader import (
            RESPONSE_TRAINING_DATASETS,
            W_PRETRAIN_ONLY_DATASETS,
        )
        assert "kang" in RESPONSE_TRAINING_DATASETS
        assert "pbmc_ifng" in RESPONSE_TRAINING_DATASETS
        assert "frangieh" in W_PRETRAIN_ONLY_DATASETS
        assert "replogle" in W_PRETRAIN_ONLY_DATASETS
        # Frangieh must NOT be in response training
        assert "frangieh" not in RESPONSE_TRAINING_DATASETS


# ─── Multi-Perturbation Loader Fixes ───

class TestMultiPerturbationLoaderFixes:

    def test_dataset_ids_has_pbmc_ifng(self):
        """DATASET_IDS['pbmc_ifng'] == 4."""
        from aivc.data.multi_perturbation_loader import DATASET_IDS
        assert DATASET_IDS["pbmc_ifng"] == 4

    def test_w_pretrain_only_flag_set_for_frangieh(self):
        """load_frangieh() must set USE_FOR_W_ONLY=True on all cells."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader

        # Create mock Frangieh h5ad
        adata = _make_mock_frangieh()
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = f.name
            adata.write_h5ad(path)

        try:
            loader = MultiPerturbationLoader(
                frangieh_path=path,
                gene_universe=adata.var_names.tolist(),
            )
            result = loader.load_frangieh()
            assert result is not None
            assert result.obs["USE_FOR_W_ONLY"].all(), (
                "All Frangieh cells must have USE_FOR_W_ONLY=True"
            )
            assert (result.obs["use_for_response_training"] == False).all(), (
                "No Frangieh cells should have use_for_response_training=True"
            )
        finally:
            os.unlink(path)

    def test_validate_pbmc_ifng_rejects_cancer_cells(self):
        """Mock AnnData with insufficient genes -> validation fails."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
        loader = MultiPerturbationLoader()

        # Create tiny adata (too few genes)
        rng = np.random.RandomState(SEED)
        adata = ad.AnnData(
            X=rng.random((10, 20)).astype(np.float32),
            obs=pd.DataFrame({"cell_type": "melanoma"}, index=[f"c{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"G{i}" for i in range(20)]),
        )
        result = loader._validate_pbmc_ifng_dataset(adata)
        assert not result["valid"]

    def test_validate_pbmc_ifng_passes_pbmc_cells(self):
        """Mock AnnData with STAT1, JAK1, ISGs, 2 donors -> passes."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader
        loader = MultiPerturbationLoader()

        rng = np.random.RandomState(SEED)
        gene_names = ["STAT1", "JAK1", "IFIT1", "MX1", "ISG15"] + [
            f"G{i}" for i in range(995)
        ]
        adata = ad.AnnData(
            X=rng.random((50, 1000)).astype(np.float32),
            obs=pd.DataFrame({
                "cell_type": "CD14+ Monocytes",
                "donor_id": [f"d{i % 3}" for i in range(50)],
            }, index=[f"c{i}" for i in range(50)]),
            var=pd.DataFrame(index=gene_names),
        )
        result = loader._validate_pbmc_ifng_dataset(adata)
        assert result["valid"], f"Validation failed: {result['errors']}"

    def test_synthetic_fallback_is_flagged(self):
        """Synthetic IFN-G fallback must have SYNTHETIC_IFNG=True."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader

        kang = _make_mock_kang()
        loader = MultiPerturbationLoader()
        loader._datasets["kang"] = kang
        result = loader._make_synthetic_ifng_fallback(kang)

        assert result is not None
        assert result.obs["SYNTHETIC_IFNG"].all()
        assert result.uns.get("synthetic_ifng") is True

    def test_synthetic_fallback_oas1_reduced(self):
        """Synthetic IFN-G: OAS1 mean must be < OAS1 mean in original stim."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader

        kang = _make_mock_kang(n_cells=100, n_genes=50)
        loader = MultiPerturbationLoader()
        loader._datasets["kang"] = kang

        # Get original stim OAS1
        stim_mask = kang.obs["label"] == "stim"
        X_orig = kang.X.toarray() if sp.issparse(kang.X) else kang.X
        oas1_idx = kang.var_names.tolist().index("OAS1")
        orig_oas1 = X_orig[stim_mask][:, oas1_idx].mean()

        synth = loader._make_synthetic_ifng_fallback(kang)
        X_synth = synth.X.toarray() if sp.issparse(synth.X) else synth.X
        synth_oas1_idx = synth.var_names.tolist().index("OAS1")
        synth_oas1 = X_synth[:, synth_oas1_idx].mean()

        assert synth_oas1 < orig_oas1, (
            f"Synthetic OAS1 ({synth_oas1:.4f}) should be < original ({orig_oas1:.4f})"
        )

    def test_build_corpus_excludes_frangieh_from_response(self):
        """After load_frangieh(), all Frangieh cells must have USE_FOR_W_ONLY=True."""
        from aivc.data.multi_perturbation_loader import MultiPerturbationLoader

        # Create mock files
        kang = _make_mock_kang()
        frangieh = _make_mock_frangieh()

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as fk:
            kang_path = fk.name
            kang.write_h5ad(kang_path)
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as ff:
            frangieh_path = ff.name
            frangieh.write_h5ad(frangieh_path)

        try:
            loader = MultiPerturbationLoader(
                kang_path=kang_path,
                frangieh_path=frangieh_path,
                gene_universe=kang.var_names.tolist(),
            )
            loader.load_kang()
            loader.load_frangieh()

            corpus = loader.build_combined_corpus(include_frangieh=True)

            # All Frangieh cells must be W-only
            frangieh_mask = corpus.obs["dataset_id"] == 1
            if frangieh_mask.any():
                assert corpus.obs.loc[frangieh_mask, "USE_FOR_W_ONLY"].all(), (
                    "All Frangieh cells must have USE_FOR_W_ONLY=True in combined corpus"
                )
        finally:
            os.unlink(kang_path)
            os.unlink(frangieh_path)


# ─── Domain Contamination Guard ───

class TestDomainContaminationGuard:

    def test_guard_blocks_melanoma_cells(self):
        """Corpus with melanoma cells in response training -> clean=False."""
        from aivc.qc.domain_contamination_guard import DomainContaminationGuard

        rng = np.random.RandomState(SEED)
        obs = pd.DataFrame({
            "cell_type": ["melanoma"] * 20 + ["CD4+ T cells"] * 20,
            "USE_FOR_W_ONLY": [False] * 40,
            "use_for_response_training": [True] * 40,
            "dataset_id": [1] * 20 + [0] * 20,
        })
        adata = ad.AnnData(X=rng.random((40, 10)).astype(np.float32), obs=obs)

        guard = DomainContaminationGuard()
        result = guard.check(adata, raise_on_contamination=False)
        assert not result["clean"]
        assert "melanoma" in result["contaminated_types"]
        assert result["n_contaminated"] == 20

    def test_guard_passes_pbmc_cells(self):
        """Corpus with only PBMC cells -> clean=True."""
        from aivc.qc.domain_contamination_guard import DomainContaminationGuard

        rng = np.random.RandomState(SEED)
        obs = pd.DataFrame({
            "cell_type": ["CD14+ Monocytes"] * 20 + ["NK cells"] * 20,
            "USE_FOR_W_ONLY": [False] * 40,
            "use_for_response_training": [True] * 40,
        })
        adata = ad.AnnData(X=rng.random((40, 10)).astype(np.float32), obs=obs)

        guard = DomainContaminationGuard()
        result = guard.check(adata, raise_on_contamination=False)
        assert result["clean"]

    def test_guard_ignores_w_only_cells(self):
        """Frangieh cells with USE_FOR_W_ONLY=True should be ignored."""
        from aivc.qc.domain_contamination_guard import DomainContaminationGuard

        rng = np.random.RandomState(SEED)
        obs = pd.DataFrame({
            "cell_type": ["melanoma"] * 20 + ["CD4+ T cells"] * 20,
            "USE_FOR_W_ONLY": [True] * 20 + [False] * 20,
            "use_for_response_training": [False] * 20 + [True] * 20,
            "dataset_id": [1] * 20 + [0] * 20,
        })
        adata = ad.AnnData(X=rng.random((40, 10)).astype(np.float32), obs=obs)

        guard = DomainContaminationGuard()
        result = guard.check(adata, raise_on_contamination=False)
        assert result["clean"], (
            "Guard should ignore melanoma cells marked USE_FOR_W_ONLY=True. "
            f"Got: {result}"
        )
