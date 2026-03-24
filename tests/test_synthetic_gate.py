"""
Tests for synthetic IFN-G data gate in curriculum training.
All tests use mock data. No real datasets. < 5 seconds.

Run: pytest tests/test_synthetic_gate.py -v
"""
import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

from aivc.orchestration.perturbation_curriculum import PerturbationCurriculum
from aivc.data.multi_perturbation_loader import MultiPerturbationLoader

SEED = 42


class TestSyntheticGate:

    def test_advance_stage2_blocked_with_synthetic_data(self):
        """Stage 2 with synthetic IFN-G -> stays at stage 2."""
        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(
            current_stage=2,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=True,
        )
        assert next_stage == 2, "Stage 2 with synthetic data must be blocked"

    def test_advance_stage2_passes_with_real_data(self):
        """Stage 2 with real IFN-G -> advances to stage 3."""
        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(
            current_stage=2,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=False,
        )
        assert next_stage == 3, "Stage 2 with real data should advance"

    def test_advance_stage1_unaffected_by_synthetic_flag(self):
        """Stage 1 has no IFN-G. Synthetic flag is irrelevant."""
        curriculum = PerturbationCurriculum()
        next_stage = curriculum.advance_stage(
            current_stage=1,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=True,  # irrelevant at stage 1
        )
        assert next_stage == 2, "Stage 1 should advance regardless of synthetic flag"

    def test_stage_result_records_synthetic_flag(self):
        """Blocked stage 2 result records synthetic metadata."""
        curriculum = PerturbationCurriculum()
        curriculum.advance_stage(
            current_stage=2,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=True,
        )
        result = curriculum.history[-1]
        assert result.data_quality == "synthetic"
        assert result.synthetic_data_used is True
        assert result.passed is False

    def test_report_shows_synthetic_warning(self):
        """get_report() must contain 'SYNTHETIC DATA' when synthetic used."""
        curriculum = PerturbationCurriculum()
        curriculum.advance_stage(
            current_stage=2,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=True,
        )
        report = curriculum.get_report()
        assert "SYNTHETIC DATA" in report

    def test_stage2_reason_contains_download_instructions(self):
        """Blocked reason must mention OneK1K or GSE90063."""
        curriculum = PerturbationCurriculum()
        curriculum.advance_stage(
            current_stage=2,
            current_pearson_r=0.880,
            jakstat_recovery=8,
            synthetic_ifng_used=True,
        )
        reason = curriculum.history[-1].reason
        assert "OneK1K" in reason or "GSE90063" in reason


class TestIsPbmcIfngSynthetic:

    def test_returns_true_for_synthetic(self):
        """AnnData with uns['synthetic_ifng']=True -> True."""
        rng = np.random.RandomState(SEED)
        adata = ad.AnnData(
            X=rng.random((10, 5)).astype(np.float32),
            obs=pd.DataFrame({"SYNTHETIC_IFNG": True}, index=[f"c{i}" for i in range(10)]),
        )
        adata.uns["synthetic_ifng"] = True

        loader = MultiPerturbationLoader()
        loader._datasets["pbmc_ifng"] = adata
        assert loader.is_pbmc_ifng_synthetic() is True

    def test_returns_false_for_real(self):
        """AnnData without synthetic flag -> False."""
        rng = np.random.RandomState(SEED)
        adata = ad.AnnData(
            X=rng.random((10, 5)).astype(np.float32),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        )

        loader = MultiPerturbationLoader()
        loader._datasets["pbmc_ifng"] = adata
        assert loader.is_pbmc_ifng_synthetic() is False
