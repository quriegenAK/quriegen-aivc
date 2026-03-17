"""
tests/test_critics.py — Tests for all three critics.

Tests use mock SkillResult objects — no real data needed.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.interfaces import SkillResult
from aivc.critics.statistical import StatisticalCritic
from aivc.critics.methodological import MethodologicalCritic
from aivc.critics.biological import BiologicalCritic


class TestStatisticalCritic:
    """Tests for StatisticalCritic."""

    def test_passes_on_valid_result(self):
        critic = StatisticalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={
                "pearson_r_mean": 0.873,
                "final_loss": 0.013,
            },
            metadata={"random_seed_used": 42},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is True
        assert report.critic_name == "StatisticalCritic"

    def test_fails_on_nan_in_outputs(self):
        critic = StatisticalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={
                "pearson_r_mean": float("nan"),
            },
            metadata={"random_seed_used": 42},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("NaN" in c or "range" in c for c in report.checks_failed)

    def test_fails_on_nan_array(self):
        critic = StatisticalCritic()
        arr = np.array([1.0, float("nan"), 3.0])
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"predictions": arr},
            metadata={"random_seed_used": 42},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("NaN" in c for c in report.checks_failed)

    def test_fails_on_wrong_seed(self):
        critic = StatisticalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={},
            metadata={"random_seed_used": 123},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("seed" in c.lower() for c in report.checks_failed)

    def test_fails_on_invalid_pearson_r(self):
        critic = StatisticalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"pearson_r_mean": 1.5},
            metadata={"random_seed_used": 42},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("range" in c.lower() for c in report.checks_failed)

    def test_passes_edge_index_shape(self):
        import torch
        critic = StatisticalCritic()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"edge_index": edge_index},
            metadata={"random_seed_used": 42},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is True
        assert any("edge_index" in c for c in report.checks_passed)


class TestMethodologicalCritic:
    """Tests for MethodologicalCritic."""

    def test_passes_on_correct_methodology(self):
        critic = MethodologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={},
            metadata={
                "pearson_r_on_raw": True,
                "split_method": "donor_based",
                "benchmark_dataset": "kang2018_GSE96583",
                "train_donors": ["A", "B", "C"],
                "test_donors": ["D", "E"],
            },
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is True

    def test_fails_on_random_split(self):
        critic = MethodologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={},
            metadata={"split_method": "random"},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("donor" in c.lower() or "split" in c.lower()
                    for c in report.checks_failed)

    def test_fails_on_log_transformed_pearson(self):
        critic = MethodologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={},
            metadata={"pearson_r_on_raw": False},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("transformed" in c.lower() for c in report.checks_failed)

    def test_fails_on_donor_leakage(self):
        critic = MethodologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={},
            metadata={
                "train_donors": ["A", "B", "C"],
                "test_donors": ["B", "D"],  # B is in both!
            },
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("leakage" in c.lower() for c in report.checks_failed)


class TestBiologicalCritic:
    """Tests for BiologicalCritic."""

    def test_passes_on_good_jakstat_recovery(self):
        critic = BiologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"jakstat_recovery_score": 8},
            metadata={},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is True
        assert report.biological_score == 8 / 15.0

    def test_fails_on_ifit1_suppressed(self):
        critic = BiologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={
                "ifit1_predicted_fc": 0.5,  # suppressed!
                "jakstat_recovery_score": 10,
            },
            metadata={},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("IFIT1" in c for c in report.checks_failed)

    def test_passes_on_partial_jakstat(self):
        critic = BiologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"jakstat_recovery_score": 6},
            metadata={},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is True  # 6 >= JAKSTAT_MINIMUM_RECOVERY (5)

    def test_fails_on_low_jakstat(self):
        critic = BiologicalCritic()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={"jakstat_recovery_score": 3},
            metadata={},
            warnings=[], errors=[],
        )
        report = critic.validate(result)
        assert report.passed is False
        assert any("JAK-STAT" in c for c in report.checks_failed)
