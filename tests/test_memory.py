"""
tests/test_memory.py — Tests for memory stores.

Tests use temporary files — cleaned up after each test.
"""

import pytest
import os
import sys
import tempfile
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.interfaces import SkillResult
from aivc.memory.session import SessionMemory
from aivc.memory.experimental import ExperimentalMemory
from aivc.memory.active_learning import ActiveLearningMemory


class TestSessionMemory:
    """Tests for SessionMemory."""

    def test_log_step_complete(self):
        mem = SessionMemory()
        result = SkillResult(
            skill_name="test", version="1.0.0", success=True,
            outputs={}, metadata={}, warnings=[], errors=[],
        )
        mem.log_step_complete("test_skill", result)
        assert len(mem.steps_completed) == 1
        assert mem.steps_completed[0]["skill_name"] == "test_skill"

    def test_log_error(self):
        mem = SessionMemory()
        mem.log_error("test_skill", "Something went wrong")
        assert len(mem.errors) == 1
        assert "Something went wrong" in mem.errors[0]["error"]

    def test_log_validation_failure(self):
        mem = SessionMemory()
        mem.log_validation_failure("test_skill", ["StatisticalCritic"])
        assert len(mem.validation_failures) == 1
        assert "StatisticalCritic" in mem.validation_failures[0]["failed_critics"]

    def test_lock_test_donors(self):
        mem = SessionMemory()
        mem.lock_test_donors({"donor_A", "donor_B"})
        assert mem.test_donors == {"donor_A", "donor_B"}
        assert mem._test_donors_locked is True

    def test_lock_test_donors_twice_raises(self):
        mem = SessionMemory()
        mem.lock_test_donors({"donor_A"})
        with pytest.raises(RuntimeError, match="already locked"):
            mem.lock_test_donors({"donor_B"})

    def test_assert_test_donors_clean_passes(self):
        mem = SessionMemory()
        mem.lock_test_donors({"donor_A", "donor_B"})
        # These candidates don't overlap
        mem.assert_test_donors_clean({"donor_C", "donor_D"})

    def test_assert_test_donors_clean_fails(self):
        mem = SessionMemory()
        mem.lock_test_donors({"donor_A", "donor_B"})
        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            mem.assert_test_donors_clean({"donor_B", "donor_C"})

    def test_get_summary(self):
        mem = SessionMemory()
        mem.log_error("test", "error")
        summary = mem.get_summary()
        assert summary["errors"] == 1
        assert summary["steps_completed"] == 0


class TestExperimentalMemory:
    """Tests for ExperimentalMemory with temporary storage."""

    def test_save_and_retrieve_run(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem = ExperimentalMemory(storage_path=path)

            result = SkillResult(
                skill_name="test", version="1.0.0", success=True,
                outputs={"pearson_r_mean": 0.873},
                metadata={"lfc_beta": 0.01},
                warnings=[], errors=[],
            )
            mem.save_run("test_workflow", {"test_skill": result})

            assert len(mem.get_all_runs()) == 1
            best = mem.get_best_run("pearson_r_mean")
            assert best is not None
            assert best["metrics"]["pearson_r_mean"] == 0.873
        finally:
            os.unlink(path)

    def test_persistence_across_instances(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem1 = ExperimentalMemory(storage_path=path)
            result = SkillResult(
                skill_name="test", version="1.0.0", success=True,
                outputs={"pearson_r_mean": 0.850},
                metadata={},
                warnings=[], errors=[],
            )
            mem1.save_run("wf1", {"skill": result})

            # New instance should load from disk
            mem2 = ExperimentalMemory(storage_path=path)
            assert len(mem2.get_all_runs()) == 1
            assert mem2.get_best_run()["metrics"]["pearson_r_mean"] == 0.850
        finally:
            os.unlink(path)

    def test_regression_detection_triggers(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem = ExperimentalMemory(storage_path=path)

            # Save a good run
            result = SkillResult(
                skill_name="test", version="1.0.0", success=True,
                outputs={"pearson_r_mean": 0.873},
                metadata={},
                warnings=[], errors=[],
            )
            mem.save_run("wf", {"skill": result})

            # Test regression detection
            warning = mem.detect_regression(0.840, threshold=0.02)
            assert warning is not None
            assert "REGRESSION" in warning

            # No regression for close value
            no_warning = mem.detect_regression(0.870, threshold=0.02)
            assert no_warning is None
        finally:
            os.unlink(path)

    def test_no_regression_on_empty_history(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem = ExperimentalMemory(storage_path=path)
            result = mem.detect_regression(0.500)
            assert result is None  # No history to compare against
        finally:
            os.unlink(path)


class TestActiveLearningMemory:
    """Tests for ActiveLearningMemory."""

    def test_update_and_retrieve_queue(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem = ActiveLearningMemory(storage_path=path)

            result = SkillResult(
                skill_name="uncertainty", version="1.0.0", success=True,
                outputs={
                    "active_learning_recommendations": [
                        {"gene": "STAT3", "uncertainty_score": 0.8, "pathway": "JAK-STAT"},
                        {"gene": "IRF1", "uncertainty_score": 0.6, "pathway": "Interferon"},
                        {"gene": "MX2", "uncertainty_score": 0.3, "pathway": "Interferon"},
                    ],
                },
                metadata={},
                warnings=[], errors=[],
            )

            mem.update_queue(result)

            assert mem.get_queue_size() == 3
            top = mem.get_next_experiment_recommendations(n=2)
            assert len(top) == 2
            # Should be sorted by uncertainty score descending
            assert top[0]["gene"] == "STAT3"
            assert top[1]["gene"] == "IRF1"
        finally:
            os.unlink(path)

    def test_mark_measured_removes_from_queue(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem = ActiveLearningMemory(storage_path=path)

            result = SkillResult(
                skill_name="uncertainty", version="1.0.0", success=True,
                outputs={
                    "active_learning_recommendations": [
                        {"gene": "STAT3", "uncertainty_score": 0.8, "pathway": "JAK-STAT"},
                        {"gene": "IRF1", "uncertainty_score": 0.6, "pathway": "Interferon"},
                    ],
                },
                metadata={},
                warnings=[], errors=[],
            )
            mem.update_queue(result)

            assert mem.get_queue_size() == 2

            # Mark STAT3 as measured
            mem.mark_measured(["STAT3"])
            assert mem.get_queue_size() == 1
            assert mem.get_measured_count() == 1

            remaining = mem.get_next_experiment_recommendations(n=5)
            assert remaining[0]["gene"] == "IRF1"
        finally:
            os.unlink(path)

    def test_persistence_across_instances(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem1 = ActiveLearningMemory(storage_path=path)
            result = SkillResult(
                skill_name="unc", version="1.0.0", success=True,
                outputs={
                    "active_learning_recommendations": [
                        {"gene": "STAT3", "uncertainty_score": 0.8, "pathway": "JAK-STAT"},
                    ],
                },
                metadata={},
                warnings=[], errors=[],
            )
            mem1.update_queue(result)

            mem2 = ActiveLearningMemory(storage_path=path)
            assert mem2.get_queue_size() == 1
            assert mem2.get_next_experiment_recommendations()[0]["gene"] == "STAT3"
        finally:
            os.unlink(path)
