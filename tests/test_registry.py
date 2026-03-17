"""
tests/test_registry.py — Tests for the skill registration system.

Tests use mock data only — no real AnnData files needed.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.interfaces import (
    AIVCSkill, BiologicalDomain, ComputeProfile, SkillResult,
    ValidationReport, ComputeCost,
)
from aivc.registry import AIVCSkillRegistry


class TestSkillRegistration:
    """Test skill registration with decorator."""

    def test_register_skill(self):
        """Test that a skill can be registered via decorator."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="test_skill",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=["input_a", "input_b"],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class TestSkill(AIVCSkill):
            pass

        assert "test_skill" in reg._skills
        skill = reg.get("test_skill")
        assert skill.name == "test_skill"
        assert skill.version == "1.0.0"

    def test_retrieve_by_name(self):
        """Test skill retrieval by name."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="retrieval_test",
            domain=BiologicalDomain.EVALUATION,
            version="2.0.0",
            requires=["data"],
            compute_profile=ComputeProfile.GPU_REQUIRED,
        )
        class RetrievalSkill(AIVCSkill):
            pass

        skill = reg.get("retrieval_test")
        assert skill.name == "retrieval_test"
        assert skill.version == "2.0.0"

    def test_retrieve_nonexistent_raises(self):
        """Test that retrieving a non-existent skill raises KeyError."""
        reg = AIVCSkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent_skill")

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="duplicate_skill",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class DuplicateSkill1(AIVCSkill):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register(
                name="duplicate_skill",
                domain=BiologicalDomain.TRANSCRIPTOMICS,
                version="1.0.0",
                requires=[],
                compute_profile=ComputeProfile.CPU_LIGHT,
            )
            class DuplicateSkill2(AIVCSkill):
                pass

    def test_list_skills_no_filter(self):
        """Test listing all skills without filtering."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="list_test_1",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class ListSkill1(AIVCSkill):
            pass

        @reg.register(
            name="list_test_2",
            domain=BiologicalDomain.EVALUATION,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.GPU_REQUIRED,
        )
        class ListSkill2(AIVCSkill):
            pass

        skills = reg.list_skills()
        assert len(skills) == 2
        names = {s["name"] for s in skills}
        assert names == {"list_test_1", "list_test_2"}

    def test_list_skills_filter_by_domain(self):
        """Test listing skills filtered by domain."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="domain_filter_1",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class DomainSkill1(AIVCSkill):
            pass

        @reg.register(
            name="domain_filter_2",
            domain=BiologicalDomain.PROTEOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class DomainSkill2(AIVCSkill):
            pass

        trans_skills = reg.list_skills(
            domain=BiologicalDomain.TRANSCRIPTOMICS
        )
        assert len(trans_skills) == 1
        assert trans_skills[0]["name"] == "domain_filter_1"

    def test_list_skills_filter_by_profile(self):
        """Test listing skills filtered by compute profile."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="profile_filter_1",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class ProfileSkill1(AIVCSkill):
            pass

        @reg.register(
            name="profile_filter_2",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=[],
            compute_profile=ComputeProfile.GPU_INTENSIVE,
        )
        class ProfileSkill2(AIVCSkill):
            pass

        gpu_skills = reg.list_skills(
            profile=ComputeProfile.GPU_INTENSIVE
        )
        assert len(gpu_skills) == 1
        assert gpu_skills[0]["name"] == "profile_filter_2"

    def test_check_requirements_all_present(self):
        """Test requirement checking with all data available."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="req_test",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=["adata_path", "gene_to_idx"],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class ReqSkill(AIVCSkill):
            pass

        can_run, missing = reg.check_requirements(
            "req_test",
            {"adata_path": "/data/test.h5ad", "gene_to_idx": {}},
        )
        assert can_run is True
        assert missing == []

    def test_check_requirements_missing(self):
        """Test requirement checking with missing data."""
        reg = AIVCSkillRegistry()

        @reg.register(
            name="req_missing_test",
            domain=BiologicalDomain.TRANSCRIPTOMICS,
            version="1.0.0",
            requires=["adata_path", "gene_to_idx", "edge_index"],
            compute_profile=ComputeProfile.CPU_LIGHT,
        )
        class ReqMissingSkill(AIVCSkill):
            pass

        can_run, missing = reg.check_requirements(
            "req_missing_test",
            {"adata_path": "/data/test.h5ad"},
        )
        assert can_run is False
        assert "gene_to_idx" in missing
        assert "edge_index" in missing


class TestGlobalRegistry:
    """Test the global registry with real registered skills."""

    def test_global_registry_has_skills(self):
        """Test that importing skills populates the global registry."""
        from aivc.registry import registry
        import aivc.skills  # noqa: F401

        skills = registry.list_skills()
        assert len(skills) >= 10  # at least 10 core skills

    def test_global_registry_has_core_skills(self):
        """Test specific core skills are registered."""
        from aivc.registry import registry
        import aivc.skills  # noqa: F401

        core_skills = [
            "scRNA_preprocessor",
            "graph_builder",
            "ot_cell_pairer",
            "gat_trainer",
            "perturbation_predictor",
            "attention_extractor",
            "uncertainty_estimator",
            "biological_plausibility_scorer",
            "benchmark_evaluator",
            "two_audience_renderer",
        ]

        for name in core_skills:
            skill = registry.get(name)
            assert skill is not None, f"Skill '{name}' not registered"
            assert skill.name == name
