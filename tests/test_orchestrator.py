"""
tests/test_orchestrator.py — Tests for the workflow orchestrator.

Tests use mock skills and results — no real data or GPU needed.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import AIVCSkillRegistry
from aivc.orchestration.orchestrator import AIVCOrchestrator, CriticSuite
from aivc.orchestration.workflows import WorkflowStep, Workflow, WORKFLOWS
from aivc.critics.statistical import StatisticalCritic
from aivc.critics.methodological import MethodologicalCritic
from aivc.critics.biological import BiologicalCritic
from aivc.context import SessionContext


def make_mock_skill(name, success=True, outputs=None, metadata=None):
    """Create a mock skill that returns a fixed result."""

    class MockSkill(AIVCSkill):
        pass

    MockSkill.name = name
    MockSkill.version = "1.0.0"
    MockSkill.biological_domain = BiologicalDomain.TRANSCRIPTOMICS
    MockSkill.compute_profile = ComputeProfile.CPU_LIGHT

    def mock_execute(self, inputs, context):
        return SkillResult(
            skill_name=name, version="1.0.0",
            success=success,
            outputs=outputs or {},
            metadata=metadata or {"random_seed_used": 42},
            warnings=[], errors=[] if success else ["Mock failure"],
        )

    def mock_estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=1.0, gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0, can_run_on_cpu=True,
        )

    MockSkill.execute = mock_execute
    MockSkill.estimate_cost = mock_estimate_cost
    return MockSkill()


class TestOrchestratorExecution:
    """Test workflow execution order and logic."""

    def test_workflow_executes_all_steps(self):
        reg = AIVCSkillRegistry()

        # Register two mock skills
        skill_a = make_mock_skill("step_a", outputs={"data_a": "value_a"})
        skill_b = make_mock_skill("step_b", outputs={"data_b": "value_b"})
        reg._skills["step_a"] = skill_a
        reg._metadata["step_a"] = {
            "name": "step_a", "domain": "transcriptomics",
            "version": "1.0.0", "requires": [],
            "profile": "cpu_light",
        }
        reg._skills["step_b"] = skill_b
        reg._metadata["step_b"] = {
            "name": "step_b", "domain": "transcriptomics",
            "version": "1.0.0", "requires": [],
            "profile": "cpu_light",
        }

        # Create workflow
        wf = Workflow(
            name="test_wf",
            description="Test workflow",
            steps=[
                WorkflowStep(skill="step_a"),
                WorkflowStep(skill="step_b"),
            ],
        )

        # Monkey-patch WORKFLOWS
        from aivc.orchestration import workflows as wf_mod
        original = dict(wf_mod.WORKFLOWS)
        wf_mod.WORKFLOWS["test_wf"] = wf

        ctx = SessionContext.create_default()
        critics = CriticSuite(
            statistical=StatisticalCritic(),
            methodological=MethodologicalCritic(),
            biological=BiologicalCritic(),
        )

        orch = AIVCOrchestrator(
            registry=reg, memory=ctx.memory,
            critics=critics, budget_usd=50.0,
        )

        result = orch.run_workflow("test_wf", {}, ctx)

        assert result.success is True
        assert ctx.memory.session.steps_completed

        # Restore
        wf_mod.WORKFLOWS.clear()
        wf_mod.WORKFLOWS.update(original)

    def test_orchestrator_fails_on_skill_failure(self):
        reg = AIVCSkillRegistry()
        skill_fail = make_mock_skill("fail_step", success=False)
        reg._skills["fail_step"] = skill_fail
        reg._metadata["fail_step"] = {
            "name": "fail_step", "domain": "transcriptomics",
            "version": "1.0.0", "requires": [],
            "profile": "cpu_light",
        }

        wf = Workflow(
            name="fail_wf", description="Failing workflow",
            steps=[WorkflowStep(skill="fail_step")],
        )

        from aivc.orchestration import workflows as wf_mod
        original = dict(wf_mod.WORKFLOWS)
        wf_mod.WORKFLOWS["fail_wf"] = wf

        ctx = SessionContext.create_default()
        critics = CriticSuite(
            statistical=StatisticalCritic(),
            methodological=MethodologicalCritic(),
            biological=BiologicalCritic(),
        )

        orch = AIVCOrchestrator(
            registry=reg, memory=ctx.memory,
            critics=critics,
        )

        result = orch.run_workflow("fail_wf", {}, ctx)
        assert result.success is False

        wf_mod.WORKFLOWS.clear()
        wf_mod.WORKFLOWS.update(original)

    def test_budget_gate_blocks_expensive_skill(self):
        reg = AIVCSkillRegistry()

        class ExpensiveSkill(AIVCSkill):
            name = "expensive"
            version = "1.0.0"
            compute_profile = ComputeProfile.GPU_INTENSIVE

            def execute(self, inputs, context):
                return SkillResult(
                    skill_name="expensive", version="1.0.0",
                    success=True, outputs={},
                    metadata={"random_seed_used": 42},
                    warnings=[], errors=[],
                )

            def estimate_cost(self, inputs):
                return ComputeCost(
                    estimated_minutes=60.0, gpu_memory_gb=80.0,
                    profile=ComputeProfile.GPU_INTENSIVE,
                    estimated_usd=100.0,  # Over budget!
                    can_run_on_cpu=False,
                )

        skill = ExpensiveSkill()
        reg._skills["expensive"] = skill
        reg._metadata["expensive"] = {
            "name": "expensive", "domain": "transcriptomics",
            "version": "1.0.0", "requires": [],
            "profile": "gpu_intensive",
        }

        wf = Workflow(
            name="expensive_wf", description="Expensive",
            steps=[WorkflowStep(skill="expensive")],
        )

        from aivc.orchestration import workflows as wf_mod
        original = dict(wf_mod.WORKFLOWS)
        wf_mod.WORKFLOWS["expensive_wf"] = wf

        ctx = SessionContext.create_default()
        critics = CriticSuite(
            statistical=StatisticalCritic(),
            methodological=MethodologicalCritic(),
            biological=BiologicalCritic(),
        )

        orch = AIVCOrchestrator(
            registry=reg, memory=ctx.memory,
            critics=critics, budget_usd=50.0,
        )

        result = orch.run_workflow("expensive_wf", {}, ctx)
        assert result.success is False
        assert "budget" in result.error_message.lower()

        wf_mod.WORKFLOWS.clear()
        wf_mod.WORKFLOWS.update(original)

    def test_unknown_workflow_fails(self):
        from aivc.registry import registry as global_reg
        import aivc.skills  # noqa: F401

        ctx = SessionContext.create_default()
        critics = CriticSuite(
            statistical=StatisticalCritic(),
            methodological=MethodologicalCritic(),
            biological=BiologicalCritic(),
        )

        orch = AIVCOrchestrator(
            registry=global_reg, memory=ctx.memory,
            critics=critics,
        )

        result = orch.run_workflow("nonexistent_workflow", {}, ctx)
        assert result.success is False
        assert "Unknown" in result.error_message


class TestWorkflowDefinitions:
    """Test that all workflow definitions are valid."""

    def test_all_workflows_have_steps(self):
        import aivc.skills  # noqa: F401
        for name, wf in WORKFLOWS.items():
            assert len(wf.steps) > 0, f"Workflow '{name}' has no steps"

    def test_rna_baseline_has_9_steps(self):
        wf = WORKFLOWS["rna_baseline_to_demo"]
        assert len(wf.steps) == 9

    def test_active_learning_has_steps(self):
        wf = WORKFLOWS["active_learning_loop"]
        assert len(wf.steps) >= 2
