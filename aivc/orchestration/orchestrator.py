"""
aivc/orchestration/orchestrator.py — Main workflow engine.

Receives a workflow name, decomposes it into ordered skill sequence,
validates at each gate, manages GPU cost, handles failures with replanning.

Core principles:
- estimate_cost() is called before every gpu_intensive skill
- No result passes to the next step without passing all three critics
- Failures trigger replanning, never silent continuation
- All decisions logged to session memory
"""

import logging
from dataclasses import dataclass

from aivc.interfaces import (
    ComputeProfile, SkillResult, WorkflowResult,
)
from aivc.orchestration.workflows import WORKFLOWS

logger = logging.getLogger(__name__)


@dataclass
class CriticSuite:
    """Container for all three critics."""
    statistical: object
    methodological: object
    biological: object


class AIVCOrchestrator:
    """
    Main workflow engine. Receives a workflow name or scientific query,
    decomposes it into ordered skill sequence, validates at each gate,
    manages GPU cost, handles failures with replanning.
    """

    def __init__(self, registry, memory, critics: CriticSuite,
                 budget_usd: float = 50.0):
        """
        Args:
            registry: AIVCSkillRegistry instance
            memory: MemoryStores instance
            critics: CriticSuite with statistical, methodological, biological
            budget_usd: total GPU spend allowed for this run
        """
        self.registry = registry
        self.memory = memory
        self.critics = critics
        self.budget_usd = budget_usd

    def run_workflow(self, workflow_name: str, inputs: dict,
                     context) -> WorkflowResult:
        """
        Main entry point. Executes a named workflow with validation
        gates at every step.
        """
        # 1 — Load workflow definition
        if workflow_name not in WORKFLOWS:
            return WorkflowResult.failed(
                f"Unknown workflow: {workflow_name}. "
                f"Available: {list(WORKFLOWS.keys())}"
            )

        workflow = WORKFLOWS[workflow_name]
        steps = workflow.get_steps(inputs)

        # 2 — Pre-flight check
        for step in steps:
            can_run, missing = self.registry.check_requirements(
                step.skill, inputs
            )
            if not can_run:
                logger.warning(
                    f"Pre-flight: {step.skill} missing inputs: {missing}. "
                    "Will attempt to resolve from previous step outputs."
                )

        results = {}
        total_cost = 0.0

        # 3 — Execute each step
        for step in steps:
            skill = self.registry.get(step.skill)

            # 3a — Cost gate for GPU-intensive skills
            if skill.compute_profile == ComputeProfile.GPU_INTENSIVE:
                try:
                    cost = skill.estimate_cost(inputs)
                    if total_cost + cost.estimated_usd > self.budget_usd:
                        msg = (
                            f"GPU budget exceeded: "
                            f"${total_cost + cost.estimated_usd:.2f} "
                            f"> ${self.budget_usd:.2f}"
                        )
                        logger.error(msg)
                        return WorkflowResult.failed(msg, step=step.skill)
                    total_cost += cost.estimated_usd
                    logger.info(
                        f"Cost gate passed for {step.skill}: "
                        f"${cost.estimated_usd:.2f} "
                        f"(total: ${total_cost:.2f})"
                    )
                except NotImplementedError:
                    # Placeholder skill — will fail at execute
                    pass

            # 3b — Execute skill
            step_inputs = step.prepare_inputs(inputs, results)

            try:
                result = skill.execute(step_inputs, context)
            except NotImplementedError as e:
                msg = f"Skill {step.skill} not implemented: {e}"
                logger.error(msg)
                if hasattr(self.memory, "session"):
                    self.memory.session.log_error(step.skill, msg)
                return WorkflowResult.failed(msg, step=step.skill)
            except Exception as e:
                logger.error(f"Skill {step.skill} failed: {e}")
                if hasattr(self.memory, "session"):
                    self.memory.session.log_error(step.skill, str(e))

                # Attempt replan
                replan = self._replan(step, e, inputs, context)
                if replan is None:
                    return WorkflowResult.failed(
                        f"Skill {step.skill} failed: {e}",
                        step=step.skill,
                    )
                result = replan

            # Check if execution itself reported failure
            if not result.success:
                logger.error(
                    f"Skill {step.skill} reported failure: {result.errors}"
                )
                if hasattr(self.memory, "session"):
                    self.memory.session.log_error(
                        step.skill, str(result.errors)
                    )
                return WorkflowResult.failed(
                    f"Skill {step.skill} failed: {result.errors}",
                    step=step.skill,
                )

            # 3c — Run all three critics
            stat_report = self.critics.statistical.validate(result)
            meth_report = self.critics.methodological.validate(result)
            bio_report = self.critics.biological.validate(result)

            # 3d — Validation gate: ALL THREE must pass
            all_passed = (
                stat_report.passed
                and meth_report.passed
                and bio_report.passed
            )

            if not all_passed:
                failed_critics = [
                    r.critic_name
                    for r in [stat_report, meth_report, bio_report]
                    if not r.passed
                ]
                logger.warning(
                    f"Validation failed at {step.skill}: {failed_critics}"
                )

                if hasattr(self.memory, "session"):
                    self.memory.session.log_validation_failure(
                        step.skill, failed_critics
                    )

                # Attempt replan for validation failure
                replan_result = self._replan_validation(
                    step, result, failed_critics, context
                )
                if replan_result is None:
                    return WorkflowResult.failed(
                        f"Validation failed at {step.skill}: "
                        f"{failed_critics}",
                        step=step.skill,
                    )
                result = replan_result

            # 3e — Store result and update inputs for next step
            results[step.skill] = result
            inputs.update(result.outputs)

            if hasattr(self.memory, "session"):
                self.memory.session.log_step_complete(step.skill, result)

            logger.info(f"Step complete: {step.skill}")

        # 4 — Log to experimental memory
        if hasattr(self.memory, "experimental"):
            self.memory.experimental.save_run(
                workflow_name, results, context
            )

        # 5 — Build final output
        final_outputs = {}
        for skill_name, result in results.items():
            final_outputs.update(result.outputs)

        # Add result references for renderer
        for skill_name, result in results.items():
            final_outputs[f"{skill_name}_result"] = result

        return WorkflowResult.success(final_outputs, total_cost)

    def _replan(self, failed_step, error, inputs, context):
        """
        Replanning logic for execution failures.
        Try once with reduced inputs.
        """
        error_str = str(error)
        skill_name = failed_step.skill

        # If OT pairing failed: fall back to pseudo-bulk
        if skill_name == "ot_cell_pairer":
            logger.info(
                "OT pairing failed. Falling back to pseudo-bulk."
            )
            inputs["force_pseudo_bulk"] = True
            try:
                skill = self.registry.get(skill_name)
                return skill.execute(inputs, context)
            except Exception:
                return None

        # If GPU OOM: reduce batch size by 50%
        if "out of memory" in error_str.lower():
            batch_size = inputs.get("batch_size", 8)
            new_batch_size = max(1, batch_size // 2)
            logger.info(
                f"OOM detected. Reducing batch size: "
                f"{batch_size} -> {new_batch_size}"
            )
            inputs["batch_size"] = new_batch_size
            try:
                skill = self.registry.get(skill_name)
                return skill.execute(inputs, context)
            except Exception:
                return None

        # If import error: log and return None
        if "import" in error_str.lower():
            logger.error(
                f"Import error in {skill_name}: {error_str}. "
                "Check dependencies."
            )
            return None

        return None

    def _replan_validation(self, step, result, failed_critics, context):
        """
        Replanning logic for validation failures.
        """
        # If only statistical critic failed with NaN: try fresh seed
        if (failed_critics == ["StatisticalCritic"]
                and any("NaN" in c for c in
                        [str(result.outputs.get(k))
                         for k in result.outputs])):
            logger.info(
                "Statistical critic failed (NaN). "
                "Cannot replan — investigation required."
            )
            return None

        # If methodological critic failed on split: cannot auto-fix
        if "MethodologicalCritic" in failed_critics:
            logger.warning(
                "Methodological critic failed. "
                "Check donor split and data transform."
            )
            # Don't abort if other critics passed — methodology
            # warnings are important but may not block
            return None

        # If biological critic failed: investigate
        if "BiologicalCritic" in failed_critics:
            logger.warning(
                "Biological critic failed. "
                "Quarantining low-plausibility outputs."
            )
            return None

        # If all three failed: abort
        if len(failed_critics) == 3:
            return None

        return None
