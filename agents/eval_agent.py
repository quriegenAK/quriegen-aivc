"""agents/eval_agent.py — deterministic eval wrapper.

See docs/aivc_spec_agents_v1.md §4.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from agents.base_agent import AgentResult, AgentTask, BaseAgent

logger = logging.getLogger("aivc.agents.eval_agent")


class EvalAgent(BaseAgent):
    agent_name = "eval_agent"

    def run(self, task: AgentTask) -> AgentResult:
        payload = task.payload
        run_id = task.run_id

        checkpoint_path = payload.get("checkpoint_path")
        if not checkpoint_path:
            reg = self.read_latest_checkpoint()
            checkpoint_path = reg.get("checkpoint_path")

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = self.queue_dir / f"eval_agent_{run_id}_{ts}.json"

        if not checkpoint_path:
            return AgentResult(
                agent_name="eval_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary="EvalAgent: no checkpoint_path and no latest_checkpoint.json",
                error="no checkpoint available",
            )

        from eval.eval_runner import run_eval_suite

        try:
            suite = run_eval_suite(
                checkpoint_path,
                run_id=run_id,
                kang_adata=payload.get("kang_adata", "data/kang2018_pbmc_fixed.h5ad"),
                norman_adata=payload.get("norman_adata", "data/norman2019.h5ad"),
                device=payload.get("device"),
            )
        except Exception as e:
            logger.exception("EvalAgent: run_eval_suite raised")
            return AgentResult(
                agent_name="eval_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary="EvalAgent: run_eval_suite raised exception",
                error=str(e),
            )

        # Always write EvalSuite JSON — even on failure.
        try:
            out_path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning(f"EvalAgent: could not write suite JSON: {e}")

        # Success criteria
        error = None
        kang = suite.kang
        norman = suite.norman

        if not kang.regression_guard_passed:
            success = False
            error = f"kang regression guard failed: {kang.failure_reason}"
        elif norman is not None and norman.delta_nonzero_pct == 0:
            success = False
            error = "norman delta collapse"
        else:
            success = bool(suite.overall_passed)

        summary = (
            f"EvalAgent: kang.passed={kang.passed} "
            f"kang.r={kang.pearson_r:.4f} "
            f"overall_passed={suite.overall_passed}"
        )

        return AgentResult(
            agent_name="eval_agent",
            run_id=run_id,
            success=success,
            output_path=str(out_path),
            summary=summary,
            error=error,
        )
