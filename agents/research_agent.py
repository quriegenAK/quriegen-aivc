"""agents/research_agent.py — success-path follow-up + next-experiment note.

See docs/aivc_spec_agents_v1.md §6.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from agents.base_agent import AgentResult, AgentTask, BaseAgent
from aivc_platform.tracking.experiment_logger import AGENT_PROMPTS

logger = logging.getLogger("aivc.agents.research_agent")

CLAUDE_OUTPUTS_DIR = Path("Claude Outputs/aivc_genelink")


class ResearchAgent(BaseAgent):
    agent_name = "research_agent"
    claude_model = "claude-sonnet-4-20250514"

    def _fetch_wandb(self, wandb_url: str) -> dict:
        try:
            import wandb
            api = wandb.Api()
            run = api.run(wandb_url)
            summary = dict(run.summary) if run.summary else {}
            return {"wandb_summary": summary}
        except Exception as e:
            logger.warning(f"ResearchAgent: W&B fetch failed: {e}")
            return {}

    def _call_claude(self, prompt: str) -> str:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _maybe_write_obsidian(
        self,
        run_id: str,
        meta_obj,
        suite_obj,
    ) -> Optional[str]:
        """Backfill Obsidian experiment note if absent and meta/suite available."""
        try:
            from aivc_platform.memory.vault import ObsidianConfig
        except Exception as e:
            logger.warning(f"ResearchAgent: vault import failed: {e}")
            return None

        try:
            config = ObsidianConfig()
            vault = config.resolved_vault()
            note_path = vault / "experiments" / f"{run_id}.md"
        except Exception as e:
            logger.warning(f"ResearchAgent: vault resolution failed: {e}")
            return None

        if note_path.exists():
            logger.info(f"ResearchAgent: Obsidian note exists, skipping: {note_path}")
            return str(note_path)

        if meta_obj is None or suite_obj is None:
            logger.warning(
                f"ResearchAgent: Obsidian note missing for {run_id} "
                "but meta/suite not in payload — backfill not possible"
            )
            return None

        try:
            from aivc_platform.memory.context_updater import update_context
            from aivc_platform.memory.obsidian_writer import write_experiment_note
            written = write_experiment_note(meta_obj, suite_obj, ObsidianConfig())
            try:
                update_context(meta_obj, suite_obj)
            except Exception as e:
                logger.warning(f"ResearchAgent: update_context failed: {e}")
            logger.info(f"ResearchAgent: backfilled Obsidian note for {run_id}")
            return str(written)
        except Exception as e:
            logger.warning(f"ResearchAgent: Obsidian backfill failed: {e}")
            return None

    def run(self, task: AgentTask) -> AgentResult:
        payload = task.payload
        run_id = task.run_id
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Deserialise meta/suite if dispatcher sent them.
        from aivc_platform.tracking.schemas import RunMetadata
        from eval.eval_runner import EvalSuite

        meta_obj = None
        suite_obj = None
        if payload.get("meta_json"):
            try:
                meta_obj = RunMetadata.model_validate_json(payload["meta_json"])
            except Exception as e:
                logger.warning(f"ResearchAgent: could not deserialise meta: {e}")
        if payload.get("suite_json"):
            try:
                suite_obj = EvalSuite.model_validate_json(payload["suite_json"])
            except Exception as e:
                logger.warning(f"ResearchAgent: could not deserialise suite: {e}")

        extra: dict = {}

        wandb_url = payload.get("wandb_url", "")
        if wandb_url:
            extra.update(self._fetch_wandb(wandb_url))

        obsidian_path = self._maybe_write_obsidian(run_id, meta_obj, suite_obj)
        if obsidian_path:
            extra["obsidian_note"] = obsidian_path

        template = AGENT_PROMPTS["research_agent"]
        safe_payload = {
            "run_id": run_id,
            "dataset": payload.get("dataset", ""),
            "pearson_r": payload.get("pearson_r", ""),
            "delta_nonzero_pct": payload.get("delta_nonzero_pct", ""),
            "w_scale_range": payload.get("w_scale_range", ""),
            "neumann_k": payload.get("neumann_k", ""),
            "checkpoint_path": payload.get("checkpoint_path", ""),
            "wandb_url": wandb_url,
        }
        prompt = template.format(**safe_payload)

        if not os.getenv("ANTHROPIC_API_KEY"):
            return AgentResult(
                agent_name="research_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary="ResearchAgent: ANTHROPIC_API_KEY missing",
                error="no api key",
                extra=extra,
            )

        try:
            response_text = self._call_claude(prompt)
        except Exception as e:
            return AgentResult(
                agent_name="research_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary=f"ResearchAgent: Claude API failed — {e}",
                error=str(e),
                extra=extra,
            )

        ts = int(time.time())
        raw_path = self.queue_dir / f"research_agent_{run_id}_{ts}_response.md"
        try:
            raw_path.write_text(response_text, encoding="utf-8")
        except OSError as e:
            logger.warning(f"ResearchAgent: could not write raw response: {e}")

        CLAUDE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        next_exp_path = CLAUDE_OUTPUTS_DIR / f"{run_id}_next_experiment.md"
        try:
            next_exp_path.write_text(response_text, encoding="utf-8")
        except OSError as e:
            logger.warning(f"ResearchAgent: could not write next-experiment: {e}")

        return AgentResult(
            agent_name="research_agent",
            run_id=run_id,
            success=True,
            output_path=str(next_exp_path),
            summary=f"ResearchAgent: next-experiment → {next_exp_path}",
            error=None,
            extra=extra,
        )
