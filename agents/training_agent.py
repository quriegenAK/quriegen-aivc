"""agents/training_agent.py — diagnose failure, emit retry sweep config.

See docs/aivc_spec_agents_v1.md §5.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from agents.base_agent import AgentResult, AgentTask, BaseAgent
from aivc_platform.tracking.experiment_logger import AGENT_PROMPTS

logger = logging.getLogger("aivc.agents.training_agent")

_YAML_FENCE_RE = re.compile(r"```yaml\s*\n(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)

_STUB_YAML = """method: bayes
parameters:
  w_scale:
    min: 0.0
    max: 0.1
  neumann_k:
    values: [2, 3]
"""


def _parse_w_scale_range(raw) -> tuple[float, float]:
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                return float(parsed[0]), float(parsed[1])
        except (ValueError, SyntaxError):
            pass
    return (0.0, 0.0)


def _rule_based_diagnosis(payload: dict) -> str:
    try:
        w_scale_range = _parse_w_scale_range(payload.get("w_scale_range"))
        delta_nz = float(payload.get("delta_nonzero_pct", 0.0) or 0.0)
        neumann_k = int(payload.get("neumann_k", 0) or 0)
        pearson_r = float(payload.get("pearson_r", 0.0) or 0.0)
        ctrl_mem = float(payload.get("ctrl_memorisation_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return "unknown; defer to Claude"

    if ctrl_mem > 0.95:
        return "ctrl label memorisation"
    if w_scale_range[1] >= 0.2 and delta_nz == 0:
        return "W scale collapse"
    if neumann_k <= 2 and pearson_r < 0.5:
        return "K too small"
    return "unknown; defer to Claude"


class TrainingAgent(BaseAgent):
    agent_name = "training_agent"
    claude_model = "claude-sonnet-4-20250514"

    def __init__(
        self,
        retry_config_path: str = "configs/sweep_w_scale_retry.yaml",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.retry_config_path = Path(retry_config_path)

    def _call_claude(self, prompt: str) -> str:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                msg = client.messages.create(
                    model=self.claude_model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception as e:
                last_exc = e
                logger.warning(f"Claude API attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    time.sleep(2)
        assert last_exc is not None
        raise last_exc

    def run(self, task: AgentTask) -> AgentResult:
        payload = task.payload
        run_id = task.run_id
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        diagnosis = _rule_based_diagnosis(payload)

        # Safe prompt format: tolerate missing payload keys.
        template = AGENT_PROMPTS["training_agent"]
        safe_payload = {
            "run_id": run_id,
            "dataset": payload.get("dataset", ""),
            "pearson_r": payload.get("pearson_r", ""),
            "delta_nonzero_pct": payload.get("delta_nonzero_pct", ""),
            "ctrl_memorisation_score": payload.get("ctrl_memorisation_score", ""),
            "neumann_k": payload.get("neumann_k", ""),
            "failure_note_path": payload.get("failure_note_path", ""),
        }
        prompt = template.format(**safe_payload) + f"\n\n## Pre-diagnosis\n{diagnosis}\n"

        if not os.getenv("ANTHROPIC_API_KEY"):
            return AgentResult(
                agent_name="training_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary=f"TrainingAgent: diagnosis={diagnosis}; ANTHROPIC_API_KEY missing",
                error="no api key",
                extra={"diagnosis": diagnosis},
            )

        try:
            response_text = self._call_claude(prompt)
        except Exception as e:
            return AgentResult(
                agent_name="training_agent",
                run_id=run_id,
                success=False,
                output_path=None,
                summary=f"TrainingAgent: Claude API failed — {e}",
                error=str(e),
                extra={"diagnosis": diagnosis},
            )

        ts = int(time.time())
        response_path = self.queue_dir / f"training_agent_{run_id}_{ts}_response.md"
        try:
            response_path.write_text(response_text, encoding="utf-8")
        except OSError as e:
            logger.warning(f"TrainingAgent: could not write response: {e}")

        # Extract YAML
        m = _YAML_FENCE_RE.search(response_text)
        extra = {"diagnosis": diagnosis}
        if m:
            yaml_body = m.group("body")
        else:
            yaml_body = _STUB_YAML
            extra["no_yaml_from_claude"] = True

        self.retry_config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.retry_config_path.write_text(yaml_body, encoding="utf-8")
        except OSError as e:
            logger.warning(f"TrainingAgent: could not write retry config: {e}")

        summary = (
            f"TrainingAgent: diagnosis={diagnosis}; "
            f"retry config → {self.retry_config_path}"
        )

        return AgentResult(
            agent_name="training_agent",
            run_id=run_id,
            success=True,
            output_path=str(self.retry_config_path),
            summary=summary,
            error=None,
            extra=extra,
        )
