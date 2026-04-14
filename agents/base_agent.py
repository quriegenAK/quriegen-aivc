"""agents/base_agent.py — Base schemas and abstract agent class.

See docs/aivc_spec_agents_v1.md §2.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger("aivc.agents.base")

AgentName = Literal["data_agent", "training_agent", "eval_agent", "research_agent"]


class AgentTask(BaseModel):
    agent_name: AgentName
    run_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    agent_name: AgentName
    run_id: str
    success: bool
    output_path: Optional[str] = None
    summary: str = ""
    error: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class MCPTool(Protocol):
    name: str
    description: str

    def execute(self, input: dict) -> dict: ...


_PEARSON_LINE_RE = re.compile(r"Pearson r:\s*([0-9eE.+-]+)")
_RUN_ID_LINE_RE = re.compile(r"Run ID:\s*(\S+)")


class BaseAgent(ABC):
    agent_name: AgentName

    def __init__(
        self,
        tools: Optional[list[MCPTool]] = None,
        context_path: str = "docs/context_snapshot.md",
        registry_path: str = "artifacts/registry/latest_checkpoint.json",
        queue_dir: str = "artifacts/agent_queue",
    ) -> None:
        self.tools: list[MCPTool] = tools or []
        self.context_path = Path(context_path)
        self.registry_path = Path(registry_path)
        self.queue_dir = Path(queue_dir)

    # ---- memory -----------------------------------------------------
    def read_context(self) -> str:
        if not self.context_path.exists():
            return ""
        try:
            return self.context_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"read_context failed: {e}")
            return ""

    def read_latest_checkpoint(self) -> dict:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"read_latest_checkpoint failed: {e}")
            return {}

    # ---- history ----------------------------------------------------
    def query_run_history(self, min_pearson: float = 0.0) -> list[dict]:
        """Scan queue_dir for prompt files and parse `Pearson r:` lines.

        Fail-soft: skip unparseable files, never raise.
        Returns list sorted by file mtime descending.
        """
        if not self.queue_dir.exists():
            return []
        entries: list[tuple[float, dict]] = []
        for p in self.queue_dir.glob("*.md"):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            m = _PEARSON_LINE_RE.search(text)
            if not m:
                continue
            try:
                pearson = float(m.group(1))
            except ValueError:
                continue
            if pearson < min_pearson:
                continue
            rid_m = _RUN_ID_LINE_RE.search(text)
            run_id = rid_m.group(1) if rid_m else p.stem
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            entries.append((mtime, {
                "run_id": run_id,
                "pearson_r": pearson,
                "file": str(p),
            }))
        entries.sort(key=lambda x: x[0], reverse=True)
        return [e[1] for e in entries]

    # ---- entrypoint -------------------------------------------------
    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult: ...
