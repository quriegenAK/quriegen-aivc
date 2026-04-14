"""Tests for agents.base_agent."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.base_agent import AgentResult, AgentTask, BaseAgent


class _StubAgent(BaseAgent):
    agent_name = "data_agent"  # type: ignore[assignment]

    def run(self, task):
        return AgentResult(agent_name="data_agent", run_id=task.run_id, success=True)


def test_agent_task_missing_required_field():
    with pytest.raises(ValidationError):
        AgentTask(agent_name="data_agent")  # type: ignore[call-arg]


def test_agent_result_success_without_output_path():
    r = AgentResult(agent_name="data_agent", run_id="r1", success=True)
    assert r.success is True
    assert r.output_path is None


def test_read_context_missing_returns_empty(tmp_path):
    a = _StubAgent(
        context_path=str(tmp_path / "does_not_exist.md"),
        registry_path=str(tmp_path / "reg.json"),
        queue_dir=str(tmp_path / "queue"),
    )
    assert a.read_context() == ""


def test_read_latest_checkpoint_missing_returns_empty_dict(tmp_path):
    a = _StubAgent(
        context_path=str(tmp_path / "ctx.md"),
        registry_path=str(tmp_path / "nope.json"),
        queue_dir=str(tmp_path / "queue"),
    )
    assert a.read_latest_checkpoint() == {}


def test_query_run_history_missing_dir_returns_empty_list(tmp_path):
    a = _StubAgent(
        context_path=str(tmp_path / "ctx.md"),
        registry_path=str(tmp_path / "reg.json"),
        queue_dir=str(tmp_path / "no_such_dir"),
    )
    assert a.query_run_history() == []
