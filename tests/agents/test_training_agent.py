"""Tests for agents.training_agent."""
from __future__ import annotations

import os

import pytest

from agents.base_agent import AgentTask
from agents.training_agent import TrainingAgent


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("C", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kw):
        return _FakeMsg(self._text)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def _make_task():
    return AgentTask(
        agent_name="training_agent",
        run_id="r1",
        payload={
            "dataset": "kang2018",
            "pearson_r": 0.5,
            "delta_nonzero_pct": 0.0,
            "ctrl_memorisation_score": 0.99,
            "w_scale_range": "(0.01, 0.3)",
            "neumann_k": 3,
            "failure_note_path": "x",
        },
    )


def test_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = TrainingAgent(
        queue_dir=str(tmp_path / "q"),
        retry_config_path=str(tmp_path / "retry.yaml"),
    )
    result = agent.run(_make_task())
    assert result.success is False
    assert result.error == "no api key"


def test_valid_yaml_extracted(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    yaml_text = "method: bayes\nparameters:\n  w_scale: {min: 0.0, max: 0.05}\n"
    claude_response = f"Diagnosis: W collapse.\n\n```yaml\n{yaml_text}```\n"

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient(claude_response))

    cfg = tmp_path / "retry.yaml"
    agent = TrainingAgent(
        queue_dir=str(tmp_path / "q"),
        retry_config_path=str(cfg),
    )
    result = agent.run(_make_task())
    assert result.success is True
    assert cfg.exists()
    assert "method: bayes" in cfg.read_text()
    assert result.extra.get("no_yaml_from_claude") is not True


def test_no_yaml_block_writes_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("no yaml here"))

    cfg = tmp_path / "retry.yaml"
    agent = TrainingAgent(
        queue_dir=str(tmp_path / "q"),
        retry_config_path=str(cfg),
    )
    result = agent.run(_make_task())
    assert result.success is True
    assert result.extra.get("no_yaml_from_claude") is True
    assert "method: bayes" in cfg.read_text()


def test_rule_based_diagnosis_ctrl_memorisation(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = TrainingAgent(
        queue_dir=str(tmp_path / "q"),
        retry_config_path=str(tmp_path / "retry.yaml"),
    )
    task = AgentTask(
        agent_name="training_agent",
        run_id="r2",
        payload={
            "dataset": "kang2018",
            "pearson_r": 0.5,
            "delta_nonzero_pct": 5.0,
            "ctrl_memorisation_score": 0.97,
            "w_scale_range": "(0.01, 0.05)",
            "neumann_k": 3,
            "failure_note_path": "x",
        },
    )
    result = agent.run(task)
    assert result.extra.get("diagnosis") == "ctrl label memorisation"
