"""Tests for agents.research_agent."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.base_agent import AgentTask
from agents.research_agent import ResearchAgent


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
        agent_name="research_agent",
        run_id="r_success_1",
        payload={
            "dataset": "kang2018",
            "pearson_r": 0.88,
            "delta_nonzero_pct": 50.0,
            "w_scale_range": "(0.01, 0.1)",
            "neumann_k": 3,
            "checkpoint_path": "ckpt.pt",
            "wandb_url": "",
        },
    )


def _setup_agent(tmp_path, monkeypatch):
    # Redirect Claude Outputs to tmp for test isolation.
    from agents import research_agent as ra
    monkeypatch.setattr(ra, "CLAUDE_OUTPUTS_DIR", tmp_path / "claude_out")
    return ResearchAgent(queue_dir=str(tmp_path / "q"))


def test_obsidian_note_exists_skips_write(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("next experiment text"))

    # Fake vault with an existing note
    vault = tmp_path / "vault"
    (vault / "experiments").mkdir(parents=True)
    (vault / "experiments" / "r_success_1.md").write_text("existing")

    from aivc_platform.memory import vault as vault_mod
    monkeypatch.setattr(
        vault_mod.ObsidianConfig,
        "resolved_vault",
        lambda self: vault,
    )
    called = {"write": 0}
    import aivc_platform.memory.obsidian_writer as ow
    def _should_not_be_called(*a, **k):
        called["write"] += 1
        return Path("x")
    monkeypatch.setattr(ow, "write_experiment_note", _should_not_be_called)

    agent = _setup_agent(tmp_path, monkeypatch)
    result = agent.run(_make_task())
    assert result.success is True
    assert called["write"] == 0


def _make_meta_suite_json():
    from aivc_platform.tracking.schemas import RunMetadata
    from eval.benchmarks.kang_eval import KangEvalReport
    from eval.benchmarks.norman_eval import NormanEvalReport
    from eval.eval_runner import EvalSuite

    meta = RunMetadata(run_id="r_success_1", pearson_r=0.88, delta_nonzero_pct=50.0)
    suite = EvalSuite(
        kang=KangEvalReport(
            run_id="r_success_1", pearson_r=0.88, delta_nonzero_pct=50.0,
            regression_guard_passed=True, passed=True,
        ),
        norman=NormanEvalReport(
            run_id="r_success_1", pearson_r_ctrl_sub=0.4,
            delta_nonzero_pct=20.0, ctrl_memorisation_score=0.5, passed=True,
        ),
        overall_passed=True,
    )
    return meta.model_dump_json(), suite.model_dump_json()


def test_obsidian_note_missing_triggers_backfill(monkeypatch, tmp_path):
    """When note is missing AND meta_json/suite_json are in payload,
    write_experiment_note IS called."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("next experiment text"))

    vault = tmp_path / "vault"
    (vault / "experiments").mkdir(parents=True)
    from aivc_platform.memory import vault as vault_mod
    monkeypatch.setattr(
        vault_mod.ObsidianConfig, "resolved_vault", lambda self: vault,
    )

    called = {"write": 0, "ctx": 0}
    import agents.research_agent as ra_mod
    # Patch the deferred imports inside _maybe_write_obsidian
    import aivc_platform.memory.obsidian_writer as ow
    import aivc_platform.memory.context_updater as cu

    def _fake_write(meta, suite, config):
        called["write"] += 1
        return vault / "experiments" / f"{meta.run_id}.md"

    def _fake_update(meta, suite=None, **kw):
        called["ctx"] += 1

    monkeypatch.setattr(ow, "write_experiment_note", _fake_write)
    monkeypatch.setattr(cu, "update_context", _fake_update)

    meta_json, suite_json = _make_meta_suite_json()
    task = AgentTask(
        agent_name="research_agent",
        run_id="r_success_1",
        payload={
            "dataset": "kang2018", "pearson_r": 0.88,
            "delta_nonzero_pct": 50.0, "w_scale_range": "(0.01, 0.1)",
            "neumann_k": 3, "checkpoint_path": "ckpt.pt", "wandb_url": "",
            "meta_json": meta_json, "suite_json": suite_json,
        },
    )
    agent = _setup_agent(tmp_path, monkeypatch)
    result = agent.run(task)
    assert result.success is True
    assert called["write"] == 1


def test_obsidian_backfill_skipped_when_payload_missing_meta(monkeypatch, tmp_path, caplog):
    """When note is missing and no meta_json/suite_json in payload,
    write_experiment_note is NOT called and a warning is logged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("next experiment text"))

    vault = tmp_path / "vault"
    (vault / "experiments").mkdir(parents=True)
    from aivc_platform.memory import vault as vault_mod
    monkeypatch.setattr(
        vault_mod.ObsidianConfig, "resolved_vault", lambda self: vault,
    )

    called = {"write": 0}
    import aivc_platform.memory.obsidian_writer as ow
    monkeypatch.setattr(ow, "write_experiment_note",
                        lambda *a, **k: called.__setitem__("write", called["write"] + 1))

    agent = _setup_agent(tmp_path, monkeypatch)
    with caplog.at_level("WARNING"):
        result = agent.run(_make_task())
    assert result.success is True
    assert called["write"] == 0
    assert any("backfill not possible" in rec.message for rec in caplog.records)


def test_next_experiment_written(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("NEXT_EXP_TEXT"))

    vault = tmp_path / "vault"
    (vault / "experiments").mkdir(parents=True)
    (vault / "experiments" / "r_success_1.md").write_text("exists")
    from aivc_platform.memory import vault as vault_mod
    monkeypatch.setattr(
        vault_mod.ObsidianConfig, "resolved_vault", lambda self: vault,
    )

    from agents import research_agent as ra
    monkeypatch.setattr(ra, "CLAUDE_OUTPUTS_DIR", tmp_path / "claude_out")
    agent = ResearchAgent(queue_dir=str(tmp_path / "q"))
    result = agent.run(_make_task())
    assert result.success is True
    out = Path(result.output_path)
    assert out.exists()
    assert "NEXT_EXP_TEXT" in out.read_text()


def test_wandb_failure_proceeds(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    class _BoomApi:
        def run(self, url):
            raise RuntimeError("wandb boom")

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key=None: _FakeClient("response"))

    # Inject a stub wandb module if missing.
    import sys, types
    wandb = sys.modules.get("wandb")
    if wandb is None:
        wandb = types.ModuleType("wandb")
        sys.modules["wandb"] = wandb
    monkeypatch.setattr(wandb, "Api", lambda: _BoomApi(), raising=False)

    vault = tmp_path / "vault"
    (vault / "experiments").mkdir(parents=True)
    (vault / "experiments" / "r_success_1.md").write_text("exists")
    from aivc_platform.memory import vault as vault_mod
    monkeypatch.setattr(
        vault_mod.ObsidianConfig, "resolved_vault", lambda self: vault,
    )

    from agents import research_agent as ra
    monkeypatch.setattr(ra, "CLAUDE_OUTPUTS_DIR", tmp_path / "claude_out")
    agent = ResearchAgent(queue_dir=str(tmp_path / "q"))

    task = AgentTask(
        agent_name="research_agent",
        run_id="r_success_1",
        payload={
            "dataset": "kang2018", "pearson_r": 0.88,
            "delta_nonzero_pct": 50.0, "w_scale_range": "(0.01, 0.1)",
            "neumann_k": 3, "checkpoint_path": "ckpt.pt",
            "wandb_url": "fake/wandb/url",
        },
    )
    result = agent.run(task)
    assert result.success is True
