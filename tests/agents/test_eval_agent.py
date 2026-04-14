"""Tests for agents.eval_agent."""
from __future__ import annotations

import json

import pytest

from agents.base_agent import AgentTask
from agents.eval_agent import EvalAgent
from eval.benchmarks.kang_eval import KangEvalReport
from eval.benchmarks.norman_eval import NormanEvalReport
from eval.eval_runner import EvalSuite


def _kang_pass(rid="r"):
    return KangEvalReport(
        run_id=rid, pearson_r=0.88, delta_nonzero_pct=50.0,
        regression_guard_passed=True, passed=True,
    )


def _kang_fail(rid="r"):
    return KangEvalReport(
        run_id=rid, pearson_r=0.5, regression_guard_passed=False, passed=False,
        failure_reason="Kang regression guard: r=0.5 < 0.873",
    )


def _norman_pass(rid="r"):
    return NormanEvalReport(
        run_id=rid, pearson_r_ctrl_sub=0.4, delta_nonzero_pct=20.0,
        ctrl_memorisation_score=0.5, passed=True,
    )


def _norman_collapse(rid="r"):
    return NormanEvalReport(
        run_id=rid, pearson_r_ctrl_sub=0.0, delta_nonzero_pct=0.0,
        ctrl_memorisation_score=0.995, passed=False,
        failure_reason="delta collapse",
    )


def test_kang_guard_failure(monkeypatch, tmp_path):
    def fake_run(*a, **k):
        return EvalSuite(kang=_kang_fail(), norman=None, overall_passed=False,
                        halt_reason="Kang regression guard: r=0.5 < 0.873")
    monkeypatch.setattr("eval.eval_runner.run_eval_suite", fake_run)

    agent = EvalAgent(queue_dir=str(tmp_path / "q"))
    result = agent.run(AgentTask(
        agent_name="eval_agent", run_id="r1",
        payload={"checkpoint_path": "fake.pt"},
    ))
    assert result.success is False
    assert result.output_path is not None
    assert json.loads(open(result.output_path).read())["kang"]["regression_guard_passed"] is False


def test_norman_delta_collapse(monkeypatch, tmp_path):
    def fake_run(*a, **k):
        return EvalSuite(kang=_kang_pass(), norman=_norman_collapse(),
                        overall_passed=False, halt_reason="delta collapse")
    monkeypatch.setattr("eval.eval_runner.run_eval_suite", fake_run)

    agent = EvalAgent(queue_dir=str(tmp_path / "q"))
    result = agent.run(AgentTask(
        agent_name="eval_agent", run_id="r2",
        payload={"checkpoint_path": "fake.pt"},
    ))
    assert result.success is False
    assert result.error == "norman delta collapse"
    assert result.output_path is not None


def test_all_pass(monkeypatch, tmp_path):
    def fake_run(*a, **k):
        return EvalSuite(kang=_kang_pass(), norman=_norman_pass(),
                        overall_passed=True)
    monkeypatch.setattr("eval.eval_runner.run_eval_suite", fake_run)

    agent = EvalAgent(queue_dir=str(tmp_path / "q"))
    result = agent.run(AgentTask(
        agent_name="eval_agent", run_id="r3",
        payload={"checkpoint_path": "fake.pt"},
    ))
    assert result.success is True
    assert result.output_path is not None


def test_fallback_to_registry(monkeypatch, tmp_path):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"checkpoint_path": "registry_ckpt.pt"}))
    seen = {}

    def fake_run(checkpoint_path, **k):
        seen["ckpt"] = checkpoint_path
        return EvalSuite(kang=_kang_pass(), norman=_norman_pass(), overall_passed=True)

    monkeypatch.setattr("eval.eval_runner.run_eval_suite", fake_run)

    agent = EvalAgent(
        queue_dir=str(tmp_path / "q"),
        registry_path=str(reg),
    )
    result = agent.run(AgentTask(
        agent_name="eval_agent", run_id="r4",
        payload={},  # no checkpoint_path
    ))
    assert result.success is True
    assert seen["ckpt"] == "registry_ckpt.pt"
