"""Tests for agents.data_agent."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from agents.base_agent import AgentTask
from agents.data_agent import DataAgent


def _make_adata(tmp_path, n_vars=3010, obs_cols=("label", "replicate", "cell_type")):
    n_obs = 100
    X = np.zeros((n_obs, n_vars), dtype=np.float32)
    obs = pd.DataFrame(
        {col: ["a"] * n_obs for col in obs_cols},
        index=[f"c{i}" for i in range(n_obs)],
    )
    adata = anndata.AnnData(X=X, obs=obs)
    path = tmp_path / "test.h5ad"
    adata.write_h5ad(path)
    return str(path)


def _make_cert_dir(tmp_path, which=("kang2018",)):
    d = tmp_path / "certs"
    d.mkdir()
    for name in which:
        if name == "kang2018":
            (d / "kang2018.json").write_text(json.dumps({"experiment_id": "kang"}))
        elif name == "quriegen_pending":
            (d / "quriegen_pending.json").write_text(json.dumps({"experiment_id": "q"}))
    return str(d)


def test_kang_3010_with_cert_succeeds(tmp_path):
    h5ad = _make_adata(tmp_path)
    cert_dir = _make_cert_dir(tmp_path, which=("kang2018",))
    agent = DataAgent(cert_dir=cert_dir, queue_dir=str(tmp_path / "queue"))
    result = agent.run(AgentTask(
        agent_name="data_agent",
        run_id="r1",
        payload={"h5ad_path": h5ad, "dataset": "kang2018", "require_cert": True},
    ))
    assert result.success is True
    assert result.output_path is not None


def test_kang_3009_genes_fails(tmp_path):
    h5ad = _make_adata(tmp_path, n_vars=3009)
    cert_dir = _make_cert_dir(tmp_path, which=("kang2018",))
    agent = DataAgent(cert_dir=cert_dir, queue_dir=str(tmp_path / "queue"))
    result = agent.run(AgentTask(
        agent_name="data_agent",
        run_id="r1",
        payload={"h5ad_path": h5ad, "dataset": "kang2018", "require_cert": True},
    ))
    assert result.success is False
    assert "n_vars" in (result.error or "")


def test_missing_required_obs_fails(tmp_path):
    h5ad = _make_adata(tmp_path, obs_cols=("label", "replicate"))  # missing cell_type
    cert_dir = _make_cert_dir(tmp_path, which=("kang2018",))
    agent = DataAgent(cert_dir=cert_dir, queue_dir=str(tmp_path / "queue"))
    result = agent.run(AgentTask(
        agent_name="data_agent",
        run_id="r1",
        payload={"h5ad_path": h5ad, "dataset": "kang2018", "require_cert": True},
    ))
    assert result.success is False
    assert "missing required obs" in (result.error or "")


def test_require_cert_missing_cert_fails(tmp_path):
    h5ad = _make_adata(tmp_path)
    cert_dir = tmp_path / "empty_certs"
    cert_dir.mkdir()
    agent = DataAgent(cert_dir=str(cert_dir), queue_dir=str(tmp_path / "queue"))
    result = agent.run(AgentTask(
        agent_name="data_agent",
        run_id="r1",
        payload={"h5ad_path": h5ad, "dataset": "kang2018", "require_cert": True},
    ))
    assert result.success is False
    assert result.error == "pairing certificate missing"


def test_quriegen_pending_cert_flagged(tmp_path, caplog):
    # Use default obs cols (kang-valid — REQUIRED_OBS_BY_DATASET returns empty
    # for unknown dataset so any columns pass).
    h5ad = _make_adata(tmp_path)
    cert_dir = _make_cert_dir(tmp_path, which=("quriegen_pending",))
    agent = DataAgent(cert_dir=cert_dir, queue_dir=str(tmp_path / "queue"))
    with caplog.at_level("WARNING"):
        result = agent.run(AgentTask(
            agent_name="data_agent",
            run_id="r1",
            payload={"h5ad_path": h5ad, "dataset": "quriegen_batch1", "require_cert": True},
        ))
    assert result.success is False
    assert result.error == "cert_pending"
    assert any("PENDING" in rec.message for rec in caplog.records)
