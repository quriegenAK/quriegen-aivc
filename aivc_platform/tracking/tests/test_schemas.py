"""
Acceptance tests for platform/tracking/.

Tests schemas validation, post_run_hook routing, agent dispatch,
apply_frozen_modules, and idempotent finish().

All external services (wandb, mlflow, anthropic) are mocked.
No live network calls.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from pydantic import ValidationError

from aivc_platform.tracking.schemas import RunMetadata, RunStatus, PostRunDecision
from aivc_platform.tracking.experiment_logger import (
    AgentDispatcher,
    ExperimentLogger,
    apply_frozen_modules,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_artifacts(tmp_path, monkeypatch):
    """Redirect all artifact writes to a temp directory."""
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def mock_model():
    """Build a minimal PerturbationPredictor-like model with genelink + feature_expander."""
    from perturbation_model import PerturbationPredictor
    model = PerturbationPredictor(n_genes=100, feature_dim=16, hidden1_dim=16,
                                   hidden2_dim=8, num_head1=2, num_head2=1,
                                   decoder_hidden=32)
    return model


def _make_meta(**overrides) -> RunMetadata:
    """Helper to build RunMetadata with sensible defaults."""
    defaults = {
        "run_id": "test-run-001",
        "dataset": "kang2018_pbmc",
        "pearson_r": 0.88,
        "delta_nonzero_pct": 5.0,
        "checkpoint_path": "/tmp/model.pt",
        "w_scale_range": (0.01, 0.3),
        "neumann_k": 3,
    }
    defaults.update(overrides)
    return RunMetadata(**defaults)


# ── Test 1: frozen_modules must include genelink ─────────────────────

def test_frozen_modules_requires_genelink():
    """RunMetadata(frozen_modules=["feature_expander"]) raises ValidationError."""
    with pytest.raises(ValidationError, match="genelink"):
        RunMetadata(
            run_id="test",
            frozen_modules=["feature_expander"],
        )


# ── Test 2: w_scale_range must be ordered ─────────────────────────────

def test_w_scale_range_ordered():
    """RunMetadata(w_scale_range=(0.3, 0.1)) raises ValidationError."""
    with pytest.raises(ValidationError, match="w_scale_range"):
        RunMetadata(
            run_id="test",
            w_scale_range=(0.3, 0.1),
        )


# ── Test 3: escape hatch allows unfreezing genelink ──────────────────

def test_escape_hatch_allows_unfreeze(monkeypatch):
    """AIVC_ALLOW_GAT_UNFREEZE=1 allows frozen_modules=[]."""
    monkeypatch.setenv("AIVC_ALLOW_GAT_UNFREEZE", "1")
    meta = RunMetadata(run_id="test", frozen_modules=[])
    assert meta.frozen_modules == []


# ── Test 4: failure path (ctrl memorisation) ──────────────────────────

@mock.patch("aivc_platform.tracking.experiment_logger.init_wandb", return_value=None)
@mock.patch.dict("sys.modules", {"mlflow": mock.MagicMock()})
def test_post_run_hook_failure_path(mock_wandb):
    """delta_nonzero_pct=0 → failure note + training_agent prompt, no registry."""
    logger = ExperimentLogger()
    meta = _make_meta(
        delta_nonzero_pct=0.0,
        pearson_r=0.9,
        ctrl_memorisation_score=0.99,
    )
    logger.start(meta)
    logger.finish(meta)

    # Failure note should exist
    failure_notes = list(Path("artifacts/failure_notes").glob("failure_*.json"))
    assert len(failure_notes) == 1, f"Expected 1 failure note, found {len(failure_notes)}"

    # Training agent prompt should exist
    agent_prompts = list(Path("artifacts/agent_queue").glob("training_agent_*.md"))
    assert len(agent_prompts) >= 1, "Expected training_agent prompt in queue"

    # Registry should NOT exist
    registry = Path("artifacts/registry/latest_checkpoint.json")
    assert not registry.exists(), "Registry should not be written on failure"


# ── Test 5: success path ──────────────────────────────────────────────

@mock.patch("aivc_platform.tracking.experiment_logger.init_wandb", return_value=None)
@mock.patch.dict("sys.modules", {"mlflow": mock.MagicMock()})
def test_post_run_hook_success_path(mock_wandb):
    """pearson_r=0.88 + delta_nonzero_pct=5.0 → registry + research_agent."""
    logger = ExperimentLogger()
    meta = _make_meta(
        pearson_r=0.88,
        delta_nonzero_pct=5.0,
        checkpoint_path="/tmp/model.pt",
    )
    logger.start(meta)
    logger.finish(meta)

    # Registry should be written
    registry = Path("artifacts/registry/latest_checkpoint.json")
    assert registry.exists(), "Registry should be written on success"
    data = json.loads(registry.read_text())
    assert data["run_id"] == "test-run-001"
    assert data["pearson_r"] == 0.88
    assert data["checkpoint_path"] == "/tmp/model.pt"
    assert "registered_at" in data
    assert "wandb_url" in data

    # Research agent prompt should exist
    agent_prompts = list(Path("artifacts/agent_queue").glob("research_agent_*.md"))
    assert len(agent_prompts) >= 1, "Expected research_agent prompt in queue"

    # mlflow.register_model should NOT have been called
    mlflow_mock = __import__("sys").modules["mlflow"]
    if hasattr(mlflow_mock, "register_model"):
        mlflow_mock.register_model.assert_not_called()


# ── Test 6: finish() is idempotent ────────────────────────────────────

@mock.patch("aivc_platform.tracking.experiment_logger.init_wandb", return_value=None)
@mock.patch.dict("sys.modules", {"mlflow": mock.MagicMock()})
def test_finish_idempotent(mock_wandb):
    """finish() called twice with same run_id dispatches only once."""
    logger = ExperimentLogger()
    meta = _make_meta(delta_nonzero_pct=0.0, pearson_r=0.9)
    logger.start(meta)
    logger.finish(meta)
    logger.finish(meta)  # second call should be a no-op

    failure_notes = list(Path("artifacts/failure_notes").glob("failure_*.json"))
    assert len(failure_notes) == 1, f"Expected exactly 1 failure note, found {len(failure_notes)}"

    agent_prompts = list(Path("artifacts/agent_queue").glob("training_agent_*.md"))
    assert len(agent_prompts) == 1, f"Expected exactly 1 agent prompt, found {len(agent_prompts)}"


# ── Test 7: apply_frozen_modules with nonexistent module ──────────────

def test_apply_frozen_modules_nonexistent(mock_model):
    """apply_frozen_modules(model, ["nonexistent"]) raises AttributeError."""
    with pytest.raises(AttributeError):
        apply_frozen_modules(mock_model, ["nonexistent"])


# ── Test 8: apply_frozen_modules freezes genelink only ────────────────

def test_apply_frozen_modules_genelink_only(mock_model):
    """Freeze genelink → all genelink params frozen, feature_expander untouched."""
    # Before freeze: all params should require grad
    for p in mock_model.genelink.parameters():
        assert p.requires_grad is True

    apply_frozen_modules(mock_model, ["genelink"])

    # After freeze: genelink frozen
    for p in mock_model.genelink.parameters():
        assert p.requires_grad is False, "genelink param should be frozen"

    # feature_expander should still be trainable
    for p in mock_model.feature_expander.parameters():
        assert p.requires_grad is True, "feature_expander param should remain trainable"
