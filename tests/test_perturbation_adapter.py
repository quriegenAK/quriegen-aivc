"""Unit tests for aivc.skills.adapter.PerturbationAdapter."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from aivc.skills.adapter import (
    PerturbationAdapter,
    save_adapter,
    load_adapter,
    ADAPTER_CKPT_SCHEMA_VERSION,
)


@pytest.fixture
def adapter():
    torch.manual_seed(0)
    return PerturbationAdapter(d=256)


def test_init_param_count(adapter):
    """Spec §3.1: ~131K params at d=256 (= 2*d*d + 2*d + 2*d for LN).
    Exact count: 2 * 256² (linear weights) + 2 * 256 (linear biases) + 2 * 256 (LN affine).
    """
    total = sum(p.numel() for p in adapter.parameters())
    # 2*256*256 = 131072 (linear weights)
    # 2*256 = 512 (linear biases)
    # 256 + 256 = 512 (LN gamma + beta)
    expected = 2 * 256 * 256 + 2 * 256 + 2 * 256
    assert total == expected, f"got {total}, expected {expected}"


def test_forward_shape(adapter):
    z = torch.randn(32, 256)
    out = adapter(z)
    assert out.shape == (32, 256)


def test_forward_batch_dims(adapter):
    """Adapter should handle (B, d) input. Other shapes raise."""
    for shape in [(1, 256), (32, 256), (128, 256)]:
        out = adapter(torch.randn(*shape))
        assert out.shape == shape


def test_forward_wrong_dim_raises(adapter):
    with pytest.raises(ValueError, match="last dim"):
        adapter(torch.randn(32, 128))   # wrong d


def test_init_d_validation():
    with pytest.raises(ValueError, match="positive"):
        PerturbationAdapter(d=0)
    with pytest.raises(ValueError, match="positive"):
        PerturbationAdapter(d=-1)


def test_backward_gradients_flow(adapter):
    """All 4 sublayer parameters should receive gradients on backward."""
    z = torch.randn(8, 256, requires_grad=False)
    out = adapter(z)
    # Use .pow(2).sum() not .sum() — LayerNorm zeros out .sum() gradient
    # by construction (mean-centering). See feedback_layernorm_gradient_trap.md.
    loss = out.pow(2).sum()
    loss.backward()
    for name, p in adapter.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_no_residual_connection(adapter):
    """Spec §3.1: adapter is NOT a residual layer. Output should NOT be
    close to input for a random init.
    """
    z = torch.randn(32, 256)
    out = adapter(z)
    cos_sim = torch.nn.functional.cosine_similarity(z, out, dim=-1).mean()
    # Random Linear → LN → GELU → Linear should produce output uncorrelated
    # with input. Even if cos sim isn't exactly 0, it shouldn't be > 0.5.
    assert cos_sim.abs() < 0.5, f"adapter output too close to input (cos sim {cos_sim:.4f})"


def test_eval_mode_deterministic(adapter):
    """In eval mode, two forwards on the same input must match exactly."""
    adapter.eval()
    z = torch.randn(8, 256)
    out_a = adapter(z)
    out_b = adapter(z)
    assert torch.allclose(out_a, out_b)


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    adapter = PerturbationAdapter(d=256, dropout=0.5)
    adapter.train()
    z = torch.randn(8, 256)
    out_a = adapter(z)
    out_b = adapter(z)
    # With dropout=0.5 in train mode, two forwards diverge
    assert not torch.allclose(out_a, out_b)


def test_dropout_inactive_in_eval_mode():
    torch.manual_seed(0)
    adapter = PerturbationAdapter(d=256, dropout=0.5)
    adapter.eval()
    z = torch.randn(8, 256)
    out_a = adapter(z)
    out_b = adapter(z)
    assert torch.allclose(out_a, out_b)


def test_save_load_roundtrip(adapter, tmp_path):
    """Checkpoint envelope schema_version=1 round-trip."""
    ckpt_path = tmp_path / "adapter.pt"
    save_adapter(adapter, ckpt_path, extra_meta={"trained_on": "mimitou"})
    loaded = load_adapter(ckpt_path)
    assert isinstance(loaded, PerturbationAdapter)
    assert loaded.d == adapter.d
    # State dict matches
    for (k_a, v_a), (k_b, v_b) in zip(
        adapter.state_dict().items(), loaded.state_dict().items()
    ):
        assert k_a == k_b
        assert torch.allclose(v_a, v_b)


def test_load_rejects_missing_schema(tmp_path):
    """A bare torch.save (no envelope) should fail at load."""
    bad_path = tmp_path / "bare.pt"
    torch.save({"weights": "garbage"}, bad_path)
    with pytest.raises(ValueError, match="schema_version"):
        load_adapter(bad_path)


def test_load_rejects_wrong_schema_version(tmp_path):
    bad_path = tmp_path / "wrong_version.pt"
    torch.save(
        {
            "schema_version": 999,
            "kind": "stage3a_adapter",
            "config": {"d": 256},
            "state_dict": PerturbationAdapter(d=256).state_dict(),
        },
        bad_path,
    )
    with pytest.raises(ValueError, match="schema_version 999"):
        load_adapter(bad_path)


def test_load_rejects_wrong_kind(tmp_path):
    """Reject checkpoints from other model types (defense-in-depth)."""
    wrong_kind = tmp_path / "wrong_kind.pt"
    torch.save(
        {
            "schema_version": ADAPTER_CKPT_SCHEMA_VERSION,
            "kind": "stage3b_temporal_ode",
            "config": {"d": 256},
            "state_dict": PerturbationAdapter(d=256).state_dict(),
        },
        wrong_kind,
    )
    with pytest.raises(ValueError, match="kind="):
        load_adapter(wrong_kind)
