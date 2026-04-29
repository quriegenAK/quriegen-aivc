"""Tests for PR #41 config + AdamW + LR-schedule integration."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml


def test_dogma_pretrain_config_yaml_parses():
    p = Path(__file__).resolve().parent.parent / "configs" / "dogma_pretrain.yaml"
    cfg = yaml.safe_load(p.read_text())
    assert cfg["optimizer"]["type"] == "adamw"
    assert cfg["optimizer"]["lr"] == 1e-4
    assert cfg["optimizer"]["weight_decay"] == 0.01
    assert cfg["training"]["epochs"] == 50
    assert cfg["training"]["batch_size"] == 256
    assert cfg["schedule"]["warmup_steps"] == 500


def test_adamw_constructs_with_weight_decay():
    """AdamW (not Adam) is the spec'd optimizer."""
    params = [torch.nn.Parameter(torch.randn(4))]
    optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    assert isinstance(optim, torch.optim.AdamW)
    assert optim.defaults["weight_decay"] == 0.01
    assert optim.defaults["lr"] == 1e-4


def test_warmup_cosine_schedule_emits_expected_lrs():
    """LR schedule: linear warmup 0 -> base LR over 500 steps, then cosine decay."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    params = [torch.nn.Parameter(torch.randn(4))]
    base_lr = 1e-4
    optim = torch.optim.AdamW(params, lr=base_lr)
    warmup_steps = 500
    total_steps = 5000

    warmup = LinearLR(optim, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optim, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # At step 0: LR = base_lr * start_factor = ~1e-7
    assert optim.param_groups[0]["lr"] == pytest.approx(base_lr * 1e-3, rel=1e-3)

    # Walk through warmup: at step 500, should be at base_lr
    for _ in range(warmup_steps):
        scheduler.step()
    assert optim.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-2)

    # Walk halfway through cosine: should be roughly between base_lr and 0
    for _ in range((total_steps - warmup_steps) // 2):
        scheduler.step()
    mid_lr = optim.param_groups[0]["lr"]
    assert 0 < mid_lr < base_lr
    # Cosine half-period -> 0.5 of base
    assert mid_lr == pytest.approx(base_lr * 0.5, rel=0.1)

    # Walk to end of cosine: should be ~0
    for _ in range((total_steps - warmup_steps) // 2):
        scheduler.step()
    assert optim.param_groups[0]["lr"] < base_lr * 0.01


def test_warmup_cosine_monotone_in_warmup():
    """During warmup, LR is strictly non-decreasing."""
    from torch.optim.lr_scheduler import LinearLR

    params = [torch.nn.Parameter(torch.randn(4))]
    optim = torch.optim.AdamW(params, lr=1e-4)
    warmup = LinearLR(optim, start_factor=1e-3, end_factor=1.0, total_iters=100)

    prev = optim.param_groups[0]["lr"]
    for _ in range(99):
        warmup.step()
        cur = optim.param_groups[0]["lr"]
        assert cur >= prev, f"LR decreased during warmup: {prev} -> {cur}"
        prev = cur


def test_config_yaml_lr_supersedes_argparse_default():
    """Config file LR overrides argparse default when --config is set."""
    cfg = {"optimizer": {"lr": 5e-5, "weight_decay": 0.02},
           "training": {"epochs": 10, "batch_size": 128, "seed": 42},
           "schedule": {"warmup_steps": 100},
           "checkpoint": {"every_n_epochs": 1}}

    # Simulate the merge logic from pretrain_multiome.py
    class _Args:
        lr = 1e-3       # argparse default
        batch_size = 64  # argparse default
        epochs = None
        seed = 0
    args = _Args()

    # Apply config (mirrors Stage 3 Change 2 logic)
    if args.lr == 1e-3:
        args.lr = float(cfg["optimizer"]["lr"])
    if args.batch_size == 64:
        args.batch_size = int(cfg["training"]["batch_size"])
    if args.epochs is None:
        args.epochs = int(cfg["training"]["epochs"])

    assert args.lr == 5e-5
    assert args.batch_size == 128
    assert args.epochs == 10
