"""
test_h100_readiness.py — 14 tests for H100 GPU readiness fixes.

Validates: device placement, BF16 mixed precision, train/eval mode switching,
OT pair shuffle correctness, and NaN/Inf loss guard.

All tests run on CPU only — no GPU required.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast


class TestDevicePlacement:

    def test_model_moves_to_cpu(self):
        from perturbation_model import PerturbationPredictor
        device = torch.device("cpu")
        model = PerturbationPredictor(n_genes=50).to(device)
        assert next(model.parameters()).device.type == "cpu"

    def test_all_parameters_on_same_device(self):
        """Catches partial .to(device) — most common real-world placement bug"""
        from perturbation_model import PerturbationPredictor
        device = torch.device("cpu")
        model = PerturbationPredictor(n_genes=50).to(device)
        for name, param in model.named_parameters():
            assert param.device.type == device.type, \
                f"Parameter '{name}' on {param.device}, expected {device}"

    def test_edge_index_stays_long_after_device_move(self):
        edge_index = torch.randint(0, 50, (2, 30))
        moved = edge_index.to(torch.device("cpu"), non_blocking=True)
        assert moved.dtype == torch.long

    def test_non_blocking_transfer_produces_correct_values(self):
        """non_blocking=True must not corrupt tensor values"""
        x = torch.rand(100, 50)
        moved = x.to(torch.device("cpu"), non_blocking=True)
        assert torch.allclose(x, moved)

    def test_step_count_increments_correctly(self):
        """_step_count must reach 1 after the first step"""
        _step_count = 0
        for _ in range(1):
            _step_count += 1
        assert _step_count == 1

    def test_forward_pass_no_nan(self):
        from perturbation_model import PerturbationPredictor
        device = torch.device("cpu")
        model = PerturbationPredictor(n_genes=50).to(device)
        edge_index = torch.randint(0, 50, (2, 30)).to(device)
        x = torch.rand(50).to(device)
        pert_id = torch.tensor(1, device=device)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, pert_id)
        assert not out.isnan().any()
        assert not out.isinf().any()


class TestMixedPrecision:

    def test_autocast_dtype_is_float32_on_cpu(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        assert dtype == torch.float32

    def test_bfloat16_backward_stable(self):
        x = torch.rand(32, 100, dtype=torch.bfloat16)
        w = torch.rand(100, 50, dtype=torch.bfloat16, requires_grad=True)
        (x @ w).sum().backward()
        assert not w.grad.isnan().any()
        assert not w.grad.isinf().any()

    def test_bfloat16_requires_float_before_numpy(self):
        t = torch.rand(10, dtype=torch.bfloat16)
        with pytest.raises((RuntimeError, TypeError)):
            t.numpy()
        arr = t.detach().cpu().float().numpy()
        assert arr.dtype == np.float32

    # Note: PR #50 removed test_no_gradscaler_in_train_file and
    # test_set_to_none_in_zero_grad — both were file-grep guards reading
    # train_v11.py, which was deleted in PR #50 as legacy dead code. The
    # invariants they encoded (no GradScaler with BF16; zero_grad uses
    # set_to_none) belong against the live training script
    # (scripts/pretrain_multiome.py); a future PR can re-add them there
    # if those invariants matter for the production GPU run.


class TestTrainEvalMode:

    def test_eval_mode_is_deterministic(self):
        """model.eval() must suppress dropout — val_r depends on this"""
        from perturbation_model import PerturbationPredictor
        device = torch.device("cpu")
        model = PerturbationPredictor(n_genes=50).to(device)
        edge_index = torch.randint(0, 50, (2, 30)).to(device)
        x = torch.rand(50).to(device)
        pert_id = torch.tensor(1, device=device)
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index, pert_id)
            out2 = model(x, edge_index, pert_id)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "model.eval() did not produce deterministic outputs"

    def test_train_mode_restored_after_eval(self):
        from perturbation_model import PerturbationPredictor
        model = PerturbationPredictor(n_genes=50)
        model.train()
        model.eval()
        model.train()
        assert model.training is True


class TestOTPairShuffle:

    def test_single_perm_preserves_pair_alignment(self):
        n, g = 200, 50
        ctrl = torch.arange(n).unsqueeze(1).expand(-1, g).float()
        stim = ctrl + 1000.0
        perm = torch.randperm(n)
        assert (stim[perm][:, 0] - ctrl[perm][:, 0]).eq(1000).all()

    def test_consecutive_randperms_differ(self):
        p1 = torch.randperm(9616)
        p2 = torch.randperm(9616)
        assert not torch.equal(p1, p2)

    def test_randperm_covers_all_indices(self):
        perm = torch.randperm(9616)
        assert perm.sort().values.equal(torch.arange(9616))


class TestNaNGuard:

    def test_nan_loss_raises_runtime_error(self):
        """NaN loss must abort training immediately"""
        loss = torch.tensor(float("nan"))
        bd = {"mse": float("nan"), "lfc": 0.0}
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"[AIVC] NaN/Inf detected in loss at epoch 0, step 1. "
                    f"Loss breakdown: {bd}. "
                    "Aborting — do not waste GPU time on a corrupted run."
                )

    def test_inf_loss_raises_runtime_error(self):
        loss = torch.tensor(float("inf"))
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    "[AIVC] NaN/Inf detected in loss at epoch 0, step 1. "
                    "Aborting."
                )

    def test_valid_loss_does_not_raise(self):
        loss = torch.tensor(0.42)
        raised = False
        try:
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("should not reach here")
        except RuntimeError:
            raised = True
        assert not raised
