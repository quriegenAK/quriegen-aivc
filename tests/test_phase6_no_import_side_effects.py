"""Phase 6 import-time invariant check.

Mirrors tests/test_phase5_no_import_side_effects.py: importing the
Phase 6 modules must NOT mutate global torch state, and train_week3.py
must NOT reference any Phase 6 symbol.
"""
import os
import sys
import importlib

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


PHASE6_MODULES = (
    "aivc.training.ckpt_loader",
)

PHASE6_FORBIDDEN_SYMBOLS = (
    "linear_probe_pretrain",
    "ckpt_loader",
    "exp1_linear_probe_ablation",
    "load_pretrained_simple_rna_encoder",
    "CheckpointSchemaError",
)


def _snapshot():
    return {
        "default_dtype": torch.get_default_dtype(),
        "rng_state_hash": hash(torch.random.get_rng_state().numpy().tobytes()),
        "num_threads": torch.get_num_threads(),
        "grad_enabled": torch.is_grad_enabled(),
    }


def test_phase6_imports_do_not_mutate_torch_state():
    torch.manual_seed(12345)
    before = _snapshot()

    for mod_name in PHASE6_MODULES:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    after = _snapshot()
    assert before == after, (
        f"Phase 6 imports mutated torch state:\n"
        f"  before={before}\n  after={after}"
    )


def test_train_week3_does_not_reference_phase6_symbols():
    tw3_path = os.path.join(_REPO_ROOT, "train_week3.py")
    with open(tw3_path) as fh:
        src = fh.read()
    for forbidden in PHASE6_FORBIDDEN_SYMBOLS:
        assert forbidden not in src, (
            f"train_week3.py references Phase 6 symbol {forbidden!r} — "
            f"new modules are no longer dead code."
        )
