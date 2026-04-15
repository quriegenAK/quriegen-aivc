"""Phase 5 import-time invariant check.

Mirrors tests/test_phase4_no_import_side_effects.py: importing the Phase 5
modules must NOT mutate global torch state, and train_week3.py must NOT
reference any Phase 5 symbol (new modules remain dead code).
"""
import os
import sys
import importlib

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


PHASE5_MODULES = (
    "aivc.training.pretrain_heads",
    "aivc.training.pretrain_losses",
)

PHASE5_FORBIDDEN_SYMBOLS = (
    "MultiomePretrainHead",
    "pretrain_multiome",
    "pretrain_heads",
    "pretrain_losses",
    "masked_rna_recon",
    "masked_atac_recon",
    "cross_modal_infonce",
    "peak_to_gene_aux",
    "register_pretrain_terms",
)


def _snapshot():
    return {
        "default_dtype": torch.get_default_dtype(),
        "rng_state_hash": hash(torch.random.get_rng_state().numpy().tobytes()),
        "num_threads": torch.get_num_threads(),
        "grad_enabled": torch.is_grad_enabled(),
    }


def test_phase5_imports_do_not_mutate_torch_state():
    torch.manual_seed(12345)
    before = _snapshot()

    for mod_name in PHASE5_MODULES:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    after = _snapshot()
    assert before == after, (
        f"Phase 5 imports mutated torch state:\n"
        f"  before={before}\n  after={after}"
    )


def test_train_week3_does_not_reference_phase5_symbols():
    tw3_path = os.path.join(_REPO_ROOT, "train_week3.py")
    with open(tw3_path) as fh:
        src = fh.read()
    for forbidden in PHASE5_FORBIDDEN_SYMBOLS:
        assert forbidden not in src, (
            f"train_week3.py references Phase 5 symbol {forbidden!r} — "
            f"new modules are no longer dead code."
        )
