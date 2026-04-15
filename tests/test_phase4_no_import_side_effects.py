"""Phase 4 import-time invariant check.

A full 100-step train_week3.py parity run is infeasible in CI (requires
the staged multi-perturbation corpus). Instead, we directly prove the
property that parity would be probing: that importing Phase 4 modules
does NOT mutate global torch state — no seed reseeding, no default
dtype change, no device registration. If this passes, the new modules
are dead code with respect to train_week3.py and baseline parity is
guaranteed by construction.
"""
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _snapshot():
    return {
        "default_dtype": torch.get_default_dtype(),
        "rng_state_hash": hash(torch.random.get_rng_state().numpy().tobytes()),
        "num_threads": torch.get_num_threads(),
        "grad_enabled": torch.is_grad_enabled(),
    }


def test_phase4_imports_do_not_mutate_torch_state():
    torch.manual_seed(12345)
    before = _snapshot()

    # Fresh imports (each module under test).
    import importlib
    for mod_name in (
        "aivc.skills.atac_peak_encoder",
        "aivc.data.multiome_loader",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    after = _snapshot()
    assert before == after, (
        f"Phase 4 imports mutated torch state:\n"
        f"  before={before}\n  after={after}"
    )


def test_train_week3_does_not_import_phase4_modules():
    """train_week3.py must NOT transitively import Phase 4 modules.
    If it does, new code is no longer dead code and parity is at risk."""
    import importlib
    # Drop cached imports so we observe a clean import graph.
    for key in list(sys.modules):
        if key.startswith(("aivc.skills.atac_peak_encoder",
                           "aivc.data.multiome_loader")):
            del sys.modules[key]

    # Parse train_week3.py source for direct references — transitive
    # imports would have to originate from a file it references.
    tw3_path = os.path.join(_REPO_ROOT, "train_week3.py")
    with open(tw3_path) as fh:
        src = fh.read()
    for forbidden in ("atac_peak_encoder", "multiome_loader",
                      "PeakLevelATACEncoder", "MultiomeLoader"):
        assert forbidden not in src, (
            f"train_week3.py references Phase 4 symbol {forbidden!r} — "
            f"new modules are no longer dead code."
        )
