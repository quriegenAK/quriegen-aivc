"""Phase 5 registration tests for the four pretrain loss terms."""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aivc.training.loss_registry import (
    LossRegistry,
    LossTerm,
    _is_term_active,
    _PRETRAIN_FORBIDDEN_NAMES,
)
from aivc.training.pretrain_losses import (
    PRETRAIN_TERM_NAMES,
    register_pretrain_terms,
    _guard_pretrain_name,
)


EXPECTED_WEIGHTS = {
    "masked_rna_recon": 1.0,
    "masked_atac_recon": 1.0,
    "cross_modal_infonce": 0.5,
    "peak_to_gene_aux": 0.1,
}


def _fresh_registry() -> LossRegistry:
    r = LossRegistry()
    register_pretrain_terms(r)
    return r


def test_all_four_registered_under_pretrain():
    reg = _fresh_registry()
    terms = {t.name: t for t in reg.terms()}
    for name in PRETRAIN_TERM_NAMES:
        assert name in terms, f"missing term {name}"
        assert terms[name].stage == "pretrain", (
            f"{name} must be stage='pretrain', got {terms[name].stage!r}"
        )
        assert terms[name].weight == EXPECTED_WEIGHTS[name]


def test_all_four_pass_allow_list():
    """Phase 3 allow-list: stage='pretrain' must be active under
    requested_stage='pretrain'."""
    reg = _fresh_registry()
    for t in reg.terms():
        assert _is_term_active(t.stage, "pretrain"), (
            f"{t.name} tagged {t.stage!r} does not pass pretrain allow-list"
        )


def test_none_trigger_forbidden_name_block():
    """Phase 3 forbidden-name block: no new pretrain term may share a
    name with _PRETRAIN_FORBIDDEN_NAMES."""
    for name in PRETRAIN_TERM_NAMES:
        assert name not in _PRETRAIN_FORBIDDEN_NAMES, (
            f"{name!r} collides with _PRETRAIN_FORBIDDEN_NAMES"
        )


def test_synthetic_leak_causal_ordering_leak_is_rejected():
    """If a future PR attempts to register a causal-adjacent term under
    stage='pretrain', the Phase 5 registration guard must RAISE so the
    leak is caught at PR time, not at compute() time."""
    with pytest.raises(ValueError) as exc:
        _guard_pretrain_name("causal_ordering_leak")
    assert "causal" in str(exc.value).lower()

    # Also verify that register_pretrain_terms stays clean after the
    # synthetic guard trip — no residual state leaked into a registry.
    reg = LossRegistry()
    register_pretrain_terms(reg)
    names = [t.name for t in reg.terms()]
    assert "causal_ordering_leak" not in names
    assert set(names) == set(PRETRAIN_TERM_NAMES)
