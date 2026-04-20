"""Phase 6.5e — tests for the contrastive fine-tune pipeline (E1-rev1)."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.training.loss_registry import LossRegistry, _is_term_active
from aivc.training.pretrain_losses import (
    E1_STAGE,
    E1_WEIGHT_MASK,
    PRETRAIN_TERM_NAMES,
    _cross_modal_infonce,
    register_joint_contrastive_only_e1_terms,
    register_pretrain_terms,
)


# --------------------------------------------------------------------- #
# Reuse contract — E1 stage uses the SAME cross_modal_infonce loss.
# --------------------------------------------------------------------- #
def test_e1_reuses_parent_contrastive_loss_fn():
    """E1's registered contrastive term must be the exact ``_cross_modal_infonce``
    callable used by the parent pretrain stage. No new loss module."""
    pretrain_reg = LossRegistry()
    register_pretrain_terms(pretrain_reg)
    parent_contrastive = next(
        t for t in pretrain_reg.terms() if t.name == "cross_modal_infonce"
    )

    e1_reg = LossRegistry()
    register_joint_contrastive_only_e1_terms(e1_reg)
    e1_contrastive = next(
        t for t in e1_reg.terms() if t.name == "cross_modal_infonce"
    )

    assert e1_contrastive.fn is parent_contrastive.fn, (
        "E1 must reuse the parent's cross_modal_infonce callable byte-for-byte."
    )
    # Qualified-name check (robust under pytest module-reload side effects).
    assert e1_contrastive.fn.__qualname__ == _cross_modal_infonce.__qualname__
    assert e1_contrastive.fn.__module__ == "aivc.training.pretrain_losses"


# --------------------------------------------------------------------- #
# Weight mask contract (T6).
# --------------------------------------------------------------------- #
def test_e1_weight_mask_is_locked():
    """The E1 weight mask is locked to {recon:0, recon:0, contrastive:1.0,
    aux:0} — this is tripwire T6 at the module level."""
    assert E1_WEIGHT_MASK == {
        "masked_rna_recon": 0.0,
        "masked_atac_recon": 0.0,
        "cross_modal_infonce": 1.0,
        "peak_to_gene_aux": 0.0,
    }
    # Every key in the mask is a known pretrain term name.
    for name in E1_WEIGHT_MASK:
        assert name in PRETRAIN_TERM_NAMES, f"{name!r} not a pretrain term"


def test_e1_stage_registration_produces_contrastive_only():
    """Only cross_modal_infonce is registered under the E1 stage
    (weight 1.0); recon / aux have implicit weight 0."""
    reg = LossRegistry()
    register_joint_contrastive_only_e1_terms(reg)
    e1_terms = {t.name: t for t in reg.terms() if t.stage == E1_STAGE}
    assert set(e1_terms.keys()) == {"cross_modal_infonce"}, (
        f"E1 stage should have exactly {{'cross_modal_infonce'}}; "
        f"got {set(e1_terms.keys())}."
    )
    assert e1_terms["cross_modal_infonce"].weight == 1.0
    assert _is_term_active(E1_STAGE, E1_STAGE)


def test_register_does_not_touch_pretrain_registrations():
    """Registering the E1 term must NOT remove or modify the four
    pretrain-stage registrations."""
    reg = LossRegistry()
    register_pretrain_terms(reg)
    pretrain_names_before = sorted(t.name for t in reg.terms())
    pretrain_weights_before = {t.name: t.weight for t in reg.terms()}
    register_joint_contrastive_only_e1_terms(reg)
    pretrain_after = {
        t.name: t.weight for t in reg.terms() if t.stage == "pretrain"
    }
    assert sorted(pretrain_after.keys()) == pretrain_names_before
    assert pretrain_after == pretrain_weights_before


# --------------------------------------------------------------------- #
# Loss behavior sanity (parent loss, re-tested under E1 interface).
# --------------------------------------------------------------------- #
def test_cross_modal_infonce_identity_low_loss():
    """When z_rna == z_atac (perfectly aligned after L2 norm), the
    diagonal dominates the softmax and loss is small compared to the
    random-inputs baseline."""
    torch.manual_seed(0)
    B, D = 256, 128
    z = torch.randn(B, D)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss = _cross_modal_infonce(z_rna=z, z_atac=z, infonce_temperature=0.1)
    assert torch.isfinite(loss)
    # At τ=0.1 on L2-normalized identical views, loss << log(B).
    assert loss.item() < 1.0, f"identity-view loss={loss.item():.3f} too high"


def test_cross_modal_infonce_random_loss_near_logB():
    """Random L2-normalized views produce a loss near log(B) at τ=0.1."""
    torch.manual_seed(1)
    B, D = 256, 128
    za = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    zb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    loss = _cross_modal_infonce(z_rna=za, z_atac=zb, infonce_temperature=0.1)
    assert torch.isfinite(loss)
    # log(256) ≈ 5.545; allow wide band since random alignment is noisy.
    assert 2.5 < loss.item() < 8.0, (
        f"random-view loss={loss.item():.3f} outside [2.5, 8.0]"
    )


def test_cross_modal_infonce_symmetric():
    """Swapping z_rna and z_atac leaves the symmetric InfoNCE unchanged."""
    torch.manual_seed(2)
    B, D = 128, 64
    za = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    zb = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    l1 = _cross_modal_infonce(z_rna=za, z_atac=zb, infonce_temperature=0.07)
    l2 = _cross_modal_infonce(z_rna=zb, z_atac=za, infonce_temperature=0.07)
    assert torch.allclose(l1, l2, atol=1e-6), (
        f"asymmetric: {l1.item()} vs {l2.item()}"
    )


# --------------------------------------------------------------------- #
# Script-level: the fine-tune script's projection-init helpers.
# --------------------------------------------------------------------- #
def test_projections_from_parent_deepcopy():
    """The 'parent' projection-init path must return deepcopies — editing
    the fine-tune projections must NOT mutate the parent's head."""
    from aivc.training.pretrain_heads import MultiomePretrainHead
    from scripts.phase6_5e_finetune_contrastive import (
        _build_projections_from_parent,
    )

    head = MultiomePretrainHead(rna_dim=128, atac_dim=64, proj_dim=128, n_genes=36601)
    rna_proj, atac_proj, path = _build_projections_from_parent(head)
    assert path == "parent"

    parent_first_w = head.rna_proj[0].weight.detach().clone()
    # Mutate the copy; parent must remain untouched.
    with torch.no_grad():
        rna_proj[0].weight.add_(torch.ones_like(rna_proj[0].weight))
    assert torch.allclose(head.rna_proj[0].weight, parent_first_w), (
        "parent head was mutated by edits to the deepcopied projection."
    )


def test_fresh_projections_seeded():
    """Fresh projections under the locked seed are reproducible."""
    from scripts.phase6_5e_finetune_contrastive import (
        _build_projections_fresh,
        _set_global_seed,
    )

    _set_global_seed(3)
    a_rna, a_atac, path = _build_projections_fresh(
        rna_dim=128, atac_dim=64, proj_dim=128, hidden_dim=256
    )
    _set_global_seed(3)
    b_rna, b_atac, _ = _build_projections_fresh(
        rna_dim=128, atac_dim=64, proj_dim=128, hidden_dim=256
    )
    assert path == "seed3_fresh"
    for pa, pb in zip(a_rna.parameters(), b_rna.parameters()):
        assert torch.allclose(pa, pb)
    for pa, pb in zip(a_atac.parameters(), b_atac.parameters()):
        assert torch.allclose(pa, pb)


# --------------------------------------------------------------------- #
# Checkpoint loader contract (persisted ckpt layout).
# --------------------------------------------------------------------- #
def _make_minimal_e1_ckpt(tmp_path: Path) -> Path:
    """Build a minimal schema-v1 ckpt that mimics the E1 output layout,
    saved via torch.save directly (test-only helper)."""
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder
    from aivc.training.pretrain_heads import MultiomePretrainHead

    n_genes, n_peaks = 200, 500
    rna = SimpleRNAEncoder(n_genes=n_genes, hidden_dim=32, latent_dim=16)
    atac = PeakLevelATACEncoder(n_peaks=n_peaks, svd_dim=10, hidden_dim=16, attn_dim=8)
    head = MultiomePretrainHead(
        rna_dim=16, atac_dim=8, proj_dim=16, n_genes=n_genes, hidden_dim=16
    )
    ckpt_path = tmp_path / "e1_ckpt.pt"
    torch.save(
        {
            "schema_version": 1,
            "rna_encoder": rna.state_dict(),
            "atac_encoder": atac.state_dict(),
            "pretrain_head": head.state_dict(),
            "rna_encoder_class": "aivc.skills.rna_encoder.SimpleRNAEncoder",
            "atac_encoder_class": "aivc.skills.atac_peak_encoder.PeakLevelATACEncoder",
            "pretrain_head_class": "aivc.training.pretrain_heads.MultiomePretrainHead",
            "config": {
                "n_genes": n_genes,
                "hidden_dim": 32,
                "latent_dim": 16,
                "n_peaks": n_peaks,
                "atac_attn_dim": 8,
                "proj_dim": 16,
                "pretrain_stage": E1_STAGE,
                "parent_ckpt_sha": "0" * 64,
                "epochs_finetuned": 1,
                "batch_size": 256,
                "temperature": 0.1,
                "seed": 3,
                "projection_init_path": "parent",
                "weight_mask": E1_WEIGHT_MASK,
                "loss": "cross_modal_infonce",
            },
        },
        ckpt_path,
    )
    return ckpt_path


def test_e1_ckpt_loads_via_strict_loader(tmp_path):
    """Output E1 ckpt must load cleanly via the strict RNA loader —
    schema-v1, RNA encoder architecture unchanged, loadable as-is."""
    from aivc.training.ckpt_loader import load_pretrained_simple_rna_encoder

    path = _make_minimal_e1_ckpt(tmp_path)
    enc = load_pretrained_simple_rna_encoder(path)
    assert enc.n_genes == 200
    assert enc.latent_dim == 16


def test_e1_ckpt_has_no_projection_heads_persisted(tmp_path):
    """The persisted ckpt must not contain rna_proj / atac_proj under
    top-level keys (projections are discarded after fine-tuning)."""
    path = _make_minimal_e1_ckpt(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # top-level: no rna_proj / atac_proj keys
    assert "rna_proj" not in ckpt
    assert "atac_proj" not in ckpt
    # rna_encoder state_dict: no proj keys
    for k in ckpt["rna_encoder"]:
        assert "proj" not in k
    for k in ckpt["atac_encoder"]:
        assert "proj" not in k
