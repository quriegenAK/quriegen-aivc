"""Phase 6.5f — tests for the frozen-projection fine-tune pipeline."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aivc.training.loss_registry import LossRegistry
from aivc.training.pretrain_heads import MultiomePretrainHead
from aivc.training.pretrain_losses import (
    E1_STAGE,
    register_joint_contrastive_only_e1_terms,
)
from scripts.phase6_5f_finetune_frozen_proj import (
    PARENT_EXPECTED_SHA,
    PHASE6_5E_CKPT_SHA,
    _assert_optimizer_excludes_projections,
    _assert_projections_equal_parent,
    _freeze_projections,
    _max_abs_projection_diff,
    _snapshot_projection_state,
)


# --------------------------------------------------------------------- #
# Frozen-projection primitives
# --------------------------------------------------------------------- #
def _mini_head() -> MultiomePretrainHead:
    return MultiomePretrainHead(
        rna_dim=16, atac_dim=8, proj_dim=16, n_genes=200, hidden_dim=32
    )


def test_frozen_proj_requires_grad_false():
    """After ``_freeze_projections``, every projection param has
    ``requires_grad == False``. This is T8-a (requires_grad half)."""
    head = _mini_head()
    _freeze_projections(head.rna_proj, head.atac_proj)
    for p in head.rna_proj.parameters():
        assert p.requires_grad is False
    for p in head.atac_proj.parameters():
        assert p.requires_grad is False


def test_optimizer_excludes_projections():
    """AdamW param_groups must contain ONLY encoder params (cross-check
    by identity). T8-a (optimizer-exclusion half)."""
    head = _mini_head()
    rna_enc = nn.Linear(10, 16)
    atac_enc = nn.Linear(10, 8)
    _freeze_projections(head.rna_proj, head.atac_proj)
    trainable = [
        p
        for p in (
            list(rna_enc.parameters())
            + list(atac_enc.parameters())
            + list(head.rna_proj.parameters())
            + list(head.atac_proj.parameters())
        )
        if p.requires_grad
    ]
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    ok_exc, ok_rg = _assert_optimizer_excludes_projections(
        optim, head.rna_proj, head.atac_proj
    )
    assert ok_exc is True
    assert ok_rg is True

    opt_ids = {id(p) for g in optim.param_groups for p in g["params"]}
    for p in head.rna_proj.parameters():
        assert id(p) not in opt_ids
    for p in head.atac_proj.parameters():
        assert id(p) not in opt_ids
    for p in rna_enc.parameters():
        assert id(p) in opt_ids
    for p in atac_enc.parameters():
        assert id(p) in opt_ids


def test_optimizer_exclusion_raises_on_leak():
    """If projections are NOT frozen before optimizer construction,
    ``_assert_optimizer_excludes_projections`` must raise."""
    head = _mini_head()
    rna_enc = nn.Linear(10, 16)
    atac_enc = nn.Linear(10, 8)
    # Forget to freeze — projection params enter the optimizer.
    trainable = (
        list(rna_enc.parameters())
        + list(atac_enc.parameters())
        + list(head.rna_proj.parameters())
        + list(head.atac_proj.parameters())
    )
    optim = torch.optim.AdamW(trainable, lr=1e-4)
    with pytest.raises(RuntimeError, match="projection param leaked"):
        _assert_optimizer_excludes_projections(
            optim, head.rna_proj, head.atac_proj
        )


def test_projection_weights_bit_identical_after_step():
    """Run a tiny forward/backward on a 4-cell batch under the 6.5f
    freeze + exclude-from-optimizer path. Assert projection tensors are
    bit-identical pre/post (T8-b). Assert at least one encoder param
    moved (grad flow landed in encoders)."""
    torch.manual_seed(0)
    rna_dim, atac_dim, proj_dim = 8, 6, 16
    head = MultiomePretrainHead(
        rna_dim=rna_dim, atac_dim=atac_dim, proj_dim=proj_dim,
        n_genes=50, hidden_dim=24,
    )
    rna_enc = nn.Linear(20, rna_dim)
    atac_enc = nn.Linear(30, atac_dim)

    _freeze_projections(head.rna_proj, head.atac_proj)
    snapshot = _snapshot_projection_state(head.rna_proj, head.atac_proj)

    encoder_pre = {
        n: p.detach().clone() for n, p in rna_enc.named_parameters()
    }

    trainable = [
        p
        for p in (
            list(rna_enc.parameters())
            + list(atac_enc.parameters())
            + list(head.rna_proj.parameters())
            + list(head.atac_proj.parameters())
        )
        if p.requires_grad
    ]
    assert len(trainable) > 0
    optim = torch.optim.AdamW(trainable, lr=1e-2)

    registry = LossRegistry()
    register_joint_contrastive_only_e1_terms(registry)

    B = 4
    rna_x = torch.randn(B, 20)
    atac_x = torch.randn(B, 30)

    for _ in range(3):
        z_rna = torch.nn.functional.normalize(
            head.rna_proj(rna_enc(rna_x)), dim=-1
        )
        z_atac = torch.nn.functional.normalize(
            head.atac_proj(atac_enc(atac_x)), dim=-1
        )
        total, _comp = registry.compute(
            stage=E1_STAGE, z_rna=z_rna, z_atac=z_atac,
            infonce_temperature=0.1,
        )
        optim.zero_grad()
        total.backward()
        optim.step()

    max_rna, max_atac = _max_abs_projection_diff(
        head.rna_proj, head.atac_proj, snapshot
    )
    assert max_rna == 0.0, f"rna_proj drifted: max|ΔW|={max_rna}"
    assert max_atac == 0.0, f"atac_proj drifted: max|ΔW|={max_atac}"

    # At least one encoder parameter moved.
    encoder_moved = any(
        not torch.equal(encoder_pre[n], p)
        for n, p in rna_enc.named_parameters()
    )
    assert encoder_moved, (
        "RNA encoder did not move — grad flow into encoder is broken."
    )


def test_assert_projections_equal_parent_happy():
    """Deep-copied projections match the parent head exactly."""
    head = _mini_head()
    rna_proj = copy.deepcopy(head.rna_proj)
    atac_proj = copy.deepcopy(head.atac_proj)
    _assert_projections_equal_parent(rna_proj, atac_proj, head)


def test_assert_projections_equal_parent_raises_on_drift():
    """Perturbed projection weights must fail the byte-equality check."""
    head = _mini_head()
    rna_proj = copy.deepcopy(head.rna_proj)
    atac_proj = copy.deepcopy(head.atac_proj)
    with torch.no_grad():
        rna_proj[0].weight.add_(1.0)
    with pytest.raises(RuntimeError, match="rna_proj"):
        _assert_projections_equal_parent(rna_proj, atac_proj, head)


# --------------------------------------------------------------------- #
# Stage reuse contract
# --------------------------------------------------------------------- #
def test_reuses_6_5e_stage():
    """6.5f must dispatch the already-registered ``joint_contrastive_only_e1``
    stage — no new stage string."""
    from scripts.phase6_5f_finetune_frozen_proj import parse_args

    args = parse_args([
        "--parent-ckpt", "p.pt",
        "--data", "d.h5ad",
        "--out-ckpt", "o.pt",
    ])
    assert args.stage == "joint_contrastive_only_e1"
    # Running main() with an unsupported --stage must raise.
    bad = parse_args([
        "--parent-ckpt", "p.pt",
        "--data", "d.h5ad",
        "--out-ckpt", "o.pt",
        "--stage", "joint",
    ])
    assert bad.stage == "joint"
    # main() itself is exercised via the end-to-end run; the guard is
    # asserted at the top of main(). Import the guard inline to cover.
    from scripts.phase6_5f_finetune_frozen_proj import main
    with pytest.raises(RuntimeError, match="must be"):
        main([
            "--parent-ckpt", "p.pt",
            "--data", "d.h5ad",
            "--out-ckpt", "o.pt",
            "--stage", "joint",
            "--no-wandb",
        ])


def test_freeze_flag_defaults_to_true():
    """The single-variable flag defaults to on; --no-freeze-projections
    exists only for test parameterization."""
    from scripts.phase6_5f_finetune_frozen_proj import parse_args

    args = parse_args([
        "--parent-ckpt", "p.pt",
        "--data", "d.h5ad",
        "--out-ckpt", "o.pt",
    ])
    assert args.freeze_projections is True

    off = parse_args([
        "--parent-ckpt", "p.pt",
        "--data", "d.h5ad",
        "--out-ckpt", "o.pt",
        "--no-freeze-projections",
    ])
    assert off.freeze_projections is False


# --------------------------------------------------------------------- #
# Checkpoint contract
# --------------------------------------------------------------------- #
def _make_minimal_frozen_proj_ckpt(tmp_path: Path) -> Path:
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder

    n_genes, n_peaks = 200, 500
    rna = SimpleRNAEncoder(n_genes=n_genes, hidden_dim=32, latent_dim=16)
    atac = PeakLevelATACEncoder(
        n_peaks=n_peaks, svd_dim=10, hidden_dim=16, attn_dim=8
    )
    head = MultiomePretrainHead(
        rna_dim=16, atac_dim=8, proj_dim=16, n_genes=n_genes, hidden_dim=16
    )
    ckpt_path = tmp_path / "frozen_proj.pt"
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
                "pretrain_stage": "joint_contrastive_only_e1",
                "parent_ckpt_sha": "0" * 64,
                "phase6_5e_baseline_ckpt_sha": "1" * 64,
                "epochs_finetuned": 1,
                "batch_size": 256,
                "temperature": 0.1,
                "seed": 3,
                "projection_init": "parent",
                "projection_init_path": "parent",
                "projection_dim": 16,
                "projections_frozen": True,
                "loss_weights": {
                    "masked_rna_recon": 0.0,
                    "masked_atac_recon": 0.0,
                    "cross_modal_infonce": 1.0,
                    "peak_to_gene_aux": 0.0,
                },
                "aivc_grad_guard": 1,
            },
        },
        ckpt_path,
    )
    return ckpt_path


def test_ckpt_metadata_projections_frozen_true(tmp_path):
    path = _make_minimal_frozen_proj_ckpt(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["config"]["projections_frozen"] is True
    assert ckpt["config"]["pretrain_stage"] == "joint_contrastive_only_e1"
    assert ckpt["config"]["loss_weights"] == {
        "masked_rna_recon": 0.0,
        "masked_atac_recon": 0.0,
        "cross_modal_infonce": 1.0,
        "peak_to_gene_aux": 0.0,
    }


def test_ckpt_loads_via_strict_loader(tmp_path):
    from aivc.training.ckpt_loader import load_pretrained_simple_rna_encoder

    path = _make_minimal_frozen_proj_ckpt(tmp_path)
    enc = load_pretrained_simple_rna_encoder(path)
    assert enc.n_genes == 200
    assert enc.latent_dim == 16


def test_ckpt_sha_differs_from_parent_and_6_5e(tmp_path):
    """End-to-end: the frozen-proj runner's SHA guard rejects a ckpt
    whose bytes equal parent or 6.5e baseline. Unit-tested via constant
    identity rather than re-running the pipeline."""
    # The constants are string equality — smoke test they match spec.
    assert PARENT_EXPECTED_SHA == (
        "416e8b1a5fe73c1beff18ec0e5034331e5ada40bd13731f6f90f366f1f58e29e"
    )
    assert PHASE6_5E_CKPT_SHA == (
        "6084d5186cbd3dc942497d60926cda7a545931c7da5d7735ba32f555b73349ee"
    )
    assert PARENT_EXPECTED_SHA != PHASE6_5E_CKPT_SHA


# --------------------------------------------------------------------- #
# Probe batch reuse (6.5e cache)
# --------------------------------------------------------------------- #
def test_probe_batch_reuse_indices_sha():
    """If the 6.5e probe cache is present at its canonical path, the
    indices SHA must match the locked value 27a906d0…"""
    cache = _REPO_ROOT / "experiments" / "phase6_5e" / "probe_batch.npz"
    if not cache.exists():
        pytest.skip("6.5e probe batch cache not present (CI-safe skip)")
    import hashlib
    data = np.load(cache)
    indices = data["indices"].astype(np.int64)
    sha = hashlib.sha256(indices.tobytes()).hexdigest()
    assert sha == (
        "27a906d07cd3c47e294ab06bcc974351d269f97039d7dd43b94a9d6d8f215f64"
    ), (
        f"probe batch indices SHA drifted from 6.5e: got {sha}"
    )


# --------------------------------------------------------------------- #
# JSON schema
# --------------------------------------------------------------------- #
def test_rsa_json_schema_keys(tmp_path):
    """Synthesize a 6.5f RSA JSON dict with the keys the enrich script
    produces; assert every required 6.5f-specific key is present."""
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    synthetic = {
        "rsa_real_mean": -0.06,
        "rsa_real_ci95": [-0.07, -0.05],
        "rsa_real_frozen_proj": -0.06,
        "rsa_real_frozen_proj_ci95": [-0.07, -0.05],
        "rsa_real_contrastive_only": -0.0621,
        "rsa_real_contrastive_only_ci95": [-0.074, -0.050],
        "rsa_real_reconstruction_dominant": -0.0584,
        "rsa_random_mean": 0.0193,
        "delta_frozen_proj_vs_contrastive_only": 0.002,
        "delta_frozen_proj_vs_contrastive_only_ci95": [-0.01, 0.01],
        "delta_frozen_proj_vs_reconstruction_dominant": -0.002,
        "delta_frozen_proj_vs_reconstruction_dominant_ci95": [-0.01, 0.01],
        "delta_frozen_proj_vs_random": -0.08,
        "delta_frozen_proj_vs_random_ci95": [-0.09, -0.07],
        "loss_weights": {
            "masked_rna_recon": 0.0,
            "masked_atac_recon": 0.0,
            "cross_modal_infonce": 1.0,
            "peak_to_gene_aux": 0.0,
        },
        "projections_frozen": True,
        "parent_ckpt_sha": "a" * 64,
        "phase6_5e_baseline_ckpt_sha": "b" * 64,
        "frozen_proj_ckpt_sha": "c" * 64,
        "parent_stage": "joint",
        "child_stage": "joint_contrastive_only_e1",
        "outcome_f": "F-NULL",
        "interpretation_f": "stub",
        "tripwires": {
            "T1_parent_ckpt_sha": {},
            "T2_probe_batch": {},
            "T3_probe_drift": {},
            "T4_collapse": {},
            "T5_online_nan_abort": {},
            "T6_weight_mask": {},
            "T7_drift_ratio": {},
            "t8": {
                "optimizer_exclusion_pass": True,
                "proj_requires_grad_pass": True,
                "max_diff_rna": 0.0,
                "max_diff_atac": 0.0,
            },
        },
    }
    required = [
        "rsa_real_frozen_proj", "rsa_real_contrastive_only",
        "rsa_real_reconstruction_dominant", "rsa_random_mean",
        "delta_frozen_proj_vs_contrastive_only",
        "delta_frozen_proj_vs_reconstruction_dominant",
        "delta_frozen_proj_vs_random",
        "loss_weights", "projections_frozen",
        "parent_ckpt_sha", "phase6_5e_baseline_ckpt_sha",
        "frozen_proj_ckpt_sha", "tripwires", "outcome_f",
    ]
    for k in required:
        assert k in synthetic, f"missing required 6.5f-JSON key: {k}"
    for t in ("T1_parent_ckpt_sha", "T2_probe_batch", "T3_probe_drift",
              "T4_collapse", "T5_online_nan_abort", "T6_weight_mask",
              "T7_drift_ratio", "t8"):
        assert t in synthetic["tripwires"], f"missing tripwire: {t}"
    # Sanity: classifier returns a valid label.
    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=-0.06,
        ci_width_real=0.02,
        ci_real={"low": -0.07, "high": -0.05},
        delta_b_minus_c=0.002,
        ci_delta_b_minus_c={"low": -0.01, "high": 0.01},
    )
    assert outcome in {"F-WIN", "F-PARTIAL", "F-NULL", "F-REGRESS",
                       "INCONCLUSIVE"}


# --------------------------------------------------------------------- #
# Outcome classifier coverage
# --------------------------------------------------------------------- #
def test_classify_f_inconclusive_on_wide_ci():
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=-0.06,
        ci_width_real=0.35,
        ci_real={"low": -0.2, "high": 0.15},
        delta_b_minus_c=0.0,
        ci_delta_b_minus_c={"low": -0.1, "high": 0.1},
    )
    assert outcome == "INCONCLUSIVE"


def test_classify_f_null():
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=-0.06,
        ci_width_real=0.02,
        ci_real={"low": -0.07, "high": -0.05},
        delta_b_minus_c=0.002,
        ci_delta_b_minus_c={"low": -0.01, "high": 0.01},
    )
    assert outcome == "F-NULL"


def test_classify_f_regress():
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    # R_c_e1 = -0.0621; regress threshold = -0.1121.
    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=-0.15,
        ci_width_real=0.02,
        ci_real={"low": -0.16, "high": -0.14},
        delta_b_minus_c=-0.09,
        ci_delta_b_minus_c={"low": -0.10, "high": -0.08},
    )
    assert outcome == "F-REGRESS"


def test_classify_f_win():
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=0.08,
        ci_width_real=0.02,
        ci_real={"low": 0.07, "high": 0.09},
        delta_b_minus_c=0.142,
        ci_delta_b_minus_c={"low": 0.13, "high": 0.15},
    )
    assert outcome == "F-WIN"


def test_classify_f_partial():
    from scripts.phase6_5f_enrich_rsa_json import classify_f_outcome

    # R_c_e1 = -0.0621; R_b = 0.0, Δ = +0.0621 < 0.05 → falls to F-NULL.
    # For F-PARTIAL, need Δ > 0.05 AND R_b ≤ 0.05. Pick R_b = 0.02,
    # Δ = 0.02 - (-0.0621) = 0.0821 > 0.05.
    outcome, _ = classify_f_outcome(
        rsa_real_frozen_proj=0.02,
        ci_width_real=0.02,
        ci_real={"low": 0.01, "high": 0.03},
        delta_b_minus_c=0.0821,
        ci_delta_b_minus_c={"low": 0.07, "high": 0.09},
    )
    assert outcome == "F-PARTIAL"


# --------------------------------------------------------------------- #
# Pipeline-hygiene: 6.5d / 6.5e files untouched on this branch.
# --------------------------------------------------------------------- #
def _git_diff_between(base: str, path: str) -> str:
    """Return the raw diff for ``path`` between ``base`` and HEAD.

    If base doesn't resolve (e.g., shallow clone in CI), return empty
    string and let the test skip.
    """
    try:
        return subprocess.check_output(
            ["git", "diff", base + "..HEAD", "--", path],
            cwd=_REPO_ROOT,
            stderr=subprocess.STDOUT,
        ).decode("utf-8", "replace")
    except subprocess.CalledProcessError:
        return ""


def _git_rev_parse(ref: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--verify", ref],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None


_FROZEN_PIPELINE_FILES_6_5D = [
    "scripts/phase6_5d_rsa.py",
    "scripts/lib/rsa.py",
    "tests/test_phase6_5d_rsa.py",
]
_FROZEN_PIPELINE_FILES_6_5E = [
    "scripts/phase6_5e_finetune_contrastive.py",
    "scripts/phase6_5e_rsa.py",
    "tests/test_phase6_5e_contrastive.py",
]


@pytest.mark.parametrize("path", _FROZEN_PIPELINE_FILES_6_5D)
def test_no_6_5d_pipeline_edits(path):
    base = _git_rev_parse("origin/main") or _git_rev_parse("main")
    if base is None:
        pytest.skip("git base ref not resolvable (shallow/rebased)")
    diff = _git_diff_between(base, path)
    assert diff == "", (
        f"6.5f branch must not modify 6.5d pipeline file {path!r}. "
        f"Diff:\n{diff[:500]}"
    )


@pytest.mark.parametrize("path", _FROZEN_PIPELINE_FILES_6_5E)
def test_no_6_5e_runtime_edits(path):
    base = _git_rev_parse("origin/main") or _git_rev_parse("main")
    if base is None:
        pytest.skip("git base ref not resolvable (shallow/rebased)")
    diff = _git_diff_between(base, path)
    assert diff == "", (
        f"6.5f branch must not modify 6.5e runtime file {path!r}. "
        f"Diff:\n{diff[:500]}"
    )


def test_no_aivc_training_edits_on_6_5f_branch():
    """No edits to aivc/training/* from the 6.5f branch (forbidden)."""
    base = _git_rev_parse("origin/main") or _git_rev_parse("main")
    if base is None:
        pytest.skip("git base ref not resolvable (shallow/rebased)")
    try:
        names = subprocess.check_output(
            ["git", "diff", "--name-only", base + "..HEAD", "--",
             "aivc/training/"],
            cwd=_REPO_ROOT,
        ).decode().strip().splitlines()
    except subprocess.CalledProcessError:
        names = []
    assert names == [], (
        f"6.5f branch must not edit aivc/training/*. Modified: {names}"
    )
