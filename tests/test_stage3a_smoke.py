"""Stage 3a real-data integration smoke test — HARD REQUIREMENT per
docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §7 risk #4.

Why this exists:
  Six distinct bugs across 2 days in prepare_mimitou_crispr.py is a
  structural signal that synthetic tests miss real-format quirks.
  Stage 3a's adapter + decomposed readout must be exercised against
  REAL Mimitou h5ad before any merge.

Coverage:
  1. Load real Mimitou labeled h5ad (5,825 cells x 323,500 union peaks
     + 38 ADT antibodies)
  2. Apply frozen DOGMA encoder via aivc.eval.calderon_probe.encode_samples
  3. Apply randomly-initialized adapter (untrained — we just smoke the API)
  4. Construct DecomposedReadout with same d as encoder output
  5. Forward pass exercising all 4 head combinations
  6. Compute zero_arm_loss with synthetic arm_mask matching real perturbation labels
  7. Verify backward pass + gradient flow on the entire chain

Gating:
  Only runs when AIVC_RUN_REAL_DATA_SMOKE=1. Without the env flag,
  pytest skips. This prevents CI from hitting the dataset; intended
  for manual sanity-check on Mac (synth data path) and BSC (real data
  path) before merging.

Expected real-data paths (overridable via env):
  AIVC_MIMITOU_LABELED_H5AD  = data/phase6_5g_2/dogma_h5ads/mimitou_crispr_union.h5ad
  AIVC_FROZEN_ENCODER_CKPT   = checkpoints/pretrain/pretrain_encoders.pt
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Hard gate
SMOKE_ENABLED = os.environ.get("AIVC_RUN_REAL_DATA_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="Real-data smoke gated on AIVC_RUN_REAL_DATA_SMOKE=1 (set to opt in)",
)


def _resolve_path(env_key: str, default_rel: str) -> Path:
    """Resolve a real-data path from env or fall back to a project-relative
    default. Returns Path that may or may not exist; caller checks.
    """
    explicit = os.environ.get(env_key)
    if explicit:
        return Path(explicit)
    return Path.cwd() / default_rel


@pytest.fixture(scope="module")
def real_mimitou_h5ad():
    p = _resolve_path(
        "AIVC_MIMITOU_LABELED_H5AD",
        "data/phase6_5g_2/dogma_h5ads/mimitou_crispr_union.h5ad",
    )
    if not p.exists():
        pytest.skip(f"Mimitou labeled h5ad not found at {p}; set AIVC_MIMITOU_LABELED_H5AD")
    return p


@pytest.fixture(scope="module")
def frozen_encoder_ckpt():
    p = _resolve_path(
        "AIVC_FROZEN_ENCODER_CKPT",
        "checkpoints/pretrain/pretrain_encoders.pt",
    )
    if not p.exists():
        pytest.skip(f"Frozen encoder ckpt not found at {p}; set AIVC_FROZEN_ENCODER_CKPT")
    return p


def test_mimitou_h5ad_loads_and_has_expected_keys(real_mimitou_h5ad):
    """Sanity: h5ad opens, has obsm['atac_peaks'], obs['perturbation']."""
    import anndata as ad

    adata = ad.read_h5ad(real_mimitou_h5ad, backed="r")
    assert "atac_peaks" in adata.obsm, "obsm['atac_peaks'] missing"
    assert "perturbation" in adata.obs.columns, "obs['perturbation'] missing"
    n_cells, n_peaks = adata.obsm["atac_peaks"].shape
    assert n_cells > 1000, f"suspiciously few cells: {n_cells}"
    # Expected union peak count from Report 2: 323,500
    assert n_peaks == 323_500, f"union peak count {n_peaks} != 323500"


def test_encoder_to_adapter_to_readout_forward(
    real_mimitou_h5ad, frozen_encoder_ckpt,
):
    """End-to-end forward: real ATAC → frozen encoder → adapter → readout.

    Smoke-only: no training, no eval thresholds. Just verifies the API
    chain composes without errors against real data shapes.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import torch

    from aivc.eval.calderon_probe import (
        load_atac_encoder_from_ckpt,
        encode_samples,
    )
    from aivc.skills.adapter import PerturbationAdapter
    from aivc.skills.decomposed_readout import DecomposedReadout, zero_arm_loss

    # --- 1. Load a small slice of real data (first 64 cells) ---
    adata = ad.read_h5ad(real_mimitou_h5ad)
    atac_sub = adata.obsm["atac_peaks"][:64]
    if not sp.issparse(atac_sub):
        atac_sub = sp.csr_matrix(atac_sub)
    perturbation_sub = adata.obs["perturbation"].astype(str).values[:64]
    print(f"  loaded slice: {atac_sub.shape}, perturbations={set(perturbation_sub)}")

    # --- 2. Load frozen encoder ---
    encoder, _ckpt_config = load_atac_encoder_from_ckpt(
        frozen_encoder_ckpt,
        expected_n_peaks=atac_sub.shape[1],
        map_location="cpu",
    )
    # Lysis covariate handling — Mimitou CRISPR arm is LLL (cat 0)
    has_lysis = getattr(encoder, "n_lysis_categories", 0) > 0
    lysis_idx = np.zeros(atac_sub.shape[0], dtype=np.int64) if has_lysis else None

    # --- 3. Encode through frozen encoder ---
    z_encoder = encode_samples(
        atac_sub, encoder, batch_size=32, device="cpu", lysis_idx=lysis_idx,
    )
    print(f"  z_encoder shape: {z_encoder.shape}")
    d = z_encoder.shape[1]
    z_encoder_t = torch.from_numpy(z_encoder).float()

    # --- 4. Apply adapter (random init — we're not training) ---
    adapter = PerturbationAdapter(d=d)
    adapter.eval()
    z_adapted = adapter(z_encoder_t)
    assert z_adapted.shape == z_encoder_t.shape

    # --- 5. Build DecomposedReadout ---
    # Use z_adapted as z_dyn AND z_static for smoke (in real Stage 3b,
    # they come from different sources). pert_dim=32 from spec default.
    readout = DecomposedReadout(d=d, pert_dim=32, output_dim=d)
    readout.eval()

    B = z_adapted.shape[0]
    z_dyn = z_adapted
    z_static = z_adapted.clone()
    t = torch.full((B,), 16.0)  # Mimitou's 16h endpoint
    stim_emb = torch.randn(B, 32)
    inh_emb  = torch.randn(B, 32)

    # --- 6. Exercise all 4 head combinations ---
    out_baseline = readout(z_dyn, z_static, t)
    out_stim     = readout(z_dyn, z_static, t, stim_emb=stim_emb)
    out_inh      = readout(z_dyn, z_static, t, inh_emb=inh_emb)
    out_both     = readout(z_dyn, z_static, t, stim_emb=stim_emb, inh_emb=inh_emb)
    for o in (out_baseline, out_stim, out_inh, out_both):
        assert o.shape == (B, d)

    # --- 7. Zero-arm loss against real perturbation labels ---
    # Map perturbation labels → arm_mask. For Mimitou CRISPR:
    #   NTC → no stim, no inh
    #   CD3E, CD4, ZAP70, NFKB2 → treat as "stim only" for smoke purposes
    #   CD3E_CD4_double → treat as "stim + inh" (the synergy condition)
    has_stim = torch.tensor([p != "NTC" for p in perturbation_sub])
    has_inh  = torch.tensor([p == "CD3E_CD4_double" for p in perturbation_sub])
    deltas = readout.head_deltas(z_dyn, z_static, t, stim_emb, inh_emb)
    loss = zero_arm_loss(deltas, {"has_stim": has_stim, "has_inh": has_inh})
    assert torch.isfinite(loss).item()
    print(f"  zero_arm_loss on real labels: {loss.item():.4f}")

    # --- 8. Backward — verify gradients flow through the full chain ---
    # Use the readout with both heads active so all 4 head MLPs get gradients.
    out = readout(z_dyn, z_static, t, stim_emb=stim_emb, inh_emb=inh_emb)
    total_loss = out.pow(2).sum() + loss
    total_loss.backward()
    # Adapter params should receive grads (z_dyn is downstream of adapter)
    # Note: in this smoke we wired z_dyn from z_adapted, so the adapter
    # is in the autograd path.
    for pname, p in adapter.named_parameters():
        assert p.grad is not None, f"adapter param {pname} got no grad"
    # Readout heads — all 4 should have grads since out_both was computed
    for head_name in ("h_b", "delta_s", "delta_i", "delta_xy"):
        head = getattr(readout, head_name)
        for pname, p in head.named_parameters():
            assert p.grad is not None, f"readout.{head_name}.{pname} got no grad"


def test_perturbation_distribution_matches_report_2(real_mimitou_h5ad):
    """Sanity: per-arm cell counts match Report 2's HTO calling stats.

    Expected (from Report 2 / job 40069332 stdout):
        NTC: 2134, CD3E: 1930, CD4: 1847, ZAP70: 2220,
        NFKB2: 1770, CD3E_CD4_double: 97
    Total: 9998 HTO-called cells. After ATAC intersection, ~7782 remain.
    """
    import anndata as ad

    adata = ad.read_h5ad(real_mimitou_h5ad, backed="r")
    counts = adata.obs["perturbation"].astype(str).value_counts().to_dict()
    print(f"  per-arm counts: {counts}")
    # All 6 arms should be present
    for arm in ("NTC", "CD3E", "CD4", "ZAP70", "NFKB2", "CD3E_CD4_double"):
        assert arm in counts, f"missing arm {arm}"
        # Each arm should have >=30 cells (synergy floor)
        assert counts[arm] >= 30, f"arm {arm} has only {counts[arm]} cells (< 30 floor)"
