"""PR #54b2 integration smoke: loader → collate → _dogma_pretrain_loss with
SupCon + VICReg active.

Synthetic-data smoke test that exercises the full data path:
  1. MultiomeLoader with labels_obs_col + masked_classes + min_confidence
  2. dogma_collate stacks cell_type_idx + supcon_eligible_mask
  3. _dogma_pretrain_loss with z_supcon + cell_type_idx threads through
     to _supcon_loss and _vicreg_variance via the LossRegistry
  4. Backward gradient flows to z_supcon, z_rna, z_atac, z_protein

Also locks the BACKWARD COMPAT contract: when z_supcon is not threaded
through, _dogma_pretrain_loss yields the original 4-term composition
unchanged (no supcon/vicreg keys in components dict).
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from aivc.data.collate import dogma_collate
from aivc.data.multiome_loader import MultiomeLoader

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from losses import _dogma_pretrain_loss  # noqa: E402


PEAK_SET_TSV = (
    "chr1\t100\t200\tpeak_0\n"
    "chr1\t300\t400\tpeak_1\n"
    "chr1\t500\t600\tpeak_2\n"
    "chr1\t700\t800\tpeak_3\n"
    "chr1\t900\t1000\tpeak_4\n"
)


@pytest.fixture
def synth_loader(tmp_path: Path) -> MultiomeLoader:
    """16-cell synthetic loader with cell_type / cell_type_confidence."""
    rng = np.random.RandomState(42)
    n_cells, n_genes, n_peaks, n_proteins = 16, 20, 5, 4

    cell_types = (
        ["CD4_T"] * 4
        + ["CD8_T"] * 4
        + ["B"] * 3
        + ["NK"] * 2
        + ["other"] * 2  # masked
        + ["other_T"] * 1  # masked
    )
    confidences = (
        [0.95] * 4   # CD4_T high
        + [0.85] * 4 # CD8_T high
        + [0.95] * 3 # B
        + [0.55] * 2 # NK below threshold
        + [0.99] * 2 # other (masked anyway)
        + [0.99] * 1 # other_T (masked anyway)
    )
    assert len(cell_types) == n_cells
    assert len(confidences) == n_cells

    obs = pd.DataFrame({
        "lysis_protocol": ["LLL"] * n_cells,
        "cell_type": pd.Categorical(cell_types),
        "cell_type_confidence": np.asarray(confidences, dtype=np.float32),
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["atac_peaks"] = rng.poisson(1.0, size=(n_cells, n_peaks)).astype(np.float32)
    adata.obsm["protein"] = rng.poisson(5.0, size=(n_cells, n_proteins)).astype(np.float32)

    h5ad_path = tmp_path / "synth.h5ad"
    adata.write_h5ad(h5ad_path, compression=None)
    peak_path = tmp_path / "peaks.tsv"
    peak_path.write_text(PEAK_SET_TSV)

    return MultiomeLoader(
        h5ad_path=str(h5ad_path),
        schema="obsm_atac",
        peak_set_path=str(peak_path),
        atac_key="atac_peaks",
        protein_obsm_key="protein",
        lysis_protocol="LLL",
        protein_panel_id="totalseq_a_210",
        labels_obs_col="cell_type",
        confidence_obs_col="cell_type_confidence",
        masked_classes=["other", "other_T"],
        min_confidence=0.6,
    )


def _build_synthetic_z_and_recon(B: int, latent_dim: int = 16):
    """Return (z_rna, z_atac, z_protein, z_supcon, all recons) all requires_grad."""
    torch.manual_seed(0)
    z_rna = torch.randn(B, latent_dim, requires_grad=True)
    z_atac = torch.randn(B, latent_dim, requires_grad=True)
    z_protein = torch.randn(B, latent_dim, requires_grad=True)
    # z_supcon: L2-normalized mean of the three modalities (PR #54c will
    # build this in the orchestrator; smoke just needs SOMETHING coherent)
    z_joint = (
        F.normalize(z_rna, dim=-1)
        + F.normalize(z_atac, dim=-1)
        + F.normalize(z_protein, dim=-1)
    ) / 3.0
    z_supcon = F.normalize(z_joint, dim=-1)

    rna_recon = torch.randn(B, 20, requires_grad=True)
    rna_target = torch.randn(B, 20)
    atac_recon = torch.randn(B, 5, requires_grad=True)
    atac_target = torch.randn(B, 5)
    protein_recon = torch.randn(B, 4, requires_grad=True)
    protein_target = torch.randn(B, 4)
    return {
        "z_rna": z_rna, "z_atac": z_atac, "z_protein": z_protein,
        "z_supcon": z_supcon,
        "rna_recon": rna_recon, "rna_target": rna_target,
        "atac_recon": atac_recon, "atac_target": atac_target,
        "protein_recon": protein_recon, "protein_target": protein_target,
    }


# --- Backward compat: no z_supcon → 4-term composition unchanged ------------

def test_dogma_loss_backward_compat_no_supcon(synth_loader):
    items = [synth_loader[i] for i in range(8)]
    batch = dogma_collate(items)
    B = batch["rna"].shape[0]
    Z = _build_synthetic_z_and_recon(B)

    total, components = _dogma_pretrain_loss(
        rna_recon=Z["rna_recon"], rna_target=Z["rna_target"],
        atac_recon=Z["atac_recon"], atac_target=Z["atac_target"],
        protein_recon=Z["protein_recon"], protein_target=Z["protein_target"],
        z_rna=Z["z_rna"], z_atac=Z["z_atac"], z_protein=Z["z_protein"],
        modality_mask=batch["modality_mask"],
        # NO z_supcon, cell_type_idx, weights default to 0
    )
    assert torch.isfinite(total)
    # 4 base terms + 'total' key
    assert "supcon" not in components
    assert "vicreg_variance" not in components
    assert "masked_rna_recon" in components
    assert "cross_modal_infonce_triad" in components


# --- Active SupCon + VICReg with eligibility filtering ----------------------

def test_dogma_loss_with_supcon_and_vicreg_finite(synth_loader):
    items = [synth_loader[i] for i in range(16)]
    batch = dogma_collate(items)
    B = batch["rna"].shape[0]
    assert "cell_type_idx" in batch
    assert "supcon_eligible_mask" in batch

    Z = _build_synthetic_z_and_recon(B)

    total, components = _dogma_pretrain_loss(
        rna_recon=Z["rna_recon"], rna_target=Z["rna_target"],
        atac_recon=Z["atac_recon"], atac_target=Z["atac_target"],
        protein_recon=Z["protein_recon"], protein_target=Z["protein_target"],
        z_rna=Z["z_rna"], z_atac=Z["z_atac"], z_protein=Z["z_protein"],
        modality_mask=batch["modality_mask"],
        z_supcon=Z["z_supcon"],
        cell_type_idx=batch["cell_type_idx"],
        supcon_eligible_mask=batch["supcon_eligible_mask"],
        w_supcon=0.5,
        w_vicreg=0.1,
        supcon_temperature=0.07,
    )
    assert torch.isfinite(total)
    assert "supcon" in components
    assert "vicreg_variance" in components
    assert all(np.isfinite(components[k]) for k in ("supcon", "vicreg_variance"))


def test_dogma_loss_supcon_gradient_flows_to_z_supcon(synth_loader):
    items = [synth_loader[i] for i in range(16)]
    batch = dogma_collate(items)
    B = batch["rna"].shape[0]

    # Build z_supcon directly as a leaf (skip the through-fusion path so
    # we can directly check gradient on z_supcon)
    torch.manual_seed(1)
    z_supcon = F.normalize(torch.randn(B, 16), dim=-1).detach().requires_grad_(True)
    z_rna = torch.randn(B, 16)
    z_atac = torch.randn(B, 16)
    z_protein = torch.randn(B, 16)
    rna_recon = torch.randn(B, 20)
    rna_target = torch.randn(B, 20)
    atac_recon = torch.randn(B, 5)
    atac_target = torch.randn(B, 5)
    protein_recon = torch.randn(B, 4)
    protein_target = torch.randn(B, 4)

    total, components = _dogma_pretrain_loss(
        rna_recon=rna_recon, rna_target=rna_target,
        atac_recon=atac_recon, atac_target=atac_target,
        protein_recon=protein_recon, protein_target=protein_target,
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=batch["modality_mask"],
        z_supcon=z_supcon,
        cell_type_idx=batch["cell_type_idx"],
        supcon_eligible_mask=batch["supcon_eligible_mask"],
        w_supcon=0.5, w_vicreg=0.1, supcon_temperature=0.07,
        # zero out base terms so we isolate gradient through SupCon+VICReg
        w_rna=0.0, w_atac=0.0, w_protein=0.0, w_triad=0.0,
    )
    assert torch.isfinite(total)
    total.backward()
    assert z_supcon.grad is not None
    assert torch.isfinite(z_supcon.grad).all()
    assert z_supcon.grad.abs().sum() > 0


def test_dogma_loss_supcon_skipped_when_weight_zero(synth_loader):
    """w_supcon=0 → supcon NOT registered → no key in components."""
    items = [synth_loader[i] for i in range(8)]
    batch = dogma_collate(items)
    B = batch["rna"].shape[0]
    Z = _build_synthetic_z_and_recon(B)

    total, components = _dogma_pretrain_loss(
        rna_recon=Z["rna_recon"], rna_target=Z["rna_target"],
        atac_recon=Z["atac_recon"], atac_target=Z["atac_target"],
        protein_recon=Z["protein_recon"], protein_target=Z["protein_target"],
        z_rna=Z["z_rna"], z_atac=Z["z_atac"], z_protein=Z["z_protein"],
        modality_mask=batch["modality_mask"],
        z_supcon=Z["z_supcon"],
        cell_type_idx=batch["cell_type_idx"],
        supcon_eligible_mask=batch["supcon_eligible_mask"],
        w_supcon=0.0,   # disabled
        w_vicreg=0.1,   # vicreg still active
    )
    assert "supcon" not in components
    assert "vicreg_variance" in components


def test_dogma_loss_eligibility_mask_is_threaded(synth_loader):
    """16-cell batch: 5 cells should be ineligible (2 'other' + 1 'other_T'
    + 2 NK with conf=0.55 < 0.6). 11 cells eligible.

    Just verifies eligibility shape + count; the SupCon math itself is
    locked by tests/test_supcon_vicreg.py.
    """
    items = [synth_loader[i] for i in range(16)]
    batch = dogma_collate(items)
    elig = batch["supcon_eligible_mask"]
    assert elig.dtype == torch.bool
    assert elig.shape == (16,)
    # 4 CD4_T + 4 CD8_T + 3 B = 11 eligible. NK fails confidence,
    # other/other_T fail class mask.
    assert int(elig.sum().item()) == 11, (
        f"expected 11 eligible, got {int(elig.sum().item())}"
    )
