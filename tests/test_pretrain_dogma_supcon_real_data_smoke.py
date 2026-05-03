"""PR #54c real-data smoke: load the actual labeled h5ads and run a
single forward+backward with SupCon+VICReg active.

GATED: skipped by default. Set AIVC_RUN_REAL_DATA_SMOKE=1 to enable.

Why gated: loading both DOGMA arms via the joint factory densifies
obsm['atac_peaks'] in memory (current loader behavior at _to_dense),
which means ~41 GB peak RAM. Macs jetsam (Killed: 9). Cluster nodes
with 256+ GB RAM run it cleanly. Run this on BSC compute or a beefy
workstation; the 24 synthetic tests in PR #54a/#54b1/#54b2 cover the
math invariants without needing real data.

Validates the full pipeline against production data shape:
  - obsm['atac_peaks'] sparse CSR loaded correctly
  - obs['cell_type'] -> cell_type_idx via manifest
  - obs['cell_type_confidence'] (DIG only) -> supcon_eligible_mask
  - dogma_collate stacks all keys
  - _dogma_pretrain_loss with SupCon+VICReg active produces finite loss
  - backward gradient flows
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Resolve repo root for `losses` import (it's at the repo root, not under aivc/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

JOINT_H5AD = _REPO_ROOT / "data/phase6_5g_2/dogma_h5ads/dogma_joint_labeled.h5ad"
LLL_H5AD = _REPO_ROOT / "data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad"
DIG_H5AD = _REPO_ROOT / "data/phase6_5g_2/dogma_h5ads/dogma_dig_union_labeled.h5ad"
PEAK_MANIFEST = _REPO_ROOT / "data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json"
CLASS_MANIFEST = _REPO_ROOT / "data/phase6_5g_2/dogma_h5ads/cell_type_index.json"


_REAL_DATA_GATE = os.environ.get("AIVC_RUN_REAL_DATA_SMOKE") == "1"
_FILES_PRESENT = (LLL_H5AD.exists() and DIG_H5AD.exists()
                  and PEAK_MANIFEST.exists() and CLASS_MANIFEST.exists())

pytestmark = pytest.mark.skipif(
    not (_REAL_DATA_GATE and _FILES_PRESENT),
    reason=(
        "Real-data smoke is opt-in. Set AIVC_RUN_REAL_DATA_SMOKE=1 and "
        "ensure labeled h5ads + manifests exist on disk. Beware: peak "
        "RAM on the joint test is ~41 GB; macOS jetsams below ~64 GB."
    ),
)


@pytest.fixture(scope="module")
def real_joint_loader():
    """Module-scoped — load the joint loader once, reuse across both tests.

    This avoids the OOM caused by re-loading both arms in two separate
    fixture instantiations.
    """
    from aivc.data.multiome_loader import MultiomeLoader
    return MultiomeLoader.make_dogma_joint_union(
        use_labeled=True,
        labels_obs_col="cell_type",
        confidence_obs_col="cell_type_confidence",
        masked_classes=["other", "other_T"],
        min_confidence=0.6,
        class_index_manifest=str(CLASS_MANIFEST),
    )


def test_joint_loader_with_supcon_kwargs_loads_real_data(real_joint_loader):
    """Joint factory with use_labeled=True should yield 31874 cells with
    supcon labels."""
    ds = real_joint_loader
    assert len(ds) == 31874, f"expected 31874 joint cells, got {len(ds)}"
    assert ds.lll.n_classes == ds.dig.n_classes
    assert ds.lll.manifest_fingerprint is not None

    # Sample a cell from each arm and verify supcon keys present.
    item_lll = ds[0]
    assert "cell_type_idx" in item_lll
    assert "supcon_eligible" in item_lll
    item_dig = ds[ds.n_lll]  # first DIG cell
    assert item_dig["lysis_idx"] == 1


def test_full_loss_pipeline_on_real_batch(real_joint_loader):
    """Pull 32 cells, run encoders + projection, compute joint loss with
    SupCon+VICReg active, backprop."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from aivc.data.collate import dogma_collate
    from losses import _dogma_pretrain_loss

    ds = real_joint_loader

    # Build a 32-cell batch via dogma_collate. Pick cells across both arms.
    np.random.seed(0)
    sample_idx = np.random.choice(len(ds), size=32, replace=False)
    items = [ds[int(i)] for i in sample_idx]
    batch = dogma_collate(items)
    B = batch["rna"].shape[0]
    assert B == 32
    assert "cell_type_idx" in batch
    assert "supcon_eligible_mask" in batch

    # Synthetic encoders (skip real encoder construction; we just need
    # gradient-bearing tensors of the right shapes for loss verification)
    n_genes = ds.n_genes
    n_peaks = ds.n_peaks
    n_proteins = ds.n_proteins
    proj_dim = 64

    rna_recon_proj = nn.Linear(n_genes, n_genes)
    atac_recon_proj = nn.Linear(n_peaks, n_peaks)
    prot_recon_proj = nn.Linear(n_proteins, n_proteins)
    rna_z_proj = nn.Linear(n_genes, proj_dim)
    atac_z_proj = nn.Linear(n_peaks, proj_dim)
    prot_z_proj = nn.Linear(n_proteins, proj_dim)

    rna_target = batch["rna"].float()
    atac_target = batch["atac_peaks"].float()
    prot_target = batch["protein"].float()

    rna_recon = rna_recon_proj(rna_target)
    atac_recon = atac_recon_proj(atac_target)
    prot_recon = prot_recon_proj(prot_target)

    z_rna = rna_z_proj(rna_target)
    z_atac = atac_z_proj(atac_target)
    z_protein = prot_z_proj(prot_target)

    z_supcon = F.normalize(
        (F.normalize(z_rna, dim=-1)
         + F.normalize(z_atac, dim=-1)
         + F.normalize(z_protein, dim=-1)) / 3.0,
        dim=-1,
    )

    modality_mask = torch.tensor(
        [[1, 0, 1, 1]], dtype=torch.float32
    ).expand(B, 4)

    total, components = _dogma_pretrain_loss(
        rna_recon=rna_recon, rna_target=rna_target,
        atac_recon=atac_recon, atac_target=atac_target,
        protein_recon=prot_recon, protein_target=prot_target,
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=modality_mask,
        z_supcon=z_supcon,
        cell_type_idx=batch["cell_type_idx"],
        supcon_eligible_mask=batch["supcon_eligible_mask"],
        w_supcon=0.5, w_vicreg=0.1, supcon_temperature=0.07,
    )
    assert torch.isfinite(total)
    assert "supcon" in components
    assert "vicreg_variance" in components
    print(f"  components: {components}")

    total.backward()
    # Verify backward by checking at least one parameter has a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for proj in (rna_recon_proj, atac_recon_proj, prot_recon_proj,
                     rna_z_proj, atac_z_proj, prot_z_proj)
        for p in proj.parameters()
    )
    assert has_grad, "no parameter received a gradient"
