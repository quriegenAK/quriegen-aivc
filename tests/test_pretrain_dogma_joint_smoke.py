"""Real-data smoke for joint LLL+DIG training with lysis covariate.

Verifies:
  - Joint loader yields cells from both arms with correct lysis_idx
  - Encoders accept the covariate without crash
  - Forward+backward through all 3 encoders + covariate produces grads

Adapted from PROMPT 43 spec: ProteinEncoder uses ``embed_dim`` (not
``attn_dim`` per prompt draft).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.real_data
@pytest.mark.slow
def test_dogma_joint_forward_backward_with_lysis_covariate():
    lll_h5ad = Path(os.environ.get(
        "DOGMA_LLL_UNION_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad"))
    dig_h5ad = Path(os.environ.get(
        "DOGMA_DIG_UNION_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad"))
    if not lll_h5ad.exists() or not dig_h5ad.exists():
        pytest.skip("DOGMA union h5ads missing")

    from aivc.data.multiome_loader import MultiomeLoader
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.protein_encoder import ProteinEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder

    ds = MultiomeLoader.make_dogma_joint_union()
    assert ds.n_lll > 0 and ds.n_dig > 0
    print(f"\nJoint loader: {len(ds)} cells (LLL={ds.n_lll}, DIG={ds.n_dig})")

    # Sample a mixed batch: some from LLL slice, some from DIG slice
    n = 16
    half = n // 2
    indices = list(range(half)) + list(range(ds.n_lll, ds.n_lll + half))

    rna = torch.stack([torch.from_numpy(ds[i]["rna"]) for i in indices])
    atac = torch.stack([torch.from_numpy(ds[i]["atac_peaks"]) for i in indices])
    prot = torch.stack([torch.from_numpy(ds[i]["protein"]) for i in indices])
    lysis = torch.tensor([ds[i]["lysis_idx"] for i in indices], dtype=torch.long)

    # Verify mixed: 8 zeros + 8 ones
    assert lysis.tolist() == [0] * half + [1] * half

    # Construct covariate-aware encoders (n_lysis_categories=2)
    rna_enc = SimpleRNAEncoder(n_genes=ds.n_genes, latent_dim=32, n_lysis_categories=2)
    atac_enc = PeakLevelATACEncoder(n_peaks=ds.n_peaks, attn_dim=16, n_lysis_categories=2)
    prot_enc = ProteinEncoder(n_proteins=210, embed_dim=16, n_lysis_categories=2)

    # Forward all three with covariate. SimpleRNAEncoder returns (z, recon).
    rna_z, _ = rna_enc(rna.float(), lysis_idx=lysis)
    atac_z = atac_enc(atac.float(), lysis_idx=lysis)
    prot_z = prot_enc(prot.float(), lysis_idx=lysis)

    assert rna_z.shape == (n, 32)
    assert atac_z.shape == (n, 16)
    assert prot_z.shape == (n, 16)
    for z in (rna_z, atac_z, prot_z):
        assert torch.isfinite(z).all()

    # Backward. Use L2 so prot_enc's terminal LayerNorm doesn't zero out
    # the gradient (LayerNorm(x).sum(dim=-1) is identically 0 by mean-
    # centering; .sum() would propagate zero gradient through the affine
    # path and the lysis_emb grad assertions below would falsely pass.)
    loss = rna_z.pow(2).sum() + atac_z.pow(2).sum() + prot_z.pow(2).sum()
    loss.backward()

    # Verify lysis embedding gradients in each encoder
    assert rna_enc.lysis_emb.weight.grad is not None
    assert rna_enc.lysis_emb.weight.grad.abs().sum() > 0
    assert atac_enc.lysis_emb.weight.grad is not None
    assert atac_enc.lysis_emb.weight.grad.abs().sum() > 0
    assert prot_enc.lysis_emb.weight.grad is not None
    assert prot_enc.lysis_emb.weight.grad.abs().sum() > 0

    print(f"Joint covariate smoke OK: rna_z={rna_z.shape}, atac_z={atac_z.shape}, "
          f"prot_z={prot_z.shape}, all lysis_emb grads non-zero")
