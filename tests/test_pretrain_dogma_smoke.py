"""Real_data smoke for DOGMA tri-modal pretrain integration (logical PR #42).

Loads the LLL union h5ad, pulls a small batch, runs forward + backward
through all three encoders, verifies gradients flow.

Gated by @pytest.mark.real_data — opt-in only.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.real_data
@pytest.mark.slow
def test_dogma_trimodal_forward_backward_smoke():
    """End-to-end: LLL union h5ad → ProteinEncoder + ATAC + RNA → grads flow."""
    h5ad_path = Path(os.environ.get(
        "DOGMA_LLL_UNION_H5AD",
        "data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad"))
    if not h5ad_path.exists():
        pytest.skip(f"DOGMA LLL union h5ad missing: {h5ad_path}")

    from aivc.data.multiome_loader import MultiomeLoader
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.protein_encoder import ProteinEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder

    # Default args populate canonical Path A artifact paths.
    ds = MultiomeLoader.make_dogma_lll_union()
    assert ds.n_genes == 36601
    # Union dim from PR #41a production
    assert ds.n_peaks == 323500
    assert ds.n_proteins == 210

    # Pull a small batch (32 cells)
    n = 32
    rna = torch.stack([torch.from_numpy(ds[i]["rna"]) for i in range(n)])
    atac = torch.stack([torch.from_numpy(ds[i]["atac_peaks"]) for i in range(n)])
    prot = torch.stack([torch.from_numpy(ds[i]["protein"]) for i in range(n)])

    assert rna.shape == (n, 36601)
    assert atac.shape == (n, 323500)
    assert prot.shape == (n, 210)

    # Construct encoders (CPU-only for smoke; small dims).
    # NOTE: existing ProteinEncoder uses `embed_dim=` (not `attn_dim=` from
    # the original PR #42 prompt draft). Setting embed_dim=64 for parity with
    # PeakLevelATACEncoder.attn_dim — protein latents need to project to a
    # common dim before the 3-way InfoNCE in _dogma_pretrain_loss.
    rna_enc = SimpleRNAEncoder(n_genes=ds.n_genes, latent_dim=128)
    atac_enc = PeakLevelATACEncoder(n_peaks=ds.n_peaks, attn_dim=64)
    prot_enc = ProteinEncoder(n_proteins=210, embed_dim=64)

    # Forward all three. SimpleRNAEncoder returns (latent, recon) — take latent.
    rna_z, _ = rna_enc(rna.float())
    atac_z = atac_enc(atac.float())
    prot_z = prot_enc(prot.float(), rna_emb=None)

    assert rna_z.shape == (n, 128)
    assert atac_z.shape == (n, 64)
    assert prot_z.shape == (n, 64)
    assert torch.isfinite(rna_z).all()
    assert torch.isfinite(atac_z).all()
    assert torch.isfinite(prot_z).all()

    # Backward through each — proxy loss is sum of latents (cheap).
    loss = rna_z.sum() + atac_z.sum() + prot_z.sum()
    loss.backward()

    # Verify gradients flow to first parameter of each encoder.
    rna_grad = next(rna_enc.parameters()).grad
    atac_grad = next(atac_enc.parameters()).grad
    prot_grad = next(prot_enc.parameters()).grad

    assert rna_grad is not None and rna_grad.abs().sum() > 0
    assert atac_grad is not None and atac_grad.abs().sum() > 0
    assert prot_grad is not None and prot_grad.abs().sum() > 0

    print(f"\nTri-modal smoke OK: rna_z={rna_z.shape}, atac_z={atac_z.shape}, "
          f"prot_z={prot_z.shape}, all grads non-zero")
