"""
aivc/skills/multimodal_smoke_test.py — Integration smoke test.

Verifies ProteinEncoder + PhosphoEncoder + ATACSeqEncoder + RNA
all plug into TemporalCrossModalFusion correctly with mock data.

Run: python aivc/skills/multimodal_smoke_test.py
"""
import torch
from aivc.skills.protein_encoder import ProteinEncoder
from aivc.skills.phospho_encoder import PhosphoEncoder
from aivc.skills.atac_encoder import ATACSeqEncoder
from aivc.skills.fusion import TemporalCrossModalFusion
from aivc.skills.contrastive_loss import PairedModalityContrastiveLoss

BATCH = 32
N_PROTEINS = 200
N_SITES = 400
N_TFS = 781
N_GENES = 3010


def check(name, tensor, expected_shape):
    assert tensor.shape == torch.Size(expected_shape), \
        f"FAIL {name}: expected {expected_shape}, got {list(tensor.shape)}"
    assert not tensor.isnan().any(), f"FAIL {name}: contains NaN"
    assert not tensor.isinf().any(), f"FAIL {name}: contains Inf"
    print(f"  ok {name}: {list(tensor.shape)}")


def run():
    print("\nAIVC Multi-Modal Integration Smoke Test")
    print("=" * 45)

    torch.manual_seed(42)

    adt_raw = torch.rand(BATCH, N_PROTEINS).abs() * 100
    phospho_raw = torch.rand(BATCH, N_SITES).abs() * 1e5
    tf_scores = torch.randn(BATCH, N_TFS)
    mock_edges = torch.randint(0, N_SITES, (2, 500))
    rna_emb = torch.randn(BATCH, 128)

    check("RNA encoder (mock)", rna_emb, [BATCH, 128])

    # ProteinEncoder
    prot_enc = ProteinEncoder(n_proteins=N_PROTEINS)
    prot_emb = prot_enc(adt_raw, rna_emb=rna_emb)
    check("Protein encoder", prot_emb, [BATCH, 128])

    prot_emb_no_rna = prot_enc(adt_raw, rna_emb=None)
    check("Protein encoder (no RNA)", prot_emb_no_rna, [BATCH, 128])

    # PhosphoEncoder
    phos_enc = PhosphoEncoder(n_sites=N_SITES)
    phos_emb = phos_enc(phospho_raw, edge_index=mock_edges)
    check("Phospho encoder", phos_emb, [BATCH, 64])

    phos_emb_no_graph = phos_enc(phospho_raw, edge_index=None)
    check("Phospho encoder (no graph)", phos_emb_no_graph, [BATCH, 64])

    # ATACSeqEncoder
    atac_enc = ATACSeqEncoder(n_tfs=N_TFS)
    atac_emb = atac_enc(tf_scores)
    check("ATAC encoder", atac_emb, [BATCH, 64])

    # Fusion
    fusion = TemporalCrossModalFusion()
    fused = fusion(
        rna_emb=rna_emb,
        protein_emb=prot_emb,
        phospho_emb=phos_emb,
        atac_emb=atac_emb,
    )
    check("Fusion output (all 4 modalities)", fused, [BATCH, 384])

    # Causal mask
    attn = fusion.get_attention_weights(rna_emb, prot_emb, phos_emb, atac_emb)
    upper = torch.triu(attn.mean(0).mean(0), diagonal=1)
    assert upper.abs().max().item() < 1e-5, \
        f"FAIL causal mask: upper triangle not zero ({upper.abs().max().item():.6f})"
    print(f"  ok Causal mask: upper triangle = {upper.abs().max().item():.2e}")

    # Contrastive loss
    cl = PairedModalityContrastiveLoss(temperature=0.07)
    cl_loss = cl(rna_emb, prot_emb)
    assert not cl_loss.isnan().any(), "FAIL contrastive loss: NaN"
    print(f"  ok Contrastive loss (RNA<>Protein): {cl_loss.item():.4f}")

    print("\n  " + "=" * 43)
    print("  All checks passed. Platform ready for QuRIE-seq data.")
    print()


if __name__ == "__main__":
    run()
