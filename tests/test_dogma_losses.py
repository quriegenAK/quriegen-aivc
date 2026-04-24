"""Tests for Day 2 DOGMA losses additions."""
import torch
import torch.nn.functional as F
import pytest

from aivc.data.modality_mask import ModalityKey, build_mask
from aivc.training.pretrain_losses import (
    _masked_rna_recon,
    _masked_atac_recon,
    _masked_protein_recon,
    _cross_modal_infonce,
    _cross_modal_infonce_triad,
    DOGMA_PRETRAIN_TERM_NAMES,
)


def test_protein_recon_no_mask_is_mse():
    torch.manual_seed(0)
    pred = torch.randn(8, 210)
    target = torch.randn(8, 210)
    loss = _masked_protein_recon(protein_recon=pred, protein_target=target)
    expected = F.mse_loss(pred, target)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_protein_recon_silent_skip_all_absent():
    torch.manual_seed(0)
    pred = torch.randn(4, 210)
    target = torch.randn(4, 210)
    mask = build_mask({ModalityKey.RNA, ModalityKey.ATAC}, batch_size=4)
    loss = _masked_protein_recon(
        protein_recon=pred, protein_target=target, modality_mask=mask,
    )
    assert loss.item() == 0.0


def test_protein_recon_only_present_cells_contribute():
    torch.manual_seed(0)
    pred = torch.randn(4, 210)
    target = torch.randn(4, 210)
    mask = torch.tensor([
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    loss = _masked_protein_recon(
        protein_recon=pred, protein_target=target, modality_mask=mask,
    )
    expected = ((pred[:2] - target[:2]) ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-5), \
        f"expected {expected.item()}, got {loss.item()}"


def test_protein_recon_token_mask_combines_with_modality_mask():
    torch.manual_seed(0)
    pred = torch.randn(4, 210)
    target = torch.randn(4, 210)
    mask = build_mask({ModalityKey.RNA, ModalityKey.ATAC, ModalityKey.PROTEIN}, 4)
    token_mask = torch.zeros(4, 210)
    token_mask[:, :100] = 1.0
    loss = _masked_protein_recon(
        protein_recon=pred, protein_target=target,
        modality_mask=mask, protein_mask=token_mask,
    )
    expected = ((pred[:, :100] - target[:, :100]) ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_triad_pairwise_average_all_three_present():
    torch.manual_seed(42)
    B, D = 8, 32
    z_rna = F.normalize(torch.randn(B, D), dim=-1)
    z_atac = F.normalize(torch.randn(B, D), dim=-1)
    z_protein = F.normalize(torch.randn(B, D), dim=-1)
    mask = build_mask({ModalityKey.RNA, ModalityKey.ATAC, ModalityKey.PROTEIN}, B)
    loss = _cross_modal_infonce_triad(
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=mask, infonce_temperature=0.1,
    )
    t = 0.1
    pairs = [(z_rna, z_atac), (z_rna, z_protein), (z_atac, z_protein)]
    nces = []
    for a, b in pairs:
        logits = a @ b.t() / t
        labels = torch.arange(B)
        nces.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)))
    expected = torch.stack(nces).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_triad_bimodal_fallback_protein_absent():
    torch.manual_seed(42)
    B, D = 8, 32
    z_rna = F.normalize(torch.randn(B, D), dim=-1)
    z_atac = F.normalize(torch.randn(B, D), dim=-1)
    mask = build_mask({ModalityKey.RNA, ModalityKey.ATAC}, B)
    loss = _cross_modal_infonce_triad(
        z_rna=z_rna, z_atac=z_atac, modality_mask=mask, infonce_temperature=0.1,
    )
    logits = z_rna @ z_atac.t() / 0.1
    labels = torch.arange(B)
    expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    assert torch.allclose(loss, expected, atol=1e-5)


def test_triad_silent_skip_when_no_pairs():
    torch.manual_seed(0)
    z_rna = F.normalize(torch.randn(4, 16), dim=-1)
    mask = build_mask({ModalityKey.RNA}, 4)
    loss = _cross_modal_infonce_triad(z_rna=z_rna, modality_mask=mask)
    assert loss.item() == 0.0


def test_triad_no_mask_backward_compat():
    torch.manual_seed(7)
    B, D = 6, 16
    z_rna = F.normalize(torch.randn(B, D), dim=-1)
    z_atac = F.normalize(torch.randn(B, D), dim=-1)
    z_protein = F.normalize(torch.randn(B, D), dim=-1)
    loss_no_mask = _cross_modal_infonce_triad(
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein, infonce_temperature=0.1,
    )
    mask_all_on = build_mask({ModalityKey.RNA, ModalityKey.ATAC, ModalityKey.PROTEIN}, B)
    loss_all_on = _cross_modal_infonce_triad(
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=mask_all_on, infonce_temperature=0.1,
    )
    assert torch.allclose(loss_no_mask, loss_all_on, atol=1e-6)


def test_combined_multimodal_backward_compat_no_mask():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from losses import combined_loss_multimodal

    torch.manual_seed(0)
    pred = torch.randn(4, 100)
    stim = torch.randn(4, 100)
    ctrl = torch.randn(4, 100)
    total_no_mask, bd_no = combined_loss_multimodal(pred, stim, ctrl)
    total_none, bd_none = combined_loss_multimodal(pred, stim, ctrl, modality_mask=None)
    assert torch.allclose(total_no_mask, total_none, atol=1e-6)
    assert bd_no["mse"] == bd_none["mse"]
    assert bd_no["lfc"] == bd_none["lfc"]
    assert bd_no["cosine"] == bd_none["cosine"]


def test_combined_multimodal_rna_absent_silent_skip():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from losses import combined_loss_multimodal

    torch.manual_seed(0)
    pred = torch.randn(4, 100)
    stim = torch.randn(4, 100)
    ctrl = torch.randn(4, 100)
    mask = build_mask({ModalityKey.ATAC}, batch_size=4)
    total, bd = combined_loss_multimodal(pred, stim, ctrl, modality_mask=mask)
    assert bd["mse"] == 0.0
    assert bd["lfc"] == 0.0
    assert bd["cosine"] == 0.0


def test_dogma_pretrain_loss_runs_and_backprops():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from losses import _dogma_pretrain_loss

    torch.manual_seed(0)
    B = 8
    rna_recon = torch.randn(B, 500, requires_grad=True)
    rna_target = torch.randn(B, 500)
    atac_recon = torch.randn(B, 1000, requires_grad=True)
    atac_target = torch.randn(B, 1000)
    protein_recon = torch.randn(B, 210, requires_grad=True)
    protein_target = torch.randn(B, 210)
    z_rna = F.normalize(torch.randn(B, 32), dim=-1).requires_grad_(True)
    z_atac = F.normalize(torch.randn(B, 32), dim=-1).requires_grad_(True)
    z_protein = F.normalize(torch.randn(B, 32), dim=-1).requires_grad_(True)
    mask = build_mask({ModalityKey.RNA, ModalityKey.ATAC, ModalityKey.PROTEIN}, B)

    total, bd = _dogma_pretrain_loss(
        rna_recon=rna_recon, rna_target=rna_target,
        atac_recon=atac_recon, atac_target=atac_target,
        protein_recon=protein_recon, protein_target=protein_target,
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=mask,
    )
    for name in DOGMA_PRETRAIN_TERM_NAMES:
        assert name in bd, f"missing term {name} in breakdown"
    assert total.requires_grad
    total.backward()
    assert rna_recon.grad is not None
    assert protein_recon.grad is not None


def test_dogma_pretrain_loss_protein_absent_uses_bimodal():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from losses import _dogma_pretrain_loss

    torch.manual_seed(0)
    B = 6
    total, bd = _dogma_pretrain_loss(
        rna_recon=torch.randn(B, 100, requires_grad=True),
        rna_target=torch.randn(B, 100),
        atac_recon=torch.randn(B, 200, requires_grad=True),
        atac_target=torch.randn(B, 200),
        protein_recon=torch.randn(B, 210, requires_grad=True),
        protein_target=torch.randn(B, 210),
        z_rna=F.normalize(torch.randn(B, 16), dim=-1),
        z_atac=F.normalize(torch.randn(B, 16), dim=-1),
        z_protein=F.normalize(torch.randn(B, 16), dim=-1),
        modality_mask=build_mask({ModalityKey.RNA, ModalityKey.ATAC}, B),
    )
    assert bd["masked_protein_recon"] == 0.0
    assert bd["cross_modal_infonce_triad"] > 0.0


def test_dogma_pretrain_term_names_pass_guard():
    from aivc.training.pretrain_losses import _guard_pretrain_name
    for name in DOGMA_PRETRAIN_TERM_NAMES:
        _guard_pretrain_name(name)
