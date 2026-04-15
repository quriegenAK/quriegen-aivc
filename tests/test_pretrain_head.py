"""Phase 5 unit tests for MultiomePretrainHead."""
import os
import sys

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aivc.training.pretrain_heads import MultiomePretrainHead
from aivc.skills.neumann_propagation import NeumannPropagation


def _make_head(rna_dim=128, atac_dim=64, n_genes=200):
    return MultiomePretrainHead(
        rna_dim=rna_dim,
        atac_dim=atac_dim,
        proj_dim=64,
        n_genes=n_genes,
        hidden_dim=128,
    )


def test_forward_shapes_and_no_nan():
    torch.manual_seed(0)
    head = _make_head()
    batch = {"rna": torch.randn(4, 200), "atac_peaks": torch.randn(4, 5000)}
    rna_latent = torch.randn(4, 128)
    atac_latent = torch.randn(4, 64)
    out = head(batch, rna_latent, atac_latent)
    assert out["z_rna"].shape == (4, 64)
    assert out["z_atac"].shape == (4, 64)
    assert out["gene_pred"].shape == (4, 200)
    for v in out.values():
        assert not torch.isnan(v).any()


def test_missing_rna_key_raises_value_error():
    head = _make_head()
    batch = {"atac_peaks": torch.randn(4, 5000)}  # no 'rna'
    with pytest.raises(ValueError) as exc:
        head(batch, torch.randn(4, 128), torch.randn(4, 64))
    msg = str(exc.value)
    assert "rna" in msg, f"error message must name missing key 'rna': {msg!r}"


def test_missing_atac_peaks_key_raises_value_error():
    head = _make_head()
    batch = {"rna": torch.randn(4, 200)}  # no 'atac_peaks'
    with pytest.raises(ValueError) as exc:
        head(batch, torch.randn(4, 128), torch.randn(4, 64))
    msg = str(exc.value)
    assert "atac_peaks" in msg, (
        f"error message must name missing key 'atac_peaks': {msg!r}"
    )


def test_graph_independence_from_W():
    """Sibling-style model with NeumannPropagation; backward through the
    head alone must leave NeumannPropagation.W.grad untouched."""
    torch.manual_seed(0)
    n_genes_grn, n_edges = 8, 12
    edge_index = torch.stack(
        [torch.randint(0, n_genes_grn, (n_edges,)),
         torch.randint(0, n_genes_grn, (n_edges,))],
        dim=0,
    )
    edge_attr = torch.rand(n_edges) * 500.0

    class Sibling(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = _make_head(rna_dim=32, atac_dim=16, n_genes=8)
            self.neumann = NeumannPropagation(
                n_genes=n_genes_grn, edge_index=edge_index, edge_attr=edge_attr,
                K=2, lambda_l1=0.001,
            )

    model = Sibling()
    model.zero_grad(set_to_none=True)

    batch = {"rna": torch.randn(4, 8), "atac_peaks": torch.randn(4, 5000)}
    out = model.head(batch, torch.randn(4, 32), torch.randn(4, 16))
    loss = out["z_rna"].pow(2).mean() + out["gene_pred"].pow(2).mean()
    loss.backward()

    assert model.neumann.W.grad is None, (
        "NeumannPropagation.W received gradient from head-only backward — "
        "structural independence violated."
    )
    head_grads = [p.grad for p in model.head.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in head_grads)
