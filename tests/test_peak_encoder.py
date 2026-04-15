"""Phase 4 unit tests for PeakLevelATACEncoder."""
import os
import sys

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
from aivc.skills.neumann_propagation import NeumannPropagation


def test_forward_shape_and_variance():
    torch.manual_seed(0)
    enc = PeakLevelATACEncoder(n_peaks=5000, attn_dim=64)
    # Sparse-ish count matrix: ~2% nonzero, integer counts.
    x = (torch.rand(32, 5000) < 0.02).float() * torch.randint(1, 5, (32, 5000)).float()
    z = enc(x)
    assert z.shape == (32, 64)
    assert not torch.isnan(z).any()
    # LSI-collapse guard: latent must carry variance.
    assert z.var().item() > 1e-4, f"latent variance collapsed: {z.var().item()}"


def test_no_batchnorm_in_module_tree():
    enc = PeakLevelATACEncoder(n_peaks=512, attn_dim=32)
    for m in enc.modules():
        assert not isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        ), f"BatchNorm leaked into encoder: {type(m).__name__}"


def test_graph_independence_from_W():
    """Mirrors the Phase 3 counterfactual pattern: encoder and
    NeumannPropagation are unrelated siblings. A backward() through the
    encoder alone must leave NeumannPropagation.W.grad untouched."""
    torch.manual_seed(0)
    n_genes, n_edges, n_peaks = 8, 12, 256

    edge_index = torch.stack(
        [torch.randint(0, n_genes, (n_edges,)),
         torch.randint(0, n_genes, (n_edges,))],
        dim=0,
    )
    edge_attr = torch.rand(n_edges) * 500.0

    class Sibling(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = PeakLevelATACEncoder(n_peaks=n_peaks, attn_dim=16)
            self.neumann = NeumannPropagation(
                n_genes=n_genes, edge_index=edge_index, edge_attr=edge_attr,
                K=2, lambda_l1=0.001,
            )

    model = Sibling()
    model.zero_grad(set_to_none=True)

    x = torch.rand(4, n_peaks)
    z = model.encoder(x)
    loss = z.pow(2).mean()
    loss.backward()

    assert model.neumann.W.grad is None, (
        "NeumannPropagation.W received gradient from encoder-only backward — "
        "structural independence violated."
    )
    # Encoder params did receive gradient.
    enc_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in enc_grads)
