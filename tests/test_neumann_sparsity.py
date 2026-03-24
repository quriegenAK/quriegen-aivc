"""
Tests for NeumannPropagation sparsity enforcement.
Run: pytest tests/test_neumann_sparsity.py -v
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from aivc.skills.neumann_propagation import NeumannPropagation


def make_mock_neumann(n_genes=50, n_edges=100, K=3, lambda_l1=0.001):
    """Create a small NeumannPropagation module with random edges."""
    torch.manual_seed(42)
    src = torch.randint(0, n_genes, (n_edges,))
    dst = torch.randint(0, n_genes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.rand(n_edges) * 700 + 300  # STRING-like scores 300-1000
    return NeumannPropagation(
        n_genes=n_genes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        K=K,
        lambda_l1=lambda_l1,
    )


class TestEnforceSparsity:

    def test_enforce_sparsity_reduces_density(self):
        """enforce_sparsity must reduce density when near-zero weights exist."""
        neumann = make_mock_neumann(n_genes=50, n_edges=100)
        # Manually inject near-zero values (simulating Adam L1 drift)
        with torch.no_grad():
            neumann.W.data[:50] = 5e-5  # near-zero, should be pruned
            neumann.W.data[50:] = 0.01  # above threshold, should stay
        report = neumann.enforce_sparsity(threshold=1e-4)
        assert report["n_pruned"] == 50, (
            f"Expected 50 edges pruned, got {report['n_pruned']}"
        )
        assert report["density_after"] < report["density_before"], (
            "Density must decrease after sparsity enforcement"
        )

    def test_enforce_sparsity_exact_zeros(self):
        """Pruned edges must be exactly 0.0, not just near-zero."""
        neumann = make_mock_neumann()
        with torch.no_grad():
            neumann.W.data[:] = 5e-5  # all near-zero
        neumann.enforce_sparsity(threshold=1e-4)
        # All pruned values must be exactly 0
        pruned_mask = neumann.W.abs() < 1e-9
        assert pruned_mask.all(), "Pruned edges must be exactly 0.0"

    def test_enforce_sparsity_preserves_large_weights(self):
        """Edges above threshold must not be affected."""
        neumann = make_mock_neumann()
        large_val = 0.05
        with torch.no_grad():
            neumann.W.data[:] = large_val  # all above threshold
        report = neumann.enforce_sparsity(threshold=1e-4)
        assert report["n_pruned"] == 0, (
            f"No edges should be pruned, but {report['n_pruned']} were"
        )
        assert torch.allclose(
            neumann.W.abs(),
            torch.full_like(neumann.W, large_val)
        ), "Large weights must not change"

    def test_density_never_increases_after_enforcement(self):
        """Repeated enforcement calls must be idempotent (non-increasing density)."""
        neumann = make_mock_neumann()
        # Apply multiple rounds
        report1 = neumann.enforce_sparsity(threshold=1e-4)
        report2 = neumann.enforce_sparsity(threshold=1e-4)
        assert report2["density_after"] <= report1["density_after"], (
            "Density must not increase on repeated enforcement"
        )
        assert report2["n_pruned"] == 0, (
            "Second enforcement should prune nothing (already zero)"
        )

    def test_enforce_sparsity_does_not_affect_frozen_W(self):
        """
        When W is frozen (requires_grad=False), enforce_sparsity still works
        but the calling code in train_v11.py skips it via the epoch >= 11 guard.
        Verify enforce_sparsity() itself works regardless of grad state.
        """
        neumann = make_mock_neumann()
        neumann.freeze_W()
        with torch.no_grad():
            neumann.W.data[:50] = 5e-5
        # enforce_sparsity uses torch.no_grad() internally so it works either way
        report = neumann.enforce_sparsity(threshold=1e-4)
        assert report["n_pruned"] == 50

    def test_l1_without_enforcement_does_not_produce_zeros(self):
        """
        Confirm the original bug: L1 + Adam does NOT produce exact zeros.
        This test documents WHY enforce_sparsity is needed.
        """
        torch.manual_seed(42)
        neumann = make_mock_neumann(n_genes=20, n_edges=40, lambda_l1=0.01)
        neumann.unfreeze_W()
        # Simulate training: gradient descent with L1 penalty for 50 steps
        optimizer = torch.optim.Adam([neumann.W], lr=5e-4)
        direct = torch.randn(4, 20)  # mock direct effects
        for _ in range(50):
            optimizer.zero_grad()
            output = neumann(direct)
            loss = output.mean() + neumann.l1_penalty()
            loss.backward()
            optimizer.step()
        # After 50 steps with L1, check density WITHOUT enforcement
        density = neumann.get_effective_W_density()
        # L1 + Adam should NOT produce exact zeros — density stays high
        # This is the bug. density > 0.5 confirms Adam does not zero weights.
        assert density > 0.3, (
            f"Expected density > 0.3 (Adam L1 bug), got {density:.4f}. "
            "If this fails, L1 is unexpectedly working — re-check."
        )


class TestSparsityReport:

    def test_sparsity_report_structure(self):
        """get_sparsity_report must return all expected keys."""
        neumann = make_mock_neumann()
        report = neumann.get_sparsity_report()
        required_keys = [
            "n_edges_total", "n_edges_active", "n_edges_large",
            "density", "density_large", "w_abs_mean", "w_abs_max", "w_abs_min",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_sparsity_report_density_range(self):
        """Density values must be in [0, 1]."""
        neumann = make_mock_neumann()
        report = neumann.get_sparsity_report()
        assert 0.0 <= report["density"] <= 1.0
        assert 0.0 <= report["density_large"] <= 1.0


class TestGetTopEdges:

    def test_get_top_edges_count(self):
        """get_top_edges must return exactly n edges."""
        neumann = make_mock_neumann(n_edges=100)
        edges = neumann.get_top_edges(n=10)
        assert len(edges) == 10

    def test_get_top_edges_sorted_descending(self):
        """Edges must be sorted by abs_weight descending."""
        neumann = make_mock_neumann(n_edges=100)
        edges = neumann.get_top_edges(n=20)
        for i in range(len(edges) - 1):
            assert edges[i]["abs_weight"] >= edges[i + 1]["abs_weight"], (
                "Edges must be sorted by abs_weight descending"
            )

    def test_get_top_edges_with_gene_names(self):
        """Gene names must be attached when provided."""
        n_genes = 50
        neumann = make_mock_neumann(n_genes=n_genes)
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        edges = neumann.get_top_edges(n=5, gene_names=gene_names)
        for edge in edges:
            assert edge["src_name"] is not None
            assert edge["dst_name"] is not None
            assert edge["src_name"].startswith("GENE_")

    def test_get_top_edges_rank_starts_at_1(self):
        """Rank must start at 1."""
        neumann = make_mock_neumann()
        edges = neumann.get_top_edges(n=5)
        assert edges[0]["rank"] == 1
        assert edges[4]["rank"] == 5
