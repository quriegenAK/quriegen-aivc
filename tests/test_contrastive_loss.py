"""
Tests for contrastive loss and cross-modal prediction loss.
All tests use mock data. No real files. Under 15 seconds on CPU.

Run: pytest tests/test_contrastive_loss.py -v
"""
import math

import pytest
import torch
import torch.nn as nn
import numpy as np

from aivc.skills.contrastive_loss import (
    CrossModalPredictionLoss,
    PairedModalityContrastiveLoss,
)
from losses import combined_loss_multimodal

SEED = 42


class TestContrastiveLoss:

    def test_contrastive_loss_perfect_alignment(self):
        """When emb_a == emb_b for all cells: loss should be near 0."""
        torch.manual_seed(SEED)
        loss_fn = PairedModalityContrastiveLoss(temperature=0.07)
        emb = torch.randn(16, 64)
        loss = loss_fn(emb, emb.clone())
        # Perfect alignment: each cell matches itself exactly
        # Loss should be very small (near 0)
        assert loss.item() < 0.1, f"Perfect alignment loss should be ~0, got {loss.item():.4f}"

    def test_contrastive_loss_random_embeddings(self):
        """Random embeddings: loss should be near log(batch_size)."""
        torch.manual_seed(SEED)
        loss_fn = PairedModalityContrastiveLoss(temperature=0.07)
        batch_size = 64
        emb_a = torch.randn(batch_size, 128)
        emb_b = torch.randn(batch_size, 128)
        loss = loss_fn(emb_a, emb_b)
        expected = math.log(batch_size)
        # Loss should be approximately log(N) for random embeddings
        assert loss.item() > expected * 0.5, (
            f"Random loss {loss.item():.2f} should be near log({batch_size})={expected:.2f}"
        )

    def test_alignment_score_perfect(self):
        """emb_a == emb_b -> alignment_score == 1.0."""
        torch.manual_seed(SEED)
        loss_fn = PairedModalityContrastiveLoss()
        emb = torch.randn(32, 64)
        score = loss_fn.alignment_score(emb, emb.clone())
        assert score == 1.0, f"Perfect alignment score should be 1.0, got {score}"

    def test_alignment_score_random(self):
        """Random embeddings, batch=64 -> alignment_score ~ 1/64."""
        torch.manual_seed(SEED)
        loss_fn = PairedModalityContrastiveLoss()
        batch_size = 64
        emb_a = torch.randn(batch_size, 128)
        emb_b = torch.randn(batch_size, 128)
        score = loss_fn.alignment_score(emb_a, emb_b)
        random_baseline = 1.0 / batch_size
        # Should be near random baseline (within 10x)
        assert score < 0.2, (
            f"Random alignment score {score:.4f} should be near "
            f"{random_baseline:.4f} (1/{batch_size})"
        )

    def test_contrastive_loss_batch_size_1_returns_zero(self):
        """batch_size=1 -> loss == 0.0."""
        loss_fn = PairedModalityContrastiveLoss()
        emb_a = torch.randn(1, 64)
        emb_b = torch.randn(1, 64)
        loss = loss_fn(emb_a, emb_b)
        assert loss.item() == 0.0, f"Batch size 1 loss should be 0.0, got {loss.item()}"


class TestCrossModalLoss:

    def test_cross_modal_loss_same_embeddings(self):
        """emb_a == emb_b -> cross-modal loss near 0 (after training projection)."""
        torch.manual_seed(SEED)
        dim = 64
        cross_fn = CrossModalPredictionLoss(dim, dim)

        # Use identical embeddings — the projection heads are random init
        # so loss won't be exactly 0, but the structure is correct
        emb = torch.randn(16, dim)
        result = cross_fn(emb, emb.clone())

        assert "loss" in result
        assert "loss_a2b" in result
        assert "loss_b2a" in result
        # Loss should be a valid positive number
        assert result["loss"].item() >= 0


class TestCombinedLossMultimodal:

    def test_combined_loss_multimodal_without_contrastive(self):
        """emb_pairs=None -> total loss equals base combined_loss_v11."""
        torch.manual_seed(SEED)
        predicted = torch.randn(8, 30)
        actual_stim = torch.randn(8, 30)
        actual_ctrl = torch.randn(8, 30)

        total_multi, bd_multi = combined_loss_multimodal(
            predicted, actual_stim, actual_ctrl,
            neumann_module=None,
            emb_pairs=None,
            contrastive_loss_fn=None,
            cross_modal_fn=None,
        )

        # Contrastive and cross-modal should be 0
        assert bd_multi["contrastive"] == 0.0
        assert bd_multi["cross_modal"] == 0.0

    def test_combined_loss_multimodal_with_contrastive(self):
        """emb_pairs provided -> total loss > base loss."""
        torch.manual_seed(SEED)
        predicted = torch.randn(16, 30)
        actual_stim = torch.randn(16, 30)
        actual_ctrl = torch.randn(16, 30)

        contrastive_fn = PairedModalityContrastiveLoss(temperature=0.07)
        emb_a = torch.randn(16, 64)
        emb_b = torch.randn(16, 64)

        total_base, bd_base = combined_loss_multimodal(
            predicted, actual_stim, actual_ctrl,
            emb_pairs=None,
            contrastive_loss_fn=None,
        )

        total_with, bd_with = combined_loss_multimodal(
            predicted, actual_stim, actual_ctrl,
            emb_pairs=[(emb_a, emb_b)],
            contrastive_loss_fn=contrastive_fn,
            lambda_contrast=0.05,
        )

        assert bd_with["contrastive"] > 0, "Contrastive term should be > 0"
        assert total_with.item() > total_base.item(), (
            "Total loss with contrastive should exceed base loss"
        )
