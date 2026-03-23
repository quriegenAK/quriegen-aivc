"""
Gradient Reversal Layer (GRL) domain adaptation for multi-dataset training.

Prevents the encoder from learning dataset-specific shortcuts when
training on Kang 2018 (PBMCs) + Frangieh 2021 (melanoma) + ImmPort (PBMCs).

Architecture:
  - Shared encoder (existing GATTrainer)
  - Domain classifier head: embed_dim -> 128 -> n_datasets
  - GRL: reverses gradient from domain classifier into shared encoder
  - Encoder learns features that CANNOT distinguish datasets
  - Perturbation decoder learns features that CAN distinguish responses

Loss:
  L_domain = CrossEntropy(predicted_dataset, actual_dataset)
  Total loss += lambda_domain * L_domain (with gradient reversal)

Stage-specific activation:
  Stage 1: lambda_domain = 0.0 (single dataset)
  Stage 2+: lambda_domain ramps from 0.0 to 0.1 over 20 epochs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal layer — reverses gradient sign during backprop."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class MultiDatasetDomainAdaptation(nn.Module):
    """
    Domain adaptation via gradient reversal.

    The domain classifier tries to predict which dataset a cell came from.
    The GRL reverses gradients so the encoder CANNOT encode dataset identity.
    Result: encoder features are dataset-invariant.

    Args:
        n_datasets: Number of datasets (2 for Stage 2, 3 for Stage 3+).
        embed_dim: Input embedding dimension. Default 384 (from fusion).
        hidden_dim: Domain classifier hidden dimension. Default 128.
    """

    def __init__(self, n_datasets: int, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()

        self.n_datasets = n_datasets
        self.grl = GradientReversalLayer()

        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_datasets),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        dataset_ids: torch.Tensor,
        alpha: float = 1.0,
    ) -> tuple:
        """
        Compute domain adversarial loss.

        Args:
            embeddings: (batch, embed_dim) — fused cell state.
            dataset_ids: (batch,) — int dataset labels.
            alpha: GRL scaling factor. Ramp from 0 to 1 over training.

        Returns:
            (domain_loss, domain_accuracy):
              domain_loss: CrossEntropy loss for domain classification.
              domain_accuracy: float, current classifier accuracy.
        """
        self.grl.alpha = alpha

        # Reverse gradient
        reversed_emb = self.grl(embeddings)

        # Classify dataset
        logits = self.domain_classifier(reversed_emb)  # (batch, n_datasets)

        # Loss
        domain_loss = F.cross_entropy(logits, dataset_ids.long())

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == dataset_ids).float().mean().item()

        return domain_loss, accuracy

    @staticmethod
    def compute_lambda_domain(epoch: int, ramp_epochs: int = 20, max_lambda: float = 0.1) -> float:
        """
        Compute domain adaptation weight with ramp schedule.

        Stage 1: lambda = 0.0 (single dataset, disabled)
        Stage 2+: ramps from 0.0 to max_lambda over ramp_epochs

        Args:
            epoch: Current training epoch.
            ramp_epochs: Epochs over which to ramp lambda. Default 20.
            max_lambda: Maximum lambda value. Default 0.1.

        Returns:
            lambda_domain for this epoch.
        """
        if epoch <= 0:
            return 0.0
        progress = min(1.0, epoch / ramp_epochs)
        return max_lambda * progress
