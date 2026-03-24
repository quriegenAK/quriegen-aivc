"""
Paired contrastive loss for physically co-measured modalities.

ONLY valid when modalities are physically paired (same cell, same lysis).
Applying this to computationally paired data actively misleads the encoder.

InfoNCE objective: same-cell embeddings from different modalities
should be closer than cross-cell embeddings.

Reference: Chen et al. 2020 (SimCLR), Radford et al. 2021 (CLIP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedModalityContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for physically paired modality embeddings.

    Same-cell (i, i) pairs are positives.
    Cross-cell (i, j!=i) pairs are negatives.
    Both directions computed (A->B and B->A) and averaged.

    Args:
        temperature:   float -- softmax temperature tau. Default 0.07.
        learnable_tau: bool -- if True, tau is a learnable parameter.
    """

    def __init__(self, temperature: float = 0.07, learnable_tau: bool = False):
        super().__init__()
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_tau", torch.tensor(temperature).log())
        self.learnable_tau = learnable_tau

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional InfoNCE loss.

        Args:
            emb_a: (batch, dim) -- embeddings from modality A
            emb_b: (batch, dim) -- embeddings from modality B
                   emb_a[i] and emb_b[i] are from the same physical cell.

        Returns:
            Scalar loss -- average of A->B and B->A InfoNCE losses.
        """
        batch_size = emb_a.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=emb_a.device, requires_grad=True)

        z_a = F.normalize(emb_a, dim=-1)
        z_b = F.normalize(emb_b, dim=-1)

        sim = torch.matmul(z_a, z_b.T) / self.temperature
        labels = torch.arange(batch_size, device=emb_a.device)

        loss_a2b = F.cross_entropy(sim, labels)
        loss_b2a = F.cross_entropy(sim.T, labels)

        return (loss_a2b + loss_b2a) / 2.0

    def alignment_score(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """
        Fraction of cells where same-cell embedding is the nearest neighbour.
        1.0 = perfect, 1/N = random baseline.
        """
        with torch.no_grad():
            z_a = F.normalize(emb_a, dim=-1)
            z_b = F.normalize(emb_b, dim=-1)
            sim = torch.matmul(z_a, z_b.T)
            predicted = sim.argmax(dim=1)
            correct = torch.arange(len(emb_a), device=emb_a.device)
            return (predicted == correct).float().mean().item()


class CrossModalPredictionLoss(nn.Module):
    """
    Cross-modal prediction: predict modality B from A (and vice versa).
    Complementary to contrastive loss -- learns directional A->B relationships.

    Args:
        dim_a: int -- dimension of modality A embedding
        dim_b: int -- dimension of modality B embedding
    """

    def __init__(self, dim_a: int, dim_b: int):
        super().__init__()
        self.proj_a2b = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.GELU(),
            nn.Linear(dim_a, dim_b),
        )
        self.proj_b2a = nn.Sequential(
            nn.Linear(dim_b, dim_b),
            nn.GELU(),
            nn.Linear(dim_b, dim_a),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> dict:
        """
        Compute cross-modal prediction loss in both directions.

        Returns dict with 'loss', 'loss_a2b', 'loss_b2a'.
        """
        pred_b = self.proj_a2b(emb_a)
        pred_a = self.proj_b2a(emb_b)

        loss_a2b = (1.0 - F.cosine_similarity(pred_b, emb_b, dim=-1)).mean()
        loss_b2a = (1.0 - F.cosine_similarity(pred_a, emb_a, dim=-1)).mean()

        return {
            "loss": (loss_a2b + loss_b2a) / 2.0,
            "loss_a2b": loss_a2b,
            "loss_b2a": loss_b2a,
        }
