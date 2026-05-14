"""Stage 3 — composite perturbation loss + SupCon for adapter pretraining.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §4.1.

Loss composition for joint adapter + readout training (Stage 3a step 2,
Stage 3b/3c full training):

    L = L_recon
      + λ_zero    * L_zero_arm        # inductive bias for zero-shot synergy
      + λ_pathway * L_pathway         # cross-modality pathway consistency
      + λ_smooth  * L_smooth          # Stage 3b only (trajectory smoothness)

For Stage 3a adapter PRE-training (step 1 of spec §4.2), use supcon_loss()
in isolation — no readout, no decoder, just SupCon over perturbation arms.

Recon target structure (per modality):
  rna_pred:      (B, n_genes)       vs rna_true       (B, n_genes)
  protein_pred:  (B, n_proteins)    vs protein_true   (B, n_proteins)
  phospho_pred:  (B, n_phospho)     vs phospho_true   (B, n_phospho)
Any modality may be None (e.g., Mimitou has no RNA in DOGMA-compatible
form; the loss skips it silently — silent-zero semantics like the
combined_loss_multimodal pattern in losses.py).

Pathway consistency:
  Forces RNA pathway_scores ≈ Protein pathway_scores ≈ Phospho pathway_scores
  (where defined). MSE between each pair, summed over pairs with at
  least one common-pathway score.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- L_recon ---

def l_recon(
    rna_pred: Optional[torch.Tensor] = None,
    rna_true: Optional[torch.Tensor] = None,
    protein_pred: Optional[torch.Tensor] = None,
    protein_true: Optional[torch.Tensor] = None,
    phospho_pred: Optional[torch.Tensor] = None,
    phospho_true: Optional[torch.Tensor] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """Per-modality reconstruction MSE with silent-zero on missing modalities.

    Returns a dict with keys 'rna', 'protein', 'phospho', 'total' (sum of
    present-modality contributions, weighted). Each per-modality value
    is a scalar tensor; missing modalities are absent from the dict.
    """
    weights = weights or {"rna": 1.0, "protein": 1.0, "phospho": 1.0}
    out = {}
    if rna_pred is not None and rna_true is not None:
        out["rna"] = F.mse_loss(rna_pred, rna_true) * weights.get("rna", 1.0)
    if protein_pred is not None and protein_true is not None:
        out["protein"] = F.mse_loss(protein_pred, protein_true) * weights.get("protein", 1.0)
    if phospho_pred is not None and phospho_true is not None:
        out["phospho"] = F.mse_loss(phospho_pred, phospho_true) * weights.get("phospho", 1.0)
    if not out:
        # Defensive: at least one modality must be present
        raise ValueError("l_recon: all modalities are None — no recon signal")
    out["total"] = sum(out.values())
    return out


# --- L_pathway: cross-modality pathway-score consistency ---

def l_pathway_consistency(
    rna_pathway: Optional[torch.Tensor] = None,
    protein_pathway: Optional[torch.Tensor] = None,
    phospho_pathway: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between pairs of modality-level pathway scores. Returns 0 if
    fewer than 2 modalities have pathway scores (no consistency to enforce).
    """
    present = [(name, s) for name, s in
               (("rna", rna_pathway), ("protein", protein_pathway), ("phospho", phospho_pathway))
               if s is not None]
    if len(present) < 2:
        # Return a 0 tensor that participates in autograd (single-element)
        # — caller can still .backward() through it without error.
        device = present[0][1].device if present else torch.device("cpu")
        return torch.zeros((), device=device)
    loss = torch.zeros((), device=present[0][1].device)
    pairs = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            loss = loss + F.mse_loss(present[i][1], present[j][1])
            pairs += 1
    return loss / pairs


# --- L_smooth: trajectory smoothness (Stage 3b only) ---

def l_smooth(
    y_traj: torch.Tensor,  # (B, T, output_dim)
    times: torch.Tensor,    # (T,)
) -> torch.Tensor:
    """Penalize high-frequency oscillation in ŷ(t). Squared first-difference
    normalized by squared timestep, averaged over batch + time.
    """
    if y_traj.dim() != 3 or times.dim() != 1:
        raise ValueError(
            f"l_smooth expects y_traj (B,T,D) + times (T,); got {y_traj.shape}, {times.shape}"
        )
    if y_traj.shape[1] != times.shape[0]:
        raise ValueError(
            f"y_traj T dim {y_traj.shape[1]} != times length {times.shape[0]}"
        )
    if times.shape[0] < 2:
        return torch.zeros((), device=y_traj.device)
    dt = (times[1:] - times[:-1]).clamp(min=1e-6)        # (T-1,)
    dy = (y_traj[:, 1:] - y_traj[:, :-1]) / dt.view(1, -1, 1)  # (B, T-1, D)
    return dy.pow(2).mean()


# --- SupCon (Khosla 2020) for adapter pretraining ---

def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss for the adapter pretraining stage.

    Following Khosla et al. 2020 (NeurIPS):
        L_supcon = -log[ Σ_{p∈P(i)} exp(sim(z_i, z_p) / τ) /
                         Σ_{a∈A(i)} exp(sim(z_i, z_a) / τ) ]
    where P(i) is the set of positives (same label) excluding i itself,
    and A(i) is everything except i.

    Args:
        z: (B, d) — adapter outputs. Will be L2-normalized internally.
        labels: (B,) — integer perturbation labels.
        temperature: scaling. Default 0.07 matches DOGMA SupCon pretraining.

    Returns:
        Scalar loss.

    Notes:
      - Cells with no positives in the batch are dropped (cannot compute
        the positive-set ratio). Caller's batch sampler should ensure ≥2
        cells per arm per batch.
      - τ=0.07 is standard; lower τ produces sharper similarities, higher
        τ smoother.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be (B, d); got shape {z.shape}")
    if labels.dim() != 1 or labels.shape[0] != z.shape[0]:
        raise ValueError(f"labels must be (B,); got shape {labels.shape}")
    if labels.dtype not in (torch.long, torch.int32, torch.int64):
        raise ValueError(
            f"labels must be integer dtype (torch.long / int32 / int64); "
            f"got {labels.dtype}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    z_norm = F.normalize(z, dim=-1)
    B = z.shape[0]
    sim = (z_norm @ z_norm.T) / temperature  # (B, B)

    # Self-mask: exclude i==i from denominator + numerator
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    # Subtract a large value on diagonal so exp(diag) ≈ 0
    sim = sim.masked_fill(self_mask, -1e9)

    # log-softmax across all-but-self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)

    # Positive mask: same-label-but-not-self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # (B, B)
    pos_count = pos_mask.sum(dim=1)                                       # (B,)
    has_positives = pos_count > 0

    if not has_positives.any():
        # No anchor has positives — return 0 loss (caller's sampler must fix)
        return torch.zeros((), device=z.device)

    # Average log_prob over the positive set per anchor
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1)

    # Final loss: mean over anchors that have at least one positive
    return -(pos_log_prob[has_positives]).mean()


# --- Composite PerturbationLoss module ---

class PerturbationLoss(nn.Module):
    """Composite loss for Stage 3a/3b joint training.

    Stage 3a (Mimitou, no temporal):
        rna_pred=None (Mimitou lacks RNA), phospho_pred=None
        lambda_smooth=0.0 (no trajectory)
        Total = lambda_recon * L_recon[protein]
              + lambda_zero  * L_zero_arm
              + lambda_pathway * L_pathway (if pool present)

    Stage 3b (QurieSeq Phase 1 temporal):
        All modalities present, lambda_smooth > 0.

    Args:
        lambda_recon, lambda_zero, lambda_pathway, lambda_smooth:
            scalar weights on each component. Per spec §4.2:
              Stage 3a defaults: 1.0, 1.0, 0.5, 0.0
              Stage 3b defaults: 1.0, 1.0, 1.0, 0.1
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_zero: float = 1.0,
        lambda_pathway: float = 0.5,
        lambda_smooth: float = 0.0,
        recon_weights: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        for name, val in (("lambda_recon", lambda_recon),
                          ("lambda_zero", lambda_zero),
                          ("lambda_pathway", lambda_pathway),
                          ("lambda_smooth", lambda_smooth)):
            if val < 0:
                raise ValueError(f"{name} must be >= 0; got {val}")
        self.lambda_recon = lambda_recon
        self.lambda_zero = lambda_zero
        self.lambda_pathway = lambda_pathway
        self.lambda_smooth = lambda_smooth
        self.recon_weights = recon_weights or {"rna": 1.0, "protein": 1.0, "phospho": 1.0}

    def forward(
        self,
        # Recon targets — any may be None for silent-zero
        rna_pred=None, rna_true=None,
        protein_pred=None, protein_true=None,
        phospho_pred=None, phospho_true=None,
        # Zero-arm head deltas — required when lambda_zero > 0
        deltas: Optional[dict[str, torch.Tensor]] = None,
        arm_mask: Optional[dict[str, torch.Tensor]] = None,
        # Pathway scores per modality — required when lambda_pathway > 0
        rna_pathway=None, protein_pathway=None, phospho_pathway=None,
        # Trajectory + times for smoothness — required when lambda_smooth > 0
        y_traj: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict with named loss components + 'total'. Components
        absent due to None inputs or zero lambda are still in the dict
        (set to a 0 tensor) so logging is shape-stable.
        """
        from aivc.skills.decomposed_readout import zero_arm_loss

        out: dict[str, torch.Tensor] = {}

        # Pick a device anchor — first non-None input we see
        device = self._device_anchor(
            rna_pred, protein_pred, phospho_pred,
            (deltas or {}).get("delta_s"),
        )

        # L_recon
        recon_out = l_recon(
            rna_pred=rna_pred, rna_true=rna_true,
            protein_pred=protein_pred, protein_true=protein_true,
            phospho_pred=phospho_pred, phospho_true=phospho_true,
            weights=self.recon_weights,
        )
        for k in ("rna", "protein", "phospho"):
            out[f"recon_{k}"] = recon_out.get(k, torch.zeros((), device=device))
        out["L_recon"] = recon_out["total"] * self.lambda_recon

        # L_zero_arm — only if we have deltas + arm_mask
        if self.lambda_zero > 0 and deltas is not None and arm_mask is not None:
            l_zero_raw = zero_arm_loss(deltas, arm_mask)
            out["L_zero_arm"] = l_zero_raw * self.lambda_zero
        else:
            out["L_zero_arm"] = torch.zeros((), device=device)

        # L_pathway
        if self.lambda_pathway > 0:
            l_pw_raw = l_pathway_consistency(rna_pathway, protein_pathway, phospho_pathway)
            out["L_pathway"] = l_pw_raw * self.lambda_pathway
        else:
            out["L_pathway"] = torch.zeros((), device=device)

        # L_smooth
        if self.lambda_smooth > 0 and y_traj is not None and times is not None:
            l_sm_raw = l_smooth(y_traj, times)
            out["L_smooth"] = l_sm_raw * self.lambda_smooth
        else:
            out["L_smooth"] = torch.zeros((), device=device)

        out["total"] = out["L_recon"] + out["L_zero_arm"] + out["L_pathway"] + out["L_smooth"]
        return out

    @staticmethod
    def _device_anchor(*tensors) -> torch.device:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                return t.device
        return torch.device("cpu")
