"""Stage 3 — 4-head decomposed readout for perturbation prediction.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §3.4.

The decomposed readout produces the per-cell prediction at time t as:

    ŷ(t) = h_b(z_dyn, z_static, t)
         + I[stim] · Δ_s(z_dyn, z_static, t, stim_emb)
         + I[inh]  · Δ_i(z_dyn, z_static, t, inh_emb)
         + I[stim ∧ inh] · Δ_xy(z_dyn, z_static, t, stim_emb, inh_emb)

This decomposition is the **load-bearing inductive bias** for zero-shot
synergy. Each head learns one factor of the perturbation response:
  - h_b: vehicle-baseline drift
  - Δ_s: stim-induced delta from vehicle
  - Δ_i: inhibitor-induced delta from vehicle
  - Δ_xy: synergistic correction beyond additive (Δ_s + Δ_i)

The zero-arm constraint (enforced by L_zero_arm loss in training) forces:
  - NTC + vehicle cells:    Δ_s = Δ_i = Δ_xy = 0
  - stim-only cells:        Δ_i = Δ_xy = 0
  - inh-only cells:         Δ_s = Δ_xy = 0
  - stim + inh cells:       no constraint (training signal for Δ_xy)

Without the zero-arm constraint, the 4 heads collapse mathematically
to a single conditional head, and zero-shot synergy prediction fails.

Time encoding: sinusoidal positional encoding of normalized time
(t / T_max). Continuous-time encoding so the readout integrates with
Neural ODE solvers at arbitrary t.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_time_encoding(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Sinusoidal time encoding for continuous-time conditioning.

    Args:
        t:   (B,) or scalar tensor of time values, in arbitrary units.
             For Stage 3 we use minutes (0–180 typical range).
        dim: encoding dimension (even number). Default 16.

    Returns:
        (B, dim) sinusoidal encoding.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even; got {dim}")
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    # Standard transformer-style sinusoidal frequencies, scaled to a typical
    # 0–180 min biological range. Period: 2π * 10000^(2i/dim).
    div = torch.exp(
        torch.arange(0, half, dtype=t.dtype, device=t.device)
        * (-math.log(10000.0) / half)
    )
    args = t.unsqueeze(-1) * div  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP: Linear → GELU → Linear. Used uniformly for all 4 heads."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DecomposedReadout(nn.Module):
    """4-head decomposed readout: baseline + stim Δ + inh Δ + synergy Δ.

    All 4 heads are 2-layer MLPs operating on a concatenation of
    (z_dyn, z_static, t_enc, [stim_emb, inh_emb]). The hidden dim is
    2*d by default.

    Args:
        d:           latent dimension matching z_dyn and z_static (default 256).
        pert_dim:    perturbation embedding dimension (default 32). Both
                     stim_emb and inh_emb have this dim.
        output_dim:  output dim of each head. Default = d (i.e., the readout
                     produces a latent vector that downstream decoders
                     convert to per-modality predictions).
        t_enc_dim:   sinusoidal time encoding dim. Default 16.
        hidden_mult: hidden dim is hidden_mult * d. Default 2.

    Forward signature:
        readout(z_dyn, z_static, t, stim_emb=None, inh_emb=None) → (B, output_dim)

    The stim_emb / inh_emb arguments may be None individually:
        - both None → vehicle baseline only (h_b)
        - stim_emb only → h_b + Δ_s
        - inh_emb only → h_b + Δ_i
        - both provided → h_b + Δ_s + Δ_i + Δ_xy

    For computing the zero-arm loss, see the head_deltas() method which
    returns the four components separately.
    """

    def __init__(
        self,
        d: int = 256,
        pert_dim: int = 32,
        output_dim: Optional[int] = None,
        t_enc_dim: int = 16,
        hidden_mult: int = 2,
    ):
        super().__init__()
        if d <= 0 or pert_dim <= 0:
            raise ValueError(f"d and pert_dim must be positive; got d={d}, pert_dim={pert_dim}")
        if t_enc_dim % 2 != 0:
            raise ValueError(f"t_enc_dim must be even; got {t_enc_dim}")
        self.d = d
        self.pert_dim = pert_dim
        self.t_enc_dim = t_enc_dim
        self.output_dim = output_dim if output_dim is not None else d
        hidden = hidden_mult * d

        base_in = d + d + t_enc_dim                               # z_dyn, z_static, t
        stim_in = base_in + pert_dim                              # + stim
        inh_in  = base_in + pert_dim                              # + inh
        syn_in  = base_in + pert_dim + pert_dim                   # + both

        self.h_b      = _build_mlp(base_in, hidden, self.output_dim)
        self.delta_s  = _build_mlp(stim_in, hidden, self.output_dim)
        self.delta_i  = _build_mlp(inh_in,  hidden, self.output_dim)
        self.delta_xy = _build_mlp(syn_in,  hidden, self.output_dim)

    def _time_enc(self, t: torch.Tensor) -> torch.Tensor:
        return sinusoidal_time_encoding(t, self.t_enc_dim)

    def forward(
        self,
        z_dyn: torch.Tensor,        # (B, d) — dynamic state at time t
        z_static: torch.Tensor,     # (B, d) — static donor context
        t: torch.Tensor,            # (B,) — time scalar per cell
        stim_emb: Optional[torch.Tensor] = None,   # (B, pert_dim) or None
        inh_emb: Optional[torch.Tensor] = None,    # (B, pert_dim) or None
    ) -> torch.Tensor:
        """Compute ŷ(t) with the appropriate active heads.

        I[stim] / I[inh] are determined by whether stim_emb / inh_emb are
        non-None. For mixed batches where some cells have stim and others
        don't, the caller should split the batch (cleaner) or pass a
        zero-padded embedding with a separate mask (more efficient).
        """
        self._validate_shapes(z_dyn, z_static, t, stim_emb, inh_emb)
        t_enc = self._time_enc(t)

        out = self.h_b(torch.cat([z_dyn, z_static, t_enc], dim=-1))
        if stim_emb is not None:
            out = out + self.delta_s(
                torch.cat([z_dyn, z_static, t_enc, stim_emb], dim=-1)
            )
        if inh_emb is not None:
            out = out + self.delta_i(
                torch.cat([z_dyn, z_static, t_enc, inh_emb], dim=-1)
            )
        if stim_emb is not None and inh_emb is not None:
            out = out + self.delta_xy(
                torch.cat([z_dyn, z_static, t_enc, stim_emb, inh_emb], dim=-1)
            )
        return out

    def head_deltas(
        self,
        z_dyn: torch.Tensor,
        z_static: torch.Tensor,
        t: torch.Tensor,
        stim_emb: torch.Tensor,
        inh_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all 4 head outputs separately. Use this in training
        when computing the L_zero_arm loss — pass stim_emb and inh_emb
        even for cells where they should be zero (the loss penalizes
        non-zero deltas in those cases).
        """
        self._validate_shapes(z_dyn, z_static, t, stim_emb, inh_emb)
        t_enc = self._time_enc(t)
        return {
            "h_b":      self.h_b(torch.cat([z_dyn, z_static, t_enc], dim=-1)),
            "delta_s":  self.delta_s(torch.cat([z_dyn, z_static, t_enc, stim_emb], dim=-1)),
            "delta_i":  self.delta_i(torch.cat([z_dyn, z_static, t_enc, inh_emb], dim=-1)),
            "delta_xy": self.delta_xy(torch.cat([z_dyn, z_static, t_enc, stim_emb, inh_emb], dim=-1)),
        }

    def _validate_shapes(
        self, z_dyn, z_static, t, stim_emb, inh_emb,
    ) -> None:
        B = z_dyn.shape[0]
        if z_dyn.shape != (B, self.d):
            raise ValueError(f"z_dyn shape {z_dyn.shape} != (B={B}, d={self.d})")
        if z_static.shape != (B, self.d):
            raise ValueError(f"z_static shape {z_static.shape} != (B={B}, d={self.d})")
        if t.shape != (B,):
            raise ValueError(f"t shape {t.shape} != (B={B},)")
        for name, emb in (("stim_emb", stim_emb), ("inh_emb", inh_emb)):
            if emb is not None and emb.shape != (B, self.pert_dim):
                raise ValueError(
                    f"{name} shape {emb.shape} != (B={B}, pert_dim={self.pert_dim})"
                )


def zero_arm_loss(
    deltas: dict[str, torch.Tensor],
    arm_mask: dict[str, torch.Tensor],
) -> torch.Tensor:
    """L_zero_arm: L2 penalty on inactive head outputs per cell.

    Implements the spec's zero-arm constraint:
      - NTC + vehicle:  ||Δ_s||² + ||Δ_i||² + ||Δ_xy||²
      - stim-only:      ||Δ_i||² + ||Δ_xy||²
      - inh-only:       ||Δ_s||² + ||Δ_xy||²
      - stim + inh:     0 (no constraint)

    Args:
        deltas: output of DecomposedReadout.head_deltas() — must contain
                'delta_s', 'delta_i', 'delta_xy' each shape (B, output_dim).
        arm_mask: dict with keys 'has_stim' (B,) bool, 'has_inh' (B,) bool.

    Returns:
        Scalar loss, averaged over batch dimension.
    """
    has_stim = arm_mask["has_stim"].float()      # (B,)
    has_inh  = arm_mask["has_inh"].float()       # (B,)
    # Penalize Δ_s for cells without stim
    mask_s = (1.0 - has_stim).unsqueeze(-1)       # (B, 1)
    # Penalize Δ_i for cells without inh
    mask_i = (1.0 - has_inh).unsqueeze(-1)
    # Penalize Δ_xy for cells without both perturbations active
    mask_xy = (1.0 - has_stim * has_inh).unsqueeze(-1)

    loss_s  = (deltas["delta_s"]  * mask_s ).pow(2).sum(dim=-1).mean()
    loss_i  = (deltas["delta_i"]  * mask_i ).pow(2).sum(dim=-1).mean()
    loss_xy = (deltas["delta_xy"] * mask_xy).pow(2).sum(dim=-1).mean()
    return loss_s + loss_i + loss_xy
