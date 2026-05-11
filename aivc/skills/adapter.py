"""Stage 3a — frozen-encoder adapter for perturbation discrimination.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §3.1.

The adapter is a small 2-layer projection (Linear → LayerNorm → GELU →
Linear) that maps the frozen DOGMA encoder's latent to a
perturbation-discriminative subspace. It is trained on Mimitou CRISPR
perturbation labels via SupCon, then frozen for Stages 3b and 3c.

Decision rationale (per Report 2 verdict 0.5676 synergy 4-class):
  - 0.5676 is in the ADAPTER_RECOMMENDED range (0.50-0.80)
  - Fine-tune was rejected because it would degrade the validated
    cell-type discrimination (0.7308 centroid-NN)
  - Adapter preserves the frozen encoder's lineage structure while
    adding a perturbation-discriminative head

Architecture invariant: this is NOT a residual layer (no z + adapter(z)
skip). Empirical: raw encoder achieved 0.5676; residual would
re-introduce raw encoder's cell-type-only bias. Pure projection lets
the adapter learn freely. If post-training the adapter ends up close
to identity, that's a learned property, not an imposed constraint.

Parameter count at d=256: 2 * d * d + 2 * d = ~131K. Trains in ~1 day
on Mimitou on a single H100.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PerturbationAdapter(nn.Module):
    """Frozen-encoder → perturbation-discriminative projection.

    Forward path:
        z_in  ── Linear(d, d) ── LayerNorm ── GELU ── Linear(d, d) ── z_out

    All four sublayers initialized via PyTorch defaults. No identity
    initialization — see spec §3.1 for the rationale.

    Args:
        d: latent dimension matching the frozen encoder's output. Default
           256 matches DOGMA SupCon encoder.
        dropout: optional dropout after GELU. Default 0.0 (no dropout for
           the spec'd version; included as a knob for ablations).

    Example:
        >>> adapter = PerturbationAdapter(d=256)
        >>> z = torch.randn(32, 256)         # batch of encoder outputs
        >>> z_proj = adapter(z)              # → (32, 256) projection
        >>> z_proj.shape
        torch.Size([32, 256])

    Checkpoint contract:
        Use save_adapter() / load_adapter() to persist with the canonical
        schema_version=1 envelope; bare torch.load is forbidden by
        tests/test_no_bare_torch_load.py.
    """

    def __init__(self, d: int = 256, dropout: float = 0.0):
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive; got {d}")
        self.d = d
        self.proj_1 = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj_2 = nn.Linear(d, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.d:
            raise ValueError(
                f"adapter input last dim {z.shape[-1]} != d={self.d}"
            )
        return self.proj_2(self.dropout(self.act(self.ln(self.proj_1(z)))))


# --- Checkpoint envelope (schema_version=1) ---

ADAPTER_CKPT_SCHEMA_VERSION = 1


def save_adapter(adapter: PerturbationAdapter, path, extra_meta: Optional[dict] = None) -> None:
    """Save adapter checkpoint with schema_version=1 envelope.

    Layout matches the project's checkpoint contract (aivc/training/ckpt_loader.py):
        {
            "schema_version": 1,
            "kind": "stage3a_adapter",
            "config": {"d": 256, "dropout": 0.0},
            "state_dict": ...,
            "meta": {...optional provenance...},
        }
    """
    payload = {
        "schema_version": ADAPTER_CKPT_SCHEMA_VERSION,
        "kind": "stage3a_adapter",
        "config": {"d": adapter.d},
        "state_dict": adapter.state_dict(),
        "meta": extra_meta or {},
    }
    torch.save(payload, path)


def load_adapter(path, map_location: str = "cpu") -> PerturbationAdapter:
    """Load adapter from a schema_version=1 checkpoint.

    Uses the same conservative loader pattern as
    aivc.training.ckpt_loader (weights_only=False is permissible for
    *trusted* internal checkpoints; we don't load external ones here).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError(f"{path}: not a versioned checkpoint (missing schema_version key)")
    if payload["schema_version"] != ADAPTER_CKPT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {payload['schema_version']} != "
            f"{ADAPTER_CKPT_SCHEMA_VERSION}"
        )
    if payload.get("kind") != "stage3a_adapter":
        raise ValueError(f"{path}: kind={payload.get('kind')!r} != 'stage3a_adapter'")
    cfg = payload["config"]
    adapter = PerturbationAdapter(d=cfg["d"])
    adapter.load_state_dict(payload["state_dict"])
    return adapter
