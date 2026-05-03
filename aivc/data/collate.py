"""
Canonical collate function for multi-modal AIVC DataLoader batches.

Uses canonical keys from aivc.data.modality_mask. Handles:
  - RNA / ATAC / Protein tensor stacking (Protein optional)
  - modality_mask stacking to (B, 4)
  - lysis_protocol string -> integer tensor (LLL=0, DIG=1, unknown=-1)
  - protein_panel_id + dataset_kind single-value consistency checks

DD2 policy (strict raise):
  Heterogeneous batches (mixed Protein presence, mixed panel_id, mixed
  dataset_kind) RAISE ValueError. Upstream sampler is responsible for
  ensuring batch-level homogeneity. This complements the D2 loss-level
  silent-zero: per-cell heterogeneity within a batch is a construction
  error (raise); batch-level modality absence is a valid training signal
  (silent-zero at the loss).
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from aivc.data.modality_mask import (
    RNA_KEY,
    ATAC_KEY,
    PROTEIN_KEY,
    MASK_KEY,
    LYSIS_KEY,
    PROTEIN_PANEL_KEY,
)


# Canonical lysis_protocol -> integer code mapping for batch-covariate
# heads (scVI-style categorical injection). The "unknown" sentinel
# supports non-DOGMA datasets that lack a meaningful lysis protocol.
LYSIS_PROTOCOL_CODES: Dict[str, int] = {
    "LLL": 0,
    "DIG": 1,
    "unknown": -1,
}


def lysis_protocol_to_code(s: str) -> int:
    """Map lysis_protocol string to integer code.

    Unknown strings -> -1 (same as the "unknown" sentinel). Policy:
    non-DOGMA datasets pass through as -1 and are ignored by the
    batch-covariate head (which keys on codes 0 and 1).
    """
    return LYSIS_PROTOCOL_CODES.get(s, -1)


def dogma_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate MultiomeLoader items into a batched dict.

    Expected item shape (per MultiomeLoader.__getitem__):
        {
          RNA_KEY: (n_genes,) tensor,
          ATAC_KEY: (n_peaks,) tensor,
          PROTEIN_KEY: (n_ab,) tensor,     # optional
          MASK_KEY: (4,) float tensor,
          LYSIS_KEY: str,
          PROTEIN_PANEL_KEY: str,
          "dataset_kind": str,
        }

    Returns:
        dict with tensors stacked along dim 0 and scalar metadata as
        single values. LYSIS_KEY is converted to an (B,) LongTensor of
        integer codes.

    Raises:
        ValueError on empty batch, mixed Protein presence, mixed
        protein_panel_id, or mixed dataset_kind (DD2 strict-raise).
    """
    if not batch:
        raise ValueError("dogma_collate called on empty batch")

    first = batch[0]
    has_protein = PROTEIN_KEY in first

    # Two-layer design — do not collapse:
    #   Collate (this raise): heterogeneous batches (mixed modality presence
    #   within a batch) are a construction error. Upstream sampler must
    #   enforce batch-level modality homogeneity.
    #   Loss (silent-zero in loss_registry / _dogma_pretrain_loss): a
    #   homogeneous RNA-only batch hitting a trimodal loss term is expected;
    #   the loss returns 0 contribution without raising.
    # If you are writing a mixed-corpus sampler: segregate batches by
    # modality presence upstream; do not attempt to downgrade this raise.
    for i, item in enumerate(batch[1:], start=1):
        if (PROTEIN_KEY in item) != has_protein:
            raise ValueError(
                f"Heterogeneous Protein presence in batch: item 0 "
                f"has_protein={has_protein}, item {i} "
                f"has_protein={PROTEIN_KEY in item}. Upstream sampler "
                f"must segregate batches by modality presence."
            )

    out: Dict[str, Any] = {
        RNA_KEY: torch.stack([torch.as_tensor(it[RNA_KEY]) for it in batch]),
        ATAC_KEY: torch.stack([torch.as_tensor(it[ATAC_KEY]) for it in batch]),
        MASK_KEY: torch.stack([torch.as_tensor(it[MASK_KEY]) for it in batch]),
    }
    if has_protein:
        out[PROTEIN_KEY] = torch.stack(
            [torch.as_tensor(it[PROTEIN_KEY]) for it in batch]
        )

    # Lysis protocol: string -> integer batch-covariate tensor
    out[LYSIS_KEY] = torch.tensor(
        [lysis_protocol_to_code(it[LYSIS_KEY]) for it in batch],
        dtype=torch.long,
    )

    # protein_panel_id: single value per batch; reject heterogeneous
    panel_ids = {it.get(PROTEIN_PANEL_KEY) for it in batch}
    panel_ids.discard(None)
    if len(panel_ids) > 1:
        raise ValueError(
            f"Heterogeneous protein_panel_id in batch: {panel_ids}. "
            f"Panel-translation is not supported at collate time; "
            f"upstream sampler must segregate batches by panel."
        )
    out[PROTEIN_PANEL_KEY] = panel_ids.pop() if panel_ids else None

    # dataset_kind: single value per batch
    kinds = {it.get("dataset_kind") for it in batch}
    kinds.discard(None)
    if len(kinds) > 1:
        raise ValueError(
            f"Heterogeneous dataset_kind in batch: {kinds}. Upstream "
            f"sampler must segregate observational from interventional."
        )
    out["dataset_kind"] = kinds.pop() if kinds else None

    # PR #54b1: SupCon label pass-through. Optional — only stacked when ALL
    # items have the key (homogeneous batch). Mixed presence raises (DD2
    # strict-raise pattern, consistent with Protein presence handling).
    has_label_idx = "cell_type_idx" in first
    for i, item in enumerate(batch[1:], start=1):
        if ("cell_type_idx" in item) != has_label_idx:
            raise ValueError(
                f"Heterogeneous cell_type_idx presence in batch: item 0 "
                f"has={has_label_idx}, item {i} has={'cell_type_idx' in item}. "
                f"Upstream sampler must segregate labeled from unlabeled cells."
            )
    if has_label_idx:
        out["cell_type_idx"] = torch.tensor(
            [int(it["cell_type_idx"]) for it in batch], dtype=torch.long,
        )
        # supcon_eligible defaults to True if individual items don't set it
        out["supcon_eligible_mask"] = torch.tensor(
            [bool(it.get("supcon_eligible", True)) for it in batch], dtype=torch.bool,
        )

    return out
