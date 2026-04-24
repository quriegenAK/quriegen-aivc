"""
Canonical contract for multi-modal data flow in AIVC.

All loaders, collate functions, and loss functions MUST import string
keys from this module. No hardcoded string keys anywhere else. This
prevents naming drift (e.g. "rna_target" vs "rna_targets") between
producers and consumers.

Temporal order MUST match TemporalCrossModalFusion.TEMPORAL_ORDER in
aivc/skills/fusion.py:
    ATAC (t=0) -> PHOSPHO (t=1) -> RNA (t=2) -> PROTEIN (t=3)
"""
from __future__ import annotations

from enum import IntEnum
from typing import Set

import torch


class ModalityKey(IntEnum):
    """Temporal-order indices for the 4 AIVC modalities.

    IntEnum values ARE the column indices used in modality_mask tensors.
    Must match TemporalCrossModalFusion.TEMPORAL_ORDER exactly.
    """
    ATAC = 0
    PHOSPHO = 1
    RNA = 2
    PROTEIN = 3


# Canonical temporal order — used by fusion, collate, mask construction.
TEMPORAL_ORDER = [
    ModalityKey.ATAC,
    ModalityKey.PHOSPHO,
    ModalityKey.RNA,
    ModalityKey.PROTEIN,
]


# Canonical batch-dict keys. Import these constants — never hardcode
# the string values in loaders / losses / collate.
RNA_KEY = "rna"
ATAC_KEY = "atac_peaks"
PROTEIN_KEY = "protein"
PHOSPHO_KEY = "phospho"
MASK_KEY = "modality_mask"
LYSIS_KEY = "lysis_protocol"
PROTEIN_PANEL_KEY = "protein_panel_id"


def build_mask(present: Set[ModalityKey], batch_size: int) -> torch.Tensor:
    """Build a (batch_size, 4) modality-presence float mask.

    Args:
        present: set of ModalityKey values present in the batch. All cells
                 in a batch share the same presence set (loader-level
                 assumption; cell-level variation requires a different
                 batch layout than this module supports).
        batch_size: number of cells.

    Returns:
        (batch_size, 4) float32 tensor. Column i is 1.0 if ModalityKey(i)
        is in `present`, else 0.0. Column order = TEMPORAL_ORDER.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    row = torch.zeros(4, dtype=torch.float32)
    for mk in present:
        row[int(mk)] = 1.0
    return row.unsqueeze(0).expand(batch_size, -1).clone()



def mask_from_obs(obs_row) -> torch.Tensor:
    """Build a (4,) modality_mask from boolean has_* columns of an obs row.

    Reads has_rna / has_atac / has_protein / has_phospho from a pandas
    Series, dict, or mapping-like obs row and returns the canonical
    (4,) float mask tensor in TEMPORAL_ORDER. Missing columns default
    to False (modality absent).

    Used by collate / dataset wrappers to derive per-cell modality_mask
    from the obs-column tags set by MultiPerturbationLoader._stamp_modality_tags.
    """
    def _get(key: str) -> bool:
        try:
            v = obs_row[key]
        except (KeyError, TypeError, AttributeError):
            return False
        return bool(v)

    present = set()
    if _get("has_rna"):     present.add(ModalityKey.RNA)
    if _get("has_atac"):    present.add(ModalityKey.ATAC)
    if _get("has_protein"): present.add(ModalityKey.PROTEIN)
    if _get("has_phospho"): present.add(ModalityKey.PHOSPHO)
    return build_mask(present, batch_size=1).squeeze(0)
