"""
MultiomeLoader — paired (RNA, ATAC ± Protein) loader for multiome
observational data (10x PBMC Multiome demo, DOGMA-seq).

Stamps every batch with ``dataset_kind = DatasetKind.OBSERVATIONAL.value``
per Phase 2 convention. Peak harmonization is OUT OF SCOPE here:
supply a pre-computed unified peak set via ``peak_set_path`` (see
scripts/harmonize_peaks.py).

Tri-modal (DOGMA) extension
---------------------------
Protein (ADT) can be loaded via either:
  - ``protein_obsm_key``: loaded from ``adata.obsm[key]`` of the main h5ad
  - ``protein_path``: loaded from a separate h5ad's ``.X`` (must share barcodes)

Exactly one of the two may be set. If neither is set, loader operates
in bimodal RNA+ATAC mode (backward-compatible with existing callers).

All batch-dict string keys are imported from ``aivc.data.modality_mask`` —
NEVER hardcoded. See that module for the canonical contract.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from aivc.data.dataset_kind import DatasetKind
from aivc.data.modality_mask import (
    ModalityKey,
    build_mask,
    RNA_KEY,
    ATAC_KEY,
    PROTEIN_KEY,
    MASK_KEY,
    LYSIS_KEY,
    PROTEIN_PANEL_KEY,
)


_PEAK_SET_ERR = (
    "MultiomeLoader requires a pre-computed unified peak set. "
    "Pass peak_set_path=<path-to-peak-set-artifact>. "
    "Generate this artifact by running scripts/harmonize_peaks.py "
    "(MACS2 on pooled fragments). MACS2 is intentionally NOT run "
    "inside the loader."
)


class MultiomeLoader(Dataset):
    """Load paired RNA + ATAC (+ optional Protein) from a multiome file.

    Parameters
    ----------
    h5ad_path : str
        Path to the multiome file.
    schema : {"mudata", "obsm_atac"}
        - "mudata"    : MuData-style layout with separate modalities
                        keyed under `mod["rna"]` and `mod["atac"]`.
        - "obsm_atac" : AnnData with RNA in `.X` and peaks in
                        `obsm[atac_key]`.
    peak_set_path : str
        REQUIRED. Path to pre-computed unified peak set artifact.
    rna_key : str
        MuData modality key for RNA (default "rna"). Unused for obsm_atac.
    atac_key : str
        MuData modality key OR obsm key for ATAC peak matrix.
    lazy : bool
        If True, defer file reads until first __getitem__.
    protein_obsm_key : Optional[str]
        If set, load protein from ``adata.obsm[key]`` of the main h5ad.
        Mutually exclusive with ``protein_path``.
    protein_path : Optional[str]
        If set, load protein from a separate h5ad's ``.X``. Barcodes
        must match the main h5ad 1:1 (strict shape-equality check).
        Mutually exclusive with ``protein_obsm_key``.
    lysis_protocol : str
        Tag emitted per batch item (DOGMA: "LLL" | "DIG"; else "unknown").
    protein_panel_id : str
        Protein antibody panel identifier (e.g. "totalseq_a_210").
    """

    def __init__(
        self,
        h5ad_path: str,
        schema: str = "obsm_atac",
        peak_set_path: Optional[str] = None,
        rna_key: str = "rna",
        atac_key: str = "atac",
        lazy: bool = False,
        protein_obsm_key: Optional[str] = None,
        protein_path: Optional[str] = None,
        lysis_protocol: str = "unknown",
        protein_panel_id: str = "dogma_adt",
    ):
        if peak_set_path is None:
            raise ValueError(_PEAK_SET_ERR)
        if schema not in ("mudata", "obsm_atac"):
            raise ValueError(
                f"Unknown schema {schema!r}. Expected 'mudata' or 'obsm_atac'."
            )
        if protein_obsm_key is not None and protein_path is not None:
            raise ValueError(
                "Only one of protein_obsm_key / protein_path may be set "
                "(received both)."
            )

        self.h5ad_path = h5ad_path
        self.schema = schema
        self.peak_set_path = peak_set_path
        self.rna_key = rna_key
        self.atac_key = atac_key
        self.protein_obsm_key = protein_obsm_key
        self.protein_path = protein_path
        self.lysis_protocol = lysis_protocol
        self.protein_panel_id = protein_panel_id

        self._rna = None
        self._atac = None
        self._protein = None  # None if bimodal; ndarray if tri-modal

        # Modality presence set — populated during _load
        self._present = {ModalityKey.RNA, ModalityKey.ATAC}

        if not lazy:
            self._load()

    def _load(self):
        if self.schema == "mudata":
            try:
                import mudata as md
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "mudata is required for schema='mudata'. "
                    "Install via `pip install mudata`."
                ) from e
            mdata = md.read_h5mu(self.h5ad_path) if self.h5ad_path.endswith(".h5mu") \
                else md.read(self.h5ad_path)
            rna_ad = mdata.mod[self.rna_key]
            atac_ad = mdata.mod[self.atac_key]
            self._rna = _to_dense(rna_ad.X)
            self._atac = _to_dense(atac_ad.X)
            adata = rna_ad  # for protein_obsm_key lookups below
        else:  # obsm_atac
            import anndata as ad
            adata = ad.read_h5ad(self.h5ad_path)
            self._rna = _to_dense(adata.X)
            if self.atac_key not in adata.obsm:
                raise KeyError(
                    f"atac_key {self.atac_key!r} not found in obsm. "
                    f"Available: {list(adata.obsm.keys())}"
                )
            self._atac = _to_dense(adata.obsm[self.atac_key])

        if self._rna.shape[0] != self._atac.shape[0]:
            raise ValueError(
                f"RNA ({self._rna.shape[0]}) and ATAC ({self._atac.shape[0]}) "
                f"have mismatched cell counts."
            )

        # Protein loading (optional)
        if self.protein_obsm_key is not None:
            if self.protein_obsm_key not in adata.obsm:
                raise KeyError(
                    f"protein_obsm_key {self.protein_obsm_key!r} not found in obsm. "
                    f"Available: {list(adata.obsm.keys())}"
                )
            self._protein = _to_dense(adata.obsm[self.protein_obsm_key])
            if self._protein.shape[0] != self._rna.shape[0]:
                raise ValueError(
                    f"Protein ({self._protein.shape[0]}) and RNA "
                    f"({self._rna.shape[0]}) have mismatched cell counts."
                )
            self._present.add(ModalityKey.PROTEIN)

        elif self.protein_path is not None:
            import anndata as ad
            prot_ad = ad.read_h5ad(self.protein_path)
            self._protein = _to_dense(prot_ad.X)
            if self._protein.shape[0] != self._rna.shape[0]:
                raise ValueError(
                    f"Protein from {self.protein_path} has "
                    f"{self._protein.shape[0]} cells; RNA has "
                    f"{self._rna.shape[0]}. Barcodes must align 1:1."
                )
            self._present.add(ModalityKey.PROTEIN)

    def __len__(self) -> int:
        return 0 if self._rna is None else self._rna.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            RNA_KEY: self._rna[idx],
            ATAC_KEY: self._atac[idx],
            LYSIS_KEY: self.lysis_protocol,
            PROTEIN_PANEL_KEY: self.protein_panel_id,
            "dataset_kind": DatasetKind.OBSERVATIONAL.value,
        }
        if self._protein is not None:
            item[PROTEIN_KEY] = self._protein[idx]
        # (4,) per-cell mask; collate stacks to (batch, 4)
        item[MASK_KEY] = build_mask(self._present, 1).squeeze(0)
        return item

    # ------------------------------------------------------------------
    # Factory methods for Mimitou 2021 DOGMA-seq arms
    # ------------------------------------------------------------------
    @classmethod
    def make_dogma_lll(
        cls,
        base_path: str,
        peak_set_path: str,
        lazy: bool = False,
    ) -> "MultiomeLoader":
        """Factory for DOGMA-LLL arm (Mimitou 2021, GSE156478).

        Expects a pre-aligned h5ad at ``{base_path}/dogma_lll.h5ad`` with:
          - RNA in ``.X``
          - ATAC peaks in ``obsm['atac_peaks']``
          - Protein (TotalSeq-A n=210) in ``obsm['protein']``
        Barcodes must match across all three modalities.
        """
        return cls(
            h5ad_path=os.path.join(base_path, "dogma_lll.h5ad"),
            schema="obsm_atac",
            peak_set_path=peak_set_path,
            atac_key="atac_peaks",
            protein_obsm_key="protein",
            lysis_protocol="LLL",
            protein_panel_id="totalseq_a_210",
            lazy=lazy,
        )

    @classmethod
    def make_dogma_dig(
        cls,
        base_path: str,
        peak_set_path: str,
        lazy: bool = False,
    ) -> "MultiomeLoader":
        """Factory for DOGMA-DIG arm (Mimitou 2021, GSE156478).

        Same structural expectation as make_dogma_lll, with lysis_protocol="DIG".
        """
        return cls(
            h5ad_path=os.path.join(base_path, "dogma_dig.h5ad"),
            schema="obsm_atac",
            peak_set_path=peak_set_path,
            atac_key="atac_peaks",
            protein_obsm_key="protein",
            lysis_protocol="DIG",
            protein_panel_id="totalseq_a_210",
            lazy=lazy,
        )


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)
