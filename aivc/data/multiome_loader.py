"""
MultiomeLoader — paired (RNA, peak-matrix) loader for multiome
observational data (e.g. 10x PBMC Multiome demo).

Stamps every batch with `dataset_kind = DatasetKind.OBSERVATIONAL.value`
to match the Phase 2 stamping convention. Peak harmonization is
OUT OF SCOPE here: a pre-computed unified peak set must be supplied
via `peak_set_path` (see scripts/harmonize_peaks.py).

Dead code until Phase 5 wires the loader into training.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from aivc.data.dataset_kind import DatasetKind


_PEAK_SET_ERR = (
    "MultiomeLoader requires a pre-computed unified peak set. "
    "Pass peak_set_path=<path-to-peak-set-artifact>. "
    "Generate this artifact by running scripts/harmonize_peaks.py "
    "(MACS2 on pooled fragments). MACS2 is intentionally NOT run "
    "inside the loader."
)


class MultiomeLoader(Dataset):
    """Load paired RNA + peak-matrix from a single .h5ad/MuData file.

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
        REQUIRED. Path to the pre-computed unified peak set artifact.
        Raises ValueError if None.
    rna_key : str
        Modality key for RNA under MuData (default "rna"). Unused for
        obsm_atac schema.
    atac_key : str
        Either the MuData modality key for ATAC (default "atac") or
        the obsm key holding the peak matrix for obsm_atac schema.
    """

    def __init__(
        self,
        h5ad_path: str,
        schema: str = "obsm_atac",
        peak_set_path: Optional[str] = None,
        rna_key: str = "rna",
        atac_key: str = "atac",
        lazy: bool = False,
    ):
        if peak_set_path is None:
            raise ValueError(_PEAK_SET_ERR)
        if schema not in ("mudata", "obsm_atac"):
            raise ValueError(
                f"Unknown schema {schema!r}. Expected 'mudata' or 'obsm_atac'."
            )

        self.h5ad_path = h5ad_path
        self.schema = schema
        self.peak_set_path = peak_set_path
        self.rna_key = rna_key
        self.atac_key = atac_key
        self._rna = None
        self._atac = None

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

    def __len__(self) -> int:
        return 0 if self._rna is None else self._rna.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "rna": self._rna[idx],
            "atac_peaks": self._atac[idx],
            # Match Phase 2 loader stamping convention exactly:
            "dataset_kind": DatasetKind.OBSERVATIONAL.value,
        }


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)
