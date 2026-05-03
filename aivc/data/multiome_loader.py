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
        # PR #54b1: optional supervised-label loading for SupCon.
        # If labels_obs_col is None, no label loading happens — preserves
        # backward compat with all existing factory callers.
        labels_obs_col: Optional[str] = None,
        confidence_obs_col: Optional[str] = None,
        masked_classes: Optional[list] = None,
        min_confidence: float = 0.0,
        class_index_manifest: Optional[str] = None,
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

        # PR #54b1: SupCon label state. None if labels_obs_col not set.
        self.labels_obs_col = labels_obs_col
        self.confidence_obs_col = confidence_obs_col
        self.masked_classes = set(masked_classes) if masked_classes else set()
        self.min_confidence = float(min_confidence)
        self.class_index_manifest_path = class_index_manifest
        self._cell_type_idx = None      # (N,) np.int64 or None
        self._supcon_eligible = None    # (N,) np.bool_ or None
        self._class_to_idx = None       # dict[str, int] or None
        self._manifest_fingerprint = None  # SHA-256 from manifest, or None

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

        # PR #54b1: optional supervised-label loading. Only fires when
        # labels_obs_col is configured. The h5ad reload is wasteful but
        # avoids an additional path through the schema branches above —
        # acceptable for the small obs read.
        if self.labels_obs_col is not None:
            import anndata
            adata_for_labels = anndata.read_h5ad(self.h5ad_path)
            self._load_supcon_labels(adata_for_labels)

    def _load_supcon_labels(self, adata) -> None:
        """Read cell_type + cell_type_confidence from obs and build SupCon
        eligibility mask. Called from _load() when labels_obs_col is set."""
        import json
        import numpy as np

        if self.labels_obs_col not in adata.obs.columns:
            raise ValueError(
                f"labels_obs_col {self.labels_obs_col!r} not in obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        # --- Class index manifest (canonical {class_name: int}) ---
        if self.class_index_manifest_path is None:
            # In-memory fallback for tests: build from this h5ad alone.
            classes = sorted(adata.obs[self.labels_obs_col].astype(str).unique())
            self._class_to_idx = {cls: i for i, cls in enumerate(classes)}
            self._manifest_fingerprint = None
        else:
            mp = self.class_index_manifest_path
            with open(mp) as f:
                manifest = json.load(f)
            self._class_to_idx = dict(manifest["class_to_idx"])
            self._manifest_fingerprint = manifest.get("fingerprint_sha256")

        # --- Map per-cell label to integer ---
        labels = adata.obs[self.labels_obs_col].astype(str).values
        unknown_classes = set(labels) - set(self._class_to_idx)
        if unknown_classes:
            raise ValueError(
                f"Cells in {self.h5ad_path} have classes {sorted(unknown_classes)} "
                f"absent from class_index_manifest. Either rebuild the manifest "
                f"to cover both arms or check the labeled h5ad."
            )
        self._cell_type_idx = np.array(
            [self._class_to_idx[c] for c in labels], dtype=np.int64
        )

        # --- Build supcon_eligible_mask ---
        eligible = np.ones(len(labels), dtype=np.bool_)
        if self.masked_classes:
            class_mask_set = set(self.masked_classes)
            eligible &= ~np.isin(labels, list(class_mask_set))
        if self.min_confidence > 0.0:
            if self.confidence_obs_col is None:
                # User asked for a confidence threshold but didn't tell us
                # which column. Treat all cells as max-confidence (1.0).
                conf = np.ones(len(labels), dtype=np.float32)
            elif self.confidence_obs_col not in adata.obs.columns:
                # Column requested but not present in this h5ad. Common
                # case for the LLL arm of DOGMA: published Azimuth labels
                # are gold-standard so no per-cell confidence was stamped;
                # only DIG's kNN-transferred labels carry confidence. Fall
                # back to max-confidence (no cells dropped on this arm).
                # This is a graceful no-op rather than a hard error so the
                # joint factory can pass the same kwargs to both arms.
                import sys as _sys
                print(
                    f"  [supcon] confidence_obs_col {self.confidence_obs_col!r} "
                    f"not present in {self.h5ad_path}; treating all cells as "
                    f"max confidence (1.0). This is the expected path for the "
                    f"LLL arm of DOGMA.",
                    file=_sys.stderr,
                )
                conf = np.ones(len(labels), dtype=np.float32)
            else:
                conf = adata.obs[self.confidence_obs_col].fillna(1.0).values.astype(np.float32)
            eligible &= (conf >= self.min_confidence)
        self._supcon_eligible = eligible

    def __len__(self) -> int:
        return 0 if self._rna is None else self._rna.shape[0]

    # ------------------------------------------------------------------
    # PR #42 (logical): public shape accessors for downstream encoder
    # construction. Keeps `_rna`/`_atac`/`_protein` private while letting
    # callers size encoders without poking at internals.
    # ------------------------------------------------------------------
    @property
    def n_genes(self) -> int:
        if self._rna is None:
            raise RuntimeError("MultiomeLoader._rna not loaded yet (lazy=True?)")
        return int(self._rna.shape[1])

    @property
    def n_peaks(self) -> int:
        if self._atac is None:
            raise RuntimeError("MultiomeLoader._atac not loaded yet (lazy=True?)")
        return int(self._atac.shape[1])

    @property
    def n_proteins(self) -> int | None:
        """None if protein modality absent."""
        return None if self._protein is None else int(self._protein.shape[1])

    # PR #54b1: SupCon label accessors.
    @property
    def n_classes(self) -> int | None:
        """Number of distinct cell_type classes, or None if labels not loaded."""
        return None if self._class_to_idx is None else len(self._class_to_idx)

    @property
    def class_to_idx(self) -> dict | None:
        """Class-name → integer mapping, or None if labels not loaded."""
        return self._class_to_idx if self._class_to_idx is None else dict(self._class_to_idx)

    @property
    def manifest_fingerprint(self) -> str | None:
        """SHA-256 of the class_index_manifest, or None if not loaded from a manifest."""
        return self._manifest_fingerprint

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
        # PR #54b1: supervised labels (only present when labels_obs_col was set)
        if self._cell_type_idx is not None:
            item["cell_type_idx"] = int(self._cell_type_idx[idx])
        if self._supcon_eligible is not None:
            item["supcon_eligible"] = bool(self._supcon_eligible[idx])
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

    # ------------------------------------------------------------------
    # PR #41a: union-peak-space factories for joint LLL+DIG training
    # ------------------------------------------------------------------
    @classmethod
    def make_dogma_lll_union(
        cls,
        base_path: str = "data/phase6_5g_2/dogma_h5ads",
        peak_set_path: str = "data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json",
        lazy: bool = False,
        # PR #54c: use the labeled h5ad and thread SupCon kwargs through.
        use_labeled: bool = False,
        labels_obs_col: Optional[str] = None,
        confidence_obs_col: Optional[str] = None,
        masked_classes: Optional[list] = None,
        min_confidence: float = 0.0,
        class_index_manifest: Optional[str] = None,
    ) -> "MultiomeLoader":
        """Factory for DOGMA-LLL arm in union peak space (PR #41a).

        Use this for joint LLL+DIG training where both arms must share
        the encoder's input dim. Arm-unique peaks are zero-filled in
        the other arm's cells.

        File selection: ``dogma_lll_union.h5ad`` by default; flip
        ``use_labeled=True`` to load ``dogma_lll_union_labeled.h5ad``
        which includes obs['cell_type'] for SupCon training (PR #54c).
        """
        filename = "dogma_lll_union_labeled.h5ad" if use_labeled else "dogma_lll_union.h5ad"
        return cls(
            h5ad_path=os.path.join(base_path, filename),
            schema="obsm_atac",
            peak_set_path=peak_set_path,
            atac_key="atac_peaks",
            protein_obsm_key="protein",
            lysis_protocol="LLL",
            protein_panel_id="totalseq_a_210",
            lazy=lazy,
            labels_obs_col=labels_obs_col,
            confidence_obs_col=confidence_obs_col,
            masked_classes=masked_classes,
            min_confidence=min_confidence,
            class_index_manifest=class_index_manifest,
        )

    @classmethod
    def make_dogma_dig_union(
        cls,
        base_path: str = "data/phase6_5g_2/dogma_h5ads",
        peak_set_path: str = "data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json",
        lazy: bool = False,
        # PR #54c: parallel to make_dogma_lll_union — same supcon plumbing.
        use_labeled: bool = False,
        labels_obs_col: Optional[str] = None,
        confidence_obs_col: Optional[str] = None,
        masked_classes: Optional[list] = None,
        min_confidence: float = 0.0,
        class_index_manifest: Optional[str] = None,
    ) -> "MultiomeLoader":
        """Factory for DOGMA-DIG arm in union peak space (PR #41a).

        See make_dogma_lll_union. ``use_labeled=True`` switches to
        ``dogma_dig_union_labeled.h5ad`` for SupCon training (PR #54c).
        """
        filename = "dogma_dig_union_labeled.h5ad" if use_labeled else "dogma_dig_union.h5ad"
        return cls(
            h5ad_path=os.path.join(base_path, filename),
            schema="obsm_atac",
            peak_set_path=peak_set_path,
            atac_key="atac_peaks",
            protein_obsm_key="protein",
            lysis_protocol="DIG",
            protein_panel_id="totalseq_a_210",
            lazy=lazy,
            labels_obs_col=labels_obs_col,
            confidence_obs_col=confidence_obs_col,
            masked_classes=masked_classes,
            min_confidence=min_confidence,
            class_index_manifest=class_index_manifest,
        )

    # ------------------------------------------------------------------
    # PR #43 (logical): joint LLL+DIG factory. See DogmaJointLoader below.
    # ------------------------------------------------------------------
    @classmethod
    def make_dogma_joint_union(
        cls,
        base_path: str = "data/phase6_5g_2/dogma_h5ads",
        peak_set_path: str = "data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json",
        # PR #54c: thread SupCon kwargs to both child arm factories.
        use_labeled: bool = False,
        labels_obs_col: Optional[str] = None,
        confidence_obs_col: Optional[str] = None,
        masked_classes: Optional[list] = None,
        min_confidence: float = 0.0,
        class_index_manifest: Optional[str] = None,
    ) -> "DogmaJointLoader":
        """Construct a joint LLL+DIG loader over the union peak space.

        Cells are concatenated; ``lysis_idx`` is stamped per cell
        (LLL=0, DIG=1 — matching ``aivc.data.collate.LYSIS_PROTOCOL_CODES``).
        Use ``--arm joint`` in scripts/pretrain_multiome.py to invoke.

        ``use_labeled=True`` + supcon kwargs propagate to both child arms.
        Both arms MUST be either-labeled-or-not (the alternative would be
        an asymmetric label space which DogmaJointLoader cannot represent
        cleanly). The kwargs are threaded as-is.
        """
        kwargs = dict(
            base_path=base_path, peak_set_path=peak_set_path,
            use_labeled=use_labeled,
            labels_obs_col=labels_obs_col,
            confidence_obs_col=confidence_obs_col,
            masked_classes=masked_classes,
            min_confidence=min_confidence,
            class_index_manifest=class_index_manifest,
        )
        lll = cls.make_dogma_lll_union(**kwargs)
        dig = cls.make_dogma_dig_union(**kwargs)
        return DogmaJointLoader(lll, dig)


# ----------------------------------------------------------------------
# PR #43 (logical): joint LLL+DIG dataset wrapper for shared-encoder
# training with lysis_protocol as a scVI-style categorical batch covariate.
# Codes follow LYSIS_PROTOCOL_CODES (LLL=0, DIG=1) — must stay in sync.
# ----------------------------------------------------------------------
class DogmaJointLoader(Dataset):
    """Joint LLL+DIG dataset wrapper. Stamps lysis_idx per cell.

    LLL cells get ``lysis_idx=0``; DIG cells get ``lysis_idx=1``.
    Length = len(lll_loader) + len(dig_loader).

    Properties n_genes / n_peaks / n_proteins are required to match
    across arms (after PR #41a peak union, ATAC dim does match;
    RNA n_genes and Protein n_proteins always match by panel design).

    Validation: dim mismatch raises at construction time with a clear
    message naming the offending property — fail loud at the contract
    boundary rather than during forward.
    """

    LYSIS_LLL = 0
    LYSIS_DIG = 1

    def __init__(self, lll_loader, dig_loader):
        for prop in ("n_genes", "n_peaks", "n_proteins"):
            lll_val = getattr(lll_loader, prop)
            dig_val = getattr(dig_loader, prop)
            if lll_val != dig_val:
                raise ValueError(
                    f"DogmaJointLoader: {prop} mismatch — "
                    f"LLL={lll_val} vs DIG={dig_val}. "
                    "Have you run the PR #41a peak union production?"
                )
        self.lll = lll_loader
        self.dig = dig_loader
        self.n_lll = len(lll_loader)
        self.n_dig = len(dig_loader)

    def __len__(self):
        return self.n_lll + self.n_dig

    def __getitem__(self, idx):
        if idx < self.n_lll:
            item = dict(self.lll[idx])
            item["lysis_idx"] = self.LYSIS_LLL
        else:
            item = dict(self.dig[idx - self.n_lll])
            item["lysis_idx"] = self.LYSIS_DIG
        return item

    @property
    def n_genes(self):
        return self.lll.n_genes

    @property
    def n_peaks(self):
        return self.lll.n_peaks

    @property
    def n_proteins(self):
        return self.lll.n_proteins


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)
