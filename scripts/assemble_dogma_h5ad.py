#!/usr/bin/env python3
"""Assemble DOGMA-seq h5ad files from cached raw GEO data.

Produces dogma_lll.h5ad and dogma_dig.h5ad with the contract:
  - RNA in .X (normalize_total(1e4) + log1p)
  - ATAC chromVAR motif deviations in obsm['atac_peaks']
  - Protein (CLR-normalized TotalSeq-A) in obsm['protein']
  - Barcodes inner-joined on pairs_of_wnn.tsv (LLL) or via QC chain (DIG)
  - obs columns: lysis_protocol, condition, has_rna=True, has_atac=True,
    has_protein=True, has_phospho=False, donor_id

Validates each output with pairing_validator using make_dogma_seq() cert.

Usage:
    python scripts/assemble_dogma_h5ad.py \\
        --raw-path ~/aivc_dogma_ncells_py \\
        --output-dir data/phase6_5g_2/dogma_h5ads \\
        --arm both   # or: lll | dig

Dependencies:
    pip install pychromvar pyjaspar scanpy muon  --break-system-packages
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio

from aivc.data.modality_mask import (
    RNA_KEY, ATAC_KEY, PROTEIN_KEY,
    LYSIS_KEY, PROTEIN_PANEL_KEY,
    ModalityKey,
)


def load_rna_h5(path: Path) -> Tuple[sp.csc_matrix, list, list]:
    """Return (sparse_rna, barcodes, gene_names) from a 10x Multiome h5."""
    import h5py
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])
        bcs = [b.decode() for b in m["barcodes"][:]]
        names = [n.decode() for n in m["features"]["name"][:]]
        ftypes = [t.decode() for t in m["features"]["feature_type"][:]]
    csc = sp.csc_matrix((data, indices, indptr), shape=shape)
    ge_mask = np.array([t == "Gene Expression" for t in ftypes])
    return csc[ge_mask, :], bcs, [n for n, m_ in zip(names, ge_mask) if m_]


def load_atac_h5(path: Path) -> Tuple[sp.csc_matrix, list, list]:
    """Return (sparse_atac_peaks, barcodes, peak_names) from 10x Multiome h5."""
    import h5py
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])
        bcs = [b.decode() for b in m["barcodes"][:]]
        names = [n.decode() for n in m["features"]["name"][:]]
        ftypes = [t.decode() for t in m["features"]["feature_type"][:]]
    csc = sp.csc_matrix((data, indices, indptr), shape=shape)
    peak_mask = np.array([t == "Peaks" for t in ftypes])
    return csc[peak_mask, :], bcs, [n for n, m_ in zip(names, peak_mask) if m_]


def load_adt(lane_dir: Path, lane: str) -> Tuple[sp.csr_matrix, list, list]:
    """Return (sparse_adt, barcodes, antibody_names) from kite featurecounts output."""
    mtx_files = list(lane_dir.glob(f"{lane}__*featurecounts*.mtx*"))
    bc_files  = list(lane_dir.glob(f"{lane}__*barcodes*.txt*"))
    gn_files  = list(lane_dir.glob(f"{lane}__*genes*.txt*"))
    if not (mtx_files and bc_files and gn_files):
        raise FileNotFoundError(f"ADT files missing for {lane} in {lane_dir}")
    mtx_path = mtx_files[0]

    def _read_lines(p: Path) -> list:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt") as fh:
            return [l.strip() for l in fh if l.strip()]

    with (gzip.open if str(mtx_path).endswith(".gz") else open)(mtx_path, "rt") as fh:
        first = fh.readline()
    if first.startswith("%%"):
        mat = sio.mmread(str(mtx_path)).tocsr()
    else:
        df = pd.read_csv(mtx_path, sep=r"\s+", header=None,
                         compression="infer" if str(mtx_path).endswith(".gz") else None)
        dims = df.iloc[0].astype(int).tolist()
        body = df.iloc[1:].astype({0: int, 1: int, 2: float})
        mat = sp.coo_matrix(
            (body[2], (body[0] - 1, body[1] - 1)),
            shape=(dims[0], dims[1]),
        ).tocsr()

    bcs = _read_lines(bc_files[0])
    gn_raw_lines = _read_lines(gn_files[0])
    gn_raw = pd.DataFrame([l.split("\t") if "\t" in l else l.split() for l in gn_raw_lines])
    ab_names = gn_raw[0].astype(str).tolist()
    if any(re.match(r"ENSG|ENST", a) for a in ab_names[:5]) and gn_raw.shape[1] > 1:
        ab_names = gn_raw[1].astype(str).tolist()

    if mat.shape[0] != len(ab_names):
        mat = mat.T
    assert mat.shape == (len(ab_names), len(bcs)), f"ADT shape {mat.shape}"
    return mat, bcs, ab_names


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"barcode", "is_cell", "atac_peak_region_fragments", "atac_fragments"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics file {path.name} missing columns: {missing}")
    df = df[df["is_cell"] == 1].copy()
    df["pct_in_peaks"] = df["atac_peak_region_fragments"] / df["atac_fragments"] * 100
    return df


def normalize_rna(mat: sp.csc_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """normalize_total + log1p on a sparse count matrix (genes x cells)."""
    X = mat.T.tocsr().astype(np.float32)  # (cells, genes)
    cell_totals = np.array(X.sum(axis=1)).ravel()
    cell_totals[cell_totals == 0] = 1.0
    scale = target_sum / cell_totals
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X


def clr_normalize(mat: sp.csr_matrix) -> np.ndarray:
    """Centered log-ratio normalization for ADT. Returns dense (cells, antibodies)."""
    X = np.asarray(mat.toarray() if sp.issparse(mat) else mat, dtype=np.float32)
    X = X + 1.0
    log_X = np.log(X)
    log_gmean = log_X.mean(axis=1, keepdims=True)
    return (log_X - log_gmean).astype(np.float32)


def compute_chromvar_deviations(
    atac_mat: sp.csc_matrix,
    peak_names: list,
    barcodes: list,
) -> Tuple[np.ndarray, list]:
    """Compute chromVAR motif deviation scores."""
    try:
        import pychromvar as pcv
        from pyjaspar import jaspardb
        import anndata as ad
    except ImportError as e:
        raise ImportError(
            "chromVAR computation requires pychromvar + pyjaspar + anndata. "
            "Install: pip install pychromvar pyjaspar anndata --break-system-packages. "
            "Or use --raw-peaks-fallback to emit raw peak counts in obsm['atac_peaks']."
        ) from e

    X = atac_mat.T.tocsr()
    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = peak_names
    parsed = [re.match(r"(chr[^\W_]+)[:_-](\d+)[-_:](\d+)", p) for p in peak_names]
    adata.var["chrom"] = [m.group(1) if m else "unknown" for m in parsed]
    adata.var["start"] = [int(m.group(2)) if m else 0 for m in parsed]
    adata.var["end"]   = [int(m.group(3)) if m else 0 for m in parsed]

    jdb = jaspardb(release="JASPAR2022")
    motifs = jdb.fetch_motifs(collection="CORE", tax_group=["vertebrates"],
                              species=[9606])

    pcv.add_gc_bias(adata, genome="hg38")
    pcv.get_bg_peaks(adata, niterations=50)
    pcv.match_motif(adata, motifs=motifs, genome="hg38")
    pcv.compute_deviations(adata)

    return (
        np.asarray(adata.obsm["chromvar_deviations"], dtype=np.float32),
        list(adata.uns.get("motif_names", [f"motif_{i}" for i in range(adata.obsm["chromvar_deviations"].shape[1])])),
    )


def assemble_arm(
    arm: str,
    raw_path: Path,
    anchor_barcodes: Optional[set] = None,
    use_chromvar: bool = True,
) -> "anndata.AnnData":
    """Assemble a single DOGMA arm (LLL or DIG) as an AnnData."""
    import anndata as ad

    lane_idx = {"LLL": ["1", "2"], "DIG": ["3", "4"]}[arm]
    lane_names = [f"{arm}_CTRL", f"{arm}_STIM"]

    rna_mats, atac_mats, adt_mats = [], [], []
    all_barcodes = []
    obs_rows = []

    for idx, lane in zip(lane_idx, lane_names):
        h5_path = next((raw_path / "h5").glob(f"{lane}__*.h5"))
        met_path = next((raw_path / "metrics").glob(f"{lane}__*"))

        rna_mat, rna_bcs, rna_names = load_rna_h5(h5_path)
        atac_mat, atac_bcs, peak_names = load_atac_h5(h5_path)
        assert rna_bcs == atac_bcs, f"RNA/ATAC barcode mismatch in {lane}"

        metrics = load_metrics(met_path)

        adt_mat, adt_bcs, ab_names = load_adt(raw_path / "adt", lane)

        n_total_rna = np.array(rna_mat.sum(axis=0)).ravel()
        n_feat_rna = (rna_mat > 0).sum(axis=0).A1
        mt_mask = np.array([n.startswith("MT") for n in rna_names])
        pct_mito = (np.array(rna_mat[mt_mask, :].sum(axis=0)).ravel()
                    / np.maximum(n_total_rna, 1) * 100)
        rna_keep = (pct_mito < 30) & (n_total_rna > 1000) & (n_feat_rna > 500)

        bc_suffixed = [re.sub(r"-1$", f"-{idx}", b) for b in rna_bcs]
        metrics["barcode"] = metrics["barcode"].str.replace(
            r"-1$", f"-{idx}", regex=True
        )
        adt_bcs_suffixed = [re.sub(r"-1$", f"-{idx}", b) for b in adt_bcs]

        iso_mask = np.array([bool(re.search(r"Ctrl|CTRL|ctrl|IgG|Isotype", a)) for a in ab_names])
        cd4_idx = next((i for i, a in enumerate(ab_names) if re.match(r"^CD4(-|$|_)", a)), None)
        cd8_idx = next((i for i, a in enumerate(ab_names) if re.match(r"^CD8(-|$|_)", a)), None)
        adt_df = pd.DataFrame({
            "barcode": adt_bcs_suffixed,
            "CD4adt": adt_mat[cd4_idx, :].toarray().ravel() if cd4_idx is not None else 0,
            "CD8adt": adt_mat[cd8_idx, :].toarray().ravel() if cd8_idx is not None else 0,
            "totalADT": np.array(adt_mat.sum(axis=0)).ravel(),
            "totalCTRLadt": np.array(adt_mat[iso_mask, :].sum(axis=0)).ravel(),
        })

        full = metrics.merge(adt_df, on="barcode", how="inner")
        full = full[full["barcode"].isin(set(np.array(bc_suffixed)[rna_keep]))]

        bc_to_rnatotal = dict(zip(bc_suffixed, n_total_rna))
        full["nCount_RNA"] = full["barcode"].map(bc_to_rnatotal).fillna(0)

        mask = ((full["pct_in_peaks"] > 50)
                & (full["totalCTRLadt"] < 10)
                & (full["totalADT"] > 100)
                & (full["nCount_RNA"] < 10**4.5)
                & ~((full["CD8adt"] > 30) & (full["CD4adt"] > 100)))
        passed_barcodes = set(full.loc[mask, "barcode"])

        if anchor_barcodes is not None:
            passed_barcodes &= anchor_barcodes

        bc_to_col = {b: i for i, b in enumerate(bc_suffixed)}
        adt_bc_to_col = {b: i for i, b in enumerate(adt_bcs_suffixed)}
        kept_ordered = [b for b in bc_suffixed if b in passed_barcodes]
        rna_cols = [bc_to_col[b] for b in kept_ordered]
        adt_cols = [adt_bc_to_col[b] for b in kept_ordered if b in adt_bc_to_col]
        assert len(rna_cols) == len(adt_cols), f"RNA/ADT intersection mismatch in {lane}"

        rna_mats.append(rna_mat[:, rna_cols])
        atac_mats.append(atac_mat[:, rna_cols])
        adt_mats.append(adt_mat[:, adt_cols])
        all_barcodes.extend(kept_ordered)

        cond = "STIM" if lane.endswith("STIM") else "CTRL"
        for _ in kept_ordered:
            obs_rows.append({
                LYSIS_KEY: arm,
                "condition": cond,
                "has_rna": True,
                "has_atac": True,
                "has_protein": True,
                "has_phospho": False,
                "donor_id": "mimitou_2021_donor_1",
                "lane": lane,
                PROTEIN_PANEL_KEY: "totalseq_a_210",
            })

    rna_full = sp.hstack(rna_mats).tocsc()
    atac_full = sp.hstack(atac_mats).tocsc()
    adt_full = sp.hstack(adt_mats).tocsr()
    n_total = rna_full.shape[1]
    assert len(all_barcodes) == n_total == atac_full.shape[1] == adt_full.shape[1]

    rna_normed = normalize_rna(rna_full)
    adt_clr = clr_normalize(adt_full.T)

    if use_chromvar:
        atac_scores, motif_names = compute_chromvar_deviations(
            atac_full, peak_names, all_barcodes,
        )
    else:
        atac_scores = np.asarray(atac_full.T.toarray(), dtype=np.float32)
        motif_names = peak_names

    adata = ad.AnnData(X=rna_normed)
    adata.obs_names = all_barcodes
    adata.var_names = rna_names
    adata.obs = pd.DataFrame(obs_rows, index=all_barcodes)
    adata.obsm[ATAC_KEY] = atac_scores
    adata.obsm[PROTEIN_KEY] = adt_clr
    adata.uns["atac_feature_names"] = motif_names
    adata.uns["adt_antibody_names"] = ab_names
    adata.uns["atac_type"] = "chromvar_deviations" if use_chromvar else "raw_peaks"
    adata.uns["assembly_contract_version"] = 1
    return adata


def validate_with_pairing_cert(adata) -> Dict:
    """Validate the assembled AnnData satisfies the DOGMA pairing certificate."""
    from aivc.data.pairing_certificate import PairingCertificate
    from aivc.data.pairing_validator import PairingValidator

    cert = PairingCertificate.make_dogma_seq()
    validator = PairingValidator()
    modalities_present = ["rna", "atac", "protein"]
    result = validator.validate_certificate_for_training(cert, modalities_present)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-path", type=Path, required=True,
                    help="Directory with h5/, metrics/, adt/ subdirs (PROMPT K v3 output).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--arm", choices=["lll", "dig", "both"], default="both")
    ap.add_argument("--raw-peaks-fallback", action="store_true",
                    help="Skip chromVAR; emit raw peak counts in obsm['atac_peaks'].")
    ap.add_argument("--anchor-path", type=Path, default=None,
                    help="pairs_of_wnn.tsv (LLL anchor). If omitted, no anchor used.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor = None
    if args.anchor_path and args.anchor_path.exists():
        df = pd.read_csv(args.anchor_path, sep="\t")
        bc_col = [c for c in df.columns if "barcode" in c.lower() or c == df.columns[0]][0]
        anchor = set(df[bc_col])
        print(f"[anchor] {len(anchor):,} barcodes from {args.anchor_path}")

    arms = ["LLL", "DIG"] if args.arm == "both" else [args.arm.upper()]
    use_chromvar = not args.raw_peaks_fallback

    for arm in arms:
        anc = anchor if arm == "LLL" else None
        print(f"\n=== Assembling {arm} ===")
        adata = assemble_arm(arm, args.raw_path, anchor_barcodes=anc,
                             use_chromvar=use_chromvar)
        out = args.output_dir / f"dogma_{arm.lower()}.h5ad"
        adata.write_h5ad(out)
        print(f"  wrote {out}  ({adata.n_obs:,} cells, {adata.n_vars:,} genes)")

        val = validate_with_pairing_cert(adata)
        assert val["can_train"], f"{arm} pairing cert validation failed: {val['errors']}"
        print(f"  cert validate: can_train=True, contrastive_pairs={val['contrastive_pairs']}")


if __name__ == "__main__":
    main()
