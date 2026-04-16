"""
scripts/build_pbmc10k_h5ad.py — build the Phase 6.7b real-data Multiome
AnnData input expected by MultiomeLoader(schema="obsm_atac").

Inputs
------
- 10x Cell Ranger ARC filtered_feature_bc_matrix.h5 (paired RNA peak counts).
- 10x ATAC fragments tsv.gz  (not re-parsed here; hash-logged for provenance).
- Harmonized peak set (Phase 6.7 artifact, 115720 peaks, chr + start + end).

Output
------
- <output>.h5ad  (AnnData with X=RNA sparse counts, obsm["atac"]=peak counts
                  aligned to the harmonized peak set).
- <output>.meta.json  (input hashes, peak_set hash, n_cells, n_genes,
                       n_peaks_aligned, timestamp_utc, schema tag).

ATAC peak reindexing strategy
-----------------------------
Pure-Python interval intersection via sorted arrays + numpy searchsorted
(no pybedtools / pyranges dependency). For every 10x peak (chrom, s_i, e_i)
we locate harmonized peaks on the same chromosome whose [s_j, e_j) intervals
overlap [s_i, e_i). A sparse projection matrix P (n_10x_peaks x n_harm) is
built with P[i, j] = 1 where overlap holds, then aligned = ATAC_cells @ P
gives an (n_cells x n_harm) sparse peak-count matrix. This is the standard
"union peak" lift used in scATAC analyses; documented choice.

The schema="obsm_atac" branch of aivc/data/multiome_loader.py requires
RNA in .X and peaks in .obsm["atac"] — that is exactly what we write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_interval_bytes(intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse 10x feature 'interval' byte strings of form b'chr1:9790-10675'.

    Returns three numpy arrays (chrom, start, end). For features without an
    interval annotation (rare for genes with no genomic coord), returns '' and
    (-1, -1) so downstream code can mask them out.
    """
    n = intervals.shape[0]
    chrom = np.empty(n, dtype=object)
    start = np.full(n, -1, dtype=np.int64)
    end = np.full(n, -1, dtype=np.int64)
    for i, raw in enumerate(intervals):
        s = raw.decode("ascii") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if not s or ":" not in s or "-" not in s:
            chrom[i] = ""
            continue
        try:
            c, rest = s.split(":", 1)
            a, b = rest.split("-", 1)
            chrom[i] = c
            start[i] = int(a)
            end[i] = int(b)
        except ValueError:
            chrom[i] = ""
    return chrom, start, end


def _build_projection(
    peak_intervals_10x: Tuple[np.ndarray, np.ndarray, np.ndarray],
    peak_set_df: pd.DataFrame,
) -> sparse.csr_matrix:
    """Sparse P of shape (n_10x_peaks, n_harmonized) with P[i,j]=1 on overlap.

    Uses per-chromosome sorted-array lookup: for peak_10x with end e_i, start
    s_i on chromosome c, the overlapping harmonized peaks are those whose
    start < e_i AND end > s_i. We binary-search the harmonized sorted-by-start
    array for the upper bound, then linearly scan backward over the sorted-by-
    end secondary array. This is O(N log M + K) where K is total overlaps.
    """
    chrom_10x, start_10x, end_10x = peak_intervals_10x
    n_10x = len(chrom_10x)
    n_harm = len(peak_set_df)

    # Group harmonized peaks by chromosome and sort by start.
    peak_set_df = peak_set_df.reset_index(drop=False).rename(columns={"index": "harm_row"})
    groups = peak_set_df.groupby("chrom", sort=False)
    per_chrom = {}
    for chrom, grp in groups:
        order = grp["start"].to_numpy().argsort()
        per_chrom[chrom] = {
            "starts": grp["start"].to_numpy()[order],
            "ends": grp["end"].to_numpy()[order],
            "rows": grp["harm_row"].to_numpy()[order],
        }

    rows_acc: list[int] = []
    cols_acc: list[int] = []
    n_overlap_cum = 0
    n_peaks_with_hit = 0
    n_peaks_no_chrom = 0

    for i in range(n_10x):
        c = chrom_10x[i]
        s_i, e_i = start_10x[i], end_10x[i]
        if not c or s_i < 0:
            continue
        g = per_chrom.get(c)
        if g is None:
            n_peaks_no_chrom += 1
            continue
        starts = g["starts"]
        ends = g["ends"]
        rows_h = g["rows"]
        # Harmonized peaks with start < e_i are candidates (needed for overlap).
        hi = np.searchsorted(starts, e_i, side="left")
        if hi == 0:
            continue
        cand_ends = ends[:hi]
        cand_rows = rows_h[:hi]
        mask = cand_ends > s_i  # overlap condition
        hit_rows = cand_rows[mask]
        if hit_rows.size:
            rows_acc.extend([i] * hit_rows.size)
            cols_acc.extend(hit_rows.tolist())
            n_overlap_cum += hit_rows.size
            n_peaks_with_hit += 1

    data = np.ones(len(rows_acc), dtype=np.float32)
    P = sparse.csr_matrix(
        (data, (rows_acc, cols_acc)), shape=(n_10x, n_harm)
    )
    print(
        f"[align] 10x peaks: {n_10x}  harmonized peaks: {n_harm}  "
        f"overlapping edges: {n_overlap_cum}  "
        f"10x peaks with ≥1 hit: {n_peaks_with_hit} "
        f"({n_peaks_with_hit / max(n_10x, 1):.1%}) "
        f"no-chrom (alt contigs missing from peak set): {n_peaks_no_chrom}"
    )
    return P


def _load_10x_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        shape = matrix["shape"][:]  # (n_features, n_cells)
        n_features, n_cells = int(shape[0]), int(shape[1])
        indptr = matrix["indptr"][:]
        indices = matrix["indices"][:]
        data = matrix["data"][:]
        feat = matrix["features"]
        feat_type = feat["feature_type"][:]
        feat_id = feat["id"][:]
        feat_name = feat["name"][:]
        feat_interval = feat["interval"][:]
        barcodes = matrix["barcodes"][:]
    # CSC stored with one indptr per cell column (len = n_cells + 1).
    counts_csc = sparse.csc_matrix(
        (data, indices, indptr), shape=(n_features, n_cells)
    )
    return {
        "counts_csc": counts_csc,
        "feat_type": feat_type,
        "feat_id": feat_id,
        "feat_name": feat_name,
        "feat_interval": feat_interval,
        "barcodes": barcodes,
        "n_features": n_features,
        "n_cells": n_cells,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5_in",
        default=str(_REPO_ROOT / "data/raw/pbmc10k_multiome/filtered_feature_bc_matrix.h5"),
    )
    p.add_argument(
        "--fragments",
        default=str(_REPO_ROOT / "data/raw/pbmc10k_multiome/atac_fragments.tsv.gz"),
        help="Hash-logged only; not re-parsed (peak counts come from the 10x h5).",
    )
    p.add_argument(
        "--peak_set",
        default=str(_REPO_ROOT / "data/peak_sets/pbmc10k_hg38_20260415.tsv"),
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data/pbmc10k_multiome.h5ad"),
    )
    args = p.parse_args(argv)

    h5_in = Path(args.h5_in).resolve()
    fragments = Path(args.fragments).resolve()
    peak_set_path = Path(args.peak_set).resolve()
    output = Path(args.output).resolve()

    for path in (h5_in, fragments, peak_set_path):
        if not path.exists():
            print(f"[error] required input missing: {path}", file=sys.stderr)
            return 2

    t0 = time.time()
    print(f"[hash] {h5_in.name} ...")
    h5_sha = _sha256_file(h5_in)
    print(f"        sha256={h5_sha}")
    print(f"[hash] {fragments.name} ...")
    frag_sha = _sha256_file(fragments)
    print(f"        sha256={frag_sha}")
    print(f"[hash] {peak_set_path.name} ...")
    peak_set_sha = _sha256_file(peak_set_path)
    print(f"        sha256={peak_set_sha}")

    # Load peak set
    peak_set = pd.read_csv(peak_set_path, sep="\t")
    expected_cols = {"peak_id", "chrom", "start", "end"}
    missing = expected_cols - set(peak_set.columns)
    if missing:
        raise RuntimeError(f"peak_set missing columns: {missing}")
    n_harm = len(peak_set)
    print(f"[peak_set] {n_harm} harmonized peaks from {peak_set_path}")

    # Load 10x h5
    print(f"[10x] reading {h5_in}")
    x10 = _load_10x_h5(h5_in)
    print(f"[10x] n_features={x10['n_features']} n_cells={x10['n_cells']}")

    is_gene = x10["feat_type"] == b"Gene Expression"
    is_peak = x10["feat_type"] == b"Peaks"
    n_genes = int(is_gene.sum())
    n_peaks_10x = int(is_peak.sum())
    print(f"[10x] n_genes={n_genes} n_peaks_10x={n_peaks_10x}")

    # RNA: features × cells CSC → transpose to cells × features CSR
    counts_csc = x10["counts_csc"]
    rna_features_x_cells = counts_csc[is_gene, :]
    atac_features_x_cells = counts_csc[is_peak, :]
    rna_cells_x_genes = rna_features_x_cells.T.tocsr()
    atac_cells_x_peaks = atac_features_x_cells.T.tocsr()
    print(
        f"[10x] rna_cells_x_genes shape={rna_cells_x_genes.shape} "
        f"nnz={rna_cells_x_genes.nnz}  "
        f"atac_cells_x_peaks shape={atac_cells_x_peaks.shape} "
        f"nnz={atac_cells_x_peaks.nnz}"
    )

    # Peak intervals for the 10x Peaks features
    peak_intervals_10x = _parse_interval_bytes(x10["feat_interval"][is_peak])

    # Build projection
    P = _build_projection(peak_intervals_10x, peak_set)
    # Align ATAC peak counts to harmonized peak set
    aligned_atac = atac_cells_x_peaks @ P  # (n_cells, n_harm) sparse
    aligned_atac = aligned_atac.tocsr()
    # Diagnostic: how many cells land with ≥1 aligned peak count?
    row_nnz = np.diff(aligned_atac.indptr)
    cells_empty = int((row_nnz == 0).sum())
    col_nnz = (aligned_atac > 0).sum(axis=0)
    col_nnz = np.asarray(col_nnz).ravel()
    peaks_empty = int((col_nnz == 0).sum())
    print(
        f"[align] aligned_atac shape={aligned_atac.shape} nnz={aligned_atac.nnz}  "
        f"cells_with_0_peaks={cells_empty} harmonized_peaks_with_0_cells={peaks_empty}"
    )
    pct_empty_rows = cells_empty / aligned_atac.shape[0]
    if pct_empty_rows > 0.5:
        print(
            f"[error] >50% of cells have zero aligned peaks "
            f"({pct_empty_rows:.1%}). Likely chrom-naming or liftover drift. "
            "Aborting — see FAILURE HANDLING in Phase 6.7b task.",
            file=sys.stderr,
        )
        return 3

    # Build AnnData. Index .var by Ensembl gene_id (unique); keep the
    # human-readable gene_name as a regular column. Using gene_name as the
    # index fails because 10x ships occasional duplicated gene names.
    gene_ids = x10["feat_id"][is_gene].astype("U")
    gene_names = x10["feat_name"][is_gene].astype("U")
    obs = pd.DataFrame(index=pd.Index(x10["barcodes"].astype("U"), name="barcode"))
    var = pd.DataFrame(
        {"gene_name": gene_names},
        index=pd.Index(gene_ids, name="gene_id"),
    )
    adata = ad.AnnData(
        X=rna_cells_x_genes.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.obsm["atac"] = aligned_atac.astype(np.float32)
    adata.uns["peak_set_sha256"] = peak_set_sha
    adata.uns["peak_set_path"] = str(
        peak_set_path.relative_to(_REPO_ROOT)
        if peak_set_path.is_relative_to(_REPO_ROOT)
        else peak_set_path
    )
    adata.uns["schema"] = "obsm_atac"
    adata.uns["source_h5_sha256"] = h5_sha
    adata.uns["fragments_sha256"] = frag_sha
    adata.uns["n_peaks_aligned"] = int(aligned_atac.shape[1])

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {output}")
    adata.write_h5ad(output, compression="gzip")

    meta = {
        "output_h5ad": str(output.relative_to(_REPO_ROOT) if output.is_relative_to(_REPO_ROOT) else output),
        "schema": "obsm_atac",
        "n_cells": int(rna_cells_x_genes.shape[0]),
        "n_genes": int(rna_cells_x_genes.shape[1]),
        "n_peaks_10x": int(n_peaks_10x),
        "n_peaks_aligned": int(aligned_atac.shape[1]),
        "cells_with_0_peaks": cells_empty,
        "harmonized_peaks_with_0_cells": peaks_empty,
        "input_h5_sha256": h5_sha,
        "input_fragments_sha256": frag_sha,
        "peak_set_path": str(peak_set_path.relative_to(_REPO_ROOT) if peak_set_path.is_relative_to(_REPO_ROOT) else peak_set_path),
        "peak_set_sha256": peak_set_sha,
        "reindex_method": "pure-Python sorted-array searchsorted (per chromosome)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    print(f"[write] {meta_path}")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    print(f"[done] elapsed={time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
