"""Compute Calderon -> DOGMA peak projection matrix.

Calderon 2019 (GSE118189) and DOGMA-seq (GSE156478) call peaks from
different read pileups; their peak sets are not directly comparable.
This script computes a Calderon-normalized fractional-overlap projection
matrix that maps Calderon raw counts into DOGMA peak space:

    X_dogma_space[sample, j] = sum_i X_calderon[sample, i] * M[i, j]
    where M[i, j] = overlap_length(C_i n D_j) / length(C_i)

Properties:
    - M is sparse (CSR), shape (n_calderon, n_dogma).
    - Row sums M[i, :].sum() <= 1.0 -- fractional mass distribution.
    - M[i, :].sum() == 1.0 iff Calderon peak C_i is fully covered by
      DOGMA peaks (otherwise coverage gap = 1 - row_sum).
    - Total mass-preserving: sum(X @ M) <= sum(X), equality on full coverage.

Usage:
    python scripts/harmonize_calderon_peaks.py \\
        --dogma data/phase6_5g_2/dogma_lll.h5ad \\
        --calderon data/calderon2019/calderon_atac.h5ad \\
        --out-projection data/calderon2019/calderon_to_dogma_M.npz \\
        --out-stats data/calderon2019/harmonization_stats.json

Output:
    - projection.npz -- scipy.sparse.csr_matrix saved via save_npz
    - stats.json -- coverage diagnostics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse as sp


def _var_to_pyranges(var_df: pd.DataFrame, name: str) -> pr.PyRanges:
    """Convert AnnData .var (chrom/start/end columns) to PyRanges with
    integer position index preserved as `peak_idx`."""
    required = {"chrom", "start", "end"}
    missing = required - set(var_df.columns)
    if missing:
        raise ValueError(f"{name}.var missing columns: {missing}")

    df = pd.DataFrame({
        "Chromosome": var_df["chrom"].astype(str).values,
        "Start": var_df["start"].astype(np.int64).values,
        "End": var_df["end"].astype(np.int64).values,
        "peak_idx": np.arange(len(var_df), dtype=np.int64),
    })
    if (df["End"] <= df["Start"]).any():
        raise ValueError(f"{name}: found peaks with End <= Start")
    return pr.PyRanges(df)


def compute_projection_matrix(
    dogma_var: pd.DataFrame,
    calderon_var: pd.DataFrame,
) -> tuple[sp.csr_matrix, dict]:
    """Compute the (n_calderon, n_dogma) Calderon-normalized projection.

    Returns
    -------
    M : sp.csr_matrix
        Shape (n_calderon, n_dogma). Row i sums to <= 1.0.
    stats : dict
        Coverage diagnostics for sanity checks and PR-body reporting.
    """
    n_c = len(calderon_var)
    n_d = len(dogma_var)

    pr_c = _var_to_pyranges(calderon_var, "calderon")
    pr_d = _var_to_pyranges(dogma_var, "dogma")

    joined = pr_c.join(pr_d, suffix="_d").df
    if len(joined) == 0:
        M = sp.csr_matrix((n_c, n_d), dtype=np.float32)
        stats = {
            "n_calderon": n_c, "n_dogma": n_d,
            "n_overlaps": 0, "n_orphan_calderon": n_c,
            "frac_calderon_with_any_overlap": 0.0,
            "mean_row_sum": 0.0, "max_row_sum": 0.0,
        }
        return M, stats

    overlap_start = np.maximum(joined["Start"].values, joined["Start_d"].values)
    overlap_end = np.minimum(joined["End"].values, joined["End_d"].values)
    overlap_len = (overlap_end - overlap_start).astype(np.float64)
    if (overlap_len <= 0).any():
        raise RuntimeError("PyRanges join returned non-overlapping rows")

    calderon_len = (joined["End"].values - joined["Start"].values).astype(np.float64)
    weights = (overlap_len / calderon_len).astype(np.float32)

    rows = joined["peak_idx"].values.astype(np.int64)
    cols = joined["peak_idx_d"].values.astype(np.int64)

    M = sp.coo_matrix((weights, (rows, cols)), shape=(n_c, n_d)).tocsr()

    row_sum = np.asarray(M.sum(axis=1)).ravel()
    if (row_sum > 1.0 + 1e-5).any():
        raise RuntimeError(
            f"Row sums exceed 1.0 (max={row_sum.max():.6f}) -- "
            "indicates duplicate Calderon entries or join bug"
        )
    has_overlap = row_sum > 0
    overlaps_per_c = np.asarray((M > 0).sum(axis=1)).ravel()

    stats = {
        "n_calderon": int(n_c),
        "n_dogma": int(n_d),
        "n_overlaps": int(len(joined)),
        "n_orphan_calderon": int((~has_overlap).sum()),
        "frac_calderon_with_any_overlap": float(has_overlap.mean()),
        "mean_row_sum": float(row_sum.mean()),
        "max_row_sum": float(row_sum.max()),
        "mean_overlaps_per_calderon": float(overlaps_per_c.mean()),
        "max_overlaps_per_calderon": int(overlaps_per_c.max()),
    }
    return M, stats


def apply_projection(
    calderon_X,
    M: sp.csr_matrix,
) -> sp.csr_matrix:
    """X_dogma_space = X_calderon @ M. Shape (n_samples, n_dogma)."""
    if calderon_X.shape[1] != M.shape[0]:
        raise ValueError(
            f"calderon_X.shape[1]={calderon_X.shape[1]} != "
            f"M.shape[0]={M.shape[0]}"
        )
    if sp.issparse(calderon_X):
        return (calderon_X @ M).tocsr()
    return sp.csr_matrix(np.asarray(calderon_X) @ M)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dogma", required=True, type=Path)
    p.add_argument("--calderon", required=True, type=Path)
    p.add_argument("--out-projection", required=True, type=Path)
    p.add_argument("--out-stats", required=True, type=Path)
    args = p.parse_args()

    print(f"Loading DOGMA: {args.dogma}")
    dogma = ad.read_h5ad(args.dogma, backed="r")
    print(f"Loading Calderon: {args.calderon}")
    calderon = ad.read_h5ad(args.calderon, backed="r")

    M, stats = compute_projection_matrix(dogma.var.copy(), calderon.var.copy())
    print(f"Projection: shape={M.shape}, nnz={M.nnz}")
    print(f"Coverage: {stats['frac_calderon_with_any_overlap']:.3%} of "
          f"Calderon peaks have >=1 DOGMA overlap")
    print(f"Orphan Calderon peaks: {stats['n_orphan_calderon']}")

    args.out_projection.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_projection, M)
    args.out_stats.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {args.out_projection}, {args.out_stats}")


if __name__ == "__main__":
    main()
