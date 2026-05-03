"""Compute Calderon -> DOGMA peak projection matrix.

Calderon-normalized fractional-overlap projection:
    M[i, j] = overlap_length(C_i n D_j) / length(C_i)
    X_dogma_space = X_calderon @ M

Row-sum semantics (PR #38 update): row sums M[i,:].sum() are <= 1.0
when DOGMA peaks form a partition of the genome (synthetic / ideal case).
Real DOGMA peak sets contain overlapping peaks (~78% of DOGMA LLL peaks
overlap another peak); when two DOGMA peaks both cover the same
Calderon peak, that Calderon peak's row sum can exceed 1.0. The
projection still represents "how much of Calderon peak i lands on
DOGMA peak j", just with overcount in DOGMA-redundant regions.
Stats include n_calderon_with_dogma_redundancy for tracking.

Peak coordinates may be stored in two AnnData layouts (extract helper
handles both):

1. .var-style: peaks in .var with chrom/start/end columns.
   Used by Calderon prep (scripts/prepare_calderon_2019.py).

2. DOGMA-style: peak names in .uns['atac_feature_names'] as a flat
   string array. Counts in .obsm['atac_peaks']. Peak names use either
   colon-dash (chr1:1000-2000) or underscore (chr1_1000_2000) format.
   Used by DOGMA assembly (scripts/assemble_dogma_h5ad.py, PR #30).

Usage:
    python scripts/harmonize_calderon_peaks.py \\
        --dogma data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad \\
        --calderon data/calderon2019/calderon_atac.h5ad \\
        --out-projection data/calderon2019/calderon_to_dogma_lll_M.npz \\
        --out-stats data/calderon2019/harmonization_stats_lll.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse as sp


# Accepts both `chr1:1000-2000` (legacy) and `chr1_1000_2000` (DOGMA + Calderon).
# Also accepts unplaced contigs (e.g. `GL000194.1:100-200`) which appear in
# real DOGMA peak sets; they won't match Calderon (chr-prefixed only) but
# should not crash the dispatcher.
PEAK_RE = re.compile(r"^(?P<chrom>[\w.]+)[_:](?P<start>\d+)[_-](?P<end>\d+)$")


def extract_peaks_to_var_format(adata: ad.AnnData, name: str = "") -> pd.DataFrame:
    """Extract peak coordinates into a uniform (chrom, start, end) DataFrame.

    Dispatches on AnnData layout:
    1. DOGMA-style: peaks in adata.uns['atac_feature_names'] as strings.
    2. .var-style: peaks in adata.var with chrom/start/end columns.

    Returns
    -------
    pd.DataFrame with columns chrom (str), start (int64), end (int64),
    indexed by peak_id (the raw string from input).
    """
    # PR #38: Try DOGMA-style first (uns wins if both present).
    if "atac_feature_names" in adata.uns:
        feat = list(adata.uns["atac_feature_names"])
        if not feat:
            raise ValueError(f"{name}: .uns['atac_feature_names'] is empty")
        peak_ids, chroms, starts, ends = [], [], [], []
        for i, p in enumerate(feat):
            m = PEAK_RE.match(str(p))
            if m is None:
                raise ValueError(
                    f"{name}: cannot parse peak name {p!r} at index {i}"
                )
            peak_ids.append(p)
            chroms.append(m.group("chrom"))
            starts.append(int(m.group("start")))
            ends.append(int(m.group("end")))
        df = pd.DataFrame(
            {"chrom": chroms, "start": starts, "end": ends},
            index=pd.Index(peak_ids, name="peak_id"),
        )
        return df

    # Fall back to .var-style.
    required = {"chrom", "start", "end"}
    if required.issubset(adata.var.columns):
        df = adata.var[["chrom", "start", "end"]].copy()
        df["chrom"] = df["chrom"].astype(str)
        df["start"] = df["start"].astype(np.int64)
        df["end"] = df["end"].astype(np.int64)
        return df

    raise ValueError(
        f"{name}: unrecognized peak layout -- neither "
        ".uns['atac_feature_names'] nor .var with chrom/start/end. "
        f"var_cols={list(adata.var.columns)[:8]}, "
        f"uns_keys={list(adata.uns.keys())[:8]}"
    )


def _var_to_pyranges(var_df: pd.DataFrame, name: str) -> pr.PyRanges:
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
    """Compute (n_calderon, n_dogma) Calderon-normalized projection."""
    n_c = len(calderon_var)
    n_d = len(dogma_var)

    pr_c = _var_to_pyranges(calderon_var, "calderon")
    pr_d = _var_to_pyranges(dogma_var, "dogma")

    joined = pr_c.join(pr_d, suffix="_d").df
    if len(joined) == 0:
        M = sp.csr_matrix((n_c, n_d), dtype=np.float32)
        return M, {
            "n_calderon": n_c, "n_dogma": n_d,
            "n_overlaps": 0, "n_orphan_calderon": n_c,
            "frac_calderon_with_any_overlap": 0.0,
            "mean_row_sum": 0.0, "max_row_sum": 0.0,
        }

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

    # PR #38: real DOGMA peak sets are NOT partitions of the genome -- 78% of
    # peaks overlap another peak (nested/border). When two DOGMA peaks both
    # cover the same Calderon peak, weights legitimately sum > 1. Track the
    # redundancy as a stat instead of raising.
    row_sum_pre_clamp = np.asarray(M.sum(axis=1)).ravel()
    max_row_sum = float(row_sum_pre_clamp.max())
    # PR #52: removed hard guard. Real peak sets with significant DOGMA
    # self-overlap (e.g., union of LLL+DIG = 323,500 peaks) can produce
    # row sums up to N where N = number of DOGMA peaks overlapping a
    # single Calderon peak. Empirical max on union: 4.0. Soft warning
    # at 5.0 catches genuinely unusual cases without blocking production.
    if max_row_sum > 5.0:
        import warnings
        warnings.warn(
            f"Row sums unusually high (max={max_row_sum:.4f}). "
            "This is acceptable for highly-overlapping peak sets but "
            "worth investigating if unexpected. Check stats output for "
            "n_calderon_with_dogma_redundancy and mean_overlaps_per_calderon.",
            RuntimeWarning,
            stacklevel=2,
        )
    has_overlap = row_sum_pre_clamp > 0
    n_redundant = int((row_sum_pre_clamp > 1.0 + 1e-5).sum())
    overlaps_per_c = np.asarray((M > 0).sum(axis=1)).ravel()

    return M, {
        "n_calderon": int(n_c),
        "n_dogma": int(n_d),
        "n_overlaps": int(len(joined)),
        "n_orphan_calderon": int((~has_overlap).sum()),
        "frac_calderon_with_any_overlap": float(has_overlap.mean()),
        "mean_row_sum": float(row_sum_pre_clamp.mean()),
        "max_row_sum": float(row_sum_pre_clamp.max()),
        "n_calderon_with_dogma_redundancy": n_redundant,
        "frac_calderon_with_dogma_redundancy": float(n_redundant / n_c),
        "mean_overlaps_per_calderon": float(overlaps_per_c.mean()),
        "max_overlaps_per_calderon": int(overlaps_per_c.max()),
    }


def apply_projection(calderon_X, M: sp.csr_matrix) -> sp.csr_matrix:
    if calderon_X.shape[1] != M.shape[0]:
        raise ValueError(
            f"calderon_X.shape[1]={calderon_X.shape[1]} != M.shape[0]={M.shape[0]}"
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

    # PR #38: layout-aware peak extraction (handles DOGMA's .uns layout).
    dogma_var = extract_peaks_to_var_format(dogma, name="dogma")
    calderon_var = extract_peaks_to_var_format(calderon, name="calderon")

    M, stats = compute_projection_matrix(dogma_var, calderon_var)
    print(f"Projection: shape={M.shape}, nnz={M.nnz}")
    print(f"Coverage: {stats['frac_calderon_with_any_overlap']:.3%}")
    print(f"Orphan Calderon peaks: {stats['n_orphan_calderon']}")

    args.out_projection.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.out_projection, M)
    args.out_stats.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {args.out_projection}, {args.out_stats}")


if __name__ == "__main__":
    main()
