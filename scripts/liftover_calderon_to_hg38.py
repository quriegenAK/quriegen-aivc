"""Lift Calderon AnnData from hg19 to hg38 via pyliftover.

Calderon 2019 (GSE118189) was published with hg19 coordinates. DOGMA
peak calling (PR #30) uses hg38. Path A diagnostic showed Calderon
hg19 + DOGMA hg38 produces only 9.81% peak overlap (mostly coincidental
alignment in stable regions). After lifting Calderon to hg38, expected
frac_overlap rises to 60-90%.

UCSC chain file (hg19ToHg38.over.chain.gz, ~1 MB) is downloaded by
pyliftover on first use and cached at ~/.pyliftover/.

Strategy:
- Lift each peak's start AND end positions independently.
- Drop peak if either lift fails, chromosomes diverge after lift, or
  end <= start after lift.
- Subset .X along peak axis; preserve obs.

Usage:
    python scripts/liftover_calderon_to_hg38.py \\
        --in data/calderon2019/calderon_atac.h5ad \\
        --out data/calderon2019/calderon_atac_hg38.h5ad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _make_liftover(from_build: str = "hg19", to_build: str = "hg38"):
    """Lazy import + construction (so unit tests can mock cleanly)."""
    from pyliftover import LiftOver
    return LiftOver(from_build, to_build)


def lift_var_coordinates(
    var_df: pd.DataFrame,
    liftover_obj=None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Lift (chrom, start, end) coordinates in var_df from hg19 to hg38.

    Parameters
    ----------
    var_df : pd.DataFrame
        Must have chrom, start, end columns.
    liftover_obj : optional
        A pyliftover.LiftOver instance, or any object with
        .convert_coordinate(chrom, pos) -> [(new_chrom, new_pos, _, _)].
        If None, constructs hg19->hg38 from pyliftover.

    Returns
    -------
    new_var_df : pd.DataFrame
        Subset of input, with chrom/start/end replaced by hg38.
    keep_idx : np.ndarray of int64
        Original positional indices of peaks that lifted cleanly. Use
        to subset adata.X / adata.obsm.
    stats : dict
        n_input, n_lifted, n_dropped_no_lift, n_dropped_chrom_change,
        n_dropped_invalid_range, frac_lifted.
    """
    lo = liftover_obj if liftover_obj is not None else _make_liftover()
    n = len(var_df)
    if n == 0:
        empty = var_df.iloc[0:0].copy()
        return empty, np.asarray([], dtype=np.int64), {
            "n_input": 0,
            "n_lifted": 0,
            "n_dropped_no_lift": 0,
            "n_dropped_chrom_change": 0,
            "n_dropped_invalid_range": 0,
            "frac_lifted": 0.0,
        }

    chroms = var_df["chrom"].astype(str).values
    starts = var_df["start"].astype(np.int64).values
    ends = var_df["end"].astype(np.int64).values

    new_chroms, new_starts, new_ends, keep_idx = [], [], [], []
    n_dropped_no_lift = 0
    n_dropped_chrom_change = 0
    n_dropped_invalid_range = 0

    for i in range(n):
        s_hits = lo.convert_coordinate(chroms[i], int(starts[i]))
        e_hits = lo.convert_coordinate(chroms[i], int(ends[i]))
        if not s_hits or not e_hits:
            n_dropped_no_lift += 1
            continue
        ns_chrom, ns_pos, _, _ = s_hits[0]
        ne_chrom, ne_pos, _, _ = e_hits[0]
        if ns_chrom != ne_chrom:
            n_dropped_chrom_change += 1
            continue
        # pyliftover sometimes flips order on inversions
        new_s = min(ns_pos, ne_pos)
        new_e = max(ns_pos, ne_pos)
        if new_e <= new_s:
            n_dropped_invalid_range += 1
            continue
        new_chroms.append(ns_chrom)
        new_starts.append(new_s)
        new_ends.append(new_e)
        keep_idx.append(i)

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    new_var = var_df.iloc[keep_idx_arr].copy()
    new_var["chrom"] = new_chroms
    new_var["start"] = new_starts
    new_var["end"] = new_ends

    return new_var, keep_idx_arr, {
        "n_input": int(n),
        "n_lifted": int(len(keep_idx_arr)),
        "n_dropped_no_lift": int(n_dropped_no_lift),
        "n_dropped_chrom_change": int(n_dropped_chrom_change),
        "n_dropped_invalid_range": int(n_dropped_invalid_range),
        "frac_lifted": float(len(keep_idx_arr) / n),
    }


def lift_anndata(
    adata: ad.AnnData,
    liftover_obj=None,
) -> tuple[ad.AnnData, dict]:
    """Lift adata to hg38: subset .X along peak axis."""
    new_var, keep_idx, stats = lift_var_coordinates(adata.var.copy(), liftover_obj)
    new_X = adata.X[:, keep_idx] if adata.X is not None else None
    new_obs = adata.obs.copy()
    new_uns = dict(adata.uns)
    new_uns["liftover_hg19_to_hg38_stats"] = stats
    new_uns["build"] = "hg38"

    new_adata = ad.AnnData(
        X=new_X,
        obs=new_obs,
        var=new_var,
        uns=new_uns,
    )
    return new_adata, stats


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    print(f"Loading {args.in_path}")
    adata = ad.read_h5ad(args.in_path)
    print(f"Input shape: {adata.shape}, build assumption: hg19")

    new_adata, stats = lift_anndata(adata)
    print(f"Output shape: {new_adata.shape}")
    print(f"Lift stats: {json.dumps(stats, indent=2)}")
    print(f"frac_lifted: {stats['frac_lifted']:.3%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_adata.write_h5ad(args.out, compression="gzip")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
