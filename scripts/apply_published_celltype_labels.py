"""Apply published Azimuth cell-type labels to DOGMA h5ads.

Path X (post-2026-05-03 Mimitou-paper deep-dive): the asap_reproducibility
repo publishes per-barcode Azimuth `predicted.celltype.l1` labels for the
LLL arm at:
  https://raw.githubusercontent.com/caleblareau/asap_reproducibility/master/pbmc_stim_multiome/output/LLL_module_scores.csv

The CSV row index is the cellranger barcode (suffix -1 = control, -2 = stim).
We checked: 100% barcode-set equality with `dogma_lll_union.h5ad` (13,763
cells in both). This script joins by barcode and writes a labeled h5ad.

LABEL CANONICALIZATION:
The paper uses space-separated labels ("CD4 T", "CD8 T", "other T"). We
canonicalize to underscore form ("CD4_T", "CD8_T", "other_T") for
Python-identifier-friendliness in downstream loss/loader code.

DIG ARM:
The asap_reproducibility repo does NOT publish DIG labels. This script
labels LLL only. Two follow-up paths:
  (a) Run Azimuth in R on DIG cells (matches paper methodology exactly).
  (b) Train a kNN/LR classifier on per-marker z-scored protein with LLL
      Azimuth labels as ground truth, predict on DIG. Same panel, same
      donors, so transfer should work well.
  (c) Skip DIG labeling: SupCon trains on LLL only; DIG cells remain in
      the contrastive InfoNCE/recon path without label loss. Acceptable
      because DIG provides batch diversity + lysis covariate signal.

Usage:
    python scripts/apply_published_celltype_labels.py \\
        --h5ad data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad \\
        --labels_csv data/phase6_5g_2/external_evidence/asap_repro_labels_2026-05-03/LLL_module_scores.csv \\
        --out data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


# Map paper labels (space-separated) -> Python-identifier-friendly form.
# Order matches the paper's predicted.celltype.l1 categories.
LABEL_CANONICAL_MAP = {
    "CD4 T":   "CD4_T",
    "CD8 T":   "CD8_T",
    "B":       "B",
    "NK":      "NK",
    "Mono":    "Monocyte",
    "DC":      "DC",
    "other T": "other_T",
    "other":   "other",
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--h5ad", required=True, type=Path,
                   help="Input DOGMA h5ad (must use cellranger barcode index)")
    p.add_argument("--labels_csv", required=True, type=Path,
                   help="LLL_module_scores.csv with predicted.celltype.l1")
    p.add_argument("--out", required=True, type=Path,
                   help="Output h5ad with obs['cell_type'] populated")
    p.add_argument("--label_column", default="predicted.celltype.l1",
                   help="CSV column name with cell-type labels")
    p.add_argument("--strict", action="store_true",
                   help="Raise on any unmatched barcode (default: report + warn)")
    args = p.parse_args()

    print(f"Loading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    print(f"\nLoading labels from {args.labels_csv}")
    labels_df = pd.read_csv(args.labels_csv, index_col=0)
    print(f"  rows: {len(labels_df)}")
    print(f"  columns: {list(labels_df.columns)}")
    if args.label_column not in labels_df.columns:
        raise ValueError(
            f"--label_column={args.label_column!r} not in CSV columns: "
            f"{list(labels_df.columns)}"
        )

    # --- Coverage check ---
    h5ad_bcs = set(adata.obs.index)
    csv_bcs = set(labels_df.index)
    matched = h5ad_bcs & csv_bcs
    h5ad_only = h5ad_bcs - csv_bcs
    csv_only = csv_bcs - h5ad_bcs

    print(f"\n=== Barcode coverage ===")
    print(f"  h5ad cells:        {len(h5ad_bcs)}")
    print(f"  CSV labeled cells: {len(csv_bcs)}")
    print(f"  matched:           {len(matched)}  ({100 * len(matched) / len(h5ad_bcs):.2f}%)")
    print(f"  h5ad-only (will be 'Unknown'):  {len(h5ad_only)}")
    print(f"  CSV-only (will be discarded):   {len(csv_only)}")

    if len(matched) == 0:
        raise ValueError(
            "Zero barcode overlap. Check barcode format consistency: "
            f"h5ad sample={list(h5ad_bcs)[:3]!r}, csv sample={list(csv_bcs)[:3]!r}"
        )
    if args.strict and (h5ad_only or csv_only):
        raise ValueError(
            f"--strict mode: {len(h5ad_only)} h5ad-only + {len(csv_only)} CSV-only barcodes"
        )

    # --- Join ---
    raw_labels = labels_df[args.label_column].reindex(adata.obs.index)
    n_unmatched = int(raw_labels.isna().sum())
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched} cells have no label, will receive 'Unknown'")

    # Canonicalize labels
    print(f"\n=== Label canonicalization ===")
    raw_distinct = raw_labels.dropna().value_counts()
    print(f"  Raw label distribution (from CSV):")
    for lbl, cnt in raw_distinct.items():
        canonical = LABEL_CANONICAL_MAP.get(lbl, lbl)
        marker = "" if lbl in LABEL_CANONICAL_MAP else "  <-- NOT in canonical map; will pass through"
        print(f"    {lbl:<10s} -> {canonical:<12s} ({cnt:>6d}){marker}")

    canonical_labels = raw_labels.map(
        lambda x: LABEL_CANONICAL_MAP.get(x, x) if pd.notna(x) else "Unknown"
    )
    canonical_labels = canonical_labels.fillna("Unknown")
    canonical_labels.index = adata.obs.index
    canonical_labels = canonical_labels.astype(str)

    # --- Stamp into obs ---
    adata.obs["cell_type"] = canonical_labels.values
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    adata.obs["cell_type_source"] = "azimuth_l1_published_2021"

    # Also stamp other useful columns for downstream
    if "stim" in labels_df.columns:
        stim_aligned = labels_df["stim"].reindex(adata.obs.index)
        # Paper uses "Control"/"Stim"; canonicalize to lowercase
        adata.obs["stim_published"] = (
            stim_aligned.fillna("unknown").str.lower().astype("category")
        )

    # --- Final distribution ---
    final_dist = adata.obs["cell_type"].value_counts()
    print(f"\n=== Final cell_type distribution ===")
    n_total = len(adata.obs)
    for lbl, cnt in final_dist.items():
        pct = 100 * cnt / n_total
        print(f"  {lbl:<14s}: {cnt:>6d}  ({pct:.1f}%)")

    n_unknown = int((adata.obs["cell_type"] == "Unknown").sum())
    n_distinct = len(final_dist) - (1 if "Unknown" in final_dist.index else 0)
    print(f"\n=== Summary ===")
    print(f"  total cells: {n_total}")
    print(f"  labeled:     {n_total - n_unknown}  ({100 * (1 - n_unknown / n_total):.2f}%)")
    print(f"  unknown:     {n_unknown}  ({100 * n_unknown / n_total:.2f}%)")
    print(f"  distinct labels (excl Unknown): {n_distinct}")

    # --- Write ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.out}")
    adata.write_h5ad(args.out, compression=None)
    print(f"Done.")


if __name__ == "__main__":
    main()
