"""Concatenate LLL+DIG labeled h5ads into a single joint h5ad for SupCon training.

PR #54 will load `dogma_joint_labeled.h5ad` (31,874 cells = 13,763 LLL +
18,111 DIG) with both `cell_type` and `lysis_protocol` columns intact.

Sanity checks:
  - var (gene set) must match between arms
  - obsm keys present in both arms must have matching dim-1 (peak/protein
    panel sizes — must already be union-harmonized per PR #41)
  - no obs index collisions (LLL barcodes use suffix -1/-2; DIG -3/-4 per
    code/15_LLLandDIG.R — preserved from assemble_dogma_h5ad.py)
  - lysis_protocol values are consistent (LLL→'LLL', DIG→'DIG')

Usage:
    python scripts/merge_dogma_labeled_arms.py \\
        --lll data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad \\
        --dig data/phase6_5g_2/dogma_h5ads/dogma_dig_union_labeled.h5ad \\
        --out data/phase6_5g_2/dogma_h5ads/dogma_joint_labeled.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--lll", required=True, type=Path)
    p.add_argument("--dig", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    print(f"Loading LLL: {args.lll}")
    lll = ad.read_h5ad(args.lll)
    print(f"  shape: {lll.shape}")
    print(f"  obs cols: {list(lll.obs.columns)}")
    print(f"  obsm keys: {list(lll.obsm.keys())}")

    print(f"\nLoading DIG: {args.dig}")
    dig = ad.read_h5ad(args.dig)
    print(f"  shape: {dig.shape}")
    print(f"  obs cols: {list(dig.obs.columns)}")
    print(f"  obsm keys: {list(dig.obsm.keys())}")

    # --- Pre-merge sanity ---
    print("\n=== Pre-merge sanity checks ===")
    if not lll.var_names.equals(dig.var_names):
        n_overlap = len(set(lll.var_names) & set(dig.var_names))
        raise ValueError(
            f"var_names mismatch: LLL n_var={lll.n_vars}, DIG n_var={dig.n_vars}, "
            f"overlap={n_overlap}. Re-run assemble_dogma_h5ad.py with same gene set."
        )
    print(f"  var_names: identical ({lll.n_vars} genes) ✓")

    # var_names duplicate handling: Cell Ranger collapses some Ensembl IDs to the
    # same gene symbol. anndata.concat with join='outer' reindexes var across
    # objects and pandas refuses to reindex non-unique indices. Both arms came
    # from the same assembly so var_names_make_unique() produces identical
    # suffixed names on both sides — safe to call.
    n_dup_before = int(lll.var_names.duplicated().sum())
    if n_dup_before > 0:
        print(f"  var_names has {n_dup_before} duplicates; "
              f"calling var_names_make_unique() on both arms")
        lll.var_names_make_unique()
        dig.var_names_make_unique()
        if not lll.var_names.equals(dig.var_names):
            raise ValueError(
                "var_names_make_unique() produced divergent names between arms. "
                "Should be impossible if both came from same assembly."
            )
        print(f"  post-uniquification: var_names still identical ({lll.n_vars} genes) ✓")

    common_obsm = set(lll.obsm.keys()) & set(dig.obsm.keys())
    for key in sorted(common_obsm):
        l_shape = lll.obsm[key].shape
        d_shape = dig.obsm[key].shape
        if l_shape[1] != d_shape[1]:
            raise ValueError(
                f"obsm['{key}'] dim-1 mismatch: LLL={l_shape}, DIG={d_shape}"
            )
        print(f"  obsm['{key}']: LLL={l_shape}, DIG={d_shape} (dim-1 match) ✓")

    # Index collisions
    common_bcs = set(lll.obs.index) & set(dig.obs.index)
    if common_bcs:
        raise ValueError(
            f"{len(common_bcs)} barcode collisions between LLL and DIG. "
            f"Sample: {list(common_bcs)[:3]}"
        )
    print(f"  no barcode collisions ✓")

    # lysis_protocol expected values
    lll_lyses = set(lll.obs["lysis_protocol"].astype(str).unique())
    dig_lyses = set(dig.obs["lysis_protocol"].astype(str).unique())
    print(f"  LLL lysis_protocol: {lll_lyses}")
    print(f"  DIG lysis_protocol: {dig_lyses}")
    if lll_lyses != {"LLL"}:
        print(f"  WARNING: LLL h5ad has unexpected lysis_protocol values {lll_lyses}")
    if dig_lyses != {"DIG"}:
        print(f"  WARNING: DIG h5ad has unexpected lysis_protocol values {dig_lyses}")

    # cell_type presence
    if "cell_type" not in lll.obs.columns:
        raise ValueError("LLL missing obs['cell_type']")
    if "cell_type" not in dig.obs.columns:
        raise ValueError("DIG missing obs['cell_type']")

    # --- Concatenate ---
    print("\n=== Concatenating ===")
    joint = ad.concat(
        {"LLL": lll, "DIG": dig},
        axis=0,
        join="outer",
        merge="first",     # for var-level metadata
        uns_merge="first", # keep first uns; DIG-specific manifest dropped (acceptable)
        label="arm",       # adds obs['arm'] = LLL or DIG
        index_unique=None, # barcodes are already disjoint per pre-merge check
    )
    print(f"  joint shape: {joint.shape}")
    print(f"  joint obs cols: {list(joint.obs.columns)}")
    print(f"  joint obsm keys: {list(joint.obsm.keys())}")

    # --- Post-merge sanity ---
    if joint.n_obs != lll.n_obs + dig.n_obs:
        raise ValueError(
            f"obs count mismatch after concat: "
            f"{joint.n_obs} != {lll.n_obs} + {dig.n_obs}"
        )

    # cell_type distribution by arm
    print("\n=== Joint cell_type × arm cross-tab ===")
    ctab = pd.crosstab(joint.obs["cell_type"], joint.obs["arm"], margins=True)
    print(ctab.to_string())

    # lysis_protocol sanity
    print("\n=== Joint lysis_protocol distribution ===")
    print(joint.obs["lysis_protocol"].value_counts().to_string())

    # cell_type_source if present
    if "cell_type_source" in joint.obs.columns:
        print("\n=== cell_type_source counts ===")
        print(joint.obs["cell_type_source"].value_counts().to_string())

    # --- Write ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.out}")
    joint.write_h5ad(args.out, compression=None)
    print("Done.")


if __name__ == "__main__":
    main()
