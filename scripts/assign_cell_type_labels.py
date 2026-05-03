"""Assign cell-type labels to DOGMA h5ad via protein-marker gating.

Path B architecture pivot (post-PR #51 E2-NULL): supervised contrastive
pretraining needs per-cell cell_type labels. DOGMA-seq has 210-D
TotalSeq-A surface markers in obsm['protein']; canonical lineage gating
produces labels without external sources.

CLR normalization: log(x+1) - mean(log(x+1)) per cell. Thresholds at 0.5
on CLR scale. Rules applied in order -- first match wins; unmatched
cells get "Unknown".

Usage:
    python scripts/assign_cell_type_labels.py \\
        --h5ad data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad \\
        --out data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad

Success criteria:
    - Unknown fraction < 20%
    - At least 6 distinct cell types assigned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np


# Gating rules (locked per PR #53 spec). Applied in order; first match wins.
GATING_RULES = {
    "CD4_T":         {"pos": ["CD3-1", "CD4-1"],                  "neg": ["CD8", "CD19", "CD56(NCAM)"]},
    "CD8_T":         {"pos": ["CD3-1", "CD8", "CD8a"],            "neg": ["CD4-1", "CD19"]},
    "NK":            {"pos": ["CD56(NCAM)"],                      "neg": ["CD3-1", "CD19"]},
    "B":             {"pos": ["CD19"],                            "neg": ["CD3-1", "CD56(NCAM)"]},
    "Monocyte_CD14": {"pos": ["CD14"],                            "neg": ["CD3-1", "CD19", "CD56(NCAM)"]},
    "Monocyte_CD16": {"pos": ["CD16"],                            "neg": ["CD3-1", "CD19", "CD14"]},
    "DC":            {"pos": ["CD11c", "HLA-DR"],                 "neg": ["CD3-1", "CD19", "CD14", "CD56(NCAM)"]},
    "Treg":          {"pos": ["CD3-1", "CD4-1", "CD25", "CD127"], "neg": ["CD8"]},
    "Naive_CD4":     {"pos": ["CD3-1", "CD4-1", "CD45RA"],        "neg": ["CD8", "CD45RO"]},
    "Memory_CD4":    {"pos": ["CD3-1", "CD4-1", "CD45RO"],        "neg": ["CD8", "CD45RA"]},
    "Naive_CD8":     {"pos": ["CD3-1", "CD8", "CD45RA"],          "neg": ["CD4-1", "CD45RO"]},
    "Memory_CD8":    {"pos": ["CD3-1", "CD8", "CD45RO"],          "neg": ["CD4-1", "CD45RA"]},
}


def clr_normalize(adt_matrix: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Centered log-ratio per cell: log(x+eps) - mean(log(x+eps)) per row.

    Rows sum to ~0 by construction. Stable for low-count antibodies.

    Pre-normalization detection: if the input contains any negative values,
    we assume it has already been CLR-normalized at write time (DOGMA h5ad
    convention; see assemble_dogma_h5ad.py) and pass through unchanged.
    Re-applying log(x+1) on already-CLR data produces NaN for x < -1
    (cf. PR #53 production failure 2026-05-03: 9415/13763 cells -> Unknown
    via NaN propagation).
    """
    arr = np.asarray(adt_matrix, dtype=np.float64)
    if (arr < 0).any():
        n_neg = int((arr < 0).sum())
        total = arr.size
        print(
            f"  [clr_normalize] input already contains {n_neg}/{total} negative values "
            f"(min={arr.min():.4f}, max={arr.max():.4f}). "
            f"Assuming pre-CLR-normalized; skipping in-script CLR.",
            file=sys.stderr,
        )
        return arr
    log_x = np.log(arr + eps)
    per_cell_mean = log_x.mean(axis=1, keepdims=True)
    return log_x - per_cell_mean


def build_marker_index(antibody_names: list[str]) -> dict[str, int]:
    """Map antibody name -> column index in the protein matrix."""
    return {name: i for i, name in enumerate(antibody_names)}


def assign_labels(
    clr_matrix: np.ndarray,
    antibody_names: list[str],
    rules: dict[str, dict[str, list[str]]] = GATING_RULES,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply gating rules in order; first match wins; unmatched -> 'Unknown'.

    Parameters
    ----------
    clr_matrix : (n_cells, n_proteins) CLR-normalized protein values.
    antibody_names : column names matching clr_matrix.shape[1].
    rules : ordered dict of {label: {"pos": [marker, ...], "neg": [marker, ...]}}.
    threshold : CLR threshold for marker-positive call.

    Returns
    -------
    labels : (n_cells,) numpy str array.
    """
    n_cells = clr_matrix.shape[0]
    marker_idx = build_marker_index(antibody_names)
    labels = np.full(n_cells, "Unknown", dtype=object)
    assigned = np.zeros(n_cells, dtype=bool)

    for cell_type, rule in rules.items():
        all_markers = rule["pos"] + rule["neg"]
        missing = [m for m in all_markers if m not in marker_idx]
        if missing:
            print(f"  [skip rule {cell_type}] missing markers: {missing}",
                  file=sys.stderr)
            continue

        pos_idx = [marker_idx[m] for m in rule["pos"]]
        neg_idx = [marker_idx[m] for m in rule["neg"]]

        pos_mask = np.all(clr_matrix[:, pos_idx] > threshold, axis=1)
        neg_mask = np.all(clr_matrix[:, neg_idx] <= threshold, axis=1)
        rule_match = pos_mask & neg_mask & ~assigned

        labels[rule_match] = cell_type
        assigned |= rule_match

    return labels


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--h5ad", required=True, type=Path,
                   help="Input DOGMA h5ad with obsm['protein'] + uns['adt_antibody_names']")
    p.add_argument("--out", required=True, type=Path,
                   help="Output h5ad with obs['cell_type'] column added")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="CLR threshold for marker-positive call (default 0.5)")
    args = p.parse_args()

    print(f"Loading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")

    if "protein" not in adata.obsm:
        raise ValueError(f"obsm['protein'] missing from {args.h5ad}")
    if "adt_antibody_names" not in adata.uns:
        raise ValueError(f"uns['adt_antibody_names'] missing from {args.h5ad}")

    adt = np.asarray(adata.obsm["protein"])
    antibody_names = [str(n) for n in adata.uns["adt_antibody_names"]]
    print(f"  ADT shape: {adt.shape}, n_antibodies: {len(antibody_names)}")
    if adt.shape[1] != len(antibody_names):
        raise ValueError(
            f"protein dim {adt.shape[1]} != n_antibodies {len(antibody_names)}"
        )

    print(
        f"  ADT input stats: min={adt.min():.4f}, max={adt.max():.4f}, "
        f"mean={adt.mean():.4f}, median={np.median(adt):.4f}"
    )

    print(f"CLR-normalizing protein matrix (or passing through if pre-CLR)...")
    clr = clr_normalize(adt)

    n_nan = int(np.isnan(clr).sum())
    n_inf = int(np.isinf(clr).sum())
    if n_nan > 0 or n_inf > 0:
        raise ValueError(
            f"Post-CLR matrix has {n_nan} NaN and {n_inf} Inf values "
            f"-- gating will silently send affected cells to Unknown. "
            f"Investigate input distribution before re-running."
        )
    print(
        f"  Post-CLR stats: min={clr.min():.4f}, max={clr.max():.4f}, "
        f"mean={clr.mean():.4f}, median={np.median(clr):.4f}"
    )

    print(f"Applying {len(GATING_RULES)} gating rules at threshold={args.threshold}")
    labels = assign_labels(clr, antibody_names, threshold=args.threshold)

    adata.obs["cell_type"] = labels.astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    n_total = len(labels)
    print(f"\n=== Label distribution ===")
    label_counts = adata.obs["cell_type"].value_counts(dropna=False)
    for lbl, cnt in label_counts.items():
        pct = 100 * cnt / n_total
        print(f"  {lbl:<18s}: {cnt:>6d}  ({pct:.1f}%)")

    n_unknown = int((labels == "Unknown").sum())
    n_assigned = n_total - n_unknown
    distinct_labels = [l for l in label_counts.index if l != "Unknown"]
    n_distinct = len(distinct_labels)
    frac_unknown = n_unknown / n_total

    print(f"\n=== Summary ===")
    print(f"  total cells: {n_total}")
    print(f"  assigned: {n_assigned}  (frac: {1-frac_unknown:.4f})")
    print(f"  unknown: {n_unknown}    (frac: {frac_unknown:.4f})")
    print(f"  distinct labels: {n_distinct}")
    print(f"  success criteria: frac_unknown < 0.20 = {'PASS' if frac_unknown < 0.20 else 'FAIL'}")
    print(f"                   distinct >= 6      = {'PASS' if n_distinct >= 6 else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.out}")
    adata.write_h5ad(args.out, compression=None)
    print(f"Done.")


if __name__ == "__main__":
    main()
