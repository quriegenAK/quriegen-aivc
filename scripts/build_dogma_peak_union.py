"""Build DOGMA LLL+DIG unified peak set, re-emit h5ads in union space.

Spec section 3 requires a single shared ATAC encoder over both arms
with lysis_protocol as a covariate. Path A inspection showed LLL/DIG
peaks are nearly disjoint (66 of ~323k shared). This script unifies
them via deduplicated set union, then re-indexes per-arm ATAC counts
into the union dim with zero-fill at peaks the arm did not call.

Output:
- {out_lll, out_dig}: h5ads with .obsm['atac_peaks'] in union dim,
  .uns['atac_feature_names'] = canonical union peak strings,
  .uns['atac_peak_set'] = 'union_lll_dig'.
- manifest.json: peak counts + shared/unique fractions for diagnostics.

Usage:
    python scripts/build_dogma_peak_union.py \\
        --lll data/phase6_5g_2/dogma_h5ads/dogma_lll.h5ad \\
        --dig data/phase6_5g_2/dogma_h5ads/dogma_dig.h5ad \\
        --out-lll data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad \\
        --out-dig data/phase6_5g_2/dogma_h5ads/dogma_dig_union.h5ad \\
        --manifest data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


# Same flexible PEAK_RE as PR #38 (accepts both `_` and `:` separators
# and unplaced contigs like GL000194.1).
PEAK_RE = re.compile(r"^(?P<chrom>[\w.]+)[_:](?P<start>\d+)[_-](?P<end>\d+)$")


def parse_peaks(feat_names) -> list[tuple[str, int, int]]:
    """Parse peak strings to (chrom, start, end) tuples."""
    out = []
    for name in feat_names:
        m = PEAK_RE.match(str(name))
        if m is None:
            raise ValueError(f"unparseable peak: {name!r}")
        out.append((m.group("chrom"), int(m.group("start")), int(m.group("end"))))
    return out


def build_union_peaks(arm_peaks_list):
    """Build union peak set + per-arm local->global index maps.

    Returns
    -------
    union_sorted : list[tuple[str, int, int]]
        Deduplicated, deterministically sorted (chrom, start, end) tuples.
    per_arm_maps : list[np.ndarray]
        For each input arm, an int64 array mapping arm-local idx to union idx.
    """
    union_set = set()
    for arm_peaks in arm_peaks_list:
        union_set.update(arm_peaks)
    union_sorted = sorted(union_set, key=lambda x: (x[0], x[1], x[2]))
    pos = {peak: idx for idx, peak in enumerate(union_sorted)}
    per_arm_maps = [
        np.asarray([pos[p] for p in arm_peaks], dtype=np.int64)
        for arm_peaks in arm_peaks_list
    ]
    return union_sorted, per_arm_maps


def project_to_union(X_arm, idx_map: np.ndarray, n_union: int) -> np.ndarray:
    """Scatter arm-local counts into union dim with zero-fill.

    X_arm : (n_cells, n_arm_peaks). idx_map : (n_arm_peaks,) local->global.
    Returns dense (n_cells, n_union).
    """
    if sp.issparse(X_arm):
        X_dense = X_arm.toarray()
    else:
        X_dense = np.asarray(X_arm)
    n_cells = X_dense.shape[0]
    X_union = np.zeros((n_cells, n_union), dtype=X_dense.dtype)
    X_union[:, idx_map] = X_dense
    return X_union


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--lll", required=True, type=Path)
    p.add_argument("--dig", required=True, type=Path)
    p.add_argument("--out-lll", required=True, type=Path)
    p.add_argument("--out-dig", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    args = p.parse_args()

    print(f"Loading LLL: {args.lll}")
    lll = ad.read_h5ad(args.lll)
    print(f"Loading DIG: {args.dig}")
    dig = ad.read_h5ad(args.dig)

    lll_peaks = parse_peaks(list(lll.uns["atac_feature_names"]))
    dig_peaks = parse_peaks(list(dig.uns["atac_feature_names"]))

    print(f"LLL peaks: {len(lll_peaks)}")
    print(f"DIG peaks: {len(dig_peaks)}")

    union, [lll_map, dig_map] = build_union_peaks([lll_peaks, dig_peaks])
    n_union = len(union)
    n_shared = len(set(lll_peaks) & set(dig_peaks))
    print(f"Union: {n_union}  Shared: {n_shared}")

    union_feat_names = np.asarray(
        [f"{c}:{s}-{e}" for (c, s, e) in union], dtype=object
    )

    print("Projecting LLL ATAC -> union...")
    lll.obsm["atac_peaks"] = project_to_union(lll.obsm["atac_peaks"], lll_map, n_union)
    lll.uns["atac_feature_names"] = union_feat_names
    lll.uns["atac_peak_set"] = "union_lll_dig"
    args.out_lll.parent.mkdir(parents=True, exist_ok=True)
    lll.write_h5ad(args.out_lll, compression="gzip")
    print(f"Wrote {args.out_lll}")

    print("Projecting DIG ATAC -> union...")
    dig.obsm["atac_peaks"] = project_to_union(dig.obsm["atac_peaks"], dig_map, n_union)
    dig.uns["atac_feature_names"] = union_feat_names
    dig.uns["atac_peak_set"] = "union_lll_dig"
    dig.write_h5ad(args.out_dig, compression="gzip")
    print(f"Wrote {args.out_dig}")

    manifest = {
        "n_lll_peaks": int(len(lll_peaks)),
        "n_dig_peaks": int(len(dig_peaks)),
        "n_union_peaks": int(n_union),
        "n_shared": int(n_shared),
        "frac_lll_unique": float((len(lll_peaks) - n_shared) / len(lll_peaks)),
        "frac_dig_unique": float((len(dig_peaks) - n_shared) / len(dig_peaks)),
        "out_lll": str(args.out_lll),
        "out_dig": str(args.out_dig),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
