"""Build a canonical {cell_type: int} index manifest for SupCon training.

Both DOGMA arms (LLL + DIG) MUST share an identical integer encoding
for cell_type so that SupCon's same-class positive set is consistent
across the joint batch. This script unions all observed cell_type
labels from both labeled h5ads, sorts deterministically, and writes
a JSON manifest with the int<->name mapping plus a SHA-256 fingerprint
so PR #54c eval can verify the encoding is unchanged at inference time.

Usage:
    python scripts/build_class_index_manifest.py \\
        --h5ads data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad \\
                data/phase6_5g_2/dogma_h5ads/dogma_dig_union_labeled.h5ad \\
        --out data/phase6_5g_2/dogma_h5ads/cell_type_index.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--h5ads", nargs="+", required=True, type=Path,
                   help="One or more labeled h5ads with obs['cell_type']")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSON manifest path")
    p.add_argument("--label_col", default="cell_type",
                   help="obs column name with class labels (default cell_type)")
    args = p.parse_args()

    all_classes: set[str] = set()
    per_file_dist: dict[str, dict[str, int]] = {}
    for h5ad_path in args.h5ads:
        if not h5ad_path.exists():
            print(f"ERROR: {h5ad_path} does not exist", file=sys.stderr)
            sys.exit(2)
        print(f"Loading {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        if args.label_col not in adata.obs.columns:
            raise ValueError(
                f"obs[{args.label_col!r}] missing from {h5ad_path}. "
                f"Available: {list(adata.obs.columns)}"
            )
        classes = adata.obs[args.label_col].astype(str).unique().tolist()
        print(f"  {h5ad_path.name}: {len(adata.obs)} cells, "
              f"{len(classes)} distinct classes: {sorted(classes)}")
        all_classes.update(classes)
        per_file_dist[str(h5ad_path)] = {
            cls: int(cnt) for cls, cnt in
            adata.obs[args.label_col].astype(str).value_counts().items()
        }

    # Deterministic sort = lexicographic. Reserve "Unknown" at index 0
    # if present (so masking via `== 0` is convenient downstream).
    sorted_classes = sorted(all_classes)
    if "Unknown" in sorted_classes:
        sorted_classes.remove("Unknown")
        sorted_classes = ["Unknown"] + sorted_classes

    class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    # Fingerprint over ordered class list (the load-bearing contract)
    fingerprint_payload = "|".join(f"{i}:{cls}" for i, cls in idx_to_class.items())
    fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()

    manifest = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(i): cls for i, cls in idx_to_class.items()},
        "n_classes": len(class_to_idx),
        "source_h5ads": [str(p) for p in args.h5ads],
        "fingerprint_sha256": fingerprint,
        "label_column": args.label_col,
        "per_file_distribution": per_file_dist,
        "sort_order": "lexicographic, with 'Unknown' pinned to index 0 if present",
        "build_date_utc": "2026-05-03",
        "consumer_contract": (
            "Loaders, losses, and eval that depend on cell_type integer codes "
            "MUST verify fingerprint_sha256 matches the one stamped at training "
            "time. Mismatch == retrain or hard error."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print()
    print(f"Wrote manifest to {args.out}")
    print(f"  n_classes: {len(class_to_idx)}")
    print(f"  fingerprint_sha256: {fingerprint}")
    print(f"  class_to_idx:")
    for cls, i in class_to_idx.items():
        print(f"    {i:>2d}: {cls}")


if __name__ == "__main__":
    main()
