"""Stage 3 prep — backfill DOGMA union peak list into UNION_MANIFEST.json.

CONTEXT:
The current UNION_MANIFEST.json at data/phase6_5g_2/dogma_h5ads/ is a
311-byte summary file (counts + h5ad pointers) that lacks the actual
peak-string list. Downstream consumers (prepare_mimitou_crispr.py,
project_calderon_to_dogma_space helpers) expect a `union_peaks` key
inside the manifest. This helper extracts the peak list from one of
the labeled union h5ads and writes an enriched manifest that satisfies
that contract.

WHY OPTION 2 (enrich manifest) OVER OPTION 1 (patch prep to read h5ad):
Cleaner separation. The manifest's job is to be the canonical pointer
for the union peak set; consumers should never need to know whether
the peak list lives in an h5ad's var_names, obsm metadata, or uns.

PEAK-LIST SOURCE PRIORITY (try in order, first hit wins):
  1. adata.uns['atac_feature_names']     ← most likely (prep scripts write here)
  2. adata.uns['union_peaks']             ← alternate convention
  3. var_names where var.feature_types == 'Peaks'  ← Cell Ranger ARC mixed-modality
  4. var_names whose strings parse as peak format  ← format-based fallback

SANITY CHECK:
After extracting the peak list, length must match the existing manifest's
`n_union_peaks` field. If they differ → ABORT and print both sides for
investigation. We do NOT silently overwrite a contract value.

OUTPUT:
Writes the manifest IN PLACE (keeps original keys + adds union_peaks).
The original counts + h5ad pointers are preserved so downstream consumers
that already read those fields don't break.

Usage (BSC, single-shot — only needs to run once per union peak rebuild):
    cd /gpfs/scratch/ehpc748/quri020505/aivc_genelink
    python scripts/build_union_manifest.py \\
        --labeled_h5ad data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad \\
        --manifest data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad


# Same peak-string regex used by build_dogma_peak_union.py and the
# Mimitou prep script. Accepts "chr1:1234-5678" or "chr1_1234_5678".
PEAK_RE = re.compile(r"^(?P<chrom>[\w.]+)[_:](?P<start>\d+)[_-](?P<end>\d+)$")


def looks_like_peak(s: str) -> bool:
    return PEAK_RE.match(str(s)) is not None


def extract_peak_list(adata: ad.AnnData) -> tuple[list[str], str]:
    """Try the four sources in priority order. Return (peak_list, source_label)."""
    # Source 1: uns['atac_feature_names']
    if "atac_feature_names" in adata.uns:
        peaks = list(adata.uns["atac_feature_names"])
        # numpy array of bytes is possible — coerce to str
        peaks = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                 for s in peaks]
        return peaks, "uns['atac_feature_names']"

    # Source 2: uns['union_peaks']
    if "union_peaks" in adata.uns:
        peaks = list(adata.uns["union_peaks"])
        peaks = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                 for s in peaks]
        return peaks, "uns['union_peaks']"

    # Source 3: var_names where feature_types == 'Peaks' (CR-ARC convention)
    if "feature_types" in adata.var.columns:
        is_peak = adata.var["feature_types"].astype(str) == "Peaks"
        if is_peak.any():
            peaks = adata.var_names[is_peak].astype(str).tolist()
            return peaks, "var_names[feature_types=='Peaks']"

    # Source 4: format-based fallback — any var_name that looks like a peak
    candidates = [s for s in adata.var_names.astype(str).tolist()
                  if looks_like_peak(s)]
    if candidates:
        return candidates, "var_names matching PEAK_RE"

    raise ValueError(
        "Could not locate the union peak list in any expected place:\n"
        "  - uns['atac_feature_names']: missing\n"
        "  - uns['union_peaks']: missing\n"
        "  - var with feature_types=='Peaks': missing or empty\n"
        "  - var_names matching PEAK_RE: zero matches\n\n"
        f"Available uns keys: {list(adata.uns.keys())}\n"
        f"Available var columns: {list(adata.var.columns)}\n"
        f"First 5 var_names: {adata.var_names[:5].tolist()}"
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--labeled_h5ad", required=True, type=Path,
                   help="Source labeled union h5ad "
                        "(e.g., dogma_lll_union_labeled.h5ad — smaller, "
                        "faster than dogma_dig_union_labeled).")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to UNION_MANIFEST.json — modified in place.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be written; don't modify the manifest.")
    args = p.parse_args()

    if not args.labeled_h5ad.exists():
        raise FileNotFoundError(f"--labeled_h5ad not found: {args.labeled_h5ad}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"--manifest not found: {args.manifest}")

    # --- 1. Read existing manifest ---
    print(f"Reading existing manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"  current keys: {sorted(manifest.keys())}")
    expected_n = manifest.get("n_union_peaks")
    if expected_n is None:
        print("  WARN: manifest has no 'n_union_peaks' field — sanity check "
              "will be skipped (cannot detect peak count drift).")
    else:
        print(f"  n_union_peaks (expected): {expected_n}")

    if "union_peaks" in manifest:
        print(f"  manifest already has 'union_peaks' "
              f"(len={len(manifest['union_peaks'])}); will overwrite.")

    # --- 2. Load labeled h5ad (use backed mode if very large) ---
    print(f"\nLoading labeled h5ad (this can take ~30-60s for the full file): "
          f"{args.labeled_h5ad}")
    # backed='r' avoids materializing X into memory — we only need uns/var
    adata = ad.read_h5ad(args.labeled_h5ad, backed="r")
    print(f"  shape: {adata.shape}")

    # --- 3. Extract peak list ---
    peaks, source = extract_peak_list(adata)
    print(f"\nPeak list extracted from: {source}")
    print(f"  count: {len(peaks)}")
    print(f"  first 3: {peaks[:3]}")
    print(f"  last 3:  {peaks[-3:]}")

    # Validate format
    bad = [s for s in peaks if not looks_like_peak(s)]
    if bad:
        print(f"  WARN: {len(bad)}/{len(peaks)} peaks fail PEAK_RE format check.")
        print(f"  First 5 unparseable: {bad[:5]}")
        # Continue anyway — downstream may handle, or this surfaces a bug

    # --- 4. Sanity: count matches manifest's n_union_peaks ---
    if expected_n is not None and len(peaks) != expected_n:
        raise ValueError(
            f"PEAK COUNT MISMATCH:\n"
            f"  manifest['n_union_peaks'] = {expected_n}\n"
            f"  extracted from {source} = {len(peaks)}\n"
            f"\n"
            f"This is a hard failure — one source is wrong. Investigate before "
            f"writing. Possible causes:\n"
            f"  (a) labeled h5ad was rebuilt with different union; manifest stale\n"
            f"  (b) extraction source is wrong (e.g., CR-ARC h5ad has both peaks\n"
            f"      and genes in var; we may have grabbed the wrong subset)\n"
            f"  (c) manifest's n_union_peaks was set incorrectly at build time"
        )
    print(f"\n  sanity check: peak count matches manifest's n_union_peaks ✓")

    # --- 5. Backfill + write ---
    manifest["union_peaks"] = peaks
    manifest["union_peaks_source_h5ad"] = str(args.labeled_h5ad)
    manifest["union_peaks_source_field"] = source

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(peaks)} peaks into {args.manifest}")
        print(f"  enriched manifest keys: {sorted(manifest.keys())}")
        return

    args.manifest.write_text(json.dumps(manifest, indent=2))
    new_size = args.manifest.stat().st_size
    print(f"\nWrote enriched manifest ({new_size} bytes; was 311-ish).")
    print(f"  added: union_peaks ({len(peaks)} entries)")
    print(f"  added: union_peaks_source_h5ad")
    print(f"  added: union_peaks_source_field")
    print(f"  preserved: {sorted(k for k in manifest.keys() if k not in {'union_peaks', 'union_peaks_source_h5ad', 'union_peaks_source_field'})}")


if __name__ == "__main__":
    main()
