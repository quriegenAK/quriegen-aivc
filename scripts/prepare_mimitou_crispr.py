"""Stage 3 Part 1 Report 2 prep — harmonize Mimitou CD4 CRISPR ASAP-seq
data into a DOGMA-encoder-compatible labeled h5ad.

Mimitou 2021 CD4 CRISPR arm (GSE156478, CD4_CRISPR_asapseq sub-study):
  - ~5,825 perturbed cells from primary human CD4+ T cells
  - 4 single CRISPR-Cas9 KO conditions: CD3E, CD4, ZAP70, NFKB2
  - 1 double KO: CD3E + CD4
  - Plus NTC (non-targeting control) gRNAs as vehicle equivalent
  - Anti-CD3/CD28 stim post-perturbation, 16h endpoint
  - ASAP-seq output: ATAC peaks + 27-protein TotalSeq-A panel + hashtag

This script ingests the asap_reproducibility processed outputs and
produces a labeled h5ad whose obsm['atac_peaks'] is in the DOGMA union
peak space (323,500 peaks). The output is directly consumable by the
validated DOGMA encoder via the same encode_samples pipeline used for
Calderon eval and the Test 1.5 pseudo-bulk centroid-NN.

Output schema (matches dogma_lll_union_labeled.h5ad shape contract):
  obs['cell_type']        = 'CD4_T' for all (CRISPR is on sorted CD4+ T cells)
  obs['perturbation']     = NTC | CD3E | CD4 | ZAP70 | NFKB2 | CD3E_CD4_double
  obs['lysis_protocol']   = 'LLL' (CRISPR arm uses LLL protocol per paper)
  obs['guide_RNA']        = parsed guide identity (sgRNA1, sgRNA2, etc.)
  obs['donor_id']         = donor_X (for held-out replicate eval)
  obsm['atac_peaks']      = (n_cells, 323_500) CSR — DOGMA union peak space
  obsm['protein']         = (n_cells, n_ab)    — TotalSeq-A panel
  uns['source']           = 'mimitou_2021_cd4_crispr_GSE156478'
  uns['perturbation_categories'] = ['NTC', 'CD3E', 'CD4', 'ZAP70', 'NFKB2', 'CD3E_CD4_double']

Usage (BSC):
  python scripts/prepare_mimitou_crispr.py \\
      --mimitou_crispr_dir /gpfs/scratch/.../mimitou_crispr_raw/ \\
      --union_manifest data/phase6_5g_2/dogma_h5ads/UNION_MANIFEST.json \\
      --out data/phase6_5g_2/dogma_h5ads/mimitou_crispr_union.h5ad

Expected input directory contents (produced by
scripts/download_mimitou_crispr.sh from caleblareau/asap_reproducibility):
  filtered_peak_bc_matrix.h5             # cellranger ATAC output
  adt/featurecounts.mtx                  # kite-format ADT (protein) counts
  adt/featurecounts.barcodes.txt
  adt/featurecounts.genes.txt
  hto/featurecounts.mtx                  # kite-format HTO counts
                                         # (encodes per-cell sgRNA target via
                                         #  TotalSeq hashtag multiplexing)
  hto/featurecounts.barcodes.txt
  hto/featurecounts.genes.txt
  hashtag_list.txt                       # HTO → sgRNA target map (reference)

Per-cell perturbation labels are DERIVED from the HTO matrix using a
top-target-HTO call with min-count threshold + CD3E+CD4 double-detection
rule (see derive_perturbation_from_hto). There is no per-cell guide CSV
in the source data — the experimental design encodes target identity in
the HTO reads themselves.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# Re-use the peak-string regex from build_dogma_peak_union.py to keep
# parsing consistent with the production union manifest.
PEAK_RE = re.compile(r"^(?P<chrom>[\w.]+)[_:](?P<start>\d+)[_-](?P<end>\d+)$")


# NOTE: Guide-RNA assignments are NOT direct in Mimitou's design — they
# are encoded as TotalSeq HTO multiplexing (see derive_perturbation_from_hto
# below). The mapping table HTO → sgRNA target is in the data dir's
# kallisto/hashtag_list.txt (downloaded by scripts/download_mimitou_crispr.sh).


def parse_peak_string(s: str) -> Optional[tuple[str, int, int]]:
    m = PEAK_RE.match(str(s))
    if m is None:
        return None
    return (m.group("chrom"), int(m.group("start")), int(m.group("end")))


def load_atac_peaks(atac_h5_path: Path):
    """Load Cell Ranger ATAC filtered_peak_bc_matrix.h5.
    Returns (counts: csr_matrix, barcodes: list[str], peaks: list[(chrom,start,end)]).
    """
    import h5py

    print(f"Loading ATAC: {atac_h5_path}")
    with h5py.File(atac_h5_path, "r") as f:
        # Cell Ranger ATAC HDF5 layout: matrix/{data, indices, indptr, shape},
        # matrix/barcodes, matrix/features/{name, id, ...}
        if "matrix" not in f:
            raise ValueError(f"{atac_h5_path}: expected 'matrix' group "
                             f"(Cell Ranger ATAC h5 format)")
        mtx = f["matrix"]
        data = mtx["data"][:]
        indices = mtx["indices"][:]
        indptr = mtx["indptr"][:]
        shape = mtx["shape"][:]
        # Cell Ranger packs as (n_features, n_cells) — transpose to (cells, peaks)
        counts = sp.csr_matrix(
            (data, indices, indptr),
            shape=(int(shape[0]), int(shape[1])),
        ).T.tocsr()
        barcodes = [b.decode() for b in mtx["barcodes"][:]]
        # Peak names: in Cell Ranger ATAC, features/name are usually "chr:start-end"
        # but can also be at features/id. Try both.
        peak_names = []
        if "features" in mtx and "name" in mtx["features"]:
            peak_names = [n.decode() for n in mtx["features"]["name"][:]]
        elif "features" in mtx and "id" in mtx["features"]:
            peak_names = [n.decode() for n in mtx["features"]["id"][:]]
        else:
            raise ValueError(f"No features/name or features/id in {atac_h5_path}")
    peaks = [parse_peak_string(s) for s in peak_names]
    n_unparseable = sum(1 for p in peaks if p is None)
    if n_unparseable:
        raise ValueError(
            f"{n_unparseable}/{len(peaks)} peaks unparseable. "
            f"Sample bad: {[s for s, p in zip(peak_names, peaks) if p is None][:5]}"
        )
    print(f"  ATAC shape: {counts.shape}  ({len(barcodes)} cells × {len(peaks)} peaks)")
    return counts, barcodes, peaks


# load_protein_counts removed — use load_kite_mtx (defined below) for both
# the ADT (protein) and HTO matrices since they share the kite output format.


# --- HTO-based guide demux (Mimitou's actual experimental design) ---
#
# Mimitou 2021 CD4 CRISPR ASAP-seq encodes sgRNA target identity as
# TotalSeq HTO (hashtag oligo) reads, not direct guide capture. From
# kallisto/hashtag_list.txt:
#
#   Hashtag01 → sgNTC      (target HTO)
#   Hashtag02 → sgCD4      (target HTO)
#   Hashtag03 → sgNFKB2    (target HTO)
#   Hashtag04 → sgCD3E     (target HTO)
#   Hashtag05 → sgZAP70    (target HTO)
#   Hashtag12 → sgGuide1   (guide-variant HTO; one of the 2 sgRNA designs)
#   Hashtag13 → sgGuide2   (guide-variant HTO; the other sgRNA design)
#
# Per-cell expected pattern: 1 target HTO + 1 guide-variant HTO. The
# CD3E+CD4 double KO is created by intentional co-tagging — cells that
# receive BOTH Hashtag02 (sgCD4) AND Hashtag04 (sgCD3E) are the doubles.
# It is NOT a separate sgRNA construct.
#
# Calling rule (top-2-target with min-count threshold):
#   - Compute per-cell target-HTO counts (5 targets only)
#   - Top target HTO count must be >= MIN_TARGET_HTO_COUNT
#   - If top target / total_target >= SINGLE_DOMINANCE → single perturbation
#   - Else if top-2 are sgCD3E + sgCD4 (in either order) and their sum
#     dominates total target counts → CD3E_CD4_double
#   - Else: drop (multiplet of unhandled combination)

TARGET_HTO_TO_PERT = {
    "TotalSeq_Human_Hashtag01": "NTC",
    "TotalSeq_Human_Hashtag02": "CD4",
    "TotalSeq_Human_Hashtag03": "NFKB2",
    "TotalSeq_Human_Hashtag04": "CD3E",
    "TotalSeq_Human_Hashtag05": "ZAP70",
}

# Tuning knobs — defaults match conservative HTODemux convention.
# Lower thresholds = more cells kept at expense of label noise.
MIN_TARGET_HTO_COUNT = 5      # cell must have >=5 reads on top target HTO
SINGLE_DOMINANCE = 0.70       # top target / total_target >= 0.70 -> single
DOUBLE_PAIR_DOMINANCE = 0.85  # (top2_targets) / total_target >= 0.85 -> double


def load_kite_mtx(mtx_path: Path, barcodes_path: Path, features_path: Path):
    """Load kite-format featurecounts (matrix.mtx + barcodes + genes).
    Same loader signature as load_protein_counts for symmetry.
    """
    from scipy.io import mmread
    print(f"Loading kite mtx: {mtx_path}")
    mat = mmread(str(mtx_path)).tocsr()
    barcodes = pd.read_csv(barcodes_path, header=None)[0].astype(str).tolist()
    features = pd.read_csv(features_path, header=None)[0].astype(str).tolist()
    if mat.shape[0] == len(features) and mat.shape[1] == len(barcodes):
        mat = mat.T.tocsr()  # → (cells, features)
    print(f"  shape: {mat.shape} ({len(barcodes)} cells × {len(features)} features)")
    return mat, barcodes, features


def derive_perturbation_from_hto(
    hto_mtx, hto_barcodes, hto_features,
):
    """Call per-cell perturbation labels from HTO matrix using top-target
    + double-detection logic. Returns pd.Series indexed by barcode.

    Stages:
      1. Identify target-HTO columns via TARGET_HTO_TO_PERT mapping
      2. For each cell: take target-HTO counts, find top-1 and top-2
      3. Apply MIN_TARGET_HTO_COUNT + SINGLE_DOMINANCE thresholds
      4. CD3E+CD4 double rule (top-2 are exactly sgCD3E + sgCD4)
    """
    print(f"Deriving per-cell perturbation labels from HTO matrix...")
    print(f"  {len(hto_features)} HTO features detected:")
    for i, f in enumerate(hto_features):
        target = TARGET_HTO_TO_PERT.get(f, "(non-target)")
        print(f"    [{i}] {f}  → {target}")

    target_idx = [i for i, f in enumerate(hto_features) if f in TARGET_HTO_TO_PERT]
    target_pert = [TARGET_HTO_TO_PERT[hto_features[i]] for i in target_idx]
    if len(target_idx) < 5:
        print(f"  WARN: only {len(target_idx)} target HTOs found; expected 5")
        if len(target_idx) == 0:
            raise ValueError("No target HTOs match TARGET_HTO_TO_PERT keys. "
                             "HTO feature names may differ from expected.")

    # Per-cell target counts: shape (n_cells, n_target_HTOs)
    target_counts = np.asarray(hto_mtx[:, target_idx].todense())
    n_cells = target_counts.shape[0]

    # Sort each cell's targets by count descending
    sorted_idx = np.argsort(target_counts, axis=1)[:, ::-1]
    top1_count = target_counts[np.arange(n_cells), sorted_idx[:, 0]]
    top2_count = target_counts[np.arange(n_cells), sorted_idx[:, 1]] if target_counts.shape[1] > 1 else np.zeros(n_cells)
    total = target_counts.sum(axis=1)

    # Avoid division by zero
    total_safe = np.where(total > 0, total, 1)
    top1_share = top1_count / total_safe
    top2_share = (top1_count + top2_count) / total_safe

    # Classify
    labels = []
    pert_array = np.array(target_pert)
    n_dropped_low = 0
    n_dropped_multiplet = 0
    n_double = 0
    for i in range(n_cells):
        if top1_count[i] < MIN_TARGET_HTO_COUNT:
            labels.append(None)
            n_dropped_low += 1
            continue
        top1_pert = pert_array[sorted_idx[i, 0]]
        top2_pert = pert_array[sorted_idx[i, 1]] if target_counts.shape[1] > 1 else None

        # Single perturbation
        if top1_share[i] >= SINGLE_DOMINANCE:
            labels.append(top1_pert)
            continue

        # CD3E+CD4 double KO
        pair = tuple(sorted([top1_pert, top2_pert]))
        if pair == ("CD3E", "CD4") and top2_share[i] >= DOUBLE_PAIR_DOMINANCE:
            labels.append("CD3E_CD4_double")
            n_double += 1
            continue

        # Other multiplets — drop
        labels.append(None)
        n_dropped_multiplet += 1

    print(f"  HTO calling stats:")
    print(f"    cells low-count (<{MIN_TARGET_HTO_COUNT}):       {n_dropped_low}")
    print(f"    cells multiplet (other):           {n_dropped_multiplet}")
    print(f"    cells called CD3E_CD4_double:      {n_double}")
    print(f"    cells called single perturbation:  "
          f"{n_cells - n_dropped_low - n_dropped_multiplet - n_double}")

    series = pd.Series(labels, index=hto_barcodes, name="perturbation")
    return series.dropna()


def project_atac_to_union(atac_counts, atac_peaks, union_peaks):
    """Project arm-local ATAC peaks to the union peak set.

    atac_counts: (n_cells, n_arm_peaks) sparse CSR
    atac_peaks: list of (chrom, start, end) for arm-local columns
    union_peaks: list of (chrom, start, end) for union (e.g., 323,500)

    Returns: (n_cells, n_union) sparse CSR with zero-fill at peaks not in arm
    """
    print(f"Projecting ATAC: {atac_counts.shape[1]} arm-local peaks → "
          f"{len(union_peaks)} union peaks")
    union_idx = {p: i for i, p in enumerate(union_peaks)}
    # Build per-arm-local → union-global mapping; -1 = peak not in union
    local_to_union = np.array(
        [union_idx.get(p, -1) for p in atac_peaks], dtype=np.int64
    )
    n_in_union = int((local_to_union != -1).sum())
    print(f"  {n_in_union}/{len(atac_peaks)} arm peaks present in union "
          f"({100*n_in_union/len(atac_peaks):.1f}%)")
    if n_in_union == 0:
        raise ValueError("ZERO arm peaks overlap union — peak set mismatch. "
                         "Check that union manifest came from same hg38 reference.")

    # Filter arm-local peaks to those in union
    keep = local_to_union != -1
    arm_kept = atac_counts[:, keep]
    union_dest = local_to_union[keep]

    # Scatter into union space (sparse)
    n_cells = arm_kept.shape[0]
    n_union = len(union_peaks)
    # Convert to COO for scatter, then back to CSR
    arm_coo = arm_kept.tocoo()
    new_cols = union_dest[arm_coo.col]
    union_counts = sp.csr_matrix(
        (arm_coo.data, (arm_coo.row, new_cols)),
        shape=(n_cells, n_union),
    )
    return union_counts


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mimitou_crispr_dir", type=Path,
                   help="Directory containing the standard Mimitou CRISPR file layout. "
                        "Override with --atac_h5 etc. for non-standard layouts.")
    p.add_argument("--atac_h5", type=Path,
                   help="Cell Ranger ATAC filtered_peak_bc_matrix.h5")
    p.add_argument("--protein_mtx", type=Path,
                   help="kite-format protein count matrix (.mtx)")
    p.add_argument("--protein_barcodes", type=Path,
                   help="kite barcodes file (.txt or .tsv)")
    p.add_argument("--protein_features", type=Path,
                   help="kite feature names file (.txt or .tsv)")
    p.add_argument("--hto_mtx", type=Path,
                   help="kite-format HTO count matrix (encodes per-cell sgRNA target)")
    p.add_argument("--hto_barcodes", type=Path)
    p.add_argument("--hto_features", type=Path)
    p.add_argument("--donor_csv", type=Path, default=None,
                   help="Optional: per-cell donor_id CSV (barcode, donor_id). "
                        "If absent, all cells get donor_id='unknown'. "
                        "(Mimitou CD4 CRISPR sub-study is single-donor; "
                        "leaving unset is fine.)")
    p.add_argument("--union_manifest", required=True, type=Path,
                   help="DOGMA union peak manifest (UNION_MANIFEST.json or peaks.tsv)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output h5ad path")
    args = p.parse_args()

    # --- Resolve input paths ---
    # Default layout matches scripts/download_mimitou_crispr.sh output:
    #   <dir>/filtered_peak_bc_matrix.h5
    #   <dir>/adt/featurecounts.{mtx,barcodes.txt,genes.txt}
    #   <dir>/hto/featurecounts.{mtx,barcodes.txt,genes.txt}
    if args.mimitou_crispr_dir is not None:
        d = args.mimitou_crispr_dir
        args.atac_h5 = args.atac_h5 or d / "filtered_peak_bc_matrix.h5"
        args.protein_mtx = args.protein_mtx or d / "adt" / "featurecounts.mtx"
        args.protein_barcodes = args.protein_barcodes or d / "adt" / "featurecounts.barcodes.txt"
        args.protein_features = args.protein_features or d / "adt" / "featurecounts.genes.txt"
        args.hto_mtx = args.hto_mtx or d / "hto" / "featurecounts.mtx"
        args.hto_barcodes = args.hto_barcodes or d / "hto" / "featurecounts.barcodes.txt"
        args.hto_features = args.hto_features or d / "hto" / "featurecounts.genes.txt"
    for fld in ("atac_h5", "protein_mtx", "protein_barcodes",
                "protein_features", "hto_mtx", "hto_barcodes",
                "hto_features", "union_manifest"):
        path = getattr(args, fld)
        if path is None or not path.exists():
            raise FileNotFoundError(f"--{fld} not found: {path}")

    # --- Load union peak set ---
    print(f"Loading union peak manifest: {args.union_manifest}")
    if str(args.union_manifest).endswith(".json"):
        with open(args.union_manifest) as f:
            manifest = json.load(f)
        peak_strs = manifest.get("union_peaks") or manifest.get("atac_feature_names")
        if peak_strs is None:
            raise ValueError("Manifest JSON has no union_peaks / atac_feature_names key")
    else:
        # Assume TSV with chrom, start, end (or single column with peak strings)
        df = pd.read_csv(args.union_manifest, sep="\t", header=None)
        if df.shape[1] >= 3:
            peak_strs = [f"{r[0]}_{r[1]}_{r[2]}" for r in df.itertuples(index=False)]
        else:
            peak_strs = df[0].tolist()
    union_peaks = [parse_peak_string(s) for s in peak_strs]
    n_bad = sum(1 for p in union_peaks if p is None)
    if n_bad:
        raise ValueError(f"{n_bad} unparseable union peaks")
    print(f"  Union peak count: {len(union_peaks)}")

    # --- Load ATAC ---
    atac_counts, atac_barcodes, atac_peaks = load_atac_peaks(args.atac_h5)

    # --- Load Protein (kite ADT) ---
    prot_counts, prot_barcodes, prot_features = load_kite_mtx(
        args.protein_mtx, args.protein_barcodes, args.protein_features,
    )

    # --- Load HTO + derive perturbation labels ---
    hto_counts, hto_barcodes, hto_features = load_kite_mtx(
        args.hto_mtx, args.hto_barcodes, args.hto_features,
    )
    perturbation_labels = derive_perturbation_from_hto(
        hto_counts, hto_barcodes, hto_features,
    )
    print(f"\nPerturbation distribution (across {len(perturbation_labels)} cells):")
    print(perturbation_labels.value_counts().to_string())

    # --- Optional: load donor_id ---
    donor_map = {}
    if args.donor_csv is not None and args.donor_csv.exists():
        donor_df = pd.read_csv(args.donor_csv)
        donor_map = dict(zip(donor_df["barcode"], donor_df["donor_id"]))

    # --- Intersect barcodes across ATAC + Protein + HTO-derived guide ---
    atac_set = set(atac_barcodes)
    prot_set = set(prot_barcodes)
    guide_set = set(perturbation_labels.index)
    common = atac_set & prot_set & guide_set
    print(f"\nBarcode intersection:")
    print(f"  ATAC:    {len(atac_set)}")
    print(f"  Protein: {len(prot_set)}")
    print(f"  Guide:   {len(guide_set)}")
    print(f"  COMMON:  {len(common)}")
    if len(common) == 0:
        raise ValueError("Zero shared barcodes — check barcode-suffix consistency "
                         "between ATAC, protein, and HTO files. Common gotcha: "
                         "Cell Ranger appends '-1' suffix; kite output may not.")

    # Sort common barcodes deterministically
    common_sorted = sorted(common)
    # Use dicts for O(1) barcode lookup (was O(n) per barcode → 32K^2)
    atac_pos = {b: i for i, b in enumerate(atac_barcodes)}
    prot_pos = {b: i for i, b in enumerate(prot_barcodes)}
    atac_idx = np.array([atac_pos[b] for b in common_sorted])
    prot_idx = np.array([prot_pos[b] for b in common_sorted])

    atac_kept = atac_counts[atac_idx]
    prot_kept = prot_counts[prot_idx]

    # --- Project ATAC to union peak space ---
    atac_union = project_atac_to_union(atac_kept, atac_peaks, union_peaks)

    # --- Build obs ---
    perts = [perturbation_labels[b] for b in common_sorted]
    donors = [donor_map.get(b, "unknown") for b in common_sorted]
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(["CD4_T"] * len(common_sorted)),
        "perturbation": pd.Categorical(perts, categories=[
            "NTC", "CD3E", "CD4", "ZAP70", "NFKB2", "CD3E_CD4_double"
        ]),
        "lysis_protocol": pd.Categorical(["LLL"] * len(common_sorted)),
        "donor_id": donors,
        "has_rna": False,    # CRISPR arm has no RNA in DOGMA-compatible form
        "has_atac": True,
        "has_protein": True,
        "has_phospho": False,
    }, index=common_sorted)

    # --- Build var (gene-side empty for CRISPR ATAC-only arm) ---
    # Use a placeholder var to satisfy AnnData; X is empty
    var = pd.DataFrame(index=[])
    X = sp.csr_matrix((len(common_sorted), 0), dtype=np.float32)

    # --- Assemble ---
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["atac_peaks"] = atac_union
    adata.obsm["protein"] = prot_kept

    # uns metadata
    adata.uns["source"] = "mimitou_2021_cd4_crispr_GSE156478"
    adata.uns["perturbation_categories"] = [
        "NTC", "CD3E", "CD4", "ZAP70", "NFKB2", "CD3E_CD4_double"
    ]
    adata.uns["atac_peak_set"] = "dogma_union"
    adata.uns["atac_feature_names"] = peak_strs
    adata.uns["protein_feature_names"] = prot_features

    # --- Distribution summary ---
    print(f"\n=== Output h5ad summary ===")
    print(f"  shape: {adata.shape}")
    print(f"  obsm['atac_peaks']: {adata.obsm['atac_peaks'].shape}, "
          f"nnz={adata.obsm['atac_peaks'].nnz}")
    print(f"  obsm['protein']: {adata.obsm['protein'].shape}")
    print(f"  perturbation distribution:")
    print(adata.obs["perturbation"].value_counts().to_string())

    # --- Sanity check: 4-arm completeness for CD3E+CD4 synergy demo ---
    need_arms = ["NTC", "CD3E", "CD4", "CD3E_CD4_double"]
    n_per_arm = adata.obs["perturbation"].value_counts()
    print(f"\n  4-arm completeness for synergy-head demo (CD3E + CD4):")
    for arm in need_arms:
        n = int(n_per_arm.get(arm, 0))
        flag = "✓" if n >= 30 else "⚠ LOW"
        print(f"    {arm:<20s}: n={n:>5d}  {flag}")
    missing = [arm for arm in need_arms if int(n_per_arm.get(arm, 0)) < 30]
    if missing:
        print(f"\n  WARNING: arms with n < 30 may not support reliable synergy "
              f"eval: {missing}. Lower bound for held-out per-perturbation "
              f"centroid-NN is ~30 cells.")

    # --- Write ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.out}")
    adata.write_h5ad(args.out, compression=None)
    print("Done.")


if __name__ == "__main__":
    main()
