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

Expected input directory contents (from GSE156478 supplementary or
asap_reproducibility/CD4_CRISPR_asapseq/data/):
  filtered_peak_bc_matrix.h5         # cellranger ATAC output
  fragments.tsv.gz                   # ATAC fragments (optional, for QC)
  protein_counts.mtx                 # kite-format protein counts
  protein_barcodes.tsv               # cell barcodes for protein
  protein_features.tsv               # antibody names
  guide_demux.csv                    # per-cell guide assignment
                                     # columns: barcode, guide_RNA, n_guides

If file names differ in your download, override via flags or pass paths
directly via --atac_h5, --protein_mtx, --guide_csv.
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


# Guide-RNA → target-gene mapping (per Mimitou 2021 §"Multiplexed CRISPR"
# and asap_reproducibility/CD4_CRISPR_asapseq/code/ harmonization scripts).
# Each target has multiple sgRNA variants; we collapse to target gene name
# for the perturbation label.
GUIDE_TO_TARGET = {
    # Non-targeting controls
    "NTC1": "NTC", "NTC2": "NTC", "sgNTC": "NTC",
    "Non-targeting": "NTC", "Non-targeting_1": "NTC", "Non-targeting_2": "NTC",
    # Targeting guides (Mimitou's panel)
    "sgCD3E_1": "CD3E", "sgCD3E_2": "CD3E", "sgCD3E": "CD3E",
    "sgCD4_1": "CD4", "sgCD4_2": "CD4", "sgCD4": "CD4",
    "sgZAP70_1": "ZAP70", "sgZAP70_2": "ZAP70", "sgZAP70": "ZAP70",
    "sgNFKB2_1": "NFKB2", "sgNFKB2_2": "NFKB2", "sgNFKB2": "NFKB2",
}


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


def load_protein_counts(mtx_path: Path, barcodes_path: Path, features_path: Path):
    """Load kite-format protein counts (matrix.mtx + barcodes + features)."""
    print(f"Loading protein counts: {mtx_path}")
    from scipy.io import mmread
    mat = mmread(str(mtx_path)).tocsr()  # likely (n_features, n_cells)
    barcodes = pd.read_csv(barcodes_path, header=None)[0].tolist()
    features = pd.read_csv(features_path, header=None)[0].tolist()
    if mat.shape[0] == len(features) and mat.shape[1] == len(barcodes):
        mat = mat.T.tocsr()  # → (cells, features)
    print(f"  protein shape: {mat.shape} ({len(barcodes)} cells × {len(features)} features)")
    return mat, barcodes, features


def load_guide_demux(guide_csv_path: Path) -> pd.DataFrame:
    """Load per-cell guide-RNA assignments. Expected columns:
    barcode, guide_RNA, n_guides (some variants: 'cell', 'sgRNA', 'guide').
    """
    print(f"Loading guide demux: {guide_csv_path}")
    df = pd.read_csv(guide_csv_path)
    # Normalize column names
    barcode_col = next((c for c in df.columns if c.lower() in ("barcode", "cell", "cell_barcode", "cell_id")), None)
    guide_col = next((c for c in df.columns if c.lower() in ("guide_rna", "sgrna", "guide", "guide_id")), None)
    if barcode_col is None or guide_col is None:
        raise ValueError(f"Could not identify barcode/guide columns in "
                         f"{guide_csv_path}; got columns {list(df.columns)}")
    df = df.rename(columns={barcode_col: "barcode", guide_col: "guide_RNA"})
    n_guides_col = next((c for c in df.columns if c.lower() in ("n_guides", "num_guides", "n_perts")), None)
    if n_guides_col is None:
        df["n_guides"] = 1  # assume single-guide if not specified
    else:
        df = df.rename(columns={n_guides_col: "n_guides"})
    print(f"  {len(df)} guide-cell rows; columns: {list(df.columns)}")
    return df[["barcode", "guide_RNA", "n_guides"]]


def map_guides_to_perturbation(guides_df: pd.DataFrame) -> pd.Series:
    """Collapse per-cell guide assignments to a single perturbation label.

    Single-guide cells: target gene of that guide (or 'NTC').
    Double-guide cells: 'CD3E_CD4_double' if matches the Mimitou double KO;
    otherwise drop (we only have CD3E+CD4 as a clean double in the panel).
    """
    # Determine target per row
    targets = guides_df["guide_RNA"].map(GUIDE_TO_TARGET)
    unmapped = guides_df.loc[targets.isna(), "guide_RNA"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: {len(unmapped)} unmapped guide names "
              f"(first 10: {list(unmapped[:10])}); these cells will be dropped.")

    # Group by barcode → set of targets
    df = guides_df.copy()
    df["target"] = targets
    df = df.dropna(subset=["target"])
    grouped = df.groupby("barcode")["target"].agg(lambda x: tuple(sorted(set(x))))

    # Map target tuples → perturbation label
    def _label(target_tuple):
        if target_tuple == ("NTC",):
            return "NTC"
        if len(target_tuple) == 1:
            return target_tuple[0]
        if target_tuple == ("CD3E", "CD4"):
            return "CD3E_CD4_double"
        return None  # other multi-guide combos: drop

    return grouped.map(_label).dropna()


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
    p.add_argument("--guide_csv", type=Path,
                   help="Per-cell guide_RNA assignment CSV")
    p.add_argument("--donor_csv", type=Path, default=None,
                   help="Optional: per-cell donor_id CSV (barcode, donor_id). "
                        "If absent, all cells get donor_id='unknown'.")
    p.add_argument("--union_manifest", required=True, type=Path,
                   help="DOGMA union peak manifest (UNION_MANIFEST.json or peaks.tsv)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output h5ad path")
    args = p.parse_args()

    # --- Resolve input paths ---
    if args.mimitou_crispr_dir is not None:
        d = args.mimitou_crispr_dir
        args.atac_h5 = args.atac_h5 or d / "filtered_peak_bc_matrix.h5"
        args.protein_mtx = args.protein_mtx or d / "protein_counts.mtx"
        args.protein_barcodes = args.protein_barcodes or d / "protein_barcodes.tsv"
        args.protein_features = args.protein_features or d / "protein_features.tsv"
        args.guide_csv = args.guide_csv or d / "guide_demux.csv"
    for fld in ("atac_h5", "protein_mtx", "protein_barcodes",
                "protein_features", "guide_csv", "union_manifest"):
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

    # --- Load Protein ---
    prot_counts, prot_barcodes, prot_features = load_protein_counts(
        args.protein_mtx, args.protein_barcodes, args.protein_features,
    )

    # --- Load guide demux ---
    guides_df = load_guide_demux(args.guide_csv)
    perturbation_labels = map_guides_to_perturbation(guides_df)
    print(f"\nPerturbation distribution (across {len(perturbation_labels)} cells):")
    print(perturbation_labels.value_counts().to_string())

    # --- Optional: load donor_id ---
    donor_map = {}
    if args.donor_csv is not None and args.donor_csv.exists():
        donor_df = pd.read_csv(args.donor_csv)
        donor_map = dict(zip(donor_df["barcode"], donor_df["donor_id"]))

    # --- Intersect barcodes across ATAC + Protein + guide ---
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
                         "between ATAC, protein, and guide files.")

    # Sort common barcodes deterministically
    common_sorted = sorted(common)
    atac_idx = np.array([atac_barcodes.index(b) for b in common_sorted])
    prot_idx = np.array([prot_barcodes.index(b) for b in common_sorted])

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
        "guide_RNA": [
            ",".join(sorted(guides_df.query("barcode == @b")["guide_RNA"].unique()))
            for b in common_sorted
        ],
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
