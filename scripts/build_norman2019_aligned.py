"""
scripts/build_norman2019_aligned.py — Gene-vocabulary alignment for Norman 2019.

Takes the raw Norman 2019 h5ad (K562 Perturb-seq, ~5k–91k cells, ~5k–8k genes)
and produces an aligned AnnData whose gene axis matches the 36,601-gene PBMC10k
vocabulary expected by the pretrained SimpleRNAEncoder.

Steps performed
---------------
1. Read raw Norman 2019 h5ad from data/raw/norman2019/.
2. Read the PBMC10k gene vocabulary from data/pbmc10k_multiome.h5ad.
3. Compute gene-name intersection (HGNC symbols, case-normalised).
   STOP with an informative message if intersection < 2,000.
4. Build aligned AnnData:
     shape = (n_cells_norman, 36601)
     intersection genes → copy expression values
     non-intersection genes → 0.0
     gene order matches PBMC10k var_names exactly
5. Preserve obs["perturbation"] (or nearest equivalent).
   Add obs["n_genes_detected_norman"] (QC: nonzero genes in original data).
6. Compute top-50 DE genes per perturbation (abs log-fold-change vs control).
   Store in uns["top50_de_per_perturbation"] and save as
   data/norman2019_top50_de.json.
7. Save to data/norman2019_aligned.h5ad.
8. Write sibling data/norman2019_aligned.h5ad.meta.json with full provenance.

Usage
-----
python scripts/build_norman2019_aligned.py \
    [--norman_path  data/raw/norman2019/NormanWeissman2019_filtered.h5ad] \
    [--pbmc_path    data/pbmc10k_multiome.h5ad] \
    [--out_path     data/norman2019_aligned.h5ad] \
    [--de_json_path data/norman2019_top50_de.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Minimum gene intersection before we refuse to proceed.
MIN_INTERSECTION = 2_000
# Number of top DE genes to record per perturbation.
TOP_K_DE = 50
# Label used for control cells (scperturb standardises to this).
CONTROL_LABELS = {"control", "ctrl", "non-targeting", "nt"}


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(msg: str) -> None:
    print(f"[build-norman2019] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Step 1 – Load raw Norman 2019
# --------------------------------------------------------------------------- #

def _find_raw_h5ad(norm_path: Path) -> Path:
    """Resolve the raw Norman h5ad path; search common filenames if a dir given."""
    if norm_path.is_file():
        return norm_path
    if norm_path.is_dir():
        candidates = sorted(norm_path.glob("*.h5ad"))
        if not candidates:
            raise FileNotFoundError(
                f"No .h5ad found in {norm_path}. "
                "Run scripts/download_norman2019.sh first."
            )
        if len(candidates) > 1:
            warnings.warn(
                f"Multiple .h5ad files in {norm_path}; using {candidates[0]}.",
                stacklevel=2,
            )
        return candidates[0]
    raise FileNotFoundError(
        f"Norman 2019 path not found: {norm_path}. "
        "Run scripts/download_norman2019.sh first."
    )


def _detect_perturbation_column(adata) -> str:
    """Return the obs column that carries perturbation labels."""
    preferred = ["perturbation", "condition", "gene", "guide_ids", "target_gene_name"]
    for col in preferred:
        if col in adata.obs.columns:
            return col
    raise KeyError(
        f"Cannot find perturbation column in obs. "
        f"Available columns: {list(adata.obs.columns)}"
    )


# --------------------------------------------------------------------------- #
# Step 2 – Gene vocabulary alignment
# --------------------------------------------------------------------------- #

def _normalise_gene_names(names) -> list[str]:
    """Upper-case strip; the most common case normalisation."""
    return [str(g).strip().upper() for g in names]


def _align_to_pbmc_vocab(
    norman_adata,
    pbmc_var_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (aligned_X, intersection_genes_pbmc_order, unmatched_pbmc_genes).

    aligned_X shape: (n_cells_norman, len(pbmc_var_names))
    Intersection genes → Norman expression.
    Non-intersection → 0.0.
    """
    import scipy.sparse as sp

    # Build case-normalised lookup: upper_symbol → original Norman var_name
    norman_upper = _normalise_gene_names(norman_adata.var_names)
    pbmc_upper   = _normalise_gene_names(pbmc_var_names)

    norman_upper_to_idx: dict[str, int] = {}
    for idx, sym in enumerate(norman_upper):
        if sym not in norman_upper_to_idx:
            norman_upper_to_idx[sym] = idx
        # If duplicate, keep first occurrence; warn about collisions
    # Warn on duplicates
    seen: set[str] = set()
    dupes = []
    for sym in norman_upper:
        if sym in seen:
            dupes.append(sym)
        seen.add(sym)
    if dupes:
        warnings.warn(
            f"Norman 2019 var_names has {len(set(dupes))} duplicate "
            f"upper-cased symbols (e.g. {dupes[:3]}); keeping first occurrence.",
            stacklevel=3,
        )

    # Identify intersection (preserving PBMC order)
    intersected_pbmc_indices: list[int] = []
    intersected_norman_indices: list[int] = []
    intersection_genes: list[str] = []
    unmatched: list[str] = []

    for pbmc_idx, sym_upper in enumerate(pbmc_upper):
        if sym_upper in norman_upper_to_idx:
            norm_idx = norman_upper_to_idx[sym_upper]
            intersected_pbmc_indices.append(pbmc_idx)
            intersected_norman_indices.append(norm_idx)
            intersection_genes.append(pbmc_var_names[pbmc_idx])
        else:
            unmatched.append(pbmc_var_names[pbmc_idx])

    n_intersect = len(intersection_genes)
    _log(f"Norman genes    : {len(norman_adata.var_names):,}")
    _log(f"PBMC10k genes   : {len(pbmc_var_names):,}")
    _log(f"Intersection    : {n_intersect:,}")
    _log(f"PBMC-only genes : {len(unmatched):,}  (will be zero-filled)")

    if n_intersect < MIN_INTERSECTION:
        raise RuntimeError(
            f"Gene intersection ({n_intersect}) is below the minimum required "
            f"({MIN_INTERSECTION}) for a meaningful evaluation.\n"
            f"Norman var_names sample (first 5): "
            f"{list(norman_adata.var_names[:5])}\n"
            f"PBMC10k var_names sample (first 5): {pbmc_var_names[:5]}\n"
            "Likely cause: gene name format mismatch (Ensembl IDs vs symbols, "
            "or case mismatch that was not resolved by upper-casing). "
            "Inspect both var_names and fix the alignment logic."
        )

    # Build aligned dense matrix (float32)
    n_cells = norman_adata.shape[0]
    n_pbmc  = len(pbmc_var_names)

    X_norman = norman_adata.X
    if sp.issparse(X_norman):
        X_norman = X_norman.toarray()
    X_norman = np.asarray(X_norman, dtype=np.float32)

    X_aligned = np.zeros((n_cells, n_pbmc), dtype=np.float32)
    X_aligned[:, intersected_pbmc_indices] = X_norman[:, intersected_norman_indices]

    return X_aligned, intersection_genes, unmatched


# --------------------------------------------------------------------------- #
# Step 3 – Top-50 DE genes per perturbation
# --------------------------------------------------------------------------- #

def _compute_top50_de(
    X_aligned: np.ndarray,
    perturbation_labels: np.ndarray,
    pbmc_var_names: list[str],
    top_k: int = TOP_K_DE,
) -> dict[str, list[int]]:
    """Return {perturbation: [gene_indices in 36601-dim space]} for top-K |LFC|.

    Control cells (label in CONTROL_LABELS, case-insensitive) are used as
    the reference.  If no control cells are found, falls back to the grand
    mean across all cells and emits a warning.
    """
    labels = np.asarray(perturbation_labels, dtype=str)

    # Find control indices
    control_mask = np.zeros(len(labels), dtype=bool)
    for ctrl_label in CONTROL_LABELS:
        control_mask |= (np.char.lower(labels) == ctrl_label.lower())

    if control_mask.sum() == 0:
        warnings.warn(
            "No control cells found (tried labels: "
            f"{sorted(CONTROL_LABELS)}). "
            "Falling back to grand mean as reference. "
            "Verify obs['perturbation'] label for controls.",
            stacklevel=2,
        )
        mean_ctrl = X_aligned.mean(axis=0)  # shape: (n_genes,)
    else:
        n_ctrl = int(control_mask.sum())
        _log(f"Control cells   : {n_ctrl:,}")
        mean_ctrl = X_aligned[control_mask].mean(axis=0)

    unique_perts = [
        p for p in np.unique(labels)
        if p.lower() not in {c.lower() for c in CONTROL_LABELS}
    ]
    _log(f"Perturbations   : {len(unique_perts):,} unique (excluding controls)")

    top50_de: dict[str, list[int]] = {}
    eps = 1e-6

    for pert in unique_perts:
        mask = labels == pert
        if mask.sum() == 0:
            continue
        mean_pert = X_aligned[mask].mean(axis=0)
        lfc = np.log2(mean_pert + eps) - np.log2(mean_ctrl + eps)
        k = min(top_k, lfc.shape[0])
        top_idx = np.argsort(-np.abs(lfc))[:k]
        top50_de[str(pert)] = sorted(top_idx.tolist())

    return top50_de


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--norman_path",
        type=Path,
        default=Path("data/raw/norman2019"),
        help="Path to raw Norman h5ad file or directory containing it.",
    )
    p.add_argument(
        "--pbmc_path",
        type=Path,
        default=Path("data/pbmc10k_multiome.h5ad"),
        help="Path to the PBMC10k Multiome h5ad (for gene vocabulary).",
    )
    p.add_argument(
        "--out_path",
        type=Path,
        default=Path("data/norman2019_aligned.h5ad"),
        help="Output aligned h5ad path.",
    )
    p.add_argument(
        "--de_json_path",
        type=Path,
        default=Path("data/norman2019_top50_de.json"),
        help="Output JSON mapping perturbation → top-50 DE gene indices.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import anndata as ad

    args = _parse_args(argv)

    # ---- Step 1: load Norman 2019 ----------------------------------------
    raw_path = _find_raw_h5ad(args.norman_path)
    _log(f"Loading Norman 2019 from: {raw_path}")
    norman = ad.read_h5ad(raw_path)
    _log(f"Raw shape: {norman.shape[0]:,} cells × {norman.shape[1]:,} genes")

    pert_col = _detect_perturbation_column(norman)
    _log(f"Perturbation column: obs['{pert_col}']")
    n_unique_perts = norman.obs[pert_col].nunique()
    _log(f"Unique perturbations: {n_unique_perts:,}")

    input_sha = _sha256(raw_path)
    _log(f"Input SHA-256: {input_sha}")

    # ---- Step 2: load PBMC10k gene vocab ---------------------------------
    if not args.pbmc_path.exists():
        raise FileNotFoundError(
            f"PBMC10k h5ad not found at {args.pbmc_path}. "
            "Run scripts/build_pbmc10k_h5ad.py first."
        )
    _log(f"Loading PBMC10k gene vocabulary from: {args.pbmc_path}")
    pbmc = ad.read_h5ad(args.pbmc_path, backed="r")
    pbmc_var_names = list(pbmc.var_names)
    pbmc.file.close()
    _log(f"PBMC10k vocabulary: {len(pbmc_var_names):,} genes")
    assert len(pbmc_var_names) == 36_601, (
        f"Expected 36601 PBMC10k genes, got {len(pbmc_var_names)}. "
        "Wrong file?"
    )

    # ---- Step 2 (cont.): align gene vocab --------------------------------
    _log("Aligning gene vocabularies ...")
    X_aligned, intersection_genes, unmatched_genes = _align_to_pbmc_vocab(
        norman, pbmc_var_names
    )
    n_intersect = len(intersection_genes)

    # QC metric: nonzero genes per cell in original Norman data
    import scipy.sparse as sp
    X_orig = norman.X
    if sp.issparse(X_orig):
        n_genes_detected = np.asarray((X_orig > 0).sum(axis=1)).ravel()
    else:
        n_genes_detected = (np.asarray(X_orig, dtype=np.float32) > 0).sum(axis=1)

    # ---- Build aligned AnnData -------------------------------------------
    _log("Building aligned AnnData ...")
    obs_df = norman.obs[[pert_col]].copy()
    obs_df.columns = ["perturbation"]
    obs_df["n_genes_detected_norman"] = n_genes_detected.astype(np.int32)

    aligned = ad.AnnData(
        X=X_aligned,
        obs=obs_df,
        var=ad.pd.DataFrame(index=pbmc_var_names),
    )
    assert aligned.shape == (norman.shape[0], 36_601), (
        f"Aligned shape {aligned.shape} unexpected."
    )
    _log(f"Aligned shape: {aligned.shape[0]:,} × {aligned.shape[1]:,}  ✓")

    # ---- Step 3: top-50 DE per perturbation ------------------------------
    _log("Computing top-50 DE genes per perturbation ...")
    top50_de = _compute_top50_de(
        X_aligned,
        aligned.obs["perturbation"].to_numpy(),
        pbmc_var_names,
        top_k=TOP_K_DE,
    )
    aligned.uns["top50_de_per_perturbation"] = top50_de
    _log(f"DE computed for {len(top50_de):,} perturbations.")

    # ---- Step 4: save aligned h5ad ---------------------------------------
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Writing aligned h5ad to: {args.out_path}")
    aligned.write_h5ad(args.out_path)

    output_sha = _sha256(args.out_path)
    _log(f"Output SHA-256: {output_sha}")

    # ---- Step 4 (cont.): save DE JSON ------------------------------------
    _log(f"Writing DE JSON to: {args.de_json_path}")
    args.de_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.de_json_path, "w") as fh:
        json.dump(top50_de, fh, indent=2)

    # ---- Step 5: write .meta.json ----------------------------------------
    gene_intersection_txt = args.out_path.with_suffix(".h5ad.gene_intersection.txt")
    with open(gene_intersection_txt, "w") as fh:
        fh.write("\n".join(intersection_genes))
    _log(f"Gene intersection list written to: {gene_intersection_txt}")

    meta = {
        "input_sha256": input_sha,
        "input_path": str(raw_path),
        "output_sha256": output_sha,
        "output_path": str(args.out_path),
        "n_cells": int(aligned.shape[0]),
        "n_genes_aligned": int(aligned.shape[1]),
        "n_genes_intersection": int(n_intersect),
        "n_genes_norman_only": int(len(norman.var_names) - n_intersect),
        "n_genes_pbmc_only": int(len(unmatched_genes)),
        "gene_intersection_file": str(gene_intersection_txt),
        "n_perturbations": int(len(top50_de)),
        "de_json_path": str(args.de_json_path),
        "perturbation_column_source": pert_col,
        "source_url": "https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = args.out_path.parent / (args.out_path.name + ".meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    _log(f"Meta JSON written to: {meta_path}")

    # ---- Final summary ---------------------------------------------------
    print("\n=== Norman 2019 alignment complete ===")
    print(f"  Input  : {raw_path}  (SHA: {input_sha[:16]}...)")
    print(f"  Output : {args.out_path}  (SHA: {output_sha[:16]}...)")
    print(f"  Shape  : {aligned.shape[0]:,} cells × {aligned.shape[1]:,} genes")
    print(f"  Intersection : {n_intersect:,} / 36,601 PBMC genes matched in Norman")
    print(f"  Perturbations: {len(top50_de):,} unique (top-50 DE computed for each)")
    print(f"  Meta   : {meta_path}")
    print()
    n_pert_unique = aligned.obs["perturbation"].nunique()
    print(f"  Unique perturbation values: {n_pert_unique:,}")
    print(f"  Control cells: {(aligned.obs['perturbation'].str.lower().isin(CONTROL_LABELS)).sum():,}")
    print()

    # Guard: shape invariant
    if aligned.shape[1] != 36_601:
        raise RuntimeError(
            f"BUG: aligned n_genes={aligned.shape[1]}, expected 36601. "
            "The alignment script has a bug."
        )
    print("Shape invariant OK: n_genes == 36,601  ✓")


if __name__ == "__main__":
    main()
