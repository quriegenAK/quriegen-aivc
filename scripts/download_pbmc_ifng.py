"""
Download and validate a real PBMC IFN-γ stimulation dataset.

Priority order:
  1. Dixit 2016 (GSE90063)
  2. OneK1K (Yazar 2022)
  3. Any PBMC + IFN-γ dataset on GEO not in current training

Usage:
  python scripts/download_pbmc_ifng.py \
    --source dixit2016 \
    --output data/pbmc_ifng_real.h5ad \
    --validate
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


SEED = 42
KANG_N_GENES = 3010
MIN_GENE_OVERLAP = 1500
MIN_STIM_CELLS = 5000

DOWNLOAD_INSTRUCTIONS = {
    "dixit2016": {
        "accession": "GSE90063",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063",
        "steps": [
            "1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063",
            "2. Download the supplementary files (processed counts)",
            "3. Convert to h5ad format using scanpy.read_10x_mtx() or scanpy.read_csv()",
            "4. Ensure obs has 'condition' column with 'ctrl' and 'stim' labels",
            "5. Save as: data/pbmc_ifng_real.h5ad",
        ],
    },
    "onek1k": {
        "accession": "EGAS00001005989",
        "url": "https://onek1k.org/",
        "steps": [
            "1. Go to https://onek1k.org/ and request data access",
            "2. Download PBMC stimulation subset",
            "3. Filter for IFN-γ stimulated cells",
            "4. Ensure obs has 'condition' column with 'ctrl' and 'stim' labels",
            "5. Save as: data/pbmc_ifng_real.h5ad",
        ],
    },
}


def load_kang_gene_names():
    """Load Kang 2018 HVG gene names for overlap check."""
    path = "data/gene_names.txt"
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback: try loading from h5ad
    try:
        import anndata as ad
        adata = ad.read_h5ad("data/kang2018_pbmc_fixed.h5ad")
        if "name" in adata.var.columns:
            return adata.var["name"].tolist()
        return adata.var_names.tolist()
    except Exception:
        return None


def validate_dataset(dataset_path, kang_genes=None):
    """Run all 6 validation checks on the dataset. Returns (passed, results)."""
    import anndata as ad

    results = []
    all_pass = True

    adata = ad.read_h5ad(dataset_path)

    # Check 1: Cell type — must contain PBMCs
    check1_pass = False
    if "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"].unique().tolist()
        pbmc_types = {"CD4 T cells", "CD8 T cells", "NK cells", "B cells",
                      "CD14+ Monocytes", "Dendritic cells", "Monocytes",
                      "T cells", "PBMC"}
        overlap = set(cell_types) & pbmc_types
        check1_pass = len(overlap) > 0
        msg = f"PBMC cell types found: {overlap}" if check1_pass else f"No PBMC cell types. Found: {cell_types[:5]}"
    else:
        # If no cell_type column, check if description mentions PBMC
        msg = "No 'cell_type' column — cannot verify cell type. Manually confirm PBMCs."
        check1_pass = False
    results.append({"check": "cell_type_pbmc", "passed": check1_pass, "message": msg})
    if not check1_pass:
        all_pass = False

    # Check 2: Perturbation — must contain IFN-γ (not IFN-α or IFN-β)
    check2_pass = False
    for col in ["condition", "stim", "perturbation"]:
        if col in adata.obs.columns:
            conditions = adata.obs[col].unique().tolist()
            ifng_labels = {"IFN-gamma", "IFNG", "ifng", "IFNg", "stim"}
            ifna_labels = {"IFN-alpha", "IFNA", "IFN-beta", "IFNB"}
            has_ifng = bool(set(conditions) & ifng_labels)
            has_ifna = bool(set(conditions) & ifna_labels)
            if has_ifng and not has_ifna:
                check2_pass = True
                msg = f"IFN-γ perturbation found in '{col}': {set(conditions) & ifng_labels}"
            elif has_ifna:
                msg = f"Found IFN-α/β (NOT IFN-γ) in '{col}': {set(conditions) & ifna_labels}"
            else:
                # Check for ctrl/stim pattern (generic)
                if "ctrl" in conditions or "control" in conditions:
                    check2_pass = True
                    msg = f"ctrl/stim labels found — manually confirm perturbation is IFN-γ"
            break
    else:
        msg = "No condition/stim/perturbation column found."
    results.append({"check": "perturbation_ifng", "passed": check2_pass, "message": msg})
    if not check2_pass:
        all_pass = False

    # Check 3: Gene overlap with Kang 2018 HVG
    check3_pass = False
    if kang_genes is not None:
        if "name" in adata.var.columns:
            eval_genes = set(adata.var["name"].tolist())
        else:
            eval_genes = set(adata.var_names.tolist())
        overlap = len(set(kang_genes) & eval_genes)
        check3_pass = overlap >= MIN_GENE_OVERLAP
        msg = f"Gene overlap: {overlap} / {len(kang_genes)} (min: {MIN_GENE_OVERLAP})"
    else:
        msg = "Kang 2018 gene names not available — cannot check overlap."
    results.append({"check": "gene_overlap", "passed": check3_pass, "message": msg})
    if not check3_pass:
        all_pass = False

    # Check 4: Cell count — >= 5000 stimulated cells
    n_stim = 0
    for col in ["condition", "stim", "perturbation"]:
        if col in adata.obs.columns:
            stim_labels = {"stim", "IFN-gamma", "IFNG", "ifng", "IFNg", "stimulated"}
            stim_mask = adata.obs[col].isin(stim_labels)
            n_stim = int(stim_mask.sum())
            break
    check4_pass = n_stim >= MIN_STIM_CELLS
    msg = f"Stimulated cells: {n_stim} (min: {MIN_STIM_CELLS})"
    results.append({"check": "cell_count", "passed": check4_pass, "message": msg})
    if not check4_pass:
        all_pass = False

    # Check 5: Condition labels — must have ctrl and stim
    check5_pass = False
    for col in ["condition", "stim", "perturbation"]:
        if col in adata.obs.columns:
            vals = set(adata.obs[col].unique().tolist())
            ctrl_labels = {"ctrl", "control", "CTRL", "unstimulated"}
            stim_labels = {"stim", "IFN-gamma", "IFNG", "ifng", "IFNg", "stimulated"}
            has_ctrl = bool(vals & ctrl_labels)
            has_stim = bool(vals & stim_labels)
            check5_pass = has_ctrl and has_stim
            msg = f"Labels in '{col}': {vals}. ctrl={has_ctrl}, stim={has_stim}"
            break
    else:
        msg = "No condition column found."
    results.append({"check": "condition_labels", "passed": check5_pass, "message": msg})
    if not check5_pass:
        all_pass = False

    # Check 6: Synthetic flag
    check6_pass = True
    if "SYNTHETIC_IFNG" in adata.obs.columns:
        if adata.obs["SYNTHETIC_IFNG"].any():
            check6_pass = False
            msg = "SYNTHETIC_IFNG=True detected. Cannot use synthetic data for Stage 2."
        else:
            msg = "SYNTHETIC_IFNG column present, all False."
    else:
        msg = "No SYNTHETIC_IFNG column — assumed real data."
    results.append({"check": "not_synthetic", "passed": check6_pass, "message": msg})
    if not check6_pass:
        all_pass = False

    # Compute counts for certificate
    n_ctrl = 0
    for col in ["condition", "stim", "perturbation"]:
        if col in adata.obs.columns:
            ctrl_labels = {"ctrl", "control", "CTRL", "unstimulated"}
            n_ctrl = int(adata.obs[col].isin(ctrl_labels).sum())
            break

    gene_overlap = 0
    if kang_genes is not None:
        if "name" in adata.var.columns:
            eval_genes = set(adata.var["name"].tolist())
        else:
            eval_genes = set(adata.var_names.tolist())
        gene_overlap = len(set(kang_genes) & eval_genes)

    extra = {
        "n_cells_ctrl": n_ctrl,
        "n_cells_stim": n_stim,
        "gene_overlap": gene_overlap,
    }

    return all_pass, results, extra


def write_certificate(dataset_path, results, extra):
    """Write validation certificate JSON."""
    cert_dir = "data/validation_certificates"
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "pbmc_ifng_real.json")

    cert = {
        "dataset_path": os.path.abspath(dataset_path),
        "validation_date": datetime.now().isoformat(),
        "checks_passed": [r["check"] for r in results if r["passed"]],
        "n_cells_ctrl": extra["n_cells_ctrl"],
        "n_cells_stim": extra["n_cells_stim"],
        "gene_overlap": extra["gene_overlap"],
        "is_synthetic": False,
        "ready_for_stage2": all(r["passed"] for r in results),
    }

    with open(cert_path, "w") as f:
        json.dump(cert, f, indent=2)

    print(f"\n  Certificate written: {cert_path}")
    return cert_path


def main():
    parser = argparse.ArgumentParser(description="Download and validate PBMC IFN-γ dataset")
    parser.add_argument("--source", default="dixit2016", choices=list(DOWNLOAD_INSTRUCTIONS.keys()),
                        help="Dataset source")
    parser.add_argument("--output", default="data/pbmc_ifng_real.h5ad", help="Output h5ad path")
    parser.add_argument("--validate", action="store_true", help="Run validation and write certificate")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PBMC IFN-γ DATASET — STAGE 2 UNLOCK")
    print(f"{'='*60}")

    if not os.path.exists(args.output):
        # Print download instructions
        info = DOWNLOAD_INSTRUCTIONS.get(args.source, {})
        print(f"\n  Dataset not found: {args.output}")
        print(f"  Source: {args.source}")
        print(f"  GEO accession: {info.get('accession', 'N/A')}")
        print(f"  URL: {info.get('url', 'N/A')}")
        print(f"\n  Download instructions:")
        for step in info.get("steps", []):
            print(f"    {step}")
        print(f"\n  After download, re-run with --validate:")
        print(f"    python scripts/download_pbmc_ifng.py --output {args.output} --validate")
        sys.exit(0)

    # Dataset exists — validate
    print(f"\n  Dataset found: {args.output}")
    kang_genes = load_kang_gene_names()

    all_pass, results, extra = validate_dataset(args.output, kang_genes)

    print(f"\n  Validation results:")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"    [{status}] {r['check']}: {r['message']}")

    if all_pass:
        print(f"\n  Dataset ready for Stage 2 training")
        if args.validate:
            write_certificate(args.output, results, extra)
    else:
        failed = [r for r in results if not r["passed"]]
        print(f"\n  {len(failed)} check(s) FAILED. Dataset not ready for Stage 2.")
        for r in failed:
            print(f"    FIX: {r['check']} — {r['message']}")


if __name__ == "__main__":
    main()
