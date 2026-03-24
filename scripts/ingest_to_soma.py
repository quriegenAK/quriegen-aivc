"""
CLI: Ingest all AIVC datasets into the TileDB-SOMA store.

Usage:
  python scripts/ingest_to_soma.py \\
    --kang    data/kang2018_pbmc_fixed.h5ad \\
    --store   data/aivc_soma/ \\
    --mode    local

  python scripts/ingest_to_soma.py \\
    --kang    data/kang2018_pbmc_fixed.h5ad \\
    --store   data/aivc_soma/ \\
    --dry-run
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aivc.ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest AIVC datasets into TileDB-SOMA store")
    parser.add_argument("--kang", required=True, help="Path to Kang 2018 h5ad")
    parser.add_argument("--store", default="data/aivc_soma/", help="Output SOMA store path")
    parser.add_argument("--mode", default="local", choices=["local", "s3"], help="Storage mode")
    parser.add_argument("--frangieh", default=None, help="Path to Frangieh h5ad (optional)")
    parser.add_argument("--replogle", default=None, help="Path to Replogle h5ad (optional)")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate after ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, don't write")
    args = parser.parse_args()

    print("=" * 70)
    print("AIVC TileDB-SOMA Ingestion Pipeline")
    print("=" * 70)

    # ── Check datasets ──
    datasets = []
    if args.kang and os.path.exists(args.kang):
        import anndata as ad
        adata = ad.read_h5ad(args.kang)
        datasets.append({
            "name": "kang_2018",
            "path": args.kang,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "dataset_id": 0,
        })
        del adata
    else:
        logger.error(f"Kang 2018 not found: {args.kang}")
        sys.exit(1)

    if args.frangieh and os.path.exists(args.frangieh):
        adata = ad.read_h5ad(args.frangieh)
        datasets.append({
            "name": "frangieh_2021",
            "path": args.frangieh,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "dataset_id": 1,
            "w_only": True,
        })
        del adata
    elif args.frangieh:
        logger.warning(f"Frangieh not found: {args.frangieh}. Skipping.")

    # ── Schema summary ──
    from aivc.data.soma_store import AIVCSomaStore

    print(f"\nStore path: {args.store}")
    print(f"Mode: {args.mode}")
    print(f"\nOBS Schema ({len(AIVCSomaStore.OBS_SCHEMA)} columns):")
    for col, dtype in AIVCSomaStore.OBS_SCHEMA.items():
        print(f"  {col:<35} {dtype.__name__}")

    print(f"\nVAR Schema ({len(AIVCSomaStore.VAR_SCHEMA)} columns):")
    for col, dtype in AIVCSomaStore.VAR_SCHEMA.items():
        print(f"  {col:<35} {dtype.__name__}")

    print(f"\nDatasets to ingest:")
    total_cells = 0
    for ds in datasets:
        w_only = " (W_ONLY)" if ds.get("w_only") else ""
        print(f"  {ds['name']:<20} {ds['n_cells']:>8,} cells x {ds['n_genes']:>6,} genes{w_only}")
        total_cells += ds["n_cells"]
    print(f"  {'TOTAL':<20} {total_cells:>8,} cells")

    if args.dry_run:
        print(f"\n*** DRY RUN — no data written ***")
        print(f"To ingest, run without --dry-run flag.")

        # Validate schema on Kang
        adata = ad.read_h5ad(args.kang)
        store = AIVCSomaStore(args.store, mode="w")
        schema_check = store.validate_obs_schema(adata)
        print(f"\nSchema validation (Kang 2018):")
        print(f"  Valid:    {schema_check['valid']}")
        print(f"  Missing:  {schema_check['missing']}")
        print(f"  Warnings: {schema_check['warnings']}")

        # Partition key preview
        cell_ids = adata.obs_names[:10].tolist()
        print(f"\nPartition key preview (first 10 cells):")
        for cid in cell_ids:
            pk = hash(cid) % 1024
            print(f"  {cid:<30} -> partition_key={pk}")

        print(f"\nJAK-STAT genes in dataset:")
        gene_names = adata.var_names.tolist() if "name" not in adata.var.columns else adata.var["name"].tolist()
        jakstat_present = [g for g in AIVCSomaStore.JAKSTAT_GENES if g in gene_names]
        print(f"  {len(jakstat_present)}/15: {jakstat_present}")

        print(f"\n{'='*70}")
        print(f"DRY RUN COMPLETE")
        print(f"{'='*70}")
        return

    # ── Actual ingestion ──
    try:
        import tiledbsoma
    except ImportError:
        print(
            "\ntiledbsoma not installed. Install with:\n"
            "  pip install tiledbsoma\n"
            "Then re-run this script."
        )
        sys.exit(1)

    store = AIVCSomaStore(args.store, mode="w")

    for ds in datasets:
        t0 = time.time()
        print(f"\n[Ingesting] {ds['name']}...")
        adata = ad.read_h5ad(ds["path"])
        store.ingest_from_anndata(adata, measurement_name="RNA")
        elapsed = time.time() - t0
        print(f"  Done: {ds['n_cells']:,} cells in {elapsed:.1f}s")

    if args.validate:
        print(f"\n[Validating] {args.store}...")
        result = store.validate_store()
        if not result["valid"]:
            logger.error("Validation FAILED!")
            sys.exit(1)
        print("  Validation PASSED")

    print(f"\n{'='*70}")
    print(f"INGESTION COMPLETE: {args.store}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
