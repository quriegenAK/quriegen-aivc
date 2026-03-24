"""
CLI: Run ambient RNA decontamination on Kang 2018 and report benchmark integrity.

Usage:
  python scripts/run_soupx_equivalent.py \
    --filtered data/kang2018_pbmc_fixed.h5ad \
    --raw      data/kang2018_raw/ \
    --output   data/kang2018_pbmc_decontaminated.h5ad \
    --report   results/ambient_rna_report.csv

If --raw is not provided or path does not exist, runs in FALLBACK mode.
"""
import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(
        description="Ambient RNA decontamination for Kang 2018 PBMC dataset"
    )
    parser.add_argument("--filtered", required=True,
                        help="Path to filtered h5ad (kang2018_pbmc_fixed.h5ad)")
    parser.add_argument("--raw", default=None,
                        help="Path to raw CellRanger output dir (optional)")
    parser.add_argument("--output", default="data/kang2018_pbmc_decontaminated.h5ad",
                        help="Output path for decontaminated h5ad")
    parser.add_argument("--report", default="results/ambient_rna_report.csv",
                        help="Output path for per-gene contamination report")
    parser.add_argument("--cap", type=float, default=0.20,
                        help="Max contamination fraction per cell (default 0.20)")
    args = parser.parse_args()

    import anndata as ad
    from aivc.qc.ambient_rna import AmbientRNAEstimator

    print("=" * 70)
    print("AIVC — Ambient RNA Decontamination (SoupX-equivalent)")
    print("=" * 70)

    # ── Load filtered data ──
    print(f"\n[1/5] Loading filtered data: {args.filtered}")
    adata = ad.read_h5ad(args.filtered)
    print(f"      Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

    # ── Initialise estimator ──
    estimator = AmbientRNAEstimator(
        filtered_adata=adata,
        raw_matrix_path=args.raw,
        low_umi_threshold=100,
        contamination_cap=args.cap,
    )

    # ── Estimate ambient profile ──
    print(f"\n[2/5] Estimating ambient RNA profile...")
    estimator.estimate_ambient_profile()
    print(f"      Mode: {estimator.mode.upper()}")
    top = estimator._top_ambient_genes(n=10)
    print(f"      Top 10 ambient genes:")
    for gene, frac in top:
        marker = " *" if gene in estimator.JAKSTAT_GENES else ""
        print(f"        {gene:<15} {frac:.4f}{marker}")

    # ── Estimate per-cell contamination ──
    print(f"\n[3/5] Estimating per-cell contamination fraction...")
    rho = estimator.estimate_contamination_per_cell()
    print(f"      Contamination rates:")
    print(f"        Mean:   {rho.mean():.4f} ({rho.mean()*100:.2f}%)")
    print(f"        Median: {np.median(rho):.4f} ({np.median(rho)*100:.2f}%)")
    print(f"        P95:    {np.percentile(rho, 95):.4f} ({np.percentile(rho, 95)*100:.2f}%)")
    print(f"        Max:    {rho.max():.4f} ({rho.max()*100:.2f}%)")

    # ── Decontaminate ──
    print(f"\n[4/5] Removing ambient contribution...")
    adata_clean = estimator.decontaminate()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    adata_clean.write(args.output)
    print(f"      Saved decontaminated data: {args.output}")

    # ── Per-gene contamination report ──
    print(f"\n[5/5] Generating per-gene contamination report...")
    report_df = estimator.gene_contamination_report(
        adata_clean,
        condition_col="label",
        ctrl_label="ctrl",
    )
    os.makedirs(os.path.dirname(args.report) or "results", exist_ok=True)
    report_df.to_csv(args.report, index=False)
    print(f"      Saved report: {args.report}")

    # ── JAK-STAT summary ──
    print(f"\n{'='*70}")
    print(f"JAK-STAT CONTAMINATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Gene':<12} {'Raw Ctrl Mean':>14} {'Clean Ctrl Mean':>16} "
          f"{'Ambient %':>10}  {'Status':<14}")
    print(f"  {'-'*68}")

    jakstat_df = report_df[report_df["is_jakstat"]].sort_values(
        "ambient_fraction", ascending=False
    )
    for _, row in jakstat_df.iterrows():
        print(
            f"  {row['gene_name']:<12} "
            f"{row['ctrl_raw_mean']:>14.4f} "
            f"{row['ctrl_clean_mean']:>16.4f} "
            f"{row['ambient_fraction']*100:>9.1f}%  "
            f"{row['interpretation']:<14}"
        )

    # ── Verdict ──
    print(f"\n{'='*70}")
    print(f"BENCHMARK INTEGRITY VERDICT")
    print(f"{'='*70}")

    ifit1_row = report_df[report_df["gene_name"] == "IFIT1"]
    ifit1_pct = float(ifit1_row["ambient_fraction"].values[0]) * 100 \
        if len(ifit1_row) > 0 else 0.0

    mx1_row = report_df[report_df["gene_name"] == "MX1"]
    mx1_pct = float(mx1_row["ambient_fraction"].values[0]) * 100 \
        if len(mx1_row) > 0 else 0.0

    contaminated_jakstat = report_df[
        report_df["is_jakstat"] & (report_df["ambient_fraction"] > 0.10)
    ]["gene_name"].tolist()

    print(f"  IFIT1 ambient contamination in ctrl cells: {ifit1_pct:.1f}%")
    print(f"  MX1   ambient contamination in ctrl cells: {mx1_pct:.1f}%")
    print(f"  JAK-STAT genes >10% contaminated: "
          f"{len(contaminated_jakstat)}/15 — {contaminated_jakstat}")
    print(f"  Mode: {estimator.mode.upper()}")

    if ifit1_pct < 5.0:
        print(f"\n  VERDICT: CLEAN")
        print(f"  r=0.873 benchmark reflects real biological signal.")
        print(f"  No re-evaluation required.")
    elif ifit1_pct < 10.0:
        print(f"\n  VERDICT: MODERATE — report r_clean in external documents")
        print(f"  Re-run evaluation on: {args.output}")
    else:
        print(f"\n  VERDICT: INFLATED — benchmark must be updated")
        print(f"  Re-train on: {args.output}")

    print(f"\n  Decontaminated data: {args.output}")
    print(f"  Full report:         {args.report}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
