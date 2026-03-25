"""
ATAC+RNA Multiome Processing Pipeline for AIVC v3.0.

Replaces Wout Megchelenbrink's 8-step R pipeline (Seurat + Signac)
with a production-grade Python implementation.

Pipeline steps:
  01. Load 10x Multiome data (filtered_feature_bc_matrix.h5 + fragments)
  02. Joint QC filter (RNA: min genes, max mt%; ATAC: min TSS, min fragments)
  03. Annotate cell types (Hoa et al. PBMC reference, discard pred_score < 0.5)
  04. Integrate modalities (WNN-equivalent, VISUALISATION ONLY)
  05. Link peaks to genes (co-accessibility at 300kb, NOT 100kb)
  06. Compute TF motif enrichment (chromVAR-equivalent — the critical step)
  07. Export as MuData .h5mu for downstream ATACSeqEncoder

Key design decisions:
  - window_kb=300 is the default (Wout flagged 100 as testing shortcut)
  - TSS score < 4.0 → discard (failed ATAC library)
  - Prediction score < 0.5 → discard (doublets + ambient RNA)
  - WNN is for visualisation only — use MOFA+/MultiVI for encoder fusion

KNOWN STUB: Step 6 (compute_chromvar_scores in tf_motif_scanner.py) raises
NotImplementedError. The ATAC pipeline will crash at Step 6 when run on real
10x Multiome data. Steps 1-5 are functional. To run the full pipeline:
implement tf_motif_scanner.py with a chromVAR-equivalent using JASPAR 2024
PWMs. The pipeline runs end-to-end on mock data (tests use mocks).
  - ATACSeqEncoder input = chromvar_scores, NOT raw peak counts
"""
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import (
    SEED, JAKSTAT_GENES, StepTimer, log_cell_counts,
    compute_mt_fraction, compute_tss_enrichment,
)
from .peak_gene_linker import link_peaks_to_genes
from .tf_motif_scanner import compute_chromvar_scores, validate_ifn_beta_enrichment
from .cell_type_annotator import transfer_labels

logger = logging.getLogger("aivc.preprocessing")


class ATACRNAPipeline:
    """
    Processes 10x Multiome ATAC+RNA data for AIVC v3.0 training.

    Absorbs and extends Wout Megchelenbrink's R-based pipeline.
    Outputs MuData .h5mu compatible with ATACSeqEncoder.
    """

    def __init__(self):
        self.qc_report = {}
        self.pipeline_params = {}
        self._step_times = {}

    # ─────────────────────────────────────────────────────────
    # Step 01: Load multiome data
    # ─────────────────────────────────────────────────────────
    def step01_load_multiome(self, cellranger_output_dir: str):
        """
        Load filtered_feature_bc_matrix.h5 and atac_fragments.tsv.gz.
        Create MuData with two modalities: rna and atac.

        Computes:
          - RNA: mt gene fraction (genes starting with MT-)
          - ATAC: TSS enrichment score using fragment file
        """
        import muon as mu
        import anndata as ad

        with StepTimer("step01_load_multiome") as t:
            h5_path = os.path.join(cellranger_output_dir, "filtered_feature_bc_matrix.h5")
            fragments_path = os.path.join(cellranger_output_dir, "atac_fragments.tsv.gz")

            if os.path.exists(h5_path):
                logger.info(f"  Loading from {h5_path}")
                mdata = mu.read_10x_h5(h5_path)
            else:
                raise FileNotFoundError(
                    f"filtered_feature_bc_matrix.h5 not found in {cellranger_output_dir}"
                )

            # Compute QC metrics
            if "rna" in mdata.mod:
                compute_mt_fraction(mdata["rna"])
            if "atac" in mdata.mod:
                frag = fragments_path if os.path.exists(fragments_path) else None
                compute_tss_enrichment(mdata["atac"], frag)

            mdata.update()
            info = log_cell_counts(mdata, "step01_load")
            self.qc_report["step01"] = info

        self._step_times["step01"] = t.elapsed
        return mdata

    def step01_load_from_mudata(self, mdata):
        """
        Alternative entry: accept a pre-built MuData (for testing/mock data).
        Computes QC metrics if not already present.
        """
        with StepTimer("step01_load_from_mudata") as t:
            if "rna" in mdata.mod:
                rna = mdata["rna"]
                if "pct_mt" not in rna.obs.columns:
                    compute_mt_fraction(rna)
            if "atac" in mdata.mod:
                atac = mdata["atac"]
                if "tss_score" not in atac.obs.columns:
                    compute_tss_enrichment(atac)

            mdata.update()
            info = log_cell_counts(mdata, "step01_load")
            self.qc_report["step01"] = info

        self._step_times["step01"] = t.elapsed
        return mdata

    # ─────────────────────────────────────────────────────────
    # Step 02: Joint QC filter
    # ─────────────────────────────────────────────────────────
    def step02_filter_cells(
        self,
        mdata,
        min_genes: int = 200,
        max_mt_pct: float = 20.0,
        min_tss_score: float = 4.0,
        min_atac_fragments: int = 1000,
    ):
        """
        Joint QC filter — cell must pass BOTH RNA and ATAC thresholds.

        TSS score < 4.0 = failed ATAC library. Do NOT keep these cells.
        """
        with StepTimer("step02_filter_cells") as t:
            n_before = mdata.n_obs
            rna = mdata["rna"]
            atac = mdata["atac"]

            # RNA filters
            rna_keep = np.ones(rna.n_obs, dtype=bool)

            if "n_genes_by_counts" in rna.obs.columns:
                low_genes = rna.obs["n_genes_by_counts"] < min_genes
                rna_keep &= ~low_genes
                logger.info(f"  Low gene count (<{min_genes}): {low_genes.sum()} cells")
            elif "n_genes" in rna.obs.columns:
                low_genes = rna.obs["n_genes"] < min_genes
                rna_keep &= ~low_genes
                logger.info(f"  Low gene count (<{min_genes}): {low_genes.sum()} cells")
            else:
                # Compute gene counts from expression matrix
                X = rna.X
                if sparse.issparse(X):
                    n_genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
                else:
                    n_genes_per_cell = (X > 0).sum(axis=1)
                rna.obs["n_genes"] = n_genes_per_cell
                low_genes = n_genes_per_cell < min_genes
                rna_keep &= ~low_genes
                logger.info(f"  Low gene count (<{min_genes}): {low_genes.sum()} cells")

            if "pct_mt" in rna.obs.columns:
                high_mt = rna.obs["pct_mt"] > max_mt_pct
                rna_keep &= ~high_mt
                logger.info(f"  High mt% (>{max_mt_pct}%): {high_mt.sum()} cells")

            # ATAC filters
            atac_keep = np.ones(atac.n_obs, dtype=bool)

            if "tss_score" in atac.obs.columns:
                low_tss = atac.obs["tss_score"] < min_tss_score
                atac_keep &= ~low_tss
                logger.info(f"  Low TSS score (<{min_tss_score}): {low_tss.sum()} cells")

            if "n_fragments" in atac.obs.columns:
                low_frag = atac.obs["n_fragments"] < min_atac_fragments
                atac_keep &= ~low_frag
                logger.info(f"  Low ATAC fragments (<{min_atac_fragments}): {low_frag.sum()} cells")

            # Joint filter: cell must pass BOTH
            rna_barcodes = set(rna.obs_names[rna_keep])
            atac_barcodes = set(atac.obs_names[atac_keep])
            keep_barcodes = rna_barcodes & atac_barcodes

            # Apply filter to MuData
            keep_mask = np.array([bc in keep_barcodes for bc in mdata.obs_names])
            mdata_filtered = mdata[keep_mask].copy()
            mdata_filtered.update()

            n_after = mdata_filtered.n_obs
            logger.info(f"  Cells before: {n_before}, after: {n_after}, dropped: {n_before - n_after}")

            if n_after < n_before * 0.5:
                logger.warning(
                    f"  >50% of cells dropped ({n_before - n_after}/{n_before}). "
                    "Check QC thresholds."
                )

            info = log_cell_counts(mdata_filtered, "step02_filter")
            info["n_before"] = n_before
            info["n_after"] = n_after
            info["n_dropped"] = n_before - n_after
            self.qc_report["step02"] = info
            self.pipeline_params.update({
                "min_genes": min_genes,
                "max_mt_pct": max_mt_pct,
                "min_tss_score": min_tss_score,
                "min_atac_fragments": min_atac_fragments,
            })

        self._step_times["step02"] = t.elapsed
        return mdata_filtered

    # ─────────────────────────────────────────────────────────
    # Step 03: Annotate cell types
    # ─────────────────────────────────────────────────────────
    def step03_annotate_cell_types(
        self,
        mdata,
        reference_path: Optional[str] = None,
        min_pred_score: float = 0.5,
    ):
        """
        Transfer cell type labels from Hoa et al. PBMC multimodal reference.
        Discard cells with prediction score < 0.5.
        """
        with StepTimer("step03_annotate_cell_types") as t:
            n_before = mdata.n_obs

            rna = mdata["rna"]
            rna_annotated, discard_report = transfer_labels(
                rna, reference_path, min_pred_score
            )

            # Propagate cell type labels to MuData level
            if "cell_type" in rna_annotated.obs.columns:
                # Filter MuData to keep only retained cells
                keep_barcodes = set(rna_annotated.obs_names)
                keep_mask = np.array([bc in keep_barcodes for bc in mdata.obs_names])
                mdata_filtered = mdata[keep_mask].copy()
                mdata_filtered.obs["cell_type"] = rna_annotated.obs["cell_type"].values
                if "pred_score" in rna_annotated.obs.columns:
                    mdata_filtered.obs["pred_score"] = rna_annotated.obs["pred_score"].values
                mdata_filtered.update()
            else:
                mdata_filtered = mdata

            n_after = mdata_filtered.n_obs
            logger.info(f"  Cell type annotation: {n_before} → {n_after} cells")

            info = log_cell_counts(mdata_filtered, "step03_annotate")
            info["discard_report"] = discard_report
            self.qc_report["step03"] = info
            self.pipeline_params["min_pred_score"] = min_pred_score

        self._step_times["step03"] = t.elapsed
        return mdata_filtered

    # ─────────────────────────────────────────────────────────
    # Step 04: Integrate modalities (VISUALISATION ONLY)
    # ─────────────────────────────────────────────────────────
    def step04_integrate_modalities(self, mdata):
        """
        WNN-equivalent integration for QC VISUALISATION ONLY.
        DO NOT use as fusion mechanism for the AIVC model.

        Exports per-cell modality weights as mdata.obs['rna_weight'] and
        mdata.obs['atac_weight'] for use as training-time attention masks.
        """
        with StepTimer("step04_integrate_modalities") as t:
            n_cells = mdata.n_obs

            # Compute modality weights based on data quality
            rna_quality = np.ones(n_cells, dtype=np.float32)
            atac_quality = np.ones(n_cells, dtype=np.float32)

            rna = mdata["rna"]
            atac = mdata["atac"]

            # RNA weight: based on gene detection rate
            if "n_genes" in rna.obs.columns:
                gene_counts = rna.obs["n_genes"].values.astype(float)
                rna_quality = np.clip(gene_counts / max(np.median(gene_counts), 1), 0.1, 1.0)
            elif "n_genes_by_counts" in rna.obs.columns:
                gene_counts = rna.obs["n_genes_by_counts"].values.astype(float)
                rna_quality = np.clip(gene_counts / max(np.median(gene_counts), 1), 0.1, 1.0)

            # ATAC weight: based on TSS enrichment
            if "tss_score" in atac.obs.columns:
                tss = atac.obs["tss_score"].values.astype(float)
                atac_quality = np.clip(tss / max(np.median(tss), 1), 0.1, 1.0)

            # Normalise to sum to 1 per cell
            total = rna_quality + atac_quality
            mdata.obs["rna_weight"] = (rna_quality / total).astype(np.float32)
            mdata.obs["atac_weight"] = (atac_quality / total).astype(np.float32)

            logger.info(
                f"  Modality weights: RNA mean={mdata.obs['rna_weight'].mean():.3f}, "
                f"ATAC mean={mdata.obs['atac_weight'].mean():.3f}"
            )
            logger.info("  NOTE: WNN integration is for QC visualisation only.")
            logger.info("  Use MOFA+ or MultiVI for encoder fusion in production.")

            info = log_cell_counts(mdata, "step04_integrate")
            self.qc_report["step04"] = info

        self._step_times["step04"] = t.elapsed
        return mdata

    # ─────────────────────────────────────────────────────────
    # Step 05: Link peaks to genes
    # ─────────────────────────────────────────────────────────
    def step05_link_peaks_to_genes(self, mdata, window_kb: int = 300):
        """
        Assign ATAC peaks to genes via co-accessibility.
        CRITICAL: Use window_kb=300, NOT 100.
        """
        with StepTimer("step05_link_peaks_to_genes") as t:
            if window_kb < 300:
                logger.warning(
                    f"window_kb={window_kb} is below recommended 300kb. "
                    "At 100kb, ~35% of known PBMC enhancer-gene pairs are missed."
                )

            links_df = link_peaks_to_genes(mdata, window_kb=window_kb)

            # Store in MuData
            mdata.uns["peak_gene_links"] = links_df

            n_links = len(links_df)
            n_genes = links_df["gene_name"].nunique() if n_links > 0 else 0
            mean_ppg = links_df.groupby("gene_name").size().mean() if n_links > 0 else 0

            info = {
                "step": "step05_link_peaks",
                "n_links": n_links,
                "n_genes_with_links": n_genes,
                "mean_peaks_per_gene": float(mean_ppg),
                "window_kb": window_kb,
            }

            # JAK-STAT coverage
            jakstat_covered = [
                g for g in JAKSTAT_GENES
                if n_links > 0 and g in links_df["gene_name"].values
            ]
            info["jakstat_coverage"] = len(jakstat_covered)
            info["jakstat_covered"] = jakstat_covered

            self.qc_report["step05"] = info
            self.pipeline_params["window_kb"] = window_kb

        self._step_times["step05"] = t.elapsed
        return mdata

    # ─────────────────────────────────────────────────────────
    # Step 06: Compute TF motif enrichment (chromVAR-equivalent)
    # ─────────────────────────────────────────────────────────
    def step06_compute_tf_motif_enrichment(
        self, mdata, jaspar_db: str = "JASPAR2024"
    ):
        """
        Compute per-cell TF motif enrichment scores.
        THIS IS THE BIOLOGICALLY CRITICAL STEP.

        These scores are the ATACSeqEncoder input, NOT raw peak counts.
        """
        with StepTimer("step06_tf_motif_enrichment") as t:
            atac = mdata["atac"]

            scores, tf_names = compute_chromvar_scores(
                atac, jaspar_db=jaspar_db
            )

            # Store in MuData
            atac.obsm["chromvar_scores"] = scores
            atac.uns["tf_names"] = tf_names

            # Validate IFN-beta enrichment if condition info available
            if "condition" in mdata.obs.columns:
                conditions = mdata.obs["condition"].values
                validation = validate_ifn_beta_enrichment(
                    scores, tf_names, conditions
                )
                mdata.uns["tf_validation"] = validation
                self.qc_report["step06_validation"] = validation

            info = {
                "step": "step06_tf_motif",
                "n_tfs": len(tf_names),
                "score_shape": list(scores.shape),
                "score_range": [float(scores.min()), float(scores.max())],
                "jaspar_db": jaspar_db,
            }
            self.qc_report["step06"] = info
            self.pipeline_params["jaspar_db"] = jaspar_db

        self._step_times["step06"] = t.elapsed
        return mdata

    # ─────────────────────────────────────────────────────────
    # Step 07: Export MuData
    # ─────────────────────────────────────────────────────────
    def step07_export_mudata(self, mdata, output_path: str) -> None:
        """
        Export final processed MuData as .h5mu.

        Required contents:
          mdata['rna'].X              → normalised log1p RNA counts
          mdata['atac'].X             → binarised peak accessibility
          mdata['atac'].obsm['chromvar_scores'] → TF motif enrichment
          mdata.obs['cell_type']      → Hoa et al. labels
          mdata.obs['donor_id']       → for train/val/test split
          mdata.obs['condition']      → 'ctrl' or 'stim'
          mdata.obs['atac_weight']    → WNN ATAC modality weight
          mdata.uns['peak_gene_links'] → peak-gene co-accessibility
          mdata.uns['pipeline_params'] → all parameters used
          mdata.uns['qc_report']       → all QC statistics
        """
        with StepTimer("step07_export_mudata") as t:
            # Store pipeline metadata
            mdata.uns["pipeline_params"] = self.pipeline_params
            mdata.uns["qc_report"] = {
                k: (v if isinstance(v, (dict, str, int, float))
                     else str(v))
                for k, v in self.qc_report.items()
            }

            # Validate required fields
            required_obs = ["cell_type"]
            for field in required_obs:
                if field not in mdata.obs.columns:
                    logger.warning(f"  Missing mdata.obs['{field}']")

            if "atac" in mdata.mod:
                if "chromvar_scores" not in mdata["atac"].obsm:
                    logger.warning("  Missing mdata['atac'].obsm['chromvar_scores']")

            # Export
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            import muon as mu
            mdata.write(output_path)
            logger.info(f"  Exported MuData to: {output_path}")

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"  File size: {file_size_mb:.1f} MB")

        self._step_times["step07"] = t.elapsed

    # ─────────────────────────────────────────────────────────
    # Run full pipeline
    # ─────────────────────────────────────────────────────────
    def run_full_pipeline(
        self,
        cellranger_dir: str,
        output_path: str,
        window_kb: int = 300,
        condition: str = "stim",
        reference_path: Optional[str] = None,
    ):
        """
        Run all 7 steps end-to-end.
        Logs progress at each step with n_cells, n_features, time elapsed.
        """
        logger.info("=" * 70)
        logger.info("AIVC v3.0 ATAC-RNA Pipeline — Full Run")
        logger.info("=" * 70)

        t_start = time.time()

        # Step 1: Load
        mdata = self.step01_load_multiome(cellranger_dir)

        # Step 2: Filter
        mdata = self.step02_filter_cells(mdata)

        # Step 3: Annotate
        mdata = self.step03_annotate_cell_types(mdata, reference_path)

        # Set condition if not already present
        if "condition" not in mdata.obs.columns:
            mdata.obs["condition"] = condition

        # Step 4: Integrate
        mdata = self.step04_integrate_modalities(mdata)

        # Step 5: Link peaks to genes
        mdata = self.step05_link_peaks_to_genes(mdata, window_kb=window_kb)

        # Step 6: TF motifs
        mdata = self.step06_compute_tf_motif_enrichment(mdata)

        # Step 7: Export
        self.step07_export_mudata(mdata, output_path)

        t_total = time.time() - t_start
        logger.info(f"\nPipeline complete in {t_total:.1f}s")
        logger.info(f"  Final cells: {mdata.n_obs}")
        logger.info(f"  Output: {output_path}")

        return mdata
