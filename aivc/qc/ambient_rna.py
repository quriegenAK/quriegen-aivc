"""
Python-native ambient RNA decontamination for droplet-based scRNA-seq.

Implements the core SoupX logic (Young & Behjati, Genome Biology 2020):
  1. Estimate ambient RNA profile from empty droplets
  2. Compute per-cell contamination fraction
  3. Subtract expected ambient contribution

Two modes:
  PRIMARY:  Uses raw CellRanger output (empty droplets available)
  FALLBACK: Uses low-UMI cells within filtered matrix as proxy
"""
import logging
import os

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("aivc.qc.ambient")


class AmbientRNAEstimator:
    """
    Ambient RNA decontamination for droplet-based scRNA-seq.

    Args:
        filtered_adata:     AnnData — filtered cell matrix
        raw_matrix_path:    str — path to raw CellRanger output dir
                            (expects barcodes.tsv.gz, features.tsv.gz,
                             matrix.mtx.gz in standard 10x format)
        low_umi_threshold:  int — UMI cutoff for empty droplet detection
                            (default 100: cells with < 100 UMIs are empty)
        contamination_cap:  float — max estimated contamination fraction
                            (default 0.2: cap at 20% per cell)
        seed:               int — reproducibility (default 42)
    """

    JAKSTAT_GENES = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
        "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    def __init__(
        self,
        filtered_adata,
        raw_matrix_path: str = None,
        low_umi_threshold: int = 100,
        contamination_cap: float = 0.20,
        seed: int = 42,
    ):
        self.adata = filtered_adata.copy()
        self.raw_matrix_path = raw_matrix_path
        self.low_umi_threshold = low_umi_threshold
        self.contamination_cap = contamination_cap
        self.seed = seed
        self.mode = None
        self.ambient_profile = None
        self.contamination_rate = None
        self._log = []

    def _log_step(self, msg: str):
        logger.info(msg)
        self._log.append(msg)

    # ─────────────────────────────────────────────────────────
    # Step 1: Estimate ambient RNA profile
    # ─────────────────────────────────────────────────────────

    def estimate_ambient_profile(self) -> np.ndarray:
        """
        Estimate the ambient RNA profile from empty droplets.

        PRIMARY MODE: Load raw CellRanger output, identify empty droplets
            (total UMI < low_umi_threshold), compute mean expression.
        FALLBACK MODE: Use cells in filtered matrix with total UMI in
            bottom 5% as proxies for empty droplets.

        Returns:
            np.ndarray (n_genes,) — ambient RNA probability distribution
            summing to 1.
        """
        if self.raw_matrix_path and self._raw_matrix_exists():
            self.mode = "primary"
            self._log_step(f"Mode: PRIMARY (raw matrix at {self.raw_matrix_path})")
            ambient_counts = self._load_empty_droplets()
        else:
            self.mode = "fallback"
            self._log_step(
                "Mode: FALLBACK (raw matrix not found). "
                "Results are APPROXIMATE. Download raw matrix from GSE96583 "
                "for precise decontamination.\n"
                "Download instructions:\n"
                "  1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583\n"
                "  2. Download: GSM2560245_*.tar.gz files (ctrl and stim)\n"
                "  3. Extract to: data/kang2018_raw/\n"
                "  4. Re-run this script"
            )
            ambient_counts = self._estimate_ambient_from_low_umi_cells()

        # Compute mean ambient profile
        if sp.issparse(ambient_counts):
            ambient_mean = np.asarray(ambient_counts.mean(axis=0)).flatten()
        else:
            ambient_mean = np.asarray(ambient_counts).mean(axis=0).flatten()

        # Normalise to probability distribution
        total = ambient_mean.sum()
        if total < 1e-10:
            raise ValueError(
                "Ambient profile sums to near-zero. "
                "Check that empty droplets contain some RNA signal."
            )
        self.ambient_profile = ambient_mean / total

        self._log_step(
            f"Ambient profile estimated from "
            f"{'empty droplets' if self.mode == 'primary' else 'low-UMI cells'}. "
            f"Top 5 ambient genes (by fraction): "
            f"{self._top_ambient_genes(n=5)}"
        )
        return self.ambient_profile

    def _raw_matrix_exists(self) -> bool:
        if not self.raw_matrix_path:
            return False
        required = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
        return all(
            os.path.exists(os.path.join(self.raw_matrix_path, f))
            for f in required
        )

    def _load_empty_droplets(self):
        """Load empty droplets from raw CellRanger output."""
        import gzip
        import scipy.io

        mtx_path = os.path.join(self.raw_matrix_path, "matrix.mtx.gz")
        with gzip.open(mtx_path, "rb") as f:
            raw_matrix = scipy.io.mmread(f).T.tocsr()  # cells x genes

        umi_per_barcode = np.asarray(raw_matrix.sum(axis=1)).flatten()
        empty_mask = umi_per_barcode < self.low_umi_threshold
        n_empty = empty_mask.sum()

        self._log_step(
            f"Empty droplets (UMI < {self.low_umi_threshold}): "
            f"{n_empty:,} / {len(umi_per_barcode):,}"
        )
        if n_empty < 100:
            raise ValueError(
                f"Only {n_empty} empty droplets found with threshold "
                f"{self.low_umi_threshold}. Lower the threshold."
            )
        return raw_matrix[empty_mask]

    def _estimate_ambient_from_low_umi_cells(self):
        """
        FALLBACK: Use bottom 5% UMI cells in filtered matrix as
        ambient-dominated droplet proxies.
        """
        X = self.adata.X
        if sp.issparse(X):
            umi_per_cell = np.asarray(X.sum(axis=1)).flatten()
        else:
            umi_per_cell = np.asarray(X).sum(axis=1)

        threshold = np.percentile(umi_per_cell, 5)
        low_umi_mask = umi_per_cell <= threshold
        n_low = int(low_umi_mask.sum())

        self._log_step(
            f"FALLBACK: Using {n_low} low-UMI cells "
            f"(bottom 5%, UMI <= {threshold:.0f}) as ambient proxy"
        )

        if sp.issparse(X):
            return X[low_umi_mask]
        else:
            return X[low_umi_mask]

    def _top_ambient_genes(self, n: int = 10) -> list:
        """Return top-n genes by ambient fraction."""
        if self.ambient_profile is None:
            return []
        gene_names = self.adata.var_names.tolist()
        top_idx = np.argsort(self.ambient_profile)[::-1][:n]
        return [(gene_names[i], float(self.ambient_profile[i])) for i in top_idx]

    # ─────────────────────────────────────────────────────────
    # Step 2: Estimate per-cell contamination fraction
    # ─────────────────────────────────────────────────────────

    def estimate_contamination_per_cell(self) -> np.ndarray:
        """
        Estimate what fraction of each cell's RNA is ambient contamination.

        Uses IFN-response marker genes (IFIT1, MX1, ISG15) that should be
        near-zero in control cells. Any observed expression in ctrl is
        likely ambient contamination from lysed stimulated cells.

        Returns:
            np.ndarray (n_cells,) — contamination fraction per cell,
            clipped to [0, contamination_cap].
        """
        if self.ambient_profile is None:
            raise RuntimeError("Call estimate_ambient_profile() first.")

        gene_names = self.adata.var_names.tolist()
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Select IFN-response marker genes for contamination estimation
        marker_genes_for_rho = [
            g for g in ["IFIT1", "MX1", "ISG15", "IFIT3", "IFITM1", "MX2"]
            if g in gene_to_idx
        ]

        if len(marker_genes_for_rho) < 2:
            top_ambient = self._top_ambient_genes(n=10)
            marker_genes_for_rho = [g for g, _ in top_ambient if g in gene_to_idx][:5]

        self._log_step(f"Contamination estimation marker genes: {marker_genes_for_rho}")

        marker_idx = [gene_to_idx[g] for g in marker_genes_for_rho]
        ambient_marker = self.ambient_profile[marker_idx]

        X = self.adata.X
        if sp.issparse(X):
            X_dense = np.asarray(X.todense(), dtype=np.float64)
        else:
            X_dense = np.array(X, dtype=np.float64)

        umi_per_cell = X_dense.sum(axis=1)
        observed_marker = X_dense[:, marker_idx]

        # rho_c = observed[c, g] / (ambient_profile[g] * total_UMI_c)
        rho_estimates = np.zeros((len(X_dense), len(marker_idx)))
        for j, (m_idx, amb_frac) in enumerate(zip(marker_idx, ambient_marker)):
            expected_if_rho1 = amb_frac * umi_per_cell
            safe_expected = np.maximum(expected_if_rho1, 1e-6)
            rho_estimates[:, j] = observed_marker[:, j] / safe_expected

        rho_per_cell = np.median(rho_estimates, axis=1)
        rho_per_cell = np.clip(rho_per_cell, 0.0, self.contamination_cap)
        self.contamination_rate = rho_per_cell

        self._log_step(
            f"Contamination rates — "
            f"mean: {rho_per_cell.mean():.4f} "
            f"median: {np.median(rho_per_cell):.4f} "
            f"p95: {np.percentile(rho_per_cell, 95):.4f} "
            f"max: {rho_per_cell.max():.4f}"
        )
        return self.contamination_rate

    # ─────────────────────────────────────────────────────────
    # Step 3: Remove ambient contribution
    # ─────────────────────────────────────────────────────────

    def decontaminate(self):
        """
        Subtract expected ambient contribution from each cell.

        For each cell c and gene g:
          decontaminated[c, g] = observed[c, g]
                                 - rho_c * ambient_profile[g] * total_UMI_c

        Counts clipped to >= 0. Returns a NEW AnnData (original unchanged).
        """
        import anndata as ad

        if self.contamination_rate is None:
            raise RuntimeError(
                "Call estimate_ambient_profile() and "
                "estimate_contamination_per_cell() first."
            )

        X = self.adata.X
        if sp.issparse(X):
            X_dense = np.asarray(X.todense(), dtype=np.float32)
        else:
            X_dense = np.array(X, dtype=np.float32)

        umi_per_cell = X_dense.sum(axis=1)

        ambient_expected = (
            self.contamination_rate[:, np.newaxis]
            * self.ambient_profile[np.newaxis, :]
            * umi_per_cell[:, np.newaxis]
        )

        X_clean = np.maximum(X_dense - ambient_expected, 0.0)

        adata_clean = ad.AnnData(
            X=sp.csr_matrix(X_clean),
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy(),
        )
        adata_clean.uns["ambient_decontamination"] = {
            "mode":               self.mode,
            "contamination_mean": float(self.contamination_rate.mean()),
            "contamination_p50":  float(np.median(self.contamination_rate)),
            "contamination_p95":  float(np.percentile(self.contamination_rate, 95)),
            "threshold_used":     self.low_umi_threshold,
            "genes_affected":     int((ambient_expected > 0.1).sum()),
            "log":                self._log,
        }

        self._log_step(
            f"Decontamination complete. "
            f"Clean matrix shape: {X_clean.shape}. "
            f"Negative values clipped to 0: "
            f"{int((X_dense - ambient_expected < 0).sum()):,} entries."
        )
        return adata_clean

    # ─────────────────────────────────────────────────────────
    # Step 4: Per-gene contamination report
    # ─────────────────────────────────────────────────────────

    def gene_contamination_report(
        self,
        adata_clean,
        condition_col: str = "label",
        ctrl_label: str = "ctrl",
    ):
        """
        Per-gene report: what fraction of expression in ctrl cells
        is estimated to be ambient contamination?

        Returns pd.DataFrame with columns: gene_name, ctrl_raw_mean,
        ctrl_clean_mean, ambient_mean, ambient_fraction, is_jakstat,
        interpretation.
        """
        import pandas as pd

        gene_names = self.adata.var_names.tolist()

        if condition_col in self.adata.obs.columns:
            ctrl_mask = (self.adata.obs[condition_col] == ctrl_label).values
        else:
            ctrl_mask = np.ones(self.adata.n_obs, dtype=bool)
            self._log_step(
                f"Warning: condition column '{condition_col}' not found. "
                "Using all cells."
            )

        n_ctrl = int(ctrl_mask.sum())
        self._log_step(f"Control cells for report: {n_ctrl}")

        X_orig = self.adata.X
        if sp.issparse(X_orig):
            X_orig = np.asarray(X_orig.todense())
        ctrl_raw_mean = np.asarray(X_orig[ctrl_mask]).mean(axis=0).flatten()

        X_clean = adata_clean.X
        if sp.issparse(X_clean):
            X_clean = np.asarray(X_clean.todense())
        ctrl_clean_mean = np.asarray(X_clean[ctrl_mask]).mean(axis=0).flatten()

        ambient_contribution = ctrl_raw_mean - ctrl_clean_mean
        ambient_fraction = np.where(
            ctrl_raw_mean > 1e-6,
            ambient_contribution / ctrl_raw_mean,
            0.0,
        )
        ambient_fraction = np.clip(ambient_fraction, 0.0, 1.0)

        jakstat_set = set(self.JAKSTAT_GENES)
        rows = []
        for i, gene in enumerate(gene_names):
            af = float(ambient_fraction[i])
            if af < 0.05:
                interp = "CLEAN"
            elif af < 0.10:
                interp = "MODERATE"
            else:
                interp = "CONTAMINATED"
            rows.append({
                "gene_name":        gene,
                "ctrl_raw_mean":    float(ctrl_raw_mean[i]),
                "ctrl_clean_mean":  float(ctrl_clean_mean[i]),
                "ambient_mean":     float(ambient_contribution[i]),
                "ambient_fraction": af,
                "is_jakstat":       gene in jakstat_set,
                "interpretation":   interp,
            })

        df = pd.DataFrame(rows).sort_values(
            "ambient_fraction", ascending=False
        ).reset_index(drop=True)
        return df
