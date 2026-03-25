"""
TileDB-SOMA store manager for AIVC multi-omics data.

Replaces h5ad file loading with a scalable cell store.
Migration strategy: ADDITIVE — existing h5ad path unchanged.
Switch via USE_SOMA_STORE flag in soma_training_bridge.py.

tiledbsoma is an OPTIONAL dependency. If not installed:
  - All tests pass (mocked)
  - Ingestion scripts print clear installation instructions
  - Training falls back to h5ad automatically

KNOWN BUG (open): ingest_from_anndata() enriches local obs/var DataFrames
with is_jakstat, is_hvg, USE_FOR_W_ONLY, etc., but passes the ORIGINAL adata
(not the enriched copy) to sio.from_anndata(). All metadata enrichment is
silently dropped on ingestion. All SOMA queries on is_jakstat / USE_FOR_W_ONLY
return wrong results until this is fixed.
FIX (not yet committed): add adata.obs = obs; adata.var = var immediately
before the sio.from_anndata() call.

Scale projections (reference, not tested):
  10k cells:  SOMA 0.5s vs h5ad 1.0s (2x)
  30k cells:  SOMA 1.5s vs h5ad 3.0s (2x)
  1M cells:   SOMA 15s  vs h5ad 480s (32x), h5ad needs 60GB RAM
  10M cells:  SOMA 120s vs h5ad infeasible (would need 600GB RAM)
"""
import logging
from typing import Optional

logger = logging.getLogger("aivc.data.soma")

SCALE_PROJECTIONS = {
    "10k_cells": {
        "X_matrix_sparse_gb": 0.06,
        "X_matrix_dense_gb": 0.22,
        "obs_metadata_mb": 2.0,
        "soma_load_time_s": 0.5,
        "h5ad_load_time_s": 1.0,
    },
    "30k_cells": {
        "X_matrix_sparse_gb": 0.18,
        "X_matrix_dense_gb": 0.65,
        "obs_metadata_mb": 6.0,
        "soma_load_time_s": 1.5,
        "h5ad_load_time_s": 3.0,
    },
    "1M_cells": {
        "X_matrix_sparse_gb": 6.0,
        "X_matrix_dense_gb": 22.0,
        "obs_metadata_mb": 200.0,
        "soma_load_time_s": 15.0,
        "h5ad_load_time_s": 480.0,
        "h5ad_ram_required_gb": 60.0,
    },
    "10M_cells": {
        "X_matrix_sparse_gb": 60.0,
        "soma_load_time_s": 120.0,
        "h5ad_load_time_s": "infeasible",
    },
}


class AIVCSomaStore:
    """
    TileDB-SOMA store manager for AIVC multi-omics data.

    Args:
        store_path: str — path to TileDB-SOMA experiment directory.
        mode:       str — "r" (read-only) or "w" (write/create).
    """

    OBS_SCHEMA = {
        "cell_id": str,
        "donor_id": str,
        "cell_type": str,
        "condition": str,
        "perturbation_id": int,
        "dataset_id": int,
        "in_test_set": bool,
        "USE_FOR_W_ONLY": bool,
        "use_for_response_training": bool,
        "SYNTHETIC_IFNG": bool,
        "ambient_decontaminated": bool,
        "partition_key": int,
    }

    VAR_SCHEMA = {
        "gene_id": str,
        "gene_name": str,
        "is_hvg": bool,
        "in_string_ppi": bool,
        "is_jakstat": bool,
        "is_housekeeping": bool,
    }

    JAKSTAT_GENES = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
        "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    def __init__(self, store_path: str, mode: str = "r"):
        self.store_path = store_path
        self.mode = mode
        self._experiment = None

    # ─── WRITE PATH ───

    def ingest_from_anndata(
        self,
        adata,
        measurement_name: str = "RNA",
        layer: str = "raw",
        gene_universe: list = None,
        string_ppi_genes: set = None,
        overwrite: bool = False,
    ) -> None:
        """Ingest an AnnData into the SOMA store."""
        try:
            import tiledbsoma
            import tiledbsoma.io as sio
        except ImportError:
            raise ImportError(
                "tiledbsoma not installed. Install with: pip install tiledbsoma"
            )

        import numpy as np

        obs = adata.obs.copy()
        defaults = {
            "donor_id": "unknown",
            "cell_type": "unknown",
            "condition": "unknown",
            "perturbation_id": -1,
            "dataset_id": -1,
            "in_test_set": False,
            "USE_FOR_W_ONLY": False,
            "use_for_response_training": True,
            "SYNTHETIC_IFNG": False,
            "ambient_decontaminated": False,
        }
        for col, default in defaults.items():
            if col not in obs.columns:
                obs[col] = default

        obs["cell_id"] = adata.obs_names.tolist()
        obs["partition_key"] = [hash(cid) % 1024 for cid in obs["cell_id"]]

        # Prepare var
        var = adata.var.copy()
        gene_names = adata.var_names.tolist()

        if "gene_name" not in var.columns:
            var["gene_name"] = gene_names
        if "gene_id" not in var.columns:
            var["gene_id"] = var.get("gene_ids", var["gene_name"])

        if gene_universe is not None:
            hv_set = set(gene_universe)
            var["is_hvg"] = [g in hv_set for g in gene_names]
        else:
            var["is_hvg"] = True

        if string_ppi_genes is not None:
            var["in_string_ppi"] = [g in string_ppi_genes for g in gene_names]
        else:
            var["in_string_ppi"] = False

        jakstat_set = set(self.JAKSTAT_GENES)
        var["is_jakstat"] = [g in jakstat_set for g in gene_names]

        from aivc.data.housekeeping_genes import get_housekeeping_genes
        hk_set = get_housekeeping_genes()
        var["is_housekeeping"] = [g in hk_set for g in gene_names]

        if "chromosome" not in var.columns:
            var["chromosome"] = ""

        # Update adata metadata before ingestion
        adata.obs = obs
        adata.var = var

        logger.info(
            f"Ingesting {adata.n_obs:,} cells x {adata.n_vars:,} genes "
            f"into {self.store_path} [{measurement_name}/{layer}]"
        )

        sio.from_anndata(
            experiment_uri=self.store_path,
            anndata=adata,
            measurement_name=measurement_name,
        )
        logger.info(f"Ingestion complete: {self.store_path}")

    def ingest_kang2018(self, adata_path: str, gene_universe: list = None) -> None:
        """Convenience method: ingest Kang 2018 with correct metadata."""
        import anndata as ad

        logger.info(f"Loading Kang 2018: {adata_path}")
        adata = ad.read_h5ad(adata_path)

        if "perturbation_id" not in adata.obs.columns:
            adata.obs["perturbation_id"] = adata.obs.get(
                "label", "ctrl"
            ).map({"ctrl": 0, "stim": 1, "control": 0, "stimulated": 1}).fillna(0).astype(int)

        adata.obs["dataset_id"] = 0
        adata.obs["USE_FOR_W_ONLY"] = False
        adata.obs["use_for_response_training"] = True
        adata.obs["SYNTHETIC_IFNG"] = False
        adata.obs["ambient_decontaminated"] = False

        self.ingest_from_anndata(adata, gene_universe=gene_universe)

    # ─── READ PATH ───

    def open(self):
        """Open the SOMA experiment for reading."""
        try:
            import tiledbsoma as soma
        except ImportError:
            raise ImportError("pip install tiledbsoma")
        self._experiment = soma.Experiment.open(self.store_path)
        return self

    def close(self):
        """Close the SOMA experiment."""
        if self._experiment is not None:
            self._experiment.close()
            self._experiment = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def query(
        self,
        measurement_name: str = "RNA",
        obs_query: str = None,
        var_query: str = None,
    ):
        """
        Query the SOMA store with optional filters.

        Args:
            measurement_name: "RNA", "protein", etc.
            obs_query: SQL-like filter on cell metadata.
            var_query: SQL-like filter on gene metadata.
        """
        if self._experiment is None:
            raise RuntimeError("Call open() or use as context manager first.")

        import tiledbsoma as soma
        return self._experiment.axis_query(
            measurement_name=measurement_name,
            obs_query=soma.AxisQuery(value_filter=obs_query) if obs_query else soma.AxisQuery(),
            var_query=soma.AxisQuery(value_filter=var_query) if var_query else soma.AxisQuery(),
        )

    # ─── PYTORCH INTEGRATION ───

    def get_pytorch_loader(
        self,
        obs_query: str = None,
        var_query: str = "is_hvg == True",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        """
        Create a PyTorch DataLoader that streams cells from SOMA.

        Returns DataLoader yielding dicts with:
          X, cell_type_idx, perturbation_id, dataset_id, condition, donor_id
        """
        try:
            from tiledbsoma.ml import ExperimentDataPipe
            import torch
            import torch.utils.data as tud
        except ImportError:
            raise ImportError("pip install 'tiledbsoma[ml]'")

        query = self.query(obs_query=obs_query, var_query=var_query)

        datapipe = ExperimentDataPipe(
            query,
            X_name="raw",
            obs_column_names=[
                "cell_type", "perturbation_id", "dataset_id",
                "condition", "donor_id", "in_test_set",
            ],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

        def collate_fn(batch):
            X_batch, obs_batch = batch
            return {
                "X": X_batch.to_dense().float(),
                "cell_type_idx": torch.zeros(X_batch.shape[0], dtype=torch.long),
                "perturbation_id": torch.tensor(obs_batch["perturbation_id"].tolist(), dtype=torch.long),
                "dataset_id": torch.tensor(obs_batch["dataset_id"].tolist(), dtype=torch.long),
                "condition": obs_batch["condition"].tolist(),
                "donor_id": obs_batch["donor_id"].tolist(),
            }

        return tud.DataLoader(
            datapipe, batch_size=None, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    # ─── VALIDATION ───

    def validate_store(self) -> dict:
        """Validate the SOMA store contents."""
        checks = []
        try:
            with self as store:
                checks.append(("store_opens", True, ""))
                try:
                    query = store.query()
                    obs_df = query.obs().concat().to_pandas()
                    missing = [c for c in self.OBS_SCHEMA if c not in obs_df.columns]
                    checks.append(("obs_columns", len(missing) == 0, f"Missing: {missing}" if missing else ""))

                    n_cells = len(obs_df)
                    checks.append(("n_cells", n_cells > 0, f"{n_cells:,} cells"))

                    test_donors = set(obs_df[obs_df["in_test_set"]]["donor_id"].unique())
                    train_donors = set(obs_df[~obs_df["in_test_set"]]["donor_id"].unique())
                    overlap = test_donors & train_donors
                    checks.append(("no_test_leakage", len(overlap) == 0, f"Overlap: {overlap}" if overlap else ""))
                except Exception as e:
                    checks.append(("query_check", False, str(e)))
        except Exception as e:
            checks.append(("store_opens", False, str(e)))

        valid = all(c[1] for c in checks)
        for name, passed, msg in checks:
            logger.info(f"  [{'PASS' if passed else 'FAIL'}] {name}: {msg}")
        return {"valid": valid, "checks": checks}

    def validate_obs_schema(self, adata) -> dict:
        """
        Validate AnnData has required obs columns before ingestion.
        Warns if Frangieh cells are marked for response training.
        """
        generated_cols = {"cell_id", "partition_key"}
        missing = [
            col for col in self.OBS_SCHEMA
            if col not in adata.obs.columns and col not in generated_cols
        ]
        warnings = []

        if "USE_FOR_W_ONLY" in adata.obs.columns and "dataset_id" in adata.obs.columns:
            frangieh_mask = adata.obs["dataset_id"] == 1
            if frangieh_mask.any():
                frangieh_response = (
                    frangieh_mask & (~adata.obs["USE_FOR_W_ONLY"])
                ).sum()
                if frangieh_response > 0:
                    warnings.append(
                        f"WARNING: {frangieh_response} Frangieh cells "
                        f"have USE_FOR_W_ONLY=False. "
                        f"These should NOT enter response training."
                    )

        return {"valid": len(missing) == 0, "missing": missing, "warnings": warnings}
