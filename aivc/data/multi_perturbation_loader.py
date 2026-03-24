"""
Multi-Perturbation Data Loader for AIVC v1.1.

Unified loader for multi-perturbation training corpus:
  - Kang 2018 (IFN-B, PBMCs) — primary benchmark
  - Frangieh 2021 (IFN-G + CRISPR, melanoma) — shared JAK-STAT
  - ImmPort SDY702 (cytokines, PBMCs) — W training only
  - Replogle 2022 (CRISPRi, K562) — GRN causal directions only

CRITICAL RULES:
  1. Kang 2018 test donors are LOCKED and SACRED.
  2. Perturbation IDs never overlap: Kang [0,1], Frangieh [2..752], ImmPort [753+]
  3. Cell type embeddings shared for PBMCs only (Kang + ImmPort), not Frangieh.
  4. Normalisation identical: log1p, 10k counts, Kang 2018 HVG universe.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aivc.data")

SEED = 42

# Hard-coded perturbation ID ranges — never auto-assigned
PERT_ID_RANGES = {
    "kang": {"ctrl": 0, "stim": 1},
    "frangieh": {"ctrl": 2, "ifng": 3},  # CRISPR: 4..752
    "immport": {"ctrl": 753, "IL-2": 754, "IL-4": 755, "IL-6": 756,
                "IL-10": 757, "IFN-gamma": 758, "TNF-alpha": 759, "GM-CSF": 760},
}

DATASET_IDS = {
    "kang":        0,
    "frangieh":    1,   # W pretraining only (GRN edges, melanoma — NOT for response training)
    "immport":     2,   # W pretraining only (cytokine GRN)
    "replogle":    3,   # W pretraining only (CRISPRi GRN)
    "pbmc_ifng":   4,   # Response training (PBMC-native IFN-G)
}

# Which datasets can be used for perturbation response prediction training
RESPONSE_TRAINING_DATASETS = {"kang", "pbmc_ifng"}
# Which datasets are used ONLY for Neumann W matrix pre-training
W_PRETRAIN_ONLY_DATASETS = {"frangieh", "immport", "replogle"}


class MultiPerturbationLoader:
    """
    Unified data loader for multi-perturbation training corpus.

    Args:
        kang_path: Path to Kang 2018 h5ad.
        frangieh_path: Path to Frangieh 2021 h5ad (or empty to skip).
        immport_paths: List of ImmPort h5ad/csv paths (or empty to skip).
        replogle_path: Path to Replogle 2022 h5ad (or empty to skip).
        kang_test_donors: LOCKED test set donor IDs. NEVER altered.
        gene_universe: Kang 2018 HVG gene list. All datasets subset to this.
    """

    def __init__(
        self,
        kang_path: str = "",
        frangieh_path: str = "",
        immport_paths: Optional[list] = None,
        replogle_path: str = "",
        kang_test_donors: Optional[list] = None,
        gene_universe: Optional[list] = None,
    ):
        self.kang_path = kang_path
        self.frangieh_path = frangieh_path
        self.immport_paths = immport_paths or []
        self.replogle_path = replogle_path
        self.kang_test_donors = set(kang_test_donors or [])
        self.gene_universe = gene_universe
        self._datasets = {}

    def load_kang(self):
        """
        Load Kang 2018. Never alter condition column or test donors.

        Returns: AnnData with obs columns:
          perturbation_id, dataset_id, donor_id, condition, in_test_set
        """
        import anndata as ad

        adata = ad.read_h5ad(self.kang_path)
        logger.info(f"Kang 2018: {adata.n_obs} cells, {adata.n_vars} genes")

        # Set perturbation IDs
        pert_ids = []
        for label in adata.obs.get("label", adata.obs.get("condition", ["ctrl"] * adata.n_obs)):
            if label in ("stim", "stimulated"):
                pert_ids.append(PERT_ID_RANGES["kang"]["stim"])
            else:
                pert_ids.append(PERT_ID_RANGES["kang"]["ctrl"])
        adata.obs["perturbation_id"] = pert_ids
        adata.obs["dataset_id"] = DATASET_IDS["kang"]

        # Prefix donor IDs
        if "donor_id" not in adata.obs.columns:
            adata.obs["donor_id"] = "kang_unknown"
        else:
            adata.obs["donor_id"] = "kang_" + adata.obs["donor_id"].astype(str)

        # Mark test set
        adata.obs["in_test_set"] = adata.obs["donor_id"].isin(
            {f"kang_{d}" for d in self.kang_test_donors} | self.kang_test_donors
        )
        adata.obs["USE_FOR_W_ONLY"] = False

        n_test = adata.obs["in_test_set"].sum()
        logger.info(f"  Test cells (locked): {n_test}")

        if self.gene_universe is None:
            self.gene_universe = adata.var_names.tolist()

        self._datasets["kang"] = adata
        return adata

    def load_frangieh(self):
        """
        Load Frangieh 2021 via pertpy or from local h5ad.
        Subset to CTRL and IFN-G treated cells for initial integration.
        Remap genes to Kang 2018 gene universe.

        Returns: AnnData with Frangieh-specific perturbation IDs.
        """
        import anndata as ad
        import scanpy as sc

        if not self.frangieh_path:
            logger.info("Frangieh path not provided. Skipping.")
            return None

        adata = ad.read_h5ad(self.frangieh_path)
        logger.info(f"Frangieh 2021 raw: {adata.n_obs} cells, {adata.n_vars} genes")

        # Subset to ctrl + IFN-G only for Stage 2
        if "condition" in adata.obs.columns:
            mask = adata.obs["condition"].isin(["control", "ctrl", "IFN-gamma", "IFNG"])
            adata = adata[mask].copy()
        elif "perturbation" in adata.obs.columns:
            mask = adata.obs["perturbation"].isin(["control", "non-targeting", "IFN-gamma"])
            adata = adata[mask].copy()

        logger.info(f"  After ctrl/IFN-G filter: {adata.n_obs} cells")

        # Remap to Kang gene universe
        if self.gene_universe:
            common = list(set(adata.var_names) & set(self.gene_universe))
            if len(common) > 100:
                adata = adata[:, common].copy()
                logger.info(f"  Common genes with Kang: {len(common)}")
            else:
                logger.warning(f"  Only {len(common)} common genes. Frangieh may not help.")

        # Normalise
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Set perturbation IDs
        pert_ids = []
        for row in adata.obs.itertuples():
            cond = getattr(row, "condition", getattr(row, "perturbation", "control"))
            if cond in ("control", "ctrl", "non-targeting"):
                pert_ids.append(PERT_ID_RANGES["frangieh"]["ctrl"])
            else:
                pert_ids.append(PERT_ID_RANGES["frangieh"]["ifng"])
        adata.obs["perturbation_id"] = pert_ids
        adata.obs["dataset_id"] = DATASET_IDS["frangieh"]
        adata.obs["donor_id"] = "frangieh_" + adata.obs.index.astype(str)
        adata.obs["in_test_set"] = False
        adata.obs["USE_FOR_W_ONLY"] = True
        adata.obs["use_for_response_training"] = False

        if "cell_type" not in adata.obs.columns:
            adata.obs["cell_type"] = "melanoma"

        self._datasets["frangieh"] = adata
        return adata

    def load_immport(self):
        """
        Load ImmPort cytokine PBMC datasets.
        Convert bulk RNA to pseudo-single-cells (50 per sample).
        Mark as USE_FOR_W_ONLY = True.

        Returns: AnnData with pseudo-cells, or None if paths empty.
        """
        import anndata as ad
        import scanpy as sc

        if not self.immport_paths:
            logger.info("ImmPort paths not provided. Skipping.")
            return None

        rng = np.random.RandomState(SEED)
        all_cells = []

        for path in self.immport_paths:
            try:
                if path.endswith(".h5ad"):
                    bulk = ad.read_h5ad(path)
                elif path.endswith(".csv") or path.endswith(".tsv"):
                    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", index_col=0)
                    bulk = ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index),
                                      var=pd.DataFrame(index=df.columns))
                else:
                    logger.warning(f"  Unknown format: {path}. Skipping.")
                    continue

                logger.info(f"  ImmPort file: {path} ({bulk.n_obs} samples)")

                # Generate pseudo-cells: 50 per bulk sample
                for i in range(bulk.n_obs):
                    sample_expr = bulk.X[i] if not hasattr(bulk.X, "toarray") else bulk.X[i].toarray().flatten()
                    sample_var = np.abs(sample_expr) * 0.1 + 0.01
                    for _ in range(50):
                        pseudo = sample_expr + rng.normal(0, np.sqrt(sample_var))
                        pseudo = np.maximum(pseudo, 0)
                        all_cells.append(pseudo)
            except Exception as e:
                logger.error(f"  ImmPort load failed for {path}: {e}")

        if not all_cells:
            logger.info("  No ImmPort cells loaded.")
            return None

        X = np.array(all_cells, dtype=np.float32)
        n_pseudo = X.shape[0]
        logger.info(f"  ImmPort pseudo-cells: {n_pseudo}")

        adata = ad.AnnData(X=X)
        adata.obs["perturbation_id"] = PERT_ID_RANGES["immport"]["ctrl"]
        adata.obs["dataset_id"] = DATASET_IDS["immport"]
        adata.obs["donor_id"] = [f"immport_{i // 50}" for i in range(n_pseudo)]
        adata.obs["cell_type"] = "PBMC"
        adata.obs["in_test_set"] = False
        adata.obs["USE_FOR_W_ONLY"] = True
        adata.obs["condition"] = "ctrl"

        # Normalise
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        self._datasets["immport"] = adata
        return adata

    def load_replogle_for_grn(self) -> pd.DataFrame:
        """
        Load Replogle 2022. Extract causal direction matrix for W init.

        Returns: DataFrame (knockdown_gene x affected_gene) with
                 values = mean expression change direction (+1/-1/0).
                 Used ONLY in NeumannPropagation W initialisation.
        """
        if not self.replogle_path:
            logger.info("Replogle path not provided. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            import anndata as ad
            adata = ad.read_h5ad(self.replogle_path)
            logger.info(f"Replogle 2022: {adata.n_obs} cells, {adata.n_vars} genes")

            # Identify knockdown gene per cell
            ko_col = None
            for col in ["gene", "perturbation", "guide_identity", "target_gene"]:
                if col in adata.obs.columns:
                    ko_col = col
                    break

            if ko_col is None:
                logger.warning("No knockdown column found in Replogle. Returning empty.")
                return pd.DataFrame()

            # Get control cells
            ctrl_labels = ["non-targeting", "control", "NT", ""]
            ctrl_mask = adata.obs[ko_col].isin(ctrl_labels)
            if ctrl_mask.sum() == 0:
                logger.warning("No control cells in Replogle.")
                return pd.DataFrame()

            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()

            ctrl_mean = X[ctrl_mask].mean(axis=0)

            # Compute direction matrix for each knockdown
            ko_genes = [g for g in adata.obs[ko_col].unique() if g not in ctrl_labels]
            logger.info(f"  Knockdown genes: {len(ko_genes)}")

            # Limit to JAK-STAT-related genes for speed if dataset is huge
            gene_names = adata.var_names.tolist()
            directions = {}

            for ko_gene in ko_genes[:500]:  # cap at 500 for memory
                ko_mask = adata.obs[ko_col] == ko_gene
                if ko_mask.sum() < 5:
                    continue
                ko_mean = X[ko_mask].mean(axis=0)
                delta = ko_mean - ctrl_mean
                direction = np.sign(delta)
                directions[ko_gene] = direction

            if directions:
                dir_df = pd.DataFrame(directions, index=gene_names).T
                logger.info(f"  Direction matrix: {dir_df.shape}")
                return dir_df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Replogle load failed: {e}")
            return pd.DataFrame()

    def load_pbmc_ifng(self, path: str = None):
        """
        Load a PBMC-native IFN-G stimulation dataset.

        Priority:
          1. Local file at `path` (if provided and exists)
          2. pertpy built-in PBMC datasets (search for IFN-G)
          3. Synthetic fallback from Kang 2018 IFN-B data

        Returns: AnnData or None.
        """
        import os

        # Attempt 1: local file
        if path and os.path.exists(path):
            import anndata as ad
            logger.info(f"Loading PBMC IFN-G from local file: {path}")
            adata = ad.read_h5ad(path)
            validation = self._validate_pbmc_ifng_dataset(adata)
            if validation["valid"]:
                return self._prepare_pbmc_ifng(adata)
            else:
                logger.warning(f"Local file failed validation: {validation['errors']}")

        # Attempt 2: pertpy built-in
        try:
            import pertpy as pt
            for dataset_fn_name in ["pbmc_ifn", "kang_2018", "sc_sim_adata"]:
                try:
                    fn = getattr(pt.data, dataset_fn_name, None)
                    if fn is not None:
                        candidate = fn()
                        if self._has_ifng_condition(candidate):
                            validation = self._validate_pbmc_ifng_dataset(candidate)
                            if validation["valid"]:
                                logger.info(f"Using pertpy.data.{dataset_fn_name}() as PBMC IFN-G dataset.")
                                return self._prepare_pbmc_ifng(candidate)
                except Exception:
                    continue
        except ImportError:
            logger.warning("pertpy not installed. Cannot load built-in datasets.")

        # Attempt 3: Synthetic fallback
        logger.warning(
            "PBMC IFN-G dataset not found from local or pertpy sources.\n"
            "USING SYNTHETIC FALLBACK: scaling Kang 2018 IFN-B response "
            "to approximate IFN-G signature.\n"
            "Suggested real sources: OneK1K (Yazar 2022), Dixit 2016 (GSE90063)"
        )
        if "kang" in self._datasets:
            return self._make_synthetic_ifng_fallback(self._datasets["kang"])
        else:
            logger.error("Kang 2018 not loaded. Cannot create synthetic IFN-G fallback.")
            return None

    def _validate_pbmc_ifng_dataset(self, adata) -> dict:
        """Validate that a dataset meets PBMC IFN-G criteria."""
        errors = []

        if adata.n_vars < 500:
            errors.append(f"Too few genes: {adata.n_vars} (minimum 500)")

        donor_col = next(
            (c for c in ["donor_id", "donor", "replicate", "patient"]
             if c in adata.obs.columns), None
        )
        if donor_col:
            if adata.obs[donor_col].nunique() < 2:
                errors.append("Only 1 donor. Need >= 2 for cross-donor split.")

        gene_names = adata.var_names.tolist()
        for g in ["STAT1", "JAK1"]:
            if g not in gene_names:
                errors.append(f"Required gene {g} not in dataset.")

        n_isg = sum(1 for g in ["IFIT1", "MX1", "ISG15", "IFIT3", "MX2"] if g in gene_names)
        if n_isg == 0:
            errors.append("No IFN-stimulated genes (IFIT1/MX1/ISG15) found.")

        return {"valid": len(errors) == 0, "errors": errors}

    def _has_ifng_condition(self, adata) -> bool:
        """Check if AnnData contains IFN-G condition label."""
        for col in ["condition", "label", "perturbation", "stimulation"]:
            if col in adata.obs.columns:
                vals = adata.obs[col].astype(str).str.lower()
                if vals.str.contains("ifng|ifn.g|interferon.gamma", regex=True).any():
                    return True
        return False

    def _prepare_pbmc_ifng(self, adata):
        """Standardise a PBMC IFN-G dataset to match Kang 2018 format."""
        import scanpy as sc

        adata = adata.copy()

        if self.gene_universe:
            common = list(set(adata.var_names) & set(self.gene_universe))
            if len(common) >= 500:
                adata = adata[:, common].copy()

        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        pert_ids = []
        for row in adata.obs.itertuples():
            is_ctrl = False
            for col in ["condition", "label", "perturbation"]:
                val = str(getattr(row, col, "")).lower()
                if any(c in val for c in ["ctrl", "control", "unstim", "resting"]):
                    is_ctrl = True
                    break
            pert_ids.append(PERT_ID_RANGES["frangieh"]["ctrl"] if is_ctrl else PERT_ID_RANGES["frangieh"]["ifng"])

        adata.obs["perturbation_id"] = pert_ids
        adata.obs["dataset_id"] = DATASET_IDS["pbmc_ifng"]
        adata.obs["in_test_set"] = False
        adata.obs["USE_FOR_W_ONLY"] = False
        adata.obs["use_for_response_training"] = True

        if "donor_id" not in adata.obs.columns:
            donor_col = next(
                (c for c in ["donor", "replicate", "patient", "individual"]
                 if c in adata.obs.columns), None
            )
            if donor_col:
                adata.obs["donor_id"] = "pbmc_ifng_" + adata.obs[donor_col].astype(str)
            else:
                adata.obs["donor_id"] = "pbmc_ifng_unknown"

        logger.info(f"PBMC IFN-G dataset prepared: {adata.n_obs} cells, {adata.n_vars} genes")
        self._datasets["pbmc_ifng"] = adata
        return adata

    def _make_synthetic_ifng_fallback(self, kang_adata):
        """
        SYNTHETIC FALLBACK: approximate IFN-G from Kang 2018 IFN-B.
        OAS1/OAS2 scaled to 50% (weaker in IFN-G). Noise added.
        Flagged as SYNTHETIC_IFNG. Not for production benchmarks.
        """
        import anndata as ad
        import scipy.sparse as sp

        logger.warning(
            "Creating SYNTHETIC IFN-G fallback from Kang 2018. "
            "Flag all results as SYNTHETIC_IFNG."
        )
        rng = np.random.RandomState(SEED)

        stim_mask = kang_adata.obs.get("label", kang_adata.obs.get("condition")) == "stim"
        adata_stim = kang_adata[stim_mask].copy()

        X = adata_stim.X
        if sp.issparse(X):
            X = X.toarray().astype(np.float32)
        else:
            X = X.astype(np.float32)

        gene_names = adata_stim.var_names.tolist()
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        for g in ["OAS1", "OAS2", "OAS3", "OASL"]:
            if g in gene_to_idx:
                X[:, gene_to_idx[g]] *= 0.5

        noise = rng.normal(0, 0.1, X.shape).astype(np.float32)
        X = np.maximum(X + noise, 0.0)

        adata_synth = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=adata_stim.obs.copy(),
            var=adata_stim.var.copy(),
        )
        adata_synth.obs["perturbation_id"] = PERT_ID_RANGES["frangieh"]["ifng"]
        adata_synth.obs["dataset_id"] = DATASET_IDS["pbmc_ifng"]
        adata_synth.obs["in_test_set"] = False
        adata_synth.obs["USE_FOR_W_ONLY"] = False
        adata_synth.obs["use_for_response_training"] = True
        adata_synth.obs["SYNTHETIC_IFNG"] = True
        adata_synth.uns["synthetic_ifng"] = True

        logger.warning(f"Synthetic IFN-G: {adata_synth.n_obs} cells. OAS1/OAS2 scaled 50%.")
        self._datasets["pbmc_ifng"] = adata_synth
        return adata_synth

    def build_combined_corpus(
        self,
        include_frangieh: bool = True,
        include_immport: bool = True,
    ):
        """
        Concatenate all loaded datasets into one AnnData.

        Returns: combined AnnData with unified obs columns:
          perturbation_id, dataset_id, cell_type, donor_id,
          condition, USE_FOR_W_ONLY, in_test_set
        """
        import anndata as ad

        parts = []

        if "kang" in self._datasets:
            parts.append(self._datasets["kang"])

        if include_frangieh and "frangieh" in self._datasets:
            parts.append(self._datasets["frangieh"])

        if include_immport and "immport" in self._datasets:
            parts.append(self._datasets["immport"])

        if not parts:
            logger.warning("No datasets loaded. Call load_* methods first.")
            return None

        # Ensure consistent columns before concat
        required_cols = [
            "perturbation_id", "dataset_id", "donor_id",
            "in_test_set", "USE_FOR_W_ONLY",
        ]
        for part in parts:
            for col in required_cols:
                if col not in part.obs.columns:
                    part.obs[col] = False if col.startswith(("in_", "USE_")) else 0

        # Find common genes
        if self.gene_universe:
            for i, part in enumerate(parts):
                common = list(set(part.var_names) & set(self.gene_universe))
                if common:
                    parts[i] = part[:, common].copy()

        combined = ad.concat(parts, join="inner", label="dataset_source")
        combined.obs_names_make_unique()

        # Verify test donor integrity
        n_test = combined.obs["in_test_set"].sum()
        kang_test = combined[combined.obs["in_test_set"]]
        if n_test > 0:
            test_datasets = kang_test.obs["dataset_id"].unique()
            assert all(d == DATASET_IDS["kang"] for d in test_datasets), \
                "Test cells must only come from Kang 2018!"

        logger.info(f"Combined corpus: {combined.n_obs} cells, {combined.n_vars} genes")
        for ds_name, ds_id in DATASET_IDS.items():
            n = (combined.obs["dataset_id"] == ds_id).sum()
            if n > 0:
                logger.info(f"  {ds_name}: {n} cells")

        return combined

    def get_perturbation_id_map(self) -> dict:
        """Return the full perturbation ID mapping."""
        return PERT_ID_RANGES.copy()

    def verify_no_test_leakage(self, corpus) -> bool:
        """
        Verify Kang 2018 test donors never appear in training data.

        Returns True if clean, raises AssertionError if leakage detected.
        """
        if corpus is None:
            return True

        test_cells = corpus[corpus.obs["in_test_set"]]
        train_cells = corpus[~corpus.obs["in_test_set"]]

        if len(test_cells) == 0:
            return True

        test_donors = set(test_cells.obs["donor_id"].unique())
        train_donors = set(train_cells.obs["donor_id"].unique())

        overlap = test_donors & train_donors
        if overlap:
            raise AssertionError(
                f"TEST DONOR LEAKAGE DETECTED! "
                f"Donors in both test and train: {overlap}"
            )

        logger.info(f"  No test leakage: {len(test_donors)} test donors, "
                     f"{len(train_donors)} train donors, 0 overlap.")
        return True
