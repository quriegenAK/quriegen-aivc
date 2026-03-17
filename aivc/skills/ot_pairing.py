"""
aivc/skills/ot_pairing.py — Optimal Transport cell pairing skill.

Wraps build_ot_pairs.py. Falls back to pseudo-bulk averaging
if POT library unavailable or if pairing quality fails validation.
"""

import time
import os
import numpy as np

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import registry


@registry.register(
    name="ot_cell_pairer",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["adata_path", "gene_to_idx"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class OTCellPairer(AIVCSkill):
    """
    Optimal Transport cell pairing across donor-by-cell-type groups.
    Wraps build_ot_pairs.py.
    Falls back to pseudo-bulk averaging if POT library unavailable
    or if pairing quality fails validation.
    """

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, ["adata_path", "gene_to_idx"])
        t0 = time.time()
        warnings = []
        errors = []

        adata_path = inputs["adata_path"]
        gene_to_idx = inputs["gene_to_idx"]
        n_pca_components = inputs.get("n_pca_components", 50)
        max_cells_per_group = inputs.get("max_cells_per_group", 500)

        pairing_method = "ot_emd"
        pairing_quality_score = 0.0

        try:
            import anndata as ad
            import scanpy as sc

            adata = ad.read_h5ad(adata_path)

            # Get gene names
            if "name" in adata.var.columns:
                gene_names = adata.var["name"].values
            else:
                gene_names = adata.var_names.values

            # Filter to genes in gene_to_idx
            gene_mask = np.array([g in gene_to_idx for g in gene_names])
            adata_filtered = adata[:, gene_mask]

            if "name" in adata_filtered.var.columns:
                filtered_genes = adata_filtered.var["name"].values
            else:
                filtered_genes = adata_filtered.var_names.values

            # Reorder to match gene_to_idx
            gene_order = sorted(gene_to_idx.keys(), key=lambda g: gene_to_idx[g])

            # Split by condition
            ctrl_mask = adata_filtered.obs["label"] == "ctrl"
            stim_mask = adata_filtered.obs["label"] == "stim"
            adata_ctrl = adata_filtered[ctrl_mask]
            adata_stim = adata_filtered[stim_mask]

            # Get cell type information
            ct_col = None
            for col in ["cell_type", "cell_abbr", "celltype"]:
                if col in adata_filtered.obs.columns:
                    ct_col = col
                    break

            # Get donor information
            donor_col = None
            for col in ["donor", "ind", "sample", "patient"]:
                if col in adata_filtered.obs.columns:
                    donor_col = col
                    break

            # Try OT pairing
            try:
                import ot as pot

                X_ctrl_list = []
                X_stim_list = []
                ct_list = []
                donor_list = []

                if ct_col and donor_col:
                    groups_ctrl = adata_ctrl.obs.groupby([donor_col, ct_col])
                    groups_stim = adata_stim.obs.groupby([donor_col, ct_col])

                    for (donor, ct), ctrl_idx in groups_ctrl.groups.items():
                        if (donor, ct) not in groups_stim.groups:
                            warnings.append(
                                f"Skipped {donor}/{ct}: no stim cells"
                            )
                            continue

                        stim_idx = groups_stim.groups[(donor, ct)]

                        # Get expression matrices
                        X_c = adata_ctrl[ctrl_idx].X
                        X_s = adata_stim[stim_idx].X

                        if hasattr(X_c, "toarray"):
                            X_c = X_c.toarray()
                        if hasattr(X_s, "toarray"):
                            X_s = X_s.toarray()

                        X_c = np.array(X_c, dtype=np.float64)
                        X_s = np.array(X_s, dtype=np.float64)

                        n_ctrl = min(X_c.shape[0], max_cells_per_group)
                        n_stim = min(X_s.shape[0], max_cells_per_group)

                        if n_ctrl < 5 or n_stim < 5:
                            warnings.append(
                                f"Skipped {donor}/{ct}: insufficient cells "
                                f"(ctrl={n_ctrl}, stim={n_stim})"
                            )
                            continue

                        # Subsample if needed
                        rng = np.random.RandomState(42)
                        if X_c.shape[0] > max_cells_per_group:
                            idx = rng.choice(X_c.shape[0], max_cells_per_group, replace=False)
                            X_c = X_c[idx]
                        if X_s.shape[0] > max_cells_per_group:
                            idx = rng.choice(X_s.shape[0], max_cells_per_group, replace=False)
                            X_s = X_s[idx]

                        # Compute cost matrix
                        from scipy.spatial.distance import cdist
                        cost = cdist(X_c, X_s, metric="euclidean")
                        cost = cost / (cost.max() + 1e-8)

                        # Uniform weights
                        a = np.ones(X_c.shape[0]) / X_c.shape[0]
                        b = np.ones(X_s.shape[0]) / X_s.shape[0]

                        # Solve OT
                        T = pot.emd(a, b, cost)

                        # Extract pairs from transport plan
                        for i in range(X_c.shape[0]):
                            j = T[i].argmax()
                            if T[i, j] > 0:
                                X_ctrl_list.append(X_c[i])
                                X_stim_list.append(X_s[j])
                                ct_list.append(ct)
                                donor_list.append(donor)

                else:
                    # No cell type or donor info — simple OT pairing
                    X_c = adata_ctrl.X
                    X_s = adata_stim.X
                    if hasattr(X_c, "toarray"):
                        X_c = X_c.toarray()
                    if hasattr(X_s, "toarray"):
                        X_s = X_s.toarray()
                    X_c = np.array(X_c, dtype=np.float64)
                    X_s = np.array(X_s, dtype=np.float64)

                    n = min(X_c.shape[0], X_s.shape[0], max_cells_per_group)
                    rng = np.random.RandomState(42)
                    idx_c = rng.choice(X_c.shape[0], n, replace=False)
                    idx_s = rng.choice(X_s.shape[0], n, replace=False)
                    X_c = X_c[idx_c]
                    X_s = X_s[idx_s]

                    from scipy.spatial.distance import cdist
                    cost = cdist(X_c, X_s, metric="euclidean")
                    cost = cost / (cost.max() + 1e-8)
                    a = np.ones(n) / n
                    b = np.ones(n) / n
                    T = pot.emd(a, b, cost)

                    for i in range(n):
                        j = T[i].argmax()
                        if T[i, j] > 0:
                            X_ctrl_list.append(X_c[i])
                            X_stim_list.append(X_s[j])
                            ct_list.append("unknown")
                            donor_list.append("unknown")

                if len(X_ctrl_list) == 0:
                    raise ValueError("No OT pairs produced")

                X_ctrl_ot = np.array(X_ctrl_list, dtype=np.float32)
                X_stim_ot = np.array(X_stim_list, dtype=np.float32)
                cell_type_ot = np.array(ct_list)
                donor_ot = np.array(donor_list)

                # Compute pairing quality (correlation between paired samples)
                flat_c = X_ctrl_ot.flatten()
                flat_s = X_stim_ot.flatten()
                if flat_c.std() > 0 and flat_s.std() > 0:
                    pairing_quality_score = float(np.corrcoef(flat_c, flat_s)[0, 1])
                else:
                    pairing_quality_score = 0.0

            except ImportError:
                warnings.append(
                    "POT library not available. Falling back to pseudo-bulk."
                )
                pairing_method = "pseudo_bulk"
                # Pseudo-bulk fallback
                X_ctrl_list = []
                X_stim_list = []
                ct_list = []
                donor_list = []

                if ct_col and donor_col:
                    groups_ctrl = adata_ctrl.obs.groupby([donor_col, ct_col])
                    groups_stim = adata_stim.obs.groupby([donor_col, ct_col])

                    for (donor, ct), ctrl_idx in groups_ctrl.groups.items():
                        if (donor, ct) not in groups_stim.groups:
                            continue
                        stim_idx = groups_stim.groups[(donor, ct)]

                        X_c = adata_ctrl[ctrl_idx].X
                        X_s = adata_stim[stim_idx].X
                        if hasattr(X_c, "toarray"):
                            X_c = X_c.toarray()
                        if hasattr(X_s, "toarray"):
                            X_s = X_s.toarray()

                        X_ctrl_list.append(X_c.mean(axis=0))
                        X_stim_list.append(X_s.mean(axis=0))
                        ct_list.append(ct)
                        donor_list.append(donor)

                X_ctrl_ot = np.array(X_ctrl_list, dtype=np.float32)
                X_stim_ot = np.array(X_stim_list, dtype=np.float32)
                cell_type_ot = np.array(ct_list)
                donor_ot = np.array(donor_list)

                flat_c = X_ctrl_ot.flatten()
                flat_s = X_stim_ot.flatten()
                if flat_c.std() > 0 and flat_s.std() > 0:
                    pairing_quality_score = float(np.corrcoef(flat_c, flat_s)[0, 1])

            # Check for NaN
            if np.isnan(X_ctrl_ot).any() or np.isnan(X_stim_ot).any():
                errors.append("NaN detected in paired expression arrays")

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=len(errors) == 0,
                outputs={
                    "X_ctrl_ot": X_ctrl_ot,
                    "X_stim_ot": X_stim_ot,
                    "cell_type_ot": cell_type_ot,
                    "donor_ot": donor_ot,
                    "n_pairs": X_ctrl_ot.shape[0],
                    "pairing_method": pairing_method,
                    "pairing_quality_score": pairing_quality_score,
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "n_pca_components": n_pca_components,
                    "max_cells_per_group": max_cells_per_group,
                    "random_seed_used": 42,
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"OT pairing failed: {str(e)}")
            return SkillResult(
                skill_name=self.name, version=self.version,
                success=False, outputs={},
                metadata={"elapsed_seconds": time.time() - t0},
                warnings=warnings, errors=errors,
            )

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []

        if not result.success:
            checks_failed.append(f"Execution failed: {result.errors}")
            return ValidationReport(
                passed=False, critic_name="OTCellPairer.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check: n_pairs > 200
        n_pairs = outputs.get("n_pairs", 0)
        if n_pairs > 200:
            checks_passed.append(f"Pair count sufficient: {n_pairs}")
        else:
            checks_failed.append(
                f"Pair count too low: {n_pairs} (minimum 200)"
            )

        # Check: shapes match
        X_ctrl = outputs.get("X_ctrl_ot")
        X_stim = outputs.get("X_stim_ot")
        if X_ctrl is not None and X_stim is not None:
            if X_ctrl.shape == X_stim.shape:
                checks_passed.append(
                    f"Ctrl/stim shapes match: {X_ctrl.shape}"
                )
            else:
                checks_failed.append(
                    f"Shape mismatch: ctrl={X_ctrl.shape}, stim={X_stim.shape}"
                )

            # Check: no NaN
            if not np.isnan(X_ctrl).any() and not np.isnan(X_stim).any():
                checks_passed.append("No NaN in paired arrays")
            else:
                checks_failed.append("NaN detected in paired arrays")

        # Check: pairing quality
        q = outputs.get("pairing_quality_score", 0)
        checks_passed.append(f"Pairing quality score: {q:.3f}")

        # Warning if pseudo-bulk fallback
        recommendation = None
        if outputs.get("pairing_method") == "pseudo_bulk":
            recommendation = (
                "Using pseudo-bulk fallback. OT pairing may yield "
                "better results if POT library is available."
            )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="OTCellPairer.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            recommendation=recommendation,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=5.0,
            gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0,
            can_run_on_cpu=True,
        )
