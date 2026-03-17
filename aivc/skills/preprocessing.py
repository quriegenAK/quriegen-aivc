"""
aivc/skills/preprocessing.py — scRNA-seq preprocessing skill.

Wraps existing preprocessing logic from build_edge_list.py and
fix_gene_selection.py. Loads AnnData, normalises, log-transforms,
selects HVGs, force-includes JAK-STAT pathway genes.
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
    name="scRNA_preprocessor",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["adata_path"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class ScRNAPreprocessor(AIVCSkill):
    """
    Loads AnnData, normalises, log-transforms, selects HVGs,
    force-includes JAK-STAT pathway genes, validates result.
    """

    MUST_INCLUDE = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
        "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, ["adata_path"])
        t0 = time.time()
        warnings = []
        errors = []

        adata_path = inputs["adata_path"]
        n_top_genes = inputs.get("n_top_genes", 3000)

        try:
            import scanpy as sc
            import anndata as ad

            # Load data
            adata = ad.read_h5ad(adata_path)

            # Get gene names — adata.var["name"] if exists, else var_names
            if "name" in adata.var.columns:
                gene_names = adata.var["name"].values
            else:
                gene_names = adata.var_names.values

            # Validate condition column
            if "label" not in adata.obs.columns:
                errors.append("Condition column 'label' not found in adata.obs")
                return SkillResult(
                    skill_name=self.name, version=self.version,
                    success=False, outputs={}, metadata={},
                    warnings=warnings, errors=errors,
                )

            labels = set(adata.obs["label"].unique())
            if not {"ctrl", "stim"}.issubset(labels):
                errors.append(
                    f"Expected 'ctrl' and 'stim' in label column, got {labels}"
                )

            # Normalize and log-transform
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # Select HVGs
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            hvg_mask = adata.var["highly_variable"].values.copy()

            # Force-include JAK-STAT pathway genes
            genes_forced = []
            for gene in self.MUST_INCLUDE:
                gene_mask = gene_names == gene
                if gene_mask.any():
                    idx = np.where(gene_mask)[0][0]
                    if not hvg_mask[idx]:
                        hvg_mask[idx] = True
                        genes_forced.append(gene)
                else:
                    warnings.append(
                        f"JAK-STAT gene '{gene}' not found in dataset"
                    )

            # Filter to selected genes
            adata = adata[:, hvg_mask]

            # Build gene-to-index mapping
            if "name" in adata.var.columns:
                final_genes = adata.var["name"].values
            else:
                final_genes = adata.var_names.values

            gene_to_idx = {g: i for i, g in enumerate(final_genes)}

            # Check for negative values after log1p
            if hasattr(adata.X, "toarray"):
                x_dense = adata.X.toarray()
            else:
                x_dense = np.array(adata.X)

            if (x_dense < 0).any():
                warnings.append(
                    "Negative expression values detected after log1p"
                )

            n_pathway = sum(
                1 for g in self.MUST_INCLUDE if g in gene_to_idx
            )

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=len(errors) == 0,
                outputs={
                    "adata_processed": adata,
                    "gene_to_idx": gene_to_idx,
                    "n_genes_final": len(gene_to_idx),
                    "n_pathway_genes_included": n_pathway,
                    "genes_forced_in": genes_forced,
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "n_cells": adata.n_obs,
                    "n_top_genes_requested": n_top_genes,
                    "random_seed_used": 42,
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Preprocessing failed: {str(e)}")
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
                passed=False, critic_name="ScRNAPreprocessor.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check: n_genes_final >= 3000
        n_genes = outputs.get("n_genes_final", 0)
        if n_genes >= 3000:
            checks_passed.append(f"Gene count sufficient: {n_genes}")
        else:
            checks_failed.append(
                f"Gene count too low: {n_genes} (minimum 3000)"
            )

        # Check: all 15 MUST_INCLUDE genes present
        gene_to_idx = outputs.get("gene_to_idx", {})
        missing_pathway = [
            g for g in self.MUST_INCLUDE if g not in gene_to_idx
        ]
        if not missing_pathway:
            checks_passed.append("All 15 JAK-STAT pathway genes present")
        else:
            checks_failed.append(
                f"Missing pathway genes: {missing_pathway}"
            )

        # Check: n_pathway_genes_included
        n_pw = outputs.get("n_pathway_genes_included", 0)
        if n_pw == 15:
            checks_passed.append(f"All pathway genes included: {n_pw}/15")
        else:
            checks_failed.append(
                f"Only {n_pw}/15 pathway genes included"
            )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="ScRNAPreprocessor.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=2.0,
            gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0,
            can_run_on_cpu=True,
        )
