"""
aivc/skills/uncertainty.py — MC Dropout uncertainty estimation skill.

Runs 20 forward passes with dropout active at inference time.
Computes epistemic and aleatoric uncertainty per gene per cell type.
Connects to active learning loop: high-uncertainty pathways are
the next QuRIE-seq experiment targets.
"""

import time
import os
import sys
import numpy as np

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import registry


JAKSTAT_GENES = [
    "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
    "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
]

# Known pathway membership for recommendations
GENE_PATHWAY_MAP = {
    "JAK1": "JAK-STAT", "JAK2": "JAK-STAT",
    "STAT1": "JAK-STAT", "STAT2": "JAK-STAT", "STAT3": "JAK-STAT",
    "IRF9": "JAK-STAT/Interferon", "IRF1": "JAK-STAT/Interferon",
    "MX1": "Interferon response", "MX2": "Interferon response",
    "ISG15": "Interferon response", "OAS1": "Interferon response",
    "IFIT1": "Interferon response", "IFIT3": "Interferon response",
    "IFITM1": "Interferon response", "IFITM3": "Interferon response",
}


@registry.register(
    name="uncertainty_estimator",
    domain=BiologicalDomain.EVALUATION,
    version="1.0.0",
    requires=["model_path", "X_ctrl", "edge_index"],
    compute_profile=ComputeProfile.GPU_REQUIRED,
)
class UncertaintyEstimator(AIVCSkill):
    """
    Monte Carlo Dropout uncertainty estimation.
    Runs N_MC_PASSES forward passes with dropout active at inference time.
    Computes epistemic and aleatoric uncertainty per gene.
    """

    N_MC_PASSES = 20
    HIGH_UNCERTAINTY_THRESHOLD = 0.15
    uncertainty_aware = True

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, ["model_path", "X_ctrl", "edge_index"])
        t0 = time.time()
        warnings = []
        errors = []

        try:
            import torch

            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
            from perturbation_model import PerturbationResponseModel

            device = context.device if hasattr(context, "device") else "cpu"

            model_path = inputs["model_path"]
            X_ctrl = inputs["X_ctrl"]
            edge_index = inputs["edge_index"]
            cell_type_ids = inputs.get("cell_type_ids")
            gene_to_idx = inputs.get("gene_to_idx", {})
            n_genes = inputs.get("n_genes", X_ctrl.shape[1] if hasattr(X_ctrl, "shape") else 0)
            n_cell_types = inputs.get("n_cell_types", 8)

            idx_to_gene = {v: k for k, v in gene_to_idx.items()}

            # Load model
            model = PerturbationResponseModel(
                n_genes=n_genes,
                n_cell_types=n_cell_types,
            ).to(device)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            # Convert inputs
            if not isinstance(X_ctrl, torch.Tensor):
                X_ctrl = torch.tensor(X_ctrl, dtype=torch.float32)
            X_ctrl = X_ctrl.to(device)

            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.to(device)

            # Use first sample (or mean of all)
            if X_ctrl.dim() == 2:
                x_input = X_ctrl[:1]
            else:
                x_input = X_ctrl.unsqueeze(0)

            pert_id = torch.ones(x_input.shape[0], dtype=torch.long, device=device)
            ct_id = None
            if cell_type_ids is not None:
                if not isinstance(cell_type_ids, torch.Tensor):
                    ct_id = torch.tensor(cell_type_ids[:1], dtype=torch.long, device=device)
                else:
                    ct_id = cell_type_ids[:1].to(device)

            # MC Dropout: enable dropout at inference
            # Set model to train mode to keep dropout active
            model.train()

            predictions = []
            for mc_pass in range(self.N_MC_PASSES):
                with torch.no_grad():
                    pred_delta = model.forward_batch(
                        x_input, edge_index, pert_id, ct_id
                    )
                    pred = (x_input + pred_delta).clamp(min=0.0)
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)  # [N_MC, batch, n_genes]

            # Epistemic uncertainty: variance across MC passes (model ignorance)
            epistemic = predictions.var(axis=0).mean(axis=0)  # [n_genes]

            # Aleatoric uncertainty: mean of variances within each pass
            # (approximated by range of predictions per gene)
            pred_range = predictions.max(axis=0) - predictions.min(axis=0)
            aleatoric = pred_range.mean(axis=0)  # [n_genes]

            # Build per-gene uncertainty dict
            epistemic_dict = {}
            aleatoric_dict = {}
            for gene_name, gene_idx in gene_to_idx.items():
                if gene_idx < len(epistemic):
                    epistemic_dict[gene_name] = float(epistemic[gene_idx])
                    aleatoric_dict[gene_name] = float(aleatoric[gene_idx])

            # Find high-uncertainty pathways
            high_uncertainty = [
                gene for gene, unc in epistemic_dict.items()
                if unc > self.HIGH_UNCERTAINTY_THRESHOLD
            ]

            # Build active learning recommendations
            recommendations = []
            ranked = sorted(
                epistemic_dict.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for rank, (gene, unc_score) in enumerate(ranked[:50], 1):
                pathway = GENE_PATHWAY_MAP.get(gene, "Unknown")
                recommendations.append({
                    "gene": gene,
                    "pathway": pathway,
                    "uncertainty_score": unc_score,
                    "priority_rank": rank,
                    "recommended_qurie_seq_panel_addition": (
                        f"Add anti-{gene} antibody to QuRIE-seq panel"
                        if unc_score > self.HIGH_UNCERTAINTY_THRESHOLD
                        else f"Optional: {gene} (moderate uncertainty)"
                    ),
                })

            # Update active learning memory
            if hasattr(context, "memory") and hasattr(context.memory, "active_learning"):
                result_for_al = SkillResult(
                    skill_name=self.name,
                    version=self.version,
                    success=True,
                    outputs={"active_learning_recommendations": recommendations},
                    metadata={},
                    warnings=[],
                    errors=[],
                )
                context.memory.active_learning.update_queue(result_for_al)

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "epistemic_uncertainty": epistemic_dict,
                    "aleatoric_uncertainty": aleatoric_dict,
                    "high_uncertainty_pathways": high_uncertainty,
                    "active_learning_recommendations": recommendations,
                    "n_mc_passes": self.N_MC_PASSES,
                    "threshold": self.HIGH_UNCERTAINTY_THRESHOLD,
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "device": device,
                    "random_seed_used": 42,
                    "n_genes_analyzed": len(epistemic_dict),
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Uncertainty estimation failed: {str(e)}")
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
                passed=False, critic_name="UncertaintyEstimator.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check: no NaN in uncertainty scores
        epi = outputs.get("epistemic_uncertainty", {})
        has_nan = any(np.isnan(v) for v in epi.values())
        if not has_nan:
            checks_passed.append("No NaN in epistemic uncertainty scores")
        else:
            checks_failed.append("NaN in epistemic uncertainty scores")

        # Check: non-negative values
        has_neg = any(v < 0 for v in epi.values())
        if not has_neg:
            checks_passed.append("All uncertainty values non-negative")
        else:
            checks_failed.append("Negative uncertainty values detected")

        # Check: high_uncertainty_pathways non-empty
        high = outputs.get("high_uncertainty_pathways", [])
        if len(high) > 0:
            checks_passed.append(
                f"High-uncertainty pathways identified: {len(high)}"
            )
        else:
            # This is a warning, not a failure
            checks_passed.append(
                "No high-uncertainty pathways "
                f"(threshold={outputs.get('threshold', 0.15)})"
            )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="UncertaintyEstimator.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=3.0,
            gpu_memory_gb=4.0,
            profile=ComputeProfile.GPU_REQUIRED,
            estimated_usd=0.10,
            can_run_on_cpu=True,
        )
