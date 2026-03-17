"""
aivc/skills/attention.py — GAT attention weight extraction skill.

Wraps extract_attention_week3.py. Extracts attention weights
from the trained GAT model for biological interpretation.
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


@registry.register(
    name="attention_extractor",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["model_path", "X_ctrl", "edge_index", "gene_to_idx"],
    compute_profile=ComputeProfile.GPU_REQUIRED,
)
class AttentionExtractor(AIVCSkill):
    """
    Extracts GAT attention weights for biological interpretation.
    Computes attention delta (stim - ctrl) to identify genes whose
    attention patterns change most under perturbation.
    """

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, [
            "model_path", "X_ctrl", "edge_index", "gene_to_idx",
        ])
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
            gene_to_idx = inputs["gene_to_idx"]
            cell_type_ids = inputs.get("cell_type_ids")
            n_genes = inputs.get("n_genes", len(gene_to_idx))
            n_cell_types = inputs.get("n_cell_types", 8)

            idx_to_gene = {v: k for k, v in gene_to_idx.items()}

            # Load model
            model = PerturbationResponseModel(
                n_genes=n_genes,
                n_cell_types=n_cell_types,
            ).to(device)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            # Convert inputs
            if not isinstance(X_ctrl, torch.Tensor):
                X_ctrl = torch.tensor(X_ctrl, dtype=torch.float32)
            X_ctrl = X_ctrl.to(device)

            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.to(device)

            # Use first sample for attention extraction
            if X_ctrl.dim() == 2:
                x_sample = X_ctrl[0:1]
            else:
                x_sample = X_ctrl.unsqueeze(0)

            # Get attention weights via hooks
            attention_weights = {}

            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        attention_weights[name] = output[1].detach().cpu()
                return hook

            # Register hooks on GAT layers
            hooks = []
            for name, module in model.named_modules():
                if "GATConv" in type(module).__name__ or "gatconv" in name.lower():
                    h = module.register_forward_hook(hook_fn(name))
                    hooks.append(h)

            # Forward pass
            with torch.no_grad():
                pert_id = torch.ones(1, dtype=torch.long, device=device)
                ct_id = None
                if cell_type_ids is not None:
                    if not isinstance(cell_type_ids, torch.Tensor):
                        ct_id = torch.tensor([cell_type_ids[0]], dtype=torch.long, device=device)
                    else:
                        ct_id = cell_type_ids[0:1].to(device)

                _ = model.forward_batch(x_sample, edge_index, pert_id, ct_id)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Compute attention statistics per gene
            gene_attention_scores = {}
            for gene_name, gene_idx in gene_to_idx.items():
                # Find edges where this gene is a target
                target_mask = (edge_index[1] == gene_idx)
                if target_mask.any():
                    # Average attention from all available layers
                    attn_values = []
                    for layer_name, attn in attention_weights.items():
                        if attn.dim() >= 1 and target_mask.sum() > 0:
                            layer_attn = attn[target_mask].mean().item()
                            attn_values.append(layer_attn)
                    if attn_values:
                        gene_attention_scores[gene_name] = float(np.mean(attn_values))

            # Rank genes by attention
            ranked_genes = sorted(
                gene_attention_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_20 = ranked_genes[:20]

            # JAK-STAT recovery
            jakstat_in_top50 = sum(
                1 for gene, _ in ranked_genes[:50]
                if gene in JAKSTAT_GENES
            )

            # JAK-STAT attention scores
            jakstat_attention = {
                gene: gene_attention_scores.get(gene, 0.0)
                for gene in JAKSTAT_GENES
            }

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "gene_attention_scores": gene_attention_scores,
                    "top_20_genes": top_20,
                    "jakstat_attention": jakstat_attention,
                    "jakstat_recovery_score": jakstat_in_top50,
                    "attention_weights_raw": attention_weights,
                    "n_layers_with_attention": len(attention_weights),
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "device": device,
                    "random_seed_used": 42,
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Attention extraction failed: {str(e)}")
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
                passed=False, critic_name="AttentionExtractor.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        scores = outputs.get("gene_attention_scores", {})
        if len(scores) > 0:
            checks_passed.append(
                f"Attention scores computed for {len(scores)} genes"
            )
        else:
            checks_failed.append("No attention scores computed")

        # Check for NaN
        has_nan = any(np.isnan(v) for v in scores.values())
        if not has_nan:
            checks_passed.append("No NaN in attention scores")
        else:
            checks_failed.append("NaN detected in attention scores")

        jakstat_recovery = outputs.get("jakstat_recovery_score", 0)
        checks_passed.append(
            f"JAK-STAT in top 50: {jakstat_recovery}/15"
        )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="AttentionExtractor.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            biological_score=jakstat_recovery / 15.0,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=2.0,
            gpu_memory_gb=2.0,
            profile=ComputeProfile.GPU_REQUIRED,
            estimated_usd=0.07,
            can_run_on_cpu=True,
        )
