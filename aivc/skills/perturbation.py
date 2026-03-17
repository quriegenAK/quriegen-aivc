"""
aivc/skills/perturbation.py — Perturbation response prediction skill.

Wraps perturbation_model.py for inference.
Loads a trained model and predicts stimulated expression from control.
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


@registry.register(
    name="perturbation_predictor",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["model_path", "X_ctrl", "edge_index"],
    compute_profile=ComputeProfile.GPU_REQUIRED,
)
class PerturbationPredictor(AIVCSkill):
    """
    Loads a trained GAT model and predicts perturbation responses.
    Uses residual learning: predicted = ctrl + model_output.
    """

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
            n_genes = inputs.get("n_genes", X_ctrl.shape[1] if hasattr(X_ctrl, "shape") else 0)
            n_cell_types = inputs.get("n_cell_types", 8)

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

            pert_id = torch.ones(X_ctrl.shape[0], dtype=torch.long, device=device)

            if cell_type_ids is not None:
                if not isinstance(cell_type_ids, torch.Tensor):
                    cell_type_ids = torch.tensor(cell_type_ids, dtype=torch.long)
                cell_type_ids = cell_type_ids.to(device)

            # Predict with residual learning
            with torch.no_grad():
                predicted_delta = model.forward_batch(
                    X_ctrl, edge_index, pert_id, cell_type_ids
                )
                predicted = (X_ctrl + predicted_delta).clamp(min=0.0)

            predicted_np = predicted.cpu().numpy()

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "predicted_stim": predicted_np,
                    "predicted_delta": predicted_delta.cpu().numpy(),
                    "n_samples": predicted_np.shape[0],
                    "n_genes": predicted_np.shape[1],
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "device": device,
                    "random_seed_used": 42,
                    "residual_learning": True,
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Perturbation prediction failed: {str(e)}")
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
                passed=False, critic_name="PerturbationPredictor.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs
        pred = outputs.get("predicted_stim")
        if pred is not None:
            if not np.isnan(pred).any():
                checks_passed.append("No NaN in predictions")
            else:
                checks_failed.append("NaN detected in predictions")

            if (pred >= 0).all():
                checks_passed.append("All predictions non-negative")
            else:
                checks_failed.append("Negative predictions detected")

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="PerturbationPredictor.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=1.0,
            gpu_memory_gb=2.0,
            profile=ComputeProfile.GPU_REQUIRED,
            estimated_usd=0.03,
            can_run_on_cpu=True,
        )
