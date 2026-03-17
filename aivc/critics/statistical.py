"""
aivc/critics/statistical.py — Numerical correctness validation.

Validates numerical correctness of every SkillResult.
Runs independently and in parallel with other critics.
"""

import numpy as np

from aivc.interfaces import SkillResult, ValidationReport


class StatisticalCritic:
    """
    Validates numerical correctness of every SkillResult.
    Runs independently and in parallel with other critics.
    """

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []

        # Check 1: No NaN or Inf in any numeric output
        for key, value in result.outputs.items():
            if hasattr(value, "dtype"):  # numpy or torch
                arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
                if np.isnan(arr).any() or np.isinf(arr).any():
                    checks_failed.append(f"NaN/Inf in {key}")
                else:
                    checks_passed.append(f"No NaN/Inf in {key}")
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, float):
                        if np.isnan(subval) or np.isinf(subval):
                            checks_failed.append(
                                f"NaN/Inf in {key}.{subkey}"
                            )
            elif isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    checks_failed.append(f"NaN/Inf in {key}")

        # Check 2: Pearson r is in valid range (-1 to 1)
        if "pearson_r_mean" in result.outputs:
            r = result.outputs["pearson_r_mean"]
            if isinstance(r, (int, float)):
                if -1.0 <= r <= 1.0:
                    checks_passed.append(
                        f"Pearson r in valid range: {r:.4f}"
                    )
                else:
                    checks_failed.append(
                        f"Pearson r out of range: {r}"
                    )

        # Check 3: Loss is non-negative and finite
        if "final_loss" in result.outputs:
            loss = result.outputs["final_loss"]
            if isinstance(loss, (int, float)):
                if loss >= 0 and np.isfinite(loss):
                    checks_passed.append(f"Loss valid: {loss:.6f}")
                else:
                    checks_failed.append(f"Invalid loss: {loss}")

        # Check 4: Tensor shapes are correct
        if "edge_index" in result.outputs:
            ei = result.outputs["edge_index"]
            if hasattr(ei, "shape"):
                if ei.shape[0] == 2:
                    checks_passed.append(
                        "edge_index shape correct [2, n_edges]"
                    )
                else:
                    checks_failed.append(
                        f"edge_index wrong shape: {ei.shape}"
                    )

        # Check 5: Reproducibility seed confirmation
        if "random_seed_used" in result.metadata:
            if result.metadata["random_seed_used"] == 42:
                checks_passed.append("Correct random seed: 42")
            else:
                checks_failed.append(
                    f"Wrong seed: {result.metadata['random_seed_used']}, "
                    "expected 42"
                )

        # Check 6: Prediction arrays have valid shape
        if "predicted_stim" in result.outputs:
            pred = result.outputs["predicted_stim"]
            if hasattr(pred, "shape"):
                if len(pred.shape) == 2:
                    checks_passed.append(
                        f"Prediction shape valid: {pred.shape}"
                    )
                else:
                    checks_failed.append(
                        f"Prediction shape unexpected: {pred.shape}"
                    )

        # Check 7: Best val Pearson r in valid range
        if "best_val_pearson_r" in result.outputs:
            r = result.outputs["best_val_pearson_r"]
            if isinstance(r, (int, float)):
                if -1.0 <= r <= 1.0:
                    checks_passed.append(
                        f"Best val Pearson r in range: {r:.4f}"
                    )
                else:
                    checks_failed.append(
                        f"Best val Pearson r out of range: {r}"
                    )

        passed = len(checks_failed) == 0
        return ValidationReport(
            passed=passed,
            critic_name="StatisticalCritic",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
