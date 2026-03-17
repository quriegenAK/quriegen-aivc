"""
aivc/skills/evaluation.py — Benchmark evaluation skill.

Full benchmark evaluation against CPA and scGEN published results.
Wraps evaluate_week3.py.
CRITICAL: Pearson r must be computed on RAW values, not log-transformed.
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
    name="benchmark_evaluator",
    domain=BiologicalDomain.EVALUATION,
    version="1.0.0",
    requires=["model_path", "X_ctrl_test", "X_stim_test",
              "edge_index", "cell_type_test"],
    compute_profile=ComputeProfile.GPU_REQUIRED,
)
class BenchmarkEvaluator(AIVCSkill):
    """
    Full benchmark evaluation against CPA and scGEN published results.
    Wraps evaluate_week3.py.
    CRITICAL: Pearson r computed on RAW values, not log-transformed.
    """

    CPA_BENCHMARK = 0.856
    SCGEN_BENCHMARK = 0.820
    DEMO_THRESHOLD = 0.800
    ABORT_THRESHOLD = 0.700

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, [
            "model_path", "X_ctrl_test", "X_stim_test",
            "edge_index", "cell_type_test",
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
            X_ctrl_test = inputs["X_ctrl_test"]
            X_stim_test = inputs["X_stim_test"]
            edge_index = inputs["edge_index"]
            cell_type_test = inputs["cell_type_test"]
            gene_to_idx = inputs.get("gene_to_idx", {})
            n_genes = inputs.get("n_genes", X_ctrl_test.shape[1])
            n_cell_types = inputs.get("n_cell_types", 8)
            ct_to_idx = inputs.get("ct_to_idx", {})

            # Load model
            model = PerturbationResponseModel(
                n_genes=n_genes,
                n_cell_types=n_cell_types,
            ).to(device)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            # Convert inputs
            if not isinstance(X_ctrl_test, torch.Tensor):
                X_ctrl_test = torch.tensor(X_ctrl_test, dtype=torch.float32)
            X_ctrl_test = X_ctrl_test.to(device)

            if not isinstance(X_stim_test, torch.Tensor):
                X_stim_test = torch.tensor(X_stim_test, dtype=torch.float32)
            X_stim_test = X_stim_test.to(device)

            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.to(device)

            # Cell type IDs
            if ct_to_idx:
                ct_ids = torch.tensor(
                    [ct_to_idx.get(ct, 0) for ct in cell_type_test],
                    dtype=torch.long,
                ).to(device)
            else:
                unique_ct = sorted(set(cell_type_test))
                auto_ct_to_idx = {ct: i for i, ct in enumerate(unique_ct)}
                ct_ids = torch.tensor(
                    [auto_ct_to_idx[ct] for ct in cell_type_test],
                    dtype=torch.long,
                ).to(device)

            pert_id = torch.ones(
                X_ctrl_test.shape[0], dtype=torch.long, device=device
            )

            # Predict with residual learning
            with torch.no_grad():
                pred_delta = model.forward_batch(
                    X_ctrl_test, edge_index, pert_id, ct_ids
                )
                predicted = (X_ctrl_test + pred_delta).clamp(min=0.0)

            pred_np = predicted.cpu().numpy()
            actual_np = X_stim_test.cpu().numpy()
            ctrl_np = X_ctrl_test.cpu().numpy()

            # Overall Pearson r (on RAW values, NOT log-transformed)
            pearson_rs = []
            for i in range(pred_np.shape[0]):
                p = pred_np[i].flatten()
                a = actual_np[i].flatten()
                if p.std() > 0 and a.std() > 0:
                    r = np.corrcoef(p, a)[0, 1]
                    pearson_rs.append(r)

            pearson_r_mean = float(np.mean(pearson_rs)) if pearson_rs else 0.0
            pearson_r_std = float(np.std(pearson_rs)) if pearson_rs else 0.0

            # R2 score
            ss_res = np.sum((actual_np - pred_np) ** 2)
            ss_tot = np.sum((actual_np - actual_np.mean()) ** 2)
            r2 = 1 - (ss_res / max(ss_tot, 1e-8))

            # MSE
            mse = float(np.mean((pred_np - actual_np) ** 2))

            # Cell-type stratified Pearson r
            cell_type_r = {}
            unique_cts = sorted(set(cell_type_test))
            for ct in unique_cts:
                ct_mask = np.array([c == ct for c in cell_type_test])
                if ct_mask.sum() > 0:
                    ct_pred = pred_np[ct_mask].flatten()
                    ct_actual = actual_np[ct_mask].flatten()
                    if ct_pred.std() > 0 and ct_actual.std() > 0:
                        cell_type_r[ct] = float(
                            np.corrcoef(ct_pred, ct_actual)[0, 1]
                        )

            # Cell-type spread
            if cell_type_r:
                ct_spread = max(cell_type_r.values()) - min(cell_type_r.values())
            else:
                ct_spread = 0.0

            # Top-50 DE gene recall
            # Find top-50 actual DE genes by mean fold change
            mean_actual = actual_np.mean(axis=0)
            mean_ctrl = ctrl_np.mean(axis=0)
            actual_fc = np.abs(mean_actual - mean_ctrl)
            top50_actual_idx = np.argsort(actual_fc)[-50:]

            mean_pred = pred_np.mean(axis=0)
            pred_fc = np.abs(mean_pred - mean_ctrl)
            top50_pred_idx = np.argsort(pred_fc)[-50:]

            top50_recall = len(
                set(top50_actual_idx) & set(top50_pred_idx)
            ) / 50.0

            # JAK-STAT LFC errors
            jakstat_lfc_errors = {}
            ifit1_predicted_fc = 1.0
            for gene in JAKSTAT_GENES:
                if gene in gene_to_idx:
                    idx = gene_to_idx[gene]
                    if idx < pred_np.shape[1]:
                        pred_val = float(mean_pred[idx])
                        actual_val = float(mean_actual[idx])
                        ctrl_val = float(mean_ctrl[idx])

                        pred_fc_gene = (
                            pred_val / max(ctrl_val, 1e-8)
                        )
                        actual_fc_gene = (
                            actual_val / max(ctrl_val, 1e-8)
                        )

                        lfc_error = abs(
                            np.log2(max(pred_fc_gene, 1e-8))
                            - np.log2(max(actual_fc_gene, 1e-8))
                        )
                        jakstat_lfc_errors[gene] = {
                            "predicted_fc": pred_fc_gene,
                            "actual_fc": actual_fc_gene,
                            "lfc_error": float(lfc_error),
                        }

                        if gene == "IFIT1":
                            ifit1_predicted_fc = pred_fc_gene

            # JAK-STAT recovery count (within 3x fold change)
            jakstat_recovery = 0
            for gene, info in jakstat_lfc_errors.items():
                if info["actual_fc"] > 0:
                    ratio = info["predicted_fc"] / max(info["actual_fc"], 1e-8)
                    if 1/3.0 <= ratio <= 3.0:
                        jakstat_recovery += 1

            # Demo readiness
            demo_ready = pearson_r_mean >= self.DEMO_THRESHOLD

            # Benchmark comparison
            benchmark_table = {
                "AIVC_GeneLink": pearson_r_mean,
                "CPA_published": self.CPA_BENCHMARK,
                "scGEN_published": self.SCGEN_BENCHMARK,
                "beats_CPA": pearson_r_mean > self.CPA_BENCHMARK,
                "beats_scGEN": pearson_r_mean > self.SCGEN_BENCHMARK,
            }

            if pearson_r_mean < self.ABORT_THRESHOLD:
                warnings.append(
                    f"ABORT: Pearson r = {pearson_r_mean:.4f} < {self.ABORT_THRESHOLD}. "
                    "Demo should be blocked."
                )

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "pearson_r_mean": pearson_r_mean,
                    "pearson_r_std": pearson_r_std,
                    "r2_score": float(r2),
                    "mse": mse,
                    "cell_type_pearson_r": cell_type_r,
                    "cell_type_spread": ct_spread,
                    "top50_recall": top50_recall,
                    "jakstat_lfc_errors": jakstat_lfc_errors,
                    "jakstat_recovery_score": jakstat_recovery,
                    "ifit1_predicted_fc": ifit1_predicted_fc,
                    "demo_ready": demo_ready,
                    "benchmark_table": benchmark_table,
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "device": device,
                    "n_test_samples": pred_np.shape[0],
                    "pearson_r_on_raw": True,
                    "random_seed_used": 42,
                    "split_method": "donor_based",
                    "benchmark_dataset": "kang2018_GSE96583",
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Benchmark evaluation failed: {str(e)}")
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
                passed=False, critic_name="BenchmarkEvaluator.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # HARD GATE: pearson_r_mean < ABORT_THRESHOLD -> FAIL
        r = outputs.get("pearson_r_mean", 0)
        if r >= self.ABORT_THRESHOLD:
            checks_passed.append(
                f"Pearson r = {r:.4f} >= {self.ABORT_THRESHOLD} abort threshold"
            )
        else:
            checks_failed.append(
                f"ABORT: Pearson r = {r:.4f} < {self.ABORT_THRESHOLD}. "
                "Demo blocked."
            )

        # Check: demo readiness
        if r >= self.DEMO_THRESHOLD:
            checks_passed.append(
                f"Demo ready: r = {r:.4f} >= {self.DEMO_THRESHOLD}"
            )
        else:
            checks_passed.append(
                f"Demo not ready: r = {r:.4f} < {self.DEMO_THRESHOLD}"
            )

        # Check: cell-type spread
        spread = outputs.get("cell_type_spread", 0)
        if spread > 0.05:
            checks_passed.append(
                f"Cell-type spread: {spread:.3f} > 0.05 (model is cell-type aware)"
            )
        else:
            checks_failed.append(
                f"Cell-type spread too narrow: {spread:.3f}. "
                "Model may not be cell-type aware."
            )

        # Check: CD14 monocyte r
        ct_r = outputs.get("cell_type_pearson_r", {})
        mono_r = ct_r.get("CD14+ Monocytes", 0)
        if mono_r > 0.70:
            checks_passed.append(
                f"CD14+ Monocyte r = {mono_r:.4f} > 0.70"
            )
        else:
            checks_failed.append(
                f"CD14+ Monocyte r = {mono_r:.4f} < 0.70. "
                "Most critical cell type underperforming."
            )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="BenchmarkEvaluator.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=2.0,
            gpu_memory_gb=2.0,
            profile=ComputeProfile.GPU_REQUIRED,
            estimated_usd=0.07,
            can_run_on_cpu=True,
        )
