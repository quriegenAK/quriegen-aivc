"""
aivc/skills/reporting.py — Two-audience output renderer skill.

Produces two distinct outputs from the same SkillResult data.
Skills never format their own output. This renderer is the only
component that produces user-facing content.
"""

import time
import os
import json
import numpy as np

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import registry


@registry.register(
    name="two_audience_renderer",
    domain=BiologicalDomain.REPORTING,
    version="1.0.0",
    requires=["evaluation_result", "attention_result",
              "uncertainty_result", "plausibility_result"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class TwoAudienceRenderer(AIVCSkill):
    """
    Produces two distinct outputs from the same SkillResult data.

    R&D Director output:
        - Response score (0 to 1) with confidence interval
        - Go/No-Go recommendation
        - Cell-type specific scores
        - Comparison to historical runs

    Discovery Scientist output:
        - JAK-STAT attention heatmap
        - Top 20 activated genes ranked by attention delta
        - Fold change accuracy table
        - Pearson r per cell type with benchmark reference
        - Active learning recommendation
    """

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, [
            "evaluation_result", "attention_result",
            "uncertainty_result", "plausibility_result",
        ])
        t0 = time.time()
        warnings = []
        errors = []

        eval_result = inputs["evaluation_result"]
        attn_result = inputs["attention_result"]
        unc_result = inputs["uncertainty_result"]
        plaus_result = inputs["plausibility_result"]

        # Extract outputs from results
        eval_out = eval_result.outputs if hasattr(eval_result, "outputs") else eval_result
        attn_out = attn_result.outputs if hasattr(attn_result, "outputs") else attn_result
        unc_out = unc_result.outputs if hasattr(unc_result, "outputs") else unc_result
        plaus_out = plaus_result.outputs if hasattr(plaus_result, "outputs") else plaus_result

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs", exist_ok=True)

        # ─── R&D Director Report ───
        pearson_r = eval_out.get("pearson_r_mean", 0.0)
        pearson_std = eval_out.get("pearson_r_std", 0.0)
        benchmark = eval_out.get("benchmark_table", {})
        demo_ready = eval_out.get("demo_ready", False)

        # Go/No-Go decision
        if pearson_r >= 0.85:
            recommendation = "GO — Strong performance, ready for demo and R&D presentation."
        elif pearson_r >= 0.80:
            recommendation = "CONDITIONAL GO — Meets minimum threshold. Monitor cell-type spread."
        elif pearson_r >= 0.70:
            recommendation = "NO-GO for demo — Internal iteration needed."
        else:
            recommendation = "ABORT — Model performance critically low."

        rd_report = {
            "report_type": "R&D Director",
            "timestamp": timestamp,
            "summary": {
                "response_score": pearson_r,
                "confidence_interval": f"{pearson_r:.3f} ± {pearson_std:.3f}",
                "recommendation": recommendation,
                "demo_ready": demo_ready,
            },
            "cell_type_scores": eval_out.get("cell_type_pearson_r", {}),
            "benchmark_comparison": benchmark,
            "biological_plausibility": {
                "mean_score": plaus_out.get("mean_plausibility_score", 0),
                "quarantined_fraction": plaus_out.get("quarantine_fraction", 0),
                "jakstat_recovery": plaus_out.get("jakstat_recovery_in_predictions", 0),
            },
            "uncertainty_summary": {
                "n_high_uncertainty_genes": len(
                    unc_out.get("high_uncertainty_pathways", [])
                ),
                "top_uncertain_pathways": unc_out.get(
                    "high_uncertainty_pathways", []
                )[:5],
            },
        }

        # Check experimental memory for historical comparison
        if hasattr(context, "memory") and hasattr(context.memory, "experimental"):
            best = context.memory.experimental.get_best_run()
            if best:
                rd_report["historical_comparison"] = {
                    "best_historical_r": best.get("metrics", {}).get(
                        "pearson_r_mean", "N/A"
                    ),
                    "regression_check": (
                        context.memory.experimental.detect_regression(pearson_r)
                    ),
                }

        # ─── Discovery Scientist Report ───
        discovery_report = {
            "report_type": "Discovery Scientist",
            "timestamp": timestamp,
            "attention_analysis": {
                "top_20_genes": attn_out.get("top_20_genes", []),
                "jakstat_attention": attn_out.get("jakstat_attention", {}),
                "jakstat_recovery_in_top50": attn_out.get(
                    "jakstat_recovery_score", 0
                ),
            },
            "fold_change_accuracy": {},
            "cell_type_pearson_r": eval_out.get("cell_type_pearson_r", {}),
            "benchmark_reference": {
                "CPA": 0.856,
                "scGEN": 0.820,
                "AIVC": pearson_r,
            },
            "active_learning": {
                "recommendations": unc_out.get(
                    "active_learning_recommendations", []
                )[:10],
                "high_uncertainty_genes": unc_out.get(
                    "high_uncertainty_pathways", []
                ),
            },
        }

        # Fold change accuracy for key genes
        jakstat_lfc = eval_out.get("jakstat_lfc_errors", {})
        for gene in ["IFIT1", "MX1", "ISG15", "OAS1"]:
            if gene in jakstat_lfc:
                info = jakstat_lfc[gene]
                discovery_report["fold_change_accuracy"][gene] = {
                    "predicted_fc": info.get("predicted_fc", "N/A"),
                    "actual_fc": info.get("actual_fc", "N/A"),
                    "lfc_error": info.get("lfc_error", "N/A"),
                }

        # Save reports
        rd_path = f"outputs/rd_report_{timestamp}.json"
        disc_path = f"outputs/discovery_report_{timestamp}.json"

        with open(rd_path, "w") as f:
            json.dump(rd_report, f, indent=2, default=str)
        with open(disc_path, "w") as f:
            json.dump(discovery_report, f, indent=2, default=str)

        output_paths = [rd_path, disc_path]

        # Generate visualizations if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # R&D Dashboard
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Panel 1: Benchmark comparison
            models = list(benchmark.keys())
            values = []
            for k in models:
                v = benchmark[k]
                if isinstance(v, (int, float)):
                    values.append(v)
                else:
                    models.remove(k)
            if values:
                colors = ["#2ecc71" if v >= 0.85 else "#e67e22" if v >= 0.80 else "#e74c3c" for v in values]
                axes[0].barh(models[:len(values)], values, color=colors)
                axes[0].set_xlim(0.6, 1.0)
                axes[0].set_title("Benchmark Comparison")

            # Panel 2: Cell-type Pearson r
            ct_r = eval_out.get("cell_type_pearson_r", {})
            if ct_r:
                cts = list(ct_r.keys())
                rs = list(ct_r.values())
                axes[1].barh(cts, rs, color="#3498db")
                axes[1].axvline(x=0.80, color="red", linestyle="--", label="Target")
                axes[1].set_xlim(0.5, 1.0)
                axes[1].set_title("Cell-Type Pearson r")
                axes[1].legend()

            # Panel 3: Go/No-Go
            axes[2].text(0.5, 0.5, recommendation.split(" — ")[0],
                        ha="center", va="center", fontsize=24,
                        fontweight="bold",
                        color="#2ecc71" if "GO" in recommendation and "NO" not in recommendation else "#e74c3c")
            axes[2].set_title("Decision")
            axes[2].axis("off")

            plt.tight_layout()
            rd_png = f"outputs/rd_dashboard_{timestamp}.png"
            plt.savefig(rd_png, dpi=150)
            plt.close()
            output_paths.append(rd_png)

            # Discovery Heatmap (JAK-STAT attention)
            jakstat_attn = attn_out.get("jakstat_attention", {})
            if jakstat_attn:
                fig, ax = plt.subplots(figsize=(10, 6))
                genes = list(jakstat_attn.keys())
                scores = [jakstat_attn[g] for g in genes]
                colors = ["#e74c3c" if s > 0.1 else "#3498db" for s in scores]
                ax.barh(genes, scores, color=colors)
                ax.set_title("JAK-STAT Gene Attention Scores")
                ax.set_xlabel("Mean Attention Weight")
                plt.tight_layout()
                disc_png = f"outputs/discovery_heatmap_{timestamp}.png"
                plt.savefig(disc_png, dpi=150)
                plt.close()
                output_paths.append(disc_png)

        except ImportError:
            warnings.append(
                "matplotlib not available. Skipping visualization generation."
            )

        elapsed = time.time() - t0
        return SkillResult(
            skill_name=self.name,
            version=self.version,
            success=True,
            outputs={
                "rd_report": rd_report,
                "discovery_report": discovery_report,
                "output_paths": output_paths,
                "recommendation": recommendation,
                "demo_ready": demo_ready,
                "pearson_r_mean": pearson_r,
            },
            metadata={
                "elapsed_seconds": elapsed,
                "random_seed_used": 42,
            },
            warnings=warnings,
            errors=errors,
        )

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []

        if not result.success:
            checks_failed.append(f"Execution failed: {result.errors}")
            return ValidationReport(
                passed=False, critic_name="TwoAudienceRenderer.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check reports exist
        paths = outputs.get("output_paths", [])
        for path in paths:
            if os.path.exists(path):
                checks_passed.append(f"Output exists: {path}")
            else:
                checks_failed.append(f"Output missing: {path}")

        # Check recommendation is present
        rec = outputs.get("recommendation", "")
        if rec:
            checks_passed.append(f"Recommendation: {rec.split(' — ')[0]}")
        else:
            checks_failed.append("No recommendation generated")

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="TwoAudienceRenderer.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=0.5,
            gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0,
            can_run_on_cpu=True,
        )
