"""
aivc/critics/biological.py — Biological plausibility validation.

Validates biological plausibility.
The hardest critic. Most dangerous failure mode is a model that is
statistically correct but biologically impossible.
"""

from aivc.interfaces import SkillResult, ValidationReport


class BiologicalCritic:
    """
    Validates biological plausibility.
    The hardest critic. Most dangerous failure mode is a model that is
    statistically correct but biologically impossible.
    """

    JAKSTAT_MINIMUM_RECOVERY = 5  # minimum genes in top 50 to pass

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []
        quarantined = []
        warnings = []
        bio_score = 1.0

        # Check 1: JAK-STAT recovery (primary biological validation)
        if "jakstat_recovery_score" in result.outputs:
            n_recovered = result.outputs["jakstat_recovery_score"]
            if n_recovered >= 8:
                checks_passed.append(
                    f"JAK-STAT recovery: {n_recovered}/15 (PASS)"
                )
            elif n_recovered >= self.JAKSTAT_MINIMUM_RECOVERY:
                checks_passed.append(
                    f"JAK-STAT recovery: {n_recovered}/15 "
                    "(PARTIAL - demo possible)"
                )
            else:
                checks_failed.append(
                    f"JAK-STAT recovery: {n_recovered}/15 below minimum "
                    f"{self.JAKSTAT_MINIMUM_RECOVERY}. "
                    "Model not learning IFN-b biology."
                )
            bio_score = n_recovered / 15.0

        # Check 2: Quarantine low-plausibility predictions
        if "scored_interactions" in result.outputs:
            for interaction in result.outputs["scored_interactions"]:
                if isinstance(interaction, dict):
                    p_score = interaction.get("plausibility_score", 1.0)
                    if p_score < 0.3:
                        gene_pair = interaction.get(
                            "gene_pair", "unknown"
                        )
                        quarantined.append(gene_pair)
                        checks_failed.append(
                            f"Quarantined: {gene_pair} "
                            f"(plausibility {p_score:.2f})"
                        )

        # Check 3: IFIT1 fold change direction
        if "ifit1_predicted_fc" in result.outputs:
            ifit1_pred = result.outputs["ifit1_predicted_fc"]
            if isinstance(ifit1_pred, (int, float)):
                if ifit1_pred < 1.0:
                    checks_failed.append(
                        f"IFIT1 predicted fold change {ifit1_pred:.2f} "
                        "is SUPPRESSED under IFN-b. Known biology: "
                        "IFIT1 is strongly INDUCED (60x). "
                        "Model has wrong direction."
                    )
                else:
                    checks_passed.append(
                        f"IFIT1 fold change direction correct: "
                        f"{ifit1_pred:.2f}x (induced)"
                    )

        # Check 4: Monocyte response vs B cell
        if "cell_type_pearson_r" in result.outputs:
            ct_r = result.outputs["cell_type_pearson_r"]
            if isinstance(ct_r, dict):
                mono_r = ct_r.get("CD14+ Monocytes", 0)
                bcell_r = ct_r.get("B cells", 0)

                if mono_r > 0:
                    checks_passed.append(
                        f"CD14+ Monocyte r = {mono_r:.3f}"
                    )

                # Monocytes are strongest IFN-b responders
                if bcell_r > mono_r + 0.1:
                    warnings.append(
                        f"B cell r ({bcell_r:.3f}) exceeds Monocyte r "
                        f"({mono_r:.3f}) by more than 0.1. "
                        "Monocytes are primary IFN-b responders. "
                        "Investigate cell-type embedding."
                    )

        # Check 5: Mean plausibility score
        if "mean_plausibility_score" in result.outputs:
            mean_p = result.outputs["mean_plausibility_score"]
            if isinstance(mean_p, (int, float)):
                if mean_p > 0.4:
                    checks_passed.append(
                        f"Mean plausibility: {mean_p:.3f} > 0.4"
                    )
                else:
                    checks_failed.append(
                        f"Mean plausibility too low: {mean_p:.3f}"
                    )

        # Check 6: Quarantine fraction
        if "quarantine_fraction" in result.outputs:
            q_frac = result.outputs["quarantine_fraction"]
            if isinstance(q_frac, (int, float)):
                if q_frac < 0.3:
                    checks_passed.append(
                        f"Quarantine fraction: {q_frac:.3f} < 0.3"
                    )
                else:
                    checks_failed.append(
                        f"Quarantine fraction too high: {q_frac:.3f} "
                        "(max 0.3)"
                    )

        # If no biological checks were applicable, pass by default
        if not checks_passed and not checks_failed:
            checks_passed.append(
                "No biological checks applicable for this skill"
            )

        passed = len(checks_failed) == 0
        return ValidationReport(
            passed=passed,
            critic_name="BiologicalCritic",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            biological_score=bio_score,
            uncertainty_flags=warnings,
            quarantined_outputs=quarantined,
        )
