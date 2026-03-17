"""
aivc/critics/methodological.py — Scientific methodology validation.

Validates scientific methodology.
Prevents the most common sources of inflated or non-comparable results.
"""

from aivc.interfaces import SkillResult, ValidationReport


class MethodologicalCritic:
    """
    Validates scientific methodology.
    Prevents the most common sources of inflated or non-comparable results.
    """

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []

        # Check 1: Pearson r computed on raw values not log-transformed
        if "pearson_r_on_raw" in result.metadata:
            if result.metadata["pearson_r_on_raw"]:
                checks_passed.append(
                    "Pearson r computed on raw values (correct)"
                )
            else:
                checks_failed.append(
                    "CRITICAL: Pearson r computed on transformed values. "
                    "CPA and scGEN benchmarks use raw values. "
                    "Results not comparable."
                )

        # Check 2: Donor-based split used (not random)
        if "split_method" in result.metadata:
            if result.metadata["split_method"] == "donor_based":
                checks_passed.append(
                    "Donor-based split used (correct)"
                )
            else:
                checks_failed.append(
                    f"Split method is '{result.metadata['split_method']}', "
                    "not donor-based. Random splits cause data leakage "
                    "and inflate Pearson r."
                )

        # Check 3: OT pairing quality
        if "pairing_quality_score" in result.outputs:
            q = result.outputs["pairing_quality_score"]
            baseline = result.outputs.get("unpaired_baseline", 0)
            if q > baseline + 0.05:
                checks_passed.append(
                    f"OT pairing quality ({q:.3f}) exceeds "
                    f"baseline ({baseline:.3f}) by >= 0.05"
                )
            else:
                checks_failed.append(
                    f"OT pairing quality ({q:.3f}) not better than "
                    f"unpaired baseline ({baseline:.3f}) by 0.05. "
                    "Pseudo-bulk fallback should be used."
                )

        # Check 4: Test donors never appeared in training
        if ("train_donors" in result.metadata
                and "test_donors" in result.metadata):
            train = set(result.metadata["train_donors"])
            test = set(result.metadata["test_donors"])
            overlap = train & test
            if overlap:
                checks_failed.append(
                    f"DATA LEAKAGE: donors in both train and test: "
                    f"{overlap}"
                )
            else:
                checks_passed.append(
                    "No donor overlap between train and test sets"
                )

        # Check 5: Benchmark uses identical dataset
        if "benchmark_dataset" in result.metadata:
            if result.metadata["benchmark_dataset"] == "kang2018_GSE96583":
                checks_passed.append(
                    "Benchmark on Kang 2018 GSE96583 (correct)"
                )
            else:
                checks_failed.append(
                    f"Benchmark not on Kang 2018 GSE96583 "
                    f"(got: {result.metadata['benchmark_dataset']}). "
                    "Comparison to CPA/scGEN invalid on different dataset."
                )

        # Check 6: Residual learning flag
        if "residual_learning" in result.metadata:
            if result.metadata["residual_learning"]:
                checks_passed.append("Residual learning enabled")

        # If no methodology checks were applicable, pass by default
        if not checks_passed and not checks_failed:
            checks_passed.append(
                "No methodology checks applicable for this skill"
            )

        passed = len(checks_failed) == 0
        return ValidationReport(
            passed=passed,
            critic_name="MethodologicalCritic",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
