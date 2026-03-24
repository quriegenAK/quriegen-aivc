"""
Benchmark integrity check: compare Pearson r on raw vs decontaminated Kang 2018.

Answers the question: how much of r=0.873 is real biology vs ambient noise?

Thresholds:
  CLEAN:    IFIT1 ambient < 5%  AND  r_delta < 0.005
  MODERATE: IFIT1 ambient < 10% OR   r_delta < 0.02
  INFLATED: IFIT1 ambient > 10% AND  r_delta >= 0.02
"""
import logging

logger = logging.getLogger("aivc.qc.integrity")


class BenchmarkIntegrityChecker:
    """
    Compare model performance on contaminated vs decontaminated data.

    Workflow:
      1. Run ambient decontamination on Kang 2018
      2. Re-evaluate the v1.0 model on clean test set
      3. Compare r_raw vs r_clean
      4. Classify benchmark status
    """

    THRESHOLDS = {
        "clean":    0.05,   # ambient fraction < 5% -> benchmark is clean
        "moderate": 0.10,   # 5-10% -> document and monitor
        "inflated": 0.10,   # > 10% -> benchmark is inflated
    }

    def run_integrity_check(
        self,
        contamination_report,
        r_raw: float,
        r_clean: float,
    ) -> dict:
        """
        Given the contamination report and both Pearson r values,
        compute the integrity verdict.

        Args:
            contamination_report: pd.DataFrame from gene_contamination_report()
            r_raw:   Pearson r on original (contaminated) data
            r_clean: Pearson r on decontaminated data

        Returns:
            dict with: verdict, r_raw, r_clean, r_delta,
            ifit1_ambient_pct, jakstat_contaminated, recommendation,
            external_report.
        """
        r_delta = r_raw - r_clean

        # JAK-STAT contamination
        jakstat_df = contamination_report[
            contamination_report["is_jakstat"]
        ].copy()

        ifit1_row = contamination_report[
            contamination_report["gene_name"] == "IFIT1"
        ]
        ifit1_pct = float(
            ifit1_row["ambient_fraction"].values[0] * 100
        ) if len(ifit1_row) > 0 else 0.0

        jakstat_contaminated = jakstat_df[
            jakstat_df["ambient_fraction"] > self.THRESHOLDS["inflated"]
        ]["gene_name"].tolist()

        # Verdict
        if ifit1_pct < 5.0 and r_delta < 0.005:
            verdict = "CLEAN"
            recommendation = (
                "Benchmark is clean. r=0.873 reflects real biological signal. "
                "No re-run required. Report r_raw in all external documents."
            )
        elif ifit1_pct < 10.0 or r_delta < 0.02:
            verdict = "MODERATE"
            recommendation = (
                f"Moderate contamination detected (IFIT1 ambient: {ifit1_pct:.1f}%). "
                f"r drops by {r_delta:.4f} on clean data. "
                f"Report r_clean = {r_clean:.4f} as the conservative benchmark. "
                "Note contamination in methods section."
            )
        else:
            verdict = "INFLATED"
            recommendation = (
                f"Significant contamination detected (IFIT1 ambient: {ifit1_pct:.1f}%). "
                f"r drops by {r_delta:.4f} on clean data. "
                f"External benchmark must be updated to r_clean = {r_clean:.4f}. "
                f"Re-run training on decontaminated data."
            )

        # External-safe report text
        if verdict == "CLEAN":
            external_report = (
                f"Pearson r = {r_raw:.4f} on held-out donors (Kang 2018 PBMC, "
                f"GSE96583). Ambient RNA decontamination confirmed < 5% "
                f"contamination of IFN-response genes in control cells. "
                f"Benchmark reflects real biological signal."
            )
        else:
            external_report = (
                f"Pearson r = {r_clean:.4f} on held-out donors after ambient "
                f"RNA decontamination (Kang 2018 PBMC, GSE96583). "
                f"Pre-decontamination r = {r_raw:.4f}. "
                f"IFIT1 ambient contamination in control cells: {ifit1_pct:.1f}%."
            )

        return {
            "verdict":              verdict,
            "r_raw":                r_raw,
            "r_clean":              r_clean,
            "r_delta":              r_delta,
            "ifit1_ambient_pct":    ifit1_pct,
            "jakstat_contaminated": jakstat_contaminated,
            "recommendation":       recommendation,
            "external_report":      external_report,
        }
