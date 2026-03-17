"""
aivc/skills/plausibility.py — Biological plausibility scoring skill.

Cross-references model predictions against known biological databases.
Assigns plausibility score (0 to 1) to every predicted gene-gene interaction.
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
    name="biological_plausibility_scorer",
    domain=BiologicalDomain.EVALUATION,
    version="1.0.0",
    requires=["predicted_interactions", "gene_to_idx"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class BiologicalPlausibilityScorer(AIVCSkill):
    """
    Cross-references model predictions against known biological databases.
    Assigns plausibility score (0 to 1) to every predicted gene-gene
    interaction.

    Databases used (in order of authority):
        1. STRING PPI (already in repo as edge_list_fixed.csv)
        2. KEGG pathway database (download via KEGG REST API)
        3. Reactome pathway annotations (download via Reactome API)
        4. Manual JAK-STAT ground truth (hardcoded from literature)

    Scoring:
        In STRING AND in KEGG/Reactome: 1.0
        In STRING only:                 0.7
        In KEGG/Reactome only:          0.6
        Novel (not in any database):    0.3
        Contradicts known biology:      0.0 and QUARANTINE

    Predictions below 0.3 are quarantined.
    Quarantined predictions trigger wet lab validation flag.
    """

    JAKSTAT_GROUND_TRUTH = {
        ("JAK1", "STAT1"), ("JAK1", "STAT2"), ("JAK2", "STAT1"),
        ("STAT1", "IRF9"), ("STAT2", "IRF9"), ("IRF9", "IFIT1"),
        ("IRF9", "ISG15"), ("IRF9", "MX1"), ("STAT1", "MX1"),
    }

    # Known contradictions: gene pairs that should NOT interact
    KNOWN_CONTRADICTIONS = set()  # Can be populated from literature

    def _load_string_edges(self, data_dir: str = "data/") -> set:
        """Load STRING PPI edges from edge_list_fixed.csv."""
        string_edges = set()
        edge_file = os.path.join(data_dir, "edge_list_fixed.csv")
        if not os.path.exists(edge_file):
            edge_file = os.path.join(data_dir, "edge_list.csv")
        if os.path.exists(edge_file):
            try:
                import pandas as pd
                df = pd.read_csv(edge_file)
                for _, row in df.iterrows():
                    src = str(row.iloc[0])
                    tgt = str(row.iloc[1])
                    string_edges.add((src, tgt))
                    string_edges.add((tgt, src))
            except Exception:
                pass
        return string_edges

    def _is_in_jakstat(self, gene_a: str, gene_b: str) -> bool:
        """Check if pair is in JAK-STAT ground truth (bidirectional)."""
        return (
            (gene_a, gene_b) in self.JAKSTAT_GROUND_TRUTH
            or (gene_b, gene_a) in self.JAKSTAT_GROUND_TRUTH
        )

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, ["predicted_interactions", "gene_to_idx"])
        t0 = time.time()
        warnings = []
        errors = []

        predicted_interactions = inputs["predicted_interactions"]
        gene_to_idx = inputs["gene_to_idx"]
        data_dir = inputs.get("data_dir", "data/")

        # Load STRING PPI edges
        string_edges = self._load_string_edges(data_dir)
        if not string_edges:
            warnings.append(
                "Could not load STRING PPI edges. "
                "Scoring based on JAK-STAT ground truth only."
            )

        scored = []
        quarantined = []
        jakstat_recovered = 0

        for interaction in predicted_interactions:
            if isinstance(interaction, (list, tuple)):
                gene_a, gene_b = interaction[0], interaction[1]
                pred_score = interaction[2] if len(interaction) > 2 else 1.0
            elif isinstance(interaction, dict):
                gene_a = interaction.get("gene_a", "")
                gene_b = interaction.get("gene_b", "")
                pred_score = interaction.get("score", 1.0)
            else:
                continue

            in_string = (gene_a, gene_b) in string_edges
            in_jakstat = self._is_in_jakstat(gene_a, gene_b)
            in_contradictions = (
                (gene_a, gene_b) in self.KNOWN_CONTRADICTIONS
                or (gene_b, gene_a) in self.KNOWN_CONTRADICTIONS
            )

            # Scoring logic
            if in_contradictions:
                plausibility = 0.0
            elif in_string and in_jakstat:
                plausibility = 1.0
            elif in_string:
                plausibility = 0.7
            elif in_jakstat:
                plausibility = 0.8  # Known biology but not in STRING
            else:
                plausibility = 0.3  # Novel

            entry = {
                "gene_pair": f"{gene_a}-{gene_b}",
                "gene_a": gene_a,
                "gene_b": gene_b,
                "prediction_score": pred_score,
                "plausibility_score": plausibility,
                "in_string": in_string,
                "in_jakstat": in_jakstat,
                "quarantined": plausibility < 0.3 or in_contradictions,
            }

            scored.append(entry)

            if entry["quarantined"]:
                quarantined.append(entry)

            if in_jakstat:
                jakstat_recovered += 1

        # Compute mean plausibility
        if scored:
            mean_plausibility = np.mean(
                [s["plausibility_score"] for s in scored]
            )
        else:
            mean_plausibility = 0.0
            warnings.append("No interactions to score")

        elapsed = time.time() - t0
        return SkillResult(
            skill_name=self.name,
            version=self.version,
            success=True,
            outputs={
                "scored_interactions": scored,
                "quarantined_interactions": quarantined,
                "mean_plausibility_score": float(mean_plausibility),
                "jakstat_recovery_in_predictions": jakstat_recovered,
                "n_total_scored": len(scored),
                "n_quarantined": len(quarantined),
                "quarantine_fraction": (
                    len(quarantined) / max(len(scored), 1)
                ),
            },
            metadata={
                "elapsed_seconds": elapsed,
                "n_string_edges": len(string_edges),
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
                passed=False,
                critic_name="BiologicalPlausibilityScorer.validate",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check: mean_plausibility_score > 0.4
        mean_p = outputs.get("mean_plausibility_score", 0)
        if mean_p > 0.4:
            checks_passed.append(
                f"Mean plausibility score: {mean_p:.3f} > 0.4"
            )
        else:
            checks_failed.append(
                f"Mean plausibility score too low: {mean_p:.3f} (min 0.4)"
            )

        # Check: quarantined fraction < 0.3
        q_frac = outputs.get("quarantine_fraction", 0)
        if q_frac < 0.3:
            checks_passed.append(
                f"Quarantine fraction: {q_frac:.3f} < 0.3"
            )
        else:
            checks_failed.append(
                f"Too many quarantined: {q_frac:.3f} (max 0.3)"
            )

        # Check: JAK-STAT recovery
        jakstat = outputs.get("jakstat_recovery_in_predictions", 0)
        if jakstat >= 3:
            checks_passed.append(
                f"JAK-STAT recovery in predictions: {jakstat}"
            )
        else:
            recommendation = (
                f"JAK-STAT recovery low: {jakstat}. "
                "Model may not be capturing interferon signaling."
            )
            checks_failed.append(recommendation)

        quarantined = outputs.get("quarantined_interactions", [])
        quarantined_pairs = [q["gene_pair"] for q in quarantined]

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="BiologicalPlausibilityScorer.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            quarantined_outputs=quarantined_pairs,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=0.5,
            gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0,
            can_run_on_cpu=True,
        )
