"""
aivc/skills/graph_builder.py — STRING PPI graph construction skill.

Wraps build_edge_list.py logic. Builds edge_index tensor from STRING PPI
filtered to HVG gene set. Validates edge count and JAK-STAT connectivity.
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
    name="graph_builder",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["gene_to_idx", "string_ppi_path"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class GraphBuilder(AIVCSkill):
    """
    Builds edge_index tensor from STRING PPI filtered to HVG gene set.
    Wraps build_edge_list.py logic.
    """

    JAKSTAT_GENES = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
        "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, ["gene_to_idx", "string_ppi_path"])
        t0 = time.time()
        warnings = []
        errors = []

        gene_to_idx = inputs["gene_to_idx"]
        string_ppi_path = inputs["string_ppi_path"]
        score_threshold = inputs.get("score_threshold", 700)

        try:
            import torch
            import pandas as pd

            # Read STRING PPI
            if not os.path.exists(string_ppi_path):
                errors.append(f"STRING PPI file not found: {string_ppi_path}")
                return SkillResult(
                    skill_name=self.name, version=self.version,
                    success=False, outputs={},
                    metadata={"elapsed_seconds": time.time() - t0},
                    warnings=warnings, errors=errors,
                )

            # Parse STRING PPI file
            edges = []
            jak_stat_edges = 0
            jak_stat_genes_in_graph = set()

            # Try CSV format first (edge_list.csv from build_edge_list.py)
            try:
                df = pd.read_csv(string_ppi_path)
                if "source" in df.columns and "target" in df.columns:
                    for _, row in df.iterrows():
                        src = str(row["source"])
                        tgt = str(row["target"])
                        if src in gene_to_idx and tgt in gene_to_idx:
                            src_idx = gene_to_idx[src]
                            tgt_idx = gene_to_idx[tgt]
                            if src_idx != tgt_idx:  # no self-loops
                                edges.append([src_idx, tgt_idx])
                                edges.append([tgt_idx, src_idx])
                                if src in self.JAKSTAT_GENES or tgt in self.JAKSTAT_GENES:
                                    jak_stat_edges += 1
                                    if src in self.JAKSTAT_GENES:
                                        jak_stat_genes_in_graph.add(src)
                                    if tgt in self.JAKSTAT_GENES:
                                        jak_stat_genes_in_graph.add(tgt)
                elif "gene_a" in df.columns and "gene_b" in df.columns:
                    score_col = "score" if "score" in df.columns else None
                    for _, row in df.iterrows():
                        if score_col and row[score_col] < score_threshold:
                            continue
                        src = str(row["gene_a"])
                        tgt = str(row["gene_b"])
                        if src in gene_to_idx and tgt in gene_to_idx:
                            src_idx = gene_to_idx[src]
                            tgt_idx = gene_to_idx[tgt]
                            if src_idx != tgt_idx:
                                edges.append([src_idx, tgt_idx])
                                edges.append([tgt_idx, src_idx])
                                if src in self.JAKSTAT_GENES or tgt in self.JAKSTAT_GENES:
                                    jak_stat_edges += 1
                                    if src in self.JAKSTAT_GENES:
                                        jak_stat_genes_in_graph.add(src)
                                    if tgt in self.JAKSTAT_GENES:
                                        jak_stat_genes_in_graph.add(tgt)
            except Exception:
                # Try space/tab-separated STRING format
                with open(string_ppi_path, "r") as f:
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            gene_a = parts[0].split(".")[-1] if "." in parts[0] else parts[0]
                            gene_b = parts[1].split(".")[-1] if "." in parts[1] else parts[1]
                            score = int(parts[-1])
                            if score >= score_threshold:
                                if gene_a in gene_to_idx and gene_b in gene_to_idx:
                                    src_idx = gene_to_idx[gene_a]
                                    tgt_idx = gene_to_idx[gene_b]
                                    if src_idx != tgt_idx:
                                        edges.append([src_idx, tgt_idx])
                                        edges.append([tgt_idx, src_idx])

            if not edges:
                errors.append("No edges found after filtering")
                return SkillResult(
                    skill_name=self.name, version=self.version,
                    success=False, outputs={},
                    metadata={"elapsed_seconds": time.time() - t0},
                    warnings=warnings, errors=errors,
                )

            # Deduplicate
            edge_set = set()
            unique_edges = []
            for e in edges:
                key = (e[0], e[1])
                if key not in edge_set:
                    edge_set.add(key)
                    unique_edges.append(e)

            edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()

            # Check JAK-STAT genes without edges
            for gene in self.JAKSTAT_GENES:
                if gene in gene_to_idx and gene not in jak_stat_genes_in_graph:
                    warnings.append(
                        f"JAK-STAT gene '{gene}' has no edges in graph"
                    )

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "edge_index": edge_index,
                    "n_edges": edge_index.shape[1],
                    "jak_stat_edges": jak_stat_edges,
                    "jak_stat_genes_in_graph": list(jak_stat_genes_in_graph),
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "score_threshold": score_threshold,
                    "random_seed_used": 42,
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Graph building failed: {str(e)}")
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
                passed=False, critic_name="GraphBuilder.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # Check: n_edges > 5000
        n_edges = outputs.get("n_edges", 0)
        if n_edges > 5000:
            checks_passed.append(f"Edge count sufficient: {n_edges}")
        else:
            checks_failed.append(
                f"Edge count too low: {n_edges} (minimum 5000). "
                "Graph may be disconnected."
            )

        # Check: edge_index shape is [2, n_edges]
        edge_index = outputs.get("edge_index")
        if edge_index is not None and edge_index.shape[0] == 2:
            checks_passed.append("edge_index shape correct [2, n_edges]")
        else:
            shape = edge_index.shape if edge_index is not None else "None"
            checks_failed.append(f"edge_index wrong shape: {shape}")

        # Check: no self-loops
        if edge_index is not None:
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            if self_loops == 0:
                checks_passed.append("No self-loops detected")
            else:
                checks_failed.append(f"Found {self_loops} self-loops")

        # Check: jak_stat_edges > 0
        jak_edges = outputs.get("jak_stat_edges", 0)
        if jak_edges > 0:
            checks_passed.append(f"JAK-STAT edges present: {jak_edges}")
        else:
            checks_failed.append("No JAK-STAT edges in graph")

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="GraphBuilder.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        return ComputeCost(
            estimated_minutes=1.0,
            gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0,
            can_run_on_cpu=True,
        )
