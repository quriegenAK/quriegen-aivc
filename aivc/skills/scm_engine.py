"""
aivc/skills/scm_engine.py — Causal intervention engine for AIVC.

Implements do(X) interventions on the learned Neumann W matrix.
Produces counterfactual predictions: "what would gene expression look
like if gene X were silenced / enhanced / the pathway were blocked?"

This is NOT a full Pearl do-calculus SCM. It is a principled
intervention on the learned gene regulatory network (W matrix) that
produces testable, interpretable counterfactuals using the existing
Neumann propagation infrastructure.

Supported interventions:
  do_gene_ko(gene):         silence a gene completely
  do_gene_oe(gene, fc):     overexpress a gene by fold-change fc
  do_pathway_block(genes):  block an entire pathway simultaneously
  do_peak_closed(peak):     ATAC accessibility intervention (stub)
"""
import logging
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger("aivc.skills.scm")


@dataclass
class CounterfactualResult:
    """Result of a single do(X) intervention."""
    intervention_type: str
    intervened_genes: List[str]
    baseline_pred: torch.Tensor
    counterfactual_pred: torch.Tensor
    delta: torch.Tensor
    top_affected_genes: List[Dict]
    causal_path: List[Dict]
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    w_density_at_inference: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "intervention_type": self.intervention_type,
            "intervened_genes": self.intervened_genes,
            "top_affected_genes": self.top_affected_genes,
            "causal_path": self.causal_path,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "w_density": self.w_density_at_inference,
            "delta_nonzero": int((self.delta.abs() > 0.01).sum()),
            "max_delta_gene": (
                self.top_affected_genes[0]["gene_name"]
                if self.top_affected_genes else "none"
            ),
            "max_delta_fc": (
                self.top_affected_genes[0]["fold_change"]
                if self.top_affected_genes else 0.0
            ),
        }


class SCMEngine:
    """
    Causal intervention engine built on the AIVC Neumann W matrix.

    Operates in inference mode (torch.no_grad). Never permanently
    modifies the trained W matrix — interventions use temporary copies.

    Args:
        neumann:          Trained NeumannPropagation module.
        response_decoder: nn.Module producing (batch, n_genes) direct effects.
        gene_names:       List of gene name strings (len = n_genes).
        min_w_density:    Min W density to consider model trained. Default 0.05.
    """

    JAKSTAT_PANEL = [
        "JAK1", "JAK2", "TYK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IRF3", "IRF7",
        "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "ISG15", "ISG20",
    ]

    KNOWN_CAUSAL_PATHS = {
        "IFN_JAK_STAT": [
            ("JAK1", "STAT1"), ("JAK2", "STAT1"), ("TYK2", "STAT2"),
            ("STAT1", "IRF9"), ("STAT2", "IRF9"), ("IRF9", "IFIT1"),
            ("STAT1", "IFIT1"), ("STAT1", "MX1"), ("STAT1", "ISG15"),
        ],
    }

    def __init__(
        self,
        neumann,
        response_decoder: nn.Module,
        gene_names: Optional[List[str]] = None,
        min_w_density: float = 0.05,
    ):
        self.neumann = neumann
        self.response_decoder = response_decoder
        self.gene_names = gene_names
        self.min_w_density = min_w_density
        self.n_genes = neumann.n_genes
        self._gene_to_idx: Dict[str, int] = {}
        if gene_names is not None:
            self._gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def _validate_readiness(self) -> List[str]:
        warnings = []
        density = self.neumann.get_effective_W_density()
        if density < self.min_w_density:
            warnings.append(
                f"W matrix density = {density:.4f} < {self.min_w_density} "
                f"(min_w_density threshold). W is not yet sufficiently trained. "
                f"Counterfactual result marked is_valid=False."
            )
        return warnings

    def _get_gene_idx(self, gene: str) -> Optional[int]:
        if gene in self._gene_to_idx:
            return self._gene_to_idx[gene]
        try:
            idx = int(gene)
            if 0 <= idx < self.n_genes:
                return idx
        except (ValueError, TypeError):
            pass
        return None

    def _build_intervention_mask(self, ko_gene_indices: List[int]) -> torch.Tensor:
        """Build edge mask zeroing all W edges incoming to KO genes."""
        mask = torch.ones(len(self.neumann.W), dtype=torch.bool,
                          device=self.neumann.W.device)
        for gene_idx in ko_gene_indices:
            incoming = (self.neumann.edge_dst == gene_idx)
            mask[incoming] = False
        return mask

    def _compute_direct_effects(self, ctrl_expr: torch.Tensor) -> torch.Tensor:
        x = ctrl_expr
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.response_decoder(x)

    def _run_counterfactual(self, ctrl_expr, masked_W, modified_d_p,
                            original_pred, top_k):
        original_W = self.neumann.W.data.clone()
        try:
            if masked_W is not None:
                self.neumann.W.data = masked_W
            d_p = (modified_d_p if modified_d_p is not None
                   else self._compute_direct_effects(ctrl_expr))
            counterfactual = self.neumann(d_p)
            if counterfactual.dim() == 2:
                counterfactual = counterfactual.squeeze(0)
            delta = counterfactual - original_pred
            top_affected = self._get_top_affected(delta, top_k)
        finally:
            self.neumann.W.data = original_W
        return counterfactual, delta, top_affected

    def _get_top_affected(self, delta: torch.Tensor, top_k: int = 20) -> List[Dict]:
        abs_delta = delta.abs()
        top_vals, top_idxs = torch.topk(abs_delta, min(top_k, len(delta)))
        result = []
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist())):
            gene_name = self.gene_names[idx] if self.gene_names else f"gene_{idx}"
            signed = delta[idx].item()
            result.append({
                "rank": rank + 1,
                "gene_name": gene_name,
                "gene_idx": idx,
                "delta": round(signed, 4),
                "fold_change": round(abs(signed), 4),
                "direction": "up" if signed > 0 else "down",
                "is_jakstat": gene_name in self.JAKSTAT_PANEL,
            })
        return result

    def _get_causal_path(self, ko_gene_indices: List[int]) -> List[Dict]:
        if not self.gene_names:
            return []
        ko_names = {self.gene_names[i] for i in ko_gene_indices
                    if i < len(self.gene_names)}
        path_edges = []
        for path_name, edges in self.KNOWN_CAUSAL_PATHS.items():
            for src, dst in edges:
                if src in ko_names or dst in ko_names:
                    path_edges.append({
                        "path": path_name, "src": src, "dst": dst,
                        "is_cut": src in ko_names,
                        "w_weight": self._get_edge_weight(src, dst),
                    })
        return path_edges

    def _get_edge_weight(self, src_name: str, dst_name: str) -> float:
        src_idx = self._gene_to_idx.get(src_name)
        dst_idx = self._gene_to_idx.get(dst_name)
        if src_idx is None or dst_idx is None:
            return 0.0
        with torch.no_grad():
            mask = ((self.neumann.edge_src == src_idx) &
                    (self.neumann.edge_dst == dst_idx))
            if mask.any():
                return round(self.neumann.W[mask][0].item(), 6)
        return 0.0

    # ─── Public API ────────────────────────────────────────────────

    def do_gene_ko(self, gene: str, ctrl_expr: torch.Tensor,
                   top_k: int = 20) -> CounterfactualResult:
        """Knock out a single gene. Returns CounterfactualResult."""
        warnings = self._validate_readiness()
        is_valid = not any("not yet sufficiently trained" in w for w in warnings)

        gene_idx = self._get_gene_idx(gene)
        if gene_idx is None:
            return CounterfactualResult(
                intervention_type="do_gene_ko", intervened_genes=[gene],
                baseline_pred=torch.zeros(self.n_genes),
                counterfactual_pred=torch.zeros(self.n_genes),
                delta=torch.zeros(self.n_genes),
                top_affected_genes=[], causal_path=[],
                is_valid=False,
                warnings=[f"Gene '{gene}' not found in gene universe."],
            )

        with torch.no_grad():
            d_p_baseline = self._compute_direct_effects(ctrl_expr)
            baseline_pred = self.neumann(d_p_baseline)
            if baseline_pred.dim() == 2:
                baseline_pred = baseline_pred.squeeze(0)

            d_p_ko = d_p_baseline.clone()
            d_p_ko[:, gene_idx] = 0.0

            mask = self._build_intervention_mask([gene_idx])
            masked_W = self.neumann.W.data.clone()
            masked_W[~mask] = 0.0

            counterfactual, delta, top_affected = self._run_counterfactual(
                ctrl_expr, masked_W, d_p_ko, baseline_pred, top_k,
            )

        causal_path = self._get_causal_path([gene_idx])
        density = self.neumann.get_effective_W_density()

        if gene == "JAK1" and "IFIT1" in self._gene_to_idx:
            ifit1_delta = delta[self._gene_to_idx["IFIT1"]].item()
            if ifit1_delta > 0:
                warnings.append(
                    f"WARNING: JAK1 KO increased IFIT1 by {ifit1_delta:.4f}. "
                    f"Expected decrease. W[JAK1->STAT1] may not be positive."
                )

        return CounterfactualResult(
            intervention_type="do_gene_ko", intervened_genes=[gene],
            baseline_pred=baseline_pred.cpu(),
            counterfactual_pred=counterfactual.cpu(),
            delta=delta.cpu(), top_affected_genes=top_affected,
            causal_path=causal_path, is_valid=is_valid,
            warnings=warnings, w_density_at_inference=density,
        )

    def do_pathway_block(self, genes: List[str], ctrl_expr: torch.Tensor,
                         top_k: int = 20) -> CounterfactualResult:
        """Block entire pathway by KO of all listed genes simultaneously."""
        warnings = self._validate_readiness()
        is_valid = not any("not yet sufficiently trained" in w for w in warnings)

        gene_indices, missing = [], []
        for gene in genes:
            idx = self._get_gene_idx(gene)
            if idx is not None:
                gene_indices.append(idx)
            else:
                missing.append(gene)
        if missing:
            warnings.append(f"Genes not found in universe: {missing}.")
        if not gene_indices:
            return CounterfactualResult(
                intervention_type="do_pathway_block", intervened_genes=genes,
                baseline_pred=torch.zeros(self.n_genes),
                counterfactual_pred=torch.zeros(self.n_genes),
                delta=torch.zeros(self.n_genes),
                top_affected_genes=[], causal_path=[],
                is_valid=False, warnings=warnings + ["No valid gene indices."],
            )

        with torch.no_grad():
            d_p_baseline = self._compute_direct_effects(ctrl_expr)
            baseline_pred = self.neumann(d_p_baseline)
            if baseline_pred.dim() == 2:
                baseline_pred = baseline_pred.squeeze(0)

            d_p_blocked = d_p_baseline.clone()
            for idx in gene_indices:
                d_p_blocked[:, idx] = 0.0

            mask = self._build_intervention_mask(gene_indices)
            masked_W = self.neumann.W.data.clone()
            masked_W[~mask] = 0.0

            counterfactual, delta, top_affected = self._run_counterfactual(
                ctrl_expr, masked_W, d_p_blocked, baseline_pred, top_k,
            )

        return CounterfactualResult(
            intervention_type="do_pathway_block", intervened_genes=genes,
            baseline_pred=baseline_pred.cpu(),
            counterfactual_pred=counterfactual.cpu(),
            delta=delta.cpu(), top_affected_genes=top_affected,
            causal_path=self._get_causal_path(gene_indices),
            is_valid=is_valid, warnings=warnings,
            w_density_at_inference=self.neumann.get_effective_W_density(),
        )

    def do_gene_oe(self, gene: str, fold_change: float,
                   ctrl_expr: torch.Tensor, top_k: int = 20) -> CounterfactualResult:
        """Overexpress a gene by scaling its direct effect."""
        warnings = self._validate_readiness()
        is_valid = not any("not yet sufficiently trained" in w for w in warnings)

        if fold_change <= 0:
            return CounterfactualResult(
                intervention_type="do_gene_oe", intervened_genes=[gene],
                baseline_pred=torch.zeros(self.n_genes),
                counterfactual_pred=torch.zeros(self.n_genes),
                delta=torch.zeros(self.n_genes),
                top_affected_genes=[], causal_path=[],
                is_valid=False,
                warnings=[f"fold_change must be > 0. Got {fold_change}."],
            )

        gene_idx = self._get_gene_idx(gene)
        if gene_idx is None:
            return CounterfactualResult(
                intervention_type="do_gene_oe", intervened_genes=[gene],
                baseline_pred=torch.zeros(self.n_genes),
                counterfactual_pred=torch.zeros(self.n_genes),
                delta=torch.zeros(self.n_genes),
                top_affected_genes=[], causal_path=[],
                is_valid=False,
                warnings=[f"Gene '{gene}' not found."],
            )

        with torch.no_grad():
            d_p_baseline = self._compute_direct_effects(ctrl_expr)
            baseline_pred = self.neumann(d_p_baseline)
            if baseline_pred.dim() == 2:
                baseline_pred = baseline_pred.squeeze(0)

            d_p_oe = d_p_baseline.clone()
            d_p_oe[:, gene_idx] *= fold_change

            counterfactual, delta, top_affected = self._run_counterfactual(
                ctrl_expr, None, d_p_oe, baseline_pred, top_k,
            )

        return CounterfactualResult(
            intervention_type="do_gene_oe", intervened_genes=[gene],
            baseline_pred=baseline_pred.cpu(),
            counterfactual_pred=counterfactual.cpu(),
            delta=delta.cpu(), top_affected_genes=top_affected,
            causal_path=self._get_causal_path([gene_idx]),
            is_valid=is_valid, warnings=warnings,
            w_density_at_inference=self.neumann.get_effective_W_density(),
        )

    def do_peak_closed(self, peak_description: str,
                       ctrl_expr: torch.Tensor) -> CounterfactualResult:
        """STUB: ATAC intervention. Returns is_valid=False."""
        return CounterfactualResult(
            intervention_type="do_peak_closed",
            intervened_genes=[peak_description],
            baseline_pred=torch.zeros(self.n_genes),
            counterfactual_pred=torch.zeros(self.n_genes),
            delta=torch.zeros(self.n_genes),
            top_affected_genes=[], causal_path=[],
            is_valid=False,
            warnings=[
                "STUB: ATAC intervention requires trained ATACSeqEncoder. "
                "Prerequisites: (1) 10x Multiome data, "
                "(2) tf_motif_scanner.py implemented, "
                "(3) ATACSeqEncoder trained, "
                "(4) Full 4-modality forward pass. "
                "Current: RNA encoder only. Target: v3.0."
            ],
        )

    def counterfactual_report(self, ctrl_expr: torch.Tensor,
                              perturbation: str = "IFN_beta",
                              gene_panel: Optional[List[str]] = None,
                              top_k: int = 20) -> Dict[str, Any]:
        """Run all single-gene KO interventions for a panel."""
        if gene_panel is None:
            gene_panel = ["JAK1", "JAK2", "STAT1", "STAT2", "IRF9", "TYK2"]

        results = []
        for gene in gene_panel:
            result = self.do_gene_ko(gene, ctrl_expr, top_k=top_k)
            ifit1_delta = None
            if "IFIT1" in self._gene_to_idx:
                ifit1_delta = result.delta[self._gene_to_idx["IFIT1"]].item()
            results.append({
                "gene_ko": gene,
                "is_valid": result.is_valid,
                "n_downstream": int((result.delta.abs() > 0.01).sum()),
                "max_affected": result.top_affected_genes[:3],
                "ifit1_delta": ifit1_delta,
                "warnings": result.warnings,
            })

        summary_lines = [
            f"Counterfactual report: {perturbation} stimulation",
            f"W density: {self.neumann.get_effective_W_density():.3f}",
            f"Interventions run: {len(results)}",
        ]
        for r in results:
            if r["gene_ko"] in ["JAK1", "STAT1"] and r["ifit1_delta"] is not None:
                d = "down" if r["ifit1_delta"] < 0 else "up"
                summary_lines.append(
                    f"  do({r['gene_ko']}=KO) -> IFIT1 {d} {r['ifit1_delta']:+.4f}"
                )

        return {
            "perturbation": perturbation,
            "n_genes": self.n_genes,
            "w_density": self.neumann.get_effective_W_density(),
            "interventions": results,
            "summary": "\n".join(summary_lines),
        }
