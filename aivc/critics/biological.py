"""
aivc/critics/biological.py — Biological plausibility validation.

The hardest critic. Most dangerous failure mode: a model that is
statistically correct (r=0.873) but biologically wrong (STAT3 > STAT1).

Checks performed:
  1. JAK-STAT recovery score (primary — >=8/15 genes within 3x FC)
  2. Quarantine low-plausibility predictions (plausibility_score < 0.3)
  3. IFIT1 fold change direction (must be induced, not suppressed)
  4. CD14+ monocyte vs B-cell r (monocytes are primary IFN-B responders)
  5. Mean plausibility score (> 0.4)
  6. Quarantine fraction (< 0.3)
  7. ATAC JAK-STAT coverage (v3.0, >=10/15 genes with peak-gene links)
  8. IRF/STAT motif enrichment direction (v3.0)
  9. ATAC causal attention ordering (v3.0)

Mechanistic direction checks (added Phase 3):
  These require pred_np, ctrl_np, gene_to_idx in result.outputs.
  If absent, checks are skipped (not failed).
  Severity: WARNING only — not blocking until 5+ perturbations are trained.
  Rationale: on single-perturbation (IFN-B only) training, directional
  ratios may be marginal due to correlation rather than causation.

  _check_stat1_over_stat3():
    IFN-B in PBMCs activates STAT1 homodimer (type I IFN response).
    STAT3 is activated by IL-6 family cytokines, NOT primarily by IFN-B.
    Rule: predicted STAT1 FC > predicted STAT3 FC.
    Failure: model may be confusing IFN-B with IL-6 response.

  _check_ifit1_over_oas2():
    IFIT1 is among the top 3 most induced ISGs in PBMC IFN-B stimulation.
    OAS2 is induced but at lower fold change than IFIT1.
    Rule: predicted IFIT1 FC > predicted OAS2 FC.
    Failure: ISG induction hierarchy is wrong.
    Source: Rusinova et al. 2013, IFN response atlas.

  _check_jak1_upstream_of_ifit1():
    Causal path: JAK1 -> STAT1 -> IFIT1 (IFN-B signalling in PBMCs).
    Rule: W[JAK1->STAT1] > 0 AND W[STAT1->IFIT1] > 0.
    Failure: Neumann W matrix has not yet learned the causal cascade.
    Expected after proper training: both edges in top-20 W edges.

Inputs (all optional in result.outputs):
  "pred_np":     np.ndarray (n_cells, n_genes) predicted stim expression
  "ctrl_np":     np.ndarray (n_cells, n_genes) ctrl expression
  "gene_to_idx": dict gene_name -> column index
  "W_top_edges": list from neumann_module.get_top_edges(n=100, gene_names)
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

        # Check 7 (v3.0): ATAC JAK-STAT coverage
        if "atac_jakstat_coverage" in result.outputs:
            coverage = result.outputs["atac_jakstat_coverage"]
            if isinstance(coverage, (int, float)):
                if coverage >= 10:
                    checks_passed.append(
                        f"ATAC JAK-STAT coverage: {coverage}/15 (PASS)"
                    )
                else:
                    checks_failed.append(
                        f"ATAC JAK-STAT coverage: {coverage}/15 "
                        "(need >= 10/15 with peak-gene links)"
                    )

        # Check 8 (v3.0): IRF/STAT motif direction
        if "irf_stat_motif_direction" in result.outputs:
            motif_dir = result.outputs["irf_stat_motif_direction"]
            if isinstance(motif_dir, dict):
                critical_tfs = ["STAT1", "IRF3"]
                for tf in critical_tfs:
                    tf_result = motif_dir.get(tf, {})
                    enriched = tf_result.get("enriched", False)
                    if enriched:
                        checks_passed.append(
                            f"{tf} motif enriched in stim (PASS)"
                        )
                    elif tf in motif_dir:
                        checks_failed.append(
                            f"{tf} motif NOT enriched in stim. "
                            "ATACSeqEncoder may be learning noise."
                        )

        # Check 9 (v3.0): ATAC causal ordering
        if "atac_causal_attention" in result.outputs:
            attn = result.outputs["atac_causal_attention"]
            if isinstance(attn, dict):
                atac_to_rna = attn.get("atac_to_rna", 0)
                rna_to_atac = attn.get("rna_to_atac", 0)
                if atac_to_rna > rna_to_atac:
                    checks_passed.append(
                        f"ATAC->RNA attention ({atac_to_rna:.3f}) > "
                        f"RNA->ATAC ({rna_to_atac:.3f}) — causal ordering OK"
                    )
                else:
                    warnings.append(
                        f"RNA->ATAC attention ({rna_to_atac:.3f}) >= "
                        f"ATAC->RNA ({atac_to_rna:.3f}). "
                        "Causal consistency loss may need higher weight."
                    )

        # ── Mechanistic direction checks ─────────────────────────
        # Require pred_np + ctrl_np + gene_to_idx in result.outputs.
        # If absent: checks are skipped (not failed).
        # Severity: WARNING only — not blocking until Stage 3+ training.
        _pred_np     = result.outputs.get("pred_np")
        _ctrl_np     = result.outputs.get("ctrl_np")
        _gene_to_idx = result.outputs.get("gene_to_idx")
        _W_top_edges = result.outputs.get("W_top_edges")

        _direction_results = {}
        _direction_n_passed = 0
        _direction_n_run    = 0

        if (
            _pred_np is not None
            and _ctrl_np is not None
            and _gene_to_idx is not None
        ):
            _dc_stat1 = self._check_stat1_over_stat3(
                _pred_np, _ctrl_np, _gene_to_idx
            )
            _dc_ifit1 = self._check_ifit1_over_oas2(
                _pred_np, _ctrl_np, _gene_to_idx
            )
            _direction_results["stat1_over_stat3"] = _dc_stat1
            _direction_results["ifit1_over_oas2"]  = _dc_ifit1

            if _W_top_edges is not None:
                _dc_jak1 = self._check_jak1_upstream_of_ifit1(
                    _W_top_edges, _gene_to_idx
                )
                _direction_results["jak1_upstream_ifit1"] = _dc_jak1

            for _name, _dc in _direction_results.items():
                if _dc.get("skipped"):
                    continue
                _direction_n_run += 1
                if _dc["passed"]:
                    _direction_n_passed += 1
                    checks_passed.append(
                        f"[direction] {_dc['message']}"
                    )
                else:
                    warnings.append(
                        f"[direction WARN] {_dc['message']}"
                    )
                    bio_score = max(0.0, bio_score - 0.10)

        _direction_ok = (
            _direction_n_run == 0
            or _direction_n_passed == _direction_n_run
        )
        if not _direction_ok:
            _n_failed = _direction_n_run - _direction_n_passed
            _mech_rec = (
                f"{_n_failed}/{_direction_n_run} mechanistic direction "
                f"checks failed. Model achieves correct aggregate metrics "
                f"but pathway ordering may be wrong. "
                f"Required: STAT1 > STAT3 (IFN-B not IL-6), "
                f"IFIT1 > OAS2 (ISG hierarchy), "
                f"W[JAK1->STAT1] > 0 (causal path positive). "
                f"Run 5+ perturbations before trusting direction checks."
            )
        else:
            _mech_rec = None
        # ─────────────────────────────────────────────────────────

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
            recommendation=_mech_rec,
        )

    # ── Mechanistic direction check methods ──────────────────────

    def _check_stat1_over_stat3(self, pred_np, ctrl_np, gene_to_idx) -> dict:
        """
        IFN-B: STAT1 FC > STAT3 FC in PBMCs.
        If STAT3 > STAT1: model may confuse IFN-B with IL-6.
        Source: Stark & Darnell 2012.
        """
        import numpy as np
        eps = 1e-6
        stat1_idx = gene_to_idx.get("STAT1")
        stat3_idx = gene_to_idx.get("STAT3")
        if stat1_idx is None or stat3_idx is None:
            return {"passed": True, "stat1_fc": 0.0, "stat3_fc": 0.0,
                    "message": "STAT1 or STAT3 not in gene universe. Skipped.",
                    "skipped": True}
        stat1_fc = float((pred_np[:, stat1_idx].mean() + eps) /
                         (ctrl_np[:, stat1_idx].mean() + eps))
        stat3_fc = float((pred_np[:, stat3_idx].mean() + eps) /
                         (ctrl_np[:, stat3_idx].mean() + eps))
        passed = stat1_fc > stat3_fc
        return {
            "passed": passed, "stat1_fc": stat1_fc, "stat3_fc": stat3_fc,
            "skipped": False,
            "message": (
                f"STAT1 FC {stat1_fc:.2f}x {'>' if passed else '<='} "
                f"STAT3 FC {stat3_fc:.2f}x"
                + ("" if passed else " — FAIL: model may confuse IFN-B with IL-6")
            ),
        }

    def _check_ifit1_over_oas2(self, pred_np, ctrl_np, gene_to_idx) -> dict:
        """
        ISG induction hierarchy: IFIT1 FC > OAS2 FC in PBMC IFN-B.
        Source: Rusinova et al. 2013.
        """
        import numpy as np
        eps = 1e-6
        ifit1_idx = gene_to_idx.get("IFIT1")
        oas2_idx = gene_to_idx.get("OAS2")
        if ifit1_idx is None or oas2_idx is None:
            return {"passed": True, "ifit1_fc": 0.0, "oas2_fc": 0.0,
                    "message": "IFIT1 or OAS2 not in gene universe. Skipped.",
                    "skipped": True}
        ifit1_fc = float((pred_np[:, ifit1_idx].mean() + eps) /
                         (ctrl_np[:, ifit1_idx].mean() + eps))
        oas2_fc = float((pred_np[:, oas2_idx].mean() + eps) /
                        (ctrl_np[:, oas2_idx].mean() + eps))
        passed = ifit1_fc > oas2_fc
        return {
            "passed": passed, "ifit1_fc": ifit1_fc, "oas2_fc": oas2_fc,
            "skipped": False,
            "message": (
                f"IFIT1 FC {ifit1_fc:.2f}x {'>' if passed else '<='} "
                f"OAS2 FC {oas2_fc:.2f}x"
                + ("" if passed else " — FAIL: ISG induction hierarchy incorrect")
            ),
        }

    def _check_jak1_upstream_of_ifit1(self, W_top_edges, gene_to_idx) -> dict:
        """
        Causal path: W[JAK1->STAT1] > 0 AND W[STAT1->IFIT1] > 0.
        Both edges must be positive (activating causal relationship).
        """
        jak1_stat1_w = 0.0
        stat1_ifit1_w = 0.0
        for edge in (W_top_edges or []):
            src = edge.get("src_name", "")
            dst = edge.get("dst_name", "")
            w = float(edge.get("weight", 0.0))
            if src == "JAK1" and dst == "STAT1":
                jak1_stat1_w = w
            if src == "STAT1" and dst == "IFIT1":
                stat1_ifit1_w = w
        jak1_ok = jak1_stat1_w > 0
        stat1_ok = stat1_ifit1_w > 0
        passed = jak1_ok and stat1_ok
        edge_desc = (
            f"W[JAK1->STAT1]={jak1_stat1_w:.4f}, "
            f"W[STAT1->IFIT1]={stat1_ifit1_w:.4f}"
        )
        if passed:
            msg = f"{edge_desc} — both positive, causal path confirmed"
        elif not jak1_ok and not stat1_ok:
            msg = f"{edge_desc} — FAIL: both causal edges not yet learned"
        elif not jak1_ok:
            msg = f"{edge_desc} — FAIL: W[JAK1->STAT1] not positive"
        else:
            msg = f"{edge_desc} — FAIL: W[STAT1->IFIT1] not positive"
        return {
            "passed": passed,
            "jak1_stat1_weight": jak1_stat1_w,
            "stat1_ifit1_weight": stat1_ifit1_w,
            "skipped": False,
            "message": msg,
        }
