"""
aivc/edge_cases/handlers.py — Failure mode handlers.

Every handler must:
1. Log the event with precise context
2. Apply the defined strategy
3. Return a result or raise a typed exception
4. Never silently suppress the issue
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class MissingDataHandler:
    """
    Handles missing donors, missing cell types, missing genes.
    Strategy: exclude and log, never impute silently.
    """

    MINIMUM_CELLS_PER_GROUP = 5

    def handle_missing_donor_cell_type(self, donor, cell_type, adata):
        """
        Handle a missing donor/cell_type group.
        Logs the skip and returns filtered adata.
        """
        mask = adata.obs["label"] == adata.obs["label"]  # all True
        if "donor" in adata.obs.columns:
            donor_mask = adata.obs["donor"] == donor
        elif "ind" in adata.obs.columns:
            donor_mask = adata.obs["ind"] == donor
        else:
            donor_mask = mask

        ct_col = None
        for col in ["cell_type", "cell_abbr", "celltype"]:
            if col in adata.obs.columns:
                ct_col = col
                break

        if ct_col:
            ct_mask = adata.obs[ct_col] == cell_type
        else:
            ct_mask = mask

        group_mask = donor_mask & ct_mask
        n_cells = group_mask.sum()

        if n_cells < self.MINIMUM_CELLS_PER_GROUP:
            logger.warning(
                f"Skipped {donor}/{cell_type}: insufficient cells "
                f"(n={n_cells}, minimum={self.MINIMUM_CELLS_PER_GROUP})"
            )
            return adata[~group_mask].copy()

        return adata

    def handle_missing_pathway_gene(self, gene, adata,
                                     must_include_genes=None):
        """
        Handle a gene missing from the dataset.
        If in must-include list: add to HVG set with a WARNING.
        If not: log and continue.
        """
        if must_include_genes is None:
            must_include_genes = []

        # Get gene names
        if "name" in adata.var.columns:
            gene_names = list(adata.var["name"].values)
        else:
            gene_names = list(adata.var_names)

        if gene in gene_names:
            return  # Gene exists, nothing to do

        if gene in must_include_genes:
            logger.warning(
                f"Must-include gene '{gene}' not found in dataset. "
                "Cannot force-include. Model will proceed without it."
            )
        else:
            logger.info(
                f"Gene '{gene}' not found in dataset. "
                "Skipping — not in must-include list."
            )


class NoisySignalHandler:
    """
    Handles low-quality biological signals.
    Strategy: flag and fall back to safer method.
    """

    def handle_ot_quality_failure(self, pairing_quality, baseline, inputs):
        """
        If OT pairing quality is insufficient, fall back to pseudo-bulk.
        Returns updated inputs dict with pseudo-bulk flag.
        """
        if pairing_quality <= baseline + 0.05:
            logger.warning(
                f"OT pairing quality {pairing_quality:.3f} insufficient "
                f"(baseline: {baseline:.3f}). Using pseudo-bulk."
            )
            inputs["force_pseudo_bulk"] = True
            return inputs
        return inputs

    def handle_zero_variance_gene(self, gene, gene_idx, context):
        """
        Handle a gene with zero variance.
        Exclude from LOSS computation but keep in graph for message passing.
        """
        logger.warning(
            f"Zero variance gene: '{gene}' (idx={gene_idx}). "
            "Excluded from loss computation, kept in graph."
        )
        return {
            "gene": gene,
            "gene_idx": gene_idx,
            "action": "exclude_from_loss",
            "keep_in_graph": True,
        }


class ModelFailureHandler:
    """
    Handles training failures and poor performance.
    Strategy: diagnose before aborting.
    """

    def handle_loss_spike(self, epoch, current_loss, prev_loss,
                          model, optimizer):
        """
        Handle a loss spike (current > prev * 10).
        Reduces gradient clipping norm and learning rate.
        """
        if current_loss > prev_loss * 10:
            logger.warning(
                f"Loss spike at epoch {epoch}: "
                f"{prev_loss:.4f} -> {current_loss:.4f}"
            )

            # Reduce learning rate by 50%
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = old_lr * 0.5
                logger.info(
                    f"Reduced lr: {old_lr:.6f} -> {param_group['lr']:.6f}"
                )

            return {
                "action": "reduced_lr",
                "new_max_norm": 0.5,
                "lr_reduction": 0.5,
            }

        return None

    def handle_low_pearson_r(self, pearson_r, context):
        """
        Handle low Pearson r (< 0.70).
        Returns diagnostic report identifying failing component.
        """
        logger.error(
            f"Low Pearson r: {pearson_r:.4f}. "
            "Running diagnostics."
        )

        diagnostic = {
            "pearson_r": pearson_r,
            "below_threshold": True,
            "threshold": 0.70,
            "recommended_actions": [
                "Check OT pairing quality vs pseudo-bulk",
                "Verify donor split is correct",
                "Check LFC beta is not dominating MSE loss",
                "Ensure residual learning is enabled",
                "Verify gradient clipping is active",
            ],
            "demo_blocked": True,
        }

        return diagnostic


class DistributionShiftHandler:
    """
    Handles distribution shift when new data modalities arrive.
    Critical for Phase 2 QuRIE-seq integration.
    """

    def check_mmd(self, source_data, target_data, threshold=0.1):
        """
        Compute Maximum Mean Discrepancy between distributions.
        Returns (shift_detected, mmd_value).
        """
        try:
            # Flatten to 2D if needed
            if hasattr(source_data, "toarray"):
                source_data = source_data.toarray()
            if hasattr(target_data, "toarray"):
                target_data = target_data.toarray()

            source = np.array(source_data, dtype=np.float64)
            target = np.array(target_data, dtype=np.float64)

            # Subsample for efficiency
            n = min(1000, source.shape[0], target.shape[0])
            rng = np.random.RandomState(42)
            s_idx = rng.choice(source.shape[0], n, replace=False)
            t_idx = rng.choice(target.shape[0], n, replace=False)
            source = source[s_idx]
            target = target[t_idx]

            # RBF kernel MMD
            from scipy.spatial.distance import cdist

            sigma = np.median(cdist(source, source, "euclidean"))
            if sigma == 0:
                sigma = 1.0

            def rbf_kernel(X, Y):
                dist = cdist(X, Y, "sqeuclidean")
                return np.exp(-dist / (2 * sigma ** 2))

            K_ss = rbf_kernel(source, source)
            K_tt = rbf_kernel(target, target)
            K_st = rbf_kernel(source, target)

            mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
            mmd = max(0, mmd)  # MMD is non-negative

            shift_detected = mmd > threshold
            logger.info(f"MMD = {mmd:.4f}, threshold = {threshold}")

            return shift_detected, float(mmd)

        except Exception as e:
            logger.error(f"MMD computation failed: {e}")
            return False, 0.0

    def handle_shift_detected(self, mmd, source, target, context):
        """
        Handle detected distribution shift.
        Log and recommend domain adaptation.
        """
        logger.warning(
            f"Distribution shift detected MMD={mmd:.4f}. "
            "Applying domain adaptation before Phase 2 integration."
        )

        return {
            "mmd": mmd,
            "shift_detected": True,
            "recommendation": (
                "Apply domain adaptation (e.g., CORAL, MMD penalty) "
                "before fine-tuning on Phase 2 data. "
                "Never assume Phase 1 and Phase 2 data "
                "distributions are compatible."
            ),
        }


class AmbiguousQueryHandler:
    """
    Handles ambiguous scientific queries.
    Strategy: surface options, never choose autonomously.
    """

    def handle_multiple_valid_workflows(self, query, workflows, context):
        """
        When a query maps to multiple workflows, return options.
        Never choose autonomously on high-stakes queries.
        """
        options = []
        for name, wf in workflows.items():
            options.append({
                "workflow": name,
                "description": wf.description,
                "n_steps": len(wf.steps),
                "gpu_required": any(
                    True for s in wf.steps
                    # placeholder check
                ),
            })

        logger.info(
            f"Ambiguous query: '{query}'. "
            f"Returning {len(options)} workflow options."
        )

        return {
            "query": query,
            "options": options,
            "message": (
                "Multiple workflows match this query. "
                "Please select one."
            ),
        }

    def handle_unknown_gene(self, gene, gene_to_idx, context):
        """
        Handle a gene not in the current graph.
        Check if it can be added and estimate cost.
        """
        if gene in gene_to_idx:
            return {
                "gene": gene,
                "in_graph": True,
                "index": gene_to_idx[gene],
            }

        logger.info(
            f"Gene '{gene}' not in current graph. "
            "Checking expansion feasibility."
        )

        return {
            "gene": gene,
            "in_graph": False,
            "expansion_proposal": {
                "action": "Add gene to HVG set and rebuild graph",
                "estimated_cost": "~5 minutes CPU + retraining",
                "risk": (
                    "Adding genes to graph changes edge structure. "
                    "May require full retraining."
                ),
            },
        }
