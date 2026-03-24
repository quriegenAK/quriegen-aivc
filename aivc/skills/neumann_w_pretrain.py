"""
Neumann W matrix pre-training from Replogle 2022 causal directions.

Runs BEFORE PBMC training to give W a causally informed initialisation.
Uses 9,867 CRISPRi knockdowns to learn cascade structure:
  - If knocking down JAK1 abolishes STAT1/IFIT1 activation ->
    W[JAK1, STAT1] > 0 and W[JAK1, IFIT1] > 0 (causal evidence)
  - If knocking down STAT1 abolishes IFIT1 but not JAK1 ->
    W[STAT1, IFIT1] > 0, W[STAT1, JAK1] ~ 0 (ordering confirmed)

After pre-training: W[JAK1, STAT1] > W[STAT1, JAK1] (direction verified).
"""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("aivc.skills")

SEED = 42


def pretrain_W_from_replogle(
    W: nn.Parameter,
    edge_index: torch.Tensor,
    replogle_direction_matrix: pd.DataFrame,
    gene_to_idx: dict,
    n_epochs: int = 5,
    lr: float = 0.001,
    top_k_genes: int = 200,
) -> nn.Parameter:
    """
    Pre-train Neumann W from Replogle causal direction evidence.

    For each gene knockdown:
      1. Set d_p[knockdown_gene] = -1 (gene suppressed)
      2. Run Neumann propagation: effect = sum(W^k * d_p) for k=1..3
      3. Loss: MSE between predicted effect and observed Replogle effect
         for top_k_genes most variable affected genes
      4. Gradient update on W only

    Args:
        W: nn.Parameter — the Neumann edge weight vector.
        edge_index: (2, n_edges) — graph structure.
        replogle_direction_matrix: DataFrame (knockdown x gene) with +1/-1 values.
        gene_to_idx: Dict mapping gene names to indices.
        n_epochs: Number of pre-training epochs. Default 5.
        lr: Learning rate for W. Default 0.001.
        top_k_genes: Number of most-affected genes per knockdown. Default 200.

    Returns:
        Updated W parameter.
    """
    if replogle_direction_matrix is None or len(replogle_direction_matrix) == 0:
        logger.warning("Empty Replogle direction matrix. Skipping W pre-training.")
        return W

    # ── VALIDATE NO JAK-STAT GENES IN PRETRAINING DATA ──
    # Runtime guard: if the direction matrix somehow contains JAK-STAT
    # genes (should not happen after housekeeping filter), remove them.
    from aivc.data.housekeeping_genes import get_blocked_jakstat_genes
    blocked_jakstat = get_blocked_jakstat_genes()
    df_genes = set(replogle_direction_matrix.index.tolist())
    jakstat_present = df_genes & blocked_jakstat
    if jakstat_present:
        logger.warning(
            f"JAK-STAT genes found in W pretraining data: {jakstat_present}. "
            f"These should have been filtered by filter_ko_genes_for_w_pretrain(). "
            f"Removing them now."
        )
        safe_index = [
            g for g in replogle_direction_matrix.index
            if g not in blocked_jakstat
        ]
        replogle_direction_matrix = replogle_direction_matrix.loc[safe_index]
        logger.info(f"After JAK-STAT removal: {len(replogle_direction_matrix)} KO genes remain.")

        if len(replogle_direction_matrix) == 0:
            logger.warning("No KO genes remain after JAK-STAT removal. Skipping W pre-training.")
            return W
    # ─────────────────────────────────────────────────────────────

    device = W.device
    edge_src = edge_index[0].to(device)
    edge_dst = edge_index[1].to(device)
    n_genes = max(edge_src.max().item(), edge_dst.max().item()) + 1

    # Map Replogle genes to our gene indices
    replogle_genes = replogle_direction_matrix.index.tolist()
    target_genes = replogle_direction_matrix.columns.tolist()

    # Build training examples
    examples = []
    for ko_gene in replogle_genes:
        if ko_gene not in gene_to_idx:
            continue
        ko_idx = gene_to_idx[ko_gene]

        directions = replogle_direction_matrix.loc[ko_gene]
        affected_genes = []
        for g in target_genes:
            if g in gene_to_idx and abs(directions.get(g, 0)) > 0:
                affected_genes.append((gene_to_idx[g], float(directions[g])))

        if len(affected_genes) >= 5:
            # Take top_k by absolute direction
            affected_genes.sort(key=lambda x: abs(x[1]), reverse=True)
            affected_genes = affected_genes[:top_k_genes]
            examples.append((ko_idx, affected_genes))

    if not examples:
        logger.warning("No valid Replogle examples mapped to gene universe.")
        return W

    logger.info(f"W pre-training: {len(examples)} knockdown examples, {n_epochs} epochs")

    optimizer = torch.optim.Adam([W], lr=lr)
    K = 3  # Neumann steps

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_examples = 0

        for ko_idx, affected in examples:
            # Direct effect: knockout gene suppressed
            d_p = torch.zeros(1, n_genes, device=device)
            d_p[0, ko_idx] = -1.0

            # Neumann propagation
            propagated = d_p.clone()
            for k in range(K):
                src_vals = propagated[:, edge_src]
                weighted = src_vals * W.unsqueeze(0)
                result = torch.zeros_like(propagated)
                result.scatter_add_(1, edge_dst.unsqueeze(0), weighted)
                propagated = propagated + result

            # Loss: predicted effect vs observed direction for affected genes
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for gene_idx, direction in affected:
                if gene_idx < n_genes:
                    predicted = propagated[0, gene_idx]
                    target = torch.tensor(direction, device=device)
                    loss = loss + (predicted - target) ** 2

            loss = loss / max(len(affected), 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_examples += 1

        avg_loss = epoch_loss / max(n_examples, 1)
        logger.info(f"  W pre-train epoch {epoch}: loss={avg_loss:.6f}")

    # Validate JAK-STAT direction
    _validate_jak_stat_direction(W, edge_index, gene_to_idx, device)

    return W


def _validate_jak_stat_direction(W, edge_index, gene_to_idx, device):
    """Check W[JAK1, STAT1] > W[STAT1, JAK1] after pre-training."""
    edge_src = edge_index[0]
    edge_dst = edge_index[1]

    jak1_idx = gene_to_idx.get("JAK1")
    stat1_idx = gene_to_idx.get("STAT1")

    if jak1_idx is None or stat1_idx is None:
        logger.info("  JAK1 or STAT1 not in gene universe. Skipping direction check.")
        return

    # Find edge JAK1 -> STAT1
    jak_stat_mask = (edge_src == jak1_idx) & (edge_dst == stat1_idx)
    stat_jak_mask = (edge_src == stat1_idx) & (edge_dst == jak1_idx)

    w_jak_stat = W[jak_stat_mask].sum().item() if jak_stat_mask.any() else 0
    w_stat_jak = W[stat_jak_mask].sum().item() if stat_jak_mask.any() else 0

    if w_jak_stat > w_stat_jak:
        logger.info(
            f"  JAK-STAT direction CORRECT: W[JAK1->STAT1]={w_jak_stat:.4f} > "
            f"W[STAT1->JAK1]={w_stat_jak:.4f}"
        )
    else:
        logger.warning(
            f"  JAK-STAT direction INCORRECT: W[JAK1->STAT1]={w_jak_stat:.4f} <= "
            f"W[STAT1->JAK1]={w_stat_jak:.4f}. "
            "More pre-training epochs may be needed."
        )
