"""
Multi-dataset evaluation suite for AIVC v1.1.

Reports per-dataset metrics and Neumann W statistics.
Primary benchmark: Kang 2018 Pearson r must NEVER fall below 0.873.
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("aivc.skills")


def compute_pearson_r(predicted, actual):
    """Compute mean Pearson r across pairs."""
    rs = []
    for i in range(predicted.shape[0]):
        p, a = predicted[i], actual[i]
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            rs.append(0.0)
            continue
        r = np.corrcoef(p, a)[0, 1]
        rs.append(0.0 if np.isnan(r) else r)
    return float(np.mean(rs)), float(np.std(rs))


def compute_fold_change(pred_np, actual_np, ctrl_np, gene_idx_map, gene_list):
    """Compute fold change metrics for a gene list."""
    eps = 1e-6
    results = {}
    within_3x = 0
    within_10x = 0

    for g in gene_list:
        if g not in gene_idx_map:
            continue
        idx = gene_idx_map[g]
        c = ctrl_np[:, idx].mean()
        p = pred_np[:, idx].mean()
        a = actual_np[:, idx].mean()
        pfc = (p + eps) / (c + eps)
        afc = (a + eps) / (c + eps)
        ratio = max(pfc, afc) / max(min(pfc, afc), eps) if min(pfc, afc) > 0.01 else float("inf")
        results[g] = {"pred_fc": float(pfc), "actual_fc": float(afc), "ratio": float(ratio)}
        if ratio <= 3.0:
            within_3x += 1
        if ratio <= 10.0:
            within_10x += 1

    return results, within_3x, within_10x


def evaluate_neumann_W(W_param, edge_index, gene_to_idx) -> dict:
    """
    Report Neumann W statistics.

    Returns:
        Dict with JAK-STAT edge weights, sparsity, top-10 edges.
    """
    import torch

    W = W_param.detach().cpu()
    edge_src = edge_index[0].cpu()
    edge_dst = edge_index[1].cpu()

    # Build reverse lookup
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}

    # Sparsity
    w_sparsity = (W.abs() > 0.01).float().mean().item()

    # JAK-STAT specific edges
    jakstat_edges = {}
    jakstat_pairs = [
        ("JAK1", "STAT1"), ("JAK2", "STAT1"),
        ("STAT1", "IFIT1"), ("STAT2", "IFIT1"),
        ("STAT1", "MX1"), ("STAT1", "ISG15"),
    ]

    for src_gene, dst_gene in jakstat_pairs:
        src_idx = gene_to_idx.get(src_gene)
        dst_idx = gene_to_idx.get(dst_gene)
        if src_idx is not None and dst_idx is not None:
            mask = (edge_src == src_idx) & (edge_dst == dst_idx)
            weight = W[mask].sum().item() if mask.any() else 0.0
            jakstat_edges[f"{src_gene}->{dst_gene}"] = weight

    # Top-10 highest weight edges
    top_k = min(10, len(W))
    top_indices = torch.topk(W.abs(), top_k).indices
    top_edges = []
    for idx in top_indices:
        src = idx_to_gene.get(edge_src[idx].item(), f"gene_{edge_src[idx].item()}")
        dst = idx_to_gene.get(edge_dst[idx].item(), f"gene_{edge_dst[idx].item()}")
        top_edges.append({"src": src, "dst": dst, "weight": W[idx].item()})

    return {
        "w_sparsity": w_sparsity,
        "jakstat_edges": jakstat_edges,
        "top_10_edges": top_edges,
        "w_mean_abs": float(W.abs().mean()),
        "w_max_abs": float(W.abs().max()),
        "n_active_edges": int((W.abs() > 0.01).sum()),
        "n_total_edges": len(W),
    }


def evaluate_multi_dataset(
    predictions_by_dataset: dict,
    gene_to_idx: dict,
    W_param=None,
    edge_index=None,
) -> dict:
    """
    Full multi-dataset evaluation.

    Args:
        predictions_by_dataset: Dict of dataset_name -> {
            "pred": np.ndarray, "actual": np.ndarray, "ctrl": np.ndarray,
            "cell_types": list (optional)
        }
        gene_to_idx: Gene name -> index mapping.
        W_param: Neumann W parameter (optional, for W stats).
        edge_index: Graph edge index (required if W_param provided).

    Returns:
        Dict with per-dataset metrics and W statistics.
    """
    jakstat_genes = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IRF1", "MX1", "MX2", "ISG15",
        "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    results = {}

    for ds_name, ds_data in predictions_by_dataset.items():
        pred = ds_data["pred"]
        actual = ds_data["actual"]
        ctrl = ds_data.get("ctrl", actual)

        # Pearson r
        r_mean, r_std = compute_pearson_r(pred, actual)

        # Fold changes
        fc_results, within_3x, within_10x = compute_fold_change(
            pred, actual, ctrl, gene_to_idx, jakstat_genes
        )

        ifit1 = fc_results.get("IFIT1", {})

        ds_result = {
            "pearson_r": r_mean,
            "pearson_std": r_std,
            "jakstat_within_3x": within_3x,
            "jakstat_within_10x": within_10x,
            "ifit1_pred_fc": ifit1.get("pred_fc", 0),
            "ifit1_actual_fc": ifit1.get("actual_fc", 0),
            "fold_changes": fc_results,
        }

        # Cell-type stratified r
        if "cell_types" in ds_data and ds_data["cell_types"]:
            ct_r = {}
            cts = ds_data["cell_types"]
            for ct in sorted(set(cts)):
                mask = [i for i, c in enumerate(cts) if c == ct]
                if len(mask) >= 2:
                    r, _ = compute_pearson_r(pred[mask], actual[mask])
                    ct_r[ct] = r
            ds_result["cell_type_r"] = ct_r
            ds_result["cd14_monocyte_r"] = ct_r.get("CD14+ Monocytes", 0)

        results[ds_name] = ds_result

    # Neumann W stats
    if W_param is not None and edge_index is not None:
        results["neumann_W_stats"] = evaluate_neumann_W(W_param, edge_index, gene_to_idx)

    # Regression check
    kang = results.get("kang_2018", {})
    if kang:
        kang_r = kang.get("pearson_r", 0)
        if kang_r < 0.873:
            logger.error(
                f"REGRESSION: Kang 2018 Pearson r = {kang_r:.4f} < 0.873. "
                "Stage advance BLOCKED."
            )
        results["regression_check"] = {
            "kang_r": kang_r,
            "passed": kang_r >= 0.873,
            "baseline": 0.873,
        }

    return results
