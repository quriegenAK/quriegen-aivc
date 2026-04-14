"""
Zero-shot perturbation response evaluation.
Tests the model on a perturbation it was never trained on.

Usage:
  python scripts/evaluate_zero_shot.py \
    --checkpoint models/v1.1/model_v11_best.pt \
    --dataset     data/dixit2016_pbmc.h5ad \
    --perturbation IFNG \
    --output      results/zero_shot_v1.1.json

Output JSON:
  {
    "pearson_r":           float,
    "pearson_r_baseline":  0.873,
    "jakstat_3x":          int,
    "ifit1_pred_fc":       float,
    "generalisation_delta": float,
    "is_overfit":          bool,
    "verdict":             str
  }
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


SEED = 42
BASELINE_R = 0.873
N_GENES = 3010


def load_checkpoint(checkpoint_path):
    """Load trained model from checkpoint. Returns (model, gene_names, edge_index) or exits."""
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("  Run the v1.1 sweep first: python train_v11.py")
        sys.exit(1)

    from perturbation_model import PerturbationPredictor, CellTypeEmbedding
    from aivc.skills.neumann_propagation import NeumannPropagation

    model = PerturbationPredictor(
        n_genes=N_GENES, num_perturbations=2, feature_dim=64,
        hidden1_dim=64, hidden2_dim=32, num_head1=3, num_head2=2,
        decoder_hidden=256,
    )
    model.cell_type_embedding = CellTypeEmbedding(
        num_cell_types=20, embedding_dim=model.feature_dim,
    )

    # Load edge index for Neumann
    edge_index = torch.randint(0, N_GENES, (2, 100))  # fallback
    import pandas as pd
    for path in ["data/edge_list_fixed.csv", "data/edge_list.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "src_idx" in df.columns and "dst_idx" in df.columns:
                    edge_index = torch.tensor(
                        df[["src_idx", "dst_idx"]].values.T, dtype=torch.long
                    )
                    break
            except Exception:
                continue

    try:
        sd = torch.load(checkpoint_path, map_location="cpu")
        # Check if checkpoint includes Neumann weights
        has_neumann = any(k.startswith("neumann.") for k in sd.keys())
        if has_neumann:
            model.neumann = NeumannPropagation(
                n_genes=N_GENES, edge_index=edge_index, K=3, lambda_l1=0.001,
            )
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # Try loading without Neumann (v1.0 checkpoints)
        try:
            model.load_state_dict(sd, strict=False)
            print(f"  [INFO] Loaded checkpoint without Neumann weights (v1.0 model)")
        except Exception as e2:
            print(f"[ERROR] Failed to load checkpoint: {e2}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)

    model.eval()

    # Load gene names
    gene_names = [f"GENE_{i}" for i in range(N_GENES)]
    if os.path.exists("data/gene_names.txt"):
        with open("data/gene_names.txt") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == N_GENES:
            gene_names = names

    return model, gene_names, edge_index


def align_gene_universe(train_genes, eval_genes):
    """Find intersection of gene universes. Returns (shared_genes, train_idx, eval_idx)."""
    train_set = {g: i for i, g in enumerate(train_genes)}
    shared = []
    train_idx = []
    eval_idx = []
    for i, g in enumerate(eval_genes):
        if g in train_set:
            shared.append(g)
            train_idx.append(train_set[g])
            eval_idx.append(i)
    return shared, np.array(train_idx), np.array(eval_idx)


def detect_condition_column(adata):
    """Auto-detect the condition column and control label in an AnnData.

    Returns (condition_col, ctrl_label) by inspecting obs columns.
    Raises ValueError with helpful message if not found.
    """
    candidates = ['condition', 'perturbation', 'guide_identity',
                  'treatment', 'stimulation', 'perturbation_name',
                  'stim', 'label']
    ctrl_labels = ['ctrl', 'control', 'unperturbed', 'NT',
                   'non-targeting', 'CTRL', 'Control']

    for col in candidates:
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).unique()
            for label in ctrl_labels:
                if label in vals:
                    return col, label

    # Last resort: scan all string/categorical columns
    for col in adata.obs.columns:
        try:
            vals = adata.obs[col].astype(str).unique()
        except Exception:
            continue
        for label in ctrl_labels:
            if label in vals:
                return col, label

    raise ValueError(
        f"Could not detect condition column in obs. "
        f"Available columns: {list(adata.obs.columns)}. "
        f"Expected one of {candidates} with a control label from {ctrl_labels}."
    )


def compute_verdict(pearson_r, generalisation_delta):
    """Determine zero-shot verdict."""
    if generalisation_delta < -0.20:
        return "MEMORISED"
    if pearson_r >= 0.70:
        return "GENERALISES"
    if pearson_r >= 0.50:
        return "PARTIAL"
    return "MEMORISED"


def compute_dryrun_verdict(dry_run_r):
    """Determine dry-run verdict (held-out donor, same perturbation)."""
    if dry_run_r >= 0.87:
        return "CONSISTENT"
    if dry_run_r >= 0.80:
        return "ACCEPTABLE"
    return "DEGRADED"


def evaluate_dry_run(model, gene_names, edge_index, kang_h5ad_path):
    """Run held-out donor validation on Kang 2018 test-set cells.

    This is NOT zero-shot — same perturbation (IFN-β), unseen donors.
    It validates the checkpoint loads correctly and produces expected r.
    """
    import h5py
    from scipy.stats import pearsonr

    with h5py.File(kang_h5ad_path, "r") as f:
        # Read donor and label info
        rep_cats = f["obs"]["replicate"]["categories"][:]
        rep_codes = f["obs"]["replicate"]["codes"][:]
        label_cats = f["obs"]["label"]["categories"][:]
        label_codes = f["obs"]["label"]["codes"][:]

        donors_all = [rep_cats[c].decode() for c in rep_codes]
        labels_all = [label_cats[c].decode() for c in label_codes]

        # Read gene names from var
        h5_gene_names = [g.decode() for g in f["var"]["name"][:]]

        # Read X matrix (dense or sparse)
        X = f["X"]
        if hasattr(X, "toarray"):
            X_dense = X[:]
        elif "data" in X:
            # CSR sparse
            import scipy.sparse as sp
            data = X["data"][:]
            indices = X["indices"][:]
            indptr = X["indptr"][:]
            shape = tuple(f["X"].attrs["shape"])
            X_dense = sp.csr_matrix((data, indices, indptr), shape=shape).toarray()
        else:
            X_dense = X[:]

    # Determine test donors using the same 60/20/20 split as train_v11.py
    unique_donors = sorted(set(donors_all))
    n_donors = len(unique_donors)
    n_train = max(1, int(n_donors * 0.6))
    n_val = max(1, int(n_donors * 0.2))
    test_donors = set(unique_donors[n_train + n_val:])

    if not test_donors:
        raise ValueError(
            "No test donors found. The 60/20/20 split produced an empty test set. "
            f"Donors: {unique_donors}"
        )

    print(f"  All donors: {unique_donors}")
    print(f"  Test donors: {sorted(test_donors)}")

    # Filter to test-set cells
    test_ctrl_rows = []
    test_stim_rows = []
    for i in range(len(donors_all)):
        if donors_all[i] in test_donors:
            if labels_all[i] == "ctrl":
                test_ctrl_rows.append(i)
            else:
                test_stim_rows.append(i)

    if not test_ctrl_rows or not test_stim_rows:
        raise ValueError(
            f"Test donors {test_donors} have no ctrl or stim cells. "
            f"ctrl: {len(test_ctrl_rows)}, stim: {len(test_stim_rows)}"
        )

    print(f"  Test cells: ctrl={len(test_ctrl_rows)}, stim={len(test_stim_rows)}")

    # Compute ctrl mean and stim mean from test-set cells
    ctrl_mean = X_dense[test_ctrl_rows].mean(axis=0).astype(np.float32)
    stim_mean = X_dense[test_stim_rows].mean(axis=0).astype(np.float32)

    # For dry run, same dataset so genes are 1:1. If gene_names are
    # fallback GENE_0..GENE_N, we use h5ad genes directly as ground truth.
    if len(h5_gene_names) == len(gene_names):
        # Same gene count — use positional alignment (same dataset)
        n_shared = len(gene_names)
        all_idx = np.arange(n_shared)
        ctrl_aligned = ctrl_mean.astype(np.float32)
        stim_aligned = stim_mean.astype(np.float32)
        # Update gene_names to real names for JAK-STAT lookup
        gene_names = h5_gene_names
        print(f"  Gene alignment: {n_shared} / {n_shared} (same dataset, positional)")
    else:
        shared, train_idx, eval_idx = align_gene_universe(gene_names, h5_gene_names)
        print(f"  Gene alignment: {len(shared)} / {len(gene_names)}")
        ctrl_aligned = np.zeros(len(gene_names), dtype=np.float32)
        ctrl_aligned[train_idx] = ctrl_mean[eval_idx]
        stim_aligned = np.zeros(len(gene_names), dtype=np.float32)
        stim_aligned[train_idx] = stim_mean[eval_idx]
        n_shared = len(shared)

    # Run model
    ctrl_tensor = torch.tensor(ctrl_aligned, dtype=torch.float32)
    pert_id = torch.tensor([1])

    model.eval()
    with torch.no_grad():
        pred_delta = model(ctrl_tensor, edge_index, pert_id)
        pred_full = (ctrl_tensor + pred_delta).clamp(min=0.0)

    pred_np = pred_full.detach().cpu().float().numpy()

    # Pearson r on all genes
    r, _ = pearsonr(pred_np, stim_aligned)
    r = float(r) if not np.isnan(r) else 0.0

    verdict = compute_dryrun_verdict(r)

    # JAK-STAT recovery
    gene_to_idx_local = {g: i for i, g in enumerate(gene_names)}
    jakstat_genes = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IRF1", "MX1", "MX2", "ISG15",
        "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]
    eps = 1e-6
    within_3x = 0
    ifit1_pred_fc = 0.0
    for g in jakstat_genes:
        if g not in gene_to_idx_local:
            continue
        idx = gene_to_idx_local[g]
        c = ctrl_aligned[idx]
        p = pred_np[idx]
        a = stim_aligned[idx]
        pfc = (p + eps) / (c + eps)
        afc = (a + eps) / (c + eps)
        ratio = max(pfc, afc) / max(min(pfc, afc), eps) if min(pfc, afc) > 0.01 else float("inf")
        if ratio <= 3.0:
            within_3x += 1
        if g == "IFIT1":
            ifit1_pred_fc = float(pfc)

    result = {
        "mode": "dry_run",
        "dry_run_r": round(r, 4),
        "pearson_r_baseline": BASELINE_R,
        "jakstat_3x": within_3x,
        "ifit1_pred_fc": round(ifit1_pred_fc, 2),
        "test_donors": sorted(test_donors),
        "n_ctrl_cells": len(test_ctrl_rows),
        "n_stim_cells": len(test_stim_rows),
        "gene_overlap": n_shared,
        "verdict": verdict,
    }

    return result


def evaluate_zero_shot(model, gene_names, edge_index, dataset_path, perturbation):
    """Run zero-shot evaluation on an unseen perturbation dataset."""
    import anndata as ad

    adata = ad.read_h5ad(dataset_path)

    # Determine gene names in eval dataset
    if "name" in adata.var.columns:
        eval_genes = adata.var["name"].tolist()
    else:
        eval_genes = adata.var_names.tolist()

    # Align gene universes
    shared_genes, train_idx, eval_idx = align_gene_universe(gene_names, eval_genes)
    n_overlap = len(shared_genes)
    print(f"  Gene overlap: {n_overlap} / {len(gene_names)} (train) x {len(eval_genes)} (eval)")

    if n_overlap < 500:
        print(f"  [WARN] Gene overlap < 500. Results may be unreliable.")

    # Auto-detect condition column and control label
    try:
        cond_col, ctrl_label = detect_condition_column(adata)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"  Condition column: '{cond_col}', control label: '{ctrl_label}'")
    ctrl_mask = adata.obs[cond_col].astype(str) == ctrl_label
    stim_mask = ~ctrl_mask
    print(f"  Ctrl cells: {ctrl_mask.sum()}, Stim/perturbed cells: {stim_mask.sum()}")

    X_ctrl = adata[ctrl_mask].X
    X_stim = adata[stim_mask].X

    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
    if hasattr(X_stim, "toarray"):
        X_stim = X_stim.toarray()

    # Compute means over shared genes
    ctrl_mean = X_ctrl[:, eval_idx].mean(axis=0)
    stim_mean = X_stim[:, eval_idx].mean(axis=0)

    # Build aligned input for model (fill non-shared genes with zeros)
    ctrl_full = np.zeros(len(gene_names), dtype=np.float32)
    ctrl_full[train_idx] = ctrl_mean

    ctrl_tensor = torch.tensor(ctrl_full, dtype=torch.float32)
    pert_id = torch.tensor([1])

    with torch.no_grad():
        pred_delta = model(ctrl_tensor, edge_index, pert_id)
        pred_full = (ctrl_tensor + pred_delta).clamp(min=0.0)

    pred_shared = pred_full.detach().cpu().float().numpy()[train_idx]
    actual_shared = stim_mean

    # Compute Pearson r
    from scipy.stats import pearsonr
    r, _ = pearsonr(pred_shared, actual_shared)
    r = float(r) if not np.isnan(r) else 0.0

    generalisation_delta = r - BASELINE_R
    verdict = compute_verdict(r, generalisation_delta)

    # JAK-STAT recovery (simplified)
    gene_to_idx = {g: i for i, g in enumerate(shared_genes)}
    jakstat_genes = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IRF1", "MX1", "MX2", "ISG15",
        "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]
    eps = 1e-6
    within_3x = 0
    ifit1_pred_fc = 0.0
    for g in jakstat_genes:
        if g not in gene_to_idx:
            continue
        idx = gene_to_idx[g]
        c = ctrl_mean[idx]
        p = pred_shared[idx]
        a = actual_shared[idx]
        pfc = (p + eps) / (c + eps)
        afc = (a + eps) / (c + eps)
        ratio = max(pfc, afc) / max(min(pfc, afc), eps) if min(pfc, afc) > 0.01 else float("inf")
        if ratio <= 3.0:
            within_3x += 1
        if g == "IFIT1":
            ifit1_pred_fc = float(pfc)

    result = {
        "pearson_r": round(r, 4),
        "pearson_r_baseline": BASELINE_R,
        "jakstat_3x": within_3x,
        "ifit1_pred_fc": round(ifit1_pred_fc, 2),
        "generalisation_delta": round(generalisation_delta, 4),
        "is_overfit": generalisation_delta < -0.15,
        "verdict": verdict,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Zero-shot perturbation response evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default=None, help="Path to evaluation h5ad dataset")
    parser.add_argument("--perturbation", default="IFNG", help="Perturbation name")
    parser.add_argument("--output", default="results/zero_shot_v1.1.json", help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on held-out Kang 2018 test-set donors (not zero-shot).")
    parser.add_argument("--n-cells", type=int, default=500,
                        help="Ignored — dry run uses all test-set cells.")
    args = parser.parse_args()

    if not args.dry_run and args.dataset is None:
        parser.error("--dataset is required unless --dry-run is specified.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("HELD-OUT DONOR VALIDATION (DRY RUN — NOT ZERO-SHOT)")
    else:
        print("ZERO-SHOT PERTURBATION RESPONSE EVALUATION")
    print(f"{'='*60}")
    print(f"  Checkpoint:    {args.checkpoint}")

    model, gene_names, edge_index = load_checkpoint(args.checkpoint)

    if args.dry_run:
        kang_path = "data/kang2018_pbmc_fixed.h5ad"
        if not os.path.exists(kang_path):
            print(f"[ERROR] Kang 2018 h5ad not found: {kang_path}")
            sys.exit(1)
        print(f"  Dataset:       {kang_path} (held-out test donors)")
        result = evaluate_dry_run(model, gene_names, edge_index, kang_path)

        # Write output
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Output: {args.output}")

        print(f"\n  DRY RUN RESULTS (held-out donors, same perturbation):")
        print(f"    dry_run_r:        {result['dry_run_r']}")
        print(f"    Baseline r:       {result['pearson_r_baseline']}")
        print(f"    Test donors:      {result['test_donors']}")
        print(f"    Test cells:       ctrl={result['n_ctrl_cells']}, stim={result['n_stim_cells']}")
        print(f"    Gene overlap:     {result['gene_overlap']}")
        print(f"    JAK-STAT 3x:      {result['jakstat_3x']}/15")
        print(f"    IFIT1 pred FC:    {result['ifit1_pred_fc']}x")
        print(f"    VERDICT:          {result['verdict']}")
        print(f"\n  NOTE: This is NOT zero-shot. Same perturbation (IFN-B), unseen donors.")
        print(f"  Real zero-shot requires a different perturbation dataset.")
    else:
        print(f"  Dataset:       {args.dataset}")
        print(f"  Perturbation:  {args.perturbation}")
        result = evaluate_zero_shot(model, gene_names, edge_index, args.dataset, args.perturbation)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Output: {args.output}")

        print(f"\n  ZERO-SHOT RESULTS:")
        print(f"    Pearson r:        {result['pearson_r']}")
        print(f"    Baseline r:       {result['pearson_r_baseline']}")
        print(f"    Delta:            {result['generalisation_delta']:+.4f}")
        print(f"    JAK-STAT 3x:      {result['jakstat_3x']}/15")
        print(f"    IFIT1 pred FC:    {result['ifit1_pred_fc']}x")
        print(f"    Overfit:          {result['is_overfit']}")
        print(f"    VERDICT:          {result['verdict']}")

    print()


if __name__ == "__main__":
    main()
