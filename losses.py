"""
losses.py — Loss function library for AIVC perturbation response prediction.

Contains three loss functions:
    1. log_fold_change_loss: penalises errors in log-fold-change space
    2. cosine_loss: penalises wrong expression profile shape
    3. combined_loss: weighted combination of MSE + LFC + cosine

The LFC loss directly addresses the Week 2 fold-change compression problem
where MSE alone treated 238x IFIT1 as an outlier to be smoothed.

All loss functions are documented, tested, and importable.
"""
import torch
import torch.nn.functional as F


def log_fold_change_loss(
    predicted: torch.Tensor,
    actual_stim: torch.Tensor,
    actual_ctrl: torch.Tensor,
    epsilon: float = 1e-6,
    high_responder_weight: float = 3.0,
    high_responder_threshold: float = 2.0,
) -> torch.Tensor:
    """
    Loss on log2 fold change: log2(stim/ctrl).

    This loss treats a 238x fold change as equally important
    to a 1.2x fold change. MSE alone treats them very differently.

    Args:
        predicted:               (batch, n_genes) — predicted stim expression.
        actual_stim:             (batch, n_genes) — actual stim expression.
        actual_ctrl:             (batch, n_genes) — ctrl expression (input).
        epsilon:                 floor to prevent log(0), default 1e-6.
        high_responder_weight:   multiplier for genes with actual FC > threshold.
        high_responder_threshold: FC threshold for high-responder weighting.

    Returns:
        Scalar loss tensor.

    Why log2 not log:
        log2 makes fold changes human-interpretable.
        A loss of 1.0 = wrong by exactly 1 doubling step.
        This is the unit biologists use when reading results.
    """
    # Expression mask: only compute LFC where at least one of ctrl/stim
    # is meaningfully expressed. Single-cell data has many exact zeros;
    # log2(eps/eps) = 0 but with predictions != 0 this creates noise.
    expressed_mask = (actual_ctrl > epsilon) | (actual_stim > epsilon)

    # Clamp inputs to non-negative before log2 — model predictions and
    # expression values should never be negative, but gradients or noise
    # can produce small negatives. log2(negative) = NaN.
    predicted_safe = predicted.clamp(min=0.0)
    actual_stim_safe = actual_stim.clamp(min=0.0)
    actual_ctrl_safe = actual_ctrl.clamp(min=0.0)

    # Compute log2 fold changes with clamping for numerical stability
    predicted_lfc = torch.log2(predicted_safe + epsilon) - torch.log2(actual_ctrl_safe + epsilon)
    actual_lfc = torch.log2(actual_stim_safe + epsilon) - torch.log2(actual_ctrl_safe + epsilon)

    # Clamp to prevent extreme values from dominating
    predicted_lfc = predicted_lfc.clamp(-20.0, 20.0)
    actual_lfc = actual_lfc.clamp(-20.0, 20.0)

    # Squared error in log-fold-change space
    base_loss = (predicted_lfc - actual_lfc) ** 2  # (batch, n_genes)

    # Weight high-responder genes more heavily
    log2_threshold = torch.log2(torch.tensor(high_responder_threshold))
    weights = torch.where(
        actual_lfc.abs() > log2_threshold,
        torch.tensor(high_responder_weight),
        torch.tensor(1.0),
    )

    # Apply expression mask: zero out loss for unexpressed genes
    weights = weights * expressed_mask.float()
    n_expressed = expressed_mask.float().sum()

    if n_expressed > 0:
        weighted_loss = (base_loss * weights).sum() / n_expressed
    else:
        # No expressed genes in this batch — return zero loss with grad
        weighted_loss = (base_loss * 0.0).sum()

    return weighted_loss


def cosine_loss(
    predicted: torch.Tensor,
    actual_stim: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine similarity loss: 1 - cos_sim(predicted, actual).

    Penalises wrong expression profile shape regardless of scale.

    Args:
        predicted:   (batch, n_genes) — predicted stim expression.
        actual_stim: (batch, n_genes) — actual stim expression.

    Returns:
        Scalar loss tensor.
    """
    return 1.0 - F.cosine_similarity(predicted, actual_stim, dim=1).mean()


def combined_loss(
    predicted: torch.Tensor,
    actual_stim: torch.Tensor,
    actual_ctrl: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> tuple:
    """
    Combined loss for perturbation response prediction.

    L = alpha * MSE + beta * LFC + gamma * cosine

    Returns total loss AND a breakdown dict for logging:
        {"mse": float, "lfc": float, "cosine": float, "total": float}

    The breakdown is essential for debugging:
        If lfc loss is high but mse is low: model predicts correct magnitude
            but wrong direction.
        If cosine loss is high: model predicts wrong expression profile shape.
        If mse loss is high: model is off on absolute values.

    Default weights (alpha=1.0, beta=0.5, gamma=0.1) are starting points.
    Do not change them without biological justification.

    Args:
        predicted:   (batch, n_genes) — predicted stim expression.
        actual_stim: (batch, n_genes) — actual stim expression.
        actual_ctrl: (batch, n_genes) — ctrl expression (input).
        alpha: MSE weight.
        beta:  log-fold-change weight.
        gamma: cosine similarity weight.

    Returns:
        (total_loss, breakdown_dict)
    """
    mse = F.mse_loss(predicted, actual_stim)
    lfc = log_fold_change_loss(predicted, actual_stim, actual_ctrl)
    cos = cosine_loss(predicted, actual_stim)

    total = alpha * mse + beta * lfc + gamma * cos

    breakdown = {
        "mse": mse.item(),
        "lfc": lfc.item(),
        "cosine": cos.item(),
        "total": total.item(),
    }

    return total, breakdown


def combined_loss_v11(
    predicted: torch.Tensor,
    actual_stim: torch.Tensor,
    actual_ctrl: torch.Tensor,
    neumann_module=None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> tuple:
    """
    Combined loss for v1.1 with Neumann L1 sparsity penalty.

    L = alpha * MSE + beta * LFC + gamma * cosine + lambda * ||W||_1

    The L1 penalty encourages sparsity in the learned GRN matrix W,
    matching the Tahoe TX1-CD approach.

    Args:
        predicted:       (batch, n_genes) predicted stim expression.
        actual_stim:     (batch, n_genes) actual stim expression.
        actual_ctrl:     (batch, n_genes) ctrl expression (input).
        neumann_module:  NeumannPropagation instance (for L1 penalty).
                         If None, falls back to combined_loss behavior.
        alpha: MSE weight.
        beta:  LFC weight.
        gamma: cosine weight.

    Returns:
        (total_loss, breakdown_dict)
    """
    mse = F.mse_loss(predicted, actual_stim)
    lfc = log_fold_change_loss(predicted, actual_stim, actual_ctrl)
    cos = cosine_loss(predicted, actual_stim)

    total = alpha * mse + beta * lfc + gamma * cos

    breakdown = {
        "mse": mse.item(),
        "lfc": lfc.item(),
        "cosine": cos.item(),
        "l1": 0.0,
        "total": total.item(),
    }

    if neumann_module is not None:
        l1 = neumann_module.l1_penalty()
        total = total + l1
        breakdown["l1"] = l1.item()
        breakdown["total"] = total.item()

    return total, breakdown


# =========================================================================
# Unit tests
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("UNIT TESTS: losses.py")
    print("=" * 70)

    all_pass = True

    # Test 1 — Perfect prediction
    print("\n  Test 1: Perfect prediction (all losses should be ~0)")
    ctrl = torch.ones(1, 10)
    stim = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    predicted = stim.clone()

    total, breakdown = combined_loss(predicted, stim, ctrl)
    print(f"    MSE:    {breakdown['mse']:.8f}")
    print(f"    LFC:    {breakdown['lfc']:.8f}")
    print(f"    Cosine: {breakdown['cosine']:.8f}")
    print(f"    Total:  {breakdown['total']:.8f}")

    if breakdown["mse"] < 1e-6 and breakdown["lfc"] < 1e-4 and breakdown["cosine"] < 1e-6:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    # Test 2 — Compressed fold change (Week 2 failure mode)
    print("\n  Test 2: Compressed fold change (IFIT1 = 238x, predicted = 7.76x)")
    ctrl2 = torch.ones(1, 10)
    stim2 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 238.0]])
    pred2 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 7.76]])

    mse_only = F.mse_loss(pred2, stim2)
    lfc_only = log_fold_change_loss(pred2, stim2, ctrl2)
    total2, bd2 = combined_loss(pred2, stim2, ctrl2)

    print(f"    MSE loss:  {mse_only.item():.2f}")
    print(f"    LFC loss:  {lfc_only.item():.4f}")
    print(f"    Total:     {total2.item():.4f}")
    print(f"    MSE dominated by 238x gene: MSE={mse_only.item():.0f}")
    print(f"    LFC captures log-scale error: log2(238)={torch.log2(torch.tensor(238.)).item():.2f}, "
          f"log2(7.76)={torch.log2(torch.tensor(7.76)).item():.2f}")

    if lfc_only.item() > 1.0 and not torch.isnan(lfc_only) and not torch.isinf(lfc_only):
        print("    PASS — LFC penalises fold-change compression")
    else:
        print("    FAIL")
        all_pass = False

    # Test 3 — All genes unchanged (null perturbation)
    print("\n  Test 3: Null perturbation (all losses should be ~0)")
    ctrl3 = torch.ones(1, 10)
    stim3 = torch.ones(1, 10)
    pred3 = torch.ones(1, 10)

    total3, bd3 = combined_loss(pred3, stim3, ctrl3)
    print(f"    MSE:    {bd3['mse']:.8f}")
    print(f"    LFC:    {bd3['lfc']:.8f}")
    print(f"    Cosine: {bd3['cosine']:.8f}")
    print(f"    Total:  {bd3['total']:.8f}")

    if bd3["mse"] < 1e-6 and bd3["lfc"] < 1e-4 and bd3["cosine"] < 1e-6:
        print("    PASS")
    else:
        print("    FAIL")
        all_pass = False

    # Test 4 — Numerical stability (zero expression)
    print("\n  Test 4: Numerical stability (zero-expression genes)")
    ctrl4 = torch.zeros(1, 10)
    stim4 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]])
    pred4 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0]])

    total4, bd4 = combined_loss(pred4, stim4, ctrl4)
    print(f"    MSE:    {bd4['mse']:.8f}")
    print(f"    LFC:    {bd4['lfc']:.4f}")
    print(f"    Cosine: {bd4['cosine']:.8f}")
    print(f"    Total:  {bd4['total']:.4f}")

    has_nan = any(torch.isnan(torch.tensor(v)) for v in bd4.values())
    has_inf = any(torch.isinf(torch.tensor(v)) for v in bd4.values())
    if not has_nan and not has_inf:
        print("    PASS — No NaN or Inf with epsilon floor")
    else:
        print(f"    FAIL — NaN={has_nan}, Inf={has_inf}")
        all_pass = False

    # Test 5 — Sparse single-cell data (many zeros — the NaN bug scenario)
    print("\n  Test 5: Sparse single-cell data (90% zeros, like real scRNA-seq)")
    torch.manual_seed(42)
    n_genes = 3010
    ctrl5 = torch.zeros(4, n_genes)
    stim5 = torch.zeros(4, n_genes)
    # Only ~10% of genes are expressed (like real single-cell data)
    expressed_idx = torch.randperm(n_genes)[:300]
    ctrl5[:, expressed_idx] = torch.rand(4, 300) * 3.0
    stim5[:, expressed_idx] = torch.rand(4, 300) * 3.0
    # A few high-responder genes (like IFIT1)
    stim5[:, expressed_idx[:5]] = ctrl5[:, expressed_idx[:5]] * 50.0

    pred5 = ctrl5.clone() + torch.randn_like(ctrl5) * 0.1  # noisy prediction

    total5, bd5 = combined_loss(pred5, stim5, ctrl5)
    print(f"    MSE:    {bd5['mse']:.4f}")
    print(f"    LFC:    {bd5['lfc']:.4f}")
    print(f"    Cosine: {bd5['cosine']:.4f}")
    print(f"    Total:  {bd5['total']:.4f}")

    has_nan5 = any(torch.isnan(torch.tensor(v)) for v in bd5.values())
    has_inf5 = any(torch.isinf(torch.tensor(v)) for v in bd5.values())
    if not has_nan5 and not has_inf5 and bd5['total'] < 1e6:
        print("    PASS — No NaN/Inf with sparse single-cell data")
    else:
        print(f"    FAIL — NaN={has_nan5}, Inf={has_inf5}, Total={bd5['total']:.4f}")
        all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("OVERALL: PASS — All 5 unit tests passed")
    else:
        print("OVERALL: FAIL — Some tests failed")
