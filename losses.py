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

from aivc.data.modality_mask import ModalityKey


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


def causal_ordering_loss(
    attention_weights: torch.Tensor,
    temporal_order: list = None,
) -> torch.Tensor:
    """
    Penalises attention patterns that violate biological causal ordering.

    Soft constraint: discourages upper-triangular attention weights
    (future modalities attending to past modalities).
    The hard mask in fusion.py provides the primary enforcement.

    Temporal order:
      Index 0 = ATAC, 1 = Phospho, 2 = RNA, 3 = Protein

    Args:
        attention_weights: (batch, n_heads, n_mod, n_mod)
        temporal_order:    list of modality indices. Default [0,1,2,3].

    Returns:
        Scalar — mean upper-triangular attention weight.
        Minimising this = enforcing causal order.
    """
    if temporal_order is None:
        temporal_order = [0, 1, 2, 3]

    n_mod = len(temporal_order)
    device = attention_weights.device

    # Upper triangular mask (strict: diagonal excluded)
    upper_mask = torch.triu(
        torch.ones(n_mod, n_mod, device=device, dtype=torch.bool),
        diagonal=1,
    )

    violation_weights = attention_weights[..., upper_mask]
    return violation_weights.abs().mean()


def combined_loss_multimodal(
    predicted: torch.Tensor,
    actual_stim: torch.Tensor,
    actual_ctrl: torch.Tensor,
    neumann_module=None,
    emb_pairs: list = None,
    contrastive_loss_fn=None,
    cross_modal_fn=None,
    attn_weights: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.1,
    lambda_contrast: float = 0.05,
    lambda_cross: float = 0.05,
    lambda_causal: float = 0.1,
    modality_mask: torch.Tensor = None,
) -> tuple:
    """
    Extended combined loss for multi-modal training.

    Adds contrastive and cross-modal terms to the existing v1.1 loss.
    CRITICAL: contrastive and cross-modal terms are ONLY added when
    emb_pairs is not None AND the corresponding loss fn is not None.

    For RNA-only training (Kang 2018), emb_pairs=None and this function
    is identical to combined_loss_v11().

    Args:
        predicted, actual_stim, actual_ctrl, neumann_module,
        alpha, beta, gamma: Same as combined_loss_v11().
        emb_pairs:          list of (emb_a, emb_b) tuples --
                            one per physically paired modality pair.
                            None -> contrastive loss disabled.
        contrastive_loss_fn: PairedModalityContrastiveLoss instance.
                             None -> contrastive loss disabled.
        cross_modal_fn:     CrossModalPredictionLoss instance.
                            None -> cross-modal loss disabled.
        lambda_contrast:    float -- contrastive loss weight. Default 0.05.
        lambda_cross:       float -- cross-modal loss weight. Default 0.05.

    Returns:
        (total_loss, breakdown_dict)
    """
    from aivc.training.loss_registry import LossRegistry, LossTerm

    # Term functions — each accepts **kwargs so the registry can pass a
    # superset of batch fields without breaking signature contracts.
    #
    # Day 2 additions: modality_mask gating on MSE / LFC / cosine / contrastive.
    # Per D2 semantics, when modality_mask indicates the RNA column is all-zero
    # for the batch, MSE/LFC/cosine silently return 0 contribution (they operate
    # on the RNA-reconstruction axis). Contrastive returns 0 if fewer than 2
    # modalities have any present cells in the batch.
    # _l1_fn, _cross_modal_fn, _causal_fn are unchanged.
    def _mse_fn(predicted, actual_stim, modality_mask=None, **_):
        if modality_mask is not None:
            rna_present = modality_mask[:, int(ModalityKey.RNA)]
            if float(rna_present.sum()) < 1.0:
                return torch.zeros((), device=predicted.device, requires_grad=True)
            w = rna_present.unsqueeze(-1)  # (B, 1)
            sq = (predicted - actual_stim) ** 2 * w
            denom = w.expand_as(sq).sum().clamp_min(1.0)
            return sq.sum() / denom
        return F.mse_loss(predicted, actual_stim)

    def _lfc_fn(predicted, actual_stim, actual_ctrl, modality_mask=None, **_):
        if modality_mask is not None:
            rna_present = modality_mask[:, int(ModalityKey.RNA)]
            if float(rna_present.sum()) < 1.0:
                return torch.zeros((), device=predicted.device, requires_grad=True)
            keep = rna_present.bool()
            return log_fold_change_loss(
                predicted[keep], actual_stim[keep], actual_ctrl[keep],
            )
        return log_fold_change_loss(predicted, actual_stim, actual_ctrl)

    def _cosine_fn(predicted, actual_stim, modality_mask=None, **_):
        if modality_mask is not None:
            rna_present = modality_mask[:, int(ModalityKey.RNA)]
            if float(rna_present.sum()) < 1.0:
                return torch.zeros((), device=predicted.device, requires_grad=True)
            keep = rna_present.bool()
            return cosine_loss(predicted[keep], actual_stim[keep])
        return cosine_loss(predicted, actual_stim)

    def _l1_fn(predicted, neumann_module=None, **_):
        if neumann_module is not None:
            return neumann_module.l1_penalty()
        return torch.tensor(0.0, device=predicted.device)

    def _contrastive_fn(
        predicted, emb_pairs=None, contrastive_loss_fn=None,
        modality_mask=None, **_,
    ):
        term = torch.tensor(0.0, device=predicted.device)
        if emb_pairs is None or contrastive_loss_fn is None:
            return term
        # D2: skip contrastive if <2 modalities have any present cells
        if modality_mask is not None:
            n_present_modalities = int((modality_mask.sum(dim=0) > 0).sum().item())
            if n_present_modalities < 2:
                return term
        for emb_a, emb_b in emb_pairs:
            term = term + contrastive_loss_fn(emb_a, emb_b)
        if len(emb_pairs) > 0:
            term = term / len(emb_pairs)
        return term

    def _cross_modal_fn(predicted, emb_pairs=None, cross_modal_fn=None, **_):
        term = torch.tensor(0.0, device=predicted.device)
        if emb_pairs is not None and cross_modal_fn is not None:
            for emb_a, emb_b in emb_pairs:
                cm_result = cross_modal_fn(emb_a, emb_b)
                term = term + cm_result["loss"]
            if len(emb_pairs) > 0:
                term = term / len(emb_pairs)
        return term

    def _causal_fn(predicted, attn_weights=None, **_):
        if attn_weights is not None:
            return causal_ordering_loss(attn_weights)
        return torch.tensor(0.0, device=predicted.device)

    # Registration order mirrors legacy summation order exactly:
    # α*mse + β*lfc + γ*cos + 1.0*l1 + λ_c*contrast + λ_x*cross + λ_z*causal
    registry = LossRegistry()
    registry.register(LossTerm("mse",         _mse_fn,         alpha,           "joint"))
    registry.register(LossTerm("lfc",         _lfc_fn,         beta,            "joint"))
    registry.register(LossTerm("cosine",      _cosine_fn,      gamma,           "joint"))
    registry.register(LossTerm("l1",          _l1_fn,          1.0,             "joint"))
    registry.register(LossTerm("contrastive", _contrastive_fn, lambda_contrast, "joint"))
    registry.register(LossTerm("cross_modal", _cross_modal_fn, lambda_cross,    "joint"))
    registry.register(LossTerm("causal",      _causal_fn,      lambda_causal,   "joint"))

    total, components = registry.compute(
        stage="joint",
        predicted=predicted,
        actual_stim=actual_stim,
        actual_ctrl=actual_ctrl,
        neumann_module=neumann_module,
        emb_pairs=emb_pairs,
        contrastive_loss_fn=contrastive_loss_fn,
        cross_modal_fn=cross_modal_fn,
        attn_weights=attn_weights,
        modality_mask=modality_mask,
    )

    breakdown = {
        "mse":         components["mse"],
        "lfc":         components["lfc"],
        "cosine":      components["cosine"],
        "l1":          components["l1"],
        "contrastive": components["contrastive"],
        "cross_modal": components["cross_modal"],
        "causal":      components["causal"],
        "total":       total.item(),
    }

    return total, breakdown


def _dogma_pretrain_loss(
    rna_recon: torch.Tensor,
    rna_target: torch.Tensor,
    atac_recon: torch.Tensor,
    atac_target: torch.Tensor,
    protein_recon: torch.Tensor,
    protein_target: torch.Tensor,
    z_rna: torch.Tensor,
    z_atac: torch.Tensor,
    z_protein: torch.Tensor,
    modality_mask: torch.Tensor,
    rna_mask: torch.Tensor = None,
    atac_mask: torch.Tensor = None,
    protein_mask: torch.Tensor = None,
    infonce_temperature: float = 0.1,
    w_rna: float = 1.0,
    w_atac: float = 1.0,
    w_protein: float = 1.0,
    w_triad: float = 0.5,
) -> tuple:
    """DOGMA pretrain loss — 3-modal reconstruction + 3-way InfoNCE.

    Composition:
      w_rna     * masked_rna_recon
    + w_atac    * masked_atac_recon
    + w_protein * masked_protein_recon       (D2 mask-gated)
    + w_triad   * cross_modal_infonce_triad  (D1 pairwise-average)

    NO Neumann L1 (DOGMA is pretrain, not causal).
    NO causal_ordering term (same reason).

    Registers under stage="pretrain" via LossRegistry. Each term name
    passes _guard_pretrain_name (causal-adjacent substring guard) at
    registration — defense-in-depth per the Phase 5 PR contract.
    """
    from aivc.training.loss_registry import LossRegistry, LossTerm
    from aivc.training.pretrain_losses import (
        _masked_rna_recon,
        _masked_atac_recon,
        _masked_protein_recon,
        _cross_modal_infonce_triad,
        _guard_pretrain_name,
    )

    spec = (
        ("masked_rna_recon",          _masked_rna_recon,          w_rna),
        ("masked_atac_recon",         _masked_atac_recon,         w_atac),
        ("masked_protein_recon",      _masked_protein_recon,      w_protein),
        ("cross_modal_infonce_triad", _cross_modal_infonce_triad, w_triad),
    )
    registry = LossRegistry()
    for name, fn, weight in spec:
        _guard_pretrain_name(name)
        registry.register(LossTerm(name=name, fn=fn, weight=weight, stage="pretrain"))

    total, components = registry.compute(
        stage="pretrain",
        predicted=rna_recon,  # device-fallback anchor for all-skip edge case
        rna_recon=rna_recon, rna_target=rna_target, rna_mask=rna_mask,
        atac_recon=atac_recon, atac_target=atac_target, atac_mask=atac_mask,
        protein_recon=protein_recon, protein_target=protein_target,
        protein_mask=protein_mask,
        z_rna=z_rna, z_atac=z_atac, z_protein=z_protein,
        modality_mask=modality_mask,
        infonce_temperature=infonce_temperature,
    )
    components["total"] = total.item() if isinstance(total, torch.Tensor) else float(total)
    return total, components


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
