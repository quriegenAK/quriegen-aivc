"""Calderon linear-probe eval pipeline.

End-to-end:
  1. project_calderon_to_dogma_space — apply (n_calderon × n_dogma) M
     to (n_samples, n_calderon) counts → (n_samples, n_dogma)
  2. encode_samples — run an encoder (MockEncoder or real-checkpoint
     loaded encoder) on projected counts → (n_samples, latent_dim)
  3. run_linear_probe — fit logistic regression on cell_type labels
     with leave-one-donor-out (or arbitrary group) CV → metrics

Mock encoder is random-init Linear; produces near-chance probe accuracy.
Real-checkpoint loading is deferred to a future PR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# --- Projection ---------------------------------------------------------------

def project_calderon_to_dogma_space(
    calderon_X: sp.spmatrix | np.ndarray,
    M: sp.spmatrix,
) -> sp.csr_matrix:
    """Project Calderon counts into DOGMA peak space.

    Parameters
    ----------
    calderon_X : (n_samples, n_calderon_peaks)
    M : (n_calderon_peaks, n_dogma_peaks) — sparse projection matrix.

    Returns
    -------
    (n_samples, n_dogma_peaks) — sparse CSR.
    """
    if calderon_X.shape[1] != M.shape[0]:
        raise ValueError(
            f"calderon_X.shape[1]={calderon_X.shape[1]} != M.shape[0]={M.shape[0]}"
        )
    if sp.issparse(calderon_X):
        return (calderon_X @ M).tocsr()
    return sp.csr_matrix(np.asarray(calderon_X) @ M)


# --- Mock encoder (interface-compatible with PeakLevelATACEncoder) -----------

class MockEncoder(nn.Module):
    """Random-init linear projection mimicking the trained encoder interface.

    Mimics aivc.skills.atac_peak_encoder.PeakLevelATACEncoder's input/output
    shapes: (batch, n_peaks) -> (batch, latent_dim). For plumbing/scaffold
    tests only — produces near-chance probe accuracy on real data.
    """

    def __init__(self, n_peaks: int, latent_dim: int = 64, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(n_peaks, latent_dim)
        with torch.no_grad():
            self.linear.weight.data = torch.empty_like(self.linear.weight).normal_(generator=gen)
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_atac_encoder_from_ckpt(
    ckpt_path,
    expected_n_peaks: int = None,
    map_location: str = "cpu",
):
    """Load PeakLevelATACEncoder from a pretrain checkpoint.

    Used by scripts/eval_calderon_linear_probe.py when --encoder_ckpt is
    provided. Replaces MockEncoder with the trained encoder.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to checkpoint produced by scripts/pretrain_multiome.py.
    expected_n_peaks : int, optional
        If provided, asserts ckpt's n_peaks matches. Catches coordinate-
        system / peak-set mismatches at load time instead of at encoder
        forward where shape errors are cryptic.
    map_location : str
        torch.load map_location. Default 'cpu'.

    Returns
    -------
    encoder : PeakLevelATACEncoder (in eval mode)
    config : dict (training config from ckpt for provenance)
    """
    from pathlib import Path
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.training.ckpt_loader import load_pretrain_ckpt_raw

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_pretrain_ckpt_raw(ckpt_path, map_location=map_location)
    config = ckpt.get("config", {})

    n_peaks = int(config.get("n_peaks", 0))
    if n_peaks == 0 and "atac_encoder" in ckpt:
        # Recover from LSI weight matrix shape if config didn't stamp it
        lsi_w = ckpt["atac_encoder"].get("lsi.weight")
        if lsi_w is not None:
            n_peaks = int(lsi_w.shape[1])
    if n_peaks == 0:
        raise ValueError(f"Could not determine n_peaks from {ckpt_path}")

    if expected_n_peaks is not None and expected_n_peaks != n_peaks:
        raise ValueError(
            f"n_peaks mismatch: ckpt={n_peaks}, expected={expected_n_peaks}. "
            "Likely coordinate-system or peak-set mismatch between trained "
            "encoder and the projection target. Verify projection M was "
            "rebuilt against the same DOGMA peak set as the trained encoder."
        )

    attn_dim = int(config.get("atac_latent", 64))
    n_lysis_categories = int(config.get("n_lysis_categories", 0))
    lysis_cov_dim = int(config.get("lysis_cov_dim", 8))

    encoder = PeakLevelATACEncoder(
        n_peaks=n_peaks,
        attn_dim=attn_dim,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=lysis_cov_dim,
    )
    encoder.load_state_dict(ckpt["atac_encoder"], strict=True)
    encoder.eval()
    return encoder, config


# --- PR Phase 6.5g.3 Pivot A: joint-fusion eval -----------------------------

def load_full_encoders_from_ckpt(
    ckpt_path,
    expected_n_peaks: int = None,
    map_location: str = "cpu",
) -> dict:
    """Load all three encoders + projections from a pretrain ckpt.

    Required for Pivot A (joint-fusion Calderon eval): forward Calderon
    ATAC through the full encoder stack (with zero-padded RNA/Protein)
    to extract z_supcon — the same joint embedding SupCon was trained on.

    Returns dict with keys:
        rna_encoder, atac_encoder, protein_encoder,
        rna_proj, atac_proj, protein_proj,
        config, n_genes, n_peaks, n_proteins, proj_dim
    All encoders/projections in eval mode, on `map_location`.
    """
    from pathlib import Path
    import torch.nn as nn
    from aivc.skills.atac_peak_encoder import PeakLevelATACEncoder
    from aivc.skills.rna_encoder import SimpleRNAEncoder
    from aivc.skills.protein_encoder import ProteinEncoder
    from aivc.training.ckpt_loader import load_pretrain_ckpt_raw

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_pretrain_ckpt_raw(ckpt_path, map_location=map_location)
    config = ckpt.get("config", {})

    n_peaks = int(config.get("n_peaks", 0))
    if n_peaks == 0 and "atac_encoder" in ckpt:
        lsi_w = ckpt["atac_encoder"].get("lsi.weight")
        if lsi_w is not None:
            n_peaks = int(lsi_w.shape[1])
    n_genes = int(config.get("n_genes", 0))
    n_proteins = int(config.get("n_proteins", 0))
    rna_latent = int(config.get("rna_latent", 128))
    atac_latent = int(config.get("atac_latent", 64))
    protein_latent = int(config.get("protein_latent", rna_latent))
    proj_dim = int(config.get("proj_dim", 128))
    n_lysis_categories = int(config.get("n_lysis_categories", 0))
    lysis_cov_dim = int(config.get("lysis_cov_dim", 8))

    if expected_n_peaks is not None and expected_n_peaks != n_peaks:
        raise ValueError(
            f"n_peaks mismatch: ckpt={n_peaks}, expected={expected_n_peaks}."
        )

    rna_enc = SimpleRNAEncoder(
        n_genes=n_genes,
        latent_dim=rna_latent,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=lysis_cov_dim,
    )
    atac_enc = PeakLevelATACEncoder(
        n_peaks=n_peaks,
        attn_dim=atac_latent,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=lysis_cov_dim,
    )
    protein_enc = ProteinEncoder(
        n_proteins=n_proteins,
        embed_dim=protein_latent,
        n_lysis_categories=n_lysis_categories,
        lysis_cov_dim=lysis_cov_dim,
    )

    missing = [k for k in ("rna_encoder", "atac_encoder", "protein_encoder",
                            "rna_proj", "atac_proj", "protein_proj")
               if k not in ckpt]
    if missing:
        raise ValueError(
            f"ckpt {ckpt_path} missing keys {missing}. The joint-fusion "
            f"eval requires all three encoders + projections. "
            f"Available keys: {sorted(ckpt.keys())}"
        )

    rna_enc.load_state_dict(ckpt["rna_encoder"], strict=True)
    atac_enc.load_state_dict(ckpt["atac_encoder"], strict=True)
    protein_enc.load_state_dict(ckpt["protein_encoder"], strict=True)

    rna_proj = nn.Linear(rna_latent, proj_dim)
    atac_proj = nn.Linear(atac_latent, proj_dim)
    protein_proj = nn.Linear(protein_latent, proj_dim)
    rna_proj.load_state_dict(ckpt["rna_proj"], strict=True)
    atac_proj.load_state_dict(ckpt["atac_proj"], strict=True)
    protein_proj.load_state_dict(ckpt["protein_proj"], strict=True)

    for m in (rna_enc, atac_enc, protein_enc, rna_proj, atac_proj, protein_proj):
        m.eval().to(map_location)

    return {
        "rna_encoder": rna_enc,
        "atac_encoder": atac_enc,
        "protein_encoder": protein_enc,
        "rna_proj": rna_proj,
        "atac_proj": atac_proj,
        "protein_proj": protein_proj,
        "config": config,
        "n_genes": n_genes,
        "n_peaks": n_peaks,
        "n_proteins": n_proteins,
        "proj_dim": proj_dim,
    }


def encode_samples_via_joint_fusion(
    X_atac_dogma: sp.spmatrix | np.ndarray,
    encoders: dict,
    batch_size: int = 64,
    device: str = "cpu",
    lysis_idx=None,
) -> np.ndarray:
    """Forward ATAC-only inputs through the FULL encoder stack with
    zero-padded RNA/Protein; return z_supcon (Pivot A diagnostic).

    z_supcon = L2_normalize(mean(L2(z_rna) + L2(z_atac) + L2(z_protein)))

    Same fusion formula used at training time (PR #54c). For Calderon
    eval, only ATAC is observed — RNA + Protein are zero-padded. This
    tests whether the cross-modal alignment learned during DOGMA
    pretrain helps at inference when only ATAC is available.

    Returns: (n_samples, proj_dim) numpy array.
    """
    import torch.nn.functional as F

    rna_enc = encoders["rna_encoder"].to(device).eval()
    atac_enc = encoders["atac_encoder"].to(device).eval()
    protein_enc = encoders["protein_encoder"].to(device).eval()
    rna_proj = encoders["rna_proj"].to(device).eval()
    atac_proj = encoders["atac_proj"].to(device).eval()
    protein_proj = encoders["protein_proj"].to(device).eval()
    n_genes = encoders["n_genes"]
    n_proteins = encoders["n_proteins"]

    n = X_atac_dogma.shape[0]
    n_lysis_categories = int(encoders["config"].get("n_lysis_categories", 0))
    pass_lysis = n_lysis_categories > 0

    if pass_lysis:
        if lysis_idx is None:
            lysis_full = torch.zeros(n, dtype=torch.long, device=device)
        else:
            lysis_full = torch.as_tensor(lysis_idx, dtype=torch.long).to(device)
            assert lysis_full.shape[0] == n

    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            atac_b = X_atac_dogma[i:i + batch_size]
            if sp.issparse(atac_b):
                atac_b = atac_b.toarray()
            atac_t = torch.from_numpy(np.asarray(atac_b, dtype=np.float32)).to(device)
            B = atac_t.shape[0]

            rna_t = torch.zeros(B, n_genes, dtype=torch.float32, device=device)
            prot_t = torch.zeros(B, n_proteins, dtype=torch.float32, device=device)

            kw = {"lysis_idx": lysis_full[i:i + batch_size]} if pass_lysis else {}

            rna_latent, _ = rna_enc(rna_t, **kw)
            atac_latent = atac_enc(atac_t, **kw)
            protein_latent = protein_enc(prot_t, rna_emb=rna_latent, **kw)

            z_rna = rna_proj(rna_latent)
            z_atac = atac_proj(atac_latent)
            z_protein = protein_proj(protein_latent)

            z_supcon = F.normalize(
                (
                    F.normalize(z_rna, dim=-1)
                    + F.normalize(z_atac, dim=-1)
                    + F.normalize(z_protein, dim=-1)
                ) / 3.0,
                dim=-1,
            )
            out.append(z_supcon.cpu().numpy())

    return np.concatenate(out, axis=0)


# --- end Pivot A additions --------------------------------------------------


def encode_samples(
    X: sp.spmatrix | np.ndarray | torch.Tensor,
    encoder: nn.Module,
    batch_size: int = 64,
    device: str = "cpu",
    lysis_idx=None,
) -> np.ndarray:
    """Encode samples through encoder in batches; return numpy (n, latent_dim).

    If the encoder was constructed with `n_lysis_categories > 0` (joint-arm
    training), the forward path requires `lysis_idx`. For cross-corpus eval
    (e.g., Calderon) where the source dataset has no lysis_protocol info,
    we default to all-zeros (category 0) — lysis is a batch effect for
    DOGMA's LLL/DIG split, not a cell-type signal, so the constant offset
    is acceptable for downstream cell-type probing.

    Parameters
    ----------
    lysis_idx : array-like, optional
        Per-sample lysis category (0 = LLL, 1 = DIG). If None and encoder
        has n_lysis_categories > 0, defaults to all-zeros. If None and
        encoder has no covariate, no lysis_idx is passed.
    """
    encoder = encoder.to(device).eval()
    n = X.shape[0]
    out = []

    # Detect whether the encoder has lysis covariate (PR #43+ encoders).
    n_lysis_categories = getattr(encoder, "n_lysis_categories", 0)
    if n_lysis_categories > 0:
        if lysis_idx is None:
            # Default: assume LLL (category 0) for the entire batch.
            # Cross-corpus eval doesn't have lysis info; this constant
            # offset is the cleanest no-info default.
            lysis_idx_full = torch.zeros(n, dtype=torch.long, device=device)
        else:
            if isinstance(lysis_idx, np.ndarray):
                lysis_idx_full = torch.from_numpy(lysis_idx).long().to(device)
            elif isinstance(lysis_idx, torch.Tensor):
                lysis_idx_full = lysis_idx.long().to(device)
            else:
                lysis_idx_full = torch.tensor(lysis_idx, dtype=torch.long, device=device)
            assert lysis_idx_full.shape[0] == n, (
                f"lysis_idx length {lysis_idx_full.shape[0]} != n_samples {n}"
            )
    else:
        lysis_idx_full = None

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size]
            if sp.issparse(batch):
                batch = batch.toarray()
            elif isinstance(batch, torch.Tensor):
                batch = batch.cpu().numpy()
            t = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(device)
            if lysis_idx_full is not None:
                lysis_batch = lysis_idx_full[i:i + batch_size]
                z = encoder(t, lysis_idx=lysis_batch)
            else:
                z = encoder(t)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


# --- Linear probe -------------------------------------------------------------

@dataclass
class FoldMetric:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float
    test_groups: list


def run_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    max_iter: int = 1000,
    random_state: int = 0,
) -> dict:
    """Fit logistic regression with leave-one-group-out (or stratified k-fold).

    If groups is provided, uses LeaveOneGroupOut (one held-out group per fold).
    Else uses StratifiedKFold(cv_folds).

    Returns metrics dict with per-fold and aggregate accuracy + f1_macro.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    if groups is not None:
        cv = LeaveOneGroupOut()
        splits = list(cv.split(embeddings, labels, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(embeddings, labels))

    fold_metrics: list[FoldMetric] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(embeddings[train_idx], labels[train_idx])
        pred = clf.predict(embeddings[test_idx])
        held_groups = (
            sorted(set(groups[test_idx].tolist())) if groups is not None else []
        )
        fold_metrics.append(FoldMetric(
            fold=fold_idx,
            n_train=int(len(train_idx)),
            n_test=int(len(test_idx)),
            accuracy=float(accuracy_score(labels[test_idx], pred)),
            f1_macro=float(f1_score(labels[test_idx], pred, average="macro", zero_division=0)),
            test_groups=held_groups,
        ))

    return {
        "fold_metrics": [m.__dict__ for m in fold_metrics],
        "mean_accuracy": float(np.mean([m.accuracy for m in fold_metrics])),
        "std_accuracy": float(np.std([m.accuracy for m in fold_metrics])),
        "mean_f1_macro": float(np.mean([m.f1_macro for m in fold_metrics])),
        "n_folds": len(fold_metrics),
        "n_classes": int(len(np.unique(labels))),
    }
