"""Stage 3a — Mimitou CRISPR DataLoader + arm-balanced sampler + perturbation embedder.

Per docs/specs/stage3_part2_architecture_proposal_2026_05_06.md §6 (Stage 3a deliverable).

Single source of truth for Mimitou's HTO/4-arm structure. Downstream
training scripts and future tests reuse this module rather than redoing
the arm-mapping logic.

The 4-arm synergy design (for the held-out CD3E+CD4 zero-shot eval):
  NTC                  → has_stim=False, has_inh=False  (vehicle baseline)
  CD3E (alone)         → has_stim=True,  has_inh=False  (perturbation A)
  CD4  (alone)         → has_stim=False, has_inh=True   (perturbation B)
  CD3E_CD4_double      → has_stim=True,  has_inh=True   (synergy condition)

The CD3E→stim, CD4→inh role assignment is arbitrary (the architecture
treats stim/inh symmetrically); chose CD3E as "stim" because TCR signaling
block produces a larger chromatin shift (CD3E 0.91 vs CD4 0.39 from
Report 2), making it the more informative anchor.

Additional Mimitou arms (ZAP70, NFKB2) are treated as "other stim
perturbations" by default — useful for adapter SupCon training (more
classes for contrastive separation) but excluded from the held-out
synergy eval.

Module exports:
  MimitouArmMap            — static arm → (has_stim, has_inh, stim_id, inh_id) map
  MimitouDataset           — torch Dataset on the labeled h5ad
  ArmBalancedBatchSampler  — yields batches with ≥2 cells per arm (SupCon req)
  PerturbationEmbedder     — nn.Embedding wrapper for stim+inh embeddings

This file is the canonical entry-point for any Mimitou-specific data
loading. If future Stage 3 work needs Mimitou data, import from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, Sampler


# --- Arm → role assignment (canonical) ---

# Stim agents (perturbation role A)
STIM_VOCAB = ("CD3E", "ZAP70", "NFKB2")
# Inh agents (perturbation role B)
INH_VOCAB = ("CD4",)
# Double-KO arm sentinel
DOUBLE_KO_NAME = "CD3E_CD4_double"


def _build_index(vocab: Sequence[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(vocab)}


STIM_ID = _build_index(STIM_VOCAB)
INH_ID = _build_index(INH_VOCAB)


@dataclass
class ArmAssignment:
    """Per-cell arm → role mapping used by the decomposed readout.

    has_stim / has_inh feed the zero-arm loss directly.
    stim_id / inh_id index into PerturbationEmbedder (sentinel -1 = absent).
    """

    has_stim: bool
    has_inh: bool
    stim_id: int   # -1 if has_stim=False
    inh_id: int    # -1 if has_inh=False


def map_arm_to_roles(arm: str) -> ArmAssignment:
    """Map a perturbation arm string to the canonical 4-arm role assignment.

    Raises ValueError if the arm isn't recognized. NTC and single KOs in
    STIM_VOCAB/INH_VOCAB resolve to single-role assignments; CD3E_CD4_double
    resolves to both roles active (the synergy training signal).
    """
    if arm == "NTC":
        return ArmAssignment(has_stim=False, has_inh=False, stim_id=-1, inh_id=-1)
    if arm == DOUBLE_KO_NAME:
        return ArmAssignment(
            has_stim=True, has_inh=True,
            stim_id=STIM_ID["CD3E"], inh_id=INH_ID["CD4"],
        )
    if arm in STIM_ID:
        return ArmAssignment(has_stim=True, has_inh=False,
                             stim_id=STIM_ID[arm], inh_id=-1)
    if arm in INH_ID:
        return ArmAssignment(has_stim=False, has_inh=True,
                             stim_id=-1, inh_id=INH_ID[arm])
    raise ValueError(
        f"Unknown arm {arm!r}. Known arms: NTC, {DOUBLE_KO_NAME}, "
        f"{list(STIM_ID)}, {list(INH_ID)}"
    )


# Canonical static arm map (for downstream tests + scripts that need
# the structure without instantiating a dataset)
MimitouArmMap = {
    "NTC":               map_arm_to_roles("NTC"),
    "CD3E":              map_arm_to_roles("CD3E"),
    "CD4":               map_arm_to_roles("CD4"),
    "ZAP70":             map_arm_to_roles("ZAP70"),
    "NFKB2":             map_arm_to_roles("NFKB2"),
    DOUBLE_KO_NAME:      map_arm_to_roles(DOUBLE_KO_NAME),
}


# --- PerturbationEmbedder ---

class PerturbationEmbedder(torch.nn.Module):
    """Learnable embeddings for stim + inh perturbation IDs.

    Stage 3a uses Mimitou's small panel (3 stim, 1 inh). Stage 3b extends
    to QurieSeq's drug + stimulus vocabulary.

    forward(stim_id, inh_id): returns (stim_emb, inh_emb) where each is
    either a tensor of shape (B, pert_dim) or None when the corresponding
    id is -1 (sentinel "absent").

    Args:
        n_stim:    size of the stim vocabulary (default len(STIM_VOCAB)=3)
        n_inh:     size of the inh vocabulary (default len(INH_VOCAB)=1)
        pert_dim:  embedding dimension (default 32 per spec §3.3)
    """

    def __init__(self, n_stim: int = len(STIM_VOCAB),
                 n_inh: int = len(INH_VOCAB), pert_dim: int = 32):
        super().__init__()
        self.n_stim = n_stim
        self.n_inh = n_inh
        self.pert_dim = pert_dim
        # +1 for the "absent" slot at index 0; legitimate IDs start at 1
        self.stim_emb = torch.nn.Embedding(n_stim + 1, pert_dim, padding_idx=0)
        self.inh_emb = torch.nn.Embedding(n_inh + 1, pert_dim, padding_idx=0)

    def forward(
        self,
        stim_id: torch.Tensor,  # (B,) int64, -1 = absent
        inh_id:  torch.Tensor,  # (B,) int64, -1 = absent
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (stim_emb, inh_emb) for the batch.

        The decomposed readout expects None-valued args when the
        perturbation is absent; but the embedder is called once per
        batch, so we always return a tensor and let the caller mask
        per-cell using arm_mask. Convention here:
          stim_id == -1 → emit zero embedding (padding_idx behavior)
          stim_id >= 0  → emit learned embedding for that id

        Caller should ALSO pass arm_mask to the readout/loss so the
        zero-arm constraint can verify these are zero where required.
        """
        # Shift IDs by +1 so -1 → 0 (padding slot = zero vector)
        stim_idx = (stim_id + 1).clamp(min=0)
        inh_idx = (inh_id + 1).clamp(min=0)
        return self.stim_emb(stim_idx), self.inh_emb(inh_idx)


# --- Dataset ---

class MimitouDataset(Dataset):
    """Torch Dataset over a Mimitou labeled h5ad.

    Returns per-cell dicts with:
        cell_idx:    int — for traceability
        atac_row:    sparse row (1, n_peaks) — for downstream encoder forward
        protein:     dense (n_proteins,) tensor — Stage 3a recon target
        perturbation: str — arm name
        has_stim, has_inh: bool — flags for zero-arm masking
        stim_id, inh_id: int — for PerturbationEmbedder lookup (-1 = absent)
        arm_label:   int — dense label index for SupCon (over included arms)
        time:        float — single endpoint for Mimitou (16 hours = 960 min)

    Args:
        h5ad_path:        path to dogma_lll_union_labeled-style h5ad with
                          obs['perturbation'], obsm['atac_peaks'], obsm['protein']
        included_arms:    arms to include in the dataset. Default = synergy
                          trio + control + other CRISPRs. Pass a tighter
                          set (e.g., synergy_only=True equivalent) to filter.
        exclude_double:   if True, drop CD3E_CD4_double for held-out synergy eval.
                          Default False (caller chooses).
    """

    DEFAULT_ARMS = ("NTC", "CD3E", "CD4", "ZAP70", "NFKB2", DOUBLE_KO_NAME)
    MIMITOU_TIME_MINUTES = 16 * 60.0  # 16 hour endpoint

    def __init__(
        self,
        h5ad_path,
        included_arms: Sequence[str] = DEFAULT_ARMS,
        exclude_double: bool = False,
    ):
        import anndata as ad

        self.h5ad_path = h5ad_path
        arms = list(included_arms)
        if exclude_double and DOUBLE_KO_NAME in arms:
            arms.remove(DOUBLE_KO_NAME)
        self.included_arms = tuple(arms)
        self.arm_to_label: dict[str, int] = {a: i for i, a in enumerate(self.included_arms)}

        adata = ad.read_h5ad(h5ad_path)
        if "perturbation" not in adata.obs.columns:
            raise ValueError("obs['perturbation'] missing — not a Mimitou labeled h5ad")
        if "atac_peaks" not in adata.obsm:
            raise ValueError("obsm['atac_peaks'] missing")

        perts = adata.obs["perturbation"].astype(str).values
        keep_mask = np.isin(perts, self.included_arms)
        n_kept = int(keep_mask.sum())
        if n_kept == 0:
            raise ValueError(
                f"Zero cells match included_arms {self.included_arms}. "
                f"h5ad arms: {sorted(set(perts))}"
            )

        # Cache the filtered slices for fast __getitem__
        self.atac = adata.obsm["atac_peaks"][keep_mask]
        if not sp.issparse(self.atac):
            self.atac = sp.csr_matrix(self.atac)
        if "protein" in adata.obsm:
            prot = adata.obsm["protein"][keep_mask]
            self.protein = (
                np.asarray(prot.todense()) if sp.issparse(prot) else np.asarray(prot)
            ).astype(np.float32)
        else:
            self.protein = None
        self.perturbations = perts[keep_mask]
        self.assignments = [map_arm_to_roles(a) for a in self.perturbations]
        self.arm_labels = np.array(
            [self.arm_to_label[a] for a in self.perturbations], dtype=np.int64
        )

    def __len__(self) -> int:
        return self.atac.shape[0]

    def __getitem__(self, idx: int) -> dict:
        arm = self.perturbations[idx]
        a = self.assignments[idx]
        item = {
            "cell_idx": int(idx),
            "atac_row": self.atac[idx],         # sparse CSR row (1, n_peaks)
            "perturbation": arm,
            "has_stim": a.has_stim,
            "has_inh": a.has_inh,
            "stim_id": a.stim_id,
            "inh_id": a.inh_id,
            "arm_label": int(self.arm_labels[idx]),
            "time": self.MIMITOU_TIME_MINUTES,
        }
        if self.protein is not None:
            item["protein"] = self.protein[idx]   # (n_proteins,) np.float32
        return item

    def per_arm_counts(self) -> dict[str, int]:
        """Diagnostic: per-arm cell count in the kept slice."""
        unique, counts = np.unique(self.perturbations, return_counts=True)
        return {str(u): int(c) for u, c in zip(unique, counts)}


# --- Batch sampler ---

class ArmBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches with ≥min_per_arm cells from each arm.

    SupCon requires ≥2 same-class samples per batch to compute positives.
    For Stage 3a Mimitou: synergy arm (CD3E_CD4_double) has only 74 cells
    after train split — sampler emits with replacement for under-resourced
    arms to maintain a balanced batch composition.

    Args:
        labels:        per-cell arm labels (length = dataset size)
        batch_size:    total batch size
        min_per_arm:   minimum cells per arm per batch (default 2 for SupCon)
        with_replacement: if True, under-resourced arms repeat. Default True.
        n_batches:     number of batches per epoch. Default = ceil(len/batch_size).
        seed:          rng seed.

    Iteration: each batch is a list of `batch_size` indices, with the
    arm composition described above.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        min_per_arm: int = 2,
        with_replacement: bool = True,
        n_batches: Optional[int] = None,
        seed: int = 0,
    ):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.min_per_arm = min_per_arm
        self.with_replacement = with_replacement
        self.seed = seed
        n_total = len(labels)
        self.n_batches = n_batches if n_batches is not None else max(1, (n_total + batch_size - 1) // batch_size)

        # Group indices by arm label
        self.indices_by_label: dict[int, np.ndarray] = {}
        for arm_label in np.unique(labels):
            self.indices_by_label[int(arm_label)] = np.where(labels == arm_label)[0]
        self.n_arms = len(self.indices_by_label)

        if batch_size < self.n_arms * min_per_arm:
            raise ValueError(
                f"batch_size={batch_size} too small for {self.n_arms} arms × "
                f"{min_per_arm} min_per_arm. Increase batch_size to "
                f"{self.n_arms * min_per_arm} or reduce min_per_arm."
            )

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_batches):
            batch = []
            # First, place the min per arm
            for arm_label, arm_idx in self.indices_by_label.items():
                if len(arm_idx) >= self.min_per_arm:
                    chosen = rng.choice(arm_idx, size=self.min_per_arm, replace=False)
                else:
                    chosen = rng.choice(arm_idx, size=self.min_per_arm, replace=True)
                batch.extend(chosen.tolist())

            # Fill remaining slots proportionally to arm sizes
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                arm_sizes = np.array([len(v) for v in self.indices_by_label.values()],
                                     dtype=np.float32)
                probs = arm_sizes / arm_sizes.sum()
                arm_keys = list(self.indices_by_label.keys())
                extra_arm_choices = rng.choice(len(arm_keys), size=remaining,
                                               replace=True, p=probs)
                for arm_pick in extra_arm_choices:
                    arm_idx = self.indices_by_label[arm_keys[arm_pick]]
                    batch.append(int(rng.choice(arm_idx, replace=self.with_replacement)))

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# --- Collate helpers ---

def sparse_collate(batch: list[dict]) -> dict:
    """Custom collate for MimitouDataset batches.

    Stacks ATAC rows into a single sparse CSR (B, n_peaks); other fields
    into dense tensors; perturbation strings retained as a list.
    """
    atac_stack = sp.vstack([item["atac_row"] for item in batch]).tocsr()
    out = {
        "cell_idx":  torch.tensor([item["cell_idx"] for item in batch], dtype=torch.long),
        "atac":      atac_stack,
        "perturbation": [item["perturbation"] for item in batch],
        "has_stim":  torch.tensor([item["has_stim"] for item in batch], dtype=torch.bool),
        "has_inh":   torch.tensor([item["has_inh"] for item in batch], dtype=torch.bool),
        "stim_id":   torch.tensor([item["stim_id"] for item in batch], dtype=torch.long),
        "inh_id":    torch.tensor([item["inh_id"] for item in batch], dtype=torch.long),
        "arm_label": torch.tensor([item["arm_label"] for item in batch], dtype=torch.long),
        "time":      torch.tensor([item["time"] for item in batch], dtype=torch.float32),
    }
    if "protein" in batch[0]:
        out["protein"] = torch.from_numpy(
            np.stack([item["protein"] for item in batch])
        ).float()
    return out
