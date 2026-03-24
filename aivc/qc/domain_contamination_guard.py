"""
Domain contamination guard for perturbation response training set.

Blocks training if non-PBMC data (melanoma, cancer cell lines) is present
in the response training corpus. Runs BEFORE Stage 2+ training starts.

PBMC-ALLOWED: CD4+ T cells, CD8+ T cells, NK cells, B cells, Monocytes, DC, etc.
BLOCKED: melanoma, tumor, cancer, K562, HeLa, Jurkat, TIL, etc.
"""
import logging

import numpy as np

logger = logging.getLogger("aivc.qc.domain")


class DomainContaminationGuard:
    """
    Validates that response training data contains only PBMC-compatible
    cell types. Blocks training if non-PBMC contamination is detected.
    """

    PBMC_ALLOWED = {
        "cd4+ t cells", "cd4 t cells", "cd4t", "cd4",
        "cd8+ t cells", "cd8 t cells", "cd8t", "cd8",
        "nk cells", "natural killer", "nk",
        "b cells", "b cell", "b_cell",
        "cd14+ monocytes", "cd14 monocytes", "monocytes",
        "cd16+ monocytes", "cd16 monocytes",
        "fcgr3a+ monocytes",
        "dendritic cells", "dc", "myeloid dc", "plasmacytoid dc", "pdc",
        "megakaryocytes", "platelet",
        "regulatory t cells", "treg",
        "hematopoietic stem cells", "hsc",
        "pbmc", "immune", "unknown",
        "t cell", "t cells",
    }

    BLOCKED_PATTERNS = [
        "melanoma", "tumor", "tumour", "cancer",
        "mcf7", "k562", "rpe1", "hek293", "hela", "jurkat", "a549",
        "til",  # tumor-infiltrating lymphocyte (melanoma context)
        "malignant",
    ]

    def check(self, combined_corpus, raise_on_contamination: bool = True) -> dict:
        """
        Check all response-training cells for PBMC compatibility.

        Args:
            combined_corpus:        AnnData with all cells.
            raise_on_contamination: If True, raises ValueError on contamination.

        Returns:
            dict with clean, contaminated_types, n_contaminated,
            contaminated_datasets, message.
        """
        # Only check response training cells
        if "use_for_response_training" in combined_corpus.obs.columns:
            response_mask = combined_corpus.obs["use_for_response_training"] == True
        elif "USE_FOR_W_ONLY" in combined_corpus.obs.columns:
            response_mask = combined_corpus.obs["USE_FOR_W_ONLY"] == False
        else:
            response_mask = np.ones(combined_corpus.n_obs, dtype=bool)

        if "cell_type" not in combined_corpus.obs.columns:
            return {
                "clean": True,
                "contaminated_types": [],
                "n_contaminated": 0,
                "contaminated_datasets": [],
                "message": "No cell_type column found. Skipping contamination check.",
            }

        response_adata = combined_corpus[response_mask]
        cell_types = response_adata.obs["cell_type"].astype(str).str.lower().unique()

        contaminated = []
        for ct in cell_types:
            for pattern in self.BLOCKED_PATTERNS:
                if pattern in ct:
                    contaminated.append(ct)
                    break

        contaminated_datasets = []
        n_contaminated = 0
        if contaminated:
            for ct in contaminated:
                ct_mask = response_adata.obs["cell_type"].astype(str).str.lower() == ct
                n_contaminated += int(ct_mask.sum())
                if "dataset_id" in response_adata.obs.columns:
                    ds_ids = response_adata.obs.loc[
                        ct_mask.values, "dataset_id"
                    ].unique().tolist()
                    contaminated_datasets.extend(ds_ids)

        clean = len(contaminated) == 0
        message = (
            f"Domain contamination check: {'CLEAN' if clean else 'CONTAMINATED'}. "
            f"Response training cells: {int(response_mask.sum())}. "
            + (
                f"Blocked cell types found: {contaminated}. "
                f"n_contaminated: {n_contaminated}."
                if not clean
                else "No non-PBMC cell types detected."
            )
        )

        if not clean and raise_on_contamination:
            raise ValueError(
                f"DOMAIN CONTAMINATION DETECTED in response training set.\n"
                f"Blocked cell types: {contaminated}\n"
                f"Remove these cells before training.\n"
                f"Affected datasets: {set(contaminated_datasets)}\n"
                f"Fix: Set USE_FOR_W_ONLY=True or exclude these cells."
            )

        return {
            "clean": clean,
            "contaminated_types": contaminated,
            "n_contaminated": n_contaminated,
            "contaminated_datasets": list(set(contaminated_datasets)),
            "message": message,
        }
