"""
AIVC v3.0 Preprocessing — ATAC+RNA multiome pipeline.

Replaces Wout Megchelenbrink's R-based Seurat/Signac pipeline
with a production-grade Python implementation using muon/scanpy.
"""
from .atac_rna_pipeline import ATACRNAPipeline
from .validate_atac_pipeline import (
    validate_jakstat_coverage,
    validate_motif_enrichment_direction,
    validate_cell_type_retention,
)

__all__ = [
    "ATACRNAPipeline",
    "validate_jakstat_coverage",
    "validate_motif_enrichment_direction",
    "validate_cell_type_retention",
]
