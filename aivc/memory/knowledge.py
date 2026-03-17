"""
aivc/memory/knowledge.py — Biological databases and papers.

Stores known biological facts used by critics and plausibility scoring.
Hardcoded ground truth from literature for JAK-STAT pathway.
API integration stubs for KEGG and Reactome.
"""

from typing import Optional


class KnowledgeBase:
    """
    Biological knowledge store.
    Provides ground truth for validation and plausibility scoring.
    """

    # JAK-STAT pathway ground truth interactions (from literature)
    JAKSTAT_INTERACTIONS = {
        ("JAK1", "STAT1"), ("JAK1", "STAT2"), ("JAK2", "STAT1"),
        ("STAT1", "IRF9"), ("STAT2", "IRF9"), ("IRF9", "IFIT1"),
        ("IRF9", "ISG15"), ("IRF9", "MX1"), ("STAT1", "MX1"),
    }

    # JAK-STAT pathway genes (must be included in all analyses)
    JAKSTAT_GENES = [
        "JAK1", "JAK2", "STAT1", "STAT2", "STAT3", "IRF9", "IRF1",
        "MX1", "MX2", "ISG15", "OAS1", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
    ]

    # Known IFN-beta responsive genes with expected fold change direction
    IFNB_RESPONSE = {
        "IFIT1": {"direction": "up", "expected_fc_range": (30, 120)},
        "ISG15": {"direction": "up", "expected_fc_range": (10, 60)},
        "MX1": {"direction": "up", "expected_fc_range": (5, 40)},
        "OAS1": {"direction": "up", "expected_fc_range": (5, 30)},
        "IFIT3": {"direction": "up", "expected_fc_range": (10, 50)},
        "STAT1": {"direction": "up", "expected_fc_range": (2, 8)},
        "IRF1": {"direction": "up", "expected_fc_range": (2, 10)},
    }

    # Cell type expected response hierarchy for IFN-beta
    # Monocytes are strongest responders
    CELL_TYPE_RESPONSE_ORDER = [
        "CD14+ Monocytes",
        "FCGR3A+ Monocytes",
        "Dendritic cells",
        "NK cells",
        "CD4 T cells",
        "CD8 T cells",
        "B cells",
        "Megakaryocytes",
    ]

    def is_jakstat_interaction(self, gene_a: str, gene_b: str) -> bool:
        """Check if a gene pair is a known JAK-STAT interaction."""
        return (
            (gene_a, gene_b) in self.JAKSTAT_INTERACTIONS
            or (gene_b, gene_a) in self.JAKSTAT_INTERACTIONS
        )

    def get_expected_direction(self, gene: str) -> Optional[str]:
        """Return expected fold change direction for IFN-beta response."""
        info = self.IFNB_RESPONSE.get(gene)
        return info["direction"] if info else None

    def query_kegg(self, pathway_id: str) -> list[tuple[str, str]]:
        """
        PLACEHOLDER: Query KEGG REST API for pathway interactions.
        Will be implemented when KEGG API access is configured.
        """
        # TODO: Implement KEGG REST API query
        # URL: https://rest.kegg.jp/get/{pathway_id}/kgml
        return []

    def query_reactome(self, pathway_name: str) -> list[tuple[str, str]]:
        """
        PLACEHOLDER: Query Reactome API for pathway annotations.
        Will be implemented when Reactome API access is configured.
        """
        # TODO: Implement Reactome content service API query
        # URL: https://reactome.org/ContentService/data/pathway/{id}/containedEvents
        return []
