"""
Housekeeping gene set for safe cross-cell-type W matrix pretraining.

These genes have consistent regulatory relationships in K562 and PBMCs.
Used to filter Replogle 2022 CRISPRi knockdowns before Neumann W pretraining.

Sources:
  1. MSigDB HALLMARK_HOUSEKEEPING (200 curated genes)
  2. Eisenberg & Levanon 2013 (doi:10.1016/j.tig.2013.05.010)
  3. Manual curation of core cellular machinery
"""
import logging

logger = logging.getLogger("aivc.data")

# ─── SECTION 1: Hard-coded safe housekeeping genes ───

# Ribosomal large subunit proteins (RPL*)
RPL_GENES = [
    "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL7A", "RPL8",
    "RPL9", "RPL10", "RPL10A", "RPL11", "RPL12", "RPL13",
    "RPL13A", "RPL14", "RPL15", "RPL17", "RPL18", "RPL18A",
    "RPL19", "RPL21", "RPL22", "RPL23", "RPL23A", "RPL24",
    "RPL26", "RPL27", "RPL27A", "RPL28", "RPL29", "RPL30",
    "RPL31", "RPL32", "RPL34", "RPL35", "RPL35A", "RPL36",
    "RPL36A", "RPL37", "RPL37A", "RPL38", "RPL39", "RPL41",
]

# Ribosomal small subunit proteins (RPS*)
RPS_GENES = [
    "RPS2", "RPS3", "RPS3A", "RPS4X", "RPS5", "RPS6",
    "RPS7", "RPS8", "RPS9", "RPS10", "RPS11", "RPS12",
    "RPS13", "RPS14", "RPS15", "RPS15A", "RPS16", "RPS17",
    "RPS18", "RPS19", "RPS20", "RPS21", "RPS23", "RPS24",
    "RPS25", "RPS26", "RPS27", "RPS27A", "RPS28", "RPS29",
]

# Core splicing factors
SPLICING_GENES = [
    "SNRPA", "SNRPA1", "SNRPB", "SNRPB2", "SNRPC", "SNRPD1",
    "SNRPD2", "SNRPD3", "SNRPE", "SNRPF", "SNRPG",
    "SF3A1", "SF3A2", "SF3A3", "SF3B1", "SF3B2", "SF3B3",
    "SF3B4", "SF3B5", "SF3B6",
    "PRPF3", "PRPF4", "PRPF6", "PRPF8", "PRPF19", "PRPF31",
    "U2AF1", "U2AF2",
]

# Translation initiation factors (EIF*)
EIF_GENES = [
    "EIF1", "EIF1A", "EIF1AX", "EIF2A", "EIF2B1", "EIF2B2",
    "EIF2B3", "EIF2B4", "EIF2B5", "EIF2S1", "EIF2S2", "EIF2S3",
    "EIF3A", "EIF3B", "EIF3C", "EIF3D", "EIF3E", "EIF3F",
    "EIF3G", "EIF3H", "EIF3I", "EIF3J", "EIF3K", "EIF3L",
    "EIF3M", "EIF4A1", "EIF4A2", "EIF4B", "EIF4E", "EIF4G1",
    "EIF4H", "EIF5", "EIF5A", "EIF5B", "EIF6",
]

# Core RNA polymerase II subunits
POLR2_GENES = [
    "POLR2A", "POLR2B", "POLR2C", "POLR2D", "POLR2E",
    "POLR2F", "POLR2G", "POLR2H", "POLR2I", "POLR2J",
    "POLR2K", "POLR2L",
]

# General transcription factors
GTF_GENES = [
    "TBP", "TAF1", "TAF2", "TAF3", "TAF4", "TAF5",
    "TAF6", "TAF7", "TAF9", "TAF10", "TAF11", "TAF12",
    "TAF13", "TAF15",
]

# 26S Proteasome subunits
PROTEASOME_GENES = [
    "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7",
    "PSMB1", "PSMB2", "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7",
    "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
    "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD6", "PSMD7",
    "PSMD8", "PSMD11", "PSMD12", "PSMD13", "PSMD14",
]

# Core DNA replication machinery
DNA_REPLICATION_GENES = [
    "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "RFC1", "RFC2", "RFC3", "RFC4", "RFC5",
    "POLA1", "POLA2", "POLE", "POLE2", "POLE3", "POLE4",
    "PRIM1", "PRIM2",
]

# Core histones
HISTONE_GENES = [
    "HIST1H1C", "HIST1H1D", "HIST1H1E",
    "H2AFX", "H2AFZ",
    "HIST1H2AB", "HIST1H2AC", "HIST1H2AD",
    "HIST1H3A", "HIST1H3B", "HIST1H4A", "HIST1H4B",
]

# Chaperones and protein folding
CHAPERONE_GENES = [
    "HSPA5", "HSPA8", "HSP90AA1", "HSP90AB1",
    "HSPD1", "HSPE1",
    "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8",
    "CANX", "CALR",
]

# Ubiquitin pathway (core E1/E2 enzymes)
UBIQUITIN_GENES = [
    "UBA1", "UBA2", "UBA52",
    "UBB", "UBC",
    "UBE2A", "UBE2B", "UBE2D1", "UBE2D2", "UBE2D3",
    "UBE2E1", "UBE2K", "UBE2L3", "UBE2N",
]

# Mitochondrial core metabolism
MITO_GENES = [
    "MT-CO1", "MT-CO2", "MT-CO3",
    "MT-ND1", "MT-ND2", "MT-ND4", "MT-ND5",
    "MT-CYB",
    "UQCRC1", "UQCRC2", "UQCRFS1",
    "COX4I1", "COX5A", "COX5B", "COX6A1", "COX7A2",
]

# ─── SECTION 2: Genes EXPLICITLY BLOCKED from W pretraining ───

JAK_STAT_BLOCKED = [
    "JAK1", "JAK2", "JAK3", "TYK2",
    "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
    "IRF1", "IRF3", "IRF7", "IRF9",
    "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM3",
    "MX1", "MX2", "ISG15", "ISG20", "OAS1", "OAS2", "OAS3",
]

BCR_ABL_BLOCKED = [
    "ABL1", "BCR", "CRKL", "GRB2", "SOS1",
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1",
    "AKT1", "AKT2", "AKT3",
]

ONCOGENE_BLOCKED = [
    "MYC", "MYCN", "MYCL",
    "TP53", "RB1", "CDKN2A",
    "BCL2", "BCL2L1", "BCL2L11", "MCL1",
    "CDK4", "CDK6", "CCND1", "CCND2", "CCND3",
    "MDM2", "MDM4",
    "KRAS", "NRAS", "HRAS", "BRAF",
]

IMMUNE_BLOCKED = [
    "IFNAR1", "IFNAR2", "IFNGR1", "IFNGR2",
    "IL6ST", "IL6R", "IL2RA", "IL2RB", "IL2RG",
    "TNFRSF1A", "TNFRSF1B",
    "TLR3", "TLR4", "TLR7", "TLR9",
    "MHC", "HLA-A", "HLA-B", "HLA-C",
]

ALL_BLOCKED_GENES = set(
    JAK_STAT_BLOCKED + BCR_ABL_BLOCKED + ONCOGENE_BLOCKED + IMMUNE_BLOCKED
)

# ─── SECTION 3: Assembled safe set ───

HOUSEKEEPING_SAFE_SET = set(
    RPL_GENES + RPS_GENES + SPLICING_GENES + EIF_GENES +
    POLR2_GENES + GTF_GENES + PROTEASOME_GENES +
    DNA_REPLICATION_GENES + HISTONE_GENES +
    CHAPERONE_GENES + UBIQUITIN_GENES + MITO_GENES
)

# ─── SECTION 4: Public API ───


def get_housekeeping_genes(source: str = "builtin") -> set:
    """
    Return the housekeeping gene set.

    Args:
        source: "builtin" (default, no deps) or "msigdb" (via gseapy).
    """
    if source == "msigdb":
        try:
            return _load_msigdb_housekeeping()
        except Exception as e:
            logger.warning(f"MSigDB load failed ({e}). Using built-in set.")
    return HOUSEKEEPING_SAFE_SET.copy()


def _load_msigdb_housekeeping() -> set:
    """Load MSigDB HALLMARK_HOUSEKEEPING via gseapy."""
    import gseapy as gp
    gene_sets = gp.get_library(name="MsigDB_Hallmark_2020", organism="Human")
    hk_genes = set()
    for set_name, genes in gene_sets.items():
        if any(k in set_name.upper() for k in ["HOUSEKEEP", "CONSTITUTIVE"]):
            hk_genes.update(genes)
    if len(hk_genes) < 50:
        raise ValueError(f"MSigDB returned only {len(hk_genes)} housekeeping genes.")
    return hk_genes


def filter_ko_genes_for_w_pretrain(
    ko_genes: list,
    gene_universe: list = None,
    source: str = "builtin",
    verbose: bool = True,
) -> dict:
    """
    Filter knockdown genes to the safe housekeeping set.

    Logic:
      1. Intersect ko_genes with HOUSEKEEPING_SAFE_SET
      2. Remove any genes in ALL_BLOCKED_GENES
      3. If gene_universe provided: restrict to genes in the model

    Returns dict with safe_ko_genes, counts, and filter_report.
    """
    hk_set = get_housekeeping_genes(source=source)
    blocked = ALL_BLOCKED_GENES

    n_input = len(ko_genes)
    blocked_genes = []
    not_hk_genes = []
    not_in_universe = []
    safe_genes = []

    for g in ko_genes:
        if g in blocked:
            blocked_genes.append(g)
            continue
        if g not in hk_set:
            not_hk_genes.append(g)
            continue
        if gene_universe is not None and g not in set(gene_universe):
            not_in_universe.append(g)
            continue
        safe_genes.append(g)

    n_safe = len(safe_genes)
    n_blocked = len(blocked_genes)
    n_not_hk = len(not_hk_genes)
    n_not_universe = len(not_in_universe)

    report = (
        f"Replogle KO gene filter:\n"
        f"  Input:           {n_input:>5} knockdown genes\n"
        f"  Hard blocked:    {n_blocked:>5} (JAK-STAT/BCR-ABL/oncogenes/immune)\n"
        f"  Not housekeeping:{n_not_hk:>5}\n"
        f"  Not in universe: {n_not_universe:>5}\n"
        f"  SAFE for W train:{n_safe:>5}\n"
    )

    if n_safe == 0:
        report += (
            "  WARNING: 0 safe genes found. "
            "Check that Replogle gene names match HGNC symbols.\n"
            "  W pretraining will be skipped."
        )

    if verbose:
        logger.info(report)

    return {
        "safe_ko_genes": safe_genes,
        "n_input": n_input,
        "n_safe": n_safe,
        "n_blocked": blocked_genes,
        "n_not_hk": n_not_hk,
        "n_not_in_universe": n_not_universe,
        "filter_report": report,
    }


def get_blocked_jakstat_genes() -> set:
    """Return the JAK-STAT blocked gene set."""
    return set(JAK_STAT_BLOCKED)


def get_safe_set_size() -> int:
    """Return the size of the built-in housekeeping safe set."""
    return len(HOUSEKEEPING_SAFE_SET)
