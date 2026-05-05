"""Stage 3 Part 1 Report 3 — pathway annotation prep.

Pulls MSigDB Hallmark (50 sets) + KEGG immune signaling pathways
into a canonical mapping table for use by the Stage 3 pathway-aware
output head.

Pathways pulled:
  - All 50 MSigDB Hallmark gene sets (h.all.v*.symbols)
  - 8 KEGG immune signaling pathways:
      hsa04620  Toll-like receptor signaling
      hsa04064  NF-kappa B signaling
      hsa04630  Jak-STAT signaling
      hsa04010  MAPK signaling
      hsa04151  PI3K-Akt signaling
      hsa04150  mTOR signaling
      hsa04662  B cell receptor signaling
      hsa04660  T cell receptor signaling

Outputs:
  data/pathway_annotations/msigdb_hallmark.gmt
  data/pathway_annotations/kegg_immune_signaling.gmt
  data/pathway_annotations/gene_to_pathway_map.csv      # gene → pathway
  data/pathway_annotations/pathway_to_genes_map.json    # pathway → gene list
  data/pathway_annotations/pathway_metadata.json        # provenance + stats

Usage:
    pip install gseapy>=1.0 requests --break-system-packages
    python scripts/prepare_pathway_annotations.py \\
        --output_dir data/pathway_annotations
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests


# KEGG pathways aligned to CEO's CD3/BCR/TLR/NF-kB/JAK-STAT/MAPK/PI3K-AKT-mTOR list
KEGG_IMMUNE_PATHWAYS = {
    "hsa04620": "TLR_signaling",
    "hsa04064": "NFkB_signaling",
    "hsa04630": "JAK_STAT_signaling",
    "hsa04010": "MAPK_signaling",
    "hsa04151": "PI3K_AKT_signaling",
    "hsa04150": "mTOR_signaling",
    "hsa04662": "BCR_signaling",
    "hsa04660": "TCR_signaling",
}


def fetch_msigdb_hallmark(out_dir: Path) -> dict:
    """Fetch MSigDB Hallmark gene sets via gseapy.
    Returns {pathway_name: [gene_symbols]} dict.
    """
    try:
        from gseapy import Msigdb
    except ImportError:
        print("ERROR: gseapy not installed. Run: pip install gseapy", file=sys.stderr)
        sys.exit(1)

    print("Fetching MSigDB Hallmark (h.all)...")
    msig = Msigdb()
    # gseapy's Msigdb wrapper. dbver defaults to latest; pin if needed for reproducibility.
    hallmark = msig.get_gmt(category="h.all", dbver="2023.2.Hs")
    if not hallmark:
        print("WARN: hallmark dict is empty; trying alternate dbver", file=sys.stderr)
        hallmark = msig.get_gmt(category="h.all", dbver="2024.1.Hs")
    print(f"  {len(hallmark)} hallmark sets fetched")

    # Write GMT
    gmt_path = out_dir / "msigdb_hallmark.gmt"
    with open(gmt_path, "w") as f:
        for name, genes in hallmark.items():
            line = f"{name}\thttp://www.gsea-msigdb.org/gsea/msigdb/cards/{name}\t" + "\t".join(genes) + "\n"
            f.write(line)
    print(f"  Wrote {gmt_path}")

    return hallmark


def fetch_kegg_pathway_genes(pathway_id: str, retries: int = 3) -> list:
    """Pull gene symbols for a KEGG pathway via REST API.

    Endpoint: https://rest.kegg.jp/get/<pathway_id>
    Parses GENE section, extracts symbols (e.g., "TLR4").
    """
    url = f"https://rest.kegg.jp/get/{pathway_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} after error: {e}", file=sys.stderr)
            time.sleep(2)
    text = r.text

    # Parse GENE section (lines starting with "GENE" or whitespace + entry id + symbol)
    genes = []
    in_gene_section = False
    for line in text.splitlines():
        if line.startswith("GENE"):
            in_gene_section = True
            line = line[len("GENE"):]
        elif in_gene_section and not line.startswith(" "):
            # New section started (e.g., "COMPOUND", "REL_PATHWAY")
            break
        if in_gene_section:
            # KEGG GENE line format: "  <entrez_id>  <symbol>; <description>"
            m = re.match(r"\s+\d+\s+(\S+?);", line)
            if m:
                sym = m.group(1).strip()
                if sym:
                    genes.append(sym)
    return genes


def fetch_kegg_immune(out_dir: Path) -> dict:
    """Pull all 8 KEGG immune pathways via REST API."""
    print("\nFetching KEGG immune signaling pathways via REST API...")
    kegg = {}
    for pid, name in KEGG_IMMUNE_PATHWAYS.items():
        try:
            genes = fetch_kegg_pathway_genes(pid)
            kegg[f"KEGG_{name}"] = sorted(set(genes))
            print(f"  {pid} ({name}): {len(kegg[f'KEGG_{name}'])} unique gene symbols")
            time.sleep(0.5)  # polite to KEGG; <1 req/sec
        except Exception as e:
            print(f"  ERROR fetching {pid}: {e}", file=sys.stderr)

    gmt_path = out_dir / "kegg_immune_signaling.gmt"
    with open(gmt_path, "w") as f:
        for name, genes in kegg.items():
            line = f"{name}\thttps://www.kegg.jp/pathway/{name.replace('KEGG_','')}\t" + "\t".join(genes) + "\n"
            f.write(line)
    print(f"  Wrote {gmt_path}")
    return kegg


def build_gene_to_pathway_map(hallmark: dict, kegg: dict) -> tuple[dict, dict]:
    """Build canonical {gene: [pathways]} and {pathway: [genes]} maps."""
    pathway_to_genes = {}
    gene_to_pathways = {}
    for src, sets in (("hallmark", hallmark), ("kegg", kegg)):
        for name, genes in sets.items():
            pathway_to_genes[name] = list(genes)
            for g in genes:
                gene_to_pathways.setdefault(g, []).append(name)
    return gene_to_pathways, pathway_to_genes


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. MSigDB Hallmark ---
    hallmark = fetch_msigdb_hallmark(args.output_dir)

    # --- 2. KEGG immune ---
    kegg = fetch_kegg_immune(args.output_dir)

    # --- 3. Build mapping ---
    gene_to_pw, pw_to_genes = build_gene_to_pathway_map(hallmark, kegg)
    print(f"\nUnique genes covered: {len(gene_to_pw)}")
    print(f"Total pathways: {len(pw_to_genes)} ({len(hallmark)} hallmark + {len(kegg)} kegg)")

    # --- 4. Outputs ---
    # 4a. CSV: gene_symbol, pathway_name, source_db
    csv_path = args.output_dir / "gene_to_pathway_map.csv"
    with open(csv_path, "w") as f:
        f.write("gene_symbol,pathway_name,source_db\n")
        for gene, pathways in sorted(gene_to_pw.items()):
            for p in pathways:
                src = "hallmark" if p in hallmark else "kegg"
                f.write(f"{gene},{p},{src}\n")
    print(f"Wrote {csv_path}")

    # 4b. JSON: pathway → genes
    json_path = args.output_dir / "pathway_to_genes_map.json"
    with open(json_path, "w") as f:
        json.dump(pw_to_genes, f, indent=2)
    print(f"Wrote {json_path}")

    # 4c. Metadata
    meta = {
        "build_date_utc": "2026-05-04",
        "hallmark_n": len(hallmark),
        "kegg_n": len(kegg),
        "kegg_pathways": KEGG_IMMUNE_PATHWAYS,
        "n_unique_genes": len(gene_to_pw),
        "sources": {
            "hallmark": "https://www.gsea-msigdb.org/gsea/msigdb/ (h.all collection)",
            "kegg": "https://rest.kegg.jp/get/<pathway_id>",
        },
        "note": (
            "Pathway-aware output head for Stage 3 perturbation predictor. "
            "Phospho readouts (QurieSeq Phase 2) align to these pathways at "
            "the readout level (pJAK1 → JAK_STAT, pERK → MAPK, etc.)."
        ),
    }
    meta_path = args.output_dir / "pathway_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
