"""
Demo: SCM do(X) interventions on AIVC Neumann W matrix.

Uses mock W (is_valid=False) to demonstrate the interface.
After v1.1 training: replace with trained checkpoint for real results.

Usage:
  python aivc/skills/scm_demo.py
"""
import torch
import torch.nn as nn
from aivc.skills.neumann_propagation import NeumannPropagation
from aivc.skills.scm_engine import SCMEngine

SEED = 42
N_GENES = 20


def run():
    torch.manual_seed(SEED)

    gene_names = [
        "JAK1", "JAK2", "TYK2", "STAT1", "STAT2", "STAT3",
        "IRF9", "IFIT1", "IFIT2", "MX1", "MX2", "OAS2", "ISG15",
        "GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F", "GENE_G",
    ]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Build edges including JAK1->STAT1 and STAT1->IFIT1
    edges = [
        [0, 3],   # JAK1 -> STAT1
        [1, 3],   # JAK2 -> STAT1
        [3, 7],   # STAT1 -> IFIT1
        [3, 9],   # STAT1 -> MX1
        [3, 12],  # STAT1 -> ISG15
        [4, 6],   # STAT2 -> IRF9
        [6, 7],   # IRF9 -> IFIT1
    ]
    # Add some random edges
    rng_edges = torch.randint(0, N_GENES, (2, 50))
    src = torch.cat([torch.tensor([e[0] for e in edges]), rng_edges[0]])
    dst = torch.cat([torch.tensor([e[1] for e in edges]), rng_edges[1]])
    edge_index = torch.stack([src, dst])

    neumann = NeumannPropagation(N_GENES, edge_index, K=3, lambda_l1=0.001)

    # Set known biological edges to meaningful weights
    with torch.no_grad():
        for i in range(len(neumann.W)):
            s, d = neumann.edge_src[i].item(), neumann.edge_dst[i].item()
            if s == 0 and d == 3:  # JAK1->STAT1
                neumann.W[i] = 0.05
            elif s == 3 and d == 7:  # STAT1->IFIT1
                neumann.W[i] = 0.03
            elif s == 3 and d == 9:  # STAT1->MX1
                neumann.W[i] = 0.02

    decoder = nn.Linear(N_GENES, N_GENES)
    engine = SCMEngine(neumann, decoder, gene_names=gene_names,
                       min_w_density=0.05)

    ctrl_expr = torch.rand(N_GENES) * 2.0

    print("\n" + "=" * 60)
    print("AIVC SCM Engine Demo — do(X) Causal Interventions")
    print("=" * 60)

    # 1. JAK1 KO
    print("\n--- do(JAK1 = KO) ---")
    r = engine.do_gene_ko("JAK1", ctrl_expr)
    print(f"  is_valid: {r.is_valid}")
    print(f"  Top 5 affected genes:")
    for g in r.top_affected_genes[:5]:
        print(f"    {g['gene_name']:<10} delta={g['delta']:+.4f} ({g['direction']})"
              + (" *JAK-STAT*" if g['is_jakstat'] else ""))
    if r.causal_path:
        print(f"  Causal path edges cut:")
        for e in r.causal_path[:3]:
            print(f"    {e['src']}->{e['dst']} (cut={e['is_cut']}, w={e['w_weight']:.4f})")
    for w in r.warnings:
        print(f"  WARNING: {w}")

    # 2. Pathway block
    print("\n--- do([JAK1, JAK2, TYK2] = pathway block) ---")
    r2 = engine.do_pathway_block(["JAK1", "JAK2", "TYK2"], ctrl_expr)
    print(f"  is_valid: {r2.is_valid}")
    print(f"  Genes affected (|delta| > 0.01): {int((r2.delta.abs() > 0.01).sum())}")
    for g in r2.top_affected_genes[:3]:
        print(f"    {g['gene_name']:<10} delta={g['delta']:+.4f}")

    # 3. STAT1 OE
    print("\n--- do(STAT1 = 5x overexpression) ---")
    r3 = engine.do_gene_oe("STAT1", fold_change=5.0, ctrl_expr=ctrl_expr)
    print(f"  is_valid: {r3.is_valid}")
    for g in r3.top_affected_genes[:3]:
        print(f"    {g['gene_name']:<10} delta={g['delta']:+.4f}")

    # 4. ATAC stub
    print("\n--- do(peak chr2:48700000 = closed) ---")
    r4 = engine.do_peak_closed("chr2:48700000 near JAK1", ctrl_expr)
    print(f"  is_valid: {r4.is_valid}")
    for w in r4.warnings:
        print(f"  {w}")

    # 5. Full report
    print("\n--- Counterfactual Report ---")
    report = engine.counterfactual_report(ctrl_expr)
    print(report["summary"])

    print(f"\n{'='*60}")
    print("SCM engine ready. Run v1.1 training to get valid results.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
