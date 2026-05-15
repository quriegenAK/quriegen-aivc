# Stage 3 Part 1 — Encoder Probe Verdict (2026-05-11)

## Verdict
**ADAPTER_RECOMMENDED** — Synergy 4-class accuracy 0.5676 vs chance 0.2500 (2.27× chance).
Per pre-registered thresholds: 0.50–0.80 = adapter (Linear+LN+GELU on top of frozen latent).

## Architecture Implications
- Frozen DOGMA encoder preserved (no retraining)
- Add lightweight adapter: ~130K params at d=256
- BTK+JAK synergy demo target mathematically viable via CD3E+CD4 substitute (74 cells post-split)
- Architecture invariants intact: temporal state-transition system, decomposed residual heads, donor-specific static ATAC context, pathway-aware output head

## Probe Numbers (Mimitou ASAP-seq CRISPR, GSE156478)
- Per-class accuracy: CD3E 0.91, CD3E_CD4_double 0.68, NTC 0.39, CD4 0.39
- Random projection baseline (sanity): 0.29
- Raw TF-IDF baseline (ceiling): 0.50
- Peak overlap (post-fix): 77,267/94,838 arm peaks → DOGMA union)

## Methodology Lessons Banked
1. Exact-string peak matching fails for cross-corpus ATAC — use genomic interval overlap (PyRanges-style sweep)
2. Cell Ranger barcodes use `-N` lane suffix; kite outputs do not — normalize before intersection
3. Cell Ranger ATAC h5 layout is features × cells (csc), not cells × features
4. HTO-based perturbation calling: top-1 target + MIN_TARGET_HTO_COUNT=5 floor; CD3E+CD4 double via co-tagging (Hashtag02+Hashtag04)
5. UNION_MANIFEST.json summary file is NOT the peak list — peaks live in dogma_lll_union_labeled.h5ad as var_names

## Status Across Reports
- Report 1 (datasets): Partial — CIPHER-seq killed, L3 reframed to B-cell CRISPR, L4 → QurieSeq direct
- Report 2 (encoder probe): COMPLETE — this verdict
- Report 3 (pathways): COMPLETE — 50 hallmark + 8 KEGG, 4798 unique genes
- Report 4 (BTK+JAK feasibility): COMPLETE — infeasible on public, CD3E+CD4 substitute
- Report 5 (B-cell investigation): COMPLETE — stand-## Production Artifacts
- Encoder ckpt: /gpfs/scratch/ehpc748/quri020505/checkpoints/pretrain/pretrain_encoders.pt
- Class manifest fingerprint: f4c7dc2136bb77fb2d762363a61a93b43468d59bdb6f90047bb914505dfbb8f2
- Probe verdict JSON: /gpfs/scratch/ehpc748/quri020505/results/stage3_prep/mimitou_perturbation_probe.json

## What Unblocks
- Stage 3 Part 2 architecture spec finalized (commit b0fdacf, v1.1)
- Stage 3a Day 1 PR landed (commit 87d6a9a) — adapter, decomposed_readout, pathway pool builder
- Stage 3a Day 2 PR landed (commit aca6b09) — pathway decoder, perturbation loss, Mimitou loader
- Day 3 unblocked: training script + eval script + load_stage3_ckpt_raw refactor
