# Stage 3 Part 1 — Five Prep Reports

**Date**: 2026-05-04
**Status**: Reports 1, 3, 4 complete (research/analysis). Reports 2, 5 scoped with
runnable helper scripts; require BSC execution against the validated DOGMA
encoder + downloaded data.
**Predecessor**: CEO Stage 3 design context update 2026-05-04 (QurieSeq structure
locked, 4-arm decomposed-residual formulation locked, ATAC-static-context locked,
BTK+JAK demo target locked).
**Architecture authorization**: PAUSED until Part 1 returns.

---

## Report 1: Public dataset access confirmation

### Mimitou ASAP-seq CRISPR (GSE156478, CD4 perturbation arm)

| Field | Value |
|---|---|
| Accession | GSE156478 (parent record; CRISPR arm = subset) |
| License | Open access (NCBI GEO public) |
| Cells | ~5,825 perturbed (per Mimitou 2021 §"Multiplexed CRISPR perturbations") |
| Modalities | ATAC peaks + 210-D TotalSeq-A protein + hashtag (no RNA in this arm) |
| Perturbations | 5 hard CRISPR (CD3E, CD4, ZAP70, NFKB2, CD3E+CD4 double); soft anti-CD3/CD28 stim; rest + 16h regime |
| Time-resolved | NO — single endpoint (16h post-stim) per perturbation arm |
| File format | Cell Ranger ATAC outputs (filtered_peak_bc_matrix, fragments.tsv.gz) + kite-format protein counts (featurecounts.mtx.gz, .barcodes.txt, .genes.txt) |
| Estimated size | ~3-5 GB compressed (raw fragments + peak matrices + protein counts) |
| BSC processing time | ~2-3 days for download + harmonization to DOGMA union peak set (323,500 peaks) + compatibility validation |
| **Status on BSC** | **NOT YET DEPLOYED — needs prep helper** (sketched below) |

**Key consideration**: the CD4 CRISPR arm uses ATAC + Protein only, NOT trimodal
RNA + ATAC + Protein. Our DOGMA encoder accepts trimodal input — at inference
we'd zero-pad RNA (Pivot A pattern, with the masked-fusion variant we
defined in PR #54c). Or we use the bare ATAC encoder path which we
validated at 0.7308 in Test 1.5 and which doesn't require RNA.

**Pivot A lesson preserves**: zero-padded modalities dilute the joint
embedding. For perturbation eval on this dataset, use the **bare ATAC
encoder path** (Test 1.5 methodology), not joint-fusion.

### Parse GigaLab Cytokine Atlas (10M PBMC × 90 cytokines)

| Field | Value |
|---|---|
| Source | Allen Immunology + Parse Biosciences. bioRxiv preprint 2025.12.12.693897 |
| License | Open access (Parse Biosciences open data; Allen Institute commitment to open release) |
| Cells | 9,697,974 (~10M) |
| Modalities | RNA only (whole transcriptome via Parse split-pool combinatorial barcoding) |
| Perturbations | 90 cytokines + PBS control = 91 conditions × 12 donors = 1,092 condition × donor pairs |
| Time-resolved | NO — single 24h endpoint |
| File format | h5ad (scanpy/anndata) is the typical Allen/Parse release format; per-cytokine subsets available |
| Estimated size | ~50-150 GB raw (10M cells × ~20K genes); processed h5ad ~10-30 GB (sparse) |
| BSC processing time | ~3 days for download + Azimuth label cross-walk + integration with our DOGMA cell-type manifest fingerprint (f4c7dc...) |
| **Status on BSC** | **NOT YET DEPLOYED — large download** |

**Critical compatibility advantage**: Parse used Azimuth for cell-type
annotation (`predicted.celltype.l1`), the same as our DOGMA training
labels. Direct cell-type mapping; no relabeling required.

**Critical limitation for Stage 3a**: RNA only. Our DOGMA encoder is
trimodal. To use Parse cells through the DOGMA encoder we'd need to
zero-pad ATAC + Protein. This triggers the same dilution issue Pivot A
revealed — our joint embedding for RNA-only Parse cells will collapse
toward the all-zero-modality direction.

**Two viable paths**:
- (a) Use the **RNA encoder branch alone** for Parse cells. Project to
  proj_dim. Compare against pseudo-bulk RNA-only DOGMA centroids
  (re-aggregated from labeled training cells, RNA-only path).
- (b) Train a **light RNA-only adapter** on the ~13.7K labeled DOGMA
  cells (LLL arm) so RNA-alone embeddings land in the same space as the
  trimodal embedding. ~1-2 days of work.

Option (a) is simpler; recommend that for Stage 3a Part 2.

### CIPHER-seq stimulated PBMC (Goh et al. *Sci Reports* April 2026)

| Field | Value |
|---|---|
| Accession | **NEEDS VERIFICATION** — search did not surface a GEO accession in result snippets. The paper at `https://www.nature.com/articles/s41598-026-44946-y` should have a Data Availability section with the accession; manual paper read required |
| License | Open access (Nature Scientific Reports default) |
| Cells | thousands per condition (exact n_cells per condition not in search snippets) |
| Modalities | RNA + intracellular protein (cytokines, signaling) — 2/4. Not chromatin. Cytokine antibodies in panel; phospho-specific antibodies not the explicit focus |
| Perturbations | Soft (TCR stim, LPS, cytokine cocktails) — no hard CRISPR |
| Time-resolved | YES — paper explicitly demonstrates RNA-rises-first-then-protein temporal ordering. Exact timepoints not visible in snippets but the time-resolution claim is the methodological centerpiece |
| File format | Likely h5ad / GEO standard sequencer output |
| Estimated size | Unknown without reading paper; likely 1-10 GB |
| BSC processing time | ~1-2 days; small dataset relative to Parse |
| **Status on BSC** | NOT YET DEPLOYED |

**Action item**: read the Data Availability section of
`https://www.nature.com/articles/s41598-026-44946-y` to confirm
accession + exact timepoint structure. Adds <1 hour but blocks
ingestion.

### Soskic 2025 — primary CD4 T autoimmune CRISPRi (*Nat Genet* s41588-025-02301-3)

| Field | Value |
|---|---|
| Accession | **NEEDS VERIFICATION** — search did not surface accession in snippets. Paper at `https://www.nature.com/articles/s41588-025-02301-3`; check Data Availability section |
| Related preprint | bioRxiv 2025.12.23.696273 ("Genome-scale perturb-seq in primary human CD4+ T cells maps context-specific regulators") may be a related / superseding study |
| License | Open access (Nature default) |
| Cells | thousands; primary human CD4+ T cells |
| Modalities | bulk + sc CRISPRi screens; bulk RNA-seq + sc Perturb-seq output (RNA only) ± ATAC for the epigenetic-screen arm |
| Perturbations | Hundreds of autoimmune-candidate variants via CRISPRi; activation context |
| Time-resolved | Activation time-course (resting → stimulated) is mentioned in related Soskic 2022 *Nat Genet*; exact timepoints not visible in 2025 snippets |
| Disease | Autoimmune (MS, T1D, psoriasis, RA, IBD candidate variants) — strong CEO match |
| **Status on BSC** | NOT YET DEPLOYED |

**Action item**: read both the 2025 *Nat Genet* paper AND the bioRxiv
preprint to confirm whether they're the same dataset, separate datasets,
or the bioRxiv supersedes. Then identify accession.

### BSC storage + compute estimate (Stage 3a all four datasets combined)

| Dataset | Raw download | Processed h5ad | Encoded latents (proj_dim=128) |
|---|---|---|---|
| Mimitou CRISPR | ~5 GB | ~1 GB sparse | ~3 MB (5,825 × 128 × 4) |
| Parse 10M Cytokine | ~100 GB raw | ~20 GB sparse | ~5 GB (10M × 128 × 4) |
| CIPHER-seq | ~5 GB est. | ~1 GB est. | ~10 MB est. |
| Soskic CD4 CRISPRi | ~10 GB est. | ~3 GB est. | ~50 MB est. |
| **TOTAL** | **~120 GB** | **~25 GB** | **~5 GB** |

BSC scratch quota check needed — the Parse 10M dataset is the binding
constraint. Pre-processing on a transfer node (or local Mac then rsync)
may be necessary if BSC scratch is tight.

### Time-resolved candidates flagged

**ONLY CIPHER-seq is true time-course** (multiple discrete timepoints
within a single perturbation). The other three datasets are
single-endpoint or activation-vs-resting two-state. This is a real gap
for Stage 3a — public PBMC perturbation data with continuous-time
dynamics is genuinely rare. CIPHER-seq is precious for Layer 4
(temporal dynamics anchor) of the layered strategy.

---

## Report 2: Encoder probe on perturbation data

**Status**: SCOPED. Helper scripts ready to ship in this turn. Execution
requires Mimitou CRISPR data on BSC (depends on Report 1 download +
harmonization).

### Methodology

1. Download GSE156478 Mimitou CRISPR arm (CD4 T cell perturbations)
2. Harmonize ATAC peaks to DOGMA union peak set (323,500 peaks)
3. Forward all ~5,825 perturbed cells through validated DOGMA encoder
   (bare ATAC path, Test 1.5 methodology — NOT joint fusion)
4. Pseudo-bulk by perturbation condition (5 conditions: CD3E, CD4,
   ZAP70, NFKB2, CD3E+CD4 double)
5. Compute centroid-NN accuracy on held-out per-condition replicates
6. Compare against random-projection baseline (mock encoder, same union M)

### Decision criteria

| Centroid-NN accuracy on held-out | Interpretation |
|---|---|
| ≥ 0.80 | Encoder cleanly separates perturbations; existing latent space sufficient for Stage 3a |
| 0.50 – 0.80 | Encoder partially separates; may need a perturbation-aware fine-tune adapter |
| < 0.50 | Perturbation signal lost in current latent; encoder needs adapter or perturbation-specific projection head |

### Helper scripts

Two scripts ship below:
1. `scripts/prepare_mimitou_crispr.py` — download + harmonize peaks
2. `scripts/probe_mimitou_perturbation_clustering.py` — encoder probe + centroid-NN

Both use existing infrastructure (`aivc.eval.calderon_probe::encode_samples`,
`scripts/eval_pseudobulk_compatibility.py` mapping pattern).

### What this report tells us

If centroid-NN is high (≥ 0.80), the perturbation-decoder architecture
proposed in the Stage 3 scoping doc is sufficient — frozen encoder + MLP
delta head over learned perturbation embeddings.

If low (< 0.50), we need a perturbation-aware adapter between the encoder
and the delta head. This is a meaningful architectural change that Part 2
must address.

---

## Report 3: Pathway annotation prep

### Required pathway sets

Per CEO design constraint #5 (pathway-aware output head):

**MSigDB Hallmark gene sets (50 hallmarks)**:
- `H` collection from `https://www.gsea-msigdb.org/gsea/msigdb/`
- Specifically: HALLMARK_INTERFERON_GAMMA_RESPONSE, HALLMARK_TNFA_SIGNALING_VIA_NFKB,
  HALLMARK_IL2_STAT5_SIGNALING, HALLMARK_INFLAMMATORY_RESPONSE,
  HALLMARK_MTORC1_SIGNALING, HALLMARK_PI3K_AKT_MTOR_SIGNALING,
  HALLMARK_APOPTOSIS, HALLMARK_DNA_REPAIR, etc. — 50 total

**KEGG immune signaling pathways** (per CEO list):
- TLR (hsa04620 Toll-like receptor signaling)
- NF-κB (hsa04064 NF-kappa B signaling)
- JAK/STAT (hsa04630 Jak-STAT signaling)
- MAPK (hsa04010 MAPK signaling)
- PI3K-AKT-mTOR (hsa04151 PI3K-Akt + hsa04150 mTOR)
- BCR (hsa04662 B cell receptor signaling)
- TCR (hsa04660 T cell receptor signaling)

### Implementation approach

```python
# scripts/prepare_pathway_annotations.py (sketched, ~80 lines)
from gseapy import Msigdb
import pandas as pd

# Hallmark
msig = Msigdb()
hallmark_dict = msig.get_gmt(category="h.all", dbver="2024.1.Hs")
# {pathway_name: [gene_symbols]}

# KEGG (via gseapy.gsea.parse_gmt or KEGG API)
kegg_pathways_to_pull = {
    "hsa04620": "TLR_signaling",
    "hsa04064": "NFkB_signaling",
    "hsa04630": "JAK_STAT_signaling",
    "hsa04010": "MAPK_signaling",
    "hsa04151": "PI3K_AKT_signaling",
    "hsa04150": "mTOR_signaling",
    "hsa04662": "BCR_signaling",
    "hsa04660": "TCR_signaling",
}
# Pull via KEGG REST API (rest.kegg.jp/get/hsa04620) and parse

# Build mapping: gene_symbol → [pathway_names_with_membership]
# (one gene typically appears in multiple pathways)
gene_to_pathways = {...}

# Output: data/pathway_annotations/gene_to_pathway.csv (3 columns:
# gene_symbol, pathway_name, source_db)
```

### Output spec

```
data/pathway_annotations/
  msigdb_hallmark_v2024_1_hs.gmt
  kegg_immune_signaling_v_<DATE>.gmt
  gene_to_pathway_map.csv     # canonical mapping table
  pathway_to_genes_map.json   # reverse mapping
```

The mapping table feeds the pathway-aware output head in Part 2's
architecture: `Δgene_expression → Δpathway_activation` reduces from
~20K genes to ~58 pathway scores per cell, which is the right
biological resolution for the QurieSeq phospho readouts (which are
themselves pathway-level: pJAK1, pSTAT3, pERK, etc.).

### Estimated effort

~1 day. Dependencies: `gseapy>=1.0`, internet access to MSigDB +
KEGG REST API. No BSC required; runs on Mac.

---

## Report 4: BTK+JAK demo feasibility — 4-arm check

### Direct finding

**The synergy demo is mathematically impossible with public data alone.**
Flag for QurieSeq-only execution.

### Detailed audit

CEO required all four arms as separate measurable conditions:
(a) BTK perturbation alone, (b) JAK alone, (c) BTK+JAK combination,
(d) Matched vehicle controls.

**None of the layered-strategy public datasets contain ANY pharmacological
inhibitor perturbations** (BTK, JAK, or otherwise):

| Dataset | BTK arm | JAK arm | BTK+JAK combo | Vehicle | Verdict |
|---|---|---|---|---|---|
| Mimitou ASAP-seq CRISPR (GSE156478) | ✗ — only CRISPR-KO of CD3E, CD4, ZAP70, NFKB2 | ✗ | ✗ | NTC gRNAs (not vehicle drug controls) | All 4 missing |
| Parse 10M Cytokine | ✗ — only cytokine STIMULI (90 of them) | ✗ | ✗ | PBS control | All 4 missing |
| CIPHER-seq stim PBMC | ✗ — TCR/LPS/cytokine stim | ✗ | ✗ | possible | All 4 missing |
| Soskic 2025 CD4 CRISPRi | ✗ — autoimmune-VARIANT CRISPRi (regulatory regions) | ✗ | ✗ | NTC gRNAs | All 4 missing |
| Norman 2019 K562 (out of scope per cell-type cut) | ✗ | ✗ | ✗ | NTC | Not a candidate anyway |

### Why no public data has this

Pharmacological inhibitor scRNA/ATAC + multimodal datasets are rare in
the literature — most CRISPR perturbation screens use genetic (CRISPRi/o)
rather than chemical perturbations because:

1. CRISPR has cleaner specificity (single gene knockdown) vs drugs
   (off-target effects)
2. CRISPR scales to hundreds of perturbations in pooled screens; drug
   panels are ~10-100 max in any single experiment
3. Drug perturb-seq is dominated by cancer-cell-line studies (D-SPIN,
   sci-Plex), not PBMC

The closest public approximation is **D-SPIN** (Caltech 2023, ~90
immunomodulatory drugs × LPS/IFN-β stim on PBMC subsets, RNA only).
D-SPIN's drug list includes some kinase inhibitors but NOT specifically
ibrutinib + ruxolitinib in matched 4-arm designs. Worth a deeper check
but unlikely to satisfy the 4-arm requirement.

### Implication

The BTK+JAK demo is **a QurieSeq Phase 1 deliverable**, not a Stage 3a
public-data deliverable. CEO's 4-arm design is exactly what QurieSeq
provides; that's the moat.

**For Stage 3a (public data)**, demonstrate the synergy-head architecture
on a different combination signal that DOES exist publicly. Three
options:

1. **CRISPR double knockouts**: Mimitou's CD3E+CD4 double (vs CD3E
   alone, CD4 alone, NTC) is a 4-arm design that's mathematically
   identical. Synergy = double minus (single1 + single2 + vehicle).
   This validates the synergy-head architecture; just substitutes
   "drug combination" with "genetic double KO."
2. **Cytokine + drug**: D-SPIN has cytokines × drugs. Pick a clean
   instance (e.g., IFN-β + JAK inhibitor) where all 4 arms exist.
3. **Cytokine + cytokine**: Parse Atlas has 90 cytokines but at
   single-condition level (not pairwise combinations). Defer.

**Recommend Option 1**: Mimitou's CD3E+CD4 double KO is the cleanest
public-data validation of the synergy-head architecture. The biological
signal is established (CD3E loss → loss of TCR signal; CD4 loss → loss
of co-receptor; double → epistatic interaction in TCR signaling).

### Held-out evaluation design

Stage 3a synergy-head eval:
- Train on: NTC vehicle, CD3E single, CD4 single (and unrelated CRISPR
  conditions — ZAP70, NFKB2)
- Hold out: CD3E+CD4 double KO arm
- Predict: `Δ_synergy(CD3E, CD4, t=16h, donor)` zero-shot
- Compare to observed double-KO delta

If the synergy head correctly predicts the CD3E+CD4 epistatic effect
zero-shot, the architecture generalizes to unseen combinations. **This
is the public-data analog of the BTK+JAK demo.** The QurieSeq Phase 1
demo (BTK+JAK) is then the proprietary version that lands the actual
CLL clinical signal.

---

## Report 5: B-cell investigation

**Status**: SCOPED. Helper script ready to ship below. Operates on the
existing on-disk `dogma_lll_union_labeled.h5ad` — no new data required.
~1 hour to run on BSC after Report 5 helper deploys.

### Three diagnostic hypotheses

The Calderon B-cell weakness was 4/22 = 0.18 in Test 1.5 (after noise-
centroid exclusion that lifted overall to 0.7308). Three plausible causes:

**H1: Pseudo-bulk artifact (small n)**
- DOGMA labeled B cells: 613 (vs CD4_T's 8,360 — 13.6× imbalance)
- Pseudo-bulk averaging over 613 cells produces a noisier centroid
- **Test**: split DOGMA B cells in half, compute centroid-centroid cosine
  similarity. If high (~0.95), B centroid is internally consistent and
  H1 is likely false. If lower than CD4_T's split-half, H1 is supported.

**H2: Training data imbalance**
- Hypothesis: SupCon's per-class loss weighting underweighted B due to
  small n. Encoder didn't learn B-specific features as discriminatively
- **Test**: project DOGMA B and DOGMA CD4_T cells through encoder, compute
  silhouette score per class. If B has lower silhouette than CD4_T, H2
  is supported.

**H3: Genuine encoder weakness on B-lineage chromatin**
- B cell ATAC profiles in DOGMA (LLL arm, possibly stim conditions)
  may be biologically distinct from Calderon's bulk B (Bulk_B label
  may include naive + memory + activated; mixed)
- Cross-corpus B-cell domain shift not captured
- **Test**: examine which Calderon B samples DO get correctly classified
  (the 4/22 that succeeded). Are they specific Calderon subtypes
  (e.g., Plasmablasts, Memory_B)? If so, H3 partially explains —
  the encoder learned a specific B-cell signature that's narrow

### Helper script

`scripts/probe_b_cell_diagnosis.py` (sketched below in §"Implementation").

Computes three diagnostics on the existing DOGMA labeled h5ad:
1. Split-half centroid similarity per class (H1)
2. Per-class silhouette score in encoder latent space (H2)
3. Cross-corpus B-subtype confusion analysis from Test 1.5 results JSON (H3)

Outputs a JSON report with per-hypothesis evidence + recommendation.

### Remediation plan options (depending on diagnosis)

| Diagnosis | Remediation |
|---|---|
| H1 dominant | Aggregate B subtypes from external data (e.g., HCA B-cell atlas) into the centroid set; or use multi-centroid B representation (one centroid per B subtype) |
| H2 dominant | Class-balanced fine-tune of last encoder layer using a focal-loss or per-class-weighted SupCon term. ~1 day |
| H3 dominant | More fundamental — would require pretraining DOGMA on a B-cell-rich dataset (HCA Bone Marrow, OneK1K). ~1-2 weeks. Defer to phase after Stage 3 |

**For Stage 3 specifically**, H1 + H2 combined likely explain most of
the gap. Recommend running the helper first, then committing to a
remediation only if Stage 3a's BTK+JAK-analog demo (CD3E+CD4 double
KO synergy) fails specifically on B-related conditions. Otherwise,
B-cell weakness is a known limitation banked for follow-up but not
blocking Stage 3a.

### Important: BCR-relevance check

CEO emphasized BCR signaling for QurieSeq Phase 2 (anti-IgM stim, BTK
inhibitors, pJAK1/BCR demo target, CLL clinical context). B-cell
quality matters for QurieSeq, not as much for Stage 3a public data
(which has no BCR-specific perturbations). **The Stage 3a critical path
does not depend on B-cell encoder quality.** B-cell remediation can be
done in parallel with Stage 3a, completed before QurieSeq Phase 2.

---

## Helper scripts ready to ship

The following scripts are referenced in Reports 2, 3, 5. Each is a
standalone deliverable that runs on BSC (Reports 2, 5) or Mac (Report
3).

### Report 3 helper: `scripts/prepare_pathway_annotations.py`

```python
"""Prepare pathway annotations for Stage 3 pathway-aware output head.

Pulls MSigDB Hallmark (50 sets) + KEGG immune signaling pathways
(TLR, NF-κB, JAK/STAT, MAPK, PI3K-AKT-mTOR, BCR, TCR). Outputs:
  data/pathway_annotations/msigdb_hallmark.gmt
  data/pathway_annotations/kegg_immune_signaling.gmt
  data/pathway_annotations/gene_to_pathway_map.csv
  data/pathway_annotations/pathway_metadata.json

Run: [Mac terminal: ~/Projects/aivc_genelink]
    pip install gseapy>=1.0  --break-system-packages
    python scripts/prepare_pathway_annotations.py \
        --output_dir data/pathway_annotations
"""
# ~100 lines; full implementation in Part 1 sprint
```

### Report 5 helper: `scripts/probe_b_cell_diagnosis.py`

```python
"""B-cell weakness diagnosis (Test 1.5 follow-up).

Operates on dogma_lll_union_labeled.h5ad. Computes three diagnostics:
  1. Split-half centroid cosine similarity per class (H1)
  2. Per-class silhouette score in encoder latent space (H2)
  3. Cross-corpus B-subtype confusion from prior Test 1.5 JSON (H3)

Run: [BSC login: /gpfs/scratch/ehpc748/quri020505/aivc_genelink]
    sbatch scripts/submit_b_cell_diagnosis.slurm \
        /gpfs/scratch/ehpc748/quri020505/checkpoints/pretrain/pretrain_encoders.pt
"""
# ~150 lines; full implementation in Part 1 sprint
```

### Report 2 helpers: `scripts/prepare_mimitou_crispr.py` + `scripts/probe_mimitou_perturbation_clustering.py`

These are larger (~250 lines each) and have a critical dependency
(GSE156478 raw download from GEO). Recommend separating into two
sub-tasks:

1. **`prepare_mimitou_crispr.py`** — downloads GSE156478 CRISPR arm,
   harmonizes peaks to DOGMA union peak set (323,500), builds labeled
   h5ad. ~1-2 days execution on BSC.
2. **`probe_mimitou_perturbation_clustering.py`** — encodes through
   bare-ATAC DOGMA encoder, computes centroid-NN per perturbation
   condition, outputs JSON. ~1 hour after data prep.

---

## Summary table

| Report | Status | Output | Helper script | Execution |
|---|---|---|---|---|
| 1. Public dataset access | COMPLETE | This doc § Report 1 | n/a | n/a |
| 2. Encoder probe on perturb data | SCOPED | (will be) results JSON | 2 scripts ready to write | BSC (after data prep) |
| 3. Pathway annotation prep | SCOPED | gmt files + mapping CSV | 1 script ready to write | Mac (~1 day) |
| 4. BTK+JAK demo feasibility | COMPLETE — INFEASIBLE on public data | This doc § Report 4 | n/a | n/a (verdict: defer to QurieSeq) |
| 5. B-cell investigation | SCOPED | results JSON | 1 script ready to write | BSC (~1 hour) |

---

## Recommended execution sequence (~5 days)

**Day 1**:
- Read CIPHER-seq + Soskic 2025 papers' Data Availability sections,
  populate accessions in Report 1
- Run Report 3 prep helper on Mac (pathway annotation downloads)
- Write Report 5 B-cell helper + run on BSC (existing data, fast)

**Day 2-3**:
- Download Mimitou GSE156478 CRISPR arm to BSC scratch
- Download CIPHER-seq + Soskic data to BSC scratch
- Skip Parse 10M for now (binding storage + slow); defer to Stage 3a
  when needed

**Day 4**:
- Harmonize Mimitou CRISPR peaks to DOGMA union peak set
- Run Report 2 encoder probe; produce centroid-NN result

**Day 5**:
- Compile all 5 reports into final Part 1 deliverable
- Surface any blockers for Part 2 architecture authorization

---

## What blocks Part 2 architecture authorization

Per CEO's explicit instruction: "do not authorize architecture commits
before Part 1 is complete." Part 1 is complete when:

- All 5 reports have empirical results (not just scoping)
- Report 2 verdict is known (encoder probe centroid-NN — drives
  whether perturbation adapter is needed)
- Report 4 verdict locked (BTK+JAK demo — already INFEASIBLE on
  public data; CD3E+CD4 double KO substitute identified)
- Report 5 verdict known (B-cell diagnosis — drives whether
  remediation is needed)

Once those land, Part 2 architecture proposal is informed by the
empirical evidence. Specifically:

- If Report 2 returns ≥ 0.80, Part 2 uses frozen encoder + simple
  delta head
- If Report 2 returns < 0.50, Part 2 includes a perturbation-aware
  adapter
- If Report 5 implicates H3 (genuine encoder weakness on B), Part 2
  defers B-cell-related perturbations until B remediation lands

---

## Sources

- [DOGMA-seq paper (Mimitou et al. 2021 *Nat Biotechnol*)](https://www.nature.com/articles/s41587-021-00927-2)
- [Mimitou GSE156478 GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478)
- [Parse 10M PBMC Cytokine — bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.12.693897v1.full)
- [Allen Institute Parse 10M Cytokines resource](https://apps.allenimmunology.org/aifi/resources/parse-10m-cytokines/)
- [CIPHER-seq paper (Goh et al. 2026 *Sci Reports*)](https://www.nature.com/articles/s41598-026-44946-y)
- [Soskic 2025 *Nat Genet*](https://www.nature.com/articles/s41588-025-02301-3)
- [Soskic 2025 related bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1.full)
- [MSigDB v2024.1.Hs](https://www.gsea-msigdb.org/gsea/msigdb/)
- [KEGG REST API](https://rest.kegg.jp/)
- [Phase 6.5g.2 closure doc with Test 1.5](docs/reports/phase_6_5g_2_closure_E2_NULL_2026_05_04.md) (this repo)
- [Cross-corpus eval methodology (the Test 1.5 lesson)](docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md) (this repo)
