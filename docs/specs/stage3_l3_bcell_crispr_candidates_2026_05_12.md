# Stage 3 L3 — B-cell CRISPR Perturbation Dataset Candidates

**Date:** 2026-05-12
**Owner:** Ash Khan (AIVC) — for review by Kinga / Thiago
**Scope:** Public-data candidates for L3 layer after Thiago's CRISPR / B-cell-line reframing
**Sources searched:** PubMed (via NCBI), GEO, scperturb.org, PerturBase, PerturbSeq.db, bioRxiv, recent (2023–2026) literature in Nature/Cell/Blood/Sci Reports/Front Immunol
**L4 (temporal):** KILLED per Thiago/Kinga — QurieSeq Phase 1 will provide. Not searched.

---

## Headline finding (read before tables)

**The public landscape for "B-cell + CRISPR + single-cell RNA" is structurally thin.**

After a wide-net pass across PubMed/GEO/PerturBase/scperturb.org/bioRxiv, **only one published dataset cleanly satisfies all three L3 must-haves (B-cell + CRISPR perturbation + single-cell omic readout): Rubin et al. 2018 Perturb-ATAC (GSE116249) in GM12878 lymphoblasts** — and even that one provides ATAC + sgRNA (no RNA).

Every other candidate fails at least one must-have:
- Bulk pooled CRISPR screens in DLBCL lines (CREBBP/EP300, FOXP1, CD20×CD3 bispecific) — not single-cell.
- Genome-scale Perturb-seq (Replogle 2022) — K562 (BCR-ABL⁺ CML, lymphoid lineage debatable); not strictly B-cell per Kinga's allowlist.
- Multiome Perturb-seq (Metzner 2024) — RPE-1 epithelial; not B-cell.
- ECCITE-seq lymphoma exemplar (Mimitou 2019) — CTCL is T-cell lymphoma; the B-cell application uses a controlled CRISPR perturbation in a lymphoblastoid line but is small (~few hundred cells).
- SDR-seq (Schmidt 2024/2025) — primary NHL B-cell lymphoma scDNA+RNA, **no CRISPR** (natural variants).
- Hodson/Staudt lab arrayed CRISPR knockins — primary tonsil GCB-B with multi-modal readout but **arrayed**, not pooled-screen geometry.

**Strategic implication for AIVC:** L3 cannot be filled by a single high-fidelity public dataset the way DOGMA (Mimitou 2021) fills L2. Pragmatic options below in the synthesis section.

---

## Candidate 1 — Rubin et al. 2018 Perturb-ATAC (GM12878)

| Field | Notes |
|---|---|
| **Dataset name** | Coupled Single-Cell CRISPR Screening and Epigenomic Profiling Reveals Causal Gene Regulatory Networks |
| **Accession ID** | **GSE116249** (GEO) — confirmed via OmicsDI |
| **Paper citation** | Rubin AJ, Parker KR, Satpathy AT, …, Greenleaf WJ, Chang HY, Khavari PA. *Cell* 176, 361–376.e17 (2019). DOI: [10.1016/j.cell.2018.11.022](https://doi.org/10.1016/j.cell.2018.11.022). PMID: 30580963 |
| **Cell type** | GM12878 — EBV-transformed lymphoblastoid B cell line (immortalized, not malignancy proper). Also primary keratinocyte arm (irrelevant) |
| **Modalities** | scATAC-seq + direct sgRNA capture (Perturb-ATAC). **No RNA. No protein.** |
| **CRISPR targets** | ~30 genes per the published phenotype-genotype table: transcription factors (PAX5, EBF1, IRF4, SPI1, BCL11A, RUNX3, BATF, IKZF1, MEF2B …), chromatin-modifying factors (EZH2, HDAC9, KAT6A, CHD2 …), and ncRNAs. 63 genotype–phenotype relationships total |
| **n_donors / cells** | Single donor (GM12878 clone); **~2,936 B-lymphoblast cells passing QC** (paper text). Plate-based, lower throughput than 10x |
| **Disease context** | Lymphoblastoid — EBV-immortalized. **Not malignancy.** TF hierarchy is the key biological output, including disease-associated cis-elements ID'd from GWAS overlap |
| **License** | NCBI GEO open access; Cell publication standard reuse |
| **Fit score** | **4.0 / 5** |
| **Strengths** | (i) Only published dataset I could verify with B-cell + CRISPR + single-cell omics. (ii) ATAC readout complements DOGMA's RNA+ATAC+Protein on the same chromatin axis. (iii) TF perturbations include B-cell identity factors that overlap with the DOGMA pretrain corpus's stim-axis biology. (iv) Establishes a reusable Perturb-ATAC peak-overlap pipeline (relevant given Phase 6.5g.2's 81.5% peak overlap fix) |
| **Weaknesses / gotchas** | (i) **No RNA modality** — fails Kinga's "RNA at minimum" must-have strictly. (ii) Plate-based, ~3k cells is small vs. 10x-scale (~30k DOGMA). (iii) EBV background = atypical chromatin state. (iv) Not a B-cell malignancy / CLL / lymphoma model — Thiago's wet-lab B-cell line will be more relevant. (v) 2018 vintage; pre-temporal-fusion conventions; will need bespoke loader rather than reuse of DOGMA loader. (vi) TF targets, not BCR-pathway-focused (no BTK/SYK/CD79b in screened set) |

---

## Candidate 2 — Replogle Genome-scale Perturb-seq (K562)

| Field | Notes |
|---|---|
| **Dataset name** | Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq |
| **Accession ID** | Figshare DOI [10.25452/figshare.plus.20029387](https://doi.org/10.25452/figshare.plus.20029387) (processed h5ad); raw on GEO (Weissman lab page links). Also hosted at gwps.wi.mit.edu and CZI Virtual Cell Models |
| **Paper citation** | Replogle JM, Saunders RA, …, Weissman JS. *Cell* 185(14), 2559–2575 (2022). DOI: [10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013). PMID: 35688146 |
| **Cell type** | **K562 — BCR-ABL⁺ chronic myeloid leukemia (CML)**. Lineage classification is contested: erythroid markers dominant, some pro-B/pre-B markers, also myeloid. **Not on Kinga's B-cell allowlist** (Ramos / OCI-Ly / BJAB / Daudi). Included as a methodological reference, not a clean L3 fit |
| **Modalities** | 10x 3′ scRNA-seq + direct sgRNA capture (CROP-seq vector with capture sequence). No ATAC, no protein |
| **CRISPR targets** | 9,866 expressed-gene CRISPRi (day-8) screen + 2,057 essential-gene CRISPRi (day-6) screen. Includes BCL2 family, BTK, SYK only if expressed in K562 (BTK low/absent in K562) |
| **n_donors / cells** | Single cell line; **~2.5M cells across both screens** (largest single CRISPR-Perturb-seq published) |
| **Disease context** | CML (chronic myeloid leukemia) — adjacent to CLL biology (both BCR-ABL / kinase-driven), but lineage mismatch |
| **License** | Open access (CC-BY); processed data on Figshare under permissive terms |
| **Fit score** | **2.5 / 5** |
| **Strengths** | (i) Gold-standard single-cell CRISPR + RNA pipeline at genome scale. (ii) Cleanest data normalization in field; CZI virtual-cell benchmark uses it. (iii) Methodological template for AIVC's own L3 — if we ever generate Ramos Perturb-seq, this is the cookbook. (iv) BCL2-family and apoptosis-pathway genes covered in the 9,866-gene set |
| **Weaknesses / gotchas** | (i) **K562 is not a B-cell line** per Kinga's explicit list; using it requires arguing CML/BCR-ABL → CLL/BTK relevance which is weak. (ii) Most BCR-signaling components (BTK, CD79A/B, BLNK) are low/absent in K562 — perturbation signal will be noise. (iii) Cell-of-origin mismatch will confound cross-corpus pseudo-bulk centroid-NN evaluation (the canonical metric per `docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md`) |

---

## Candidate 3 — Mimitou ECCITE-seq method paper (CRISPR-compatible CITE-seq)

| Field | Notes |
|---|---|
| **Dataset name** | Multiplexed detection of proteins, transcriptomes, clonotypes and CRISPR perturbations in single cells |
| **Accession ID** | **GSE126863** (GEO — verification needed; paper Data Availability cites this for human samples). The methodology demo on B-cell-like data is small-scale |
| **Paper citation** | Mimitou EP, Cheng A, …, Smibert P. *Nat Methods* 16, 409–412 (2019). DOI: [10.1038/s41592-019-0392-0](https://doi.org/10.1038/s41592-019-0392-0). PMID: 31011186 |
| **Cell type** | Healthy PBMC + cutaneous T-cell lymphoma (CTCL) patient sample as the lymphoma exemplar. **Note: CTCL is T-cell lymphoma, not B-cell.** The CRISPR-perturbation demo arm uses HEK293 + a lymphoblastoid context for sgRNA capture validation |
| **Modalities** | scRNA + 49-marker surface protein panel + sample HTO + TCR/BCR clonotype + sgRNA capture. **5-modality** in a single droplet — most modality-complete CITE-seq-family protocol |
| **CRISPR targets** | Small demo set (~10 sgRNAs) — methodology paper, not a screen |
| **n_donors / cells** | Few thousand demo cells; CTCL patient sample dominant |
| **Disease context** | CTCL (cutaneous T-cell lymphoma) — does not meet L3 cancer/B-cell focus |
| **License** | Open access |
| **Fit score** | **2.0 / 5** |
| **Strengths** | (i) Protocol-of-record for combining CRISPR + 5-modality readout that **exactly matches AIVC's L2 DOGMA channel structure** (RNA + Protein + sgRNA + HTO). (ii) Adapting this protocol to Ramos / OCI-Ly is the most natural in-house path to a Phase-aligned L3 dataset |
| **Weaknesses / gotchas** | (i) Methodology paper, not a B-cell-disease screen — using it as L3 means using the protocol, not the data. (ii) The cancer exemplar is T-cell lymphoma. (iii) Small-cell-count demo arms |

---

## Candidate 4 — SDR-seq primary NHL B-cell lymphoma

| Field | Notes |
|---|---|
| **Dataset name** | Functional phenotyping of genomic variants using joint multiomic single-cell DNA–RNA sequencing (SDR-seq) |
| **Accession ID** | bioRxiv 2024.05.31.596895; final published *Nature Methods* 2025 (DOI [10.1038/s41592-025-02805-0](https://doi.org/10.1038/s41592-025-02805-0)). **GEO/EGA accession not located in 24h search** — likely European Genome-Phenome Archive (EGA, controlled access for patient data); verify with authors / paper supplement |
| **Paper citation** | Schmidt et al. *Nat Methods* (2025). Affiliations include DKFZ Heidelberg / Roider group |
| **Cell type** | **Primary non-Hodgkin lymphoma (NHL) B-cell** patient samples |
| **Modalities** | Single-cell joint targeted DNA (genotype) + RNA. No protein, no ATAC, no CRISPR |
| **CRISPR targets** | **None — natural mutations as "perturbations"** (CD79B, MYD88, EZH2, BCL2, KMT2D, TP53, etc. recurrent in DLBCL/FL) |
| **n_donors / cells** | Multi-patient cohort (numbers in paper); first such joint DNA+RNA on primary B-cell lymphoma at single-cell resolution |
| **Disease context** | DLBCL / follicular lymphoma — **strong cancer + B-cell + hematological-malignancy fit** |
| **License** | Likely restricted (patient data via EGA-class access) |
| **Fit score** | **2.0 / 5** |
| **Strengths** | (i) Directly addresses the BTK/JAK CLL-adjacent story from Thiago via natural CD79B/MYD88/EZH2 mutations. (ii) Primary patient samples — strongest disease-relevance of any candidate. (iii) Variant-as-perturbation framing is a useful complement to CRISPR-as-perturbation for L3 |
| **Weaknesses / gotchas** | (i) **No CRISPR** — strictly fails Kinga's must-have. Only justifiable if L3 is reframed as "perturbation = natural variant OR CRISPR." (ii) Likely controlled access (patient data). (iii) No ATAC / protein |

---

## Candidate 5 — Decker 2019 genome-scale CRISPRa CD20×CD3 bispecific screen

| Field | Notes |
|---|---|
| **Dataset name** | Genome-scale CRISPR activation screen uncovers tumor-intrinsic modulators of CD3 bispecific antibody efficacy |
| **Accession ID** | GEO accession **not surfaced in 24h search** — paper Data Availability would name it; check supplement |
| **Paper citation** | Decker CE, Young T, Pasnikowski E, Chiu J, Song H, Wei Y, et al. *Sci Rep* 9, 20068 (2019). DOI: [10.1038/s41598-019-56670-x](https://doi.org/10.1038/s41598-019-56670-x). PMID: 31882897 |
| **Cell type** | CD20×CD3-sensitive **human B-lymphoma cell line(s)** (paper text mentions Raji and others among DLBCL lines) |
| **Modalities** | **Bulk pooled CRISPRa screen** with sgRNA read-counts as the only readout. **No single-cell omic, no RNA at cell resolution** |
| **CRISPR targets** | Genome-scale (~20k genes) CRISPRa SAM library |
| **n_donors / cells** | Single cell-line backbone; not single-cell resolved |
| **Disease context** | B-cell lymphoma + immuno-oncology (T-cell-redirecting bispecific) — **clean cancer + B-cell fit** |
| **License** | Open access |
| **Fit score** | **1.5 / 5** |
| **Strengths** | (i) Cleanest cancer + B-cell + multi-gene CRISPR perturbation on Kinga's allowlist of cell lines. (ii) CRISPRa modality covers gain-of-function space that AIVC otherwise lacks. (iii) Reveals tumor-intrinsic resistance hits useful as feature priors |
| **Weaknesses / gotchas** | (i) **Bulk screen, not single-cell** — strictly fails Kinga's "RNA at minimum" must-have unless we treat sgRNA-count fitness signal as a pseudo-readout. (ii) Useful as feature engineering / target prior, not as a training dataset. (iii) GEO accession not confirmed — needs paper-supplement lookup before commit |

---

## Ranked summary

| Rank | Dataset | Fit | Verdict |
|------|---------|-----|---------|
| 1 | **Rubin Perturb-ATAC GSE116249** (GM12878) | 4.0 | Only verified B-cell + CRISPR + single-cell omic. Use for chromatin axis. |
| 2 | **Replogle K562 Genome-scale Perturb-seq** | 2.5 | Methodological template only. Lineage mismatch (CML/myeloid). |
| 3 | **Mimitou ECCITE-seq protocol** (GSE126863, T.B.V.) | 2.0 | Adopt the protocol; data exemplar is T-cell not B-cell. |
| 4 | **SDR-seq primary NHL** | 2.0 | Variant-as-perturbation, no CRISPR. Reframing required. |
| 5 | **Decker 2019 CRISPRa CD20×CD3** | 1.5 | Bulk screen, useful as feature prior only. |

---

## Synthesis & strategic options

The L3 layer cannot be filled the way L2 (Mimitou DOGMA) was. **Three live options:**

**Option A — Adopt Rubin Perturb-ATAC as canonical L3, accept "ATAC-only L3."**
Pros: only verified dataset; clean ATAC channel; B-cell-identity TFs overlap with DOGMA stim biology.
Cons: no RNA → asymmetric vs. L2; small n_cells (~3k); EBV-immortalized not malignancy.
Pre-registration: L3 metric becomes chromatin-axis perturbation prediction, not transcriptome.

**Option B — Generate AIVC's own Ramos / OCI-Ly Perturb-seq + CITE-seq dataset (ECCITE-seq protocol per Candidate 3).**
Pros: Phase-aligned with Thiago's wet-lab B-cell-line plans; can target BCR + apoptosis pathway (BTK, SYK, CD79B, BCL2, MCL1); 5-modality readout matches L2.
Cons: Adds wet-lab arc; timeline impact ≥ 1 quarter; budget impact.
Pre-registration: standard CRISPR-perturbation prediction with proper held-out gRNA evaluation.

**Option C — Defer L3 until QurieSeq Phase 2 / Phase 3 produces in-house CRISPR data.**
Pros: avoids public-data forcing; QurieSeq already on roadmap.
Cons: leaves the platform without any L3 signal until late 2026.

**My recommendation: Option A as floor + Option B as ceiling.** Use Rubin GSE116249 as L3-v0 (chromatin only) to keep the platform moving; commit to in-house Ramos/OCI-Ly ECCITE-seq as L3-v1 aligned with Thiago's wet-lab arc. Reject Option C — it leaves too long a gap and doesn't actually close the loop on validated CRISPR perturbations.

---

## Key Risks

1. **L3 public-data shortage is structural, not search-effort-limited.** A 24h additional search will not find a hidden "DOGMA-equivalent for CRISPR in B cells." 5 days of deeper search will yield at most one or two more candidates of similar (≤2.0) fit. Plan around the gap.
2. **Cross-corpus mismatch already burned us once** (B-vs-DC stim-driven chromatin shift, see `project_aivc_bcell_diagnosis_2026_05_05`). Using Rubin GM12878 + DOGMA primary B in the same evaluation will hit a new corpus-corpus shift (EBV-transformed vs. primary stim-driven). Pseudo-bulk centroid-NN methodology will need a B-lymphoblast lineage label to stay valid.
3. **Accession verification incomplete for Candidates 3, 4, 5.** GSE126863 (Mimitou) needs paper-supplement confirmation; SDR-seq likely EGA (controlled access); Decker 2019 GEO ID not surfaced. Do not lock these into a spec until accession is hand-verified.
4. **K562 inclusion temptation.** The Replogle data is so much cleaner than everything else that there will be pressure to include it as "B-lineage adjacent." Resist unless we explicitly add a CML/myeloid lineage label and reject the "B-cell only" framing — otherwise it pollutes cross-corpus evaluation.
5. **Decker/SDR-seq involve reframing L3 must-haves.** SDR-seq trades "CRISPR" for "natural variant"; Decker trades "single-cell" for "bulk." Either reframing needs explicit pre-registration before we count the data toward the L3 evaluation.

## Recommended Next Step

Kinga + Thiago decision call on Option A vs A+B vs C, framed as "what is L3-v0 and what is L3-v1, with what timeline?" Concrete asks:
1. Approve Rubin GSE116249 as L3-v0 with chromatin-only metric, or reject and demand B as the only valid path.
2. If Option B: confirm Thiago's wet-lab B-cell line identity (Ramos? OCI-Ly1/3/7/10? BJAB?) and lock the gRNA-target list to BCR + apoptosis pathway (proposed: BTK, SYK, LYN, BLNK, CD79A, CD79B, CD19, BCL2, MCL1, BAX, BAK + NTC + safe-targeting controls).
3. Adjudicate the K562 / Replogle question: hard-no, or accept as methodology-only reference?

## What I need from you

- **Confirmation that the L3 must-haves cannot be relaxed** — specifically "RNA at minimum" and "CRISPR perturbation." If those can flex, SDR-seq (variants) and Rubin (ATAC-only) move up the ranking substantially.
- **Decision on Option A vs B vs A+B vs C** — drives whether I close out L3 against public data or start scoping an in-house experiment.
- **Wet-lab B-cell-line identity from Thiago** — so any in-house L3-v1 spec aligns with his actual chassis.
- **Accession-verification budget** — 4–6h to hand-check GSE126863 / Decker GEO / SDR-seq EGA in Box / Synapse / EGA portal access if needed.
