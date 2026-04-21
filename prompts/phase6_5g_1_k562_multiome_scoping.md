# Phase 6.5g.1 — K562 Multiome Source Scoping (Gap 1 resolution)

## STATUS

- **Branch**: `phase-6.5g.1-scoping` cut from `main` at `8e3a52744b07a45a14b2307a6714c184e84313bd` (=`phase-6.5-closed^{commit}`)
- **Parent phase**: 6.5g/E2 (K562 lineage-pretraining domain-invariance) blocked on Gap 1 — original spec's "10x K562 Multiome ARC demo" verified non-existent (4-probe 10x catalog scan, 2026-04-20)
- **Purpose**: Bounded, pre-registered scan of public K562 paired RNA+ATAC single-cell datasets. Outcome is binary (FOUND / EMPTY) and locked before execution. No training code, no GPU, no data download beyond metadata/manifest inspection within this phase.
- **Sibling branch state**: `phase-6.5g-e2` paused at `6d56f610a6834a8813551ea2cda27a3a96c40cea` awaiting outcome; E2 spec (`prompts/phase6_5g_e2_k562_pretrain.md`, SHA-256 `ab25340b21c61bf2f1f1c44e28f569429d45041c948e71629ddd4f869d117c1d`) unchanged by this phase — only Gap 1 line is rewritten on FOUND, or the whole E2 branch is deferred on EMPTY.

## CONTEXT

Phase 6.5g/E2 requires a K562 dataset with paired RNA+ATAC single-cell measurements to isolate the tissue-of-origin variable against the 6.5e PBMC baseline. Path (ii) — hg19 SHARE-seq barnyard + liftover — was rejected on build-mismatch + cell-count grounds. Path (i) — naive K562-ATAC-only swap — was rejected on two-variable confound grounds (varies tissue AND modality simultaneously vs. 6.5e). Path (iii) — bounded public-catalog scoping — is this phase. If path (iii) returns EMPTY, path (i-extended) — two-arm modality-control (K562-ATAC-only + PBMC-RNA-only) — auto-triggers as a staged fallback in `phase-6.5g.2-e2-iextended` but does not execute without explicit `GO`.

## ACCEPTANCE CRITERIA (pre-registered outcomes)

### FOUND — all must hold for at least one candidate

1. Protocol ∈ **R1 allowlist** (see below)
2. Barcode-level RNA+ATAC pairing verifiable per **T_pair**
3. Repository ∈ **R2 surface** (see below)
4. Genome build hg38 native (preferred) OR hg19→hg38 CrossMap liftover feasible with documented coordinate-loss bound ≤ 2%
5. Processed data format: `.h5ad`, `.h5mu`, 10x `outs/` (`filtered_feature_bc_matrix.h5` + `atac_fragments.tsv.gz`), or `.mtx` with paired barcodes. **FASTQ-only is insufficient** (re-processing blows Gap 4's 12h early-stop envelope).
6. Cell count ≥ 2,000 post-QC (floor chosen to exceed SHARE-seq demo's ~1.5–2k and ensure minimum statistical traction on Norman RSA)
7. License compatible: **processed matrices** (h5ad/h5mu/mtx/10x outs) must be publicly redistributable without DUA. FASTQ-level gating is acceptable since FASTQ is out of scope for this phase (see T_license).

### EMPTY — all must hold

1. **R3** 3-probe protocol completed for every lab in the **Lab scan set** (§LAB SCAN SET)
2. All **R2** catalog-level surfaces probed (CZI CellxGene Census, 4DN, ENCODE, HCA, NeMO)
3. Per-probe mini-ledger row written with query-string, hit-count, and candidate-URL fields — no blank cells
4. Negative determination passes **T_cover**

> **Scope note**: "EMPTY" means no qualifying K562 paired RNA+ATAC dataset was found **within the bounds defined by R1 (protocol allowlist), R2 (repository surface), and R3 (lab scan set)**. This is a bounded-scope determination, not a claim of universal absence. Datasets outside R1/R2/R3 may exist but are out of scope for this phase; widening the scope is a separate re-registration.

## REPOSITORY SURFACE (R2 — locked)

**Included**:

- GEO (NCBI)
- ArrayExpress (EBI)
- Zenodo
- CZI CellxGene + Census
- Human Cell Atlas (HCA) Data Browser
- ENCODE Project
- 4DN Data Portal
- NeMO (brain — included for completeness; K562 hit unlikely)
- Synapse (with T_license gate — DUA-free entries only)
- Lab GitHub organizations + lab Zenodo community pages

**Excluded**:

- SRA-only listings without processed data (re-processing blows 12h Gap 4 cap)
- FASTQ-only releases without paired count/peak matrices
- Any repo where the **processed data itself** requires institutional DUA or payment (FASTQ-level gating with public processed data is allowed — see T_license)

## PROTOCOL ALLOWLIST (R1 — locked)

Paired RNA + ATAC single-cell methods:

- 10x Genomics Multiome ARC (Chromium Next GEM Single Cell Multiome ATAC + Gene Expression)
- SHARE-seq (Ma, Chen, Buenrostro 2020)
- sci-CAR (Cao, Shendure 2018)
- ISSAAC-seq (Xu, Fang 2022)
- Paired-seq (Zhu, Ren 2019)
- Paired-Tag (Zhu, Ren 2021)
- SNARE-seq / SNARE-seq2 (Chen, Lake 2019 / 2022)

**Pairing verification (T_pair) — structural, not documentary**:

Compute `overlap_fraction = |barcodes_RNA ∩ barcodes_ATAC| / |barcodes_RNA ∪ barcodes_ATAC|` (Jaccard) across:

- `.h5ad` / `.h5mu` with shared `obs.index` across RNA and ATAC AnnData/MuData modalities
- 10x `outs/` `per_barcode_metrics.csv` or `barcodes.tsv.gz` across `filtered_feature_bc_matrix` (GEX) and `atac_*` outputs
- `.mtx` with explicit barcode file shared across RNA and ATAC count matrices

**Tiered admission** (applies to T_pair — all tiers require structural evidence, not documentary alone):

| `overlap_fraction` | Verdict |
|---|---|
| `[0.90, 1.00]` | **pass** — admit to FOUND |
| `[0.85, 0.90)` | **borderline_candidate** — manual review required; escalate to Ash for one-line decision before FOUND admission |
| `< 0.85` | **fail** — demote to `excluded_format` |

## LAB SCAN SET

### Catalog-level probes

| Probe | Query | Surface |
|---|---|---|
| C1 | `K562` + `multiome` filter | CZI CellxGene Explorer + Census |
| C2 | `K562` cell-line filter + `scATAC-seq` or `Multiome` assay | 4DN Data Portal |
| C3 | `K562` biosample + `single cell` + `ATAC-seq`/`RNA-seq` | ENCODE Project |
| C4 | `K562` + species filter | HCA Data Browser |
| C5 | `K562` + `multiome OR SHARE-seq OR sci-CAR` | NeMO (low-prior, completeness) |
| C6 | `K562 multiome` full-text | Zenodo |
| C7 | `K562` + Multiome-assay filter | GEO (`DatasetType[Properties] AND "K562"[All Fields] AND ("multiome" OR "SHARE-seq" OR ...)`) |
| C8 | `K562` + single-cell + paired-modality filter | ArrayExpress |

### Per-lab probes (R3 3-probe protocol)

For each lab below:

- **(a) Google Scholar**: `"K562" AND ("Multiome" OR "SHARE-seq" OR "sci-CAR" OR "Paired-seq" OR "Paired-Tag" OR "SNARE-seq" OR "ISSAAC-seq") AND [PI name]`
- **(b) Lab website** publications/datasets tab
- **(c) Lab GitHub org + lab Zenodo community**

**Labs**:

| # | Lab | Rationale |
|---|---|---|
| L1 | Shendure (UW) | sci-CAR, SHARE-seq collab (Ma 2020) |
| L2 | Buenrostro (Harvard) | SHARE-seq, scATAC pioneering |
| L3 | Greenleaf (Stanford) | scATAC-seq, multi-omics; K562 ATAC benchmarks |
| L4 | Chang (Stanford) | SHARE-seq co-origin, K562 multiome plausibility |
| L5 | Ren (UCSD/LJI) | Paired-seq, Paired-Tag |
| L6 | Satpathy (Stanford) | ASAP-seq, CITE-seq extensions; K562 |
| L7 | Adey (OHSU) | sci-CAR variants, K562 spike-ins |
| L8 | Lander / Regev (Broad) | Multi-omics atlases; K562 frequently used as reference |
| L9 | Weissman (WI/Broad) | Perturb-seq K562 — may have Multiome companion |
| L10 | Gasperini / Jost | K562 CRISPR screens — may publish paired companion |
| L11 | Norman / Adamson | K562 Perturb-seq — check for companion multiome |
| L12 | Lake (UCSD) | SNARE-seq, SNARE-seq2 |
| L13 | Fang (CSHL) | ISSAAC-seq |
| L14 | Zhu / Ren | Paired-Tag first-author companions |

## SCAN PROTOCOL (TASK)

For each catalog probe Cn and each lab Lm, produce one row in `data/phase6_5g/scoping_ledger.tsv` with schema:

```
source_id    probe_class    probe_id    query_string    hit_count    candidate_accession    candidate_url    protocol    genome_build    format    n_cells    license_flag    verdict    notes
```

`verdict` ∈ `{candidate, excluded_protocol, excluded_build, excluded_format, excluded_license, excluded_cells, no_hits}`.

A candidate row is elevated to FOUND only after **T_pair** verification against the downloaded processed file's structural metadata (not FASTQ — metadata-only download budget).

Terminate when either:

- ≥1 row reaches FOUND and passes T_pair → emit `data/phase6_5g/outcome_found.yaml`
- All rows filled with non-`candidate` or T_pair-failed verdicts AND T_cover satisfied → emit `data/phase6_5g/outcome_empty.yaml`

No intermediate early-exit. Ledger must be complete before outcome emission.

## TRIPWIRES

| ID | Condition | Action if violated |
|---|---|---|
| **T_cover** | Every (Cn, Lm) row in `scoping_ledger.tsv` has non-null `query_string` and `hit_count` before EMPTY emission | Abort outcome emission; fill gaps; re-attempt |
| **T_pair** | No candidate admitted to FOUND without structural barcode-identity evidence (see R1 pairing verification). Tiered: `overlap_fraction ∈ [0.90, 1.00]` → pass; `[0.85, 0.90)` → borderline_candidate (escalate manual review); `< 0.85` → fail | Pass → admit; borderline → pause + escalate; fail → demote to `excluded_format`; continue scan |
| **T_staged** | No training code, GPU command, or model-weight download reachable from this phase's scripts; merge choreography auto-steps stop before training | Revoke auto-draft; block until GO |
| **T_pause_tag** | On EMPTY outcome: annotated tag `phase-6.5g-e2-paused-gap1` MUST exist at `phase-6.5g-e2` HEAD **before** `phase-6.5g.1-scoping` merges to main | Block merge; create tag; retry |
| **T_license** | Admission requires **processed matrices** (h5ad/h5mu/mtx/10x outs) to be publicly redistributable without DUA. FASTQ-level gating is permitted — FASTQ is out of scope for this phase. Applies to Synapse and any other gated-access repo. | If processed data is DUA-gated → demote to `excluded_license`. If only FASTQ is gated → proceed, note in ledger `license_flag=fastq_gated_ok`. |
| **T_race** | `phase-6.5g-e2` rebase target after scoping merge is `phase-6.5g.1-closed^{commit}` (tag-dereferenced), NOT `main` HEAD at an arbitrary later time | Abort rebase; re-target from tag |

## COMPUTE BUDGET

- **Scoping itself**: 0 GPU-h (metadata/manifest reads only)
- **Metadata download cap**: 1 GB total across all candidate inspections (processed `.h5ad`/`.h5mu`/`outs` summary files)
- **Fallback (path i-extended) budget** — triggered by EMPTY, executes only in `phase-6.5g.2-e2-iextended` with `GO`:
  - Arm A (K562-ATAC-only pretrain): 12 GPU-h ceiling — Gap 4 early-stop semantics (checkpoint + continue to eval, do NOT abort)
  - Arm B (PBMC-RNA-only pretrain): 12 GPU-h ceiling — Gap 4 early-stop semantics
  - Total ceiling: 24 GPU-h

## ARM DEFINITIONS (fallback path i-extended only — not executed here)

- **Arm A — K562-ATAC-only**: new pretrain on K562 scATAC (source TBD: Buenrostro 2015 bulk+pseudo-single-cell, or ENCODE K562 single-cell ATAC). ATAC encoder active; RNA encoder zero-masked; modality flag = ATAC-only.
- **Arm B — PBMC-RNA-only**: reuse `data/pbmc10k_multiome.h5ad` (existing SHA-anchored artifact in REAL_DATA_BLOCKERS.md). RNA encoder active; ATAC zero-masked; modality flag = RNA-only. **No new data anchor required** (R6).
- **Comparison probe**: both arms evaluated on Norman 2019 RSA probe using the 6.5e evaluation pipeline. Isolates single variable (modality) vs. 6.5e RNA-only PBMC baseline; tissue variable holds (PBMC for Arm B, K562 for Arm A) — two-arm design recovers modality-control interpretation.

## FAILURE HANDLING — merge choreography

### Path FOUND

1. Commit scoping ledger + `outcome_found.yaml` + T_pair structural evidence to `phase-6.5g.1-scoping`
2. Open PR `phase-6.5g.1-scoping` → `main`; body per `.github/PR_BODY_phase6_5g_1_scoping.md`
3. On merge: annotated tag `phase-6.5g.1-closed` at merge commit
4. `git checkout phase-6.5g-e2 && git rebase 'phase-6.5g.1-closed^{commit}'`
5. Amend `prompts/phase6_5g_e2_k562_pretrain.md` Gap 1 line with locked source accession + SHA-256 + provenance → new E2 spec SHA recorded
6. Update `data/phase6_5g/metadata/` with new source manifest
7. Resume E2 Step 1

### Path EMPTY

1. Commit scoping ledger + `outcome_empty.yaml` (exhaustive negative) to `phase-6.5g.1-scoping`
2. **T_pause_tag**: create annotated tag `phase-6.5g-e2-paused-gap1` at `phase-6.5g-e2` HEAD (record full 64-char SHA in PR body)
3. Open PR `phase-6.5g.1-scoping` → `main`; body documents EMPTY determination
4. On merge: annotated tag `phase-6.5g.1-closed` at merge commit
5. **Auto-trigger (staged, T_staged-compliant)**:
   - `git branch phase-6.5g.2-e2-iextended 'phase-6.5g.1-closed^{commit}'`
   - Auto-draft `prompts/phase6_5g_2_e2_iextended.md` (two-arm modality-control spec)
   - Auto-draft `.github/PR_BODY_phase6_5g_2_iextended.md`
   - Open Draft PR against `main`
6. **HOLD**: no further commits, no training code, no GPU runs until explicit `GO` token from Ash after spec review

## PR PREPARATION

- PR body: `.github/PR_BODY_phase6_5g_1_scoping.md`
- Required sections: Status, Decisions (FOUND or EMPTY), Tripwire ledger (T_cover/T_pair/T_staged/T_pause_tag/T_license/T_race — pass/fail per row), Anchors (full 64-char SHA for parent commit + this PR's merge commit + this spec + scoping ledger + outcome yaml + paused-tag if EMPTY), Next-phase section (FOUND: E2 rebase instructions; EMPTY: Draft PR link for 6.5g.2)
- Target: `main`
- Merge mode: manual (per pre-merge-gate convention)

## APPENDIX

### Anchors at branch cut

- Parent commit: `8e3a52744b07a45a14b2307a6714c184e84313bd` (= `phase-6.5-closed^{commit}`)
- `phase-6.5-closed` annotated tag object: `880a40bb...` (tag object, not commit)
- Sibling paused branch: `phase-6.5g-e2` HEAD `6d56f610a6834a8813551ea2cda27a3a96c40cea`
- E2 spec SHA-256: `ab25340b21c61bf2f1f1c44e28f569429d45041c948e71629ddd4f869d117c1d` (unchanged by this phase)
- MEMORY.md entries: 8 (as of 2026-04-20)

### SHA-anchor ledger (to be filled on completion)

- This spec SHA-256: `<computed at first commit>`
- `data/phase6_5g/scoping_ledger.tsv` SHA-256: `<filled at commit>`
- `data/phase6_5g/outcome_{found,empty}.yaml` SHA-256: `<filled at commit>`

### Inline schemas

`outcome_found.yaml`:

```yaml
outcome: FOUND
source:
  accession: <GEO/ArrayExpress/Zenodo ID>
  protocol: <one of R1 allowlist>
  repository: <one of R2 surface>
  genome_build: <hg38 | hg19-liftover>
  format: <h5ad | h5mu | 10x-outs | mtx>
  n_cells_post_qc: <int>
  license: <string>
  url: <string>
  sha256_processed_manifest: <64-char>
tripwires:
  T_cover: pass
  T_pair: pass
  T_license: pass
pair_verification:
  method: <h5ad-shared-obs | mtx-shared-barcodes | 10x-barcode-intersection>
  overlap_fraction: <float in [0.0, 1.0]>
  tier: <pass | borderline_candidate>   # pass = [0.90, 1.00]; borderline = [0.85, 0.90); fail is rejected pre-FOUND
  manual_review_decision: <null | approved | rejected>   # required iff tier == borderline_candidate
```

`outcome_empty.yaml`:

```yaml
outcome: EMPTY
rows_scanned: <int>
probes_catalog: <int>
probes_lab: <int>
tripwires:
  T_cover: pass
  T_pause_tag: pass
next_phase: phase-6.5g.2-e2-iextended
ledger_sha256: <64-char>
```

### Cross-references

- Parent phase spec: `prompts/phase6_5g_e2_k562_pretrain.md`
- Real-data blockers registry: `.github/REAL_DATA_BLOCKERS.md`
- Pre-merge gate template: memory `feedback_premerge_gate_template.md`
- 10x catalog gap record: memory `project_aivc_k562_multiome_catalog_gap.md`
- Source-vetting rule: memory `feedback_verify_data_source_before_lock.md`
