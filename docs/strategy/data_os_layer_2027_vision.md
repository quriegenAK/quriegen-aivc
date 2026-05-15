# AIVC Data Operating System — 2027 Vision

**Status**: DEFERRED — design document, not active implementation
**Source**: ChatGPT-architect strategic deep-dive, May 2026
**Decision date**: 2026-05-11
**Decision**: Bank as 2027 vision. Build only 3 primitives now.
**Owner of revisit**: CTO / Bioinformatics Platform Lead (future hire)

---

## Decision Summary

ChatGPT-architect delivered a comprehensive enterprise-grade biological data OS architecture covering discovery, literature parsing, QC scoring, harmonization, ontology mapping, canonical schema, storage, training staging, registry, orchestration, and lineage tracking.

The full proposal is sound as a 2027+ platform vision. Building it now would be premature optimization on a problem we don't have yet.

**Current scale**: 6 datasets (Mimitou, Calderon, Parse, CIPHER-seq killed, Soskic killed, QurieSeq pending).
**Critical pathStage 3 model validation through Q3 2026, then QurieSeq integration.
**Competitive moat**: QurieSeq proprietary data + trained models. NOT general-purpose data infrastructure.

Burning engineering capacity on commodity infrastructure ahead of model validation misaligns with where competitive value accrues.

---

## What We ARE Building Now (Stage 3a/3b Byproducts)

Three primitives that earn their existence at current scale:

### 1. Canonical Biological Schema
- **Format**: Pydantic spec + docs/specs/canonical_schema.md
- **Effort**: 3-5 days
- **Why now**: Stage 3 readout heads already implicitly assume donor, timepoint, perturbation, arm, modality structure. Formalizing it now prevents organic divergence across 5+ datasets.
- **Status**: PENDING (Stage 3a Day 2-3 deliverable)

### 2. Per-Modality Loaders (aivc/io/)
- **Modules**: atac_loader, protein_loader, hto_loader, rna_loader
- **Effort**: Emerges as Stage 3a byproduct
- **Why now**: Direct response to 6 sequential bugs in prepare_mimitou_crispr.py. Each loader produces canonical-schema output regardless of source format.
- **Status**: IN PROGRESS (Day 2 PR landed mimitou_loader.py; ATAC/HTO/RNA next)

### 3. Lightweight Dataset Registry
- **Format**: data/REGISTRY.yaml
- **Effort**: 1 day
- **Why now**: Investor-grade provenance ("what datasets is your model trained on?"). Reproducibility requirement.
- **Status**: PENDING (Stage 3a Day 3-4)

---

## What's DEFERRED (Revisit Triggers Below)

From the full ChatGPT-architect proposal, the following components are NOT being built now:

- Discovery agents (literature crawling, dataset surfacing)
- Literature parsing agents (paper → metadata extraion)
- QC legitimacy scoring system
- Ontology mapping infrastructure (Cell Ontology, MONDO, etc.)
- Graph DB / vector store for biological relationships
- Multi-agent orchestration layer
- Continuous ingestion pipelines
- Embedding storage for retrieval
- Lineage/provenance tracking beyond basic registry
- Workflow engines (Airflow, Prefect)
- Multimodal metadata stores
- Human-in-the-loop validation infrastructure
- Biological scoring systems
- Automated harmonization planners

---

## Revisit Triggers

When ANY of these fire, this document becomes active scope:

1. **Dataset count exceeds 20** in active training/eval portfolio
2. **Manual normalization effort exceeds 1 person-week per dataset** (current: ~hours per dataset)
3. **Series A closes** + bioinformatics team scales to 3+ engineers
4. **Multiple product lines need shared data infrastructure** (current: single product line)
5. **Regulatory / clinical deployment** requires formal provenance + ontology compliance
6. **Client / partner datasets** start arriving (current: only Quriegen-internal proprietary)

---

## Strategic Notes

- The full ChatGPT-architect proposal is preserved BELOW as the long-term design.
- Don't dismiss this thinking when triggers fire — it's high-quality architecture work.
- DO push back if anyone wants to build a subset of this BEFORE triggers fire.
- The 3 primitives above are NOT a partial implementation of this proposal — they're disciplined Stage 3 hygienat happens to be a strict subset of what this vision would later subsume.

---

## Full ChatGPT-Architect Proposal (Preserved)

### Design Objective

A biologically-aware AI data operating system for continuous discovery, validation, harmonization, and ingestion of public multi-omics datasets into the training ecosystem.

NOT a simple dataset search app. The goal is an extensible biological data infrastructure layer that:
- discovers relevant public datasets automatically
- evaluates biological legitimacy and relevance
- extracts experimental metadata from papers/protocols
- harmonizes modalities into a canonical schema
- prepares datasets for training/inference pipelines
- continuously feeds the platform's multi-omics models

### Major Subsystems (Originally Proposed)

1. **Discovery layer** — automated dataset surfacing from GEO/SRA/ArrayExpress/CELLxGENE/ImmPort/ENCODE/HCA/PubMed/BioRxiv
2. **Literature understanding** — paper → experimental metadata extraction
3. **Metadata extraction** — protocol identification, sample composition, perturbation conditions
4. **QC/legitimacy scoring** — dataset trustworthiness and biological relevance
5. **Harmonization layer** — modality alignment, normalization, genome build reconciliatology mapping** — Cell Ontology, MONDO, EFO, Chebi, etc.
7. **Canonical schema** — donor / timepoint / perturbation / modality / pathway / cell type / disease unified structure
8. **Storage layer** — relational + graph + vector hybrid for biological relationships and embeddings
9. **Training staging** — dataset → training-ready conversion pipeline
10. **Dataset registry** — central source of truth with versioning
11. **Orchestration** — multi-agent or workflow-engine coordination
12. **Lineage/provenance tracking** — full audit trail per dataset

### Agent Architecture (Originally Proposed)

Specialized agents per subsystem:
- Dataset discovery re parsing agent
- Biological metadata agent
- Harmonization planner
- QC validation agent
- Modality normalization agent
- Ontology mapping agent
- Training staging agent

Each agent with: responsibilities, inputs/outputs, memory/context requirements, failure boundaries, validation requirements.

### Canonical Biological Schema (Originally Proposed)

Capable of representing:
- donor structure (anonymized, longitudinal-ready)
- timepoints (continuous, irregular)
- perturbations (genetic, pharmacological, environmental)
- inhibitors (target, concentration, kinetics)
- controls (vehicle, NTC, scramble)
- modalities (RNA, ATAC, Protein, Phospho, VDJ, future)
- pathways (Hallmark, KEGG, Reactome, custom)
- cell types (Cell Ontology-grounded)
- disease states (MONDO-grounded)
- sequencing protocols (per-modality)
- genome builds (hg38/hg19/T2T)
- antibody panels (CITE-seq markers)
- clonotypes (TCR/BCR, future)

Support for: future modalities, longitudinal experiments, causal modeling, dataset provenance, reproducibility.

### Multi-Omics Harmonization Strategy (Originally Proposed)

Per-modality:
- **RNA**: normalization (CPM/log/SCT), batch correction (Harmony, scVI), reference mapping
- **ATAC**: union peak handling across peak callers (genomic interval overlap, NOT exact-string matching — lesson from Phase 6.5g.2)
- **Protein**: antibody panel reconciliatio CLR transformation, ADT denoising
- **Phospho**: pathway-level pooling, antibody-specific calibration
- **VDJ**: variable-length sequence handling, clonotype clustering, immune repertoire metrics

### Biological Legitimacy + QC System (Originally Proposed)

Quantitative scoring framework covering:
- biological quality (cell-type purity, viability markers, mitochondrial fraction)
- protocol quality (technical replicates, batch design, control inclusion)
- donor quality (n_donors, demographic diversity, health status)
- modality completeness (which modalities co-measured)
- perturbation validity (NTC controls, dose-response, time-course)
- reproducibility (independent replication, code/data availability)
- compatibility with platform goals (PBMC focus, immune relevance, etc.)

With hard rejection criteria and human review boundaries.

### Storage + Indexing Strategy (Originally Proposed)

- Relational: dataset registry, sample metadata, experimental design
- Graph: biological relationships (gene → pathway → disease → drug)
- Vector: embeddings for retrieval (dataset similarity, cell similarity)
- Provenance/lineage: dataset versioning, derived-from tracking
- Retrieval: hybrid search (metadata filter + semantic vector)

### Training Pipeline Integration (Originally Proposed)
Discovery → Harmonization → Validation → Staging → Training
|             |              |           |          |
agent        deterministic  human-in-the- automated   model
+ agent       loop boundary             checkpoint
With versioning, rollback, reproducibility guarantees.

### Multi-Agent + Claude Role (Originally Proposed)

- Claude (this conversation) = architect / researcher / systems designer / computational biology collaborator
- Future agents: discovery, literature, QC, harmonization, etc.
- Hybrid: deterministic services for normalization, agents for discovery/curation

### Future-Readiness Extensions (Originally Proposed)

- Causal inference (counterfactual data structure)
- Pathway simulation (kinetic model integration)
- Drug-target reasoning (chembl/pubchem linking)
- Sub-molecular biophysics (structure-aware embeddings)
- Protein language models (ESM family)
- VDJ reasoning (clonotype evolution)
- Clinical datasets (HIPAA-compliant provenance)
- Target discovery (perturbation → phenotype prediction)
- Longitudinal patient modeling (multi-visit, treatment-aware)

### Phased Roadmap (Originally Proposed)

- **MVP**: Discovery + canonical schema + manual harmonization + basic registry
- **Phase 2**: QC scoring literature parsing + ontology mapping
- **Phase 3**: Multi-agent orchestration + graph DB + embedding retrieval
- **Long-term**: Continuous ingestion + clinical readiness + causal-modeling integration

---

## Cross-References

- Current execution plan: docs/specs/stage3_part2_architecture_proposal_2026_05_06.md (Stage 3 architecture)
- Methodology learnings: docs/eval_methodology/cross_corpus_pseudobulk_centroid_nn.md
- Repository hygiene: docs/TODO_stage3a_wrap_cleanup.md (related entropy concerns)
- This week's verdict: docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md
