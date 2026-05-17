# Prior Knowledge Placement Decisions — 2026-05-16

**Status**: Locked
**Inputs**: 
- `docs/deck/research/kinga_slides_8_37_extraction.md` (audit, commit `7aa9460`)
- Ash + Claude convene 2026-05-16
**Outputs this drives**: 
- New A5 content spec + SVG + pptx v3 reassembly
- Phase 4 polish prompt with concrete content additions per slide

---

## The 7 Decisions

### Q1 — Stage 3c causal architecture (Neumann / sparse GRN / direct-effect log-FC head)

**Decision**: **Path A — committed Stage 3c architecture**

**Implication**: New appendix slide **A5 — Causal Architecture Internals** added to Section A.

**Rationale**: Neumann propagation `(I−W)⁻¹ dₚ`, sparse learned GRN, and direct-effect log-FC head are the load-bearing components of the Stage 3c causal-readiness layer. Without surfacing them as a first-class slide, the appendix has the "causal-ready" claim (in E1, D1) but no visible mechanism for how causality is implemented. Investors doing diligence on the causal claim need to see the architecture, not a footnote.

**Cost**: A5 spec + SVG + pptx v3 reassembly = ~2-3 hours of work before Phase 4 begins.

---

### Q2 — STRING database role

**Decision**: **Structural prior on GRN sparsity (A5 anchor)**

**Implication**: A5 surfaces STRING as the structural prior providing edge-existence constraints on the sparse learned GRN. STRING-supported edges get L1 regularization anchors; non-STRING edges face higher sparsity pressure but are not categorically excluded.

**Rationale**: With Q1 Path A locked, STRING's architectural role becomes concrete. The "structural prior + learned sparsity" pattern is well-established in graph-neural-network and GRN-inference literature; STRING is the canonical database for this role.

---

### Q3 — Per-database output mapping (IMGT→targets, ENCODE→druggables, etc.)

**Decision**: **No A6 slide. Distributed speaker notes per architectural placement.**

**Rationale**: Kinga's per-database output mapping is investor-narrative compression, not committed output schema. Replicating it as a structured table oversells. Each database has a specific architectural role; speaker notes at the correct placement preserve technical accuracy without inventing schema commitments.

**Phase 4 actions**:
- A2 speaker notes: ENCODE peak harmonization (explicit), IMGT for VDJ (Phase 2 Q&A)
- A3 speaker notes: GO pathway enrichment 
- A5 speaker notes: STRING as GRN structural prior
- C1 speaker notes: Reactome for cell-state transition pathways

---

### Q4 — Consolidated PK stack slide vs distributed notes

**Decision**: **Distributed speaker notes. No standalone PK stack slide.**

**Rationale**: Each prior-knowledge component has a different architectural placement. Bundling them onto one slide flattens architectural meaning. The 4 prior-knowledge databases (ENCODE/IMGT/GO/Reactome) get spread across A2, A3, C1 speaker notes based on where they actually plug into the architecture. STRING anchors A5 as the GRN sparsity prior.

**Phase 4 absorbs this entirely** — no new slide needed beyond A5 (which exists for Stage 3c architecture reasons, not for PK stack reasons).

---

### Q5 — Causal attention mask `ATAC → Phospho → RNA → Protein` on A1

**Decision**: **Add as footer strip on A1 visual.**

**Implementation**: Below A1's existing 5-block architecture diagram, add a footer strip:

```
ARCHITECTURAL INVARIANT: ATAC → Phospho → RNA → Protein
(causal attention mask enforces dependency ordering)
```

**Rationale**: The causal attention mask is **in-use today in code** — not aspirational. Surfacing it visually reframes A1 from "5-block diagram" to "5-block diagram + load-bearing biological invariant." High technical-credibility addition. Concrete, defensible, sourced.

**Phase 4 action**: Visual polish addition to A1 SVG. Update A1 content spec to include the footer strip.

---

### Q6 — 5-layer vocabulary alignment (Kinga's Harmonize/Integrate/Represent/Infer/Predict vs A1's INPUT/ENCODER/TEMPORAL/READOUT/OUTPUT)

**Decision**: **Speaker note bridge in A1. Don't unify vocabularies across decks.**

**Mapping (for speaker note)**:

| Kinga (slide 8) | A1 |
|---|---|
| Harmonize | INPUT (multi-omics fusion) |
| Integrate | ENCODER (256-D latent) |
| Represent | TEMPORAL (Neural ODE) |
| Infer | READOUT (4-arm decomposed) |
| Predict | OUTPUT (perturbation response) |

**Speaker note text**:
> If asked: how does this map to slide 8's 5-layer cascade? Harmonize = INPUT, Integrate = ENCODER, Represent = TEMPORAL, Infer = READOUT, Predict = OUTPUT. Same architecture, different vocabulary — Kinga's frames the process; A1 frames the components.

**Rationale**: Unifying vocabularies across decks creates cross-deck dependencies that break if either deck iterates independently. Speaker note bridge solves the question without coupling.

---

### Q7 — 24-month trajectory (slide 8) vs D1's 11-quarter Gantt

**Decision**: **D1 11-quarter Gantt is canonical. Kinga's 24-month stays as investor summary.**

**Speaker note bridge in D1**:
> If asked: how does this 11-quarter view map to slide 8's 24-month trajectory? Slide 8 compresses the same plan into a 4-phase visual for investor narrative. D1 is the canonical per-quarter detail with explicit milestone dependencies.

**Rationale**: Both serve different audiences. D1 is technical detail (which Stage ships in which quarter). Kinga's slide 8 is strategic compression (4-phase rhythm). They're complementary, not contradictory.

---

## Net Effect On Appendix Structure

**Before this convene**: 13 content slides (A1-F1), 20-slide pptx v2 shipped at commit `7604343`.

**After implementation**:

- **New slide**: A5 — Causal Architecture Internals
- **Section A grows**: 4 slides → 5 slides
- **Content slides**: 13 → 14
- **Total deck slides**: 20 → 21 (1 cover + 6 dividers + 14 content)
- **A1 visual addition**: Architectural invariant footer strip
- **Speaker notes**: 8+ additions across A1, A2, A3, A5, C1, D1

---

## Items NOT Added (For The Record)

To make scope explicit, these audit-surfaced items did NOT result in slide additions:

- **Per-database output schema (IMGT→targets, ENCODE→druggables, ...)**: Kinga-narrative compression, not committed schema. Distributed to speaker notes only.
- **Consolidated PK stack slide**: Each database has different architectural role; bundling flattens meaning. No A6 slide.
- **5-layer cascade vocabulary unification**: Cross-deck coupling risk. Speaker note bridge only.
- **24-month trajectory alternative**: D1 canonical. Speaker note bridge only.
- **Slide 37's `Pre-existing knowledge: Databases, Cellular context, assay conditions` catch-all**: Already implicitly covered across A2, A5, C1 with specific named databases.

---

## Path Forward (Sequence)

```
STEP 1 (now)  → Commit this decisions doc
STEP 2        → Claude drafts A5 content spec
STEP 3        → You commit A5 content spec
STEP 4        → Cowork generates A5 SVG (~1.5-2h)
STEP 5        → Visual review + ship A5 SVG
STEP 6        → Cowork reassembles pptx v3 with A5 + new Section A divider position
STEP 7        → Verify v3 on Mac/PowerPoint
STEP 8        → Claude drafts Phase 4 polish prompt with concrete content additions baked in
STEP 9        → Phase 4 execution (Claude Design + speaker notes expansion)
STEP 10       → Investor-ready deck v4
```

A5 adds 2-3 hours of work before Phase 4 begins. Worth the cost because Stage 3c causal architecture is the load-bearing intellectual differentiation of the platform.

---

## Open For Future Decisions (Banked, Not Today)

- **AIVC_GRAD_GUARD environment variable**: in-use today, mentioned in audit. Phase 4 speaker note addition to A2 explaining frozen-encoder mechanism. No slide change.
- **`Pre-existing knowledge` catch-all from slide 37**: vocabulary alignment with our spec, covered in distributed speaker notes (Q3 decision).
- **Kinga's slide 8 cascade-step icons**: visual nuance noted in audit Layer 1 extraction. Phase 4 may borrow stylistically but won't replicate.
