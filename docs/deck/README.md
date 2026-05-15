# QurieGen Deck Workspace

**Purpose**: Source of truth for all investor + technical presentation materials.
**Owner**: Ash Khan
**Last updated**: 2026-05-12

---

## What Lives Here

| Folder | Contents | Status |
|---|---|---|
| `source/` | Kinga's primary deck (do not edit) | Active |
| `prompts/` | Reusable prompts for Claude Chat / Cowork / Design | Building |
| `content/` | Slide content drafts (markdown specs before generation) | Building |
| `assets/diagrams/` | Vector + raster architecture diagrams | Building |
| `assets/icons/` | Reusable icon set | Building |
| `templates/` | PowerPoint templates + master layouts | TBD |
| `exports/` | Final .pptx + .pdf outputs (gitignored) | Building |

---

## Working Principle

**Kinga's deck is the primary source**. We extend and supplement, never edit.

Two extension artifacts:
1. **AIVC Technical Appendix Deck** — 10-15 slides mapped to Kinga's slides 8/9/37 + roadmap + budget
2. **Architecture diagram library** — reusable visuals for deck, papers, technical docs

---

## Tool Division Of Labor

| Task | Tool | Output |
|---|---|---|
| Slide content spec (per slide markdown) | Claude Chat (Ash + Claude) | `content/<slide_name>.md` |
| .pptx generation from spec | Cowork | `exports/<deck_name>.pptx` |
| Architecture diagrams (SVG/vector) | Cowork or Claude Code | `assets/diagrams/<name>.svg` |
| Roadmap visualizations | Cowork | `assets/diagrams/roadmap_*.svg` |
| Budget charts | Cowork | `assets/diagrams/budget_*.svg` |
| Visual polish / hero diagrams | Claude Design | `assets/diagrams/<name>_design.svg` |
| Iteration + review | Claude Chat | Inline review against current state |

---

## Source Materials

- `source/QurieGen_SEED_ROUND_05_2026_new.pptx` — Kinga's primary deck (40 slides)
- See `source/README.md` for version history and update protocol

---

## Generated Deliverables

- AIVC Technical Appendix Deck (target: ~10-15 slides)
- Roadmap slides (28-month quarterly + 5-year horizon)
- Budget allocation slide (model vs lab vs team vs infra)
- Architecture diagram library (reusable across deck, papers, technical docs)

---

## Architectural Context (Phase 1/2/3 Confirmed Reality)

See `docs/specs/stage3_part2_architecture_proposal_2026_05_06.md` for canonical spec.

Phase 1 QurieSeq (Q3 2026):
- 5 donors, 5 timepoints (0/5/30/60/180 min), 4-arm design
- RNA + Protein measured; ATAC integration TBD (pending Kinga clarification)
- BTK + JAK combo CONFIRMED → headline zero-shot synergy demo

Phase 2 (Q1-Q2 2027): Phospho (~17 panels) + VDJ + 20 donors + ATAC measurement decision
Phase 3 (Q3 2027+): B-cell line CRISPR + disease-state samples + L3 internal data

Architectural trajectory:
- Stage 3a (now): Frozen encoder + adapter + decomposed readout
- Stage 3b (Q3 2026): Neural ODE temporal on QurieSeq Phase 1
- Stage 3c (Q1 2027): Phospho decoder + pathway alignment
- Stage 4 (Q2-Q4 2027): VDJ + 20 donors + cross-disease transfer
- Stage 5 (2028+): Causal-readiness + drug-target reasoning + clinical translation
