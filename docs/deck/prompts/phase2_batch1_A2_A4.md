# Phase 2 Batch 1 — A2 / A3 / A4 SVG Generation

**Owner**: Cowork (execution)
**Estimated time**: 60-90 min for all 3 diagrams + PNG previews
**Batch scope**: Section A remaining slides (A2 + A3 + A4)
**Strategy**: 2-chunk batching. This chunk completes Section A's visual style; Section B-E batch follows after Ash review.

---

## Context

A1 v2 visual anchor approved (commit `bf605de` SVG + `<png-commit>` PNG). The visual style for the entire appendix is now locked:

- **Color palette**: `docs/deck/assets/color_palette.md` — dark navy bg, neon cyan/purple/lavender accents, slide-37 step-number rotation
- **Typography**: `docs/deck/assets/typography.md` — Inter (titles) + Arial (body) with declared fallbacks
- **Layout conventions**: 16:9, 1920×1080 viewBox, `APPENDIX <SLIDE> · <SECTION>` header eyebrow, `<slide_id> / 12` pagination indicator, source citation footer
- **Visual idioms**: 5-block horizontal flow with step numbers, body bullets prefixed with `›`, validation/status rows, pill-shaped invariant labels

A2-A4 must match A1's aesthetic. Investors should see Section A as a coherent set.

---

## Hard Requirements (Apply To All 3 Diagrams)

### Style coherence with A1

- Same dark background `#070A14` + corner radial glows (cyan + purple)
- Same Inter title typography, Arial body typography
- Same `APPENDIX <ID> · <SECTION>` cyan eyebrow header (40-80 chars max)
- Same `<id> / 12` cyan pagination indicator top-right
- Same source citation footer at y≈980
- Same dark card background `#0F1428` with 1.5px stroke at 0.65 opacity
- Same body-bullet typography (Arial 15pt, `›` chevron in step color)

### Paired SVG + PNG (HARD REQUIREMENT)

For every diagram, ship BOTH artifacts in the same commit:
- `docs/deck/assets/diagrams/<slide_id>.svg`
- `docs/deck/assets/diagrams/<slide_id>_preview.png` (rendered at 1920×1080 via cairosvg or equivalent)

GitHub's SVG rendering doesn't honor external `font-family` declarations — paired PNG is the only reliable way for visual review on GitHub and downstream pipelines.

This rule applies to A2-A4 in this batch AND to B1-E1 in the next batch. Make it part of the build script convention.

### Build script per diagram

Same pattern as `_build_a1.py`:
- One `_build_<slide_id>.py` per diagram
- Generates BOTH the SVG and the PNG preview
- Commits both artifacts in the same commit

### Verification before commit

For each diagram, run textual acceptance checks BEFORE staging:
- No "Trimodal" anywhere (we explicitly stripped this in A2 v4 / A1 v2)
- No `210-D panel` (replaced with range or removed everywhere)
- Source citation references real specs (architecture spec v1.1, Phase 6.5g.2 closure, etc.)
- SVG validates as well-formed XML (`xmllint --noout`)

---

## A2 — Multi-Omics Encoder: The Frozen Substrate

**Content spec**: `docs/deck/content/A2_encoder_substrate.md`
**Output**: 
- `docs/deck/assets/diagrams/A2_encoder_evidence.svg`
- `docs/deck/assets/diagrams/A2_encoder_evidence_preview.png`
- `docs/deck/assets/diagrams/_build_a2.py`

### Layout — three-zone

**Left zone (top, ~50% width)** — multi-omics encoder schematic:

```
Modalities (per cell):           Encoder              Latent space
───────────                       ─────                ──────────
                                                       
✅ RNA (filled, primary cyan)  ─┐                       z ∈ ℝ²⁵⁶
                                ├→ [Contrastive ─→    🔒 frozen
✅ ATAC (filled, primary cyan) ─┤    fusion]            after pretraining
                                │                       
✅ Protein (filled, primary    ─┘
   cyan)                       
                                
🟡 Phospho (outlined, dashed   ─┐  ↓ Phase 2 integration
   stroke, amber)               │
                                ─┘
🟡 VDJ (outlined, dashed,      ─┐
   amber)                       │
                                ─┘
```

- "Today" modalities (RNA + ATAC + Protein): solid fills, cyan accent, ✅ icon
- "Phase 2" modalities (Phospho + VDJ): outlined only, dashed strokes, amber `#FBBF24`, 🟡 icon
- Visual clearly distinguishes the two states

**Right zone (top, ~50% width)** — hero 73% callout:

```
┌────────────────────────────┐
│                            │
│         73 %               │  ← Large cyan number, ~96pt
│                            │
│  cross-corpus cell-type    │  ← Inter 18pt
│  accuracy on Calderon 2019 │
│                            │
│  • Independent dataset     │  ← Arial 14pt, muted
│  • Zero retraining         │
│  • Major PBMC lineages     │
│                            │
└────────────────────────────┘
```

The 73% is the slide's hero. Make it BIG. Subtle background watermark/glow optional.

**Bottom zone (full width)** — DOGMA-seq sidebar callout:

```
┌─────────────────────────────────────────────────────────────────────┐
│  DOGMA-seq (Mimitou 2021, Nat Biotech)                              │
│                                                                     │
│  • RNA + ATAC + Protein measured in the same single cell            │
│  • Primary human PBMCs (not cell lines)                             │
│  • 6 healthy donors, ~30K cells                                     │
│  • Peer-reviewed protocol                                           │
│                                                                     │
│  Source of: encoder pretraining + perturbation training             │
│  (ASAP-seq CRISPR sub-study)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

Smaller text size than the main zones (Arial 13-14pt). Functions as a credibility footer.

### Speaker-note nuances to preserve

- The word "Trimodal" must NOT appear (use "RNA + ATAC + Protein" or "Multi-omics")
- B-cell weakness handled in speaker notes only (NOT on slide)
- Per-cell-type accuracy breakdown is NOT shown (only overall 73%)

---

## A3 — Decomposed Readout: How Synergy Generalizes

**Content spec**: `docs/deck/content/A3_decomposed_readout.md`
**Output**:
- `docs/deck/assets/diagrams/A3_decomposed_readout.svg`
- `docs/deck/assets/diagrams/A3_decomposed_readout_preview.png`
- `docs/deck/assets/diagrams/_build_a3.py`

### Layout — the equation is the slide

**Center top — the 4-arm equation**, rendered LaTeX-quality (math typography):

```
ŷ(c, s, i, t)  =   h_base(z, t)                          ← always active
                  
                 + 𝟙[s] · Δ_stim(z, s, t)                 ← stim present
                 
                 + 𝟙[i] · Δ_inh(z, i, t)                  ← inh present
                 
                 + 𝟙[s ∧ i] · Δ_synergy(z, s, i, t)       ← combination only
```

- Color-code the four heads consistently for the entire deck (this becomes the reference):
  - `h_base` = pale text (`#EAF6FF`) — baseline
  - `Δ_stim` = green `#4ADE80` — activation
  - `Δ_inh` = brand purple `#8B5CF6` — inhibition
  - `Δ_synergy` = brand cyan accent `#26DDF9` — the synergy color (reuse on C2)
- Indicator functions `𝟙[·]` rendered explicitly (math notation)
- Right-side annotations (← always active, etc.) in muted grey

**Below equation — zero-arm constraint block** (boxed, theorem-style):

```
┌─────────────────────────────────────────────────────────────────────┐
│  Load-bearing constraint                                            │
│                                                                     │
│  For NTC (no stim, no inh):     Δ_stim, Δ_inh, Δ_synergy = 0        │
│  For stim-only cells:           Δ_inh, Δ_synergy = 0                │
│  For inhibitor-only cells:      Δ_stim, Δ_synergy = 0               │
│                                                                     │
│  Penalty:  L_zero_arm = λ · Σ‖Δ‖²    where condition fails          │
│  λ = 1.0  (architecture spec v1.1, §3.2.2)                          │
└─────────────────────────────────────────────────────────────────────┘
```

Visual: subdued card background, slight indent. Looks like a math theorem or lemma box.

**Right-side / inset — generalization table** (3 rows):

| Training data | Inference target | What this enables |
|---|---|---|
| Mimitou: CD3E + CD4 single KO + CD3E+CD4 double | Predict held-out CD3E+CD4 double | 0.68 validated, Stage 3 Part 1 |
| QurieSeq: BTK + JAK singles + 4-arm controls | Predict **BTK+JAK combo zero-shot** | Headline demo Q4 2026 |
| Mimitou: single perturbations | Predict any pairwise combination | Compositional generalization |

Make BTK+JAK row visually emphasized (synergy cyan accent).

---

## A4 — Temporal Dynamics via Neural ODE

**Content spec**: `docs/deck/content/A4_temporal_neural_ode.md`
**Output**:
- `docs/deck/assets/diagrams/A4_temporal_dynamics.svg`
- `docs/deck/assets/diagrams/A4_temporal_dynamics_preview.png`
- `docs/deck/assets/diagrams/_build_a4.py`

### Layout — two-zone

**Top zone — continuous trajectory plot**:

A 2D plot of `z(t)` over time t ∈ [0, 180] minutes. Show:
- Smooth continuous curve (not a polyline) — represents the latent state evolving via Neural ODE
- 5 sample points marked on the curve at t = 0, 5, 30, 60, 180 min
- Each sample point a different color reflecting biological interpretation:
  - 0 min: neutral grey — baseline
  - 5 min: cyan accent — early signaling (phospho-ready for Phase 2)
  - 30 min: brand purple — transcriptional onset
  - 60 min: lavender — peak response window
  - 180 min: pale text — stable phenotype
- x-axis labels: `0 min  5 min  30 min  60 min  180 min` (non-uniform spacing — critical visual)
- y-axis: latent state magnitude (label as `z(t)` or "Latent activation")

Annotation below trajectory:
- 0 min: baseline state (pre-perturbation)
- 5 min: early signaling (phospho-active, RNA latent)
- 30 min: transcriptional onset (RNA dynamics begin)
- 60 min: peak response window
- 180 min: stable phenotype (RNA + Protein equilibrium)

**Bottom zone — 3-card architecture-choice row**:

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Neural ODE          │  │  Latent SDE          │  │  Discrete Transformer│
│  (PRIMARY)           │  │  (FALLBACK)          │  │  (REJECTED)          │
│                      │  │                      │  │                      │
│  ✓ Continuous time   │  │  ✓ Stochastic        │  │  ✗ Fixed timesteps   │
│  ✓ Irregular sampling│  │    dynamics support  │  │  ✗ Interpolation     │
│  ✓ Deterministic     │  │  ✓ Triggers          │  │    artifacts         │
│    trajectories      │  │    documented:       │  │  ✗ Architectural     │
│  ✓ Reuses ~130K param│  │    NaN >3/100,       │  │    invariant         │
│    adapter            │  │    plateau >5 epochs,│  │    violation         │
│                      │  │    spectral radius   │  │                      │
│                      │  │    >5.0              │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
       PRIMARY                  FALLBACK                   REJECTED
       (filled brand)           (outlined)                  (greyed)
```

Visual treatment:
- **Neural ODE (PRIMARY)**: filled with brand accent background tint, full color text
- **Latent SDE (FALLBACK)**: outlined only, muted text
- **Discrete Transformer (REJECTED)**: greyed out, with subtle red `#FF4D6D` X icons on each row

Visual hierarchy makes the choice obvious in 5 seconds.

---

## Per-Slide Acceptance Criteria

Before staging, verify each SVG passes these textual checks:

### A2 acceptance
- ✓ "Multi-omics" present in title or body (NOT "trimodal")
- ✓ "Trimodal" NOT present anywhere
- ✓ "73 %" or "73%" rendered as hero number
- ✓ "DOGMA-seq" referenced with "Mimitou 2021"
- ✓ "Phospho" + "VDJ" present with Phase 2 framing
- ✓ "B cell" or per-cell-type accuracy breakdown NOT present (speaker notes only)

### A3 acceptance
- ✓ Equation: `h_base`, `Δ_stim`, `Δ_inh`, `Δ_synergy` all present
- ✓ Indicator functions `𝟙[·]` rendered (use Unicode 𝟙 or styled "1" with subscript)
- ✓ "λ = 1.0" or "λ_zero = 1.0" present in constraint block
- ✓ "0.68" present (Mimitou validation number)
- ✓ "BTK+JAK" or "BTK + JAK" present (demo target)

### A4 acceptance
- ✓ "Neural ODE" present
- ✓ "0", "5", "30", "60", "180" all present as timepoint labels
- ✓ "PRIMARY", "FALLBACK", "REJECTED" all present
- ✓ "SDE" present in fallback card
- ✓ "Discrete Transformer" present in rejected card

---

## Build Script Pattern (Same For All 3)

```python
#!/usr/bin/env python3
"""Build A<N> SVG + PNG preview."""

# ... imports, colors from color_palette.md, fonts from typography.md ...

def build_svg():
    """Generate the SVG XML."""
    # ... layout code ...
    return svg_string

def build_png_preview(svg_path, png_path):
    """Render SVG to PNG via cairosvg."""
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                     output_width=1920, output_height=1080)

if __name__ == "__main__":
    svg_path = "A<N>_<topic>.svg"
    png_path = "A<N>_<topic>_preview.png"
    
    with open(svg_path, "w") as f:
        f.write(build_svg())
    
    build_png_preview(svg_path, png_path)
    print(f"Built {svg_path} + {png_path}")
```

PNG generation is part of the build — not a separate manual step.

---

## Deliverable Sequence

### Single ship script per diagram OR one combined script

Pick whichever pattern matches the A1 v2 ship workflow. If one combined script: handle each diagram + verify checks + stage + commit independently within the script.

### Commits (one per diagram, OR one batch commit)

Cowork's call. Both acceptable:

**Pattern A — 3 separate commits**:
```
git commit -m "docs(deck): A2 multi-omics encoder SVG + preview"
git commit -m "docs(deck): A3 decomposed readout SVG + preview"
git commit -m "docs(deck): A4 temporal Neural ODE SVG + preview"
git push origin main
```

**Pattern B — one batch commit**:
```
git commit -m "docs(deck): A2-A4 batch — Section A diagrams complete

- A2: Multi-omics encoder + 73% hero callout + DOGMA-seq footer
- A3: 4-arm decomposed readout equation + zero-arm constraint
- A4: Continuous-time trajectory + Neural ODE PRIMARY card

Each diagram ships paired SVG + preview PNG. Style coherent with
A1 v2 (commit bf605de) — same colors, typography, layout idioms."
```

---

## What Ash Will Check On Review

For each of the 3 diagrams:
1. **Style coherence with A1**: opens A1 + A2/A3/A4 side-by-side. Same aesthetic family.
2. **Content accuracy**: every claim traces to its content spec (`A2_encoder_substrate.md`, etc.)
3. **Visual hierarchy**: hero element (73% for A2, equation for A3, trajectory for A4) dominates appropriately
4. **PNG renders cleanly**: opens the PNG in any viewer, looks right at slide scale
5. **No "Trimodal", no `210-D panel`, no other banned terms** (per A1 v2 decisions)

---

## What's Out Of Scope For This Batch

- B1, B2, B3, C1, C2, D1, D2, E1 — these are the next batch after A2-A4 review
- Modifications to A1 v2 — already locked
- Updates to color_palette.md or typography.md — already locked
- Anything Claude Design polish — Phase 4, after all 12 SVGs land

---

## Risks To Flag

1. **A3's equation rendering**: math typography in SVG is finicky. If Inter doesn't have proper math glyphs (Δ, 𝟙), fall back to italic style for variables and Unicode for symbols. Document any font substitutions used.

2. **A4's continuous curve**: avoid making the curve look like a polyline. Use a Bezier or spline that visually conveys "smooth differential equation trajectory", not "linear interpolation between sample points."

3. **A2's modality grid**: the "Phase 2 outlined/dashed" treatment must be visually obvious — not just a slight stroke difference. Investors should see at a glance "5 modalities, 3 today, 2 coming."

4. **PNG file sizes**: each PNG will be ~200-300KB. Three PNGs in this batch = ~700KB additional repo size. Acceptable.

5. **Build script timing**: cairosvg PNG generation takes a few seconds per diagram. Account for this in any timeout assumptions.

---

## What Comes After This Batch

If A2-A4 land clean and Ash approves:
- **Batch 2**: B1, B2, B3, C1, C2, D1, D2, E1 (8 diagrams, the validation + QurieSeq + roadmap + horizon sections)
- Then **Phase 3**: .pptx assembly via pptx skill
- Then **Phase 4**: Claude Design visual polish on hero diagrams

If any of A2-A4 needs iteration, fix and re-ship before Batch 2 starts.

---

## Tool Selection Confirmation

**Cowork** (Python matplotlib/svgwrite + cairosvg PNG render) for all three.

Not Claude Design — that's Phase 4 polish, not Phase 2 generation.
