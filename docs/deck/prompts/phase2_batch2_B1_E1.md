# Phase 2 Batch 2 — B1 / B2 / B3 / C1 / C2 / D1 / D2 / E1

**Owner**: Cowork (execution)
**Estimated time**: 3-4 hours for 8 diagrams + PNG previews
**Batch scope**: Sections B (validation) + C (QurieSeq) + D (roadmap+budget) + E (horizon)
**Strategy**: Final SVG batch before Phase 3 .pptx assembly. After this lands, Section A + B + C + D + E = 12 diagrams complete.

---

## Context

Section A visual style is **locked** across 4 diagrams (commits `bf605de`, `199f29e`, `c1fe71c`):

- A1 v2 — system architecture (visual anchor)
- A2 — multi-omics encoder + 73% hero
- A3 v2 — decomposed readout equation
- A4 — temporal Neural ODE trajectory

Style assets:
- Colors: `docs/deck/assets/color_palette.md`
- Typography: `docs/deck/assets/typography.md`
- Icons: `docs/deck/assets/icon_inventory.md`

All 8 diagrams in this batch must visually match Section A. Same dark navy bg `#070A14`, same Inter title + Arial body typography, same `APPENDIX <ID> · <SECTION>` header pattern, same `<id> / 12` pagination, same footer divider/citation conventions.

---

## Hard Requirements (Apply To All 8 Diagrams)

### Style coherence with Section A

- Same dark background `#070A14` + corner radial glows (cyan top-right, purple bottom-left)
- Same Inter title typography (32-44pt), Arial body (14-18pt)
- Same `APPENDIX <ID> · <SECTION>` cyan eyebrow header at y=55
- Same large title at y=93 (~40pt)
- Same sub-headline at y=136 (~18pt muted)
- Same `<id> / 12` cyan pagination indicator at top-right
- Same source citation footer at y=H-100 (1980 px from top in 1080-height)
- Same card style: dark `#0F1428` fill, 1.5px stroke, rx=14, stroke-opacity 0.65
- Same body-bullet typography (Arial 15pt, `›` chevron in section accent color)
- 1920×1080 viewBox

### Section accent rotation

Each section gets a dominant accent that anchors its visual identity:

| Section | Primary accent | Secondary accent | Step number color rotation |
|---|---|---|---|
| A — Architecture (LOCKED) | cyan `#26DDF9` | purple `#8B5CF6` | 01 cyan → 02 lavender → 03 cyan → 04 purple → 05 white |
| B — Validation | green `#4ADE80` | cyan `#26DDF9` | B1 green / B2 cyan / B3 cyan-with-amber-accent |
| C — QurieSeq | cyan `#26DDF9` (proprietary moat = cyan = our color) | purple `#8B5CF6` | C1 cyan / C2 cyan |
| D — Roadmap+Budget | lavender `#B47DF0` | white `#EAF6FF` | D1 lavender / D2 white |
| E — Horizon | white `#EAF6FF` / pale gradient | cyan `#26DDF9` | E1 white-to-pale |

The visual rhythm Section A → B → C → D → E should feel like **progression** through accent colors, not random.

### Paired SVG + PNG (HARD REQUIREMENT)

For every diagram, ship BOTH artifacts in the same commit:
- `docs/deck/assets/diagrams/<slide_id>.svg`
- `docs/deck/assets/diagrams/<slide_id>_preview.png` (rendered at 1920×1080 via cairosvg)

Build scripts (`_build_<slide_id>.py`) generate BOTH artifacts. PNG generation is part of the build, not a separate manual step.

### Verification before commit

For each diagram, run textual acceptance checks BEFORE staging:
- No "Trimodal" anywhere (banned term)
- No `210-D panel` (banned term)
- All specs match the content spec in `docs/deck/content/<slide_id>.md`
- SVG validates as well-formed XML (`xmllint --noout`)
- Visual width calculations use **visible character count**, not HTML tspan markup length (per A3 v2 lesson)

### Build script pattern

Same template all 8 follow:

```python
#!/usr/bin/env python3
"""Build <ID> SVG + PNG preview."""

# imports, colors from color_palette.md, fonts from typography.md
# common helpers shared across builders (consider extracting to _common.py)

def build_svg():
    """Generate SVG XML."""
    # ... layout code with visible-char-width calculations ...
    return svg_string

def build_png_preview(svg_path, png_path):
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                     output_width=1920, output_height=1080)

if __name__ == "__main__":
    svg_path = "<ID>_<topic>.svg"
    png_path = "<ID>_<topic>_preview.png"
    with open(svg_path, "w") as f:
        f.write(build_svg())
    build_png_preview(svg_path, png_path)
```

---

## Per-Diagram Specifications

For each slide below, the **content spec** in `docs/deck/content/` is canonical for headline, sub-headline, body, claims, source data, and speaker notes. The visual spec below tells you the SVG-specific layout.

---

### B1 — Methodology: Three Datasets

**Content spec**: `docs/deck/content/B1_methodology_rigor.md`
**Output**:
- `docs/deck/assets/diagrams/B1_three_datasets_methodology.svg`
- `docs/deck/assets/diagrams/B1_three_datasets_methodology_preview.png`
- `docs/deck/assets/diagrams/_build_b1.py`

**Headline**: "Methodology rigor is the moat before the moat."
**Section accent**: green `#4ADE80`

**Layout**:

**Top zone — Three dataset cards (3-column grid)**:

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│  DOGMA-seq             │  │  Calderon 2019         │  │  Mimitou ASAP-seq      │
│  Mimitou 2021          │  │  Immune cell atlas     │  │  CRISPR sub-study      │
│                        │  │                        │  │                        │
│  RNA + ATAC + Protein  │  │  Bulk + scATAC,        │  │  ATAC + Protein + HTO  │
│  6 donors              │  │  stim-driven PBMCs     │  │  CRISPR-perturbed       │
│  ~30K cells            │  │  different protocol    │  │  CD4 T cells           │
│                        │  │                        │  │                        │
│  ┌──────────────────┐  │  │  ┌──────────────────┐  │  │  ┌──────────────────┐  │
│  │ ROLE             │  │  │  │ ROLE             │  │  │  │ ROLE             │  │
│  │ Encoder          │  │  │  │ Cross-corpus     │  │  │  │ Perturbation     │  │
│  │ pretraining      │  │  │  │ validation       │  │  │  │ adapter probe    │  │
│  └──────────────────┘  │  │  └──────────────────┘  │  │  └──────────────────┘  │
│                        │  │                        │  │                        │
│  RESULT: encoder       │  │  RESULT: 73% cell-type │  │  RESULT: 0.57 synergy  │
│  → 256-D latent        │  │  accuracy              │  │  4-class accuracy      │
│                        │  │                        │  │  → ADAPTER_RECOMMENDED │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
        ↓                              ↑                              ↑
   pretrain encoder              validate encoder              probe encoder on
                                                                  perturbations
                              ──────────────────────
                              No data overlap between roles
                              ──────────────────────
```

Each card gets a distinct accent tone:
- DOGMA-seq: deep cyan `#26DDF9`
- Calderon: green `#4ADE80` (the section accent — the validation dataset)
- Mimitou CRISPR: lavender `#B47DF0`

Reinforces independence visually.

The "No data overlap between roles" callout in the center: visible horizontal divider with the text in muted-white below it. This is the **most important sentence on the slide** — make it readable but not flashy.

**Bottom zone — Pre-registration workflow** (left-to-right):

```
[1. Spec written]  →  [2. Eval defined]  →  [3. Results generated]  →  [4. Verdict applied]
   architecture          pseudo-bulk            pretrained encoder         ADAPTER_RECOMMENDED
   spec v1.1             centroid-NN            on Calderon                per pre-registered
   May 2026              methodology            + Mimitou CRISPR           threshold
                         (docs/eval_meth*)

   ←──── methodology pre-registered BEFORE results ────→
```

4 stepped boxes with arrows between them. Use a thin connecting line under boxes with the "←──── methodology pre-registered BEFORE results ────→" annotation.

**Acceptance checks**:
- "DOGMA-seq", "Calderon", "Mimitou" all present as dataset names
- "Mimitou 2021" attribution present
- "ROLE" appears 3x (once per card)
- "73%" present
- "0.57" present (Mimitou result)
- "ADAPTER_RECOMMENDED" present
- "No data overlap" present in central callout
- "pre-registered" appears at least once (lowercase or uppercase)
- Banned: NO "Trimodal", NO "210-D panel"

---

### B2 — Encoder Probe: ADAPTER_RECOMMENDED

**Content spec**: `docs/deck/content/B2_encoder_probe_verdict.md`
**Output**:
- `docs/deck/assets/diagrams/B2_adapter_verdict.svg`
- `docs/deck/assets/diagrams/B2_adapter_verdict_preview.png`
- `docs/deck/assets/diagrams/_build_b2.py`

**Headline**: "0.57 synergy accuracy on held-out perturbations. Verdict: ADAPTER_RECOMMENDED."
**Section accent**: cyan `#26DDF9` (the result is the moat)

**Layout**:

**Top zone — pre-registered threshold table** (the visual hero):

```
┌────────────────────────────────────────────────────────────────────────┐
│  Pre-registered verdict thresholds  (architecture spec v1.1, §5)       │
│                                                                        │
│  Synergy 4-class accuracy        Verdict                  Action       │
│  ────────────────────────       ─────────────────────    ─────────────│
│                                                                        │
│   ≥ 0.80                         FROZEN_ENCODER_OK         Use as-is   │
│                                                                        │
│  ◆ 0.50 — 0.80 ◆                 ◆ ADAPTER_RECOMMENDED ◆   Train       │
│  ◆ ← WE ARE HERE                                           lightweight │
│  ◆     0.57 ◆                                              adapter     │
│                                                                        │
│   < 0.50                         FINE_TUNE_REQUIRED        Re-train    │
│                                                            encoder     │
└────────────────────────────────────────────────────────────────────────┘
```

Visual treatment:
- Threshold table looks like a structured protocol document — boxed, semi-monospaced numbers, clean alignment
- The middle row (ADAPTER_RECOMMENDED) gets a **strong visual emphasis**: cyan-tinted fill background, brighter stroke, "WE ARE HERE" pointer or badge
- The 0.57 number inside that row should be visually prominent (large, cyan)

**Bottom zone — per-class bar chart**:

A horizontal bar chart:

```
Arm:                              Accuracy:       Chance = 0.25
                              0.0      0.5     1.0
CD3E (TCR pathway)            ████████████████  0.91   ← strong
CD3E + CD4 (double KO)        █████████████     0.68   ← synergy demo target (highlighted)
NTC (no perturbation)         ██████            0.39
CD4 (single KO)               ██████            0.39
                              ─────|──────────
                              chance baseline 0.25

Caption: 0.57 overall 4-class accuracy = 2.27× chance. Random projection baseline 0.29 (sanity). Raw TF-IDF 0.50 (encoder near input ceiling, the right regime for adapter strategy).
```

Bars colored:
- CD3E: muted green `#4ADE80` (strong baseline)
- CD3E+CD4 double-KO: **brand cyan `#26DDF9` accent — the synergy demo target**, with a small "synergy" badge to its right
- NTC: neutral grey
- CD4: neutral grey

The 0.25 chance line drawn as a dashed vertical line through the chart.

**Acceptance checks**:
- "0.57" present (overall accuracy)
- "0.91", "0.68", "0.39" all present (per-class)
- "0.25" present (chance baseline)
- "0.29" present (random projection baseline)
- "0.50" present (TF-IDF baseline)
- "ADAPTER_RECOMMENDED" present
- "FROZEN_ENCODER_OK" present
- "FINE_TUNE_REQUIRED" present
- "WE ARE HERE" present
- "2.27×" or "2.27x" present
- Banned: NO "Trimodal", NO "210-D panel"

---

### B3 — Mechanism Validation: Public-Data Substitute

**Content spec**: `docs/deck/content/B3_synergy_pre_demo.md`
**Output**:
- `docs/deck/assets/diagrams/B3_mechanism_pre_demo.svg`
- `docs/deck/assets/diagrams/B3_mechanism_pre_demo_preview.png`
- `docs/deck/assets/diagrams/_build_b3.py`

**Headline**: "The synergy mechanism validates on public data — before BTK+JAK runs on our own."
**Section accent**: cyan `#26DDF9` + amber `#FBBF24` (for "coming Q3 2026" framing)

**Layout**:

**Top zone — Two parallel columns** (the structural substitution):

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  PUBLIC-DATA SUBSTITUTE          │    │  QURIESEQ PHASE 1                │
│  (Mimitou CRISPR — today)        │    │  (Q3 2026)                       │
│                                  │    │                                  │
│  Training arms (seen):           │    │  Training arms (will be seen):   │
│  • CD3E single KO                │    │  • BTK inhibitor alone           │
│  • CD4 single KO                 │    │  • JAK inhibitor alone           │
│  • ZAP70 single KO               │    │  • Other inhibitor singles       │
│  • NFKB2 single KO               │    │  • All 4-arm controls            │
│  • NTC                            │    │                                  │
│                                  │    │                                  │
│  ━━━━━━━━━━━━━━━━━━━━            │    │  ━━━━━━━━━━━━━━━━━━━━            │
│                                  │    │                                  │
│  ◆ HELD OUT:                     │    │  ◆ HELD OUT:                     │
│  CD3E + CD4 double KO            │    │  BTK + JAK combination           │
│  (architectural substitute       │    │  (the clinical demo target)      │
│   for BTK+JAK test)              │    │                                  │
│                                  │    │                                  │
│        ↓                          │    │        ↓                          │
│      PREDICT                     │    │      PREDICT                     │
│      zero-shot                   │    │      zero-shot                   │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

Left column: cyan-toned accent (current). Right column: amber-toned accent (future).
The "HELD OUT" callouts (◆) in both columns use the same brand cyan to visually link them — same mechanism, different perturbation.

**Middle zone — 3-card substitute justification row** (smaller, supporting):

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  ARCHITECTURE        │  │  DATA                │  │  MECHANISM           │
│                      │  │                      │  │                      │
│  Same 4-arm          │  │  Mimitou double-KO   │  │  Each single        │
│  decomposed readout  │  │  has 74 cells post-  │  │  perturbation       │
│                      │  │  split — sufficient  │  │  alters TCR         │
│  Same zero-arm L2    │  │  for pre-registered  │  │  signaling          │
│  constraint          │  │  eval with bootstrap │  │                     │
│                      │  │  CI                  │  │  Double KO yields   │
│  Same synergy head   │  │                      │  │  non-additive       │
│  trained only on     │  │  Target ≥0.70 zero-  │  │  phenotype the      │
│  residual            │  │  shot synergy        │  │  synergy head must  │
│                      │  │  accuracy            │  │  learn              │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

These cards are supporting evidence — keep them smaller than the top zone. Subdued visual weight.

**Bottom zone — Clinical grounding footer** (single row, citation-style):

```
WHY THIS COMBINATION MATTERS:
• Ibrutinib (BTK inhibitor) + Ruxolitinib (JAK1/2 inhibitor) — CLL Phase Ib/II trial NCT02912754
• Maddocks 2016, Blood (PMID 26819050) — published clinical rationale
• Thiago wet-lab finding (Quriegen IP): pJAK1 unexpectedly active in BCR pathway
```

Small text size, citation aesthetic. Functions as a credibility footer linking the architectural test to real clinical context.

**Acceptance checks**:
- "Mimitou" and "QurieSeq" both present
- "CD3E", "CD4", "ZAP70" or "NFKB2" present (training arm names)
- "BTK" and "JAK" both present
- "HELD OUT" appears at least twice (once per column)
- "PREDICT" appears at least twice (once per column)
- "NCT02912754" present (trial ID)
- "PMID 26819050" or "Maddocks" present
- "pJAK1" present (wet-lab finding)
- "≥0.70" or "0.70" present (Stage 3a target)
- Banned: NO "Trimodal", NO "210-D panel"

---

### C1 — QurieSeq Phase 1 Experimental Design

**Content spec**: `docs/deck/content/C1_phase1_design.md`
**Output**:
- `docs/deck/assets/diagrams/C1_phase1_experimental_design.svg`
- `docs/deck/assets/diagrams/C1_phase1_experimental_design_preview.png`
- `docs/deck/assets/diagrams/_build_c1.py`

**Headline**: "The data architected for the model. Phase 1 lands Q3 2026."
**Section accent**: cyan `#26DDF9` (proprietary moat = our color)

**Layout**:

**Top zone — Experimental design grid**:

Render this as a structured table grid:

```
                            T I M E P O I N T S
                  0min     5min     30min    60min    180min
                ┌────────┬────────┬────────┬────────┬────────┐
DONOR 1         │ V|S|I|C│ V|S|I|C│ V|S|I|C│ V|S|I|C│ V|S|I|C│  ← RNA + Protein per cell
                ├────────┴────────┴────────┴────────┴────────┤
                │ ◆ ATAC (chromatin signature) at t=0 only   │
                └─────────────────────────────────────────────┘

DONOR 2         │ ... same 4-arm × 5-timepoint × RNA+Protein × ATAC at t=0 ...
DONOR 3         │ ... same ...
DONOR 4         │ ... same ...
DONOR 5         │ ... same ...

LEGEND: V = vehicle, S = stim alone, I = inhibitor alone, C = stim + inhibitor combo

TOTAL: 5 donors × 5 timepoints × 4 arms × ~5,000 cells = ~500,000 cells
MODALITIES: RNA + Protein per cell. Donor-level ATAC at t=0.
```

The grid visual should:
- 5 donor rows (showing detail only for Donor 1; rows 2-5 abbreviated with "same as above")
- 5 timepoint columns
- Each cell shows the 4-arm structure as `V|S|I|C` notation
- ATAC bar spans the full row for Donor 1 showing it's a donor-level static input
- Large "~500,000 cells" total prominently displayed
- "BTK + JAK combo CONFIRMED" callout with green checkmark visible

**Bottom zone — 3-card "WHY" row**:

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│  WHY 5 TIMEPOINTS      │  │  WHY 4-ARM PER PERT    │  │  WHY 5 DONORS          │
│                        │  │                        │  │                        │
│  • Phospho-level       │  │  • Vehicle = baseline  │  │  • Donor-conditioned   │
│    signaling at 5 min  │  │  • Stim = activation   │  │    static context      │
│    (Phase 2 ready)     │  │    only                │  │    (chromatin per      │
│                        │  │  • Inh = inhibition    │  │    donor)              │
│  • Transcriptional     │  │    only                │  │                        │
│    onset at 30 min     │  │  • Stim+Inh = synergy  │  │  • 5 biological        │
│                        │  │                        │  │    replicates of the   │
│  • Stable phenotype    │  │  • Direct match to     │  │    full 5×4 grid       │
│    at 180 min          │  │    decomposed readout  │  │                        │
│                        │  │    architecture (A3)   │  │  • Phase 2 scales to   │
│                        │  │                        │  │    20 donors for       │
│                        │  │  • Held-out arm = zero-│  │    cross-donor         │
│                        │  │    shot synergy demo   │  │    generalization      │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
```

Each card connects an experimental design choice to a specific architectural need.

**Critical visual element**:

A prominent "BTK + JAK combo CONFIRMED for Phase 1" callout — green check icon, cyan accent text, positioned near the top of the experimental grid so it reads immediately. **This is the most important element on the slide for investor confidence.**

**Acceptance checks**:
- "5 donors" present
- "5 timepoints" or "5 timepoint" present
- "0 min" or "0min", "5 min" or "5min", "30 min", "60 min", "180 min" all present
- "500,000" or "500K" or "~500K" present (total cell count)
- "RNA + Protein" or "RNA+Protein" present
- "ATAC" present
- "BTK + JAK" or "BTK+JAK" present with "CONFIRMED" near it
- "vehicle", "stim", "inhibitor" or "inh" all present (4-arm names)
- Banned: NO "Trimodal", NO "210-D panel"

---

### C2 — BTK+JAK Demo: Pre-Registered Eval

**Content spec**: `docs/deck/content/C2_btk_jak_demo.md`
**Output**:
- `docs/deck/assets/diagrams/C2_btk_jak_demo_plan.svg`
- `docs/deck/assets/diagrams/C2_btk_jak_demo_plan_preview.png`
- `docs/deck/assets/diagrams/_build_c2.py`

**Headline**: "The eval that defines the platform's first investor-grade demo."
**Section accent**: cyan `#26DDF9` (continues C1) + GREEN/AMBER/RED for thresholds

**Layout**:

**Top zone — Eval flow diagram** (left-to-right):

```
   STAGE 3b TRAINING                          STAGE 3b EVAL
   ─────────────────                          ─────────────

   ┌──────────────────────┐
   │ Single-arm data      │
   │ ─────────────────    │
   │ • BTK alone          │
   │ • JAK alone          │──┐
   │ • IKK16 alone        │  │
   │ • Idelalisib alone   │  │      ┌──────────────────────┐
   │ • Rapamycin alone    │  │      │ Trained model:       │     ┌─────────────────┐
   │ • Vehicle controls   │  │      │                      │     │ Predict          │
   │ • All stimuli        │  │      │ • Frozen encoder     │     │ zero-shot:       │
   │   (LPS, IFN, etc.)   │  ├──→   │ • Trained adapter    │ ──→ │                  │
   │ • Other combos       │  │      │ • 4-arm readout      │     │ BTK + JAK combo  │
   │   (NOT BTK+JAK)      │  │      │ • Neural ODE         │     │ response         │
   └──────────────────────┘  │      │   temporal           │     │ trajectory       │
                              │      └──────────────────────┘     │ 0 → 180 min      │
                              │                                    │                  │
                              │                                    │ Score vs         │
                              │                                    │ measured combo   │
   ┌──────────────────────┐  │                                    └─────────────────┘
   │ HELD OUT during       │  │                                           │
   │ training:             │  │                                           │
   │                       │──┘                                           │
   │ ◆ BTK + JAK combo    ◀───────────────────────────────────────────────┘
   │                       │                                       score
   └──────────────────────┘
```

Visual: held-out box bypasses training, arrow goes directly to the prediction step on the right. Makes "zero-shot" structurally obvious.

**Middle zone — Pre-registered verdict thresholds table**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Pre-registered Stage 3b verdict thresholds  (spec v1.1, §5.1)           │
│                                                                          │
│  Zero-shot synergy accuracy            Verdict       Action              │
│  ─────────────────────────             ─────────     ────────────────    │
│                                                                          │
│  ≥ 0.75                                ◆ GREEN ◆     Demo ready,         │
│                                                      publish + show      │
│                                                                          │
│  0.65 — 0.75   bootstrap CI            ◆ GREEN ◆     Demo ready,         │
│                  includes 0.70                       publish with CI     │
│                                                                          │
│  0.65 — 0.75   bootstrap CI            ◆ AMBER ◆     Expand sample,      │
│                  excludes 0.70                       re-run               │
│                                                                          │
│  0.55 — 0.65   regardless of CI        ◆ AMBER ◆     Reduce λ_zero,      │
│                                                      re-train             │
│                                                                          │
│  < 0.55                                ◆ RED ◆       Architecture-class  │
│                                                      pivot — SDE fallback│
│                                                      (spec §7.1)         │
└──────────────────────────────────────────────────────────────────────────┘
```

Color-code GREEN rows green `#4ADE80`, AMBER rows amber `#FBBF24`, RED row red `#FF4D6D`. Diamond bullet `◆` per row in the verdict color.

**Bottom zone — Clinical context bridge** (single line):

```
WHY THIS COMBINATION MATTERS:
Ibrutinib (BTK) + Ruxolitinib (JAK1/2) — CLL Phase Ib/II trial NCT02912754 · 
Maddocks 2016, Blood (PMID 26819050) · Thiago wet-lab IP: pJAK1 unexpectedly 
active in BCR pathway → biological rationale for the combination
```

**Acceptance checks**:
- "HELD OUT" present
- "BTK + JAK" or "BTK+JAK" present
- "zero-shot" or "zero shot" present
- "0.75", "0.70", "0.65", "0.55" all present (thresholds)
- "GREEN", "AMBER", "RED" all present
- "NCT02912754" present
- "Maddocks" or "PMID 26819050" present
- "pJAK1" present
- "SDE" present (fallback mention)
- Banned: NO "Trimodal", NO "210-D panel"

---

### D1 — Quarterly Roadmap Q3 2026 → Q4 2028

**Content spec**: `docs/deck/content/D1_quarterly_roadmap.md`
**Output**:
- `docs/deck/assets/diagrams/D1_quarterly_roadmap.svg`
- `docs/deck/assets/diagrams/D1_quarterly_roadmap_preview.png`
- `docs/deck/assets/diagrams/_build_d1.py`

**Headline**: "11 quarters. 5 stages. Two drug pipelines. One coherent platform plan."
**Section accent**: lavender `#B47DF0`

**Layout**:

**Full-slide horizontal Gantt with 4 swimlanes**:

```
TIMELINE          Q3'26    Q4'26    Q1'27    Q2'27    Q3'27    Q4'27    Q1'28    Q2'28    Q3'28    Q4'28
                 ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
                 │      │      │      │      │      │      │      │      │      │      │

LANE 1           │██████████│
WET LAB          │ Phase 1   │█████████████████│
                 │ delivery  │ Phase 2 (phospho│███████████████████████│
                 │           │ + VDJ + 20      │  Phase 3 (B-cell line  │
                 │           │ donors)         │  + disease samples)    │
                 │           │                  │                         │
                 ↓                              ↓                          ↓
LANE 2           │██████│   │
MODEL            │ 3a    │██████│
ARCHITECTURE     │       │ 3b   │ ◆◆◆ BTK+JAK ZERO-SHOT DEMO ← (Q4 2026)
                 │       │  │██████│
                 │       │  │ 3c  │██████████████│
                 │       │       │    Stage 4   │█████████████████│
                 │       │       │              │   Stage 5       │
                 │       │       │              │                  │
LANE 3           │       │       │██████████████████████████████████│
DRUG PIPELINES   │       │       │ Pipeline 1                       │
                 │       │       │ (starts Q1-Q2 2027)              │
                 │       │       │                  │██████████████│
                 │       │       │                  │ Pipeline 2   │
                 │       │       │                  │ + Pipeline 1 │
                 │       │       │                  │ target valid │
                 │       │       │                                  │
LANE 4           │██████████│   │                  │██████████████│
PUBLICATIONS     │ Stage 3   │   │                  │ Stage 4 + 5  │
& INVESTOR DEMOS │ verdict + │   │                  │ peer-reviewed│
                 │ BTK+JAK   │   │                  │ publication  │
                 │ demo deck │   │                  │              │
                 └─────────────────────────────────────────────────────────────────────────┘
                 Q3'26    Q4'26    Q1'27    Q2'27    Q3'27    Q4'27    Q1'28    Q2'28    Q3'28    Q4'28

Milestone markers (◆ diamond) at top labeled:
◆ Q3 2026 — QurieSeq Phase 1 lands
◆ Q4 2026 — BTK+JAK ZERO-SHOT DEMO ← visual anchor (largest marker)
◆ Q1 2027 — Phase 2 phospho onboarded
◆ Q2 2027 — Drug pipeline 1 starts
◆ Q4 2027 — Stage 4 wraps (VDJ + 20 donors)
◆ Q2 2028 — Drug pipeline 2 starts, pipeline 1 target validation
◆ Q4 2028 — Stage 5 wraps (causal + clinical-ready)
```

**Critical visual elements**:
- Q4 2026 BTK+JAK demo bar gets brand cyan `#26DDF9` accent — make it visually unmistakable as the slide's anchor milestone
- 4 swimlane colors: warm (wet lab) / cool (model) / accent (pipelines) / neutral (publications)
- Dotted dependency arrows between lanes (Phase 1 → 3b; Phase 2 phospho → 3c; Phase 2 VDJ → Stage 4)
- Quarter dividers as faint vertical lines
- 7 diamond ◆ milestone markers at the top with brief labels

**Acceptance checks**:
- "Q3 2026", "Q4 2026", "Q1 2027", "Q2 2027", "Q3 2027", "Q4 2027", "Q1 2028", "Q2 2028", "Q3 2028", "Q4 2028" — all 10 quarters present
- "Phase 1", "Phase 2", "Phase 3" all present
- "Stage 3a", "Stage 3b", "Stage 3c", "Stage 4", "Stage 5" all present
- "BTK+JAK" or "BTK + JAK" present with "DEMO" near it
- "Pipeline 1" and "Pipeline 2" both present
- "QurieSeq" present
- "VDJ" present
- "phospho" present
- "causal" present (Stage 5)
- Banned: NO "Trimodal", NO "210-D panel", NO "Series A", NO "IPO"

---

### D2 — Seed Allocation: Where The $10M Goes

**Content spec**: `docs/deck/content/D2_seed_allocation.md`
**Output**:
- `docs/deck/assets/diagrams/D2_seed_allocation.svg`
- `docs/deck/assets/diagrams/D2_seed_allocation_preview.png`
- `docs/deck/assets/diagrams/_build_d2.py`

**Headline**: "$10M seed → 10 quarters of platform execution."
**Section accent**: lavender `#B47DF0` (continues D1) + white `#EAF6FF`

**Layout**:

**Top zone — Stacked horizontal bar** showing $10M allocation:

```
$10M SEED ROUND ALLOCATION
──────────────────────────

WET LAB (Phase 1 + 2 prep)            ████████████████████░░░░░░░░░░  40%   $4.0M
AI/ML TEAM + COMPUTE                   █████████████░░░░░░░░░░░░░░░░  25%   $2.5M  ← AI investment
WET LAB TEAM                           ███████░░░░░░░░░░░░░░░░░░░░░░  15%   $1.5M
BUSINESS DEVELOPMENT                   █████░░░░░░░░░░░░░░░░░░░░░░░░  10%   $1.0M
G&A + IP + LEGAL                       █████░░░░░░░░░░░░░░░░░░░░░░░░  10%   $1.0M
                                       ─────────────────────────────────────────
                                                                       100%   $10M
```

The 25% AI/ML row should be visually emphasized — investors care that the model gets serious investment.

Use distinct accent colors per category:
- Wet Lab Phase 1+2: warm tone (cyan `#26DDF9`)
- AI/ML Team + Compute: lavender `#B47DF0` (brand accent — the model)
- Wet Lab Team: muted cyan
- BD: green `#4ADE80`
- G&A + IP + Legal: neutral grey

**Bottom zone — 3-card strategic re-grouping**:

```
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│  DATA ENGINE            │  │  MODEL ENGINE          │  │  COMMERCIAL BACKBONE   │
│  $5.5M (55%)            │  │  $2.5M (25%)            │  │  $2.0M (20%)            │
│                        │  │                        │  │                        │
│  • Phase 1 delivery    │  │  • Stage 3a/3b/3c       │  │  • Pharma BD pipeline  │
│    Q3 2026             │  │    training (3 stages) │  │    2027 onwards         │
│  • Phase 2 phospho     │  │  • BTK+JAK demo Q4 2026│  │  • IP filings on        │
│    + VDJ onboarding    │  │  • Stage 4 + 5         │  │    architecture +       │
│    Q1-Q2 2027          │  │    extensions           │  │    QurieSeq protocol    │
│  • Phase 3 wet lab     │  │  • Compute infra (BSC + │  │  • Regulatory readiness │
│    Q3 2027+            │  │    cloud burst)         │  │  • Office + G&A         │
│  • CITE-seq + inhibitor│  │                        │  │                        │
│    procurement         │  │                        │  │                        │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
        55%                          25%                          20%
        of $10M                      of $10M                      of $10M
   compounding moat              executing platform           supporting infra
```

DATA ENGINE = Wet Lab Phase 1+2 + Wet Lab Team (40+15=55%).
MODEL ENGINE = AI/ML Team + Compute (25%).
COMMERCIAL BACKBONE = BD + G&A (10+10=20%).

**Footer note** (small, visible):

> Allocation estimates pending CEO confirmation. See speaker notes for budget assumptions.

This is the **honesty signal** — numbers are estimates, Kinga to confirm.

**Acceptance checks**:
- "$10M" or "$10 million" present
- "40%" or "$4M" or "$4.0M" present (wet lab Phase 1+2)
- "25%" or "$2.5M" present (AI/ML)
- "15%" or "$1.5M" present (wet lab team)
- "10%" or "$1.0M" or "$1M" present (BD, G&A — 2 occurrences ok)
- "Wet Lab", "AI/ML", "Business Development" all present
- "DATA ENGINE", "MODEL ENGINE", "COMMERCIAL BACKBONE" all present
- "estimates pending" or "CEO confirmation" present (honesty disclosure)
- Banned: NO "Trimodal", NO "210-D panel", NO "Series A", NO "IPO"

---

### E1 — 5-Year Trajectory: Pipeline + Clinical Maturation

**Content spec**: `docs/deck/content/E1_five_year_trajectory.md`
**Output**:
- `docs/deck/assets/diagrams/E1_five_year_trajectory.svg`
- `docs/deck/assets/diagrams/E1_five_year_trajectory_preview.png`
- `docs/deck/assets/diagrams/_build_e1.py`

**Headline**: "From validated platform to first-in-class candidates — 2026 to 2031."
**Section accent**: white `#EAF6FF` / pale gradient (the horizon, the future)

**Layout**:

**Top zone — 4-phase horizontal progression**:

```
2026                  2027                    2028                    2029-2031
─────                 ─────                   ─────                   ─────────

┌─────────────────┐  ┌─────────────────┐    ┌─────────────────┐      ┌──────────────────┐
│ PHASE 1         │  │ PHASE 2         │    │ PHASE 3         │      │ PHASE 4          │
│ VALIDATION      │  │ EXTENSION       │    │ MATURATION      │      │ TRANSLATION      │
│                 │  │                 │    │                 │      │                  │
│ • BTK+JAK demo  │  │ • Phospho       │    │ • Causal-       │      │ • First-in-class │
│ • Phospho       │  │   integration   │    │   readiness     │      │   candidates     │
│   integration   │  │ • VDJ + 20-     │    │   layer         │      │ • Pharma         │
│ • Stage 3 wraps │  │   donor scale   │    │ • Pipeline 1    │      │   partnerships   │
│                 │  │ • Pipeline 1    │    │   target        │      │   scale          │
│                 │  │   starts        │    │   validation    │      │ • Pipeline 1     │
│                 │  │ • Pipeline 2    │    │ • Clinical      │      │   target-valid'd │
│                 │  │   starts        │    │   framework     │      │ • Pipeline 2     │
│                 │  │ • Stage 4 wraps │    │ • Stage 5       │      │   lead selection │
│                 │  │                 │    │   wraps         │      │ • Platform = OS  │
└─────────────────┘  └─────────────────┘    └─────────────────┘      └──────────────────┘
   most filled         fully colored        colored, lighter           outlined only
   (high conf.)        (high conf.)         (planned, contingent)     (directional)
```

Visual rhythm: 2026 has highest visual confidence (filled brand color), 2031 has lowest (outline only with pale gradient). **Conveys "we know exactly where we ship now, less precisely far out" — appropriate honesty about decreasing future visibility.**

**Bottom zone — 3-card compounding loops row**:

```
┌────────────────────────────┐  ┌────────────────────────────┐  ┌────────────────────────────┐
│  DATA COMPOUNDS             │  │  MODEL COMPOUNDS           │  │  CLINICAL INFRA COMPOUNDS  │
│                             │  │                            │  │                            │
│  Phase 1 (5 donors,         │  │  3 modalities → 5          │  │  Regulatory-grade          │
│  3 modal)                   │  │  modalities                 │  │  provenance                │
│                             │  │                            │  │                            │
│  Phase 2 (20 donors,        │  │  Single donor → 20-donor   │  │  Computational diligence   │
│  5 modal)                   │  │  scale                     │  │  package                   │
│                             │  │                            │  │                            │
│  Phase 3 (B-cell + disease  │  │  Static → temporal Neural  │  │  Audit trails + version    │
│  samples)                   │  │  ODE                       │  │  control                   │
│                             │  │                            │  │                            │
│  Every quarter adds wet-lab │  │  Correlation → causal-     │  │  Every milestone adds      │
│  data to training corpus    │  │  readiness                  │  │  clinical partnership      │
│                             │  │                            │  │  readiness                 │
│                             │  │  Every stage adds          │  │                            │
│                             │  │  capability without re-    │  │                            │
│                             │  │  architecting              │  │                            │
└────────────────────────────┘  └────────────────────────────┘  └────────────────────────────┘
```

These cards justify "why does the platform compound over time?" — the question that defends the 5-year investment thesis.

**Critical**:
- NO IPO/exit/Series A mentions anywhere (per Ash's strategic direction)
- 2029-2031 phase should be visually less specific — appropriate honesty about future visibility
- "Platform = OS for immune-system drug discovery" as the 2031 endpoint — ambitious but defensible

**Acceptance checks**:
- "2026", "2027", "2028", "2029" all present (year labels)
- "PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4" all present
- "VALIDATION", "EXTENSION", "MATURATION", "TRANSLATION" all present (phase titles)
- "BTK+JAK" or "BTK + JAK" present (Phase 1)
- "Pipeline 1" and "Pipeline 2" both present
- "Stage 4" and "Stage 5" both present
- "causal" present (Stage 5)
- "first-in-class" or "first in class" present (Phase 4)
- "DATA COMPOUNDS", "MODEL COMPOUNDS", "CLINICAL INFRA COMPOUNDS" all present
- Banned: NO "IPO", NO "Series A", NO "exit", NO "Trimodal", NO "210-D panel"

---

## Deliverable Sequence

Two options for commit pattern — Cowork's call:

**Pattern A — Per-diagram commits (8 commits)**:
```
docs(deck): B1 three datasets methodology SVG + preview
docs(deck): B2 encoder probe verdict SVG + preview
docs(deck): B3 mechanism pre-demo SVG + preview
docs(deck): C1 phase 1 experimental design SVG + preview
docs(deck): C2 BTK+JAK demo plan SVG + preview
docs(deck): D1 quarterly roadmap SVG + preview
docs(deck): D2 seed allocation SVG + preview
docs(deck): E1 five-year trajectory SVG + preview
```

**Pattern B — Section-grouped commits (4 commits)**:
```
docs(deck): Section B SVGs (B1+B2+B3) + previews
docs(deck): Section C SVGs (C1+C2) + previews
docs(deck): Section D SVGs (D1+D2) + previews
docs(deck): Section E SVG (E1) + preview
```

**Pattern C — Single batch commit**:
```
docs(deck): Phase 2 Batch 2 — B1-E1 SVGs + previews
```

Pattern C is fine if all 8 pass acceptance checks together. Pattern A is fine if you want per-diagram debugging granularity. Cowork's call.

---

## Self-Review Discipline (per A3 v2 lesson)

A3 v1 had a latent positioning bug that the annotation-overlap symptom masked. Cowork found it on iteration. For Batch 2, **proactively look for similar issues**:

1. **Visible character width vs HTML tspan markup width** — always count visible glyphs, not markup
2. **Off-card text** — verify every text element's x+width stays within its parent card bounds
3. **Right-edge clipping** — text with `text-anchor="end"` near the right edge can clip if x is at or past `viewBox-width`
4. **Greek/math glyph fallback** — if Inter doesn't have a glyph, the SVG `font-family` fallback chain matters; PowerPoint will render correctly on Kinga's Mac if fonts are installed

Run xmllint validation on every SVG before staging. Run textual acceptance grep checks. Both should pass before commit.

---

## What Ash Will Check On Review

For each of the 8 diagrams:

1. **Style coherence with A1-A4**: opens all 12 SVGs side-by-side. Coherent visual family across sections, with appropriate section accents.
2. **Content accuracy**: every claim traces to its content spec in `docs/deck/content/<slide_id>.md`
3. **Visual hierarchy**: hero element dominates appropriately per slide
4. **PNG renders cleanly**: opens each PNG, looks right at slide-fill scale
5. **No banned terms**: no "Trimodal", no "210-D panel", no "IPO", no "Series A" anywhere
6. **No off-card text or clipping**: every text element renders within its bounds
7. **Section-accent rotation works**: A=cyan/purple, B=green/cyan, C=cyan, D=lavender, E=white/pale reads as deliberate progression

If any diagram needs iteration, fix and re-ship before considering Batch 2 complete.

---

## What's Out Of Scope For This Batch

- A1, A2, A3, A4 — already locked, do not modify
- Any updates to `color_palette.md`, `typography.md`, `icon_inventory.md`
- Phase 3 .pptx assembly
- Phase 4 Claude Design visual polish

---

## What Comes After Batch 2

If all 8 land clean:

**Phase 3 — .pptx assembly** (~30-60 min)
Use Cowork's pptx skill to assemble final `aivc_appendix_v1.pptx` from:
- 12 content specs (`docs/deck/content/*.md`)
- 12 SVGs (`docs/deck/assets/diagrams/*.svg`)
- Speaker notes from each content spec embedded as slide notes

Output: `docs/deck/exports/aivc_appendix_v1.pptx`

**Phase 4 — Visual polish** (optional, variable time)
Claude Design refinement on hero diagrams (A1, A2, A3, B1, C1, C2 are highest priority for polish). Skip if Phase 3 output is investor-ready as-is.

**Total path to first investor-ready draft from where we are**: 4-6 hours of work after Batch 2 completes.

---

## Tool Selection Confirmation

**Cowork** (Python svgwrite/matplotlib + cairosvg PNG render) for all 8 diagrams.

Not Claude Design — that's Phase 4 polish, after structural SVGs are stable.

---

## Risks To Flag

1. **D1 Gantt density**: 4 swimlanes × 10 quarters = a lot of content. If the Gantt feels cluttered at slide-fill scale, consider reducing to 3 swimlanes (merge "Pipelines" and "Publications") or compressing Q3 2027 - Q4 2028 into a smaller horizontal slice. Cowork's call after seeing the rendered output.

2. **E1's "decreasing visual confidence" rhythm**: Phase 1 (2026) should look most-filled, Phase 4 (2029-2031) least-filled. Use opacity gradients on card fills, not just color shifts. The visual rhythm conveys our honesty about future visibility.

3. **C1's experimental grid**: rendering 5 donors × 5 timepoints × 4 arms cleanly is non-trivial. The simplification "show Donor 1 in detail, abbreviate rows 2-5" is canonical — do NOT try to show all 100 cells in the grid.

4. **D2's $10M math**: the 3-card strategic re-grouping (DATA $5.5M / MODEL $2.5M / COMMERCIAL $2M) must mathematically equal the 5-row line items (4+2.5+1.5+1+1 = $10M). Verify before commit.

5. **PNG file sizes**: 8 PNGs × ~250KB each = ~2MB additional repo size. Acceptable.

6. **Time budget**: 3-4 hours for 8 diagrams. If running long, prioritize B1, C1, D1, E1 (one per section) for first ship, then B2/B3/C2/D2 in a follow-up. Don't compromise quality for speed.
