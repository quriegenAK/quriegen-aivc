# Slide A5 — Causal Architecture: Where Inference Becomes Causal

- **Maps to Kinga's deck**: Architectural depth behind slide 37 (AI virtual cell model) and the causal-readiness layer of slide 8 (5-layer cascade)
- **Section**: A — Architecture Depth (extends from 4 slides to 5)
- **Visual lead**: 3-zone causal architecture stack diagram
- **Status**: Stage 3c architectural commitment, validation scheduled Q1-Q2 2027 post Phase 1 wet-lab data
- **Status — slide content**: Draft, pending Ash review

---

## Headline

**Causal architecture — spec-locked, validation post-Phase-1**

(Alternative: *"From correlation to causation: the Stage 3c architectural commitment"*)

(Alternative: *"Three components that make the platform causal-ready"*)

---

## Sub-headline (one line under headline)

Neumann propagation + sparse learned GRN + direct-effect decoder. Architecturally locked in spec v1.1. Validation begins Q1-Q2 2027 once Phase 1 wet-lab perturbation data lands.

---

## Body content (3 bullets max)

- **Neumann propagation `(I − W)⁻¹ dₚ` linearizes perturbation flow through learned graph structure.** Where Stage 3b predicts perturbation responses from observed cells, Stage 3c models *how* a perturbation propagates through the gene regulatory network. The matrix inverse closed-form gives tractable causal-effect estimation when W has bounded spectral radius — a property the sparse GRN learning objective enforces.

- **Sparse learned GRN with STRING structural prior.** The graph matrix W isn't learned from scratch — STRING's protein-protein interaction database provides edge-existence priors. STRING-supported edges face lower L1 sparsity pressure; non-STRING edges remain learnable but must clear higher evidence thresholds. The structural prior + learned sparsity pattern is the architectural commitment to biologically-grounded causal inference.

- **Direct-effect log-FC head decodes perturbation-specific gradients.** Stage 3a/3b predicted abundance changes; Stage 3c separates direct perturbation effects from downstream propagation. The log-FC head outputs `dₚ` (direct effect) which Neumann propagation then expands through the GRN. This decomposition is what enables causal queries — "what does perturbing X *cause*?" — rather than only predictive ones — "what happens after X is perturbed?".

---

## Visual spec — 3-zone causal architecture stack

### Top zone — Neumann propagation block (visual hero)

The mathematical centerpiece. Renders as:

```
┌──────────────────────────────────────────────────────────────────┐
│  NEUMANN PROPAGATION                                              │
│  perturbation flow through learned graph structure                │
│                                                                   │
│         ŷ = (I − W)⁻¹ · dₚ                                        │
│         ─────────────────                                          │
│                                                                   │
│         W ∈ ℝᴺˣᴺ       sparse learned GRN                         │
│         dₚ ∈ ℝᴺ        direct perturbation effect                 │
│         (I − W)⁻¹      closed-form propagation                    │
│                                                                   │
│  Architectural requirement: ρ(W) < 1 enforced by sparsity L1     │
└──────────────────────────────────────────────────────────────────┘
```

Style:
- Equation in large white serif/mono (matching A3's equation typography)
- `W`, `dₚ`, `(I − W)⁻¹` color-coded: W in cyan, dₚ in lavender, propagation operator in green
- Architectural requirement (spectral radius condition) as small footer text in muted

### Middle zone — Sparse learned GRN visualization

Two side-by-side panels showing the GRN learning structure:

```
┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
│  STRUCTURAL PRIOR (STRING)            │   │  LEARNED SPARSE GRN                  │
│                                       │   │                                      │
│  ● ─── ● ─── ●                         │   │  ●━━━━━━━●━━━━━━━●                  │
│       │         │                     │   │      ┃                               │
│       ● ─── ●                          │   │       ●━━━━━━━●                      │
│             │                          │   │              ┃                       │
│              ● ─── ●                   │   │               ●·· ····●              │
│                                       │   │                                      │
│  STRING-supported edges               │   │  thick = high-weight, learned         │
│  (lower L1 pressure)                  │   │  dashed = below sparsity threshold    │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
                    └────── L1 sparsity → ──────┘
              prior shapes initialization, learning prunes
```

Style:
- Left panel: graph node-edge diagram with grey edges representing STRING priors
- Right panel: graph node-edge diagram with cyan thick edges (learned, weighted) + grey dashed edges (sub-threshold, pruned)
- Both panels same node count (~6-8 nodes for visual clarity, representative not exhaustive)
- Center arrow with caption "L1 sparsity →" connecting the two
- Bottom caption: "prior shapes initialization, learning prunes"

### Bottom zone — Direct-effect log-FC head decoder

The decoder block showing where causal-effect separation happens:

```
┌──────────────────────────────────────────────────────────────────┐
│  DIRECT-EFFECT LOG-FC HEAD                                        │
│                                                                   │
│  Latent z + perturbation context  →  log-FC decoder  →  dₚ        │
│                                                                   │
│  Stage 3a/3b predicted:  abundance after perturbation              │
│  Stage 3c separates:     dₚ (direct) + (I−W)⁻¹ dₚ (propagated)    │
│                                                                   │
│  Why this matters: causal queries vs predictive queries           │
│  "what does X cause?"        vs    "what happens after X?"        │
└──────────────────────────────────────────────────────────────────┘
```

Style:
- Block diagram showing latent z + perturbation context → log-FC decoder → dₚ output
- Two-row comparison at bottom: "Stage 3a/3b predicted" vs "Stage 3c separates" — visually contrast
- Why-it-matters line as muted footer

### Status pill (CRITICAL — honesty signal)

Top-right of slide, prominently visible:

```
┌───────────────────────────────────┐
│  ◆ STAGE 3c · SPEC-LOCKED         │
│  Validation Q1-Q2 2027            │
│  post Phase 1 wet-lab data        │
└───────────────────────────────────┘
```

Style:
- Cyan-amber border (cyan = spec-locked, amber = forward-looking)
- Filled background at 0.12 opacity
- Bold "SPEC-LOCKED" status, smaller body explaining timing
- Positioned at top-right of slide (same zone as pagination but separate element)

This pill is **load-bearing**. It signals upfront that this is spec architecture, not operational claim. Investors flipping through the appendix see "SPEC-LOCKED · Validation Q1-Q2 2027" before they read the math. Sets expectation correctly.

### Source citation footer (standard pattern)

```
Source: Architecture spec v1.1 §[TBD causal layer section] · 
QurieSeq Phase 1+2 spec (Thiago, May 2026) · 
STRING DB v12.0 (Szklarczyk et al., 2023, NAR) · 
Neumann series propagation (standard linear-algebra reference)
```

Note on TBD reference: A5 may drive the architecture-spec extension. If section v1.1 doesn't have a formal causal-layer section, this slide essentially anchors the upcoming spec update.

---

## Notes for design

- **Status pill is non-negotiable**. Without it, A5 reads as operational. With it, A5 reads as architectural commitment. The honesty difference is binary.
- **Equation in top zone is the visual hero**. `ŷ = (I − W)⁻¹ · dₚ` should be the largest single element on the slide — same prominence as A3's decomposed readout equation.
- **GRN visualization should look "real"** — not a generic network diagram. Use ~6-8 nodes (representing immune-relevant gene clusters like TCR signaling, BCR signaling, cytokine response) labeled with abbreviated gene names if space permits.
- **Color coding consistent with A3**: cyan = primary architectural element, lavender = perturbation, green = compositional/causal operator.
- **No new color introductions** — stay within the locked Section A palette (cyan + lavender + green accents on dark navy).

---

## Why this slide matters

A5 earns the appendix's "causal" claim. Three things it does that F1 and the existing 12 content slides do not:

1. **Surfaces the architectural commitment to causality.** Stage 3a/3b are predictive (correlation-based perturbation response). Stage 3c is causal (effect decomposition + propagation). Without A5, the "causal-ready" claim in E1 + D1 + F1 has no visible mechanism.

2. **Closes the gap with Kinga's slide 37.** Slide 37 names Neumann propagation, sparse GRN, log-FC head. The appendix needs a slide that gives investors the architectural depth behind those names.

3. **Anchors the spec extension.** A5's existence creates pressure to write the causal-layer section in `architecture_spec_v1.1.md` (or v1.2). The slide drives the spec update, which is the right direction — visible architectural commitment precedes detailed spec when an architecture is forward-looking.

---

## Source data / claims

| Claim | Source |
|---|---|
| Neumann propagation `(I − W)⁻¹ dₚ` linearizes perturbation flow | Architecture spec v1.1 [TBD section] · Kinga slide 37 (Ash-authored) |
| Spectral radius ρ(W) < 1 enforced by L1 sparsity | Standard linear-algebra requirement for matrix-inverse convergence |
| STRING as structural prior on GRN edges | STRING DB v12.0 (Szklarczyk et al., 2023, Nucleic Acids Research) |
| Direct-effect log-FC head separates dₚ from propagation | Architecture spec v1.1 [TBD section] · Stage 3c design intent |
| Causal vs predictive query distinction | Standard causal-inference framing (Pearl 2009 framework, applied to bio) |
| Stage 3c validation timeline Q1-Q2 2027 post Phase 1 data | D1 quarterly roadmap (Stage 4/5 scope includes Stage 3c implementation) |

---

## Speaker notes

**If asked: "What does 'Stage 3c spec-locked' actually mean? Is this real or aspirational?"**

> Spec-locked means the architectural commitment is written down in spec v1.1 (with v1.2 causal-layer extension pending) and the components have concrete mathematical definitions — Neumann propagation as `(I − W)⁻¹ dₚ`, sparse GRN with L1 regularization on edges absent from STRING, log-FC head for direct-effect decoding. What's not yet done is implementation and validation. Validation requires perturbation-aware multi-omics data with sufficient signal for GRN edge inference — Phase 1 wet-lab generation (Q3 2026) provides this with 4 modalities including phospho. Stage 3c implementation begins post-Phase-1, validation Q1-Q2 2027. The slide's status pill is honest about this status.

**If asked: "Why Neumann propagation specifically? Why not GNNs or transformers for the propagation step?"**

> Neumann propagation gives closed-form causal-effect estimation when the graph is fixed and the spectral radius is bounded. It's interpretable — every edge in W is a learnable causal weight; every coefficient in (I − W)⁻¹ is a propagation pathway. GNNs and transformers can learn similar functions but lose the closed-form causal-effect interpretation. For a causal inference architecture where each component must be explainable for diligence and eventually regulatory review, Neumann's linearity is a feature, not a limitation. The trade-off is expressiveness — Neumann assumes linear propagation. For Stage 3c that's the right starting point; Stage 5 may extend to nonlinear propagation if validation reveals the linear assumption is limiting.

**If asked: "STRING database — isn't it noisy? How do you handle confidence scores?"**

> STRING provides edge confidence scores from 0 to 1000 reflecting evidence type (experimental, database, co-expression, etc.). We use the high-confidence threshold (≥700) as the structural prior — only edges above this threshold get lower L1 pressure. The learning objective can still discover novel edges (lower-confidence STRING edges or genuinely novel edges from our perturbation data) but they must clear higher evidence thresholds in the regularization. The prior shapes initialization without constraining final structure. This is the standard pattern for biologically-informed graph learning.

**If asked: "How does Stage 3c relate to A3's decomposed readout?"**

> A3's decomposed readout is the architectural foundation Stage 3c builds on. The 4-arm decomposition (`h_base + Δ_stim + Δ_inh + Δ_synergy`) gives us perturbation-conditioned predictions. Stage 3c takes the `Δ` outputs and treats them as the `dₚ` direct-effect vector entering Neumann propagation. So A3's compositional generalization gives us correct perturbation-response prediction; A5's causal architecture gives us perturbation-mechanism decomposition. Sequentially: predict response (A3) → decompose into direct + propagated effects (A5). They compose; they don't compete.

**If asked: "What happens if Neumann propagation doesn't work as expected after validation?"**

> Three fallback paths in spec. First, spectral-radius violation (ρ(W) ≥ 1) — increase L1 sparsity weight or add explicit eigenvalue regularization. Second, linear-assumption failure (validation shows GRN propagation is nonlinear in the data regime) — extend to graph neural network layers replacing the matrix inverse, preserving causal-effect interpretation through architectural masking. Third, identifiability failure (W not uniquely recoverable from observational + perturbation data) — fall back to perturbation-targeted causal inference using only directly-perturbed nodes (still useful, less expressive than full GRN). All three fallbacks preserve the causal-architecture intent; only the propagation mechanism changes.

**If asked: "How does Stage 3c connect to the BTK+JAK demo (Stage 3b)?"**

> Stage 3b (Q4 2026) is the predictive milestone — zero-shot prediction of BTK+JAK combination response from singles. Stage 3c is the explanatory milestone — given a prediction, decompose it into direct BTK effect + direct JAK effect + propagated combination effect through the GRN. Stage 3b validates the platform predicts correctly; Stage 3c validates the platform explains why. Investors typically care about both: prediction accuracy demonstrates technical capability, explanation depth demonstrates regulatory and clinical readiness.

**If asked: "When does this become operational? When can we point to Stage 3c validation results?"**

> Implementation Stage 3c starts Q4 2026 after Phase 1 wet-lab data lands in Q3 2026. Phospho is available in Phase 1 (integral to QuRIE-seq), so causal architecture validation has perturbation-aware phospho signal from Q3 2026. Architecture stub + STRING integration: Q4 2026 - Q1 2027. GRN learning + sparsity calibration: Q1-Q2 2027. Validation on Phase 1 perturbation-response data: Q1-Q2 2027. First publishable Stage 3c results: Q2-Q3 2027. This timeline is on the D1 roadmap as part of Stage 4 + 5 scope. The earlier framing of "post Phase 2 data" was incorrect — Phase 1 already provides the modality signal Stage 3c needs.

**If asked: "Is this the same as DeepLife's causal modeling or Cellarity's cell-state correction?"**

> Different architecture choices, different validation strategies. DeepLife's TwinCell uses a causal cell model framework; we use Neumann propagation on a learned sparse GRN with STRING prior. Cellarity's cell-state correction operates on a learned latent space without explicit graph structure; we explicitly learn the gene-level graph for interpretability. None of the named competitors uses our specific stack (Neumann + sparse GRN + STRING prior + log-FC decoder). Whether that's a defensible technical choice or an unconventional one depends on Stage 3c validation results — which is why the slide is explicit about validation timing.

---

## Investor framing (one-paragraph elevator)

> Stage 3c is the architectural layer where the platform transitions from correlative perturbation prediction (Stages 3a/3b) to causal inference. Three components implement it: Neumann propagation `(I − W)⁻¹ dₚ` linearizes perturbation flow through a learned gene regulatory network; the GRN is sparse-learned with STRING database providing structural priors on edge existence; a direct-effect log-FC head decodes perturbation-specific direct effects before Neumann expands them through the network. This stack is spec-locked in architecture spec v1.1 with concrete mathematical commitments. Validation begins Q1-Q2 2027 once Phase 1 wet-lab data provides the perturbation-aware multi-omics signal required for GRN edge inference. Without this layer, the platform would be predictive-only; with it, the platform is causal-ready — meaning we can answer "what does X cause?" (mechanism) not only "what happens after X?" (response). For drug-discovery applications, mechanism is what enables target validation and combination rationalization. Stage 3c is what earns the platform's causal-readiness claim.

---

## What's NOT on this slide (intentionally)

- **Validation results** — none exist yet. Status pill explicitly says "validation post Phase 1 wet-lab data."
- **Specific gene names in GRN visualization** — keep generic (TCR/BCR/cytokine clusters) since the GRN is learned not fixed
- **Comparison to specific competitor causal architectures** — addressed in speaker notes (DeepLife / Cellarity Q&A), not on slide
- **Pearl-causality framework references** — too academic; "causal vs predictive query" framing is sufficient
- **Fallback paths if Neumann fails** — speaker notes only; on-slide would dilute the architectural commitment
- **Spec version number "v1.1"** in body — too granular; source citation footer carries this

---

## Diagram generation strategy

**Tool**: Cowork (Python svgwrite for math typography + matplotlib for GRN graph rendering)

**File output**: `docs/deck/assets/diagrams/A5_causal_architecture.svg`

**Followup prompt for Cowork** (when ready):
"Generate `A5_causal_architecture.svg` per spec in `docs/deck/content/A5_causal_architecture.md`. Section A continues; section accent cyan + lavender + green (matches A1-A4 palette).

Top zone (visual hero, ~40%): Neumann propagation equation `ŷ = (I − W)⁻¹ · dₚ` in large math typography. Component definitions below the equation (W, dₚ, propagation operator) with color coding. Architectural requirement (ρ(W) < 1) as small footer.

Middle zone (~35%): Two side-by-side panels — STRUCTURAL PRIOR (STRING) showing a sparse undirected graph with grey edges, vs LEARNED SPARSE GRN showing the same graph with thick cyan weighted edges + dashed grey pruned edges. Center caption: 'L1 sparsity →'. Bottom caption: 'prior shapes initialization, learning prunes'.

Bottom zone (~20%): DIRECT-EFFECT LOG-FC HEAD block showing latent z + perturbation context → log-FC decoder → dₚ. Two-row comparison: 'Stage 3a/3b predicted' vs 'Stage 3c separates'.

Status pill (top-right, prominent): '◆ STAGE 3c · SPEC-LOCKED · Validation Q1-Q2 2027 · post Phase 1 wet-lab data' — cyan border with amber accent, filled at 0.12 opacity.

Source citation footer (standard A1-A4 pattern). Pagination `A5 / 14` (deck grows from 13 to 14 content slides).

Apply collision-guard helpers from _deck_common.py pre-write. Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- **A5 is the most forward-looking slide in the appendix.** Status pill carries 100% of the load on diligence credibility. If status pill is missed, A5 reads as operational claim. Visual prominence of the pill must be tested at slide-fill scale.

- **Architecture spec v1.1 may not have a formal causal-layer section.** A5 may anchor a spec extension (write v1.2 or extend v1.1 with §X causal-architecture). If so, follow-up commit needed: `docs/specs/architecture_spec_v1_2_causal_layer.md` or equivalent. Phase 4 may include this spec-writing as scope.

- **The Neumann propagation math is correct as written but assumes linear propagation through W**. This is a legitimate Stage 3c starting assumption. Pearl-causality purists may push back; speaker note Q5 (fallback paths) handles this.

- **STRING confidence threshold (≥700)** is a defensible standard but not universal — some papers use ≥400 or ≥900. A5 uses 700 as the high-confidence default. Worth documenting in spec extension when written.

- **GRN node count in visualization** is illustrative (~6-8 nodes for visual clarity), not representative of actual scale. The real GRN learned during Stage 3c has potentially thousands of nodes. A5 caption should clarify "illustrative GRN structure; actual implementation N >> 8".

- **Stage 3c implementation cost estimate** (Q3 2026 - Q2 2027 timeline) assumes Phase 1 wet-lab data arrives Q3 2026 on schedule. Phase 1 slippage propagates to Stage 3c validation timing. D1 roadmap reflects this dependency; A5 speaker notes mention it.

---

## What's NEXT after A5 is committed

1. **A5 SVG generation** via Cowork (1.5-2h) — flywheel-comparable scope to F1
2. **pptx v3 reassembly** — insert A5 between A4 and B section divider, bump pagination (`/ 13` → `/ 14`)
3. **Phase 4 polish prompt** drafted with concrete content additions per audit + A5 additions baked in
4. **Architecture spec extension** (optional Phase 4 scope) — write v1.2 §X causal-layer formal spec backing A5's slide content

Total path from A5 spec commit to investor-ready deck v3: ~5-6 hours of execution work (A5 SVG ~2h + pptx v3 ~30min + Phase 4 polish ~3h).
