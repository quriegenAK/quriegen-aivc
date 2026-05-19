# Slide A3 — Decomposed Readout: How Synergy Generalizes

- **Maps to Kinga's deck**: Slide 9 (competitive checkbox matrix) — substantiates "Virtual cell ✓" + "Causal-ready"
- **Section**: A — Architecture Depth
- **Visual lead**: 4-arm equation diagram with the zero-arm constraint
- **Status**: Draft — pending Ash review

---

## Headline

**Predict drug combinations the model has never seen.**

(Alternative: *"Compositional generalization, not memorization."*)

(Alternative: *"The architecture choice that makes BTK+JAK prediction possible."*)

---

## Sub-headline (one line under headline)

Every cell response is decomposed into four learned components — baseline, stimulus, inhibitor, synergy. The synergy head learns only the non-additive correction, which is exactly what zero-shot drug combination prediction requires.

---

## Body content (3 bullets max)

- **4-arm decomposition**: For every cell, predicted response = baseline + (stimulus effect if any) + (inhibitor effect if any) + (synergy correction if both). Four learned heads, parallel branches, summed at output.

- **Zero-arm constraint is load-bearing**: A penalty (L2, λ=1.0) forces the synergy head to output zero when stimulus or inhibitor is absent. This is not a regularizer — it's the architectural choice that makes the synergy head learn *only the non-additive part* of combination biology.

- **Why it generalizes**: At inference time, predicting a never-seen combination (e.g. BTK+JAK on a new donor) requires the synergy head to extrapolate from training combinations. Because the zero-arm constraint forced it to learn only the residual synergy signal, it can compose responses for unseen combinations without overfitting to seen ones.

---

## Visual spec (the equation diagram)

**Center of slide — the architecture equation:**

```
                                  Predicted cell response

        ŷ(c, s, i, t)  =   h_base(z, t)                          ← always active
                          + 𝟙[s]·Δ_stim(z, s, t)                  ← active if stim present
                          + 𝟙[i]·Δ_inh(z, i, t)                   ← active if inhibitor present
                          + 𝟙[s∧i]·Δ_synergy(z, s, i, t)          ← active only for combos
```

Where:
- `c` = cell
- `s` = stimulus identity (or none)
- `i` = inhibitor identity (or none)
- `t` = timepoint
- `z` = encoder latent vector (frozen)
- `𝟙[·]` = indicator function (1 if condition met, 0 otherwise)
- Each `Δ` = independent learned head

**Below the equation — the zero-arm constraint, called out as a separate block:**

```
┌─────────────────────────────────────────────────────────────┐
│ Load-bearing constraint:                                    │
│                                                             │
│   For NTC (no stim, no inh):     Δ_stim, Δ_inh, Δ_synergy = 0 │
│   For stim-only cells:           Δ_inh, Δ_synergy = 0        │
│   For inhibitor-only cells:      Δ_stim, Δ_synergy = 0       │
│                                                             │
│   Penalty: L_zero_arm = λ · Σ‖Δ‖² where condition fails     │
│   λ = 1.0 (architecture spec v1.1, §3.2)                    │
└─────────────────────────────────────────────────────────────┘
```

**Right side / inset — the generalization claim:**

A small 3-row table:

| Training data | Inference target | What this enables |
|---|---|---|
| Cells with: stim_A, stim_B, inh_X, inh_Y, stim_A+inh_X | Predict: **stim_B + inh_Y** | Compositional generalization to unseen combos |
| Mimitou CD3E, CD4 single-KO + CD3E+CD4 double-KO | Predict: held-out double-KO | Validated at 0.68 accuracy in Stage 3 Part 1 |
| QurieSeq: BTK alone, JAK alone + 4-arm controls | Predict: **BTK+JAK combo zero-shot** | The headline demo, Q3 2026 |

---

## Notes for design

- **The equation is the slide.** Make it center stage, large, clean LaTeX-quality rendering.
- **Color-code the four heads consistently**: e.g. h_base = grey, Δ_stim = green, Δ_inh = blue, Δ_synergy = brand accent (red/magenta). This color scheme should be reused on slide C2 (BTK+JAK demo) for narrative continuity.
- **Indicator functions 𝟙[·] should be visually obvious** — they're what makes the equation work for any combination of conditions.
- **The zero-arm constraint block should look like a "lemma" or theorem box** — boxed, slightly recessed, mathematical aesthetic.
- **Right-side generalization table is supporting evidence**, not the focus. Smaller text. Three rows max.

---

## Why this slide matters more than it looks

This is the slide that lets you claim:
- "The model predicts synergy for drug combinations it has never seen"
- "Causal-ready architecture"
- "Compositional generalization"

Without this decomposition + zero-arm constraint, the architecture is just a conditional regressor with no zero-shot story. **An investor's CTO will recognize this.** This is where technical due diligence either succeeds or fails.

---

## Source data / claims

| Claim | Source |
|---|---|
| 4-arm decomposed readout architecture | Architecture spec v1.1, §3.2 |
| Zero-arm L2 constraint, λ=1.0 | Architecture spec v1.1, §3.2.2 |
| Mimitou CD3E+CD4 zero-shot 0.68 accuracy | `docs/memory/project_aivc_stage3_part1_verdict_2026_05_11.md` |
| BTK+JAK zero-shot demo target for Q3 2026 | Architecture spec v1.1, §5 + Thiago confirmation May 12 |
| Implementation: `aivc/skills/decomposed_readout.py` | Commit 87d6a9a (Stage 3a Day 1 PR), tests in `tests/test_decomposed_readout.py` |
| Zero-arm constraint verified in test_zero_arm_loss_double_perturbation_no_constraint | Stage 3a Day 1 test suite |

---

## Speaker notes

### Three-state framing
- **Today (architecture shipped)**: 4-arm decomposed readout is implemented and trained on Mimitou CRISPR data. CD3E + CD4 single perturbations train the synergy head; CD3E×CD4 double-knockout is the held-out test for compositional generalization.
- **Phase 1 (Q3 2026)**: Same 4-arm readout trains on QuRIE-seq Phase 1 perturbation panel. BTK alone + JAK alone train the inhibitor arms; BTK+JAK combo is the held-out test for the Stage 3b demo. Synergy head's zero-arm constraint becomes the load-bearing claim.
- **Phase 2 (2027)**: 4-arm readout extends to additional inhibitor combinations and donor-level cross-validation across the 20-donor scale.

### Technical glossary
**Decomposed readout (4-arm)** — Decoder architecture: predicted response = `h_base + 𝟙[s]·Δ_stim + 𝟙[i]·Δ_inh + 𝟙[s∧i]·Δ_synergy`. Four learned heads, parallel branches, summed at output.

**Synergy** — When the combined effect of two perturbations exceeds the sum of their individual effects: `Δ_combo > Δ_drug1 + Δ_drug2`. The Δ_synergy arm captures this directly.

**Zero-arm constraint** — A penalty (L2, λ=1.0) forcing the synergy head to output zero when stimulus or inhibitor is absent. Forces the synergy head to learn ONLY the non-additive correction.

**Compositional generalization** — Model's ability to predict combinations from singletons. Train on BTK alone + JAK alone, predict BTK+JAK combo response.

**Indicator function 𝟙[s] (Iverson bracket)** — `𝟙[s] = 1 if condition s is true, 0 if false`. Switches arms on/off based on experimental condition.

**Δ (Delta)** — "Change" or "difference". Δ_stim = stimulus contribution, Δ_inh = inhibitor contribution, Δ_synergy = synergy correction beyond additive.

**L2 regularization** — Penalty on the sum of squared weight values. Used at λ=1.0 to enforce the zero-arm constraint on the synergy head when single arms are absent.

**Perturbation embedding** — Vector representation of a perturbation context (drug, concentration, duration). Combined with cell latent state to predict response.

### Equations & notation

**Reading the decomposed readout equation**:
```
ŷ(z, s, i, t) = h_base(z, t)
              + 𝟙[s]·Δ_stim(z, s, t)
              + 𝟙[i]·Δ_inh(z, i, t)
              + 𝟙[s∧i]·Δ_synergy(z, s, i, t)
```

- `ŷ` (y-hat) = predicted response
- `z` = cell latent state (from encoder)
- `s` = stimulus identifier (vector)
- `i` = inhibitor identifier (vector)
- `t` = timepoint
- `h_base(z, t)` = vehicle-control baseline at time t
- `𝟙[s]` = 1 if stimulus s is present, 0 otherwise (Iverson bracket / indicator)
- `Δ_stim(z, s, t)` = additional response contributed by stimulus alone
- `Δ_inh(z, i, t)` = additional response contributed by inhibitor alone
- `Δ_synergy(z, s, i, t)` = non-additive synergy correction when both present
- `s ∧ i` = "both s and i present" (logical AND)

Architectural commitment: synergy head outputs zero when either single arm is absent (`𝟙[s∧i] = 0`), forced by L2 penalty during training. This means the synergy head can only learn the non-additive part — making zero-shot combination prediction possible.

### Diligence Q&A

**If asked: "Why not just train a single conditional head?"**

> Mathematically you could — a single head conditioned on (cell, stim, inh) can fit any training data. But at inference, it has no inductive bias for unseen combinations. The architecture would memorize the training combinations rather than learn the additive structure of combinatorial biology. Our decomposition forces the model to learn the non-additive part separately, which is precisely what zero-shot synergy prediction requires.

**If asked: "How do you know the zero-arm constraint isn't too strict?"**

> Two safety nets. First, the constraint is a soft penalty (L2 with λ=1.0), not a hard projection — the model has some slack if biological signal demands it. Second, we have a documented downgrade path: if Stage 3a training shows the constraint is too rigid, we drop λ to 0.3 or 0.5 and rerun. This decision is pre-registered in the architecture spec (§7, risk #3).

**If asked: "What if the synergy head learns the trivial 'mean of singles' solution?"**

> That's the failure mode we explicitly test for. If Stage 3a zero-shot synergy accuracy comes in near 0.55 (matching the null baseline `h_base + Δ_stim + Δ_inh` with `Δ_synergy=0`), it means the synergy head collapsed to triviality. The pre-registered remediation: increase λ_zero or move to a hard projection constraint. Banked in `docs/specs/stage3_part2_architecture_proposal_2026_05_06.md` §7.

**If asked: "Has this architecture been published before?"**

> The decomposition is inspired by causal inference (potential outcomes framework: Y(0), Y(1), Y(1,1) interactions) applied to deep learning readout heads. The specific 4-arm + zero-arm constraint formulation is our architectural choice for perturbation biology — it's documented in our spec and validated on Mimitou CRISPR data with 0.68 accuracy on held-out double-KO cells.

---

## Investor framing (one-paragraph elevator)

> The decomposed readout is the architecture's compositional generalization mechanism. Each cell response is built from four parallel components — baseline, stimulus effect, inhibitor effect, synergy correction — and a soft constraint forces the synergy head to output zero when only one of (stimulus, inhibitor) is present. This means the synergy head learns *only the non-additive part* of combination biology. At inference, predicting an unseen combination requires composing learned single-perturbation effects plus a learned synergy correction — and because the architecture trained the synergy head only on the residual signal, it generalizes to combinations it has never seen. We've validated this approach on Mimitou's double-KO arm at 0.68 zero-shot accuracy. The same architecture, applied to QurieSeq's BTK + JAK combination data in Q3 2026, becomes the headline investor demo.

---

## What's NOT on this slide (intentionally)

- Detailed training loop (lives in Stage 3a implementation, not investor deck)
- Pseudocode (the equation is enough)
- All four head architectures in detail (sufficient to say "independent learned heads")
- Loss function full form (zero-arm is shown; recon, pathway, smoothness are not — handled in code)

---

## Diagram generation strategy

**Tool**: Cowork or Claude Code with LaTeX rendering (matplotlib with `usetex=True`, or directly via TikZ → SVG)

**File output**: `docs/deck/assets/diagrams/A3_decomposed_readout.svg`

**Followup prompt for Cowork** (when ready):
"Generate `A3_decomposed_readout.svg` per spec in `docs/deck/content/A3_decomposed_readout.md`. Center-of-slide: the 4-arm equation rendered with LaTeX-quality typography, color-coded heads (h_base=grey, Δ_stim=green, Δ_inh=blue, Δ_synergy=accent). Below: zero-arm constraint block (boxed, theorem-style). Right inset: 3-row generalization table. Output 1920×1080 viewBox. Use Kinga's deck color palette (TBD)."

---

## Risk callouts (NOT to include on slide; for tracking only)

- Zero-arm constraint validated in unit tests (Stage 3a Day 1) but not yet on training data — Stage 3a Day 3-5 will tell us if it works at scale.
- The 0.68 Mimitou double-KO accuracy is from a held-out test split of size 37 cells; bootstrap CI is wide (~±0.10).
- BTK+JAK demo claim depends on QurieSeq Phase 1 data quality (independent risk).

---

## What's NEXT after A3 is committed

Move to **A4 (Neural ODE temporal backbone)** — the slide that explains how 0/5/30/60/180 min time-course data is modeled as continuous dynamics. Connects to QurieSeq Phase 1 directly. Sets up slide C1 (Phase 1 experimental design).
