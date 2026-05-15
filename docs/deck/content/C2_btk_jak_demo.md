# Slide C2 — BTK + JAK Headline Demo: Pre-Registered Eval

- **Maps to Kinga's deck**: New slide — closes the BTK+JAK arc that runs through slides 3-4 (Aduro story), 8/9/37 (architecture), and 14 (roadmap)
- **Section**: C — QurieSeq Phase 1 (closing slide)
- **Visual lead**: Eval flow diagram + pre-registered verdict thresholds + result interpretation table
- **Status**: Draft — pending Ash review

---

## Headline

**The eval that defines the platform's first investor-grade demo.**

(Alternative: *"Pre-registered. Mechanical verdict. Q3 2026."*)

(Alternative: *"Train on singles, hold out the combo, predict zero-shot."*)

---

## Sub-headline (one line under headline)

When QurieSeq Phase 1 lands in Q3 2026, this is exactly what the BTK + JAK demo will run — protocol fixed in advance, thresholds locked, verdict mechanical.

---

## Body content (3 bullets max)

- **The eval protocol**: Train the adapter + decomposed readout on Phase 1 single-arm data (BTK alone, JAK alone, plus other inhibitor singles and the 4-arm controls). **Hold out the BTK + JAK combination arm entirely** during training. At test time, predict the held-out combo response using the synergy head — true zero-shot. Score against measured combo data.

- **Pre-registered verdict thresholds**: Synergy accuracy ≥ 0.75 = GREEN (production-ready); 0.65–0.75 = GREEN if bootstrap CI includes 0.70 / AMBER otherwise; 0.55–0.65 = AMBER (reduce λ_zero, re-run); below 0.55 = RED (architecture-class pivot). Locked in architecture spec v1.1, §5.1 before any QurieSeq data is collected.

- **What success means clinically**: A zero-shot synergy prediction at ≥0.70 means the model predicts how a BTK + JAK combination affects PBMC signaling **without seeing the combination during training**. Grounded in the Ibrutinib + Ruxolitinib CLL trial (NCT02912754, PMID 26819050) and Thiago's pJAK1/BCR pathway finding. The platform's first answer to "can it predict drug combinations we care about?"

---

## Visual spec (the eval flow diagram)

A three-part layout:

**Top — the eval flow (left-to-right):**

```
   STAGE 3b TRAINING                            STAGE 3b EVAL
   ─────────────────                            ─────────────
   
   QurieSeq Phase 1 data
   ──────────────────────
   
   ┌────────────────────┐
   │ Single-arm data     │
   │ ───────────────     │
   │ • BTK alone         │──┐
   │ • JAK alone         │  │
   │ • IKK16 alone       │  │
   │ • Idelalisib alone  │  │
   │ • Rapamycin alone   │  │      ┌──────────────────────────┐
   │ • Vehicle controls  │  │      │ Trained model:           │      ┌─────────────────────┐
   │ • All stimuli       │  ├──→   │ frozen encoder           │ ──→  │ Predict (zero-shot):│
   │   (LPS, IFN, etc.)  │  │      │ + trained adapter        │      │                     │
   │ • Other combos      │  │      │ + trained 4-arm readout  │      │ BTK + JAK combo     │
   │   (NOT BTK+JAK)     │  │      │ + Neural ODE temporal    │      │ response trajectory │
   │                     │  │      │                          │      │ 0 → 180 min         │
   └────────────────────┘  │      └──────────────────────────┘      │                     │
                            │                                          │ Score vs measured   │
   ┌────────────────────┐  │                                          │ combo data          │
   │ HELD OUT during    │  │                                          └─────────────────────┘
   │ training:           │  │                                                    │
   │                     │──┘                                                    │
   │ ◆ BTK + JAK combo  ◀──────────────────────────────────────────────────────┘
   │                     │                                            score
   └────────────────────┘
```

**Middle — pre-registered verdict thresholds (table):**

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Pre-registered Stage 3b verdict thresholds (spec v1.1, §5.1)             │
│                                                                          │
│ Zero-shot BTK+JAK synergy accuracy        Verdict      Action            │
│ ─────────────────────────────────         ────────     ────────────────  │
│                                                                          │
│  ≥ 0.75                                    GREEN        Demo ready,      │
│                                                         publish + show   │
│                                                                          │
│  0.65 — 0.75   with bootstrap CI            GREEN        Demo ready,      │
│                including 0.70                           publish with CI  │
│                                                                          │
│  0.65 — 0.75   with bootstrap CI            AMBER        Expand sample,   │
│                excluding 0.70                           re-run            │
│                                                                          │
│  0.55 — 0.65   regardless of CI             AMBER        Reduce λ_zero   │
│                                                         constraint, re-run│
│                                                                          │
│  < 0.55                                    RED          Architecture-     │
│                                                         class pivot       │
│                                                         (SDE fallback,    │
│                                                         spec §7.1)        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Bottom — clinical context bridge (single row):**

```
Why this combination matters:
• Ibrutinib (BTK inhibitor) + Ruxolitinib (JAK1/2 inhibitor) — CLL Phase Ib/II trial NCT02912754
• Maddocks 2016, Blood (PMID 26819050) — published clinical rationale  
• Thiago wet-lab finding (Quriegen IP): pJAK1 unexpectedly active in BCR pathway,
  providing biological rationale for the combination beyond independent pathway inhibition
```

---

## Notes for design

- **The eval flow diagram is the slide.** Left-to-right movement makes the training/eval separation visually obvious. The held-out box on the bottom-left, with an arrow that bypasses the trained model and feeds straight to the eval at the right, is the visual proof of "zero-shot."
- **The verdict thresholds table should look like a real protocol document** — boxed, semi-monospaced, structured. Conveys pre-registration credibility.
- **Color coding for GREEN/AMBER/RED rows** in the threshold table — quick visual scan tells investors "we have a graduated decision rule, not a single pass-fail."
- **The clinical context footer** should be small but visible — keeps the demo grounded in real medicine.
- **No actual numbers from real BTK+JAK data on this slide.** The data doesn't exist yet. This is the *plan*, not the *result*.

---

## Why this slide matters

C2 is the single slide that **converts the entire technical appendix into an investor commitment.**

Every previous slide builds capability. C2 says "and here's the specific test that will tell you whether the capability is real, on a specific date, against a specific threshold, with a specific verdict logic."

Three things it earns:

1. **Mechanical commitment**: We're not saying "we'll predict drug combinations." We're saying "when Phase 1 lands, we'll run THIS eval with THESE thresholds and the result will be GREEN, AMBER, or RED."
2. **Pre-registration as moat**: The thresholds were locked before any data was collected. Investors who've been burned by AI-pitch theatrics will recognize the credibility delta.
3. **Bridges to the roadmap (D1)**: This slide's Q3 2026 commitment is the anchor for the quarterly roadmap that follows.

---

## Source data / claims

| Claim | Source |
|---|---|
| Stage 3b BTK+JAK demo target | Architecture spec v1.1, §5.1 |
| Pre-registered verdict thresholds (GREEN/AMBER/RED) | Architecture spec v1.1, §5.1 (added in v1.1 amendment) |
| Bootstrap CI interpretation logic | Architecture spec v1.1, §5.1 |
| Eval protocol: train on singles, hold out combo, predict zero-shot | Architecture spec v1.1, §5 |
| QurieSeq Phase 1 includes BTK+JAK combo | Thiago confirmation, May 12 |
| Ibrutinib + Ruxolitinib CLL Phase Ib/II | NCT02912754 (ClinicalTrials.gov) |
| Maddocks 2016 published rationale | PMID 26819050, Blood |
| pJAK1/BCR pathway finding (Quriegen IP) | Thiago wet-lab finding |
| SDE fallback for RED verdict | Architecture spec v1.1, §7.1 |
| λ_zero reduction protocol for AMBER | Architecture spec v1.1, §7 |

---

## Speaker notes

**If asked: "Why pre-register before having the data?"**

> Two reasons. First, it eliminates post-hoc threshold drift — the moment you see a 0.62 result, the temptation to argue "well, 0.60 is actually impressive given the difficulty" becomes overwhelming. Pre-registration cuts that off. Second, the threshold table includes graduated responses (GREEN/AMBER/RED) with specific remediations per level. We're not betting the company on a single number; we have documented remediation paths for each outcome. This is what scientific rigor looks like in deep tech.

**If asked: "What if the result lands at exactly 0.65 — neither GREEN nor RED?"**

> The graduated threshold logic handles this. 0.65 with bootstrap CI including 0.70 (i.e., the upper CI bound reaches 0.70) is GREEN — we still meet the threshold with confidence. 0.65 with CI excluding 0.70 (the upper bound stays below) is AMBER — we expand the sample size and re-run, or reduce λ_zero on the zero-arm constraint and re-train. Either way the action is pre-defined; there's no judgment call at result time.

**If asked: "What does 'SDE fallback' mean if we hit RED?"**

> The Neural ODE temporal backbone (slide A4) is our primary choice. We've documented a fallback to latent SDE — same drift function `f_θ` reused, zero-initialized diffusion term, switching procedure pre-registered in spec §7.1. The fallback handles cases where deterministic dynamics prove insufficient on real data. We don't need to discover that ODE failed mid-Q3 and panic; the fallback is planned and authorized. Architecture-class pivot is the explicit decision, not "let me think about what to do."

**If asked: "What's the timing?"**

> QurieSeq Phase 1 data is targeted for delivery Q3 2026 — Thiago confirmed scheduling. Once data is in our hands, training the model on single-arm data takes ~3-4 weeks on BSC GPUs (compute already secured). The zero-shot BTK+JAK eval runs within days of training completion. The full Stage 3b execution from data delivery to verdict is targeted for Q4 2026.

**If asked: "Is 0.70 a typical threshold for this kind of prediction?"**

> For zero-shot perturbation prediction with a 4-class output structure (chance = 0.25), 0.70 is materially above chance and above the encoder probe result (0.57 from B2). 0.70 is also the threshold above which downstream applications — target prioritization, combination screening, lead selection — become useful at production scale. Below 0.70, the model is informative but not yet sufficient for production decisions. The threshold is calibrated to the use case, not arbitrary.

**If asked: "What if Thiago's wet-lab plan slips? What if Phase 1 is delayed?"**

> The architecture is built to be application-agnostic — Phase 1 delays slip the BTK+JAK demo to Q4 2026 or Q1 2027 but don't break the platform. In parallel, the Stage 3a dress rehearsal on Mimitou (slide B3) provides public-data validation of the same architectural mechanism, so we're not entirely dependent on Phase 1 timing for the technical capability story. Phase 1 delays affect demo timing, not architectural validity.

---

## Investor framing (one-paragraph elevator)

> When QurieSeq Phase 1 data lands in Q3 2026, we'll run a specific pre-registered evaluation: train the adapter, decomposed readout, and Neural ODE temporal backbone on Phase 1 single-arm data (BTK alone, JAK alone, all the 4-arm controls), with the BTK + JAK combination arm held out entirely from training. At test time, predict the held-out combo response zero-shot. Pre-registered thresholds map the result to a verdict: ≥0.75 GREEN with publication; 0.65-0.75 GREEN if bootstrap CI includes 0.70; 0.55-0.65 AMBER with documented remediation; below 0.55 RED with SDE fallback. The mechanical decision logic was locked in our architecture spec before any QurieSeq data was collected — there's no post-hoc threshold drift. Clinically, success means predicting how a BTK + JAK combination affects PBMC signaling without seeing the combination during training, grounded in the Ibrutinib + Ruxolitinib CLL trial evidence and Thiago's pJAK1/BCR wet-lab finding.

---

## What's NOT on this slide (intentionally)

- Specific Stage 3a results (will appear before deck v2 if Mimitou training completes in time)
- Detailed BSC compute usage and training cost — D2 budget slide territory
- Phase 2 phospho extensions of the BTK+JAK demo — E1 horizon slide
- The "what if we miss?" deep-dive — speaker notes handle this
- Comparison to other AI bio companies' demos — E1 territory if at all

---

## Diagram generation strategy

**Tool**: Cowork (matplotlib) — 3-zone vertical layout.

**File output**: `docs/deck/assets/diagrams/C2_btk_jak_demo_plan.svg`

**Followup prompt for Cowork** (when ready):
"Generate `C2_btk_jak_demo_plan.svg` per spec in `docs/deck/content/C2_btk_jak_demo.md`. Top: left-to-right eval flow showing (left) Phase 1 single-arm training data in one box, BTK+JAK held-out arm in a separate accented box, with arrows showing training data flowing INTO the trained model in the middle, held-out arm bypassing training and going directly to (right) the zero-shot prediction + score-vs-measured step. Middle: pre-registered verdict threshold table with 5 rows (≥0.75 GREEN, 0.65-0.75 GREEN-if-CI, 0.65-0.75 AMBER-otherwise, 0.55-0.65 AMBER, <0.55 RED), color-coded. Bottom: clinical context single-line footer citing NCT02912754, PMID 26819050, and Quriegen IP pJAK1/BCR finding. Output 1920×1080 viewBox."

---

## Risk callouts (NOT to include on slide; for tracking only)

- The 0.70 target threshold is aggressive for zero-shot synergy on a Phase 1 data scale of 5 donors. Bootstrap CI is the documented mitigation but still meaningful uncertainty.
- The architecture-class pivot for RED (SDE fallback) is documented but untested in our environment. Latent SDE training on QurieSeq's specific data structure has not been validated.
- Phase 1 wet-lab timing slippage cascades to Q3 demo timing.
- The slide's pre-registration claim depends on the architecture spec actually being timestamped in git BEFORE QurieSeq Phase 1 data is collected. Spec v1.1 committed May 6, 2026; Phase 1 wet lab begins later. Audit trail is clean — but verifiable, so don't overstate.
- We need to verify Maddocks 2016 PMID and the NCT trial number before final deck commit. Cowork or one of us should hand-check these.

---

## What's NEXT after C2 is committed

Section C complete. Move to **Section D — Roadmap + Budget**:
- **D1**: Quarterly Roadmap Q3 2026 → Q4 2028 — extends Kinga's slide 14
- **D2**: Seed Allocation — Where The $10M Goes — extends Kinga's slide 17

Then **E1** (5-Year Horizon — Pipeline + Clinical Maturation) closes the appendix.
