# Cover Note — v5 Deck To Kinga + Jan (REVISED)

**Drafted**: 2026-05-17
**Purpose**: Cover message accompanying `aivc_appendix_v5.pptx` + `aivc_appendix_v5_speaker_notes.md` shared with Kinga Matuła + Jan for content/technical review.
**Revision**: Adds explicit how-to-access notes + feedback mechanism sections.

---

## Suggested Message (Edit Before Sending)

> Hi Kinga, Jan —
>
> Attached: AIVC GeneLink technical appendix **v5** with the comprehensive speaker notes you asked for.
>
> **Two files attached:**
> 1. `aivc_appendix_v5.pptx` — 21-slide investor appendix, speaker notes embedded per slide
> 2. `aivc_appendix_v5_speaker_notes.md` — same speaker notes as a standalone document (mobile-readable, no PowerPoint required)
>
> ---
>
> ## How To Access The Speaker Notes
>
> The notes are extensive (~124k characters / 84 diligence Q&As). Two ways to read them:
>
> **In PowerPoint** (recommended for deck review):
> - Open `aivc_appendix_v5.pptx`
> - Either: View → Notes (Mac) / View → Notes Page (Windows) for full slide+notes view per page
> - Or: View → Presenter View to see slide + notes side-by-side
>
> **As standalone document** (recommended for mobile/cross-platform):
> - Open `aivc_appendix_v5_speaker_notes.md` in any markdown viewer, text editor, or browser
> - Slide-by-slide, no PowerPoint needed
>
> ---
>
> ## What's New In v5 (vs the v3 you saw last)
>
> **1. Phospho-in-Phase-1 correction propagated everywhere.** Based on Kinga's clarification that phosphoproteins are integral to QuRIE-seq Phase 1, I reconciled the full deck. A2 / C1 / D1 / F1 visuals updated. Phase 1 is now correctly framed as 4 modalities (RNA + Protein + Phospho at all 5 timepoints; ATAC at t=0 and t=180 only). Phase 2 adds VDJ as 5th modality. The earlier framing of "phospho deferred to Phase 2" was rooted in our public-data layer discussion (no public phospho data on PBMCs) — the deck had inherited that incorrectly as a wet-lab timing claim.
>
> **2. Comprehensive speaker notes across all 14 content slides:**
> - **Three-state framing per slide** — what's validated today on public data / what Phase 1 adds (Q3 2026) / what Phase 2 adds (2027). Public-data substrate stays visible; QuRIE-seq is the upgrade.
> - **Technical glossary per slide** — every term, abbreviation, equation, library, and biology concept used on the slide, defined in plain English with reading-order math.
> - **Diligence Q&As** — 84 total across the deck, covering predictable investor questions.
>
> **3. Master glossary** — canonical definitions for ~100 terms, in the repo at `docs/deck/research/glossary_2026_05_17.md`.
>
> ---
>
> ## What's NOT In v5 (Intentional, Tracked For Phase 4)
>
> Visual polish is deferred until I have your content feedback. Tracked items:
> - A1 capsule tabs text overflow at bottom
> - A2 left-right zone balance (73% appears too large vs right-side panel)
> - D1 Stage 3a/3b/3c row compactness with text bleed
> - F1 Quriegen bordered row vertical padding
> - A5 equation color coding (currently white-only)
> - Pagination unification across all slides
> - Font sizing audit ("some text too small at presentation scale")
>
> These don't change content — only polish how it looks. Better to land your content feedback first, then polish in a final pass.
>
> ---
>
> ## Three Items I Need Your Confirmation On
>
> **1. Kinga — "Sci" reference.** In your speaker notes ask you mentioned "Sci" as a term to define. I did a systematic scan of all 14 slides and content specs and could not find a "Sci"-prefix library, methodology, or term anywhere. Marked as `[PENDING IDENTIFICATION]` in the master glossary. Possibilities: SciPlex, sci-RNA-seq, Sci-Hub reference, or potentially a misread. Could you point me to which slide or section?
>
> **2. Kinga — D2 budget timing.** The ~$1M phospho antibody panel is now framed as a Phase 1 (Q3 2026) wet-lab cost rather than a Phase 2 (Q1 2027) prep cost — because phospho is integral to QuRIE-seq from Phase 1. Total dollar amount unchanged (~$1M); only timing shifts earlier by one quarter. Want to confirm Phase 1 budget can absorb this in Q3 2026, or if the timing reshuffles other line items.
>
> **3. Thiago — Phase 1 perturbation panel size.** Throughout the deck I use "~15-20 conditions" as an estimate for the Phase 1 perturbation panel (vehicle + stimulus + inhibitor singles + combinations including BTK+JAK confirmed). Architecture is sized to support this range, but if you've finalized the exact count, I'll align the deck.
>
> ---
>
> ## Feedback Format — Whatever Works For You
>
> Send comments in whatever format is easiest:
> - Annotated PDF
> - Slack thread or email summary
> - Redlined .pptx (direct edits in PowerPoint are fine for your review process)
> - GitHub comments or PRs if you prefer
>
> **Note on the build pipeline**: the canonical source for the deck is our content specs in the repo (`docs/deck/content/*.md`). I'll apply your edits via our build pipeline and ship v6, ensuring your changes persist through future rebuilds. Direct edits to the .pptx are useful for your review process, but I'll re-apply them to the specs.
>
> ---
>
> ## Where Everything Lives
>
> All on GitHub at `quriegenAK/quriegen-aivc` (main branch):
> - **The deck**: `docs/deck/exports/aivc_appendix_v5.pptx`
> - **Standalone speaker notes**: `docs/deck/exports/aivc_appendix_v5_speaker_notes.md`
> - **Master glossary**: `docs/deck/research/glossary_2026_05_17.md`
> - **Phase 1 modality correction canonical doc**: `docs/deck/research/phase1_modality_correction_2026_05_17.md`
> - **Content specs (source of truth)**: `docs/deck/content/*.md`
>
> ---
>
> Take whatever time you need. The v5 framing is substantially tighter than v3 because of your phospho clarification — thanks for the careful read.
>
> Ash

---

## What This Revised Cover Note Adds vs Original

| Addition | Why |
|---|---|
| Explicit "How to access speaker notes" section with PowerPoint Mac + Windows menus | Without this, they might not realize notes are embedded |
| Mention of standalone markdown companion doc | Gives mobile-readable option |
| "What's NOT in v5" section listing tracked polish items | Saves them from spending feedback on items already tracked |
| Explicit feedback format note | Sets correct expectations about how their feedback flows back |
| Build pipeline clarification | Prevents confusion when they ask "did my .pptx edits land?" — answer is "your feedback informed v6 via the specs" |
| Full file paths for everything | They can navigate the repo directly if they want |

---

## Optional Shorter Variant (Slack)

> Hi Kinga, Jan —
>
> v5 deck attached + companion speaker notes markdown. **Notes are embedded in the pptx** — open via View → Notes (Mac) or View → Notes Page (Windows). Standalone .md file is mobile-readable if easier.
>
> What's new: phospho-in-Phase-1 correction propagated across A2/C1/D1/F1 visuals, plus comprehensive speaker notes across all 14 slides (three-state framing + technical glossary + 84 diligence Q&As, ~124k chars total).
>
> Three confirmation items in the deck cover note: Sci reference, D2 budget timing for $1M phospho panel now in Phase 1, Thiago on exact perturbation panel size.
>
> Visual polish deferred to Phase 4 after your feedback. Format your feedback however works — annotated PDF, redlined .pptx, Slack thread, all fine. I'll apply via build pipeline to v6.

---

## Files To Attach When You Send

1. **`aivc_appendix_v5.pptx`** — from `docs/deck/exports/` on GitHub
2. **`aivc_appendix_v5_speaker_notes.md`** — coming from the next Cowork iteration

Don't send until the markdown doc lands.
