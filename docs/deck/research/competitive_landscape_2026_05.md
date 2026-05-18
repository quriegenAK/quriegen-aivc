# Competitive Landscape — AI Virtual Cell / Multi-Omics Foundation Models

**Compiled**: 2026-05-15
**Scope**: 7 competitors named on Kinga's slide 9 + 3 adjacent competitors discovered during research
**Method**: Web search, primary-source verification (company press, peer-reviewed publications, funding announcements). Every factual claim traces to a numbered source at the bottom of each entry.
**Refresh cadence**: Competitive landscape moves fast — this snapshot has a shelf life of ~3 months. Refresh before any external use beyond Q3 2026.

---

## Reading guide

Each entry follows the same fields. Where information is unavailable from public sources, the entry says `unclear from public information` rather than guessing. Differentiation claims are written as "what *they* would say against QurieGen" — these are competitor-perspective claims, not endorsements.

The honest-gap section is the load-bearing part of each entry. If a competitor has a capability QurieGen lacks, it's flagged.

---

# 1. TAHOE Therapeutics (formerly Vevo Therapeutics) — *named on Kinga's slide 9*

**URL**: https://www.tahoebio.ai/
**Founded**: 2022 [1]
**Funding stage**: Series A
**Recent raise**: $30M Series A in August 2025 led by Amplify Partners + Databricks Ventures, General Catalyst, Mubadala Capital [2]. Total funding ≈ $42M [2]. Reported valuation: $120M [2].
**Headquarters**: South San Francisco, CA [1]
**Primary public claim**: Generating the world's largest single-cell perturbation atlas to power virtual cell models for AI-driven drug discovery [3].

**Modality coverage**:
- RNA: yes (transcriptomic profiles only) [3]
- ATAC: not currently — no public claim of chromatin data
- Protein: not currently — no public claim of surface or intracellular protein
- Phospho: not currently
- VDJ: not currently
- Other: small-molecule perturbations (1,100+ compounds × 50 cancer cell lines in Tahoe-100M) [3]

**Data strategy**:
- Public data only? No — Tahoe runs its own wet-lab data generation
- Proprietary wet lab? Yes — "Mosaic Technology" for pan-cancer pertubation testing [3][4]. Open-sourced Tahoe-100M (100M transcriptomic profiles) February 2025 [4]. Announced 300M-cell generation effort with Parse Biosciences GigaLab + 120M additional cells with Arc Institute and Biohub in 2026 [5][6].
- Partnerships for data? Yes — Arc Institute (Virtual Cell Atlas), Chan Zuckerberg Biohub, Parse Biosciences [5][6].

**Model architecture**:
- Foundation model? Yes — Tahoe-x1 (3 billion parameters) [7]
- Architecture type: masked-expression generative objective with drug-token incorporation; joint representations of genes, cells, and compounds [7]. Decoder-style transformer per published preprint [7].
- Modalities trained on: RNA-seq + drug-token only [7]
- Public model weights? Yes — Tahoe-100M dataset open-sourced via HuggingFace (`tahoebio/Tahoe-100M`) [8]; Tahoe-x1 model details published as bioRxiv preprint [7].

**Validation evidence**:
- Peer-reviewed publications? Two bioRxiv preprints (not yet peer-reviewed at time of writing): Tahoe-100M dataset paper [3] and Tahoe-x1 model paper [7].
- Cross-corpus generalization? Unclear — published benchmarks focus on internal Tahoe-100M splits.
- Perturbation prediction validation? Yes — Tahoe-x1 reports state-of-the-art on internal perturbation tasks per [7].
- Specific benchmarks: internal Tahoe-100M splits; comparisons in [7] to scGPT and Geneformer.

**Differentiation TAHOE would claim against QurieGen**:
- "We have 100M cells; you have 500K (200× scale advantage)"
- "Our foundation model is 3B parameters and open-sourced — yours isn't either of those things"
- "We're partnered with Arc Institute, Biohub, and Parse Biosciences — top-tier infrastructure"
- "Cancer is a clearer commercial market than PBMC + immune perturbation"
- "Our dataset is public, fueling external benchmark adoption"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: 100M-cell pretraining corpus (vs our 500K Phase 1 plan — 200× larger); 3B-param open-source foundation model with state-of-the-art perturbation benchmarks; cancer cell-line breadth (50 lines); Arc Institute partnership; brand recognition in the virtual-cell community.
- **What we have that they don't**: Multi-omics in the same single cell (they're RNA-only) [3]; ATAC chromatin readouts; protein readouts; planned phospho integration; PBMC primary cells (vs cancer cell lines — disease relevance argument); decomposed-readout architecture for compositional synergy prediction (not a Tahoe-x1 claim).
- **Roughly equivalent**: foundation-model claim, wet-lab data-generation claim, lab-in-the-loop iteration claim (both claim it).

**Sources**:
1. Tahoe Therapeutics Crunchbase, https://www.crunchbase.com/organization/vevo-therapeutics
2. "Tahoe Therapeutics Raises $30 Million to Expand Single-Cell Data Production for AI Models", BiopharmaTrend, https://www.biopharmatrend.com/news/tahoe-therapeutics-raises-30-million-to-expand-single-cell-data-production-for-ai-models-1337/
3. Zhang et al., "Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas for Context-Dependent Gene Function and Cellular Modeling", bioRxiv 2025.02.20.639398, https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1
4. "Tahoe 100M: The World's Largest Single-Cell Dataset, Open-Sourced", Tahoe, https://www.tahoebio.ai/news/tahoe-100m-the-worlds-largest-open-source-single-cell-dataset-as-the-inaugural-contribution-to-arc-institutes-new-virtual-cell-atlas
5. "Tahoe Therapeutics Selects Parse Biosciences' GigaLab to Generate 300 Million Single Cell Profiles", Parse Biosciences press, https://www.parsebiosciences.com/news/tahoe-therapeutics-selects-parse-biosciences-gigalab-to-generate-300-million-single-cell-profiles-for-large-scale-perturbation-atlas/
6. "Tahoe Therapeutics, Arc Institute, and Biohub Partner to Generate the Largest Perturbation Dataset for Virtual Cell Models", Arc Institute, https://arcinstitute.org/news/tahoe-arc-biohub
7. "Tahoe-x1: Scaling Perturbation-Trained Single-Cell Foundation Models to 3 Billion Parameters", bioRxiv 2025.10.23.683759, https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1
8. tahoebio/Tahoe-100M HuggingFace dataset, https://huggingface.co/datasets/tahoebio/Tahoe-100M

**Last researched**: 2026-05-15

---

# 2. DeepLife — *named on Kinga's slide 9*

**URL**: https://www.deeplife.co/
**Founded**: 2019 [1]
**Funding stage**: Series A
**Recent raise**: $10M Series A in 2025 led by Turenne Groupe + YZR Capital, with Beiersdorf Venture Capital and Relyens Innovation Santé [2][3]. Earlier funding not publicly broken out.
**Headquarters**: Paris, France [1]
**Primary public claim**: Build "digital twins" of human cells using interpretable AI to identify drug candidates that restore cells from disease to healthy state [4].

**Modality coverage**:
- RNA: yes (single-cell transcriptomics) [4]
- ATAC: not currently surfaced in public materials
- Protein: not currently surfaced
- Phospho: not currently
- VDJ: not currently
- Other: multi-omics integration claim is generic ("state-of-the-art multi-omics data") — specific modalities not detailed publicly [4]

**Data strategy**:
- Public data only? Yes, primarily — DeepLife builds atlases from public + literature data [4]. No proprietary wet lab claim.
- Proprietary wet lab? No — kept consistent with Kinga's slide 9 mark (— for PROPRIETARY DATA GEN).
- Partnerships for data? Yes — recently announced partnership with cultivated-meat company Gourmey to digitally engineer production cells [5] (signals broad-domain platform positioning vs strict drug discovery).

**Model architecture**:
- Foundation model? Yes — "TwinCell" branded as a Large Causal Cell Model (LCCM) [4].
- Architecture type: graph + mechanistic hybrid; claims interpretable AI (not black-box transformer) [4]. Specific architecture details not in public materials.
- Modalities trained on: transcriptomics primarily; other modalities claimed but unspecified.
- Public model weights? No — model is product-facing; no public weights or model card.

**Validation evidence**:
- Peer-reviewed publications? Limited public surface — featured in *Nature* "How digital twins of human cells are accelerating drug discovery" (2022 feature, not a peer-reviewed primary publication) [6].
- Cross-corpus generalization? Unclear from public information.
- Perturbation prediction validation? Unclear — pitch deck language describes "billions of drug combinations" predicted but no public benchmark numbers found.
- Specific benchmarks: none surfaced in search.

**Differentiation DeepLife would claim against QurieGen**:
- "Interpretable AI vs your black-box deep network — pharma diligence will prefer mechanism-aware models"
- "Causal model claim grounded in genetic regulatory networks — yours is a residual decomposition trick"
- "We're already mulling US relocation and expanding beyond drug discovery (Gourmey) — broader platform"
- "Capital-efficient: $10M raised vs your $10M ask"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: 4 years longer in market; published autoimmune + neurodegenerative disease use cases; explicit "interpretability" framing that pharma diligence likes; cross-industry validation (food/biotech adjacency).
- **What we have that they don't**: Trimodal same-cell data (RNA + ATAC + Protein from DOGMA-seq); proprietary wet-lab data generation; phospho integration path; published cross-corpus generalization number (73% Calderon); peer-reviewed-style closure reports.
- **Roughly equivalent**: "virtual cell model" / "digital twin" positioning (both claim it); causal-modeling claim (both claim it; definitions differ).

**Sources**:
1. DeepLife Crunchbase, https://www.crunchbase.com/organization/deeplife
2. "French biotech startup DeepLife raises $10M Series A", Vestbee, https://www.vestbee.com/insights/articles/deep-life-raises-10-m
3. "DeepLife secures $10M for AI-driven healthcare innovation using digital twins", liveforever.club, https://liveforever.club/article/deeplife-secures-10m-for-ai-driven-healthcare-innovation-using-digital-twins
4. DeepLife Tech page, https://www.deeplife.co/tech
5. "DeepLife and Gourmey Partner to Digitally Engineer the Future of Cultivated Meat", https://www.deeplife.co/media-post/deeplife-and-gourmey-partner-to-digitally-engineer-the-future-of-cultivated-meat
6. "How digital twins of human cells are accelerating drug discovery", *Nature* news feature, 2022, https://www.nature.com/articles/d43747-022-00108-3

**Last researched**: 2026-05-15

---

# 3. Turbine AI — *named on Kinga's slide 9*

**URL**: https://turbine.ai/
**Founded**: ~2018 (Budapest University spinout; first funding round Nov 2019) [1]
**Funding stage**: Series B
**Recent raise**: $25M Series B in February 2026 [2]. Earlier €20M Series A [3] + €25.5M oversubscribed extension [4]. Total ≈ $36.7M [1]. Investors include Accel, Accenture, Boston Millennia, MSD Global Health Innovation, Mercia [1][2].
**Headquarters**: Budapest, Hungary [1]
**Primary public claim**: "Simulated Cell" virtual lab for in-silico cancer drug discovery, partnered with AstraZeneca / Bayer / Ono Pharma / Cancer Research Horizons [5].

**Modality coverage**:
- RNA: yes (multi-omics data harmonized onto cell model) [5]
- ATAC: claimed in marketing "multi-omics" but specifics unclear
- Protein: surface + signaling, claimed
- Phospho: signaling claim implies some level of phospho modeling; specifics not public
- VDJ: not surfaced
- Other: pathway / signaling network simulation as core differentiator [5]

**Data strategy**:
- Public data only? No — Turbine harmonizes external + partner data onto its own model framework
- Proprietary wet lab? **Per Kinga's slide 9: NO** (— mark in PROPRIETARY DATA GEN column). Verifies: Turbine's positioning is in-silico simulation, not wet lab. Pharma partners provide experimental data; Turbine simulates [5].
- Partnerships for data? Yes — AstraZeneca (ADC discovery [6]), Bayer, Ono, Cancer Research Horizons.

**Model architecture**:
- Foundation model? Yes — "Simulated Cell™" platform, a foundational cell model trained on decade of research [5].
- Architecture type: hybrid — combines mechanistic / signaling network with machine learning. Described as "models molecular interactions within and around cells" [5]. Not a pure transformer.
- Modalities trained on: signaling-pathway-centric (kinase activity, receptor signaling); RNA / phospho / protein integration claimed.
- Public model weights? No — closed platform.

**Validation evidence**:
- Peer-reviewed publications? Limited — partnerships generate joint publications but primary platform paper not surfaced in search.
- Cross-corpus generalization? Internal validation through pharma partner programs; not externally benchmarked.
- Perturbation prediction validation? Yes (claimed) — partnership with AstraZeneca on antibody-drug conjugate discovery [6] uses Turbine for in-silico perturbation modeling.
- Specific benchmarks: none publicly disclosed.

**Differentiation Turbine would claim against QurieGen**:
- "Mechanistic + ML hybrid vs pure deep-learning — interpretable for pharma decisions"
- "Eight years of platform development; you're at year 3"
- "Five major pharma partnerships already (AstraZeneca, Bayer, Ono, Cancer Research Horizons, plus Accenture investment)"
- "Cancer market is concrete; PBMC is platform-y"
- "Lab-in-the-loop with our pharma partners; you're early-stage"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: 8-year head start; named pharma partnerships at scale (AstraZeneca, Bayer, Ono); Accenture investment signals enterprise relationships; mechanistic interpretability story aligned with diligence preferences.
- **What we have that they don't**: Same-cell multi-omics (RNA + ATAC + Protein) per Kinga's slide 9 marks (Turbine — / —); proprietary wet-lab data generation; phospho explicit Phase 2; PBMC immune cell focus (vs Turbine's cancer-cell focus); peer-reviewed-style cross-corpus generalization (73% Calderon).
- **Roughly equivalent**: virtual cell model claim; lab-in-the-loop claim (both claim it; Turbine's loop is partner-driven, ours is in-house).

**Sources**:
1. Turbine Crunchbase, https://www.crunchbase.com/organization/turbine-2
2. "Hungary's Turbine raises $25M Series B to build virtual experiments", Endpoints News, https://endpoints.news/ai-biotech-turbine-raises-25m-series-b-to-sell-virtual-lab-to-pharma/
3. "Turbine Raises €20 Million in Series A Financing", https://turbine.ai/news/turbine-raises-e20-million-in-series-a-financing-to-advance-programs-partnerships-towards-the-clinic-with-worlds-first-cancer-cell-simulation-platform/
4. "Budapest University spinout Turbine AI raises €25.5M for merging biology and AI", TFN, https://techfundingnews.com/budapest-university-spinout-turbine-ai-raises-e25-5m-for-merging-biology-and-ai-for-cancer-treatments/
5. "Turbine Launches the World's First Virtual Lab Using Cell Simulations", https://turbine.ai/news/virtual-lab-launch-2025/
6. "AI discovery outfit Turbine looks to turbocharge AstraZeneca's ADC efforts", FierceBiotech, https://www.fiercebiotech.com/medtech/turbine-looks-turbo-charge-astrazenecas-adc-discovery

**Last researched**: 2026-05-15

---

# 4. CytoReason — *named on Kinga's slide 9*

**URL**: https://cytoreason.com/
**Founded**: 2016 [1]
**Funding stage**: Series C
**Recent raise**: $80M (cited via Times of Israel headline) backed by Nvidia + Pfizer [2]. Total funding $130M across 3 rounds [3]. Founders Elina Starosvetsky, David Harel, Yuval Kalugny [3].
**Headquarters**: Haifa, Israel [1]
**Primary public claim**: AI-driven disease-intelligence platform for drug discovery, with mechanistic simulation models of the human immune system. Trusted by 5 of top 10 pharma companies [4].

**Modality coverage**:
- RNA: yes (bulk + single-cell transcriptomics) [4]
- ATAC: not surfaced as core capability
- Protein: yes (cytokine + cell-protein-gene level mechanistic models) [4]
- Phospho: not surfaced as core (consistent with Kinga's "DEEP INTRACELLULAR PROTEOMICS" — mark)
- VDJ: not surfaced
- Other: immune cell ontology + cytokine networks as platform foundation [4]

**Data strategy**:
- Public data only? Mostly — CytoReason builds models on proprietary + public clinical data integrated with literature [4]. Some proprietary clinical-trial-derived data via pharma partnerships.
- Proprietary wet lab? **Per Kinga's slide 9: YES (✓)**. Verified: CytoReason positions as analyzing multi-omic clinical data (their own + partner-derived) — but they do not run wet labs themselves [4]. The Kinga ✓ may reflect data-curation infrastructure rather than wet-lab capability — worth flagging.
- Partnerships for data? Pfizer ($110M+ deal extended 2027 [5]), Sanofi ($16M extension 2026 [6]), Japanese pharma (unnamed in some sources [7]).

**Model architecture**:
- Foundation model? Partial — disease-specific computational simulators, not a generic single foundation model.
- Architecture type: mechanistic + machine-learning hybrid; builds cell-protein-gene level disease models [4]. Specific architecture details proprietary.
- Modalities trained on: transcriptomics + clinical metadata + literature integration.
- Public model weights? No.

**Validation evidence**:
- Peer-reviewed publications? Several with pharma partners; specific count not surfaced from search.
- Cross-corpus generalization? Unclear — disease-specific models, not a general cross-corpus benchmark.
- Perturbation prediction validation? Yes — drug-response prediction is a core platform claim; specific public benchmarks limited.
- Specific benchmarks: 5 of top 10 pharma adoption is the strongest commercial validation signal [4].

**Differentiation CytoReason would claim against QurieGen**:
- "Half of top 10 pharma already use us — that's enterprise validation"
- "Series C, $130M raised, profitable partnerships ($110M Pfizer alone)"
- "10 years of platform development"
- "Pfizer equity investment ($20M equity portion of $110M deal [5]) is strategic validation"
- "Disease-specific models tuned to real clinical trial data — not lab cell lines"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: 10× our funding raise; commercial enterprise relationships at scale; pharma equity backing (Nvidia + Pfizer); Series C maturity vs our seed stage; clinical-trial-grade data partnerships.
- **What we have that they don't**: Same-cell multi-omics (RNA + ATAC + Protein) per Kinga's slide 9 (CytoReason — / —); proprietary wet-lab data engine for the multi-omic axes they lack; foundation-model framing (vs their mechanistic-hybrid); explicit lab-in-the-loop (per Kinga slide 9 CytoReason — mark for LAB-IN-LOOP).
- **Roughly equivalent**: clinical / immune-system focus (both immune-centric); pharma-partnership trajectory (we aspire, they have it).

**Sources**:
1. CytoReason Tracxn profile, https://tracxn.com/d/companies/cytoreason/__Lf0ZMG45nqKkQgUWP44yxVmCo_W5jkA5CFTEOWJB3Ic
2. "Backed by Nvidia and Pfizer, Israeli AI medical startup raises $80m in fresh capital", Times of Israel, https://www.timesofisrael.com/backed-by-nvidia-and-pfizer-israeli-ai-medical-startup-raises-80m-in-fresh-capital/
3. CytoReason PitchBook profile, https://pitchbook.com/profiles/company/435026-80
4. "Supporting Drug Discovery with Cell-centered Models of the Immune System", Technology Networks interview, https://www.technologynetworks.com/drug-discovery/blog/supporting-drug-discovery-with-cell-centered-models-of-the-immune-system-314664
5. "CytoReason, Pfizer ink $110M, 5-year extension of AI-powered drug development deal", FierceBiotech, https://www.fiercebiotech.com/medtech/cytoreason-pfizer-ink-110m-five-year-extension-ai-powered-drug-development-deal
6. "CytoReason extends its collaboration with Sanofi to advance AI-driven drug discovery", https://cytoreason.com/resources/cytoreason-extends-its-collaboration-with-sanofi-to-advance-ai-driven-drug-discovery/
7. "Japanese pharmaceutical firm taps Israel's CytoReason for AI drug development", Times of Israel, https://www.timesofisrael.com/japanese-pharmaceutical-firm-taps-israels-cytoreason-for-ai-drug-development/

**Last researched**: 2026-05-15

---

# 5. Valo Health — *named on Kinga's slide 9*

**URL**: https://www.valohealth.com/
**Founded**: 2019 (Flagship Pioneering origination) [1]
**Funding stage**: Late-stage private (planned SPAC merger paused) [2][3]
**Recent raise**: $300M Series B closed March 2021 (incl. $110M from Koch Disruptive Technologies) [3]. Total funding > $450M [3]. Announced $2.8B SPAC deal with Khosla Ventures Acquisition Co [2] (deal status: paused / status uncertain — verify before deck use).
**Recent expanded deals**: Novo Nordisk collaboration expanded — $190M upfront + up to $4.6B in milestones [4].
**Headquarters**: Boston, MA (with offices in San Francisco / Princeton / Branford / Lexington) [1]
**Primary public claim**: Opal Computational Platform combines human longitudinal + omics data for AI-driven drug discovery across therapeutic areas [5].

**Modality coverage**:
- RNA: yes (transcriptomics + clinical genomics)
- ATAC: not explicit in public materials
- Protein: yes (proteomics in Opal data stack) [5]
- Phospho: not explicit
- VDJ: not explicit
- Other: longitudinal human clinical data (EHR + genomic) is platform differentiator [5]

**Data strategy**:
- Public data only? No
- Proprietary wet lab? **Per Kinga's slide 9: YES (✓)**. Verified — Valo has its own tissue production / cellular engineering arm via the Opal platform's "chemistry + tissue production" pillar [5].
- Partnerships for data? Novo Nordisk ($4.6B potential) [4]; Charles River; other unnamed.

**Model architecture**:
- Foundation model? Not framed as one — Opal is platform-of-platforms (chemistry + computation + tissue production) rather than a single foundation model.
- Architecture type: machine learning across multi-modal omics + EHR + chemistry; specific architecture details proprietary.
- Modalities trained on: longitudinal patient omics + clinical data + chemistry.
- Public model weights? No.

**Validation evidence**:
- Peer-reviewed publications? Limited public-facing technical publications relative to platform claims.
- Cross-corpus generalization? Unclear — the Opal value proposition is patient-data-grounded, not cross-corpus generalization in the foundation-model sense.
- Perturbation prediction validation? Internal pipeline programs serve as validation surface (e.g., obesity / cardiometabolic with Novo Nordisk).
- Specific benchmarks: none publicly disclosed in foundation-model benchmark format.

**Differentiation Valo would claim against QurieGen**:
- "We have $450M+ raised and an active SPAC track — capital depth"
- "$4.6B Novo Nordisk milestone potential — that's the kind of pharma deal that defines the segment"
- "Longitudinal human EHR + omics — closer to clinical translation than your cell-line and PBMC data"
- "Internal pipeline + therapeutics (per Kinga ✓ on THERAPEUTICS column) — we ship drugs, not just platforms"
- "Cross-therapeutic breadth (cardiometabolic, obesity, others) vs your immune-only focus"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: ~45× our planned raise; live pharma partnerships at multi-billion-dollar scale; longitudinal human clinical data; internal drug pipeline with named therapeutic areas.
- **What we have that they don't**: Same-cell multi-omics (RNA + ATAC + Protein); explicit foundation-model architecture (vs Valo's platform-of-platforms framing); lab-in-the-loop iterative training (per Kinga ✓ for us, — for Valo); immune-specific causal architecture.
- **Roughly equivalent**: proprietary data engine (both have wet labs at scale).

**Sources**:
1. "Valo | Flagship Pioneering", https://www.flagshippioneering.com/companies/valo
2. "Valo Health and Khosla Ventures Acquisition Co. to Combine and Create Publicly Traded Company", https://www.valohealth.com/press/valo-health-and-khosla-ventures-acquisition-co-to-combine-and-create-publicly-traded-company-focused-on-transforming-the-drug-discovery-and-development-process
3. "Valo Health Receives $110 Million In Funding From Koch Disruptive Technologies To Close Series B", PRNewswire, https://www.prnewswire.com/news-releases/valo-health-receives-110-million-in-funding-from-koch-disruptive-technologies-to-close-series-b-301243086.html
4. "Novo Nordisk, Valo Health Ink Expanded Up-to-$4.6B AI Collaboration", GEN, https://www.genengnews.com/topics/artificial-intelligence/novo-nordisk-valo-health-ink-expanded-up-to-4-6b-ai-collaboration/
5. Valo Opal platform page, https://www.valohealth.com/what-we-do/platform

**Last researched**: 2026-05-15

---

# 6. Noetik — *named on Kinga's slide 9*

**URL**: https://www.noetik.ai/
**Founded**: ~2021 (seed 2023)
**Funding stage**: Series A
**Recent raise**: $40M Series A in August 2024 led by Polaris Partners (Amy Schulman) with Khosla Ventures, Wittington Ventures, Breakout Ventures, DCVC, Zetta Venture Partners [1]. Earlier $14M seed led by DCVC in 2023 [2]. Plus $50M GSK partnership for oncology AI foundation models [3].
**Total funding**: ≈ $54M + GSK collaboration
**Primary public claim**: AI-native biotech building multimodal spatial-biology platform ("OCTO" — Oncology Counterfactual Therapeutics Oracle) for cancer immunotherapy [1][4].

**Modality coverage**:
- RNA: yes (spatial transcriptomics)
- ATAC: not surfaced as core
- Protein: yes (spatial protein imaging — IHC / multiplex)
- Phospho: not surfaced
- VDJ: not surfaced
- Other: tissue-level spatial multi-omics; in-vivo CRISPR Perturb-Map [1]; non-small cell lung cancer atlas of 1,000+ cases [4]

**Data strategy**:
- Public data only? No
- Proprietary wet lab? **Per Kinga's slide 9: YES (✓)**. Verified — Noetik runs in-vivo CRISPR Perturb-Map and proprietary spatial multi-omics tissue profiling [1].
- Partnerships for data? GSK $50M oncology deal [3].

**Model architecture**:
- Foundation model? Yes — multimodal cancer foundation model trained via self-supervised learning on OCTO platform [4].
- Architecture type: self-supervised learning combining spatial biology + tissue imaging + multi-omics. Specific architecture (transformer / CNN / hybrid) not surfaced in detail in public materials.
- Modalities trained on: spatial transcriptomics + spatial protein + tissue imaging.
- Public model weights? No — closed platform.

**Validation evidence**:
- Peer-reviewed publications? Limited public surface; GSK partnership announcement [3] cites their oncology foundation model as licensable.
- Cross-corpus generalization? Unclear — focus is on NSCLC atlas-specific modeling.
- Perturbation prediction validation? Yes — in-vivo CRISPR Perturb-Map is a perturbation platform; specific benchmarks not public.
- Specific benchmarks: NSCLC tumor microenvironment characterization.

**Differentiation Noetik would claim against QurieGen**:
- "Spatial biology — tissue-level context vs your dissociated single cells"
- "In-vivo CRISPR perturbations on real tumor tissue — closer to clinical truth than cell lines or sorted PBMC"
- "$50M GSK deal already in hand — pharma validation"
- "Cancer market is concrete; immune perturbation is one application within a bigger oncology framing"
- "AI-native from day one — same generation as you, but raised more"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: Spatial context (tissue-level + spatial information we don't capture); in-vivo perturbation platform; $50M pharma partnership already closed; explicit oncology clinical target.
- **What we have that they don't**: Same-cell multi-omics (Noetik is spatial-multimodal but not same-cell multi-omic in the DOGMA sense — different stack); ATAC chromatin readouts; planned phospho integration; immune-system perturbation focus (we target immune signaling; they target tumor microenvironment); decomposed-readout architecture for compositional synergy.
- **Roughly equivalent**: AI-native framing (both); funding stage (Series A); foundation-model claim.

**Sources**:
1. "Noetik Secures $40 Million Series A Financing to Advance Precision Cancer Therapies", Business Wire, https://www.businesswire.com/news/home/20240829428359/en/Noetik-Secures-$40-Million-Series-A-Financing-to-Advance-Precision-Cancer-Therapies
2. "Noetik Raises $14 Million Seed Financing to Revolutionize Cancer Immunotherapy Using Artificial Intelligence", BioSpace, https://www.biospace.com/noetik-raises-14-million-seed-financing-to-revolutionize-cancer-immunotherapy-using-artificial-intelligence
3. "AI Foundation Models in Pharma: GSK-Noetik Oncology Deal", IntuitionLabs, https://intuitionlabs.ai/articles/ai-foundation-models-pharma-gsk-noetik-oncology
4. Noetik Lung Cancer Atlas page, https://www.noetik.ai/lungcanceratlas

**Last researched**: 2026-05-15

---

# 7. Immunai — *named on Kinga's slide 9*

**URL**: https://www.immunai.com/
**Founded**: 2018 [1]
**Funding stage**: Series B (unicorn status as of 2021)
**Recent raise**: $215M Series B closed 2021 (Schusterman, Talos Capital, Viola Group, Dexcel, others) achieving $1B+ unicorn valuation [2][3]. Total funding ≈ $295M as of 2025 [1].
**Recent pharma deals**: AstraZeneca $37.5M expanded oncology + IBD collaboration (2024/2025) [4]; Parker Institute for Cancer Immunotherapy single-cell cohort [5].
**Headquarters**: New York City [1]
**Primary public claim**: AMICA (Annotated Multiomic Immune Cell Atlas) — world's largest immune-focused harmonized single-cell database, powering immune drug discovery via IDE™ engine [6].

**Modality coverage**:
- RNA: yes (single-cell RNA-seq core)
- ATAC: yes (multi-omic claim; specifics not always public but ATAC is part of the AMICA stack per [6])
- Protein: yes (surface protein via CITE-seq-style readouts)
- Phospho: not surfaced as core
- VDJ: yes — TCR / BCR repertoire is part of Immunai's immune-cell atlas [6]
- Other: clinical metadata + functional genomics

**Data strategy**:
- Public data only? No
- Proprietary wet lab? **Per Kinga's slide 9: NO (— mark)**. Worth verifying — Immunai's AMICA is built via partner-derived clinical samples + clinical-trial integration, not via in-house wet lab generation at the same scale as Tahoe or Noetik [6]. May be more nuanced than a binary mark; specific clinical-sample-acquisition pipeline is proprietary.
- Partnerships for data? AstraZeneca [4]; Parker Institute (Parker-CITN, largest single-cohort dataset in cancer immunotherapy) [5]; multiple unnamed pharma collaborations.

**Model architecture**:
- Foundation model? Yes — IDE engine; AMICA database powers the model [6].
- Architecture type: not fully public; described as multi-omic foundation model.
- Modalities trained on: harmonized single-cell multi-omic immune data.
- Public model weights? No.

**Validation evidence**:
- Peer-reviewed publications? Multiple in single-cell immunology — Immunai team has contributed to high-profile immune-atlas publications.
- Cross-corpus generalization? Yes — by design, AMICA harmonizes across studies; specific cross-corpus benchmarks not always disclosed externally.
- Perturbation prediction validation? Unclear — Immunai is more about immune characterization than perturbation prediction.
- Specific benchmarks: Parker Institute partnership produces named single-cell cohort data.

**Differentiation Immunai would claim against QurieGen**:
- "$1B+ unicorn valuation; $295M raised; we're at scale you aspire to"
- "AMICA is the largest harmonized immune cell atlas in the world"
- "VDJ + multi-omic + clinical metadata — broader axes than you currently cover"
- "AstraZeneca and Parker Institute both publicly partnered with us — pharma + academic credibility"
- "Five years of platform development vs your three"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: 10× our planned funding; $1B+ valuation; VDJ as a real operational modality (we have it as Phase 2 plan); larger immune-data corpus; named pharma + academic partnerships.
- **What we have that they don't**: Same-cell trimodal DOGMA data (RNA + ATAC + Protein measured simultaneously in same cell — Immunai's AMICA harmonizes across studies, less single-cell same-modality coverage at scale); proprietary wet-lab data engine (per Kinga slide 9 Immunai — for PROPRIETARY DATA GEN); explicit perturbation-prediction architecture with decomposed readout + zero-arm constraint; planned phospho integration.
- **Roughly equivalent**: immune-system focus; lab-in-the-loop claim (per Kinga ✓ for both); virtual cell model claim.

**Sources**:
1. Immunai Tracxn / Crunchbase, https://tracxn.com/d/companies/immunai/__67EyQ7bi9KEsshAtS4atAUevx9YjXzpxWcyAMaIrOiQ
2. "Immunai Scores $215 Million to Accelerate Mapping of the Immune System", BioSpace, https://www.biospace.com/immunai-scores-215-million-to-accelerate-mapping-of-the-immune-system
3. "Biotech startup Immunai earns unicorn status with $215 million Series B", CTech, https://www.calcalistech.com/ctech/articles/0,7340,L-3921148,00.html
4. "AstraZeneca expands AI partnership with Immunai in deal worth up to $37.5 million", CTech, https://www.calcalistech.com/ctechnews/article/bkhaclqcwe
5. "Immunai and the Parker Institute Collaborate to Build One of the Largest Single-Cell Datasets in Cancer", https://www.parkerici.org/the-latest/immunai-and-the-parker-institute-collaborate-to-build-one-of-the-largest-single-cell-datasets-in-cancer/
6. Immunai company site, https://www.immunai.com/

**Last researched**: 2026-05-15

---

# Competitors NOT on Kinga's slide 9 — discovered during research

These three are well-known adjacencies in the AI-virtual-cell / foundation-model space. They may not appear on Kinga's slide because they're either too large (Recursion is public; Insitro raised $643M) or strategically positioned differently (Cellarity targets cell-state correction broadly, not immune perturbation). Including them honestly because a technical investor reading the slide will notice they're missing.

---

# 8. Recursion Pharmaceuticals — *adjacent, not on Kinga's slide 9*

**URL**: https://www.recursion.com/
**Founded**: 2013
**Funding stage**: **Public** (NASDAQ: RXRX)
**Current market cap**: ≈ $1.79B (May 2026 snapshot) [1], down from $2.18B in January 2026.
**2025 revenue**: $74.68M (up 26.92% YoY); 2025 net loss: −$644.76M [1].
**Primary public claim**: Industrial-scale phenomics (cell-imaging) platform + foundation models for drug discovery; "building the first virtual cell" framing [2].

**Modality coverage**:
- RNA: yes (Trekseq industrialized transcriptomics) [3]
- ATAC: not core
- Protein: indirect via cell painting imaging
- Phospho: not core
- VDJ: not core
- Other: phenomics imaging (Cell Painting + Brightfield) at 2.2M experiments / week [3]; functional genomics

**Data strategy**:
- Public data only? No
- Proprietary wet lab? Yes — industrial-scale (2.2M experiments per week, automated robotic labs) [3].
- Partnerships for data? Nvidia (BioNeMo) [4]; Bayer; Roche; Helix; multiple.

**Model architecture**:
- Foundation model? Yes — Phenom-1, Phenom-Beta foundation models on NVIDIA BioNeMo [4]
- Architecture type: vision transformer backbone for cell-image foundation modeling; multi-modal extensions [4]
- Modalities trained on: cell painting images + brightfield + transcriptomics; not multi-omics in single-cell DOGMA sense
- Public model weights? Yes — Phenom-1 model card distributed via NVIDIA BioNeMo [4]; RxRx3 dataset publicly available [5].

**Validation evidence**:
- Peer-reviewed publications? Multiple — Phenom-Beta paper published, RxRx datasets benchmarked widely [5].
- Cross-corpus generalization? Yes — RxRx3-trained models generalize across cell-painting variants and JUMP-CP [5].
- Perturbation prediction validation? Yes — multiple internal benchmarks on gene-knockout + chemical perturbation.
- Specific benchmarks: RxRx-series public datasets; internal target ID pipeline metrics.

**Differentiation Recursion would claim against QurieGen**:
- "Public company. $1.8B market cap. You're seed-stage."
- "2.2M experiments per week — three orders of magnitude more throughput than you'll ever have"
- "Two foundation models open-sourced on NVIDIA BioNeMo (Phenom-1, Phenom-Beta) — your weights aren't public"
- "Pipeline programs in real clinical trials — we're past virtual cell into actual therapeutics"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: Public-company scale; industrial wet-lab throughput; deployed foundation models with public weights; multi-billion-dollar partnership track record; brand recognition.
- **What we have that they don't**: Same-cell multi-omics (RNA + ATAC + Protein per cell — Recursion's stack is imaging + bulk transcriptomics, not same-cell multi-omics); immune-system / PBMC focus (Recursion is broader); decomposed-readout / synergy-prediction architecture; phospho integration plan.
- **Roughly equivalent**: foundation-model framing; wet-lab proprietary claim.

**Why Recursion may not be on Kinga's slide**: Recursion's primary modality is cell-imaging (Cell Painting) not single-cell multi-omics. Kinga's slide criteria emphasize "OWN SINGLE-CELL MULTI-OMICS" — Recursion doesn't fit that column natively. Still, for a sophisticated investor, "where's Recursion?" is a likely question.

**Sources**:
1. Recursion Pharmaceuticals (RXRX) Market Cap, Stock Analysis, https://stockanalysis.com/stocks/rxrx/market-cap/
2. "Since Its Inception, Recursion has Been Building the Foundation for the First Virtual Cell", Recursion, https://www.recursion.com/news/since-its-inception-recursion-has-been-building-the-foundation-for-the-first-virtual-cell
3. "How Phenomic Foundation Models Are Empowering the Brightfield Comeback", Recursion, https://www.recursion.com/news/how-phenomic-foundation-models-are-empowering-the-brightfield-comeback
4. "Nothing Short of Phenomenal: New Deep Learning Model Available on NVIDIA's BioNeMo Platform", Recursion, https://www.recursion.com/news/nothing-short-of-phenomenal-new-deep-learning-model-available-on-nvidias-bionemo-platform
5. "Recursion's Phenom-Beta aims to upend phenomics research", DrugDiscoveryTrends, https://www.drugdiscoverytrends.com/recursion-phenom-beta-phenomics/

**Last researched**: 2026-05-15

---

# 9. Insitro — *adjacent, not on Kinga's slide 9*

**URL**: https://www.insitro.com/
**Founded**: 2018 by Daphne Koller (former Coursera co-founder + Stanford CS professor)
**Funding stage**: Series C
**Total funding**: $643M across 3 rounds [1][2]
**Recent valuation**: $2.4B set at Series C close (2021) [2]
**2024 revenue**: $69M [2]

**Primary public claim**: Integrates machine learning + multi-omics + human genetics to discover and develop targeted therapeutics across multiple disease areas [3].

**Modality coverage**:
- RNA: yes
- ATAC: not surfaced as core
- Protein: yes (via POSH platform — pooled optical screening)
- Phospho: not surfaced
- VDJ: not core
- Other: high-content imaging + CRISPR pooled screens (POSH) [4]; human genetics integration

**Data strategy**:
- Public data only? No
- Proprietary wet lab? Yes — POSH platform combines pooled CRISPR + high-content imaging + ML [4]; internal stem cell + disease-model labs.
- Partnerships for data? Bristol Myers Squibb [2]; Lilly TuneLab AI partnership announced Sep 2025 [5]; Genentech historically.

**Model architecture**:
- Foundation model? Multiple ML models across modalities; CellPaint-POSH model described in Nature Communications 2025 [4].
- Architecture type: self-supervised on cell morphology; multi-modal extensions; specific architectures proprietary per use case.
- Modalities trained on: high-content imaging + pooled-screen perturbations + genetics + transcriptomics.
- Public model weights? Limited — research publications describe methods, not weights.

**Validation evidence**:
- Peer-reviewed publications? Yes — Nature Communications 2025 (POSH) [4], multiple others.
- Cross-corpus generalization? Yes — POSH demonstrates reconstructing gene function without predefined biomarkers [4].
- Perturbation prediction validation? Yes — pooled CRISPR + imaging perturbation predictions validated in publications.
- Specific benchmarks: published in [4].

**Differentiation Insitro would claim against QurieGen**:
- "Daphne Koller — name recognition for ML credibility"
- "$643M raised, $2.4B valuation; we're orders of magnitude beyond seed"
- "Lilly TuneLab + BMS partnerships — real pharma traction"
- "Nature Communications publication on POSH — peer-reviewed at top venue"
- "Multi-disease pipeline (cancer, ALS, liver, others) — therapeutic breadth"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: Series C + multi-billion-dollar pharma deals; high-profile founder; peer-reviewed top-tier publications; multi-therapeutic-area pipeline.
- **What we have that they don't**: Same-cell DOGMA multi-omics (Insitro is imaging + screens + transcriptomics, not RNA+ATAC+Protein in same cell); explicit immune-system / PBMC focus (Insitro is generalist); decomposed-readout perturbation-prediction architecture; planned phospho integration.
- **Roughly equivalent**: ML-as-foundation framing; wet-lab proprietary platform.

**Why Insitro may not be on Kinga's slide**: Insitro's positioning is "ML for drug discovery" generally, not specifically "single-cell multi-omics virtual cell" — so they're adjacent rather than direct.

**Sources**:
1. "Insitro raises $400M for machine learning-powered drug discovery efforts", FierceBiotech, https://www.fiercebiotech.com/medtech/insitro-raises-400m-for-machine-learning-powered-drug-discovery-efforts
2. "insitro Revenue 2024: $69M ARR, $2.4B Valuation", GetLatka, https://getlatka.com/companies/insitro
3. "Daphne Koller: How Insitro is Reprogramming Drug Discovery", Klover.ai, https://www.klover.ai/daphne-koller-how-insitro-is-reprogramming-drug-discovery/
4. "insitro Validates AI-Enabled POSH Platform in Nature Communications", https://www.insitro.com/news/insitro-validates-ai-enabled-posh-platform-in-nature-communications-bridging-critical-gap-in-drug-discovery/
5. "insitro partners with Lilly to build first-in-kind machine learning models", Business Wire, https://www.businesswire.com/news/home/20250908958690/en/insitro-Partners-with-Lilly-to-Build-First-in-Kind-Machine-Learning-Models-to-Advance-Small-Molecule-Drug-Discovery

**Last researched**: 2026-05-15

---

# 10. Cellarity — *adjacent, not on Kinga's slide 9*

**URL**: https://cellarity.com/
**Founded**: 2019 (Flagship Pioneering origination)
**Funding stage**: Late-stage private; lead asset (CLY-124) in Phase 1
**Recent raise**: Specifics not surfaced in primary search; Flagship-backed with multiple closes. Public data is less granular than other peers.
**Primary public claim**: Cell-state-correcting medicines — discover drugs that move whole-cell states from disease back to healthy [1][2].

**Modality coverage**:
- RNA: yes (high-dimensional transcriptomics at single-cell resolution) [3]
- ATAC: yes (recently released single-cell multi-omic hematopoiesis atlas combining RNA + ATAC + surface receptors) [3]
- Protein: yes (surface receptors in the same atlas) [3]
- Phospho: not core
- VDJ: not core
- Other: dynamic AI modeling of gene networks

**Data strategy**:
- Public data only? No — proprietary perturbational datasets (1.26M cells, 1,700+ samples published in *Science* 2025) [3]
- Proprietary wet lab? Yes — internal perturbation dataset generation
- Partnerships for data? Flagship Pioneering origination; specific pharma partnerships not loud in public.

**Model architecture**:
- Foundation model? Dynamic AI modeling framework for cell-state transitions; specific architecture described in *Science* publication [3].
- Architecture type: cell-state trajectory modeling combining transcriptomics + perturbation response.
- Modalities trained on: RNA-seq + perturbation + recently multi-omic (RNA + ATAC + surface).
- Public model weights? Limited — research-paper-level disclosure; not a publicly distributed model.

**Validation evidence**:
- Peer-reviewed publications? Yes — *Science* publication October 2025 on cell-state-correcting framework [1][2][3]. This is the strongest peer-reviewed validation in the competitive set (top-tier journal).
- Cross-corpus generalization? Demonstrated in [3] — perturbational transcriptomic dataset enables cross-cell-type drug response mapping.
- Perturbation prediction validation? Yes — published benchmark methodology in [3].
- Specific benchmarks: 1.26M-cell perturbational dataset released as community resource [3].

**Clinical asset**: CLY-124 in Phase 1 for sickle cell disease via globin-switching mechanism [3] — they're actually in the clinic.

**Differentiation Cellarity would claim against QurieGen**:
- "*Science* publication on our framework — peer-reviewed at the highest tier"
- "Already in Phase 1 clinical trials (CLY-124, sickle cell)"
- "1.26M-cell published perturbational dataset — open to community"
- "Multi-omic hematopoiesis atlas combining RNA + ATAC + surface receptors at single-cell resolution"
- "Flagship Pioneering backing — same family as Moderna, Indigo, Valo"

**Gap vs QurieGen (honest)**:
- **What they have that we don't**: Phase 1 clinical asset; *Science* publication; multi-omic single-cell dataset already published (RNA + ATAC + surface receptors); broader disease coverage (hematopoiesis + others).
- **What we have that they don't**: Phospho integration plan; explicit foundation-model + decomposed-readout architecture for compositional generalization; immune-system / PBMC perturbation focus (Cellarity is hematopoiesis + cell-state generally); lab-in-the-loop framing.
- **Roughly equivalent**: multi-omic single-cell capability (Cellarity's hematopoiesis atlas is closest peer to our DOGMA stack on modality breadth — this is the most direct head-to-head). [Honest takeaway: Cellarity's atlas is more like our DOGMA pretraining substrate than any other competitor's data.]

**Why Cellarity may not be on Kinga's slide**: Cellarity's positioning is "cell-state-correcting medicines" — a therapeutic framing, not explicitly a "virtual cell" or "lab-in-the-loop" framing. They are still highly relevant for the modality-coverage and validation-rigor comparisons.

**Sources**:
1. "Cellarity Publishes Framework for Discovery of Cell State-Correcting Medicines in Science", Business Wire, https://www.businesswire.com/news/home/20251021180543/Cellarity-Publishes-Framework-for-Discovery-of-Cell-State-Correcting-Medicines-in-Science
2. "Cellarity publishes framework for discovery of cell state-correcting medicines in Science", EurekAlert!, https://www.eurekalert.org/news-releases/1102610
3. "Cellarity Publishes Framework for Discovery of Cell State-Correcting Medicines in Science" (full press), Cellarity site, https://cellarity.com/news_item/cellarity-publishes-framework-for-discovery-of-cell-state-correcting-medicines-in-science/

**Last researched**: 2026-05-15

---

# Aggregate comparison table

| Company | Funding stage | Total raised / mkt cap | Wet lab | Multi-omics RNA+ATAC+Protein in same cell | VDJ | Phospho | Peer-reviewed pubs at top venues | Cross-corpus generalization shown | Therapeutics pipeline |
|---|---|---|---|---|---|---|---|---|---|
| TAHOE | Series A | $42M / $120M val | ✓ | ✗ (RNA only) | ✗ | ✗ | preprints only | unclear | ✗ (data/platform) |
| DeepLife | Series A | ~$10M+ | ✗ | unclear | ✗ | ✗ | ✗ | unclear | ✗ |
| Turbine | Series B | ~$37M | ✗ | unclear | ✗ | partial | limited | ✗ | ✗ |
| CytoReason | Series C | ~$130M | partial (curation) | ✗ | ✗ | ✗ | yes | ✗ | ✗ |
| Valo | Late-private | $450M+ | ✓ | unclear | ✗ | ✗ | limited | unclear | ✓ |
| Noetik | Series A | ~$54M + $50M GSK | ✓ (spatial + in-vivo) | spatial multi-omics ≠ same-cell | ✗ | ✗ | limited | unclear | ✓ |
| Immunai | Series B (unicorn) | $295M / $1B+ | partial | partial (AMICA) | ✓ | ✗ | yes | partial | ✗ |
| Recursion | Public (RXRX) | $1.79B mkt cap | ✓ (industrial) | ✗ (imaging + bulk RNA) | ✗ | ✗ | yes | yes (RxRx3) | ✓ |
| Insitro | Series C | $643M / $2.4B val | ✓ | ✗ | ✗ | ✗ | yes (*Nat Comms*) | yes | ✓ |
| Cellarity | Late-private | unclear | ✓ | partial (RNA+ATAC+surface) | ✗ | ✗ | yes (*Science*) | yes | ✓ (Phase 1) |
| **QurieGen** | **Seed (in raise)** | **$10M ask** | **✓ (planned Phase 1)** | **✓ (DOGMA + planned QurieSeq)** | **✗ (Phase 2 plan)** | **✓ (Phase 2 plan)** | **closure reports (not peer-reviewed)** | **73% Calderon (pre-registered)** | **✗ (planned 2027)** |

Note: rows where a competitor has the capability are marked from the competitor's own claims; ✗ means it isn't publicly claimed; "unclear" means search didn't surface a definitive yes/no.

---

## Synthesis — Patterns Worth Noting

### Where the competitive set converges (table-stakes capabilities)
Every direct competitor + every adjacent competitor claims at least the following:
1. **Some form of single-cell or cellular data at scale** — universal table stakes.
2. **Some form of "AI" or "foundation model" framing** — every competitor uses it, often loosely.
3. **At least one pharma partnership** — pharma BD is the de facto commercial validation everyone seeks.
4. **A "virtual cell" or "digital twin" or "cell-state" framing** — generic naming; specific architecture differs widely.
5. **Lab-in-the-loop language** in marketing — though execution differs (Turbine = partner-driven loop; Recursion = industrial in-house; CytoReason = data-curation loop).

### Where the competitive set diverges (real differentiators)
1. **Data modality stack**:
   - **RNA-only**: TAHOE (the largest-scale player is single-modality)
   - **Spatial multimodal (image + transcriptomics + protein)**: Noetik, Recursion
   - **Same-cell multi-omic (RNA + ATAC + Protein)**: Cellarity (partial), QurieGen (planned)
   - **Multi-omic via harmonization across studies (not same-cell)**: Immunai (AMICA), CytoReason
2. **Data scale**:
   - 100M+ cells: TAHOE (and partners)
   - Industrial-throughput proprietary lab: Recursion (2.2M experiments/week)
   - Mid-scale proprietary: Insitro, Cellarity, Noetik, Valo, Immunai
   - Small / public-data: DeepLife, Turbine
3. **Therapeutic stage**:
   - In clinical trials: Cellarity (CLY-124 Phase 1), Recursion (multiple programs)
   - Pipeline announced: Valo, Noetik
   - Platform / data only: TAHOE, DeepLife, Turbine, CytoReason, Immunai, QurieGen
4. **Funding stage** (gross asymmetry):
   - Public / multi-billion: Recursion, Valo ($450M+), Insitro ($643M)
   - Unicorn private: Immunai ($1B+)
   - Mid-stage private: CytoReason ($130M), Turbine ($37M)
   - Seed / Series A: TAHOE ($42M), Noetik ($54M), DeepLife ($10M), QurieGen (current raise)

### Capabilities NO competitor currently has (QurieGen's defensible territory)
Per Kinga's slide 9 framing + research verification:
1. **Same-cell multi-omics including PHOSPHO**. No competitor in this set claims phospho integration. Even multi-modal players (Cellarity, Immunai) don't have phospho. **This is the cleanest white space** — phospho would be a real new capability that competitors don't currently match. Caveat: Cellarity's hematopoiesis atlas adds RNA + ATAC + surface to one cell type; if they extend to phospho, our gap closes.
2. **PBMC immune-perturbation focus with planned compositional drug-combination generalization**. Noetik is oncology / spatial; Immunai is immune characterization (not perturbation prediction); Turbine is cancer cell simulation. Nobody in this exact intersection.
3. **Decomposed-readout architecture with zero-arm constraint for compositional synergy**. This is a specific architectural claim. The exact 4-arm decomposition is QurieGen-original per the architecture spec; no competitor publishes this approach. (Note: many competitors claim "compositional" generally; the specific zero-arm L2 constraint is ours.)
4. **Pre-registered cross-corpus generalization threshold (e.g., 0.70 Calderon pre-registered)**. Methodological rigor as a claim — most competitors don't publish pre-registered thresholds before running evals. This is a credibility differentiator, not a capability differentiator.

### Capabilities multiple competitors claim that we lack (honest gaps)
1. **Data scale 100M+ cells**: TAHOE's 100M-cell open dataset dwarfs our 500K Phase 1 plan by ~200×. Reframing: "scale isn't the right axis — same-cell multi-omics is" — but a sophisticated investor will press on this.
2. **Peer-reviewed top-tier publication**: Cellarity (*Science*), Insitro (*Nat Comms*), Immunai (*Nat Immunol* and similar) — we have closure reports, not peer-reviewed papers yet. Pre-registered methodology helps narratively but doesn't substitute for a *Nature* publication.
3. **Therapeutic pipeline in clinic**: Cellarity (Phase 1), Recursion (multiple), Valo (multiple), Noetik (preclinical). Our therapeutics are Phase 4 of the roadmap — 2029+.
4. **Major pharma partnership at $50M+ scale**: Noetik (GSK $50M), CytoReason (Pfizer $110M), Valo (Novo $4.6B), Immunai (AstraZeneca $37.5M), Insitro (Lilly TuneLab), Turbine (AstraZeneca/Bayer). Every named competitor has at least one $50M+ pharma deal; we have zero today.
5. **VDJ as operational modality**: Immunai has VDJ in production. Ours is Phase 2 plan.
6. **Industrial wet-lab throughput**: Recursion (2.2M/week), TAHOE (300M+ cells planned), Noetik (in-vivo Perturb-Map at scale). Our 500K Phase 1 is modest in comparison.
7. **Public foundation model weights**: TAHOE-x1 (3B params), Recursion Phenom on NVIDIA BioNeMo. Both have open weights; we don't currently.

### QurieGen's strongest defensible angles (ranked, with reasoning)

1. **Phospho-integrated same-cell multi-omics (Phase 2 plan)** — *Why*: this is the only capability with zero competitor coverage in the entire research set. Phospho is hard (it requires QurieSeq-style proprietary protocol). Even if a competitor decides to add phospho tomorrow, they don't have our protocol. **Caveat: this is a Phase 2 plan, not a shipped capability** — must be framed honestly as roadmap, not current state.

2. **Compositional synergy prediction with zero-arm-constrained decomposed readout** — *Why*: the architecture is original; nobody else publishes this exact 4-arm formulation. The BTK+JAK demo plan is concrete, pre-registered, and biologically grounded (Maddocks 2016 + Thiago pJAK1 finding). Competitors who claim "compositional" generally don't have this specific construct.

3. **Methodological rigor as institutional discipline** — *Why*: pre-registered thresholds, public closure reports including failure modes, cross-corpus methodology documented before results were generated. This is process-credibility, not a single capability. Stronger than most competitors' marketing. Caveat: it's a *credibility* differentiator more than a capability moat — it makes our claims more believable but doesn't directly produce a unique product.

4. **Immune-system focus with planned 5-modality scope (RNA + ATAC + Protein + Phospho + VDJ)** — *Why*: no competitor has all 5 as a same-cell stack. Immunai is closest (their AMICA includes VDJ + multi-omic but harmonized across studies, not same-cell). If we ship Phase 2 + 3 as planned, we have a unique stack. Caveat: today we have 3 of 5 in operation; the 5-modality story is forward-looking.

5. **Decomposed readout (zero-arm constraint) as architectural choice** — *Why*: specific to perturbation prediction; the 4-arm decomposition is what enables zero-shot synergy. Most competitors do generic embedding / latent models without this specific separability. Caveat: same as #2 — overlap.

### Open questions for Ash + Claude (strategic decisions, not research)

1. **Should the F1 slide call out specific competitors by name, or talk in categories?** Kinga's slide 9 names 7 competitors with binary marks. Research above shows the binary marks oversimplify (e.g., CytoReason ✓ for PROPRIETARY DATA GEN is debatable; Immunai's "harmonized multi-omic" vs "same-cell" distinction matters). A named-comparison slide invites diligence; a category-comparison slide protects credibility but may feel evasive.

2. **How do we handle "scale" honestly?** TAHOE's 100M cells is 200× our planned 500K. Either we (a) reframe to "scale isn't the right axis — same-cell multi-omics is", (b) acknowledge the scale gap and emphasize quality + modality, or (c) commit to a roadmap scale-up number that closes the gap. The slide design needs a strategic choice here.

3. **Recursion / Insitro / Cellarity adjacency — call them out or not?** They're not on Kinga's slide 9. If we omit them, sophisticated investors may flag the absence as a credibility gap. If we include them, we acknowledge stronger players exist. Recommendation: include with a "different segment but related" frame (Recursion is imaging, Insitro is genetics + screens, Cellarity is cell-state). Show that we know the landscape.

4. **Should the slide lead with what we have today vs what we plan?** Phospho is our cleanest white space, but it's Phase 2 plan, not shipped. If we lead with phospho, we lead with a roadmap claim — investors should know that. If we lead with "DOGMA + decomposed readout + 73% cross-corpus", we lead with shipped capability but it's less differentiated.

5. **Cellarity as the closest peer on modality breadth — explicit comparison?** Their hematopoiesis atlas (RNA + ATAC + surface receptors) is the closest to our DOGMA stack. They also have a *Science* publication. If we name them, we elevate them; if we don't, investors who know them may notice. Strategic call.

6. **How do we handle the "no peer-reviewed publication" gap?** Cellarity (*Science*), Insitro (*Nat Comms*), Immunai have peer-reviewed papers; we have closure reports. The slide could either (a) emphasize pre-registration as a substitute, (b) commit to a publication target by a specific date, or (c) say nothing on this dimension.

7. **TAHOE-x1's 3B-param open-source foundation model is a real threat**. Even if its modality is RNA-only, the open availability + 100M training corpus means academic researchers will benchmark on it freely. Do we (a) frame it as "complementary, not competing" (they're tools, we're a vertically-integrated platform), or (b) head-to-head on a benchmark where same-cell multi-omics matters?

---

*End of competitive landscape research, 2026-05-15.*

*Next: F1 content spec to be authored separately by Ash + Claude based on findings above.*
