#!/usr/bin/env python3
"""
DOGMA arm classifier for GSE156478 (Mimitou 2021 superseries) — schema v2.

Changes from v1 (phase-6.5g.1 disaggregation, committed 8982fd0):
  (a) Tokenize Sample/Title on [_\-\s/]+ before matching. v1's \b-anchored
      regex failed because Python treats '_' as a word character, so
      `\basap\b` never matches the deposit's `_ASAP_` convention.
  (b) Inferred-DOGMA rule. DOGMA is not a title token in this deposit
      (author uses perm+protocol-implicit naming: LLL-DOGMA, OMNI-DOGMA).
      If perm ∈ {DIG, LLL, OMNI} and no explicit ASAP/CITE token, emit
      protocol=DOGMA at confidence=LOW — inferred, not stated.
  (c) Paper-anchored GSM→arm map (external file) can override inferred LOW
      classifications to HIGH. Until that map is filled, HIGH-confidence
      DOGMA arms remain empty per no-inference discipline.

Schema contract identical to v1: every classified field is
  {value, evidence, source_field, confidence, [is_authoritative]}
with confidence ∈ {HIGH, MEDIUM, LOW, AMBIGUOUS, UNKNOWN}.

Usage: python3 dogma_classifier_v3.py
Exit: 0 on successful YAML emit; non-zero on schema-validation failure.
HALT conditions are handled by the calling shell, not by this script.
"""
import re, sys, json, pathlib, xml.etree.ElementTree as ET
import yaml

SCHEMA_VERSION = 2
CONFIDENCE = ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "UNKNOWN")
NS = {"m": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}

EVID = pathlib.Path("data/phase6_5g_2/external_evidence/dogma_disaggregation_2026-04-23")
XML  = EVID / "GSE156478.full.miniml.xml"
YAML_OUT  = EVID / "GSE156478.arm_disaggregation.yaml"
JSON_OUT  = EVID / "GSE156478.samples.json"
PAPER_MAP_PATH = pathlib.Path("data/phase6_5g_2/external_evidence/dogma_gsm_map_manual_2026-04-23.yaml")

PROTO_TOKENS_EXPLICIT = {
    "ASAP":  {"asap", "asapseq"},
    "CITE":  {"cite", "citeseq"},
    "DOGMA": {"dogma", "dogmaseq"},
}
PERM_TOKENS = {
    "DIG":  {"dig", "digitonin"},
    "LLL":  {"lll"},
    "OMNI": {"omni"},
}
STIM_TOKENS = {
    "stim":    {"stim", "stimulated", "cd3cd28"},
    "control": {"control", "resting", "unstim", "unstimulated", "ctrl"},
}

def classif(value=None, evidence=None, source_field=None, confidence="UNKNOWN"):
    assert confidence in CONFIDENCE
    return {"value": value, "evidence": evidence,
            "source_field": source_field, "confidence": confidence}

def tokenize(s):
    return [t for t in re.split(r"[_\-\s/]+", (s or "").lower()) if t]

def classify_by_token(tokens, token_map, source_field):
    hits = {name for name, toks in token_map.items() if tokens & toks}
    if len(hits) == 1:
        name = list(hits)[0]
        matched = sorted(tokens & token_map[name])[0]
        return classif(value=name, evidence=f"token:{matched}",
                       source_field=source_field, confidence="MEDIUM")
    if len(hits) > 1:
        return classif(evidence=f"multi-match:{sorted(hits)}",
                       source_field=source_field, confidence="AMBIGUOUS")
    return classif(confidence="UNKNOWN")

def classify_protocol_with_inference(title_tokens, perm_classif):
    """Explicit protocol wins; else infer DOGMA from perm token presence."""
    explicit = classify_by_token(title_tokens, PROTO_TOKENS_EXPLICIT, "Sample/Title")
    if explicit["confidence"] in ("MEDIUM", "AMBIGUOUS"):
        return explicit
    if perm_classif["confidence"] == "MEDIUM" and perm_classif["value"] in ("DIG", "LLL", "OMNI"):
        return classif(
            value="DOGMA",
            evidence=f"inferred from permeabilization={perm_classif['value']} "
                     f"(no explicit ASAP/CITE/DOGMA token in title)",
            source_field="inferred:Sample/Title.permeabilization_token",
            confidence="LOW",
        )
    return explicit

def classify_n_cells_hint(title, chars, supp):
    for field, blob in [("Sample/Title", title or ""),
                        ("Sample/Supplementary-Data", " ".join(supp)),
                        ("Sample/Characteristics", " ".join(chars))]:
        m = re.search(r"(\d{3,7})\s*(cells?|nuclei)\b", blob, re.I)
        if m:
            return {**classif(value=int(m.group(1)), evidence=m.group(0),
                              source_field=field, confidence="LOW"),
                    "is_authoritative": False,
                    "caveat": "heuristic regex over free text; paper pass authoritative"}
    return {**classif(confidence="UNKNOWN"),
            "is_authoritative": False,
            "caveat": "no numeric hint found; paper pass required"}

paper_map = {}
if PAPER_MAP_PATH.exists():
    paper_map_doc = yaml.safe_load(PAPER_MAP_PATH.read_text()) or {}
    paper_map = paper_map_doc.get("gsm_to_arm", {})
    print(f"[info] Paper-anchored map loaded: {len(paper_map)} GSM entries")
else:
    print(f"[info] No paper-anchored map at {PAPER_MAP_PATH} — HIGH-confidence "
          f"DOGMA promotion will not occur")

tree = ET.parse(XML)
root = tree.getroot()
samples = []
for s in root.findall(".//m:Sample", NS):
    iid   = s.get("iid", "")
    title = (s.findtext("m:Title", default="", namespaces=NS) or "").strip()
    srcs  = [x.text.strip() for x in s.findall(".//m:Source", NS) if x.text]
    chrs  = [x.text.strip() for x in s.findall(".//m:Characteristics", NS) if x.text]
    lib   = (s.findtext(".//m:Library-Strategy", default="", namespaces=NS) or "").strip()
    mol   = (s.findtext(".//m:Molecule", default="", namespaces=NS) or "").strip()
    org   = (s.findtext("m:Channel/m:Organism", default="", namespaces=NS) or "").strip()
    supp  = [sf.text for sf in s.findall(".//m:Supplementary-Data", NS) if sf.text]

    title_toks = set(tokenize(title))
    perm       = classify_by_token(title_toks, PERM_TOKENS, "Sample/Title")
    protocol   = classify_protocol_with_inference(title_toks, perm)
    stim       = classify_by_token(title_toks, STIM_TOKENS, "Sample/Title")

    if iid in paper_map:
        pm = paper_map[iid]
        if "protocol" in pm:
            protocol = classif(value=pm["protocol"],
                               evidence=f"paper_anchored_map@{PAPER_MAP_PATH.name}",
                               source_field="paper:Mimitou2021_SuppTable",
                               confidence="HIGH")
        if "permeabilization" in pm:
            perm = classif(value=pm["permeabilization"],
                           evidence=f"paper_anchored_map@{PAPER_MAP_PATH.name}",
                           source_field="paper:Mimitou2021_SuppTable",
                           confidence="HIGH")

    samples.append({
        "gsm": iid, "title": title, "organism": org,
        "library_strategy": lib, "molecule": mol,
        "sources_raw": srcs, "characteristics_raw": chrs,
        "supplementary_files": supp,
        "title_tokens": sorted(title_toks),
        "protocol": protocol,
        "permeabilization": perm,
        "stim": stim,
        "n_cells_hint": classify_n_cells_hint(title, chrs, supp),
    })

from collections import defaultdict
def usable(c): return c["confidence"] in ("HIGH", "MEDIUM")

arms_hm = defaultdict(list)
arms_low = defaultdict(list)
unclassified = []
for s in samples:
    p, pr = s["protocol"], s["permeabilization"]
    if usable(p) and usable(pr):
        arms_hm[(p["value"], pr["value"])].append(s["gsm"])
    elif p["confidence"] == "AMBIGUOUS" or pr["confidence"] == "AMBIGUOUS":
        unclassified.append(s["gsm"])
    elif p["confidence"] == "UNKNOWN" and pr["confidence"] == "UNKNOWN":
        unclassified.append(s["gsm"])
    else:
        arms_low[(p["value"] or "unknown", pr["value"] or "unknown")].append(s["gsm"])

def aggregate_arm(gsms, tier):
    ncells_vals = [s["n_cells_hint"]["value"] for s in samples
                   if s["gsm"] in gsms and s["n_cells_hint"]["value"] is not None]
    stim_count = {"stim": 0, "control": 0, "unknown_or_low": 0}
    for s in samples:
        if s["gsm"] not in gsms: continue
        if usable(s["stim"]):
            stim_count[s["stim"]["value"]] += 1
        else:
            stim_count["unknown_or_low"] += 1
    return {
        "n_samples": len(gsms),
        "gsms": sorted(gsms),
        "confidence_tier": tier,
        "n_cells_hint_sum": {
            "value_sum": sum(ncells_vals) if ncells_vals else None,
            "n_samples_contributing": len(ncells_vals),
            "is_authoritative": False,
            "caveat": "Sum of LOW-confidence heuristic hits. Paper pass required.",
        },
        "stim_breakdown_usable": stim_count,
    }

arms_hm_out  = [{"protocol": p, "permeabilization": pr, **aggregate_arm(g, "HIGH_MEDIUM")}
                for (p, pr), g in sorted(arms_hm.items())]
arms_low_out = [{"protocol": p, "permeabilization": pr, **aggregate_arm(g, "LOW")}
                for (p, pr), g in sorted(arms_low.items())]

report = {
    "schema_version": SCHEMA_VERSION,
    "accession": "GSE156478",
    "fetched_at": "2026-04-23",
    "fetch_mode": "view=full",
    "classifier_version": "v3 (token-based + inferred-DOGMA rule + paper-map hook)",
    "paper_map_loaded": bool(paper_map),
    "paper_map_entries": len(paper_map),
    "n_samples_total": len(samples),
    "classification_summary": {
        "high_medium_confidence_samples": sum(len(v) for v in arms_hm.values()),
        "low_confidence_samples":         sum(len(v) for v in arms_low.values()),
        "ambiguous_or_unknown_samples":   len(unclassified),
    },
    "arms_high_medium_confidence": arms_hm_out,
    "arms_low_confidence":         arms_low_out,
    "samples_unclassified":        sorted(unclassified),
    "always_unresolved_paper_required": [
        "paper_anchored_gsm_to_arm_map (see dogma_gsm_map_manual_2026-04-23.yaml.TEMPLATE)",
        "per_arm_n_cells_post_qc (Mimitou 2021 Figures/Supp Tables)",
        "license_explicit_terms",
        "FASTQ_dbGaP_gating_status",
        "TotalSeq_A_panel_confirmation (expected n=210)",
    ],
    "downstream_usage_rule": (
        "E2-iextended spec cites arms_high_medium_confidence only. With no paper map "
        "loaded, that list is empty for DOGMA — LOW-confidence inferred DOGMA arms "
        "surface in arms_low_confidence and require paper-map resolution before spec "
        "inclusion."
    ),
    "samples": samples,
}

YAML_OUT.write_text("---\n" + yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
JSON_OUT.write_text(json.dumps(samples, indent=2))
reread = yaml.safe_load(YAML_OUT.read_text())
assert reread["schema_version"] == SCHEMA_VERSION
for s in reread["samples"]:
    for field in ("protocol", "permeabilization", "stim", "n_cells_hint"):
        assert {"value","evidence","source_field","confidence"} <= set(s[field].keys())
        assert s[field]["confidence"] in CONFIDENCE
    assert s["n_cells_hint"]["is_authoritative"] is False
print(f"Schema v{SCHEMA_VERSION} — VALIDATED on re-parse.")
print(f"\nClassification summary: {report['classification_summary']}")
print(f"\nHIGH+MEDIUM arms ({len(arms_hm_out)}):")
for a in arms_hm_out:
    print(f"  {a['protocol']:6s} {a['permeabilization']:6s}  n={a['n_samples']:3d}")
print(f"\nLOW arms ({len(arms_low_out)}):")
for a in arms_low_out:
    print(f"  {a['protocol']:10s} {a['permeabilization']:10s}  n={a['n_samples']:3d}")
print(f"\nUnclassified: {len(unclassified)}")

sys.exit(0)
