"""B-cell weakness diagnosis — Stage 3 Part 1 Report 5.

Test 1.5 showed B cells at 4/22 = 0.18 accuracy after noise-centroid
exclusion that brought overall accuracy to 0.7308. Three diagnostic
hypotheses to disambiguate:

  H1: Pseudo-bulk artifact — small B-cell training count (613) produces
      a noisy centroid relative to CD4_T (8360 cells)
  H2: Training data imbalance — SupCon's per-class loss weighting
      underweighted B; encoder didn't learn B-specific features as well
  H3: Genuine encoder weakness on B-lineage chromatin — possibly due to
      Mimitou DOGMA-LLL stim conditions producing B chromatin states
      distinct from Calderon's bulk B label heterogeneity

Computes:
  D1: Split-half centroid cosine similarity per class (tests H1)
       — if B's split-half similarity is meaningfully lower than CD4_T's,
         the centroid is internally inconsistent → H1 supported
  D2: Per-class silhouette score in encoder latent space (tests H2)
       — if B has lower silhouette than CD4_T even on raw cell-level
         embeddings, encoder under-resolved B → H2 supported
  D3: Per-Calderon-B-subtype prediction analysis (tests H3)
       — read prior Test 1.5 result JSON; check which Calderon B
         samples succeeded (4/22) vs failed (18/22), look for subtype
         pattern

Usage:
    python scripts/probe_b_cell_diagnosis.py \\
        --dogma_labeled data/phase6_5g_2/dogma_h5ads/dogma_lll_union_labeled.h5ad \\
        --encoder_ckpt /gpfs/scratch/.../pretrain_encoders.pt \\
        --test1_5_results_json /gpfs/scratch/.../calderon_eval/pretrain_encoders_pseudobulk_test1_excl_other_other_T.json \\
        --calderon data/calderon2019/calderon_atac_hg38.h5ad \\
        --out results/b_cell_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def split_half_centroid_similarity(
    atac_matrix, labels, encoder, device="cpu", seed=0
):
    """For each class, randomly split cells in half, encode each half's
    pseudo-bulk centroid, compute cosine sim between the two halves.
    High sim = internally consistent class; low sim = noisy centroid (H1).
    """
    from aivc.eval.calderon_probe import encode_samples
    rng = np.random.RandomState(seed)
    classes = sorted(set(labels))
    results = {}
    for cls in classes:
        idx = np.where(labels == cls)[0]
        n = len(idx)
        if n < 10:
            results[cls] = {"n": int(n), "split_half_sim": None,
                            "note": "too few cells for split"}
            continue
        rng.shuffle(idx)
        half_a = idx[: n // 2]
        half_b = idx[n // 2 : 2 * (n // 2)]
        bulk_a = np.asarray(atac_matrix[half_a].sum(axis=0)).ravel()
        bulk_b = np.asarray(atac_matrix[half_b].sum(axis=0)).ravel()
        # Encode both pseudo-bulk profiles
        z = encode_samples(
            sp.csr_matrix(np.vstack([bulk_a, bulk_b])),
            encoder, batch_size=2, device=device,
        )
        # Cosine similarity between half_a and half_b centroid embeddings
        norm_a = np.linalg.norm(z[0]) + 1e-12
        norm_b = np.linalg.norm(z[1]) + 1e-12
        sim = float(np.dot(z[0], z[1]) / (norm_a * norm_b))
        results[cls] = {"n": int(n), "n_half": int(n // 2),
                        "split_half_sim": sim}
    return results


def per_class_silhouette(
    atac_matrix, labels, encoder, device="cpu",
    sample_size=500, seed=0,
):
    """Sample up to N cells per class, encode them individually (cell-level,
    not pseudo-bulk), compute silhouette score per class. High silhouette
    = cells of this class cluster tightly and are well-separated from
    other classes. Low silhouette = encoder under-resolves this class (H2).
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    from aivc.eval.calderon_probe import encode_samples
    rng = np.random.RandomState(seed)
    classes = sorted(set(labels))

    # Sample cells per class
    sampled_idx = []
    sampled_labels = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        n_take = min(sample_size, len(idx))
        chosen = rng.choice(idx, size=n_take, replace=False)
        sampled_idx.extend(chosen.tolist())
        sampled_labels.extend([cls] * n_take)
    sampled_idx = np.asarray(sampled_idx)
    sampled_labels = np.asarray(sampled_labels)

    # Encode at cell level (each cell as its own pseudo-bulk row)
    print(f"  encoding {len(sampled_idx)} sampled cells...")
    z = encode_samples(
        atac_matrix[sampled_idx], encoder,
        batch_size=64, device=device,
    )

    # Per-class silhouette (mean over each class's samples)
    sample_silhouettes = silhouette_samples(z, sampled_labels, metric="cosine")
    per_class = {}
    for cls in classes:
        mask = sampled_labels == cls
        per_class[cls] = {
            "n_sampled": int(mask.sum()),
            "mean_silhouette": float(sample_silhouettes[mask].mean()),
            "median_silhouette": float(np.median(sample_silhouettes[mask])),
        }
    overall = float(silhouette_score(z, sampled_labels, metric="cosine"))
    return {"overall_silhouette": overall, "per_class": per_class}


def calderon_b_subtype_analysis(test1_5_json_path, calderon_h5ad_path):
    """From the Test 1.5 result JSON + Calderon obs, identify which
    specific B-subtype labels in Calderon got correctly classified
    (kept after lineage map AND argmax → B centroid) vs failed.

    Tests H3: if the 4/22 successes are all e.g. Plasmablasts, and the
    18/22 failures are all naive_B / memory_B, the encoder learned a
    narrow B signature.
    """
    if not Path(test1_5_json_path).exists():
        return {"error": f"Test 1.5 result JSON not found at {test1_5_json_path}"}

    with open(test1_5_json_path) as f:
        t15 = json.load(f)

    pred_labels = t15.get("predicted_centroid_labels", [])
    cal_lineage = t15.get("calderon_kept_lineage_labels", [])
    cal_25 = t15.get("calderon_kept_25class_labels", [])
    if not (pred_labels and cal_lineage and cal_25):
        return {"error": "Test 1.5 JSON missing expected keys "
                         "(predicted_centroid_labels, calderon_kept_lineage_labels, "
                         "calderon_kept_25class_labels)"}

    # B-only subset
    b_mask = np.array([ln == "B" for ln in cal_lineage])
    b_25 = np.array(cal_25)[b_mask]
    b_pred = np.array(pred_labels)[b_mask]
    correct = b_pred == "B"

    # Per-subtype breakdown
    per_subtype = {}
    for subtype in sorted(set(b_25)):
        s_mask = b_25 == subtype
        n = int(s_mask.sum())
        n_correct = int((s_mask & correct).sum())
        # What did wrong predictions go to?
        wrong_pred_dist = pd.Series(b_pred[s_mask & ~correct]).value_counts().to_dict()
        per_subtype[subtype] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n > 0 else None,
            "wrong_predictions_distribution": wrong_pred_dist,
        }

    return {
        "n_total_b": int(b_mask.sum()),
        "n_correct_b": int(correct.sum()),
        "overall_b_accuracy": float(correct.mean()) if len(correct) > 0 else None,
        "per_subtype": per_subtype,
    }


def diagnose(d1_results, d2_results, d3_results, b_class="B", reference_class="CD4_T"):
    """Apply decision rules to per-hypothesis evidence to land a verdict."""
    h1_score = 0  # 0 = not supported, 1 = weakly, 2 = strongly
    h2_score = 0
    h3_score = 0
    notes = []

    # H1: split-half similarity
    b_sim = d1_results.get(b_class, {}).get("split_half_sim")
    ref_sim = d1_results.get(reference_class, {}).get("split_half_sim")
    if b_sim is not None and ref_sim is not None:
        gap = ref_sim - b_sim
        if gap > 0.10:
            h1_score = 2
            notes.append(f"H1 STRONG: B split-half sim {b_sim:.3f} vs {reference_class} {ref_sim:.3f} "
                         f"(gap {gap:.3f}). B centroid is internally inconsistent.")
        elif gap > 0.03:
            h1_score = 1
            notes.append(f"H1 WEAK: B split-half sim {b_sim:.3f} vs {reference_class} {ref_sim:.3f} "
                         f"(gap {gap:.3f}).")
        else:
            notes.append(f"H1 NOT SUPPORTED: B split-half sim {b_sim:.3f} comparable to "
                         f"{reference_class} {ref_sim:.3f}.")

    # H2: silhouette
    b_sil = d2_results.get("per_class", {}).get(b_class, {}).get("mean_silhouette")
    ref_sil = d2_results.get("per_class", {}).get(reference_class, {}).get("mean_silhouette")
    if b_sil is not None and ref_sil is not None:
        gap = ref_sil - b_sil
        if gap > 0.15:
            h2_score = 2
            notes.append(f"H2 STRONG: B silhouette {b_sil:.3f} vs {reference_class} {ref_sil:.3f} "
                         f"(gap {gap:.3f}). Encoder under-resolves B at cell level.")
        elif gap > 0.05:
            h2_score = 1
            notes.append(f"H2 WEAK: B silhouette {b_sil:.3f} vs {reference_class} {ref_sil:.3f} "
                         f"(gap {gap:.3f}).")
        else:
            notes.append(f"H2 NOT SUPPORTED: B silhouette {b_sil:.3f} comparable to "
                         f"{reference_class} {ref_sil:.3f}.")

    # H3: subtype heterogeneity
    if not d3_results.get("error"):
        per_subtype = d3_results.get("per_subtype", {})
        accs = [s.get("accuracy") for s in per_subtype.values() if s.get("accuracy") is not None]
        if accs:
            spread = max(accs) - min(accs)
            if spread > 0.40:
                h3_score = 2
                notes.append(f"H3 STRONG: per-subtype B accuracy spread {spread:.2f} "
                             f"(some subtypes succeed, others fail). Narrow B signature learned.")
            elif spread > 0.15:
                h3_score = 1
                notes.append(f"H3 WEAK: per-subtype B accuracy spread {spread:.2f}.")
            else:
                notes.append(f"H3 NOT SUPPORTED: B accuracy uniform across subtypes "
                             f"(spread {spread:.2f}); failure mode is class-wide, not subtype-specific.")

    # Verdict
    dominant = max([("H1", h1_score), ("H2", h2_score), ("H3", h3_score)],
                   key=lambda x: x[1])
    verdict = {
        "h1_score": h1_score,
        "h2_score": h2_score,
        "h3_score": h3_score,
        "dominant_hypothesis": dominant[0] if dominant[1] > 0 else "NONE / unclear",
        "notes": notes,
    }

    # Remediation recommendation
    if dominant[0] == "H1" and dominant[1] >= 1:
        verdict["recommendation"] = (
            "Split B into multiple sub-centroids (B_naive, B_memory, "
            "B_plasmablast) using external B-cell atlas. ~1 day."
        )
    elif dominant[0] == "H2" and dominant[1] >= 1:
        verdict["recommendation"] = (
            "Class-balanced fine-tune of last encoder layer with "
            "focal-loss SupCon term. ~1-2 days. Defer until BTK+JAK-analog "
            "demo (CD3E+CD4) fails specifically on B-related conditions."
        )
    elif dominant[0] == "H3" and dominant[1] >= 1:
        verdict["recommendation"] = (
            "Pretrain DOGMA on B-cell-rich corpus (HCA Bone Marrow, "
            "OneK1K). ~1-2 weeks. Defer to phase after Stage 3."
        )
    else:
        verdict["recommendation"] = (
            "B-cell weakness is mild or class-wide-uniform; not blocking "
            "Stage 3a critical path. Bank as known limitation."
        )
    return verdict


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dogma_labeled", required=True, type=Path)
    p.add_argument("--encoder_ckpt", required=True, type=Path)
    p.add_argument("--test1_5_results_json", required=True, type=Path,
                   help="JSON output of prior pseudobulk test 1.5 run")
    p.add_argument("--calderon", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from aivc.eval.calderon_probe import load_atac_encoder_from_ckpt

    # --- Load DOGMA labeled h5ad ---
    print(f"Loading DOGMA labeled h5ad: {args.dogma_labeled}")
    dogma = ad.read_h5ad(args.dogma_labeled)
    if "atac_peaks" not in dogma.obsm:
        raise ValueError("obsm['atac_peaks'] missing")
    atac = dogma.obsm["atac_peaks"]
    if not sp.issparse(atac):
        atac = sp.csr_matrix(atac)
    labels = dogma.obs["cell_type"].astype(str).values
    classes_with_real_lineage = ["B", "CD4_T", "CD8_T", "DC", "Monocyte", "NK"]
    keep_mask = np.isin(labels, classes_with_real_lineage)
    atac = atac[keep_mask]
    labels = labels[keep_mask]
    print(f"  {len(labels)} cells across {len(set(labels))} real-lineage classes")

    # --- Load encoder ---
    print(f"\nLoading encoder: {args.encoder_ckpt}")
    encoder, _ = load_atac_encoder_from_ckpt(
        args.encoder_ckpt,
        expected_n_peaks=atac.shape[1],
        map_location=args.device,
    )

    # --- D1: split-half centroid similarity ---
    print("\n=== D1: split-half centroid cosine similarity (tests H1) ===")
    d1 = split_half_centroid_similarity(
        atac, labels, encoder, device=args.device, seed=args.seed,
    )
    for cls, r in d1.items():
        sim = r.get("split_half_sim")
        print(f"  {cls:<12s}: n={r['n']:>5d}  split-half-sim={sim:.4f}" if sim is not None
              else f"  {cls:<12s}: n={r['n']:>5d}  ({r.get('note')})")

    # --- D2: per-class silhouette ---
    print("\n=== D2: per-class silhouette in encoder latent (tests H2) ===")
    d2 = per_class_silhouette(
        atac, labels, encoder, device=args.device, sample_size=500, seed=args.seed,
    )
    print(f"  overall silhouette: {d2['overall_silhouette']:.4f}")
    for cls, r in d2["per_class"].items():
        print(f"  {cls:<12s}: n={r['n_sampled']:>4d}  silhouette={r['mean_silhouette']:.4f}  "
              f"(median={r['median_silhouette']:.4f})")

    # --- D3: Calderon B-subtype analysis ---
    print("\n=== D3: per-Calderon-B-subtype prediction analysis (tests H3) ===")
    d3 = calderon_b_subtype_analysis(args.test1_5_results_json, args.calderon)
    if "error" in d3:
        print(f"  {d3['error']}")
    else:
        print(f"  {d3['n_correct_b']}/{d3['n_total_b']} = {d3['overall_b_accuracy']:.4f} overall B accuracy")
        for subtype, r in d3["per_subtype"].items():
            acc = r["accuracy"]
            print(f"  {subtype:<24s}: n={r['n']:>3d}  acc={acc:.4f}  "
                  f"wrongs→{r['wrong_predictions_distribution']}" if acc is not None
                  else f"  {subtype:<24s}: n={r['n']:>3d}")

    # --- Verdict ---
    print("\n=== Verdict ===")
    verdict = diagnose(d1, d2, d3, b_class="B", reference_class="CD4_T")
    print(f"  Dominant hypothesis: {verdict['dominant_hypothesis']}")
    for note in verdict["notes"]:
        print(f"  - {note}")
    print(f"\n  Recommendation: {verdict['recommendation']}")

    # --- Write JSON ---
    out_payload = {
        "test": "b_cell_diagnosis",
        "ckpt_path": str(args.encoder_ckpt),
        "dogma_labeled_path": str(args.dogma_labeled),
        "d1_split_half_centroid_sim": d1,
        "d2_per_class_silhouette": d2,
        "d3_calderon_b_subtype_analysis": d3,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
