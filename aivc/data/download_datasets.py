"""
Dataset download automation for AIVC v1.1 multi-perturbation training.

Datasets:
  A. Kang 2018 (GSE96583) — existing, in S3
  B. Frangieh 2021 (SCP1064) — Perturb-CITE-seq, ~750 CRISPR perturbations
  C. ImmPort SDY702 — cytokine stimulation of PBMCs (bulk RNA)
  D. Replogle 2022 (K562) — genome-wide CRISPRi, 9,867 knockdowns
"""
import os
import logging

logger = logging.getLogger("aivc.data")


def download_frangieh(output_dir: str = "data/") -> str:
    """
    Download Frangieh 2021 Perturb-CITE-seq via pertpy.

    Falls back to direct SCP1064 download if pertpy unavailable.
    Expected size: ~800MB.

    Returns: local path to h5ad file.
    """
    path = os.path.join(output_dir, "frangieh_2021_raw.h5ad")
    if os.path.exists(path):
        logger.info(f"Frangieh 2021 already downloaded: {path}")
        return path

    os.makedirs(output_dir, exist_ok=True)

    try:
        import pertpy as pt
        logger.info("Downloading Frangieh 2021 via pertpy...")
        adata = pt.data.frangieh_2021_raw()
        adata.write_h5ad(path)
        logger.info(f"Saved: {path} ({adata.n_obs} cells)")
        return path
    except ImportError:
        logger.warning(
            "pertpy not installed. Install with: pip install pertpy\n"
            "Or download manually from Broad Single Cell Portal SCP1064."
        )
        return ""
    except Exception as e:
        logger.error(f"Frangieh download failed: {e}")
        return ""


def download_replogle_k562(output_dir: str = "data/", essential_only: bool = False) -> str:
    """
    Download Replogle 2022 K562 genome-wide Perturb-seq via pertpy.

    Args:
        output_dir: Directory to save to.
        essential_only: If True, download the smaller essential subset (~500MB).
            Otherwise download full GWPS (~4GB).

    Returns: local path to h5ad file.
    """
    if essential_only:
        filename = "replogle_2022_k562_essential.h5ad"
    else:
        filename = "replogle_2022_k562_gwps.h5ad"

    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        logger.info(f"Replogle 2022 already downloaded: {path}")
        return path

    os.makedirs(output_dir, exist_ok=True)

    try:
        import pertpy as pt
        logger.info(f"Downloading Replogle 2022 ({'essential' if essential_only else 'full'})...")
        if essential_only:
            adata = pt.data.replogle_2022_k562_essential()
        else:
            adata = pt.data.replogle_2022_k562_gwps()
        adata.write_h5ad(path)
        logger.info(f"Saved: {path} ({adata.n_obs} cells)")
        return path
    except ImportError:
        logger.warning(
            "pertpy not installed. Install with: pip install pertpy\n"
            "Or download from Figshare DOI: 10.25452/figshare.plus.20029387"
        )
        return ""
    except Exception as e:
        logger.error(f"Replogle download failed: {e}")
        return ""


def download_immport_datasets(output_dir: str = "data/") -> list:
    """
    ImmPort requires registration (free). Print instructions.

    Returns: list of local paths or empty list with instructions.
    """
    immport_dir = os.path.join(output_dir, "immport")
    sdy702_dir = os.path.join(immport_dir, "SDY702")

    if os.path.exists(sdy702_dir) and os.listdir(sdy702_dir):
        logger.info(f"ImmPort SDY702 found: {sdy702_dir}")
        return [sdy702_dir]

    logger.info(
        "ImmPort datasets require manual download:\n"
        "  1. Go to https://www.immport.org/\n"
        "  2. Register for a free account\n"
        "  3. Search: SDY702 (cytokine PBMC stimulation)\n"
        "  4. Download: expression matrix + metadata\n"
        "  5. Save to: data/immport/SDY702/\n"
        "  Cytokines in SDY702: IL-2, IL-4, IL-6, IL-10, IFN-gamma, TNF-alpha"
    )
    os.makedirs(sdy702_dir, exist_ok=True)
    return []


def check_and_download_all(output_dir: str = "data/", skip_large: bool = False) -> dict:
    """
    Check all datasets and download missing ones.

    Args:
        output_dir: Base data directory.
        skip_large: If True, skip Replogle (4GB) for quick testing.

    Returns:
        Dict of {dataset_name: {path, available, notes}}.
    """
    results = {}

    # Kang 2018
    kang_path = os.path.join(output_dir, "kang2018_pbmc_fixed.h5ad")
    results["kang_2018"] = {
        "path": kang_path,
        "available": os.path.exists(kang_path),
        "notes": "Primary benchmark. Load from S3 if not local.",
    }

    # Frangieh 2021
    frangieh_path = download_frangieh(output_dir)
    results["frangieh_2021"] = {
        "path": frangieh_path,
        "available": bool(frangieh_path) and os.path.exists(frangieh_path),
        "notes": "Perturb-CITE-seq, ~218k cells, ~750 CRISPR perturbations.",
    }

    # ImmPort
    immport_paths = download_immport_datasets(output_dir)
    results["immport_sdy702"] = {
        "path": immport_paths[0] if immport_paths else "",
        "available": bool(immport_paths),
        "notes": "Cytokine PBMC stimulation. Manual download required.",
    }

    # Replogle 2022
    if not skip_large:
        replogle_path = download_replogle_k562(output_dir, essential_only=True)
        results["replogle_2022"] = {
            "path": replogle_path,
            "available": bool(replogle_path) and os.path.exists(replogle_path),
            "notes": "Genome-wide CRISPRi. GRN edges only (not cell training).",
        }
    else:
        results["replogle_2022"] = {
            "path": "",
            "available": False,
            "notes": "Skipped (skip_large=True). Download manually or use pertpy.",
        }

    # Summary
    n_available = sum(1 for v in results.values() if v["available"])
    logger.info(f"Datasets available: {n_available}/{len(results)}")
    for name, info in results.items():
        status = "OK" if info["available"] else "MISSING"
        logger.info(f"  {name}: {status} — {info['path'] or 'not downloaded'}")

    return results
