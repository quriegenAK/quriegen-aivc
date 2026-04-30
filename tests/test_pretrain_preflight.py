"""Pre-flight smoke tests for the pretrain pipeline.

Exercises the full main() -> save -> load -> main(--resume) cycle.

The kill-and-resume cycle test is the load-bearing integration check
before any 24-h GPU run. Marked @pytest.mark.slow because it invokes
main() twice with full encoder construction on mock data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _mock_args(checkpoint_dir, max_steps=None, resume=None, extra=None):
    """Build argv list for pretrain main() in mock-data mode."""
    argv = [
        "--no_wandb",
        "--steps", "10",
        "--epochs", "5",
        "--n_cells", "50",
        "--n_genes", "100",
        "--n_peaks", "200",
        "--checkpoint_dir", str(checkpoint_dir),
        "--seed", "0",
    ]
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    if resume is not None:
        argv += ["--resume", str(resume)]
    if extra:
        argv += list(extra)
    return argv


def _find_ckpt(checkpoint_dir):
    """Return path to the .pt checkpoint produced by main()."""
    pts = sorted(Path(checkpoint_dir).glob("*.pt"))
    assert len(pts) >= 1, f"No .pt files in {checkpoint_dir}"
    return pts[-1]


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# --- Synthetic tests --------------------------------------------------------

def test_max_steps_terminates_at_exact_count(tmp_path):
    """--max-steps N -> resume_state.global_step == N."""
    from pretrain_multiome import main
    main(_mock_args(tmp_path, max_steps=4))

    ckpt = _load_ckpt(_find_ckpt(tmp_path))
    assert "resume_state" in ckpt
    assert ckpt["resume_state"]["global_step"] == 4


def test_max_steps_supersedes_epochs_steps_product(tmp_path):
    """--epochs 5 --steps 10 (would give 50 iters) but --max-steps 3 caps at 3."""
    from pretrain_multiome import main
    main(_mock_args(tmp_path, max_steps=3))
    ckpt = _load_ckpt(_find_ckpt(tmp_path))
    # 5 * 10 = 50 implied; max-steps caps at 3
    assert ckpt["resume_state"]["global_step"] == 3


def test_max_steps_zero_or_negative_raises(tmp_path):
    """--max-steps 0 or negative is invalid."""
    from pretrain_multiome import main
    with pytest.raises(ValueError, match=r"max-steps must be positive"):
        main(_mock_args(tmp_path, max_steps=0))


@pytest.mark.slow
def test_kill_and_resume_cycle_advances_global_step(tmp_path):
    """LOAD-BEARING: full save->load->save cycle through main().

    Run 1: --max-steps 4 -> saves ckpt with gstep=4.
    Run 2: --resume <ckpt> --max-steps 8 -> loads, runs 4 more, saves with gstep=8.

    Verifies:
    - First ckpt has resume_state.global_step == 4
    - Second ckpt has resume_state.global_step == 8 (not 4 - resume actually advanced)
    - Optimizer state changed between the two checkpoints (training continued)
    """
    from pretrain_multiome import main

    # Run 1: kill at step 4
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    main(_mock_args(run1_dir, max_steps=4))
    ckpt_a_path = _find_ckpt(run1_dir)
    ckpt_a = _load_ckpt(ckpt_a_path)
    assert ckpt_a["resume_state"]["global_step"] == 4
    saved_optim_a = ckpt_a["resume_state"]["optimizer"]

    # Run 2: resume + advance to step 8
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    main(_mock_args(run2_dir, max_steps=8, resume=ckpt_a_path))
    ckpt_b_path = _find_ckpt(run2_dir)
    ckpt_b = _load_ckpt(ckpt_b_path)
    assert ckpt_b["resume_state"]["global_step"] == 8, \
        f"Resume failed to advance gstep: {ckpt_b['resume_state']['global_step']}"

    # Sanity: optimizer state evolved (4 more steps ran)
    saved_optim_b = ckpt_b["resume_state"]["optimizer"]
    # Compare a representative tensor in optimizer state - should differ
    if "state" in saved_optim_a and len(saved_optim_a["state"]) > 0:
        # Adam-style state has 'exp_avg' tensors per param
        first_param_state_a = next(iter(saved_optim_a["state"].values()))
        first_param_state_b = next(iter(saved_optim_b["state"].values()))
        if "exp_avg" in first_param_state_a:
            ea_a = first_param_state_a["exp_avg"]
            ea_b = first_param_state_b["exp_avg"]
            assert not torch.allclose(ea_a, ea_b, atol=1e-7), \
                "Optimizer exp_avg unchanged across resume - training did not continue"


# --- Real-data smoke (single-arm LLL for fast load) -------------------------

@pytest.mark.real_data
@pytest.mark.slow
def test_real_data_kill_and_resume_lll(tmp_path):
    """Kill-and-resume against actual LLL union h5ad. Single-arm for speed
    (joint loader loads ~30k cells; lll alone is ~13.7k -> ~2.3x faster)."""
    h5ad = Path("data/phase6_5g_2/dogma_h5ads/dogma_lll_union.h5ad")
    if not h5ad.exists():
        pytest.skip(f"DOGMA LLL union h5ad missing: {h5ad}")

    from pretrain_multiome import main

    common = [
        "--config", "configs/dogma_pretrain.yaml",
        "--arm", "lll",
        "--no_wandb",
        "--steps", "10",
        "--epochs", "1",
        "--seed", "0",
    ]

    # Run 1: max-steps=2
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    main(common + ["--checkpoint_dir", str(run1_dir), "--max-steps", "2"])
    ckpt_a = _find_ckpt(run1_dir)
    ca = _load_ckpt(ckpt_a)
    assert ca["resume_state"]["global_step"] == 2
    # Trimodal: protein_encoder must be in ckpt
    assert "protein_encoder" in ca

    # Run 2: resume + extend to 4
    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    main(common + ["--checkpoint_dir", str(run2_dir),
                   "--resume", str(ckpt_a), "--max-steps", "4"])
    ckpt_b = _find_ckpt(run2_dir)
    cb = _load_ckpt(ckpt_b)
    assert cb["resume_state"]["global_step"] == 4
    assert "protein_encoder" in cb

    print(f"\nLLL kill-and-resume cycle OK: gstep 2 -> 4, "
          f"protein_encoder + resume_state preserved")
