#!/usr/bin/env python3
"""
QurieGen AIVC — Workflow Runner

Usage:
  python run_workflow.py --workflow rna_baseline_to_demo
  python run_workflow.py --workflow active_learning_loop
  python run_workflow.py --workflow multimodal_integration
  python run_workflow.py --list-skills
  python run_workflow.py --list-workflows
  python run_workflow.py --check-gpu
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="QurieGen AIVC Workflow Runner"
    )
    parser.add_argument(
        "--workflow",
        choices=["rna_baseline_to_demo", "multimodal_integration",
                 "active_learning_loop"],
        help="Workflow to execute",
    )
    parser.add_argument(
        "--list-skills", action="store_true",
        help="List all registered skills",
    )
    parser.add_argument(
        "--list-workflows", action="store_true",
        help="List all available workflows",
    )
    parser.add_argument(
        "--check-gpu", action="store_true",
        help="Check GPU availability and specs",
    )
    parser.add_argument(
        "--budget-usd", type=float, default=50.0,
        help="Maximum GPU budget in USD (default: 50.0)",
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Data directory (default: data/)",
    )
    parser.add_argument(
        "--adata-path", default=None,
        help="Path to AnnData h5ad file",
    )
    parser.add_argument(
        "--string-ppi-path", default=None,
        help="Path to STRING PPI file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import after path setup to ensure proper resolution
    from aivc.registry import registry
    # Trigger skill registration by importing skills package
    import aivc.skills  # noqa: F401
    from aivc.orchestration.workflows import WORKFLOWS
    from aivc.orchestration.orchestrator import AIVCOrchestrator, CriticSuite
    from aivc.critics.statistical import StatisticalCritic
    from aivc.critics.methodological import MethodologicalCritic
    from aivc.critics.biological import BiologicalCritic
    from aivc.context import SessionContext

    if args.list_skills:
        print("\n  QurieGen AIVC — Registered Skills")
        print("  " + "=" * 70)
        skills = registry.list_skills()
        for skill in skills:
            print(
                f"  {skill['name']:40s} "
                f"{skill['domain']:20s} "
                f"{skill['profile']}"
            )
        print(f"\n  Total: {len(skills)} skills registered\n")
        return

    if args.list_workflows:
        print("\n  QurieGen AIVC — Available Workflows")
        print("  " + "=" * 70)
        for name, wf in WORKFLOWS.items():
            print(f"  {name}: {len(wf.steps)} steps")
            print(f"    {wf.description[:80]}...")
            for i, step in enumerate(wf.steps, 1):
                print(f"    {i}. {step.skill}")
            print()
        return

    if args.check_gpu:
        print("\n  QurieGen AIVC — GPU Check")
        print("  " + "=" * 40)
        try:
            import torch
            available = torch.cuda.is_available()
            print(f"  GPU available: {available}")
            if available:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {name}")
                print(f"  Memory: {mem:.1f} GB")

                # Check for TF32 support (H100, A100)
                if "H100" in name or "A100" in name:
                    print(f"  TF32 support: Yes")
                else:
                    print(f"  TF32 support: No")
            else:
                print("  Running on CPU (training will be slower)")
        except ImportError:
            print("  PyTorch not installed")
        print()
        return

    if args.workflow is None:
        parser.print_help()
        return

    # Determine device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    # Build context
    context = SessionContext.create_default(
        data_dir=args.data_dir,
        device=device,
    )

    # Build critics
    critics = CriticSuite(
        statistical=StatisticalCritic(),
        methodological=MethodologicalCritic(),
        biological=BiologicalCritic(),
    )

    # Build orchestrator
    orchestrator = AIVCOrchestrator(
        registry=registry,
        memory=context.memory,
        critics=critics,
        budget_usd=args.budget_usd,
    )

    # Prepare inputs
    inputs = {
        "data_dir": args.data_dir,
    }
    if args.adata_path:
        inputs["adata_path"] = args.adata_path
    else:
        # Default path
        default_path = os.path.join(args.data_dir, "kang2018_pbmc.h5ad")
        if os.path.exists(default_path):
            inputs["adata_path"] = default_path

    if args.string_ppi_path:
        inputs["string_ppi_path"] = args.string_ppi_path
    else:
        # Default path
        for fname in ["edge_list_fixed.csv", "edge_list.csv"]:
            default_path = os.path.join(args.data_dir, fname)
            if os.path.exists(default_path):
                inputs["string_ppi_path"] = default_path
                break

    # Run workflow
    print(f"\n  Running workflow: {args.workflow}")
    print(f"  Device: {device}")
    print(f"  Budget: ${args.budget_usd:.2f}")
    print("  " + "=" * 40)

    result = orchestrator.run_workflow(args.workflow, inputs, context)

    if result.success:
        print(f"\n  Workflow complete!")
        print(f"  Pearson r: {result.pearson_r:.4f}")
        print(f"  GPU cost: ${result.total_cost:.2f}")
        print(f"  Demo ready: {result.demo_ready}")
        if result.output_paths:
            print(f"  Reports:")
            for p in result.output_paths:
                print(f"    {p}")
    else:
        print(f"\n  Workflow failed: {result.error_message}")
        if result.failed_step:
            print(f"  Failed at step: {result.failed_step}")
        if result.diagnostic_report:
            print(f"  Diagnostic: {result.diagnostic_report}")

    print()


if __name__ == "__main__":
    main()
