"""
aivc/orchestration/cost_estimator.py — GPU credit cost estimation.

Estimates compute cost for training and inference based on empirical
benchmarks and GPU pricing.
"""

from aivc.interfaces import ComputeCost, ComputeProfile


GPU_COSTS_PER_HOUR = {
    "lambda_a100": 1.10,    # Lambda Cloud A100
    "nebius_h100": 2.00,    # Nebius H100 SXM5
    "aws_p3_v100": 3.06,    # AWS p3.2xlarge V100
    "aws_p3_8x": 12.24,     # AWS p3.8xlarge 4x V100
    "colab_t4": 0.00,       # Free tier
}

# CPU-to-GPU speedup factors
GPU_SPEEDUP = {
    "lambda_a100": 80.0,
    "nebius_h100": 100.0,
    "aws_p3_v100": 40.0,
    "aws_p3_8x": 80.0,
    "colab_t4": 20.0,
}


def estimate_training_cost(
    n_pairs: int,
    n_epochs: int,
    gpu_type: str = "nebius_h100",
) -> ComputeCost:
    """
    Estimate training cost based on empirical benchmarks.

    Empirical baseline: 60 pairs, 200 epochs = 5 min on CPU.
    H100 is approximately 60x faster than CPU for this workload.
    """
    # CPU time estimate (minutes)
    cpu_minutes = (n_pairs / 60.0) * (n_epochs / 200.0) * 5.0

    # GPU time estimate
    speedup = GPU_SPEEDUP.get(gpu_type, 60.0)
    gpu_minutes = cpu_minutes / speedup

    # Cost estimate
    cost_per_hour = GPU_COSTS_PER_HOUR.get(gpu_type, 2.0)
    estimated_usd = (gpu_minutes / 60.0) * cost_per_hour

    # GPU memory estimate (empirical: ~2 GB base + 0.5 GB per 1000 pairs)
    gpu_memory_gb = 2.0 + (n_pairs / 1000.0) * 0.5

    can_run_on_cpu = cpu_minutes < 30  # feasible on CPU if under 30 min

    return ComputeCost(
        estimated_minutes=gpu_minutes,
        gpu_memory_gb=gpu_memory_gb,
        profile=ComputeProfile.GPU_INTENSIVE,
        estimated_usd=estimated_usd,
        can_run_on_cpu=can_run_on_cpu,
    )


def estimate_inference_cost(
    n_samples: int,
    gpu_type: str = "nebius_h100",
) -> ComputeCost:
    """Estimate inference cost for prediction or evaluation."""
    # Inference is ~10x faster than training per sample
    cpu_minutes = (n_samples / 60.0) * 0.5
    speedup = GPU_SPEEDUP.get(gpu_type, 60.0)
    gpu_minutes = cpu_minutes / speedup
    cost_per_hour = GPU_COSTS_PER_HOUR.get(gpu_type, 2.0)
    estimated_usd = (gpu_minutes / 60.0) * cost_per_hour

    return ComputeCost(
        estimated_minutes=gpu_minutes,
        gpu_memory_gb=2.0,
        profile=ComputeProfile.GPU_REQUIRED,
        estimated_usd=estimated_usd,
        can_run_on_cpu=True,
    )
