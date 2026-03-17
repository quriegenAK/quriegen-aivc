"""
aivc/skills/gat_training.py — GAT model training orchestration skill.

Wraps train_week3.py. Three-phase GAT training with perturbation
and cell-type embeddings. Saves checkpoints, applies gradient clipping,
triggers early stopping, enables TF32 if H100 detected.
"""

import time
import os
import sys
import numpy as np

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import registry


@registry.register(
    name="gat_trainer",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["X_ctrl_ot", "X_stim_ot", "edge_index",
              "cell_type_ot", "donor_ot"],
    compute_profile=ComputeProfile.GPU_INTENSIVE,
)
class GATTrainer(AIVCSkill):
    """
    GAT training with perturbation and cell-type embeddings.
    Wraps train_week3.py.
    Saves checkpoint every 50 epochs.
    Applies gradient clipping max_norm=1.0 throughout.
    Triggers early stopping if val r does not improve for 30 epochs.
    Enables TF32 if H100 is detected.
    """

    def execute(self, inputs: dict, context) -> SkillResult:
        self._check_inputs(inputs, [
            "X_ctrl_ot", "X_stim_ot", "edge_index",
            "cell_type_ot", "donor_ot",
        ])
        t0 = time.time()
        warnings = []
        errors = []

        n_epochs = inputs.get("n_epochs", 200)
        batch_size = inputs.get("batch_size", 8)
        lfc_beta = inputs.get("lfc_beta", 0.01)
        checkpoint_dir = inputs.get("checkpoint_dir", "models/")

        try:
            import torch

            # Enable TF32 if H100 detected
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if "H100" in gpu_name:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    warnings.append(f"TF32 enabled for {gpu_name}")

            device = context.device if hasattr(context, "device") else "cpu"

            # Set random seeds
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            # Import model components from existing codebase
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
            from perturbation_model import PerturbationResponseModel
            from losses import combined_loss

            X_ctrl = inputs["X_ctrl_ot"]
            X_stim = inputs["X_stim_ot"]
            edge_index = inputs["edge_index"]
            cell_type_ot = inputs["cell_type_ot"]
            donor_ot = inputs["donor_ot"]

            n_genes = X_ctrl.shape[1]

            # Determine unique cell types and donors
            unique_ct = sorted(set(cell_type_ot))
            ct_to_idx = {ct: i for i, ct in enumerate(unique_ct)}
            n_cell_types = len(unique_ct)

            unique_donors = sorted(set(donor_ot))

            # Donor-based split: 60% train, 20% val, 20% test
            n_train = max(1, int(len(unique_donors) * 0.6))
            n_val = max(1, int(len(unique_donors) * 0.2))
            train_donors = set(unique_donors[:n_train])
            val_donors = set(unique_donors[n_train:n_train + n_val])
            test_donors = set(unique_donors[n_train + n_val:])

            # Lock test donors in session memory
            if hasattr(context, "memory") and hasattr(context.memory, "session"):
                context.memory.session.lock_test_donors(test_donors)

            # Split data
            train_mask = np.array([d in train_donors for d in donor_ot])
            val_mask = np.array([d in val_donors for d in donor_ot])
            test_mask = np.array([d in test_donors for d in donor_ot])

            # Convert to tensors
            X_ctrl_t = torch.tensor(X_ctrl, dtype=torch.float32).to(device)
            X_stim_t = torch.tensor(X_stim, dtype=torch.float32).to(device)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.to(device)

            ct_ids = torch.tensor(
                [ct_to_idx[ct] for ct in cell_type_ot], dtype=torch.long
            ).to(device)

            # Build model
            model = PerturbationResponseModel(
                n_genes=n_genes,
                n_cell_types=n_cell_types,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-6
            )

            # Training loop
            best_val_r = -1.0
            best_epoch = 0
            epoch_of_best = 0
            early_stop_patience = 30
            early_stopped = False
            first_loss = None
            final_loss = None
            training_log = []

            os.makedirs(checkpoint_dir, exist_ok=True)

            LFC_START = 30

            for epoch in range(1, n_epochs + 1):
                model.train()

                # Mini-batch training on train split
                train_idx = np.where(train_mask)[0]
                np.random.shuffle(train_idx)

                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(train_idx), batch_size):
                    batch_idx = train_idx[start:start + batch_size]
                    ctrl_b = X_ctrl_t[batch_idx]
                    stim_b = X_stim_t[batch_idx]
                    ct_b = ct_ids[batch_idx]

                    pert_id = torch.ones(
                        len(batch_idx), dtype=torch.long, device=device
                    )

                    # Forward pass (residual learning)
                    predicted_delta = model.forward_batch(
                        ctrl_b, edge_index, pert_id, ct_b
                    )
                    predicted = (ctrl_b + predicted_delta).clamp(min=0.0)

                    # Compute loss
                    alpha = 1.0
                    beta = lfc_beta if epoch >= LFC_START else 0.0
                    gamma = 0.1

                    loss = combined_loss(
                        predicted, stim_b, ctrl_b,
                        alpha=alpha, beta=beta, gamma=gamma,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                avg_loss = epoch_loss / max(n_batches, 1)
                if first_loss is None:
                    first_loss = avg_loss
                final_loss = avg_loss

                # Validation
                model.eval()
                with torch.no_grad():
                    val_idx = np.where(val_mask)[0]
                    if len(val_idx) > 0:
                        ctrl_v = X_ctrl_t[val_idx]
                        stim_v = X_stim_t[val_idx]
                        ct_v = ct_ids[val_idx]
                        pert_v = torch.ones(
                            len(val_idx), dtype=torch.long, device=device
                        )

                        pred_delta_v = model.forward_batch(
                            ctrl_v, edge_index, pert_v, ct_v
                        )
                        pred_v = (ctrl_v + pred_delta_v).clamp(min=0.0)

                        # Compute Pearson r on raw values
                        pred_np = pred_v.cpu().numpy().flatten()
                        stim_np = stim_v.cpu().numpy().flatten()
                        if pred_np.std() > 0 and stim_np.std() > 0:
                            val_r = float(np.corrcoef(pred_np, stim_np)[0, 1])
                        else:
                            val_r = 0.0
                    else:
                        val_r = 0.0

                training_log.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "val_r": val_r,
                })

                if val_r > best_val_r:
                    best_val_r = val_r
                    epoch_of_best = epoch
                    # Save best model
                    model_path = os.path.join(checkpoint_dir, "model_best.pt")
                    torch.save(model.state_dict(), model_path)

                # Checkpoint every 50 epochs
                if epoch % 50 == 0:
                    ckpt_path = os.path.join(
                        checkpoint_dir, f"model_epoch_{epoch}.pt"
                    )
                    torch.save(model.state_dict(), ckpt_path)

                # Early stopping (after LFC phase starts)
                if epoch >= LFC_START:
                    if epoch - epoch_of_best >= early_stop_patience:
                        early_stopped = True
                        warnings.append(
                            f"Early stopped at epoch {epoch}. "
                            f"Best val r = {best_val_r:.4f} at epoch {epoch_of_best}."
                        )
                        break

            # Save final model
            final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
            torch.save(model.state_dict(), final_model_path)

            # Save training log
            log_path = os.path.join(checkpoint_dir, "training_log.txt")
            with open(log_path, "w") as f:
                f.write("epoch,loss,val_r\n")
                for entry in training_log:
                    f.write(
                        f"{entry['epoch']},{entry['loss']:.6f},"
                        f"{entry['val_r']:.4f}\n"
                    )

            elapsed = time.time() - t0
            return SkillResult(
                skill_name=self.name,
                version=self.version,
                success=True,
                outputs={
                    "model_path": os.path.join(checkpoint_dir, "model_best.pt"),
                    "best_val_pearson_r": best_val_r,
                    "final_loss": final_loss,
                    "final_loss_breakdown": {
                        "first_epoch_loss": first_loss,
                        "final_epoch_loss": final_loss,
                    },
                    "training_log_path": log_path,
                    "epoch_of_best": epoch_of_best,
                    "early_stopped": early_stopped,
                    "train_donors": list(train_donors),
                    "val_donors": list(val_donors),
                    "test_donors": list(test_donors),
                    "n_genes": n_genes,
                    "n_cell_types": n_cell_types,
                    "ct_to_idx": ct_to_idx,
                },
                metadata={
                    "elapsed_seconds": elapsed,
                    "n_epochs_run": len(training_log),
                    "batch_size": batch_size,
                    "lfc_beta": lfc_beta,
                    "device": device,
                    "random_seed_used": 42,
                    "split_method": "donor_based",
                    "train_donors": list(train_donors),
                    "test_donors": list(test_donors),
                },
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Training failed: {str(e)}")
            return SkillResult(
                skill_name=self.name, version=self.version,
                success=False, outputs={},
                metadata={"elapsed_seconds": time.time() - t0},
                warnings=warnings, errors=errors,
            )

    def validate(self, result: SkillResult) -> ValidationReport:
        checks_passed = []
        checks_failed = []

        if not result.success:
            checks_failed.append(f"Execution failed: {result.errors}")
            return ValidationReport(
                passed=False, critic_name="GATTrainer.validate",
                checks_passed=checks_passed, checks_failed=checks_failed,
            )

        outputs = result.outputs

        # HARD GATE: if best_val_pearson_r < 0.70 -> FAIL
        val_r = outputs.get("best_val_pearson_r", 0)
        if val_r >= 0.70:
            checks_passed.append(
                f"Validation Pearson r sufficient: {val_r:.4f}"
            )
        else:
            checks_failed.append(
                f"HARD GATE: Validation Pearson r = {val_r:.4f} < 0.70. "
                "Model performance insufficient for downstream use."
            )

        # Check: loss went down overall
        breakdown = outputs.get("final_loss_breakdown", {})
        first = breakdown.get("first_epoch_loss")
        final = breakdown.get("final_epoch_loss")
        if first is not None and final is not None:
            if final < first:
                checks_passed.append(
                    f"Loss decreased: {first:.6f} -> {final:.6f}"
                )
            else:
                checks_failed.append(
                    f"Loss did not decrease: {first:.6f} -> {final:.6f}"
                )

        # Check: no NaN in final loss
        final_loss = outputs.get("final_loss")
        if final_loss is not None:
            if np.isfinite(final_loss):
                checks_passed.append(f"Final loss finite: {final_loss:.6f}")
            else:
                checks_failed.append(f"Final loss is NaN/Inf: {final_loss}")

        # Check: checkpoint file exists
        model_path = outputs.get("model_path", "")
        if os.path.exists(model_path):
            checks_passed.append(f"Checkpoint exists: {model_path}")
        else:
            checks_failed.append(f"Checkpoint missing: {model_path}")

        recommendation = None
        if outputs.get("early_stopped"):
            recommendation = (
                f"Training early-stopped at epoch "
                f"{result.metadata.get('n_epochs_run', '?')}. "
                f"Best val r = {val_r:.4f} at epoch "
                f"{outputs.get('epoch_of_best', '?')}."
            )

        return ValidationReport(
            passed=len(checks_failed) == 0,
            critic_name="GATTrainer.validate",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            recommendation=recommendation,
        )

    def estimate_cost(self, inputs: dict) -> ComputeCost:
        n_pairs = inputs.get("n_pairs", 60)
        n_epochs = inputs.get("n_epochs", 200)

        # Empirical: 60 pairs, 200 epochs = 5 min on CPU
        # H100 is approximately 60x faster than CPU
        cpu_minutes = (n_pairs / 60.0) * (n_epochs / 200.0) * 5.0
        gpu_minutes = cpu_minutes / 60.0

        # Based on $2/hr H100
        estimated_usd = (gpu_minutes / 60.0) * 2.0

        return ComputeCost(
            estimated_minutes=gpu_minutes,
            gpu_memory_gb=4.0,
            profile=ComputeProfile.GPU_INTENSIVE,
            estimated_usd=estimated_usd,
            can_run_on_cpu=True,  # slow but possible
        )
