"""
aivc/orchestration/workflows.py — Named end-to-end workflows.

Defines three named workflows:
  1. rna_baseline_to_demo — Full pipeline from raw scRNA to demo-ready output
  2. multimodal_integration — Phase 2 QuRIE-seq integration (partial placeholder)
  3. active_learning_loop — Uncertainty-driven experiment recommendation
"""

from dataclasses import dataclass, field


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    skill: str
    input_mapping: dict = field(default_factory=dict)
    # input_mapping: maps skill input names to either:
    #   - literal keys in the global inputs dict
    #   - "skill_name.output_key" to reference previous skill outputs

    def prepare_inputs(self, global_inputs: dict,
                       results: dict) -> dict:
        """
        Prepare inputs for this step from global inputs and
        previous step results.
        """
        step_inputs = dict(global_inputs)  # start with all globals

        # Add outputs from previous steps
        for skill_name, result in results.items():
            if hasattr(result, "outputs"):
                for key, value in result.outputs.items():
                    step_inputs[key] = value
                # Also expose the result itself
                step_inputs[f"{skill_name}_result"] = result

        # Apply explicit mappings
        for target_key, source_key in self.input_mapping.items():
            if "." in source_key:
                skill_name, output_key = source_key.split(".", 1)
                if skill_name in results:
                    result = results[skill_name]
                    if hasattr(result, "outputs"):
                        if output_key in result.outputs:
                            step_inputs[target_key] = (
                                result.outputs[output_key]
                            )
            elif source_key in step_inputs:
                step_inputs[target_key] = step_inputs[source_key]

        return step_inputs


@dataclass
class Workflow:
    """A named workflow with ordered steps."""
    name: str
    description: str
    steps: list[WorkflowStep]

    def get_steps(self, inputs: dict) -> list[WorkflowStep]:
        """Return the ordered list of steps for this workflow."""
        return self.steps


# ─── WORKFLOW 1: RNA Baseline to Demo ───

WORKFLOW_1 = Workflow(
    name="rna_baseline_to_demo",
    description=(
        "Full pipeline from raw scRNA-seq to demo-ready output. "
        "9 steps: preprocessing, graph building, OT pairing, "
        "GAT training, uncertainty estimation, benchmark evaluation, "
        "attention extraction, biological plausibility scoring, "
        "two-audience rendering."
    ),
    steps=[
        WorkflowStep(
            skill="scRNA_preprocessor",
            input_mapping={
                "adata_path": "adata_path",
            },
        ),
        WorkflowStep(
            skill="graph_builder",
            input_mapping={
                "gene_to_idx": "gene_to_idx",
                "string_ppi_path": "string_ppi_path",
            },
        ),
        WorkflowStep(
            skill="ot_cell_pairer",
            input_mapping={
                "adata_path": "adata_path",
                "gene_to_idx": "gene_to_idx",
            },
        ),
        WorkflowStep(
            skill="gat_trainer",
            input_mapping={
                "X_ctrl_ot": "X_ctrl_ot",
                "X_stim_ot": "X_stim_ot",
                "edge_index": "edge_index",
                "cell_type_ot": "cell_type_ot",
                "donor_ot": "donor_ot",
            },
        ),
        WorkflowStep(
            skill="uncertainty_estimator",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl": "X_ctrl_ot",
                "edge_index": "edge_index",
            },
        ),
        WorkflowStep(
            skill="benchmark_evaluator",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl_test": "X_ctrl_ot",  # uses test split internally
                "X_stim_test": "X_stim_ot",
                "edge_index": "edge_index",
                "cell_type_test": "cell_type_ot",
            },
        ),
        WorkflowStep(
            skill="attention_extractor",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl": "X_ctrl_ot",
                "edge_index": "edge_index",
                "gene_to_idx": "gene_to_idx",
            },
        ),
        WorkflowStep(
            skill="biological_plausibility_scorer",
            input_mapping={
                "predicted_interactions": "predicted_interactions",
                "gene_to_idx": "gene_to_idx",
            },
        ),
        WorkflowStep(
            skill="two_audience_renderer",
            input_mapping={
                "evaluation_result": "benchmark_evaluator_result",
                "attention_result": "attention_extractor_result",
                "uncertainty_result": "uncertainty_estimator_result",
                "plausibility_result": "biological_plausibility_scorer_result",
            },
        ),
    ],
)


# ─── WORKFLOW 2: Multimodal Integration ───

WORKFLOW_2 = Workflow(
    name="multimodal_integration",
    description=(
        "Phase 2 QuRIE-seq integration pipeline. "
        "Steps 2-3 are PLACEHOLDER skills — available May 2026 "
        "when QuRIE-seq data arrives."
    ),
    steps=[
        WorkflowStep(
            skill="scRNA_preprocessor",
            input_mapping={"adata_path": "adata_path"},
        ),
        WorkflowStep(
            skill="proteomics_loader",
            input_mapping={"qurie_seq_path": "qurie_seq_path"},
        ),
        WorkflowStep(
            skill="distribution_shift_checker",
            input_mapping={
                "source_data": "adata_processed",
                "target_data": "proteomics_data",
            },
        ),
        WorkflowStep(
            skill="gat_trainer",
            input_mapping={
                "X_ctrl_ot": "X_ctrl_ot",
                "X_stim_ot": "X_stim_ot",
                "edge_index": "edge_index",
                "cell_type_ot": "cell_type_ot",
                "donor_ot": "donor_ot",
            },
        ),
        WorkflowStep(
            skill="benchmark_evaluator",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl_test": "X_ctrl_ot",
                "X_stim_test": "X_stim_ot",
                "edge_index": "edge_index",
                "cell_type_test": "cell_type_ot",
            },
        ),
        WorkflowStep(
            skill="uncertainty_estimator",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl": "X_ctrl_ot",
                "edge_index": "edge_index",
            },
        ),
        WorkflowStep(
            skill="two_audience_renderer",
            input_mapping={
                "evaluation_result": "benchmark_evaluator_result",
                "attention_result": "uncertainty_estimator_result",
                "uncertainty_result": "uncertainty_estimator_result",
                "plausibility_result": "uncertainty_estimator_result",
            },
        ),
    ],
)


# ─── WORKFLOW 3: Active Learning Loop ───

WORKFLOW_3 = Workflow(
    name="active_learning_loop",
    description=(
        "Uncertainty-driven experiment recommendation. "
        "Runs uncertainty estimation, ranks pathways by Gini entropy, "
        "and outputs ranked list for next QuRIE-seq experiment."
    ),
    steps=[
        WorkflowStep(
            skill="uncertainty_estimator",
            input_mapping={
                "model_path": "model_path",
                "X_ctrl": "X_ctrl",
                "edge_index": "edge_index",
            },
        ),
        WorkflowStep(
            skill="biological_plausibility_scorer",
            input_mapping={
                "predicted_interactions": "active_learning_recommendations",
                "gene_to_idx": "gene_to_idx",
            },
        ),
    ],
)


# ─── Placeholder Skills (registered as stubs) ───

from aivc.interfaces import (
    AIVCSkill, SkillResult, ValidationReport, ComputeCost,
    BiologicalDomain, ComputeProfile,
)
from aivc.registry import registry


@registry.register(
    name="proteomics_loader",
    domain=BiologicalDomain.PROTEOMICS,
    version="1.0.0",
    requires=["qurie_seq_path"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class ProteomicsLoader(AIVCSkill):
    """
    PLACEHOLDER — QuRIE-seq proteomics data loader.
    Available from May 2026 when QuRIE-seq data arrives.
    """

    def execute(self, inputs, context):
        raise NotImplementedError(
            "QuRIE-seq proteomics loader not yet implemented. "
            "Available from May 2026 when QuRIE-seq data arrives. "
            "Requires: QuRIE-seq CITE-seq h5ad files from wet lab."
        )

    def validate(self, result):
        return ValidationReport(
            passed=False, critic_name="ProteomicsLoader.validate",
            checks_passed=[], checks_failed=["Not implemented"],
        )

    def estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=5.0, gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0, can_run_on_cpu=True,
        )


@registry.register(
    name="distribution_shift_checker",
    domain=BiologicalDomain.MULTIMODAL,
    version="1.0.0",
    requires=["source_data", "target_data"],
    compute_profile=ComputeProfile.CPU_LIGHT,
)
class DistributionShiftChecker(AIVCSkill):
    """
    PLACEHOLDER — MMD distribution shift check between RNA and protein.
    Available from May 2026 when QuRIE-seq data arrives.
    """

    def execute(self, inputs, context):
        raise NotImplementedError(
            "Distribution shift checker not yet implemented. "
            "Available from May 2026 when QuRIE-seq data arrives. "
            "Will compute Maximum Mean Discrepancy (MMD) between "
            "RNA and protein modality distributions."
        )

    def validate(self, result):
        return ValidationReport(
            passed=False, critic_name="DistributionShiftChecker.validate",
            checks_passed=[], checks_failed=["Not implemented"],
        )

    def estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=2.0, gpu_memory_gb=0.0,
            profile=ComputeProfile.CPU_LIGHT,
            estimated_usd=0.0, can_run_on_cpu=True,
        )


# ─── Extension Examples (Phase 3 & 4 Placeholders) ───

@registry.register(
    name="atac_seq_encoder",
    domain=BiologicalDomain.EPIGENOMICS,
    version="1.0.0",
    requires=["atac_h5ad_path", "tf_motif_database_path"],
    compute_profile=ComputeProfile.GPU_INTENSIVE,
)
class ATACSeqEncoder(AIVCSkill):
    """
    PLACEHOLDER — Phase 3 implementation.
    Integrates 100,000+ chromatin peaks via TF motif scanning (JASPAR).
    Adds epigenomic node type to the heterogeneous graph.
    Zero changes to orchestrator, critics, or memory layers required.
    """

    def execute(self, inputs, context):
        raise NotImplementedError(
            "ATAC-seq encoder not yet implemented. "
            "Available in Phase 3 (late 2026) when in-house ATAC-seq data "
            "is generated. Requires JASPAR TF motif database and TF network "
            "inference expertise from collaborator."
        )

    def validate(self, result):
        return ValidationReport(
            passed=False, critic_name="ATACSeqEncoder.validate",
            checks_passed=[], checks_failed=["Not implemented"],
        )

    def estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=30.0, gpu_memory_gb=16.0,
            profile=ComputeProfile.GPU_INTENSIVE,
            estimated_usd=1.0, can_run_on_cpu=False,
        )


@registry.register(
    name="foundation_model_trainer",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["parse_10m_h5ad_path", "multi_gpu_config"],
    compute_profile=ComputeProfile.GPU_INTENSIVE,
)
class FoundationModelTrainer(AIVCSkill):
    """
    PLACEHOLDER — Phase 4 implementation.
    Pre-trains on Parse Biosciences 10M PBMC dataset.
    9.7 million cells, 90 cytokines, 12 donors.
    Requires multi-GPU H100 cluster via NCCL distributed training.
    Fine-tunes on QuRIE-seq proprietary data.
    """

    def execute(self, inputs, context):
        raise NotImplementedError(
            "Foundation model trainer not yet implemented. "
            "Available 2027 when Parse 10M dataset pre-training "
            "infrastructure is in place. Requires NVIDIA DGX Cloud "
            "multi-GPU H100 cluster."
        )

    def validate(self, result):
        return ValidationReport(
            passed=False, critic_name="FoundationModelTrainer.validate",
            checks_passed=[], checks_failed=["Not implemented"],
        )

    def estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=480.0, gpu_memory_gb=80.0,
            profile=ComputeProfile.GPU_INTENSIVE,
            estimated_usd=160.0, can_run_on_cpu=False,
        )


@registry.register(
    name="cytokine_perturbation_IL6",
    domain=BiologicalDomain.TRANSCRIPTOMICS,
    version="1.0.0",
    requires=["adata_path", "perturbation_id"],
    compute_profile=ComputeProfile.GPU_REQUIRED,
)
class IL6PerturbationSkill(AIVCSkill):
    """
    Extends perturbation response prediction to IL-6 stimulation.
    Adds a new row to the PerturbationEmbedding lookup table.
    Zero changes to model architecture — just a new embedding index.
    """
    PERTURBATION_ID = 2  # 0=ctrl, 1=IFN-b, 2=IL-6

    def execute(self, inputs, context):
        raise NotImplementedError(
            "IL-6 perturbation skill not yet implemented. "
            "Requires IL-6 stimulated scRNA-seq data. "
            "When available, adds perturbation_id=2 to the "
            "PerturbationEmbedding lookup table. Zero changes to "
            "model architecture required."
        )

    def validate(self, result):
        return ValidationReport(
            passed=False, critic_name="IL6PerturbationSkill.validate",
            checks_passed=[], checks_failed=["Not implemented"],
        )

    def estimate_cost(self, inputs):
        return ComputeCost(
            estimated_minutes=10.0, gpu_memory_gb=4.0,
            profile=ComputeProfile.GPU_REQUIRED,
            estimated_usd=0.33, can_run_on_cpu=True,
        )


# ─── Workflow Registry ───

WORKFLOWS = {
    "rna_baseline_to_demo": WORKFLOW_1,
    "multimodal_integration": WORKFLOW_2,
    "active_learning_loop": WORKFLOW_3,
}
