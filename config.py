"""
Configuration file for the VLM-PwC project.
All paths, model names, and hyperparameters are defined here.

Model IDs live ONLY in this file (the model registry below). Every other
module — backend.py, models/, judges/, config_generator.py — derives its
model references from here, so renaming a model is a single-line change.
"""

from dataclasses import dataclass

VIDEO_ID = "YazxN78R_6M"

# File paths
VIDEO_PATH = "input/video.mp4"
TRANSCRIPT_PATH = "input/transcript.json"
FRAME_DIR = "keyframes"
OUTPUT_FILE = "output/extracted_requirements.json"
CHECKPOINT_FILE = "output/extracted_requirements_checkpoint.jsonl"
CONFIG_FILE = "output/project_config.json"
DOCX_FILE = "output/BRS.docx"

# ── Model registry ─────────────


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one model used by the pipeline.

    Attributes:
        mlx_id:     Model ID passed to the MLX backend (and referenced by
                    MLXBackend's _NO_THINK/_VLM_BACKED classification).
        hf_id:      Canonical HuggingFace repo ID used by the HF backend
                    (the _MLX_TO_HF mapping).
        no_think:   Whether the model needs a "/no_think" suffix to disable
                    CoT thinking blocks.
        vlm_backed: Whether the model is loaded via mlx_vlm even for
                    text-only LLM calls (unified VLM-as-LLM routing).
    """

    mlx_id: str
    hf_id: str
    no_think: bool = False
    vlm_backed: bool = False


PROPOSER = ModelSpec(
    mlx_id="lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
    hf_id="Qwen/Qwen3-VL-8B-Instruct",
    no_think=True,
)
JUDGE = ModelSpec(
    mlx_id="lmstudio-community/Qwen3.5-9B-MLX-4bit",
    hf_id="Qwen/Qwen3.5-9B",
    no_think=True,
    vlm_backed=True,
)
# Shared small LLM. NOTE: despite the name, DEDUP_MODEL powers priority
# classification AND config generation in addition to dedup resolution —
# the pipeline loads it once and shares the handle across those stages.
DEDUP = ModelSpec(
    mlx_id="mlx-community/Qwen3-1.7B-4bit",
    hf_id="Qwen/Qwen3-1.7B",
    no_think=True,
)

ALL_MODELS = [PROPOSER, JUDGE, DEDUP]

ENCODER_MODEL = "all-MiniLM-L6-v2"

# Backwards-compat aliases — keep consumer imports unchanged. Each maps to a
# single registry entry so renaming a model is a one-line change above.
MODEL_NAME = PROPOSER.mlx_id
JUDGE_MODEL = JUDGE.mlx_id
DEDUP_MODEL = DEDUP.mlx_id

# Chunking configuration
CHUNK_SIZE_SEC = 20
FRAME_INTERVAL_SEC = 10

FRAME_SIZE = (448, 448)

# Generation configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
REPETITION_CONTEXT_SIZE = 20

# Per-stage inference budgets
JUDGE_MAX_TOKENS = 150
JUDGE_MAX_BULLETS = 5
PRIORITY_BATCH_SIZE = 15
PRIORITY_MAX_TOKENS = 120
DEDUP_MAX_TOKENS = 500
CONFIG_MAX_TOKENS = 1200

# Prompts
PROPOSER_PROMPT = (
    "You are a strict systems analyst watching a software tutorial. Your job is to translate "
    "the observed UI actions into functional system capabilities. Do NOT infer business goals, "
    "ROI, or marketing claims. Combine granular micro-actions (e.g., 'click line', 'click circle') "
    "into broader feature sets (e.g., 'The system must provide 2D geometric sketching tools'). "
    "Output ONLY a valid JSON array of strings in the form: 'The system shall allow users to [perform specific technical action]'.\n\n"
    "Transcript: {transcript}"
)

# Judges configuration — single judge, replacing the 3-judge panel
JUDGE_SYSTEM = (
    "You are a systems engineer refining raw VLM-extracted requirements into a "
    "clean, testable functional requirements list.\n"
    "- Every requirement must describe a specific, testable system behavior.\n"
    "- Delete any requirement containing marketing language, ROI claims, or generic "
    "business statements (e.g. 'improve collaboration', 'accelerate innovation', "
    "'establish trust').\n"
    "- Consolidate granular UI micro-actions into one cohesive feature requirement "
    "rather than listing every sub-step separately.\n"
    "- Do NOT add any example, detail, or illustrative parenthetical that is not "
    "explicitly present in the input. If a term is ambiguous, preserve it exactly "
    "as given rather than inferring what it might mean.\n"
    "- Rewrite vague or infeasible items as realistic, testable technical "
    "constraints instead of deleting them outright.\n"
    "No preamble, plain text only. Maximum 5 bullets."
)
