"""
Configuration file for the VLM-PwC project.
All paths, model names, and hyperparameters are defined here.
"""

VIDEO_ID = "YazxN78R_6M"

# File paths
VIDEO_PATH = "input/video.mp4"
TRANSCRIPT_PATH = "input/transcript.json"
FRAME_DIR = "keyframes"
OUTPUT_FILE = "output/extracted_requirements.json"

# Model configuration
MODEL_NAME = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit"

# Chunking configuration
CHUNK_SIZE_SEC = 20
FRAME_INTERVAL_SEC = 10

FRAME_SIZE = (448, 448)

# Generation configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
REPETITION_CONTEXT_SIZE = 20

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
JUDGE_MODEL = "lmstudio-community/Qwen3.5-9B-MLX-4bit"
DEDUP_MODEL = "mlx-community/Qwen3-1.7B-4bit"

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
