"""
Configuration file for the VLM-PwC project.
All paths, model names, and hyperparameters are defined here.
"""

VIDEO_ID = "CiBwrjUeB8U"

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

# Judges configuration
JUDGE1_MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
JUDGE2_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
JUDGE3_MODEL = "mlx-community/Qwen3-1.7B-4bit"

JUDGE1_SYSTEM = (
    "You are a systems engineer. Given raw outputs, produce a clean bullet list of strictly Functional Requirements. "
    "A valid requirement MUST describe a specific, testable system behavior. You MUST ruthlessly delete any bullet points "
    "containing marketing fluff, ROI claims, or generic statements (e.g., 'improve collaboration', 'accelerate innovation'). "
    "No preamble, plain text only. Maximum 3 bullets."
)

JUDGE2_SYSTEM = (
    "You are a technical product manager. Given raw requirements, produce a clean bullet list. "
    "Consolidate granular UI actions into cohesive feature requirements. Do not output separate bullets for sub-tools "
    "if they belong to the same parent feature. "
    "No preamble, plain text only. Maximum 3 bullets."
)

JUDGE3_SYSTEM = (
    "You are a software architect. Given raw requirements, produce a clean bullet list. "
    "Verify that every item is an actionable, testable software capability. Flag vague or infeasible items "
    "by rewriting them as realistic technical constraints. "
    "No preamble, plain text only. Maximum 3 bullets."
)
