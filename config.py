"""
Configuration file for the VLM-PwC project.
All paths, model names, and hyperparameters are defined here.

Model IDs live ONLY in this file (the model registry below). Every other
module — backend.py, models/, judges/, config_generator.py — derives its
model references from here, so renaming a model is a single-line change.
"""

import os
from dataclasses import dataclass

VIDEO_PATH = "input/video.mp4"
TRANSCRIPT_PATH = "input/transcript.json"
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

ALL_MODELS = [JUDGE, DEDUP]

# Backwards-compat aliases — keep consumer imports unchanged. Each maps to a
# single registry entry so renaming a model is a one-line change above.
JUDGE_MODEL = JUDGE.mlx_id
DEDUP_MODEL = DEDUP.mlx_id

# Chunking configuration
CHUNK_SIZE_SEC = 20

# Single-pass AVI mode (Foundry friendly). When enabled, the entire video's
# OCR + transcript is fed to the judge as ONE block instead of per-20s chunks.
# This drops per-block time attribution but reduces judge round-trips and the
# local 20s-bucket output shape. The proposer stage still needs chunk
# boundaries IF the assembled full context is too large — see the fallback in
# server/pipeline_runner.py. Toggle via AVI_SINGLE_PASS=1.
AVI_SINGLE_PASS = os.getenv("AVI_SINGLE_PASS", "0") == "1"
# Per-judge-call output budget for a single-pass prompt. The default
# JUDGE_MAX_TOKENS (150) is far too small to emit ALL bullets for a long video.
SINGLE_PASS_JUDGE_MAX_TOKENS = 8000
# Conservative input-budget guard: if the assembled full context is estimated
# to exceed this many tokens, single-pass falls back to coarse chunking instead
# of overflowing the hosted model's context window.
SINGLE_PASS_INPUT_TOKEN_CAP = 100000
# Downstream dedup/priority/config are bounded per block. With one whole-video
# block the bullet list can be large; cap it so the O(n^2) Jaccard prefilter and
# per-cluster LLM calls stay reasonable.
SINGLE_PASS_MAX_BULLETS = 500
# Pre-merge input bound for the whole-video block's dedup. Generous enough that
# unique requirements are rarely dropped, tight enough that the O(n^2) Jaccard
# prefilter and per-cluster LLM calls stay bounded. The final SINGLE_PASS_MAX_BULLETS
# cap is applied AFTER merging, so duplicates cannot crowd out unique bullets.
SINGLE_PASS_DEDUP_INPUT_CAP = 1500
# Lower Jaccard prefilter threshold for the whole-video block. The per-chunk
# default (CANDIDATE_THRESHOLD=0.40) misses near-paraphrases; a lower threshold
# raises recall so the LLM adjudicator actually sees similar-but-differently-worded
# bullets and can merge them.
SINGLE_PASS_CANDIDATE_THRESHOLD = 0.25
# Strip clearly non-functional OCR noise (bare URLs/domains, emails, watermarks,
# copyright/footer, version strings) before the judge sees the screen text.
AVI_OCR_NOISE_FILTER = os.getenv("AVI_OCR_NOISE_FILTER", "1") == "1"

# Generation configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
REPETITION_CONTEXT_SIZE = 20

# Per-stage inference budgets
JUDGE_MAX_TOKENS = 150
PRIORITY_BATCH_SIZE = 15
PRIORITY_MAX_TOKENS = 120
DEDUP_MAX_TOKENS = 500
CONFIG_MAX_TOKENS = 1200

# ── Azure Video Indexer ─────────────────────────
AVI_LANGUAGE = "en-US"
AVI_UPLOAD_NAME = "brs-pipeline-demo"
AVI_INDEX_FILE = "avi_index.json"
AVI_INDEX_TIMEOUT_SEC = 600  # max seconds to wait for AVI to finish indexing
AVI_POLL_INTERVAL_SEC = 10
# Transport resilience for the AVI HTTP client (retries on transient
# connection/read failures — e.g. corporate proxies/firewalls that abort
# mid-upload with WinError 10053).
AVI_MAX_RETRIES = 3
AVI_CONNECT_TIMEOUT = 30
AVI_READ_TIMEOUT = 180
AVI_RETRY_BACKOFF = 2.0
AVI_UPLOAD_TIMEOUT_SEC = 900  # generous read timeout for large video uploads

# Env vars consumed by avi.py (loaded via python-dotenv):
#   AZURE_VIDEO_INDEXER_LOCATION        e.g. "trial" or a region like "eastus"
#   AZURE_VIDEO_INDEXER_ACCOUNT_ID      AVI account GUID
#   AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN   contributor account access token
#   AZURE_VIDEO_INDEXER_ENDPOINT        optional data-plane host override for
#                                       private-endpoint accounts (default
#                                       "api.videoindexer.ai"). Set to your
#                                       account FQDN that resolves to the
#                                       private endpoint IP, e.g.
#                                       "<account>.api.videoindexer.ai".
#   AZURE_VIDEO_INDEXER_RESOLVE_IP      optional private endpoint IP to connect
#                                       to for AZURE_VIDEO_INDEXER_ENDPOINT when
#                                       DNS still resolves the FQDN publicly on
#                                       a locked-down laptop (no hosts-file
#                                       access). App-level DNS override; SNI/Host
#                                       keep the FQDN.

# ── Azure Foundry / Azure OpenAI (judge backend) ─────────────
# Optional hosted text-LLM backend for the judge, dedup resolution, MoSCoW
# priority, and config generation. Set BACKEND=foundry to use it instead of
# the local MLX/HF models. Env vars consumed by backend.FoundryBackend:
#   FOUNDRY_ENDPOINT      base URL, e.g. https://<resource>.openai.azure.com/openai/v1/
#   FOUNDRY_API_KEY       API key for the endpoint
#   FOUNDRY_DEPLOYMENT    deployment name sent as the "model" field (e.g. gpt-4o)
#   FOUNDRY_VERIFY_SSL=0  optional: disable TLS verification (corporate interception)
#   FOUNDRY_NO_PROXY=1    optional: ignore HTTP(S)_PROXY / NO_PROXY
FOUNDRY_MAX_RETRIES = 3
FOUNDRY_CONNECT_TIMEOUT = 30
FOUNDRY_READ_TIMEOUT = 180
FOUNDRY_RETRY_BACKOFF = 2.0

# ── Prompts ─────────────────────────────────────────────────────────────
# The AVI judge reads RAW OCR (on-screen UI text) + transcript for a chunk
# and EMITS "The system shall ..." bullets — the job the local frame-based
# VLM proposer used to do. This replaces the local VLM proposer; the judge
# LLM now converts screen text into testable capabilities.
AVI_JUDGE_SYSTEM = (
    "You are a strict systems analyst watching a software demo. Below you will be "
    "given OCR (on-screen text captured from the video frames) and the spoken "
    "transcript for one segment of a product demo.\n"
    "- Translate the observed UI and narrated behaviour into distinct, testable "
    "system capabilities.\n"
    "- Do NOT infer business goals, ROI, or marketing claims.\n"
    "- Combine granular micro-actions into broader feature sets (e.g. 'click line', "
    "'click circle' → 'The system must provide 2D geometric sketching tools').\n"
    "- Each bullet must be phrased as: "
    "'The system shall allow users to [specific technical action]'.\n"
    "- Prefer on-screen UI text (labels, buttons, menus) when it is present — it "
    "usually names the feature more precisely than speech.\n"
    "- Each OCR line is prefixed with the on-screen region where the text was "
    "captured (e.g. '(top-right)'). Use these positions to describe the screen "
    "layout — where buttons, menus, and panels sit (top/bottom, left/center/right, "
    "and relative ordering) — like a visual review of the demo.\n"
    "- Only report UI and layout that the OCR text and positions actually show; "
    "do not invent controls that are not present.\n"
    "- Drop duplication and generic statements. No preamble, plain text only.\n"
)

# Compose the per-chunk raw context the AVI judge consumes. Kept here so the
# exact formatting is tunable in one place. OCR lines carry a normalized-screen
# region prefix (e.g. "(top-right)") so the judge can reconstruct UI placement.
AVI_RAW_TEMPLATE = (
    "SCREEN TEXT (on-screen UI labels with normalized screen region):\n"
    "{screen_text}\n\n"
    "SPOKEN TRANSCRIPT:\n{transcript}"
)

# Single-pass judge prompt: the SAME raw OCR + transcript as AVI_JUDGE_SYSTEM,
# but for the WHOLE video in one call rather than one 20s segment. The model
# must enumerate every distinct capability across the full product demo without
# relying on a single short window to bound the output.
AVI_SINGLE_PASS_SYSTEM = (
    "You are a strict systems analyst watching an entire product demo video. "
    "Below you will be given OCR (on-screen text captured from the video "
    "frames) and the spoken transcript for the WHOLE video, in chronological "
    "order. Segment lines carry a start timestamp in square brackets, e.g. "
    "'[45.0s]', and OCR lines carry the on-screen region, e.g. '(top-right)'.\n"
    "- Translate the observed UI and narrated behaviour into distinct, testable "
    "system capabilities for the ENTIRE video. Do not limit yourself to the "
    "first part — capture capabilities from the beginning, middle, and end.\n"
    "- Do NOT infer business goals, ROI, or marketing claims.\n"
    "- IGNORE non-functional, incidental screen text: URLs and domain names "
    "(e.g. 'example.com'), email addresses, watermarks, copyright and footer "
    "text, version or build numbers, timestamps, cookie/consent banners, "
    "brand slogans, and search-placeholder or breadcrumb strings. Only convert "
    "actual interactive controls (buttons, menus, forms, toggles) and core "
    "narrated behaviour into requirements.\n"
    "- Combine granular micro-actions into broader feature sets (e.g. 'click "
    "line', 'click circle' → 'The system must provide 2D geometric sketching "
    "tools').\n"
    "- Each bullet must be phrased as: "
    "'The system shall allow users to [specific technical action]'.\n"
    "- Prefer on-screen UI text (labels, buttons, menus) when it is present — it "
    "usually names the feature more precisely than speech.\n"
    "- Use OCR region prefixes to describe screen layout (top/bottom, "
    "left/center/right, and relative ordering) where the text actually shows it.\n"
    "- Only report UI and layout that the OCR text and positions actually show; "
    "do not invent controls that are not present.\n"
    "- If the same capability or feature recurs at several timestamps across "
    "the video, describe it ONCE (consolidating the recurrences into a single "
    "bullet); do not restate the same requirement per timestamp.\n"
    "- Drop duplication and generic statements. No preamble, plain text only. "
    "Use timestamped, fact-based bullet points.\n"
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
    "No preamble, plain text only."
)
