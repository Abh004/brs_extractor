"""
Configuration file for the BRS extraction pipeline.
All paths, deployment labels, and hyperparameters are defined here.

The pipeline is Azure + Foundry only. All LLM (and vision) inference goes to
a hosted Azure Foundry / Azure OpenAI deployment identified by
FOUNDRY_DEPLOYMENT; JUDGE_MODEL / DEDUP_MODEL below are label-only — callers
pass them as a model name for token-usage reporting, but FoundryBackend ignores
the value and always uses the deployment from the environment.
"""

import os

VIDEO_PATH = "input/video.mp4"
OUTPUT_FILE = "output/extracted_requirements.json"
CHECKPOINT_FILE = "output/extracted_requirements_checkpoint.jsonl"
CONFIG_FILE = "output/project_config.json"
DOCX_FILE = "output/BRS.docx"

# ── Deployment labels (label-only; FoundryBackend always uses FOUNDRY_DEPLOYMENT) ──
JUDGE_MODEL = "foundry-judge"
# Despite the name, DEDUP_MODEL powers dedup, priority classification, AND config generation.
DEDUP_MODEL = "foundry-dedup"

# Chunking configuration
CHUNK_SIZE_SEC = 20

# Single-pass AVI mode (default on; disable with AVI_SINGLE_PASS=0). Feeds the whole video's
# OCR + transcript to the judge as ONE block, falling back to 20s chunks if the assembled
# context exceeds SINGLE_PASS_INPUT_TOKEN_CAP (see server/pipeline_runner.py).
AVI_SINGLE_PASS = os.getenv("AVI_SINGLE_PASS", "1") != "0"
# Per-judge-call output budget for a single-pass prompt (default JUDGE_MAX_TOKENS is too small).
SINGLE_PASS_JUDGE_MAX_TOKENS = 8000
# Input-budget guard: fall back to chunking if the assembled context exceeds this many tokens.
SINGLE_PASS_INPUT_TOKEN_CAP = 1_000_000
# Per-block output cap so the O(n^2) Jaccard prefilter and per-cluster LLM calls stay bounded.
SINGLE_PASS_MAX_BULLETS = 500
# Pre-merge input bound for the whole-video block's dedup (final cap applies AFTER merging).
SINGLE_PASS_DEDUP_INPUT_CAP = 1500
# Lower Jaccard prefilter threshold for whole-video blocks (raises recall for near-paraphrases).
SINGLE_PASS_CANDIDATE_THRESHOLD = 0.25
# Strip non-functional OCR noise (URLs/domains, emails, watermarks, footers, version strings).
AVI_OCR_NOISE_FILTER = os.getenv("AVI_OCR_NOISE_FILTER", "1") == "1"

AVI_UI_DESCRIPTION_FILE = "avi_ui_description.json"
# Keyframes per vision request (Azure caps ~10 images/request); cost scales with keyframe count.
AVI_UI_BATCH_SIZE = 10
# Vision image resolution: low (cheap) | auto | high.
AVI_UI_FRAME_DETAIL = os.getenv("AVI_UI_FRAME_DETAIL", "low")
# Output budget per vision call (one batch).
AVI_UI_DESCRIPTION_MAX_TOKENS = 2000
# Safety rail on the CONCATENATED description (trimmed to its head with a warning if exceeded).
AVI_UI_MAX_DESC_TOKENS = 20000
# System prompt for the vision pass, focused on component PLACEMENT only.
AVI_UI_DESCRIPTION_SYSTEM = (
    "You are a UI analyst reviewing keyframes from a software product demo. "
    "The attached images are keyframes from the demo video, in chronological "
    "order. For each distinct screen, describe the UI layout precisely: the "
    "placement of buttons, menus, panels and dialogs (top/bottom, left/center/"
    "right and relative ordering), and the position of non-text graphical "
    "components such as logos and icons. Only describe what is actually visible "
    "in the frames. Be factual and concise; no preamble."
)
# Section template prepending the VLM keyframe analysis to OCR + transcript in single-pass.
AVI_UI_SECTION_TEMPLATE = (
    "VLM SCREEN ANALYSIS (component placement from video keyframes):\n"
    "{ui_description}\n\n"
)

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
# Transport resilience for the AVI HTTP client (retries on transient connect/read failures).
AVI_MAX_RETRIES = 3
AVI_CONNECT_TIMEOUT = 30
AVI_READ_TIMEOUT = 180
AVI_RETRY_BACKOFF = 2.0
AVI_UPLOAD_TIMEOUT_SEC = 900  # generous read timeout for large video uploads

FOUNDRY_MAX_RETRIES = 3
FOUNDRY_CONNECT_TIMEOUT = 30
FOUNDRY_READ_TIMEOUT = 180
FOUNDRY_RETRY_BACKOFF = 2.0

# ── Prompts ─────────────────────────────────────────────────────────────
# The AVI judge converts raw OCR + transcript for a chunk into "The system shall ..." bullets.
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

# Per-chunk raw context the AVI judge consumes (OCR lines carry a normalized-screen region prefix).
AVI_RAW_TEMPLATE = (
    "SCREEN TEXT (on-screen UI labels with normalized screen region):\n"
    "{screen_text}\n\n"
    "SPOKEN TRANSCRIPT:\n{transcript}"
)

# Single-pass judge prompt: the same raw OCR + transcript as AVI_JUDGE_SYSTEM, but for the
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
    "- If a 'VLM SCREEN ANALYSIS' block is present above the OCR, treat it as an "
    "authoritative description of component placement (buttons, menus, panels, "
    "logos, icons, and their positions). Reconcile it with the OCR: prefer the "
    "OCR for the exact label text, and the VLM analysis for where each component "
    "sits and for non-text graphical components.\n"
    "- Only report UI and layout that the OCR text, OCR positions, and any VLM "
    "screen analysis actually show; do not invent controls that are not present.\n"
    "- If the same capability or feature recurs at several timestamps across "
    "the video, describe it ONCE (consolidating the recurrences into a single "
    "bullet); do not restate the same requirement per timestamp.\n"
    "- Order your bullets chronologically: list them by the time each "
    "capability FIRST appears in the video, earliest first.\n"
    "- Drop duplication and generic statements. No preamble, plain text only. "
    "Use timestamped, fact-based bullet points.\n"
)
