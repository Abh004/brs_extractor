# Project Overview

This project extracts functional business requirements from product demo videos by combining
a Vision-Language Model proposer (Qwen3-VL-8B) with a single LLM judge (Qwen3.5-9B),
embedding-based deduplication, and MoSCoW priority classification. The pipeline splits a
YouTube video + transcript into 20s chunks, generates candidate requirements per chunk,
refines them via the judge LLM, merges semantically equivalent bullets across time blocks,
classifies each as Must/Should/Could, infers BRS document structure via LLM, and renders a
structured Word document. Runs on Apple Silicon via MLX (`BACKEND=mlx`, default) or on any
CUDA/CPU box (`BACKEND=hf`). Includes a FastAPI web UI with WebSocket progress streaming.

## Repository Structure

- `main.py` — Thin CLI wrapper that delegates to `server.pipeline_runner.PipelineRunner`.
  Run directly: `python main.py`. Or via the web UI: `uvicorn server.app:app`.
- `config.py` — Single source of truth for paths, model names, chunk sizes, prompts,
  and gen parameters.
- `backend.py` — Inference backend abstraction: `InferenceBackend` ABC with two
  implementations — `MLXBackend` (Apple Silicon, `mlx-vlm`/`mlx_lm`, supports unified
  VLM-as-LLM for Qwen3.5) and `HFBackend` (cross-platform, HuggingFace transformers +
  Pillow + optional flash-attention). Select via `BACKEND` env var. Contains model ID
  mapping (`_MLX_TO_HF`) for cross-backend compatibility, `_NO_THINK_MODELS` for models
  needing `/no_think` suffix, and `_VLM_BACKED_TEXT_MODELS` for unified VLM-as-LLM routing.
- `utils.py` — Shared utilities: text cleaning (`strip_special_tokens`,
  `strip_think_blocks`, `clean_output`), spell-checking (`flag_unknown_words`,
  compound-word heuristics, domain-term whitelist), JSON extraction (`extract_balanced_json`,
  brace-depth counting), embedding normalization (`normalize_embeddings`).
- `config_generator.py` — Qwen3.5-9B inference pass (via `JUDGE_MODEL`) that reads the
  final bullet list and writes `output/project_config.json` for docx generation. Accepts
  optional shared LLM handle to avoid redundant model loads.
- `extractor.py` — Downloads YouTube transcript JSON for the configured `VIDEO_ID`.
  Uses `YouTubeTranscriptApi().fetch()` (newer API). Skips re-download if file exists
  unless `force=True`.
- `generate_docx.js` — Node.js script that reads `extracted_requirements.json` and
  `project_config.json` to produce `BRS.docx`. Requires Node.js >= 18 and `docx` npm package.
  All content driven by config JSON — no hardcoded categories or product names.
- `pyproject.toml` — Package metadata and pinned Python dependencies. Optional `[web]`
  extras for FastAPI/uvicorn/websockets.
- `server/` — FastAPI web application with three modules:
  - `app.py` — FastAPI app with REST endpoints (`/api/run`, `/api/status/{job_id}`,
    `/api/results/{job_id}`, `/api/cancel/{job_id}`, `/api/jobs`, `/api/download/{filename}`)
    and WebSocket (`/ws/{job_id}`) for progress streaming. Serves the static SPA at `/`.
  - `pipeline_runner.py` — **Actual pipeline logic** with `PipelineRunner` class.
    Contains all 7 pipeline stages, retry/timeout constants (`_VLM_MAX_RETRIES=2`,
    `_JUDGE_MAX_RETRIES=2`, `_VLM_TIMEOUT_SEC=120`, `_JUDGE_TIMEOUT_SEC=60`),
    checkpoint management, and progress streaming via `StageUpdate` callback.
  - `job_manager.py` — `JobManager` singleton with in-memory job registry, concurrency
    control (one job at a time, GPU exclusivity), `JobStatus` enum (pending/running/
    completed/failed/cancelled), and thread-based job execution.
- `static/` — Web UI frontend:
  - `index.html` — Input tabs (YouTube ID / Upload Video / Transcript Only), progress
    display with stage indicators, results view with collapsible blocks and download links.
  - `app.js` — WebSocket connection, status polling fallback, stage indicator updates,
    results rendering with MoSCoW color badges and flagged-word highlighting.
  - `styles.css` — Dark theme CSS with stage pills, progress bar, priority badges.
- `data_prep/` — Video frame extraction (OpenCV, resize to 448×448 at save time) and
  transcript chunking. Keyframes cached with auto-invalidation via `.video_id` marker file.
- `models/` — `VLMProposer` class wrapping backend VLM inference.
- `judges/` — Single LLM refinement (`JudgesPanel`) with sequential evaluation, bullet
  validation, and embedding-based dedup pipeline.
- `judges/panel.py` — `JudgesPanel`: preload/load/evaluate/unload lifecycle for the single
  judge model (Qwen3.5-9B). Uses `extract_bullets` → `filter_meta_bullets` →
  `filter_truncated_bullets` → `filter_substrings` for output validation.
- `judges/dedup.py` — Bullet deduplication via embedding prefilter (`all-MiniLM-L6-v2`
  on CPU, no MLX GPU dependency) + LLM cluster resolution. Supports auto-accept skip for
  high-cosine pairs (AUTO_ACCEPT_THRESHOLD=0.85) above the recall prefilter
  (CANDIDATE_THRESHOLD=0.35). Used for cross-block dedup (per-bullet against a running
  pool of accepted bullets, with LLM adjudication for ambiguous near-dupes).
- `judges/priority.py` — Batched MoSCoW priority classification (M/S/C) via Qwen3-1.7B,
  15 bullets per LLM call. Accepts optional shared LLM handle. Defaults failed parses to
  "S" to avoid all-Must-Have bias.
- `input/` — Input assets: `video.mp4` and `transcript.json`.
- `keyframes/` — Cached frame images (448×448, JPEG q90), marked by `.video_id`.
- `output/` — Generated `extracted_requirements.json`, `project_config.json`,
  `BRS.docx`, and `*_checkpoint.jsonl`.

## Build & Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Install Python dependencies
pip install -e .

# Install web UI dependencies (optional, for FastAPI server)
pip install -e '.[web]'

# Install Node.js dependency (one-time)
npm install -g docx

# Fetch YouTube transcript (requires VIDEO_ID in config.py)
python extractor.py

# Run full pipeline (CLI)
python main.py

# Run with checkpoint resume (skips VLM inference, re-runs judges + docx)
python main.py

# Run web UI
uvicorn server.app:app --reload

# Clear checkpoint to force re-inference
rm output/extracted_requirements_checkpoint.jsonl

# Regenerate docx only (after editing project_config.json)
node generate_docx.js output/extracted_requirements.json output/BRS.docx output/project_config.json

# Use HuggingFace backend (cross-platform) instead of MLX
BACKEND=hf python main.py
```

## Code Style & Conventions

- **Language:** Python 3.12+ with type hints on all public functions.
- **Formatting:** Google-style docstrings (Args / Returns). Max line length ~100 chars.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_CASE` for config constants.
- **Imports:** stdlib → third-party → local, grouped by blank line.
- **Comments:** `# ── section header ──` for pipeline stages; design-rationale docstrings
  in utils.py and dedup.py.
- **Commits:** Conventional Commits format (`feat:`, `fix:`, `refactor:`, `chore:`).

## Architecture Notes

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Store       │───▶│  Data Prep    │───▶│  VLM Proposer    │───▶│  Judge LLM   │
│  Video/      │    │  (chunk +     │    │  (Qwen3-VL-8B)   │    │ (Qwen3.5-9B) │
│  Transcript  │    │  keyframe)    │    │  per chunk       │    │  sequential  │
└──────────────┘    └───────────────┘    └──────────────────┘    └──────┬───────┘
                                                                        │
                                                                ┌───────▼───────┐
                                                                │   Embedding   │
                                                                │  Dedup (CPU)  │
                                                                │  cross-block  │
                                                                └───────┬───────┘
                                                                        │
                                                                ┌───────▼───────┐
                                                                │    MoSCoW     │
                                                                │   Priority    │
                                                                │ (Qwen3-1.7B)  │
                                                                └───────┬───────┘
                                                                        │
                                                                ┌───────▼───────┐
                                                                │  Config Gen   │
                                                                │  Qwen3.5-9B   │
                                                                │  (Stage 6)    │
                                                                └───────┬───────┘
                                                                        │
                                                                ┌───────▼───────┐
                                                                │  BRS Docx     │
                                                                │  generate_    │
                                                                │  docx.js      │
                                                                │  (Stage 7)    │
                                                                └───────────────┘
```

**Backend abstraction:** All stages go through the `InferenceBackend` interface
(`backend.py`). The default `MLXBackend` runs on Apple Silicon via `mlx-vlm`/`mlx_lm`.
Supports unified VLM-as-LLM for Qwen3.5-9B (loaded via `mlx_vlm` for text-only inference
too). Set `BACKEND=hf` to use HuggingFace transformers + Pillow on any CUDA/MPS/CPU box.
Model ID mapping (`_MLX_TO_HF` in `backend.py`) translates `lmstudio-community`/`mlx-community`
IDs to canonical HuggingFace repo IDs for cross-backend compatibility. Flash-attention 2
auto-detected at import time.

**Memory flow:** VLM proposer (Qwen3-VL-8B, ~8GB) loaded for Stage 3, then unloaded before
the judge (Qwen3.5-9B, ~9GB) loads once in Stage 4. Embedding model runs on CPU (~80MB)
so it never contends for the Metal GPU stream. Qwen3-1.7B (~1GB) loads once in Stage 5
and is **shared** across cross-block dedup, priority classification, and config generation
(same handle, single load/unload cycle). Checkpoints are written incrementally (one line
per chunk, JSONL) so the expensive VLM stage can be skipped on re-run.

**Web UI:** FastAPI server (`uvicorn server.app:app`) serves the static SPA and exposes
REST endpoints for pipeline orchestration. Progress is streamed via WebSocket as
`StageUpdate` dataclass instances. The `JobManager` enforces single-job concurrency
(GPU exclusivity) and runs pipelines in a daemon thread. The SPA supports three input
modes (YouTube ID, video upload, transcript-only) and displays results with collapsible
time blocks, MoSCoW priority badges (red/gold/blue), and spell-flagged word highlighting.

**Retry & timeout:** VLM and judge calls wrap `_with_retry()` (2 retries, exponential
backoff from 2s) and `_with_timeout()` (SIGALRM, 120s VLM / 60s judge per attempt).
Located in `server/pipeline_runner.py`.

**Dedup pipeline:**

`extract_bullets` → `filter_meta_bullets` → `filter_truncated_bullets` →
`filter_substrings` (applied per-chunk inside `JudgesPanel._validate_bullets`)

Cross-block dedup iterates each bullet against a running pool of previously-accepted
bullets. Above `CANDIDATE_THRESHOLD=0.35` cosine similarity, the candidate and its
near-neighbours are sent to `llm_resolve_clusters` which first splits auto-accept pairs
(`AUTO_ACCEPT_THRESHOLD=0.85`, no LLM call) then adjudicates the ambiguous band via LLM.
If the LLM confirms the new bullet is a duplicate, it is dropped.

**Priority classification:** Batched MoSCoW (M/S/C) via Qwen3-1.7B, 15 bullets per LLM
call. Defaults failed/bad-parse entries to "S" (not "M") to avoid all-Must-Have bias.
The system prompt instructs the LLM to ignore the template word "must" and judge actual
business criticality.

**Spell checking:** Post-processing pass via `flag_unknown_words()` wraps unrecognised
words in `[[double brackets]]`. Uses pyspellchecker with domain terms + compound-word
heuristics (suffix-stripping, compound-splitting) to reduce false positives on technical
vocabulary. No auto-correction — flagging invites human review without risk of silently
altering meaning.

## Testing Strategy

No tests are implemented yet.

## Security & Compliance

- **Secrets:** No API keys or credentials in code. YouTube transcripts are public.
- **Dependency scanning:** `pip-audit` or `safety` recommended before deployment.
- **Model weights:** Downloaded from Hugging Face Hub via `mlx-vlm` / `mlx-lm` /
  `sentence-transformers`. Verify model IDs in `config.py` before use.
- **License:** MIT. Node.js `docx` package is MIT-licensed.

## Agent Guardrails

- **DO NOT** modify `output/` or `keyframes/` — they are generated artifacts.
- **DO NOT** commit model weights or `.venv/` to version control.
- **DO NOT** change `VIDEO_ID` or paths without updating the corresponding cache marker.
- **DO NOT** parallelize judge evaluation — MLX GPU streams are thread-local and
  will fail with `"There is no Stream(gpu, N) in current thread"`.
- **DO NOT** call `generate_docx.js` without first verifying `project_config.json`
  exists — `PipelineRunner._generate_docx()` already enforces this.
- **DO NOT** remove the `finally` block that unloads the shared judge handle in Stage 5/6
  — if the handle leaks, subsequent runs will fail with "model already loaded".
- A reviewer must sign off on any changes to `config.py` (prompts, thresholds,
  model names), `backend.py` (model mapping, inference logic), and `judges/panel.py`
  (judge logic, bullet validation).

## Extensibility Hooks

- **Backend:** Select MLX (Apple Silicon) or HF (CUDA/CPU) via `BACKEND` env var.
  Add new backends by implementing `InferenceBackend` ABC in `backend.py`.
- **Models:** Swap proposer or judge models by editing `MODEL_NAME`, `JUDGE_MODEL`,
  `DEDUP_MODEL` in `config.py`. For HF backend, add model ID mapping in `_MLX_TO_HF`
  dict in `backend.py`. Qwen3.5 uses the unified VLM-as-LLM path in MLXBackend;
  add new unified model IDs to `_VLM_BACKED_TEXT_MODELS`. Models needing `/no_think`
  append must be added to `_NO_THINK_MODELS` in `backend.py`.
- **Chunking:** Adjust `CHUNK_SIZE_SEC` and `FRAME_INTERVAL_SEC` in `config.py`.
- **Deduplication:** Tune `CANDIDATE_THRESHOLD` (recall prefilter) and
  `AUTO_ACCEPT_THRESHOLD` (auto-merge ceiling) in `judges/dedup.py`.
- **Prompts:** Edit `PROPOSER_PROMPT`, `JUDGE_SYSTEM`, etc. in `config.py`.
- **Judge count:** The panel now uses a single judge. To add multiple judges, extend
  `JudgesPanel.load_all()` and `evaluate()` in `judges/panel.py` and re-introduce a
  merger step (embedding dedup handles N inputs).
- **Priority:** Adjust `batch_size` in `classify_priorities()`, or modify the
  `_PRIORITY_SYSTEM` prompt in `judges/priority.py`.
- **Spell check:** Edit domain terms in `_get_spellchecker()` or suffix list in
  `_SUFFIXES` in `utils.py`.
- **Output format:** Modify JSON structure in `_dump_json()` or `_assemble_blocks()`
  in `server/pipeline_runner.py`.
- **Docx layout:** Edit `generate_docx.js` — all content is driven by
  `project_config.json`, no hardcoded product names or categories.
- **Retry/timeout:** Adjust `_VLM_MAX_RETRIES`, `_JUDGE_MAX_RETRIES`,
  `_VLM_TIMEOUT_SEC`, `_JUDGE_TIMEOUT_SEC` constants at the top of
  `server/pipeline_runner.py`.
- **Web UI:** Add new endpoints in `server/app.py`, modify `static/index.html`
  and `static/app.js` for frontend changes.

## Further Reading

- `config.py` — all tunable parameters in one file.
- `backend.py` — inference backend abstraction, two implementations, model ID mapping,
  unified VLM-as-LLM support, `/no_think` model handling.
- `utils.py` — shared text cleaning, spell checking, JSON extraction, embedding helpers.
- `server/pipeline_runner.py` — complete 7-stage pipeline implementation.
- `server/app.py` — FastAPI REST + WebSocket API.
- `static/app.js` — frontend SPA with WebSocket progress streaming.
- [MLX documentation](https://ml-explore.github.io/mlx/)
- [mlx-vlm](https://github.com/ml-explore/mlx-vlm)
- [docx npm package](https://docx.js.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
