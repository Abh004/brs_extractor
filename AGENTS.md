# Project Overview

This project extracts functional business requirements from product demo videos by combining
an Azure Video Indexer (AVI) cloud proposer — per-20s-chunk OCR screen-text + transcript —
with a single LLM judge (Qwen3.5-9B locally, or gpt-4o via Azure Foundry), within-block
deduplication, and MoSCoW priority classification. The pipeline uploads the video to AVI,
splits the indexed OCR + transcript into 20s chunks, has the judge LLM convert each chunk
into requirement bullets, merges semantically equivalent bullets across time blocks,
classifies each as Must/Should/Could, infers BRS document structure via LLM, and renders a
structured Word document. Runs on Apple Silicon via MLX (`BACKEND=mlx`), on any CUDA/CPU
box (`BACKEND=hf`), or with all text-LLM stages hosted on Azure Foundry (`BACKEND=foundry`).
Includes a FastAPI web UI with WebSocket progress streaming.

## Repository Structure

- `main.py` — Thin CLI wrapper that delegates to `server.pipeline_runner.PipelineRunner`.
  Run directly: `python main.py`. Or via the web UI: `uvicorn server.app:app`.
- `config.py` — Single source of truth for paths, model names, chunk sizes, prompts,
  and gen parameters.
- `backend.py` — Inference backend abstraction: `InferenceBackend` ABC with three
  implementations — `MLXBackend` (Apple Silicon, `mlx_lm`, supports unified
  VLM-as-LLM for Qwen3.5), `HFBackend` (cross-platform, HuggingFace transformers +
  flash-attention), and `FoundryBackend` (hosted Azure OpenAI text-LLM, no local
  models). Select via `BACKEND` env var. Contains model ID mapping (`_MLX_TO_HF`)
  for cross-backend compatibility, `_NO_THINK_MODELS` for models needing `/no_think`
  suffix, and `_VLM_BACKED_TEXT_MODELS` for unified VLM-as-LLM routing.
- `utils.py` — Shared utilities: text cleaning (`strip_special_tokens`,
  `strip_think_blocks`, `clean_output`), spell-checking (`flag_unknown_words`,
  compound-word heuristics, domain-term whitelist), JSON extraction (`extract_balanced_json`,
  brace-depth counting).
- `config_generator.py` — Judge-LLM inference pass (via the shared backend handle) that reads
  the final bullet list and writes `output/project_config.json` for docx generation. Accepts
  optional shared LLM handle to avoid redundant model loads.
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
    Contains all 7 pipeline stages, retry/timeout constants (`_JUDGE_MAX_RETRIES=2`,
    `_JUDGE_TIMEOUT_SEC=60`), checkpoint management, and progress streaming via
    `StageUpdate` callback.
  - `job_manager.py` — `JobManager` singleton with in-memory job registry, concurrency
    control (one job at a time, GPU exclusivity), `JobStatus` enum (pending/running/
    completed/failed/cancelled), and thread-based job execution.
- `static/` — Web UI frontend:
  - `index.html` — Upload-video input, progress display with stage indicators,
    results view with collapsible blocks and download links.
  - `app.js` — WebSocket connection, status polling fallback, stage indicator updates,
    results rendering with MoSCoW color badges and flagged-word highlighting.
  - `styles.css` — Dark theme CSS with stage pills, progress bar, priority badges.
- `data_prep/` — Chunking of the Azure Video Indexer index into 20s buckets
  (`data_prep/avi.py`): OCR screen-text (with normalized screen region) + transcript.
  The legacy local keyframe/OpenCV and YouTube-transcript paths were removed.
- `models/` — `AVIProposer` cloud proposer (assembles OCR + transcript per chunk; no
  local model).
- `judges/` — Single LLM refinement (`JudgesPanel`) with sequential evaluation, bullet
  validation, and within-block dedup pipeline.
- `judges/panel.py` — `JudgesPanel`: preload/load/evaluate/unload lifecycle for the single
  judge model (Qwen3.5-9B). Uses `extract_bullets` → `filter_meta_bullets` →
  `filter_truncated_bullets` → `filter_substrings` for output validation.
- `judges/dedup.py` — Within-block bullet deduplication via Jaccard prefilter
  (CANDIDATE_THRESHOLD=0.40, pure Python, no embedding model) + LLM cluster resolution
  and canonicalization. No cross-block pass — semantically equivalent bullets are
  collapsed only within the same 20s time block.
- `judges/priority.py` — Batched MoSCoW priority classification (M/S/C) via Qwen3-1.7B,
  15 bullets per LLM call. Accepts optional shared LLM handle. Defaults failed parses to
  "S" to avoid all-Must-Have bias.
- `input/` — Input assets: `video.mp4` (uploaded to AVI for indexing).
- `output/` — Generated `extracted_requirements.json`, `project_config.json`,
  `BRS.docx`, `avi_index.json`, and `*_checkpoint.jsonl`.

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

# Run full pipeline (CLI)
python main.py

# Run with checkpoint resume (skips proposer re-run, re-runs judges + docx)
python main.py

# Run web UI
uvicorn server.app:app --reload

# Clear checkpoint to force re-inference
rm output/extracted_requirements_checkpoint.jsonl

# Regenerate docx only (after editing project_config.json)
node generate_docx.js output/extracted_requirements.json output/BRS.docx output/project_config.json

# Use HuggingFace backend (cross-platform) instead of MLX
BACKEND=hf python main.py

# Use hosted Azure Foundry / Azure OpenAI (gpt-4o) for all text-LLM stages
# (judge, dedup resolution, MoSCoW priority, config gen). Requires
# FOUNDRY_ENDPOINT / FOUNDRY_API_KEY / FOUNDRY_DEPLOYMENT in .env.
BACKEND=foundry python main.py
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
│  Store       │───▶│  AVI Index    │───▶│  AVI Proposer    │───▶│  Judge LLM   │
│  Video/      │    │  (20s chunk   │    │  (OCR screen-    │    │ (Qwen3.5-9B) │
│  Upload to   │    │  OCR +        │    │  text +          │    │  sequential  │
│  Azure       │    │  transcript)  │    │  transcript)     │    └──────┬───────┘
└──────────────┘    └───────────────┘    └──────────────────┘            │
                                                                 ┌───────▼───────┐
                                                                 │  Within-block │
                                                                 │  Dedup (Jacc. │
                                                                 │  + LLM)       │
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
                                                                 │  Judge LLM    │
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
(`backend.py`). The default `MLXBackend` runs on Apple Silicon via `mlx_lm`.
Supports unified VLM-as-LLM for Qwen3.5-9B (loaded via `mlx_vlm` for text-only inference
too). Set `BACKEND=hf` to use HuggingFace transformers + Pillow on any CUDA/MPS/CPU box.
Set `BACKEND=foundry` to route all text-LLM stages (judge, dedup resolution, MoSCoW
priority, config gen) to a hosted Azure Foundry / Azure OpenAI endpoint (e.g. gpt-4o)
via `FoundryBackend`; the proposer remains AVI, so no local GPU models load and only
Qwen3-1.7B (dedup/priority/config) loads locally on CPU.
Model ID mapping (`_MLX_TO_HF` in `backend.py`) translates `lmstudio-community`/`mlx-community`
IDs to canonical HuggingFace repo IDs for cross-backend compatibility. Flash-attention 2
auto-detected at import time.

**Memory flow:** The AVI proposer is a cloud client (no local GPU model). The judge
(Qwen3.5-9B, ~9GB) loads once in Stage 4. Qwen3-1.7B (~1GB) loads once in Stage 5 and is
**shared** across within-block dedup, priority classification, and config generation
(same handle, single load/unload cycle). Dedup uses a pure-Python Jaccard prefilter (no
embedding model), so nothing runs on the Metal GPU stream during Stage 5. Under
`BACKEND=foundry` none of the local GPU models load — the judge/dedup/priority/config-gen
LLM calls go to the hosted endpoint. Checkpoints are written incrementally
(one line per chunk, JSONL) so the expensive proposer upload/index step can be skipped
on re-run.

**Web UI:** FastAPI server (`uvicorn server.app:app`) serves the static SPA and exposes
REST endpoints for pipeline orchestration. Progress is streamed via WebSocket as
`StageUpdate` dataclass instances. The `JobManager` enforces single-job concurrency
(GPU exclusivity) and runs pipelines in a daemon thread. The SPA accepts a video upload
and displays results with collapsible time blocks, MoSCoW priority badges
(red/gold/blue), and spell-flagged word highlighting.

**Retry & timeout:** Judge calls wrap `_with_retry()` (2 retries, exponential
backoff from 2s). Located in `server/pipeline_runner.py`.

**Dedup pipeline:**

`extract_bullets` → `filter_meta_bullets` → `filter_truncated_bullets` →
`filter_substrings` (applied per-chunk inside `JudgesPanel._validate_bullets`)

Within-block dedup runs per 20s time block: bullets above
`CANDIDATE_THRESHOLD=0.40` Jaccard similarity are grouped by
`cluster_candidates`, then `llm_resolve_clusters` asks the LLM to confirm true
duplicates (conservative — same-topic is not enough) and returns one canonical
bullet per confirmed group. No cross-block pass exists.

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

- **Secrets:** No API keys or credentials in code. AVI tokens and Foundry keys come
  from `.env`.
- **Dependency scanning:** `pip-audit` or `safety` recommended before deployment.
- **Model weights:** Downloaded from Hugging Face Hub via `mlx-lm`
  (`mlx-vlm` only when the unified VLM-as-LLM judge path is used). Verify model IDs in
  `config.py` before use.
- **License:** MIT. Node.js `docx` package is MIT-licensed.

## Agent Guardrails

- **DO NOT** modify `output/` — it contains generated artifacts.
- **DO NOT** commit model weights or `.venv/` to version control.
- **DO NOT** change `AVI_UPLOAD_NAME` or paths without updating the corresponding
  AVI cache marker (`output/avi_upload.json` / `avi_index.json.marker`).
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

- **Backend:** Select MLX (Apple Silicon), HF (CUDA/CPU), or Foundry (hosted
  Azure OpenAI, text-only) via `BACKEND` env var.
  Add new backends by implementing `InferenceBackend` ABC in `backend.py`.
- **Models:** Swap judge or dedup models by editing `JUDGE_MODEL`, `DEDUP_MODEL` in
  `config.py`. For HF backend, add model ID mapping in `_MLX_TO_HF` dict in `backend.py`.
  Qwen3.5 uses the unified VLM-as-LLM path in MLXBackend; add new unified model IDs to
  `_VLM_BACKED_TEXT_MODELS`. Models needing `/no_think` append must be added to
  `_NO_THINK_MODELS` in `backend.py`.
- **Chunking:** Adjust `CHUNK_SIZE_SEC` in `config.py`.
- **Deduplication:** Tune `CANDIDATE_THRESHOLD` (recall prefilter) in
  `judges/dedup.py`.
- **Prompts:** Edit `AVI_JUDGE_SYSTEM`, `AVI_RAW_TEMPLATE`, `JUDGE_SYSTEM`, etc. in `config.py`.
- **Judge count:** The panel now uses a single judge. To add multiple judges, extend
  `JudgesPanel.load_all()` and `evaluate()` in `judges/panel.py` and re-introduce a
  merger step (Jaccard-prefiltered LLM dedup handles N inputs).
- **Priority:** Adjust `batch_size` in `classify_priorities()`, or modify the
  `_PRIORITY_SYSTEM` prompt in `judges/priority.py`.
- **Spell check:** Edit domain terms in `_get_spellchecker()` or suffix list in
  `_SUFFIXES` in `utils.py`.
- **Output format:** Modify JSON structure in `_dump_json()` or `_assemble_blocks()`
  in `server/pipeline_runner.py`.
- **Docx layout:** Edit `generate_docx.js` — all content is driven by
  `project_config.json`, no hardcoded product names or categories.
- **Retry/timeout:** Adjust `_JUDGE_MAX_RETRIES`, `_JUDGE_TIMEOUT_SEC` constants at the
  top of `server/pipeline_runner.py`.
- **Web UI:** Add new endpoints in `server/app.py`, modify `static/index.html`
  and `static/app.js` for frontend changes.

## Further Reading

- `config.py` — all tunable parameters in one file.
- `backend.py` — inference backend abstraction, three implementations, model ID mapping,
  unified VLM-as-LLM support, `/no_think` model handling.
- `utils.py` — shared text cleaning, spell checking, JSON extraction.
- `server/pipeline_runner.py` — complete 7-stage pipeline implementation.
- `server/app.py` — FastAPI REST + WebSocket API.
- `static/app.js` — frontend SPA with WebSocket progress streaming.
- [MLX documentation](https://ml-explore.github.io/mlx/)
- [mlx-vlm](https://github.com/ml-explore/mlx-vlm)
- [docx npm package](https://docx.js.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
