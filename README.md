# Business Requirements Extractor

Extracts functional business requirements from product demo videos by combining
**Azure Video Indexer (AVI)** cloud OCR + transcript with a single LLM judge
(**gpt-4o via Azure Foundry**). The pipeline uploads the video, splits indexed
OCR + transcript into 20s chunks (or runs a single whole-video pass), converts
each chunk into requirement bullets, deduplicates and MoSCoW-classifies them,
infers a BRS document structure, and renders a structured Word document.

All LLM stages (judge, dedup resolution, priority classification, config
generation) and the keyframe vision pass run on hosted Azure Foundry. No local
GPU models are used.

## Prerequisites

- **Python 3.12+**
- An **Azure Video Indexer** account (trial or paid) — see `.env.example`
- An **Azure Foundry / Azure OpenAI** resource with a deployment (e.g. `gpt-4o`)
  and vision support — powers every LLM stage and the keyframe UI pass
- A product demo video (MP4, AVI, MOV, MKV)

## Installation

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate   #Linux
.venv\Scripts\Activate.ps1  #Windows Powershell

# 2. Install the package (platform-agnostic core)
pip install -e .

# 3. Install web UI extras (only needed for the FastAPI web interface)
pip install -e '.[web]'
```

## Configuration

All secrets are read from `.env` (loaded automatically at startup). Minimum
required values:

| Variable | Description |
|---|---|
| `AZURE_VIDEO_INDEXER_LOCATION` | AVI account region, e.g. `trial` or `eastus` |
| `AZURE_VIDEO_INDEXER_ACCOUNT_ID` | Your AVI account ID (UUID) |
| `AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN` | Short-lived AVI access token (~1h) |
| `FOUNDRY_ENDPOINT` | Foundry/OpenAI base URL, e.g. `https://<resource>.openai.azure.com/openai/v1/` |
| `FOUNDRY_API_KEY` | API key for the Foundry endpoint |
| `FOUNDRY_DEPLOYMENT` | Deployment name sent as the model, e.g. `gpt-4o` |

Optional tuning (also in `.env` / `config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `AVI_SINGLE_PASS` | `1` | Feed the whole video as one block; falls back to 20s chunks if the context exceeds the token cap |
| `AVI_UI_FRAME_DETAIL` | `Low` | Vision image resolution: `Low` \| `Auto` \| `High` |
| `BACKEND` | `foundry` | Only `foundry` is accepted |
| `AVI_VERIFY_SSL` / `FOUNDRY_VERIFY_SSL` | `1` | Set `0` to skip TLS verification (corporate TLS interception) |
| `AVI_NO_PROXY` / `FOUNDRY_NO_PROXY` | — | Set `1` to bypass proxy env vars for the respective client |

## Running

### Option A — Web UI (recommended)

```bash
uvicorn server.app:app --reload
```

Open http://localhost:8000/, upload a video, and click **Generate Requirements**.
Progress is streamed live via WebSocket (stage pills + progress bar); results
render as collapsible requirement blocks with MoSCoW priority badges and
spell-flagged word highlighting. Download `BRS.docx` from the results view.

### Option B — CLI

Place your video at `input/video.mp4`, then:

```bash
python main.py
```

Both entry points run the identical `PipelineRunner` pipeline.

### Checkpoint / resume

Progress is checkpointed per chunk (`output/extracted_requirements_checkpoint.jsonl`),
so a re-run skips the expensive AVI upload/index step and resumes from where it
stopped. To force a fresh full run:

```bash
rm output/extracted_requirements_checkpoint.jsonl
```

## Expected output

All artifacts are written to `output/`:

| File | Contents |
|---|---|
| `avi_index.json` | Raw Azure Video Indexer index (OCR + transcript + keyframes) |
| `extracted_requirements.json` | Summary + per-time-block requirement bullets with MoSCoW priority |
| `project_config.json` | LLM-inferred BRS document structure used for the docx |
| `BRS.docx` | Generated Word document (cover, TOC, requirement sections, appendix) |
| `extracted_requirements_checkpoint.jsonl` | Incremental per-chunk checkpoints (one JSON per line) |
| `avi_ui_description.json` | Cached keyframe UI-description output (single-pass mode) |


## Possible errors & troubleshooting

Errors surface in one of three places: the **web UI error panel** (via `/api/*`
HTTP errors or a terminal WebSocket `StageUpdate`), the **CLI stderr/traceback**,
or as **inline warnings** the pipeline logs and continues past. The most common
fatal errors, their causes, and fixes:

### Configuration / environment

| Error message | Cause | Fix |
|---|---|---|
| `Missing required Azure Video Indexer config: <KEY> ...` | `AZURE_VIDEO_INDEXER_LOCATION` / `_ACCOUNT_ID` unset | Set them in `.env` |
| `No AVI credential set. Set AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN.` | AVI account token missing or expired (~1h) | Generate a fresh token (`az video-indexer account access-token`) and set it in `.env` |
| `Missing required Foundry config: ...` | `FOUNDRY_ENDPOINT` / `FOUNDRY_API_KEY` / `FOUNDRY_DEPLOYMENT` unset | Fill these in `.env` |
| `Unknown backend '...'. This pipeline only supports 'foundry'.` | `BACKEND` env set to something else | Leave unset or set `BACKEND=foundry` |

### Video file

| Error message | Cause | Fix |
|---|---|---|
| `Video not found at input/video.mp4 ...` | No/empty video at the expected path (CLI run) | Place the video at `input/video.mp4`, or use the web UI upload |
| `Upload a video file.` (web UI) | No file selected before clicking **Generate Requirements** | Select a video file first |

### Azure Video Indexer — network / transport

| Error message | Cause | Fix |
|---|---|---|
| `AVI request timed out / connection aborted. Check your VPN/proxy/firewall, or set AVI_NO_PROXY=1` | Timeouts/connects aborted, e.g. WinError 10053 | Verify network; set `AVI_NO_PROXY=1` to bypass proxies |
| `AVI network error during <METHOD> <url>: ...` | Generic transport failure | Check network/firewall/DNS; set `AVI_NO_PROXY=1` |
| `AVI upload failed after N attempts: ...` | Upload interrupted repeatedly mid-stream | Improve connectivity; set `AVI_NO_PROXY=1` and/or `AVI_VERIFY_SSL=0` |
| `AVI upload returned no video ID: ...` | Unexpected AVI response body | Check account state in the Azure portal; retry |

### Azure Video Indexer — HTTP / indexing

| Error message | Cause | Fix |
|---|---|---|
| `AVI HTTP 401/403: check AVI credentials ...` | Bad or expired account token, or wrong account ID | Refresh the token; confirm `AZURE_VIDEO_INDEXER_ACCOUNT_ID` |
| `AVI HTTP <4xx/5xx> <METHOD> <url> -> <body>` | Various AVI API errors; also "public access has been disabled" / 404 for private-endpoint accounts | Check `AZURE_VIDEO_INDEXER_ENDPOINT` / `_RESOLVE_IP` and DNS (private endpoint should resolve to a 10.x/172.x/192.x IP) |
| `AVI indexing did not finish within Ns. Check the account in the Azure portal.` | Indexing exceeded `AVI_INDEX_TIMEOUT_SEC` (600s) | Raise `AVI_INDEX_TIMEOUT_SEC`; verify the account tier permits indexing enqueue |
| `AVI indexing failed for video <id>: ...` | AVI reported the video as failed | Inspect the account in the Azure portal; retry on a fresh video |
| `Cannot resolve videoId from AVI index.` | Parsed index has no video id | Check index freshness; clear `output/avi_index.json` to re-index |

### Foundry backend

| Error message | Cause | Fix |
|---|---|---|
| `Foundry request failed for model <deploy>: ...` | Transport/connection failure to the endpoint | Check network; set `FOUNDRY_NO_PROXY=1` / `FOUNDRY_VERIFY_SSL=0` |
| `Foundry HTTP <4xx/5xx> for model <deploy>: ...` | Bad key, wrong deployment name, or model limits | Verify `FOUNDRY_API_KEY` / `FOUNDRY_DEPLOYMENT`; ensure the deployment supports what's requested (incl. vision for keyframes) |
| `Judge model unavailable: ...` (stage 2) | `JudgesPanel.preload()` validation failed | Check `FOUNDRY_*` env / deployment access |

### Pipeline stages

| Error message | Cause | Fix |
|---|---|---|
| `Pipeline cancelled by user` | `cancel()` was requested (Cancel button or `/api/cancel`) | Expected — not an error; the job ends in `cancelled` |
| `Failed: <exception>` / `Pipeline finished` (stage-0 error) | A fatal exception in any stage | Read the traceback / error panel for the specific stage |
| `Proposer returned empty or invalid output.` | AVI chunk produced no usable context | Non-fatal per chunk; logged as a block `error` |
| `Config failed: ...` (stage 6) | Config-inference LLM call failed | Shown as stage error; `BRS.docx` may be skipped |

### Non-fatal warnings (pipeline continues)

| Warning | Behavior |
|---|---|
| `[UI] keyframe UI description failed ... continuing with OCR + transcript only.` | Vision pass skipped; runs on OCR + transcript |
| `[single-pass] context ~N tokens exceeds cap ... falling back to chunking.` | Falls back to 20s chunks instead of one block |
| `[single-pass] WARNING: full context ~N tokens ... may be too large` | Judge call may exceed the model context |
| Priority classification failure | Defaults all bullets to priority `S` to avoid all-Must bias |
| Judge `DEDUP_MODEL` load failure (`judge3_handle is None`) | Skips dedup/priority/config-gen; only writes raw JSON |
| `WARNING: python-docx not available; skipping docx` | `generate_docx.py` unavailable; JSON already written |

### Web API (HTTP status codes)

| Status | Endpoint / message | Meaning |
|---|---|---|
| `400` | `/api/cancel/{id}` — "Job not found or not running" | Cannot cancel a non-running job |
| `403` | `/api/download/{file}` — "File not available for download" | Filename not in the download whitelist |
| `404` | `/api/status`, `/api/results` — "Job not found" | Unknown `job_id` |
| `404` | `/api/download/{file}` — "<file> not found — pipeline may not have completed" | Artifact not yet generated |
| `409` | `/api/run` — "A pipeline job is already running ..." | Single-job concurrency limit; wait or cancel the current job first |
| `409` | `/api/run` — "Could not create job (concurrency limit)." | Race between the busy check and job creation |

### Client-side (static/app.js)

| Message | Cause |
|---|---|
| `Upload a video file.` | No file selected |
| `Failed to start pipeline` / backend `detail` | `/api/run` rejected the request (see HTTP codes above) |
| `Network error: <msg>` | `fetch` to `/api/run` failed (server not running / unreachable) |
| `Pipeline was cancelled.` | Job cancelled before completion |
| Job `failed`/`error` with a message | A stage-0 error update was received |

### Diagnostics

`python probe_avi_aoai.py --check-vision` verifies Foundry vision support and
dumps AVI account/video state to help isolate account-level failures.

## License

MIT
