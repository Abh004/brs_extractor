"""
models/frame_ui_proposer.py

Route B: describe on-screen UI (component placement) from AVI keyframes using the
user's own vision-capable backend (e.g. Foundry gpt-4o).

This is the source of the "ui_description" block for the single-pass context.
Instead of making AVI run its connected model (which requires a deployment name
AVI accepts), we:
  1. pull EVERY keyframe JPEG from the AVI index (avi.fetch_sample_keyframes),
  2. base64 them into data URLs,
  3. chunk them into temporally-adjacent batches of AVI_UI_BATCH_SIZE (Azure caps
     ~10 images per chat-completions request),
  4. run one backend.generate_llm(..., images=...) call per batch, sequentially,
     reusing a single loaded handle,
  5. concatenate the per-batch descriptions into one chapter,
  6. cache the result per source video.

Only FoundryBackend supports image input (the sole backend in this pipeline).
The result is cached to AVI_UI_DESCRIPTION_FILE so re-runs never re-pay, and the
frame images are cached per source so re-runs never re-download.
"""

from __future__ import annotations

import base64
import json
import os

import avi
from config import (
    AVI_UI_BATCH_SIZE,
    AVI_UI_DESCRIPTION_FILE,
    AVI_UI_DESCRIPTION_MAX_TOKENS,
    AVI_UI_DESCRIPTION_SYSTEM,
    AVI_UI_MAX_DESC_TOKENS,
    DEDUP_MODEL,
)

_UI_PROMPT = (
    "Attached are keyframes from a software product demo, in chronological order. "
    "For each distinct screen, describe the UI layout precisely: the placement of "
    "buttons, menus, panels and dialogs (top/bottom, left/center/right and "
    "relative ordering), and the position of non-text graphical components such "
    "as logos and icons. Only describe what is actually visible."
)

_BATCH_SEP = "\n\n---- next keyframe segment ----\n\n"


def _data_url(path: str) -> str:
    """Encode a local image file as an image/jpeg data URL for the vision API."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _format_seconds(seconds: float) -> str:
    """Format a seconds value as M:SS (e.g. 83.4 -> '1:23')."""
    total = max(int(round(seconds)), 0)
    return f"{total // 60}:{total % 60:02d}"


def _batch_prompt(batch_index: int, batch_count: int, frames: list[dict]) -> str:
    """Build the user prompt for one batch, scoped to its time segment."""
    if not frames:
        return _UI_PROMPT
    start = _format_seconds(min(f["start_sec"] for f in frames))
    end = _format_seconds(max(f["start_sec"] for f in frames))
    header = (
        f"This is keyframe batch {batch_index}/{batch_count} of the demo video, "
        f"covering the time segment {start} -> {end}. "
    )
    return header + _UI_PROMPT


def _cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, AVI_UI_DESCRIPTION_FILE)


def _describe_batches(backend, frames: list[dict], batch_size: int) -> str:
    """Run one vision call per batch and concatenate the descriptions.

    A single handle is loaded once and reused across all batches, sequentially.
    """
    batches = avi.chunk_keyframes(frames, batch_size)
    batch_count = len(batches)
    if batch_count == 0:
        return ""

    handle = backend.load_llm(DEDUP_MODEL)
    parts: list[str] = []
    try:
        for i, batch in enumerate(batches, start=1):
            images = [_data_url(f["path"]) for f in batch]
            print(f"  [UI] describing keyframe batch {i}/{batch_count} "
                  f"({len(images)} frames, {_batch_span(batch)})...")
            text = backend.generate_llm(
                handle,
                system_prompt=AVI_UI_DESCRIPTION_SYSTEM,
                user_prompt=_batch_prompt(i, batch_count, batch),
                max_tokens=AVI_UI_DESCRIPTION_MAX_TOKENS,
                images=images,
            )
            if not text:
                print(f"  [UI]   batch {i} returned empty output; skipped.")
                continue
            parts.append(text)
    finally:
        backend.unload_llm(handle)

    return _BATCH_SEP.join(parts).strip()


def _batch_span(batch: list[dict]) -> str:
    if not batch:
        return "?"
    start = _format_seconds(min(f["start_sec"] for f in batch))
    end = _format_seconds(max(f["start_sec"] for f in batch))
    return f"{start} -> {end}"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (word count * 1.3) for input-budget guards."""
    return max(1, int(len(text.split()) * 1.3))


def _apply_size_guard(text: str) -> str:
    """Trim the concatenated description to AVI_UI_MAX_DESC_TOKENS with a warning."""
    budget = AVI_UI_MAX_DESC_TOKENS
    if budget is None or budget <= 0:
        return text
    tokens = _estimate_tokens(text)
    if tokens <= budget:
        return text
    # Keep the head; tokens align with ~4 chars/token on average at the tail.
    keep_chars = int(budget * 20 / 4)
    head = text[:keep_chars].rsplit("\n", 1)[0]
    print(
        f"  [UI] WARNING: concatenated UI description ~{tokens} tokens exceeds "
        f"AVI_UI_MAX_DESC_TOKENS ({budget}); trimmed to its head to protect the "
        f"judge context window."
    )
    return head + "\n\n[... UI description trimmed to fit the context budget ...]"


def get_keyframe_ui_description(
    backend,
    video_path: str,
    index: dict,
    output_dir: str = "output",
    batch_size: int = AVI_UI_BATCH_SIZE,
) -> str:
    """
    Produce a UI-placement description for the video's keyframes via the backend.

    Analyzes EVERY keyframe, one ~10-frame vision call at a time, and returns the
    concatenated description. Reuses a cached description when the marker matches
    the same source video, so a re-run never re-pays for the vision calls.

    Args:
        backend:    An InferenceBackend with vision support (Foundry).
        video_path: Path of the source video file (used for the cache marker).
        index:      Parsed AVI index dict.
        output_dir: Where to persist the cached description + frames + marker.
        batch_size: Max keyframes per vision call (images/request ceiling).

    Returns:
        The concatenated UI description text, or "" if no keyframes are available.
    """
    cache_path = _cache_path(output_dir)
    marker_path = cache_path + ".marker"

    if os.path.exists(cache_path) and os.path.exists(marker_path):
        if avi._marker_matches(marker_path, video_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                text = cached.get("text")
                if text:
                    print(f"  [UI] Reusing cached keyframe UI description: {cache_path}")
                    return text
            except (OSError, ValueError):
                pass

    frames = avi.fetch_sample_keyframes(video_path, index, output_dir)
    if not frames:
        return ""

    print(f"  [UI] analyzing {len(frames)} keyframes in batches of "
          f"{batch_size} (Foundry vision)...")
    text = _describe_batches(backend, frames, batch_size)
    text = _apply_size_guard(text)

    if not text:
        print("  [UI] vision backend returned no descriptions; skipping UI pass.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, indent=2, ensure_ascii=False)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(avi._source_identity(video_path))
    print(f"  [UI] keyframe UI description cached: {cache_path}")
    return text