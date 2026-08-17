"""
models/frame_ui_proposer.py

Route B: describe on-screen UI (component placement) from AVI keyframes using the
user's own vision-capable backend (e.g. Foundry gpt-4o).

This is the source of the "ui_description" block for the single-pass context.
Instead of making AVI run its connected model (which requires a deployment name
AVI accepts), we:
  1. pull keyframe JPEGs from the AVI index (avi.fetch_sample_keyframes),
  2. base64 them into data URLs,
  3. send them + a placement prompt to backend.generate_llm(..., images=...),
  4. cache the resulting text per source video.

Only FoundryBackend supports image input (the sole backend in this pipeline).
The result is cached to AVI_UI_DESCRIPTION_FILE so re-runs never re-pay.
"""

from __future__ import annotations

import base64
import json
import os

import avi
from config import (
    AVI_UI_DESCRIPTION_FILE,
    AVI_UI_DESCRIPTION_MAX_TOKENS,
    AVI_UI_DESCRIPTION_SYSTEM,
    DEDUP_MODEL,
)

_UI_PROMPT = (
    "Attached are keyframes from a software product demo, in chronological order. "
    "For each distinct screen, describe the UI layout precisely: the placement of "
    "buttons, menus, panels and dialogs (top/bottom, left/center/right and "
    "relative ordering), and the position of non-text graphical components such "
    "as logos and icons. Only describe what is actually visible."
)


def _data_url(path: str) -> str:
    """Encode a local image file as an image/jpeg data URL for the vision API."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, AVI_UI_DESCRIPTION_FILE)


def get_keyframe_ui_description(
    backend,
    video_path: str,
    index: dict,
    output_dir: str = "output",
    max_frames: int | None = None,
) -> str:
    """
    Produce a UI-placement description for the video's keyframes via the backend.

    Reuses a cached description when the marker matches the same source video, so
    a re-run never re-pays for the vision call.

    Args:
        backend:     An InferenceBackend with vision support (Foundry).
        video_path:  Path of the source video file (used for the cache marker).
        index:       Parsed AVI index dict.
        output_dir:  Where to persist the cached description + frames + marker.
        max_frames:  Cap on keyframes fed to one vision call (None = backend cap).

    Returns:
        The VLM-style UI description text, or "" if no keyframes are available.
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

    frames = avi.fetch_sample_keyframes(
        video_path, index, output_dir, max_frames=max_frames
    )
    if not frames:
        return ""

    images = [_data_url(f["path"]) for f in frames]
    print(f"  [UI] sending {len(images)} keyframes to vision backend...")

    handle = backend.load_llm(DEDUP_MODEL)
    try:
        text = backend.generate_llm(
            handle,
            system_prompt=AVI_UI_DESCRIPTION_SYSTEM,
            user_prompt=_UI_PROMPT,
            max_tokens=AVI_UI_DESCRIPTION_MAX_TOKENS,
            images=images,
        )
    finally:
        backend.unload_llm(handle)

    if not text:
        print("  [UI] vision backend returned empty output; skipping UI description.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, indent=2, ensure_ascii=False)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(avi._source_identity(video_path))
    print(f"  [UI] keyframe UI description cached: {cache_path}")
    return text