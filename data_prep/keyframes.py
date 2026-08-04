import math
import os
import shutil

import cv2

from config import (
    CHUNK_SIZE_SEC,
    FRAME_INTERVAL_SEC,
    FRAME_SIZE,
    VIDEO_ID,
)

from .transcript import _new_chunk

_CACHE_MARKER = os.path.join(os.path.dirname(__file__), ".video_id")


def _load_cached_frames(frame_dir: str, chunks: dict) -> dict:
    """Load frame paths from an existing cache directory into chunks."""
    print(f"Keyframes already exist in '{frame_dir}', loading from cache...")
    for fname in sorted(os.listdir(frame_dir)):
        if not fname.endswith(".jpg"):
            continue
        try:
            # Filename format: frame_{bucket_index}_{time}.jpg
            parts = fname.replace(".jpg", "").split("_")
            bucket_index = int(parts[1])
        except (IndexError, ValueError):
            continue

        frame_filename = os.path.join(frame_dir, fname)
        if bucket_index not in chunks:
            chunks[bucket_index] = _new_chunk()
        chunks[bucket_index]["frame_paths"].append(frame_filename)
    return chunks


def extract_keyframes(video_path: str, chunks: dict, frame_dir: str) -> dict:
    """
    Extract keyframes from video at specified intervals, resize to FRAME_SIZE,
    and add them to chunks.

    Cache behaviour: if a valid cache exists for the current VIDEO_ID, frames
    are loaded from disk. If VIDEO_ID has changed, the old cache is deleted and
    frames are re-extracted.

    Args:
        video_path: Path to the video file
        chunks: Dictionary of chunks to populate with frame paths
        frame_dir: Directory to save extracted frames

    Returns:
        Updated chunks dictionary with frame paths
    """
    # Check cache validity
    if os.path.isdir(frame_dir) and os.path.exists(_CACHE_MARKER):
        with open(_CACHE_MARKER) as f:
            cached_id = f.read().strip()
        if cached_id == VIDEO_ID:
            return _load_cached_frames(frame_dir, chunks)
        else:
            print(
                f"Video ID changed ({cached_id} → {VIDEO_ID}), clearing keyframes cache..."
            )
            shutil.rmtree(frame_dir)

    print("Extracting keyframes from video...")
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or math.isnan(fps):
        fps = 30.0

    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                ):
                    break
                print(f"  Warning: could not read frame {frame_count}, skipping.")
                frame_count += 1
                continue

            current_time_sec = frame_count / fps

            if frame_count % math.ceil(fps * FRAME_INTERVAL_SEC) == 0:
                bucket_index = math.floor(current_time_sec / CHUNK_SIZE_SEC)

                # #1: Resize to FRAME_SIZE (448×448) at extraction time —
                # Qwen3-VL's native resolution; avoids internal rescaling during
                # inference, reducing VLM processing time per frame.
                resized = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

                frame_filename = os.path.join(
                    frame_dir, f"frame_{bucket_index}_{current_time_sec:.1f}.jpg"
                )
                # JPEG quality 90 — good fidelity, smaller than default (95)
                cv2.imwrite(frame_filename, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if bucket_index not in chunks:
                    print(
                        f"  Note: bucket {bucket_index} has frames but no transcript text "
                        f"(t={current_time_sec:.1f}s)."
                    )
                    chunks[bucket_index] = _new_chunk()

                chunks[bucket_index]["frame_paths"].append(frame_filename)

            frame_count += 1
    finally:
        cap.release()

    # Write cache marker so subsequent runs load from disk
    with open(_CACHE_MARKER, "w") as f:
        f.write(VIDEO_ID)

    return chunks
