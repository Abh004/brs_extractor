"""
data_prep/avi.py

Chunk an Azure Video Indexer index into fixed-size time buckets, mirroring the
shape of the local `data_prep.transcript` chunker so the rest of the pipeline
(get_valid_chunks / Stage 3 proposer) is unchanged.

Each chunk carries both spoken transcript lines (`text_lines`) and OCR
screen-text lines (`ocr_lines`) so a judge can use the on-screen UI text,
which usually names features more precisely than the narration.
"""

from __future__ import annotations

import math

from config import CHUNK_SIZE_SEC

from avi import get_ocr_spatial, get_transcript


def _new_chunk() -> dict:
    return {"text_lines": [], "ocr_lines": [], "frame_paths": []}


def _line(seg: dict) -> str:
    """Format one AVI segment ({start, end, text}) as a timestamped line."""
    s = seg.get("start")
    return f"[{s:.1f}s] {seg['text']}" if isinstance(s, (int, float)) else seg["text"]


def _normalize(value: float | None) -> float:
    """Coerce a possibly-invalid AVI coordinate to a 0..1 fraction."""
    if value is None:
        return 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _region(left: float, top: float, width: float, height: float) -> str:
    """Map a normalized OCR bounding box to a coarse screen region."""
    cx = _normalize(left) + (_normalize(width) if width is not None else 0.0) / 2.0
    cy = _normalize(top) + (_normalize(height) if height is not None else 0.0) / 2.0
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    if row == "middle":
        return f"middle-{col}"
    if row == "center":
        return col
    if col == "center":
        return f"{row}-center"
    return f"{row}-{col}"


def _ocr_line(seg: dict) -> str:
    """Format one spatial OCR item as a timestamped, positioned screen line."""
    ts = ""
    s = seg.get("start")
    if isinstance(s, (int, float)):
        ts = f"[{s:.1f}s] "
    pos = _region(seg.get("left"), seg.get("top"), seg.get("width"), seg.get("height"))
    return f"{ts}({pos}) {seg['text']}"


def load_avi_chunks(index: dict) -> dict:
    """
    Split AVI transcript + OCR into 20s chunks keyed by bucket index.

    Args:
        index: Parsed AVI index dict (as returned by avi.upload_and_index).

    Returns:
        Dict mapping bucket_index → {text_lines, ocr_lines, frame_paths}.
    """
    chunks: dict = {}

    for seg in get_transcript(index):
        bucket = math.floor((seg.get("start") or 0) / CHUNK_SIZE_SEC)
        chunks.setdefault(bucket, _new_chunk())
        chunks[bucket]["text_lines"].append(_line(seg))

    for seg in get_ocr_spatial(index):
        bucket = math.floor((seg.get("start") or 0) / CHUNK_SIZE_SEC)
        chunks.setdefault(bucket, _new_chunk())
        chunks[bucket]["ocr_lines"].append(_ocr_line(seg))

    return chunks


def get_valid_avi_chunks(chunks: dict) -> list:
    """Filter to non-empty chunks, preserving the (bucket_index, chunk) shape."""
    return [
        (b, chunks[b])
        for b in sorted(chunks)
        if chunks[b]["text_lines"] or chunks[b]["ocr_lines"]
    ]