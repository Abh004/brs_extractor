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

from avi import get_ocr, get_transcript


def _new_chunk() -> dict:
    return {"text_lines": [], "ocr_lines": [], "frame_paths": []}


def _line(seg: dict) -> str:
    """Format one AVI segment ({start, end, text}) as a timestamped line."""
    s = seg.get("start")
    return f"[{s:.1f}s] {seg['text']}" if isinstance(s, (int, float)) else seg["text"]


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

    for seg in get_ocr(index):
        bucket = math.floor((seg.get("start") or 0) / CHUNK_SIZE_SEC)
        chunks.setdefault(bucket, _new_chunk())
        chunks[bucket]["ocr_lines"].append(_line(seg))

    return chunks


def get_valid_avi_chunks(chunks: dict) -> list:
    """Filter to non-empty chunks, preserving the (bucket_index, chunk) shape."""
    return [
        (b, chunks[b])
        for b in sorted(chunks)
        if chunks[b]["text_lines"] or chunks[b]["ocr_lines"]
    ]