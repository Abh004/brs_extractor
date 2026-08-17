"""
data_prep/avi.py

Chunk an Azure Video Indexer index into fixed-size time buckets, mirroring the
shape of the old local chunker so the rest of the pipeline
(get_valid_avi_chunks / Stage 3 proposer) is unchanged.

Each chunk carries both spoken transcript lines (`text_lines`) and OCR
screen-text lines (`ocr_lines`) so a judge can use the on-screen UI text,
which usually names features more precisely than the narration.
"""

from __future__ import annotations

import math
import re

from config import AVI_OCR_NOISE_FILTER, CHUNK_SIZE_SEC

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


def build_avi_single_pass_chunk(index: dict, ui_description: str | None = None) -> dict:
    """
    Assemble the ENTIRE video's OCR + transcript into a single chunk dict
    (no 20s bucketing), so single-pass mode can feed the whole demo to the
    judge in one call. Shares the {text_lines, ocr_lines, frame_paths} shape
    with `_new_chunk()`, so AVIProposer.generate() consumes it unchanged.

    OCR noise (URLs, watermarks, footers, version strings) is stripped first
    (when AVI_OCR_NOISE_FILTER is enabled) so incidental screen text is never
    surfaced to the judge as a requirement.

    Args:
        index:          Parsed AVI index dict (as returned by avi.upload_and_index).
        ui_description: Optional VLM keyframe screen-analysis text (component
                        placement) from the Foundry vision pass over AVI
                        keyframes. When set, it is attached to the chunk so the
                        proposer can prepend it to the OCR + transcript for the
                        judge.

    Returns:
        A single chunk dict with all transcript + (noise-filtered) OCR lines
        in chronological order (each still carrying its start timestamp /
        screen region), plus an optional keyframe ui_description.
    """
    chunk = _new_chunk()
    chunk["text_lines"] = [_line(seg) for seg in get_transcript(index)]
    ocr_segments = get_ocr_spatial(index)
    if AVI_OCR_NOISE_FILTER:
        ocr_segments = filter_ocr_noise(ocr_segments)
    chunk["ocr_lines"] = [_ocr_line(seg) for seg in ocr_segments]
    if ui_description:
        chunk["ui_description"] = ui_description
    return chunk


# A bare line is treated as noise when its visible text, after leading
# punctuation is stripped, matches any of these — i.e. the line is PREDOMINANTLY
# non-functional incidental text (URL/domain, email, watermark/copyright,
# footer, version/build label), not a UI control label.
_NOISE_RE = re.compile(
    r"(?:"
    r"https?://\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\S+\.(?:com|net|org|io|co|ai|edu|gov|dev|app)\b"
    r"|copyright|\W©|watermark"
    r"|\bv\d+(?:\.\d+)*\b"
    r"|\b(?:ver|version|build|rev|release)\s*(?:no\.?\s*)?\d+(?:[.\s]\d+)*\b"
    r"|\b(?:all\s+rights\s+reserved|privacy\s+policy|terms\s+of\s+(?:use|service))\b"
    r")",
    re.IGNORECASE,
)


def _is_noise_line(text: str) -> bool:
    """True if a non-empty OCR line is predominantly non-functional noise.

    A line is dropped only when the matched noise token(s) account for at
    least half its visible length. This drops standalone URLs/domains/emails
    ('bellisjoneshill.com'), watermarks, and footers, while keeping real
    label lines where a domain just happens to appear mid-sentence.
    """
    stripped = text.strip().lstrip("([{").strip()
    if not stripped:
        return False
    matches = list(_NOISE_RE.finditer(stripped))
    if not matches:
        return False
    covered = sum(len(m.group(0)) for m in matches if m.group(0))
    return covered * 2 >= len(stripped)


def filter_ocr_noise(segments: list[dict]) -> list[dict]:
    """
    Drop OCR segments whose visible text is predominantly incidental noise
    (bare URLs/domains, emails, watermarks, copyright/footer text, version
    strings). A domain or number appearing mid-sentence alongside functional
    text is kept, so genuine labels are not lost.

    Args:
        segments: Raw OCR segments [{start, end, text, ...}].
    Returns:
        A new list with noise segments removed.
    """
    return [seg for seg in segments if not _is_noise_line((seg.get("text") or ""))]