"""
Data preparation module for chunking video and transcript data.
Handles video frame extraction and transcript chunking.

Frame extraction resizes each frame to FRAME_SIZE (448×448, Qwen3-VL native)
at save time so the VLM never rescales internally.

The keyframe cache is video-aware: a .video_id marker file records which
VIDEO_ID the cached frames belong to. If the ID changes, the cache is
cleared automatically.
"""

from .keyframes import extract_keyframes
from .pipeline import get_valid_chunks, prepare_data
from .transcript import load_transcript

__all__ = [
    "load_transcript",
    "extract_keyframes",
    "prepare_data",
    "get_valid_chunks",
]
