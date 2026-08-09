"""
Data preparation module for chunking video and transcript data.

The proposer is now Azure Video Indexer: the judge is fed from on-screen OCR
text + transcript via data_prep.avi. The legacy local keyframe/OpenCV path
(keyframes, pipeline) is intentionally NOT imported here so the main pipeline
has no hard OpenCV dependency. Import it explicitly if a local-frames path is
ever needed again:
    from data_prep.keyframes import extract_keyframes
"""

from .avi import get_valid_avi_chunks, load_avi_chunks
from .transcript import load_transcript

__all__ = [
    "load_transcript",
    "load_avi_chunks",
    "get_valid_avi_chunks",
]
