"""
Data preparation module for chunking Azure Video Indexer content.

The proposer is Azure Video Indexer: the judge is fed from on-screen OCR
text + transcript via data_prep.avi. The legacy local keyframe/OpenCV path
(extract_keyframes / load_transcript / prepare_data) was removed — the
pipeline has no OpenCV or YouTube-transcript dependency.
"""

from .avi import get_valid_avi_chunks, load_avi_chunks

__all__ = [
    "load_avi_chunks",
    "get_valid_avi_chunks",
]
