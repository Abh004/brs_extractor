"""
Models module — AVI proposer (cloud, no local VLM).

The proposer role is now filled by Azure Video Indexer: it assembles OCR
screen-text + transcript per chunk, and the judge LLM converts that raw
context into requirement bullets. No local GPU VLM is loaded.
"""

from .avi_proposer import AVIProposer

__all__ = [
    "AVIProposer",
]
