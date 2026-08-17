"""
Judges panel — refines VLM output through single LLM judges.

All judge calls use the single Foundry backend (loaded ONCE after the AVI
proposer stage) and are reused across all chunks. Merging is done via
Jaccard-prefiltered LLM deduplication (judges/dedup.py) — no merger LLM.

The pipeline is single-threaded, so judge evaluation runs sequentially.
"""

from .panel import JudgesPanel

__all__ = [
    "JudgesPanel",
]
