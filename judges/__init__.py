"""
Judges panel — refines VLM output through single LLM judges.

All judge models are loaded ONCE after the VLM proposer is unloaded and
reused across all chunks. Merging is done via Jaccard-prefiltered LLM
deduplication (judges/dedup.py) — no merger LLM.

NOTE on parallelism: MLX GPU streams are thread-local. ThreadPoolExecutor
workers cannot call generate() — judges run sequentially.
"""

from .panel import JudgesPanel

__all__ = [
    "JudgesPanel",
]
