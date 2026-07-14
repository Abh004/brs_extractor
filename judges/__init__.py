"""
Judges panel — refines VLM output through three sequential LLM judges.

All judge models are loaded ONCE after the VLM proposer is unloaded and
reused across all chunks. Merging is done via embedding-based cosine
deduplication (judges/dedup.py) — no merger LLM.

Memory flow:
  preload()   — validates/downloads models, loads nothing into RAM.
  load_all()  — loads J1+J2+J3 via the backend (~2GB total on MLX).
  evaluate()  — runs all three judges on one chunk's VLM output.
  unload_all()— frees all judge handles and clears accelerator cache.

NOTE on parallelism: MLX GPU streams are thread-local. ThreadPoolExecutor
workers cannot call generate() — judges run sequentially.
"""

from .panel import JudgesPanel

__all__ = [
    "JudgesPanel",
]
