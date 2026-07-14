"""
Models module — VLM proposer.

Delegates all inference to the shared InferenceBackend instance injected
at construction time. No direct imports of mlx_vlm or transformers here.
"""

from .proposer import VLMProposer

__all__ = [
    "VLMProposer",
]
