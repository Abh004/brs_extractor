from __future__ import annotations

from backend import InferenceBackend, get_backend
from config import (
    AVI_JUDGE_SYSTEM,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
)
from judges.dedup import (
    extract_bullets,
    filter_meta_bullets,
    filter_substrings,
    filter_truncated_bullets,
)


def _validate_bullets(text: str) -> str:
    bullets = extract_bullets(text)
    bullets = filter_meta_bullets(bullets)
    bullets = filter_truncated_bullets(bullets)
    bullets = filter_substrings(bullets)  # catch within-chunk near-dupes too
    return "\n".join(f"- {b}" for b in bullets) if bullets else ""


class JudgesPanel:
    """Manages the single LLM judge that converts/refines proposer output."""

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        system_prompt: str = AVI_JUDGE_SYSTEM,
    ) -> None:
        self._backend = backend or get_backend()
        self._handle = None
        self._system_prompt = system_prompt

    def preload(self) -> None:
        """Pre-download judge weights (backend-aware; no-op for remote backends)."""
        self._backend.preload(JUDGE_MODEL)

    def load_all(self) -> None:
        print("Loading judge model...")
        self._handle = self._backend.load_llm(JUDGE_MODEL)
        print("Judge loaded.\n")

    def unload_all(self) -> None:
        if self._handle is not None:
            self._backend.unload_llm(self._handle)
            self._handle = None
        self._backend.clear_cache()
        print("Judge unloaded.")

    def evaluate(self, raw_output: str, max_tokens: int | None = None) -> str:
        raw = self._backend.generate_llm(
            handle=self._handle,
            system_prompt=self._system_prompt,
            user_prompt=raw_output,
            max_tokens=max_tokens or JUDGE_MAX_TOKENS,
        )
        return _validate_bullets(raw)
