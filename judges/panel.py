from __future__ import annotations

from backend import InferenceBackend, get_backend
from config import JUDGE_MAX_BULLETS, JUDGE_MAX_TOKENS, JUDGE_MODEL, JUDGE_SYSTEM
from judges.dedup import (
    extract_bullets,
    filter_meta_bullets,
    filter_substrings,
    filter_truncated_bullets,
)


def _validate_bullets(text: str, max_bullets: int = JUDGE_MAX_BULLETS) -> str:
    bullets = extract_bullets(text)
    bullets = filter_meta_bullets(bullets)
    bullets = filter_truncated_bullets(bullets)
    bullets = filter_substrings(bullets)  # catch within-chunk near-dupes too
    bullets = bullets[:max_bullets]
    return "\n".join(f"- {b}" for b in bullets) if bullets else ""


class JudgesPanel:
    """Manages the single LLM judge that refines VLM proposer output."""

    def __init__(self, backend: InferenceBackend | None = None) -> None:
        self._backend = backend or get_backend()
        self._handle = None

    def preload(self) -> None:
        from huggingface_hub import snapshot_download

        # Check local cache first — instant if cached, no progress bar
        try:
            snapshot_download(JUDGE_MODEL, local_files_only=True)
            print(f"  ✓ Judge (cached): {JUDGE_MODEL}")
            return
        except Exception:
            pass

        try:
            snapshot_download(JUDGE_MODEL, local_files_only=False)
            print(f"  ✓ Judge: {JUDGE_MODEL}")
        except Exception as e:
            raise RuntimeError(
                f"Judge model unavailable (remote unreachable and no local cache): "
                f"{JUDGE_MODEL}"
            ) from e

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

    def evaluate(self, vlm_output: str) -> str:
        raw = self._backend.generate_llm(
            handle=self._handle,
            system_prompt=JUDGE_SYSTEM,
            user_prompt=vlm_output,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        return _validate_bullets(raw)
