from __future__ import annotations

import re

from backend import InferenceBackend, get_backend
from config import (
    JUDGE1_MODEL,
    JUDGE1_SYSTEM,
    JUDGE2_MODEL,
    JUDGE2_SYSTEM,
    JUDGE3_MODEL,
    JUDGE3_SYSTEM,
)
from judges.dedup import filter_substrings, merge_judge_outputs
from utils import clean_output

_JUDGE_MAX_TOKENS = 150
_MAX_BULLETS_PER_JUDGE = 3


def _validate_bullets(text: str, max_bullets: int = _MAX_BULLETS_PER_JUDGE) -> str:
    """
    Clean known model artefacts, extract bullet lines via regex, and cap at
    max_bullets. Allows using a lower max_tokens since we don't rely on the
    model to stop.
    """
    text = clean_output(text)
    bullets = re.findall(r"^\s*[-*•]\s+.+$", text, re.MULTILINE)
    bullets = [b.strip() for b in bullets[:max_bullets]]
    return "\n".join(bullets) if bullets else text.strip()


class JudgesPanel:
    """
    Manages three LLM judges that refine VLM proposer output.

    Args:
        backend: Shared InferenceBackend. If None, one is created via
                 get_backend() (reads the BACKEND env var).
    """

    def __init__(self, backend: InferenceBackend | None = None) -> None:
        self._backend = backend or get_backend()
        # Each entry: {"handle": <backend handle>, "system": <str>, "model_id": <str>}
        self._judges: list[dict] = []

    def preload(self) -> None:
        """
        Validate all judge models are downloaded WITHOUT loading into RAM.
        Called during Stage 2 while the VLM is still loaded.
        """
        from huggingface_hub import snapshot_download

        for label, model_id in [
            ("Judge 1", JUDGE1_MODEL),
            ("Judge 2", JUDGE2_MODEL),
            ("Judge 3", JUDGE3_MODEL),
        ]:
            try:
                snapshot_download(model_id, local_files_only=False)
                print(f"  ✓ {label}: {model_id}")
            except Exception as exc:
                raise RuntimeError(f"{label} model unavailable: {model_id}") from exc

    def load_all(self) -> None:
        """Load J1, J2, J3. Call AFTER proposer.unload()."""
        print("Loading all judge models...")
        for model_id, system in [
            (JUDGE1_MODEL, JUDGE1_SYSTEM),
            (JUDGE2_MODEL, JUDGE2_SYSTEM),
            (JUDGE3_MODEL, JUDGE3_SYSTEM),
        ]:
            handle = self._backend.load_llm(model_id)
            self._judges.append(
                {"handle": handle, "system": system, "model_id": model_id}
            )
        print("All judges loaded.\n")

    def unload_all(self) -> None:
        """Free all judge handles and clear accelerator cache. Idempotent."""
        for judge in self._judges:
            self._backend.unload_llm(judge["handle"])
        self._judges.clear()
        self._backend.clear_cache()
        print("All judges unloaded.")

    def evaluate(self, vlm_output: str) -> str:
        """
        Run J1, J2, J3 sequentially on the VLM output, validate bullets,
        then merge via embedding-prefilter + LLM cluster-and-canonicalize
        deduplication (see judges/dedup.py).

        Args:
            vlm_output: Raw string output from the VLM proposer.

        Returns:
            Merged, deduplicated markdown bullet list string.
        """
        results = []
        for judge in self._judges:
            raw = self._backend.generate_llm(
                handle=judge["handle"],
                system_prompt=judge["system"],
                user_prompt=vlm_output,
                max_tokens=_JUDGE_MAX_TOKENS,
            )
            results.append(_validate_bullets(raw))

        j1_out, j2_out, j3_out = results
        dedup_handle = self._judges[0]["handle"]

        try:
            return merge_judge_outputs(
                j1_out,
                j2_out,
                j3_out,
                backend=self._backend,
                llm_handle=dedup_handle,
                max_bullets=5,
            )
        except Exception as exc:
            print(
                f"    [panel] merge/dedup failed ({exc}); falling back to raw judge bullets."
            )
            from judges.dedup import extract_bullets

            all_b = filter_substrings(
                extract_bullets(j1_out)
                + extract_bullets(j2_out)
                + extract_bullets(j3_out)
            )
            return "\n".join(f"- {b}" for b in all_b[:7]) if all_b else None
