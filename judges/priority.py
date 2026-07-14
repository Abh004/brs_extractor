"""
priority.py

MoSCoW priority classification for extracted requirement bullets. Runs as
part of Stage 5/6, after cross-block dedup, on the final flat bullet list.

Batched (not per-bullet) to keep this cheap — one LLM call per ~15 bullets.
"""

from __future__ import annotations

from config import DEDUP_MODEL
from utils import extract_balanced_json

_VALID = {"M", "S", "C"}

_PRIORITY_SYSTEM = """You are a business analyst doing MoSCoW prioritization \
on software requirements extracted from a product demo video.

Every requirement was generated from the template "The system must enable \
[user] to [goal] in order to [reason]." IGNORE the word "must" in that \
template — it is boilerplate phrasing, not a signal of priority. Judge \
actual business criticality:
- M (Must Have): core functionality; the product doesn't work without it.
- S (Should Have): a clear, important improvement most users would expect, \
  but the product is still usable without it.
- C (Could Have): a nice-to-have, edge case, or minor convenience.

Output ONLY valid JSON, no markdown fences, no explanation, in this exact \
shape: {"priorities": ["M", "S", "C", ...]}
The list must have exactly as many entries as input bullets, in the same \
order."""


def classify_priorities(
    bullets: list[str],
    backend,
    batch_size: int = 15,
    handle=None,
) -> list[str]:
    """
    Classify each bullet as M/S/C. Returns a list the same length and order
    as `bullets`. On any per-batch failure (bad JSON, wrong length), failed
    entries default to "S" rather than "M" — defaulting to "M" would
    silently reproduce the all-Must-Have problem this function exists to fix.

    Args:
        bullets:    Flat list of requirement bullet strings to classify.
        backend:    InferenceBackend instance.
        batch_size: Bullets per LLM call.
        handle:     Optional pre-loaded handle for JUDGE3_MODEL. If given,
                    this function uses it as-is and does NOT load or unload
                    it — the caller owns that lifecycle (e.g. main.py sharing
                    one handle across priority classification AND config
                    generation, both of which use JUDGE3_MODEL, to avoid
                    loading the same model twice back-to-back). If None
                    (default), the previous self-contained behavior is
                    preserved: this function loads JUDGE3_MODEL itself and
                    unloads it before returning.
    """
    if not bullets:
        return []

    owns_handle = handle is None
    if owns_handle:
        handle = backend.load_llm(DEDUP_MODEL)

    out: list[str] = []
    try:
        for i in range(0, len(bullets), batch_size):
            batch = bullets[i : i + batch_size]
            numbered = "\n".join(f"{n}. {b}" for n, b in enumerate(batch))
            raw = backend.generate_llm(
                handle=handle,
                system_prompt=_PRIORITY_SYSTEM,
                user_prompt=numbered,
                max_tokens=120,
            )
            parsed_obj = extract_balanced_json(raw)
            parsed = parsed_obj.get("priorities", []) if parsed_obj else []

            parsed = [p if p in _VALID else "S" for p in parsed]
            if len(parsed) < len(batch):
                print(
                    f"  [priority] batch {i}-{i + len(batch)}: got "
                    f"{len(parsed)}/{len(batch)} priorities, padding with S."
                )
                parsed += ["S"] * (len(batch) - len(parsed))
            elif len(parsed) > len(batch):
                parsed = parsed[: len(batch)]

            out.extend(parsed)
    finally:
        if owns_handle:
            backend.unload_llm(handle)

    return out
