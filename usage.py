"""
usage.py

Exact token-usage accounting for the inference pipeline.

Every LLM generation in the project funnels through the backend's
generate_llm method (backend.py), so a single TokenTracker
attached to the InferenceBackend can record a per-call token count without
touching any call sites. The backend exposes exact prompt/completion counts
(from the Foundry usage payload) for precise accounting.

Stages are labelled with a small context-manager so records carry the pipeline
stage that produced them (Stage 3 VLM, Stage 4 judge, Stage 5 dedup/priority,
Stage 6 config). The pipeline is single-threaded, so a plain instance
attribute is sufficient and safe.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class UsageRecord:
    """One model generation call and its exact token counts."""

    model_id: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    n_calls: int = 1


def _aggregate(
    totals: dict[str, list[int]],
) -> dict[str, dict]:
    """Convert {name: [calls, prompt, completion]} → {name: totals-dict}."""
    out: dict[str, dict] = {}
    for name, (calls, prompt, completion) in totals.items():
        out[name] = {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return out


@dataclass
class TokenTracker:
    """
    Accumulates exact token usage across a pipeline run.

    Attach an instance to an InferenceBackend (backend.usage). The backend's
    generate_llm appends UsageRecords; pipeline stages set the
    active stage label via the `stage()` context manager.
    """

    records: list[UsageRecord] = field(default_factory=list)
    _current_stage: str = "default"

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Set the active stage label for the duration of the block."""
        previous = self._current_stage
        self._current_stage = name
        try:
            yield
        finally:
            self._current_stage = previous

    def add(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record one generation call, tagged with the active stage."""
        self.records.append(
            UsageRecord(
                model_id=model_id,
                stage=self._current_stage,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    # ── aggregation ───────────────────────────────────────────────────────────

    def by_model(self) -> dict[str, dict]:
        totals: dict[str, list[int]] = {}
        for r in self.records:
            agg = totals.setdefault(r.model_id, [0, 0, 0])
            agg[0] += r.n_calls
            agg[1] += r.prompt_tokens
            agg[2] += r.completion_tokens
        return _aggregate(totals)

    def by_stage(self) -> dict[str, dict]:
        totals: dict[str, list[int]] = {}
        for r in self.records:
            agg = totals.setdefault(r.stage, [0, 0, 0])
            agg[0] += r.n_calls
            agg[1] += r.prompt_tokens
            agg[2] += r.completion_tokens
        return _aggregate(totals)

    def total(self) -> dict:
        calls = sum(r.n_calls for r in self.records)
        prompt = sum(r.prompt_tokens for r in self.records)
        completion = sum(r.completion_tokens for r in self.records)
        return {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def summary(self) -> dict:
        """
        Aggregate records by model and by stage.

        Returns {"models": {...}, "stages": {...}, "total": {...}} where each
        totals dict has keys calls / prompt_tokens / completion_tokens /
        total_tokens.
        """
        return {
            "models": self.by_model(),
            "stages": self.by_stage(),
            "total": self.total(),
        }

    def to_console(self) -> None:
        """Print a human-readable token usage table to stdout."""
        print("\n" + "=" * 68)
        print("  TOKEN USAGE")
        print("=" * 68)

        rows: list[tuple[str, dict]] = [
            *[(f"model  {name}", agg) for name, agg in self.by_model().items()],
            *[(f"stage  {name}", agg) for name, agg in self.by_stage().items()],
            ("TOTAL", self.total()),
        ]

        header = f"{'Name':<30}{'Calls':>6}{'Prompt':>10}{'Completion':>12}{'Total':>10}"
        print(header)
        print("-" * len(header))
        for name, agg in rows:
            print(
                f"{name:<30}{agg['calls']:>6}{agg['prompt_tokens']:>10}"
                f"{agg['completion_tokens']:>12}{agg['total_tokens']:>10}"
            )
        print("=" * 68)