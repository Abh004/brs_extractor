"""
models/avi_proposer.py

AVIProposer — the cloud "proposer" that replaces the local VLM. It does not
run any on-device model. Instead it assembles the raw per-chunk context
(OCR screen-text + transcript) that the judge LLM converts into requirement
bullets.

Kept as a thin object so pipeline_runner's Stage-3 loop reads identically to
how it drove VLMProposer:
    proposer.generate(chunk) -> raw_model_output string
"""

from __future__ import annotations

from config import AVI_RAW_TEMPLATE


class AVIProposer:
    """Assemble raw OCR + transcript context for one chunk (no GPU inference)."""

    def __init__(self, index: dict | None = None) -> None:
        self._index = index

    def generate(self, chunk: dict) -> str:
        """
        Build the raw context string for a chunk.

        Args:
            chunk: A per-bucket chunk dict {text_lines, ocr_lines} from
                   data_prep.avi.load_avi_chunks.

        Returns:
            Formatted OCR + transcript text (the judge's input).
        """
        screen_text = "\n".join(chunk.get("ocr_lines") or []) or "(no on-screen text)"
        transcript = "\n".join(chunk.get("text_lines") or []) or "(no speech)"
        return AVI_RAW_TEMPLATE.format(
            screen_text=screen_text,
            transcript=transcript,
        ).strip()

    def unload(self) -> None:
        """No-op for interface parity with VLMProposer."""