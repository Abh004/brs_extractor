"""
models/avi_proposer.py

AVIProposer — the cloud "proposer" that replaces the local VLM. It does not
run any on-device model. Instead it assembles the raw per-chunk context
(OCR screen-text + transcript) that the judge LLM converts into requirement
bullets.

Kept as a thin object so pipeline_runner's Stage-3 loop reads identically to
how it drove the old local VLM proposer:
    proposer.generate(chunk) -> raw_model_output string
"""

from __future__ import annotations

from config import AVI_KEYFRAME_SECTION_TEMPLATE, AVI_RAW_TEMPLATE


class AVIProposer:
    """Assemble raw OCR + transcript context for one chunk (no GPU inference)."""

    def __init__(self, index: dict | None = None) -> None:
        self._index = index

    @staticmethod
    def render(chunk: dict) -> str:
        """
        Compose the single raw context string for a chunk.

        Prepend the optional VLM keyframe screen-analysis block (component
        placement) when present, then OCR + transcript. Kept as a static helper
        so the pipeline can estimate input tokens with the EXACT text the judge
        will receive.

        Args:
            chunk: A per-bucket chunk dict {text_lines, ocr_lines} and optional
                   ui_description, from data_prep.avi.load_avi_chunks /
                   build_avi_single_pass_chunk.

        Returns:
            Formatted VLM screen analysis (if any) + OCR + transcript text.
        """
        screen_text = "\n".join(chunk.get("ocr_lines") or []) or "(no on-screen text)"
        transcript = "\n".join(chunk.get("text_lines") or []) or "(no speech)"
        body = AVI_RAW_TEMPLATE.format(
            screen_text=screen_text,
            transcript=transcript,
        ).strip()
        ui = chunk.get("ui_description")
        if ui:
            block = AVI_KEYFRAME_SECTION_TEMPLATE.format(ui_description=ui).strip()
            body = f"{block}\n\n{body}"
        return body

    def generate(self, chunk: dict) -> str:
        """
        Build the raw context string for a chunk.

        Args:
            chunk: A per-bucket chunk dict {text_lines, ocr_lines} from
                   data_prep.avi.load_avi_chunks.

        Returns:
            Formatted OCR + transcript text (the judge's input).
        """
        return self.render(chunk)

    def unload(self) -> None:
        """No-op — the AVI proposer holds no local model."""
