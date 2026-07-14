"""
config_generator.py

Infers all docx configuration fields from the extracted requirement bullets
and writes project_config.json. Runs as Stage 6.

The backend is injected so the same backend instance used throughout the
pipeline is reused here — no redundant re-imports or re-initialisation.
"""

from __future__ import annotations

import json
import os

from backend import InferenceBackend, get_backend
from config import JUDGE3_MODEL
from utils import clean_output, extract_balanced_json, strip_think_blocks

_SYSTEM_PROMPT = """\
You are a senior business analyst. Given a list of software requirements \
extracted from a product video, infer the document configuration fields \
for a Business Requirements Specification (BRS).

Rules:
- Output ONLY valid JSON. No preamble, no explanation, no markdown fences.
- Infer everything from the requirements — do not make up features not implied.
- categories must group the provided requirements meaningfully. \
  Each category needs 3-8 keywords that match actual words in the requirements.
- non_functional_requirements must be inferred from patterns in the requirements \
  (e.g. if requirements mention real-time updates → performance NFR).
- Keep all text fields concise and professional.
- definitions should include technical terms that appear in the requirements \
  and may need explanation for stakeholders.
"""

_USER_TEMPLATE = """\
Here are the extracted requirements:

{bullets}

Output this exact JSON structure with all fields filled in:
{{
  "project_name": "short name of the software product",
  "purpose": "1-2 sentence purpose of this BRS document",
  "scope": "2-3 sentences describing what the system covers",
  "user_classes": [
    "Role Name: description of what this user does"
  ],
  "operating_environment": [
    "environment bullet 1"
  ],
  "categories": [
    {{
      "id": "01",
      "title": "Category Title",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ],
  "non_functional_requirements": {{
    "Usability": ["NFR bullet 1"],
    "Performance": ["NFR bullet 1"],
    "Compatibility": ["NFR bullet 1"]
  }},
  "constraints": [
    "constraint 1"
  ],
  "assumptions": [
    "assumption 1"
  ]
}}\
"""


def _parse_config_response(text: str) -> dict | None:
    """
    Strip thinking blocks and special tokens, then extract the first
    balanced JSON object (brace-depth counting, not a greedy regex — see
    utils.extract_balanced_json). Returns None if no valid, complete JSON
    object was found (including the truncated-output case).
    """
    text = strip_think_blocks(text)
    text = clean_output(text)  # strips special tokens + real-self artefacts
    return extract_balanced_json(text)


def _build_bullet_summary(all_bullets: list[str], max_bullets: int = 60) -> str:
    if len(all_bullets) <= max_bullets:
        selected = all_bullets
    else:
        step = len(all_bullets) / max_bullets
        selected = [all_bullets[int(i * step)] for i in range(max_bullets)]
    return "\n".join(f"- {b}" for b in selected)


def generate_config(
    all_bullets: list[str],
    config_path: str,
    backend: InferenceBackend | None = None,
    handle=None,
) -> dict:
    """
    Run LLM inference to infer docx config from the bullet list.
    Writes the result to config_path as JSON and returns the parsed dict.

    Args:
        all_bullets: Flat list of all refined requirement bullets.
        config_path: Path to write project_config.json.
        backend:     InferenceBackend to use. If None, one is created via
                     get_backend() (reads BACKEND env var).
        handle:      Optional pre-loaded handle for JUDGE3_MODEL. If given,
                     this function uses it as-is and does NOT load or unload
                     it — the caller owns that lifecycle (e.g. main.py
                     sharing one handle across priority classification and
                     config generation, both of which use JUDGE3_MODEL, to
                     avoid loading the same model twice back-to-back). If
                     None (default), the previous self-contained behavior is
                     preserved: this function loads JUDGE3_MODEL itself and
                     unloads it before returning.

    Returns:
        The parsed config dict (also written to config_path).
    """
    _backend = backend or get_backend()

    print(f"\n[STAGE 6] Generating BRS config from {len(all_bullets)} requirements...")

    owns_handle = handle is None
    if owns_handle:
        handle = _backend.load_llm(JUDGE3_MODEL)

    bullet_summary = _build_bullet_summary(all_bullets)
    user_content = _USER_TEMPLATE.format(bullets=bullet_summary)

    print("  Inferring document structure...")
    raw = _backend.generate_llm(
        handle=handle,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_content,
        max_tokens=1200,
    )

    if owns_handle:
        _backend.unload_llm(handle)
        print("  Config model unloaded.")

    cfg = _parse_config_response(raw)
    if cfg is None:
        print("  WARNING: LLM config response was not valid/complete JSON.")
        print(f"  Raw response (first 500 chars): {raw[:500]}")
        print("  Falling back to minimal default config.")
        cfg = _default_config(all_bullets)

    cfg.setdefault("version", "1.0")
    cfg.setdefault("status", "Draft")
    cfg.setdefault("prepared_by", "John")
    cfg.setdefault("confidentiality", "None")

    _ensure_base_definitions(cfg)

    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"  Config written to {config_path}")
    print(f"  Categories inferred: {len(cfg.get('categories', []))}")
    return cfg


_BASE_DEFINITIONS: list[list[str]] = [
    ["FR", "Functional Requirement"],
    ["NFR", "Non-Functional Requirement"],
]


def _default_config(bullets: list[str]) -> dict:
    return {
        "project_name": "Software System",
        "purpose": (
            "This document specifies the functional and non-functional requirements "
            "for the system under analysis."
        ),
        "scope": (
            "Requirements were automatically extracted from video content. "
            "This document covers the observed functionality of the system."
        ),
        "user_classes": [
            "End Users: interact with the system to complete primary tasks"
        ],
        "operating_environment": [
            "Desktop or web-based application",
            "Standard input devices (mouse, keyboard)",
        ],
        "categories": [
            {"id": "FR-01", "title": "General Requirements", "keywords": []}
        ],
        "non_functional_requirements": {
            "Usability": [
                "The system must provide clear feedback for all user actions."
            ],
            "Performance": [
                "The system must respond to user interactions without perceptible delay."
            ],
        },
        "constraints": [
            "Requirements were extracted from video content and may not be exhaustive.",
            "Priority defaults to Must Have pending stakeholder review.",
        ],
        "assumptions": ["Users have basic familiarity with the application domain."],
    }


def _ensure_base_definitions(cfg: dict) -> None:
    existing_terms = {d[0].upper() for d in cfg.get("definitions", [])}
    prepend = [d for d in _BASE_DEFINITIONS if d[0].upper() not in existing_terms]
    cfg["definitions"] = prepend + cfg.get("definitions", [])
