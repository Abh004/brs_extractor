"""
utils.py

Shared utilities for the pipeline.

Centralises text-cleaning helpers that were previously duplicated across
backend.py and config_generator.py:
  - _strip_special_tokens  (was copied in both files)
  - _strip_think_blocks    (was copied in both files)
  - clean_output           (was in backend.py only)
  - normalize_embeddings   (extracted from dedup.py to remove the
                            L2-norm duplication between _cosine_matrix
                            and _cosine_similarity_against_pool)
"""

from __future__ import annotations

import json
import re

import numpy as np

# ── compiled patterns (module-level — compiled once) ─────────────────────────

_SPECIAL_TOKENS = re.compile(
    r"<\|end\|>|<end_of_turn>|<\|im_end\|>|<\|eot_id\|>|<\|endoftext\|>"
    r"|<\|user\|>|<\|assistant\|>|<\|system\|>|<\|end_of_text\|>"
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Consistent VLM artefact: "real-self<word>" → "real-world"
_REAL_SELF = re.compile(r"\breal-self\w*", re.IGNORECASE)


# ── text cleaning ─────────────────────────────────────────────────────────────


def strip_special_tokens(text: str) -> str:
    """Remove known EOS / control tokens present in quantised model output."""
    return _SPECIAL_TOKENS.sub("", text).strip()


def strip_think_blocks(text: str) -> str:
    """Strip Qwen3-style <think>…</think> chain-of-thought blocks."""
    return _THINK_BLOCK.sub("", text).strip()


def is_truncated_think(text: str) -> bool:
    """Return True if output has an unclosed <think> block (truncated)."""
    return "<think>" in text and "</think>" not in text


def clean_output(text: str) -> str:
    """
    Full post-processing pass applied to every model output:
      1. Strip <think> blocks (Qwen3 CoT).
      2. Remove EOS / special tokens.
      3. Fix known VLM hallucination artefact: real-self* → real-world.
      4. Collapse 3+ consecutive newlines to 2.
    """
    text = strip_think_blocks(text)
    text = strip_special_tokens(text)
    text = _REAL_SELF.sub("real-world", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_spell = None  # lazy-loaded — pyspellchecker import is not free

# Common derivational suffixes. If stripping one of these from an unknown
# word yields a known root, treat the word as valid — pyspellchecker's
# dictionary is frequency-based and misses many correctly-derived words
# (e.g. "customizable" is unknown, but "custom" is known and "-izable" is
# a standard, productive suffix).
_SUFFIXES = (
    "izable",
    "able",
    "ible",
    "ization",
    "ing",
    "tion",
    "ment",
    "ness",
    "ed",
    "es",
    "s",
)


def _get_spellchecker():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker

        _spell = SpellChecker()
        # Domain terms that are correct but not in a general English dictionary
        _spell.word_frequency.load_words(
            [
                "vlm",
                "brs",
                "frs",
                "nfr",
                "moscow",
                "solidworks",
                "geometry",
                "geometries",
                "parametric",
                "lofting",
                "lofted",
                "sketching",
                "cad",
                "api",
                "ui",
                "ux",
                # Words pyspellchecker's frequency dictionary is missing
                # despite them being ordinary, correctly-spelled English —
                # confirmed via direct lookup, not garbling. See
                # _is_valid_compound for the general case (workflow,
                # toolchain, etc. are also caught there; this specific one
                # is listed explicitly since it isn't a clean compound).
                "customizable",
            ]
        )
    return _spell


def _is_valid_compound(word: str, spell) -> bool:
    """
    Return True if `word` is plausibly a valid English compound or
    derived word that pyspellchecker's frequency dictionary simply doesn't
    contain — e.g. "workflow" (work+flow), "toolchain" (tool+chain),
    "subcomponent" (sub+component), all confirmed real words whose parts
    ARE individually in the dictionary, unlike genuine garbling such as
    "stakeholdsel" or "real-selct" which don't split into two known words
    at any position.

    This is deliberately a soft heuristic, not a hard guarantee — the goal
    is reducing false positives on legitimate technical vocabulary, not
    catching every possible real word. Ambiguous cases fall through to
    still being flagged, which is the safer default (a human reviewer
    seeing one extra bracketed word is a minor inconvenience; a genuinely
    garbled word slipping through unflagged is worse).
    """
    lower = word.lower()

    # Suffix-strip: does removing a common suffix yield a known root?
    for suf in _SUFFIXES:
        if lower.endswith(suf) and len(lower) > len(suf) + 2:
            root = lower[: -len(suf)]
            if root in spell:
                return True

    # Compound-split: does splitting into two known words at any internal
    # position work? Require each half to be at least 3 chars so we don't
    # accept spurious splits like "a" + "nything".
    for i in range(3, len(lower) - 2):
        left, right = lower[:i], lower[i:]
        if left in spell and right in spell:
            return True

    return False


def flag_unknown_words(text: str) -> str:
    """
    Wrap words not found in an English dictionary (and not domain terms) in
    [[double brackets]] so they surface visually in review rather than being
    silently auto-corrected (auto-correct risks quietly changing the meaning
    of a requirement; flagging makes it a 30-second scan instead).

    Before flagging, checks _is_valid_compound as a false-positive filter:
    pyspellchecker's frequency-based dictionary misses many ordinary English
    compounds/derived words (confirmed via direct testing: "workflow",
    "toolchain", "subcomponent", "keypoint" are all absent despite being
    real words whose parts are individually recognized). Words passing that
    check are treated as valid and left unflagged.
    """
    spell = _get_spellchecker()
    words = re.findall(r"[A-Za-z']+", text)
    candidates = [
        w
        for w in words
        if len(w) > 3
        and w[0].islower()
        and w.lower() not in spell
        and not _is_valid_compound(w, spell)
    ]
    if not candidates:
        return text
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in candidates) + r")\b"

    def _bracket(m: re.Match) -> str:
        return f"[[{m.group(0)}]]"

    return re.sub(pattern, _bracket, text)


# ── JSON extraction ───────────────────────────────────────────────────────────


def extract_balanced_json(text: str) -> dict | None:
    """
    Pull the first COMPLETE {...} JSON object out of raw LLM output.

    Uses brace-depth counting rather than a greedy regex. Small models often
    echo fragments of their system prompt after answering (which itself
    contains example JSON with braces) — a greedy `\\{.*\\}` match would span
    from the real answer's opening brace all the way to the LAST closing
    brace anywhere in that trailing echo, stitching two JSON objects into one
    invalid blob. Scanning for the first balanced object avoids that, and
    also fails cleanly (returns None) on truncated output where the object
    never closes, rather than handing json.loads a corrupt fragment.

    Strips a single leading/trailing markdown code fence if present, since
    that's the other common wrapper small models add around JSON output.

    Used by dedup.py (cluster resolution), priority.py (batch priority
    classification), and config_generator.py (BRS config inference) — this
    used to be three separate, slightly different regex-based extractors;
    this is the one, more robust implementation all three now share.
    """
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None  # never closed — truncated output


# ── embedding helpers ─────────────────────────────────────────────────────────


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalise a (N, D) embedding matrix row-wise.
    Zero vectors are kept as-is (norm replaced with 1e-9 to avoid division
    by zero, producing a near-zero vector rather than NaN).

    Args:
        embeddings: (N, D) float array

    Returns:
        (N, D) float array where each row has unit L2 norm
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return embeddings / norms
