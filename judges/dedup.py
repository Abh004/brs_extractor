"""
Bullet deduplication: Jaccard prefilter + LLM cluster-and-canonicalize.

Within each time block, drop bullets that are semantically equivalent after
the cheap per-chunk filters have run.

Dedup pipeline per location:
  extract_bullets
  → filter_meta_bullets        (drop system-prompt echoes)
  → filter_truncated_bullets   (drop mid-sentence cutoffs)
  → filter_substrings          (drop bullets that are subsets of longer ones)
  → cluster_candidates         (cheap Jaccard prefilter, high recall)
  → llm_resolve_clusters       (LLM judges + canonicalizes each candidate cluster)
"""

from __future__ import annotations

import re

import numpy as np

from utils import extract_balanced_json
from config import DEDUP_MAX_TOKENS


# Known meta-instruction patterns that judges echo from their system prompts.
_META_PATTERNS = re.compile(
    r"(no duplicate|prevent duplicate|maximum \d+ bullet|plain text only|"
    r"no preamble|clear and concise|eliminate vague|remove duplicate|"
    r"vague statement|no explanation|focus on clear|actionable requirement|"
    r"provide clear instructions|provide feedback|help users understand|"
    r"do not include|user.facing functionality|without preamble)",
    re.IGNORECASE,
)

# Bullets ending with these words are likely truncated mid-sentence.
_TRUNCATION_ENDINGS = re.compile(
    r"\b(the|a|an|and|or|for|of|with|in|to|at|by|from|on|its|their|this|that|"
    r"is|are|was|were|be|been|being|selected|specified|limited|restricted|"
    r"defined|configured|based|applied|used|set|given|provided|allowed|"
    r"enabled|supported|ensured|included|excluded)\s*$",
    re.IGNORECASE,
)

# Jaccard prefilter threshold — deliberately LOW (high recall, low precision). Its only job is "could these plausibly be duplicates?" so we don't burn an LLM call on every single pair. The shared "The system shall" boilerplate carries little Jaccard weight, so no prefix stripping is needed here.
CANDIDATE_THRESHOLD = 0.40


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens — the prefilter measure."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


_DEDUP_SYSTEM_PROMPT = """You are deduplicating a list of software requirement bullets that describe the same underlying system.

You will be given a numbered list of requirement bullets that an embedding model has flagged as POSSIBLY describing the same requirement. Some are true duplicates (same requirement, different wording). Some are genuinely distinct requirements that happen to share vocabulary or topic. Your job is to tell them apart.

Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact shape:
{"clusters": [[0, 2], [1], [3, 4, 5]]}

Rules:
- Each inner list is a group of indices (0-based, matching the numbered list given) that describe the EXACT SAME requirement, just worded differently.
- A bullet that is NOT a duplicate of anything else gets its own single-item list.
- Every index from the input must appear in exactly one cluster.
- Be conservative: only group bullets if a reviewer would say "these are the same requirement." Sharing a topic (e.g. both about 3D navigation) is not enough — they must express the same specific requirement.
- Output nothing except the JSON object."""


def extract_bullets(text: str) -> list[str]:
    """Extract bullet text from judge output or merged markdown."""
    if not text:
        return []
    bullets = re.findall(
        r"^\s*(?:[-*•]|\d+[.)]) +(.+)$",
        text,
        re.MULTILINE,
    )
    return [b.strip() for b in bullets if b.strip()]


def filter_meta_bullets(bullets: list[str]) -> list[str]:
    """Drop bullets that echo the judge's own system-prompt instructions."""
    return [b for b in bullets if not _META_PATTERNS.search(b)]


def filter_truncated_bullets(bullets: list[str]) -> list[str]:
    """Drop bullets that appear cut off mid-sentence."""
    kept = []
    for b in bullets:
        if re.search(r"[.!?)\]\"']$", b):
            kept.append(b)
            continue
        if _TRUNCATION_ENDINGS.search(b):
            continue
        kept.append(b)
    return kept


def filter_substrings(bullets: list[str]) -> list[str]:
    """
    Drop bullets whose full text is contained within another bullet in the
    same list. Comparison is case-insensitive after punctuation normalisation.

    The longer bullet is always kept; the shorter substring is dropped.
    When two bullets are mutual substrings (identical), the first is kept.
    """
    if len(bullets) <= 1:
        return bullets

    def _norm(s: str) -> str:
        return re.sub(r"['''\"]", "", s.lower()).strip()

    indexed = sorted(
        [(i, _norm(b)) for i, b in enumerate(bullets)],
        key=lambda x: -len(x[1]),
    )
    kept: list[tuple[int, str]] = []
    for idx, norm in indexed:
        if not any(norm in other_norm for _, other_norm in kept):
            kept.append((idx, norm))
    return [bullets[i] for i, _ in kept]


# Candidate clustering (Jaccard prefilter — recall, not precision)
def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find over `n` nodes with the given edge list. Returns components
    as lists of indices, each sorted ascending, components sorted by their
    smallest index (i.e. stable w.r.t. input order)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    components = sorted(groups.values(), key=lambda g: g[0])
    return [sorted(g) for g in components]


def cluster_candidates(
    bullets: list[str], threshold: float = CANDIDATE_THRESHOLD
) -> list[list[int]]:
    """
    Group bullet indices into candidate-duplicate clusters using a
    deliberately lenient Jaccard threshold. This is a RECALL step: it only
    needs to make sure plausible duplicates end up in the same cluster so
    the LLM pass can look at them together. Precision is handled downstream.

    Singletons (no neighbour above threshold) are returned as single-item
    clusters and do not need an LLM call at all.

    Args:
        bullets:   Bullet text strings (no prefix)
        threshold: Jaccard similarity cutoff for "worth checking together"

    Returns:
        List of index-lists into `bullets` — candidate duplicate clusters.
    """
    n = len(bullets)
    if n <= 1:
        return [[i] for i in range(n)]

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = _jaccard_similarity(bullets[i], bullets[j])
            sim[i, j] = sim[j, i] = s

    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold
    ]
    return _connected_components(n, edges)


def _pick_canonical(bullets: list[str], indices: list[int]) -> str:
    """
    Choose the representative bullet text for a confirmed duplicate cluster.
    Picks the longest bullet (most detail / specificity retained) rather than
    asking the LLM to rewrite, to avoid introducing fabricated detail.
    """
    return max((bullets[i] for i in indices), key=len)


def llm_resolve_clusters(
    bullets: list[str],
    clusters: list[list[int]],
    backend,
    llm_handle,
    max_tokens: int = DEDUP_MAX_TOKENS,
) -> list[str]:
    """
    For each candidate cluster with more than one bullet, ask the LLM to
    confirm which members are true duplicates (it may split a cluster into
    several real ones) and return one canonical bullet per confirmed group.
    Singleton clusters pass through untouched with no LLM call.

    Args:
        bullets:    Full bullet list (post cheap-filter stage)
        clusters:   Candidate clusters from cluster_candidates()
        backend:    InferenceBackend instance (see backend.py)
        llm_handle: Handle returned by backend.load_llm()
        max_tokens: Generation budget for the clustering response

    Returns:
        Deduplicated bullet list, original relative order preserved.
    """
    output: list[tuple[int, str]] = []  # (first_index, text) for ordering

    for cluster in clusters:
        if len(cluster) == 1:
            idx = cluster[0]
            output.append((idx, bullets[idx]))
            continue

        numbered = "\n".join(f"{n}. {bullets[i]}" for n, i in enumerate(cluster))
        user_prompt = (
            f"Bullets (use these numbers in your answer, NOT the original "
            f"list positions):\n{numbered}"
        )

        effective_max_tokens = max(max_tokens, 30 * len(cluster))

        raw = None
        try:
            raw = backend.generate_llm(
                llm_handle,
                _DEDUP_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=effective_max_tokens,
            )
            parsed = extract_balanced_json(raw)
            sub_clusters = parsed.get("clusters") if parsed else None
            if not sub_clusters:
                raise ValueError("no clusters in LLM response")

            # sub_clusters are indices into `cluster` (the local numbering we gave the LLM) — map back to indices into the original `bullets`.
            seen_local: set[int] = set()
            for sub in sub_clusters:
                sub = [int(x) for x in sub if isinstance(x, (int, float, str))]
                sub = [x for x in sub if 0 <= x < len(cluster)]
                if not sub:
                    continue
                seen_local.update(sub)
                global_indices = [cluster[x] for x in sub]
                canonical = _pick_canonical(bullets, global_indices)
                output.append((min(global_indices), canonical))

            # Safety net: if the LLM dropped any local index, pass it through
            missing_local = set(range(len(cluster))) - seen_local
            for local_idx in missing_local:
                global_idx = cluster[local_idx]
                output.append((global_idx, bullets[global_idx]))

            display_groups = []
            for sub in sub_clusters:
                valid = [
                    x
                    for x in sub
                    if isinstance(x, (int, float, str)) and 0 <= int(x) < len(cluster)
                ]
                display_groups.append(
                    f"[{', '.join(str(bullets[cluster[int(x)]])[:40] for x in valid)}]"
                )
            print(
                f"  [dedup] adjudicated [{len(cluster)} bullets] -> "
                f"{len(sub_clusters)} group(s), {len(missing_local)} passed through: "
                + "; ".join(display_groups)
            )

        except Exception as exc:
            preview = (raw or "")[:200].replace("\n", " ")
            print(
                f"  [dedup] LLM cluster resolution failed ({exc}); "
                f"keeping {len(cluster)} bullets unmodified."
            )
            print(f"  [dedup]   raw output preview: {preview!r}")
            for idx in cluster:
                output.append((idx, bullets[idx]))

    output.sort(key=lambda pair: pair[0])
    return [text for _, text in output]


def dedup_bullets_llm(
    bullets: list[str],
    backend,
    llm_handle,
    candidate_threshold: float = CANDIDATE_THRESHOLD,
) -> list[str]:
    """
    Full dedup: cheap Jaccard prefilter for candidate clusters, then LLM
    judgment + canonicalization for the ambiguous candidate_threshold band.
    """
    if len(bullets) <= 1:
        return bullets
    clusters = cluster_candidates(bullets, candidate_threshold)
    return llm_resolve_clusters(bullets, clusters, backend, llm_handle)

