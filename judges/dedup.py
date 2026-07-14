"""
Bullet deduplication: embedding prefilter + LLM cluster-and-canonicalize.

Used in two places:
  1. Per-chunk: merge J1/J2/J3 bullet lists by dropping near-duplicate bullets
     before writing to the chunk's refined_requirements field.
  2. Cross-block: after all chunks are refined, drop bullets that are
     semantically equivalent across different time blocks.

Dedup pipeline per location:
  extract_bullets
  → filter_meta_bullets        (drop system-prompt echoes)
  → filter_truncated_bullets   (drop mid-sentence cutoffs)
  → filter_substrings          (drop bullets that are subsets of longer ones)
  → cluster_candidates         (cheap embedding prefilter, high recall)
  → llm_resolve_clusters       (LLM judges + canonicalizes each candidate cluster)
"""

from __future__ import annotations

import re

import numpy as np

from utils import extract_balanced_json, normalize_embeddings

_model = None  # lazy-loaded on first use

# ---------------------------------------------------------------------------
# Known meta-instruction patterns that judges echo from their system prompts.
# ---------------------------------------------------------------------------
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

# Embedding prefilter threshold — deliberately LOW (high recall, low
# precision). Its only job is "could these plausibly be duplicates?" so we
# don't burn an LLM call on every single pair. Calibrated against the
# diagnostic run: true duplicates were observed as low as 0.08, so even this
# is not airtight recall — but going lower starts pulling in most of the
# corpus and defeats the cost-saving purpose of the prefilter. The LLM pass
# is the actual precision mechanism; this step only needs to be cheap and
# not miss the *obvious* candidates (shared vocabulary, shared topic).
CANDIDATE_THRESHOLD = 0.35

# Cosine similarity above which two bullets are treated as obvious duplicates
# without spending an LLM call — the complement to CANDIDATE_THRESHOLD above.
# Everything between CANDIDATE_THRESHOLD and this value is the genuinely
# ambiguous band (shared vocabulary/topic but possibly distinct requirements)
# where LLM judgment earns its cost. Above this value, wording is close
# enough that a reviewer would call it the same requirement essentially by
# inspection, so we auto-merge (keeping the longer bullet, same rule
# llm_resolve_clusters already applies via _pick_canonical for confirmed
# clusters) instead of burning a call on the easy majority.
AUTO_ACCEPT_THRESHOLD = 0.85

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


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


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

    normalized = [_norm(b) for b in bullets]
    kept = []
    for i, b_norm in enumerate(normalized):
        is_sub = any(
            b_norm in other and b_norm != other
            for j, other in enumerate(normalized)
            if i != j
        )
        if not is_sub:
            kept.append(bullets[i])
    return kept


def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Return a pairwise N×N cosine similarity matrix for a set of embeddings.

    Args:
        embeddings: (N, D) float array

    Returns:
        (N, N) float array where entry [i, j] is cosine similarity of i and j
    """
    normed = normalize_embeddings(embeddings)
    return normed @ normed.T


def _cosine_similarity_against_pool(
    emb: np.ndarray,
    pool: list[np.ndarray],
) -> float:
    """
    Return the maximum cosine similarity between a single embedding and a
    pool of previously-seen embeddings. Returns 0.0 if the pool is empty.

    Args:
        emb:  (D,) float array for the candidate bullet
        pool: list of (D,) float arrays already accepted

    Returns:
        Maximum cosine similarity, or 0.0 if pool is empty
    """
    if not pool:
        return 0.0
    normed_emb = normalize_embeddings(emb[np.newaxis])[0]  # (D,)
    normed_pool = normalize_embeddings(np.stack(pool))  # (M, D)
    return float((normed_pool @ normed_emb).max())


# ---------------------------------------------------------------------------
# Candidate clustering (embedding prefilter — recall, not precision)
# ---------------------------------------------------------------------------


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
) -> tuple[list[list[int]], np.ndarray]:
    """
    Group bullet indices into candidate-duplicate clusters using a
    deliberately lenient cosine threshold. This is a RECALL step: it only
    needs to make sure plausible duplicates end up in the same cluster so
    the LLM pass can look at them together. Precision is handled downstream.

    Singletons (no neighbour above threshold) are returned as single-item
    clusters and do not need an LLM call at all.

    Args:
        bullets:   Bullet text strings (no prefix)
        threshold: Cosine similarity cutoff for "worth checking together"

    Returns:
        (clusters, embeddings) — clusters is a list of index-lists into
        `bullets`; embeddings is the raw (N, D) embedding matrix, handed
        back so callers (llm_resolve_clusters' auto-accept split) can reuse
        it instead of re-encoding the same bullets a second time.
    """
    n = len(bullets)
    if n <= 1:
        return [[i] for i in range(n)], np.zeros((n, 0))

    model = _get_model()
    embeddings = model.encode(bullets, convert_to_numpy=True, show_progress_bar=False)
    sim = _cosine_matrix(embeddings)

    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold
    ]
    return _connected_components(n, edges), embeddings


# ---------------------------------------------------------------------------
# LLM cluster resolution (precision step)
# ---------------------------------------------------------------------------


def _split_auto_accept(
    cluster: list[int], embeddings: np.ndarray, threshold: float
) -> tuple[list[list[int]], list[int]]:
    """
    Partition a candidate cluster into high-confidence duplicate sub-groups
    (cosine >= threshold — safe to auto-merge with no LLM call) and the
    remaining indices that are still ambiguous and need LLM adjudication.

    Runs the same connected-components logic as cluster_candidates, just
    restricted to this cluster's members and at the stricter AUTO_ACCEPT
    threshold instead of the lenient recall threshold.

    Args:
        cluster:    Global bullet indices belonging to one candidate cluster
        embeddings: (N, D) embedding matrix for the FULL bullet list —
                    indexed with `cluster`'s global indices
        threshold:  Cosine cutoff above which bullets are auto-merged

    Returns:
        (auto_pairs, remaining) — auto_pairs is a list of index-lists (each
        len > 1, global indices) confidently confirmed as duplicates;
        remaining is a flat list of global indices (still-singleton members
        under `threshold`) to hand to the LLM unresolved.
    """
    n = len(cluster)
    sub_embeddings = np.stack([embeddings[i] for i in cluster])
    sim = _cosine_matrix(sub_embeddings)
    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold
    ]
    components = _connected_components(n, edges)

    auto_pairs: list[list[int]] = []
    remaining: list[int] = []
    for comp in components:
        if len(comp) > 1:
            auto_pairs.append([cluster[c] for c in comp])
        else:
            remaining.append(cluster[comp[0]])
    return auto_pairs, remaining


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
    max_tokens: int = 500,
    embeddings: np.ndarray | None = None,
    auto_accept_threshold: float = AUTO_ACCEPT_THRESHOLD,
) -> list[str]:
    """
    For each candidate cluster with more than one bullet, ask the LLM to
    confirm which members are true duplicates (it may split a cluster into
    several real ones) and return one canonical bullet per confirmed group.
    Singleton clusters pass through untouched with no LLM call.

    If `embeddings` is given, each multi-member cluster is first split via
    _split_auto_accept: sub-groups whose pairwise cosine similarity is
    already >= auto_accept_threshold are merged immediately (keeping the
    longer bullet, via _pick_canonical) with no LLM call at all. Only the
    remaining ambiguous members — still-singleton after that split — are
    sent to the LLM, as one shrunk cluster in place of the original. If
    `embeddings` is None, this split is skipped and the full cluster goes
    to the LLM exactly as before (backward-compatible default).

    On any parse/call failure for a given cluster, fails safe by keeping all
    of that cluster's bullets unmodified rather than risking an incorrect
    merge.

    Args:
        bullets:               Full bullet list (post cheap-filter stage)
        clusters:              Candidate clusters from cluster_candidates()
        backend:               InferenceBackend instance (see backend.py)
        llm_handle:            Handle returned by backend.load_llm()
        max_tokens:            Generation budget for the clustering response
        embeddings:            Optional (N, D) embedding matrix aligned with
                               `bullets`, enabling the auto-accept skip above
        auto_accept_threshold: Cosine cutoff for the auto-accept skip

    Returns:
        Deduplicated bullet list, original relative order preserved.
    """
    output: list[tuple[int, str]] = []  # (first_index, text) for ordering

    for cluster in clusters:
        if len(cluster) == 1:
            idx = cluster[0]
            output.append((idx, bullets[idx]))
            continue

        if embeddings is not None:
            auto_pairs, remaining = _split_auto_accept(
                cluster, embeddings, auto_accept_threshold
            )
            for pair in auto_pairs:
                canonical = _pick_canonical(bullets, pair)
                output.append((min(pair), canonical))
            if not remaining:
                continue
            if len(remaining) == 1:
                idx = remaining[0]
                output.append((idx, bullets[idx]))
                continue
            cluster = remaining  # only the still-ambiguous members go to the LLM

        numbered = "\n".join(f"{n}. {bullets[i]}" for n, i in enumerate(cluster))
        user_prompt = (
            f"Bullets (use these numbers in your answer, NOT the original "
            f"list positions):\n{numbered}"
        )

        # Scale the token budget with cluster size — a fixed budget that's
        # fine for a 3-bullet cluster will truncate a 17-bullet one before
        # the JSON closes, which presents as "no clusters in LLM response"
        # with no indication that truncation was the actual cause.
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

            # sub_clusters are indices into `cluster` (the local numbering we
            # gave the LLM) — map back to indices into the original `bullets`.
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
    auto_accept_threshold: float = AUTO_ACCEPT_THRESHOLD,
) -> list[str]:
    """
    Full dedup: cheap embedding prefilter for candidate clusters, then an
    auto-accept skip for the obvious high-cosine majority (>=
    auto_accept_threshold, merged with no LLM call), then LLM judgment +
    canonicalization for whatever's left in the ambiguous
    candidate_threshold–auto_accept_threshold band. This replaces the old
    pure-cosine dedup_bullets() as the final pass — see module docstring
    for why.
    """
    if len(bullets) <= 1:
        return bullets
    clusters, embeddings = cluster_candidates(bullets, candidate_threshold)
    return llm_resolve_clusters(
        bullets,
        clusters,
        backend,
        llm_handle,
        embeddings=embeddings,
        auto_accept_threshold=auto_accept_threshold,
    )


def merge_judge_outputs(
    j1_text: str,
    j2_text: str,
    j3_text: str,
    backend,
    llm_handle,
    max_bullets: int = 7,
    candidate_threshold: float = CANDIDATE_THRESHOLD,
) -> str:
    """
    Merge three judge bullet lists into a single clean markdown bullet list.

    Args:
        j1_text:             Raw output from Judge 1
        j2_text:              Raw output from Judge 2
        j3_text:              Raw output from Judge 3
        backend:              InferenceBackend instance (see backend.py)
        llm_handle:           Handle returned by backend.load_llm()
        max_bullets:          Maximum bullets in merged output
        candidate_threshold:  Embedding prefilter threshold (recall step)

    Returns:
        Merged markdown bullet list string, or "" if nothing survives
    """

    def _process(text: str) -> list[str]:
        bullets = extract_bullets(text)
        bullets = filter_meta_bullets(bullets)
        bullets = filter_truncated_bullets(bullets)
        return bullets

    all_bullets = _process(j1_text) + _process(j2_text) + _process(j3_text)

    if not all_bullets:
        return ""

    all_bullets = filter_substrings(all_bullets)
    deduped = dedup_bullets_llm(all_bullets, backend, llm_handle, candidate_threshold)[
        :max_bullets
    ]
    return "\n".join(f"- {b}" for b in deduped)


def dedup_cross_block(
    refined: list[dict],
    backend,
    llm_handle,
    candidate_threshold: float = CANDIDATE_THRESHOLD,
) -> tuple[list[dict], int]:
    """
    Remove semantically duplicate bullets that appear across different time
    blocks. Returns NEW dicts — does NOT mutate the input list. The first
    block with content is always kept in full (pool is empty).

    Unlike the old pure-cosine version, the embedding pool is only used to
    find CANDIDATE cross-block duplicates (high recall, low precision); the
    LLM makes the actual keep/merge decision against the running pool of
    already-accepted bullets, the same way it does within a single chunk in
    merge_judge_outputs.

    Args:
        refined:              List of refined chunk dicts (not mutated)
        backend:              InferenceBackend instance (see backend.py)
        llm_handle:           Handle returned by backend.load_llm()
        candidate_threshold:  Embedding prefilter threshold (recall step)

    Returns:
        (new list of refined dicts with updated refined_requirements,
         total number of bullets removed across all blocks)
    """
    model = _get_model()
    pool_bullets: list[str] = []
    pool_embeddings: list[np.ndarray] = []
    total_removed = 0
    result: list[dict] = []

    for req in refined:
        text = req.get("refined_requirements")

        # Pass through blocks with no content unchanged
        if not text:
            result.append(dict(req))
            continue

        bullets = extract_bullets(text)
        if not bullets:
            result.append(dict(req))
            continue

        # Bullets arriving here have already been through filter_substrings
        # inside merge_judge_outputs, so we skip repeating that step.
        chunk_embeddings = model.encode(
            bullets, convert_to_numpy=True, show_progress_bar=False
        )

        kept_bullets: list[str] = []
        kept_embeddings: list[np.ndarray] = []

        for bullet, emb in zip(bullets, chunk_embeddings):
            max_sim = _cosine_similarity_against_pool(emb, pool_embeddings)

            if max_sim < candidate_threshold:
                # Not even a plausible candidate — keep, no LLM call needed.
                kept_bullets.append(bullet)
                kept_embeddings.append(emb)
                continue

            # Plausible candidate — find which pool bullets it's close to
            # and let the LLM decide if it's a true duplicate of any of them.
            if pool_embeddings:
                normed_emb = normalize_embeddings(emb[np.newaxis])[0]
                normed_pool = normalize_embeddings(np.stack(pool_embeddings))
                sims = normed_pool @ normed_emb
                nearby_idx = [i for i, s in enumerate(sims) if s >= candidate_threshold]
            else:
                nearby_idx = []

            if not nearby_idx:
                kept_bullets.append(bullet)
                kept_embeddings.append(emb)
                continue

            local_bullets = [pool_bullets[i] for i in nearby_idx] + [bullet]
            local_embeddings = np.stack(
                [pool_embeddings[i] for i in nearby_idx] + [emb]
            )
            new_bullet_local_idx = len(local_bullets) - 1
            clusters = [
                [i for i in range(len(local_bullets))]
            ]  # one cluster: ask LLM to confirm/split (or auto-accept, below)

            try:
                resolved = llm_resolve_clusters(
                    local_bullets,
                    clusters,
                    backend,
                    llm_handle,
                    embeddings=local_embeddings,
                )
                # If the new bullet's text survived verbatim or as part of a
                # merge that still represents it, it's a duplicate -> drop.
                # We detect "dropped" by checking whether resolving merged it
                # away: if len(resolved) < len(local_bullets) and the new
                # bullet is not present as its own canonical text, treat as
                # duplicate of an existing pool entry (already represented).
                is_new_bullet_represented_alone = bullet in resolved
                merged_away = len(resolved) < len(local_bullets)

                if merged_away and not is_new_bullet_represented_alone:
                    total_removed += 1
                    continue
                else:
                    kept_bullets.append(bullet)
                    kept_embeddings.append(emb)
            except Exception as exc:  # noqa: BLE001 — fail safe, keep bullet
                print(
                    f"  [dedup] cross-block LLM check failed ({exc}); "
                    f"keeping bullet unmodified."
                )
                kept_bullets.append(bullet)
                kept_embeddings.append(emb)

        pool_bullets.extend(kept_bullets)
        pool_embeddings.extend(kept_embeddings)

        # Build new dict — never mutate the input
        new_req = dict(req)
        new_req["refined_requirements"] = (
            "\n".join(f"- {b}" for b in kept_bullets) if kept_bullets else None
        )
        result.append(new_req)

    return result, total_removed
