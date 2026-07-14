"""
Main execution script for the VLM-PwC requirement extraction pipeline.
Orchestrates data preparation, model inference, judges refinement, and output generation.

Memory flow:
  Stage 2: VLM proposer loaded. Judge models validated/downloaded, not loaded.
  Stage 3: VLM runs inference on all chunks. Per-chunk checkpoint written.
  -> proposer.unload()
  Stage 4: All judge models loaded ONCE, reused across all chunks.
           Merger LLM replaced by embedding-prefilter + LLM cluster dedup.
  Stage 5: Cross-block dedup (JUDGE1, own load/unload cycle) -> priority
           classification -> config generation. Priority classification and
           config generation both use JUDGE3_MODEL and now SHARE one
           load/unload cycle instead of loading it twice back-to-back.
  Stage 6: Uses the JUDGE3 handle loaded in Stage 5 (see above) to infer BRS
           config from bullets, writes project_config.json.
  Stage 7: node generate_docx.js reads JSON + config → BRS.docx

Output: output/extracted_requirements.json + output/BRS.docx
"""

import json
import os
import signal
import subprocess
import time

from backend import get_backend
from config import CHUNK_SIZE_SEC, JUDGE_MODEL, DEDUP_MODEL, OUTPUT_FILE
from config_generator import generate_config
from data_prep import get_valid_chunks, prepare_data
from extractor import fetch_transcript
from judges import JudgesPanel
from judges.dedup import dedup_cross_block, extract_bullets
from judges.priority import classify_priorities
from models import VLMProposer
from utils import flag_unknown_words

_CHECKPOINT_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_checkpoint.jsonl"
_CONFIG_FILE = "output/project_config.json"
_DOCX_FILE = "output/BRS.docx"

# Retry / timeout configuration
_VLM_MAX_RETRIES = 2  # retries after first attempt fails
_JUDGE_MAX_RETRIES = 2
_VLM_TIMEOUT_SEC = 120  # per-chunk timeout for VLM generate()
_JUDGE_TIMEOUT_SEC = 60  # per-chunk timeout for judge evaluate()
_BACKOFF_BASE_SEC = 2.0  # exponential backoff base


# ── helpers ───────────────────────────────────────────────────────────────────


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Operation timed out")


def _with_timeout(fn, timeout_sec: int):
    """
    Run fn() with a hard timeout using SIGALRM (Unix only).
    Raises _TimeoutError if fn() does not return within timeout_sec.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _with_retry(fn, max_retries: int, timeout_sec: int, label: str):
    """
    Call fn() with up to max_retries retries and exponential backoff.
    Each attempt is wrapped in _with_timeout(timeout_sec).

    Args:
        fn:          Zero-argument callable to attempt
        max_retries: Number of retries after the first attempt
        timeout_sec: Per-attempt timeout in seconds
        label:       Human-readable label for log messages

    Returns:
        Return value of fn() on success

    Raises:
        Last exception if all attempts fail
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return _with_timeout(fn, timeout_sec)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = _BACKOFF_BASE_SEC * (2**attempt)
                print(
                    f"    {label} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                print(f"    {label} failed after {1 + max_retries} attempts: {e}")
    raise last_exc


def _is_empty_output(text: str | None) -> bool:
    """
    Robust empty-output detection via json.loads.
    Returns True if the proposer output is missing, blank, unparseable,
    or parses to an empty/null value.
    """
    if not text or not text.strip():
        return True
    try:
        parsed = json.loads(text.strip())
        return not parsed
    except (json.JSONDecodeError, ValueError):
        return len(text.strip()) < 10


def _load_checkpoint() -> list | None:
    """
    Load a previously saved checkpoint if it exists.
    Reads one JSON object per line (JSONL) — O(n) and resumable mid-run.
    Returns the list of extracted chunks, or None if no checkpoint found.
    """
    if not os.path.exists(_CHECKPOINT_FILE):
        return None
    try:
        chunks = []
        with open(_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        if not chunks:
            return None
        print(f"Resuming from checkpoint: {_CHECKPOINT_FILE} ({len(chunks)} chunks)\n")
        return chunks
    except (json.JSONDecodeError, KeyError):
        print("Checkpoint found but unreadable — re-running inference.\n")
        return None


def _append_checkpoint(chunk: dict) -> None:
    """
    Append a single chunk record to the checkpoint file as one JSON line.
    O(1) — opens the file in append mode, writes one line, closes.
    No read-back, no rewrite of the full array.
    """
    os.makedirs(os.path.dirname(_CHECKPOINT_FILE) or ".", exist_ok=True)
    with open(_CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _write_json(
    refined: list,
    inference_elapsed: float,
    judges_elapsed: float,
    dedup_removed: int,
    priorities: list[str] | None = None,
) -> list[str]:
    """
    Write refined requirements to the output JSON file.
    Returns a flat list of all bullet texts (for Stage 6 priority/config use).

    If `priorities` is given (same flat order as bullets are extracted, i.e.
    block-by-block in iteration order), each bullet in the JSON output
    becomes {"text": ..., "priority": ...} instead of a plain string.
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total = inference_elapsed + judges_elapsed
    successful = sum(1 for r in refined if r.get("refined_requirements"))

    blocks: list[dict] = []
    all_bullets: list[str] = []
    cursor = 0
    for req in refined:
        bullets = (
            extract_bullets(req["refined_requirements"])
            if req.get("refined_requirements")
            else []
        )
        bullets = [flag_unknown_words(b) for b in bullets]
        all_bullets.extend(bullets)

        if priorities is not None:
            slice_ = priorities[cursor : cursor + len(bullets)]
            cursor += len(bullets)
            bullet_payload = [
                {"text": b, "priority": p} for b, p in zip(bullets, slice_)
            ]
        else:
            bullet_payload = bullets

        blocks.append(
            {
                "minute_block": req["minute_block"],
                "time_range": req["time_range"],
                "bullets": bullet_payload,
                "error": req.get("error"),
            }
        )

    output = {
        "summary": {
            "total_chunks": len(refined),
            "successfully_refined": successful,
            "duplicate_bullets_removed_cross_block": dedup_removed,
            "inference_time_sec": round(inference_elapsed, 1),
            "judges_time_sec": round(judges_elapsed, 1),
            "total_time_sec": round(total, 1),
        },
        "requirements": blocks,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return all_bullets


def _generate_docx() -> None:
    """
    Call generate_docx.js via node to produce the BRS Word document.
    Verifies that _CONFIG_FILE exists before spawning node — if it was
    not written by Stage 6 (e.g. due to an error), skip docx generation
    and print a recovery command.
    """
    if not os.path.exists(_CONFIG_FILE):
        print(f"  Skipped — config file not found: {_CONFIG_FILE}")
        print(
            f"  To retry: node generate_docx.js {OUTPUT_FILE} {_DOCX_FILE} {_CONFIG_FILE}"
        )
        return

    result = subprocess.run(
        ["node", "generate_docx.js", OUTPUT_FILE, _DOCX_FILE, _CONFIG_FILE],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
    else:
        print(f"  WARNING: docx generation failed:\n{result.stderr.strip()}")


# ── pipeline ──────────────────────────────────────────────────────────────────


def main():
    # ===== STAGE 1: DATA PREPARATION =====
    print("[STAGE 1] Data Preparation")
    fetch_transcript()  # no-op if transcript already exists
    chunks = prepare_data()
    valid_chunks = get_valid_chunks(chunks)
    print(f"Prepared {len(valid_chunks)} chunks\n")

    # ===== STAGE 2: MODEL INITIALIZATION =====
    print("[STAGE 2] Model Initialization")
    backend = get_backend()  # single instance for all stages
    proposer = VLMProposer(backend=backend)
    judges_panel = JudgesPanel(backend=backend)
    print("  Validating judge models...")
    judges_panel.preload()
    print("Pipeline ready\n")

    # ===== STAGE 3: INFERENCE =====
    checkpointed = _load_checkpoint()
    inference_elapsed = 0.0

    # A checkpoint can be PARTIAL (e.g. the process crashed mid-run — MLX/
    # Metal GPU hangs are an uncatchable native abort, not a Python
    # exception, so a chunk can vanish mid-generate() with no chance for
    # try/except to record it). Treating "checkpoint exists" as "Stage 3 is
    # done" would silently drop every chunk after the crash point. Instead,
    # figure out exactly which bucket_index values are already recorded and
    # only process what's left.
    done_indices = (
        {c["minute_block"] for c in checkpointed} if checkpointed else set()
    )
    remaining_chunks = [
        (bucket_index, chunk_data)
        for bucket_index, chunk_data in valid_chunks
        if bucket_index not in done_indices
    ]
    all_extracted = list(checkpointed) if checkpointed else []

    if not remaining_chunks:
        print("[STAGE 3] Skipped (loaded from checkpoint)\n")
    else:
        if checkpointed:
            print(
                f"[STAGE 3] Resuming: {len(done_indices)}/{len(valid_chunks)} "
                f"chunk(s) already in checkpoint, processing remaining "
                f"{len(remaining_chunks)}\n"
            )
        else:
            print(f"[STAGE 3] Inference: Processing {len(valid_chunks)} chunks\n")
        inference_start = time.time()

        for i, (bucket_index, chunk_data) in enumerate(remaining_chunks, 1):
            combined_text = " ".join(chunk_data["text_lines"])
            frame_paths = chunk_data["frame_paths"]
            label = f"Chunk {bucket_index + 1} [{i}/{len(remaining_chunks)}]"

            t = time.time()
            try:
                response = _with_retry(
                    fn=lambda fp=frame_paths, tx=combined_text: proposer.generate(
                        fp, tx
                    ),
                    max_retries=_VLM_MAX_RETRIES,
                    timeout_sec=_VLM_TIMEOUT_SEC,
                    label=label,
                )
                print(
                    f"  [{i}/{len(remaining_chunks)}] Chunk {bucket_index + 1}: "
                    f"{time.time() - t:.1f}s"
                )
            except Exception as e:
                print(
                    f"  [{i}/{len(remaining_chunks)}] Chunk {bucket_index + 1}: "
                    f"FAILED after {time.time() - t:.1f}s — {e}"
                )
                response = None

            chunk_record = {
                "minute_block": bucket_index,
                "time_range": (
                    f"{bucket_index * CHUNK_SIZE_SEC}s - "
                    f"{(bucket_index + 1) * CHUNK_SIZE_SEC}s"
                ),
                "raw_model_output": response,
            }
            all_extracted.append(chunk_record)

            # Incremental checkpoint — only appends this chunk, does not
            # rewrite the entire list on every iteration
            _append_checkpoint(chunk_record)

        inference_elapsed = time.time() - inference_start
        print(f"\nInference complete. Checkpoint: {_CHECKPOINT_FILE}\n")

    # Free the VLM before loading judges
    proposer.unload()

    # ===== STAGE 4: JUDGES REFINEMENT =====
    print(f"[STAGE 4] Judges Panel: Refining {len(all_extracted)} chunks\n")
    judges_panel.load_all()
    judges_start = time.time()

    refined: list[dict] = []
    for i, req in enumerate(all_extracted, 1):
        print(
            f"  [{i}/{len(all_extracted)}] Chunk {req['minute_block'] + 1}...", end=" "
        )

        if _is_empty_output(req["raw_model_output"]):
            print("Skipped (empty output).")
            refined.append(
                {
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": None,
                    "error": "Proposer returned empty or invalid output.",
                }
            )
            continue

        label = f"Judge eval chunk {req['minute_block'] + 1}"
        try:
            raw_output = req["raw_model_output"]
            output = _with_retry(
                fn=lambda ro=raw_output: judges_panel.evaluate(ro),
                max_retries=_JUDGE_MAX_RETRIES,
                timeout_sec=_JUDGE_TIMEOUT_SEC,
                label=label,
            )
            refined.append(
                {
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": output,
                }
            )
            print("Done")
        except Exception as e:
            print(f"Failed: {e}")
            refined.append(
                {
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": None,
                    "error": str(e),
                }
            )

    judges_elapsed = time.time() - judges_start
    print()
    judges_panel.unload_all()

    failed = [r for r in refined if r.get("error")]
    if failed:
        print(f"  WARNING: {len(failed)} chunk(s) did not produce requirements:")
        for r in failed:
            print(
                f"    - Chunk {r['minute_block'] + 1} ({r['time_range']}): {r['error']}"
            )
        print()

    # ===== STAGE 5: POST-PROCESSING & OUTPUT =====
    print("[STAGE 5] Post-processing & Output")

    # Cross-block dedup, priority classification, and config generation all
    # use JUDGE3_MODEL (Qwen3-1.7B) — switched dedup over from Judge 1
    # (Phi-3-mini) since Qwen3 has a demonstrably better track record on
    # structured JSON output and a much larger native context window (32k
    # vs 4096), and doing so lets all three steps share ONE load/unload
    # cycle instead of two (dedup previously loaded JUDGE1 separately).
    dedup_removed = 0  # safe default — referenced unconditionally in the
    # summary section below, even if Judge 3 never loads.
    all_bullets: list[str] = []
    judge3_handle = None

    # Fine-grained timing for everything in Stage 5/6 — previously NONE of
    # this was tracked, so the reported "Total time" only ever summed
    # inference + judges refinement and silently dropped model load, dedup,
    # priority classification, and config generation entirely. That's the
    # likely explanation for large gaps between reported total and actual
    # wall-clock run time.
    judge3_load_elapsed = 0.0
    dedup_elapsed = 0.0
    priority_elapsed = 0.0
    config_elapsed = 0.0

    load_start = time.time()
    try:
        judge3_handle = backend.load_llm(DEDUP_MODEL)
    except Exception as e:
        print(
            f"  WARNING: failed to load {DEDUP_MODEL} for Stage 5: {e}. "
            f"Skipping dedup, priority classification, and config generation. "
            f"JSON will still be written without them."
        )
    judge3_load_elapsed = time.time() - load_start
    print(f"  Judge 3 model load: {judge3_load_elapsed:.1f}s")

    if judge3_handle is not None:
        try:
            print("  Running cross-block deduplication...")
            dedup_start = time.time()
            try:
                refined, dedup_removed = dedup_cross_block(
                    refined, backend=backend, llm_handle=judge3_handle
                )
                dedup_elapsed = time.time() - dedup_start
                print(
                    f"  Removed {dedup_removed} duplicate bullets across blocks "
                    f"({dedup_elapsed:.1f}s).\n"
                )
            except Exception as e:
                dedup_elapsed = time.time() - dedup_start
                print(
                    f"  WARNING: cross-block dedup failed after "
                    f"{dedup_elapsed:.1f}s: {e}. Skipping.\n"
                )
                dedup_removed = 0

            # First pass: get the flat bullet list (no priorities yet) to classify against
            all_bullets = _write_json(
                refined, inference_elapsed, judges_elapsed, dedup_removed
            )

            print("  Classifying requirement priorities...")
            priority_start = time.time()
            try:
                priorities = classify_priorities(
                    all_bullets, backend=backend, handle=judge3_handle
                )
                priority_elapsed = time.time() - priority_start
                print(
                    f"  Classified {len(priorities)} requirements "
                    f"(M={priorities.count('M')}, S={priorities.count('S')}, "
                    f"C={priorities.count('C')}) ({priority_elapsed:.1f}s)\n"
                )
            except Exception as e:
                priority_elapsed = time.time() - priority_start
                print(
                    f"  WARNING: priority classification failed after "
                    f"{priority_elapsed:.1f}s: {e}. Defaulting all to 'S'.\n"
                )
                priorities = ["S"] * len(all_bullets)

            # Second pass: rewrite JSON with priorities attached per bullet
            all_bullets = _write_json(
                refined,
                inference_elapsed,
                judges_elapsed,
                dedup_removed,
                priorities=priorities,
            )
            print(f"  JSON written to {OUTPUT_FILE}")

            # ===== STAGE 6: BRS CONFIG GENERATION =====
            print("\n[STAGE 6] BRS Config Generation")
            config_start = time.time()
            try:
                generate_config(
                    all_bullets, _CONFIG_FILE, backend=backend, handle=judge3_handle
                )
                config_elapsed = time.time() - config_start
                print(f"  Config generation: {config_elapsed:.1f}s")
            except Exception as e:
                config_elapsed = time.time() - config_start
                print(
                    f"  WARNING: config generation failed after "
                    f"{config_elapsed:.1f}s: {e}"
                )
                print(f"  Docx will be skipped. To retry after fixing:")
                print(
                    f"  node generate_docx.js {OUTPUT_FILE} {_DOCX_FILE} {_CONFIG_FILE}"
                )
        finally:
            backend.unload_llm(judge3_handle)
    else:
        # Judge 3 never loaded — still write the JSON so the run isn't a
        # total loss, just without dedup/priorities/config. _generate_docx()
        # already handles a missing config file gracefully (Stage 7).
        all_bullets = _write_json(
            refined, inference_elapsed, judges_elapsed, dedup_removed
        )
        print(f"  JSON written to {OUTPUT_FILE} (without dedup/priority/config)")

    # ===== STAGE 7: DOCX GENERATION =====
    print("\n[STAGE 7] Generating BRS Word Document")
    docx_start = time.time()
    _generate_docx()
    docx_elapsed = time.time() - docx_start

    # ===== SUMMARY =====
    stage5_6_elapsed = (
        judge3_load_elapsed + dedup_elapsed + priority_elapsed + config_elapsed
    )
    total = inference_elapsed + judges_elapsed + stage5_6_elapsed + docx_elapsed
    avg = (
        inference_elapsed / len(valid_chunks)
        if valid_chunks and inference_elapsed
        else 0
    )

    print(f"\n{'=' * 50}")
    print("Pipeline Summary")
    print(f"{'=' * 50}")
    print(f"Total time:                  {total:.1f}s")
    print(f"  - Inference (Stage 3):     {inference_elapsed:.1f}s")
    print(f"  - Judges refinement (S4):  {judges_elapsed:.1f}s")
    print(f"  - Judge 3 model load:      {judge3_load_elapsed:.1f}s")
    print(f"  - Cross-block dedup:       {dedup_elapsed:.1f}s")
    print(f"  - Priority classification: {priority_elapsed:.1f}s")
    print(f"  - Config generation (S6):  {config_elapsed:.1f}s")
    print(f"  - Docx generation (S7):    {docx_elapsed:.1f}s")
    print(f"Chunks processed:            {len(valid_chunks)}")
    print(f"Avg time per chunk:          {avg:.1f}s")
    print(f"Cross-block dupes removed:   {dedup_removed}")
    print(f"Output (JSON):               {OUTPUT_FILE}")
    print(f"Output (Config):             {_CONFIG_FILE}")
    print(f"Output (DOCX):               {_DOCX_FILE}")
    print(f"Checkpoint:                  {_CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
