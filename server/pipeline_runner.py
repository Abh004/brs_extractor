"""
PipelineRunner — observable async wrapper around the BRS extraction pipeline.

Instead of printing to stdout, this class yields StageUpdate dataclass
instances through an asyncio.Queue so a FastAPI WebSocket can stream
progress to the browser.

Usage:
    runner = PipelineRunner(video_id="abc123")
    async for update in runner.run():
        # update.stage, update.message, update.progress, ...
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from backend import check_hf_reachable, get_backend, print_hf_unreachable_warning
from config import (
    AVI_INDEX_FILE,
    CHECKPOINT_FILE,
    CHUNK_SIZE_SEC,
    CONFIG_FILE,
    DEDUP_MODEL,
    DOCX_FILE,
    OUTPUT_FILE,
)
import avi as avi_client
from config_generator import generate_config
from data_prep.avi import get_valid_avi_chunks, load_avi_chunks
from judges import JudgesPanel
from judges.dedup import (
    dedup_bullets_llm,
    dedup_cross_block,
    extract_bullets,
)
from judges.priority import classify_priorities
from models import AVIProposer
from utils import flag_unknown_words

try:
    import mlx.core as _mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


@dataclass
class StageUpdate:
    stage: int
    stage_name: str
    status: str  # "running" | "done" | "skipped" | "error"
    message: str
    progress: float  # 0.0–1.0
    elapsed: float = 0.0
    log: str = ""
    details: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Dict form consumed by the WebSocket / job status endpoints."""
        return {
            "type": "update",
            "stage": self.stage,
            "stage_name": self.stage_name,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "elapsed": round(self.elapsed, 1),
            "log": self.log.split("\n")[-50:],
        }


Callback = Callable[[StageUpdate], None]


_JUDGE_MAX_RETRIES = 2
_JUDGE_TIMEOUT_SEC = 60
_BACKOFF_BASE_SEC = 2.0


def _with_retry(fn, max_retries: int, label: str):
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return fn()
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
    if not text or not text.strip():
        return True
    try:
        parsed = json.loads(text.strip())
        return not parsed
    except (json.JSONDecodeError, ValueError):
        return len(text.strip()) < 10


def _load_checkpoint(checkpoint_file: str) -> list | None:
    if not os.path.exists(checkpoint_file):
        return None
    try:
        chunks = []
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        if not chunks:
            return None
        return chunks
    except (json.JSONDecodeError, KeyError):
        return None


def _append_checkpoint(chunk: dict, checkpoint_file: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


class PipelineRunner:
    def __init__(
        self,
        backend_name: str | None = None,
        video_path: str = "input/video.mp4",
        output_dir: str = "output",
    ):
        self._backend_name = backend_name
        self._video_path = video_path
        self._output_dir = output_dir

        self._checkpoint_file = os.path.join(output_dir, os.path.basename(CHECKPOINT_FILE))
        self._config_file = os.path.join(output_dir, os.path.basename(CONFIG_FILE))
        self._docx_file = os.path.join(output_dir, os.path.basename(DOCX_FILE))
        self._output_file = os.path.join(output_dir, os.path.basename(OUTPUT_FILE))
        self._avi_index_file = os.path.join(output_dir, os.path.basename(AVI_INDEX_FILE))

        self._cancel_requested = False
        self._log_lines: list[str] = []

    def cancel(self):
        self._cancel_requested = True

    def _emit(self, cb: Callback | None, update: StageUpdate):
        ts = time.strftime("%H:%M:%S")
        log_entry = f"[{ts}] [{update.stage_name}] {update.message}"
        self._log_lines.append(log_entry)
        update.log = "\n".join(self._log_lines[-200:])
        if cb:
            cb(update)

    def _check_cancelled(self):
        if self._cancel_requested:
            raise InterruptedError("Pipeline cancelled by user")

    async def run(self, cb: Callback | None = None):
        self._cancel_requested = False
        self._log_lines = []

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_sync, cb)

    @staticmethod
    def _clear_gpu_cache():
        if _HAS_MLX:
            _mx.clear_cache()

    def _run_sync(self, cb: Callback | None = None):
        try:
            self._pipeline(cb)
        except InterruptedError:
            self._emit(cb, StageUpdate(0, "Pipeline", "cancelled", "Cancelled by user", 0))
        except Exception as e:
            self._emit(cb, StageUpdate(0, "Pipeline", "error", f"Failed: {e}", 0))
            import traceback
            traceback.print_exc()

    def _clean_output_files(self) -> None:
        for path in [self._checkpoint_file, self._output_file, self._config_file, self._docx_file]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def _pipeline(self, cb: Callback | None = None):
        inference_elapsed = 0.0
        judges_elapsed = 0.0
        valid_chunks: list = []

        # Clean old output before starting fresh
        self._clean_output_files()

        # ── STAGE 1: DATA PREPARATION (Azure Video Indexer) ──
        self._emit(cb, StageUpdate(1, "Azure Video Indexer", "running",
                                    "Preparing data...", 0.0))
        stage1_start = time.time()

        if not os.path.exists(self._video_path) or os.path.getsize(self._video_path) == 0:
            raise FileNotFoundError(
                f"Video not found at {self._video_path}. "
                "Azure Video Indexer requires the actual video file — provide "
                "a YouTube ID / uploaded video or place the file at the video path."
            )

        self._emit(cb, StageUpdate(1, "Azure Video Indexer", "running",
                                    "Uploading & indexing video in Azure...", 0.2))
        self._check_cancelled()
        index = avi_client.upload_and_index(
            video_path=self._video_path,
            output_dir=self._output_dir,
            index_file=os.path.basename(self._avi_index_file),
        )
        self._check_cancelled()

        self._emit(cb, StageUpdate(1, "Azure Video Indexer", "running",
                                    "Chunking AVI transcript + OCR...", 0.9))
        chunks = load_avi_chunks(index)
        valid_chunks = get_valid_avi_chunks(chunks)
        self._emit(cb, StageUpdate(1, "Azure Video Indexer", "done",
                                    f"Prepared {len(valid_chunks)} chunks from AVI",
                                    1.0, elapsed=time.time() - stage1_start))

        # ── STAGE 2: MODEL INITIALIZATION ──
        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Initializing backend...", 0.0))
        stage2_start = time.time()
        self._check_cancelled()

        # Check HF reachability for user awareness
        if not check_hf_reachable():
            print_hf_unreachable_warning()

        # Clear any stale GPU state from previous runs (safe no-op if mlx not available)
        self._clear_gpu_cache()
        self._check_cancelled()

        backend = get_backend(self._backend_name)
        self._check_cancelled()

        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Setting up AVI proposer...", 0.3))
        self._check_cancelled()
        proposer = AVIProposer(index=index)
        self._check_cancelled()

        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Validating judge models...", 0.6))
        judges_panel = JudgesPanel(backend=backend)
        try:
            judges_panel.preload()
        except Exception as e:
            self._emit(cb, StageUpdate(2, "Model Initialization", "error",
                                        f"Judge model unavailable: {e}", 0.6,
                                        elapsed=time.time() - stage2_start))
            raise
        self._check_cancelled()
        self._emit(cb, StageUpdate(2, "Model Initialization", "done",
                                    "Models initialized", 1.0,
                                    elapsed=time.time() - stage2_start))

        # ── STAGE 3: PROPOSAL (Azure Video Indexer context) ──
        self._emit(cb, StageUpdate(3, "Proposal", "running",
                                   f"Building {len(valid_chunks)} chunk contexts...", 0.0))
        stage3_start = time.time()

        checkpointed = _load_checkpoint(self._checkpoint_file)
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
            self._emit(cb, StageUpdate(3, "Proposal", "skipped",
                                        "All chunks in checkpoint", 1.0,
                                        elapsed=time.time() - stage3_start))
        else:
            resume_info = (
                f"Resuming: {len(done_indices)}/{len(valid_chunks)} done, "
                f"{len(remaining_chunks)} remaining"
                if checkpointed else None
            )
            if resume_info:
                self._emit(cb, StageUpdate(3, "Proposal", "running",
                                            resume_info, 0.0))

            for i, (bucket_index, chunk_data) in enumerate(remaining_chunks, 1):
                self._check_cancelled()

                label = f"Chunk {bucket_index + 1} [{i}/{len(remaining_chunks)}]"

                t = time.time()
                try:
                    response = proposer.generate(chunk_data)
                    elapsed = time.time() - t
                except Exception as e:
                    elapsed = time.time() - t
                    print(f"    {label} failed: {e}")
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
                _append_checkpoint(chunk_record, self._checkpoint_file)

                progress = i / len(remaining_chunks)
                msg = f"Chunk {bucket_index + 1}/{len(valid_chunks)} ({elapsed:.1f}s)"
                self._emit(cb, StageUpdate(3, "Proposal", "running",
                                            msg, progress))

            inference_elapsed = time.time() - stage3_start
            self._emit(cb, StageUpdate(3, "Proposal", "done",
                                        f"Prepared {len(remaining_chunks)} chunks",
                                        1.0, elapsed=inference_elapsed))

        # ── STAGE 4: JUDGES REFINEMENT ──
        self._emit(cb, StageUpdate(4, "Judges Refinement", "running",
                                   f"Refining {len(all_extracted)} chunks...", 0.0))
        stage4_start = time.time()
        self._check_cancelled()

        judges_panel.load_all()

        refined: list[dict] = []
        for i, req in enumerate(all_extracted, 1):
            self._check_cancelled()

            if _is_empty_output(req["raw_model_output"]):
                refined.append({
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": None,
                    "error": "Proposer returned empty or invalid output.",
                })
                self._emit(cb, StageUpdate(4, "Judges Refinement", "running",
                                            f"Skipped chunk {req['minute_block'] + 1} (empty)",
                                            i / len(all_extracted)))
                continue

            label = f"Judge eval chunk {req['minute_block'] + 1}"
            try:
                raw_output = req["raw_model_output"]
                with backend.usage.stage("stage4_judge"):
                    output = _with_retry(
                        fn=lambda ro=raw_output: judges_panel.evaluate(ro),
                        max_retries=_JUDGE_MAX_RETRIES,
                        label=label,
                    )
                refined.append({
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": output,
                })
            except Exception as e:
                refined.append({
                    "minute_block": req["minute_block"],
                    "time_range": req["time_range"],
                    "raw_model_output": req["raw_model_output"],
                    "refined_requirements": None,
                    "error": str(e),
                })

            progress = i / len(all_extracted)
            self._emit(cb, StageUpdate(4, "Judges Refinement", "running",
                                        f"Chunk {i}/{len(all_extracted)}",
                                        progress))

        judges_elapsed = time.time() - stage4_start
        judges_panel.unload_all()

        failed = [r for r in refined if r.get("error")]
        fail_msg = f", {len(failed)} failed" if failed else ""
        self._emit(cb, StageUpdate(4, "Judges Refinement", "done",
                                    f"Refined {len(refined)} chunks{fail_msg}",
                                    1.0, elapsed=judges_elapsed))

        # ── STAGE 5+6: POST-PROCESSING, PRIORITIES, CONFIG ──
        dedup_removed, all_bullets = self._run_post_processing(
            refined, backend, judges_elapsed, inference_elapsed, cb
        )

        # ── STAGE 7: DOCX GENERATION ──
        self._emit(cb, StageUpdate(7, "DOCX Generation", "running",
                                    "Generating Word document...", 0.0))
        stage7_start = time.time()
        self._check_cancelled()

        self._generate_docx()

        self._emit(cb, StageUpdate(7, "DOCX Generation", "done",
                                    "BRS.docx generated",
                                    1.0, elapsed=time.time() - stage7_start))

        # Token usage summary
        backend.usage.to_console()

        # Done
        total = time.time() - stage1_start
        self._emit(cb, StageUpdate(0, "Complete", "done",
                                    f"Pipeline finished in {total:.1f}s",
                                    1.0, elapsed=total))

    def _extract_bullets(
        self, refined: list,
    ) -> tuple[list[list[str]], list[str]]:
        """Extract + spell-flag bullets per block. Returns (per_block, all_bullets)."""
        per_block: list[list[str]] = []
        all_bullets: list[str] = []
        for req in refined:
            bullets = (
                extract_bullets(req["refined_requirements"])
                if req.get("refined_requirements")
                else []
            )
            bullets = [flag_unknown_words(b) for b in bullets]
            per_block.append(bullets)
            all_bullets.extend(bullets)
        return per_block, all_bullets

    def _assemble_blocks(
        self,
        refined: list,
        per_block: list[list[str]],
        priorities: list[str] | None = None,
    ) -> list[dict]:
        """Build output block dicts from pre-extracted per-block bullets."""
        blocks: list[dict] = []
        cursor = 0
        for req, bullets in zip(refined, per_block):
            if priorities is not None:
                slice_ = priorities[cursor:cursor + len(bullets)]
                cursor += len(bullets)
                bullet_payload = [
                    {"text": b, "priority": p} for b, p in zip(bullets, slice_)
                ]
            else:
                bullet_payload = [{"text": b} for b in bullets]
            blocks.append({
                "minute_block": req["minute_block"],
                "time_range": req["time_range"],
                "bullets": bullet_payload,
                "error": req.get("error"),
            })
        return blocks

    def _dump_json(
        self, refined: list, inference_elapsed: float,
        judges_elapsed: float, dedup_removed: int = 0,
        priorities: list[str] | None = None,
        extracted: tuple[list[list[str]], list[str]] | None = None,
        token_usage: dict | None = None,
    ) -> list[str]:
        per_block, all_bullets = extracted or self._extract_bullets(refined)
        blocks = self._assemble_blocks(refined, per_block, priorities)
        os.makedirs(os.path.dirname(self._output_file), exist_ok=True)
        total = inference_elapsed + judges_elapsed
        successful = sum(1 for r in refined if r.get("refined_requirements"))
        summary = {
            "total_chunks": len(refined),
            "successfully_refined": successful,
            "duplicate_bullets_removed_cross_block": dedup_removed,
            "inference_time_sec": round(inference_elapsed, 1),
            "judges_time_sec": round(judges_elapsed, 1),
            "total_time_sec": round(total, 1),
        }
        if token_usage:
            summary["token_usage"] = token_usage
        output = {
            "summary": summary,
            "requirements": blocks,
        }
        with open(self._output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return all_bullets

    def _dedup_within_blocks(self, refined: list, backend, llm_handle) -> list:
        """Deduplicate semantically-equivalent bullets within each time block.

        The judge panel already applies a substring filter per chunk; this adds
        an embedding+LLM pass so near-duplicate wording in the SAME block is
        collapsed too (cross-block dedup only compares across blocks). Returns a
        new list — never mutates the input.
        """
        updated: list[dict] = []
        for req in refined:
            text = req.get("refined_requirements")
            if not text:
                updated.append(req)
                continue
            bullets = extract_bullets(text)
            if not bullets:
                updated.append(req)
                continue
            deduped = dedup_bullets_llm(bullets, backend, llm_handle)
            new_req = dict(req)
            new_req["refined_requirements"] = (
                "\n".join(f"- {b}" for b in deduped) if deduped else None
            )
            updated.append(new_req)
        return updated

    def _run_post_processing(
        self, refined: list, backend, judges_elapsed: float,
        inference_elapsed: float, cb: Callback | None = None,
    ) -> tuple[int, list[str]]:
        dedup_removed = 0
        all_bullets: list[str] = []

        self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                   "Loading DEDUP_MODEL...", 0.0))
        self._check_cancelled()

        judge3_handle = None
        try:
            judge3_handle = backend.load_llm(DEDUP_MODEL)
        except Exception:
            pass

        if judge3_handle is not None:
            try:
                self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                            "Within-block deduplication...", 0.05))
                with backend.usage.stage("stage5_dedup"):
                    refined = self._dedup_within_blocks(
                        refined, backend, judge3_handle
                    )

                self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                            "Cross-block deduplication...", 0.1))
                with backend.usage.stage("stage5_dedup"):
                    refined, dedup_removed = dedup_cross_block(
                        refined, backend=backend, llm_handle=judge3_handle
                    )

                extracted = self._extract_bullets(refined)
                all_bullets = extracted[1]

                self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                            f"Classifying {len(all_bullets)} priorities...", 0.4))
                try:
                    with backend.usage.stage("stage5_priority"):
                        priorities = classify_priorities(
                            all_bullets, backend=backend, handle=judge3_handle
                        )
                except Exception:
                    priorities = ["S"] * len(all_bullets)

                self._dump_json(refined, inference_elapsed, judges_elapsed,
                                dedup_removed, priorities=priorities,
                                extracted=extracted,
                                token_usage=backend.usage.summary())

                # Stage 6
                self._emit(cb, StageUpdate(6, "Config Generation", "running",
                                            "Inferring document structure...", 0.0))
                stage6_start = time.time()
                try:
                    with backend.usage.stage("stage6_config"):
                        generate_config(
                            all_bullets, self._config_file, backend=backend,
                            handle=judge3_handle,
                        )
                    self._emit(cb, StageUpdate(6, "Config Generation", "done",
                                                "Config generated",
                                                1.0, elapsed=time.time() - stage6_start))
                except Exception as e:
                    self._emit(cb, StageUpdate(6, "Config Generation", "error",
                                                f"Config failed: {e}",
                                                0, elapsed=time.time() - stage6_start))
            finally:
                backend.unload_llm(judge3_handle)
        else:
            all_bullets = self._dump_json(
                refined, inference_elapsed, judges_elapsed,
                token_usage=backend.usage.summary(),
            )

        self._emit(cb, StageUpdate(5, "Post-processing", "done",
                                    f"Removed {dedup_removed} dupes, "
                                    f"{len(all_bullets)} requirements",
                                    1.0, elapsed=0))
        return dedup_removed, all_bullets

    def _generate_docx(self) -> None:
        if not os.path.exists(self._config_file):
            return
        result = subprocess.run(
            ["node", "generate_docx.js", self._output_file,
             self._docx_file, self._config_file],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: docx generation failed:\n{result.stderr.strip()}")
