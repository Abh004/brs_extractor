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

from backend import get_backend
from config import (
    CHUNK_SIZE_SEC,
    DEDUP_MODEL,
    OUTPUT_FILE,
)
from config_generator import generate_config
from data_prep import get_valid_chunks, prepare_data
from extractor import fetch_transcript
from judges import JudgesPanel
from judges.dedup import dedup_cross_block, extract_bullets
from judges.priority import classify_priorities
from models import VLMProposer
from utils import flag_unknown_words


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


Callback = Callable[[StageUpdate], None]


_CHECKPOINT_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_checkpoint.jsonl"
_CONFIG_FILE = "output/project_config.json"
_DOCX_FILE = "output/BRS.docx"

_VLM_MAX_RETRIES = 2
_JUDGE_MAX_RETRIES = 2
_VLM_TIMEOUT_SEC = 120
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
        video_id: str | None = None,
        video_path: str = "input/video.mp4",
        transcript_path: str = "input/transcript.json",
        output_dir: str = "output",
        keyframe_dir: str = "keyframes",
    ):
        self._backend_name = backend_name
        self._video_id = video_id
        self._video_path = video_path
        self._transcript_path = transcript_path
        self._output_dir = output_dir
        self._keyframe_dir = keyframe_dir

        self._checkpoint_file = os.path.join(output_dir, "extracted_requirements_checkpoint.jsonl")
        self._config_file = os.path.join(output_dir, "project_config.json")
        self._docx_file = os.path.join(output_dir, "BRS.docx")
        self._output_file = os.path.join(output_dir, "extracted_requirements.json")

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
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass

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

        # ── STAGE 1: DATA PREPARATION ──
        self._emit(cb, StageUpdate(1, "Data Preparation", "running", "Preparing data...", 0.0))
        stage1_start = time.time()

        if self._video_id:
            if not os.path.exists(self._transcript_path) or os.path.getsize(self._transcript_path) == 0:
                fetch_transcript()
        else:
            if not os.path.exists(self._transcript_path):
                raise FileNotFoundError(
                    f"Transcript not found at {self._transcript_path}. "
                    "Provide a YouTube ID or upload a transcript file."
                )

        chunks = prepare_data(
            transcript_path=self._transcript_path,
            video_path=self._video_path,
            frame_dir=self._keyframe_dir,
        )
        valid_chunks = get_valid_chunks(chunks)
        self._emit(cb, StageUpdate(1, "Data Preparation", "done",
                                    f"Prepared {len(valid_chunks)} chunks",
                                    1.0, elapsed=time.time() - stage1_start))

        # ── STAGE 2: MODEL INITIALIZATION ──
        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Initializing backend...", 0.0))
        stage2_start = time.time()
        self._check_cancelled()

        # Clear any stale MLX GPU state from previous runs (safe no-op if mlx not available)
        self._clear_gpu_cache()
        self._check_cancelled()

        backend = get_backend(self._backend_name)
        self._check_cancelled()

        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Loading VLM proposer...", 0.3))
        self._check_cancelled()
        try:
            proposer = VLMProposer(backend=backend)
        except Exception as e:
            self._emit(cb, StageUpdate(2, "Model Initialization", "error",
                                        f"VLM load failed: {e}", 0,
                                        elapsed=time.time() - stage2_start))
            raise
        self._check_cancelled()

        self._emit(cb, StageUpdate(2, "Model Initialization", "running",
                                   "Validating judge models...", 0.6))
        judges_panel = JudgesPanel(backend=backend)
        judges_panel.preload()
        self._check_cancelled()
        self._emit(cb, StageUpdate(2, "Model Initialization", "done",
                                    "Models initialized", 1.0,
                                    elapsed=time.time() - stage2_start))

        # ── STAGE 3: INFERENCE ──
        self._emit(cb, StageUpdate(3, "Inference", "running",
                                   f"Processing {len(valid_chunks)} chunks...", 0.0))
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
            self._emit(cb, StageUpdate(3, "Inference", "skipped",
                                        "All chunks in checkpoint", 1.0,
                                        elapsed=time.time() - stage3_start))
        else:
            resume_info = (
                f"Resuming: {len(done_indices)}/{len(valid_chunks)} done, "
                f"{len(remaining_chunks)} remaining"
                if checkpointed else None
            )
            if resume_info:
                self._emit(cb, StageUpdate(3, "Inference", "running",
                                            resume_info, 0.0))

            for i, (bucket_index, chunk_data) in enumerate(remaining_chunks, 1):
                self._check_cancelled()

                combined_text = " ".join(chunk_data["text_lines"])
                frame_paths = chunk_data["frame_paths"]
                label = f"Chunk {bucket_index + 1} [{i}/{len(remaining_chunks)}]"

                t = time.time()
                try:
                    response = _with_retry(
                        fn=lambda fp=frame_paths, tx=combined_text: proposer.generate(fp, tx),
                        max_retries=_VLM_MAX_RETRIES,
                        label=label,
                    )
                    elapsed = time.time() - t
                except Exception as e:
                    elapsed = time.time() - t
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
                self._emit(cb, StageUpdate(3, "Inference", "running",
                                            msg, progress))

            inference_elapsed = time.time() - stage3_start
            self._emit(cb, StageUpdate(3, "Inference", "done",
                                        f"Inferred {len(remaining_chunks)} chunks",
                                        1.0, elapsed=inference_elapsed))

        proposer.unload()

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

        # ── STAGE 5: POST-PROCESSING & OUTPUT ──
        self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                   "Dedup, priorities, JSON...", 0.0))
        stage5_start = time.time()
        self._check_cancelled()

        dedup_removed = 0
        all_bullets: list[str] = []
        judge3_handle = None
        judge3_load_elapsed = 0.0
        dedup_elapsed = 0.0
        priority_elapsed = 0.0

        load_start = time.time()
        try:
            judge3_handle = backend.load_llm(DEDUP_MODEL)
        except Exception as e:
            pass
        judge3_load_elapsed = time.time() - load_start

        if judge3_handle is not None:
            try:
                self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                            "Cross-block deduplication...", 0.1))
                dedup_start = time.time()
                refined, dedup_removed = dedup_cross_block(
                    refined, backend=backend, llm_handle=judge3_handle
                )
                dedup_elapsed = time.time() - dedup_start
            except Exception as e:
                dedup_elapsed = time.time() - dedup_start
                dedup_removed = 0

            per_block, all_bullets = self._flatten_flagged_bullets(refined)

            self._emit(cb, StageUpdate(5, "Post-processing", "running",
                                        f"Classifying {len(all_bullets)} priorities...", 0.4))
            priority_start = time.time()
            try:
                priorities = classify_priorities(
                    all_bullets, backend=backend, handle=judge3_handle
                )
                priority_elapsed = time.time() - priority_start
            except Exception as e:
                priority_elapsed = time.time() - priority_start
                priorities = ["S"] * len(all_bullets)

            blocks = self._assemble_blocks(refined, per_block, priorities=priorities)
            self._dump_json(blocks, refined, inference_elapsed, judges_elapsed, dedup_removed)

            # ── STAGE 6: BRS CONFIG GENERATION ──
            self._emit(cb, StageUpdate(6, "Config Generation", "running",
                                        "Inferring document structure...", 0.0))
            stage6_start = time.time()
            try:
                generate_config(
                    all_bullets, self._config_file, backend=backend, handle=judge3_handle
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
            all_bullets = self._write_json(
                refined, inference_elapsed, judges_elapsed, dedup_removed
            )

        self._emit(cb, StageUpdate(5, "Post-processing", "done",
                                    f"Removed {dedup_removed} dupes, "
                                    f"{len(all_bullets)} requirements",
                                    1.0, elapsed=time.time() - stage5_start))

        # ── STAGE 7: DOCX GENERATION ──
        self._emit(cb, StageUpdate(7, "DOCX Generation", "running",
                                    "Generating Word document...", 0.0))
        stage7_start = time.time()
        self._check_cancelled()

        self._generate_docx()

        self._emit(cb, StageUpdate(7, "DOCX Generation", "done",
                                    "BRS.docx generated",
                                    1.0, elapsed=time.time() - stage7_start))

        # Done
        total = time.time() - stage1_start
        self._emit(cb, StageUpdate(0, "Complete", "done",
                                    f"Pipeline finished in {total:.1f}s",
                                    1.0, elapsed=total))

    def _flatten_flagged_bullets(self, refined: list) -> tuple[list[list[str]], list[str]]:
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
        self, refined: list, per_block: list[list[str]],
        priorities: list[str] | None = None,
    ) -> list[dict]:
        blocks = []
        cursor = 0
        for req, bullets in zip(refined, per_block):
            if priorities is not None:
                slice_ = priorities[cursor: cursor + len(bullets)]
                cursor += len(bullets)
                bullet_payload = [
                    {"text": b, "priority": p} for b, p in zip(bullets, slice_)
                ]
            else:
                bullet_payload = bullets
            blocks.append({
                "minute_block": req["minute_block"],
                "time_range": req["time_range"],
                "bullets": bullet_payload,
                "error": req.get("error"),
            })
        return blocks

    def _dump_json(self, blocks: list[dict], refined: list,
                    inference_elapsed: float, judges_elapsed: float,
                    dedup_removed: int) -> None:
        os.makedirs(os.path.dirname(self._output_file), exist_ok=True)
        total = inference_elapsed + judges_elapsed
        successful = sum(1 for r in refined if r.get("refined_requirements"))
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
        with open(self._output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def _write_json(self, refined: list, inference_elapsed: float,
                     judges_elapsed: float, dedup_removed: int) -> list[str]:
        total = inference_elapsed + judges_elapsed
        successful = sum(1 for r in refined if r.get("refined_requirements"))
        blocks: list[dict] = []
        all_bullets: list[str] = []
        for req in refined:
            bullets = (
                extract_bullets(req["refined_requirements"])
                if req.get("refined_requirements")
                else []
            )
            bullets = [flag_unknown_words(b) for b in bullets]
            all_bullets.extend(bullets)
            blocks.append({
                "minute_block": req["minute_block"],
                "time_range": req["time_range"],
                "bullets": bullets,
                "error": req.get("error"),
            })
        output = {
            "summary": {
                "total_chunks": len(refined),
                "successfully_refined": successful,
                "duplicate_bullets_removed_cross_block": 0,
                "inference_time_sec": round(inference_elapsed, 1),
                "judges_time_sec": round(judges_elapsed, 1),
                "total_time_sec": round(total, 1),
            },
            "requirements": blocks,
        }
        os.makedirs(os.path.dirname(self._output_file), exist_ok=True)
        with open(self._output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return all_bullets

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
