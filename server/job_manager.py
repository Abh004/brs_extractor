"""
Job manager — in-memory registry of pipeline jobs with concurrency control.

Only one job runs at a time (GPU exclusivity). Queued jobs are rejected
immediately. Each job has a unique ID, status, and a queue of StageUpdate
events that WebSocket consumers can read from.
"""

from __future__ import annotations

import asyncio
import queue
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from config import OUTPUT_FILE
from server.pipeline_runner import PipelineRunner, StageUpdate


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float
    updated_at: float
    elapsed: float = 0.0
    stage: int = 0
    stage_name: str = ""
    progress: float = 0.0
    message: str = ""
    log: str = ""
    details: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    updates: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=500))
    _runner: PipelineRunner | None = None

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": round(self.elapsed, 1),
            "stage": self.stage,
            "stage_name": self.stage_name,
            "progress": round(self.progress * 100),
            "message": self.message,
            "log": self.log,
            "error": self.error,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._current_job: Job | None = None
        self._lock = asyncio.Lock()

    @property
    def is_busy(self) -> bool:
        return self._current_job is not None and self._current_job.status == JobStatus.RUNNING

    async def create_job(
        self,
        backend_name: str | None = None,
        video_path: str = "input/video.mp4",
    ) -> Job | None:
        async with self._lock:
            if self.is_busy:
                return None

            job_id = uuid.uuid4().hex[:12]
            job = Job(
                id=job_id,
                status=JobStatus.PENDING,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self._jobs[job_id] = job
            self._current_job = job
            return job

    def start_job(
        self,
        job: Job,
        backend_name: str | None = None,
        video_path: str = "input/video.mp4",
    ):
        def on_update(update: StageUpdate):
            now = time.time()
            job.updated_at = now
            job.stage = update.stage
            job.stage_name = update.stage_name
            job.progress = update.progress
            job.message = update.message
            job.elapsed = update.elapsed
            job.log = update.log

            if update.stage > 0 and update.status in ("running", "done", "skipped"):
                if job.status not in (JobStatus.CANCELLED,):
                    job.status = JobStatus.RUNNING
            elif update.status == "error":
                job.error = update.message

            if update.stage == 0:
                if update.status == "done":
                    job.status = JobStatus.COMPLETED
                    job.elapsed = update.elapsed
                elif update.status == "error":
                    job.status = JobStatus.FAILED
                    job.error = update.message
                elif update.status == "cancelled":
                    job.status = JobStatus.CANCELLED
                    job.elapsed = update.elapsed

            try:
                job.updates.put_nowait(update)
            except queue.Full:
                try:
                    job.updates.get_nowait()
                    job.updates.put_nowait(update)
                except queue.Empty:
                    pass

        runner = PipelineRunner(
            backend_name=backend_name,
            video_path=video_path,
        )
        job._runner = runner

        import threading
        thread = threading.Thread(
            target=runner._run_sync,
            args=(on_update,),
            daemon=True,
        )
        thread.start()

    async def cancel_job(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.RUNNING:
                return False
            if job._runner:
                job._runner.cancel()
            # Don't set status to CANCELLED here — let the pipeline thread
            # propagate the status via its own StageUpdate, so the WebSocket
            # gets the final event and the transition is clean.
            return True

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_jobs(self, limit: int = 10) -> list[dict]:
        sorted_jobs = sorted(
            self._jobs.values(), key=lambda j: j.created_at, reverse=True
        )
        return [j.to_dict() for j in sorted_jobs[:limit]]

    async def load_results(self, job_id: str) -> dict | None:
        import json
        import os

        if not os.path.exists(OUTPUT_FILE):
            return None
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None


manager = JobManager()
