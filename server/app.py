"""
FastAPI application for the BRS extraction pipeline.
Serves the frontend SPA, handles file uploads, manages pipeline jobs,
and streams progress via WebSocket.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
import queue

from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG_FILE, DOCX_FILE, OUTPUT_FILE, TRANSCRIPT_PATH, VIDEO_PATH
from server.job_manager import JobStatus, manager


async def _get_update_async(job, timeout: float = 1.0):
    """Thread-safe queue poll: runs queue.get() in a thread executor without
    blocking the asyncio event loop."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, job.updates.get, timeout),
        timeout=timeout + 0.5,
    )

app = FastAPI(title="BRS Pipeline")


# ── static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ── API endpoints ─────────────────────────────────────────────────────────────


@app.post("/api/run")
async def run_pipeline(
    video_id: str | None = Form(None),
    backend: str | None = Form(None),
    video_file: UploadFile | None = None,
    transcript_file: UploadFile | None = None,
):
    if manager.is_busy:
        raise HTTPException(409, "A pipeline job is already running. Wait for it to finish or cancel it.")

    os.makedirs(os.path.dirname(VIDEO_PATH) or ".", exist_ok=True)

    video_path = VIDEO_PATH
    transcript_path = TRANSCRIPT_PATH

    if video_file and video_file.filename:
        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)

    if transcript_file and transcript_file.filename:
        with open(transcript_path, "wb") as f:
            content = await transcript_file.read()
            f.write(content)

    job = await manager.create_job(
        backend_name=backend,
        video_id=video_id or None,
        video_path=video_path,
        transcript_path=transcript_path,
    )
    if job is None:
        raise HTTPException(409, "Could not create job (concurrency limit).")

    manager.start_job(job, backend_name=backend, video_id=video_id or None,
                       video_path=video_path, transcript_path=transcript_path)

    return {"job_id": job.id, "status": "started"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return {"status": job.status.value, "message": "Pipeline still running"}
    results = await manager.load_results(job_id)
    return {"status": job.status.value, "results": results}


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    ok = await manager.cancel_job(job_id)
    if not ok:
        raise HTTPException(400, "Job not found or not running")
    return {"status": "cancelled"}


@app.get("/api/jobs")
async def list_jobs():
    return manager.get_jobs(limit=20)


@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    allowed = {
        os.path.basename(OUTPUT_FILE),
        os.path.basename(DOCX_FILE),
        os.path.basename(CONFIG_FILE),
    }
    if filename not in allowed:
        raise HTTPException(403, "File not available for download")
    filepath = os.path.join(os.path.dirname(OUTPUT_FILE), filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, f"{filename} not found — pipeline may not have completed")
    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if filename.endswith(".docx")
        else "application/json"
    )
    return FileResponse(filepath, media_type=media_type, filename=filename)


# ── WebSocket ─────────────────────────────────────────────────────────────────


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = manager.get_job(job_id)

    if job is None:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    if job.status == JobStatus.COMPLETED:
        await websocket.send_json({"type": "status", "status": "completed",
                                    "message": "Job already completed"})
        await websocket.close()
        return

    try:
        while True:
            update = None
            try:
                update = await _get_update_async(job, timeout=1.0)
            except (asyncio.TimeoutError, queue.Empty):
                pass
            if update is None:
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    break
                await websocket.send_json({"type": "ping"})
                continue

            await websocket.send_json(update.to_payload())

            if update.stage == 0 and update.status in ("done", "error", "cancelled"):
                break

        job_update = job.to_dict()
        job_update["type"] = "job_done"
        await websocket.send_json(job_update)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
