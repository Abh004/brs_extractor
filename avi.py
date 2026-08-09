"""
avi.py

Azure Video Indexer (AVI) client — the cloud "proposer" that replaces the
local VLM. Uploads a video, waits for indexing to finish, and returns the
rich index (per-20s-chunk transcript + OCR screen-text) that the judge LLM
turns into requirement bullets.

Credentials come from the environment (loaded via python-dotenv in the
pipeline entry points):

  AZURE_VIDEO_INDEXER_LOCATION          e.g. "eastus" (or "trial")
  AZURE_VIDEO_INDEXER_ACCOUNT_ID        AVI account GUID
  AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN     Contributor account access token
  AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY  optional x-trialsubscriptionkey fallback

All HTTP is done with the stdlib urllib so no third-party dependency is
required for the cloud path.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from config import (
    AVI_INDEX_FILE,
    AVI_INDEX_TIMEOUT_SEC,
    AVI_LANGUAGE,
    AVI_POLL_INTERVAL_SEC,
    AVI_UPLOAD_NAME,
)


def _env(key: str) -> str:
    """Read a required AVI env var."""
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required Azure Video Indexer config: {key}. "
            f"Set it in your environment / .env file."
        )
    return value


def _source_identity(path: str) -> str:
    """Return a stable fingerprint of the source video file (size + mtime)."""
    st = os.stat(path)
    return f"{st.st_size}::{st.st_mtime_ns}"


def _marker_matches(marker_path: str, video_path: str) -> bool:
    """True if an existing marker records the SAME source video as `video_path`."""
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return f.read().strip() == _source_identity(video_path)
    except OSError:
        return False


def _account_url(location: str, account_id: str) -> str:
    """Base URL for a Video Indexer account."""
    return f"https://api.videoindexer.ai/{location}/Accounts/{account_id}"


def _authed(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Build request headers, appending the Contributor access token."""
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault(
        "Authorization", f"Bearer {_env('AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN')}"
    )
    # Optional fallback key (e.g. the account "x-trialsubscriptionkey") used by
    # the video indexer API alongside bearer token auth.
    sub_key = os.getenv("AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY", "").strip()
    if sub_key:
        headers["x-trialsubscriptionkey"] = sub_key
    return headers


def _request(
    method: str, url: str, headers: dict[str, str], body: bytes | None = None
) -> Any:
    """Perform an HTTP request and return the parsed JSON body."""
    req = urllib.request.Request(url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"AVI HTTP {e.code} {method} {url} -> {detail}") from e
    if not raw:
        return None
    return json.loads(raw)


def upload_and_index(
    video_path: str,
    name: str = AVI_UPLOAD_NAME,
    language: str = AVI_LANGUAGE,
    output_dir: str = "output",
    index_file: str = AVI_INDEX_FILE,
) -> dict:
    """
    Upload a video to AVI, wait for it to finish indexing, and return the
    parsed index JSON. Result is cached to {output_dir}/{index_file} and
    reused on re-runs for the SAME video (matched by mtime+size), so a
    changed/uploaded video is re-indexed rather than served stale.
    """
    location = _env("AZURE_VIDEO_INDEXER_LOCATION")
    account_id = _env("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    base = _account_url(location, account_id)

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, index_file)
    marker_path = cache_path + ".marker"

    # Reuse a cached index if present AND it belongs to the same source video.
    if os.path.exists(cache_path) and os.path.exists(marker_path):
        if _marker_matches(marker_path, video_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                print(f"  [AVI] Reusing cached index: {cache_path}")
                return cached
        else:
            print("  [AVI] Source video changed; re-indexing (discarding stale cache).")

    # Upload (multipart).
    name_enc = urllib.parse.quote(name)
    upload_url = f"{base}/Videos/{name_enc}?language={urllib.parse.quote(language)}"
    print(f"  [AVI] Uploading '{os.path.basename(video_path)}' as '{name}'...")
    boundary = "----aviupload" + str(int(time.time() * 1000))
    body, content_type = _multipart(video_path, boundary)
    upload_headers = _authed({})
    upload_headers["Content-Type"] = content_type
    result = _request("POST", upload_url, upload_headers, body)
    video_id = result.get("id") if isinstance(result, dict) else None
    if not video_id:
        raise RuntimeError(f"AVI upload returned no video ID: {result}")
    print(f"  [AVI] videoId: {video_id} | status: {result.get('state')}")

    # Poll until processed.
    index_url = (
        f"{base}/Videos/{video_id}/Index?"
        f"language={urllib.parse.quote(language)}"
    )
    start = time.time()
    while True:
        if time.time() - start > AVI_INDEX_TIMEOUT_SEC:
            raise TimeoutError(
                f"AVI indexing did not finish within {AVI_INDEX_TIMEOUT_SEC}s. "
                f"Check the account in the Azure portal."
            )
        state = _request("GET", index_url, _authed())
        videos = state.get("videos") if isinstance(state, dict) else None
        video_state = videos[0].get("state") if videos else None
        if video_state == "Processed":
            break
        if video_state == "Failed":
            raise RuntimeError(f"AVI indexing failed for video {video_id}: {state}")
        print(f"  [AVI] indexing state: {video_state}. retrying in {AVI_POLL_INTERVAL_SEC}s...")
        time.sleep(AVI_POLL_INTERVAL_SEC)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(_source_identity(video_path))
    print(f"  [AVI] index saved: {cache_path}")
    return state


def _multipart(path: str, boundary: str) -> tuple[bytes, str]:
    """Build a multipart file body and its Content-Type header value."""
    with open(path, "rb") as f:
        payload = f.read()
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(path)}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + payload + tail, f"multipart/form-data; boundary={boundary}"


def _insights(index: dict) -> dict:
    """Return the AVI insights dict (empty if absent)."""
    videos = index.get("videos") if isinstance(index, dict) else None
    if not videos:
        return {}
    return videos[0].get("insights") or {}


def _segs(items: list | None) -> list[dict]:
    """Normalise AVI insights items (with start/end) to a compact list."""
    if not items:
        return []
    out = []
    for it in items:
        start = it.get("start")
        end = it.get("end")
        text = it.get("text") or ""
        if not text:
            continue
        out.append({"start": start, "end": end, "text": text})
    return out


def get_transcript(index: dict) -> list[dict]:
    """Return transcript segments [{start, end, text}]."""
    return _segs(_insights(index).get("transcript") or [])


def get_ocr(index: dict) -> list[dict]:
    """Return OCR screen-text segments [{start, end, text}]."""
    return _segs(_insights(index).get("ocr") or [])


def save_index(index: dict, output_dir: str = "output", index_file: str = AVI_INDEX_FILE) -> str:
    """Persist the AVI index JSON to disk. Returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, index_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return path


def load_index(index_path: str) -> dict:
    """Load a previously saved AVI index from disk."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)