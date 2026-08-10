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

The account access token (short-lived, ~1h) is sent as a bearer token.

HTTP uses the requests library (streaming upload + retry/backoff) so
transient connection aborts from corporate proxies/firewalls (e.g. Windows
WinError 10053) are retried instead of failing the run. Set AVI_NO_PROXY=1 to
bypass proxy env vars (HTTP(S)_PROXY / NO_PROXY) and AVI_VERIFY_SSL=0 to skip
TLS verification (GlobalProtect / corporate TLS interception) for these calls.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.parse
from typing import Any

import requests
from urllib3.util.retry import Retry

from config import (
    AVI_CONNECT_TIMEOUT,
    AVI_INDEX_FILE,
    AVI_INDEX_TIMEOUT_SEC,
    AVI_LANGUAGE,
    AVI_MAX_RETRIES,
    AVI_POLL_INTERVAL_SEC,
    AVI_READ_TIMEOUT,
    AVI_RETRY_BACKOFF,
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


def _resolved_ip(host: str) -> str:
    """Best-effort DNS resolution of `host` to a numeric IP (for diagnostics).

    A private-endpoint host should resolve to a 10.x/172.x/192.x address.
    Resolving to a public IP (or failing) is the usual cause of
    "public access has been disabled" / 404 when the account sits behind a
    private endpoint.
    """
    try:
        addrs = [
            info[4][0]
            for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ]
    except OSError as e:
        return f"<resolve error: {e}>"
    return ", ".join(dict.fromkeys(addrs)) or "<none>"


def _diagnose(method: str, url: str, resp) -> None:
    """Print a diagnostic line for an AVI call: resolved IP, status, body.

    Called on non-2xx responses so a misrouted call (public vs private
    endpoint, wrong path, auth/channel issue) is self-explanatory in the log.
    The access token is redacted from the body.
    """
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    body = getattr(resp, "text", "") or ""
    token = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN", "")
    if token and token in body:
        body = body.replace(token, "<redacted>")
    print(f"  [AVI][diag] {method} {url}")
    print(f"  [AVI][diag] resolved IP: {_resolved_ip(host)}")
    print(f"  [AVI][diag] status: {getattr(resp, 'status_code', '?')}")
    print(f"  [AVI][diag] body: {body[:2000]}")


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
    """Base URL for a Video Indexer account.

    The data-plane host defaults to the public endpoint (api.videoindexer.ai).
    For a private-endpoint account, override the host via the env var
    AZURE_VIDEO_INDEXER_ENDPOINT (e.g. "<account>.api.videoindexer.ai") so
    requests resolve to the private endpoint IP.
    """
    host = os.getenv("AZURE_VIDEO_INDEXER_ENDPOINT", "api.videoindexer.ai").strip()
    return f"https://{host}/{location}/Accounts/{account_id}"


def _verify() -> bool | str:
    """Return the TLS verification setting for AVI requests.

    Precedence:
      - AVI_VERIFY_SSL=0  -> disable verification (for GlobalProtect /
                            corporate TLS interception). Suppresses requests'
                            insecure-request warning.
      - otherwise default strict verification (True).
    """
    if os.getenv("AVI_VERIFY_SSL", "1").strip() == "0":
        import warnings

        warnings.simplefilter("ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning)
        return False
    return True


def _authed(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Build request headers using the AVI account access token.

    The account access token (AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN) is sent as a
    bearer token. A clear error names the missing config if it is not set.
    """
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")
    token = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "No AVI credential set. Set AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN."
        )
    headers["Authorization"] = f"Bearer {token}"
    return headers


def _session() -> requests.Session:
    """Return a requests.Session with retry/backoff on transient failures.

    Retries connect/read timeouts and the connection-reset class (proxy /
    firewall aborts, e.g. Windows WinError 10053). Honors HTTP(S)_PROXY /
    NO_PROXY from the environment by default; set AVI_NO_PROXY=1 to bypass
    them for these calls.
    """
    retry = Retry(
        total=AVI_MAX_RETRIES,
        connect=AVI_MAX_RETRIES,
        read=AVI_MAX_RETRIES,
        status=0,  # don't auto-retry on HTTP status; we surface those clearly
        backoff_factor=AVI_RETRY_BACKOFF,
        raise_on_status=False,
        status_forcelist=set(),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if os.getenv("AVI_NO_PROXY", "").strip() == "1":
        session.trust_env = False  # ignore HTTP(S)_PROXY / NO_PROXY
        session.proxies = {}
    return session


def _request(
    method: str, url: str, headers: dict[str, str], body: bytes | None = None
) -> Any:
    """Perform an HTTP request and return the parsed JSON body."""
    try:
        resp = _session().request(
            method,
            url,
            headers=headers,
            data=body,
            timeout=(AVI_CONNECT_TIMEOUT, AVI_READ_TIMEOUT),
            verify=_verify(),
        )
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            "AVI request timed out / connection aborted. Check your "
            "VPN/proxy/firewall, or set AVI_NO_PROXY=1 to bypass the proxy."
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"AVI network error during {method} {url}: {e}. Check your "
            f"VPN/proxy/firewall, or set AVI_NO_PROXY=1 to bypass the proxy."
        ) from e

    if resp.status_code in (401, 403):
        _diagnose(method, url, resp)
        raise RuntimeError(
            f"AVI HTTP {resp.status_code}: check AVI credentials "
            f"(AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN)."
        )
    if resp.status_code >= 400:
        _diagnose(method, url, resp)
        raise RuntimeError(
            f"AVI HTTP {resp.status_code} {method} {url} -> {resp.text[:500]}"
        )
    if not resp.text.strip():
        return None
    return resp.json()


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

    # Upload (streaming multipart via requests).
    name_enc = urllib.parse.quote(name)
    upload_url = f"{base}/Videos/{name_enc}?language={urllib.parse.quote(language)}"
    print(f"  [AVI] Uploading '{os.path.basename(video_path)}' as '{name}'...")
    upload_headers = _authed({})
    try:
        with open(video_path, "rb") as fobj:
            resp = _session().post(
                upload_url,
                headers=upload_headers,
                files={
                    "file": (
                        os.path.basename(video_path),
                        fobj,
                        "application/octet-stream",
                    )
                },
                timeout=(AVI_CONNECT_TIMEOUT, AVI_READ_TIMEOUT),
                verify=_verify(),
            )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"AVI upload failed: {e}. Check your VPN/proxy/firewall, or "
            f"set AVI_NO_PROXY=1 to bypass the proxy."
        ) from e
    if resp.status_code >= 400:
        _diagnose("POST", upload_url, resp)
        raise RuntimeError(
            f"AVI HTTP {resp.status_code} POST {upload_url} -> {resp.text[:500]}"
        )
    result = resp.json() if resp.text.strip() else None
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