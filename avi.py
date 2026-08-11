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
    AVI_UPLOAD_TIMEOUT_SEC,
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


def _redact_url(url: str) -> str:
    """Strip the access token from a URL for logs/errors."""
    token = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN", "")
    if token:
        url = url.replace(f"accessToken={token}", "accessToken=<redacted>")
    return url


def _diagnose(method: str, url: str, resp) -> None:
    """Print a diagnostic line for an AVI call: resolved IP, status, body.

    Called on non-2xx responses so a misrouted call (public vs private
    endpoint, wrong path, auth/channel issue) is self-explanatory in the log.
    The access token is redacted from the body and the printed URL.
    """
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    body = getattr(resp, "text", "") or ""
    token = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN", "")
    display_url = _redact_url(url)
    if token and token in body:
        body = body.replace(token, "<redacted>")
    print(f"  [AVI][diag] {method} {display_url}")
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


_resolve_override_applied = False


def _apply_resolve_override() -> None:
    """Route the AVI endpoint host to a fixed IP at the app layer.

    On a locked-down corporate laptop (no admin / no hosts-file edits), DNS for
    `AZURE_VIDEO_INDEXER_ENDPOINT` may still resolve to the PUBLIC endpoint of
    a private-endpoint-only account, yielding `404 resource not found`. Setting
    `AZURE_VIDEO_INDEXER_RESOLVE_IP` (the private endpoint's IP, e.g. 10.x)
    makes this process connect to that IP while keeping the FQDN in the URL, so
    the TLS SNI / Host header still identify the account correctly.

    Implemented as an idempotent monkeypatch of socket.getaddrinfo (what
    requests/urllib3 use to resolve hosts); every other hostname is passed
    through untouched.
    """
    global _resolve_override_applied
    if _resolve_override_applied:
        return
    ip = os.getenv("AZURE_VIDEO_INDEXER_RESOLVE_IP", "").strip()
    host = os.getenv(
        "AZURE_VIDEO_INDEXER_ENDPOINT", "api.videoindexer.ai"
    ).strip().lower()
    if not ip:
        return
    original = socket.getaddrinfo

    def _patched(hostname, port, *args, **kwargs):
        if hostname is not None and hostname.lower() == host:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
        return original(hostname, port, *args, **kwargs)

    socket.getaddrinfo = _patched
    _resolve_override_applied = True


_apply_resolve_override()


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


def _account_token() -> str:
    """Return the AVI account access token, raising a clear error if unset."""
    token = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "No AVI credential set. Set AZURE_VIDEO_INDEXER_ACCOUNT_TOKEN."
        )
    return token


def _append_access_token(url: str) -> str:
    """Add/substitute the account access token as the `accessToken` query param.

    The AVI data plane (including private-endpoint front doors) authenticates
    via the `?accessToken=` query parameter (as in Microsoft's SDK), not an
    `Authorization` header. Returns a new URL with the token URL-encoded, either
    appended to existing query params or added as the first one.
    """
    token = _account_token()
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query["accessToken"] = token
    encoded = urllib.parse.urlencode(query)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, encoded, parts.fragment)
    )


def _authed(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Build the base request headers for an AVI call.

    Auth is carried via the `?accessToken=` query parameter (see
    `_append_access_token`), so only a JSON content type is set here. A clear
    error names the missing config if the token is not set.
    """
    headers = dict(headers or {})
    headers.setdefault("Content-Type", "application/json")
    _account_token()
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
    _apply_resolve_override()
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
    url = _append_access_token(url)
    display_url = _redact_url(url)
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
            f"AVI network error during {method} {display_url}: {e}. Check your "
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
            f"AVI HTTP {resp.status_code} {method} {display_url} -> {resp.text[:500]}"
        )
    if not resp.text.strip():
        return None
    return resp.json()


def _upload_marker_path(output_dir: str) -> str:
    """Path of the file recording videoIds of uploads started by this tool."""
    return os.path.join(output_dir, "avi_upload.json")


def _read_upload_marker(output_dir: str) -> dict:
    try:
        with open(_upload_marker_path(output_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


def _write_upload_marker(output_dir: str, source: str, video_id: str, name: str) -> None:
    data = _read_upload_marker(output_dir)
    data[source] = {"video_id": video_id, "name": name}
    with open(_upload_marker_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _videos_by_name(base: str, name: str) -> list[dict]:
    """List account videos and return those whose name matches exactly."""
    data = _request("GET", f"{base}/Videos", _authed())
    items = data if isinstance(data, list) else (data or {}).get("results") or []
    if not isinstance(items, list):
        return []
    return [v for v in items if isinstance(v, dict) and v.get("name") == name]


def _fetch_index(base: str, video_id: str, language: str) -> dict:
    """Return the index dict for an already-uploaded video."""
    url = f"{base}/Videos/{video_id}/Index?language={urllib.parse.quote(language)}"
    return _request("GET", url, _authed())


def _save_index(cache_path: str, marker_path: str, state: dict, video_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(_source_identity(video_path))
    print(f"  [AVI] index saved: {cache_path}")


def _upload_video(
    base: str,
    video_path: str,
    name: str,
    language: str,
    output_dir: str,
    source: str,
) -> str:
    """Upload a video file (streaming multipart) with retry/backoff.

    urllib3 won't retry a connection that fails while the request body is being
    written (it can't know how much the server received), so we retry the whole
    upload ourselves. On a "video already exists" rejection, we reuse the
    existing videoId instead of failing. Returns the videoId.
    """
    name_enc = urllib.parse.quote(name)
    upload_url = (
        f"{base}/Videos?name={name_enc}&language={urllib.parse.quote(language)}"
    )
    upload_url = _append_access_token(upload_url)
    upload_headers = _authed({})
    upload_headers.pop("Content-Type", None)  # let requests set multipart boundary
    attempts = AVI_MAX_RETRIES + 1
    last_err = None
    for attempt in range(1, attempts + 1):
        print(
            f"  [AVI] Uploading '{os.path.basename(video_path)}' as '{name}' "
            f"(attempt {attempt}/{attempts})..."
        )
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
                    timeout=(AVI_CONNECT_TIMEOUT, AVI_UPLOAD_TIMEOUT_SEC),
                    verify=_verify(),
                )
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < attempts:
                delay = AVI_RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"  [AVI] upload interrupted ({e}); retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"AVI upload failed after {attempts} attempts: {last_err}. "
                f"Check your VPN/proxy/firewall, or set AVI_NO_PROXY=1 to "
                f"bypass the proxy."
            ) from last_err
        break
    if resp.status_code >= 400:
        display_url = _redact_url(upload_url)
        _diagnose("POST", upload_url, resp)
        text = (resp.text or "").lower()
        if resp.status_code in (400, 409) and any(
            token in text for token in ("exist", "conflict", "duplicate")
        ):
            existing = _videos_by_name(base, name)
            if existing and existing[0].get("id"):
                vid = existing[0]["id"]
                print(
                    f"  [AVI] Upload rejected ({resp.status_code}); video "
                    f"'{name}' already exists — reusing id {vid}."
                )
                return vid
        raise RuntimeError(
            f"AVI HTTP {resp.status_code} POST {display_url} -> {resp.text[:500]}"
        )
    result = resp.json() if resp.text.strip() else None
    video_id = result.get("id") if isinstance(result, dict) else None
    if not video_id:
        raise RuntimeError(f"AVI upload returned no video ID: {result}")
    print(f"  [AVI] videoId: {video_id} | status: {result.get('state')}")
    _write_upload_marker(output_dir, source, video_id, name)
    return video_id


def _wait_for_index(base: str, video_id: str, language: str) -> dict:
    """Poll AVI until the video is indexed (Processed) or fails."""
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
            return state
        if video_state == "Failed":
            raise RuntimeError(f"AVI indexing failed for video {video_id}: {state}")
        print(
            f"  [AVI] indexing state: {video_state}. retrying in "
            f"{AVI_POLL_INTERVAL_SEC}s..."
        )
        time.sleep(AVI_POLL_INTERVAL_SEC)


def upload_and_index(
    video_path: str,
    name: str = AVI_UPLOAD_NAME,
    language: str = AVI_LANGUAGE,
    output_dir: str = "output",
    index_file: str = AVI_INDEX_FILE,
) -> dict:
    """
    Upload a video to AVI, wait for it to finish indexing, and return the
    parsed index JSON.

    Avoids redundant uploads in this order:
      1. Reuse a cached index for the same source video.
      2. Reuse an EXISTING, fully-processed account video with the same name.
      3. Resume a videoId this tool uploaded earlier (marker file) that is
         still processing.
      4. Upload a fresh copy (retried with backoff for mid-stream aborts) and
         persist its videoId so an interrupted run never re-streams the file.
    """
    location = _env("AZURE_VIDEO_INDEXER_LOCATION")
    account_id = _env("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    base = _account_url(location, account_id)

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, index_file)
    marker_path = cache_path + ".marker"
    source = _source_identity(video_path)

    # 1. Reuse a cached index if present AND it belongs to the same source video.
    if os.path.exists(cache_path) and os.path.exists(marker_path):
        if _marker_matches(marker_path, video_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                print(f"  [AVI] Reusing cached index: {cache_path}")
                return cached
        else:
            print("  [AVI] Source video changed; re-indexing (discarding stale cache).")

    # 2. Reuse an existing, fully-processed account video with the same name.
    existing = _videos_by_name(base, name)
    if existing:
        match = existing[0]
        state_code = match.get("state")
        if state_code == "Processed":
            print(
                f"  [AVI] Found existing processed video '{name}' "
                f"({match.get('id')}); reusing it."
            )
            state = _fetch_index(base, match["id"], language)
            _save_index(cache_path, marker_path, state, video_path)
            return state
        print(
            f"  [AVI] Found existing video '{name}' but it is {state_code}; "
            f"only Processed videos are reused — continuing."
        )

    # 3. Resume a videoId this tool uploaded earlier (still processing), or
    #    upload a fresh copy.
    video_id = None
    own = _read_upload_marker(output_dir).get(source)
    if own and own.get("video_id"):
        vid = own["video_id"]
        try:
            probe = _fetch_index(base, vid, language)
            videos = probe.get("videos") if isinstance(probe, dict) else None
            st = videos[0].get("state") if videos else None
        except RuntimeError:
            st = None
        if st == "Processed":
            print(
                f"  [AVI] Previously uploaded video {vid} finished processing; "
                f"reusing it."
            )
            _save_index(cache_path, marker_path, probe, video_path)
            return probe
        if st in ("Processing", "Uploaded"):
            print(
                f"  [AVI] Resuming indexing for previously uploaded video "
                f"{vid} (state: {st})."
            )
            video_id = vid
    if video_id is None:
        video_id = _upload_video(base, video_path, name, language, output_dir, source)

    state = _wait_for_index(base, video_id, language)
    _save_index(cache_path, marker_path, state, video_path)
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


def get_ocr_spatial(index: dict) -> list[dict]:
    """Return OCR segments with normalized screen position.

    Each item: {start, end, text, left, top, width, height} where left/top/
    width/height are the AVI-normalized (0..1) bounding box of the on-screen
    text. Placement lets the judge describe button/menu locations, which the
    old keyframe VLM captured but plain OCR text cannot.
    """
    items = _insights(index).get("ocr") or []
    out: list[dict] = []
    for it in items:
        text = it.get("text") or ""
        if not text:
            continue
        inst = (it.get("instances") or [{}])
        inst = inst[0] if isinstance(inst, list) and inst else {}
        start = inst.get("start", it.get("start"))
        end = inst.get("end", it.get("end"))
        out.append({
            "start": start,
            "end": end,
            "text": text,
            "left": it.get("left"),
            "top": it.get("top"),
            "width": it.get("width"),
            "height": it.get("height"),
        })
    return out


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