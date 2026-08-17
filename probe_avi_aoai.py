"""
probe_avi_aoai.py

Standalone diagnostic for the Azure OpenAI connection on an Azure Video
Indexer account. Reuses the avi.py plumbing (access-token auth, private-endpoint
DNS override, TLS/proxy settings) and touches nothing in the pipeline.

The deployment name of the connected AOAI resource is NOT visible via the AVI
account object (it lives on the AOAI resource, which the user may not be able
to read). AVI's Create-Video-Summary API rejects unknown deployment names with
"no deployment found with this name", so we iterate likely candidates and latch
onto the first one that validates. Once found, we run a real frame-grounded
summarization (includedFrames=Keyframes) with an instruction asking for a
VLM-style UI description, poll for it, and print GPT's output.

Run:  python probe_avi_aoai.py
Optional: set AZURE_VIDEO_INDEXER_DEPLOYMENT to skip the discovery loop.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse

from dotenv import load_dotenv

import avi

from config import (
    AVI_INDEX_FILE,
    AVI_KEYFRAME_SUMMARY_LENGTH,
    AVI_KEYFRAME_SUMMARY_STYLE,
    AVI_UPLOAD_NAME,
)

PROBE_SUMMARY_TIMEOUT_SEC = 600  # max seconds to wait for a summary job

# Likely Azure OpenAI deployment names for a private company account. The
# discovery loop stops as soon as one validates (or a default exists).
_DEPLOYMENT_CANDIDATES = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-35-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
]

# Instruction appended to the frame-grounded summary so the output reads like
# the old keyframe VLM proposer (UI + button placement + requirement bullets).
_KEYFRAME_INSTRUCTIONS = (
    "Act as a strict systems analyst watching a software demo. For each screen, "
    "describe the UI layout: the placement of buttons, menus, panels and dialogs "
    "(top/bottom, left/center/right and relative ordering). Then list the observed "
    "functional requirements as distinct bullets starting with 'The system shall '."
)


def _probe(method: str, url: str):
    """Raw request; returns (status_code, body_text) without raising on 4xx/5xx."""
    url = avi._append_access_token(url)
    resp = avi._session().request(
        method,
        url,
        headers=avi._authed(),
        timeout=(avi.AVI_CONNECT_TIMEOUT, avi.AVI_READ_TIMEOUT),
        verify=avi._verify(),
    )
    return resp.status_code, (resp.text or "")


def _try_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _probe_get(base: str, suffix: str):
    return _probe("GET", f"{base}{suffix}")


def _probe_post(base: str, suffix: str, query: dict[str, str] | None = None):
    qs = urllib.parse.urlencode(query or {})
    return _probe("POST", f"{base}{suffix}{'?' + qs if qs else ''}")


# ── Phase A: account + video resolution ──────────────────────────────────────
def dump_account(base: str) -> None:
    """Print account identity and any AOAI-related fields (best-effort)."""
    code, text = _probe_get(base, "")
    print(f"  [A] GET account -> HTTP {code}")
    if code >= 400:
        print(f"  [A] body: {text[:500]}")
        return
    data = _try_json(text)
    if not data:
        print(f"  [A] non-JSON body: {text[:300]}")
        return
    print(f"  [A] account id: {data.get('id')} | name: {data.get('name')} | "
          f"location: {data.get('location')}")
    flat = json.dumps(data).lower()
    if "openai" in flat or "deployment" in flat:
        print(f"  [A] AOAI-related keys present in account object:")
        def walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if "openai" in k.lower() or "deploy" in k.lower():
                        print(f"  [A]   {key} = {json.dumps(v)[:200]}")
                    walk(v, key)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    walk(item, f"{prefix}[{i}]")
        walk(data)
    else:
        print("  [A] no AOAI/deployment fields found in account object (expected).")


def resolve_video_id(base: str) -> str:
    """Find the already-indexed demo video id from cache, else account list."""
    idx_path = os.path.join("output", AVI_INDEX_FILE)
    if os.path.exists(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                d = json.load(f)
            v = (d.get("videos") or [{}])[0]
            if v.get("id"):
                print(f"  [V] videoId from cached index ({idx_path}): {v['id']}")
                return str(v["id"])
        except Exception as e:  # noqa: BLE001
            print(f"  [V] could not read cached index ({e}); falling back to list")
    code, text = _probe_get(base, "/Videos")
    if code >= 400:
        print(f"  [V] GET /Videos -> HTTP {code}: {text[:400]}")
        raise SystemExit(f"cannot list videos: HTTP {code}")
    data = _try_json(text) or {}
    items = data if isinstance(data, list) else (data.get("results") or [])
    for v in items if isinstance(items, list) else []:
        if isinstance(v, dict) and v.get("name") == AVI_UPLOAD_NAME and v.get("id"):
            print(f"  [V] videoId from account list: {v['id']} "
                  f"(name='{AVI_UPLOAD_NAME}', state={v.get('state')})")
            return str(v["id"])
    raise SystemExit(
        f"video named '{AVI_UPLOAD_NAME}' not found in account. Upload/index it "
        f"first (python main.py) or adjust AVI_UPLOAD_NAME."
    )


# ── Phase B: deployment discovery ────────────────────────────────────────────
def _create_summary(base: str, video_id: str, deployment: str | None,
                    keyframes: bool, instructions: str | None = None):
    """POST Create-Video-Summary; returns (status_code, body_text)."""
    query: dict[str, str] = {
        "length": AVI_KEYFRAME_SUMMARY_LENGTH,
        "style": AVI_KEYFRAME_SUMMARY_STYLE,
    }
    if deployment:
        query["deploymentName"] = deployment
    if keyframes:
        query["includedFrames"] = "Keyframes"
    if instructions:
        query["addToEndOfSummaryInstructions"] = instructions
    return _probe_post(base, f"/Videos/{video_id}/Summaries/Textual", query)


def _summary_id_from(text: str) -> str | None:
    """Pull the summary id out of the create response (dict or bare string)."""
    t = text.strip()
    if not t:
        return None
    if t.startswith("{"):
        d = _try_json(t)
        if isinstance(d, dict):
            for k in ("id", "summaryId", "summary_id", "summaryID"):
                if d.get(k):
                    return str(d[k])
        return None
    if " " not in t and "{" not in t and len(t) < 64:
        return t
    return None


def discover_deployment(base: str, video_id: str) -> str | None:
    """Return a working deployment name, or None if the default is used.

    Tries no-deployment first (AVI may have a default), then the candidate
    list. A non-4xx create response means the name validated.
    """
    override = os.getenv("AZURE_VIDEO_INDEXER_DEPLOYMENT", "").strip()
    if override:
        print(f"  [D] using AZURE_VIDEO_INDEXER_DEPLOYMENT={override}")
        return override

    print("  [D] trying without deploymentName (account default) ...")
    code, text = _create_summary(base, video_id, None, keyframes=False)
    if code < 400:
        print(f"  [D] no deploymentName needed -> HTTP {code}; using default.")
        return None
    print(f"  [D]   -> HTTP {code}: {text[:200]}")

    for name in _DEPLOYMENT_CANDIDATES:
        code, text = _create_summary(base, video_id, name, keyframes=False)
        if code < 400:
            print(f"  [D] '{name}' VALIDATES -> HTTP {code}.")
            return name
        print(f"  [D]   '{name}' -> HTTP {code}: {text[:160]}")

    print("  [D] none of the candidate deployment names validated.")
    print("  [D] paste the exact error body above into the ticket / tell me, and")
    print("  [D] I can add more candidates, or set AZURE_VIDEO_INDEXER_DEPLOYMENT.")
    return None


# ── Phase C: frame-grounded quality test ─────────────────────────────────────
def run_keyframe_summary(base: str, video_id: str, deployment: str | None) -> None:
    """Create a keyframe-based summary with the UI instruction and print it."""
    print("\n[QUALITY] creating keyframe-based textual summary ...")
    code, text = _create_summary(
        base, video_id, deployment, keyframes=True, instructions=_KEYFRAME_INSTRUCTIONS
    )
    print(f"  [Q] create -> HTTP {code}: {text[:400]}")
    if code >= 400:
        print("  [Q] create failed; nothing to poll.")
        return

    sid = _summary_id_from(text)
    if sid is None:
        # The API may have returned the summary directly; check for text.
        d = _try_json(text)
        if isinstance(d, dict) and _summary_text(d):
            print("  [Q] summary returned inline:")
            print(_summary_text(d))
            return
        print(f"  [Q] could not parse a summary id from response:\n{text[:800]}")
        return

    get_url = f"/Videos/{video_id}/Summaries/Textual/{urllib.parse.quote(sid)}"
    start = time.time()
    print(f"  [Q] polling GET {get_url} ...")
    while time.time() - start < PROBE_SUMMARY_TIMEOUT_SEC:
        time.sleep(10)
        code, text = _probe_get(base, get_url)
        d = _try_json(text)
        if isinstance(d, dict):
            st = _summary_status(d)
            out = _summary_text(d)
            if out:
                print(f"  [Q] summary ready (HTTP {code}):")
                print("-" * 60)
                print(out)
                print("-" * 60)
                return
            if st in ("Failed", "Error"):
                print(f"  [Q] summary {st}:\n{text[:1500]}")
                return
            print(f"  [Q]   status: {st or 'unknown'} (HTTP {code}) ...")
        else:
            print(f"  [Q]   HTTP {code}, not ready yet ...")
    print(f"  [Q] timed out after {PROBE_SUMMARY_TIMEOUT_SEC}s. "
          f"Re-run with the same deployment to check later.")


def _summary_status(d: dict) -> str | None:
    for k in ("status", "state", "jobStatus"):
        if d.get(k):
            return str(d[k])
    return None


def _summary_text(d: dict) -> str | None:
    for k in ("text", "summary", "result", "output", "content"):
        if isinstance(d.get(k), str) and d[k].strip():
            return d[k]
    return None


# ── Optional reference: raw Prompt Content material ──────────────────────────
def dump_prompt_content(base: str, video_id: str) -> None:
    """Fetch AVI's prompt-ready content (text + keyframe ids) if present."""
    code, text = _probe_get(base, f"/Videos/{video_id}/PromptContent")
    print(f"\n[REF] GET /PromptContent -> HTTP {code}")
    if code >= 400:
        print(f"  [REF]   {text[:200]} (run the create step or ignore)")
        return
    d = _try_json(text)
    if not isinstance(d, dict):
        print(f"  [REF]   {text[:400]}")
        return
    segs = d.get("segments") or d.get("sections") or []
    print(f"  [REF]   prompt content: {len(segs)} segments; "
          f"keys: {sorted(d.keys())[:12]}")
    for seg in segs[:1]:
        content = seg.get("content", "")
        frames = seg.get("frames") or []
        print(f"  [REF]   sample segment start={seg.get('start')} frames={len(frames)}")
        print(f"  [REF]   {str(content)[:300]}")
    if segs and any(seg.get("frames") for seg in segs):
        print("  [REF]   keyframes ARE included -> a vision-capable deployment "
              "can be fed these frames via a BYO model.")


def check_foundry_vision() -> None:
    """Verify the Foundry backend's deployment accepts image input.

    Sends a tiny (1x1 JPEG) base64 image via the chat-completions vision format to
    FOUNDRY_DEPLOYMENT. A 400 here (e.g. "doesn't support images" / vision not
    enabled) means Route B can't run with that deployment.
    """
    print("\n[VISION] checking Foundry deployment for image input support ...")
    try:
        from backend import get_backend
        import backend as _b
        import types

        endpoint_given = os.getenv("FOUNDRY_ENDPOINT") and os.getenv("FOUNDRY_API_KEY") \
            and os.getenv("FOUNDRY_DEPLOYMENT")
        if not endpoint_given:
            print("  [VISION] set FOUNDRY_ENDPOINT / FOUNDRY_API_KEY / FOUNDRY_DEPLOYMENT "
                  "to run this check.")
            return
        inst = get_backend("foundry")
        captured = {}
        def fake_post(url, headers=None, json=None, timeout=None, verify=None):
            captured["payload"] = json
            class R:
                status_code = 200
                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            return R()
        inst._session = lambda: types.SimpleNamespace(post=fake_post)
        out = inst.generate_llm(
            {"model_id": inst._deployment},
            "Test vision support.",
            "Describe this image.",
            max_tokens=8,
            images=["data:image/jpeg;base64,/9j/4AAQSkZJRg=="],
        )
        parts = captured["payload"]["messages"][1]["content"]
        kinds = [p["type"] for p in parts]
        print(f"  [VISION] payload OK; content parts: {kinds}; detail="
              f"{parts[-1]['image_url'].get('detail')}")
        print(f"  [VISION] response: {out!r}")
        print("  [VISION] Foundry deployment ACCEPTS image input. Route B is viable.")
    except Exception as e:
        print(f"  [VISION] Foundry deployment REJECTED image input: {e}")
        print("  [VISION] Check that FOUNDRY_DEPLOYMENT is a vision-enabled gpt-4o "
              "deployment (image input on).")


def main() -> None:
    load_dotenv()
    if "--check-vision" in sys.argv:
        check_foundry_vision()
        return

    location = avi._env("AZURE_VIDEO_INDEXER_LOCATION")
    account_id = avi._env("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    base = avi._account_url(location, account_id)
    avi._account_token()  # fail fast with a clear error if unset

    print("=" * 60)
    print("AVI <-> Azure OpenAI probe")
    print(f"account: {account_id} @ {location} | base: {base}")
    print("=" * 60)

    dump_account(base)

    print("\n[VIDEO] resolving the demo video ...")
    video_id = resolve_video_id(base)

    print("\n[DEPLOYMENT] discovering the connected AOAI deployment ...")
    deployment = discover_deployment(base, video_id)

    run_keyframe_summary(base, video_id, deployment)

    dump_prompt_content(base, video_id)

    print("\nDone. If deployment discovery returned None but you saw a specific")
    print("error, set AZURE_VIDEO_INDEXER_DEPLOYMENT=<name> and re-run.")


if __name__ == "__main__":
    main()
