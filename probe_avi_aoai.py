"""
probe_avi_aoai.py

Standalone diagnostic for the Azure OpenAI connection on an Azure Video
Indexer account. Reuses the avi.py plumbing (access-token auth, private-endpoint
DNS override, TLS/proxy settings) and touches nothing in the pipeline.

Two modes:
  * Default: dump the account object, resolve the indexed demo video id, and
    inspect AVI's prompt-ready content (text + keyframes) — useful reference for
    the Foundry vision path.
  * --check-vision: send a tiny image through the Foundry chat-completions
    backend to confirm FOUNDRY_DEPLOYMENT accepts image input.

(AVI's native Create-Video-Summary / deployment-discovery path was removed as
part of the Foundry-only cleanup; the component-placement pass now fetches
keyframes itself and uses the user's FOUNDRY_DEPLOYMENT.)

Run:  python probe_avi_aoai.py            (account / video / prompt content)
      python probe_avi_aoai.py --check-vision
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

import avi

from config import AVI_INDEX_FILE, AVI_UPLOAD_NAME


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


# ── Vision support check ─────────────────────────────────────────────────────
def check_foundry_vision() -> None:
    """Verify the Foundry backend's deployment accepts image input.

    Sends a tiny (1x1 JPEG) base64 image via the chat-completions vision format to
    FOUNDRY_DEPLOYMENT. A 400 here (e.g. "doesn't support images" / vision not
    enabled) means the keyframe vision pass can't run with that deployment.
    """
    print("\n[VISION] checking Foundry deployment for image input support ...")
    try:
        import backend as _b
        import types
        from backend import get_backend

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
        print("  [VISION] Foundry deployment ACCEPTS image input. Vision pass is viable.")
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

    dump_prompt_content(base, video_id)

    print("\nDone.")


if __name__ == "__main__":
    main()