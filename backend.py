"""
backend.py

Inference backend abstraction for the pipeline.

The only backend is Foundry — a hosted Azure Foundry / Azure OpenAI endpoint
(e.g. gpt-4o) used for every text-LLM and vision stage. The proposer is AVI
(cloud indexes OCR + transcript), so no local GPU models are loaded at all.

Usage:
  from backend import get_backend
  backend = get_backend()           # Foundry, reads FOUNDRY_* env vars
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

import requests
from urllib3.util.retry import Retry

from config import (
    FOUNDRY_CONNECT_TIMEOUT,
    FOUNDRY_MAX_RETRIES,
    FOUNDRY_READ_TIMEOUT,
    FOUNDRY_RETRY_BACKOFF,
)
from usage import TokenTracker
from utils import clean_output, is_truncated_think


def _finalize_llm_output(raw: str) -> str:
    """
    Post-process a text-LLM generation: drop truncated think blocks (return
    "") and run the full clean_output pass on the rest.
    """
    if is_truncated_think(raw):
        return ""
    return clean_output(raw)


# ── abstract base ─────────────────────────────────────────────────────────────
class InferenceBackend(ABC):
    """
    Common interface for inference backends.

    The pipeline is single-threaded, so backends need not be safe for
    concurrent calls from worker threads.
    """

    def __init__(self) -> None:
        self.usage = TokenTracker()

    @abstractmethod
    def load_llm(self, model_id: str) -> Any:
        """
        Load a text-only LLM and return an opaque handle.
        The handle is passed back to generate_llm / unload_llm so multiple
        judge models can be managed independently.
        """

    @abstractmethod
    def generate_llm(
        self,
        handle: Any,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,
        images: list[str] | None = None,
    ) -> str:
        """Run text-only inference using a previously loaded handle.

        Args:
            images: Optional list of image data URLs (e.g. "data:image/jpeg;base64,...")
                    for multimodal backends. Text-only backends raise if set.
        """

    @abstractmethod
    def unload_llm(self, handle: Any) -> None:
        """Free an LLM handle returned by load_llm."""

    def preload(self, model_id: str) -> None:
        """
        Optionally pre-download a model's weights so a later load_llm() is fast.

        Remote backends (Foundry) keep this default no-op — nothing is loaded
        locally, so there is nothing to pre-download.
        """

    @contextmanager
    def llm_context(self, model_id: str) -> Iterator[Any]:
        """
        Load an LLM, yield its handle for the duration of the block, then
        unload it. Guarantees the handle is freed even if the body raises —
        preventing a leaked model from blocking later runs.
        """
        handle = self.load_llm(model_id)
        try:
            yield handle
        finally:
            self.unload_llm(handle)

    def clear_cache(self) -> None:
        """Optional: flush any accelerator memory caches."""


class FoundryBackend(InferenceBackend):
    """
    Azure Foundry / Azure OpenAI chat-completions backend (text + vision).

    Replaces all local judge/dedup/priority/config-gen models with a hosted
    model (e.g. gpt-4o) reached over HTTPS. Reads from the environment:

      FOUNDRY_ENDPOINT      base URL, e.g. https://<resource>.openai.azure.com/openai/v1/
      FOUNDRY_API_KEY       API key for the endpoint
      FOUNDRY_DEPLOYMENT    deployment name sent as the "model" field
      FOUNDRY_VERIFY_SSL=0  optional: disable TLS verification (corporate TLS
                            interception / GlobalProtect)
      FOUNDRY_NO_PROXY=1    optional: ignore HTTP(S)_PROXY / NO_PROXY

    load_llm is a lightweight handle factory (nothing is loaded locally); the
    deployment name is resolved from the environment, so the model_id passed by
    the pipeline is used only as a label. generate_llm POSTs an OpenAI-style
    chat/completions request and returns the assistant text. When `images` is
    passed, the user message becomes a multimodal content array so a
    vision-capable deployment can describe video keyframes.
    """

    def __init__(self) -> None:
        self.usage = TokenTracker()
        self._endpoint = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")
        self._api_key = os.getenv("FOUNDRY_API_KEY", "").strip()
        self._deployment = os.getenv("FOUNDRY_DEPLOYMENT", "").strip()
        missing = [
            name
            for name, val in (
                ("FOUNDRY_ENDPOINT", self._endpoint),
                ("FOUNDRY_API_KEY", self._api_key),
                ("FOUNDRY_DEPLOYMENT", self._deployment),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                "Missing required Foundry config: "
                + ", ".join(missing)
                + ". Set them in your environment / .env file."
            )
        self._no_proxy = os.getenv("FOUNDRY_NO_PROXY", "").strip() == "1"

    def _verify(self) -> bool:
        """Return the TLS verification setting (FOUNDRY_VERIFY_SSL=0 bypasses)."""
        if os.getenv("FOUNDRY_VERIFY_SSL", "1").strip() == "0":
            import warnings

            warnings.simplefilter(
                "ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning
            )
            return False
        return True

    def _mask(self, text: str) -> str:
        """Redact the API key from a string (for logs / error messages)."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "<redacted>")
        return text

    def _session(self) -> requests.Session:
        """Return a requests.Session with retry/backoff on transient failures."""
        retry = Retry(
            total=FOUNDRY_MAX_RETRIES,
            connect=FOUNDRY_MAX_RETRIES,
            read=FOUNDRY_MAX_RETRIES,
            status=0,  # don't auto-retry on HTTP status; we surface those clearly
            backoff_factor=FOUNDRY_RETRY_BACKOFF,
            raise_on_status=False,
            status_forcelist=set(),
            allowed_methods=frozenset({"POST"}),
        )
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        if self._no_proxy:
            session.trust_env = False
            session.proxies = {}
        return session

    def load_llm(self, model_id: str) -> dict:
        return {"model_id": self._deployment}

    def generate_llm(
        self,
        handle: dict,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
        repetition_penalty: float | None = None,  # noqa: ARG002 — not an OpenAI param
        repetition_context_size: int = 20,  # noqa: ARG002 — not an OpenAI param
        images: list[str] | None = None,
    ) -> str:
        if images:
            detail = os.getenv("AVI_UI_FRAME_DETAIL", "low")
            user_content = [{"type": "text", "text": user_prompt}]
            for url in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": detail},
                    }
                )
        else:
            user_content = user_prompt
        payload = {
            "model": handle.get("model_id") or self._deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        url = f"{self._endpoint}/chat/completions"
        headers = {"Content-Type": "application/json", "api-key": self._api_key}
        try:
            resp = self._session().post(
                url,
                headers=headers,
                json=payload,
                timeout=(FOUNDRY_CONNECT_TIMEOUT, FOUNDRY_READ_TIMEOUT),
                verify=self._verify(),
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Foundry request failed for model {self._deployment}: "
                f"{self._mask(str(e))}"
            ) from e
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Foundry HTTP {resp.status_code} for model {self._deployment} "
                f"(endpoint {self._endpoint}): {self._mask(resp.text[:500])}"
            )
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        self.usage.add(
            model_id=handle.get("model_id", self._deployment),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
        return _finalize_llm_output(raw)

    def unload_llm(self, handle: dict) -> None:
        pass


def get_backend(name: str | None = None) -> InferenceBackend:
    """
    Return an InferenceBackend instance.

    The pipeline is Foundry-only; any BACKEND value other than "foundry"
    (or an explicit name) is rejected to avoid silently choosing a local
    backend the environment can't support.
    """
    selected = (name or os.getenv("BACKEND") or "foundry").lower()
    if selected != "foundry":
        raise ValueError(
            f"Unknown backend {selected!r}. This pipeline only supports "
            f"'foundry'. Set BACKEND=foundry or leave it unset."
        )
    return FoundryBackend()