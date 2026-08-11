"""
backend.py

Inference backend abstraction for the pipeline.

Select via environment variable:
  BACKEND=mlx   (default) — Apple Silicon, MLX. Fast, native.
  BACKEND=hf    — Any CUDA/CPU box. Uses HuggingFace transformers + PIL.

Usage:
  from backend import get_backend
  backend = get_backend()
  backend = get_backend("hf")
"""

from __future__ import annotations

import os
import platform
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

import requests
from urllib3.util.retry import Retry

from config import (
    ALL_MODELS,
    FOUNDRY_CONNECT_TIMEOUT,
    FOUNDRY_MAX_RETRIES,
    FOUNDRY_READ_TIMEOUT,
    FOUNDRY_RETRY_BACKOFF,
)
from usage import TokenTracker
from utils import clean_output, is_truncated_think, strip_think_blocks


def _result_text(result) -> str:
    """Extract the text from a generation result object or plain string."""
    return str(result.text) if hasattr(result, "text") else str(result)


def _finalize_vlm_output(raw: str) -> str:
    """
    Post-process a VLM generation: drop truncated think blocks (return "")
    and strip any complete <think>…</think> blocks from the rest.
    """
    if is_truncated_think(raw):
        return ""
    return strip_think_blocks(raw)


def _finalize_llm_output(raw: str) -> str:
    """
    Post-process a text-LLM generation: drop truncated think blocks (return
    "") and run the full clean_output pass on the rest.
    """
    if is_truncated_think(raw):
        return ""
    return clean_output(raw)


def _apply_chat_template(processor, messages, **kwargs) -> Any:
    """
    Apply a chat template, disabling Qwen3 thinking. Newer transformers
    expect the flag inside processor_kwargs; older versions accept it as a
    top-level kwarg — try the new form first and fall back on TypeError.
    """
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            processor_kwargs={"enable_thinking": False},
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            **kwargs,
        )


# ── abstract base ─────────────────────────────────────────────────────────────
class InferenceBackend(ABC):
    """
    Common interface for all inference backends.

    Implementations must be safe to call from the main thread only — MLX GPU
    streams are thread-local and cannot be used from ThreadPoolExecutor workers.
    """

    def __init__(self) -> None:
        self.usage = TokenTracker()

    @abstractmethod
    def load_vlm(self, model_id: str) -> None:
        """Load a VLM into memory. Must be called before generate_vlm."""

    @abstractmethod
    def generate_vlm(
        self,
        frame_paths: list[str],
        transcript: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,
    ) -> str:
        """Run multimodal inference. Returns raw text (not cleaned)."""

    @abstractmethod
    def unload_vlm(self) -> None:
        """Free VLM weights. Safe to call even if nothing is loaded."""

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
    ) -> str:
        """Run text-only inference using a previously loaded handle."""

    @abstractmethod
    def unload_llm(self, handle: Any) -> None:
        """Free an LLM handle returned by load_llm."""

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
        """Optional: flush any accelerator memory caches (MLX Metal cache etc.)."""


class MLXBackend(InferenceBackend):
    """
    Apple Silicon backend using mlx-vlm (VLM) and mlx_lm (LLM).

    Thread safety: MLX GPU streams are thread-local. This backend must only
    be called from the main thread — do not wrap calls in ThreadPoolExecutor.
    """

    _NO_THINK_MODELS = {m.mlx_id for m in ALL_MODELS if m.no_think}
    _VLM_BACKED_TEXT_MODELS = {m.mlx_id for m in ALL_MODELS if m.vlm_backed}

    def __init__(self) -> None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        self.usage = TokenTracker()
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_config = None
        self._vlm_model_id: str | None = None

    def _mlx_load_with_fallback(self, load_fn, model_id: str, **kwargs):
        """Try loading an MLX model, falling back to local files on network failure."""
        try:
            return load_fn(model_id, **kwargs)
        except Exception as first_exc:
            print(f"  [MLX] Remote load failed for {model_id}; trying local files only.")
            try:
                return load_fn(
                    model_id,
                    hf_hub_download_kwargs={"local_files_only": True},
                    **kwargs,
                )
            except TypeError:
                print(
                    f"  [MLX] Local-only fallback not supported by this library version; "
                    f"re-raising original error."
                )
                raise first_exc

    def load_vlm(self, model_id: str) -> None:
        import warnings

        warnings.filterwarnings(
            "ignore", message="Kwargs passed to `processor.__call__`"
        )
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"  [MLX] Loading VLM: {model_id}")
        self._vlm_model, self._vlm_processor = self._mlx_load_with_fallback(load, model_id)
        self._vlm_config = self._mlx_load_with_fallback(load_config, model_id)
        self._vlm_model_id = model_id
        print("  [MLX] VLM loaded.")

    def generate_vlm(
        self,
        frame_paths: list[str],
        transcript: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if self._vlm_model is None:
            raise RuntimeError("VLM not loaded. Call load_vlm() first.")

        prompt_text = system_prompt.format(transcript=transcript)
        if self._vlm_model_id in self._NO_THINK_MODELS:
            prompt_text += " /no_think"
        prompt = apply_chat_template(
            self._vlm_processor,
            self._vlm_config,
            prompt_text,
            num_images=len(frame_paths),
        )

        try:
            result = generate(
                self._vlm_model,
                self._vlm_processor,
                prompt,
                image=frame_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                verbose=False,
            )
        except Exception:
            mx.clear_cache()
            raise

        mx.clear_cache()
        self.usage.add(
            model_id=self._vlm_model_id or "vlm",
            prompt_tokens=getattr(result, "prompt_tokens", 0),
            completion_tokens=getattr(result, "generation_tokens", 0),
        )
        return _finalize_vlm_output(_result_text(result))

    def unload_vlm(self) -> None:
        import mlx.core as mx

        if self._vlm_model is None:
            return
        del self._vlm_model, self._vlm_processor
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_config = None
        self._vlm_model_id = None
        mx.clear_cache()
        print("  [MLX] VLM unloaded.")

    def load_llm(self, model_id: str) -> dict:
        if model_id in self._VLM_BACKED_TEXT_MODELS:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            print(f"  [MLX] Loading unified VLM-as-LLM: {model_id}")
            model, processor = self._mlx_load_with_fallback(load, model_id)
            config = self._mlx_load_with_fallback(load_config, model_id)
            print(f"  [MLX] Loaded: {model_id}")
            return {
                "model": model,
                "processor": processor,
                "config": config,
                "model_id": model_id,
                "use_no_think": model_id in self._NO_THINK_MODELS,
                "vlm_backed": True,
            }

        from mlx_lm import load

        print(f"  [MLX] Loading LLM: {model_id}")
        model, tokenizer = self._mlx_load_with_fallback(load, model_id)
        handle = {
            "model": model,
            "tokenizer": tokenizer,
            "model_id": model_id,
            "use_no_think": model_id in self._NO_THINK_MODELS,
            "vlm_backed": False,
        }
        print(f"  [MLX] LLM loaded: {model_id}")
        return handle

    def generate_llm(
        self,
        handle: dict,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,
    ) -> str:
        if handle.get("vlm_backed"):
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            user_content = (
                user_prompt + " /no_think" if handle["use_no_think"] else user_prompt
            )
            prompt_text = f"{system_prompt}\n\n{user_content}"
            prompt = apply_chat_template(
                handle["processor"],
                handle["config"],
                prompt_text,
                num_images=0,
            )
            result = generate(
                handle["model"],
                handle["processor"],
                prompt,
                image=[],
                max_tokens=max_tokens,
                temperature=0.0,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                verbose=False,
            )
            self.usage.add(
                model_id=handle.get("model_id", "llm"),
                prompt_tokens=getattr(result, "prompt_tokens", 0),
                completion_tokens=getattr(result, "generation_tokens", 0),
            )
            return _finalize_llm_output(_result_text(result))

        # existing mlx_lm path, unchanged
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors

        tokenizer = handle["tokenizer"]
        user_content = (
            user_prompt + " /no_think" if handle["use_no_think"] else user_prompt
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        text_parts: list[str] = []
        usage = None
        for response in stream_generate(
            handle["model"],
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
        ):
            text_parts.append(response.text)
            usage = response
        raw = "".join(text_parts)
        self.usage.add(
            model_id=handle.get("model_id", "llm"),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "generation_tokens", 0),
        )
        return _finalize_llm_output(raw)

    def unload_llm(self, handle: dict) -> None:
        import mlx.core as mx

        if handle is None:
            return
        if handle.get("vlm_backed"):
            del handle["model"], handle["processor"]
        else:
            del handle["model"], handle["tokenizer"]
        mx.clear_cache()
        print(f"  [MLX] LLM unloaded: {handle.get('model_id', '?')}")

    def clear_cache(self) -> None:
        import mlx.core as mx

        mx.clear_cache()


def _attn_kwargs() -> dict:
    """
    Opt into flash-attention 2 if it's actually installed; otherwise return
    {} and let transformers pick its own default (sdpa/eager) exactly as it
    does today.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return {"attn_implementation": "flash_attention_2"}
    return {}


def _from_pretrained_safe(cls, model_id: str, **kwargs):
    """
    Load a pretrained HF object (config/tokenizer/processor/model),
    preferring transformers' own maintained implementation over a model
    repo's custom remote code.

    Falls back to local files only if remote access fails (HF outage).
    """
    strategies = [
        {"trust_remote_code": False, "local_files_only": False},
        {"trust_remote_code": True, "local_files_only": False},
        {"trust_remote_code": True, "local_files_only": True},
    ]

    last_exc: Exception | None = None
    for i, strat in enumerate(strategies):
        try:
            return cls.from_pretrained(model_id, **strat, **kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "gated repo" in msg or "401" in msg or "restricted" in msg:
                raise RuntimeError(
                    f"'{model_id}' is a gated or private Hugging Face repo — "
                    f"this environment isn't authenticated for it (or hasn't "
                    f"accepted the license yet). trust_remote_code can't fix "
                    f"authentication, so not retrying. Either:\n"
                    f"  1. Accept the license at "
                    f"https://huggingface.co/{model_id} while logged in, then "
                    f"authenticate this environment (huggingface_hub.login(token=...) "
                    f"or `huggingface-cli login`) with a token that has access, or\n"
                    f"  2. Point _MLX_TO_HF at an ungated mirror of this model."
                ) from exc
            if i < len(strategies) - 1:
                desc = ", ".join(f"{k}={v}" for k, v in strat.items() if not v)
                if not desc:
                    desc = "remote + trust_remote_code=False"
                print(
                    f"  [HF] Attempt {i + 1} ({desc}) failed ({exc!r}); "
                    f"trying next strategy."
                )

    msg = str(last_exc).lower() if last_exc else "unknown error"
    if "not found" in msg and "local_files_only" in msg:
        print(
            f"  [HF] Model {model_id} is not cached locally. "
            f"Download it first with `huggingface-cli download {model_id}` "
            f"or restore network access to HuggingFace Hub."
        )
    raise last_exc  # type: ignore[misc]


class HFBackend(InferenceBackend):
    """
    Cross-platform backend using HuggingFace transformers.

    Install extras:
      pip install transformers accelerate pillow bitsandbytes
    """

    def __init__(self) -> None:
        self.usage = TokenTracker()
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_model_id: str | None = None
        self._device: str = self._pick_device()
        print(f"  [HF] Device: {self._device}")

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _quant_kwargs(self) -> dict:
        """Return BitsAndBytes 4-bit kwargs when on CUDA, else empty."""
        if self._device != "cuda":
            return {}
        try:
            from transformers import BitsAndBytesConfig

            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                )
            }
        except ImportError:
            return {}

    def load_vlm(self, model_id: str) -> None:
        try:
            from transformers import AutoModelForImageTextToText as _AutoModelVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as _AutoModelVLM
        from transformers import AutoProcessor

        # Strip mlx-community prefix — map to canonical HF repo
        hf_id = _mlx_to_hf(model_id)
        print(f"  [HF] Loading VLM: {hf_id}  (mapped from {model_id})")

        self._vlm_processor = _from_pretrained_safe(AutoProcessor, hf_id)
        self._vlm_model = _from_pretrained_safe(
            _AutoModelVLM,
            hf_id,
            device_map="auto" if self._device == "cuda" else self._device,
            torch_dtype="auto",
            **self._quant_kwargs(),
            **_attn_kwargs(),
        )
        self._vlm_model_id = model_id
        print("  [HF] VLM loaded.")

    def generate_vlm(
        self,
        frame_paths: list[str],
        transcript: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,  # noqa: ARG002 — no HF equivalent; kept for interface parity with MLXBackend
    ) -> str:
        from PIL import Image

        if self._vlm_model is None:
            raise RuntimeError("VLM not loaded. Call load_vlm() first.")

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        prompt_text = system_prompt.format(transcript=transcript)

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt_text})

        inputs = _apply_chat_template(
            self._vlm_processor,
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._vlm_model.device)

        gen_kwargs: dict = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        output_ids = self._vlm_model.generate(**inputs, **gen_kwargs)
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = self._vlm_processor.decode(new_ids, skip_special_tokens=True)

        self.usage.add(
            model_id=self._vlm_model_id or "vlm",
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(new_ids.shape[0]),
        )

        return _finalize_vlm_output(raw)

    def unload_vlm(self) -> None:
        import gc

        if self._vlm_model is None:
            return
        del self._vlm_model, self._vlm_processor
        self._vlm_model = None
        self._vlm_processor = None
        gc.collect()
        self._try_cuda_empty_cache()
        print("  [HF] VLM unloaded.")

    def load_llm(self, model_id: str) -> dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = _mlx_to_hf(model_id)
        print(f"  [HF] Loading LLM: {hf_id}  (mapped from {model_id})")

        tokenizer = _from_pretrained_safe(AutoTokenizer, hf_id)
        model = _from_pretrained_safe(
            AutoModelForCausalLM,
            hf_id,
            device_map="auto" if self._device == "cuda" else self._device,
            torch_dtype="auto",
            **self._quant_kwargs(),
            **_attn_kwargs(),
        )
        return {"model": model, "tokenizer": tokenizer, "model_id": model_id}

    def generate_llm(
        self,
        handle: dict,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,  # noqa: ARG002 — no HF equivalent; kept for interface parity with MLXBackend
    ) -> str:
        import torch

        tokenizer = handle["tokenizer"]
        model = handle["model"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = _apply_chat_template(tokenizer, messages, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        self.usage.add(
            model_id=handle.get("model_id", "llm"),
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(new_ids.shape[0]),
        )
        return _finalize_llm_output(raw)

    def unload_llm(self, handle: dict) -> None:
        import gc

        if handle is None:
            return
        del handle["model"], handle["tokenizer"]
        gc.collect()
        self._try_cuda_empty_cache()
        print(f"  [HF] LLM unloaded: {handle.get('model_id', '?')}")

    def _try_cuda_empty_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def clear_cache(self) -> None:
        self._try_cuda_empty_cache()


# Maps MLX model IDs → canonical HuggingFace repo IDs. Derived from the
# config.py registry so model names live in exactly one place.
_MLX_TO_HF: dict[str, str] = {m.mlx_id: m.hf_id for m in ALL_MODELS}


def _mlx_to_hf(model_id: str) -> str:
    """
    Convert an mlx-community model ID to its canonical HuggingFace equivalent.
    """
    return _MLX_TO_HF.get(model_id, model_id)


def check_hf_reachable(timeout: float = 5.0) -> bool:
    """
    Quick connectivity check against HuggingFace Hub.
    Returns False if HF appears unreachable (network issue, not auth).
    """
    try:
        import urllib.request

        urllib.request.urlopen("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


def print_hf_unreachable_warning() -> None:
    """Print a clear warning banner when HuggingFace Hub is unreachable."""
    print(
        "\n" + "=" * 62 + "\n"
        "  WARNING: HuggingFace Hub is unreachable.\n"
        "  Pipeline will attempt to use locally cached models only.\n"
        + "=" * 62 + "\n"
    )


class FoundryBackend(InferenceBackend):
    """
    Azure Foundry / Azure OpenAI chat-completions backend (text only).

    Replaces the local judge/dedup/priority/config-gen models with a hosted
    model (e.g. gpt-4o) reached over HTTPS. Reads from the environment:

      FOUNDRY_ENDPOINT      base URL, e.g. https://<resource>.openai.azure.com/openai/v1/
      FOUNDRY_API_KEY       API key for the endpoint
      FOUNDRY_DEPLOYMENT    deployment name sent as the "model" field
      FOUNDRY_VERIFY_SSL=0  optional: disable TLS verification (corporate TLS
                            interception / GlobalProtect)
      FOUNDRY_NO_PROXY=1    optional: ignore HTTP(S)_PROXY / NO_PROXY

    load_llm is a lightweight handle factory (nothing is loaded locally); the
    deployment name is resolved from the environment, so the model_id passed by
    the pipeline (an MLX id) is ignored. generate_llm POSTs an OpenAI-style
    chat/completions request and returns the assistant text.
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

    # VLM methods are unused — the proposer is AVI, not a local model.
    def load_vlm(self, model_id: str) -> None:
        raise NotImplementedError(
            "FoundryBackend is text-only; the AVI proposer replaces VLM inference."
        )

    def generate_vlm(
        self,
        frame_paths: list[str],
        transcript: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int = 20,
    ) -> str:
        raise NotImplementedError(
            "FoundryBackend is text-only; the AVI proposer replaces VLM inference."
        )

    def unload_vlm(self) -> None:
        pass

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
    ) -> str:
        payload = {
            "model": handle.get("model_id") or self._deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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


_BACKEND_MAP = {
    "mlx": MLXBackend,
    "hf": HFBackend,
    "foundry": FoundryBackend,
}


def _detect_backend() -> str:
    """
    Auto-detect the right backend for this machine:
      - Apple Silicon (Darwin + arm64) → mlx
      - Everything else (Intel Mac, Linux, Windows, CUDA boxes, Colab) → hf

    Checked structurally (OS + CPU arch) rather than by trying to import
    mlx, so the error message stays clear if mlx is supposed to be there
    but isn't installed, instead of silently falling back to hf.
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return "mlx" if is_apple_silicon else "hf"


def get_backend(name: str | None = None) -> InferenceBackend:
    """
    Return an InferenceBackend instance.

    Resolution order:
      1. Explicit `name` argument, if given.
      2. BACKEND env var, if set.
      3. Auto-detected from platform (Apple Silicon → mlx, else → hf).
    """
    selected = (name or os.getenv("BACKEND") or _detect_backend()).lower()
    if selected not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown backend {selected!r}. Choose from: {list(_BACKEND_MAP)}"
        )
    print(f"[backend] Using: {selected} (auto-detected)" if not (name or os.getenv("BACKEND")) else f"[backend] Using: {selected}")
    return _BACKEND_MAP[selected]()
