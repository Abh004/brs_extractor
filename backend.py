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
from typing import Any

from utils import clean_output, strip_think_blocks


# ── abstract base ─────────────────────────────────────────────────────────────
class InferenceBackend(ABC):
    """
    Common interface for all inference backends.

    Implementations must be safe to call from the main thread only — MLX GPU
    streams are thread-local and cannot be used from ThreadPoolExecutor workers.
    """

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

    def clear_cache(self) -> None:
        """Optional: flush any accelerator memory caches (MLX Metal cache etc.)."""


class MLXBackend(InferenceBackend):
    """
    Apple Silicon backend using mlx-vlm (VLM) and mlx_lm (LLM).

    Thread safety: MLX GPU streams are thread-local. This backend must only
    be called from the main thread — do not wrap calls in ThreadPoolExecutor.
    """

    _NO_THINK_MODELS = {
        "mlx-community/Qwen3-VL-8B-Instruct-MLX-4bit",
        "mlx-community/Qwen3-1.7B-4bit",
    }

    def __init__(self) -> None:
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_config = None
        self._vlm_model_id: str | None = None

    def load_vlm(self, model_id: str) -> None:
        import warnings

        warnings.filterwarnings(
            "ignore", message="Kwargs passed to `processor.__call__`"
        )
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"  [MLX] Loading VLM: {model_id}")
        self._vlm_model, self._vlm_processor = load(model_id)
        self._vlm_config = load_config(model_id)
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
        raw = str(result.text) if hasattr(result, "text") else str(result)
        if "<think>" in raw and "</think>" not in raw:
            return ""
        return strip_think_blocks(raw)

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
            model, processor = load(model_id)
            config = load_config(model_id)
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
        model, tokenizer = load(model_id)
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
            raw = str(result.text) if hasattr(result, "text") else str(result)
            if "<think>" in raw and "</think>" not in raw:
                return ""
            return clean_output(raw)

        # existing mlx_lm path, unchanged
        from mlx_lm import generate
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
        raw = generate(
            handle["model"],
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            verbose=False,
        )
        if "<think>" in raw and "</think>" not in raw:
            return ""
        return clean_output(raw)

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

    # Model IDs that are natively unified VLM architectures (Qwen3.5, etc.)
    # but used here purely for text — routed through mlx_vlm with
    # num_images=0 rather than mlx_lm, since these models don't load via
    # mlx_lm's API.
    _VLM_BACKED_TEXT_MODELS = {
        "lmstudio-community/Qwen3.5-9B-MLX-4bit",
    }


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
    """
    try:
        return cls.from_pretrained(model_id, trust_remote_code=False, **kwargs)
    except Exception as exc:
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

        print(
            f"  [HF] Native load of {model_id} failed ({exc!r}); "
            f"retrying with trust_remote_code=True."
        )
        return cls.from_pretrained(model_id, trust_remote_code=True, **kwargs)


class HFBackend(InferenceBackend):
    """
    Cross-platform backend using HuggingFace transformers.

    Install extras:
      pip install transformers accelerate pillow bitsandbytes
    """

    def __init__(self) -> None:
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

        try:
            inputs = self._vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self._vlm_model.device)
        except TypeError:
            inputs = self._vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
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

        # Defensive cleanup in case enable_thinking=False was ignored (some converted/community chat templates drop the conditional entirely)
        if "<think>" in raw and "</think>" not in raw:
            return ""
        return strip_think_blocks(raw)

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
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
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
        if "<think>" in raw and "</think>" not in raw:
            return ""
        return clean_output(raw)

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


# Maps mlx-community quantised model IDs → canonical HuggingFace repo IDs.
_MLX_TO_HF: dict[str, str] = {
    "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit": "Qwen/Qwen3-VL-8B-Instruct",
    "lmstudio-community/Qwen3.5-9B-MLX-4bit": "Qwen/Qwen3.5-9B",
    "mlx-community/Qwen3-1.7B-4bit": "Qwen/Qwen3-1.7B",
}


def _mlx_to_hf(model_id: str) -> str:
    """
    Convert an mlx-community model ID to its canonical HuggingFace equivalent.
    """
    return _MLX_TO_HF.get(model_id, model_id)


_BACKEND_MAP = {
    "mlx": MLXBackend,
    "hf": HFBackend,
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
