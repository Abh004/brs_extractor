"""
backend.py

Inference backend abstraction for the pipeline.

Provides a common interface so pipeline stages (VLM proposer, judges,
config generator) never import mlx_vlm or mlx_lm directly — they call
through a backend instance instead.

Select via environment variable:
  BACKEND=mlx   (default) — Apple Silicon, MLX. Fast, native.
  BACKEND=hf    — Any CUDA/CPU box. Uses HuggingFace transformers + PIL.

Usage:
  from backend import get_backend
  backend = get_backend()               # reads BACKEND env var
  backend = get_backend("hf")           # explicit override

VLM:
  text = backend.generate_vlm(frame_paths, transcript, system_prompt)

LLM:
  text = backend.generate_llm(handle, system_prompt, user_prompt, max_tokens)

Lifecycle:
  backend.load_vlm(model_id)
  backend.unload_vlm()
  # judges load individually:
  handle = backend.load_llm(model_id)
  backend.unload_llm(handle)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from utils import clean_output, strip_think_blocks  # noqa: F401 — clean_output re-exported for callers

# ── abstract base ─────────────────────────────────────────────────────────────


class InferenceBackend(ABC):
    """
    Common interface for all inference backends.

    Implementations must be safe to call from the main thread only — MLX GPU
    streams are thread-local and cannot be used from ThreadPoolExecutor workers.
    """

    # ---------- VLM (multimodal) ----------

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
    ) -> str:
        """Run multimodal inference. Returns raw text (not cleaned)."""

    @abstractmethod
    def unload_vlm(self) -> None:
        """Free VLM weights. Safe to call even if nothing is loaded."""

    # ---------- LLM (text-only) ----------

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
    ) -> str:
        """Run text-only inference using a previously loaded handle."""

    @abstractmethod
    def unload_llm(self, handle: Any) -> None:
        """Free an LLM handle returned by load_llm."""

    # ---------- utility ----------

    def clear_cache(self) -> None:
        """Optional: flush any accelerator memory caches (MLX Metal cache etc.)."""


# ── MLX backend ───────────────────────────────────────────────────────────────


class MLXBackend(InferenceBackend):
    """
    Apple Silicon backend using mlx-vlm (VLM) and mlx_lm (LLM).

    All operations run on the Metal GPU via MLX. Not importable on Linux
    because mlx / mlx_vlm are macOS-only packages.

    Thread safety: MLX GPU streams are thread-local. This backend must only
    be called from the main thread — do not wrap calls in ThreadPoolExecutor.
    """

    # Model IDs that support the /no_think control token (Qwen3 family).
    _NO_THINK_MODELS = {
        "mlx-community/Qwen3-VL-4B-Instruct-8bit",
        "mlx-community/Qwen3-1.7B-4bit",
    }

    def __init__(self) -> None:
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_config = None
        self._vlm_model_id: str | None = None

    # ---- VLM ----

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

    # ---- LLM ----

    def load_llm(self, model_id: str) -> dict:
        from mlx_lm import load

        print(f"  [MLX] Loading LLM: {model_id}")
        model, tokenizer = load(model_id)
        handle = {
            "model": model,
            "tokenizer": tokenizer,
            "model_id": model_id,
            "use_no_think": model_id in self._NO_THINK_MODELS,
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
        del handle["model"], handle["tokenizer"]
        mx.clear_cache()
        print(f"  [MLX] LLM unloaded: {handle.get('model_id', '?')}")

    def clear_cache(self) -> None:
        import mlx.core as mx

        mx.clear_cache()


# ── HuggingFace backend ───────────────────────────────────────────────────────


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

    Supports CUDA (Linux/Windows GPU), MPS (Mac, alternative to MLX),
    and CPU. Automatically selects the best available device.

    VLM: uses AutoModelForVision2Seq + AutoProcessor (works with Qwen2-VL,
    LLaVA, InternVL, etc.). Images loaded via PIL.

    LLM: uses AutoModelForCausalLM + AutoTokenizer with 4-bit BitsAndBytes
    quantisation when on CUDA (matches MLX 4bit models). Falls back to
    float16/bfloat16 on MPS/CPU.

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

    # ---- VLM ----

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

    # ---- LLM ----

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

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
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


# ── model ID mapping ──────────────────────────────────────────────────────────

# Maps mlx-community quantised model IDs → canonical HuggingFace repo IDs.
# Add entries here as you use new models.
_MLX_TO_HF: dict[str, str] = {
    "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit": "Qwen/Qwen3-VL-8B-Instruct",
    "mlx-community/Phi-3-mini-4k-instruct-4bit": "microsoft/Phi-3-mini-4k-instruct",
    "mlx-community/Llama-3.2-1B-Instruct-4bit": "unsloth/Llama-3.2-1B-Instruct",
    "mlx-community/Qwen3-1.7B-4bit": "Qwen/Qwen3-1.7B",
}


def _mlx_to_hf(model_id: str) -> str:
    """
    Convert an mlx-community model ID to its canonical HuggingFace equivalent.
    Falls back to the original ID unchanged (useful if already a HF ID).
    """
    return _MLX_TO_HF.get(model_id, model_id)


# ── factory ───────────────────────────────────────────────────────────────────

_BACKEND_MAP = {
    "mlx": MLXBackend,
    "hf": HFBackend,
}


def get_backend(name: str | None = None) -> InferenceBackend:
    """
    Return an InferenceBackend instance.

    Args:
        name: "mlx" or "hf". If None, reads the BACKEND env var
              (defaults to "mlx").

    Raises:
        ValueError: if name is not a known backend key.
    """
    selected = (name or os.getenv("BACKEND", "mlx")).lower()
    if selected not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown backend {selected!r}. Choose from: {list(_BACKEND_MAP)}"
        )
    print(f"[backend] Using: {selected}")
    return _BACKEND_MAP[selected]()
