from backend import InferenceBackend, get_backend
from config import MAX_NEW_TOKENS, MODEL_NAME, PROPOSER_PROMPT, TEMPERATURE, REPETITION_PENALTY, REPETITION_CONTEXT_SIZE


class VLMProposer:
    """
    VLM Proposer: extracts initial requirements from video frames + transcript.

    The backend is injected so main.py can share one backend instance across
    all pipeline stages (VLM proposer, judges, config generator).

    Args:
        model_name: HF / mlx-community model ID to load.
        backend:    InferenceBackend instance. If None, one is created via
                    get_backend() (reads the BACKEND env var).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        backend: InferenceBackend | None = None,
    ):
        self.model_name = model_name
        self._backend = backend or get_backend()
        print(f"Loading VLM proposer: {model_name}")
        self._backend.load_vlm(model_name)
        print("VLM proposer ready.")

    def generate(self, frame_paths: list[str], transcript: str) -> str:
        """
        Generate requirements from video frames and transcript.

        Args:
            frame_paths: Paths to pre-resized 448×448 keyframe images.
            transcript:  Combined transcript text for this chunk.

        Returns:
            Raw model output string (JSON array of requirement strings).
        """
        return self._backend.generate_vlm(
            frame_paths=frame_paths,
            transcript=transcript,
            system_prompt=PROPOSER_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            repetition_context_size=REPETITION_CONTEXT_SIZE,
        )

    def unload(self) -> None:
        """Free VLM weights. Call before loading judge models."""
        self._backend.unload_vlm()
        print("Proposer unloaded.")
