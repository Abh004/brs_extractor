import json
import math

from config import CHUNK_SIZE_SEC


def _new_chunk() -> dict:
    """Return an empty chunk dict — shared shape used across data_prep."""
    return {"text_lines": [], "frame_paths": []}


def load_transcript(transcript_path: str) -> dict:
    """
    Load transcript from JSON file and organize into chunks.

    Args:
        transcript_path: Path to the transcript JSON file

    Returns:
        Dictionary mapping bucket_index to chunk data
    """
    print("Loading transcript...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    chunks = {}
    for item in transcript_data:
        start_time = item["start"]
        text = item["text"]

        bucket_index = math.floor(start_time / CHUNK_SIZE_SEC)
        if bucket_index not in chunks:
            chunks[bucket_index] = _new_chunk()

        chunks[bucket_index]["text_lines"].append(text)

    return chunks
