from config import FRAME_DIR, TRANSCRIPT_PATH, VIDEO_PATH

from .keyframes import extract_keyframes
from .transcript import load_transcript


def prepare_data(
    transcript_path: str = TRANSCRIPT_PATH,
    video_path: str = VIDEO_PATH,
    frame_dir: str = FRAME_DIR,
) -> dict:
    """Main function to prepare all data: load transcript and extract keyframes."""
    chunks = load_transcript(transcript_path)
    chunks = extract_keyframes(video_path, chunks, frame_dir)
    return chunks


def get_valid_chunks(chunks: dict) -> list:
    """Filter out empty chunks and return valid ones sorted by time."""
    return [
        (bucket_index, chunks[bucket_index])
        for bucket_index in sorted(chunks.keys())
        if chunks[bucket_index]["text_lines"] or chunks[bucket_index]["frame_paths"]
    ]
