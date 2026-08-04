"""
extractor.py

Fetches the YouTube transcript for VIDEO_ID and writes it to TRANSCRIPT_PATH.
"""

from config import TRANSCRIPT_PATH, VIDEO_ID


def fetch_transcript(video_id: str | None = None, force: bool = False) -> None:
    """
    Download the YouTube transcript for a video and write it to TRANSCRIPT_PATH.

    Args:
        video_id: YouTube video ID to fetch. Defaults to config.VIDEO_ID.
        force: Re-download even if TRANSCRIPT_PATH already exists.
               Default False — skips the network call if the file is present,
               allowing reruns and checkpoint resumption without re-fetching.
    """
    import os

    video_id = video_id or VIDEO_ID

    if not force and os.path.exists(TRANSCRIPT_PATH):
        print(f"  Transcript already exists: {TRANSCRIPT_PATH} (skip fetch)")
        return

    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import JSONFormatter

    print(f"  Fetching transcript for video: {video_id}")
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    json_formatted = JSONFormatter().format_transcript(transcript)

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH) or ".", exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(json_formatted)

    print(f"  Transcript written to {TRANSCRIPT_PATH}")


if __name__ == "__main__":
    # Still runnable standalone for manual use / debugging.
    fetch_transcript(force=True)
