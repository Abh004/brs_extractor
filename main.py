"""
CLI entry point for the BRS extraction pipeline.

Delegates to PipelineRunner so the same logic powers both CLI and
the FastAPI web UI. Run directly:
    python main.py

Or via the web UI:
    uvicorn server.app:app
"""

from server.pipeline_runner import PipelineRunner


def main():
    runner = PipelineRunner()
    runner._run_sync()


if __name__ == "__main__":
    main()
