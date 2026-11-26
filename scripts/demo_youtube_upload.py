"""Upload the generated demo Short to YouTube (private) and read back stats & comments."""

from __future__ import annotations

import json
from pathlib import Path

from sg_mcp.config import Settings
from sg_mcp.youtube_service import YouTubeService


def main() -> None:
    cfg = Settings.from_env()
    yt = YouTubeService(cfg)

    demo_path = Path("demo_media/demo_short.mp4")
    if not demo_path.is_file():
        raise FileNotFoundError(
            f"{demo_path} not found. Generate a demo video first or pass your own MP4."
        )

    print("Uploading demo Short to YouTube (privacy=private)...")
    uploaded = yt.upload_video(
        video_source=str(demo_path),
        title="sg-mcp Gemini 3.8 Flash TTS Demo Short",
        description="Automated demo YouTube Short created with Gemini 3.8 Flash TTS and sg-mcp.",
        video_type="short",
        privacy_status="private",
        tags=["GeminiTTS", "MCP", "AI", "Shorts"],
    )
    print("Upload result:")
    print(json.dumps(uploaded, indent=2))

    vid = uploaded["video_id"]
    print(f"\nReading video stats for {vid}...")
    print(json.dumps(yt.get_stats(video_id=vid), indent=2))

    print("\nReading channel stats...")
    print(json.dumps(yt.get_stats(max_videos=5), indent=2))

    print(f"\nReading comments for {vid}...")
    print(json.dumps(yt.get_comments(video_id=vid), indent=2))


if __name__ == "__main__":
    main()
