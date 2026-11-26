"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_api_keys() -> tuple[str, ...]:
    """Collect deduplicated Gemini API keys in priority order."""
    keys: list[str] = []

    multi = os.getenv("GEMINI_API_KEYS", "")
    for raw in multi.split(","):
        cleaned = raw.strip()
        if cleaned and cleaned not in keys:
            keys.append(cleaned)

    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        val = os.getenv(env_name, "").strip()
        if val and val not in keys:
            keys.append(val)

    return tuple(keys)


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the MCP server."""

    api_keys: tuple[str, ...]
    tts_model: str
    default_voice_id: str
    youtube_client_id: str
    youtube_client_secret: str
    youtube_refresh_token: str
    youtube_token_file: Path
    youtube_channel_email: str
    youtube_redirect_uri: str
    host: str
    port: int
    public_base_url: str
    audio_storage_dir: Path
    audio_ttl_seconds: int
    max_stored_files: int
    mcp_auth_token: str
    include_inline_audio: bool
    voice_cache_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_keys=_parse_api_keys(),
            tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts").strip()
            or "gemini-3.8-flash-tts",
            default_voice_id=os.getenv("DEFAULT_VOICE_ID", "charon").strip().lower() or "charon",
            youtube_client_id=os.getenv("YOUTUBE_CLIENT_ID", "").strip(),
            youtube_client_secret=os.getenv("YOUTUBE_CLIENT_SECRET", "").strip(),
            youtube_refresh_token=os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip(),
            youtube_token_file=Path(
                os.getenv("YOUTUBE_TOKEN_FILE", "./youtube_token.json").strip()
                or "./youtube_token.json"
            ),
            youtube_channel_email=os.getenv("YOUTUBE_CHANNEL_EMAIL", "").strip(),
            youtube_redirect_uri=os.getenv("YOUTUBE_REDIRECT_URI", "").strip(),
            host=os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(os.getenv("PORT", "8080")),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
            audio_storage_dir=Path(
                os.getenv("AUDIO_STORAGE_DIR", "./generated_audio").strip() or "./generated_audio"
            ),
            audio_ttl_seconds=max(60, int(os.getenv("AUDIO_TTL_SECONDS", "3600"))),
            max_stored_files=max(10, int(os.getenv("MAX_STORED_FILES", "200"))),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN", "").strip(),
            include_inline_audio=_parse_bool(os.getenv("INCLUDE_INLINE_AUDIO"), True),
            voice_cache_ttl_seconds=max(60, int(os.getenv("VOICE_CACHE_TTL_SECONDS", "3600"))),
        )
