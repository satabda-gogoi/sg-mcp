"""FastMCP Streamable HTTP server for Gemini 3.8 Flash TTS and YouTube Data API v3 tools."""

from __future__ import annotations

import hmac
import html
import json
import logging
from typing import Annotated, Any, Literal

import anyio
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Audio
from mcp.types import TextContent
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from sg_mcp import __version__
from sg_mcp.config import Settings
from sg_mcp.storage import AudioStorage
from sg_mcp.tts_service import GeminiTTSService
from sg_mcp.youtube_service import YouTubeService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sg_mcp")


class OptionalBearerAuthMiddleware:
    """Enforces Bearer token auth on /mcp when MCP_AUTH_TOKEN is configured."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.token:
            path = scope.get("path", "")
            if path == "/mcp" or path.startswith("/mcp/"):
                headers = dict(scope.get("headers") or [])
                auth_raw = headers.get(b"authorization", b"").decode("latin-1")
                expected = f"Bearer {self.token}"
                if not hmac.compare_digest(auth_raw, expected):
                    resp = JSONResponse(
                        {"error": "Unauthorized: invalid or missing Bearer token."},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                    await resp(scope, receive, send)
                    return

        await self.app(scope, receive, send)


def _base_url_from_request(request: Request, settings: Settings) -> str:
    if settings.public_base_url:
        return settings.public_base_url

    proto = (
        request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        or request.url.scheme
        or "https"
    )
    host = (
        request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        or request.headers.get("host", "").strip()
        or request.url.netloc
    )
    if host:
        return f"{proto}://{host}".rstrip("/")
    return f"http://127.0.0.1:{settings.port}"


def _resolve_base_url(settings: Settings) -> str:
    if settings.public_base_url:
        return settings.public_base_url

    try:
        return _base_url_from_request(get_http_request(), settings)
    except RuntimeError:
        pass

    return f"http://127.0.0.1:{settings.port}"


def _youtube_redirect_uri(request: Request, settings: Settings) -> str:
    if settings.youtube_redirect_uri:
        return settings.youtube_redirect_uri
    return f"{_base_url_from_request(request, settings)}/youtube/callback"


def build_mcp_server(
    settings: Settings | None = None,
    tts_service: GeminiTTSService | None = None,
    storage: AudioStorage | None = None,
    youtube_service: YouTubeService | None = None,
) -> tuple[FastMCP, Settings, GeminiTTSService, AudioStorage, YouTubeService]:
    """Build and configure the FastMCP server, tools, and HTTP routes."""
    cfg = settings or Settings.from_env()
    tts = tts_service or GeminiTTSService(cfg)
    store = storage or AudioStorage(
        base_dir=cfg.audio_storage_dir,
        ttl_seconds=cfg.audio_ttl_seconds,
        max_files=cfg.max_stored_files,
    )
    yt = youtube_service or YouTubeService(cfg)

    mcp = FastMCP(
        name="sg-mcp",
        version=__version__,
        instructions=(
            "Cloud MCP server providing expressive text-to-speech via gemini-3.8-flash-tts "
            "and YouTube channel management via YouTube Data API v3.\n"
            "- TTS: Call `list_voices` to browse male/female/neutral voices, or `generate_voice` "
            "(defaults to male voice 'charon' if `voice` is omitted) and share the returned "
            "downloadable WAV URL.\n"
            "- YouTube: Call `upload_youtube_video` (supports Shorts & long videos from a local "
            "path or HTTP/HTTPS URL; defaults to 'private' privacy_status unless requested "
            "otherwise), `get_youtube_stats` (channel + recent video views/likes/comments or "
            "specific video stats), and `get_youtube_comments` (top-level comments + replies)."
        ),
    )

    @mcp.tool(
        name="list_voices",
        description=(
            "List available Gemini TTS voices across male, female, and neutral genders "
            "(2,150+ voices including 30 flagship voices and multilingual personas). "
            "Supports filtering by gender, language_code, pitch, or keyword search."
        ),
    )
    async def list_voices(
        gender: Annotated[
            Literal["all", "male", "female", "neutral"] | None,
            Field(description="Filter by voice gender: 'all', 'male', 'female', or 'neutral'."),
        ] = "all",
        language_code: Annotated[
            str | None,
            Field(description="Optional BCP-47 language filter, e.g. 'en-US', 'en-IN', 'es-ES'."),
        ] = None,
        pitch: Annotated[
            Literal["low", "medium", "high"] | None,
            Field(description="Optional vocal pitch filter: 'low', 'medium', or 'high'."),
        ] = None,
        search: Annotated[
            str | None,
            Field(description="Optional keyword filter matching voice id, name, accent, or style."),
        ] = None,
        featured_only: Annotated[
            bool,
            Field(description="If true, return only the 30 flagship Gemini prebuilt voices."),
        ] = False,
        limit: Annotated[
            int,
            Field(description="Maximum voices to return per page (1-200).", ge=1, le=200),
        ] = 30,
        offset: Annotated[
            int,
            Field(description="Pagination offset.", ge=0),
        ] = 0,
    ) -> str:
        """Browse and search available Gemini TTS voices."""
        result = await anyio.to_thread.run_sync(
            lambda: tts.list_voices(
                gender=gender,
                language_code=language_code,
                pitch=pitch,
                search=search,
                featured_only=featured_only,
                limit=limit,
                offset=offset,
            )
        )
        result["default_voice_id"] = cfg.default_voice_id
        result["tts_model"] = cfg.tts_model
        return json.dumps(result, indent=2)

    @mcp.tool(
        name="generate_voice",
        description=(
            "Generate expressive speech audio (.wav) from a voice script using Gemini 3.8 "
            "Flash TTS. If `voice` is omitted or empty, uses the default male voice "
            "('charon'). Returns a direct downloadable WAV file URL, audio metadata, and "
            "inline MCP audio content."
        ),
    )
    async def generate_voice(
        script: Annotated[
            str,
            Field(description="The text script to speak aloud."),
        ],
        voice: Annotated[
            str | None,
            Field(
                description=(
                    "Optional voice ID or name (e.g. 'charon', 'puck', 'fenrir', 'kore', "
                    "'zephyr', 'en-au-assistant-6'). Defaults to male voice 'charon' if omitted."
                )
            ),
        ] = None,
        style: Annotated[
            str | None,
            Field(
                description=(
                    "Optional delivery/emotion instruction, e.g. 'warm and confident', "
                    "'cheerful and upbeat', 'calm documentary narrator'."
                )
            ),
        ] = None,
        language_code: Annotated[
            str | None,
            Field(description="Optional BCP-47 language code hint, e.g. 'en-US', 'hi-IN', 'ja-JP'."),
        ] = None,
        temperature: Annotated[
            float | None,
            Field(description="Optional sampling expressiveness (0.0 to 2.0).", ge=0.0, le=2.0),
        ] = None,
        seed: Annotated[
            int | None,
            Field(description="Optional deterministic random seed."),
        ] = None,
        filename_prefix: Annotated[
            str | None,
            Field(description="Optional custom filename prefix for the downloadable .wav file."),
        ] = None,
    ) -> ToolResult:
        """Generate a downloadable WAV voice file from script and optional voice settings."""
        generated = await anyio.to_thread.run_sync(
            lambda: tts.synthesize(
                script=script,
                voice=voice,
                style=style,
                language_code=language_code,
                temperature=temperature,
                seed=seed,
            )
        )

        stored = await anyio.to_thread.run_sync(
            lambda: store.save_wav(
                generated.wav_bytes,
                voice_id=generated.voice.id,
                filename_prefix=filename_prefix,
            )
        )

        base_url = _resolve_base_url(cfg)
        download_url = f"{base_url}/download/{stored.file_id}/{stored.filename}"
        preview_url = f"{download_url}?inline=1"

        metadata: dict[str, Any] = {
            "status": "success",
            "model": generated.model,
            "voice": {
                "id": generated.voice.id,
                "display_name": generated.voice.display_name,
                "gender": generated.voice.gender,
                "language_code": generated.voice.language_code or language_code,
                "accent": generated.voice.accent,
            },
            "used_default_voice": not bool((voice or "").strip()),
            "style": style,
            "file": {
                "file_id": stored.file_id,
                "filename": stored.filename,
                "mime_type": generated.mime_type,
                "size_bytes": stored.size_bytes,
                "duration_seconds": generated.duration_seconds,
                "sample_rate_hz": generated.sample_rate_hz,
                "download_url": download_url,
                "inline_play_url": preview_url,
                "expires_in_seconds": cfg.audio_ttl_seconds,
            },
        }

        duration_note = (
            f"{generated.duration_seconds}s" if generated.duration_seconds is not None else "unknown"
        )
        summary_text = (
            f"Generated speech audio using {generated.model}.\n"
            f"- Voice: {generated.voice.display_name} (`{generated.voice.id}`, "
            f"{generated.voice.gender})\n"
            f"- Duration: {duration_note} | Size: {stored.size_bytes} bytes | Format: WAV\n"
            f"- Download URL: {download_url}\n"
            f"- Browser Preview URL: {preview_url}\n"
            f"- Link expires in {cfg.audio_ttl_seconds} seconds."
        )

        content_blocks: list[Any] = [TextContent(type="text", text=summary_text)]
        if cfg.include_inline_audio:
            content_blocks.append(Audio(data=generated.wav_bytes, format="wav").to_audio_content())

        return ToolResult(
            content=content_blocks,
            structured_content=metadata,
        )

    @mcp.tool(
        name="upload_youtube_video",
        description=(
            "Upload a YouTube Short or standard long-form video to the connected YouTube "
            "channel from a local file path or public HTTP/HTTPS video URL. Supports title, "
            "description, tags, privacy status ('private', 'unlisted', 'public'), scheduled "
            "publish_at, and optional custom thumbnail."
        ),
    )
    async def upload_youtube_video(
        video_source: Annotated[
            str,
            Field(description="Local file path or direct HTTP/HTTPS URL of the video (.mp4, .mov, etc.)."),
        ],
        title: Annotated[
            str,
            Field(description="Video title (max 100 chars)."),
        ],
        description: Annotated[
            str,
            Field(description="Video description (max 5000 chars)."),
        ] = "",
        video_type: Annotated[
            Literal["short", "long"],
            Field(
                description=(
                    "'short' for vertical/square YouTube Shorts (<=3 min; adds #Shorts tag) "
                    "or 'long' for regular horizontal videos."
                )
            ),
        ] = "long",
        privacy_status: Annotated[
            Literal["private", "unlisted", "public"],
            Field(description="YouTube visibility: 'private' (default), 'unlisted', or 'public'."),
        ] = "private",
        tags: Annotated[
            list[str] | None,
            Field(description="Optional list of keyword tags."),
        ] = None,
        category_id: Annotated[
            str,
            Field(description="YouTube category ID (default '22' = People & Blogs; '28' = Science & Tech)."),
        ] = "22",
        made_for_kids: Annotated[
            bool,
            Field(description="Whether the video is made for kids (COPPA)."),
        ] = False,
        publish_at: Annotated[
            str | None,
            Field(description="Optional ISO-8601 UTC timestamp for scheduled publishing (e.g. '2026-12-01T15:00:00Z')."),
        ] = None,
        default_language: Annotated[
            str | None,
            Field(description="Optional BCP-47 language code of video title/description (e.g. 'en')."),
        ] = None,
        thumbnail_source: Annotated[
            str | None,
            Field(description="Optional local path or HTTP/HTTPS URL of a custom thumbnail image."),
        ] = None,
    ) -> ToolResult:
        """Upload a Short or long video to YouTube and return its URL and metadata."""
        result = await anyio.to_thread.run_sync(
            lambda: yt.upload_video(
                video_source=video_source,
                title=title,
                description=description,
                video_type=video_type,
                privacy_status=privacy_status,
                tags=tags,
                category_id=category_id,
                made_for_kids=made_for_kids,
                publish_at=publish_at,
                default_language=default_language,
                thumbnail_source=thumbnail_source,
            )
        )

        primary_url = result.get("shorts_url") or result["watch_url"]
        summary = (
            f"Uploaded YouTube {result['video_type']} video!\n"
            f"- Title: {result['title']}\n"
            f"- Video ID: `{result['video_id']}` | Privacy: `{result['privacy_status']}`\n"
            f"- Watch URL: {primary_url}\n"
            f"- YouTube Studio: {result['studio_url']}"
        )
        return ToolResult(
            content=[TextContent(type="text", text=summary)],
            structured_content=result,
        )

    @mcp.tool(
        name="get_youtube_stats",
        description=(
            "Read YouTube statistics: if `video_id` (or YouTube URL) is provided, returns "
            "that video's views, total likes, total comments, duration, and privacy status. "
            "If `video_id` is omitted, returns the connected channel's subscriber count, "
            "total views, total videos, and recent uploaded videos with their stats."
        ),
    )
    async def get_youtube_stats(
        video_id: Annotated[
            str | None,
            Field(description="Optional YouTube video ID or URL. Omit to inspect channel & recent videos."),
        ] = None,
        max_videos: Annotated[
            int,
            Field(description="Maximum recent videos to include when querying channel stats (1-50).", ge=1, le=50),
        ] = 10,
    ) -> ToolResult:
        """Fetch video or channel statistics from YouTube Data API v3."""
        result = await anyio.to_thread.run_sync(
            lambda: yt.get_stats(video_id=video_id, max_videos=max_videos)
        )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            structured_content=result,
        )

    @mcp.tool(
        name="get_youtube_comments",
        description=(
            "Read comments and replies on a YouTube video (or across the connected channel "
            "if `video_id` is omitted). Supports sorting by relevance/time and fetching "
            "all comment threads."
        ),
    )
    async def get_youtube_comments(
        video_id: Annotated[
            str | None,
            Field(description="Optional YouTube video ID or URL. Omit to list recent comments across the channel."),
        ] = None,
        max_results: Annotated[
            int,
            Field(description="Comment threads per page (1-100) when fetch_all is false.", ge=1, le=100),
        ] = 50,
        order: Annotated[
            Literal["relevance", "time"],
            Field(description="Sort order: 'relevance' or 'time' (newest first)."),
        ] = "relevance",
        include_replies: Annotated[
            bool,
            Field(description="Include nested comment replies."),
        ] = True,
        page_token: Annotated[
            str | None,
            Field(description="Optional pagination token from a previous call."),
        ] = None,
        fetch_all: Annotated[
            bool,
            Field(description="If true, paginate automatically to fetch up to 500 comment threads."),
        ] = False,
    ) -> ToolResult:
        """Fetch top-level comments and replies from YouTube."""
        result = await anyio.to_thread.run_sync(
            lambda: yt.get_comments(
                video_id=video_id,
                max_results=max_results,
                order=order,
                include_replies=include_replies,
                page_token=page_token,
                fetch_all=fetch_all,
            )
        )
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            structured_content=result,
        )

    @mcp.custom_route("/", methods=["GET"])
    async def root_info(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "service": "sg-mcp",
                "version": __version__,
                "status": "ok",
                "transport": "streamable-http",
                "mcp_endpoint": "/mcp",
                "tts_model": cfg.tts_model,
                "default_voice": cfg.default_voice_id,
                "api_keys_configured": len(cfg.api_keys),
                "youtube_oauth_client_configured": yt.has_oauth_client(),
                "youtube_authenticated": yt.is_authenticated(),
                "youtube_auth_url": f"{_base_url_from_request(request, cfg)}/youtube/auth",
                "tools": [
                    "generate_voice",
                    "list_voices",
                    "upload_youtube_video",
                    "get_youtube_stats",
                    "get_youtube_comments",
                ],
            }
        )

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    @mcp.custom_route("/download/{file_id}", methods=["GET"])
    @mcp.custom_route("/download/{file_id}/{filename}", methods=["GET"])
    async def download_audio(request: Request) -> Response:
        file_id = request.path_params.get("file_id", "")
        record = store.get(file_id)
        if not record:
            return JSONResponse(
                {"error": "Audio file not found or expired."},
                status_code=404,
            )

        inline = request.query_params.get("inline", "").lower() in {"1", "true", "yes"}
        disposition = "inline" if inline else "attachment"
        wav_bytes = await anyio.to_thread.run_sync(record.path.read_bytes)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'{disposition}; filename="{record.filename}"',
                "Cache-Control": "no-store",
            },
        )

    @mcp.custom_route("/youtube/auth", methods=["GET"])
    async def youtube_auth_start(request: Request) -> Response:
        if not yt.has_oauth_client():
            return JSONResponse(
                {
                    "error": "YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET are not configured."
                },
                status_code=500,
            )
        redirect_uri = _youtube_redirect_uri(request, cfg)
        auth_url, _ = yt.create_auth_url(redirect_uri)
        return RedirectResponse(url=auth_url, status_code=302)

    @mcp.custom_route("/youtube/callback", methods=["GET"])
    async def youtube_auth_callback(request: Request) -> Response:
        error = request.query_params.get("error")
        if error:
            return HTMLResponse(
                f"<h2>YouTube OAuth Failed</h2><p>{html.escape(error)}</p>",
                status_code=400,
            )

        code = request.query_params.get("code", "").strip()
        if not code:
            return HTMLResponse(
                "<h2>Missing OAuth authorization code.</h2>",
                status_code=400,
            )

        state = request.query_params.get("state")
        redirect_uri = _youtube_redirect_uri(request, cfg)
        try:
            tokens = await anyio.to_thread.run_sync(
                lambda: yt.exchange_code(code=code, redirect_uri=redirect_uri, state=state)
            )
        except Exception as exc:
            logger.exception("YouTube OAuth callback failed")
            return HTMLResponse(
                f"<h2>Token exchange failed</h2><pre>{html.escape(str(exc))}</pre>",
                status_code=500,
            )

        refresh_token = html.escape(tokens.get("refresh_token") or "(none returned)")
        token_file = html.escape(tokens.get("token_file") or "")
        body = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>sg-mcp YouTube Connected</title></head>
<body style="font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;">
  <h2>YouTube Channel Connected Successfully</h2>
  <p>Token saved locally to: <code>{token_file}</code></p>
  <p>For cloud deployments (Render / Google Cloud Run), copy this environment variable:</p>
  <pre style="background:#111;color:#0f0;padding:12px;border-radius:8px;overflow:auto;">YOUTUBE_REFRESH_TOKEN={refresh_token}</pre>
  <p>You can close this tab now.</p>
</body>
</html>"""
        return HTMLResponse(body, status_code=200)

    return mcp, cfg, tts, store, yt


def create_app(
    settings: Settings | None = None,
    tts_service: GeminiTTSService | None = None,
    storage: AudioStorage | None = None,
    youtube_service: YouTubeService | None = None,
) -> Any:
    """Create the production ASGI application exposing Streamable HTTP MCP + routes."""
    mcp, cfg, _, _, _ = build_mcp_server(
        settings=settings,
        tts_service=tts_service,
        storage=storage,
        youtube_service=youtube_service,
    )

    middlewares: list[Middleware] = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )
    ]
    if cfg.mcp_auth_token:
        middlewares.append(Middleware(OptionalBearerAuthMiddleware, token=cfg.mcp_auth_token))

    return mcp.http_app(
        path="/mcp",
        transport="streamable-http",
        stateless_http=True,
        host_origin_protection=False,
        middleware=middlewares,
    )


def main() -> None:
    """Run the sg-mcp Streamable HTTP server via Uvicorn."""
    cfg = Settings.from_env()
    app = create_app(settings=cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
