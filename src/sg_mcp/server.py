"""FastMCP Streamable HTTP server for Gemini 3.8 Flash TTS and future tools."""

from __future__ import annotations

import hmac
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
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from sg_mcp import __version__
from sg_mcp.config import Settings
from sg_mcp.storage import AudioStorage
from sg_mcp.tts_service import GeminiTTSService

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


def _resolve_base_url(settings: Settings) -> str:
    if settings.public_base_url:
        return settings.public_base_url

    try:
        request = get_http_request()
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
    except RuntimeError:
        pass

    return f"http://127.0.0.1:{settings.port}"


def build_mcp_server(
    settings: Settings | None = None,
    tts_service: GeminiTTSService | None = None,
    storage: AudioStorage | None = None,
) -> tuple[FastMCP, Settings, GeminiTTSService, AudioStorage]:
    """Build and configure the FastMCP server and its tools/routes."""
    cfg = settings or Settings.from_env()
    tts = tts_service or GeminiTTSService(cfg)
    store = storage or AudioStorage(
        base_dir=cfg.audio_storage_dir,
        ttl_seconds=cfg.audio_ttl_seconds,
        max_files=cfg.max_stored_files,
    )

    mcp = FastMCP(
        name="sg-mcp",
        version=__version__,
        instructions=(
            "Cloud MCP server providing expressive text-to-speech generation via "
            "gemini-3.8-flash-tts. Call `list_voices` if the user wants to browse or pick "
            "male, female, or neutral voices. Call `generate_voice` with `script` and "
            "optional `voice` (defaults to male voice 'charon' if omitted) plus optional "
            "style/language/temperature settings. Always share the returned downloadable "
            "audio URL with the user."
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
                "tools": ["generate_voice", "list_voices"],
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

    return mcp, cfg, tts, store


def create_app(
    settings: Settings | None = None,
    tts_service: GeminiTTSService | None = None,
    storage: AudioStorage | None = None,
) -> Any:
    """Create the production ASGI application exposing Streamable HTTP MCP + download routes."""
    mcp, cfg, _, _ = build_mcp_server(
        settings=settings,
        tts_service=tts_service,
        storage=storage,
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
