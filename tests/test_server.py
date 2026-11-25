"""Unit and ASGI integration tests for sg-mcp."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastmcp import Client
from starlette.testclient import TestClient

from sg_mcp.config import Settings
from sg_mcp.server import build_mcp_server, create_app
from sg_mcp.storage import AudioStorage
from sg_mcp.tts_service import GeminiTTSService
from sg_mcp.voices import VoiceCatalog


def _make_dummy_wav(duration_ms: int = 100, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    frames = b"\x00\x00" * int(rate * (duration_ms / 1000.0))
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return buf.getvalue()


def _make_settings(tmp_path: Path, **overrides) -> Settings:
    base = Settings(
        api_keys=("key-1", "key-2"),
        tts_model="gemini-3.8-flash-tts",
        default_voice_id="charon",
        host="127.0.0.1",
        port=8080,
        public_base_url="https://example.run.app",
        audio_storage_dir=tmp_path / "audio",
        audio_ttl_seconds=3600,
        max_stored_files=50,
        mcp_auth_token="",
        include_inline_audio=True,
        voice_cache_ttl_seconds=3600,
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


class _FakeClient:
    def __init__(self, key: str, call_log: list[str], wav_bytes: bytes, fail_first: bool = False):
        self.key = key
        self.call_log = call_log
        self.wav_bytes = wav_bytes
        self.fail_first = fail_first
        self.voices = self
        self.models = self

    def list(self, **kwargs):
        self.call_log.append(f"voices.list:{self.key}")
        if self.fail_first and self.key == "key-1":
            raise RuntimeError("429 Quota exceeded on key-1")
        return SimpleNamespace(
            voices=[
                SimpleNamespace(
                    id="charon",
                    display_name="Charon",
                    gender="male",
                    language_code="en-US",
                    accent="General American",
                    pitch="low",
                    persona="Informative",
                    context="Enterprise Agent",
                    description="Default male voice.",
                ),
                SimpleNamespace(
                    id="kore",
                    display_name="Kore",
                    gender="female",
                    language_code="en-US",
                    accent="General American",
                    pitch="medium",
                    persona="Firm",
                    context="Enterprise Agent",
                    description="Firm female voice.",
                ),
                SimpleNamespace(
                    id="en-au-assistant-6",
                    display_name="Digital Assistant 6",
                    gender="neutral",
                    language_code="en-AU",
                    accent="Australian",
                    pitch="medium",
                    persona="Digital Assistant",
                    context="Conversational / Edu",
                    description="Neutral digital assistant.",
                ),
            ],
            next_page_token=None,
        )

    def generate_content(self, *, model: str, contents, config):
        self.call_log.append(f"generate_content:{self.key}:{model}")
        if self.fail_first and self.key == "key-1":
            raise RuntimeError("429 Quota exceeded on key-1")
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                inline_data=SimpleNamespace(
                                    data=self.wav_bytes,
                                    mime_type="audio/wav",
                                )
                            )
                        ]
                    )
                )
            ]
        )


def test_storage_ttl_eviction_and_path_safety(tmp_path: Path) -> None:
    store = AudioStorage(tmp_path / "audio", ttl_seconds=60, max_files=2)
    wav = _make_dummy_wav()
    r1 = store.save_wav(wav, voice_id="charon")
    r2 = store.save_wav(wav, voice_id="kore")
    r3 = store.save_wav(wav, voice_id="puck")

    assert store.get(r1.file_id) is None  # evicted due to max_files=2
    assert store.get(r2.file_id) is not None
    assert store.get(r3.file_id) is not None
    assert store.get("../etc/passwd") is None
    assert store.get("invalid") is None


def test_failover_and_default_voice(tmp_path: Path) -> None:
    wav = _make_dummy_wav()
    calls: list[str] = []
    settings = _make_settings(tmp_path)

    tts = GeminiTTSService(
        settings=settings,
        catalog=VoiceCatalog(),
        client_factory=lambda key: _FakeClient(key, calls, wav, fail_first=True),
    )

    result = tts.synthesize(script="Hello world", voice=None, style="calm and clear")
    assert result.voice.id == "charon"
    assert result.voice.gender == "male"
    assert result.model == "gemini-3.8-flash-tts"
    assert result.api_key_slot == 2
    assert result.wav_bytes[:4] == b"RIFF"
    assert "generate_content:key-1:gemini-3.8-flash-tts" in calls
    assert "generate_content:key-2:gemini-3.8-flash-tts" in calls


@pytest.mark.anyio
async def test_mcp_tools_and_download_route(tmp_path: Path) -> None:
    wav = _make_dummy_wav()
    calls: list[str] = []
    settings = _make_settings(tmp_path)
    tts = GeminiTTSService(
        settings=settings,
        catalog=VoiceCatalog(),
        client_factory=lambda key: _FakeClient(key, calls, wav, fail_first=False),
    )
    store = AudioStorage(settings.audio_storage_dir, ttl_seconds=3600, max_files=20)
    mcp, _, _, _ = build_mcp_server(settings=settings, tts_service=tts, storage=store)

    async with Client(mcp) as client:
        # 1. List neutral voices
        list_res = await client.call_tool("list_voices", {"gender": "neutral"})
        payload = json.loads(list_res.content[0].text)
        assert payload["total_matching"] == 1
        assert payload["voices"][0]["id"] == "en-au-assistant-6"
        assert payload["default_voice_id"] == "charon"

        # 2. Generate voice with omitted voice -> uses default male 'charon'
        gen_res = await client.call_tool(
            "generate_voice",
            {"script": "Testing cloud voice generation.", "style": "warm"},
        )
        assert gen_res.structured_content["used_default_voice"] is True
        assert gen_res.structured_content["voice"]["id"] == "charon"
        file_id = gen_res.structured_content["file"]["file_id"]
        assert gen_res.structured_content["file"]["download_url"].startswith(
            "https://example.run.app/download/"
        )

    # 3. Verify HTTP routes (/health, /download/{file_id}, auth middleware)
    app = create_app(settings=settings, tts_service=tts, storage=store)
    with TestClient(app) as http_client:
        health = http_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        dl = http_client.get(f"/download/{file_id}/test.wav")
        assert dl.status_code == 200
        assert dl.headers["content-type"].startswith("audio/wav")
        assert dl.content[:4] == b"RIFF"


def test_optional_bearer_auth_on_mcp(tmp_path: Path) -> None:
    wav = _make_dummy_wav()
    calls: list[str] = []
    settings = _make_settings(tmp_path, mcp_auth_token="secret-token-123")
    tts = GeminiTTSService(
        settings=settings,
        catalog=VoiceCatalog(),
        client_factory=lambda key: _FakeClient(key, calls, wav),
    )
    app = create_app(settings=settings, tts_service=tts)
    with TestClient(app) as http_client:
        assert http_client.get("/health").status_code == 200
        unauth = http_client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert unauth.status_code == 401
