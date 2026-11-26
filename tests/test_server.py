"""Unit and ASGI integration tests for sg-mcp (TTS + YouTube tools)."""

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
from sg_mcp.instagram_service import InstagramService
from sg_mcp.server import build_mcp_server, create_app
from sg_mcp.storage import AudioStorage
from sg_mcp.tts_service import GeminiTTSService
from sg_mcp.voices import VoiceCatalog
from sg_mcp.youtube_service import YouTubeService, extract_video_id


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
        youtube_client_id="yt-client-id",
        youtube_client_secret="yt-client-secret",
        youtube_refresh_token="yt-refresh-token",
        youtube_token_file=tmp_path / "youtube_token.json",
        youtube_channel_email="dev.satabda@gmail.com",
        youtube_redirect_uri="",
        meta_app_id="meta-app-id",
        meta_app_secret="meta-app-secret",
        instagram_access_token="ig-access-token",
        instagram_user_id="17841400000000001",
        instagram_auth_mode="facebook",
        instagram_graph_version="v27.0",
        instagram_token_file=tmp_path / "instagram_token.json",
        instagram_redirect_uri="",
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


class _FakeExec:
    def __init__(self, payload: dict):
        self._payload = payload

    def execute(self) -> dict:
        return self._payload


class _FakeInsertReq:
    def __init__(self, payload: dict):
        self._payload = payload

    def next_chunk(self):
        return None, self._payload


class _FakeYouTubeAPI:
    def __init__(self):
        self.last_insert: dict | None = None

    def videos(self):
        api = self

        class _Videos:
            def insert(self, *, part, body, media_body):
                api.last_insert = {"part": part, "body": body}
                return _FakeInsertReq(
                    {
                        "id": "dQw4w9WgXcQ",
                        "snippet": {
                            "title": body["snippet"]["title"],
                            "description": body["snippet"]["description"],
                            "channelId": "UC123",
                            "channelTitle": "Satabda Demo",
                            "publishedAt": "2026-11-26T00:00:00Z",
                        },
                        "status": {"privacyStatus": body["status"]["privacyStatus"]},
                    }
                )

            def list(self, *, part, id):
                return _FakeExec(
                    {
                        "items": [
                            {
                                "id": id.split(",")[0],
                                "snippet": {
                                    "title": "Demo Video",
                                    "description": "Desc",
                                    "channelId": "UC123",
                                    "channelTitle": "Satabda Demo",
                                    "publishedAt": "2026-11-26T00:00:00Z",
                                    "tags": ["Shorts"],
                                },
                                "statistics": {
                                    "viewCount": "150",
                                    "likeCount": "25",
                                    "commentCount": "4",
                                    "favoriteCount": "0",
                                },
                                "contentDetails": {"duration": "PT15S"},
                                "status": {"privacyStatus": "private"},
                            }
                        ]
                    }
                )

        return _Videos()

    def channels(self):
        class _Channels:
            def list(self, *, part, mine=True):
                return _FakeExec(
                    {
                        "items": [
                            {
                                "id": "UC123",
                                "snippet": {
                                    "title": "Satabda Demo",
                                    "customUrl": "@satabdademo",
                                    "publishedAt": "2024-01-01T00:00:00Z",
                                },
                                "statistics": {
                                    "subscriberCount": "42",
                                    "hiddenSubscriberCount": False,
                                    "viewCount": "1200",
                                    "videoCount": "5",
                                },
                                "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
                            }
                        ]
                    }
                )

        return _Channels()

    def playlistItems(self):
        class _PlaylistItems:
            def list(self, *, part, playlistId, maxResults=10):
                return _FakeExec({"items": [{"contentDetails": {"videoId": "dQw4w9WgXcQ"}}]})

        return _PlaylistItems()

    def commentThreads(self):
        class _CommentThreads:
            def list(self, **kwargs):
                return _FakeExec(
                    {
                        "items": [
                            {
                                "id": "thread-1",
                                "snippet": {
                                    "totalReplyCount": 1,
                                    "topLevelComment": {
                                        "id": "c1",
                                        "snippet": {
                                            "videoId": "dQw4w9WgXcQ",
                                            "authorDisplayName": "Viewer One",
                                            "authorChannelUrl": "https://youtube.com/@v1",
                                            "textDisplay": "Great demo!",
                                            "likeCount": 3,
                                            "publishedAt": "2026-11-26T01:00:00Z",
                                            "updatedAt": "2026-11-26T01:00:00Z",
                                        },
                                    },
                                },
                                "replies": {
                                    "comments": [
                                        {
                                            "id": "c2",
                                            "snippet": {
                                                "authorDisplayName": "Satabda Demo",
                                                "textDisplay": "Thanks!",
                                                "likeCount": 1,
                                                "publishedAt": "2026-11-26T01:05:00Z",
                                            },
                                        }
                                    ]
                                },
                            }
                        ],
                        "nextPageToken": None,
                    }
                )

        return _CommentThreads()


class _FakeCreds:
    valid = True
    expired = False
    refresh_token = "yt-refresh-token"

    def refresh(self, req):
        pass


class _TestableYouTubeService(YouTubeService):
    def _load_credentials(self, refresh_if_needed: bool = True):
        return _FakeCreds()


def test_storage_ttl_eviction_and_path_safety(tmp_path: Path) -> None:
    store = AudioStorage(tmp_path / "audio", ttl_seconds=60, max_files=2)
    wav = _make_dummy_wav()
    r1 = store.save_wav(wav, voice_id="charon")
    r2 = store.save_wav(wav, voice_id="kore")
    r3 = store.save_wav(wav, voice_id="puck")

    assert store.get(r1.file_id) is None
    assert store.get(r2.file_id) is not None
    assert store.get(r3.file_id) is not None
    assert store.get("../etc/passwd") is None
    assert store.get("invalid") is None


def test_extract_video_id_formats() -> None:
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=5") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/invalid")


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
async def test_mcp_tts_and_youtube_tools(tmp_path: Path) -> None:
    wav = _make_dummy_wav()
    calls: list[str] = []
    settings = _make_settings(tmp_path)
    tts = GeminiTTSService(
        settings=settings,
        catalog=VoiceCatalog(),
        client_factory=lambda key: _FakeClient(key, calls, wav, fail_first=False),
    )
    store = AudioStorage(settings.audio_storage_dir, ttl_seconds=3600, max_files=20)
    fake_api = _FakeYouTubeAPI()
    yt = _TestableYouTubeService(settings, youtube_client_factory=lambda creds: fake_api)

    demo_mp4 = tmp_path / "demo.mp4"
    demo_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    mcp, _, _, _, _, _ = build_mcp_server(
        settings=settings, tts_service=tts, storage=store, youtube_service=yt
    )

    async with Client(mcp) as client:
        # 1. TTS voice listing & generation
        list_res = await client.call_tool("list_voices", {"gender": "neutral"})
        payload = json.loads(list_res.content[0].text)
        assert payload["total_matching"] == 1
        assert payload["voices"][0]["id"] == "en-au-assistant-6"

        gen_res = await client.call_tool(
            "generate_voice",
            {"script": "Testing cloud voice generation.", "style": "warm"},
        )
        assert gen_res.structured_content["used_default_voice"] is True
        assert gen_res.structured_content["voice"]["id"] == "charon"
        file_id = gen_res.structured_content["file"]["file_id"]

        # 2. YouTube Short upload (path + base64)
        up_res = await client.call_tool(
            "upload_youtube_video",
            {
                "video_source": str(demo_mp4),
                "title": "Demo Short",
                "description": "Testing YouTube upload via sg-mcp",
                "video_type": "short",
                "privacy_status": "private",
            },
        )
        assert up_res.structured_content["video_id"] == "dQw4w9WgXcQ"
        assert up_res.structured_content["shorts_url"] == "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert "#Shorts" in fake_api.last_insert["body"]["snippet"]["description"]

        import base64
        up_b64 = await client.call_tool(
            "upload_youtube_video",
            {
                "video_base64": base64.b64encode(demo_mp4.read_bytes()).decode(),
                "title": "Demo Base64 Short",
                "video_type": "short",
                "privacy_status": "private",
            },
        )
        assert up_b64.structured_content["video_id"] == "dQw4w9WgXcQ"

        # 3. YouTube channel & video stats
        ch_stats = await client.call_tool("get_youtube_stats", {})
        assert ch_stats.structured_content["scope"] == "channel"
        assert ch_stats.structured_content["channel"]["subscriber_count"] == 42
        assert ch_stats.structured_content["recent_videos"][0]["like_count"] == 25

        vid_stats = await client.call_tool(
            "get_youtube_stats", {"video_id": "https://youtu.be/dQw4w9WgXcQ"}
        )
        assert vid_stats.structured_content["scope"] == "video"
        assert vid_stats.structured_content["video"]["comment_count"] == 4

        # 4. YouTube comments
        comments_res = await client.call_tool(
            "get_youtube_comments", {"video_id": "dQw4w9WgXcQ"}
        )
        assert comments_res.structured_content["total_threads_returned"] == 1
        assert comments_res.structured_content["comments"][0]["text"] == "Great demo!"
        assert comments_res.structured_content["comments"][0]["replies"][0]["text"] == "Thanks!"

    # 5. Verify HTTP routes (/health, /download/{file_id}, /youtube/auth)
    app = create_app(settings=settings, tts_service=tts, storage=store, youtube_service=yt)
    with TestClient(app) as http_client:
        health = http_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        root = http_client.get("/")
        assert root.status_code == 200
        assert "upload_youtube_video" in root.json()["tools"]
        assert "upload_instagram_content" in root.json()["tools"]

        dl = http_client.get(f"/download/{file_id}/test.wav")
        assert dl.status_code == 200
        assert dl.content[:4] == b"RIFF"

        auth_redir = http_client.get("/youtube/auth", follow_redirects=False)
        assert auth_redir.status_code == 302
        assert "accounts.google.com/o/oauth2/auth" in auth_redir.headers["location"]

        ig_redir = http_client.get("/instagram/auth", follow_redirects=False)
        assert ig_redir.status_code == 302
        assert "facebook.com/v27.0/dialog/oauth" in ig_redir.headers["location"]


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


def test_oauth_exchange_accepts_expanded_google_scopes(tmp_path: Path, monkeypatch) -> None:
    import os
    from google_auth_oauthlib.flow import Flow

    settings = _make_settings(tmp_path, youtube_refresh_token="")
    yt = YouTubeService(settings)
    _, state = yt.create_auth_url("https://sg-mcp.onrender.com/youtube/callback")

    def fake_fetch_token(self, **kwargs):
        # Simulate Google returning extra openid/userinfo scopes alongside youtube.force-ssl
        assert os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE") == "1"
        self.oauth2session.token = self.oauth2session._client.parse_request_body_response(
            json.dumps(
                {
                    "access_token": "access-123",
                    "refresh_token": "refresh-456",
                    "token_type": "Bearer",
                    "expires_in": 3599,
                    "scope": (
                        "openid https://www.googleapis.com/auth/userinfo.profile "
                        "https://www.googleapis.com/auth/youtube.force-ssl "
                        "https://www.googleapis.com/auth/userinfo.email"
                    ),
                }
            ),
            scope=self.oauth2session.scope,
        )
        return self.oauth2session.token

    monkeypatch.delenv("OAUTHLIB_RELAX_TOKEN_SCOPE", raising=False)
    monkeypatch.setattr(Flow, "fetch_token", fake_fetch_token)

    result = yt.exchange_code(
        code="auth-code-xyz",
        redirect_uri="https://sg-mcp.onrender.com/youtube/callback",
        state=state,
    )
    assert result["refresh_token"] == "refresh-456"
    assert settings.youtube_token_file.is_file()


@pytest.mark.anyio
async def test_instagram_oauth_and_mcp_tools(tmp_path: Path) -> None:
    import base64

    recorded_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, str(request.url)))
        path = request.url.path
        if path.endswith("/oauth/access_token"):
            if request.url.params.get("grant_type") == "fb_exchange_token":
                return httpx.Response(200, json={"access_token": "long-user-token", "expires_in": 5184000})
            return httpx.Response(200, json={"access_token": "short-user-token"})
        if path.endswith("/me/accounts"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "page-1",
                            "name": "StupidlySimpleOne Page",
                            "access_token": "page-token-xyz",
                            "instagram_business_account": {
                                "id": "17841400000000001",
                                "username": "stupidlysimpleone",
                            },
                        }
                    ]
                },
            )
        if path.endswith("/17841400000000001/media") and request.method == "POST":
            return httpx.Response(200, json={"id": "container-999"})
        if path.endswith("/container-999"):
            return httpx.Response(200, json={"id": "container-999", "status_code": "FINISHED"})
        if path.endswith("/17841400000000001/media_publish") and request.method == "POST":
            return httpx.Response(200, json={"id": "media-777"})
        if path.endswith("/media-777/insights"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"name": "views", "values": [{"value": 320}]},
                        {"name": "reach", "values": [{"value": 280}]},
                        {"name": "saved", "values": [{"value": 12}]},
                        {"name": "shares", "values": [{"value": 5}]},
                        {"name": "total_interactions", "values": [{"value": 45}]},
                    ]
                },
            )
        if path.endswith("/media-777/comments"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "ig-c1",
                            "username": "fan_user",
                            "text": "Awesome reel!",
                            "like_count": 4,
                            "hidden": False,
                            "timestamp": "2026-11-26T10:00:00+0000",
                            "replies": {
                                "data": [
                                    {
                                        "id": "ig-r1",
                                        "username": "stupidlysimpleone",
                                        "text": "Thank you!",
                                        "like_count": 1,
                                        "timestamp": "2026-11-26T10:05:00+0000",
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
        if path.endswith("/media-777"):
            return httpx.Response(
                200,
                json={
                    "id": "media-777",
                    "caption": "Test IG Reel",
                    "media_type": "VIDEO",
                    "media_product_type": "REELS",
                    "permalink": "https://www.instagram.com/reel/ABC123/",
                    "shortcode": "ABC123",
                    "timestamp": "2026-11-26T09:55:00+0000",
                    "like_count": 28,
                    "comments_count": 2,
                },
            )
        if path.endswith("/17841400000000001/insights"):
            return httpx.Response(
                200,
                json={"data": [{"name": "reach", "total_value": {"value": 500}}]},
            )
        if path.endswith("/17841400000000001/media") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "media-777",
                            "caption": "Test IG Reel",
                            "media_type": "VIDEO",
                            "media_product_type": "REELS",
                            "permalink": "https://www.instagram.com/reel/ABC123/",
                            "timestamp": "2026-11-26T09:55:00+0000",
                            "like_count": 28,
                            "comments_count": 2,
                        }
                    ]
                },
            )
        if path.endswith("/17841400000000001"):
            return httpx.Response(
                200,
                json={
                    "id": "17841400000000001",
                    "username": "stupidlysimpleone",
                    "name": "Stupidly Simple One",
                    "followers_count": 125,
                    "follows_count": 10,
                    "media_count": 1,
                },
            )
        return httpx.Response(404, json={"error": {"message": f"Unhandled mock path {path}"}})

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    settings = _make_settings(tmp_path, instagram_access_token="", instagram_user_id="")
    store = AudioStorage(settings.audio_storage_dir, ttl_seconds=3600, max_files=20)
    ig = InstagramService(settings, storage=store, http_client=mock_http)

    # 1. Test OAuth exchange (Facebook Login -> Long-lived token -> Linked IG account)
    _, state = ig.create_auth_url("https://example.run.app/instagram/callback")
    exchanged = ig.exchange_code(
        code="fb-code-123",
        redirect_uri="https://example.run.app/instagram/callback",
        state=state,
    )
    assert exchanged["ig_user_id"] == "17841400000000001"
    assert exchanged["ig_username"] == "stupidlysimpleone"
    assert exchanged["access_token"] == "page-token-xyz"
    assert settings.instagram_token_file.is_file()

    # 2. Test MCP tools via FastMCP Client
    mcp, _, _, _, _, _ = build_mcp_server(settings=settings, storage=store, instagram_service=ig)
    async with Client(mcp) as client:
        b64_mp4 = base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
        pub = await client.call_tool(
            "upload_instagram_content",
            {"media_type": "reel", "media_base64": b64_mp4, "caption": "Test IG Reel"},
        )
        assert pub.structured_content["media_id"] == "media-777"
        assert pub.structured_content["permalink"] == "https://www.instagram.com/reel/ABC123/"

        acct_stats = await client.call_tool("get_instagram_stats", {})
        assert acct_stats.structured_content["scope"] == "account"
        assert acct_stats.structured_content["account"]["username"] == "stupidlysimpleone"
        assert acct_stats.structured_content["recent_posts"][0]["like_count"] == 28

        media_stats = await client.call_tool("get_instagram_stats", {"media_id": "media-777"})
        assert media_stats.structured_content["scope"] == "media"
        assert media_stats.structured_content["media"]["insights"]["views"] == 320

        comments = await client.call_tool("get_instagram_comments", {})
        assert comments.structured_content["total_comments_returned"] == 1
        assert comments.structured_content["comments"][0]["text"] == "Awesome reel!"
        assert comments.structured_content["comments"][0]["replies"][0]["text"] == "Thank you!"
