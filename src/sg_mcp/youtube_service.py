"""YouTube Data API v3 service for uploading videos/Shorts, reading stats, and listing comments."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from sg_mcp.config import Settings

logger = logging.getLogger(__name__)

YOUTUBE_SCOPES: list[str] = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_URI = "https://oauth2.googleapis.com/token"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_MAX_REMOTE_VIDEO_BYTES = 512 * 1024 * 1024  # 512 MiB safety guard for cloud containers


def extract_video_id(video_id_or_url: str) -> str:
    """Extract an 11-character YouTube video ID from a raw ID or YouTube URL."""
    raw = (video_id_or_url or "").strip()
    if not raw:
        raise ValueError("video_id cannot be empty.")
    if _VIDEO_ID_RE.match(raw):
        return raw

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path_parts = [p for p in parsed.path.split("/") if p]

    if "youtu.be" in host and path_parts:
        candidate = path_parts[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"] and _VIDEO_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        for prefix in ("shorts", "embed", "live", "v"):
            if len(path_parts) >= 2 and path_parts[0] == prefix and _VIDEO_ID_RE.match(path_parts[1]):
                return path_parts[1]

    raise ValueError(
        f"Could not extract a valid 11-character YouTube video ID from '{video_id_or_url}'."
    )


def _is_http_url(value: str) -> bool:
    lower = value.strip().lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _download_to_temp_file(url: str, default_suffix: str) -> Path:
    """Stream a remote media file to a temporary local path."""
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or default_suffix
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                    total += len(chunk)
                    if total > _MAX_REMOTE_VIDEO_BYTES:
                        raise ValueError(
                            f"Remote media exceeds {_MAX_REMOTE_VIDEO_BYTES // (1024 * 1024)} MiB limit."
                        )
                    tmp.write(chunk)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path


class YouTubeService:
    """Handles OAuth2 credentials, video/Shorts uploads, stats, and comment retrieval."""

    def __init__(
        self,
        settings: Settings,
        youtube_client_factory: Callable[[Credentials], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._youtube_client_factory = youtube_client_factory or (
            lambda creds: build("youtube", "v3", credentials=creds, cache_discovery=False)
        )
        self._oauth_lock = threading.Lock()
        self._pending_oauth_states: dict[str, tuple[str | None, str, float]] = {}

    def has_oauth_client(self) -> bool:
        return bool(self._settings.youtube_client_id and self._settings.youtube_client_secret)

    def _client_config(self, redirect_uri: str) -> dict[str, Any]:
        if not self.has_oauth_client():
            raise RuntimeError(
                "YouTube OAuth client is not configured. Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET."
            )
        return {
            "web": {
                "client_id": self._settings.youtube_client_id,
                "client_secret": self._settings.youtube_client_secret,
                "auth_uri": AUTH_URI,
                "token_uri": TOKEN_URI,
                "redirect_uris": [redirect_uri],
            }
        }

    def create_auth_url(self, redirect_uri: str) -> tuple[str, str]:
        """Create a Google OAuth 2.0 consent URL and track PKCE state."""
        flow = Flow.from_client_config(
            self._client_config(redirect_uri),
            scopes=YOUTUBE_SCOPES,
            redirect_uri=redirect_uri,
        )
        kwargs: dict[str, Any] = {
            "access_type": "offline",
            "prompt": "consent",
        }
        if self._settings.youtube_channel_email:
            kwargs["login_hint"] = self._settings.youtube_channel_email

        auth_url, state = flow.authorization_url(**kwargs)
        now = time.monotonic()
        with self._oauth_lock:
            # Prune expired states (> 15 min)
            stale = [s for s, (_, _, ts) in self._pending_oauth_states.items() if now - ts > 900]
            for s in stale:
                self._pending_oauth_states.pop(s, None)
            self._pending_oauth_states[state] = (flow.code_verifier, redirect_uri, now)
        return auth_url, state

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Exchange an OAuth authorization code for tokens and persist them locally."""
        code_verifier: str | None = None
        effective_redirect = redirect_uri
        if state:
            with self._oauth_lock:
                entry = self._pending_oauth_states.pop(state, None)
            if entry:
                code_verifier, effective_redirect, _ = entry

        flow = Flow.from_client_config(
            self._client_config(effective_redirect),
            scopes=YOUTUBE_SCOPES,
            redirect_uri=effective_redirect,
            code_verifier=code_verifier,
            autogenerate_code_verifier=code_verifier is None,
        )
        # Google may include additional granted scopes (e.g. openid, userinfo.email)
        # in the token response; allow oauthlib to accept expanded scopes cleanly.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        flow.fetch_token(code=code)
        creds = flow.credentials

        refresh_token = creds.refresh_token or self._settings.youtube_refresh_token
        token_payload = {
            "token": creds.token,
            "refresh_token": refresh_token,
            "token_uri": creds.token_uri or TOKEN_URI,
            "client_id": self._settings.youtube_client_id,
            "client_secret": self._settings.youtube_client_secret,
            "scopes": list(creds.scopes or YOUTUBE_SCOPES),
        }

        try:
            self._settings.youtube_token_file.parent.mkdir(parents=True, exist_ok=True)
            self._settings.youtube_token_file.write_text(
                json.dumps(token_payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not write youtube token file: %s", exc)

        return {
            "refresh_token": refresh_token,
            "scopes": token_payload["scopes"],
            "token_file": str(self._settings.youtube_token_file),
        }

    def is_authenticated(self) -> bool:
        try:
            self._load_credentials(refresh_if_needed=False)
            return True
        except Exception:
            return False

    def _load_credentials(self, refresh_if_needed: bool = True) -> Credentials:
        if not self.has_oauth_client():
            raise RuntimeError(
                "YouTube OAuth client credentials missing. Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET."
            )

        creds: Credentials | None = None
        if self._settings.youtube_refresh_token:
            creds = Credentials(
                token=None,
                refresh_token=self._settings.youtube_refresh_token,
                token_uri=TOKEN_URI,
                client_id=self._settings.youtube_client_id,
                client_secret=self._settings.youtube_client_secret,
                scopes=YOUTUBE_SCOPES,
            )
        elif self._settings.youtube_token_file.is_file():
            creds = Credentials.from_authorized_user_file(
                str(self._settings.youtube_token_file), scopes=YOUTUBE_SCOPES
            )

        if creds is None or not creds.refresh_token:
            raise RuntimeError(
                "YouTube OAuth refresh token is not configured. Open /youtube/auth in your browser "
                "(or set YOUTUBE_REFRESH_TOKEN) to authorize the channel once."
            )

        if refresh_if_needed and (not creds.valid or creds.expired):
            creds.refresh(GoogleAuthRequest())

        return creds

    def _get_client(self) -> Any:
        creds = self._load_credentials(refresh_if_needed=True)
        return self._youtube_client_factory(creds)

    def upload_video(
        self,
        *,
        video_source: str,
        title: str,
        description: str = "",
        video_type: str = "long",
        privacy_status: str = "private",
        tags: list[str] | None = None,
        category_id: str = "22",
        made_for_kids: bool = False,
        publish_at: str | None = None,
        default_language: str | None = None,
        thumbnail_source: str | None = None,
    ) -> dict[str, Any]:
        """Upload a video (Long or Short) to YouTube from a local path or HTTP(S) URL."""
        cleaned_title = (title or "").strip()
        if not cleaned_title:
            raise ValueError("Video title cannot be empty.")

        norm_type = (video_type or "long").strip().lower()
        if norm_type not in {"short", "shorts", "long", "standard"}:
            raise ValueError("video_type must be 'short' or 'long'.")
        is_short = norm_type in {"short", "shorts"}

        norm_privacy = (privacy_status or "private").strip().lower()
        if norm_privacy not in {"private", "unlisted", "public"}:
            raise ValueError("privacy_status must be 'private', 'unlisted', or 'public'.")

        temp_files: list[Path] = []
        try:
            source_str = (video_source or "").strip()
            if not source_str:
                raise ValueError("video_source (local file path or HTTP/HTTPS URL) is required.")

            if _is_http_url(source_str):
                video_path = _download_to_temp_file(source_str, ".mp4")
                temp_files.append(video_path)
            else:
                video_path = Path(source_str).expanduser()
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video file not found: {video_path}")

            cleaned_desc = (description or "").strip()
            cleaned_tags = [t.strip() for t in (tags or []) if t and t.strip()]
            if is_short:
                if "#shorts" not in cleaned_desc.lower() and "#shorts" not in cleaned_title.lower():
                    cleaned_desc = f"{cleaned_desc}\n\n#Shorts".strip()
                if not any(t.lower() == "shorts" for t in cleaned_tags):
                    cleaned_tags.append("Shorts")

            snippet: dict[str, Any] = {
                "title": cleaned_title[:100],
                "description": cleaned_desc[:5000],
                "categoryId": (category_id or "22").strip(),
            }
            if cleaned_tags:
                snippet["tags"] = cleaned_tags
            if default_language and default_language.strip():
                snippet["defaultLanguage"] = default_language.strip()

            status: dict[str, Any] = {
                "privacyStatus": norm_privacy,
                "selfDeclaredMadeForKids": bool(made_for_kids),
            }
            if publish_at and publish_at.strip():
                # Scheduled publishing requires privacyStatus='private' in YouTube Data API v3
                status["privacyStatus"] = "private"
                status["publishAt"] = publish_at.strip()

            mime_type, _ = mimetypes.guess_type(str(video_path))
            media = MediaFileUpload(
                str(video_path),
                mimetype=mime_type or "video/mp4",
                chunksize=8 * 1024 * 1024,
                resumable=True,
            )

            youtube = self._get_client()
            insert_req = youtube.videos().insert(
                part="snippet,status",
                body={"snippet": snippet, "status": status},
                media_body=media,
            )

            response = None
            while response is None:
                _, response = insert_req.next_chunk()

            video_id = response["id"]

            thumbnail_uploaded = False
            if thumbnail_source and thumbnail_source.strip():
                thumb_str = thumbnail_source.strip()
                if _is_http_url(thumb_str):
                    thumb_path = _download_to_temp_file(thumb_str, ".jpg")
                    temp_files.append(thumb_path)
                else:
                    thumb_path = Path(thumb_str).expanduser()
                    if not thumb_path.is_file():
                        raise FileNotFoundError(f"Thumbnail file not found: {thumb_path}")
                thumb_mime, _ = mimetypes.guess_type(str(thumb_path))
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(
                        str(thumb_path), mimetype=thumb_mime or "image/jpeg"
                    ),
                ).execute()
                thumbnail_uploaded = True

            resp_snippet = response.get("snippet", {})
            resp_status = response.get("status", {})
            return {
                "status": "uploaded",
                "video_id": video_id,
                "video_type": "short" if is_short else "long",
                "title": resp_snippet.get("title", cleaned_title),
                "description": resp_snippet.get("description", cleaned_desc),
                "channel_id": resp_snippet.get("channelId"),
                "channel_title": resp_snippet.get("channelTitle"),
                "privacy_status": resp_status.get("privacyStatus", status["privacyStatus"]),
                "published_at": resp_snippet.get("publishedAt"),
                "thumbnail_uploaded": thumbnail_uploaded,
                "watch_url": f"https://www.youtube.com/watch?v={video_id}",
                "shorts_url": f"https://www.youtube.com/shorts/{video_id}" if is_short else None,
                "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
            }
        finally:
            for tmp in temp_files:
                tmp.unlink(missing_ok=True)

    def get_stats(
        self,
        *,
        video_id: str | None = None,
        max_videos: int = 10,
    ) -> dict[str, Any]:
        """Read statistics for a specific YouTube video or the authenticated channel + recent videos."""
        youtube = self._get_client()

        if video_id and video_id.strip():
            vid = extract_video_id(video_id)
            resp = (
                youtube.videos()
                .list(part="snippet,statistics,contentDetails,status", id=vid)
                .execute()
            )
            items = resp.get("items") or []
            if not items:
                raise LookupError(f"YouTube video '{vid}' not found or not accessible.")
            return {"scope": "video", "video": self._format_video_item(items[0])}

        channel_resp = (
            youtube.channels()
            .list(part="snippet,statistics,contentDetails", mine=True)
            .execute()
        )
        channels = channel_resp.get("items") or []
        if not channels:
            raise LookupError("No YouTube channel found for the authenticated Google account.")

        ch = channels[0]
        ch_snippet = ch.get("snippet", {})
        ch_stats = ch.get("statistics", {})
        uploads_playlist = (
            ch.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        )

        recent_videos: list[dict[str, Any]] = []
        safe_max = max(1, min(max_videos, 50))
        if uploads_playlist:
            pl_resp = (
                youtube.playlistItems()
                .list(part="contentDetails", playlistId=uploads_playlist, maxResults=safe_max)
                .execute()
            )
            video_ids = [
                item["contentDetails"]["videoId"]
                for item in (pl_resp.get("items") or [])
                if item.get("contentDetails", {}).get("videoId")
            ]
            if video_ids:
                vids_resp = (
                    youtube.videos()
                    .list(
                        part="snippet,statistics,contentDetails,status",
                        id=",".join(video_ids),
                    )
                    .execute()
                )
                recent_videos = [self._format_video_item(v) for v in (vids_resp.get("items") or [])]

        return {
            "scope": "channel",
            "channel": {
                "channel_id": ch.get("id"),
                "title": ch_snippet.get("title"),
                "custom_url": ch_snippet.get("customUrl"),
                "published_at": ch_snippet.get("publishedAt"),
                "subscriber_count": int(ch_stats.get("subscriberCount", 0)),
                "hidden_subscriber_count": bool(ch_stats.get("hiddenSubscriberCount", False)),
                "total_view_count": int(ch_stats.get("viewCount", 0)),
                "total_video_count": int(ch_stats.get("videoCount", 0)),
            },
            "recent_videos_count": len(recent_videos),
            "recent_videos": recent_videos,
        }

    def get_comments(
        self,
        *,
        video_id: str | None = None,
        max_results: int = 50,
        order: str = "relevance",
        include_replies: bool = True,
        page_token: str | None = None,
        fetch_all: bool = False,
    ) -> dict[str, Any]:
        """Fetch top-level comments and replies for a YouTube video or across the whole channel."""
        youtube = self._get_client()

        norm_order = (order or "relevance").strip().lower()
        if norm_order not in {"relevance", "time"}:
            raise ValueError("order must be 'relevance' or 'time'.")

        list_kwargs: dict[str, Any] = {
            "part": "snippet,replies" if include_replies else "snippet",
            "textFormat": "plainText",
            "order": norm_order,
        }

        target_video_id: str | None = None
        target_channel_id: str | None = None
        if video_id and video_id.strip():
            target_video_id = extract_video_id(video_id)
            list_kwargs["videoId"] = target_video_id
        else:
            ch_resp = youtube.channels().list(part="id", mine=True).execute()
            ch_items = ch_resp.get("items") or []
            if not ch_items:
                raise LookupError("No YouTube channel found for the authenticated account.")
            target_channel_id = ch_items[0]["id"]
            list_kwargs["allThreadsRelatedToChannelId"] = target_channel_id

        threads: list[dict[str, Any]] = []
        next_token = page_token
        max_total = 500 if fetch_all else max(1, min(max_results, 100))

        while True:
            page_size = min(100, max_total - len(threads))
            if page_size <= 0:
                break
            req_kwargs = dict(list_kwargs, maxResults=page_size)
            if next_token:
                req_kwargs["pageToken"] = next_token

            resp = youtube.commentThreads().list(**req_kwargs).execute()
            for item in resp.get("items") or []:
                threads.append(self._format_comment_thread(item, include_replies=include_replies))

            next_token = resp.get("nextPageToken")
            if not fetch_all or not next_token or len(threads) >= max_total:
                break

        return {
            "video_id": target_video_id,
            "channel_id": target_channel_id,
            "order": norm_order,
            "total_threads_returned": len(threads),
            "next_page_token": next_token,
            "comments": threads,
        }

    @staticmethod
    def _format_video_item(item: dict[str, Any]) -> dict[str, Any]:
        vid = item.get("id")
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        status = item.get("status", {})
        return {
            "video_id": vid,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_id": snippet.get("channelId"),
            "channel_title": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt"),
            "tags": snippet.get("tags", []),
            "duration": details.get("duration"),
            "privacy_status": status.get("privacyStatus"),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
            "favorite_count": int(stats.get("favoriteCount", 0)),
            "watch_url": f"https://www.youtube.com/watch?v={vid}",
        }

    @staticmethod
    def _format_comment_thread(item: dict[str, Any], include_replies: bool) -> dict[str, Any]:
        top = item.get("snippet", {}).get("topLevelComment", {})
        top_snip = top.get("snippet", {})
        formatted: dict[str, Any] = {
            "thread_id": item.get("id"),
            "video_id": top_snip.get("videoId"),
            "comment_id": top.get("id"),
            "author": top_snip.get("authorDisplayName"),
            "author_channel_url": top_snip.get("authorChannelUrl"),
            "text": top_snip.get("textDisplay"),
            "like_count": int(top_snip.get("likeCount", 0)),
            "published_at": top_snip.get("publishedAt"),
            "updated_at": top_snip.get("updatedAt"),
            "total_reply_count": int(item.get("snippet", {}).get("totalReplyCount", 0)),
        }
        if include_replies:
            replies_raw = item.get("replies", {}).get("comments", [])
            formatted["replies"] = [
                {
                    "comment_id": r.get("id"),
                    "author": r.get("snippet", {}).get("authorDisplayName"),
                    "text": r.get("snippet", {}).get("textDisplay"),
                    "like_count": int(r.get("snippet", {}).get("likeCount", 0)),
                    "published_at": r.get("snippet", {}).get("publishedAt"),
                }
                for r in replies_raw
            ]
        return formatted
