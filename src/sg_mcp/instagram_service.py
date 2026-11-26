"""Instagram Graph API v27.0 service for publishing content, reading stats, and reading comments."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from sg_mcp.config import Settings
from sg_mcp.storage import AudioStorage

logger = logging.getLogger(__name__)

FACEBOOK_SCOPES: list[str] = [
    "instagram_basic",
    "instagram_content_publish",
    "instagram_manage_comments",
    "instagram_manage_insights",
    "pages_show_list",
    "pages_read_engagement",
]

INSTAGRAM_LOGIN_SCOPES: list[str] = [
    "instagram_business_basic",
    "instagram_business_content_publish",
    "instagram_business_manage_comments",
    "instagram_business_manage_insights",
]

_MAX_STAGED_MEDIA_BYTES = 250 * 1024 * 1024  # 250 MiB guard


def _is_http_url(value: str) -> bool:
    lower = value.strip().lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _extract_error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or resp.text
            user_msg = err.get("error_user_msg")
            code = err.get("code")
            subcode = err.get("error_subcode")
            parts = [f"Meta API error ({resp.status_code}, code={code}, subcode={subcode}): {msg}"]
            if user_msg:
                parts.append(user_msg)
            return " - ".join(parts)
        if isinstance(data, dict) and data.get("error_message"):
            return f"Meta OAuth error ({resp.status_code}): {data['error_message']}"
    except Exception:
        pass
    return f"Meta API HTTP {resp.status_code}: {resp.text}"


class InstagramService:
    """Handles Meta/Instagram OAuth2, media publishing, insights/stats, and comments."""

    def __init__(
        self,
        settings: Settings,
        storage: AudioStorage | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._http = http_client or httpx.Client(timeout=60.0, follow_redirects=True)
        self._lock = threading.Lock()
        self._pending_states: dict[str, tuple[str, str, float]] = {}

    @property
    def auth_mode(self) -> str:
        mode = (self._settings.instagram_auth_mode or "").strip().lower()
        if mode == "instagram":
            return "instagram"
        if mode == "unified" or (mode != "facebook" and bool(self._settings.meta_config_id)):
            return "unified"
        if mode == "facebook" and self._settings.meta_config_id:
            return "unified"
        return "facebook"

    @property
    def graph_version(self) -> str:
        return self._settings.instagram_graph_version or "v27.0"

    @property
    def api_base(self) -> str:
        host = "https://graph.instagram.com" if self.auth_mode == "instagram" else "https://graph.facebook.com"
        return f"{host}/{self.graph_version}"

    def has_app_credentials(self) -> bool:
        return bool(self._settings.meta_app_id and self._settings.meta_app_secret)

    def _load_token_data(self) -> dict[str, Any]:
        if self._settings.instagram_access_token:
            return {
                "access_token": self._settings.instagram_access_token,
                "ig_user_id": self._settings.instagram_user_id,
                "auth_mode": self.auth_mode,
            }
        if self._settings.instagram_token_file.is_file():
            try:
                data = json.loads(self._settings.instagram_token_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("access_token"):
                    if self._settings.instagram_user_id and not data.get("ig_user_id"):
                        data["ig_user_id"] = self._settings.instagram_user_id
                    return data
            except Exception as exc:
                logger.warning("Failed reading instagram token file: %s", exc)
        return {}

    def is_authenticated(self) -> bool:
        data = self._load_token_data()
        return bool(data.get("access_token"))

    def _require_auth(self) -> tuple[str, str]:
        data = self._load_token_data()
        token = (data.get("access_token") or "").strip()
        if not token:
            raise RuntimeError(
                "Instagram access token is not configured. Open /instagram/auth in your browser "
                "(or set INSTAGRAM_ACCESS_TOKEN) to authorize your Professional account."
            )

        ig_user_id = (self._settings.instagram_user_id or data.get("ig_user_id") or "").strip()
        if not ig_user_id:
            ig_user_id = self._discover_ig_user_id(token)
            data["ig_user_id"] = ig_user_id
            self._save_token_data(data)

        return token, ig_user_id

    def _save_token_data(self, payload: dict[str, Any]) -> None:
        try:
            self._settings.instagram_token_file.parent.mkdir(parents=True, exist_ok=True)
            self._settings.instagram_token_file.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not write instagram token file: %s", exc)

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._http.request(method, url, params=params, data=data)
        if resp.status_code >= 400:
            raise RuntimeError(_extract_error_message(resp))
        payload = resp.json()
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(_extract_error_message(resp))
        return payload

    def _resolve_facebook_or_unified_ig_account(self, access_token: str) -> dict[str, Any]:
        """Resolve IG account ID + best access token across Unified Login and Page-linked flows."""
        base = f"https://graph.facebook.com/{self.graph_version}"

        # 1. Unified Login (Facebook Login for Business SUAT): /me/assigned_instagram_accounts
        try:
            assigned = self._request(
                "GET",
                f"{base}/me/assigned_instagram_accounts",
                params={"fields": "id,username,name", "access_token": access_token},
            )
            items = assigned.get("data") or []
            if items and items[0].get("id"):
                return {
                    "ig_user_id": str(items[0]["id"]),
                    "ig_username": items[0].get("username"),
                    "access_token": access_token,
                    "source": "assigned_instagram_accounts",
                }
        except Exception:
            pass

        # 2. Unified Login Business Portfolio: /me?fields=client_business_id -> /{biz}/owned_instagram_accounts
        try:
            me_biz = self._request(
                "GET",
                f"{base}/me",
                params={"fields": "id,client_business_id", "access_token": access_token},
            )
            biz_id = me_biz.get("client_business_id")
            if biz_id:
                for edge in ("owned_instagram_accounts", "client_instagram_accounts"):
                    try:
                        biz_accts = self._request(
                            "GET",
                            f"{base}/{biz_id}/{edge}",
                            params={"fields": "id,username,name", "access_token": access_token},
                        )
                        items = biz_accts.get("data") or []
                        if items and items[0].get("id"):
                            return {
                                "ig_user_id": str(items[0]["id"]),
                                "ig_username": items[0].get("username"),
                                "access_token": access_token,
                                "business_id": str(biz_id),
                                "source": edge,
                            }
                    except Exception:
                        pass
        except Exception:
            pass

        # 3. Traditional Facebook Login: /me/accounts -> instagram_business_account
        pages_count = 0
        page_names: list[str] = []
        try:
            pages_resp = self._request(
                "GET",
                f"{base}/me/accounts",
                params={
                    "fields": "id,name,access_token,instagram_business_account{id,username,name,followers_count,media_count}",
                    "access_token": access_token,
                },
            )
            pages = pages_resp.get("data") or []
            pages_count = len(pages)
            page_names = [str(p.get("name") or p.get("id")) for p in pages]
            linked_page = next(
                (p for p in pages if isinstance(p.get("instagram_business_account"), dict)),
                None,
            )
            if linked_page:
                ig_acct = linked_page["instagram_business_account"]
                return {
                    "ig_user_id": str(ig_acct["id"]),
                    "ig_username": ig_acct.get("username"),
                    "access_token": linked_page.get("access_token") or access_token,
                    "page_id": linked_page.get("id"),
                    "page_name": linked_page.get("name"),
                    "source": "me_accounts",
                }
        except Exception:
            pass

        # 4. Granular scopes target_ids via /debug_token (works even when portfolio edges are restricted)
        granted_scopes: list[str] = []
        if self.has_app_credentials():
            try:
                app_token = f"{self._settings.meta_app_id}|{self._settings.meta_app_secret}"
                dbg = self._request(
                    "GET",
                    f"{base}/debug_token",
                    params={"input_token": access_token, "access_token": app_token},
                )
                dbg_data = dbg.get("data") or {}
                granted_scopes = list(dbg_data.get("scopes") or [])
                for gs in dbg_data.get("granular_scopes") or []:
                    target_ids = gs.get("target_ids") or []
                    if target_ids and gs.get("scope", "").startswith("instagram"):
                        ig_id = str(target_ids[0])
                        ig_username = None
                        try:
                            ig_info = self._request(
                                "GET",
                                f"{base}/{ig_id}",
                                params={"fields": "id,username", "access_token": access_token},
                            )
                            ig_username = ig_info.get("username")
                        except Exception:
                            pass
                        return {
                            "ig_user_id": ig_id,
                            "ig_username": ig_username,
                            "access_token": access_token,
                            "source": "debug_token_granular_scopes",
                        }
            except Exception:
                pass

        raise RuntimeError(
            "Meta Login succeeded, but no Instagram Professional account was returned "
            f"(pages_found={pages_count} {page_names}, granted_scopes={granted_scopes}). "
            "Ensure you selected your Instagram Professional account in the Meta Login popup "
            "(for Unified Login, set META_CONFIG_ID and assign the IG account in the Business Portfolio)."
        )

    def _discover_ig_user_id(self, access_token: str) -> str:
        if self.auth_mode == "instagram":
            me = self._request(
                "GET",
                f"{self.api_base}/me",
                params={"fields": "user_id,id,username", "access_token": access_token},
            )
            uid = str(me.get("user_id") or me.get("id") or "").strip()
            if not uid:
                raise RuntimeError("Could not determine Instagram user_id from /me.")
            return uid

        resolved = self._resolve_facebook_or_unified_ig_account(access_token)
        return str(resolved["ig_user_id"])

    def create_auth_url(self, redirect_uri: str, config_id: str | None = None) -> tuple[str, str]:
        """Build the Meta/Instagram OAuth 2.0 authorization URL and track CSRF state."""
        if not self.has_app_credentials():
            raise RuntimeError(
                "Meta app credentials missing. Set META_APP_ID and META_APP_SECRET."
            )

        state = secrets.token_urlsafe(24)
        effective_config_id = (config_id or self._settings.meta_config_id or "").strip()
        mode = self.auth_mode
        if effective_config_id and mode != "instagram":
            mode = "unified"

        if mode == "instagram":
            base = "https://www.instagram.com/oauth/authorize"
            params = {
                "client_id": self._settings.meta_app_id,
                "redirect_uri": redirect_uri,
                "scope": ",".join(INSTAGRAM_LOGIN_SCOPES),
                "response_type": "code",
                "state": state,
            }
        elif mode == "unified":
            if not effective_config_id:
                raise RuntimeError(
                    "META_CONFIG_ID is required for Instagram API with Unified Login. "
                    "Copy the Config ID from Meta for Developers -> Use cases -> "
                    "Manage messaging & content on Instagram -> API setup with Unified login."
                )
            base = f"https://www.facebook.com/{self.graph_version}/dialog/oauth"
            params = {
                "client_id": self._settings.meta_app_id,
                "redirect_uri": redirect_uri,
                "config_id": effective_config_id,
                "response_type": "code",
                "state": state,
            }
        else:
            base = f"https://www.facebook.com/{self.graph_version}/dialog/oauth"
            params = {
                "client_id": self._settings.meta_app_id,
                "redirect_uri": redirect_uri,
                "scope": ",".join(FACEBOOK_SCOPES),
                "response_type": "code",
                "state": state,
            }

        now = time.monotonic()
        with self._lock:
            stale = [s for s, (_, _, ts) in self._pending_states.items() if now - ts > 900]
            for s in stale:
                self._pending_states.pop(s, None)
            self._pending_states[state] = (redirect_uri, mode, now)

        return f"{base}?{urlencode(params)}", state

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Exchange OAuth code for a 60-day long-lived token and resolve the IG User ID."""
        if not self.has_app_credentials():
            raise RuntimeError(
                "Meta app credentials missing. Set META_APP_ID and META_APP_SECRET."
            )

        effective_redirect = redirect_uri
        mode = self.auth_mode
        if state:
            with self._lock:
                entry = self._pending_states.pop(state, None)
            if entry:
                effective_redirect, mode, _ = entry

        if mode == "instagram":
            short_resp = self._request(
                "POST",
                "https://api.instagram.com/oauth/access_token",
                data={
                    "client_id": self._settings.meta_app_id,
                    "client_secret": self._settings.meta_app_secret,
                    "grant_type": "authorization_code",
                    "redirect_uri": effective_redirect,
                    "code": code,
                },
            )
            short_token = short_resp["access_token"]
            long_resp = self._request(
                "GET",
                "https://graph.instagram.com/access_token",
                params={
                    "grant_type": "ig_exchange_token",
                    "client_secret": self._settings.meta_app_secret,
                    "access_token": short_token,
                },
            )
            long_token = long_resp["access_token"]
            expires_in = long_resp.get("expires_in")
            me = self._request(
                "GET",
                f"https://graph.instagram.com/{self.graph_version}/me",
                params={
                    "fields": "user_id,id,username,name,account_type,followers_count,media_count",
                    "access_token": long_token,
                },
            )
            ig_user_id = str(me.get("user_id") or me.get("id") or short_resp.get("user_id") or "")
            ig_username = me.get("username")
            payload = {
                "auth_mode": "instagram",
                "access_token": long_token,
                "expires_in": expires_in,
                "ig_user_id": ig_user_id,
                "ig_username": ig_username,
                "account_type": me.get("account_type"),
                "obtained_at": int(time.time()),
            }
            self._save_token_data(payload)
            return payload

        # Facebook Login / Unified Login flow
        token_resp = self._request(
            "GET",
            f"https://graph.facebook.com/{self.graph_version}/oauth/access_token",
            params={
                "client_id": self._settings.meta_app_id,
                "client_secret": self._settings.meta_app_secret,
                "redirect_uri": effective_redirect,
                "code": code,
            },
        )
        initial_token = token_resp["access_token"]
        expires_in = token_resp.get("expires_in")
        working_token = initial_token

        # Unified Login returns a Business Integration SUAT directly; for standard user tokens,
        # exchange for a 60-day long-lived user token (ignore cleanly if already a SUAT).
        if mode != "unified":
            try:
                long_resp = self._request(
                    "GET",
                    f"https://graph.facebook.com/{self.graph_version}/oauth/access_token",
                    params={
                        "grant_type": "fb_exchange_token",
                        "client_id": self._settings.meta_app_id,
                        "client_secret": self._settings.meta_app_secret,
                        "fb_exchange_token": initial_token,
                    },
                )
                working_token = long_resp.get("access_token") or initial_token
                expires_in = long_resp.get("expires_in") or expires_in
            except Exception as exc:
                logger.info("Skipping fb_exchange_token (likely already SUAT): %s", exc)

        resolved = self._resolve_facebook_or_unified_ig_account(working_token)
        payload = {
            "auth_mode": mode,
            "access_token": resolved["access_token"],
            "user_access_token": working_token,
            "expires_in": expires_in,
            "page_id": resolved.get("page_id"),
            "page_name": resolved.get("page_name"),
            "business_id": resolved.get("business_id"),
            "ig_user_id": resolved["ig_user_id"],
            "ig_username": resolved.get("ig_username"),
            "discovery_source": resolved.get("source"),
            "obtained_at": int(time.time()),
        }
        self._save_token_data(payload)
        return payload

    def _stage_public_media_url(
        self,
        *,
        media_url: str | None,
        media_base64: str | None,
        default_ext: str,
        public_base_url: str | None,
    ) -> str:
        """Return a publicly reachable HTTPS URL for Meta servers to fetch."""
        src = (media_url or "").strip()
        b64 = (media_base64 or "").strip()
        if src.startswith("data:") and ";base64," in src and not b64:
            header, b64 = src.split(";base64,", 1)
            mime_hint = header.split(":", 1)[1].split(";", 1)[0].strip()
            guessed = mimetypes.guess_extension(mime_hint)
            if guessed:
                default_ext = guessed
            src = ""

        if src and _is_http_url(src):
            return src

        if not self._storage or not public_base_url:
            raise RuntimeError(
                "Instagram Graph API requires a publicly reachable HTTPS `media_url` "
                "(or base64/local upload when hosted at a public HTTPS URL)."
            )

        if b64:
            raw = base64.b64decode(b64)
            ext = default_ext
        elif src:
            path = Path(src).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Media file not found: {path}")
            raw = path.read_bytes()
            ext = path.suffix or default_ext
        else:
            raise ValueError("Provide either media_url (HTTPS URL or local path) or media_base64.")

        if len(raw) > _MAX_STAGED_MEDIA_BYTES:
            raise ValueError(
                f"Media exceeds {_MAX_STAGED_MEDIA_BYTES // (1024 * 1024)} MiB limit."
            )

        if public_base_url.startswith("http://127.0.0.1") or public_base_url.startswith("http://localhost"):
            raise RuntimeError(
                "Meta's servers cannot download media from localhost. Pass a public `https://` "
                "`media_url` when running locally, or invoke the deployed Render MCP server."
            )

        stored = self._storage.save_bytes(raw, extension=ext, filename_prefix="ig_upload")
        return f"{public_base_url}/download/{stored.file_id}/{stored.filename}"

    def _wait_for_container(self, container_id: str, access_token: str, timeout_seconds: float = 90.0) -> None:
        """Poll a Meta media container until status_code is FINISHED."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            status_resp = self._request(
                "GET",
                f"{self.api_base}/{container_id}",
                params={"fields": "status_code,status", "access_token": access_token},
            )
            code = (status_resp.get("status_code") or "").upper()
            if code in {"FINISHED", "PUBLISHED"}:
                return
            if code in {"ERROR", "EXPIRED"}:
                detail = status_resp.get("status") or code
                raise RuntimeError(f"Instagram media container {container_id} failed: {detail}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Instagram container {container_id} (last status: {code or 'unknown'})."
                )
            time.sleep(2.5)

    def upload_content(
        self,
        *,
        media_type: str = "reel",
        media_url: str | None = None,
        media_base64: str | None = None,
        caption: str = "",
        share_to_feed: bool = True,
        cover_url: str | None = None,
        thumb_offset_ms: int | None = None,
        location_id: str | None = None,
        carousel_items: list[dict[str, str] | str] | None = None,
        is_ai_generated: bool | None = None,
        public_base_url: str | None = None,
    ) -> dict[str, Any]:
        """Publish an Instagram Reel, Image, Story, or Carousel via Graph API v27.0."""
        access_token, ig_user_id = self._require_auth()
        kind = (media_type or "reel").strip().lower()
        valid_kinds = {"reel", "reels", "video", "image", "photo", "story_image", "story_video", "carousel"}
        if kind not in valid_kinds:
            raise ValueError(
                "media_type must be one of: 'reel', 'image', 'story_image', 'story_video', 'carousel'."
            )

        if kind == "carousel":
            if not carousel_items or len(carousel_items) < 2:
                raise ValueError("Carousel uploads require at least 2 items in `carousel_items`.")
            if len(carousel_items) > 10:
                raise ValueError("Instagram carousels support at most 10 items.")

            child_ids: list[str] = []
            for item in carousel_items:
                if isinstance(item, str):
                    item_url = item
                    ext = Path(urlparse(item_url).path).suffix.lower()
                    item_kind = "video" if ext in {".mp4", ".mov", ".m4v"} else "image"
                else:
                    item_url = item.get("media_url") or item.get("url") or ""
                    item_kind = (item.get("media_type") or "image").strip().lower()

                staged_url = self._stage_public_media_url(
                    media_url=item_url,
                    media_base64=None,
                    default_ext=".mp4" if item_kind in {"video", "reel"} else ".jpg",
                    public_base_url=public_base_url,
                )
                child_payload: dict[str, Any] = {
                    "is_carousel_item": "true",
                    "access_token": access_token,
                }
                if item_kind in {"video", "reel"}:
                    child_payload["media_type"] = "VIDEO"
                    child_payload["video_url"] = staged_url
                else:
                    child_payload["image_url"] = staged_url

                child_resp = self._request(
                    "POST", f"{self.api_base}/{ig_user_id}/media", data=child_payload
                )
                child_id = child_resp["id"]
                self._wait_for_container(child_id, access_token)
                child_ids.append(child_id)

            parent_data: dict[str, Any] = {
                "media_type": "CAROUSEL",
                "children": ",".join(child_ids),
                "caption": (caption or "").strip()[:2200],
                "access_token": access_token,
            }
            if location_id and location_id.strip():
                parent_data["location_id"] = location_id.strip()
            if is_ai_generated is not None:
                parent_data["is_ai_generated"] = "true" if is_ai_generated else "false"

            container = self._request(
                "POST", f"{self.api_base}/{ig_user_id}/media", data=parent_data
            )
            creation_id = container["id"]
            self._wait_for_container(creation_id, access_token)

        else:
            is_video = kind in {"reel", "reels", "video", "story_video"}
            staged_url = self._stage_public_media_url(
                media_url=media_url,
                media_base64=media_base64,
                default_ext=".mp4" if is_video else ".jpg",
                public_base_url=public_base_url,
            )

            container_data: dict[str, Any] = {
                "access_token": access_token,
            }
            if caption and caption.strip() and kind not in {"story_image", "story_video"}:
                container_data["caption"] = caption.strip()[:2200]
            if location_id and location_id.strip():
                container_data["location_id"] = location_id.strip()
            if is_ai_generated is not None:
                container_data["is_ai_generated"] = "true" if is_ai_generated else "false"

            if kind in {"reel", "reels", "video"}:
                container_data["media_type"] = "REELS"
                container_data["video_url"] = staged_url
                container_data["share_to_feed"] = "true" if share_to_feed else "false"
                if cover_url and cover_url.strip():
                    container_data["cover_url"] = cover_url.strip()
                if thumb_offset_ms is not None:
                    container_data["thumb_offset"] = str(thumb_offset_ms)
            elif kind in {"story_image", "story_video"}:
                container_data["media_type"] = "STORIES"
                if kind == "story_video":
                    container_data["video_url"] = staged_url
                else:
                    container_data["image_url"] = staged_url
            else:
                container_data["image_url"] = staged_url

            container = self._request(
                "POST", f"{self.api_base}/{ig_user_id}/media", data=container_data
            )
            creation_id = container["id"]
            self._wait_for_container(creation_id, access_token)

        publish_resp = self._request(
            "POST",
            f"{self.api_base}/{ig_user_id}/media_publish",
            data={"creation_id": creation_id, "access_token": access_token},
        )
        media_id = publish_resp["id"]

        details: dict[str, Any] = {}
        try:
            details = self._request(
                "GET",
                f"{self.api_base}/{media_id}",
                params={
                    "fields": "id,permalink,shortcode,media_type,media_product_type,timestamp",
                    "access_token": access_token,
                },
            )
        except Exception:
            pass

        return {
            "status": "published",
            "media_id": media_id,
            "creation_id": creation_id,
            "ig_user_id": ig_user_id,
            "media_type": details.get("media_type", kind.upper()),
            "media_product_type": details.get("media_product_type"),
            "permalink": details.get("permalink"),
            "shortcode": details.get("shortcode"),
            "timestamp": details.get("timestamp"),
        }

    def get_stats(
        self,
        *,
        media_id: str | None = None,
        max_posts: int = 10,
    ) -> dict[str, Any]:
        """Read Instagram account/post stats, likes, comments count, and insights."""
        access_token, ig_user_id = self._require_auth()

        if media_id and media_id.strip():
            mid = media_id.strip()
            media = self._request(
                "GET",
                f"{self.api_base}/{mid}",
                params={
                    "fields": "id,caption,media_type,media_product_type,permalink,timestamp,like_count,comments_count",
                    "access_token": access_token,
                },
            )
            insights, insights_note = self._fetch_media_insights(mid, access_token)
            return {
                "scope": "media",
                "media": {
                    "media_id": media.get("id", mid),
                    "caption": media.get("caption"),
                    "media_type": media.get("media_type"),
                    "media_product_type": media.get("media_product_type"),
                    "permalink": media.get("permalink"),
                    "timestamp": media.get("timestamp"),
                    "like_count": int(media.get("like_count", 0)),
                    "comments_count": int(media.get("comments_count", 0)),
                    "insights": insights,
                    "insights_note": insights_note,
                },
            }

        profile = self._request(
            "GET",
            f"{self.api_base}/{ig_user_id}",
            params={
                "fields": "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url",
                "access_token": access_token,
            },
        )

        safe_limit = max(1, min(max_posts, 50))
        media_resp = self._request(
            "GET",
            f"{self.api_base}/{ig_user_id}/media",
            params={
                "fields": "id,caption,media_type,media_product_type,permalink,timestamp,like_count,comments_count",
                "limit": safe_limit,
                "access_token": access_token,
            },
        )
        recent_posts = [
            {
                "media_id": item.get("id"),
                "caption": item.get("caption"),
                "media_type": item.get("media_type"),
                "media_product_type": item.get("media_product_type"),
                "permalink": item.get("permalink"),
                "timestamp": item.get("timestamp"),
                "like_count": int(item.get("like_count", 0)),
                "comments_count": int(item.get("comments_count", 0)),
            }
            for item in (media_resp.get("data") or [])
        ]

        acct_insights, acct_note = self._fetch_account_insights(ig_user_id, access_token)
        return {
            "scope": "account",
            "account": {
                "ig_user_id": profile.get("id", ig_user_id),
                "username": profile.get("username"),
                "name": profile.get("name"),
                "biography": profile.get("biography"),
                "followers_count": int(profile.get("followers_count", 0)),
                "follows_count": int(profile.get("follows_count", 0)),
                "media_count": int(profile.get("media_count", 0)),
                "profile_picture_url": profile.get("profile_picture_url"),
                "insights_last_day": acct_insights,
                "insights_note": acct_note,
            },
            "recent_posts_count": len(recent_posts),
            "recent_posts": recent_posts,
        }

    def _fetch_media_insights(self, media_id: str, access_token: str) -> tuple[dict[str, Any], str | None]:
        metric_sets = (
            "views,reach,saved,shares,total_interactions",
            "reach,saved,shares,total_interactions",
            "plays,reach,saved,shares",
        )
        last_err: str | None = None
        for metrics in metric_sets:
            try:
                resp = self._request(
                    "GET",
                    f"{self.api_base}/{media_id}/insights",
                    params={"metric": metrics, "access_token": access_token},
                )
                parsed = {}
                for row in resp.get("data") or []:
                    values = row.get("values") or []
                    if values:
                        parsed[row["name"]] = values[0].get("value")
                return parsed, None
            except Exception as exc:
                last_err = str(exc)
        return {}, last_err

    def _fetch_account_insights(self, ig_user_id: str, access_token: str) -> tuple[dict[str, Any], str | None]:
        try:
            resp = self._request(
                "GET",
                f"{self.api_base}/{ig_user_id}/insights",
                params={
                    "metric": "reach,profile_views,accounts_engaged,total_interactions",
                    "period": "day",
                    "metric_type": "total_value",
                    "access_token": access_token,
                },
            )
            parsed = {}
            for row in resp.get("data") or []:
                if "total_value" in row and isinstance(row["total_value"], dict):
                    parsed[row["name"]] = row["total_value"].get("value")
                elif row.get("values"):
                    parsed[row["name"]] = row["values"][-1].get("value")
            return parsed, None
        except Exception as exc:
            return {}, f"Account insights unavailable (often requires 100+ followers): {exc}"

    def get_comments(
        self,
        *,
        media_id: str | None = None,
        max_posts: int = 5,
        limit: int = 50,
        include_replies: bool = True,
    ) -> dict[str, Any]:
        """Read comments (and nested replies) on one Instagram post or across recent posts."""
        access_token, ig_user_id = self._require_auth()
        fields = (
            "id,text,username,timestamp,like_count,hidden,replies{id,text,username,timestamp,like_count}"
            if include_replies
            else "id,text,username,timestamp,like_count,hidden"
        )
        safe_limit = max(1, min(limit, 100))

        if media_id and media_id.strip():
            mid = media_id.strip()
            comments = self._fetch_media_comments(mid, fields, safe_limit, access_token)
            return {
                "scope": "media",
                "media_id": mid,
                "total_comments_returned": len(comments),
                "comments": comments,
            }

        safe_posts = max(1, min(max_posts, 20))
        media_resp = self._request(
            "GET",
            f"{self.api_base}/{ig_user_id}/media",
            params={
                "fields": "id,caption,permalink,timestamp,comments_count",
                "limit": safe_posts,
                "access_token": access_token,
            },
        )
        all_comments: list[dict[str, Any]] = []
        for post in media_resp.get("data") or []:
            pid = post["id"]
            if int(post.get("comments_count", 0)) == 0:
                continue
            post_comments = self._fetch_media_comments(pid, fields, safe_limit, access_token)
            for c in post_comments:
                c["media_id"] = pid
                c["permalink"] = post.get("permalink")
            all_comments.extend(post_comments)

        return {
            "scope": "account_recent_posts",
            "ig_user_id": ig_user_id,
            "posts_checked": len(media_resp.get("data") or []),
            "total_comments_returned": len(all_comments),
            "comments": all_comments,
        }

    def _fetch_media_comments(
        self,
        media_id: str,
        fields: str,
        limit: int,
        access_token: str,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        url: str | None = f"{self.api_base}/{media_id}/comments"
        params: dict[str, Any] | None = {
            "fields": fields,
            "limit": min(50, limit),
            "access_token": access_token,
        }
        while url and len(collected) < limit:
            resp = self._request("GET", url, params=params)
            for item in resp.get("data") or []:
                formatted: dict[str, Any] = {
                    "comment_id": item.get("id"),
                    "username": item.get("username"),
                    "text": item.get("text"),
                    "like_count": int(item.get("like_count", 0)),
                    "hidden": bool(item.get("hidden", False)),
                    "timestamp": item.get("timestamp"),
                }
                if "replies" in item:
                    formatted["replies"] = [
                        {
                            "comment_id": r.get("id"),
                            "username": r.get("username"),
                            "text": r.get("text"),
                            "like_count": int(r.get("like_count", 0)),
                            "timestamp": r.get("timestamp"),
                        }
                        for r in (item.get("replies", {}).get("data") or [])
                    ]
                collected.append(formatted)
                if len(collected) >= limit:
                    break
            url = (resp.get("paging") or {}).get("next")
            params = None  # paging.next already includes query params
        return collected
