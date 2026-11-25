"""Gemini TTS voice catalog (dynamic API fetch with curated flagship fallback)."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable

from google import genai

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceInfo:
    """Normalized metadata for a Gemini TTS voice."""

    id: str
    display_name: str
    gender: str
    language_code: str
    accent: str
    pitch: str
    persona: str
    context: str
    description: str
    featured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Built-in fallback catalog of the 30 flagship Gemini prebuilt voices (male & female)
# used when offline or before the live 2,150+ voice catalog is fetched.
FLAGSHIP_VOICES: tuple[VoiceInfo, ...] = (
    VoiceInfo("charon", "Charon", "male", "en-US", "General American", "low", "Informative", "Enterprise Agent", "Informative, calm, and professional male voice (Default).", True),
    VoiceInfo("puck", "Puck", "male", "en-US", "General American", "medium", "Upbeat", "Conversational / Edu", "Upbeat, lively, and energetic male voice.", True),
    VoiceInfo("fenrir", "Fenrir", "male", "en-US", "General American", "low", "Excitable", "Content & Media", "Excitable, passionate, and expressive male voice.", True),
    VoiceInfo("orus", "Orus", "male", "en-US", "General American", "low", "Firm", "Enterprise Agent", "Firm, steady, and confident male voice.", True),
    VoiceInfo("enceladus", "Enceladus", "male", "en-US", "General American", "low", "Breathy", "Content & Media", "Breathy, intimate, and expressive male voice.", True),
    VoiceInfo("iapetus", "Iapetus", "male", "en-US", "General American", "medium", "Clear", "Conversational / Edu", "Clear, articulate, and crisp male voice.", True),
    VoiceInfo("umbriel", "Umbriel", "male", "en-US", "General American", "low", "Easy-going", "Conversational / Edu", "Easy-going, relaxed, and natural male voice.", True),
    VoiceInfo("algieba", "Algieba", "male", "en-US", "General American", "low", "Smooth", "Enterprise Agent", "Smooth, relaxed, and polished male voice.", True),
    VoiceInfo("algenib", "Algenib", "male", "en-US", "General American", "low", "Gravelly", "Content & Media", "Gravelly, grounded, and dramatic male voice.", True),
    VoiceInfo("rasalgethi", "Rasalgethi", "male", "en-US", "General American", "medium", "Informative", "Conversational / Edu", "Informative and clear male presenter voice.", True),
    VoiceInfo("alnilam", "Alnilam", "male", "en-US", "General American", "low", "Firm", "Enterprise Agent", "Firm, direct, and commanding male voice.", True),
    VoiceInfo("schedar", "Schedar", "male", "en-US", "General American", "medium", "Even", "Enterprise Agent", "Even, balanced, and steady male voice.", True),
    VoiceInfo("achird", "Achird", "male", "en-US", "General American", "low", "Friendly", "Conversational / Edu", "Friendly, approachable, and warm male voice.", True),
    VoiceInfo("zubenelgenubi", "Zubenelgenubi", "male", "en-US", "General American", "low", "Casual", "Conversational / Edu", "Casual, conversational male voice.", True),
    VoiceInfo("sadachbia", "Sadachbia", "male", "en-US", "General American", "medium", "Lively", "Content & Media", "Lively, vibrant, and upbeat male voice.", True),
    VoiceInfo("sadaltager", "Sadaltager", "male", "en-US", "General American", "low", "Knowledgeable", "Conversational / Edu", "Knowledgeable, reassuring male voice.", True),
    VoiceInfo("kore", "Kore", "female", "en-US", "General American", "medium", "Firm", "Enterprise Agent", "Firm, strong, and confident female voice.", True),
    VoiceInfo("zephyr", "Zephyr", "female", "en-US", "General American", "high", "Bright", "Conversational / Edu", "Bright, clear, and cheerful female voice.", True),
    VoiceInfo("leda", "Leda", "female", "en-US", "General American", "high", "Youthful", "Content & Media", "Youthful, energetic, and expressive female voice.", True),
    VoiceInfo("aoede", "Aoede", "female", "en-US", "General American", "medium", "Breezy", "Content & Media", "Breezy, light, and relaxed female voice.", True),
    VoiceInfo("callirrhoe", "Callirrhoe", "female", "en-US", "General American", "medium", "Easy-going", "Conversational / Edu", "Easy-going, warm, and natural female voice.", True),
    VoiceInfo("autonoe", "Autonoe", "female", "en-US", "General American", "high", "Bright", "Content & Media", "Bright, crisp, and engaging female voice.", True),
    VoiceInfo("despina", "Despina", "female", "en-US", "General American", "medium", "Smooth", "Enterprise Agent", "Smooth, poised, and articulate female voice.", True),
    VoiceInfo("erinome", "Erinome", "female", "en-US", "General American", "medium", "Clear", "Enterprise Agent", "Clear, direct, and helpful female voice.", True),
    VoiceInfo("laomedeia", "Laomedeia", "female", "en-US", "General American", "high", "Upbeat", "Content & Media", "Upbeat, enthusiastic, and lively female voice.", True),
    VoiceInfo("achernar", "Achernar", "female", "en-US", "General American", "high", "Soft", "Content & Media", "Soft, calm, and soothing female voice.", True),
    VoiceInfo("gacrux", "Gacrux", "female", "en-US", "General American", "low", "Mature", "Enterprise Agent", "Mature, authoritative, and warm female voice.", True),
    VoiceInfo("pulcherrima", "Pulcherrima", "female", "en-US", "General American", "medium", "Forward", "Content & Media", "Forward, confident, and expressive female voice.", True),
    VoiceInfo("vindemiatrix", "Vindemiatrix", "female", "en-US", "General American", "medium", "Gentle", "Content & Media", "Gentle, calm, and empathetic female voice.", True),
    VoiceInfo("sulafat", "Sulafat", "female", "en-US", "General American", "medium", "Warm", "Conversational / Edu", "Warm, welcoming, and friendly female voice.", True),
)

FLAGSHIP_IDS: frozenset[str] = frozenset(v.id for v in FLAGSHIP_VOICES)


def _normalize_api_voice(raw: Any) -> VoiceInfo:
    voice_id = (getattr(raw, "id", None) or getattr(raw, "display_name", None) or "").strip()
    display_name = (getattr(raw, "display_name", None) or voice_id).strip()
    gender = (getattr(raw, "gender", None) or "unknown").strip().lower()
    return VoiceInfo(
        id=voice_id,
        display_name=display_name,
        gender=gender,
        language_code=(getattr(raw, "language_code", None) or "").strip(),
        accent=(getattr(raw, "accent", None) or "").strip(),
        pitch=(getattr(raw, "pitch", None) or "").strip().lower(),
        persona=(getattr(raw, "persona", None) or "").strip(),
        context=(getattr(raw, "context", None) or "").strip(),
        description=(getattr(raw, "description", None) or "").strip(),
        featured=voice_id.lower() in FLAGSHIP_IDS,
    )


class VoiceCatalog:
    """Thread-safe TTL cache for Gemini TTS voices with API failover support."""

    def __init__(
        self,
        fetch_fn: Callable[[genai.Client], list[VoiceInfo]] | None = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._fetch_fn = fetch_fn or self._default_fetch_from_api
        self._cache_ttl = cache_ttl_seconds
        self._lock = threading.Lock()
        self._voices: list[VoiceInfo] = list(FLAGSHIP_VOICES)
        self._last_loaded_at: float = 0.0
        self._source: str = "built-in-fallback"

    @staticmethod
    def _default_fetch_from_api(client: genai.Client) -> list[VoiceInfo]:
        collected: list[VoiceInfo] = []
        page_token: str | None = None
        for _ in range(50):
            kwargs: dict[str, Any] = {"page_size": 500}
            if page_token:
                kwargs["page_token"] = page_token
            response = client.voices.list(**kwargs)
            batch = getattr(response, "voices", None) or []
            for item in batch:
                info = _normalize_api_voice(item)
                if info.id:
                    collected.append(info)
            page_token = getattr(response, "next_page_token", None)
            if not page_token:
                break
        return collected

    def ensure_loaded(
        self,
        run_with_failover: Callable[[Callable[[genai.Client], list[VoiceInfo]]], list[VoiceInfo]],
        force_refresh: bool = False,
    ) -> tuple[list[VoiceInfo], str]:
        """Return cached voices, refreshing from the Gemini API if stale."""
        now = time.monotonic()
        with self._lock:
            is_fresh = (
                not force_refresh
                and self._source == "gemini-voices-api"
                and (now - self._last_loaded_at) < self._cache_ttl
            )
            if is_fresh:
                return self._voices, self._source

        try:
            fetched = run_with_failover(self._fetch_fn)
            if fetched:
                # Sort flagship voices first, then alphabetically by id
                fetched.sort(key=lambda v: (not v.featured, v.id.lower()))
                with self._lock:
                    self._voices = fetched
                    self._last_loaded_at = time.monotonic()
                    self._source = "gemini-voices-api"
                    return self._voices, self._source
        except Exception as exc:
            logger.warning("Failed to refresh live Gemini voices list; using fallback catalog: %s", exc)

        with self._lock:
            return self._voices, self._source

    def resolve_voice(
        self,
        voice_input: str | None,
        default_voice_id: str,
        run_with_failover: Callable[[Callable[[genai.Client], list[VoiceInfo]]], list[VoiceInfo]],
    ) -> VoiceInfo:
        """Resolve a requested voice ID or display name, defaulting to default_voice_id."""
        voices, _ = self.ensure_loaded(run_with_failover)
        target = (voice_input or "").strip() or default_voice_id
        target_lower = target.lower()

        # 1. Exact ID match
        for voice in voices:
            if voice.id.lower() == target_lower:
                return voice

        # 2. Exact display_name match (prefer featured flagship voices if multiple share a display_name)
        display_matches = [v for v in voices if v.display_name.lower() == target_lower]
        if display_matches:
            display_matches.sort(key=lambda v: (not v.featured, v.id.lower()))
            return display_matches[0]

        # 3. Partial match on id or display_name
        partial_matches = [
            v
            for v in voices
            if target_lower in v.id.lower() or target_lower in v.display_name.lower()
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if len(partial_matches) > 1:
            suggestions = ", ".join(f"{v.id} ({v.gender}, {v.language_code})" for v in partial_matches[:8])
            raise ValueError(
                f"Voice '{target}' matched multiple voices. Please pick a specific voice ID: {suggestions}"
            )

        sample = ", ".join(v.id for v in FLAGSHIP_VOICES[:10])
        raise ValueError(
            f"Unknown voice '{target}'. Call list_voices to inspect available voices (e.g. {sample})."
        )

    def query_voices(
        self,
        run_with_failover: Callable[[Callable[[genai.Client], list[VoiceInfo]]], list[VoiceInfo]],
        *,
        gender: str | None = None,
        language_code: str | None = None,
        pitch: str | None = None,
        search: str | None = None,
        featured_only: bool = False,
        limit: int = 50,
        offset: int = 0,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Filter and paginate available Gemini voices."""
        voices, source = self.ensure_loaded(run_with_failover, force_refresh=force_refresh)

        gender_filter = (gender or "").strip().lower()
        if gender_filter in {"", "all", "any"}:
            gender_filter = ""

        lang_filter = (language_code or "").strip().lower()
        pitch_filter = (pitch or "").strip().lower()
        search_filter = (search or "").strip().lower()

        filtered: list[VoiceInfo] = []
        for voice in voices:
            if featured_only and not voice.featured:
                continue
            if gender_filter and voice.gender.lower() != gender_filter:
                continue
            if lang_filter and lang_filter not in voice.language_code.lower():
                continue
            if pitch_filter and voice.pitch.lower() != pitch_filter:
                continue
            if search_filter:
                haystack = " ".join(
                    [
                        voice.id,
                        voice.display_name,
                        voice.gender,
                        voice.language_code,
                        voice.accent,
                        voice.persona,
                        voice.context,
                        voice.description,
                    ]
                ).lower()
                if search_filter not in haystack:
                    continue
            filtered.append(voice)

        gender_counts = dict(Counter(v.gender for v in filtered))
        all_gender_counts = dict(Counter(v.gender for v in voices))
        safe_offset = max(0, offset)
        safe_limit = max(1, min(limit, 200))
        page = filtered[safe_offset : safe_offset + safe_limit]

        return {
            "source": source,
            "total_catalog_voices": len(voices),
            "catalog_gender_counts": all_gender_counts,
            "total_matching": len(filtered),
            "matching_gender_counts": gender_counts,
            "offset": safe_offset,
            "limit": safe_limit,
            "has_more": (safe_offset + len(page)) < len(filtered),
            "voices": [v.to_dict() for v in page],
        }
