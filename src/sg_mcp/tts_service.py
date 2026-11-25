"""Gemini 3.8 Flash TTS service with automatic multi-key failover."""

from __future__ import annotations

import base64
import io
import logging
import wave
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import types

from sg_mcp.config import Settings
from sg_mcp.voices import VoiceCatalog, VoiceInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class GeneratedSpeech:
    """Result of a Gemini TTS synthesis call."""

    wav_bytes: bytes
    mime_type: str
    voice: VoiceInfo
    model: str
    duration_seconds: float | None
    sample_rate_hz: int | None
    api_key_slot: int


def _ensure_wav_bytes(raw_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Ensure audio payload is a valid RIFF/WAV container."""
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return raw_bytes, "audio/wav"

    # If Gemini returns headerless PCM (e.g. audio/L16;rate=24000), wrap it in a WAV container
    sample_rate = 24000
    for token in mime_type.split(";"):
        token = token.strip().lower()
        if token.startswith("rate="):
            try:
                sample_rate = int(token.split("=", 1)[1])
            except ValueError:
                pass

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue(), "audio/wav"


def _inspect_wav(wav_bytes: bytes) -> tuple[float | None, int | None]:
    """Extract duration (seconds) and sample rate (Hz) from WAV bytes."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 2), rate
    except Exception:
        pass
    return None, None


class GeminiTTSService:
    """Synthesizes speech using Gemini TTS models with automatic API-key failover."""

    def __init__(
        self,
        settings: Settings,
        catalog: VoiceCatalog | None = None,
        client_factory: Callable[[str], genai.Client] | None = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog or VoiceCatalog(cache_ttl_seconds=settings.voice_cache_ttl_seconds)
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(api_key: str) -> genai.Client:
        return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    @property
    def default_voice_id(self) -> str:
        return self._settings.default_voice_id

    @property
    def default_model(self) -> str:
        return self._settings.tts_model

    def _run_with_failover(self, operation: Callable[[genai.Client], T]) -> tuple[T, int]:
        if not self._settings.api_keys:
            raise RuntimeError(
                "No Gemini API keys configured. Set GEMINI_API_KEYS or GEMINI_API_KEY in environment."
            )

        errors: list[str] = []
        for idx, api_key in enumerate(self._settings.api_keys, start=1):
            try:
                client = self._client_factory(api_key)
                result = operation(client)
                return result, idx
            except Exception as exc:
                # Never log raw API keys
                logger.warning("Gemini API key slot %d failed: %s", idx, exc)
                errors.append(f"key#{idx}: {type(exc).__name__}: {exc}")

        raise RuntimeError("All configured Gemini API keys failed -> " + " | ".join(errors))

    def _run_with_failover_value(self, operation: Callable[[genai.Client], T]) -> T:
        value, _ = self._run_with_failover(operation)
        return value

    def list_voices(
        self,
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
        return self._catalog.query_voices(
            self._run_with_failover_value,
            gender=gender,
            language_code=language_code,
            pitch=pitch,
            search=search,
            featured_only=featured_only,
            limit=limit,
            offset=offset,
            force_refresh=force_refresh,
        )

    def synthesize(
        self,
        *,
        script: str,
        voice: str | None = None,
        style: str | None = None,
        language_code: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        model: str | None = None,
    ) -> GeneratedSpeech:
        cleaned_script = (script or "").strip()
        if not cleaned_script:
            raise ValueError("Voice script cannot be empty.")

        resolved_voice = self._catalog.resolve_voice(
            voice,
            self._settings.default_voice_id,
            self._run_with_failover_value,
        )

        chosen_model = (model or "").strip() or self._settings.tts_model
        effective_lang = (language_code or "").strip() or None

        speech_metadata = (
            types.SpeechMetadata(style=style.strip()) if style and style.strip() else None
        )
        content = types.Content(
            role="user",
            parts=[types.Part(text=cleaned_script, speech_metadata=speech_metadata)],
        )

        speech_cfg_kwargs: dict[str, Any] = {
            "voice_config": types.VoiceConfig(voice=resolved_voice.id),
        }
        if effective_lang:
            speech_cfg_kwargs["language_code"] = effective_lang

        gen_cfg_kwargs: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": types.SpeechConfig(**speech_cfg_kwargs),
        }
        if temperature is not None:
            gen_cfg_kwargs["temperature"] = temperature
        if seed is not None:
            gen_cfg_kwargs["seed"] = seed

        gen_config = types.GenerateContentConfig(**gen_cfg_kwargs)

        def _call(client: genai.Client) -> Any:
            return client.models.generate_content(
                model=chosen_model,
                contents=[content],
                config=gen_config,
            )

        response, key_slot = self._run_with_failover(_call)

        raw_audio: bytes | None = None
        mime_type = "audio/wav"
        for candidate in getattr(response, "candidates", None) or []:
            content_obj = getattr(candidate, "content", None)
            for part in getattr(content_obj, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    raw_audio = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                    mime_type = getattr(inline, "mime_type", None) or "audio/wav"
                    break
            if raw_audio is not None:
                break

        if not raw_audio:
            raise RuntimeError("Gemini TTS response did not contain any audio data.")

        wav_bytes, final_mime = _ensure_wav_bytes(raw_audio, mime_type)
        duration_seconds, sample_rate_hz = _inspect_wav(wav_bytes)

        return GeneratedSpeech(
            wav_bytes=wav_bytes,
            mime_type=final_mime,
            voice=resolved_voice,
            model=chosen_model,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate_hz,
            api_key_slot=key_slot,
        )
