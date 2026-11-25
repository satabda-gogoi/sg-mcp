"""Ephemeral downloadable audio file storage with TTL and capacity eviction."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_HEX_ID_RE = re.compile(r"^[a-f0-9]{12,32}$")
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class StoredAudio:
    """Metadata for a generated audio file available for HTTP download."""

    file_id: str
    filename: str
    path: Path
    size_bytes: int
    created_at: float
    expires_at: float


def sanitize_filename_prefix(value: str | None, fallback: str = "voice") -> str:
    """Convert an arbitrary string into a URL-safe filename prefix."""
    cleaned = _SAFE_NAME_RE.sub("_", (value or "").strip()).strip("_").lower()
    return (cleaned[:40] or fallback)


class AudioStorage:
    """Thread-safe local disk store for downloadable WAV files with TTL cleanup."""

    def __init__(
        self,
        base_dir: Path,
        ttl_seconds: int = 3600,
        max_files: int = 200,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = max(60, ttl_seconds)
        self._max_files = max(1, max_files)
        self._lock = threading.Lock()
        self._index: dict[str, StoredAudio] = {}

    def save_wav(
        self,
        wav_bytes: bytes,
        *,
        voice_id: str,
        filename_prefix: str | None = None,
    ) -> StoredAudio:
        """Persist WAV bytes and return a StoredAudio descriptor."""
        now = time.time()
        file_id = uuid.uuid4().hex[:16]
        prefix = sanitize_filename_prefix(filename_prefix or voice_id, fallback="tts")
        voice_slug = sanitize_filename_prefix(voice_id, fallback="voice")
        if prefix == voice_slug:
            filename = f"{prefix}_{file_id[:8]}.wav"
        else:
            filename = f"{prefix}_{voice_slug}_{file_id[:8]}.wav"

        file_path = self._base_dir / f"{file_id}.wav"
        file_path.write_bytes(wav_bytes)

        record = StoredAudio(
            file_id=file_id,
            filename=filename,
            path=file_path,
            size_bytes=len(wav_bytes),
            created_at=now,
            expires_at=now + self._ttl_seconds,
        )

        with self._lock:
            self._index[file_id] = record
            self._cleanup_locked(now)

        return record

    def get(self, file_id: str) -> StoredAudio | None:
        """Look up a non-expired audio file by its ID."""
        cleaned_id = (file_id or "").strip().lower()
        if not _HEX_ID_RE.match(cleaned_id):
            return None

        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            record = self._index.get(cleaned_id)
            if record and record.path.is_file():
                return record
            if record:
                self._index.pop(cleaned_id, None)
        return None

    def _cleanup_locked(self, now: float) -> None:
        expired_ids = [fid for fid, rec in self._index.items() if rec.expires_at <= now]
        for fid in expired_ids:
            rec = self._index.pop(fid, None)
            if rec:
                rec.path.unlink(missing_ok=True)

        # Evict oldest entries if over max_files
        if len(self._index) > self._max_files:
            sorted_items = sorted(self._index.items(), key=lambda item: item[1].created_at)
            excess = len(self._index) - self._max_files
            for fid, rec in sorted_items[:excess]:
                self._index.pop(fid, None)
                rec.path.unlink(missing_ok=True)
