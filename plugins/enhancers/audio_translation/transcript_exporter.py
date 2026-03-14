"""Transcript exporter for saving translation sessions as SRT or plain text.

Accumulates segments (original + translated text with timestamps) during
a live session and writes them to disk on demand.  Three export formats
are supported:

* **SRT** — standard SubRip subtitle file (translated text only).
* **Dual SRT** — bilingual subtitles with the original on top and
  the translation below each timestamp block.
* **Plain text** — side-by-side original and translation, one segment
  per line, with optional timestamps.

The exporter is format-agnostic regarding the *source* of segments: it
works equally well with ``TranscriptSegment`` objects from the YouTube
transcript fetcher, raw dicts from Whisper STT, or manually constructed
``ExportSegment`` instances.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ExportSegment:
    """A single segment ready for export.

    Attributes
    ----------
    original:
        Source-language text.
    translation:
        Target-language text (may be empty if translation hasn't
        happened yet or the segment was untranslatable).
    start:
        Segment start time in seconds from the beginning of the
        session or video.
    duration:
        Segment duration in seconds.
    source_language:
        ISO 639-1 code of the original text language.
    target_language:
        ISO 639-1 code of the translation language.
    """

    original: str
    translation: str = ""
    start: float = 0.0
    duration: float = 0.0
    source_language: str = ""
    target_language: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


def _format_srt_time(seconds: float) -> str:
    """Convert a float timestamp to SRT time format ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _normalize_segments(
    raw: Sequence[Any],
) -> list[ExportSegment]:
    """Accept multiple input shapes and normalise to ``ExportSegment``.

    Handles:
    * ``ExportSegment`` instances (pass-through).
    * ``TranscriptSegment`` from ``youtube_transcript.py``
      (has ``.text``, ``.start``, ``.duration`` but no translation).
    * Plain ``dict`` with keys ``text``/``original``, ``start``,
      ``duration``, and optionally ``translation``.
    """
    out: list[ExportSegment] = []
    for item in raw:
        if isinstance(item, ExportSegment):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ExportSegment(
                original=item.get("original") or item.get("text", ""),
                translation=item.get("translation", ""),
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
                source_language=item.get("source_language", ""),
                target_language=item.get("target_language", ""),
            ))
        elif hasattr(item, "text") and hasattr(item, "start"):
            out.append(ExportSegment(
                original=item.text,
                translation=getattr(item, "translation", ""),
                start=float(item.start),
                duration=float(item.duration),
            ))
        else:
            logger.warning(
                "[TranscriptExporter] Skipping unrecognised segment type: %s",
                type(item).__name__,
            )
    return out


class TranscriptExporter:
    """Accumulates translation segments and exports them to file.

    Thread-safe: segments can be added from pipeline worker threads
    while the UI thread triggers an export.
    """

    def __init__(self) -> None:
        self._segments: list[ExportSegment] = []
        self._lock = threading.Lock()

    @property
    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    @property
    def segments(self) -> list[ExportSegment]:
        with self._lock:
            return list(self._segments)

    def add_segment(
        self,
        original: str,
        translation: str = "",
        start: float = 0.0,
        duration: float = 0.0,
        source_language: str = "",
        target_language: str = "",
    ) -> None:
        """Append a single segment to the session buffer."""
        seg = ExportSegment(
            original=original,
            translation=translation,
            start=start,
            duration=duration,
            source_language=source_language,
            target_language=target_language,
        )
        with self._lock:
            self._segments.append(seg)

    def add_segments(self, segments: Sequence[Any]) -> None:
        """Append multiple segments (any supported shape) at once."""
        normalised = _normalize_segments(segments)
        with self._lock:
            self._segments.extend(normalised)

    def clear(self) -> None:
        """Discard all accumulated segments."""
        with self._lock:
            self._segments.clear()

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def export_srt(
        self,
        path: str | Path,
        segments: Sequence[Any] | None = None,
        *,
        use_translation: bool = True,
    ) -> Path:
        """Write an SRT subtitle file.

        Parameters
        ----------
        path:
            Destination file path.
        segments:
            Explicit segments to export.  When *None*, the internally
            accumulated segments are used.
        use_translation:
            When *True* (default), write the translated text.  When
            *False*, write the original source text instead.

        Returns
        -------
        Path
            The resolved output path.
        """
        segs = self._resolve_segments(segments)
        path = Path(path)
        lines: list[str] = []

        for idx, seg in enumerate(segs, start=1):
            text = (seg.translation or seg.original) if use_translation else seg.original
            if not text.strip():
                continue
            lines.append(str(idx))
            lines.append(
                f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}"
            )
            lines.append(text.strip())
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "[TranscriptExporter] Exported SRT (%d segments) → %s", len(segs), path
        )
        return path

    def export_dual_srt(
        self,
        path: str | Path,
        segments: Sequence[Any] | None = None,
    ) -> Path:
        """Write a bilingual SRT file (original + translation).

        Each cue block contains the original text on the first line(s)
        and the translation on the line(s) below, visually separated.

        Parameters
        ----------
        path:
            Destination file path.
        segments:
            Explicit segments to export.  When *None*, uses internal
            buffer.

        Returns
        -------
        Path
            The resolved output path.
        """
        segs = self._resolve_segments(segments)
        path = Path(path)
        lines: list[str] = []

        for idx, seg in enumerate(segs, start=1):
            original = seg.original.strip()
            translation = seg.translation.strip()
            if not original and not translation:
                continue
            lines.append(str(idx))
            lines.append(
                f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}"
            )
            if original:
                lines.append(original)
            if translation and translation != original:
                lines.append(translation)
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "[TranscriptExporter] Exported dual SRT (%d segments) → %s",
            len(segs),
            path,
        )
        return path

    def export_text(
        self,
        path: str | Path,
        segments: Sequence[Any] | None = None,
        *,
        include_timestamps: bool = True,
        delimiter: str = " | ",
    ) -> Path:
        """Write a plain-text transcript with original and translation.

        Parameters
        ----------
        path:
            Destination file path.
        segments:
            Explicit segments to export.  When *None*, uses internal
            buffer.
        include_timestamps:
            Prefix each line with ``[HH:MM:SS]``.
        delimiter:
            String placed between original and translation columns.

        Returns
        -------
        Path
            The resolved output path.
        """
        segs = self._resolve_segments(segments)
        path = Path(path)
        lines: list[str] = []

        for seg in segs:
            original = seg.original.strip()
            translation = seg.translation.strip()
            if not original and not translation:
                continue

            parts: list[str] = []

            if include_timestamps:
                ts = _format_srt_time(seg.start).split(",")[0]  # HH:MM:SS
                parts.append(f"[{ts}]")

            if original and translation and translation != original:
                parts.append(f"{original}{delimiter}{translation}")
            else:
                parts.append(original or translation)

            lines.append(" ".join(parts))

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "[TranscriptExporter] Exported text (%d segments) → %s", len(segs), path
        )
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_segments(
        self, segments: Sequence[Any] | None
    ) -> list[ExportSegment]:
        if segments is not None:
            return _normalize_segments(segments)
        with self._lock:
            return list(self._segments)
