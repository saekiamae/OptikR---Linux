"""YouTube transcript fetcher and timed segment feeder.

Fetches timed subtitles from a YouTube video URL using the
``youtube-transcript-api`` library and exposes them through two
consumption interfaces:

1. **Pipeline stage** — ``YouTubeTranscriptStage`` replaces both
   ``AudioCaptureStage`` and ``SpeechToTextStage``.  Each ``execute()``
   call blocks until the next segment's timestamp is reached (relative
   to playback start), then returns the segment text as
   ``transcribed_text`` so the downstream ``AudioTextAdapterStage``
   can process it identically to STT output.

2. **Queue feeder** — ``feed_queue()`` pushes ``(text, language)``
   tuples into the ``transcription_queue`` used by ``optimizer.py``
   at the correct wall-clock offsets, bypassing both microphone capture
   and Whisper entirely.

When no transcript is available for a video the caller receives a clear
``NoTranscriptFoundError`` with a human-readable message.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/v/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
]


class NoTranscriptFoundError(Exception):
    """Raised when a YouTube video has no available transcript."""


# ------------------------------------------------------------------
# Segment data
# ------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A single timed subtitle segment from a YouTube transcript."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "duration": self.duration}


# ------------------------------------------------------------------
# Core source
# ------------------------------------------------------------------

@dataclass
class YouTubeTranscriptSource:
    """Fetches and manages a YouTube video's timed transcript.

    Parameters
    ----------
    preferred_languages:
        Ordered list of language codes to prefer when fetching the
        transcript.  The first available language wins.  When *None*,
        defaults to ``["en"]``.
    """

    preferred_languages: list[str] | None = None

    _segments: list[TranscriptSegment] = field(default_factory=list, init=False, repr=False)
    _language: str = field(default="en", init=False, repr=False)
    _video_id: str | None = field(default=None, init=False, repr=False)

    _playback_start: float | None = field(default=None, init=False, repr=False)
    _playback_offset: float = field(default=0.0, init=False, repr=False)
    _segment_index: int = field(default=0, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _feeder_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------
    # URL / video-ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_video_id(url: str) -> str | None:
        """Extract the 11-character YouTube video ID from *url*.

        Returns *None* when the URL doesn't match any known YouTube
        format.
        """
        url = url.strip()
        for pattern in _VIDEO_ID_PATTERNS:
            m = pattern.search(url)
            if m:
                return m.group(1)
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
            return url
        return None

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> list[TranscriptSegment]:
        """Fetch the timed transcript for a YouTube video.

        Parameters
        ----------
        url:
            Full YouTube URL or bare 11-character video ID.

        Returns
        -------
        list[TranscriptSegment]
            Chronologically ordered list of subtitle segments.

        Raises
        ------
        NoTranscriptFoundError
            When no transcript is available for the video in any
            requested language.
        ValueError
            When *url* is not a valid YouTube URL / video ID.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(
                f"Could not extract a YouTube video ID from: {url!r}"
            )
        self._video_id = video_id

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is not installed. "
                "Install with: pip install youtube-transcript-api"
            )

        langs = list(self.preferred_languages or ["en"])

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as exc:
            raise NoTranscriptFoundError(
                f"No transcript found for video {video_id} — "
                f"use System Audio mode instead. (Error: {exc})"
            ) from exc

        transcript = None

        # 1) Try a manually-created transcript in one of the preferred languages
        for lang in langs:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                break
            except Exception:
                continue

        # 2) Fall back to auto-generated transcript
        if transcript is None:
            for lang in langs:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    break
                except Exception:
                    continue

        # 3) Last resort: grab whatever is available
        if transcript is None:
            try:
                for t in transcript_list:
                    transcript = t
                    break
            except Exception:
                pass

        if transcript is None:
            raise NoTranscriptFoundError(
                f"No transcript found for video {video_id} — "
                "use System Audio mode instead."
            )

        raw_segments = transcript.fetch()
        self._language = transcript.language_code
        self._segments = [
            TranscriptSegment(
                text=seg["text"],
                start=float(seg["start"]),
                duration=float(seg["duration"]),
            )
            for seg in raw_segments
        ]
        self._segments.sort(key=lambda s: s.start)
        self._segment_index = 0
        self._playback_start = None

        logger.info(
            "[YouTubeTranscript] Fetched %d segments (%s) for video %s",
            len(self._segments),
            self._language,
            video_id,
        )

        return list(self._segments)

    def get_available_languages(self, url: str) -> list[dict[str, str]]:
        """Return a list of available transcript languages for *url*.

        Each entry is a dict with ``language_code`` and ``language``
        (human-readable name) keys, plus a ``is_generated`` boolean.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(
                f"Could not extract a YouTube video ID from: {url!r}"
            )

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is not installed. "
                "Install with: pip install youtube-transcript-api"
            )

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as exc:
            raise NoTranscriptFoundError(
                f"No transcript found for video {video_id}. (Error: {exc})"
            ) from exc

        languages: list[dict[str, str]] = []
        for t in transcript_list:
            languages.append({
                "language_code": t.language_code,
                "language": t.language,
                "is_generated": t.is_generated,
            })

        return languages

    # ------------------------------------------------------------------
    # Playback-synced segment delivery
    # ------------------------------------------------------------------

    @property
    def segments(self) -> list[TranscriptSegment]:
        return list(self._segments)

    @property
    def language(self) -> str:
        return self._language

    @property
    def is_playing(self) -> bool:
        return self._playback_start is not None and not self._stop_event.is_set()

    def start_playback(self, offset: float = 0.0) -> None:
        """Start the playback clock.

        *offset* lets the caller synchronise with a video that is
        already partway through (in seconds from the video start).
        """
        with self._lock:
            self._playback_offset = offset
            self._playback_start = time.monotonic() - offset
            self._stop_event.clear()

            # Advance index past segments that are already before the offset
            self._segment_index = 0
            for i, seg in enumerate(self._segments):
                if seg.end > offset:
                    self._segment_index = i
                    break
            else:
                self._segment_index = len(self._segments)

        logger.info(
            "[YouTubeTranscript] Playback started (offset=%.1fs, segments remaining=%d)",
            offset,
            len(self._segments) - self._segment_index,
        )

    def stop_playback(self) -> None:
        """Stop the playback clock and any running feeder thread."""
        self._stop_event.set()
        if self._feeder_thread is not None and self._feeder_thread.is_alive():
            self._feeder_thread.join(timeout=3.0)
        self._feeder_thread = None
        self._playback_start = None
        logger.info("[YouTubeTranscript] Playback stopped")

    def _elapsed(self) -> float:
        """Seconds elapsed since playback started."""
        if self._playback_start is None:
            return 0.0
        return time.monotonic() - self._playback_start

    def get_next_segment(self, timeout: float = 5.0) -> TranscriptSegment | None:
        """Block until the next segment's start time is reached.

        Returns *None* when playback has been stopped, all segments
        have been delivered, or *timeout* seconds pass with no segment
        due.  This is designed for the pipeline stage's ``execute()``
        loop.
        """
        if self._playback_start is None:
            self.start_playback()

        deadline = time.monotonic() + timeout

        while not self._stop_event.is_set():
            with self._lock:
                if self._segment_index >= len(self._segments):
                    return None
                seg = self._segments[self._segment_index]

            wait = seg.start - self._elapsed()
            if wait > 0:
                if time.monotonic() + wait > deadline:
                    return None
                self._stop_event.wait(wait)
                if self._stop_event.is_set():
                    return None
            else:
                # Segment's time has arrived (or passed)
                pass

            with self._lock:
                if self._segment_index >= len(self._segments):
                    return None
                seg = self._segments[self._segment_index]
                self._segment_index += 1

            logger.debug(
                "[YouTubeTranscript] Delivering segment %d @ %.1fs: %s",
                self._segment_index - 1,
                seg.start,
                seg.text[:60],
            )
            return seg

        return None

    # ------------------------------------------------------------------
    # Queue feeder (for optimizer.py)
    # ------------------------------------------------------------------

    def feed_queue(
        self,
        transcription_queue: queue.Queue,
        *,
        offset: float = 0.0,
        blocking: bool = False,
    ) -> threading.Thread | None:
        """Feed segments into a ``(text, language)`` transcription queue.

        Spawns a daemon thread that pushes segments at the correct
        wall-clock offsets so the downstream translation + TTS pipeline
        processes them in sync with the video.

        Parameters
        ----------
        transcription_queue:
            The ``transcription_queue`` from ``AudioTranslationPlugin``
            / ``optimizer.py``.
        offset:
            Seconds into the video at which playback is starting.
        blocking:
            When *True*, run synchronously on the calling thread instead
            of spawning a background thread.  Useful for testing.

        Returns
        -------
        threading.Thread | None
            The feeder thread (or *None* when *blocking=True*).
        """
        self.stop_playback()
        self.start_playback(offset=offset)

        def _run() -> None:
            lang = self._language
            logger.info(
                "[YouTubeTranscript] Queue feeder started (%d segments, lang=%s)",
                len(self._segments) - self._segment_index,
                lang,
            )
            while not self._stop_event.is_set():
                seg = self.get_next_segment(timeout=10.0)
                if seg is None:
                    if self._segment_index >= len(self._segments):
                        break
                    continue
                try:
                    transcription_queue.put((seg.text, lang), timeout=5.0)
                except queue.Full:
                    logger.warning("[YouTubeTranscript] Transcription queue full, dropping segment")

            logger.info("[YouTubeTranscript] Queue feeder finished")

        if blocking:
            _run()
            return None

        self._feeder_thread = threading.Thread(
            target=_run, daemon=True, name="yt-transcript-feeder",
        )
        self._feeder_thread.start()
        return self._feeder_thread

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop playback and release resources."""
        self.stop_playback()
        self._segments.clear()
        self._video_id = None


# ------------------------------------------------------------------
# Pipeline stage (replaces AudioCaptureStage + SpeechToTextStage)
# ------------------------------------------------------------------

class YouTubeTranscriptStage:
    """Pipeline stage that delivers YouTube transcript segments as
    ``transcribed_text`` at the correct playback timestamps.

    Replaces both ``AudioCaptureStage`` and ``SpeechToTextStage``
    when the audio source mode is ``"youtube"``.  The downstream
    ``AudioTextAdapterStage`` receives the same data shape it would
    get from Whisper STT, so no further changes are needed.
    """

    name = "youtube_transcript"

    def __init__(
        self,
        source: YouTubeTranscriptSource,
        *,
        auto_start: bool = True,
        playback_offset: float = 0.0,
    ) -> None:
        self._source = source
        self._auto_start = auto_start
        self._playback_offset = playback_offset
        self._started = False

    def execute(self, input_data: dict[str, Any]) -> "StageResult":
        from app.workflow.pipeline.types import StageResult

        start = time.perf_counter()

        if not self._started and self._auto_start:
            self._source.start_playback(offset=self._playback_offset)
            self._started = True

        seg = self._source.get_next_segment(timeout=5.0)

        elapsed = (time.perf_counter() - start) * 1000

        if seg is None:
            all_done = self._source._segment_index >= len(self._source._segments)
            return StageResult(
                success=True,
                data={
                    "transcribed_text": "",
                    "detected_language": self._source.language,
                    "transcript_finished": all_done,
                },
                duration_ms=elapsed,
            )

        return StageResult(
            success=True,
            data={
                "transcribed_text": seg.text,
                "detected_language": self._source.language,
                "segment_start": seg.start,
                "segment_duration": seg.duration,
            },
            duration_ms=elapsed,
        )

    def cleanup(self) -> None:
        self._source.stop_playback()
        self._started = False
