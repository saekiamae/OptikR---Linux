"""Volume ducking controller for lowering source audio during TTS playback.

Uses **pycaw** (Windows Core Audio API) to temporarily reduce the volume
of media applications (browsers, players) while the translated speech is
playing, then restore it afterwards.

Supports two strategies:

1. **Per-process** — targets specific processes (Chrome, Firefox, Edge,
   etc.) so only their audio is ducked while other sounds remain at
   normal volume.
2. **System-wide fallback** — adjusts the master endpoint volume when
   per-process control is unavailable.

Thread-safe: designed to be called from the TTS playback thread while
other threads continue capturing/translating.

Gracefully degrades to a no-op when pycaw or comtypes are not installed.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

BROWSER_PROCESS_NAMES: frozenset[str] = frozenset({
    "chrome.exe",
    "firefox.exe",
    "msedge.exe",
    "opera.exe",
    "brave.exe",
    "vivaldi.exe",
    "iexplore.exe",
})

MEDIA_PROCESS_NAMES: frozenset[str] = frozenset({
    "spotify.exe",
    "vlc.exe",
    "wmplayer.exe",
    "mpc-hc.exe",
    "mpc-hc64.exe",
    "foobar2000.exe",
    "musicbee.exe",
    "potplayer.exe",
    "potplayer64.exe",
})

DEFAULT_TARGET_PROCESSES: frozenset[str] = BROWSER_PROCESS_NAMES | MEDIA_PROCESS_NAMES

_pycaw_available: bool | None = None


def _check_pycaw() -> bool:
    """Return whether pycaw and comtypes are importable (cached)."""
    global _pycaw_available
    if _pycaw_available is None:
        try:
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume  # noqa: F401
            _pycaw_available = True
        except ImportError:
            _pycaw_available = False
            logger.info(
                "[VolumeDucker] pycaw not installed — volume ducking disabled. "
                "Install with: pip install pycaw"
            )
    return _pycaw_available


class VolumeDucker:
    """Temporarily duck (lower) the volume of media/browser audio.

    Parameters
    ----------
    duck_level:
        Target volume while ducked, expressed as a percentage 0–100 of the
        original level.  ``20`` means duck *to* 20 % of the original volume.
    target_processes:
        Set of executable names to duck.  When *None*, defaults to common
        browsers and media players.  Pass an empty set to force system-wide
        ducking only.
    fade_steps:
        Number of intermediate volume steps when fading down/up.
        ``1`` means instant switch (no fade).
    fade_interval_ms:
        Milliseconds between fade steps.  Total fade duration is
        approximately ``fade_steps * fade_interval_ms``.
    """

    def __init__(
        self,
        duck_level: int = 20,
        target_processes: frozenset[str] | None = None,
        fade_steps: int = 4,
        fade_interval_ms: int = 25,
    ) -> None:
        self._duck_level = max(0, min(100, duck_level)) / 100.0
        self._target_processes = (
            target_processes if target_processes is not None
            else DEFAULT_TARGET_PROCESSES
        )
        self._fade_steps = max(1, fade_steps)
        self._fade_interval = fade_interval_ms / 1000.0

        self._lock = threading.Lock()
        self._ducked = False
        self._saved_volumes: list[tuple[Any, float]] = []
        self._saved_master: float | None = None
        self._used_master_fallback = False

    @property
    def available(self) -> bool:
        """Whether volume ducking is functional on this system."""
        return _check_pycaw()

    @property
    def is_ducked(self) -> bool:
        return self._ducked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def duck(self, level: int | None = None) -> None:
        """Lower target application volumes to *level* % (0–100).

        If *level* is ``None``, uses the ``duck_level`` from construction.
        Safe to call multiple times — subsequent calls are no-ops while
        already ducked.
        """
        if not _check_pycaw():
            return

        with self._lock:
            if self._ducked:
                return
            target = (max(0, min(100, level)) / 100.0) if level is not None else self._duck_level
            self._do_duck(target)
            self._ducked = True

    def restore(self) -> None:
        """Restore volumes to their pre-duck levels.

        Safe to call even when not currently ducked (no-op).
        """
        if not _check_pycaw():
            return

        with self._lock:
            if not self._ducked:
                return
            self._do_restore()
            self._ducked = False

    @contextmanager
    def ducked(self, level: int | None = None) -> Iterator[None]:
        """Context manager that ducks on entry and restores on exit.

        Usage::

            with ducker.ducked():
                play_tts_audio()
        """
        self.duck(level)
        try:
            yield
        finally:
            self.restore()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_duck(self, target_fraction: float) -> None:
        from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

        self._saved_volumes.clear()
        self._saved_master = None
        self._used_master_fallback = False

        sessions = AudioUtilities.GetAllSessions()
        matched_any = False

        for session in sessions:
            if session.Process is None:
                continue
            proc_name = session.Process.name().lower()
            if proc_name not in self._target_processes:
                continue

            try:
                vol: ISimpleAudioVolume = session._ctl.QueryInterface(ISimpleAudioVolume)
                current = vol.GetMasterVolume()
                self._saved_volumes.append((vol, current))
                ducked_vol = current * target_fraction
                self._fade_to(vol, current, ducked_vol)
                matched_any = True
                logger.debug(
                    "[VolumeDucker] Ducked %s: %.0f%% -> %.0f%%",
                    proc_name, current * 100, ducked_vol * 100,
                )
            except Exception as exc:
                logger.debug("[VolumeDucker] Could not duck %s: %s", proc_name, exc)

        if not matched_any:
            self._duck_master(target_fraction)

    def _duck_master(self, target_fraction: float) -> None:
        """Fallback: duck the system master endpoint volume."""
        try:
            from pycaw.pycaw import AudioUtilities
            from pycaw.pycaw import IAudioEndpointVolume
            from comtypes import CLSCTX_ALL  # type: ignore[import-untyped]

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            endpoint: IAudioEndpointVolume = interface.QueryInterface(IAudioEndpointVolume)

            current = endpoint.GetMasterVolumeLevelScalar()
            self._saved_master = current
            self._used_master_fallback = True
            ducked = current * target_fraction
            self._fade_scalar(endpoint, current, ducked)
            logger.debug(
                "[VolumeDucker] Ducked master volume: %.0f%% -> %.0f%%",
                current * 100, ducked * 100,
            )
        except Exception as exc:
            logger.warning("[VolumeDucker] Master volume duck failed: %s", exc)

    def _do_restore(self) -> None:
        for vol_ctl, original in self._saved_volumes:
            try:
                current = vol_ctl.GetMasterVolume()
                self._fade_to(vol_ctl, current, original)
            except Exception as exc:
                logger.debug("[VolumeDucker] Could not restore session volume: %s", exc)
        self._saved_volumes.clear()

        if self._used_master_fallback and self._saved_master is not None:
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from comtypes import CLSCTX_ALL  # type: ignore[import-untyped]

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = interface.QueryInterface(IAudioEndpointVolume)
                current = endpoint.GetMasterVolumeLevelScalar()
                self._fade_scalar(endpoint, current, self._saved_master)
                logger.debug(
                    "[VolumeDucker] Restored master volume to %.0f%%",
                    self._saved_master * 100,
                )
            except Exception as exc:
                logger.warning("[VolumeDucker] Master volume restore failed: %s", exc)

        self._saved_master = None
        self._used_master_fallback = False

    # ------------------------------------------------------------------
    # Fade helpers
    # ------------------------------------------------------------------

    def _fade_to(self, vol_ctl: Any, start: float, end: float) -> None:
        """Gradually change an ``ISimpleAudioVolume`` from *start* to *end*."""
        if self._fade_steps <= 1 or abs(end - start) < 0.01:
            vol_ctl.SetMasterVolume(max(0.0, min(1.0, end)), None)
            return

        import time
        for i in range(1, self._fade_steps + 1):
            fraction = i / self._fade_steps
            intermediate = start + (end - start) * fraction
            vol_ctl.SetMasterVolume(max(0.0, min(1.0, intermediate)), None)
            if i < self._fade_steps:
                time.sleep(self._fade_interval)

    def _fade_scalar(self, endpoint: Any, start: float, end: float) -> None:
        """Gradually change an ``IAudioEndpointVolume`` scalar level."""
        if self._fade_steps <= 1 or abs(end - start) < 0.01:
            endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, end)), None)
            return

        import time
        for i in range(1, self._fade_steps + 1):
            fraction = i / self._fade_steps
            intermediate = start + (end - start) * fraction
            endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, intermediate)), None)
            if i < self._fade_steps:
                time.sleep(self._fade_interval)
