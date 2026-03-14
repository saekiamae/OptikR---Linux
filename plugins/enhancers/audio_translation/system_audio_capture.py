"""WASAPI loopback audio capture for system/desktop audio.

Captures audio output (e.g. browser, media player, game) via Windows
WASAPI loopback and exposes it through the ``read(frames)`` interface
expected by ``AudioCaptureStage``'s injected *audio_source*.

Resamples from the device's native sample rate to 16 kHz mono int16,
which is what Whisper expects downstream.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TARGET_RATE = 16_000
TARGET_CHANNELS = 1

# PyAudio host-API type constant for WASAPI
_WASAPI_HOST_API_TYPE = 13  # paWASAPI


@dataclass
class LoopbackDeviceInfo:
    """Metadata for an available WASAPI loopback (output) device."""
    index: int
    name: str
    default_sample_rate: int
    max_output_channels: int
    host_api_index: int
    is_default: bool = False


def _find_wasapi_host_api(pa: Any) -> int | None:
    """Return the host-API index for WASAPI, or *None* if unavailable."""
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if info.get("type") == _WASAPI_HOST_API_TYPE:
            return i
    return None


def enumerate_loopback_devices() -> list[LoopbackDeviceInfo]:
    """List output devices that support WASAPI loopback capture.

    Each returned device can be passed to ``SystemAudioCapture`` via its
    ``device_index`` parameter.
    """
    try:
        import pyaudio  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pyaudio is not installed -- cannot enumerate loopback devices")
        return []

    pa = pyaudio.PyAudio()
    try:
        wasapi_idx = _find_wasapi_host_api(pa)
        if wasapi_idx is None:
            logger.warning("WASAPI host API not found -- loopback capture unavailable")
            return []

        wasapi_info = pa.get_host_api_info_by_index(wasapi_idx)
        default_output = wasapi_info.get("defaultOutputDevice", -1)

        devices: list[LoopbackDeviceInfo] = []
        for i in range(pa.get_device_count()):
            try:
                dev = pa.get_device_info_by_index(i)
            except Exception:
                continue

            if dev.get("hostApi") != wasapi_idx:
                continue
            if dev.get("maxOutputChannels", 0) < 1:
                continue

            devices.append(LoopbackDeviceInfo(
                index=i,
                name=dev["name"],
                default_sample_rate=int(dev["defaultSampleRate"]),
                max_output_channels=dev["maxOutputChannels"],
                host_api_index=wasapi_idx,
                is_default=(i == default_output),
            ))

        return devices
    finally:
        pa.terminate()


@dataclass
class SystemAudioCapture:
    """Captures desktop audio via WASAPI loopback.

    Implements the *audio_source* interface consumed by
    ``AudioCaptureStage``:

    * ``read(frames)`` -> ``np.ndarray[int16]``  (16 kHz mono)
    * ``cleanup()``

    Parameters
    ----------
    device_index:
        PyAudio device index of the output device to capture.  When
        *None*, the default WASAPI output device is used.
    input_volume:
        Volume multiplier applied to captured audio (1.0 = unity).
    buffer_seconds:
        Maximum seconds of audio to buffer internally before dropping
        frames.
    """

    device_index: int | None = None
    input_volume: float = 1.0
    buffer_seconds: float = 10.0

    _pa: Any = field(default=None, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)
    _queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=800), init=False, repr=False)
    _device_rate: int = field(default=TARGET_RATE, init=False, repr=False)
    _device_channels: int = field(default=1, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _residual: np.ndarray | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the WASAPI loopback stream and begin capturing."""
        if self._started:
            return

        import pyaudio  # type: ignore[import-untyped]

        self._pa = pyaudio.PyAudio()

        wasapi_idx = _find_wasapi_host_api(self._pa)
        if wasapi_idx is None:
            self._pa.terminate()
            self._pa = None
            raise RuntimeError("WASAPI host API not found -- loopback capture unavailable")

        dev_index = self.device_index
        if dev_index is None:
            wasapi_info = self._pa.get_host_api_info_by_index(wasapi_idx)
            dev_index = wasapi_info.get("defaultOutputDevice", -1)
            if dev_index < 0:
                self._pa.terminate()
                self._pa = None
                raise RuntimeError("No default WASAPI output device found")

        dev_info = self._pa.get_device_info_by_index(dev_index)
        self._device_rate = int(dev_info["defaultSampleRate"])
        self._device_channels = max(dev_info.get("maxOutputChannels", 2), 1)
        logger.info(
            "[SystemAudioCapture] Opening loopback on '%s' (%d Hz, %d ch)",
            dev_info["name"], self._device_rate, self._device_channels,
        )

        frames_per_buffer = int(self._device_rate * 0.05)  # 50 ms chunks

        try:
            self._stream = self._pa.open(
                format=self._pa.get_format_from_width(2),  # int16
                channels=self._device_channels,
                rate=self._device_rate,
                input=True,
                input_device_index=dev_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._on_audio,
                as_loopback=True,
            )
        except Exception as exc:
            self._pa.terminate()
            self._pa = None
            raise RuntimeError(f"Failed to open WASAPI loopback stream: {exc}") from exc

        self._started = True
        logger.info("[SystemAudioCapture] Loopback capture started")

    def cleanup(self) -> None:
        """Stop the stream and release PyAudio resources."""
        self._started = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self._residual = None
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # PyAudio callback (runs on the audio I/O thread)
    # ------------------------------------------------------------------

    def _on_audio(
        self, in_data: bytes, frame_count: int, time_info: Any, status_flags: int,
    ) -> tuple:
        import pyaudio as _pa  # type: ignore[import-untyped]

        if status_flags:
            logger.debug("[SystemAudioCapture] stream status: %s", status_flags)

        try:
            self._queue.put_nowait(in_data)
        except queue.Full:
            pass  # drop rather than block the audio thread

        return (None, _pa.paContinue)

    # ------------------------------------------------------------------
    # read() -- public interface for AudioCaptureStage
    # ------------------------------------------------------------------

    def read(self, frames: int = 1024) -> np.ndarray:
        """Return up to *frames* samples of 16 kHz mono int16 audio.

        Blocks briefly (up to ~200 ms) waiting for data.  Returns a
        zero-length array when no audio is available rather than raising.

        The method handles:
        1. Collecting raw chunks from the internal queue
        2. Converting multi-channel audio to mono
        3. Resampling from the device's native rate to 16 kHz
        4. Applying the ``input_volume`` gain
        """
        if not self._started:
            self.start()

        raw_chunks: list[bytes] = []
        # Collect available data (at least one chunk, wait briefly)
        try:
            raw_chunks.append(self._queue.get(timeout=0.2))
        except queue.Empty:
            return np.array([], dtype=np.int16)

        while not self._queue.empty():
            try:
                raw_chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break

        audio = np.frombuffer(b"".join(raw_chunks), dtype=np.int16)
        if audio.size == 0:
            return audio

        audio = self._to_mono(audio)
        audio = self._resample(audio)
        audio = self._apply_volume(audio)

        return audio

    # ------------------------------------------------------------------
    # Internal DSP helpers
    # ------------------------------------------------------------------

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Down-mix multi-channel int16 audio to mono by averaging."""
        if self._device_channels <= 1:
            return audio
        # Reshape to (samples, channels) and average across channels
        n_samples = audio.size // self._device_channels
        trimmed = audio[: n_samples * self._device_channels]
        channels = trimmed.reshape(-1, self._device_channels)
        mono = channels.mean(axis=1)
        return mono.astype(np.int16)

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample from device native rate to 16 kHz using scipy."""
        if self._device_rate == TARGET_RATE:
            return audio
        try:
            from scipy.signal import resample_poly
            gcd = np.gcd(TARGET_RATE, self._device_rate)
            up = TARGET_RATE // gcd
            down = self._device_rate // gcd
            resampled = resample_poly(audio.astype(np.float32), up, down)
            return np.clip(resampled, -32768, 32767).astype(np.int16)
        except ImportError:
            # Fallback: simple linear interpolation via numpy
            n_out = int(len(audio) * TARGET_RATE / self._device_rate)
            if n_out == 0:
                return np.array([], dtype=np.int16)
            indices = np.linspace(0, len(audio) - 1, n_out)
            resampled = np.interp(indices, np.arange(len(audio)), audio.astype(np.float64))
            return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        """Scale audio by ``input_volume``."""
        if self.input_volume == 1.0 or audio.size == 0:
            return audio
        scaled = audio.astype(np.float32) * self.input_volume
        return np.clip(scaled, -32768, 32767).astype(np.int16)
