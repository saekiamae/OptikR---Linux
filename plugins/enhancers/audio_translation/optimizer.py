"""
Real-Time Audio Translation Plugin

Live bidirectional audio translation for calls (Zoom, Skype, etc.)

Pipeline Flow:
1. Capture audio from microphone (streaming chunks)
2. Transcribe with Whisper (Speech-to-Text) — streamed segments
3. Translate using OptikR translation engine
4. Generate speech with TTS (Text-to-Speech) — non-blocking
5. Output to virtual/physical audio device

Optimized for low latency (~1-3s end-to-end):
- Parallel pipeline: transcription, translation, and TTS run on separate threads
- Streaming VAD: processes audio as soon as speech ends (no fixed buffer)
- Whisper with fixed source language skips detection overhead
- Non-blocking TTS playback
- Translation cache via Smart Dictionary for repeated phrases
"""

from typing import Any
import os
import threading
import time
import queue
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioTranslationPlugin:
    """
    Real-time bidirectional audio translation for video calls.

    Features:
    - Microphone → Translation → Speaker output
    - Speaker input → Translation → Headphones output
    - Works with Zoom, Skype, Teams, etc.
    - Single device setup (one PC)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)

        # Always initialize core attributes so stop/list_audio_devices
        # don't crash when the plugin was never fully enabled
        self.is_running = False
        self.pyaudio = None
        self.whisper_model = None
        self.tts_engine = None
        self.vad = None
        self.input_stream = None
        self.output_stream = None
        self.translation_engine = None
        self.smart_dict = None
        self.stats = {
            'transcriptions': 0,
            'translations': 0,
            'speeches': 0,
            'errors': 0,
            'dict_hits': 0,
            'dict_misses': 0
        }

        # Pipeline queues — separate stages so they run in parallel
        self.audio_queue = queue.Queue()          # raw audio chunks from mic
        self.transcription_queue = queue.Queue()   # text ready for translation
        self.tts_queue = queue.Queue()             # translated text ready for speech

        # Worker threads
        self._transcription_thread = None
        self._translation_thread = None
        self._tts_thread = None

        # TTS lock — pyttsx3 is not thread-safe
        self._tts_lock = threading.Lock()
        self._tts_type = None
        self._whisper_device = 'cpu'
        self._use_fp16 = False
        self._selected_voice_id = None  # user-chosen voice
        self._voice_reference_file = None  # for Coqui voice cloning

        self._system_capture = None
        self._system_capture_thread = None
        self._youtube_source = None
        self._youtube_feeder_thread = None
        self._volume_ducker = None
        self._initialized = False

        if not self.enabled:
            return

        self._apply_config(config)

        logger.info("[AUDIO_TRANSLATION] Plugin initialized")
        logger.info(f"[AUDIO_TRANSLATION] Mode: {'Bidirectional' if self.bidirectional else 'Unidirectional'}")
        logger.info(f"[AUDIO_TRANSLATION] {self.source_language} ↔ {self.target_language}")

    def _apply_config(self, config: dict[str, Any]):
        """Re-apply config values (used when config changes after construction)."""
        self.config = config
        self.enabled = config.get('enabled', False)
        self.input_device = config.get('input_device', None)
        self.output_device = config.get('output_device', None)
        self.source_language = config.get('source_language', 'en')
        self.target_language = config.get('target_language', 'ja')
        self.bidirectional = config.get('bidirectional', True)
        self.whisper_model_size = config.get('whisper_model', 'base')
        self.use_gpu = config.get('use_gpu', True)
        self.vad_enabled = config.get('vad_enabled', True)
        self.vad_sensitivity = config.get('vad_sensitivity', 2)
        self.sample_rate = 16000
        # How long to accumulate audio before sending to Whisper.
        # Shorter = lower latency but less context for Whisper accuracy.
        # 0.8s is a good balance — catches short phrases quickly.
        self.chunk_duration = config.get('chunk_duration', 0.8)
        # Silence gap that triggers end-of-utterance.
        # 0.3s is aggressive but keeps latency tight.
        self.silence_threshold = config.get('silence_threshold', 0.3)
        # Minimum audio length to bother transcribing (skip tiny noise bursts)
        self.min_audio_length = config.get('min_audio_length', 0.4)

        # Voice selection
        self._selected_voice_id = config.get('voice_id', None)
        self._tts_speed = config.get('tts_speed', 170)

        # Volume controls (0-200, where 100 = unity gain)
        self.input_volume = config.get('input_volume', 100) / 100.0
        self.output_volume = config.get('output_volume', 100) / 100.0

        # Audio source mode: "microphone", "system", or "youtube"
        self.audio_source_mode = config.get('audio_source_mode', 'microphone')
        self.loopback_device = config.get('loopback_device', None)
        self.youtube_url = config.get('youtube_url', '')
        self.duck_enabled = config.get('duck_enabled', True)
        self.duck_level = config.get('duck_level', 20)
        self.auto_detect_language = config.get('auto_detect_language', False)

    def _init_tts_engine(self):
        """Initialize TTS engine based on selected voice."""
        voice_id = self._selected_voice_id

        # Custom voice (Coqui voice cloning with reference audio)
        if voice_id and voice_id.startswith("custom:"):
            try:
                from plugins.enhancers.audio_translation.voice_manager import get_custom_voices
                voice = next((v for v in get_custom_voices() if v["id"] == voice_id), None)
                if voice and os.path.exists(voice.get("reference_file", "")):
                    from TTS.api import TTS
                    self.tts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
                    self._tts_type = 'coqui_clone'
                    self._voice_reference_file = voice["reference_file"]
                    logger.info(f"[AUDIO_TRANSLATION] TTS: custom voice clone '{voice['name']}'")
                    return
            except Exception as e:
                logger.warning(f"[AUDIO_TRANSLATION] Custom voice failed, falling back: {e}")

        # Voice pack (bundled model)
        if voice_id and voice_id.startswith("pack:"):
            try:
                from plugins.enhancers.audio_translation.voice_manager import get_voice_packs
                pack = next((p for p in get_voice_packs() if p["id"] == voice_id), None)
                if pack:
                    from TTS.api import TTS
                    model_path = os.path.join(pack["path"], pack["manifest"].get("model_file", "model.pth"))
                    config_path = os.path.join(pack["path"], pack["manifest"].get("config_file", "config.json"))
                    self.tts_engine = TTS(model_path=model_path, config_path=config_path)
                    self._tts_type = 'voice_pack'
                    logger.info(f"[AUDIO_TRANSLATION] TTS: voice pack '{pack['name']}'")
                    return
            except Exception as e:
                logger.warning(f"[AUDIO_TRANSLATION] Voice pack failed, falling back: {e}")

        # Coqui neural model (selected by model id)
        if voice_id and not voice_id.startswith("pyttsx3:"):
            try:
                from TTS.api import TTS
                self.tts_engine = TTS(model_name=voice_id)
                self._tts_type = 'coqui'
                logger.info(f"[AUDIO_TRANSLATION] TTS: Coqui model '{voice_id}'")
                return
            except Exception as e:
                logger.warning(f"[AUDIO_TRANSLATION] Coqui model '{voice_id}' failed: {e}")

        # System voice via pyttsx3 (specific voice or default)
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', self._tts_speed)
            if voice_id and voice_id.startswith("pyttsx3:"):
                real_id = voice_id[len("pyttsx3:"):]
                # Map our id back to the actual SAPI voice id
                for v in self.tts_engine.getProperty("voices"):
                    if v.id == real_id or v.name == real_id:
                        self.tts_engine.setProperty("voice", v.id)
                        break
            self._tts_type = 'pyttsx3'
            logger.info("[AUDIO_TRANSLATION] TTS: pyttsx3 system voice")
            return
        except Exception as e:
            logger.warning(f"[AUDIO_TRANSLATION] pyttsx3 not available: {e}")

        # Last resort — Coqui default
        try:
            from TTS.api import TTS
            self.tts_engine = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
            self._tts_type = 'coqui'
            logger.info("[AUDIO_TRANSLATION] TTS: Coqui default fallback")
        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] No TTS engine available: {e}")

    def initialize_components(self):
        """Initialize audio translation components (lazy loading)"""
        if not self.enabled:
            return False

        try:
            logger.info("[AUDIO_TRANSLATION] Initializing components...")

            import pyaudio

            self.pyaudio = pyaudio.PyAudio()

            # Whisper + VAD only needed for mic/system modes
            if self.audio_source_mode != 'youtube':
                import whisper
                import torch

                if self.vad_enabled:
                    try:
                        import webrtcvad
                        self.vad = webrtcvad.Vad(self.vad_sensitivity)
                        logger.info("[AUDIO_TRANSLATION] VAD initialized")
                    except ImportError:
                        logger.info("[AUDIO_TRANSLATION] VAD not available, continuing without it")
                        self.vad_enabled = False

                device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
                self._whisper_device = device
                self._use_fp16 = (device == 'cuda')
                logger.info(f"[AUDIO_TRANSLATION] Loading Whisper '{self.whisper_model_size}' on {device}...")
                self.whisper_model = whisper.load_model(self.whisper_model_size, device=device)
                logger.info("[AUDIO_TRANSLATION] Whisper model loaded")
            else:
                logger.info("[AUDIO_TRANSLATION] YouTube mode — skipping Whisper initialization")

            # TTS — choose engine based on selected voice
            self._init_tts_engine()

            # Smart Dictionary
            try:
                from app.text_translation.smart_dictionary import SmartDictionary
                self.smart_dict = SmartDictionary()
                logger.info("[AUDIO_TRANSLATION] Smart Dictionary initialized")
            except Exception as e:
                logger.info(f"[AUDIO_TRANSLATION] Smart Dictionary not available: {e}")
                self.smart_dict = None

            # Translation engine
            if not self.translation_engine:
                logger.info("[AUDIO_TRANSLATION] No translation engine pre-set, will use placeholder fallback")
            else:
                logger.info("[AUDIO_TRANSLATION] Translation engine ready")

            logger.info("[AUDIO_TRANSLATION] All components initialized successfully")
            self._initialized = True
            return True

        except ImportError as e:
            logger.error(f"[AUDIO_TRANSLATION] Missing dependencies: {e}")
            logger.error("[AUDIO_TRANSLATION] Install: pip install openai-whisper pyaudio pyttsx3")
            logger.error("[AUDIO_TRANSLATION] Optional: pip install webrtcvad TTS")
            return False
        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        """Start the audio translation pipeline"""
        if self.is_running:
            logger.info("[AUDIO_TRANSLATION] Already running")
            return False

        if not self.enabled:
            self._apply_config(self.config)
            if not self.enabled:
                logger.info("[AUDIO_TRANSLATION] Plugin not enabled")
                return False

        if not self._initialized:
            if not self.initialize_components():
                logger.error("[AUDIO_TRANSLATION] Failed to initialize components")
                return False

        try:
            # Drain any stale data from previous runs
            for q in (self.audio_queue, self.transcription_queue, self.tts_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            audio_mode = self.audio_source_mode

            # Volume ducking (for system/youtube modes)
            if self.duck_enabled and audio_mode != 'microphone':
                try:
                    from plugins.enhancers.audio_translation.volume_ducker import VolumeDucker
                    self._volume_ducker = VolumeDucker(duck_level=self.duck_level)
                    if self._volume_ducker.available:
                        logger.info(
                            "[AUDIO_TRANSLATION] Volume ducking enabled (duck to %d%%)",
                            self.duck_level,
                        )
                    else:
                        self._volume_ducker = None
                except ImportError:
                    logger.info("[AUDIO_TRANSLATION] Volume ducker not available (pycaw not installed)")
                    self._volume_ducker = None

            if audio_mode == 'youtube':
                # YouTube transcript mode — bypass mic + Whisper entirely
                if not self.youtube_url:
                    logger.error("[AUDIO_TRANSLATION] YouTube mode selected but no URL provided")
                    return False

                try:
                    from plugins.enhancers.audio_translation.youtube_transcript import (
                        YouTubeTranscriptSource,
                    )
                    self._youtube_source = YouTubeTranscriptSource(
                        preferred_languages=[self.source_language],
                    )
                    self._youtube_source.fetch(self.youtube_url)
                except Exception as e:
                    logger.error(f"[AUDIO_TRANSLATION] Failed to fetch YouTube transcript: {e}")
                    return False

                self.is_running = True

                # Feed transcript segments straight into transcription_queue
                self._youtube_feeder_thread = self._youtube_source.feed_queue(
                    self.transcription_queue,
                )

                # Only translation + TTS threads needed (no capture / transcription)
                self._translation_thread = threading.Thread(
                    target=self._translation_loop, daemon=True, name="audio-translate",
                )
                self._tts_thread = threading.Thread(
                    target=self._tts_loop, daemon=True, name="audio-tts",
                )
                self._translation_thread.start()
                self._tts_thread.start()

                logger.info("[AUDIO_TRANSLATION] Started in YouTube transcript mode")
                logger.info(f"[AUDIO_TRANSLATION] Video URL: {self.youtube_url}")

            elif audio_mode == 'system':
                # System audio capture via WASAPI loopback
                try:
                    from plugins.enhancers.audio_translation.system_audio_capture import (
                        SystemAudioCapture,
                    )
                    self._system_capture = SystemAudioCapture(
                        device_index=self.loopback_device,
                        input_volume=self.input_volume,
                    )
                    self._system_capture.start()
                except Exception as e:
                    logger.error(f"[AUDIO_TRANSLATION] Failed to start system audio capture: {e}")
                    return False

                self.is_running = True

                self._system_capture_thread = threading.Thread(
                    target=self._system_capture_loop, daemon=True, name="audio-system-capture",
                )
                self._transcription_thread = threading.Thread(
                    target=self._transcription_loop, daemon=True, name="audio-transcribe",
                )
                self._translation_thread = threading.Thread(
                    target=self._translation_loop, daemon=True, name="audio-translate",
                )
                self._tts_thread = threading.Thread(
                    target=self._tts_loop, daemon=True, name="audio-tts",
                )

                self._system_capture_thread.start()
                self._transcription_thread.start()
                self._translation_thread.start()
                self._tts_thread.start()

                logger.info("[AUDIO_TRANSLATION] Started in system audio capture mode")
                logger.info(f"[AUDIO_TRANSLATION] Loopback device: {self.loopback_device or 'default'}")

            else:
                # Microphone mode (original behavior)
                self.input_stream = self.pyaudio.open(
                    format=self.pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device,
                    frames_per_buffer=int(self.sample_rate * 0.05),
                    stream_callback=self._audio_input_callback,
                )

                self.is_running = True

                self._transcription_thread = threading.Thread(
                    target=self._transcription_loop, daemon=True, name="audio-transcribe",
                )
                self._translation_thread = threading.Thread(
                    target=self._translation_loop, daemon=True, name="audio-translate",
                )
                self._tts_thread = threading.Thread(
                    target=self._tts_loop, daemon=True, name="audio-tts",
                )

                self._transcription_thread.start()
                self._translation_thread.start()
                self._tts_thread.start()

                logger.info("[AUDIO_TRANSLATION] Started in microphone mode")
                logger.info(f"[AUDIO_TRANSLATION] Input device: {self.input_device or 'default'}")

            logger.info(f"[AUDIO_TRANSLATION] Output device: {self.output_device or 'default'}")
            return True

        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] Failed to start: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
            return False

    def stop(self):
        """Stop the audio translation pipeline"""
        if not self.is_running:
            return

        logger.info("[AUDIO_TRANSLATION] Stopping...")
        self.is_running = False

        # Stop YouTube feeder
        if self._youtube_source:
            self._youtube_source.stop_playback()

        # Stop system audio capture
        if self._system_capture:
            self._system_capture.cleanup()
            self._system_capture = None

        # Stop mic stream
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None

        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None

        # Unblock worker threads waiting on queues
        for q in (self.audio_queue, self.transcription_queue, self.tts_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        # Wait for all workers
        for t in (
            self._system_capture_thread,
            self._youtube_feeder_thread,
            self._transcription_thread,
            self._translation_thread,
            self._tts_thread,
        ):
            if t and t.is_alive():
                t.join(timeout=2.0)

        self._transcription_thread = None
        self._translation_thread = None
        self._tts_thread = None
        self._system_capture_thread = None
        self._youtube_feeder_thread = None
        self._youtube_source = None
        self._volume_ducker = None

        logger.info("[AUDIO_TRANSLATION] Stopped")

    # =========================================================================
    # Stage 0: Mic capture callback (runs on PyAudio's thread)
    # =========================================================================

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream — just enqueue, never block."""
        if status:
            logger.debug(f"[AUDIO_TRANSLATION] Input status: {status}")

        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Apply input volume scaling
        if self.input_volume != 1.0:
            audio_data = np.clip(
                audio_data.astype(np.float32) * self.input_volume,
                -32768, 32767
            ).astype(np.int16)
            in_data = audio_data.tobytes()

        # VAD filtering — drop silence immediately so it never reaches Whisper
        if self.vad_enabled and self.vad:
            try:
                if not self.vad.is_speech(in_data, self.sample_rate):
                    # Still enqueue a sentinel so the transcription loop
                    # knows time is passing (for silence detection)
                    self.audio_queue.put_nowait(None)
                    return (in_data, self.pyaudio.paContinue)
            except Exception:
                pass

        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # drop frame rather than block the audio thread

        return (in_data, self.pyaudio.paContinue)

    # =========================================================================
    # Stage 0b: System audio capture loop (replaces mic callback)
    # =========================================================================

    def _system_capture_loop(self):
        """Read from SystemAudioCapture and push chunks to audio_queue."""
        logger.info("[AUDIO_TRANSLATION] System audio capture loop started")

        while self.is_running:
            try:
                audio_data = self._system_capture.read(
                    frames=int(self.sample_rate * 0.05),
                )
                if audio_data.size == 0:
                    self.audio_queue.put_nowait(None)
                    continue
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    pass
            except Exception as e:
                if self.is_running:
                    logger.error(f"[AUDIO_TRANSLATION] System capture error: {e}")
                    time.sleep(0.1)

        logger.info("[AUDIO_TRANSLATION] System audio capture loop stopped")

    # =========================================================================
    # Stage 1: Transcription (accumulate speech → Whisper)
    # =========================================================================

    def _transcription_loop(self):
        """Accumulate audio until silence gap, then transcribe with Whisper."""
        audio_buffer = []
        last_speech_time = time.time()
        buffer_duration = 0.0

        logger.info("[AUDIO_TRANSLATION] Transcription loop started")

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.05)

                if chunk is None:
                    # Silence sentinel or shutdown signal
                    if not self.is_running:
                        break
                    # Check if we should flush on silence
                    silence = time.time() - last_speech_time
                    if audio_buffer and silence > self.silence_threshold:
                        self._flush_transcription(audio_buffer, buffer_duration)
                        audio_buffer = []
                        buffer_duration = 0.0
                    continue

                audio_buffer.append(chunk)
                chunk_secs = len(chunk) / self.sample_rate
                buffer_duration += chunk_secs
                last_speech_time = time.time()

                # Also flush if buffer is getting long (max ~4s to avoid
                # Whisper processing very long segments which adds latency)
                if buffer_duration >= 4.0:
                    self._flush_transcription(audio_buffer, buffer_duration)
                    audio_buffer = []
                    buffer_duration = 0.0

            except queue.Empty:
                # No data — check for silence flush
                silence = time.time() - last_speech_time
                if audio_buffer and silence > self.silence_threshold:
                    self._flush_transcription(audio_buffer, buffer_duration)
                    audio_buffer = []
                    buffer_duration = 0.0

        logger.info("[AUDIO_TRANSLATION] Transcription loop stopped")

    def _flush_transcription(self, audio_buffer, buffer_duration):
        """Run Whisper on accumulated audio and push result to translation queue."""
        if buffer_duration < self.min_audio_length:
            return  # too short, likely noise

        try:
            audio_data = np.concatenate(audio_buffer)
            audio_float = audio_data.astype(np.float32) / 32768.0

            t0 = time.time()

            # Auto-detect: let Whisper figure out the language.
            # Otherwise pin the source language so Whisper skips detection (~30% faster).
            if self.auto_detect_language:
                lang_hint = None
            else:
                lang_hint = self.source_language if not self.bidirectional else None

            result = self.whisper_model.transcribe(
                audio_float,
                language=lang_hint,
                fp16=self._use_fp16,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,  # avoid hallucination carry-over
            )

            elapsed = time.time() - t0
            text = result['text'].strip()
            detected_lang = result.get('language', self.source_language)

            if not text:
                return

            self.stats['transcriptions'] += 1
            logger.info(
                f"[AUDIO_TRANSLATION] Transcribed ({detected_lang}, {elapsed:.2f}s): {text[:80]}"
            )

            self.transcription_queue.put((text, detected_lang))

        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] Transcription error: {e}")
            self.stats['errors'] += 1

    # =========================================================================
    # Stage 2: Translation (runs on its own thread, doesn't block STT or TTS)
    # =========================================================================

    def _translation_loop(self):
        """Pick up transcribed text, translate, push to TTS queue."""
        logger.info("[AUDIO_TRANSLATION] Translation loop started")

        while self.is_running:
            try:
                item = self.transcription_queue.get(timeout=0.05)
                if item is None:
                    if not self.is_running:
                        break
                    continue

                text, detected_lang = item

                # Determine target
                if self.bidirectional:
                    target_lang = (
                        self.target_language
                        if detected_lang == self.source_language
                        else self.source_language
                    )
                else:
                    target_lang = self.target_language

                t0 = time.time()
                translated = self._translate_text(text, detected_lang, target_lang)
                elapsed = time.time() - t0

                if translated:
                    self.stats['translations'] += 1
                    logger.info(
                        f"[AUDIO_TRANSLATION] Translated ({target_lang}, {elapsed:.2f}s): {translated[:80]}"
                    )
                    self.tts_queue.put((translated, target_lang))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[AUDIO_TRANSLATION] Translation loop error: {e}")
                self.stats['errors'] += 1

        logger.info("[AUDIO_TRANSLATION] Translation loop stopped")

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Translate text using Smart Dictionary (fast) then engine (fallback)."""
        try:
            # Smart Dictionary first — near-instant for repeated phrases
            if self.smart_dict:
                dict_entry = self.smart_dict.lookup(text, source_lang, target_lang)
                if dict_entry:
                    self.stats['dict_hits'] = self.stats.get('dict_hits', 0) + 1
                    return dict_entry.translation

            self.stats['dict_misses'] = self.stats.get('dict_misses', 0) + 1

            # AI translation engine
            if self.translation_engine:
                result = self.translation_engine.translate_text(
                    text=text,
                    src_lang=source_lang,
                    tgt_lang=target_lang
                )

                translated_text = result.translated_text

                # Learn high-confidence translations for future instant lookup
                if self.smart_dict and hasattr(result, 'confidence') and result.confidence > 0.85:
                    self.smart_dict.learn_from_translation(
                        source_text=text,
                        translation=translated_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=result.confidence
                    )

                return translated_text

            # No engine — placeholder
            logger.warning("[AUDIO_TRANSLATION] No translation engine available")
            return f"[{target_lang.upper()}] {text}"

        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] Translation error: {e}")
            return None

    # =========================================================================
    # Stage 3: TTS playback (own thread so it never blocks transcription)
    # =========================================================================

    def _tts_loop(self):
        """Pick up translated text and speak it without blocking other stages."""
        logger.info("[AUDIO_TRANSLATION] TTS loop started")

        while self.is_running:
            try:
                item = self.tts_queue.get(timeout=0.05)
                if item is None:
                    if not self.is_running:
                        break
                    continue

                text, language = item

                if self._volume_ducker:
                    self._volume_ducker.duck()
                try:
                    self._speak(text, language)
                finally:
                    if self._volume_ducker:
                        self._volume_ducker.restore()

                self.stats['speeches'] += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[AUDIO_TRANSLATION] TTS loop error: {e}")
                self.stats['errors'] += 1

        logger.info("[AUDIO_TRANSLATION] TTS loop stopped")

    def _speak(self, text: str, language: str):
        """Convert text to speech and play."""
        try:
            if self._tts_type == 'pyttsx3':
                with self._tts_lock:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            elif self._tts_type == 'coqui_clone' and self._voice_reference_file:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    self.tts_engine.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        speaker_wav=self._voice_reference_file,
                        language=language
                    )
                    self._play_audio_file(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            else:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    self.tts_engine.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        language=language
                    )
                    self._play_audio_file(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] TTS error: {e}")
            self.stats['errors'] += 1

    def _play_audio_file(self, audio_file: str):
        """Play audio file to output device."""
        try:
            from scipy.io import wavfile

            sample_rate, audio_data = wavfile.read(audio_file)

            # Apply output volume scaling
            if self.output_volume != 1.0:
                audio_data = np.clip(
                    audio_data.astype(np.float32) * self.output_volume,
                    -32768, 32767
                ).astype(np.int16)

            if not self.output_stream:
                self.output_stream = self.pyaudio.open(
                    format=self.pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    output_device_index=self.output_device
                )

            self.output_stream.write(audio_data.tobytes())

        except Exception as e:
            logger.error(f"[AUDIO_TRANSLATION] Audio playback error: {e}")

    # =========================================================================
    # Utility methods
    # =========================================================================

    def list_audio_devices(self):
        """List available audio devices."""
        if not self.pyaudio:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()

        devices = []
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'max_input_channels': info['maxInputChannels'],
                'max_output_channels': info['maxOutputChannels'],
                'default_sample_rate': info['defaultSampleRate']
            })
        return devices

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        total_lookups = self.stats.get('dict_hits', 0) + self.stats.get('dict_misses', 0)
        dict_hit_rate = self.stats.get('dict_hits', 0) / total_lookups if total_lookups > 0 else 0.0

        return {
            'enabled': self.enabled,
            'running': self.is_running,
            'audio_source_mode': getattr(self, 'audio_source_mode', 'microphone'),
            'transcriptions': self.stats['transcriptions'],
            'translations': self.stats['translations'],
            'speeches': self.stats['speeches'],
            'errors': self.stats['errors'],
            'dict_hits': self.stats.get('dict_hits', 0),
            'dict_misses': self.stats.get('dict_misses', 0),
            'dict_hit_rate': dict_hit_rate,
            'source_language': getattr(self, 'source_language', 'en'),
            'target_language': getattr(self, 'target_language', 'ja'),
            'bidirectional': getattr(self, 'bidirectional', True),
            'whisper_model': getattr(self, 'whisper_model_size', 'base'),
            'input_device': getattr(self, 'input_device', None),
            'output_device': getattr(self, 'output_device', None),
            'smart_dict_enabled': self.smart_dict is not None,
            'translation_engine_enabled': self.translation_engine is not None,
            'duck_enabled': getattr(self, 'duck_enabled', False),
            'duck_level': getattr(self, 'duck_level', 20),
            'volume_ducker_active': self._volume_ducker is not None,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'transcriptions': 0,
            'translations': 0,
            'speeches': 0,
            'errors': 0,
            'dict_hits': 0,
            'dict_misses': 0
        }

    def set_translation_engine(self, engine):
        """Set translation engine from main application."""
        self.translation_engine = engine
        logger.info(f"[AUDIO_TRANSLATION] Translation engine set: {engine.engine_name if engine else 'None'}")

    def cleanup(self):
        """Stop the plugin and release all resources."""
        self.stop()
        if self._system_capture:
            self._system_capture.cleanup()
            self._system_capture = None
        if self._youtube_source:
            self._youtube_source.cleanup()
            self._youtube_source = None
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception:
                pass
            self.pyaudio = None



# Plugin interface
def initialize(config: dict[str, Any]):
    """Initialize the audio translation plugin."""
    return AudioTranslationPlugin(config)
