"""Rolling PCM audio ring buffer, Int16-to-Float32 conversion, and hybrid Silero/RMS VAD."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit little-endian mono PCM bytes into normalized float32 [-1.0, 1.0]."""
    if not pcm_bytes or len(pcm_bytes) < 2:
        return np.empty(0, dtype=np.float32)
    even_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    pcm_i16 = np.frombuffer(memoryview(pcm_bytes)[:even_len], dtype="<i2")
    return pcm_i16.astype(np.float32) / 32768.0


def compute_rms(samples: np.ndarray) -> float:
    """Calculate root-mean-square energy of a float32 audio signal."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class HybridVADEngine:
    """
    Lightweight Voice Activity Detector combining fast RMS energy gating with Silero VAD.

    Uses an energy pre-filter so dead silence takes ~0.01ms without invoking PyTorch,
    and falls back cleanly to RMS VAD if Silero VAD is disabled or unavailable.
    """

    SILERO_WINDOW_SAMPLES = 512  # Required window size for 16kHz Silero VAD v5/v6

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = settings.vad_rms_threshold,
        speech_prob_threshold: float = settings.vad_speech_prob_threshold,
        use_silero: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.speech_prob_threshold = speech_prob_threshold
        self.use_silero = use_silero

        self._model = None
        self._torch = None
        self._load_attempted = False
        self._lock = threading.Lock()

    def _ensure_silero_loaded(self) -> bool:
        if not self.use_silero:
            return False
        if self._model is not None:
            return True
        if self._load_attempted:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True
            try:
                import torch
                from silero_vad import load_silero_vad

                self._torch = torch
                self._model = load_silero_vad()
                logger.info("Silero VAD model loaded successfully.")
                return True
            except Exception as exc:
                logger.warning("Silero VAD unavailable, falling back to adaptive RMS VAD: %s", exc)
                self._model = None
                return False

    def analyze_chunk(self, samples: np.ndarray) -> tuple[bool, float, float]:
        """
        Evaluate a 16kHz float32 audio chunk.

        Returns:
            (is_speech, rms_energy, speech_probability)
        """
        if samples.size == 0:
            return False, 0.0, 0.0

        rms = compute_rms(samples)
        # Ultra-fast silence gate: skip neural VAD when signal is below noise floor
        if rms < self.rms_threshold * 0.55:
            return False, rms, 0.0

        if self._ensure_silero_loaded() and self._model is not None and self._torch is not None:
            try:
                max_prob = 0.0
                win = self.SILERO_WINDOW_SAMPLES
                with self._lock, self._torch.no_grad():
                    if hasattr(self._model, "reset_states"):
                        self._model.reset_states()
                    total = samples.size
                    if total < win:
                        padded = np.zeros(win, dtype=np.float32)
                        padded[:total] = samples
                        tensor = self._torch.from_numpy(padded)
                        max_prob = float(self._model(tensor, self.sample_rate).item())
                    else:
                        for start in range(0, total - win + 1, win):
                            tensor = self._torch.from_numpy(samples[start : start + win])
                            prob = float(self._model(tensor, self.sample_rate).item())
                            if prob > max_prob:
                                max_prob = prob
                is_speech = max_prob >= self.speech_prob_threshold and rms >= (self.rms_threshold * 0.7)
                return is_speech, rms, max_prob
            except Exception as exc:
                logger.debug("Silero chunk inference error, using RMS fallback: %s", exc)

        # Fallback adaptive RMS energy scoring
        pseudo_prob = min(1.0, rms / max(1e-5, self.rms_threshold * 2.5))
        is_speech = rms >= self.rms_threshold
        return is_speech, rms, pseudo_prob


# Shared VAD singleton
vad_engine = HybridVADEngine()


@dataclass
class AudioBufferDecision:
    """Action directives emitted after ingesting a binary PCM chunk into RollingAudioBuffer."""

    is_speech: bool
    rms: float
    speech_prob: float
    buffer_duration_ms: float
    silence_duration_ms: float
    should_transcribe_partial: bool
    should_commit_clause: bool


class RollingAudioBuffer:
    """
    Per-client rolling 16kHz PCM audio buffer for real-time sliding-window ASR.

    Maintains a 240ms pre-roll ring during silence so leading consonants are preserved,
    accumulates active utterance audio, triggers partial sliding-window evaluations every
    `sliding_window_step_ms`, and signals clause commitment on `> silence_threshold_ms` pause
    or when `max_utterance_seconds` is reached.
    """

    def __init__(
        self,
        sample_rate: int = settings.audio_sample_rate,
        silence_threshold_ms: int = settings.silence_threshold_ms,
        sliding_window_step_ms: int = settings.sliding_window_step_ms,
        max_utterance_seconds: float = settings.max_utterance_seconds,
        pre_roll_ms: int = 240,
        vad: HybridVADEngine | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold_ms = float(silence_threshold_ms)
        self.sliding_window_step_ms = float(sliding_window_step_ms)
        self.max_utterance_samples = int(max_utterance_seconds * sample_rate)
        self.pre_roll_samples = int((pre_roll_ms / 1000.0) * sample_rate)
        self.vad = vad or vad_engine

        self._pre_roll = np.empty(0, dtype=np.float32)
        self._active_chunks: list[np.ndarray] = []
        self._active_sample_count: int = 0

        self.has_active_speech: bool = False
        self.trailing_silence_ms: float = 0.0
        self._ms_since_last_eval: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self._active_sample_count / float(self.sample_rate)) * 1000.0

    def ingest_pcm16(self, pcm_bytes: bytes) -> AudioBufferDecision:
        """Convert incoming Int16 PCM frame, run VAD, update buffer state, and return ASR directives."""
        samples = pcm16_bytes_to_float32(pcm_bytes)
        return self.ingest_float32(samples)

    def ingest_float32(self, samples: np.ndarray) -> AudioBufferDecision:
        if samples.size == 0:
            return AudioBufferDecision(
                is_speech=False,
                rms=0.0,
                speech_prob=0.0,
                buffer_duration_ms=self.duration_ms,
                silence_duration_ms=self.trailing_silence_ms,
                should_transcribe_partial=False,
                should_commit_clause=False,
            )

        chunk_ms = (samples.size / float(self.sample_rate)) * 1000.0
        is_speech, rms, speech_prob = self.vad.analyze_chunk(samples)

        if not self.has_active_speech:
            if not is_speech:
                # Keep rolling pre-roll buffer during silence
                if self.pre_roll_samples > 0:
                    combined = (
                        np.concatenate((self._pre_roll, samples))
                        if self._pre_roll.size
                        else samples.copy()
                    )
                    self._pre_roll = combined[-self.pre_roll_samples :]
                return AudioBufferDecision(
                    is_speech=False,
                    rms=rms,
                    speech_prob=speech_prob,
                    buffer_duration_ms=0.0,
                    silence_duration_ms=self.trailing_silence_ms + chunk_ms,
                    should_transcribe_partial=False,
                    should_commit_clause=False,
                )

            # Speech just started: seed active buffer with pre-roll + current chunk
            self.has_active_speech = True
            self.trailing_silence_ms = 0.0
            self._ms_since_last_eval = chunk_ms
            if self._pre_roll.size > 0:
                self._active_chunks.append(self._pre_roll)
                self._active_sample_count += int(self._pre_roll.size)
                self._pre_roll = np.empty(0, dtype=np.float32)
            self._append_active(samples)
        else:
            # Currently inside an active speech utterance
            self._append_active(samples)
            if is_speech:
                self.trailing_silence_ms = 0.0
                self._ms_since_last_eval += chunk_ms
            else:
                self.trailing_silence_ms += chunk_ms
                self._ms_since_last_eval += chunk_ms

        buf_ms = self.duration_ms
        should_commit = (
            self.has_active_speech
            and (
                self.trailing_silence_ms >= self.silence_threshold_ms
                or self._active_sample_count >= self.max_utterance_samples
            )
        )
        should_partial = (
            self.has_active_speech
            and not should_commit
            and buf_ms >= 320.0
            and self._ms_since_last_eval >= self.sliding_window_step_ms
        )
        if should_partial:
            self._ms_since_last_eval = 0.0

        return AudioBufferDecision(
            is_speech=is_speech,
            rms=rms,
            speech_prob=speech_prob,
            buffer_duration_ms=buf_ms,
            silence_duration_ms=self.trailing_silence_ms,
            should_transcribe_partial=should_partial,
            should_commit_clause=should_commit,
        )

    def _append_active(self, samples: np.ndarray) -> None:
        self._active_chunks.append(samples.copy())
        self._active_sample_count += int(samples.size)
        # Hard safety cap (keep newest max_utterance_samples * 1.25 if overrun)
        hard_cap = int(self.max_utterance_samples * 1.25)
        if self._active_sample_count > hard_cap:
            merged = self.get_audio_copy()
            trimmed = merged[-self.max_utterance_samples :]
            self._active_chunks = [trimmed]
            self._active_sample_count = int(trimmed.size)

    def get_audio_copy(self) -> np.ndarray:
        """Return a contiguous float32 array of the current active utterance buffer."""
        if not self._active_chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._active_chunks) == 1:
            return self._active_chunks[0].copy()
        merged = np.concatenate(self._active_chunks)
        # Compact internal list for faster subsequent reads
        self._active_chunks = [merged]
        self._active_sample_count = int(merged.size)
        return merged.copy()

    def pop_utterance_audio(self) -> np.ndarray:
        """Extract current active utterance audio and reset buffer for the next clause."""
        audio = self.get_audio_copy()
        self.reset_utterance()
        return audio

    def trim_committed_prefix(self, keep_tail_ms: int = 180) -> None:
        """Reset active utterance while optionally preserving a short tail context."""
        if keep_tail_ms > 0 and self._active_sample_count > 0:
            keep_samples = int((keep_tail_ms / 1000.0) * self.sample_rate)
            merged = self.get_audio_copy()
            self._pre_roll = merged[-min(keep_samples, merged.size) :]
        else:
            self._pre_roll = np.empty(0, dtype=np.float32)
        self._active_chunks.clear()
        self._active_sample_count = 0
        self.has_active_speech = False
        self.trailing_silence_ms = 0.0
        self._ms_since_last_eval = 0.0

    def reset_utterance(self) -> None:
        self._active_chunks.clear()
        self._active_sample_count = 0
        self.has_active_speech = False
        self.trailing_silence_ms = 0.0
        self._ms_since_last_eval = 0.0

    def clear_all(self) -> None:
        self.reset_utterance()
        self._pre_roll = np.empty(0, dtype=np.float32)
