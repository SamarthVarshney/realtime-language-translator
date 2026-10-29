"""Streaming ASR service supporting Hybrid Browser Interim Stream and Server-Side Faster-Whisper."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from app.config import get_language_spec, normalize_lang_code, settings

logger = logging.getLogger(__name__)

# Clause boundary punctuation across Latin, Devanagari, Arabic, and CJK scripts
_CLAUSE_PUNCT_RE = re.compile(r"[.!?;:,।،؛。？！，、](\s+|$)")
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_TOKEN_RE = re.compile(r"\S+")

# Common short-buffer Whisper hallucinations during low-energy noise
_WHISPER_HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
    "subtitles by the amara.org community",
    "you",
    "...",
    "bye.",
    "mbc 뉴스 이덕영입니다.",
}


def count_effective_words(text: str) -> int:
    """Count whitespace words plus CJK character pairs so CJK scripts trigger clause splitting too."""
    stripped = text.strip()
    if not stripped:
        return 0
    cjk_count = len(_CJK_CHAR_RE.findall(stripped))
    non_cjk = _CJK_CHAR_RE.sub(" ", stripped)
    space_words = len(_WORD_TOKEN_RE.findall(non_cjk))
    return space_words + (cjk_count + 1) // 2


def split_clause_at_boundary(text: str, max_words: int = settings.max_clause_words) -> tuple[str | None, str]:
    """
    Split an active phrase if it exceeds `max_words` at a natural punctuation boundary
    (or hard-split at a safe word boundary if it exceeds `max_words + 10` without punctuation).

    Returns:
        (committed_clause_or_None, remaining_uncommitted_text)
    """
    cleaned = text.strip()
    eff_words = count_effective_words(cleaned)
    if eff_words < max_words:
        return None, cleaned

    # Look for natural punctuation boundary after at least ~60% of max_words
    min_split_words = max(6, int(max_words * 0.6))
    best_split_idx: int | None = None

    for match in _CLAUSE_PUNCT_RE.finditer(cleaned):
        end_idx = match.end()
        prefix = cleaned[:end_idx].strip()
        if count_effective_words(prefix) >= min_split_words:
            best_split_idx = end_idx
            # Prefer the first boundary once we reach >= max_words - 2
            if count_effective_words(prefix) >= max_words - 2:
                break

    if best_split_idx is not None:
        committed = cleaned[:best_split_idx].strip()
        remainder = cleaned[best_split_idx:].strip()
        return committed, remainder

    # Fallback safety valve if speaker speaks > max_words + 10 continuously with zero punctuation
    hard_limit = max_words + 10
    if eff_words >= hard_limit:
        tokens = list(_WORD_TOKEN_RE.finditer(cleaned))
        if len(tokens) >= max_words:
            cut_pos = tokens[max_words - 1].end()
            return cleaned[:cut_pos].strip(), cleaned[cut_pos:].strip()
        # CJK continuous stream without spaces
        if len(cleaned) >= max_words * 2:
            cut_pos = max_words * 2
            return cleaned[:cut_pos].strip(), cleaned[cut_pos:].strip()

    return None, cleaned


class HybridSegmentTracker:
    """
    Tracks incremental browser SpeechRecognition interim segments so mid-segment clause splits
    and silence auto-commits never re-emit already-committed prefix words.
    """

    def __init__(self, max_clause_words: int = settings.max_clause_words) -> None:
        self.max_clause_words = max_clause_words
        self.active_segment_id: int | str | None = None
        self.committed_token_count: int = 0
        self.committed_cjk_chars: int = 0
        self._committed_first_token: str = ""
        self.last_uncommitted_text: str = ""

    def reset_segment(self) -> None:
        self.committed_token_count = 0
        self.committed_cjk_chars = 0
        self._committed_first_token = ""
        self.last_uncommitted_text = ""

    def extract_uncommitted(self, full_text: str, segment_id: int | str | None = None) -> str:
        """Slice away any prefix tokens already locked during the current recognition segment."""
        if segment_id is not None and segment_id != self.active_segment_id:
            self.active_segment_id = segment_id
            self.reset_segment()

        stripped = full_text.strip()
        if not stripped:
            return ""

        if self.committed_token_count == 0 and self.committed_cjk_chars == 0:
            return stripped

        tokens = list(_WORD_TOKEN_RE.finditer(stripped))
        # Detect brand-new utterance when segment_id is omitted
        if segment_id is None:
            first_tok = tokens[0].group(0).lower().strip(".,!?") if tokens else ""
            if len(tokens) < self.committed_token_count or (
                self._committed_first_token and first_tok != self._committed_first_token
            ):
                self.reset_segment()
                return stripped

        # CJK stream without spaces
        if len(tokens) <= 2 and self.committed_cjk_chars > 0:
            if len(stripped) <= self.committed_cjk_chars:
                return ""
            return stripped[self.committed_cjk_chars :].strip()

        if self.committed_token_count > 0:
            if len(tokens) <= self.committed_token_count:
                return ""
            slice_start = tokens[self.committed_token_count].start()
            return stripped[slice_start:].strip()

        return stripped

    def mark_prefix_committed(self, committed_chunk: str) -> None:
        """Advance token/char pointers after locking a prefix of the ongoing browser segment."""
        tokens = list(_WORD_TOKEN_RE.finditer(committed_chunk))
        if tokens and not self._committed_first_token:
            self._committed_first_token = tokens[0].group(0).lower().strip(".,!?")
        if len(tokens) <= 2 and len(_CJK_CHAR_RE.findall(committed_chunk)) >= 6:
            self.committed_cjk_chars += len(committed_chunk.strip())
        else:
            self.committed_token_count += len(tokens)


@dataclass
class TranscriptionResult:
    """Output of a sliding-window or final Faster-Whisper evaluation."""

    text: str
    detected_lang: str
    language_probability: float
    no_speech_prob: float
    duration_ms: int


class FasterWhisperTranscriber:
    """
    Lazy-loaded Faster-Whisper streaming ASR engine optimized for low-latency sliding windows.

    Automatically selects CUDA float16 when available or CPU int8 quantization otherwise,
    and executes inference in a dedicated worker thread so FastAPI's event loop stays responsive.
    """

    def __init__(
        self,
        model_size: str = settings.whisper_model_size,
        device: str = settings.whisper_device,
        compute_type: str = settings.whisper_compute_type,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type

        self.resolved_device: str = "cpu"
        self.resolved_compute_type: str = "int8"
        self._model = None
        self._loading = False
        self._load_error: str | None = None
        self._lock = threading.Lock()
        cpu_workers = max(2, min(6, (os.cpu_count() or 4) // 2))
        self._cpu_threads = cpu_workers
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-asr")

    def _resolve_hardware(self) -> tuple[str, str]:
        device = self.requested_device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        compute_type = self.requested_compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def get_status(self) -> dict[str, object]:
        device, compute = self._resolve_hardware()
        return {
            "model_size": self.model_size,
            "device": self.resolved_device if self._model else device,
            "compute_type": self.resolved_compute_type if self._model else compute,
            "loaded": self._model is not None,
            "loading": self._loading,
            "error": self._load_error,
        }

    def _load_model_sync(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            self._loading = True
            self._load_error = None
            t0 = time.perf_counter()
            try:
                try:
                    import av  # noqa: F401
                except ImportError:
                    import sys
                    import types

                    # Windows Smart App Control / WDAC can block PyAV's bundled FFmpeg DLLs (`av._core`).
                    # Faster-Whisper only uses `av` for file-path decoding, whereas our live pipeline passes
                    # 16kHz float32 `np.ndarray` buffers directly to `model.transcribe()`.
                    av_stub = types.ModuleType("av")
                    av_stub.error = types.SimpleNamespace(InvalidDataError=ValueError)
                    sys.modules["av"] = av_stub
                    logger.info("PyAV blocked by OS policy; enabled direct NumPy PCM mode for Faster-Whisper.")

                from faster_whisper import WhisperModel

                device, compute_type = self._resolve_hardware()
                self.resolved_device = device
                self.resolved_compute_type = compute_type
                logger.info(
                    "Loading Faster-Whisper model '%s' on %s (%s, threads=%d)...",
                    self.model_size,
                    device,
                    compute_type,
                    self._cpu_threads,
                )
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self._cpu_threads,
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                logger.info("Faster-Whisper '%s' ready in %d ms.", self.model_size, elapsed)
                return self._model
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load Faster-Whisper model: %s", exc)
                raise
            finally:
                self._loading = False

    async def warmup(self) -> dict[str, object]:
        """Pre-load the Faster-Whisper model asynchronously in the background worker."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_model_sync)
        return self.get_status()

    def _transcribe_sync(
        self,
        audio_f32: np.ndarray,
        source_lang: str = "auto",
        is_final: bool = False,
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        if audio_f32.size < 3200:  # < 200ms at 16kHz is too short for reliable ASR
            fallback_lang = "en" if source_lang == "auto" else normalize_lang_code(source_lang)
            return TranscriptionResult(
                text="",
                detected_lang=fallback_lang,
                language_probability=0.0,
                no_speech_prob=1.0,
                duration_ms=0,
            )

        model = self._load_model_sync()
        whisper_lang = None
        if source_lang and source_lang != "auto":
            whisper_lang = get_language_spec(source_lang).whisper_code

        segments_iter, info = model.transcribe(
            audio_f32,
            language=whisper_lang,
            task="transcribe",
            beam_size=2 if is_final else 1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=not is_final,
            vad_filter=False,
        )

        texts: list[str] = []
        max_no_speech = 0.0
        for seg in segments_iter:
            no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            if no_speech > max_no_speech:
                max_no_speech = no_speech
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            # Filter obvious silence hallucinations
            if no_speech > 0.72 and seg_text.lower() in _WHISPER_HALLUCINATIONS:
                continue
            if seg_text.lower() in _WHISPER_HALLUCINATIONS and audio_f32.size < 14000:
                continue
            texts.append(seg_text)

        combined = re.sub(r"\s+", " ", " ".join(texts)).strip()
        detected = normalize_lang_code(
            getattr(info, "language", None),
            fallback="en" if source_lang == "auto" else normalize_lang_code(source_lang),
        )
        lang_prob = float(getattr(info, "language_probability", 1.0) or 1.0)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return TranscriptionResult(
            text=combined,
            detected_lang=detected,
            language_probability=lang_prob,
            no_speech_prob=max_no_speech,
            duration_ms=elapsed_ms,
        )

    async def transcribe(
        self,
        audio_f32: np.ndarray,
        source_lang: str = "auto",
        is_final: bool = False,
    ) -> TranscriptionResult:
        """Run Faster-Whisper transcription asynchronously without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio_f32,
            source_lang,
            is_final,
        )


# Shared singleton transcriber
whisper_transcriber = FasterWhisperTranscriber()
