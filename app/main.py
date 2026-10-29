"""FastAPI entry point, REST endpoints, and full-duplex WebSocket simultaneous translation session."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.config import (
    STATIC_DIR,
    SUPPORTED_LANGUAGES,
    normalize_lang_code,
    settings,
)
from app.services.audio_stream import RollingAudioBuffer
from app.services.transcriber import (
    HybridSegmentTracker,
    TranscriptionResult,
    split_clause_at_boundary,
    whisper_transcriber,
)
from app.services.translator import has_translatable_content, translator_service
from app.services.tts_engine import tts_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("realtime_translator")

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Zero-wait real-time simultaneous speech translation over full-duplex WebSockets.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    target_lang: str = Field(default="es")


@app.get("/", include_in_schema=False)
async def serve_landing() -> FileResponse:
    landing_file = STATIC_DIR / "landing.html"
    if landing_file.exists():
        return FileResponse(landing_file)
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend landing.html not found.")
    return FileResponse(index_file)


@app.get("/app", include_in_schema=False)
async def serve_translator_app() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "translator": translator_service.get_stats(),
        "whisper": whisper_transcriber.get_status(),
        "config": {
            "silence_threshold_ms": settings.silence_threshold_ms,
            "sliding_window_step_ms": settings.sliding_window_step_ms,
            "translation_debounce_ms": settings.translation_debounce_ms,
            "max_clause_words": settings.max_clause_words,
        },
    }


@app.get("/api/languages")
async def list_languages() -> dict[str, Any]:
    languages = [spec.to_dict() for spec in SUPPORTED_LANGUAGES.values()]
    return {
        "languages": languages,
        "default_source": "auto",
        "default_target": "es",
    }


@app.post("/api/tts")
async def generate_tts(req: TTSRequest) -> dict[str, Any]:
    """On-demand neural TTS endpoint for replaying historical transcript cards."""
    target = normalize_lang_code(req.target_lang, fallback="es")
    audio_b64 = await tts_engine.synthesize_base64(req.text, target)
    return {
        "target_lang": target,
        "audio_base64": audio_b64,
    }


class TranslationSession:
    """
    Per-WebSocket stateful simultaneous translation pipeline.

    Implements:
      - Provisional vs. Committed utterance state machine
      - Monotonic `sequence_id` + `asyncio.Task` cancellation for superseded partials
      - Debounced live word-by-word translation
      - Silence (>800ms) and 15-word punctuation clause locking
      - Non-blocking parallel clause commits so speech continues uninterrupted during TTS
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self._send_lock = asyncio.Lock()
        self._closed = False

        # Dynamic session configuration
        self.source_lang: str = "auto"
        self.target_lang: str = "es"
        self.tts_enabled: bool = True
        self.engine_mode: str = "hybrid"  # "hybrid" | "whisper"
        self.bidirectional: bool = False
        self.active_speaker: str = "auto"  # "auto" | "A" | "B"
        self.auto_silence_commit: bool = True

        # Utterance & concurrency tracking
        self._utterance_counter: int = 100
        self.current_utterance_id: str = self._next_utterance_id()
        self.sequence_id: int = 0
        self._partial_task: asyncio.Task[None] | None = None
        self._silence_watchdog_task: asyncio.Task[None] | None = None
        self._whisper_partial_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Stream buffers
        self.segment_tracker = HybridSegmentTracker(max_clause_words=settings.max_clause_words)
        self.audio_buffer = RollingAudioBuffer()
        self._active_provisional_text: str = ""
        self._active_detected_lang: str | None = None
        self._has_emitted_in_utterance: bool = False

    def _next_utterance_id(self) -> str:
        self._utterance_counter += 1
        return f"utt_{self._utterance_counter}"

    def _advance_utterance(self) -> str:
        """Snapshot the active utterance_id for committing and open a fresh one immediately."""
        committed_id = self.current_utterance_id
        self.current_utterance_id = self._next_utterance_id()
        self._active_provisional_text = ""
        self._active_detected_lang = None
        self._has_emitted_in_utterance = False
        return committed_id

    def _spawn_background(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def send_json_safe(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            async with self._send_lock:
                if not self._closed:
                    await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._closed = True

    def _cancel_partial_task(self) -> None:
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None

    def _cancel_silence_watchdog(self) -> None:
        if self._silence_watchdog_task is not None and not self._silence_watchdog_task.done():
            self._silence_watchdog_task.cancel()
        self._silence_watchdog_task = None

    async def cleanup(self) -> None:
        self._closed = True
        self._cancel_partial_task()
        self._cancel_silence_watchdog()
        if self._whisper_partial_task and not self._whisper_partial_task.done():
            self._whisper_partial_task.cancel()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        self.audio_buffer.clear_all()

    async def handle_config(self, msg: dict[str, Any]) -> None:
        if "source_lang" in msg:
            raw_src = str(msg["source_lang"]).strip()
            self.source_lang = "auto" if raw_src == "auto" else normalize_lang_code(raw_src, fallback="auto")
        if "target_lang" in msg:
            tgt = normalize_lang_code(str(msg["target_lang"]), fallback="es")
            self.target_lang = "es" if tgt == "auto" else tgt
        if "tts_enabled" in msg:
            self.tts_enabled = bool(msg["tts_enabled"])
        if "bidirectional" in msg:
            self.bidirectional = bool(msg["bidirectional"])
        if "active_speaker" in msg:
            spk = str(msg["active_speaker"]).strip()
            self.active_speaker = spk if spk in ("auto", "A", "B") else "auto"
        if "auto_silence_commit" in msg:
            self.auto_silence_commit = bool(msg["auto_silence_commit"])
        if "engine_mode" in msg:
            mode = str(msg["engine_mode"]).strip().lower()
            if mode in ("hybrid", "whisper"):
                self.engine_mode = mode
                if mode == "whisper":
                    self._spawn_background(self._warmup_whisper_and_notify())

        await self.send_json_safe(
            {
                "status": "config_ack",
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "tts_enabled": self.tts_enabled,
                "engine_mode": self.engine_mode,
                "bidirectional": self.bidirectional,
                "active_speaker": self.active_speaker,
                "whisper": whisper_transcriber.get_status(),
            }
        )

    async def _warmup_whisper_and_notify(self) -> None:
        await self.send_json_safe(
            {"status": "engine_status", "whisper": {**whisper_transcriber.get_status(), "loading": True}}
        )
        try:
            status_info = await whisper_transcriber.warmup()
            await self.send_json_safe({"status": "engine_status", "whisper": status_info})
        except Exception as exc:
            await self.send_json_safe(
                {"status": "engine_status", "whisper": {**whisper_transcriber.get_status(), "error": str(exc)}}
            )

    def _schedule_partial_translation(
        self,
        text: str,
        detected_hint: str | None = None,
        debounce_override_ms: int | None = None,
    ) -> None:
        """Cancel any stale partial task and launch a fresh debounced translation."""
        self.sequence_id += 1
        seq = self.sequence_id
        utt_id = self.current_utterance_id
        self._active_provisional_text = text
        self._active_detected_lang = detected_hint

        self._cancel_partial_task()

        # First token of a fresh utterance skips debounce for ultra-snappy initial response
        if debounce_override_ms is not None:
            delay_s = debounce_override_ms / 1000.0
        elif not self._has_emitted_in_utterance:
            delay_s = 0.02
        else:
            delay_s = settings.translation_debounce_ms / 1000.0

        self._partial_task = asyncio.create_task(
            self._run_partial_translation(
                text=text,
                utt_id=utt_id,
                seq=seq,
                delay_s=delay_s,
                detected_hint=detected_hint,
            )
        )

    async def _run_partial_translation(
        self,
        text: str,
        utt_id: str,
        seq: int,
        delay_s: float,
        detected_hint: str | None = None,
    ) -> None:
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if seq != self.sequence_id or utt_id != self.current_utterance_id:
                return

            t0 = time.perf_counter()
            res = await translator_service.translate(
                text=text,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                bidirectional=self.bidirectional,
                active_speaker=self.active_speaker,
                detected_hint=detected_hint,
            )

            # Double-check monotonicity after async translation finishes
            if seq != self.sequence_id or utt_id != self.current_utterance_id:
                return

            self._has_emitted_in_utterance = True
            total_ms = int((time.perf_counter() - t0) * 1000)
            await self.send_json_safe(
                {
                    "status": "partial",
                    "utterance_id": utt_id,
                    "sequence_id": seq,
                    "source_lang": res.source_lang,
                    "target_lang": res.target_lang,
                    "speaker": res.speaker,
                    "original_text": res.original_text,
                    "translated_text": res.translated_text,
                    "audio_base64": None,
                    "cached": res.cached,
                    "latency_ms": total_ms,
                }
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Partial translation task error: %s", exc)

    def _arm_silence_watchdog(self, segment_id: int | str | None) -> None:
        """Auto-commit provisional hybrid text if speaker pauses > silence_threshold_ms."""
        self._cancel_silence_watchdog()
        if not self.auto_silence_commit:
            return
        # Slight buffer above VAD threshold so browser WebSpeech natural cadence isn't cut mid-word
        timeout_s = max(0.85, (settings.silence_threshold_ms + 150) / 1000.0)
        self._silence_watchdog_task = asyncio.create_task(
            self._silence_watchdog_fire(timeout_s, segment_id)
        )

    async def _silence_watchdog_fire(self, timeout_s: float, segment_id: int | str | None) -> None:
        try:
            await asyncio.sleep(timeout_s)
            pending = self._active_provisional_text.strip()
            if not has_translatable_content(pending):
                return
            self._cancel_partial_task()
            self.sequence_id += 1
            self.segment_tracker.mark_prefix_committed(pending)
            utt_id = self._advance_utterance()
            self._spawn_background(
                self._commit_utterance(
                    text=pending,
                    utt_id=utt_id,
                    detected_hint=self._active_detected_lang,
                )
            )
        except asyncio.CancelledError:
            return

    async def _commit_utterance(
        self,
        text: str,
        utt_id: str,
        detected_hint: str | None = None,
    ) -> None:
        """Finalize translation, synthesize optional Edge-TTS audio, and emit `status: 'final'`."""
        cleaned = text.strip()
        if not has_translatable_content(cleaned):
            return

        t0 = time.perf_counter()
        res = await translator_service.translate(
            text=cleaned,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            bidirectional=self.bidirectional,
            active_speaker=self.active_speaker,
            detected_hint=detected_hint,
        )
        mt_latency_ms = int((time.perf_counter() - t0) * 1000)

        audio_b64: str | None = None
        if self.tts_enabled and has_translatable_content(res.translated_text):
            audio_b64 = await tts_engine.synthesize_base64(
                text=res.translated_text,
                target_lang=res.target_lang,
            )

        await self.send_json_safe(
            {
                "status": "final",
                "utterance_id": utt_id,
                "source_lang": res.source_lang,
                "target_lang": res.target_lang,
                "speaker": res.speaker,
                "original_text": res.original_text,
                "translated_text": res.translated_text,
                "audio_base64": audio_b64,
                "cached": res.cached,
                "latency_ms": mt_latency_ms,
            }
        )

    async def handle_interim_speech(self, msg: dict[str, Any]) -> None:
        """Process live interim or final speech text tokens from Hybrid Browser ASR."""
        raw_text = str(msg.get("text") or "")
        is_final = bool(msg.get("is_final", False))
        segment_id = msg.get("segment_id")

        self._cancel_silence_watchdog()

        uncommitted = self.segment_tracker.extract_uncommitted(raw_text, segment_id=segment_id)

        if not has_translatable_content(uncommitted):
            if is_final:
                self.segment_tracker.reset_segment()
            return

        if is_final:
            self._cancel_partial_task()
            self.sequence_id += 1
            self.segment_tracker.reset_segment()
            utt_id = self._advance_utterance()
            self._spawn_background(self._commit_utterance(text=uncommitted, utt_id=utt_id))
            return

        # Check 15-word punctuation boundary auto-commit while user keeps speaking
        committed_clause, remainder = split_clause_at_boundary(
            uncommitted,
            max_words=settings.max_clause_words,
        )
        if committed_clause:
            self._cancel_partial_task()
            self.sequence_id += 1
            self.segment_tracker.mark_prefix_committed(committed_clause)
            utt_id = self._advance_utterance()
            self._spawn_background(self._commit_utterance(text=committed_clause, utt_id=utt_id))

            if has_translatable_content(remainder):
                self._schedule_partial_translation(remainder)
                self._arm_silence_watchdog(segment_id)
            return

        # Standard provisional update
        if uncommitted != self._active_provisional_text:
            self._schedule_partial_translation(uncommitted)
        self._arm_silence_watchdog(segment_id)

    async def handle_binary_audio(self, pcm_bytes: bytes) -> None:
        """Ingest 16kHz Int16 mono PCM frames, run VAD, and trigger Faster-Whisper sliding-window ASR."""
        decision = self.audio_buffer.ingest_pcm16(pcm_bytes)

        if decision.should_commit_clause:
            audio_f32 = self.audio_buffer.pop_utterance_audio()
            if self._whisper_partial_task and not self._whisper_partial_task.done():
                self._whisper_partial_task.cancel()
            self._cancel_partial_task()
            self.sequence_id += 1
            utt_id = self._advance_utterance()
            self._spawn_background(self._transcribe_and_commit_audio(audio_f32, utt_id))
            return

        if decision.should_transcribe_partial:
            # Coalesce sliding windows: skip launching if previous Whisper window is still evaluating
            if self._whisper_partial_task is None or self._whisper_partial_task.done():
                audio_snapshot = self.audio_buffer.get_audio_copy()
                utt_id = self.current_utterance_id
                self._whisper_partial_task = asyncio.create_task(
                    self._transcribe_partial_audio(audio_snapshot, utt_id)
                )

    async def _transcribe_partial_audio(self, audio_snapshot: Any, utt_id: str) -> None:
        try:
            asr: TranscriptionResult = await whisper_transcriber.transcribe(
                audio_snapshot,
                source_lang=self.source_lang,
                is_final=False,
            )
            if utt_id != self.current_utterance_id or not has_translatable_content(asr.text):
                return

            # Clause locking if continuous speech exceeds 15 words at punctuation
            committed_clause, remainder = split_clause_at_boundary(
                asr.text,
                max_words=settings.max_clause_words,
            )
            if committed_clause and not remainder:
                # Commit current rolling audio buffer cleanly so buffer doesn't grow indefinitely
                self.audio_buffer.trim_committed_prefix(keep_tail_ms=120)
                self._cancel_partial_task()
                self.sequence_id += 1
                locked_id = self._advance_utterance()
                self._spawn_background(
                    self._commit_utterance(
                        text=committed_clause,
                        utt_id=locked_id,
                        detected_hint=asr.detected_lang,
                    )
                )
                return

            if asr.text != self._active_provisional_text:
                # Audio window already took ~300ms, so translate immediately with 0ms extra debounce
                self._schedule_partial_translation(
                    text=asr.text,
                    detected_hint=asr.detected_lang,
                    debounce_override_ms=0,
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Sliding-window Whisper error: %s", exc)

    async def _transcribe_and_commit_audio(self, audio_f32: Any, utt_id: str) -> None:
        try:
            asr: TranscriptionResult = await whisper_transcriber.transcribe(
                audio_f32,
                source_lang=self.source_lang,
                is_final=True,
            )
            if not has_translatable_content(asr.text):
                return
            await self._commit_utterance(
                text=asr.text,
                utt_id=utt_id,
                detected_hint=asr.detected_lang,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Final Whisper commit error: %s", exc)

    async def handle_flush(self) -> None:
        """Commit any pending provisional speech when the user pauses listening."""
        self._cancel_silence_watchdog()
        self._cancel_partial_task()

        if self.audio_buffer.has_active_speech:
            audio_f32 = self.audio_buffer.pop_utterance_audio()
            utt_id = self._advance_utterance()
            self._spawn_background(self._transcribe_and_commit_audio(audio_f32, utt_id))
            return

        pending = self._active_provisional_text.strip()
        if has_translatable_content(pending):
            self.sequence_id += 1
            self.segment_tracker.reset_segment()
            utt_id = self._advance_utterance()
            self._spawn_background(
                self._commit_utterance(
                    text=pending,
                    utt_id=utt_id,
                    detected_hint=self._active_detected_lang,
                )
            )

    async def handle_reset(self) -> None:
        """Clear active provisional state and audio buffers."""
        self._cancel_silence_watchdog()
        self._cancel_partial_task()
        self.sequence_id += 1
        self.segment_tracker.reset_segment()
        self.audio_buffer.clear_all()
        self._advance_utterance()
        await self.send_json_safe({"status": "reset_ack"})


@app.websocket("/ws/translate")
async def websocket_translate_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = TranslationSession(websocket)
    logger.info("WebSocket client connected.")

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            text_data = message.get("text")
            bytes_data = message.get("bytes")

            if text_data is not None:
                try:
                    payload = json.loads(text_data)
                except json.JSONDecodeError:
                    await session.send_json_safe(
                        {"status": "error", "message": "Invalid JSON frame."}
                    )
                    continue

                if not isinstance(payload, dict):
                    continue

                action = str(payload.get("action") or "").strip().lower()
                if action == "config":
                    await session.handle_config(payload)
                elif action == "interim_speech":
                    await session.handle_interim_speech(payload)
                elif action == "flush":
                    await session.handle_flush()
                elif action == "reset":
                    await session.handle_reset()
                elif action == "ping":
                    await session.send_json_safe(
                        {"status": "pong", "ts": payload.get("ts", time.time())}
                    )
                else:
                    await session.send_json_safe(
                        {"status": "error", "message": f"Unknown action: {action}"}
                    )

            elif bytes_data is not None:
                await session.handle_binary_audio(bytes_data)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.warning("WebSocket session terminated with error: %s", exc)
    finally:
        await session.cleanup()
