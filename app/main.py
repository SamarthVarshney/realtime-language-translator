"""FastAPI entry point, REST endpoints, and full-duplex WebSocket simultaneous translation session."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.config import (
    STATIC_DIR,
    SUPPORTED_LANGUAGES,
    get_language_spec,
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

OPENAPI_TAGS = [
    {
        "name": "System & Telemetry",
        "description": "Live pipeline health, LRU cache hit/miss counters, Faster-Whisper ASR hardware status, and VAD timing parameters.",
    },
    {
        "name": "Language Registry",
        "description": "Multilingual catalog mapping 14+ canonical language codes across BCP-47 locales, Whisper ISO codes, and Edge Neural TTS voices.",
    },
    {
        "name": "Translation & Neural Speech",
        "description": "Stateless REST endpoints for low-latency text translation (with 0ms LRU phrase caching) and in-memory Edge Neural MP3 synthesis.",
    },
]

app = FastAPI(
    title="TR-01 Field Interpreter — Real-Time Simultaneous Language Translator",
    version=__version__,
    summary="Zero-wait full-duplex speech translation & neural TTS engine.",
    description=(
        "Real-time simultaneous speech translation engine powered by **FastAPI**, **Full-Duplex WebSockets** (`/ws/translate`), "
        "**Faster-Whisper (`int8`)**, **Silero VAD**, **0ms LRU Phrase Caching**, and **Edge Neural TTS**."
    ),
    openapi_tags=OPENAPI_TAGS,
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    docs_url=None,
    redoc_url=None,
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


# ==============================================================================
# Pydantic OpenAPI 3.1 Schemas
# ==============================================================================


class TranslatorStatsSchema(BaseModel):
    provider: str = Field(..., description="Active Machine Translation provider tier.", examples=["deep-translator"])
    cache_size: int = Field(..., description="Current number of cached phrase pairs in memory.", examples=[42])
    max_cache_size: int = Field(..., description="Maximum LRU phrase cache capacity.", examples=[1024])
    cache_hits: int = Field(..., description="Cumulative 0ms LRU cache hits.", examples=[128])
    cache_misses: int = Field(..., description="Cumulative upstream provider lookups.", examples=[42])


class WhisperStatusSchema(BaseModel):
    model_size: str = Field(..., description="Configured Faster-Whisper checkpoint size.", examples=["base"])
    device: str = Field(..., description="Resolved inference hardware (`cpu` or `cuda`).", examples=["cpu"])
    compute_type: str = Field(..., description="Quantization precision (`int8` or `float16`).", examples=["int8"])
    loaded: bool = Field(..., description="Whether the Whisper model is currently warm in RAM.", examples=[False])
    loading: bool = Field(..., description="Whether background model initialization is in progress.", examples=[False])
    error: str | None = Field(default=None, description="Last model initialization error, if any.")


class PipelineConfigSchema(BaseModel):
    silence_threshold_ms: int = Field(..., description="Pause duration (ms) that commits an active clause.", examples=[800])
    sliding_window_step_ms: int = Field(..., description="Interval (ms) between partial Faster-Whisper evaluations.", examples=[400])
    translation_debounce_ms: int = Field(..., description="Debounce window (ms) coalescing rapid partial speech tokens.", examples=[150])
    max_clause_words: int = Field(..., description="Word threshold triggering punctuation boundary splitting.", examples=[15])


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service health (`ok`).", examples=["ok"])
    version: str = Field(..., description="Semantic API version.", examples=["1.0.0"])
    translator: TranslatorStatsSchema
    whisper: WhisperStatusSchema
    config: PipelineConfigSchema


class LanguageItemSchema(BaseModel):
    code: str = Field(..., description="Canonical language identifier.", examples=["es"])
    name: str = Field(..., description="English display name.", examples=["Spanish"])
    native_name: str = Field(..., description="Endonym / native language label.", examples=["Español"])
    bcp47: str = Field(..., description="BCP-47 locale tag for browser SpeechRecognition.", examples=["es-ES"])
    tts_voice: str = Field(..., description="Edge Neural TTS voice identifier.", examples=["es-ES-ElviraNeural"])
    allow_as_target: bool = Field(..., description="True when selectable as a translation output language.", examples=[True])


class LanguagesResponse(BaseModel):
    languages: list[LanguageItemSchema]
    default_source: str = Field(default="auto", examples=["auto"])
    default_target: str = Field(default="es", examples=["es"])


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Source phrase or partial speech transcript to translate.",
        examples=["Simultaneous translation streams live while you are still speaking."],
    )
    source_lang: str = Field(
        default="auto",
        description="Canonical source language code (or `auto` for instant script/lexicon detection).",
        examples=["auto"],
    )
    target_lang: str = Field(
        default="es",
        description="Canonical target language code.",
        examples=["es"],
    )
    bidirectional: bool = Field(
        default=False,
        description="Enable two-speaker across-the-table automatic language routing between `source_lang` and `target_lang`.",
        examples=[False],
    )
    active_speaker: str = Field(
        default="auto",
        description="Optional speaker lock (`auto`, `A`, or `B`) when `bidirectional=True`.",
        examples=["auto"],
    )


class TranslateResponse(BaseModel):
    original_text: str = Field(..., examples=["Simultaneous translation streams live while you are still speaking."])
    translated_text: str = Field(..., examples=["La traducción simultánea se transmite en vivo mientras sigues hablando."])
    source_lang: str = Field(..., examples=["en"])
    target_lang: str = Field(..., examples=["es"])
    provider: str = Field(..., examples=["deep-translator"])
    cached: bool = Field(..., description="True if served in 0ms from the in-memory LRU cache.", examples=[False])
    latency_ms: int = Field(..., description="End-to-end translation latency in milliseconds.", examples=[78])
    speaker: str = Field(..., description="Resolved speaker channel (`A` or `B`).", examples=["A"])


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Translated clause to synthesize into spoken neural audio.",
        examples=["¡Hola! Bienvenido a la consola interactiva del intérprete TR-01."],
    )
    target_lang: str = Field(
        default="es",
        description="Canonical language code selecting the neural voice.",
        examples=["es"],
    )


class TTSResponse(BaseModel):
    target_lang: str = Field(..., examples=["es"])
    voice: str = Field(..., description="Resolved Edge Neural TTS voice name.", examples=["es-ES-ElviraNeural"])
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded MP3 audio payload ready for WebAudio or HTML5 playback.",
    )


# ==============================================================================
# HTML Pages & Custom Field-Recorder Documentation Views
# ==============================================================================


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


@app.get("/docs", include_in_schema=False)
async def serve_interactive_docs() -> FileResponse:
    docs_file = STATIC_DIR / "docs.html"
    if not docs_file.exists():
        raise HTTPException(status_code=404, detail="Documentation page docs.html not found.")
    return FileResponse(docs_file)


@app.get("/docs/swagger", include_in_schema=False)
async def serve_themed_swagger_ui() -> HTMLResponse:
    """Custom Field-Recorder Neo-Brutalist themed OpenAPI 3.1 Swagger UI."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TR-01 OpenAPI Swagger Explorer — Field Interpreter</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body {{
      margin: 0;
      background: #F4F3EF;
      color: #141413;
      font-family: 'Space Grotesk', sans-serif;
    }}
    .tr-topbar {{
      position: sticky;
      top: 0;
      z-index: 50;
      background: #D8D5CC;
      border-bottom: 2px solid #141413;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
    }}
    .tr-topbar a {{
      color: #141413;
      text-decoration: none;
      font-weight: 700;
    }}
    .tr-btn {{
      padding: 6px 12px;
      border: 2px solid #141413;
      background: #FAF9F6;
      box-shadow: 2px 2px 0 #141413;
    }}
    .tr-btn-orange {{
      background: #FF4F00;
      color: #FFFFFF !important;
    }}
    .swagger-ui {{ font-family: 'Space Grotesk', sans-serif !important; }}
    .swagger-ui .topbar {{ display: none !important; }}
    .swagger-ui .info .title {{
      font-family: 'Space Grotesk', sans-serif !important;
      color: #141413 !important;
      font-weight: 700 !important;
    }}
    .swagger-ui .info .title small.version-stamp {{
      background: #FF4F00 !important;
      border: 1.5px solid #141413 !important;
      border-radius: 0 !important;
    }}
    .swagger-ui .opblock-tag {{
      border-bottom: 2px solid #141413 !important;
      font-family: 'Space Grotesk', sans-serif !important;
      color: #141413 !important;
    }}
    .swagger-ui .opblock {{
      border: 2px solid #141413 !important;
      border-radius: 0 !important;
      box-shadow: 5px 5px 0 #141413 !important;
      background: #FAF9F6 !important;
      margin-bottom: 18px !important;
    }}
    .swagger-ui .opblock .opblock-summary-method {{
      border-radius: 0 !important;
      border: 1.5px solid #141413 !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-weight: 700 !important;
    }}
    .swagger-ui .opblock.opblock-get .opblock-summary-method {{
      background: #141413 !important;
      color: #F4F3EF !important;
    }}
    .swagger-ui .opblock.opblock-post .opblock-summary-method {{
      background: #FF4F00 !important;
      color: #FFFFFF !important;
    }}
    .swagger-ui .btn {{
      border-radius: 0 !important;
      border: 2px solid #141413 !important;
      box-shadow: 2px 2px 0 #141413 !important;
      font-family: 'JetBrains Mono', monospace !important;
    }}
    .swagger-ui .btn.execute {{
      background: #FF4F00 !important;
      color: #FFFFFF !important;
    }}
    .swagger-ui section.models {{
      border: 2px solid #141413 !important;
      border-radius: 0 !important;
      background: #FAF9F6 !important;
      box-shadow: 5px 5px 0 #141413 !important;
    }}
  </style>
</head>
<body>
  <div class="tr-topbar">
    <div style="display:flex;align-items:center;gap:14px;">
      <a href="/docs" class="tr-btn">← Interactive API &amp; WS Console</a>
      <span>TR-01 OPENAPI 3.1 SWAGGER SPEC (v{__version__})</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <a href="/openapi.json" target="_blank" class="tr-btn">openapi.json ↗</a>
      <a href="/app" class="tr-btn tr-btn-orange">Launch Studio ↗</a>
    </div>
  </div>
  <div id="swagger-ui" style="max-width:1080px;margin:0 auto;padding:16px 20px 60px;"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      deepLinking: true,
      defaultModelsExpandDepth: 1,
      docExpansion: 'list',
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# ==============================================================================
# REST API Endpoints
# ==============================================================================


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System & Telemetry"],
    summary="Inspect live system health, cache telemetry & Whisper status",
)
async def health_check() -> dict[str, Any]:
    """Return real-time operational telemetry, LRU phrase cache hit statistics, Faster-Whisper device status, and VAD timing parameters."""
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


@app.get(
    "/api/languages",
    response_model=LanguagesResponse,
    tags=["Language Registry"],
    summary="List all 14+ supported languages, BCP-47 locales & neural voices",
)
async def list_languages() -> dict[str, Any]:
    """Return the complete multilingual voice and code registry used across ASR, Machine Translation, and Edge Neural TTS."""
    languages = [spec.to_dict() for spec in SUPPORTED_LANGUAGES.values()]
    return {
        "languages": languages,
        "default_source": "auto",
        "default_target": "es",
    }


@app.post(
    "/api/translate",
    response_model=TranslateResponse,
    tags=["Translation & Neural Speech"],
    summary="Translate text instantaneously with 0ms LRU phrase caching",
)
async def translate_text(req: TranslateRequest) -> dict[str, Any]:
    """
    Stateless REST translation endpoint backed by the same high-concurrency `TranslationService`
    used in `/ws/translate`, including `<0.1ms` script/lexicon language detection, 0ms LRU caching,
    and automatic two-speaker Bi-Directional routing.
    """
    result = await translator_service.translate(
        text=req.text,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        bidirectional=req.bidirectional,
        active_speaker=req.active_speaker,
    )
    return {
        "original_text": result.original_text,
        "translated_text": result.translated_text,
        "source_lang": result.source_lang,
        "target_lang": result.target_lang,
        "provider": result.provider,
        "cached": result.cached,
        "latency_ms": result.latency_ms,
        "speaker": result.speaker,
    }


@app.post(
    "/api/tts",
    response_model=TTSResponse,
    tags=["Translation & Neural Speech"],
    summary="Synthesize Edge Neural TTS audio as Base64 MP3",
)
async def generate_tts(req: TTSRequest) -> dict[str, Any]:
    """On-demand neural Text-to-Speech endpoint generating an in-memory Base64 MP3 stream for the requested target language."""
    target = normalize_lang_code(req.target_lang, fallback="es")
    spec = get_language_spec(target, fallback="es")
    audio_b64 = await tts_engine.synthesize_base64(req.text, target)
    return {
        "target_lang": target,
        "voice": spec.tts_voice,
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
