"""Comprehensive automated unit and WebSocket integration tests for LinguaStream Live."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_language_spec, normalize_lang_code
from app.main import app
from app.services.audio_stream import (
    HybridVADEngine,
    RollingAudioBuffer,
    compute_rms,
    pcm16_bytes_to_float32,
)
from app.services.transcriber import (
    HybridSegmentTracker,
    TranscriptionResult,
    split_clause_at_boundary,
    whisper_transcriber,
)
from app.services.translator import (
     BaseTranslationProvider,
    TranslationService,
    detect_language_fast,
    has_translatable_content,
    translator_service,
)
from app.services.tts_engine import tts_engine


class MockFastProvider(BaseTranslationProvider):
    """Deterministic async mock translation provider for fast unit & concurrency tests."""

    name = "mock-mt"

    def __init__(self, delay_s: float = 0.01) -> None:
        self.delay_s = delay_s
        self.call_count = 0

    async def translate_raw(self, text: str, source_lang: str, target_lang: str) -> str:
        self.call_count += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return f"[{target_lang}] {text}"


# ==============================================================================
# 1. Config & Language Registry Tests
# ==============================================================================


def test_language_normalization_and_voices() -> None:
    assert normalize_lang_code("zh") == "zh-CN"
    assert normalize_lang_code("en-US") == "en"
    assert normalize_lang_code("ES-MX") == "es"
    assert normalize_lang_code("auto") == "auto"
    assert normalize_lang_code("unknown-xyz", fallback="fr") == "fr"

    hi_spec = get_language_spec("hi")
    assert hi_spec.tts_voice == "hi-IN-SwaraNeural"
    assert hi_spec.bcp47 == "hi-IN"


# ==============================================================================
# 2. Audio Conversion, VAD & Rolling Ring Buffer Tests
# ==============================================================================


def test_pcm16_conversion_and_rms() -> None:
    assert pcm16_bytes_to_float32(b"").size == 0
    assert pcm16_bytes_to_float32(b"\x01").size == 0

    # Half-scale sine wave @ 440Hz for 160ms (2560 samples @ 16kHz)
    t = np.arange(2560, dtype=np.float32) / 16000.0
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm_i16 = (sine * 32767.0).astype("<i2")
    raw_bytes = pcm_i16.tobytes()

    restored = pcm16_bytes_to_float32(raw_bytes)
    assert restored.shape == (2560,)
    assert np.allclose(restored, sine, atol=1e-3)
    assert 0.34 < compute_rms(restored) < 0.36


def test_rolling_audio_buffer_lifecycle() -> None:
    # Use deterministic RMS-only VAD for repeatable synthetic waveform tests
    rms_vad = HybridVADEngine(rms_threshold=0.02, use_silero=False)
    buf = RollingAudioBuffer(
        sample_rate=16000,
        silence_threshold_ms=800,
        sliding_window_step_ms=300,
        pre_roll_ms=200,
        vad=rms_vad,
    )

    chunk_samples = 2560  # 160ms @ 16kHz
    silence_pcm = np.zeros(chunk_samples, dtype="<i2").tobytes()
    t = np.arange(chunk_samples, dtype=np.float32) / 16000.0
    speech_pcm = (0.35 * np.sin(2.0 * np.pi * 300.0 * t) * 32767).astype("<i2").tobytes()

    # 1) Feed silence -> pre-roll fills, active buffer stays inactive
    d0 = buf.ingest_pcm16(silence_pcm)
    assert not d0.is_speech
    assert not buf.has_active_speech
    assert not d0.should_transcribe_partial

    # 2) Feed 3 chunks of speech (480ms + pre-roll) -> triggers sliding-window partial ASR
    partials_triggered = 0
    for _ in range(3):
        dec = buf.ingest_pcm16(speech_pcm)
        assert dec.is_speech
        if dec.should_transcribe_partial:
            partials_triggered += 1
    assert buf.has_active_speech
    assert partials_triggered >= 1
    assert buf.duration_ms >= 600.0  # 480ms speech + 160ms pre-roll

    # 3) Feed 5 chunks of silence (5 * 160ms = 800ms) -> triggers clause commit!
    committed = False
    for _ in range(5):
        dec = buf.ingest_pcm16(silence_pcm)
        if dec.should_commit_clause:
            committed = True
            break
    assert committed

    popped = buf.pop_utterance_audio()
    assert popped.size > 0
    assert not buf.has_active_speech


# ==============================================================================
# 3. Clause Boundary Splitter & Hybrid Segment Tracker Tests
# ==============================================================================


def test_clause_boundary_splitting_and_segment_tracker() -> None:
    short = "Hello how are you doing today"
    committed, rem = split_clause_at_boundary(short, max_words=15)
    assert committed is None
    assert rem == short

    long_clause = (
        "Welcome everyone to this live simultaneous translation demonstration today, "
        "we are speaking continuously without pressing any stop button"
    )
    committed, rem = split_clause_at_boundary(long_clause, max_words=15)
    assert committed is not None
    assert committed.endswith(",")
    assert "without pressing" in rem

    tracker = HybridSegmentTracker(max_clause_words=15)
    u1 = tracker.extract_uncommitted(long_clause, segment_id="seg_1")
    assert u1 == long_clause

    tracker.mark_prefix_committed(committed)
    # Next cumulative update from browser SpeechRecognition on same segment_id
    extended = long_clause + " right now"
    u2 = tracker.extract_uncommitted(extended, segment_id="seg_1")
    assert u2 == rem + " right now"
    assert "Welcome everyone" not in u2


# ==============================================================================
# 4. Translator LRU Cache, Noise Filtering & Bi-Directional Routing Tests
# ==============================================================================


def test_translator_lru_cache_noise_filter_and_bidirectional() -> None:
    async def _run() -> None:
        svc = TranslationService(max_cache_size=32)
        mock_prov = MockFastProvider(delay_s=0.03)
        svc.provider = mock_prov

        # Empty / punctuation-only filter (Section 8.4)
        assert not has_translatable_content("   ... ?!  ")
        noise_res = await svc.translate("  ...?! ", source_lang="en", target_lang="es")
        assert noise_res.translated_text == ""
        assert mock_prov.call_count == 0

        # First call misses cache, hits provider
        r1 = await svc.translate("Hello how are you", source_lang="en", target_lang="es")
        assert r1.translated_text == "[es] Hello how are you"
        assert not r1.cached
        assert mock_prov.call_count == 1

        # Second identical call resolves in ~0ms from LRU cache without hitting provider
        t0 = time.perf_counter()
        r2 = await svc.translate("Hello how are you", source_lang="en", target_lang="es")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert r2.cached
        assert r2.translated_text == "[es] Hello how are you"
        assert mock_prov.call_count == 1
        assert elapsed_ms < 5.0

        # Bi-Directional routing between English (A) and Hindi (B)
        assert detect_language_fast("नमस्ते आप कैसे हैं") == "hi"
        bi_hi = await svc.translate(
            "नमस्ते आप कैसे हैं",
            source_lang="en",
            target_lang="hi",
            bidirectional=True,
        )
        assert bi_hi.source_lang == "hi"
        assert bi_hi.target_lang == "en"
        assert bi_hi.speaker == "B"

        bi_en = await svc.translate(
            "Hello how are you doing",
            source_lang="en",
            target_lang="hi",
            bidirectional=True,
        )
        assert bi_en.source_lang == "en"
        assert bi_en.target_lang == "hi"
        assert bi_en.speaker == "A"

    asyncio.run(_run())


# ==============================================================================
# 5. REST & Full-Duplex WebSocket Protocol Tests
# ==============================================================================


def test_rest_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_prov = MockFastProvider(delay_s=0.005)
    monkeypatch.setattr(translator_service, "provider", mock_prov)

    with TestClient(app) as client:
        landing = client.get("/")
        assert landing.status_code == 200
        assert "Try Working Model" in landing.text
        assert 'href="/app"' in landing.text

        studio = client.get("/app")
        assert studio.status_code == 200
        assert "Microphone Access Required" in studio.text

        docs = client.get("/docs")
        assert docs.status_code == 200
        assert "API &amp; Streaming Protocol" in docs.text
        assert "Live WebSocket Console" in docs.text

        swagger = client.get("/docs/swagger")
        assert swagger.status_code == 200
        assert "SwaggerUIBundle" in swagger.text

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        assert "/api/translate" in openapi.json()["paths"]

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        langs = client.get("/api/languages")
        assert langs.status_code == 200
        codes = {item["code"] for item in langs.json()["languages"]}
        assert {"auto", "en", "es", "hi", "fr", "de", "zh-CN", "ja", "ar"}.issubset(codes)

        tr = client.post(
            "/api/translate",
            json={"text": "Hello world", "source_lang": "en", "target_lang": "es"},
        )
        assert tr.status_code == 200
        assert tr.json()["translated_text"] == "[es] Hello world"


def test_websocket_provisional_and_committed_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_prov = MockFastProvider(delay_s=0.01)
    monkeypatch.setattr(translator_service, "provider", mock_prov)
    monkeypatch.setattr(
        tts_engine,
        "synthesize_base64",
        AsyncMock(return_value="FAKE_MP3_BASE64_PAYLOAD"),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/translate") as ws:
            # 1. Config handshake
            ws.send_json(
                {
                    "action": "config",
                    "source_lang": "en",
                    "target_lang": "es",
                    "tts_enabled": True,
                    "auto_silence_commit": False,
                }
            )
            ack = ws.receive_json()
            assert ack["status"] == "config_ack"
            assert ack["target_lang"] == "es"

            # 2. Rapid burst of partial tokens -> stale partials cancelled, latest partial arrives
            ws.send_json({"action": "interim_speech", "text": "Hello", "is_final": False, "segment_id": "s1"})
            ws.send_json({"action": "interim_speech", "text": "Hello how are", "is_final": False, "segment_id": "s1"})
            ws.send_json(
                {"action": "interim_speech", "text": "Hello how are you", "is_final": False, "segment_id": "s1"}
            )

            # Drain partial message(s) until latest arrives
            latest_partial = None
            for _ in range(3):
                msg = ws.receive_json()
                assert msg["status"] == "partial"
                assert msg["audio_base64"] is None
                latest_partial = msg
                if msg["original_text"] == "Hello how are you":
                    break

            assert latest_partial is not None
            assert latest_partial["original_text"] == "Hello how are you"
            assert latest_partial["translated_text"] == "[es] Hello how are you"
            active_utt_id = latest_partial["utterance_id"]

            # 3. Commit final utterance -> verify Section 7.2 schema & TTS payload
            ws.send_json(
                {
                    "action": "interim_speech",
                    "text": "Hello how are you today",
                    "is_final": True,
                    "segment_id": "s1",
                }
            )
            final_msg = ws.receive_json()
            assert final_msg["status"] == "final"
            assert final_msg["utterance_id"] == active_utt_id
            assert final_msg["source_lang"] == "en"
            assert final_msg["target_lang"] == "es"
            assert final_msg["original_text"] == "Hello how are you today"
            assert final_msg["translated_text"] == "[es] Hello how are you today"
            assert final_msg["audio_base64"] == "FAKE_MP3_BASE64_PAYLOAD"
            assert isinstance(final_msg["latency_ms"], int)


def test_websocket_binary_pcm_whisper_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_prov = MockFastProvider(delay_s=0.005)
    monkeypatch.setattr(translator_service, "provider", mock_prov)
    monkeypatch.setattr(tts_engine, "synthesize_base64", AsyncMock(return_value=None))
    monkeypatch.setattr(whisper_transcriber, "warmup", AsyncMock(return_value={"loaded": True}))
    monkeypatch.setattr(
        whisper_transcriber,
        "transcribe",
        AsyncMock(
            return_value=TranscriptionResult(
                text="Live audio streamed over binary WebSockets",
                detected_lang="en",
                language_probability=0.98,
                no_speech_prob=0.02,
                duration_ms=42,
            )
        ),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/translate") as ws:
            ws.send_json(
                {
                    "action": "config",
                    "source_lang": "auto",
                    "target_lang": "fr",
                    "tts_enabled": False,
                    "engine_mode": "whisper",
                }
            )
            # Consume config_ack & async engine_status frames
            seen_ack = False
            while not seen_ack:
                frame = ws.receive_json()
                if frame["status"] == "config_ack":
                    seen_ack = True

            # Stream synthetic speech PCM frames followed by >800ms silence PCM frames
            t = np.arange(2560, dtype=np.float32) / 16000.0
            speech_bytes = (0.4 * np.sin(2.0 * np.pi * 220.0 * t) * 32767).astype("<i2").tobytes()
            silence_bytes = np.zeros(2560, dtype="<i2").tobytes()

            # Force deterministic RMS VAD inside this session test
            from app.services import audio_stream

            monkeypatch.setattr(audio_stream.vad_engine, "use_silero", False)

            for _ in range(4):
                ws.send_bytes(speech_bytes)
            for _ in range(6):
                ws.send_bytes(silence_bytes)

            got_final = None
            for _ in range(8):
                msg = ws.receive_json()
                if msg["status"] == "final":
                    got_final = msg
                    break

            assert got_final is not None
            assert got_final["original_text"] == "Live audio streamed over binary WebSockets"
            assert got_final["translated_text"] == "[fr] Live audio streamed over binary WebSockets"
            assert got_final["target_lang"] == "fr"
