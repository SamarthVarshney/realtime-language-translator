"""Edge-TTS async neural speech synthesizer with automatic retry and HTTP TTS failover."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import threading
from collections import OrderedDict

import edge_tts
import httpx

from app.config import get_language_spec, settings
from app.services.translator import has_translatable_content

logger = logging.getLogger(__name__)


class EdgeTTSEngine:
    """
    Async neural Text-to-Speech generator powered by Microsoft `edge-tts`.

    Streams MP3 audio chunks purely in memory (zero disk I/O) with:
      - In-memory LRU phrase audio cache
      - Automatic transient DNS/connection retry
      - Seamless HTTP MP3 TTS fallback if Bing Speech is unreachable on restricted networks
    """

    def __init__(
        self,
        max_cache_size: int = settings.tts_cache_size,
        rate: str = "+5%",
        timeout_seconds: float = 7.0,
    ) -> None:
        self.max_cache_size = max_cache_size
        self.rate = rate
        self.timeout_seconds = timeout_seconds
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()
        self._http_fallback: httpx.AsyncClient | None = None

    def _get_http_fallback(self) -> httpx.AsyncClient:
        if self._http_fallback is None:
            self._http_fallback = httpx.AsyncClient(
                timeout=4.5,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0.0.0"},
            )
        return self._http_fallback

    def _cache_get(self, key: tuple[str, str]) -> str | None:
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
                return val
            return None

    def _cache_put(self, key: tuple[str, str], audio_b64: str) -> None:
        with self._lock:
            self._cache[key] = audio_b64
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

    async def _synthesize_edge_tts(self, text: str, voice: str) -> bytes:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=self.rate)
        buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if data:
                    buffer.extend(data)
        return bytes(buffer)

    async def _synthesize_http_fallback(self, text: str, lang_code: str) -> bytes:
        tl = get_language_spec(lang_code, fallback="en").translator_code
        snippet = text[:200]
        resp = await self._get_http_fallback().get(
            "https://translate.google.com/translate_tts",
            params={"ie": "UTF-8", "client": "tw-ob", "tl": tl, "q": snippet},
        )
        resp.raise_for_status()
        return resp.content

    async def synthesize_base64(self, text: str, target_lang: str) -> str | None:
        """
        Synthesize speech for `text` in `target_lang` and return base64-encoded MP3 bytes.

        Returns `None` only if text has no speakable content or all TTS tiers fail.
        """
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not has_translatable_content(cleaned):
            return None

        if len(cleaned) > 550:
            cleaned = cleaned[:550].rsplit(" ", 1)[0] + "..."

        spec = get_language_spec(target_lang, fallback="en")
        voice = spec.tts_voice
        cache_key = (voice, cleaned)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        mp3_bytes: bytes = b""
        last_err: Exception | None = None

        # Primary: Microsoft Edge Neural TTS (with 1 fast retry on cold DNS lookup)
        for attempt in range(2):
            try:
                mp3_bytes = await asyncio.wait_for(
                    self._synthesize_edge_tts(cleaned, voice),
                    timeout=self.timeout_seconds,
                )
                if mp3_bytes:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_err = exc
                if attempt == 0:
                    await asyncio.sleep(0.25)

        # Secondary Fallback: Direct HTTP MP3 TTS stream if Bing Speech WSS is blocked
        if not mp3_bytes:
            try:
                mp3_bytes = await self._synthesize_http_fallback(cleaned, spec.code)
            except asyncio.CancelledError:
                raise
            except Exception as fallback_exc:
                logger.warning(
                    "All TTS tiers failed for voice=%s (edge=%r, fallback=%r)",
                    voice,
                    last_err,
                    fallback_exc,
                )
                return None

        if not mp3_bytes:
            return None

        encoded = base64.b64encode(mp3_bytes).decode("ascii")
        self._cache_put(cache_key, encoded)
        return encoded


# Shared singleton TTS engine
tts_engine = EdgeTTSEngine()
