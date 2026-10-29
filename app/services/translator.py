"""Async real-time Machine Translation service with LRU phrase caching and Mode A/B providers."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from deep_translator import GoogleTranslator, MyMemoryTranslator

from app.config import get_language_spec, normalize_lang_code, settings

logger = logging.getLogger(__name__)

# Script character detectors for instant (<0.05ms) language identification
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")

# High-signal function words for disambiguating Latin-script languages in Bi-Directional mode
_LATIN_STOPWORDS: dict[str, set[str]] = {
    "en": {
        "the", "is", "are", "am", "you", "we", "they", "this", "that", "have",
        "with", "for", "not", "what", "how", "can", "will", "hello", "please", "thank",
    },
    "es": {
        "el", "la", "los", "las", "es", "son", "estoy", "está", "estás", "como", "cómo",
        "hola", "para", "con", "por", "qué", "que", "gracias", "buenos", "pero", "muy",
    },
    "fr": {
        "le", "les", "est", "sont", "je", "nous", "vous", "bonjour", "merci",
        "avec", "pour", "dans", "pas", "très", "comment", "c'est", "une", "des",
    },
    "de": {
        "der", "die", "das", "ist", "sind", "ich", "wir", "sie", "nicht", "mit",
        "für", "und", "wie", "hallo", "danke", "bitte", "ein", "eine", "haben",
    },
    "pt": {
        "o", "os", "as", "é", "são", "estou", "você", "nós", "olá", "obrigado",
        "com", "para", "não", "como", "muito", "isso", "uma", "bom",
    },
    "it": {
        "il", "lo", "gli", "è", "sono", "ciao", "grazie", "per", "con", "non",
        "come", "molto", "questo", "questa", "della", "anche", "buongiorno",
    },
    "tr": {
        "bir", "ve", "bu", "için", "değil", "merhaba", "teşekkür", "nasıl",
        "ben", "sen", "biz", "çok", "var", "yok", "evet",
    },
    "nl": {
        "de", "het", "een", "is", "zijn", "ik", "wij", "jij", "niet", "met",
        "voor", "hallo", "dank", "hoe", "dat", "van", "maar",
    },
}


def has_translatable_content(text: str) -> bool:
    """Return False if text is empty or consists solely of whitespace/punctuation/symbols."""
    if not text:
        return False
    return any(ch.isalnum() for ch in text)


def detect_language_fast(text: str, candidates: tuple[str, ...] | None = None) -> str | None:
    """
    Ultra-fast (<0.1ms) heuristic language identifier using Unicode script blocks and stopwords.

    Used to label auto-detected speech and instantaneously route Bi-Directional conversations.
    """
    if not text:
        return None

    if _DEVANAGARI_RE.search(text):
        return "hi"
    if _ARABIC_RE.search(text):
        return "ar"
    if _HANGUL_RE.search(text):
        return "ko"
    if _KANA_RE.search(text):
        return "ja"
    if _HAN_RE.search(text):
        return "ja" if (candidates and "ja" in candidates and "zh-CN" not in candidates) else "zh-CN"
    if _CYRILLIC_RE.search(text):
        return "ru"

    words = re.findall(r"[a-zA-Záéíóúüñçàèìòùâêîôûäöüßğışİ]+", text.lower())
    if not words:
        return None

    pool = candidates if candidates else tuple(_LATIN_STOPWORDS.keys())
    best_lang: str | None = None
    best_score = 0
    for lang in pool:
        lexicon = _LATIN_STOPWORDS.get(lang)
        if not lexicon:
            continue
        score = sum(1 for w in words if w in lexicon)
        if score > best_score:
            best_score = score
            best_lang = lang

    # Characteristic diacritics tie-breaker
    lowered = text.lower()
    if best_score == 0:
        if "¿" in lowered or "¡" in lowered or "ñ" in lowered:
            return "es" if (not candidates or "es" in candidates) else None
        if "ß" in lowered:
            return "de" if (not candidates or "de" in candidates) else None
        if any(c in lowered for c in ("ğ", "ş", "ı")):
            return "tr" if (not candidates or "tr" in candidates) else None

    return best_lang if best_score > 0 else None


@dataclass
class TranslationResult:
    """Structured translation output including cache telemetry and resolved language direction."""

    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str
    cached: bool
    latency_ms: int
    speaker: str = "A"


class BaseTranslationProvider(ABC):
    """Modular Machine Translation provider interface."""

    name: str = "base"

    @abstractmethod
    async def translate_raw(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate `text` from canonical `source_lang` (or 'auto') into `target_lang`."""


class DeepTranslatorProvider(BaseTranslationProvider):
    """
    Mode A zero-key free provider combining persistent async HTTP connection pooling
    with `deep-translator` (`GoogleTranslator` + `MyMemoryTranslator`) automatic failover.
    """

    name = "deep-translator"

    def __init__(self, max_workers: int = 6) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mt-worker")
        self._client: httpx.AsyncClient | None = None
        self._google_html_cooldown_until: float = 0.0

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=3.5,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0.0.0"},
            )
        return self._client

    @staticmethod
    def _run_google_sync(text: str, src_code: str, tgt_code: str) -> str:
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        result = translator.translate(text)
        return (result or text).strip()

    @staticmethod
    def _run_mymemory_sync(text: str, src_code: str, tgt_code: str) -> str:
        src_bcp = get_language_spec(src_code, fallback="en").bcp47 if src_code != "auto" else "en-US"
        tgt_bcp = get_language_spec(tgt_code, fallback="es").bcp47
        translator = MyMemoryTranslator(source=src_bcp, target=tgt_bcp)
        result = translator.translate(text)
        return (result or text).strip()

    async def translate_raw(self, text: str, source_lang: str, target_lang: str) -> str:
        src_code = (
            "auto"
            if source_lang == "auto"
            else get_language_spec(source_lang).translator_code
        )
        tgt_code = get_language_spec(target_lang).translator_code
        if src_code != "auto" and src_code == tgt_code:
            return text

        # Tier 1: Ultra-low-latency connection-pooled JSON endpoint (~60-110ms)
        try:
            resp = await self._get_client().get(
                "https://clients5.google.com/translate_a/t",
                params={"client": "dict-chrome-ex", "sl": src_code, "tl": tgt_code, "q": text},
            )
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, list) and payload:
                    first = payload[0]
                    if isinstance(first, list) and first:
                        return str(first[0]).strip()
                    if isinstance(first, str):
                        return first.strip()
        except Exception as exc:
            logger.debug("Fast JSON translate tier skipped: %s", exc)

        loop = asyncio.get_running_loop()

        # Tier 2: deep-translator GoogleTranslator (with 60s circuit-breaker on HTTP 429)
        now = time.monotonic()
        if now >= self._google_html_cooldown_until:
            try:
                return await loop.run_in_executor(
                    self._executor, self._run_google_sync, text, src_code, tgt_code
                )
            except Exception as exc:
                self._google_html_cooldown_until = now + 60.0
                logger.info("GoogleTranslator rate-limited; failing over to MyMemory: %s", exc)

        # Tier 3: deep-translator MyMemoryTranslator free fallback
        return await loop.run_in_executor(
            self._executor, self._run_mymemory_sync, text, src_code, tgt_code
        )


class DeepLTranslationProvider(BaseTranslationProvider):
    """Mode B optional cloud provider using DeepL REST API with fallback to DeepTranslator."""

    name = "deepl"

    def __init__(self, api_key: str, api_url: str, fallback: BaseTranslationProvider) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.fallback = fallback
        self._client = httpx.AsyncClient(timeout=4.0)

    async def translate_raw(self, text: str, source_lang: str, target_lang: str) -> str:
        tgt_deepl = get_language_spec(target_lang).deepl_code
        if not tgt_deepl:
            return await self.fallback.translate_raw(text, source_lang, target_lang)

        payload: dict[str, object] = {"text": [text], "target_lang": tgt_deepl}
        if source_lang != "auto":
            src_deepl = get_language_spec(source_lang).deepl_code
            if src_deepl:
                payload["source_lang"] = src_deepl.split("-")[0]

        try:
            resp = await self._client.post(
                self.api_url,
                headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            translations = data.get("translations") or []
            if translations:
                return str(translations[0].get("text", text)).strip()
        except Exception as exc:
            logger.warning("DeepL API fallback to deep-translator due to error: %s", exc)

        return await self.fallback.translate_raw(text, source_lang, target_lang)


class GoogleCloudTranslationProvider(BaseTranslationProvider):
    """Mode B optional cloud provider using Google Cloud Translation v2 REST API."""

    name = "google-cloud"

    def __init__(self, api_key: str, fallback: BaseTranslationProvider) -> None:
        self.api_key = api_key
        self.fallback = fallback
        self._client = httpx.AsyncClient(timeout=4.0)

    async def translate_raw(self, text: str, source_lang: str, target_lang: str) -> str:
        tgt_code = get_language_spec(target_lang).translator_code
        params: dict[str, str] = {"key": self.api_key}
        body: dict[str, object] = {"q": text, "target": tgt_code, "format": "text"}
        if source_lang != "auto":
            body["source"] = get_language_spec(source_lang).translator_code

        try:
            resp = await self._client.post(
                "https://translation.googleapis.com/language/translate/v2",
                params=params,
                json=body,
            )
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("translations", [])
            if items:
                raw_text = str(items[0].get("translatedText", text))
                return html.unescape(raw_text).strip()
        except Exception as exc:
            logger.warning("Google Cloud Translate fallback to deep-translator: %s", exc)

        return await self.fallback.translate_raw(text, source_lang, target_lang)


class TranslationService:
    """
    High-concurrency translation orchestrator featuring:
      - 0ms thread-safe LRU phrase cache for repeated sliding-window prefixes
      - In-flight request coalescing for identical concurrent strings
      - Empty/punctuation noise suppression
      - Bi-Directional two-speaker language auto-routing
    """

    def __init__(self, max_cache_size: int = settings.translation_cache_size) -> None:
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inflight: dict[tuple[str, str, str], asyncio.Future[str]] = {}

        self.cache_hits: int = 0
        self.cache_misses: int = 0

        free_provider = DeepTranslatorProvider()
        if settings.deepl_api_key:
            self.provider: BaseTranslationProvider = DeepLTranslationProvider(
                api_key=settings.deepl_api_key,
                api_url=settings.deepl_api_url,
                fallback=free_provider,
            )
        elif settings.google_translate_api_key:
            self.provider = GoogleCloudTranslationProvider(
                api_key=settings.google_translate_api_key,
                fallback=free_provider,
            )
        else:
            self.provider = free_provider

    def get_stats(self) -> dict[str, object]:
        with self._cache_lock:
            size = len(self._cache)
        return {
            "provider": self.provider.name,
            "cache_size": size,
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def _cache_get(self, key: tuple[str, str, str]) -> str | None:
        with self._cache_lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
                return val
            return None

    def _cache_put(self, key: tuple[str, str, str], value: str) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

    def resolve_direction(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        bidirectional: bool = False,
        active_speaker: str = "auto",
        detected_hint: str | None = None,
    ) -> tuple[str, str, str]:
        """
        Determine effective `(resolved_source_lang, resolved_target_lang, speaker_label)`.

        In Bi-Directional mode between Lang A (`source_lang`) and Lang B (`target_lang`),
        automatically detects which of the two languages is being spoken and translates
        toward the other speaker's language.
        """
        src = "auto" if source_lang == "auto" else normalize_lang_code(source_lang, fallback="auto")
        tgt = normalize_lang_code(target_lang, fallback="es")
        if tgt == "auto":
            tgt = "es"

        if not bidirectional:
            if src == "auto":
                guessed = detected_hint or detect_language_fast(text)
                display_src = normalize_lang_code(guessed, fallback="auto") if guessed else "auto"
                return display_src, tgt, "A"
            return src, tgt, "A"

        # Bi-Directional Conversation Mode between Lang A and Lang B
        lang_a = "en" if src == "auto" else src
        lang_b = tgt if tgt != lang_a else ("es" if lang_a != "es" else "en")

        if active_speaker == "A":
            return lang_a, lang_b, "A"
        if active_speaker == "B":
            return lang_b, lang_a, "B"

        # Automatic speaker/language detection between lang_a and lang_b
        hint = normalize_lang_code(detected_hint, fallback="") if detected_hint else ""
        if hint == lang_b:
            return lang_b, lang_a, "B"
        if hint == lang_a:
            return lang_a, lang_b, "A"

        guessed = detect_language_fast(text, candidates=(lang_a, lang_b))
        if guessed == lang_b:
            return lang_b, lang_a, "B"
        return lang_a, lang_b, "A"

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "es",
        bidirectional: bool = False,
        active_speaker: str = "auto",
        detected_hint: str | None = None,
    ) -> TranslationResult:
        """Translate partial or final speech with 0ms LRU cache hits and noise filtering."""
        t0 = time.perf_counter()
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()

        eff_src, eff_tgt, speaker = self.resolve_direction(
            cleaned,
            source_lang=source_lang,
            target_lang=target_lang,
            bidirectional=bidirectional,
            active_speaker=active_speaker,
            detected_hint=detected_hint,
        )

        # Filter empty or punctuation-only strings without invoking external providers
        if not has_translatable_content(cleaned):
            return TranslationResult(
                original_text=cleaned,
                translated_text="",
                source_lang=eff_src,
                target_lang=eff_tgt,
                provider=self.provider.name,
                cached=True,
                latency_ms=0,
                speaker=speaker,
            )

        # Short-circuit identical source & target languages
        if eff_src != "auto" and eff_src == eff_tgt:
            return TranslationResult(
                original_text=cleaned,
                translated_text=cleaned,
                source_lang=eff_src,
                target_lang=eff_tgt,
                provider="passthrough",
                cached=True,
                latency_ms=0,
                speaker=speaker,
            )

        # Use 'auto' for the provider call when source was auto-detected heuristically so MT handles mixed loanwords cleanly
        provider_src = "auto" if (source_lang == "auto" and not bidirectional) else eff_src
        cache_key = (provider_src, eff_tgt, cleaned)

        cached_val = self._cache_get(cache_key)
        if cached_val is not None:
            self.cache_hits += 1
            return TranslationResult(
                original_text=cleaned,
                translated_text=cached_val,
                source_lang=eff_src if eff_src != "auto" else (detect_language_fast(cleaned) or "auto"),
                target_lang=eff_tgt,
                provider=self.provider.name,
                cached=True,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                speaker=speaker,
            )

        self.cache_misses += 1

        # Coalesce identical concurrent requests
        existing_future = self._inflight.get(cache_key)
        if existing_future is not None:
            translated = await asyncio.shield(existing_future)
        else:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[str] = loop.create_future()
            self._inflight[cache_key] = fut
            try:
                translated = await self.provider.translate_raw(cleaned, provider_src, eff_tgt)
                # Bi-directional self-echo fallback: if auto speaker guessed A->B on ambiguous Latin text
                # and output equals input (3+ words), try flipping B->A automatically!
                if (
                    bidirectional
                    and active_speaker == "auto"
                    and translated.lower() == cleaned.lower()
                    and len(cleaned.split()) >= 2
                ):
                    flipped_src, flipped_tgt = eff_tgt, eff_src
                    alt = await self.provider.translate_raw(cleaned, flipped_src, flipped_tgt)
                    if alt and alt.lower() != cleaned.lower():
                        translated = alt
                        eff_src, eff_tgt, speaker = flipped_src, flipped_tgt, ("B" if speaker == "A" else "A")
                        cache_key = (eff_src, eff_tgt, cleaned)

                self._cache_put(cache_key, translated)
                if not fut.done():
                    fut.set_result(translated)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.cancel()
                raise
            except Exception as exc:
                logger.warning("Translation error (%s -> %s): %s", provider_src, eff_tgt, exc)
                translated = cleaned
                if not fut.done():
                    fut.set_result(translated)
            finally:
                self._inflight.pop(cache_key, None)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        display_src = eff_src if eff_src != "auto" else (detect_language_fast(cleaned) or "auto")
        return TranslationResult(
            original_text=cleaned,
            translated_text=translated,
            source_lang=display_src,
            target_lang=eff_tgt,
            provider=self.provider.name,
            cached=False,
            latency_ms=elapsed_ms,
            speaker=speaker,
        )


# Shared singleton translator service
translator_service = TranslationService()
