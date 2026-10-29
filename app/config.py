"""Application configuration, environment settings, and multilingual voice/code registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load optional .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class LanguageSpec:
    """Metadata mapping a canonical language code across ASR, MT, and TTS engines."""

    code: str
    name: str
    native_name: str
    bcp47: str
    translator_code: str
    whisper_code: str | None
    deepl_code: str | None
    tts_voice: str
    allow_as_target: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "native_name": self.native_name,
            "bcp47": self.bcp47,
            "tts_voice": self.tts_voice,
            "allow_as_target": self.allow_as_target,
        }


SUPPORTED_LANGUAGES: dict[str, LanguageSpec] = {
    "auto": LanguageSpec(
        code="auto",
        name="Auto-Detect",
        native_name="Auto",
        bcp47="en-US",
        translator_code="auto",
        whisper_code=None,
        deepl_code=None,
        tts_voice="en-US-AriaNeural",
        allow_as_target=False,
    ),
    "en": LanguageSpec(
        code="en",
        name="English",
        native_name="English",
        bcp47="en-US",
        translator_code="en",
        whisper_code="en",
        deepl_code="EN",
        tts_voice="en-US-AriaNeural",
    ),
    "es": LanguageSpec(
        code="es",
        name="Spanish",
        native_name="Español",
        bcp47="es-ES",
        translator_code="es",
        whisper_code="es",
        deepl_code="ES",
        tts_voice="es-ES-ElviraNeural",
    ),
    "hi": LanguageSpec(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        bcp47="hi-IN",
        translator_code="hi",
        whisper_code="hi",
        deepl_code=None,
        tts_voice="hi-IN-SwaraNeural",
    ),
    "fr": LanguageSpec(
        code="fr",
        name="French",
        native_name="Français",
        bcp47="fr-FR",
        translator_code="fr",
        whisper_code="fr",
        deepl_code="FR",
        tts_voice="fr-FR-DeniseNeural",
    ),
    "de": LanguageSpec(
        code="de",
        name="German",
        native_name="Deutsch",
        bcp47="de-DE",
        translator_code="de",
        whisper_code="de",
        deepl_code="DE",
        tts_voice="de-DE-KatjaNeural",
    ),
    "zh-CN": LanguageSpec(
        code="zh-CN",
        name="Mandarin Chinese",
        native_name="中文 (简体)",
        bcp47="zh-CN",
        translator_code="zh-CN",
        whisper_code="zh",
        deepl_code="ZH",
        tts_voice="zh-CN-XiaoxiaoNeural",
    ),
    "ja": LanguageSpec(
        code="ja",
        name="Japanese",
        native_name="日本語",
        bcp47="ja-JP",
        translator_code="ja",
        whisper_code="ja",
        deepl_code="JA",
        tts_voice="ja-JP-NanamiNeural",
    ),
    "ar": LanguageSpec(
        code="ar",
        name="Arabic",
        native_name="العربية",
        bcp47="ar-SA",
        translator_code="ar",
        whisper_code="ar",
        deepl_code="AR",
        tts_voice="ar-SA-ZariyahNeural",
    ),
    "pt": LanguageSpec(
        code="pt",
        name="Portuguese",
        native_name="Português",
        bcp47="pt-BR",
        translator_code="pt",
        whisper_code="pt",
        deepl_code="PT-BR",
        tts_voice="pt-BR-FranciscaNeural",
    ),
    "ru": LanguageSpec(
        code="ru",
        name="Russian",
        native_name="Русский",
        bcp47="ru-RU",
        translator_code="ru",
        whisper_code="ru",
        deepl_code="RU",
        tts_voice="ru-RU-SvetlanaNeural",
    ),
    "ko": LanguageSpec(
        code="ko",
        name="Korean",
        native_name="한국어",
        bcp47="ko-KR",
        translator_code="ko",
        whisper_code="ko",
        deepl_code="KO",
        tts_voice="ko-KR-SunHiNeural",
    ),
    "it": LanguageSpec(
        code="it",
        name="Italian",
        native_name="Italiano",
        bcp47="it-IT",
        translator_code="it",
        whisper_code="it",
        deepl_code="IT",
        tts_voice="it-IT-ElsaNeural",
    ),
    "tr": LanguageSpec(
        code="tr",
        name="Turkish",
        native_name="Türkçe",
        bcp47="tr-TR",
        translator_code="tr",
        whisper_code="tr",
        deepl_code="TR",
        tts_voice="tr-TR-EmelNeural",
    ),
    "nl": LanguageSpec(
        code="nl",
        name="Dutch",
        native_name="Nederlands",
        bcp47="nl-NL",
        translator_code="nl",
        whisper_code="nl",
        deepl_code="NL",
        tts_voice="nl-NL-ColetteNeural",
    ),
}

# Alias lookup table for Whisper / BCP-47 / translator outputs
_LANGUAGE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-CN",
    "cmn": "zh-CN",
    "chinese": "zh-CN",
    "en-us": "en",
    "en-gb": "en",
    "es-es": "es",
    "es-mx": "es",
    "hi-in": "hi",
    "fr-fr": "fr",
    "de-de": "de",
    "ja-jp": "ja",
    "ar-sa": "ar",
    "pt-br": "pt",
    "pt-pt": "pt",
    "ru-ru": "ru",
    "ko-kr": "ko",
    "it-it": "it",
    "tr-tr": "tr",
    "nl-nl": "nl",
}


def normalize_lang_code(code: str | None, fallback: str = "en") -> str:
    """Normalize arbitrary ISO/BCP-47 language strings into canonical SUPPORTED_LANGUAGES keys."""
    if not code:
        return fallback
    cleaned = code.strip()
    if cleaned in SUPPORTED_LANGUAGES:
        return cleaned
    lowered = cleaned.lower()
    if lowered in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[lowered]
    if lowered in SUPPORTED_LANGUAGES:
        return lowered
    primary = lowered.split("-")[0].split("_")[0]
    if primary in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[primary]
    if primary in SUPPORTED_LANGUAGES:
        return primary
    return fallback


def get_language_spec(code: str | None, fallback: str = "en") -> LanguageSpec:
    norm = normalize_lang_code(code, fallback=fallback)
    return SUPPORTED_LANGUAGES.get(norm, SUPPORTED_LANGUAGES[fallback])


@dataclass
class Settings:
    """Runtime settings populated from environment variables with safe defaults."""

    app_name: str = "Real-Time Simultaneous Language Translator"
    audio_sample_rate: int = 16000

    # Faster-Whisper ASR configuration
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base").strip() or "base"
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto").strip() or "auto"
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto").strip() or "auto"

    # Streaming & VAD thresholds
    silence_threshold_ms: int = _env_int("SILENCE_THRESHOLD_MS", 800)
    sliding_window_step_ms: int = _env_int("SLIDING_WINDOW_STEP_MS", 400)
    translation_debounce_ms: int = _env_int("TRANSLATION_DEBOUNCE_MS", 150)
    max_clause_words: int = _env_int("MAX_CLAUSE_WORDS", 15)
    max_utterance_seconds: float = _env_float("MAX_UTTERANCE_SECONDS", 15.0)
    vad_rms_threshold: float = _env_float("VAD_RMS_THRESHOLD", 0.008)
    vad_speech_prob_threshold: float = _env_float("VAD_SPEECH_PROB_THRESHOLD", 0.45)

    # Caches
    translation_cache_size: int = _env_int("TRANSLATION_CACHE_SIZE", 4096)
    tts_cache_size: int = _env_int("TTS_CACHE_SIZE", 128)

    # Optional Mode B Cloud credentials
    deepl_api_key: str = os.getenv("DEEPL_API_KEY", "").strip()
    deepl_api_url: str = (
        os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate").strip()
        or "https://api-free.deepl.com/v2/translate"
    )
    google_translate_api_key: str = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "").strip()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()


settings = Settings()
