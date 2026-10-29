# LinguaStream Live — Real-Time Simultaneous Language Translator

A Python-powered (`FastAPI` + full-duplex `WebSockets` + `Faster-Whisper` + `Edge-TTS`) simultaneous speech translation web application that translates **live while you are speaking** — without waiting for sentence completion or a Stop button.

---

## Key Features

- **Mandatory Hardware Microphone Gate**: Locks the studio workspace on load until browser microphone permission is granted (with hardware `echoCancellation`, `noiseSuppression`, and `autoGainControl`), displaying a step-by-step browser troubleshooting guide if denied.
- **True "While-Speaking" Provisional & Committed Stream**:
  - **Provisional Stream**: Words appear and refine dynamically in-place (`200–500 ms`, `<150 ms` in Hybrid mode) inside glowing live bubbles as sentence grammar evolves.
  - **Committed Stream**: Clauses lock automatically into the permanent dual-pane transcript upon `> 800 ms` of silence or at `15+` words on a natural punctuation boundary.
- **Dual ASR Streaming Engines (Switchable Live in Header)**:
  1. **⚡ Ultra-Fast Hybrid Stream** *(Default — Zero GPU Required)*: Browser live interim speech recognition streams tokens over WebSockets to Python for debounced, LRU-cached async translation and neural TTS.
  2. **🎙️ Server-Side Whisper PCM Stream**: Inline `AudioWorklet` resamples microphone audio to `16 kHz` mono `Int16` PCM chunks (`160 ms`) and streams binary frames to Python's `RollingAudioBuffer` + `Silero/RMS VAD` + `Faster-Whisper` (`int8` on CPU / `float16` on CUDA).
- **Neural Live TTS & Smart Echo Ducking**: Synthesizes natural target-language speech in memory via Microsoft `edge-tts` across 14 global languages, with automatic microphone ducking to prevent speaker feedback loops.
- **Two-Way Conversation (Bi-Directional) Mode & Quick-Swap**: Instant language inversion plus dual-speaker shared-mic routing.
- **60 FPS Live Canvas Visualizer & Transcript Export**: Real-time frequency spectrum + dB meter, one-click per-card TTS replay, `.TXT` conversation export, and `.SRT` bilingual subtitle export.

---

## Project Structure

```text
Language Translator/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point, REST routes, and /ws/translate state machine
│   ├── config.py               # Runtime settings, .env loader, and 14-language ASR/MT/TTS registry
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_stream.py     # Rolling 16kHz PCM ring buffer, pre-roll, and Hybrid Silero/RMS VAD
│   │   ├── transcriber.py      # Clause splitter, HybridSegmentTracker, and async Faster-Whisper engine
│   │   ├── translator.py       # Async translation orchestrator with 0ms LRU cache & Mode A/B providers
│   │   └── tts_engine.py       # In-memory Edge-TTS async neural MP3 generator
│   └── static/
│       ├── index.html          # Studio UI with mandatory mic gate & split-screen translation panes
│       ├── styles.css          # Glassmorphic surfaces, provisional glow animations & equalizer bars
│       └── app.js              # AudioWorklet 16kHz PCM capture, 60fps visualizer, WS client & exporters
├── tests/
│   └── test_app.py             # Automated pytest suite for VAD, LRU MT, REST, and WebSocket concurrency
├── .env.example                # Mode A (free default) & Mode B (optional cloud API) configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Setup and execution instructions
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure `.env`

Out of the box (**Mode A**), the app runs **100% free with zero paid API keys**. To customize Whisper model sizes or enable optional **Mode B** Cloud providers (`DEEPL_API_KEY` or `GOOGLE_TRANSLATE_API_KEY`), copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in Chrome or Edge, click **Grant Access & Start**, and begin speaking naturally.

---

## Running Automated Tests

```bash
pytest -v
```
