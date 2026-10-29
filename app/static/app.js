/**
 * TR-01 FIELD INTERPRETER — Real-Time Simultaneous Language Translator
 *
 * Aesthetic: Field Recorder (#F4F3EF Warm Paper, #141413 Carbon Ink, #FF4F00 International Orange)
 * Features:
 *  - Mandatory Hardware Microphone Gate & Hardware AEC Setup
 *  - 60 FPS Interactive 3-Mode OLED Visualizer (Peak-Hold Spectrum Bars / Analog Scope / Dual Mirror)
 *  - Voice-Driven Spinning Tape Spools & Live Running Timecode
 *  - One-Tap Quick-Tune Language Matrix, Interactive Demo Phrase Streamer & Saved Phrasebook (★)
 *  - Large-Type Across-the-Table Presenter Mode & Synthesized Tactile Switch SFX
 *  - Dual ASR Engines (Ultra-Fast Hybrid Stream + 16kHz AudioWorklet Server Faster-Whisper)
 *  - Sequential Neural TTS Queue with Smart Echo Ducking & .TXT / .SRT Exporters
 */

(() => {
  "use strict";

  const DEFAULT_LANGUAGES = [
    { code: "auto", name: "Auto-Detect", native_name: "Auto", bcp47: "en-US", allow_as_target: false },
    { code: "en", name: "English", native_name: "English", bcp47: "en-US", allow_as_target: true },
    { code: "es", name: "Spanish", native_name: "Español", bcp47: "es-ES", allow_as_target: true },
    { code: "hi", name: "Hindi", native_name: "हिन्दी", bcp47: "hi-IN", allow_as_target: true },
    { code: "fr", name: "French", native_name: "Français", bcp47: "fr-FR", allow_as_target: true },
    { code: "de", name: "German", native_name: "Deutsch", bcp47: "de-DE", allow_as_target: true },
    { code: "zh-CN", name: "Mandarin Chinese", native_name: "中文", bcp47: "zh-CN", allow_as_target: true },
    { code: "ja", name: "Japanese", native_name: "日本語", bcp47: "ja-JP", allow_as_target: true },
    { code: "ar", name: "Arabic", native_name: "العربية", bcp47: "ar-SA", allow_as_target: true },
    { code: "pt", name: "Portuguese", native_name: "Português", bcp47: "pt-BR", allow_as_target: true },
    { code: "ru", name: "Russian", native_name: "Русский", bcp47: "ru-RU", allow_as_target: true },
    { code: "ko", name: "Korean", native_name: "한국어", bcp47: "ko-KR", allow_as_target: true },
    { code: "it", name: "Italian", native_name: "Italiano", bcp47: "it-IT", allow_as_target: true },
    { code: "tr", name: "Turkish", native_name: "Türkçe", bcp47: "tr-TR", allow_as_target: true },
    { code: "nl", name: "Dutch", native_name: "Nederlands", bcp47: "nl-NL", allow_as_target: true },
  ];

  const QUICK_TUNE_CODES = ["es", "hi", "fr", "de", "ja", "zh-CN", "ar", "ko", "it", "pt"];

  const DEMO_PHRASES = [
    "Hello! Could you recommend a great local coffee shop nearby?",
    "We are testing real-time simultaneous translation while speaking naturally.",
    "Excuse me, which platform does the express train depart from?",
    "Let's walk through the product roadmap and launch timeline together.",
    "It is wonderful meeting you today, thank you for your hospitality!",
    "Can we switch languages on the fly without pressing a stop button?",
  ];

  // Inline AudioWorklet processor downsampling native 44.1/48kHz audio to 16000Hz Int16 PCM (160ms frames)
  const WORKLET_CODE = `
    class Pcm16kWorkletProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.targetRate = 16000;
        this.chunkSize = 2560; // 160ms @ 16kHz
        this.outBuffer = new Int16Array(this.chunkSize);
        this.outOffset = 0;
        this.resampleAccum = 0;
        this.sampleSum = 0;
        this.sampleCount = 0;
      }

      process(inputs) {
        const input = inputs[0];
        if (!input || !input[0] || input[0].length === 0) {
          return true;
        }
        const channel = input[0];
        const ratio = sampleRate / this.targetRate;

        for (let i = 0; i < channel.length; i++) {
          this.sampleSum += channel[i];
          this.sampleCount += 1;
          this.resampleAccum += 1;

          if (this.resampleAccum >= ratio) {
            this.resampleAccum -= ratio;
            const avg = this.sampleCount > 0 ? (this.sampleSum / this.sampleCount) : channel[i];
            this.sampleSum = 0;
            this.sampleCount = 0;

            const clamped = Math.max(-1, Math.min(1, avg));
            this.outBuffer[this.outOffset++] = clamped < 0
              ? Math.round(clamped * 32768)
              : Math.round(clamped * 32767);

            if (this.outOffset >= this.chunkSize) {
              const packet = this.outBuffer.slice(0);
              this.port.postMessage(packet.buffer, [packet.buffer]);
              this.outOffset = 0;
            }
          }
        }
        return true;
      }
    }
    registerProcessor("pcm-16khz-processor", Pcm16kWorkletProcessor);
  `;

  const VIZ_MODES = ["BARS", "SCOPE", "MIRROR"];

  const state = {
    languages: [...DEFAULT_LANGUAGES],
    langMap: new Map(),
    micGranted: false,
    mediaStream: null,
    audioCtx: null,
    analyser: null,
    workletNode: null,
    scriptFallbackNode: null,

    ws: null,
    wsConnected: false,
    reconnectAttempt: 0,
    reconnectTimer: null,

    engineMode: "hybrid", // "hybrid" | "whisper"
    webSpeechSupported: Boolean(window.SpeechRecognition || window.webkitSpeechRecognition),
    recognition: null,
    recognitionRunning: false,
    recognitionEpoch: 1,

    isListening: true,
    sourceLang: "auto",
    targetLang: "es",
    lastDetectedSource: "en",
    bidirectional: false,
    activeSpeaker: "auto",

    ttsEnabled: true,
    echoDucking: true,
    sfxEnabled: true,
    focusTypeMode: false,
    showStarredOnly: false,

    isPlayingTTS: false,
    duckingCooldownUntil: 0,
    ttsQueue: [],
    currentAudio: null,

    vizModeIndex: 0,
    reelAngle: 0,
    peakHolds: new Float32Array(56),
    demoPhraseCursor: 0,
    demoSimRunning: false,

    sessionStartEpochMs: Date.now(),
    totalWordsSpoken: 0,
    exploredLangs: new Set(["es"]),
    starredIds: new Set(),
    activeProvisionalId: null,
    provisionalStartMap: new Map(),
    utterances: [],
    audioCacheByUtt: new Map(),
    currentRms: 0,
  };

  const dom = {
    micGateOverlay: document.getElementById("mic-gate-overlay"),
    micTroubleshootBox: document.getElementById("mic-troubleshoot-box"),
    micErrorHeading: document.getElementById("mic-error-heading"),
    micErrorDetail: document.getElementById("mic-error-detail"),
    grantMicBtn: document.getElementById("grant-mic-btn"),
    grantMicBtnLabel: document.getElementById("grant-mic-btn-label"),
    appShell: document.getElementById("app-shell"),

    sessionStatsSummary: document.getElementById("session-stats-summary"),
    engineHybridBtn: document.getElementById("engine-hybrid-btn"),
    engineWhisperBtn: document.getElementById("engine-whisper-btn"),
    whisperStatusPill: document.getElementById("whisper-status-pill"),
    whisperStatusText: document.getElementById("whisper-status-text"),
    latencyValue: document.getElementById("latency-value"),
    wsStatusDot: document.getElementById("ws-status-dot"),
    wsStatusLabel: document.getElementById("ws-status-label"),

    reelLeft: document.getElementById("reel-left"),
    reelRight: document.getElementById("reel-right"),
    micLiveDot: document.getElementById("mic-live-dot"),
    captureStateLabel: document.getElementById("capture-state-label"),
    timecodeDisplay: document.getElementById("timecode-display"),
    rmsMeterText: document.getElementById("rms-meter-text"),
    visualizerBox: document.getElementById("visualizer-box"),
    waveformCanvas: document.getElementById("waveform-canvas"),
    vizModeLabel: document.getElementById("viz-mode-label"),
    toggleFocusBtn: document.getElementById("toggle-focus-btn"),
    toggleSfxBtn: document.getElementById("toggle-sfx-btn"),

    sourceLangSelect: document.getElementById("source-lang-select"),
    targetLangSelect: document.getElementById("target-lang-select"),
    swapLangsBtn: document.getElementById("swap-langs-btn"),
    swapIcon: document.getElementById("swap-icon"),
    bidirectionalBtn: document.getElementById("bidirectional-btn"),
    bidirIndicator: document.getElementById("bidir-indicator"),
    speakerTurnGroup: document.getElementById("speaker-turn-group"),
    speakerABtn: document.getElementById("speaker-a-btn"),
    speakerBBtn: document.getElementById("speaker-b-btn"),
    quickLangBar: document.getElementById("quick-lang-bar"),
    filterStarredBtn: document.getElementById("filter-starred-btn"),
    starredCount: document.getElementById("starred-count"),
    shufflePromptsBtn: document.getElementById("shuffle-prompts-btn"),

    transcriptMain: document.getElementById("transcript-main"),
    activeSourceBadge: document.getElementById("active-source-badge"),
    activeTargetBadge: document.getElementById("active-target-badge"),
    utteranceCountBadge: document.getElementById("utterance-count-badge"),
    ttsPlayingIndicator: document.getElementById("tts-playing-indicator"),

    sourcePaneScroll: document.getElementById("source-pane-scroll"),
    sourceEmptyState: document.getElementById("source-empty-state"),
    promptChipsContainer: document.getElementById("prompt-chips-container"),
    sourceHistoryList: document.getElementById("source-history-list"),
    sourceProvisionalCard: document.getElementById("source-provisional-card"),
    sourceProvisionalLang: document.getElementById("source-provisional-lang"),
    sourceProvisionalText: document.getElementById("source-provisional-text"),

    targetPaneScroll: document.getElementById("target-pane-scroll"),
    targetEmptyState: document.getElementById("target-empty-state"),
    targetHistoryList: document.getElementById("target-history-list"),
    targetProvisionalCard: document.getElementById("target-provisional-card"),
    targetProvisionalLatency: document.getElementById("target-provisional-latency"),
    targetProvisionalText: document.getElementById("target-provisional-text"),

    toggleListenBtn: document.getElementById("toggle-listen-btn"),
    listenBtnIcon: document.getElementById("listen-btn-icon"),
    listenBtnLabel: document.getElementById("listen-btn-label"),
    toggleTtsBtn: document.getElementById("toggle-tts-btn"),
    ttsBtnIcon: document.getElementById("tts-btn-icon"),
    ttsBtnLabel: document.getElementById("tts-btn-label"),
    toggleDuckingBtn: document.getElementById("toggle-ducking-btn"),
    duckingBtnLabel: document.getElementById("ducking-btn-label"),
    clearTranscriptBtn: document.getElementById("clear-transcript-btn"),
    exportTxtBtn: document.getElementById("export-txt-btn"),
    exportSrtBtn: document.getElementById("export-srt-btn"),
    toastContainer: document.getElementById("toast-container"),
  };

  // =========================================================================
  // Synthesized Tactile Field-Recorder Click SFX & Toast Helpers
  // =========================================================================

  function playTactileClick(freq = 1150) {
    if (!state.sfxEnabled || !state.audioCtx) return;
    try {
      if (state.audioCtx.state === "suspended") return;
      const now = state.audioCtx.currentTime;
      const osc = state.audioCtx.createOscillator();
      const gain = state.audioCtx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(freq, now);
      osc.frequency.exponentialRampToValueAtTime(220, now + 0.016);
      gain.gain.setValueAtTime(0.03, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.018);
      osc.connect(gain);
      gain.connect(state.audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.02);
    } catch {
      // Ignore audio SFX errors
    }
  }

  function showToast(message, tone = "info") {
    const tones = {
      info: "bg-[#141413] text-[#F4F3EF] border-[#141413]",
      success: "bg-[#141413] text-[#F4F3EF] border-[#FF4F00]",
      warn: "bg-[#FFF3EE] text-[#141413] border-[#FF4F00]",
      error: "bg-[#FF4F00] text-white border-[#141413]",
    };
    const el = document.createElement("div");
    el.className = `pointer-events-auto px-3.5 py-2 rounded-xl border-[1.5px] font-mono text-[11px] font-semibold uppercase tracking-wider shadow-lg transition-all duration-300 ${tones[tone] || tones.info}`;
    el.textContent = message;
    dom.toastContainer.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(6px)";
      setTimeout(() => el.remove(), 260);
    }, 3000);
  }

  function rebuildLanguageMap() {
    state.langMap.clear();
    for (const lang of state.languages) {
      state.langMap.set(lang.code, lang);
    }
  }

  function getLangLabel(code) {
    const spec = state.langMap.get(code);
    if (!spec) return (code || "auto").toUpperCase();
    return spec.name;
  }

  function getLangBcp47(code) {
    if (code === "auto") {
      return navigator.language || "en-US";
    }
    const spec = state.langMap.get(code);
    return spec ? spec.bcp47 : "en-US";
  }

  function formatWallClock(date = new Date()) {
    return date.toTimeString().split(" ")[0];
  }

  function formatTimecode(ms) {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const h = String(Math.floor(totalSec / 3600)).padStart(2, "0");
    const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
    const s = String(totalSec % 60).padStart(2, "0");
    return `TC ${h}:${m}:${s}`;
  }

  function formatSrtTimestamp(ms) {
    const clamped = Math.max(0, Math.floor(ms));
    const hours = Math.floor(clamped / 3600000);
    const minutes = Math.floor((clamped % 3600000) / 60000);
    const seconds = Math.floor((clamped % 60000) / 1000);
    const millis = clamped % 1000;
    const pad2 = (n) => String(n).padStart(2, "0");
    const pad3 = (n) => String(n).padStart(3, "0");
    return `${pad2(hours)}:${pad2(minutes)}:${pad2(seconds)},${pad3(millis)}`;
  }

  function isMicDuckedNow() {
    if (!state.echoDucking) return false;
    return state.isPlayingTTS || Date.now() < state.duckingCooldownUntil;
  }

  function updateSessionStatsHeader() {
    const elapsedMin = Math.max(0.15, (Date.now() - state.sessionStartEpochMs) / 60000);
    const wpm = state.totalWordsSpoken > 0 ? Math.round(state.totalWordsSpoken / elapsedMin) : 0;
    const langCount = state.exploredLangs.size;
    dom.sessionStatsSummary.textContent = `${state.totalWordsSpoken} WORDS // ${wpm} WPM // ${langCount} ${langCount === 1 ? "LANG" : "LANGS"}`;
  }

  // =========================================================================
  // Language Dropdowns, Quick-Tune Matrix & Icebreaker Prompt Chips
  // =========================================================================

  function populateLanguageSelectors() {
    rebuildLanguageMap();
    const prevSource = state.sourceLang;
    const prevTarget = state.targetLang;

    dom.sourceLangSelect.innerHTML = "";
    dom.targetLangSelect.innerHTML = "";

    for (const lang of state.languages) {
      const label = lang.code === "auto"
        ? "✦ Auto-Detect Spoken"
        : `${lang.name} (${lang.native_name})`;

      const srcOpt = document.createElement("option");
      srcOpt.value = lang.code;
      srcOpt.textContent = label;
      dom.sourceLangSelect.appendChild(srcOpt);

      if (lang.allow_as_target !== false) {
        const tgtOpt = document.createElement("option");
        tgtOpt.value = lang.code;
        tgtOpt.textContent = `${lang.name} (${lang.native_name})`;
        dom.targetLangSelect.appendChild(tgtOpt);
      }
    }

    dom.sourceLangSelect.value = state.langMap.has(prevSource) ? prevSource : "auto";
    dom.targetLangSelect.value = state.langMap.has(prevTarget) && prevTarget !== "auto" ? prevTarget : "es";
    renderQuickTuneBar();
    updateLanguageBadges();
  }

  function renderQuickTuneBar() {
    dom.quickLangBar.innerHTML = "";
    for (const code of QUICK_TUNE_CODES) {
      const spec = state.langMap.get(code);
      if (!spec) continue;
      const btn = document.createElement("button");
      btn.type = "button";
      const shortCode = code === "zh-CN" ? "ZH" : code.toUpperCase();
      btn.className = `quick-lang-pill ${state.targetLang === code ? "active" : ""}`;
      btn.title = `Instant switch output to ${spec.name} (${spec.native_name})`;
      btn.textContent = `${shortCode} · ${spec.native_name}`;
      btn.onclick = () => {
        playTactileClick(1300);
        setTargetLanguage(code);
      };
      dom.quickLangBar.appendChild(btn);
    }
  }

  function setTargetLanguage(code) {
    if (!state.langMap.has(code) || code === "auto") return;
    state.targetLang = code;
    state.exploredLangs.add(code);
    dom.targetLangSelect.value = code;
    renderQuickTuneBar();
    updateLanguageBadges();
    updateSessionStatsHeader();
    pushConfigToServer();
    if (state.bidirectional && state.engineMode === "hybrid" && state.isListening) {
      startRecognitionEngine();
    }
  }

  function renderPromptChips() {
    dom.promptChipsContainer.innerHTML = "";
    const picks = DEMO_PHRASES.slice(0, 3);
    for (const phrase of picks) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "prompt-chip px-3 py-2 rounded-xl text-left text-xs font-medium leading-snug max-w-[260px]";
      chip.innerHTML = `<span class="font-mono text-[10px] font-bold text-[#FF4F00] block mb-0.5">▶ TEST LIVE</span>"${escapeHtml(phrase)}"`;
      chip.onclick = () => {
        playTactileClick(1250);
        simulateLiveDemoPhrase(phrase);
      };
      dom.promptChipsContainer.appendChild(chip);
    }
  }

  async function simulateLiveDemoPhrase(customPhrase) {
    if (state.demoSimRunning) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      showToast("Connecting live stream — try in a moment", "warn");
      return;
    }
    state.demoSimRunning = true;
    const phrase = customPhrase || DEMO_PHRASES[state.demoPhraseCursor % DEMO_PHRASES.length];
    state.demoPhraseCursor += 1;

    const words = phrase.split(" ");
    const segId = `demo_${Date.now()}`;
    const steps = [
      words.slice(0, Math.min(3, words.length)).join(" "),
      words.slice(0, Math.min(6, words.length)).join(" "),
      phrase,
    ];

    try {
      for (let i = 0; i < steps.length - 1; i++) {
        sendWsJson({
          action: "interim_speech",
          text: steps[i],
          is_final: false,
          segment_id: segId,
        });
        await new Promise((r) => setTimeout(r, 230));
      }
      sendWsJson({
        action: "interim_speech",
        text: phrase,
        is_final: true,
        segment_id: segId,
      });
    } finally {
      setTimeout(() => { state.demoSimRunning = false; }, 350);
    }
  }

  async function fetchServerLanguages() {
    try {
      const resp = await fetch("/api/languages");
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data.languages) && data.languages.length > 0) {
        state.languages = data.languages;
        populateLanguageSelectors();
      }
    } catch {
      // Built-in DEFAULT_LANGUAGES already populated
    }
  }

  function updateLanguageBadges() {
    const srcName = getLangLabel(state.sourceLang).toUpperCase();
    const tgtName = getLangLabel(state.targetLang).toUpperCase();
    dom.activeSourceBadge.textContent = state.bidirectional
      ? `${srcName === "AUTO-DETECT" ? "ENGLISH" : srcName} ⇄ ${tgtName}`
      : srcName;
    dom.activeTargetBadge.textContent = state.bidirectional
      ? "TWO-WAY AUTO"
      : tgtName;

    const langACode = state.sourceLang === "auto" ? "en" : state.sourceLang;
    dom.speakerABtn.textContent = `SPK A (${langACode.toUpperCase()})`;
    dom.speakerBBtn.textContent = `SPK B (${state.targetLang.toUpperCase()})`;
  }

  // =========================================================================
  // Mandatory Microphone Gate & Web Audio Capture (Section 2.1, 3.1, 7.4)
  // =========================================================================

  function lockStudioUI(errorTitle, errorMessage) {
    state.micGranted = false;
    dom.micGateOverlay.classList.remove("hidden", "opacity-0", "pointer-events-none");
    dom.appShell.classList.add("opacity-25", "pointer-events-none", "select-none");

    if (errorTitle) {
      dom.micTroubleshootBox.classList.remove("hidden");
      dom.micErrorHeading.textContent = errorTitle;
      if (errorMessage) {
        dom.micErrorDetail.textContent = errorMessage;
      }
      dom.grantMicBtnLabel.textContent = "Retry Microphone Check";
    }
  }

  function unlockStudioUI() {
    state.micGranted = true;
    dom.micTroubleshootBox.classList.add("hidden");
    dom.micGateOverlay.classList.add("opacity-0", "pointer-events-none");
    setTimeout(() => dom.micGateOverlay.classList.add("hidden"), 300);
    dom.appShell.classList.remove("opacity-25", "pointer-events-none", "select-none");
  }

  async function requestMicrophoneAndUnlock(isAutoCheck = false) {
    dom.grantMicBtn.disabled = true;
    dom.grantMicBtnLabel.textContent = "Arming Microphone...";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      dom.grantMicBtn.disabled = false;
      lockStudioUI(
        "Browser Audio API Unavailable",
        "navigator.mediaDevices.getUserMedia is unavailable. Open via http://localhost:8000 or HTTPS."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      state.mediaStream = stream;

      for (const track of stream.getAudioTracks()) {
        track.addEventListener("ended", () => {
          stopRecognitionEngine();
          lockStudioUI(
            "Microphone Disconnected",
            "Active microphone input ended or unplugged. Reconnect and press Retry."
          );
        });
      }

      if (navigator.permissions && navigator.permissions.query) {
        try {
          const permStatus = await navigator.permissions.query({ name: "microphone" });
          permStatus.onchange = () => {
            if (permStatus.state === "denied") {
              stopRecognitionEngine();
              lockStudioUI(
                "Microphone Permission Revoked",
                "Microphone permission was switched to Blocked in browser settings."
              );
            }
          };
        } catch {}
      }

      await setupWebAudioPipeline(stream);
      playTactileClick(1400);
      unlockStudioUI();
      connectWebSocket();

      if (!state.webSpeechSupported && state.engineMode === "hybrid") {
        setEngineMode("whisper", false);
        showToast("WebSpeech unavailable — switched to Whisper 16kHz PCM", "info");
      } else {
        syncEngineCaptureState();
      }
    } catch (err) {
      dom.grantMicBtn.disabled = false;
      const errName = (err && err.name) || "";
      if (errName === "NotAllowedError" || errName === "PermissionDeniedError") {
        lockStudioUI(
          "Microphone Permission Denied",
          "Browser microphone access was denied. Enable Microphone permission using the steps below:"
        );
      } else if (errName === "NotFoundError" || errName === "DevicesNotFoundError") {
        lockStudioUI(
          "No Microphone Detected",
          "No audio input device was found. Connect a microphone or headset and retry."
        );
      } else if (!isAutoCheck) {
        lockStudioUI(
          "Microphone Hardware Busy",
          `Unable to capture microphone (${errName || "Device locked"}). Check Windows Sound settings.`
        );
      } else {
        dom.grantMicBtnLabel.textContent = "Grant Access & Start";
      }
    }
  }

  async function setupWebAudioPipeline(stream) {
    if (state.audioCtx) {
      try { await state.audioCtx.close(); } catch {}
    }

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    state.audioCtx = ctx;

    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;
    source.connect(analyser);
    state.analyser = analyser;

    const silentSink = ctx.createGain();
    silentSink.gain.value = 0;
    silentSink.connect(ctx.destination);

    let workletReady = false;
    if (ctx.audioWorklet) {
      try {
        const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
        const blobUrl = URL.createObjectURL(blob);
        await ctx.audioWorklet.addModule(blobUrl);
        URL.revokeObjectURL(blobUrl);

        const workletNode = new AudioWorkletNode(ctx, "pcm-16khz-processor");
        workletNode.port.onmessage = (ev) => handleOutgoingPcmBuffer(ev.data);
        source.connect(workletNode);
        workletNode.connect(silentSink);
        state.workletNode = workletNode;
        workletReady = true;
      } catch (err) {
        console.warn("AudioWorklet fallback to ScriptProcessorNode:", err);
      }
    }

    if (!workletReady) {
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const ratio = ctx.sampleRate / 16000;
      let accum = 0;
      let sum = 0;
      let count = 0;
      const outBuf = new Int16Array(2560);
      let outIdx = 0;

      proc.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < ch.length; i++) {
          sum += ch[i];
          count++;
          accum++;
          if (accum >= ratio) {
            accum -= ratio;
            const avg = count > 0 ? sum / count : ch[i];
            sum = 0;
            count = 0;
            const c = Math.max(-1, Math.min(1, avg));
            outBuf[outIdx++] = c < 0 ? Math.round(c * 32768) : Math.round(c * 32767);
            if (outIdx >= outBuf.length) {
              const copy = outBuf.slice(0);
              handleOutgoingPcmBuffer(copy.buffer);
              outIdx = 0;
            }
          }
        }
      };
      source.connect(proc);
      proc.connect(silentSink);
      state.scriptFallbackNode = proc;
    }

    startFieldInstrumentLoop();
  }

  function handleOutgoingPcmBuffer(arrayBuffer) {
    if (!state.micGranted || !state.isListening) return;
    if (state.engineMode !== "whisper") return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (isMicDuckedNow()) return;

    state.ws.send(arrayBuffer);
  }

  // =========================================================================
  // 60 FPS OLED Oscilloscope, Peak-Hold Spectrum & Voice-Driven Tape Spools
  // =========================================================================

  function startFieldInstrumentLoop() {
    const canvas = dom.waveformCanvas;
    const c2d = canvas.getContext("2d");
    const freqData = new Uint8Array(state.analyser.frequencyBinCount);
    const timeData = new Uint8Array(state.analyser.fftSize);

    function render() {
      if (!state.analyser) return;
      requestAnimationFrame(render);

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const W = Math.max(1, Math.floor(rect.width * dpr));
      const H = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }

      state.analyser.getByteFrequencyData(freqData);
      state.analyser.getByteTimeDomainData(timeData);

      // Compute RMS & dB
      let sqSum = 0;
      for (let i = 0; i < timeData.length; i++) {
        const norm = (timeData[i] - 128) / 128;
        sqSum += norm * norm;
      }
      state.currentRms = Math.sqrt(sqSum / timeData.length);
      const db = Math.max(-60, Math.round(20 * Math.log10(Math.max(1e-4, state.currentRms))));
      dom.rmsMeterText.textContent = state.isListening ? `${db} dB • 16KHZ` : "STANDBY";
      dom.timecodeDisplay.textContent = formatTimecode(Date.now() - state.sessionStartEpochMs);

      const ducked = isMicDuckedNow();

      // Rotate tape spools dynamically proportional to voice energy!
      if (state.isListening && !ducked) {
        const speed = 0.8 + Math.min(14, state.currentRms * 26);
        state.reelAngle = (state.reelAngle + speed) % 360;
        const rot = `rotate(${state.reelAngle.toFixed(1)}deg)`;
        dom.reelLeft.style.transform = rot;
        dom.reelRight.style.transform = rot;
      }

      // Draw subtle OLED telemetry grid lines
      c2d.clearRect(0, 0, W, H);
      c2d.strokeStyle = "rgba(244, 243, 239, 0.06)";
      c2d.lineWidth = 1 * dpr;
      c2d.beginPath();
      c2d.moveTo(0, H / 2);
      c2d.lineTo(W, H / 2);
      c2d.stroke();

      const mode = VIZ_MODES[state.vizModeIndex];
      const active = state.isListening && !ducked;

      if (mode === "BARS") {
        const barCount = 52;
        const gap = 3 * dpr;
        const barW = Math.max(2 * dpr, (W - gap * (barCount - 1)) / barCount);

        for (let i = 0; i < barCount; i++) {
          const bin = Math.min(freqData.length - 1, Math.floor((i / barCount) * (freqData.length * 0.72)));
          const val = active ? freqData[bin] / 255 : (ducked ? (freqData[bin] / 255) * 0.2 : 0.03);
          const barH = Math.max(2 * dpr, val * (H * 0.86));

          if (val > state.peakHolds[i]) {
            state.peakHolds[i] = val;
          } else {
            state.peakHolds[i] = Math.max(0.03, state.peakHolds[i] - 0.012);
          }

          const x = i * (barW + gap);
          const y = H - barH - 2 * dpr;

          c2d.fillStyle = !state.isListening
            ? "rgba(140, 138, 132, 0.3)"
            : (val > 0.62 ? "#FF4F00" : "#F4F3EF");
          c2d.fillRect(x, y, barW, barH);

          // International Orange Peak-Hold Floating Cap
          if (state.isListening) {
            const peakY = H - Math.max(3 * dpr, state.peakHolds[i] * (H * 0.86)) - 4 * dpr;
            c2d.fillStyle = "#FF4F00";
            c2d.fillRect(x, Math.max(1, peakY), barW, 2 * dpr);
          }
        }
      } else if (mode === "SCOPE") {
        c2d.lineWidth = 2.2 * dpr;
        c2d.strokeStyle = !state.isListening ? "#8C8A84" : "#FF4F00";
        c2d.beginPath();
        const slice = W / timeData.length;
        for (let i = 0; i < timeData.length; i++) {
          const v = active ? (timeData[i] - 128) / 128 : 0;
          const y = H / 2 + v * (H * 0.44);
          const x = i * slice;
          if (i === 0) c2d.moveTo(x, y);
          else c2d.lineTo(x, y);
        }
        c2d.stroke();
      } else {
        // MIRROR mode
        const barCount = 56;
        const gap = 2.5 * dpr;
        const barW = Math.max(2 * dpr, (W - gap * (barCount - 1)) / barCount);
        const midY = H / 2;

        for (let i = 0; i < barCount; i++) {
          const bin = Math.min(freqData.length - 1, Math.floor((i / barCount) * (freqData.length * 0.7)));
          const val = active ? freqData[bin] / 255 : 0.04;
          const h = Math.max(2 * dpr, val * (H * 0.86));
          const x = i * (barW + gap);
          c2d.fillStyle = i % 3 === 0 || val > 0.45 ? "#FF4F00" : "#F4F3EF";
          c2d.fillRect(x, midY - h / 2, barW, h);
        }
      }

      updateCaptureStatusBadge(ducked);
    }

    requestAnimationFrame(render);
  }

  function updateCaptureStatusBadge(ducked) {
    if (!state.isListening) {
      dom.micLiveDot.className = "w-2.5 h-2.5 rounded-full bg-[#8C8A84]";
      dom.captureStateLabel.className = "text-xs font-bold tracking-wider uppercase text-[#8C8A84]";
      dom.captureStateLabel.textContent = "PAUSED // STANDBY";
      return;
    }
    if (ducked) {
      dom.micLiveDot.className = "w-2.5 h-2.5 rounded-full bg-amber-400";
      dom.captureStateLabel.className = "text-xs font-bold tracking-wider uppercase text-amber-300";
      dom.captureStateLabel.textContent = "VOICE OUT // MIC DUCKED";
      return;
    }
    if (state.currentRms > 0.025) {
      dom.micLiveDot.className = "w-2.5 h-2.5 rounded-full rec-dot-orange";
      dom.captureStateLabel.className = "text-xs font-bold tracking-wider uppercase text-[#FF4F00]";
      dom.captureStateLabel.textContent = "REC // CAPTURING VOICE";
    } else {
      dom.micLiveDot.className = "w-2.5 h-2.5 rounded-full rec-dot-orange";
      dom.captureStateLabel.className = "text-xs font-bold tracking-wider uppercase text-[#F4F3EF]";
      dom.captureStateLabel.textContent = state.engineMode === "hybrid"
        ? "ARMED // HYBRID LIVE"
        : "ARMED // WHISPER 16KHZ";
    }
  }

  // =========================================================================
  // Hybrid Browser Live SpeechRecognition Engine
  // =========================================================================

  function getActiveRecognitionLocale() {
    if (state.bidirectional) {
      if (state.activeSpeaker === "B") {
        return getLangBcp47(state.targetLang);
      }
      const langA = state.sourceLang === "auto" ? "en" : state.sourceLang;
      return getLangBcp47(langA);
    }
    return getLangBcp47(state.sourceLang);
  }

  function startRecognitionEngine() {
    if (!state.webSpeechSupported || state.engineMode !== "hybrid" || !state.isListening || !state.micGranted) {
      return;
    }
    stopRecognitionEngine();

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) return;

    const rec = new SpeechRec();
    state.recognitionEpoch += 1;
    const epoch = state.recognitionEpoch;

    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.lang = getActiveRecognitionLocale();

    rec.onstart = () => {
      state.recognitionRunning = true;
    };

    rec.onresult = (event) => {
      if (!state.isListening || state.engineMode !== "hybrid") return;
      if (isMicDuckedNow() && state.currentRms < 0.09) return;

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const transcript = (res[0] && res[0].transcript ? res[0].transcript : "").trim();
        if (!transcript) continue;

        sendWsJson({
          action: "interim_speech",
          text: transcript,
          is_final: Boolean(res.isFinal),
          segment_id: `${epoch}_${i}`,
        });
      }
    };

    rec.onerror = (ev) => {
      if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
        lockStudioUI(
          "Speech Recognition Blocked",
          "Microphone or speech recognition permission was blocked by the browser."
        );
      } else if (ev.error === "network") {
        setEngineMode("whisper", true);
        showToast("Browser speech offline — auto-switched to local Faster-Whisper", "warn");
      }
    };

    rec.onend = () => {
      state.recognitionRunning = false;
      if (
        state.micGranted &&
        state.isListening &&
        state.engineMode === "hybrid" &&
        state.recognition === rec
      ) {
        setTimeout(() => {
          if (state.isListening && state.engineMode === "hybrid" && state.micGranted) {
            startRecognitionEngine();
          }
        }, 120);
      }
    };

    state.recognition = rec;
    try {
      rec.start();
    } catch {}
  }

  function stopRecognitionEngine() {
    const rec = state.recognition;
    state.recognition = null;
    state.recognitionRunning = false;
    if (rec) {
      try {
        rec.onend = null;
        rec.abort();
      } catch {}
    }
  }

  function syncEngineCaptureState() {
    if (state.audioCtx && state.audioCtx.state === "suspended" && state.isListening) {
      state.audioCtx.resume().catch(() => {});
    }
    if (!state.isListening || !state.micGranted) {
      stopRecognitionEngine();
      return;
    }
    if (state.engineMode === "hybrid") {
      startRecognitionEngine();
    } else {
      stopRecognitionEngine();
    }
  }

  // =========================================================================
  // Full-Duplex WebSocket Client with Exponential Backoff
  // =========================================================================

  function setWsBadge(status) {
    if (status === "connected") {
      dom.wsStatusDot.className = "w-2 h-2 rounded-full bg-emerald-400";
      dom.wsStatusLabel.textContent = "ONLINE";
    } else if (status === "reconnecting") {
      dom.wsStatusDot.className = "w-2 h-2 rounded-full bg-[#FF4F00] animate-ping";
      dom.wsStatusLabel.textContent = "RETRYING";
    } else {
      dom.wsStatusDot.className = "w-2 h-2 rounded-full bg-rose-500";
      dom.wsStatusLabel.textContent = "OFFLINE";
    }
  }

  function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/translate`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    state.ws = ws;

    ws.onopen = () => {
      state.wsConnected = true;
      state.reconnectAttempt = 0;
      setWsBadge("connected");
      pushConfigToServer();
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      try {
        handleServerMessage(JSON.parse(event.data));
      } catch (err) {
        console.warn("Invalid server JSON:", err);
      }
    };

    ws.onclose = () => {
      state.wsConnected = false;
      setWsBadge("reconnecting");
      scheduleReconnect();
    };

    ws.onerror = () => ws.close();
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) return;
    const baseDelay = Math.min(8000, 500 * Math.pow(2, state.reconnectAttempt));
    const jitter = Math.floor(Math.random() * 250);
    state.reconnectAttempt += 1;
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connectWebSocket();
    }, baseDelay + jitter);
  }

  function sendWsJson(payload) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(payload));
    }
  }

  function pushConfigToServer() {
    sendWsJson({
      action: "config",
      source_lang: state.sourceLang,
      target_lang: state.targetLang,
      tts_enabled: state.ttsEnabled,
      engine_mode: state.engineMode,
      bidirectional: state.bidirectional,
      active_speaker: state.activeSpeaker,
    });
  }

  // =========================================================================
  // Provisional vs. Committed Transcript State Machine
  // =========================================================================

  function handleServerMessage(msg) {
    const status = msg.status;

    if (status === "config_ack" || status === "engine_status") {
      if (msg.whisper) updateWhisperStatusPill(msg.whisper);
      return;
    }

    if (status === "reset_ack") {
      clearProvisionalCards();
      return;
    }

    if (status === "partial") {
      renderPartialStream(msg);
      return;
    }

    if (status === "final") {
      renderCommittedUtterance(msg);
    }
  }

  function updateWhisperStatusPill(whisperInfo) {
    if (state.engineMode !== "whisper") {
      dom.whisperStatusPill.classList.add("hidden");
      dom.whisperStatusPill.classList.remove("flex");
      return;
    }
    dom.whisperStatusPill.classList.remove("hidden");
    dom.whisperStatusPill.classList.add("flex");

    if (whisperInfo.error) {
      dom.whisperStatusText.textContent = "WHISPER ERR";
    } else if (whisperInfo.loading) {
      dom.whisperStatusText.textContent = `LOADING WHISPER (${whisperInfo.model_size.toUpperCase()})...`;
    } else if (whisperInfo.loaded) {
      dom.whisperStatusText.textContent = `WHISPER ${whisperInfo.model_size.toUpperCase()} READY`;
    } else {
      dom.whisperStatusText.textContent = `WHISPER ${whisperInfo.model_size.toUpperCase()}`;
    }
  }

  function renderPartialStream(msg) {
    const uttId = msg.utterance_id;
    const orig = (msg.original_text || "").trim();
    const trans = (msg.translated_text || "").trim();
    if (!orig) return;

    dom.sourceEmptyState.classList.add("hidden");
    dom.targetEmptyState.classList.add("hidden");

    if (!state.provisionalStartMap.has(uttId)) {
      state.provisionalStartMap.set(uttId, Date.now() - state.sessionStartEpochMs);
    }
    state.activeProvisionalId = uttId;

    if (msg.source_lang && msg.source_lang !== "auto") {
      state.lastDetectedSource = msg.source_lang;
    }

    const latencyLabel = msg.cached ? `${msg.latency_ms} MS ⚡` : `${msg.latency_ms} MS`;
    dom.latencyValue.textContent = latencyLabel;

    dom.sourceProvisionalCard.classList.remove("hidden");
    dom.sourceProvisionalLang.textContent = (msg.source_lang || "auto").toUpperCase();
    dom.sourceProvisionalText.textContent = orig;

    dom.targetProvisionalCard.classList.remove("hidden");
    dom.targetProvisionalLatency.textContent = latencyLabel;
    dom.targetProvisionalText.textContent = trans || "...";

    scrollPanesToBottom();
  }

  function clearProvisionalCards() {
    state.activeProvisionalId = null;
    dom.sourceProvisionalCard.classList.add("hidden");
    dom.targetProvisionalCard.classList.add("hidden");
    dom.sourceProvisionalText.textContent = "";
    dom.targetProvisionalText.textContent = "";
  }

  function renderCommittedUtterance(msg) {
    const uttId = msg.utterance_id || `utt_${Date.now()}`;
    const orig = (msg.original_text || "").trim();
    const trans = (msg.translated_text || "").trim();
    if (!orig && !trans) return;

    dom.sourceEmptyState.classList.add("hidden");
    dom.targetEmptyState.classList.add("hidden");

    if (msg.source_lang && msg.source_lang !== "auto") {
      state.lastDetectedSource = msg.source_lang;
    }
    if (msg.target_lang) {
      state.exploredLangs.add(msg.target_lang);
    }

    if (!state.activeProvisionalId || state.activeProvisionalId === uttId) {
      clearProvisionalCards();
    }

    const nowRelMs = Date.now() - state.sessionStartEpochMs;
    const startRelMs = state.provisionalStartMap.get(uttId) ?? Math.max(0, nowRelMs - 1800);
    const endRelMs = Math.max(startRelMs + 900, nowRelMs);
    state.provisionalStartMap.delete(uttId);

    const latencyLabel = msg.cached ? `${msg.latency_ms} MS ⚡` : `${msg.latency_ms} MS`;
    dom.latencyValue.textContent = latencyLabel;

    if (msg.audio_base64) {
      state.audioCacheByUtt.set(uttId, msg.audio_base64);
    }

    const wordCount = orig.split(/\s+/).filter(Boolean).length;
    state.totalWordsSpoken += wordCount;
    updateSessionStatsHeader();

    const takeNumber = String(state.utterances.length + 1).padStart(2, "0");
    const record = {
      utterance_id: uttId,
      takeNumber,
      source_lang: msg.source_lang || state.sourceLang,
      target_lang: msg.target_lang || state.targetLang,
      speaker: msg.speaker || "A",
      original_text: orig,
      translated_text: trans,
      latency_ms: msg.latency_ms ?? 0,
      cached: Boolean(msg.cached),
      wallTime: formatWallClock(),
      startRelMs,
      endRelMs,
    };
    state.utterances.push(record);
    dom.utteranceCountBadge.textContent = `${String(state.utterances.length).padStart(2, "0")} ${state.utterances.length === 1 ? "TAKE" : "TAKES"}`;

    appendCommittedCardsToDom(record);
    scrollPanesToBottom();

    if (state.ttsEnabled && msg.audio_base64) {
      enqueueTtsAudio(msg.audio_base64);
    }
  }

  function appendCommittedCardsToDom(record) {
    const spkHtml = state.bidirectional
      ? `<span class="px-1.5 py-0.5 rounded bg-[#141413] text-[#F4F3EF] font-bold text-[9px]">SPK ${record.speaker}</span>`
      : "";

    // 1. Source Tape Card
    const srcCard = document.createElement("article");
    srcCard.className = "tape-entry-card rounded-xl p-3.5";
    srcCard.dataset.uttId = record.utterance_id;

    const srcHeader = document.createElement("div");
    srcHeader.className = "flex items-center justify-between font-mono text-[10px] text-[#8C8A84] mb-1.5";
    srcHeader.innerHTML = `
      <div class="flex items-center gap-1.5">
        <span class="font-bold text-[#141413]">#${record.takeNumber}</span>
        ${spkHtml}
        <span class="uppercase px-1.5 py-0.5 rounded bg-[#EAE8E1] text-[#141413] font-semibold">${escapeHtml(record.source_lang)}</span>
        <span>${escapeHtml(record.wallTime)}</span>
      </div>
    `;

    const srcCopyBtn = document.createElement("button");
    srcCopyBtn.type = "button";
    srcCopyBtn.className = "font-mono text-[10px] font-semibold text-[#8C8A84] hover:text-[#141413] px-1.5 py-0.5 rounded hover:bg-[#EAE8E1] transition-colors uppercase";
    srcCopyBtn.textContent = "COPY";
    srcCopyBtn.onclick = () => copyText(record.original_text, srcCopyBtn);
    srcHeader.appendChild(srcCopyBtn);

    const srcBody = document.createElement("p");
    srcBody.className = "clause-body-text font-speech text-[15px] sm:text-base text-[#141413] leading-relaxed";
    srcBody.textContent = record.original_text;

    srcCard.appendChild(srcHeader);
    srcCard.appendChild(srcBody);

    // 2. Target Tape Card
    const tgtCard = document.createElement("article");
    tgtCard.className = "tape-entry-card tape-entry-target rounded-xl p-3.5";
    tgtCard.dataset.uttId = record.utterance_id;

    const tgtHeader = document.createElement("div");
    tgtHeader.className = "flex items-center justify-between font-mono text-[10px] text-[#8C8A84] mb-1.5";
    tgtHeader.innerHTML = `
      <div class="flex items-center gap-1.5">
        <span class="font-bold text-[#FF4F00]">#${record.takeNumber}</span>
        <span class="uppercase px-1.5 py-0.5 rounded bg-[#141413] text-[#F4F3EF] font-bold">${escapeHtml(record.target_lang)}</span>
        <span>${record.latency_ms} MS${record.cached ? " ⚡" : ""}</span>
      </div>
    `;

    const actionsWrap = document.createElement("div");
    actionsWrap.className = "flex items-center gap-1";

    const starBtn = document.createElement("button");
    starBtn.type = "button";
    starBtn.title = "Save to starred phrasebook";
    starBtn.className = "font-mono text-xs px-1.5 py-0.5 rounded text-[#8C8A84] hover:text-[#FF4F00] transition-colors";
    starBtn.textContent = "☆";
    starBtn.onclick = () => toggleStarClause(record.utterance_id, starBtn, tgtCard);

    const replayBtn = document.createElement("button");
    replayBtn.type = "button";
    replayBtn.className = "font-mono text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-[#FF4F00]/10 hover:bg-[#FF4F00] text-[#FF4F00] hover:text-white transition-colors";
    replayBtn.textContent = "▶ PLAY";
    replayBtn.onclick = () => {
      playTactileClick(1100);
      replayUtteranceTts(record, replayBtn);
    };

    const tgtCopyBtn = document.createElement("button");
    tgtCopyBtn.type = "button";
    tgtCopyBtn.className = "font-mono text-[10px] font-semibold text-[#8C8A84] hover:text-[#141413] px-1.5 py-0.5 rounded hover:bg-[#EAE8E1] transition-colors uppercase";
    tgtCopyBtn.textContent = "COPY";
    tgtCopyBtn.onclick = () => copyText(record.translated_text, tgtCopyBtn);

    actionsWrap.appendChild(starBtn);
    actionsWrap.appendChild(replayBtn);
    actionsWrap.appendChild(tgtCopyBtn);
    tgtHeader.appendChild(actionsWrap);

    const tgtBody = document.createElement("p");
    tgtBody.className = "clause-body-text font-speech text-[15px] sm:text-base font-semibold text-[#141413] leading-relaxed";
    tgtBody.textContent = record.translated_text;

    tgtCard.appendChild(tgtHeader);
    tgtCard.appendChild(tgtBody);

    if (state.showStarredOnly && !state.starredIds.has(record.utterance_id)) {
      srcCard.classList.add("hidden");
      tgtCard.classList.add("hidden");
    }

    dom.sourceHistoryList.appendChild(srcCard);
    dom.targetHistoryList.appendChild(tgtCard);
  }

  function toggleStarClause(uttId, starBtn, tgtCard) {
    playTactileClick(1500);
    if (state.starredIds.has(uttId)) {
      state.starredIds.delete(uttId);
      starBtn.textContent = "☆";
      starBtn.classList.remove("text-[#FF4F00]");
      tgtCard.classList.remove("is-starred");
    } else {
      state.starredIds.add(uttId);
      starBtn.textContent = "★";
      starBtn.classList.add("text-[#FF4F00]");
      tgtCard.classList.add("is-starred");
    }
    dom.starredCount.textContent = String(state.starredIds.size);
    applyStarredFilter();
  }

  function applyStarredFilter() {
    const filterOn = state.showStarredOnly;
    dom.filterStarredBtn.classList.toggle("bg-[#FF4F00]", filterOn);
    dom.filterStarredBtn.classList.toggle("text-white", filterOn);
    dom.filterStarredBtn.classList.toggle("bg-[#EAE8E1]", !filterOn);
    dom.filterStarredBtn.classList.toggle("text-[#141413]", !filterOn);

    for (const list of [dom.sourceHistoryList, dom.targetHistoryList]) {
      for (const card of list.children) {
        const id = card.dataset.uttId;
        const visible = !filterOn || state.starredIds.has(id);
        card.classList.toggle("hidden", !visible);
      }
    }
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function copyText(text, btn) {
    playTactileClick(1200);
    try {
      await navigator.clipboard.writeText(text);
      const prev = btn.textContent;
      btn.textContent = "COPIED";
      setTimeout(() => { btn.textContent = prev; }, 1100);
    } catch {
      showToast("Clipboard unavailable", "warn");
    }
  }

  function scrollPanesToBottom() {
    dom.sourcePaneScroll.scrollTop = dom.sourcePaneScroll.scrollHeight;
    dom.targetPaneScroll.scrollTop = dom.targetPaneScroll.scrollHeight;
  }

  // =========================================================================
  // Neural TTS Playback Queue & Audio Ducking
  // =========================================================================

  function enqueueTtsAudio(base64Mp3) {
    if (!base64Mp3) return;
    state.ttsQueue.push(base64Mp3);
    if (!state.isPlayingTTS) {
      playNextInTtsQueue();
    }
  }

  function playNextInTtsQueue() {
    if (state.ttsQueue.length === 0) {
      state.isPlayingTTS = false;
      state.currentAudio = null;
      state.duckingCooldownUntil = Date.now() + 200;
      dom.ttsPlayingIndicator.classList.add("hidden");
      dom.ttsPlayingIndicator.classList.remove("flex");
      return;
    }

    state.isPlayingTTS = true;
    dom.ttsPlayingIndicator.classList.remove("hidden");
    dom.ttsPlayingIndicator.classList.add("flex");

    const b64 = state.ttsQueue.shift();
    const audio = new Audio(`data:audio/mp3;base64,${b64}`);
    state.currentAudio = audio;

    const finish = () => playNextInTtsQueue();
    audio.onended = finish;
    audio.onerror = finish;
    audio.play().catch(() => finish());
  }

  function stopAllTtsPlayback() {
    state.ttsQueue.length = 0;
    if (state.currentAudio) {
      try {
        state.currentAudio.pause();
        state.currentAudio.src = "";
      } catch {}
      state.currentAudio = null;
    }
    state.isPlayingTTS = false;
    dom.ttsPlayingIndicator.classList.add("hidden");
    dom.ttsPlayingIndicator.classList.remove("flex");
  }

  async function replayUtteranceTts(record, btn) {
    let b64 = state.audioCacheByUtt.get(record.utterance_id);
    if (!b64) {
      const origLabel = btn.textContent;
      btn.disabled = true;
      btn.textContent = "SYNC...";
      try {
        const resp = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: record.translated_text,
            target_lang: record.target_lang,
          }),
        });
        if (resp.ok) {
          const data = await resp.json();
          b64 = data.audio_base64;
          if (b64) state.audioCacheByUtt.set(record.utterance_id, b64);
        }
      } catch {
        showToast("TTS synthesis failed", "warn");
      } finally {
        btn.disabled = false;
        btn.textContent = origLabel;
      }
    }
    if (b64) enqueueTtsAudio(b64);
  }

  // =========================================================================
  // Transport Controls & Archival Exporters (.TXT & .SRT)
  // =========================================================================

  function setEngineMode(mode, notifyServer = true) {
    playTactileClick(1050);
    if (mode === "hybrid" && !state.webSpeechSupported) {
      showToast("WebSpeech unsupported here; using Whisper 16kHz PCM", "warn");
      mode = "whisper";
    }
    state.engineMode = mode;

    dom.engineHybridBtn.classList.toggle("active", mode === "hybrid");
    dom.engineWhisperBtn.classList.toggle("active", mode === "whisper");

    if (mode !== "whisper") {
      dom.whisperStatusPill.classList.add("hidden");
      dom.whisperStatusPill.classList.remove("flex");
    }

    if (notifyServer) {
      sendWsJson({ action: "flush" });
      pushConfigToServer();
    }
    syncEngineCaptureState();
  }

  function toggleListening() {
    playTactileClick(900);
    state.isListening = !state.isListening;
    if (state.isListening) {
      dom.toggleListenBtn.className = "hw-btn-orange px-4 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center gap-2";
      dom.listenBtnIcon.textContent = "●";
      dom.listenBtnLabel.textContent = "REC // PAUSE [SPACE]";
      syncEngineCaptureState();
    } else {
      dom.toggleListenBtn.className = "hw-btn px-4 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center gap-2";
      dom.listenBtnIcon.textContent = "▶";
      dom.listenBtnLabel.textContent = "RESUME REC [SPACE]";
      stopRecognitionEngine();
      sendWsJson({ action: "flush" });
    }
  }

  function swapLanguages() {
    playTactileClick(1350);
    dom.swapIcon.classList.toggle("rotate-180");
    const oldSrc = state.sourceLang;
    const oldTgt = state.targetLang;

    const newSrc = oldTgt;
    let newTgt = oldSrc === "auto" ? (state.lastDetectedSource || "en") : oldSrc;
    if (newTgt === newSrc) {
      newTgt = newSrc === "en" ? "es" : "en";
    }

    state.sourceLang = newSrc;
    state.targetLang = newTgt;
    state.exploredLangs.add(newTgt);
    dom.sourceLangSelect.value = newSrc;
    dom.targetLangSelect.value = newTgt;

    renderQuickTuneBar();
    updateLanguageBadges();
    updateSessionStatsHeader();
    pushConfigToServer();
    if (state.engineMode === "hybrid" && state.isListening) {
      startRecognitionEngine();
    }
  }

  function toggleBidirectionalMode() {
    playTactileClick(1200);
    state.bidirectional = !state.bidirectional;
    dom.bidirectionalBtn.setAttribute("aria-pressed", String(state.bidirectional));
    dom.bidirectionalBtn.classList.toggle("is-active", state.bidirectional);

    if (state.bidirectional) {
      dom.bidirIndicator.className = "w-2 h-2 rounded-full bg-[#FF4F00]";
      dom.speakerTurnGroup.classList.remove("hidden");
      dom.speakerTurnGroup.classList.add("flex");
      showToast("Two-Way Shared-Mic Mode Armed", "success");
    } else {
      state.activeSpeaker = "auto";
      dom.bidirIndicator.className = "w-2 h-2 rounded-full bg-[#8C8A84]";
      dom.speakerTurnGroup.classList.add("hidden");
      dom.speakerTurnGroup.classList.remove("flex");
    }

    updateLanguageBadges();
    pushConfigToServer();
    if (state.engineMode === "hybrid" && state.isListening) {
      startRecognitionEngine();
    }
  }

  function clearTranscript() {
    playTactileClick(800);
    state.utterances = [];
    state.starredIds.clear();
    state.showStarredOnly = false;
    state.totalWordsSpoken = 0;
    state.audioCacheByUtt.clear();
    state.provisionalStartMap.clear();
    state.sessionStartEpochMs = Date.now();

    dom.starredCount.textContent = "0";
    applyStarredFilter();
    updateSessionStatsHeader();

    dom.sourceHistoryList.innerHTML = "";
    dom.targetHistoryList.innerHTML = "";
    clearProvisionalCards();

    dom.sourceEmptyState.classList.remove("hidden");
    dom.targetEmptyState.classList.remove("hidden");
    dom.utteranceCountBadge.textContent = "00 TAKES";

    sendWsJson({ action: "reset" });
    showToast("Tape cleared — ready for new session", "info");
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function getExportCandidates() {
    if (state.showStarredOnly && state.starredIds.size > 0) {
      return state.utterances.filter((u) => state.starredIds.has(u.utterance_id));
    }
    return state.utterances;
  }

  function exportAsTxt() {
    playTactileClick(1100);
    const items = getExportCandidates();
    if (items.length === 0) {
      showToast("No recorded takes to export yet", "warn");
      return;
    }
    const header = [
      "======================================================================",
      "TR-01 FIELD INTERPRETER — SIMULTANEOUS TRANSLATION TAPE LOG",
      `Exported : ${new Date().toLocaleString()}`,
      `Takes    : ${items.length}`,
      "======================================================================",
      "",
    ];
    const body = items.map((u, idx) => {
      const spk = state.bidirectional ? ` [SPK ${u.speaker}]` : "";
      const star = state.starredIds.has(u.utterance_id) ? " ★" : "";
      return [
        `#${String(idx + 1).padStart(2, "0")}${star} [${u.wallTime}]${spk} (${u.source_lang.toUpperCase()} -> ${u.target_lang.toUpperCase()})`,
        `  SOURCE     : ${u.original_text}`,
        `  TRANSLATED : ${u.translated_text}`,
        "",
      ].join("\n");
    });

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    downloadBlob(`TR01_transcript_${stamp}.txt`, header.concat(body).join("\n"), "text/plain;charset=utf-8");
    showToast("Exported .TXT Field Log", "success");
  }

  function exportAsSrt() {
    playTactileClick(1100);
    const items = getExportCandidates();
    if (items.length === 0) {
      showToast("No recorded takes to export yet", "warn");
      return;
    }
    const blocks = items.map((u, idx) => {
      const start = formatSrtTimestamp(u.startRelMs);
      const end = formatSrtTimestamp(u.endRelMs);
      return `${idx + 1}\n${start} --> ${end}\n${u.translated_text}\n[${u.original_text}]\n`;
    });

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    downloadBlob(`TR01_subtitles_${stamp}.srt`, blocks.join("\n"), "application/x-subrip;charset=utf-8");
    showToast("Exported .SRT Bilingual Subtitles", "success");
  }

  // =========================================================================
  // Event Wiring & Bootstrapping
  // =========================================================================

  function bindEventListeners() {
    dom.grantMicBtn.addEventListener("click", () => requestMicrophoneAndUnlock(false));

    dom.engineHybridBtn.addEventListener("click", () => setEngineMode("hybrid"));
    dom.engineWhisperBtn.addEventListener("click", () => setEngineMode("whisper"));

    dom.visualizerBox.addEventListener("click", () => {
      playTactileClick(1450);
      state.vizModeIndex = (state.vizModeIndex + 1) % VIZ_MODES.length;
      dom.vizModeLabel.textContent = VIZ_MODES[state.vizModeIndex];
    });

    dom.toggleFocusBtn.addEventListener("click", () => {
      playTactileClick(1250);
      state.focusTypeMode = !state.focusTypeMode;
      dom.transcriptMain.classList.toggle("focus-type-active", state.focusTypeMode);
      dom.toggleFocusBtn.classList.toggle("bg-[#FF4F00]", state.focusTypeMode);
      dom.toggleFocusBtn.classList.toggle("text-white", state.focusTypeMode);
      dom.toggleFocusBtn.classList.toggle("bg-white/10", !state.focusTypeMode);
      showToast(state.focusTypeMode ? "Large-Type Presenter Mode ON" : "Standard Type View", "info");
    });

    dom.toggleSfxBtn.addEventListener("click", () => {
      state.sfxEnabled = !state.sfxEnabled;
      dom.toggleSfxBtn.setAttribute("aria-pressed", String(state.sfxEnabled));
      dom.toggleSfxBtn.textContent = state.sfxEnabled ? "SFX: ON" : "SFX: OFF";
      dom.toggleSfxBtn.classList.toggle("text-[#FF4F00]", state.sfxEnabled);
      dom.toggleSfxBtn.classList.toggle("text-[#8C8A84]", !state.sfxEnabled);
      if (state.sfxEnabled) playTactileClick(1300);
    });

    dom.sourceLangSelect.addEventListener("change", (e) => {
      playTactileClick(1150);
      state.sourceLang = e.target.value;
      updateLanguageBadges();
      pushConfigToServer();
      if (state.engineMode === "hybrid" && state.isListening) {
        startRecognitionEngine();
      }
    });

    dom.targetLangSelect.addEventListener("change", (e) => {
      playTactileClick(1150);
      setTargetLanguage(e.target.value);
    });

    dom.swapLangsBtn.addEventListener("click", swapLanguages);
    dom.bidirectionalBtn.addEventListener("click", toggleBidirectionalMode);

    dom.speakerTurnGroup.querySelectorAll(".speaker-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        playTactileClick(1280);
        state.activeSpeaker = btn.dataset.speaker || "auto";
        dom.speakerTurnGroup.querySelectorAll(".speaker-pill").forEach((b) => {
          b.classList.toggle("active", b === btn);
        });
        pushConfigToServer();
        if (state.engineMode === "hybrid" && state.isListening) {
          startRecognitionEngine();
        }
      });
    });

    dom.filterStarredBtn.addEventListener("click", () => {
      playTactileClick(1200);
      state.showStarredOnly = !state.showStarredOnly;
      applyStarredFilter();
    });

    dom.shufflePromptsBtn.addEventListener("click", () => {
      playTactileClick(1350);
      simulateLiveDemoPhrase();
    });

    dom.toggleListenBtn.addEventListener("click", toggleListening);

    dom.toggleTtsBtn.addEventListener("click", () => {
      playTactileClick(1100);
      state.ttsEnabled = !state.ttsEnabled;
      dom.toggleTtsBtn.setAttribute("aria-pressed", String(state.ttsEnabled));
      if (state.ttsEnabled) {
        dom.ttsBtnIcon.textContent = "🔊";
        dom.ttsBtnLabel.textContent = "VOICE: ON";
      } else {
        dom.ttsBtnIcon.textContent = "🔇";
        dom.ttsBtnLabel.textContent = "VOICE: MUTE";
        stopAllTtsPlayback();
      }
      pushConfigToServer();
    });

    dom.toggleDuckingBtn.addEventListener("click", () => {
      playTactileClick(1100);
      state.echoDucking = !state.echoDucking;
      dom.toggleDuckingBtn.setAttribute("aria-pressed", String(state.echoDucking));
      dom.duckingBtnLabel.textContent = state.echoDucking ? "DUCKING: ON" : "DUCKING: OFF";
      dom.toggleDuckingBtn.classList.toggle("is-active", state.echoDucking);
    });

    dom.clearTranscriptBtn.addEventListener("click", clearTranscript);
    dom.exportTxtBtn.addEventListener("click", exportAsTxt);
    dom.exportSrtBtn.addEventListener("click", exportAsSrt);

    window.addEventListener("keydown", (e) => {
      if (!state.micGranted) return;
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) {
        e.preventDefault();
        toggleListening();
      }
    });
  }

  function applyLandingQueryPresets() {
    const params = new URLSearchParams(window.location.search);
    const targetParam = params.get("target");
    if (targetParam && state.langMap.has(targetParam) && targetParam !== "auto") {
      state.targetLang = targetParam;
      state.exploredLangs.add(targetParam);
      dom.targetLangSelect.value = targetParam;
      renderQuickTuneBar();
      updateLanguageBadges();
    }

    const engineParam = params.get("engine");
    if (engineParam === "whisper" || engineParam === "hybrid") {
      setEngineMode(engineParam, false);
    }

    const modeParam = params.get("mode");
    if (modeParam === "twoway" && !state.bidirectional) {
      state.bidirectional = true;
      dom.bidirectionalBtn.setAttribute("aria-pressed", "true");
      dom.bidirectionalBtn.classList.add("is-active");
      dom.bidirIndicator.className = "w-2 h-2 rounded-full bg-[#FF4F00]";
      dom.speakerTurnGroup.classList.remove("hidden");
      dom.speakerTurnGroup.classList.add("flex");
      updateLanguageBadges();
    }
  }

  window.addEventListener("load", () => {
    populateLanguageSelectors();
    applyLandingQueryPresets();
    renderPromptChips();
    bindEventListeners();
    fetchServerLanguages();
    requestMicrophoneAndUnlock(true);
  });
})();
