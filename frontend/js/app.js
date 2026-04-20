/**
 * Streamline Vocals — Main orchestrator.
 *
 * Wires together:
 *   - Health polling (60 s interval)
 *   - Config load → model dropdowns + LoRA select
 *   - Source audio upload + pre-processing
 *   - Whisper transcription button
 *   - ACE-Step remix job (Generate button)
 *   - Result waveform rendering + bottom player integration
 *   - Save-to-output button
 *   - Run Pipeline button (process audio → ACE-Step → save)
 *   - Settings modal
 *   - Model switch (header button)
 */

// ------------------------------------------------------------------
// Toast helper (global)
// ------------------------------------------------------------------

const Toast = (() => {
  let _container = null;

  function _el() {
    if (!_container) _container = document.getElementById("toast-root");
    return _container;
  }

  function show(message, type = "info", duration = 3500) {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    _el()?.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("toast--visible"));
    if (duration > 0) setTimeout(() => dismiss(toast), duration);
    return toast;
  }

  function dismiss(toastEl) {
    toastEl.classList.add("fade-out");
    const cleanup = () => toastEl.remove();
    toastEl.addEventListener("animationend", cleanup, { once: true });
    setTimeout(cleanup, 400);
  }

  return { show, dismiss };
})();

// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------

let _acestepReady = false;
let _loadedModels = { dit_model: "", lm_model: "" };
let _activeJobId = null;
let _activeJobInterval = null;
let _resultRawPath = null;    // raw server-side path for save operation
let _resultAudioUrl = null;   // URL for bottom player
let _sourceOriginalName = ""; // input filename stem for output naming
let _resultOutputIndex = 1;   // incrementing index for saved files
let _resultWs = null;         // WaveSurfer for result waveform
let _rvcResultPath = null;    // server-side path for RVC output
let _rvcResultWs = null;      // WaveSurfer for RVC result waveform
let _pipelineRunning = false;

const _KEY_OPTIONS = [
  "C major", "C# major", "D major", "Eb major", "E major", "F major", "F# major", "G major", "Ab major", "A major", "Bb major", "B major",
  "C minor", "C# minor", "D minor", "Eb minor", "E minor", "F minor", "F# minor", "G minor", "Ab minor", "A minor", "Bb minor", "B minor",
];

const _VOCAL_LANGUAGE_OPTIONS = [
  ["en", "English"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["es", "Spanish"],
  ["fr", "French"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["ru", "Russian"],
  ["hi", "Hindi"],
  ["ar", "Arabic"],
];

// AudioUpload registry — keyed by widget element id
const _audioUploads = new Map();

// ------------------------------------------------------------------
// DOMContentLoaded
// ------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  _initAllSliders();
  _initAllAudioUploads();
  _populateMusicalDropdowns();
  _initAccordions();
  _bindTranscribeBtn();
  _bindGenerateBtn();
  _bindSaveResultBtn();
  _bindRvcRunBtn();
  _bindRvcSaveBtn();
  _bindRunPipelineBtn();
  _bindRefreshBtn();
  _bindSwitchModelsBtn();
  _bindSettings();
  _bindBrowseBtn();
  _setGenerateReady(false);
  _applyLoraSlotPatchUi();
  _applyUiTheme();

  // Initial health + config fetch
  _pollHealth();
  setTimeout(_pollHealth, 0); // immediate first poll
  setInterval(_pollHealth, 60_000);

  try {
    const config = await API.config();
    _populateModelSelects(config);
    await _syncModelSelectors();
  } catch {
    Toast.show("Could not load config from server", "warning");
  }

  // Populate RVC dropdowns
  try {
    const rvcCfg = await API.rvcConfig();
    _populateRvcSelects(rvcCfg);
    _setRvcStatus(rvcCfg.pth_files?.length > 0 ? "ok" : "partial",
      rvcCfg.pth_files?.length > 0 ? `${rvcCfg.pth_files.length} model(s)` : "No models");
  } catch {
    Toast.show("Could not load RVC config", "warning");
    _setRvcStatus("error", "Unavailable");
  }
});

// ------------------------------------------------------------------
// Health polling
// ------------------------------------------------------------------

async function _pollHealth() {
  try {
    const h = await API.health();
    _acestepReady = Boolean(h.acestep_ready);
    _setAcestepStatus(
      h.acestep_api === "ok"
        ? (_acestepReady ? "ok" : "partial")
        : "error",
      h.acestep_api === "ok"
        ? (_acestepReady ? "Ready" : "Loading…")
        : "Unavailable"
    );
    _setGenerateReady(_acestepReady && !_activeJobId);
  } catch {
    _setAcestepStatus("error", "Unreachable");
    _setGenerateReady(false);
  }
}

function _setAcestepStatus(status, text) {
  const dot = document.getElementById("status-dot");
  const txt = document.getElementById("status-text");
  if (dot) dot.dataset.status = status;
  if (txt) txt.textContent = text;
}

function _setRvcStatus(status, text) {
  const dot = document.getElementById("rvc-status-dot");
  const txt = document.getElementById("rvc-status-text");
  if (dot) dot.dataset.status = status;
  if (txt) txt.textContent = `RVC: ${text}`;
}

// ------------------------------------------------------------------
// Config → dropdowns
// ------------------------------------------------------------------

function _populateModelSelects(config) {
  const ditSelect = document.getElementById("dit-model");
  if (ditSelect && config.checkpoints?.length) {
    ditSelect.innerHTML = "";
    const filtered = config.checkpoints.filter((n) => n.toLowerCase().includes("acestep-v15"));
    const items = filtered.length ? filtered : config.checkpoints;
    for (const name of items) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = name;
      ditSelect.appendChild(opt);
    }
  }

  // Populate all 4 LoRA slot selects
  const loraOptions = config.loras || [];
  for (let i = 1; i <= 4; i++) {
    const loraSelect = document.getElementById(`lora-select-${i}`);
    if (!loraSelect) continue;
    const currentVal = loraSelect.value;
    while (loraSelect.options.length > 1) loraSelect.remove(1);
    for (const name of loraOptions) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = name;
      loraSelect.appendChild(opt);
    }
    if (currentVal && loraOptions.includes(currentVal)) loraSelect.value = currentVal;
  }
}

function _populateMusicalDropdowns() {
  const keySel = document.getElementById("keyscale");
  if (keySel) {
    keySel.innerHTML = '<option value="">auto</option>';
    for (const key of _KEY_OPTIONS) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      keySel.appendChild(opt);
    }
  }

  const langSel = document.getElementById("vocal-language");
  if (langSel) {
    langSel.innerHTML = '<option value="unknown">auto / Instrumental</option>';
    for (const [code, label] of _VOCAL_LANGUAGE_OPTIONS) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = `${label} (${code})`;
      langSel.appendChild(opt);
    }
  }
}

async function _syncModelSelectors() {
  try {
    const state = await API.modelState();
    const loaded = state.loaded || {};
    const saved  = state.saved  || {};
    _loadedModels = { dit_model: loaded.dit_model || "", lm_model: loaded.lm_model || "" };

    const ditSel = document.getElementById("dit-model");
    const pick = (loaded, saved) => loaded || saved || "";
    if (ditSel) ditSel.value = pick(loaded.dit_model, saved.dit_model);
  } catch (err) {
    console.warn("Could not sync model selectors:", err);
  }
}

// ------------------------------------------------------------------
// Slider + Audio Upload init
// ------------------------------------------------------------------

function _initAllSliders() {
  document.querySelectorAll(".tick-slider").forEach((el) => {
    if (!el.id) return;
    new TickSlider(el);
  });
}

function _initAllAudioUploads() {
  document.querySelectorAll(".audio-upload").forEach((el) => {
    if (!el.id) return;
    const widget = new AudioUpload(el);
    _audioUploads.set(el.id, widget);

    // Audio-upload play buttons are handled natively by WaveSurfer inside AudioUpload.
    // No bottom-player routing needed.
  });

  // Track source audio filename for output naming
  const sourceWidget = _audioUploads.get("source-audio");
  if (sourceWidget) {
    const fileInput = sourceWidget.container.querySelector("input[type=file]");
    fileInput?.addEventListener("change", (ev) => {
      const file = ev.target.files?.[0];
      if (file) _sourceOriginalName = file.name;
    });
    // Also hook drop
    sourceWidget.container.querySelector(".audio-upload__dropzone")?.addEventListener("drop", (ev) => {
      const file = ev.dataTransfer?.files?.[0];
      if (file) _sourceOriginalName = file.name;
    });
  }

  // Exposed for AudioUpload widgets so one playback source stays active at a time.
  window.__streamlineStopAllPlayers = (exceptKey = "") => _stopAllPlayers(exceptKey);
}

// ------------------------------------------------------------------
// Accordions
// ------------------------------------------------------------------

function _initAccordions() {
  document.querySelectorAll(".accordion__toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const acc = btn.closest(".accordion");
      const isOpen = acc.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(isOpen));
    });
  });
  _initLoraCount();
}

function _initLoraCount() {
  const countInput = document.getElementById("lora-count");
  if (!countInput) return;
  _applyLoraCount(Math.max(1, Math.min(4, parseInt(countInput.value, 10) || 1)));
  countInput.addEventListener("input", () => {
    const count = Math.max(1, Math.min(4, parseInt(countInput.value, 10) || 1));
    _applyLoraCount(count);
  });
}

function _applyLoraCount(count) {
  for (let i = 1; i <= 4; i++) {
    const row = document.getElementById(`lora-row-${i}`);
    if (row) row.style.display = i <= count ? "" : "none";
  }
}

// ------------------------------------------------------------------
// Whisper Transcription
// ------------------------------------------------------------------

function _bindTranscribeBtn() {
  const btn = document.getElementById("transcribe-btn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const sourceWidget = _audioUploads.get("source-audio");
    const path = sourceWidget?.getTempPath();
    if (!path) {
      Toast.show("Upload source audio first", "warning");
      return;
    }

    btn.disabled = true;
    btn.textContent = "⏳ Transcribing…";
    try {
      const result = await API.transcribe({
        audio_path: path,
        whisper_model: _getSettings().whisperModel || "base",
      });
      const lyricsEl = document.getElementById("lyrics");
      if (lyricsEl && result.text) {
        lyricsEl.value = result.text;
        Toast.show("Transcription complete", "success");
      }
    } catch (err) {
      Toast.show(`Transcription failed: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = "✦ Transcribe";
    }
  });
}

// ------------------------------------------------------------------
// ACE-Step Generation
// ------------------------------------------------------------------

function _bindGenerateBtn() {
  document.getElementById("generate-btn")?.addEventListener("click", _onGenerate);
}

function _setGenerateReady(ready) {
  const btn = document.getElementById("generate-btn");
  const pipelineBtn = document.getElementById("run-pipeline-btn");
  if (btn) btn.disabled = !ready;
  if (pipelineBtn) pipelineBtn.disabled = !ready;
}

async function _onGenerate() {
  if (_activeJobId || _pipelineRunning) return;

  const sourceWidget = _audioUploads.get("source-audio");
  const refWidget    = _audioUploads.get("ref-audio");

  if (!sourceWidget?.getTempPath()) {
    Toast.show("Upload source audio first", "warning");
    return;
  }

  _setGenerateReady(false);
  _showProgress(true, "Processing audio…");

  // 1. Apply pre-processing (pitch shift, filter, gate)
  let processedPath = sourceWidget.getTempPath();
  try {
    const pitchShift = _sliderVal("ts-pitch-shift");
    const applyLowCut   = document.getElementById("apply-low-cut")?.checked   ?? false;
    const applyNoiseGate = document.getElementById("apply-noise-gate")?.checked ?? false;

    const processResult = await API.processAudio({
      audio_path: processedPath,
      pitch_shift_semitones: pitchShift,
      apply_low_cut: applyLowCut,
      apply_noise_gate: applyNoiseGate,
    });
    processedPath = processResult.processed_path;
  } catch (err) {
    Toast.show(`Audio processing failed: ${err.message}`, "error");
    _showProgress(false);
    _setGenerateReady(_acestepReady);
    return;
  }

  _showProgress(true, "Starting ACE-Step…");

  // 2. Build and run generate request (single output preview)
  const params = _buildGenerateParams(processedPath, refWidget?.getTempPath() ?? null, 1);
  try {
    const job = await _runGenerateJob(params, "acestep");
    const rawPaths = job.raw_audio_paths || [];
    const audioUrls = job.audio_urls || [];
    if (rawPaths.length && audioUrls.length) {
      _resultRawPath = rawPaths[0];
      _resultAudioUrl = audioUrls[0];
      _showResult(_resultAudioUrl);
    } else {
      Toast.show("Generation done but no audio was returned", "warning");
    }
  } catch (err) {
    Toast.show(`Generation failed: ${err.message}`, "error");
    _showProgress(false);
    _setGenerateReady(_acestepReady);
    return;
  }

  _showProgress(false);
  _setGenerateReady(true);
}

async function _runGenerateJob(params, progressTarget) {
  const result = await API.generate(params);
  const jobId = result.job_id;
  _activeJobId = jobId;

  try {
    while (true) {
      await _sleep(2000);
      const job = await API.pollJob(jobId);
      _updateProgress(progressTarget, job.message);
      if (job.status === "done") return job;
      if (job.status === "error") {
        throw new Error(job.message || "Unknown ACE-Step generation error");
      }
    }
  } finally {
    _activeJobId = null;
  }
}

function _collectLoras() {
  const slotPatchEnabled = (localStorage.getItem("vocals.settings.lora_slot_patch") ?? "false") === "true";
  const maxCount = slotPatchEnabled
    ? Math.max(1, Math.min(4, parseInt(document.getElementById("lora-count")?.value, 10) || 1))
    : 1;
  const result = [];
  for (let i = 1; i <= maxCount; i++) {
    const name = _val(`lora-select-${i}`);
    const scale = _sliderVal(`ts-lora-scale-${i}`);
    if (!name) continue;
    result.push({
      name,
      scale: scale ?? 1.0,
      group_scales: slotPatchEnabled ? {
        self_attn: _sliderVal(`ts-lora-sa-${i}`) ?? 1.0,
        cross_attn: _sliderVal(`ts-lora-ca-${i}`) ?? 1.0,
        mlp: _sliderVal(`ts-lora-mlp-${i}`) ?? 1.0,
      } : { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 },
    });
  }
  return result;
}

function _buildGenerateParams(processedSourcePath, refAudioPath, batchSize = 1) {
  const loras = _collectLoras();
  return {
    dit_model: _val("dit-model") || undefined,
    loras,
    lora_name:  loras[0]?.name  || undefined,
    lora_scale: loras[0]?.scale ?? 1.0,
    ref_audio_path:    refAudioPath,
    source_audio_path: processedSourcePath,
    caption: _val("caption"),
    lyrics:  _val("lyrics"),
    thinking: false,
    lm_strength:    _sliderVal("ts-remix-strength"),
    cover_strength: _sliderVal("ts-cover-strength"),
    inference_steps:  _sliderVal("ts-inference-steps"),
    guidance_scale:   _sliderVal("ts-guidance-scale"),
    infer_method:     _val("infer-method") || "ode",
    use_adg:          document.getElementById("use-adg")?.checked ?? false,
    shift:            _sliderVal("ts-shift"),
    cfg_interval_start: _sliderVal("ts-cfg-start"),
    cfg_interval_end:   _sliderVal("ts-cfg-end"),
    keyscale:       _val("keyscale"),
    timesignature:  _val("timesignature"),
    vocal_language: _val("vocal-language") || "unknown",
    seed: parseInt(_val("seed") ?? "-1", 10) || -1,
    batch_size: batchSize,
  };
}

// ------------------------------------------------------------------
// Result waveform
// ------------------------------------------------------------------

function _showResult(audioUrl) {
  // Reset the result play button before destroying the old WaveSurfer
  const _oldResultPlayBtn = document.getElementById("result-play-btn");
  if (_oldResultPlayBtn) _oldResultPlayBtn.textContent = "▶";
  if (_resultWs) { _resultWs.destroy(); _resultWs = null; }

  document.getElementById("result-placeholder")?.style.setProperty("display", "none");
  const wrap = document.getElementById("result-waveform-wrap");
  if (wrap) wrap.style.display = "";

  const container = document.getElementById("result-wave");
  if (!container) return;

  _resultWs = WaveSurfer.create({
    container,
    waveColor: "#5ccfe6",
    progressColor: "rgba(92, 207, 230, 0.35)",
    cursorColor: "#D5D8EC",
    barWidth: 2,
    barGap: 1,
    height: 64,
    normalize: true,
    backend: "WebAudio",
  });

  _resultWs.load(audioUrl);

  _resultWs.on("ready", () => {
    _updateResultTime(0, _resultWs.getDuration());
  });

  _resultWs.on("audioprocess", (t) => {
    _updateResultTime(t, _resultWs.getDuration());
  });

  _resultWs.on("finish", () => {
    const btn = document.getElementById("result-play-btn");
    if (btn) btn.textContent = "▶";
  });

  // Click waveform → toggle playback in WaveSurfer
  container.addEventListener("click", () => {
    if (!_resultWs.isPlaying()) _stopAllPlayers("result-player");
    _resultWs.playPause();
    const btn = document.getElementById("result-play-btn");
    if (btn) btn.textContent = _resultWs.isPlaying() ? "⏸" : "▶";
  });

  // Play button
  const playBtn = document.getElementById("result-play-btn");
  if (playBtn) {
    playBtn.onclick = () => {
      if (!_resultWs.isPlaying()) _stopAllPlayers("result-player");
      _resultWs.playPause();
      playBtn.textContent = _resultWs.isPlaying() ? "⏸" : "▶";
    };
  }

  Toast.show("ACE-Step generation complete", "success");
}

function _updateResultTime(current, total) {
  const el = document.getElementById("result-time");
  if (el) el.textContent = `${_fmtSecs(current)} / ${_fmtSecs(total)}`;
}

// ------------------------------------------------------------------
// RVC controls
// ------------------------------------------------------------------

function _populateRvcSelects(rvcCfg) {
  const pthFiles = rvcCfg.pth_files || [];
  const indexFiles = rvcCfg.index_files || [];

  const modelSel = document.getElementById("rvc-model");
  const indexSel = document.getElementById("rvc-index");

  if (modelSel) {
    modelSel.innerHTML = "";
    if (pthFiles.length === 0) {
      modelSel.innerHTML = "<option value=\"\">— No models found —</option>";
    } else {
      pthFiles.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p.split("/").pop();
        modelSel.appendChild(opt);
      });
    }
  }

  if (indexSel) {
    indexSel.innerHTML = "<option value=\"\">— None —</option>";
    indexFiles.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p.split("/").pop();
      indexSel.appendChild(opt);
    });
  }
}

function _bindRvcRunBtn() {
  document.getElementById("rvc-run-btn")?.addEventListener("click", () => _onRunRvc(false));
}

function _bindRvcSaveBtn() {
  document.getElementById("rvc-save-btn")?.addEventListener("click", _onSaveRvcResult);
}

/**
 * Run RVC inference.
 * @param {boolean} [autoSave=false] - Automatically save after successful inference (pipeline mode).
 */
async function _onRunRvc(autoSave = false) {
  const modelPath = _val("rvc-model");
  if (!modelPath) {
    Toast.show("Select a voice model first", "warning");
    return;
  }

  // Use ACE-Step result if available, otherwise fall back to the processed source audio
  const sourceWidget = _audioUploads.get("source-audio");
  const inputPath = _resultRawPath || sourceWidget?.getTempPath();
  if (!inputPath) {
    Toast.show("No audio to convert — run ACE-Step or upload source audio first", "warning");
    return;
  }

  const runBtn = document.getElementById("rvc-run-btn");
  if (runBtn) { runBtn.disabled = true; runBtn.textContent = "⏳ Running…"; }
  _showRvcProgress(true);

  try {
    const result = await API.runRvc({
      input_path: inputPath,
      model_path: modelPath,
      index_path: _val("rvc-index") || "",
      pitch: Math.round(_sliderVal("ts-rvc-pitch")),
      f0_method: _val("rvc-f0-method") || "rmvpe",
      index_rate: _sliderVal("ts-rvc-index-rate"),
      volume_envelope: _sliderVal("ts-rvc-volume-envelope"),
      protect: _sliderVal("ts-rvc-protect"),
      clean_audio: document.getElementById("rvc-clean-audio")?.checked ?? true,
      clean_strength: _sliderVal("ts-rvc-clean-strength"),
      embedder_model: _val("rvc-embedder-model") || "contentvec",
      filter_radius: Math.round(_sliderVal("ts-rvc-filter-radius")),
      seed: parseInt(_val("rvc-seed") || "-1", 10),
      cuda_device: _getSettings().rvcCudaDevice || "auto",
    });

    _rvcResultPath = result.output_path;
    _showRvcResult(API.tempAudioUrl(_rvcResultPath));
    Toast.show("RVC conversion complete", "success");

    if (autoSave) {
      setTimeout(_onSaveRvcResult, 500);
    }
  } catch (err) {
    Toast.show(`RVC failed: ${err.message}`, "error");
  } finally {
    if (runBtn) { runBtn.disabled = false; runBtn.textContent = "▶ Run RVC"; }
    _showRvcProgress(false);
  }
}

function _showRvcProgress(visible) {
  const btn = document.getElementById("rvc-run-btn");
  if (btn) btn.classList.toggle("is-running", visible);
}

function _showRvcResult(audioUrl) {
  // Reset the RVC play button before destroying the old WaveSurfer
  const _oldRvcPlayBtn = document.getElementById("rvc-play-btn");
  if (_oldRvcPlayBtn) _oldRvcPlayBtn.textContent = "▶";

  const placeholder = document.getElementById("rvc-result-placeholder");
  const wrapEl = document.getElementById("rvc-result-waveform-wrap");
  const waveEl = document.getElementById("rvc-result-wave");

  if (placeholder) placeholder.style.display = "none";
  if (wrapEl) wrapEl.style.display = "";

  if (!waveEl) return;

  if (_rvcResultWs) {
    _rvcResultWs.destroy();
    _rvcResultWs = null;
  }

  _rvcResultWs = WaveSurfer.create({
    container: waveEl,
    waveColor: "#7c6fcd",
    progressColor: "#4f46e5",
    height: 56,
    barWidth: 2,
    barGap: 1,
    interact: true,
  });
  _rvcResultWs.load(audioUrl);
  _rvcResultWs.on("timeupdate", (t) => _updateRvcTime(t, _rvcResultWs.getDuration()));
  _rvcResultWs.on("ready", () => _updateRvcTime(0, _rvcResultWs.getDuration()));
  _rvcResultWs.on("finish", () => {
    const playBtn = document.getElementById("rvc-play-btn");
    if (playBtn) playBtn.textContent = "▶";
  });

  const playBtn = document.getElementById("rvc-play-btn");
  if (playBtn) {
    playBtn.onclick = () => {
      if (!_rvcResultWs.isPlaying()) _stopAllPlayers("rvc-player");
      _rvcResultWs.playPause();
      playBtn.textContent = _rvcResultWs.isPlaying() ? "⏸" : "▶";
    };
  }
}

function _updateRvcTime(current, total) {
  const el = document.getElementById("rvc-result-time");
  if (el) el.textContent = `${_fmtSecs(current)} / ${_fmtSecs(total)}`;
}

async function _onSaveRvcResult() {
  if (!_rvcResultPath) {
    Toast.show("No RVC result to save yet", "warning");
    return;
  }

  const saveBtn = document.getElementById("rvc-save-btn");
  if (saveBtn) saveBtn.disabled = true;

  try {
    const result = await API.saveResult({
      audio_src_path: _rvcResultPath,
      input_filename: _sourceOriginalName || "vocal",
      index: _resultOutputIndex,
    });
    _resultOutputIndex++;
    Toast.show(`Saved: ${result.filename}`, "success");
  } catch (err) {
    Toast.show(`Save failed: ${err.message}`, "error");
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

// ------------------------------------------------------------------
// Save result (ACE-Step)
// ------------------------------------------------------------------

function _bindSaveResultBtn() {
  document.getElementById("result-save-btn")?.addEventListener("click", _onSaveResult);
}

async function _onSaveResult() {
  if (!_resultRawPath) {
    Toast.show("No result to save yet", "warning");
    return;
  }

  const saveBtn = document.getElementById("result-save-btn");
  if (saveBtn) saveBtn.disabled = true;

  try {
    const result = await API.saveResult({
      audio_src_path: _resultRawPath,
      input_filename: _sourceOriginalName || "vocal",
      index: _resultOutputIndex,
    });
    _resultOutputIndex++;
    Toast.show(`Saved: ${result.filename}`, "success");
  } catch (err) {
    Toast.show(`Save failed: ${err.message}`, "error");
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

// ------------------------------------------------------------------
// Run Pipeline (audio processing + ACE-Step + auto-save)
// ------------------------------------------------------------------

function _bindRunPipelineBtn() {
  document.getElementById("run-pipeline-btn")?.addEventListener("click", _onRunPipeline);
}

async function _onRunPipeline() {
  if (_activeJobId || _pipelineRunning) return;

  const sourceWidget = _audioUploads.get("source-audio");
  const refWidget = _audioUploads.get("ref-audio");
  if (!sourceWidget?.getTempPath()) {
    Toast.show("Upload source audio first", "warning");
    return;
  }

  const runBtn = document.getElementById("run-pipeline-btn");
  const batchSize = Math.max(1, Math.min(8, parseInt(_val("pipeline-batch-size") || "1", 10) || 1));
  const modelPath = _val("rvc-model");

  _pipelineRunning = true;
  _setGenerateReady(false);
  _showPipelineProgress(true, "Processing audio…");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = "⏳ Running…";
  }

  try {
    let processedPath = sourceWidget.getTempPath();
    const processResult = await API.processAudio({
      audio_path: processedPath,
      pitch_shift_semitones: _sliderVal("ts-pitch-shift"),
      apply_low_cut: document.getElementById("apply-low-cut")?.checked ?? false,
      apply_noise_gate: document.getElementById("apply-noise-gate")?.checked ?? false,
    });
    processedPath = processResult.processed_path;

    _showPipelineProgress(true, `Running ACE-Step batch (${batchSize})…`);
    const params = _buildGenerateParams(processedPath, refWidget?.getTempPath() ?? null, batchSize);
    const job = await _runGenerateJob(params, "pipeline");
    const rawPaths = job.raw_audio_paths || [];
    if (!rawPaths.length) {
      throw new Error("ACE-Step returned no audio outputs");
    }

    const outputsForSave = [];
    if (modelPath) {
      for (let i = 0; i < rawPaths.length; i++) {
        _showPipelineProgress(true, `Running RVC ${i + 1}/${rawPaths.length}…`);
        const rvc = await API.runRvc({
          input_path: rawPaths[i],
          model_path: modelPath,
          index_path: _val("rvc-index") || "",
          pitch: Math.round(_sliderVal("ts-rvc-pitch")),
          f0_method: _val("rvc-f0-method") || "rmvpe",
          index_rate: _sliderVal("ts-rvc-index-rate"),
          volume_envelope: _sliderVal("ts-rvc-volume-envelope"),
          protect: _sliderVal("ts-rvc-protect"),
          clean_audio: document.getElementById("rvc-clean-audio")?.checked ?? true,
          clean_strength: _sliderVal("ts-rvc-clean-strength"),
          embedder_model: _val("rvc-embedder-model") || "contentvec",
          filter_radius: Math.round(_sliderVal("ts-rvc-filter-radius")),
          seed: parseInt(_val("rvc-seed") || "-1", 10),
        });
        outputsForSave.push(rvc.output_path);
      }
    } else {
      outputsForSave.push(...rawPaths);
    }

    for (let i = 0; i < outputsForSave.length; i++) {
      _showPipelineProgress(true, `Saving ${i + 1}/${outputsForSave.length}…`);
      await API.saveResult({
        audio_src_path: outputsForSave[i],
        input_filename: _sourceOriginalName || "vocal",
        index: _resultOutputIndex,
      });
      _resultOutputIndex++;
    }

    Toast.show(`Pipeline complete: saved ${outputsForSave.length} file(s)`, "success");
  } catch (err) {
    Toast.show(`Pipeline failed: ${err.message}`, "error");
  } finally {
    _pipelineRunning = false;
    _showPipelineProgress(false);
    _setGenerateReady(_acestepReady);
    if (runBtn) {
      runBtn.disabled = !_acestepReady;
      runBtn.textContent = "▶ Run Pipeline";
    }
  }
}

// ------------------------------------------------------------------
// Progress indicator
// ------------------------------------------------------------------

function _showProgress(visible, text = "") {
  const btn = document.getElementById("generate-btn");
  if (btn) btn.classList.toggle("is-running", visible);
  if (btn && visible) btn.textContent = "⏳ Generating…";
  if (btn && !visible) btn.textContent = "▶ Run ACE-Step";
}

function _showPipelineProgress(visible, text = "") {
  const btn = document.getElementById("run-pipeline-btn");
  if (btn) btn.classList.toggle("is-running", visible);
}

function _updateProgress(target, text) {
  // Progress bars are intentionally bar-only (no text labels).
}

// ------------------------------------------------------------------
// Browse output folder
// ------------------------------------------------------------------

function _bindBrowseBtn() {
  document.getElementById("browse-output-btn")?.addEventListener("click", async () => {
    const btn = document.getElementById("browse-output-btn");
    if (btn) btn.disabled = true;
    try {
      const current = _val("output-path") || "";
      const result = await API.pickFolder(current);
      if (result.path) {
        const el = document.getElementById("output-path");
        if (el) el.value = result.path;
      }
    } catch (err) {
      Toast.show(`Could not open folder picker: ${err.message}`, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  });
}

// ------------------------------------------------------------------
// Refresh button
// ------------------------------------------------------------------

function _bindRefreshBtn() {
  document.getElementById("refresh-btn")?.addEventListener("click", async () => {
    const btn = document.getElementById("refresh-btn");
    if (btn) btn.style.animation = "spin 0.6s linear";
    try {
      const [config] = await Promise.all([API.config(), _pollHealth()]);
      _populateModelSelects(config);
      await _syncModelSelectors();
      try { const rvcCfg = await API.rvcConfig(); _populateRvcSelects(rvcCfg);
        _setRvcStatus(rvcCfg.pth_files?.length > 0 ? "ok" : "partial",
          rvcCfg.pth_files?.length > 0 ? `${rvcCfg.pth_files.length} model(s)` : "No models");
      } catch {}
      Toast.show("Refreshed", "success", 1500);
    } catch {
      Toast.show("Refresh failed", "error");
    } finally {
      if (btn) setTimeout(() => (btn.style.animation = ""), 700);
    }
  });
}

// ------------------------------------------------------------------
// Model switch
// ------------------------------------------------------------------

function _bindSwitchModelsBtn() {
  document.getElementById("switch-models-btn")?.addEventListener("click", async () => {
    const dit = _val("dit-model") || "";

    _setGenerateReady(false);
    _setAcestepStatus("running", "Switching…");

    const serverSettings = _readServerSettings();
    try {
      const result = await API.switchModels(dit, "", serverSettings);
      if (result.status === "unchanged") {
        Toast.show("Models already loaded", "info");
      } else {
        Toast.show("ACE-Step restarting with new models…", "info", 5000);
      }
      _loadedModels = { dit_model: dit, lm_model: "" };
    } catch (err) {
      Toast.show(`Switch failed: ${err.message}`, "error");
    }
  });
}

// ------------------------------------------------------------------
// Settings modal
// ------------------------------------------------------------------

function _bindSettings() {
  const backdrop = document.getElementById("settings-backdrop");
  const closeBtn  = document.getElementById("settings-close");
  const saveBtn   = document.getElementById("settings-save");
  const openBtn   = document.getElementById("settings-btn");

  openBtn?.addEventListener("click", () => {
    _loadSettings();
    _populateCudaDevices();
    backdrop?.removeAttribute("hidden");
  });

  closeBtn?.addEventListener("click",  () => backdrop?.setAttribute("hidden", ""));
  backdrop?.addEventListener("click", (ev) => { if (ev.target === backdrop) backdrop.setAttribute("hidden", ""); });

  saveBtn?.addEventListener("click", () => {
    _saveSettings();
    _applyLoraSlotPatchUi();
    _applyUiTheme();
    backdrop?.setAttribute("hidden", "");
    Toast.show("Settings saved", "success", 1500);
  });
}

function _loadSettings() {
  const s = _getSettings();
  _setEl("setting-acestep-url",    s.acestepUrl    || "");
  _setEl("setting-whisper-model", s.whisperModel  || "base");
  _setCheck("setting-flash-attn",   s.flashAttn   ?? false);
  _setCheck("setting-compile",      s.compile     ?? false);
  _setCheck("setting-cpu-offload",  s.cpuOffload  ?? false);
  _setCheck("setting-dit-offload",  s.ditOffload  ?? false);
  _setCheck("setting-mlx-patches",  s.mlxPatches  ?? false);
  _setCheck("setting-lora-slot-patch", s.loraSlotPatch ?? false);
  const cudaSel = document.getElementById("setting-acestep-cuda-device");
  if (cudaSel) cudaSel.value = s.cudaDevice || "auto";
  const rvcCudaSel = document.getElementById("setting-rvc-cuda-device");
  if (rvcCudaSel) rvcCudaSel.value = s.rvcCudaDevice || "auto";
}

function _saveSettings() {
  const s = {
    acestepUrl:   _val("setting-acestep-url"),
    whisperModel: _val("setting-whisper-model") || "base",
    flashAttn:  _checked("setting-flash-attn"),
    compile:    _checked("setting-compile"),
    cpuOffload: _checked("setting-cpu-offload"),
    ditOffload: _checked("setting-dit-offload"),
    mlxPatches: _checked("setting-mlx-patches"),
    loraSlotPatch: _checked("setting-lora-slot-patch"),
    cudaDevice: _val("setting-acestep-cuda-device") || "auto",
    rvcCudaDevice: _val("setting-rvc-cuda-device") || "auto",
  };
  localStorage.setItem("vocals_settings", JSON.stringify(s));
  localStorage.setItem("vocals.settings.lora_slot_patch", s.loraSlotPatch ? "true" : "false");
}

function _getSettings() {
  try { return JSON.parse(localStorage.getItem("vocals_settings") || "{}"); } catch { return {}; }
}

function _readServerSettings() {
  const s = _getSettings();
  return {
    use_flash_attention: s.flashAttn  ?? undefined,
    compile_model:       s.compile    ?? undefined,
    offload_to_cpu:      s.cpuOffload ? "true" : "false",
    offload_dit_to_cpu:  s.ditOffload ?? undefined,
    mlx_patches_enabled: s.mlxPatches ?? undefined,
    lora_slot_patch_enabled: s.loraSlotPatch ?? false,
    cuda_device: s.cudaDevice || "auto",
  };
}

/**
 * Show/hide LoRA-slot-patch-only controls based on the saved setting.
 * When disabled, hides SA/CA/MLP sliders and the LoRA count input, forces count to 1.
 */
function _applyLoraSlotPatchUi() {
  const s = _getSettings();
  const enabled = s.loraSlotPatch ?? false;
  const slotEls = document.querySelectorAll(".lora-slot-only");
  for (const el of slotEls) {
    el.style.display = enabled ? "" : "none";
  }
  if (!enabled) {
    const countInput = document.getElementById("lora-count");
    if (countInput) {
      countInput.value = "1";
      for (let i = 2; i <= 4; i++) {
        const row = document.getElementById(`lora-row-${i}`);
        if (row) row.style.display = "none";
      }
    }
  }
}

/** Populate the CUDA device <select> from the /api/hardware/cuda-devices endpoint. */
async function _populateCudaDevices() {
  const devices = await _fetchCudaDevices();
  
  // Populate ACE-Step selector
  const sel = document.getElementById("setting-acestep-cuda-device");
  if (sel) {
    const savedDevice = _getSettings().cudaDevice || "auto";
    sel.innerHTML = '<option value="auto">Auto</option>';
    for (const d of (devices || [])) {
      const opt = document.createElement("option");
      opt.value = d.value;
      opt.textContent = `${d.value} — ${d.name}`;
      sel.appendChild(opt);
    }
    sel.value = savedDevice;
  }
  
  // Populate RVC selector with same devices
  const rvcSel = document.getElementById("setting-rvc-cuda-device");
  if (rvcSel) {
    const savedRvcDevice = _getSettings().rvcCudaDevice || "auto";
    rvcSel.innerHTML = '<option value="auto">Auto</option>';
    for (const d of (devices || [])) {
      const opt = document.createElement("option");
      opt.value = d.value;
      opt.textContent = `${d.value} — ${d.name}`;
      rvcSel.appendChild(opt);
    }
    rvcSel.value = savedRvcDevice;
  }
}

async function _fetchCudaDevices() {
  try {
    const { devices } = await API.cudaDevices();
    return devices || [];
  } catch {
    return [];
  }
}

/** Apply the Vite/Side-Step theme class from saved settings to <body>. */
function _applyUiTheme() {
  // SVC always uses Vite theme (vite-theme.css is always loaded).
  // Toggle the class in case the CSS file depends on it.
  document.body.classList.add("theme-vite");
  document.body.classList.remove("theme-side-step");
  const faviconEl = document.querySelector("link[rel='icon']");
  if (faviconEl) faviconEl.href = "/images/favicon.png";
}

// ------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------

function _val(id) {
  return document.getElementById(id)?.value ?? "";
}

function _checked(id) {
  return document.getElementById(id)?.checked ?? false;
}

function _setEl(id, v) {
  const el = document.getElementById(id);
  if (el) el.value = v;
}

function _setCheck(id, v) {
  const el = document.getElementById(id);
  if (el) el.checked = v;
}

function _sliderVal(id) {
  const el = document.getElementById(id);
  if (!el) return 0;
  return parseFloat(el.dataset.value ?? "0");
}

function _fmtSecs(s) {
  if (!isFinite(s)) return "0:00";
  const m   = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

function _sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function _stopAllPlayers(exceptKey = "") {
  for (const [id, widget] of _audioUploads.entries()) {
    if (exceptKey === `audio-upload:${id}`) continue;
    if (typeof widget.stopPlayback === "function") widget.stopPlayback();
  }

  if (exceptKey !== "result-player" && _resultWs) {
    _resultWs.pause();
    const btn = document.getElementById("result-play-btn");
    if (btn) btn.textContent = "▶";
  }

  if (exceptKey !== "rvc-player" && _rvcResultWs) {
    _rvcResultWs.pause();
    const btn = document.getElementById("rvc-play-btn");
    if (btn) btn.textContent = "▶";
  }
}
