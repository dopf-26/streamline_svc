/**
 * Streamline Vocals — REST API client.
 * Focused on the vocal transformation pipeline (no Ollama, no library).
 */

const API = (() => {
  const BASE = "";

  async function _fetch(url, options = {}) {
    const res = await fetch(BASE + url, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.detail || j.message || msg; } catch {}
      throw new Error(msg);
    }
    return res.json();
  }

  return {
    /** Check ACE-Step health and server readiness. */
    health: () => _fetch("/api/health"),

    /** Fetch available checkpoints and LoRA options. */
    config: () => _fetch("/api/config"),

    /** Get currently loaded and saved DiT/LM model state. */
    modelState: () => _fetch("/api/models/state"),

    /** Restart ACE-Step server with selected DiT/LM models. */
    switchModels: (ditModel, lmModel, serverSettings = {}) =>
      _fetch("/api/models/switch", {
        method: "POST",
        body: JSON.stringify({ dit_model: ditModel, lm_model: lmModel, ...serverSettings }),
      }),

    /**
     * Upload a File object to a temp location on the server.
     * @returns {{ temp_path: string }}
     */
    uploadAudio: async (file) => {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(BASE + "/api/upload-audio", { method: "POST", body: form });
      if (!res.ok) { const j = await res.json(); throw new Error(j.detail || "Upload failed"); }
      return res.json();
    },

    /**
     * Apply pitch shift / filters to source audio.
     * @param {{ audio_path, pitch_shift_semitones, apply_low_cut, apply_noise_gate }} params
     * @returns {{ processed_path: string }}
     */
    processAudio: (params) =>
      _fetch("/api/process-audio", { method: "POST", body: JSON.stringify(params) }),

    /**
     * Transcribe audio to text via Whisper large-v3.
     * @param {{ audio_path: string, language?: string }} params
     * @returns {{ text: string }}
     */
    transcribe: (params) =>
      _fetch("/api/transcribe", { method: "POST", body: JSON.stringify(params) }),

    /**
     * Start an ACE-Step remix generation job.
     * @returns {{ job_id: string, status: string, seed: number }}
     */
    generate: (params) =>
      _fetch("/api/generate", { method: "POST", body: JSON.stringify(params) }),

    /**
     * Poll generation job status.
     * @param {string} jobId
     */
    pollJob: (jobId) => _fetch(`/api/jobs/${jobId}`),

    /**
     * Save an ACE-Step result to the output directory.
     * @param {{ audio_src_path, input_filename, index }} body
     * @returns {{ saved_path: string, filename: string }}
     */
    saveResult: (body) =>
      _fetch("/api/save-result", { method: "POST", body: JSON.stringify(body) }),

    /**
     * Open a native OS folder-picker dialog and return the chosen path.
     * @param {string} [initial] - Starting directory (defaults to output dir).
     * @returns {{ path: string|null }}
     */
    pickFolder: (initial = "") =>
      _fetch(`/api/pick-folder?initial=${encodeURIComponent(initial)}`),

    /**
     * Build URL to stream a temp audio file by server-side path.
     * @param {string} path - Absolute server-side file path.
     */
    tempAudioUrl: (path) => `${BASE}/api/audio/temp?path=${encodeURIComponent(path)}`,

    /**
     * Return detected CUDA devices with indices and names.
     * @returns {{ devices: Array<{index: number, value: string, name: string}> }}
     */
    cudaDevices: () => _fetch("/api/hardware/cuda-devices"),

    /**
     * Return available .pth and .index files from applio/logs.
     * @returns {{ pth_files: string[], index_files: string[] }}
     */
    rvcConfig: () => _fetch("/api/rvc/config"),

    /**
     * Run RVC voice conversion.
     * @param {object} body - RvcRequest fields.
     * @returns {{ output_path: string }}
     */
    runRvc: (body) =>
      _fetch("/api/rvc/run", { method: "POST", body: JSON.stringify(body) }),
  };
})();

if (typeof module !== "undefined") module.exports = API;
