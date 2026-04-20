/**
 * Streamline — AudioUpload widget.
 *
 * Wraps a `.audio-upload` container element.  When a file is selected (drag-
 * drop or click), it uploads via `API.uploadAudio()` and renders a WaveSurfer
 * waveform.  The widget toggles between dropzone and waveform-player states.
 *
 * Inpaint variant (.audio-upload[data-inpaint="true"]) additionally exposes
 * region handles that let the user define start/end times as fractions (0–1).
 * The fraction values are stored in `regionStart` / `regionEnd` and synced to
 * the hidden `<input>` elements declared in index.html.
 *
 * Required WaveSurfer CDN loaded via index.html.
 */

class AudioUpload {
  /**
   * @param {HTMLElement} container  — the .audio-upload wrapper
   */
  constructor(container) {
    this.container = container;
    this.id = container.id || "";
    this.isInpaint = container.classList.contains("audio-upload--inpaint");

    this.tempPath = null;   // path on server after upload
    this.ws = null;         // WaveSurfer instance
    this.isPlaying = false;

    this.regionStart = 0;
    this.regionEnd = 1;

    this._resolveUpload = null; // pending upload promise resolver
    this._uploadInProgress = false;

    this._queryEls();
    this._bindEvents();
  }

  // ------------------------------------------------------------------
  // Public
  // ------------------------------------------------------------------

  /** Return the server temp-path of the uploaded file, or null. */
  getTempPath() { return this.tempPath; }

  /** Return region fractions {start, end} for inpaint widgets. */
  getRegion() { return { start: this.regionStart, end: this.regionEnd }; }

  /** Programmatically reset the widget to its empty state. */
  reset() {
    this.tempPath = null;
    this.isPlaying = false;
    this.ws?.destroy();
    this.ws = null;
    this.container.classList.remove("has-file", "is-playing", "is-loading");
    if (this._waveformEl) {
      this._waveformEl.innerHTML = "";
      // Re-attach region elements so they survive the innerHTML clear
      if (this.isInpaint) {
        [this._regionFill, this._regionStart, this._regionEnd].forEach((el) => {
          if (el) this._waveformEl.appendChild(el);
        });
        this._setRegionHandlePositions();
      }
    }
  }

  /** Pause playback and reset play button state without clearing uploaded file. */
  stopPlayback() {
    if (!this.ws) return;
    this.ws.pause();
    this.isPlaying = false;
    this.container.classList.remove("is-playing");
    if (this._playBtn) this._playBtn.textContent = "▶";
  }

  // ------------------------------------------------------------------
  // Internal
  // ------------------------------------------------------------------

  _queryEls() {
    this._dropzone = this.container.querySelector(".audio-upload__dropzone");
    this._fileInput = this.container.querySelector("input[type=file]");
    this._waveformEl = this.container.querySelector(".audio-upload__waveform");
    this._playBtn = this.container.querySelector(".audio-upload__play-btn");
    this._trashBtn = this.container.querySelector(".audio-upload__trash-btn");
    this._timeEl = this.container.querySelector(".audio-upload__time");
    // Region handles (inpaint variant)
    const handles = this.container.querySelectorAll(".region-handle");
    this._regionFill  = this.container.querySelector(".region-fill");
    this._regionStart = handles[0] ?? null;
    this._regionEnd   = handles[1] ?? null;
  }

  _bindEvents() {
    // Click on dropzone opens file picker
    this._dropzone?.addEventListener("click", () => this._fileInput?.click());

    // File input change
    this._fileInput?.addEventListener("change", (ev) => {
      const file = ev.target.files?.[0];
      if (file) this._handleFile(file);
      ev.target.value = "";  // reset so same file can be re-selected
    });

    // Drag and drop
    this._dropzone?.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      this._dropzone.classList.add("drag-over");
    });
    this._dropzone?.addEventListener("dragleave", () => {
      this._dropzone.classList.remove("drag-over");
    });
    this._dropzone?.addEventListener("drop", (ev) => {
      ev.preventDefault();
      this._dropzone.classList.remove("drag-over");
      const file = ev.dataTransfer?.files?.[0];
      if (file) this._handleFile(file);
    });

    // Play / pause
    this._playBtn?.addEventListener("click", () => this._togglePlay());

    // Trash (reset)
    this._trashBtn?.addEventListener("click", () => this.reset());

    // Inpaint region handles (drag based on .audio-upload__waveform width)
    if (this.isInpaint && this._regionStart) {
      this._bindRegionHandles();
    }
  }

  async _handleFile(file) {
    if (this._uploadInProgress) return;
    this._uploadInProgress = true;

    this.container.classList.add("is-loading");
    try {
      const result = await API.uploadAudio(file);
      this.tempPath = result.temp_path;

      this.container.classList.remove("is-loading");
      this.container.classList.add("has-file");

      await this._initWaveSurfer(URL.createObjectURL(file));
    } catch (err) {
      console.error("AudioUpload: upload failed", err);
      this.container.classList.remove("is-loading");
      Toast.show("Upload failed: " + err.message, "error");
    } finally {
      this._uploadInProgress = false;
    }
  }

  async _initWaveSurfer(blobUrl) {
    if (this.ws) { this.ws.destroy(); this.ws = null; }

    /** @type {import("wavesurfer.js").default} */
    this.ws = WaveSurfer.create({
      container: this._waveformEl,
      waveColor: "#5ccfe6",
      progressColor: "rgba(92, 207, 230, 0.4)",
      cursorColor: "#D5D8EC",
      barWidth: 2,
      barGap: 1,
      height: 48,
      normalize: true,
      backend: "WebAudio",
    });

    // Re-append region overlays on top of WaveSurfer's canvas
    if (this.isInpaint) {
      [this._regionFill, this._regionStart, this._regionEnd].forEach((el) => {
        if (el) this._waveformEl.appendChild(el);
      });
    }

    this.ws.load(blobUrl);

    this.ws.on("ready", () => {
      const dur = this.ws.getDuration();
      this._updateTime(0, dur);

      if (this.isInpaint) {
        this._setRegionHandlePositions();
      }
    });

    this.ws.on("audioprocess", (time) => {
      this._updateTime(time, this.ws.getDuration());
      // Stop at region end for inpaint playback
      if (this.isInpaint && this.isPlaying) {
        const endTime = this.regionEnd * this.ws.getDuration();
        if (time >= endTime - 0.05) {
          this.ws.pause();
          this.isPlaying = false;
          this.container.classList.remove("is-playing");
          if (this._playBtn) this._playBtn.textContent = "▶";
        }
      }
    });

    this.ws.on("finish", () => {
      this.isPlaying = false;
      this.container.classList.remove("is-playing");
      if (this._playBtn) this._playBtn.textContent = "▶";
    });

    // Double-click on waveform: seek to clicked position and start playback
    this._waveformEl.addEventListener("dblclick", (ev) => {
      if (!this.ws) return;
      const rect = this._waveformEl.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      const seekTime = fraction * this.ws.getDuration();

      if (typeof window.__streamlineStopAllPlayers === "function") {
        window.__streamlineStopAllPlayers(`audio-upload:${this.id}`);
      }

      if (this.isInpaint) {
        const endTime = this.regionEnd * this.ws.getDuration();
        if (seekTime < endTime) {
          this.ws.play(seekTime, endTime);
          this.isPlaying = true;
          this.container.classList.add("is-playing");
          if (this._playBtn) this._playBtn.textContent = "⏸";
        }
      } else {
        this.ws.seekTo(fraction);
        if (!this.isPlaying) { this._togglePlay(); }
      }
      ev.preventDefault();
    });
  }

  _togglePlay() {
    if (!this.ws) return;
    if (!this.isPlaying) {
      if (typeof window.__streamlineStopAllPlayers === "function") {
        window.__streamlineStopAllPlayers(`audio-upload:${this.id}`);
      }
      if (this.isInpaint) {
        const dur = this.ws.getDuration();
        this.ws.play(this.regionStart * dur, this.regionEnd * dur);
      } else {
        this.ws.play();
      }
      this.isPlaying = true;
      this.container.classList.add("is-playing");
      if (this._playBtn) this._playBtn.textContent = "⏸";
    } else {
      this.ws.pause();
      this.isPlaying = false;
      this.container.classList.remove("is-playing");
      if (this._playBtn) this._playBtn.textContent = "▶";
    }
  }

  _updateTime(current, total) {
    if (!this._timeEl) return;
    this._timeEl.textContent = `${_fmt(current)} / ${_fmt(total)}`;
  }

  // ------------------------------------------------------------------
  // Inpaint region handles
  // ------------------------------------------------------------------

  _renderRegion(_duration) {
    // Handles are always in the DOM for the inpaint variant; just ensure positions are set.
    this._setRegionHandlePositions();
  }

  _setRegionHandlePositions() {
    if (!this._regionStart || !this._regionEnd) return;
    this._regionStart.style.left = `${this.regionStart * 100}%`;
    this._regionEnd.style.left   = `${this.regionEnd   * 100}%`;
    // Keep fill overlay synced between the two handles
    if (this._regionFill) {
      this._regionFill.style.left  = `${this.regionStart * 100}%`;
      this._regionFill.style.width = `${(this.regionEnd - this.regionStart) * 100}%`;
    }
  }

  _bindRegionHandles() {
    const makeHandleDrag = (handle, isEnd) => {
      let dragging = false;

      handle?.addEventListener("pointerdown", (ev) => {
        dragging = true;
        handle.setPointerCapture(ev.pointerId);
        ev.stopPropagation();
      });

      handle?.addEventListener("pointermove", (ev) => {
        if (!dragging) return;
        const rect = this._waveformEl.getBoundingClientRect();
        const fraction = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        if (isEnd) {
          this.regionEnd = Math.max(this.regionStart + 0.05, fraction);
        } else {
          this.regionStart = Math.min(fraction, this.regionEnd - 0.05);
        }
        this._setRegionHandlePositions();
      });

      handle?.addEventListener("pointerup", () => { dragging = false; });
    };

    makeHandleDrag(this._regionStart, false);
    makeHandleDrag(this._regionEnd, true);
  }
}

// Shared time formatter mm:ss
function _fmt(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Initialize all .audio-upload elements on the page.
const AudioUploadRegistry = new Map(); // id -> AudioUpload

function initAllAudioUploads() {
  document.querySelectorAll(".audio-upload").forEach((el) => {
    if (!el.id) return;
    AudioUploadRegistry.set(el.id, new AudioUpload(el));
  });
}
