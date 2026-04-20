/**
 * Streamline Vocals — Bottom audio Player singleton.
 *
 * Simplified version: no prev/next, no continuous-play mode.
 * Any audio upload widget or result waveform can load a track by calling
 * Player.load(url, title) and Player.play().
 */

const Player = (() => {
  let _audio = null;
  let _seeking = false;
  let _currentTitle = "";

  let $title, $time, $seekTrack, $seekFill, $seekCursor, $play;

  function _queryEls() {
    _audio = document.getElementById("main-audio");
    $title = document.getElementById("player-title");
    $time = document.getElementById("player-time");
    $seekTrack = document.getElementById("player-seek-track");
    $seekFill = document.getElementById("player-seek-fill");
    $seekCursor = document.getElementById("player-seek-cursor");
    $play = document.getElementById("player-play");
  }

  function _bindAudioEvents() {
    if (!_audio) return;
    _audio.addEventListener("timeupdate", () => {
      if (_seeking) return;
      _updateSeek();
      _updateTime();
    });
    _audio.addEventListener("ended", () => _setPlayIcon(false));
    _audio.addEventListener("play",  () => _setPlayIcon(true));
    _audio.addEventListener("pause", () => _setPlayIcon(false));
    _audio.addEventListener("loadedmetadata", () => { _updateTime(); _updateSeek(); });
  }

  function _bindTransportEvents() {
    $play?.addEventListener("click", toggle);

    $seekTrack?.addEventListener("pointerdown", (ev) => {
      _seeking = true;
      $seekTrack.setPointerCapture(ev.pointerId);
      _seekToPointer(ev);
    });
    $seekTrack?.addEventListener("pointermove", (ev) => {
      if (!_seeking) return;
      _seekToPointer(ev);
    });
    $seekTrack?.addEventListener("pointerup",     () => { _seeking = false; });
    $seekTrack?.addEventListener("pointercancel", () => { _seeking = false; });
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Load a new track and auto-play it.
   * @param {string} url    — audio source URL
   * @param {string} title  — display title
   */
  function load(url, title) {
    if (!_audio) return;
    _currentTitle = title || "Untitled";
    _audio.src = url;
    _audio.load();

    document.getElementById("player-bar")?.classList.remove("player-standby");

    if ($title) {
      $title.textContent = _currentTitle.toUpperCase();
      $title.style.animation = "none";
      void $title.offsetWidth;
      $title.style.animation = "";
    }
  }

  function play()   { _audio?.play().catch(() => {}); }
  function pause()  { _audio?.pause(); }
  function toggle() { if (!_audio) return; _audio.paused ? play() : pause(); }
  function isPlaying() { return _audio ? !_audio.paused : false; }

  function init() {
    _queryEls();
    _bindAudioEvents();
    _bindTransportEvents();
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  function _updateSeek() {
    if (!_audio || !$seekFill) return;
    const frac = _audio.duration > 0 ? (_audio.currentTime / _audio.duration) : 0;
    const pct = `${frac * 100}%`;
    $seekFill.style.width = pct;
    if ($seekCursor) $seekCursor.style.left = pct;
  }

  function _updateTime() {
    if (!$time) return;
    const cur   = _audio?.currentTime ?? 0;
    const total = _audio?.duration    ?? 0;
    $time.textContent = `${_fmtSecs(cur)} / ${_fmtSecs(total)}`;
  }

  function _seekToPointer(ev) {
    const rect = $seekTrack.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    if (_audio && _audio.duration) _audio.currentTime = frac * _audio.duration;
    const pct = `${frac * 100}%`;
    if ($seekFill)   $seekFill.style.width = pct;
    if ($seekCursor) $seekCursor.style.left = pct;
  }

  function _setPlayIcon(playing) {
    if ($play) $play.textContent = playing ? "⏸" : "▶";
  }

  function _fmtSecs(s) {
    if (!isFinite(s)) return "0:00";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  }

  return { init, load, play, pause, toggle, isPlaying };
})();
