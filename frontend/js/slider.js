/**
 * Streamline — Suno-style tick-mark sliders.
 *
 * Each slider is a `.tick-slider` element with data attributes:
 *   data-min, data-max, data-value, data-step
 *
 * The track contains:
 *   .tick-slider__bg     — base track bar
 *   .tick-slider__fill   — filled portion up to thumb
 *   .tick-slider__thumb  — draggable handle
 *
 * 11 tick marks are injected at 0%, 10%, …, 100%.
 * Tick opacity formula (Suno original):
 *   opacity = max(0.1, (1 - |normalizedValue - tickPos|)^6)
 */

class TickSlider {
  /**
   * @param {HTMLElement} container  — the .tick-slider wrapper
   * @param {function}    onChange   — called with (value) when dragging
   */
  constructor(container, onChange = null) {
    this.container = container;
    this.onChange = onChange;

    this.min = parseFloat(container.dataset.min ?? 0);
    this.max = parseFloat(container.dataset.max ?? 1);
    this.step = parseFloat(container.dataset.step ?? 0.01);
    this.value = parseFloat(container.dataset.value ?? this.min);

    this.track = container.querySelector(".tick-slider__track");
    this.fill = container.querySelector(".tick-slider__fill");
    this.thumb = container.querySelector(".tick-slider__thumb");
    this.valueDisplay = container.querySelector(".tick-slider__value");

    this._ticks = [];
    this._dragging = false;

    this._injectTicks();
    this._render();
    this._bindEvents();
  }

  // ------------------------------------------------------------------
  // Public
  // ------------------------------------------------------------------

  /** Set the slider value programmatically. */
  setValue(v) {
    this.value = this._clamp(this._snap(v));
    this._render();
  }

  /** Return the current slider value. */
  getValue() { return this.value; }

  // ------------------------------------------------------------------
  // Internal
  // ------------------------------------------------------------------

  _injectTicks() {
    const TICK_COUNT = 11; // 0%, 10%, …, 100%
    for (let i = 0; i < TICK_COUNT; i++) {
      const el = document.createElement("div");
      el.className = "tick-slider__tick";
      el.style.left = `${(i / (TICK_COUNT - 1)) * 100}%`;
      this.track.insertBefore(el, this.fill);
      this._ticks.push({ el, pos: i / (TICK_COUNT - 1) });
    }
  }

  _normalize(v) {
    return (v - this.min) / (this.max - this.min);
  }

  _fromNormalized(n) {
    return this.min + n * (this.max - this.min);
  }

  _clamp(v) {
    return Math.min(this.max, Math.max(this.min, v));
  }

  _snap(v) {
    if (this.step === 0) return v;
    const snapped = Math.round((v - this.min) / this.step) * this.step + this.min;
    // Round to avoid float precision artifacts
    const decimals = (this.step.toString().split(".")[1] || "").length;
    return parseFloat(snapped.toFixed(decimals));
  }

  _render() {
    const norm = this._normalize(this.value);
    const pct = norm * 100;

    this.fill.style.width = `${pct}%`;
    this.thumb.style.left = `${pct}%`;

    // Update tick opacities
    for (const { el, pos } of this._ticks) {
      const distance = Math.abs(norm - pos);
      const opacity = Math.max(0.1, Math.pow(1 - distance, 6));
      el.style.opacity = opacity;
    }

    // Update aria + value display
    this.track.setAttribute("aria-valuenow", this.value);
    if (this.valueDisplay) {
      // Format nicely: integers show no decimals, floats show 2
      const decimals = (this.step.toString().split(".")[1] || "").length;
      this.valueDisplay.textContent = decimals > 0
        ? this.value.toFixed(decimals)
        : String(this.value);
    }

    // Propagate to data attribute so sidebar.js can read it
    this.container.dataset.value = this.value;
  }

  _valueFromPointer(ev) {
    const rect = this.track.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width;
    return this._clamp(this._snap(this._fromNormalized(Math.min(1, Math.max(0, x)))));
  }

  _bindEvents() {
    const onMove = (ev) => {
      if (!this._dragging) return;
      ev.preventDefault();
      this.value = this._valueFromPointer(ev);
      this._render();
      this.onChange?.(this.value);
    };

    const onUp = () => { this._dragging = false; };

    this.track.addEventListener("pointerdown", (ev) => {
      this._dragging = true;
      this.track.setPointerCapture(ev.pointerId);
      this.value = this._valueFromPointer(ev);
      this._render();
      this.onChange?.(this.value);
    });

    this.track.addEventListener("pointermove", onMove);
    this.track.addEventListener("pointerup", onUp);
    this.track.addEventListener("pointercancel", onUp);

    // Keyboard support
    this.track.addEventListener("keydown", (ev) => {
      const step = ev.shiftKey ? this.step * 10 : this.step;
      if (ev.key === "ArrowRight" || ev.key === "ArrowUp") {
        this.setValue(this.value + step);
        this.onChange?.(this.value);
        ev.preventDefault();
      } else if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") {
        this.setValue(this.value - step);
        this.onChange?.(this.value);
        ev.preventDefault();
      }
    });

    // Double-click the value label to edit it inline
    if (this.valueDisplay) {
      this.valueDisplay.style.cursor = "text";
      this.valueDisplay.title = "Double-click to edit";

      this.valueDisplay.addEventListener("dblclick", (ev) => {
        ev.stopPropagation();
        const decimals = (this.step.toString().split(".")[1] || "").length;
        const input = document.createElement("input");
        input.type = "number";
        input.value = this.value.toFixed(decimals);
        input.min = this.min;
        input.max = this.max;
        input.step = this.step;
        input.className = "tick-slider__inline-input";
        this.valueDisplay.replaceWith(input);
        input.focus();
        input.select();

        const commit = () => {
          const parsed = parseFloat(input.value);
          if (!isNaN(parsed)) {
            this.setValue(parsed);
            this.onChange?.(this.value);
          }
          // _render() replaces valueDisplay textContent but not the element;
          // swap input back for the original span
          input.replaceWith(this.valueDisplay);
          this._render();
        };

        input.addEventListener("blur", commit);
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") { e.preventDefault(); input.blur(); }
          if (e.key === "Escape") { input.value = this.value; input.blur(); }
        });
      });
    }
  }
}

// ------------------------------------------------------------------
// Initialize all .tick-slider elements in the document.
// Called by app.js after DOM is ready.
// ------------------------------------------------------------------

const SliderRegistry = new Map(); // id -> TickSlider

function initAllSliders() {
  document.querySelectorAll(".tick-slider").forEach((container) => {
    if (!container.id) return;
    const slider = new TickSlider(container);
    SliderRegistry.set(container.id, slider);
  });
}

function getSliderValue(id) {
  return SliderRegistry.get(id)?.getValue() ?? null;
}

function setSliderValue(id, value) {
  SliderRegistry.get(id)?.setValue(value);
}
