"""Audio pre-processing utilities for Streamline Vocals.

Provides vocal pitch shifting (WORLD vocoder), low-cut filter, and noise gate.
All processing is applied to the source audio before it is passed to ACE-Step.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_CUT_FREQ_HZ: float = 80.0
LOW_CUT_ORDER: int = 4  # 4th-order Butterworth ≈ -24 dB/oct; pre-shaped to -12 dB at cutoff

NOISE_GATE_THRESHOLD_DB: float = -40.0
NOISE_GATE_ATTACK_MS: float = 5.0
NOISE_GATE_RELEASE_MS: float = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_audio(
    input_path: str,
    *,
    pitch_shift_semitones: float = 0.0,
    apply_low_cut: bool = False,
    apply_noise_gate: bool = False,
) -> str:
    """Apply selected pre-processing steps and return a new temp file path.

    Processing order:
      1. Low-cut filter (if enabled) — removes sub-bass mud before pitch shift
      2. Noise gate (if enabled)
      3. Pitch shift (WORLD vocoder, non-zero semitones only)

    Args:
        input_path: Filesystem path to the source audio (any ffmpeg-readable format).
        pitch_shift_semitones: Semitones to shift pitch, -12 to +12 (0 = skip).
        apply_low_cut: Apply 80 Hz high-pass filter.
        apply_noise_gate: Apply -40 dB / 5 ms noise gate.

    Returns:
        Filesystem path to the processed 16-bit WAV temp file.  If no processing
        is required the original path is returned unchanged.
    """
    no_processing = (
        pitch_shift_semitones == 0.0
        and not apply_low_cut
        and not apply_noise_gate
    )
    if no_processing:
        return input_path

    audio, sr = _load_audio(input_path)

    if apply_low_cut:
        audio = _apply_low_cut(audio, sr)

    if apply_noise_gate:
        audio = _apply_noise_gate(audio, sr)

    if pitch_shift_semitones != 0.0:
        audio = _pitch_shift_world(audio, sr, pitch_shift_semitones)

    return _save_wav(audio, sr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio as a mono float64 array using soundfile.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of (samples: float64 ndarray, sample_rate: int).
    """
    audio, sr = sf.read(path, always_2d=False, dtype="float64")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), int(sr)


def _apply_low_cut(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a 2nd-order Butterworth high-pass at LOW_CUT_FREQ_HZ.

    A 2nd-order filter gives ~-12 dB/oct above the cutoff, matching the
    -12 dB specification.  The filter is applied via sosfilt for numerical
    stability.

    Args:
        audio: Mono float64 audio samples.
        sr: Sample rate in Hz.

    Returns:
        Filtered audio array.
    """
    nyq = sr / 2.0
    normalized_freq = LOW_CUT_FREQ_HZ / nyq
    # 2nd-order SOS for -12 dB/octave high-pass
    sos = butter(2, normalized_freq, btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float64)


def _apply_noise_gate(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a simple RMS-based noise gate.

    Uses NOISE_GATE_THRESHOLD_DB as the open/close threshold with linear
    attack and release envelopes of NOISE_GATE_ATTACK_MS / NOISE_GATE_RELEASE_MS.

    Args:
        audio: Mono float64 audio samples.
        sr: Sample rate in Hz.

    Returns:
        Gated audio array.
    """
    threshold_linear = 10.0 ** (NOISE_GATE_THRESHOLD_DB / 20.0)
    attack_samples = max(1, int(NOISE_GATE_ATTACK_MS * sr / 1000.0))
    release_samples = max(1, int(NOISE_GATE_RELEASE_MS * sr / 1000.0))

    envelope = np.abs(audio)
    gain = np.zeros_like(audio)
    current_gain = 0.0
    attack_step = 1.0 / attack_samples
    release_step = 1.0 / release_samples

    for i in range(len(audio)):
        if envelope[i] >= threshold_linear:
            current_gain = min(1.0, current_gain + attack_step)
        else:
            current_gain = max(0.0, current_gain - release_step)
        gain[i] = current_gain

    return (audio * gain).astype(np.float64)


def _pitch_shift_world(
    audio: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """Shift pitch using the WORLD vocoder.

    WORLD decomposes audio into F0 (pitch), spectral envelope, and aperiodicity,
    then resynthesises with the F0 scaled by the semitone ratio.  This produces
    high-quality pitch shifting that preserves formant shape and timing.

    Args:
        audio: Mono float64 audio samples (WORLD requires float64).
        sr: Sample rate in Hz.
        semitones: Number of semitones to shift (positive = up, negative = down).

    Returns:
        Pitch-shifted audio array (float64, same length as input).
    """
    try:
        import pyworld as pw  # imported lazily — optional heavy dependency
    except ImportError as exc:
        logger.warning(
            "pyworld is not installed; skipping pitch shift. "
            "Run: pip install pyworld"
        )
        raise RuntimeError("pyworld is required for pitch shifting") from exc

    ratio = 2.0 ** (semitones / 12.0)

    f0, sp, ap = pw.wav2world(audio, sr)

    # Scale non-zero F0 values; zero means unvoiced (leave as-is)
    voiced = f0 > 0
    f0[voiced] = f0[voiced] * ratio

    shifted = pw.synthesize(f0, sp, ap, sr)
    # Clip to valid range to prevent inter-sample clipping
    return np.clip(shifted, -1.0, 1.0).astype(np.float64)


def _save_wav(audio: np.ndarray, sr: int) -> str:
    """Save audio to a temp 16-bit WAV file and return its path.

    Args:
        audio: Mono float64 audio samples.
        sr: Sample rate in Hz.

    Returns:
        Filesystem path to the new temp WAV file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="vocals_proc_")
    import os
    os.close(fd)
    sf.write(out_path, audio.astype(np.float32), sr, subtype="PCM_16")
    logger.debug(f"Saved processed audio → {out_path}")
    return out_path
