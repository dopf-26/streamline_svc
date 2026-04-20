"""Whisper-based audio transcription for Streamline Vocals.

Provides a single ``transcribe()`` function that loads Whisper large-v3
on demand and returns the transcript as plain text (no timestamps needed
for the lyrics field).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_model = None
_MODEL_NAME = "base"
_loaded_model_name: str | None = None

def transcribe(audio_path: str, language: str | None = None, model_name: str = "base") -> str:
    """Transcribe audio to text using OpenAI Whisper.

    The model is loaded lazily on first call and reloaded if the requested
    model name differs from the currently cached one.

    Args:
        audio_path: Filesystem path to the audio file.
        language: Optional ISO 639-1 language code (e.g. "en").  None = auto-detect.
        model_name: Whisper model size (tiny/base/small/medium/large-v2/large-v3).

    Returns:
        Transcribed text string.

    Raises:
        RuntimeError: If whisper is not installed or transcription fails.
    """
    global _model, _loaded_model_name
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. "
            "Run: pip install openai-whisper"
        ) from exc

    if _model is None or _loaded_model_name != model_name:
        logger.info(f"Loading Whisper '{model_name}' model…")
        _model = whisper.load_model(model_name)
        _loaded_model_name = model_name
        logger.info("Whisper model loaded.")

    options: dict = {"fp16": False}
    if language:
        options["language"] = language

    logger.info(f"Transcribing {audio_path}")
    result = _model.transcribe(audio_path, **options)

    if not isinstance(result, dict):
        raise RuntimeError("Whisper transcription returned an unexpected result type")

    text = result.get("text")
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip()
