"""Pydantic request/response models for Streamline Vocals."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class LoraEntry(BaseModel):
    """Single LoRA adapter entry with optional per-layer group scales."""

    name: str
    scale: float = Field(default=1.0, ge=0.0, le=4.0)
    group_scales: Optional[dict[str, float]] = None


class RemixRequest(BaseModel):
    """Parameters for a vocal remix generation job (ACE-Step remix mode).

    BPM and duration are always auto (-1 / None).
    File format is always 16-bit WAV.
    """

    # --- Model selection ---
    dit_model: Optional[str] = Field(default=None)
    lm_model: Optional[str] = Field(default=None)

    # --- LoRA (multi-adapter) ---
    loras: Optional[list[LoraEntry]] = None
    # Legacy single-LoRA fields (maintained for backwards compatibility)
    lora_name: Optional[str] = Field(default=None)
    lora_scale: float = Field(default=1.0, ge=0.0, le=4.0)

    @model_validator(mode="after")
    def _normalize_loras(self) -> "RemixRequest":
        """Normalize multi-LoRA and legacy single-LoRA fields."""
        if not self.loras and self.lora_name:
            self.loras = [LoraEntry(name=self.lora_name, scale=self.lora_scale)]
        if self.loras and not self.lora_name:
            first = self.loras[0]
            self.lora_name = first.name
            self.lora_scale = first.scale
        return self

    # --- Audio uploads (temp paths on disk) ---
    ref_audio_path: Optional[str] = None
    source_audio_path: Optional[str] = None

    # --- Text inputs ---
    caption: str = ""
    lyrics: str = ""
    thinking: bool = True

    # --- Remix strengths ---
    lm_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="audio_cover_strength")
    cover_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="cover_noise_strength")

    # --- LM settings ---
    lm_temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    lm_cfg_scale: float = Field(default=2.0, ge=1.0, le=3.0)
    lm_top_k: int = Field(default=0, ge=0, le=100)
    lm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    lm_repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    lm_negative_prompt: str = "NO USER INPUT"

    # --- DiT settings ---
    inference_steps: int = Field(default=32, ge=1, le=200)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=15.0)
    infer_method: Literal["ode", "sde"] = "ode"
    use_adg: bool = False
    shift: float = Field(default=3.0, ge=1.0, le=5.0)
    cfg_interval_start: float = Field(default=0.0, ge=0.0, le=1.0)
    cfg_interval_end: float = Field(default=1.0, ge=0.0, le=1.0)

    # --- Optional parameters ---
    keyscale: str = ""
    timesignature: str = ""
    vocal_language: str = "unknown"
    seed: int = Field(default=-1)
    batch_size: int = Field(default=1, ge=1, le=8)

    # --- Output naming (set by server) ---
    output_filename: str = ""


class TranscribeRequest(BaseModel):
    """Request body for Whisper transcription endpoint."""

    audio_path: str = Field(description="Server-side temp path of the audio file")
    language: Optional[str] = Field(default=None, description="ISO language code hint, or null for auto")
    whisper_model: str = Field(default="base", description="Whisper model size (tiny/base/small/medium/large-v2/large-v3)")


class ProcessAudioRequest(BaseModel):
    """Request to apply pre-processing to source audio before ACE-Step."""

    audio_path: str = Field(description="Server-side temp path of the source audio")
    pitch_shift_semitones: float = Field(default=0.0, ge=-12.0, le=12.0)
    apply_low_cut: bool = False
    apply_noise_gate: bool = False


class SaveResultRequest(BaseModel):
    """Request to save an ACE-Step result to the output directory."""

    audio_src_path: str
    input_filename: str
    index: int = 1


class RvcRequest(BaseModel):
    """Parameters for an RVC voice conversion job."""

    input_path: str = Field(description="Server-side path to source audio (ACE-Step result or source audio)")
    model_path: str = Field(description="Absolute path to .pth model file")
    index_path: str = Field(default="", description="Absolute path to .index file, or empty string")

    pitch: int = Field(default=0, ge=-24, le=24, description="Pitch shift in semitones")
    f0_method: str = Field(default="rmvpe", description="F0 extraction method")
    index_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Index search feature ratio")
    volume_envelope: float = Field(default=0.3, ge=0.0, le=1.0, description="Volume envelope mix ratio")
    protect: float = Field(default=0.3, ge=0.0, le=0.5, description="Protect voiceless consonants ratio")
    clean_audio: bool = Field(default=True, description="Apply audio cleaning post-processing")
    clean_strength: float = Field(default=0.3, ge=0.0, le=1.0, description="Clean audio strength")
    embedder_model: str = Field(default="contentvec", description="Speaker embedding model")
    filter_radius: int = Field(default=3, ge=0, le=7, description="F0 median filter radius")
    seed: int = Field(default=-1, description="Random seed (-1 = random)")
    cuda_device: str = Field(default="auto", description="CUDA device selector (auto/0/1/cuda:0/etc)")

    # Hardcoded in runner: export_format=WAV, hop_length=64
