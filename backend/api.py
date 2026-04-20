"""FastAPI application for Streamline Vocals.

Routes:
  GET  /                          → index.html
  GET  /api/health                → health check (ACE-Step + whisper status)
  GET  /api/config                → checkpoint + LoRA listings
  POST /api/upload-audio          → save uploaded audio to temp on disk
  POST /api/process-audio         → apply pitch shift / filters, return new temp path
  POST /api/transcribe            → Whisper transcription of source audio
  POST /api/generate              → start ACE-Step remix job
  GET  /api/jobs/{job_id}         → poll generation status
  POST /api/save-result           → copy ACE-Step output to output directory
  GET  /api/audio/temp            → stream a temp audio file by path
  GET  /api/models/state          → loaded DiT/LM model state
  POST /api/models/switch         → restart ACE-Step with different models
  GET  /api/hardware/cuda-devices → detected CUDA devices
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import subprocess

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from . import proxy
from .api_process_manager import ApiProcessManager, DEFAULT_DIT_MODEL, DEFAULT_LM_MODEL
from .models import ProcessAudioRequest, RemixRequest, RvcRequest, SaveResultRequest, TranscribeRequest

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_output_dir: Path = Path("streamline_svc/output")
_lora_dir: Path = Path("lora")
_checkpoints_dir: Path = Path("checkpoints")
_lm_models_dir: Path = Path("lm_models")
_frontend_dir: Path = Path(__file__).parent.parent / "frontend"
_api_process_manager: ApiProcessManager | None = None

_config_cache: dict[str, Any] = {}
_config_cache_ts: float = 0.0
_CONFIG_CACHE_TTL: float = 15.0


class SwitchModelsRequest(BaseModel):
    """Request body for model switch action."""

    dit_model: str
    lm_model: str
    compile_model: bool | None = None
    use_flash_attention: bool | None = None
    offload_to_cpu: str | None = None
    offload_dit_to_cpu: bool | None = None
    mlx_patches_enabled: bool | None = None
    lora_slot_patch_enabled: bool | None = None
    cuda_device: str | None = None


def create_app(
    output_dir: Path,
    lora_dir: Path,
    checkpoints_dir: Path,
    api_process_manager: ApiProcessManager | None = None,
) -> FastAPI:
    """Build and return the FastAPI application instance.

    Args:
        output_dir: Directory where final processed files are saved.
        lora_dir: Root directory containing LoRA subfolders.
        checkpoints_dir: Root directory containing DiT checkpoint subfolders.
        api_process_manager: Optional manager for the ACE-Step child process.

    Returns:
        Configured FastAPI app ready for uvicorn.
    """
    global _output_dir, _lora_dir, _checkpoints_dir, _lm_models_dir, _api_process_manager

    _output_dir = output_dir
    _lora_dir = lora_dir
    _checkpoints_dir = checkpoints_dir
    _lm_models_dir = checkpoints_dir.parent / "lm_models"
    _api_process_manager = api_process_manager

    app = FastAPI(title="Streamline Vocals", version="0.1.0", docs_url="/api/docs")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Return server status and ACE-Step connectivity."""
        api = await proxy.health_status()
        return {
            "vocals": "ok",
            "acestep_api": "ok" if api["connected"] else "unavailable",
            "acestep_ready": api["ready"],
            "models_initialized": api["models_initialized"],
            "llm_initialized": api["llm_initialized"],
            "loaded_model": api["loaded_model"],
            "loaded_lm_model": api["loaded_lm_model"],
            "acestep_url": os.environ.get("ACESTEP_API_URL", "http://127.0.0.1:8001"),
        }

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """Return available checkpoints and LoRA options (cached 15 s)."""
        global _config_cache, _config_cache_ts
        now = time.monotonic()
        if _config_cache and (now - _config_cache_ts) < _CONFIG_CACHE_TTL:
            return _config_cache
        _config_cache = {
            "checkpoints": _scan_checkpoints(),
            "loras": _scan_loras(),
            "lm_models": _scan_lm_models(),
        }
        _config_cache_ts = now
        return _config_cache

    # ------------------------------------------------------------------
    # Model state / switch
    # ------------------------------------------------------------------

    @app.get("/api/hardware/cuda-devices")
    async def cuda_devices() -> dict[str, Any]:
        """Return detected CUDA devices with indices and GPU names."""
        devices = _detect_cuda_devices()
        return {"devices": devices}

    @app.get("/api/models/state")
    async def models_state() -> dict[str, Any]:
        """Return currently loaded and persisted DiT/LM model selections."""
        saved = (
            _api_process_manager.get_saved_models()
            if _api_process_manager
            else {"dit_model": DEFAULT_DIT_MODEL, "lm_model": DEFAULT_LM_MODEL}
        )
        loaded = await proxy.get_active_models()
        return {
            "loaded": loaded,
            "saved": saved,
            "managed": bool(_api_process_manager and _api_process_manager.owns_process),
        }

    @app.post("/api/models/switch")
    async def switch_models(req: SwitchModelsRequest) -> dict[str, Any]:
        """Restart ACE-Step API server with selected models or settings when changed."""
        dit_model = (req.dit_model or "").strip() or DEFAULT_DIT_MODEL
        lm_model = (req.lm_model or "").strip() or DEFAULT_LM_MODEL

        loaded = await proxy.get_active_models()
        models_unchanged = loaded.get("dit_model") == dit_model and loaded.get("lm_model") == lm_model

        if models_unchanged:
            # Check whether server-affecting settings also differ
            saved = _api_process_manager.get_saved_models() if _api_process_manager else {}
            req_cuda = ApiProcessManager._normalize_cuda_device(req.cuda_device)
            saved_cuda = ApiProcessManager._normalize_cuda_device(str(saved.get("cuda_device") or "auto"))
            req_lora_patch = bool(req.lora_slot_patch_enabled)
            saved_lora_patch = bool(saved.get("lora_slot_patch_enabled", False))
            settings_changed = req_cuda != saved_cuda or req_lora_patch != saved_lora_patch

            if not settings_changed:
                if _api_process_manager:
                    _api_process_manager.save_models(
                        dit_model,
                        lm_model,
                        mlx_patches_enabled=req.mlx_patches_enabled,
                        lora_slot_patch_enabled=req.lora_slot_patch_enabled,
                        cuda_device=req.cuda_device,
                    )
                return {
                    "status": "unchanged",
                    "message": "Selected models are already loaded",
                    "loaded": loaded,
                }

        if not _api_process_manager:
            raise HTTPException(500, detail="Model switch manager is not available")

        try:
            _api_process_manager.restart(
                dit_model,
                lm_model,
                compile_model=req.compile_model,
                use_flash_attention=req.use_flash_attention,
                offload_to_cpu=req.offload_to_cpu,
                offload_dit_to_cpu=req.offload_dit_to_cpu,
                mlx_patches_enabled=req.mlx_patches_enabled,
                lora_slot_patch_enabled=req.lora_slot_patch_enabled,
                cuda_device=req.cuda_device,
            )
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[Vocals] Failed to restart ACE-Step server")
            raise HTTPException(500, detail=f"Failed to restart ACE-Step server: {exc}") from exc

        global _config_cache, _config_cache_ts
        _config_cache = {}
        _config_cache_ts = 0.0

        return {
            "status": "restarted",
            "message": "ACE-Step server restarted with selected models",
            "loaded": {"dit_model": dit_model, "lm_model": lm_model},
        }

    # ------------------------------------------------------------------
    # Audio upload
    # ------------------------------------------------------------------

    @app.post("/api/upload-audio")
    async def upload_audio(file: UploadFile = File(...)) -> dict[str, str]:
        """Save an uploaded audio file to a temp location on disk."""
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="vocals_audio_")
        try:
            with os.fdopen(fd, "wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()
        return {"temp_path": temp_path}

    # ------------------------------------------------------------------
    # Audio processing (pitch shift, filters)
    # ------------------------------------------------------------------

    @app.post("/api/process-audio")
    async def process_audio_endpoint(req: ProcessAudioRequest) -> dict[str, str]:
        """Apply pitch shift and/or filters to source audio.

        Returns the server-side path to the processed temp WAV file.
        """
        if not req.audio_path or not Path(req.audio_path).exists():
            raise HTTPException(400, detail="audio_path is missing or does not exist")

        try:
            from .audio_processor import process_audio

            loop = asyncio.get_running_loop()
            out_path = await loop.run_in_executor(
                None,
                lambda: process_audio(
                    req.audio_path,
                    pitch_shift_semitones=req.pitch_shift_semitones,
                    apply_low_cut=req.apply_low_cut,
                    apply_noise_gate=req.apply_noise_gate,
                ),
            )
        except Exception as exc:
            logger.exception("[Vocals] Audio processing failed")
            raise HTTPException(500, detail=f"Audio processing failed: {exc}") from exc

        return {"processed_path": out_path}

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------

    @app.post("/api/transcribe")
    async def transcribe(req: TranscribeRequest) -> dict[str, str]:
        """Transcribe audio to text using the selected Whisper model."""
        if not req.audio_path or not Path(req.audio_path).exists():
            raise HTTPException(400, detail="audio_path is missing or does not exist")

        try:
            from .transcriber import transcribe as _transcribe

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                lambda: _transcribe(req.audio_path, language=req.language, model_name=req.whisper_model),
            )
        except RuntimeError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[Vocals] Transcription failed")
            raise HTTPException(500, detail=f"Transcription failed: {exc}") from exc

        return {"text": text}

    # ------------------------------------------------------------------
    # Generation (ACE-Step remix)
    # ------------------------------------------------------------------

    @app.post("/api/generate")
    async def generate(req: RemixRequest) -> dict[str, Any]:
        """Start an ACE-Step remix generation job and return the job ID."""
        resolved_seed = req.seed if req.seed != -1 else random.randint(0, 2_147_483_647)
        payload = _build_remix_payload(req, resolved_seed)

        active_loras = req.loras or []

        # LoRA loading — two paths depending on whether the slot-based
        # monkeypatch is enabled (persisted in state file, applied at startup).
        _slot_patch_on = bool(
            _api_process_manager and
            _api_process_manager.get_saved_models().get("lora_slot_patch_enabled", False)
        )

        if _slot_patch_on:
            # Slot-based path: smart caching, multi-adapter, per-group scales.
            if active_loras:
                current_status = await proxy.get_slots_status()
                current_slots: dict[int, dict] = {
                    int(s["slot"]): s
                    for s in current_status.get("slots", [])
                }

                slots_to_reload: list[dict[str, Any]] = []
                slots_to_rescale: list[tuple[int, Any]] = []

                for slot_idx, entry in enumerate(active_loras):
                    lora_path = str((_lora_dir / entry.name).resolve())
                    existing = current_slots.get(slot_idx)
                    if existing and existing.get("path") == lora_path:
                        slots_to_rescale.append((slot_idx, entry))
                    else:
                        gs = entry.group_scales or {}
                        use_group_scales = any(
                            abs(gs.get(k, 1.0) - 1.0) > 1e-9
                            for k in ("self_attn", "cross_attn", "mlp")
                        )
                        slots_to_reload.append({
                            "lora_path": lora_path,
                            "slot": slot_idx,
                            "scale": entry.scale,
                            "group_scales": gs if use_group_scales else None,
                        })

                requested_slot_indices = set(range(len(active_loras)))
                stale_slots = set(current_slots.keys()) - requested_slot_indices
                if stale_slots or slots_to_reload:
                    try:
                        await proxy.unload_lora_all()
                    except Exception as exc:
                        logger.warning(f"LoRA unload-all failed (continuing): {exc}")
                    for slot_idx, entry in slots_to_rescale:
                        lora_path = str((_lora_dir / entry.name).resolve())
                        gs = entry.group_scales or {}
                        use_group_scales = any(
                            abs(gs.get(k, 1.0) - 1.0) > 1e-9
                            for k in ("self_attn", "cross_attn", "mlp")
                        )
                        slots_to_reload.append({
                            "lora_path": lora_path,
                            "slot": slot_idx,
                            "scale": entry.scale,
                            "group_scales": gs if use_group_scales else None,
                        })
                    slots_to_rescale = []

                if slots_to_reload:
                    try:
                        results = await proxy.load_lora_slots_batch(slots_to_reload)
                        for r in results:
                            if r.startswith("❌"):
                                logger.warning(f"LoRA batch load: {r}")
                            else:
                                logger.debug(f"LoRA batch load: {r}")
                    except Exception as exc:
                        logger.warning(f"LoRA batch load failed (continuing): {exc}")

                for slot_idx, entry in slots_to_rescale:
                    try:
                        await proxy.update_lora_slot_scale(slot_idx, entry.scale)
                    except Exception as exc:
                        logger.warning(f"LoRA scale update slot {slot_idx} failed: {exc}")
                    gs = entry.group_scales or {}
                    use_group_scales = any(
                        abs(gs.get(k, 1.0) - 1.0) > 1e-9
                        for k in ("self_attn", "cross_attn", "mlp")
                    )
                    if use_group_scales:
                        try:
                            await proxy.update_lora_slot_group_scales(slot_idx, gs)
                        except Exception as exc:
                            logger.warning(f"LoRA group_scales update slot {slot_idx} failed: {exc}")
            else:
                current_status = await proxy.get_slots_status()
                if current_status.get("slot_count", 0) > 0:
                    try:
                        await proxy.unload_lora_all()
                    except Exception as exc:
                        logger.warning(f"LoRA unload-all failed (continuing): {exc}")

        else:
            # Simple path: native ACE-Step /v1/lora/load + /v1/lora/unload.
            first_entry = active_loras[0] if active_loras else None
            if first_entry:
                lora_path = str((_lora_dir / first_entry.name).resolve())
                try:
                    await proxy.load_lora_simple(lora_path, first_entry.scale)
                except Exception as exc:
                    logger.warning(f"LoRA load (simple) failed (continuing): {exc}")
            else:
                try:
                    await proxy.unload_lora_simple()
                except Exception as exc:
                    logger.warning(f"LoRA unload (simple) failed (continuing): {exc}")

        try:
            job_id = await proxy.start_generation(payload)
        except RuntimeError as exc:
            logger.exception("[Vocals] start_generation failed")
            raise HTTPException(503, detail=str(exc)) from exc

        if _api_process_manager and req.dit_model:
            _api_process_manager.save_models(req.dit_model, "")

        return {"job_id": job_id, "status": "queued", "seed": resolved_seed}

    @app.get("/api/jobs/{job_id}")
    async def poll_job(job_id: str) -> dict[str, Any]:
        """Poll the status of a generation job."""
        try:
            results = await proxy.query_jobs([job_id])
        except Exception as exc:
            raise HTTPException(502, detail=f"Failed to query job: {exc}") from exc

        if not results:
            return {"job_id": job_id, "status": "queued", "progress": 0.0}

        job = results[0]
        audio_paths: list[str] = job.get("audio_paths") or []
        raw_audio_paths: list[str] = job.get("raw_audio_paths") or []
        audio_urls = [
            f"{os.environ.get('ACESTEP_API_URL', 'http://127.0.0.1:8001')}{p}"
            for p in audio_paths
            if p
        ]
        return {
            "job_id": job_id,
            "status": job.get("status", "queued"),
            "progress": 0.0,
            "message": job.get("progress_text", ""),
            "audio_urls": audio_urls,
            "raw_audio_paths": raw_audio_paths,
            "seed": -1,
            "duration": job.get("duration", 0.0),
        }

    # ------------------------------------------------------------------
    # RVC config (model / index file listings)
    # ------------------------------------------------------------------

    @app.get("/api/rvc/config")
    async def rvc_config() -> dict[str, list[str]]:
        """Return available .pth and .index files from applio/logs."""
        from .rvc_runner import scan_models

        return scan_models()

    # ------------------------------------------------------------------
    # RVC inference
    # ------------------------------------------------------------------

    @app.post("/api/rvc/run")
    async def rvc_run(req: RvcRequest) -> dict[str, str]:
        """Run RVC voice conversion and return a temp path to the result WAV."""
        if not req.input_path or not Path(req.input_path).exists():
            raise HTTPException(400, detail="input_path is missing or does not exist")
        if not req.model_path or not Path(req.model_path).exists():
            raise HTTPException(400, detail="model_path does not exist")

        suffix = ".wav"
        fd, out_path = tempfile.mkstemp(suffix=suffix, prefix="vocals_rvc_")
        os.close(fd)

        try:
            from .rvc_runner import run_rvc

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: run_rvc(
                    input_path=req.input_path,
                    output_path=out_path,
                    model_path=req.model_path,
                    index_path=req.index_path or "",
                    pitch=req.pitch,
                    f0_method=req.f0_method,
                    index_rate=req.index_rate,
                    volume_envelope=req.volume_envelope,
                    protect=req.protect,
                    clean_audio=req.clean_audio,
                    clean_strength=req.clean_strength,
                    embedder_model=req.embedder_model,
                    filter_radius=req.filter_radius,
                    seed=req.seed,
                    cuda_device=req.cuda_device or "auto",
                ),
            )
        except RuntimeError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[Vocals] RVC inference failed")
            raise HTTPException(500, detail=f"RVC inference failed: {exc}") from exc

        return {"output_path": out_path}

    # ------------------------------------------------------------------
    # Native folder picker
    # ------------------------------------------------------------------

    @app.get("/api/pick-folder")
    async def pick_folder(initial: str = "") -> dict[str, str | None]:
        """Open a native OS folder-picker dialog and return the chosen path.

        Uses tkinter.filedialog on all platforms (subprocess-safe).  Returns
        ``{"path": "/abs/path"}`` on success or ``{"path": null}`` when the
        user cancels without selecting a folder.
        """
        import subprocess
        import sys

        initial_dir = initial or str(_output_dir)

        # Run tkinter in a one-shot subprocess so it doesn't block the
        # asyncio event loop and works even if the server was started without
        # a display context on some platforms.
        script = (
            "import tkinter, tkinter.filedialog; "
            "root = tkinter.Tk(); root.withdraw(); root.call('wm', 'attributes', '.', '-topmost', True); "
            f"d = tkinter.filedialog.askdirectory(initialdir={initial_dir!r}); "
            "print(d if d else '', end='')"
        )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                ).stdout.strip(),
            )
        except Exception as exc:
            logger.warning(f"[Vocals] pick-folder failed: {exc}")
            return {"path": None}

        return {"path": result if result else None}

    # ------------------------------------------------------------------
    # Save result to output directory
    # ------------------------------------------------------------------

    @app.post("/api/save-result")
    async def save_result(req: SaveResultRequest) -> dict[str, str]:
        """Copy an ACE-Step result to the output directory with a formatted name.

        The output filename follows the pattern:
            {input_filename}_processed_{index}.wav
        """
        src = Path(req.audio_src_path)
        if not src.exists():
            raise HTTPException(400, detail="audio_src_path does not exist")

        stem = Path(req.input_filename).stem if req.input_filename else "output"
        dest_name = f"{stem}_processed_{req.index}.wav"
        dest = _output_dir / dest_name
        _output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to 16-bit WAV if not already
        loop = asyncio.get_running_loop()
        saved_path = await loop.run_in_executor(None, lambda: _copy_as_wav16(src, dest))

        return {"saved_path": str(saved_path), "filename": dest_name}

    # ------------------------------------------------------------------
    # Temp audio streaming (for ACE-Step result preview)
    # ------------------------------------------------------------------

    @app.get("/api/audio/temp")
    async def stream_temp_audio(path: str) -> FileResponse:
        """Stream a temp audio file by absolute server path."""
        p = Path(path)
        if not p.exists():
            raise HTTPException(404, detail="Audio file not found")
        ext = p.suffix.lstrip(".")
        media_map = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}
        return FileResponse(str(p), media_type=media_map.get(ext, "audio/wav"))

    # ------------------------------------------------------------------
    # Static frontend (must be LAST)
    # ------------------------------------------------------------------

    if _frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    else:
        logger.warning(f"Frontend directory not found: {_frontend_dir}")

    return app


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _scan_checkpoints() -> list[str]:
    """Return available DiT checkpoint names."""
    if not _checkpoints_dir.exists():
        return []
    return sorted(d.name for d in _checkpoints_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def _detect_cuda_devices() -> list[dict[str, str | int]]:
    """Detect CUDA devices, preferring torch and falling back to nvidia-smi.

    Returns:
        List of device records with ``index``, ``value`` (cuda:N), and ``name``.
    """
    devices = _detect_cuda_devices_via_torch()
    if devices:
        return devices
    return _detect_cuda_devices_via_nvidia_smi()


def _detect_cuda_devices_via_torch() -> list[dict[str, str | int]]:
    """Detect CUDA devices through torch runtime if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count())
        return [
            {"index": idx, "value": f"cuda:{idx}", "name": str(torch.cuda.get_device_name(idx))}
            for idx in range(count)
        ]
    except Exception as exc:
        logger.debug(f"[Vocals] torch CUDA detection failed: {exc}")
        return []


def _detect_cuda_devices_via_nvidia_smi() -> list[dict[str, str | int]]:
    """Detect CUDA-capable GPUs via nvidia-smi command output."""
    candidates = [
        shutil.which("nvidia-smi"),
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    nvidia_smi_exe = next((p for p in candidates if p and Path(p).exists()), None)
    if not nvidia_smi_exe:
        return []

    try:
        result = subprocess.run(
            [str(nvidia_smi_exe), "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception as exc:
        logger.debug(f"[Vocals] nvidia-smi invocation failed: {exc}")
        return []

    if result.returncode != 0:
        return []

    devices: list[dict[str, str | int]] = []
    for line in (result.stdout or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split(",", 1)
        if len(parts) != 2 or not parts[0].strip().isdigit():
            continue
        idx = int(parts[0].strip())
        devices.append({"index": idx, "value": f"cuda:{idx}", "name": parts[1].strip()})
    return devices


def _scan_loras() -> list[str]:
    """Return available LoRA folder names."""
    if not _lora_dir.exists():
        return []
    return sorted(d.name for d in _lora_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def _scan_lm_models() -> list[str]:
    """Return available LM model names."""
    if _lm_models_dir.exists():
        return sorted(d.name for d in _lm_models_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not _checkpoints_dir.exists():
        return []
    return sorted(
        d.name
        for d in _checkpoints_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name.lower().startswith("acestep-5hz-lm-")
    )


def _seeded_auto_duration(seed: int) -> float:
    """Return a deterministic auto-duration based on seed (120–240 s)."""
    rng = random.Random(seed)
    return round(rng.uniform(120.0, 240.0), 1)


def _build_remix_payload(req: RemixRequest, resolved_seed: int) -> dict[str, Any]:
    """Translate a RemixRequest into an ACE-Step release_task payload.

    Args:
        req: Validated RemixRequest instance.
        resolved_seed: Final resolved seed value.

    Returns:
        Dict ready to POST to ACE-Step /release_task.
    """
    payload: dict[str, Any] = {
        "prompt": req.caption,
        "lyrics": req.lyrics if req.lyrics and req.lyrics.strip() else "[Instrumental]",
        "thinking": req.thinking,
        "task_type": "cover",
        "inference_steps": req.inference_steps,
        "guidance_scale": req.guidance_scale,
        "infer_method": req.infer_method,
        "use_adg": req.use_adg,
        "shift": req.shift,
        "cfg_interval_start": req.cfg_interval_start,
        "cfg_interval_end": req.cfg_interval_end,
        "lm_temperature": req.lm_temperature,
        "lm_cfg_scale": req.lm_cfg_scale,
        "lm_top_k": req.lm_top_k,
        "lm_top_p": req.lm_top_p,
        "lm_repetition_penalty": req.lm_repetition_penalty,
        "lm_negative_prompt": req.lm_negative_prompt,
        "use_cot_caption": False,
        "audio_format": "wav",
        "seed": resolved_seed,
        "use_random_seed": False,
        "vocal_language": "" if req.vocal_language == "unknown" else req.vocal_language,
        "audio_cover_strength": req.lm_strength,
        "cover_noise_strength": req.cover_strength,
        "batch_size": req.batch_size,
        "audio_duration": _seeded_auto_duration(resolved_seed),
    }

    if req.keyscale:
        payload["key_scale"] = req.keyscale
    if req.timesignature:
        payload["time_signature"] = req.timesignature
    if req.dit_model:
        payload["model"] = req.dit_model
    if req.ref_audio_path:
        payload["reference_audio_path"] = req.ref_audio_path
    if req.source_audio_path:
        payload["src_audio_path"] = req.source_audio_path

    return payload


def _copy_as_wav16(src: Path, dest: Path) -> Path:
    """Copy audio file to dest, ensuring 16-bit WAV format.

    If soundfile is available and the source is not already a 16-bit WAV,
    it is re-encoded.  Otherwise a plain file copy is performed.

    Args:
        src: Source audio file path.
        dest: Destination path (must end in .wav).

    Returns:
        The dest path.
    """
    try:
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(str(src), always_2d=False, dtype="float32")
        sf.write(str(dest), audio, sr, subtype="PCM_16")
    except Exception as exc:
        logger.warning(f"soundfile re-encode failed ({exc}), copying as-is")
        shutil.copy2(str(src), str(dest))
    return dest
