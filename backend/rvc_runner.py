"""RVC runner — drives Applio inference in an isolated subprocess.

Running inference in a subprocess (via rvc_script.py) ensures that native
crashes (MPS OOM, faiss segfault, numba JIT errors, etc.) cannot kill the
uvicorn server process.  The cwd / sys.path manipulation required by applio
is also fully contained inside the child process.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from loguru import logger

# Single-threaded BLAS/OpenMP settings passed to the subprocess.
# Multi-threaded BLAS combined with torch's own thread pool causes POSIX
# semaphore leaks and SIGSEGV on macOS ARM64 (Apple Silicon).
_SINGLE_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    # Fall back to CPU for any MPS op that isn't fully supported.
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}

_APPLIO_DIR = Path(__file__).resolve().parent.parent / "applio"
_RVC_SCRIPT = Path(__file__).resolve().parent / "rvc_script.py"
_SVC_ROOT = Path(__file__).resolve().parent.parent


def _svc_python_exe() -> str:
    """Return the streamline_svc virtualenv interpreter path.

    Raises:
        RuntimeError: If streamline_svc/.venv interpreter does not exist.
    """
    if os.name == "nt":
        venv_python = _SVC_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = _SVC_ROOT / ".venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    raise RuntimeError(
        "Streamline RVC Python interpreter not found in streamline_svc/.venv. "
        f"Expected: {venv_python}"
    )


def _normalize_cuda_device(cuda_device: str | None) -> str:
    """Normalize CUDA selector into a value suitable for CUDA_VISIBLE_DEVICES."""
    if cuda_device is None:
        return "auto"
    raw = str(cuda_device).strip().lower()
    if raw in ("", "auto", "none"):
        return "auto"
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1].strip()
    return raw if raw.isdigit() else "auto"


def run_rvc(
    input_path: str,
    output_path: str,
    model_path: str,
    index_path: str = "",
    pitch: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.5,
    volume_envelope: float = 0.3,
    protect: float = 0.3,
    hop_length: int = 64,
    clean_audio: bool = True,
    clean_strength: float = 0.3,
    embedder_model: str = "contentvec",
    filter_radius: int = 3,
    seed: int = -1,
    cuda_device: str = "auto",
) -> str:
    """Run RVC voice conversion in an isolated subprocess.

    Args:
        input_path: Absolute path to source audio file.
        output_path: Absolute path for the converted WAV output.
        model_path: Absolute path to the RVC .pth model file.
        index_path: Absolute path to the .index file, or empty string.
        pitch: Pitch shift in semitones (-24 to +24).
        f0_method: F0 extraction method (rmvpe/crepe/harvest/pm/dio).
        index_rate: Index search feature ratio (0.0-1.0).
        volume_envelope: Volume envelope mix ratio (0.0-1.0).
        protect: Protect voiceless consonants ratio (0.0-0.5).
        hop_length: Crepe hop length (hardcoded to 64 for quality).
        clean_audio: Whether to apply audio cleaning post-processing.
        clean_strength: Clean audio strength (0.0-1.0).
        embedder_model: Speaker embedding model name.
        filter_radius: Median filter radius for F0 smoothing (0-7).
        seed: Unused; reserved for future deterministic inference.
        cuda_device: CUDA device selector (auto/0/1/cuda:0/etc).

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If the applio directory / model is missing, or if the
            subprocess exits with a non-zero return code.
    """
    if not _APPLIO_DIR.exists():
        raise RuntimeError(f"Applio directory not found: {_APPLIO_DIR}")
    if not Path(model_path).exists():
        raise RuntimeError(f"RVC model not found: {model_path}")

    params = json.dumps({
        "applio_dir": str(_APPLIO_DIR),
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": model_path,
        "index_path": index_path or "",
        "pitch": pitch,
        "f0_method": f0_method,
        "index_rate": index_rate,
        "volume_envelope": volume_envelope,
        "protect": protect,
        "hop_length": hop_length,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "embedder_model": embedder_model,
        "export_format": "WAV",
        "filter_radius": filter_radius,
    })

    logger.info(f"[RVC] Starting subprocess inference -> {output_path}")
    env = {**os.environ, **_SINGLE_THREAD_ENV}
    
    # Set CUDA_VISIBLE_DEVICES if a specific device is requested
    normalized_device = _normalize_cuda_device(cuda_device)
    if normalized_device != "auto":
        env["CUDA_VISIBLE_DEVICES"] = normalized_device
    
    python_exe = _svc_python_exe()
    result = subprocess.run(
        [python_exe, str(_RVC_SCRIPT), params],
        capture_output=True,
        text=True,
        timeout=600,  # 10-minute hard cap; first run includes model downloads
        env=env,
    )

    if result.stdout:
        logger.debug(f"[RVC] stdout: {result.stdout.strip()}")
    if result.stderr:
        # applio prints tqdm progress + logs to stderr; only log as debug
        logger.debug(f"[RVC] stderr: {result.stderr.strip()}")

    if result.returncode != 0:
        # Surface the last 1000 chars of stderr for a useful error message
        tail = result.stderr.strip()[-1000:] if result.stderr else "(no stderr)"
        logger.error(f"[RVC] subprocess exited {result.returncode}:\n{tail}")
        raise RuntimeError(f"RVC subprocess failed (exit {result.returncode}): {tail}")

    logger.info(f"[RVC] Done -> {output_path}")
    return output_path


def scan_models() -> dict[str, list[str]]:
    """Scan applio/logs for available .pth and .index files.

    Returns:
        Dict with keys "pth_files" and "index_files", each a sorted list of
        absolute paths.
    """
    logs_dir = _APPLIO_DIR / "logs"
    pth_files: list[str] = []
    index_files: list[str] = []

    if logs_dir.exists():
        for path in sorted(logs_dir.rglob("*.pth")):
            pth_files.append(str(path))
        for path in sorted(logs_dir.rglob("*.index")):
            index_files.append(str(path))

    return {"pth_files": pth_files, "index_files": index_files}
