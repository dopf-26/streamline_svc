"""ACE-Step API process lifecycle manager for Streamline.

This module owns starting/stopping the local ACE-Step API server process,
persisting the selected DiT/LM models, and restoring them on next startup.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from loguru import logger

DEFAULT_DIT_MODEL = "acestep-v15-base"
DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"


class ApiProcessManager:
    """Manage a child ACE-Step API process for Streamline.

    The manager starts ACE-Step with persisted DiT/LM selections and can
    restart it with new models on demand.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        host: str,
        port: int,
        state_file: Path,
    ) -> None:
        self._project_root = project_root
        self._host = host
        self._port = port
        self._state_file = state_file
        self._process: subprocess.Popen[str] | None = None

    @property
    def owns_process(self) -> bool:
        """Return True when this manager currently owns a running process."""
        return self._process is not None and self._process.poll() is None

    def get_saved_models(self) -> dict[str, str | bool]:
        """Return persisted models and settings, defaulting to base on first startup."""
        if self._state_file.exists():
            try:
                payload = json.loads(self._state_file.read_text(encoding="utf-8"))
                dit_model = str(payload.get("dit_model") or "").strip() or DEFAULT_DIT_MODEL
                lm_model = str(payload.get("lm_model") or "").strip() or DEFAULT_LM_MODEL
                mlx_patches_enabled = bool(payload.get("mlx_patches_enabled", False))
                lora_slot_patch_enabled = bool(payload.get("lora_slot_patch_enabled", False))
                cuda_device = str(payload.get("cuda_device") or "auto").strip() or "auto"
                return {
                    "dit_model": dit_model,
                    "lm_model": lm_model,
                    "mlx_patches_enabled": mlx_patches_enabled,
                    "lora_slot_patch_enabled": lora_slot_patch_enabled,
                    "cuda_device": cuda_device,
                }
            except Exception as exc:
                logger.warning(f"[Streamline] Could not read model state file: {exc}")
        return {
            "dit_model": DEFAULT_DIT_MODEL,
            "lm_model": DEFAULT_LM_MODEL,
            "mlx_patches_enabled": False,
            "lora_slot_patch_enabled": False,
            "cuda_device": "auto",
        }

    def save_models(
        self,
        dit_model: str,
        lm_model: str,
        mlx_patches_enabled: bool | None = None,
        lora_slot_patch_enabled: bool | None = None,
        cuda_device: str | None = None,
    ) -> None:
        """Persist the selected DiT/LM models and settings for future startups."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if self._state_file.exists():
            try:
                existing = json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing["dit_model"] = dit_model
        existing["lm_model"] = lm_model
        if mlx_patches_enabled is not None:
            existing["mlx_patches_enabled"] = mlx_patches_enabled
        if lora_slot_patch_enabled is not None:
            existing["lora_slot_patch_enabled"] = lora_slot_patch_enabled
        if cuda_device is not None:
            existing["cuda_device"] = self._normalize_cuda_device(cuda_device)
        self._state_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def ensure_started(self) -> dict[str, str | bool]:
        """Start the API process if needed using persisted model selection."""
        models = self.get_saved_models()
        if self.owns_process:
            return models
        if self._is_reachable():
            logger.info(
                f"[Streamline] ACE-Step API already running at http://{self._host}:{self._port}; "
                "keeping existing process"
            )
            return models
        self.start(
            str(models["dit_model"]),
            str(models["lm_model"]),
            mlx_patches_enabled=bool(models.get("mlx_patches_enabled", False)),
            lora_slot_patch_enabled=bool(models.get("lora_slot_patch_enabled", False)),
            cuda_device=str(models.get("cuda_device") or "auto"),
        )
        return models

    def start(
        self,
        dit_model: str,
        lm_model: str,
        *,
        compile_model: bool | None = None,
        use_flash_attention: bool | None = None,
        offload_to_cpu: str | None = None,
        offload_dit_to_cpu: bool | None = None,
        mlx_patches_enabled: bool | None = None,
        lora_slot_patch_enabled: bool | None = None,
        cuda_device: str | None = None,
    ) -> None:
        """Start ACE-Step API server with explicit DiT startup settings.

        Args:
            dit_model: DiT checkpoint name or path.
            lm_model: LM model name or path (kept for API compat; not loaded).
            compile_model: Enable torch.compile acceleration. None = server default.
            use_flash_attention: Enable Flash Attention 2. None = server default.
            offload_to_cpu: "true"/"false" to force CPU offload; "auto" or None = server auto-detect.
            offload_dit_to_cpu: Enable DiT-specific CPU offload. None = server default.
            mlx_patches_enabled: Apply MLX performance patches and bypass GPU tier limits on MPS.
            lora_slot_patch_enabled: Apply slot-based LoRA weight-merging monkeypatch.
            cuda_device: CUDA device selector (e.g. "cuda:0", "0", or "auto").
        """
        if self.owns_process:
            return

        command = self._build_command(mlx_patches_enabled=mlx_patches_enabled)
        env = self._build_clean_env(command[0])
        env["ACESTEP_API_HOST"] = self._host
        env["ACESTEP_API_PORT"] = str(self._port)
        env["ACESTEP_CONFIG_PATH"] = dit_model
        env["ACESTEP_INIT_LLM"] = "false"
        env["ACESTEP_NO_INIT"] = "false"

        if compile_model is not None:
            env["ACESTEP_COMPILE_MODEL"] = "true" if compile_model else "false"
        if use_flash_attention is not None:
            env["ACESTEP_USE_FLASH_ATTENTION"] = "true" if use_flash_attention else "false"
        if offload_to_cpu is not None and offload_to_cpu != "auto":
            env["ACESTEP_OFFLOAD_TO_CPU"] = offload_to_cpu
        if offload_dit_to_cpu is not None:
            env["ACESTEP_OFFLOAD_DIT_TO_CPU"] = "true" if offload_dit_to_cpu else "false"

        normalized_cuda_device = self._normalize_cuda_device(cuda_device)
        if normalized_cuda_device != "auto":
            env["CUDA_VISIBLE_DEVICES"] = normalized_cuda_device
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        if mlx_patches_enabled:
            env["STREAMLINE_MLX_PATCHES"] = "1"
            env["STREAMLINE_BYPASS_GPU_TIER"] = "1"
            env["ACESTEP_MLX_VAE_FP16"] = "1"
        else:
            env["STREAMLINE_MLX_PATCHES"] = "0"
            env.pop("STREAMLINE_BYPASS_GPU_TIER", None)
            env.pop("ACESTEP_MLX_VAE_FP16", None)

        env["STREAMLINE_LORA_SLOT_PATCH"] = "1" if lora_slot_patch_enabled else "0"

        logger.info(
            f"[Streamline] Starting ACE-Step API (DiT={dit_model}) at "
            f"http://{self._host}:{self._port}"
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        self._process = subprocess.Popen(
            command,
            cwd=str(self._project_root),
            env=env,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
            shell=False,
        )
        self.save_models(
            dit_model,
            lm_model,
            mlx_patches_enabled=mlx_patches_enabled,
            lora_slot_patch_enabled=lora_slot_patch_enabled,
            cuda_device=normalized_cuda_device,
        )

    def stop(self) -> None:
        """Stop the managed API process if running."""
        if not self._process:
            return
        if self._process.poll() is not None:
            self._process = None
            return

        logger.info("[Streamline] Stopping ACE-Step API process")
        self._process.terminate()
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("[Streamline] ACE-Step API did not exit in time; killing")
            self._process.kill()
            self._process.wait(timeout=10)
        finally:
            self._process = None

    def restart(
        self,
        dit_model: str,
        lm_model: str,
        *,
        pause_seconds: float = 1.0,
        compile_model: bool | None = None,
        use_flash_attention: bool | None = None,
        offload_to_cpu: str | None = None,
        offload_dit_to_cpu: bool | None = None,
        mlx_patches_enabled: bool | None = None,
        lora_slot_patch_enabled: bool | None = None,
        cuda_device: str | None = None,
    ) -> None:
        """Restart ACE-Step API process with a new DiT selection.

        Args:
            dit_model: DiT checkpoint name or path.
            lm_model: LM model name or path (kept for API compat; not loaded).
            pause_seconds: Seconds to wait between stop and start.
            compile_model: Enable torch.compile acceleration. None = server default.
            use_flash_attention: Enable Flash Attention 2. None = server default.
            offload_to_cpu: "true"/"false" to force CPU offload; "auto" or None = server auto-detect.
            offload_dit_to_cpu: Enable DiT-specific CPU offload. None = server default.
            mlx_patches_enabled: Apply MLX performance patches and bypass GPU tier limits on MPS.
            lora_slot_patch_enabled: Apply slot-based LoRA weight-merging monkeypatch.
            cuda_device: CUDA device selector (e.g. "cuda:0", "0", or "auto").
        """
        if not self.owns_process:
            raise RuntimeError(
                "ACE-Step server is not managed by Streamline in this session. "
                "Start both via start_streamline scripts to enable model switching."
            )
        self.stop()
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        self.start(
            dit_model,
            lm_model,
            compile_model=compile_model,
            use_flash_attention=use_flash_attention,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            mlx_patches_enabled=mlx_patches_enabled,
            lora_slot_patch_enabled=lora_slot_patch_enabled,
            cuda_device=cuda_device,
        )

    def _build_command(self, *, mlx_patches_enabled: bool | None = None) -> list[str]:
        """Build the python command for launching the ACE-Step API server.

        Routes through streamline_svc/patches/launch_api_server.py which applies
        cross-platform tweaks.  The LM is never loaded (no --init-llm flag).
        """
        script_path = (
            self._project_root / "streamline_svc" / "patches" / "launch_api_server.py"
        )

        python_exe = self._detect_python_exe()
        return [
            python_exe,
            str(script_path),
            "--host",
            self._host,
            "--port",
            str(self._port),
        ]

    @staticmethod
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

    def _build_clean_env(self, python_exe: str) -> dict[str, str]:
        """Build a clean environment for the ACE-Step subprocess.

        Strips all Python venv markers and PYTHONHOME/PYTHONPATH from the
        parent process environment and prepends the venv Scripts/bin directory.

        Args:
            python_exe: Absolute path to the Python interpreter being used.

        Returns:
            A dict suitable for passing as ``env`` to ``subprocess.Popen``.
        """
        env = os.environ.copy()
        for var in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE",
                    "UV_INTERNAL__PYTHONHOME"):
            env.pop(var, None)

        venv_scripts = str(Path(python_exe).parent)
        sep = os.pathsep
        existing_path = env.get("PATH", "")
        cleaned_paths = [
            p for p in existing_path.split(sep)
            if r"\.venv\\" not in p.lower() and "/.venv/" not in p.lower()
        ]
        env["PATH"] = venv_scripts + sep + sep.join(cleaned_paths)
        return env

    def _detect_python_exe(self) -> str:
        """Resolve the ACE-Step interpreter, preferring embedded then project-root .venv."""
        if os.name == "nt":
            embedded = self._project_root / "python_embedded" / "python.exe"
            venv = self._project_root / ".venv" / "Scripts" / "python.exe"
        else:
            embedded = self._project_root / "python_embedded" / "bin" / "python3.11"
            venv = self._project_root / ".venv" / "bin" / "python"

        if embedded.exists():
            return str(embedded)
        if venv.exists():
            return str(venv)

        raise RuntimeError(
            "ACE-Step Python interpreter not found at project root .venv. "
            f"Expected: {venv}"
        )

    def _is_reachable(self) -> bool:
        """Check whether the target API host/port is already accepting connections."""
        try:
            with socket.create_connection((self._host, self._port), timeout=1.0):
                return True
        except OSError:
            return False
