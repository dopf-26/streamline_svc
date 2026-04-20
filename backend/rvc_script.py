"""Standalone RVC inference entry-point.

Invoked by rvc_runner.py in a subprocess:
    python rvc_script.py '<json-params>'

Parameters are passed as a single JSON string argument.  On success, prints
a JSON object {"status": "ok"} to stdout.  On failure, exits with code 1 and
prints the error message to stderr.

Running in a subprocess fully isolates the RVC inference from the uvicorn
event loop: native crashes (MPS OOM, faiss segfault, etc.) cannot kill the
server process.
"""

from __future__ import annotations

import json
import os
import sys


def _configure_single_threaded_torch() -> None:
    """Restrict torch to one intra-op thread before any model is loaded.

    On macOS ARM64, allowing PyTorch to spawn multiple BLAS/OpenMP threads
    while running inside a subprocess causes POSIX semaphore leaks and
    SIGSEGV during HuBERT / RMVPE forward passes.
    """
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001 — best-effort; not fatal if torch not ready
        pass


def main() -> None:
    """Parse params from CLI, run RVC inference, exit cleanly."""
    if len(sys.argv) < 2:
        sys.exit("Usage: rvc_script.py '<json-params>'")

    params: dict = json.loads(sys.argv[1])
    applio_dir: str = params.pop("applio_dir")

    # Set applio as cwd and add to path so all relative imports resolve.
    os.chdir(applio_dir)

    # Restrict torch thread counts BEFORE importing any torch-based library.
    _configure_single_threaded_torch()
    if applio_dir not in sys.path:
        sys.path.insert(0, applio_dir)

    # Download predictor models / embedders if missing.
    from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

    prequisites_download_pipeline(pretraineds_hifigan=False, models=True, exe=False)

    # Run inference.
    from rvc.infer.infer import VoiceConverter

    vc = VoiceConverter()
    vc.convert_audio(**params)

    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
