"""Streamline Vocals — server entry point.

Usage:
    python streamline_svc/start.py [--listen] [--port PORT]

Flags:
    --listen    Bind to 0.0.0.0 instead of 127.0.0.1.
    --port N    Override the bind port (default: 7861 or VOCALS_PORT).

Environment variables:
    VOCALS_HOST         Bind host (default: 127.0.0.1)
    VOCALS_PORT         Bind port (default: 7861)
    VOCALS_OUTPUT_DIR   Default output directory (default: streamline_svc/output)
    ACESTEP_API_URL     ACE-Step API base URL (default: http://127.0.0.1:8001)
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from loguru import logger

from streamline_svc.backend.api import create_app
from streamline_svc.backend.api_process_manager import ApiProcessManager


def _get_local_ip() -> str:
    """Best-effort local network IP for display when --listen is used."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "<your-local-ip>"


def main() -> None:
    """Configure and launch the Streamline Vocals uvicorn server."""
    # Prevent uv-injected interpreter vars from leaking into child processes.
    for key in ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME"):
        os.environ.pop(key, None)

    parser = argparse.ArgumentParser(description="Streamline Vocals server")
    parser.add_argument(
        "--listen",
        action="store_true",
        default=os.environ.get("VOCALS_LISTEN", "").lower() in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VOCALS_PORT", "7861")),
    )
    args = parser.parse_args()

    host = "0.0.0.0" if args.listen else os.environ.get("VOCALS_HOST", "127.0.0.1")
    port = args.port
    acestep_host = os.environ.get("ACESTEP_API_HOST", "127.0.0.1")
    acestep_port = int(os.environ.get("ACESTEP_API_PORT", "8001"))

    output_dir = Path(os.environ.get("VOCALS_OUTPUT_DIR", str(_HERE / "output")))
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dir = _HERE / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = _ROOT / "lora"
    checkpoints_dir = _ROOT / "checkpoints"

    api_process_manager = ApiProcessManager(
        project_root=_ROOT,
        host=acestep_host,
        port=acestep_port,
        state_file=state_dir / "acestep_model_state.json",
    )
    startup_models = api_process_manager.ensure_started()
    app = create_app(output_dir, lora_dir, checkpoints_dir, api_process_manager)

    logger.info(f"Streamline Vocals → http://{host}:{port}")
    if args.listen:
        local_ip = _get_local_ip()
        logger.info(f"Network access → http://{local_ip}:{port}")
    logger.info(f"Output dir:    {output_dir}")
    logger.info(f"ACE-Step API:  http://{acestep_host}:{acestep_port}")
    logger.info(
        f"ACE-Step startup models: DiT={startup_models['dit_model']}, "
        f"LM={startup_models['lm_model']}"
    )

    # Open the UI in the default browser after a short delay so uvicorn is ready
    ui_url = f"http://127.0.0.1:{port}"
    threading.Timer(1.5, webbrowser.open, args=[ui_url]).start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        logger.info("[Vocals] Shutting down ACE-Step API process…")
        api_process_manager.stop()


if __name__ == "__main__":
    main()
