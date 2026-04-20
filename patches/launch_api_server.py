"""Streamline Vocals wrapper launcher for the ACE-Step API server.

Applies MPS/MLX performance patches to the live module namespace before
starting the real server via runpy.  Because runpy executes api_server.py
inside the *same* process, every import that api_server.py performs picks up
the already-patched module objects — so patches are active from the very first
model initialisation call.

Always used as the entry point (MPS and non-MPS alike) so that shared
cross-platform tweaks (e.g. log filters) apply uniformly.

argv is preserved unchanged so api_server.py argparse receives the same
--host / --port / --init-llm / --lm-model-path flags that were passed here.
"""

import logging
import runpy
import sys
import warnings
from pathlib import Path

# multiprocessing.resource_tracker emits a spurious "leaked semaphore" warning
# on clean shutdown when LLM worker threads are torn down.  It's a known false
# positive on macOS and doesn't indicate data loss or corruption.
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)


class _QueryResultFilter(logging.Filter):
    """Drop uvicorn access log lines for /query_result polling requests.

    The frontend polls /query_result every ~1 s during generation; logging
    every hit produces hundreds of identical lines that bury real output.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Return False (suppress) for /query_result access log entries."""
        msg = record.getMessage()
        return "/query_result" not in msg


# Install the filter now; uvicorn.access logger is created lazily so we attach
# to the root handler chain via the named logger which uvicorn always uses.
_access_logger = logging.getLogger("uvicorn.access")
_access_logger.addFilter(_QueryResultFilter())

# Ensure the project root is on sys.path so both `acestep` and `streamline_svc`
# are importable without relying on the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

from streamline_svc.patches.mlx_optimizations import apply_mlx_patches

apply_mlx_patches()

# Apply slot-based LoRA patch if enabled
if os.environ.get("STREAMLINE_LORA_SLOT_PATCH", "0") == "1":
    from streamline_svc.patches.lora_slot_patch import apply_lora_slot_patch
    apply_lora_slot_patch()

# Hand off to the real API server entry point.
_API_SERVER = str(_PROJECT_ROOT / "acestep" / "api_server.py")
# Correct argv[0] so argparse --help and error messages show the right script name.
sys.argv[0] = _API_SERVER
runpy.run_path(_API_SERVER, run_name="__main__")
