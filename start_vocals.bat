@echo off
REM Streamline Vocals – launch script (Windows)
REM ACE-Step API must already be running on port 8001.

cd /d "%~dp0"

if not exist ".venv" (
  echo [vocals] Creating uv venv...
  uv venv
)

echo [vocals] Installing / syncing dependencies...
uv sync

echo [vocals] Starting Streamline Vocals on http://localhost:7861
".venv\Scripts\python.exe" start.py %*
