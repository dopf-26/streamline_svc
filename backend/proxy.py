"""HTTP proxy layer between Streamline and the ACE-Step API server.

All generation, job-status, and LLM utility requests are forwarded to
ACE-Step's existing API server (default http://127.0.0.1:8001).
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any

import httpx
from loguru import logger

_DEFAULT_BASE = "http://127.0.0.1:8001"
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)

# Shared long-lived client for standard requests.  Short-timeout callers
# (health_check, health_status, list_models) create their own lightweight
# clients so they do not inherit the 300 s read timeout.
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the module-level shared AsyncClient, creating it if needed."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _http_client

# ACE-Step can return Unicode sharp/flat and title-cased mode names.
# Normalise to the standard form used by the Streamline dropdown: "C# minor", "Bb major".
_UNICODE_TO_ASCII = str.maketrans({"♯": "#", "♭": "b"})


def _normalize_keyscale(raw: str) -> str:
    """Normalise ACE-Step keyscale strings to ASCII sharp/flat + lowercase mode.

    Converts Unicode ♯→# and ♭→b, and lowercases the mode word so values like
    "F♯ Minor" become "F# minor" matching the Streamline dropdown options.

    Args:
        raw: Raw keyscale string from ACE-Step metas.

    Returns:
        Normalised keyscale string, or empty string if input is empty.
    """
    if not raw:
        return ""
    if raw.strip().upper() == "N/A":
        return ""
    normalised = raw.translate(_UNICODE_TO_ASCII).strip()
    # Lowercase only the mode word (last word), preserve note-letter case.
    parts = normalised.rsplit(" ", 1)
    if len(parts) == 2:
        normalised = f"{parts[0]} {parts[1].lower()}"
    return normalised


def _base_url() -> str:
    return os.environ.get("ACESTEP_API_URL", _DEFAULT_BASE).rstrip("/")


async def health_check() -> bool:
    """Return True if the ACE-Step API server is reachable.

    Returns:
        Boolean indicating whether the server answered /health.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{_base_url()}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def health_status() -> dict[str, Any]:
    """Return detailed ACE-Step health and readiness status.

    Returns:
        Dict containing connectivity and model-init readiness flags.
    """
    default = {
        "connected": False,
        "ready": False,
        "models_initialized": False,
        "llm_initialized": False,
        "loaded_model": "",
        "loaded_lm_model": "",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            resp = await client.get(f"{_base_url()}/health")
            if resp.status_code != 200:
                return default
            body = resp.json()
    except Exception:
        return default

    payload = body.get("data") if isinstance(body, dict) else None
    if not isinstance(payload, dict):
        payload = body if isinstance(body, dict) else {}

    models_initialized = bool(payload.get("models_initialized", False))
    llm_initialized = bool(payload.get("llm_initialized", False))
    loaded_model = str(payload.get("loaded_model") or "")
    loaded_lm_model = str(payload.get("loaded_lm_model") or "")

    return {
        "connected": True,
        "ready": models_initialized,  # SVC only needs DiT (LLM not required for audio-only inference)
        "models_initialized": models_initialized,
        "llm_initialized": llm_initialized,
        "loaded_model": loaded_model,
        "loaded_lm_model": loaded_lm_model,
    }


async def start_generation(payload: dict[str, Any]) -> str:
    """POST /release_task on the ACE-Step API and return the job ID.

    Args:
        payload: Dict matching ACE-Step GenerateMusicRequest schema.

    Returns:
        Job ID string returned by the server.

    Raises:
        RuntimeError: If the request fails or the server is unreachable.
    """
    try:
        client = _get_client()
        resp = await client.post(f"{_base_url()}/release_task", json=payload)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach ACE-Step API at {_base_url()}. "
            "Please start the API server first."
        ) from exc
    data = resp.json()
    # ACE-Step wraps all responses: { "data": {...}, "code": 200, ... }
    inner = data.get("data") or data
    job_id = inner.get("task_id") or inner.get("job_id") or inner.get("id")
    if not job_id:
        raise RuntimeError(f"Unexpected response from /release_task: {data}")
    logger.debug(f"Started generation job {job_id}")
    return str(job_id)


def _decode_audio_url(file_url: str) -> str:
    """Convert a /v1/audio?path=<encoded> URL back to a raw filesystem path.

    ACE-Step stores audio paths as URL-encoded strings. This reverses that
    so the Streamline backend can access the file directly on the same machine.

    Args:
        file_url: The file/URL string from the query_result response.

    Returns:
        Raw filesystem path string, or the original value if not decodable.
    """
    if not file_url:
        return ""
    if "/v1/audio" in file_url:
        parsed = urllib.parse.urlparse(file_url)
        qs = urllib.parse.parse_qs(parsed.query)
        paths = qs.get("path", [])
        return paths[0] if paths else file_url
    return file_url


async def query_jobs(job_ids: list[str]) -> list[dict[str, Any]]:
    """POST /query_result for a list of job IDs.

    Args:
        job_ids: List of job ID strings to query.

    Returns:
        List of normalized job-status dicts with keys:
        task_id, status ("queued"|"running"|"done"|"error"),
        audio_paths, raw_audio_paths, duration, progress_text.
    """
    client = _get_client()
    resp = await client.post(
            f"{_base_url()}/query_result",
            # ACE-Step uses task_id_list (accepts a list directly)
            json={"task_id_list": job_ids},
        )
    resp.raise_for_status()

    outer = resp.json()
    # Unwrap envelope: { "data": [...], "code": 200, ... }
    items = outer.get("data") if isinstance(outer.get("data"), list) else []

    results = []
    for item in items:
        status_int = item.get("status", 0)
        # 0 = queued/running, 1 = succeeded, 2 = failed
        if status_int == 1:
            status_str = "done"
        elif status_int == 2:
            status_str = "error"
        else:
            status_str = "running" if item.get("progress_text") else "queued"

        # "result" is a JSON-encoded string containing a list of track objects
        result_raw = item.get("result", "[]")
        try:
            result_list = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
            if not isinstance(result_list, list):
                result_list = []
        except Exception:
            result_list = []

        audio_paths: list[str] = []
        raw_audio_paths: list[str] = []
        duration = 0.0
        track_metas: list[dict[str, Any]] = []
        for r in result_list:
            file_url = r.get("file", "")
            if file_url:
                audio_paths.append(file_url)
                # "file" values are URL-encoded paths: /v1/audio?path=%2Ftmp%2F...
                # Decode back to real filesystem paths so the library copy can access them.
                raw_audio_paths.append(_decode_audio_url(file_url))
            metas = r.get("metas") or {}
            # Duration: take the first non-zero value
            if duration == 0.0:
                duration_raw = metas.get("duration")
                try:
                    track_duration = float(duration_raw) if str(duration_raw) not in ("N/A", "", "None") else 0.0
                except (ValueError, TypeError):
                    track_duration = 0.0
                if track_duration > 0:
                    duration = track_duration
            # Per-track metadata generated by the model
            bpm_raw = metas.get("bpm")
            try:
                bpm: int | None = int(bpm_raw) if bpm_raw is not None and str(bpm_raw) not in ("N/A", "", "None") else None
            except (ValueError, TypeError):
                bpm = None
            track_metas.append({
                "bpm": bpm,
                "keyscale": _normalize_keyscale(metas.get("keyscale") or ""),
                "timesignature": metas.get("timesignature") or "",
            })

        results.append({
            "task_id": item.get("task_id"),
            "status": status_str,
            "audio_paths": audio_paths,
            "raw_audio_paths": raw_audio_paths,
            "duration": duration,
            "track_metas": track_metas,
            "progress_text": item.get("progress_text", ""),
            "error": item.get("error"),
        })

    return results


async def create_random_sample(query: str) -> dict[str, Any]:
    """POST /create_random_sample to let the LLM invent caption + lyrics.

    Args:
        query: Optional style/genre hint string.

    Returns:
        Dict with ``caption`` and ``lyrics`` keys returned by ACE-Step.
    """
    client = _get_client()
    resp = await client.post(
        f"{_base_url()}/create_random_sample",
        json={"sample_query": query},
    )
    resp.raise_for_status()
    return resp.json()


async def format_input(caption: str, lyrics: str) -> dict[str, Any]:
    """POST /format_input to enhance caption and lyrics via the LLM.

    Args:
        caption: Music description to enrich.
        lyrics: Lyrics text to format.

    Returns:
        Dict with ``caption`` and ``lyrics`` keys.
    """
    client = _get_client()
    resp = await client.post(
        f"{_base_url()}/format_input",
        json={"prompt": caption, "lyrics": lyrics},
    )
    resp.raise_for_status()
    return resp.json()


async def load_lora_simple(lora_path: str, scale: float = 1.0) -> None:
    """Load a LoRA adapter using ACE-Step's native /v1/lora/load endpoint.

    Used when the slot-based monkeypatch is disabled.  Unloads any existing
    adapter first, then loads the new one and sets its scale.

    Args:
        lora_path: Absolute filesystem path to the LoRA directory.
        scale: Strength scale to apply after loading.

    Raises:
        httpx.HTTPStatusError: If the server returns an error response.
    """
    client = _get_client()
    await client.post(f"{_base_url()}/v1/lora/unload", json={})
    resp = await client.post(f"{_base_url()}/v1/lora/load", json={"lora_path": lora_path})
    resp.raise_for_status()
    scale_resp = await client.post(f"{_base_url()}/v1/lora/scale", json={"scale": min(scale, 1.0)})
    scale_resp.raise_for_status()
    logger.debug(f"LoRA loaded (simple): {lora_path} scale={scale}")


async def unload_lora_simple() -> None:
    """Unload LoRA adapter via ACE-Step's native /v1/lora/unload endpoint.

    Used when the slot-based monkeypatch is disabled.

    Raises:
        httpx.HTTPStatusError: If the server returns an error response.
    """
    client = _get_client()
    resp = await client.post(f"{_base_url()}/v1/lora/unload", json={})
    resp.raise_for_status()
    logger.debug("LoRA unloaded (simple)")


async def load_lora_slots_batch(entries: list[dict[str, Any]]) -> list[str]:
    """POST /v1/lora/load-slots-batch to load multiple adapters in one call.

    Args:
        entries: List of dicts with keys: lora_path, slot, scale, group_scales.

    Returns:
        List of per-slot status strings from the handler.
    """
    client = _get_client()
    resp = await client.post(f"{_base_url()}/v1/lora/load-slots-batch", json={"entries": entries})
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


async def unload_lora_all() -> None:
    """POST /v1/lora/unload-slot with no slot parameter to clear all adapters.

    Raises:
        RuntimeError: If the request fails.
    """
    client = _get_client()
    resp = await client.post(f"{_base_url()}/v1/lora/unload-slot", json={})
    resp.raise_for_status()
    logger.debug("Unloaded all LoRA adapter slots")


async def get_slots_status() -> dict[str, Any]:
    """GET /v1/lora/slots-status — current adapter slot state.

    Returns:
        Dict with keys: use_lora, lora_loaded, slots (list), slot_count,
        base_decoder_present.  Returns empty dict on any error.
    """
    client = _get_client()
    try:
        resp = await client.get(f"{_base_url()}/v1/lora/slots-status")
        resp.raise_for_status()
        raw = resp.json()
        return raw.get("data", raw) if isinstance(raw, dict) else {}
    except Exception as exc:
        logger.debug(f"slots-status fetch failed: {exc}")
        return {}


async def update_lora_slot_scale(slot: int, scale: float) -> None:
    """POST /v1/lora/slot-scale — update strength for one slot without reloading.

    Args:
        slot: Slot index (0–3).
        scale: New strength scale.
    """
    client = _get_client()
    resp = await client.post(f"{_base_url()}/v1/lora/slot-scale", json={"slot": slot, "scale": scale})
    resp.raise_for_status()
    logger.debug(f"Updated slot {slot} scale to {scale}")


async def update_lora_slot_group_scales(slot: int, group_scales: dict[str, float]) -> None:
    """POST /v1/lora/slot-group-scales — update group scales for one slot.

    Args:
        slot: Slot index (0–3).
        group_scales: Dict with self_attn, cross_attn, mlp keys.
    """
    client = _get_client()
    resp = await client.post(
        f"{_base_url()}/v1/lora/slot-group-scales",
        json={"slot": slot, **group_scales},
    )
    resp.raise_for_status()
    logger.debug(f"Updated slot {slot} group_scales to {group_scales}")


async def list_models() -> dict[str, Any]:
    """GET /v1/models from ACE-Step for available checkpoints.

    Returns:
        JSON response from /v1/models or empty dict on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{_base_url()}/v1/models")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(f"Could not fetch model list: {exc}")
        return {}


async def get_active_models() -> dict[str, str]:
    """Return currently loaded DiT/LM model names from ACE-Step inventory.

    Returns:
        Dict with keys ``dit_model`` and ``lm_model`` (empty string when unknown).
    """
    payload = await list_models()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return {"dit_model": "", "lm_model": ""}
    dit_model = str(data.get("default_model") or "").strip()
    lm_model = str(data.get("loaded_lm_model") or "").strip()
    return {"dit_model": dit_model, "lm_model": lm_model}


async def init_models(dit_model: str | None = None, lm_model: str | None = None) -> dict[str, Any]:
    """POST /v1/init to activate requested DiT/LM models on ACE-Step server.

    Args:
        dit_model: DiT model name (e.g. "acestep-v15-turbo").
        lm_model: LM model name/path (e.g. "acestep-5Hz-lm-0.6B").

    Returns:
        Parsed ``data`` object from ACE-Step wrapped response.

    Raises:
        RuntimeError: If initialization fails.
    """
    payload: dict[str, Any] = {}
    if dit_model:
        payload["model"] = dit_model
    if lm_model:
        payload["init_llm"] = True
        payload["lm_model_path"] = lm_model
    elif dit_model:
        payload["init_llm"] = False

    if not payload:
        return {}

    client = _get_client()
    try:
        resp = await client.post(f"{_base_url()}/v1/init", json=payload)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        logger.error(f"[Streamline] /v1/init connect error. payload={payload} error={exc}")
        raise RuntimeError(f"Cannot reach ACE-Step API at {_base_url()}") from exc
    except httpx.HTTPStatusError as exc:
        body_text = ""
        try:
            body_text = exc.response.text
        except Exception:
            body_text = "<unavailable>"
        logger.error(
            "[Streamline] /v1/init HTTP error. "
            f"status={exc.response.status_code} payload={payload} body={body_text}"
        )
        raise RuntimeError(
            f"/v1/init request failed ({exc.response.status_code}): {body_text}"
        ) from exc
    except httpx.HTTPError as exc:
        logger.error(f"[Streamline] /v1/init request error. payload={payload} error={exc}")
        raise RuntimeError(f"/v1/init request failed: {exc}") from exc

    body = resp.json()
    code = int(body.get("code", 200))
    if code != 200:
        detail = body.get("error") or body.get("message") or "unknown error"
        logger.error(f"[Streamline] /v1/init returned error. payload={payload} body={body}")
        raise RuntimeError(str(detail))

    return body.get("data") or {}
