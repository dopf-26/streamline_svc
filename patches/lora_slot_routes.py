"""Monkey-patch for ACE-Step API: adds slot-based LoRA lifecycle routes.

Wraps ``register_lora_routes`` to append four new endpoints after the
existing basic LoRA endpoints are registered:

  POST /v1/lora/load-slot         load adapter into a numbered slot
  POST /v1/lora/unload-slot       unload one slot, or all slots
  POST /v1/lora/slot-scale        set scale for a slot
  POST /v1/lora/slot-group-scales set per-group scales for a slot
  GET  /v1/lora/slots-status      current slot state
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadSlotRequest(BaseModel):
    """Request payload for slot-based LoRA loading."""

    lora_path: str = Field(..., description="Path to adapter directory or .safetensors file")
    slot: Optional[int] = Field(default=None, ge=0, le=3, description="Slot ID (0–3); auto-assigned if omitted")
    scale: Optional[float] = Field(default=None, ge=0.0, le=4.0, description="Initial scale applied after load")
    group_scales: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-group scales {self_attn, cross_attn, mlp} applied after load",
    )


class UnloadSlotRequest(BaseModel):
    """Request payload to unload one or all LoRA slots."""

    slot: Optional[int] = Field(default=None, ge=0, le=3, description="Slot ID to unload; omit to unload all")


class SlotScaleRequest(BaseModel):
    """Request payload to set scale on a slot."""

    slot: Optional[int] = Field(default=None, ge=0, le=3, description="Slot ID; omit to apply to all")
    scale: float = Field(..., ge=0.0, le=4.0, description="Scale value (0.0–4.0)")


class SlotGroupScalesRequest(BaseModel):
    """Request payload to set per-group scales on a slot."""

    slot: int = Field(..., ge=0, le=3, description="Slot ID")
    self_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Self-attention scale")
    cross_attn: float = Field(default=1.0, ge=0.0, le=4.0, description="Cross-attention scale")
    mlp: float = Field(default=1.0, ge=0.0, le=4.0, description="MLP/feed-forward scale")


class LoadSlotsBatchRequest(BaseModel):
    """Request payload for batch slot loading."""

    entries: list[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Route registration patch
# ---------------------------------------------------------------------------

_original_register_lora_routes: Callable | None = None


def _require_initialized_handler(app: FastAPI) -> Any:
    """Return initialized handler or raise HTTP 500 when unavailable."""
    from acestep.handler import AceStepHandler  # type: ignore[attr-defined]
    handler: AceStepHandler = app.state.handler
    if handler is None or handler.model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    return handler


def _is_success(result: str) -> bool:
    return result.startswith("✅")


def _patched_register_lora_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    **kwargs: Any,
) -> None:
    """Replacement that calls the original then adds slot-based endpoints."""
    # Call the original registration first
    if _original_register_lora_routes is not None:
        _original_register_lora_routes(app=app, verify_api_key=verify_api_key, wrap_response=wrap_response, **kwargs)

    # ── Slot-based endpoints ────────────────────────────────────────────

    @app.post("/v1/lora/load-slot")
    async def load_slot_endpoint(request: LoadSlotRequest, _: None = Depends(verify_api_key)):
        """Load adapter into a numbered slot using weight-space merging."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "load_lora_slot"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        result: str = handler.load_lora_slot(request.lora_path, slot=request.slot)
        if not _is_success(result):
            raise HTTPException(status_code=400, detail=result)
        slot_id = request.slot
        if slot_id is None:
            slot_id = max(handler._adapter_slots.keys()) if handler._adapter_slots else 0
        if request.scale is not None:
            handler.set_lora_slot_scale(request.scale, slot_id)
        if request.group_scales is not None:
            handler.set_slot_group_scales(
                slot=slot_id,
                self_attn_scale=request.group_scales.get("self_attn", 1.0),
                cross_attn_scale=request.group_scales.get("cross_attn", 1.0),
                mlp_scale=request.group_scales.get("mlp", 1.0),
            )
        return wrap_response({"message": result, "slot": slot_id, "lora_path": request.lora_path})

    @app.post("/v1/lora/unload-slot")
    async def unload_slot_endpoint(request: UnloadSlotRequest, _: None = Depends(verify_api_key)):
        """Unload adapter from one slot, or all slots."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "unload_lora_slot"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        result: str = handler.unload_lora_slot(slot=request.slot)
        if result.startswith("❌"):
            raise HTTPException(status_code=400, detail=result)
        return wrap_response({"message": result})

    @app.post("/v1/lora/slot-scale")
    async def slot_scale_endpoint(request: SlotScaleRequest, _: None = Depends(verify_api_key)):
        """Set adapter scale for a slot or all slots."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "set_lora_slot_scale"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        result: str = handler.set_lora_slot_scale(request.scale, slot=request.slot)
        if result.startswith("❌"):
            raise HTTPException(status_code=400, detail=result)
        return wrap_response({"message": result})

    @app.post("/v1/lora/slot-group-scales")
    async def slot_group_scales_endpoint(request: SlotGroupScalesRequest, _: None = Depends(verify_api_key)):
        """Set per-group scales for a specific adapter slot."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "set_slot_group_scales"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        result: str = handler.set_slot_group_scales(
            slot=request.slot,
            self_attn_scale=request.self_attn,
            cross_attn_scale=request.cross_attn,
            mlp_scale=request.mlp,
        )
        if result.startswith("❌"):
            raise HTTPException(status_code=400, detail=result)
        return wrap_response({"message": result})

    @app.get("/v1/lora/slots-status")
    async def slots_status_endpoint(_: None = Depends(verify_api_key)):
        """Return current slot state including names, scales, and group scales."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "get_lora_slots_status"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        return wrap_response(handler.get_lora_slots_status())

    @app.post("/v1/lora/load-slots-batch")
    async def load_slots_batch_endpoint(request: LoadSlotsBatchRequest, _: None = Depends(verify_api_key)):
        """Load multiple adapters in one call with a single weight-merge pass."""
        handler = _require_initialized_handler(app)
        if not hasattr(handler, "load_lora_slots_batch"):
            raise HTTPException(status_code=500, detail="Slot-based LoRA patch not applied")
        results: list[str] = handler.load_lora_slots_batch(request.entries)
        any_failed = any(r.startswith("❌") for r in results)
        return wrap_response({"results": results, "any_failed": any_failed})

    logger.info("Registered slot-based LoRA routes (/v1/lora/load-slot, load-slots-batch, unload-slot, slot-scale, slot-group-scales)")


def apply_lora_routes_patch() -> None:
    """Wrap ``register_lora_routes`` to inject slot-based endpoints."""
    global _original_register_lora_routes
    import acestep.api.http.lora_routes as _lora_mod
    import acestep.api.route_setup as _route_mod

    _original_register_lora_routes = _lora_mod.register_lora_routes
    _lora_mod.register_lora_routes = _patched_register_lora_routes
    _route_mod.register_lora_routes = _patched_register_lora_routes
    logger.info("Applied lora_slot_routes patch (slot-based endpoints)")
