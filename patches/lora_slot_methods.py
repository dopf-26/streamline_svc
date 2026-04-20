"""Slot-based LoRA handler methods for weight-space multi-adapter loading.

These methods are injected onto AceStepHandler by lora_slot_patch.py.
Ported from HOT-Step-9000's advanced_adapter_mixin.py.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch
from loguru import logger

from streamline_svc.patches.lora_slot_core import (
    MAX_ADAPTER_SLOTS,
    _dequantize_decoder_nf4,
    _log_vram,
    _derive_adapter_name,
    apply_merged_weights,
    apply_merged_weights_with_groups,
    extract_adapter_delta,
)

_ALL_ONE = {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0, "cond_embed": 1.0}


def _group_scales_are_all_one(group_scales: dict) -> bool:
    """Return True when all group scale values equal 1.0."""
    return all(abs(v - 1.0) < 1e-9 for v in group_scales.values())


def _all_slots_use_uniform_scales(handler: Any) -> bool:
    """Return True when every loaded slot has all group scales at 1.0."""
    return all(
        _group_scales_are_all_one(s.get("group_scales", _ALL_ONE))
        for s in handler._adapter_slots.values()
    )


def _merge(handler: Any) -> None:
    """Call the fast simple merge when group scales are all 1.0, else the group-scaled merge."""
    if _all_slots_use_uniform_scales(handler):
        apply_merged_weights(handler)
    else:
        apply_merged_weights_with_groups(handler)


# ---------------------------------------------------------------------------
# Handler methods (injected onto AceStepHandler)
# ---------------------------------------------------------------------------

def load_lora_slot(self, lora_path: str, slot: Optional[int] = None) -> str:
    """Load LoRA/LoKr adapter into a numbered slot using weight-space merging.

    Args:
        lora_path: Path to adapter directory or .safetensors file.
        slot: Optional slot ID (0–3). None = auto-assign next slot.

    Returns:
        Human-readable status message starting with ✅ or ❌.
    """
    if self.model is None:
        return "❌ Model not initialized. Please initialize service first."

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."
    lora_path = lora_path.strip()

    import os
    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    if len(self._adapter_slots) >= MAX_ADAPTER_SLOTS:
        return f"❌ Maximum {MAX_ADAPTER_SLOTS} adapter slots reached. Please unload one first."

    try:
        _needs_nf4_requant = False
        if getattr(self, "quantization", None) == "nf4":
            deq_count = _dequantize_decoder_nf4(self.model)
            _needs_nf4_requant = deq_count > 0

        # Backup base decoder on first slot load only — not per-slot.
        is_first_slot = self._base_decoder is None
        if is_first_slot:
            _log_vram("before base decoder backup")
            logger.info("Backing up base decoder state_dict to CPU")
            backup: dict[str, Any] = {}
            for k, v in self.model.decoder.state_dict().items():
                try:
                    backup[k] = v.detach().cpu().clone()
                except Exception:
                    backup[k] = v.float().to(torch.bfloat16).detach().cpu().clone()
            self._base_decoder = backup
            backup_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024 ** 2)
            logger.info(f"Base decoder backed up ({backup_mb:.1f}MB)")
            _log_vram("after base decoder backup")

        logger.info(f"Extracting adapter delta from {lora_path}")
        # When this is the first slot, the decoder is already at the clean base
        # state (we just backed it up from it), so skip the redundant restore.
        result = extract_adapter_delta(self, lora_path, decoder_is_at_base=is_first_slot)

        # Determine slot ID
        if slot is None:
            slot = self._next_slot_id
        self._next_slot_id = max(self._next_slot_id, slot + 1)

        adapter_name = _derive_adapter_name(lora_path, result.get("safetensors_file"))
        self._adapter_slots[slot] = {
            "path": lora_path,
            "name": adapter_name,
            "type": result["type"],
            "delta": result["delta"],
            "scale": 1.0,
            "group_scales": dict(_ALL_ONE),
            "layer_scales": {},
        }

        self.use_lora = True
        self.lora_loaded = True
        self._merged_dirty = True

        _log_vram("before merge")
        apply_merged_weights(self)

        if _needs_nf4_requant:
            _dequantize_decoder_nf4(self.model)

        delta_keys = len(result["delta"])
        type_label = "LoRA" if result["type"] == "peft_lora" else "LoKr"
        _log_vram("adapter load complete")
        logger.info(f"Adapter loaded into slot {slot}: {adapter_name} ({type_label}, {delta_keys} keys)")
        return f"✅ {type_label} loaded into slot {slot}: {adapter_name}"

    except Exception as e:
        logger.exception("Failed to load adapter into slot")
        try:
            if self._base_decoder is not None:
                logger.info("[load_lora_slot] Recovering decoder from base backup")
                torch._dynamo.reset()
                self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
                self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
                self.model.decoder.eval()
            elif hasattr(self.model, "decoder"):
                self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
                self.model.decoder.eval()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as recovery_err:
            logger.error(f"[load_lora_slot] Recovery also failed: {recovery_err}")
        return f"❌ Failed to load adapter: {str(e)}"


def load_lora_slots_batch(
    self,
    slot_entries: list[dict[str, Any]],
) -> list[str]:
    """Load multiple LoRA adapters in batch with a single merge pass at the end.

    Compared to calling load_lora_slot N times this avoids:
    - N-1 redundant ``apply_merged_weights`` calls (only one merge at the end)
    - N-1 redundant ``load_state_dict`` restores inside extract_adapter_delta
    - duplicate NF4 dequant/requant cycles

    Args:
        slot_entries: List of dicts with keys:
            - ``lora_path`` (str)
            - ``slot`` (int, optional)
            - ``scale`` (float, optional, default 1.0)
            - ``group_scales`` (dict, optional)

    Returns:
        List of status strings (one per entry) starting with ✅ or ❌.
    """
    if self.model is None:
        return ["❌ Model not initialized."] * len(slot_entries)

    results: list[str] = []
    needs_nf4_requant = False

    if getattr(self, "quantization", None) == "nf4":
        deq_count = _dequantize_decoder_nf4(self.model)
        needs_nf4_requant = deq_count > 0

    # Backup base decoder once — only if not already present.
    if self._base_decoder is None:
        _log_vram("before base decoder backup")
        logger.info("Backing up base decoder state_dict to CPU (batch)")
        backup: dict[str, Any] = {}
        for k, v in self.model.decoder.state_dict().items():
            try:
                backup[k] = v.detach().cpu().clone()
            except Exception:
                backup[k] = v.float().to(torch.bfloat16).detach().cpu().clone()
        self._base_decoder = backup
        backup_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024 ** 2)
        logger.info(f"Base decoder backed up ({backup_mb:.1f}MB)")
        _log_vram("after base decoder backup")
        decoder_currently_at_base = True
    else:
        decoder_currently_at_base = False

    for idx, entry in enumerate(slot_entries):
        lora_path = (entry.get("lora_path") or "").strip()
        slot = entry.get("slot")
        scale = float(entry.get("scale") or 1.0)
        group_scales = entry.get("group_scales") or {}

        import os as _os
        if not lora_path or not _os.path.exists(lora_path):
            results.append(f"❌ LoRA path not found: {lora_path}")
            decoder_currently_at_base = False
            continue

        if len(self._adapter_slots) >= MAX_ADAPTER_SLOTS:
            results.append(f"❌ Maximum {MAX_ADAPTER_SLOTS} adapter slots reached")
            break

        try:
            logger.info(f"[batch {idx+1}/{len(slot_entries)}] Extracting delta: {lora_path}")
            result = extract_adapter_delta(
                self, lora_path, decoder_is_at_base=decoder_currently_at_base
            )
            # After first extraction the decoder is no longer at base state
            # (extract_adapter_delta restores it, but only via the slow path;
            # on the fast path it never touches the decoder at all).
            # Either way, after returning, the backup is intact and the
            # decoder is at base — safe to set True for subsequent slots
            # when using the fast direct-delta path.
            decoder_currently_at_base = (result.get("type") == "peft_lora")

            if slot is None:
                slot = self._next_slot_id
            self._next_slot_id = max(self._next_slot_id, slot + 1)

            gs = dict(_ALL_ONE)
            if group_scales:
                for gk in ("self_attn", "cross_attn", "mlp", "cond_embed"):
                    if gk in group_scales:
                        gs[gk] = float(group_scales[gk])

            adapter_name = _derive_adapter_name(lora_path, result.get("safetensors_file"))
            self._adapter_slots[slot] = {
                "path": lora_path,
                "name": adapter_name,
                "type": result["type"],
                "delta": result["delta"],
                "scale": scale,
                "group_scales": gs,
                "layer_scales": {},
            }
            type_label = "LoRA" if result["type"] == "peft_lora" else "LoKr"
            results.append(f"✅ {type_label} queued for slot {slot}: {adapter_name}")
        except Exception as exc:
            logger.exception(f"[batch] Failed to load {lora_path}")
            results.append(f"❌ Failed: {exc}")
            decoder_currently_at_base = False

    if self._adapter_slots:
        self.use_lora = True
        self.lora_loaded = True
        self._merged_dirty = True
        _log_vram("before batch merge")
        _merge(self)
        if needs_nf4_requant:
            _dequantize_decoder_nf4(self.model)
        _log_vram("batch merge complete")
    else:
        # All loads failed — restore decoder to base
        if self._base_decoder is not None:
            torch._dynamo.reset()
            self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

    return results


def unload_lora_slot(self, slot: Optional[int] = None) -> str:
    """Unload adapter from a specific slot or all slots.

    Args:
        slot: Slot ID to unload. None = unload all adapters.

    Returns:
        Human-readable status message starting with ✅, ⚠️, or ❌.
    """
    if not self._adapter_slots:
        return "⚠️ No adapters loaded."
    if self._base_decoder is None:
        return "❌ Base decoder backup not found."

    try:
        if slot is not None:
            if slot not in self._adapter_slots:
                return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"
            name = self._adapter_slots[slot]["name"]
            del self._adapter_slots[slot]
            self._merged_dirty = True
            apply_merged_weights(self)
            if not self._adapter_slots:
                self.use_lora = False
                self.lora_loaded = False
            logger.info(f"Unloaded adapter from slot {slot}: {name}")
            return f"✅ Unloaded slot {slot}: {name}"
        else:
            count = len(self._adapter_slots)
            _log_vram("before unload all")
            self._adapter_slots.clear()
            self._next_slot_id = 0
            self.use_lora = False
            self.lora_loaded = False
            self._merged_dirty = True
            apply_merged_weights(self)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded all {count} adapter(s)")
            _log_vram("after unload all")
            return f"✅ Unloaded all {count} adapter(s), using base model"
    except Exception as e:
        logger.exception("Failed to unload adapter")
        return f"❌ Failed to unload: {str(e)}"


def set_lora_slot_scale(self, scale: float, slot: Optional[int] = None) -> str:
    """Set adapter scale for a specific slot or all slots.

    Args:
        scale: Scale value (0.0–4.0).
        slot: Specific slot. None = apply to all slots.

    Returns:
        Human-readable status message.
    """
    if not self._adapter_slots:
        return "⚠️ No adapters loaded"

    scale = max(0.0, min(4.0, scale))
    if slot is not None:
        if slot not in self._adapter_slots:
            return f"❌ Slot {slot} not found"
        self._adapter_slots[slot]["scale"] = scale
        self._merged_dirty = True
        _merge(self)
        name = self._adapter_slots[slot]["name"]
        return f"✅ Slot {slot} ({name}) scale: {scale:.2f}"
    else:
        for s in self._adapter_slots.values():
            s["scale"] = scale
        self._merged_dirty = True
        _merge(self)
        return f"✅ All adapter scales set to {scale:.2f}"


def set_slot_group_scales(
    self,
    slot: int,
    self_attn_scale: float = 1.0,
    cross_attn_scale: float = 1.0,
    mlp_scale: float = 1.0,
    cond_embed_scale: float = 1.0,
) -> str:
    """Set per-group LoRA scales for a specific adapter slot.

    Args:
        slot: Slot ID.
        self_attn_scale: Scale for self-attention layers (0.0–4.0).
        cross_attn_scale: Scale for cross-attention layers (0.0–4.0).
        mlp_scale: Scale for MLP/feed-forward layers (0.0–4.0).
        cond_embed_scale: Scale for conditioning embedder (0.0–4.0).

    Returns:
        Human-readable status message.
    """
    if slot not in self._adapter_slots:
        return f"❌ Slot {slot} not found. Active slots: {list(self._adapter_slots.keys())}"

    scales = {
        "self_attn": max(0.0, min(4.0, self_attn_scale)),
        "cross_attn": max(0.0, min(4.0, cross_attn_scale)),
        "mlp": max(0.0, min(4.0, mlp_scale)),
        "cond_embed": max(0.0, min(4.0, cond_embed_scale)),
    }
    self._adapter_slots[slot]["group_scales"] = scales

    if self.use_lora:
        self._merged_dirty = True
        _merge(self)

    name = self._adapter_slots[slot]["name"]
    sa, ca, ml = scales["self_attn"], scales["cross_attn"], scales["mlp"]
    return f"✅ Slot {slot} ({name}) group scales: SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%}"


def set_lora_group_scales(
    self,
    self_attn_scale: float,
    cross_attn_scale: float,
    mlp_scale: float,
    cond_embed_scale: float = 1.0,
) -> str:
    """Set per-group scales for ALL adapter slots simultaneously.

    Args:
        self_attn_scale: Self-attention group scale.
        cross_attn_scale: Cross-attention group scale.
        mlp_scale: MLP/feed-forward group scale.
        cond_embed_scale: Conditioning embedder scale.

    Returns:
        Human-readable status message.
    """
    scales = {
        "self_attn": max(0.0, min(4.0, self_attn_scale)),
        "cross_attn": max(0.0, min(4.0, cross_attn_scale)),
        "mlp": max(0.0, min(4.0, mlp_scale)),
        "cond_embed": max(0.0, min(4.0, cond_embed_scale)),
    }
    self.lora_group_scales = scales
    for s in self._adapter_slots.values():
        s["group_scales"] = dict(scales)

    if self._adapter_slots and self.use_lora:
        self._merged_dirty = True
        _merge(self)

    sa, ca, ml, ce = scales["self_attn"], scales["cross_attn"], scales["mlp"], scales["cond_embed"]
    return f"✅ Group scales (all slots): SA={sa:.0%} CA={ca:.0%} MLP={ml:.0%} CE={ce:.0%}"


def get_lora_slots_status(self) -> dict:
    """Return current slot state for diagnostic and API status responses.

    Returns:
        Dict with slot info, use_lora flag, and base_decoder presence.
    """
    slots = []
    for sid, s in self._adapter_slots.items():
        slots.append({
            "slot": sid,
            "name": s["name"],
            "path": s["path"],
            "type": s["type"],
            "scale": s["scale"],
            "group_scales": s.get("group_scales", dict(_ALL_ONE)),
        })
    return {
        "use_lora": self.use_lora,
        "lora_loaded": self.lora_loaded,
        "slots": slots,
        "slot_count": len(self._adapter_slots),
        "base_decoder_present": self._base_decoder is not None,
    }
