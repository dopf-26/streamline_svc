"""Streamline SVC LoRA slot patch: weight-space multi-adapter loading.

Applies three changes to a running ACE-Step API server process:

1. **Handler init** — Adds ``_adapter_slots``, ``_next_slot_id``, and
   ``lora_group_scales`` attributes to AceStepHandler.__init__.

2. **Handler methods** — Injects ``load_lora_slot``, ``unload_lora_slot``,
   ``set_lora_slot_scale``, ``set_slot_group_scales``, ``set_lora_group_scales``,
   and ``get_lora_slots_status`` onto AceStepHandler.

3. **API routes** — Wraps ``register_lora_routes`` to append slot-based
   endpoints without modifying existing basic LoRA routes.

Replaces the old PEFT-based multi-adapter patches (lora_multi_adapter_fix,
lora_registry_fix, lora_unload_fix) with a weight-space merging approach
ported from HOT-Step-9000.  The approach:
  - Backs up base decoder weights on first slot load.
  - For each adapter: load → merge → compute delta = merged − base → store CPU.
  - At inference: decoder = base + Σ(scale × group_scale × delta).
"""

from __future__ import annotations

from loguru import logger


def apply_lora_slot_patch() -> None:
    """Inject the slot-based LoRA implementation into AceStepHandler and API routes.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    from acestep.handler import AceStepHandler  # type: ignore[attr-defined]

    # Guard: skip if already patched
    if getattr(AceStepHandler, "_streamline_slot_patch_applied", False):
        logger.debug("lora_slot_patch already applied — skipping")
        return

    # ── 1. Handler __init__ extension ───────────────────────────────────
    _original_init = AceStepHandler.__init__

    def _patched_init(self) -> None:
        _original_init(self)
        # Slot-based adapter state (not present in base AceStepHandler)
        self._adapter_slots: dict = {}          # slot_id → {name, path, type, delta, scale, group_scales, layer_scales}
        self._base_decoder: dict = None         # backup of base decoder state_dict (CPU)
        self._next_slot_id: int = 0
        self.lora_group_scales: dict = {
            "self_attn": 1.0,
            "cross_attn": 1.0,
            "mlp": 1.0,
            "cond_embed": 1.0,
        }
        self._merged_dirty: bool = False

    AceStepHandler.__init__ = _patched_init

    # ── 2. Inject handler methods ────────────────────────────────────────
    from streamline_svc.patches.lora_slot_methods import (
        get_lora_slots_status,
        load_lora_slot,
        load_lora_slots_batch,
        set_lora_group_scales,
        set_lora_slot_scale,
        set_slot_group_scales,
        unload_lora_slot,
    )

    AceStepHandler.load_lora_slot = load_lora_slot
    AceStepHandler.load_lora_slots_batch = load_lora_slots_batch
    AceStepHandler.unload_lora_slot = unload_lora_slot
    AceStepHandler.set_lora_slot_scale = set_lora_slot_scale
    AceStepHandler.set_slot_group_scales = set_slot_group_scales
    AceStepHandler.set_lora_group_scales = set_lora_group_scales
    AceStepHandler.get_lora_slots_status = get_lora_slots_status

    # ── 3. Patch API route registration ─────────────────────────────────
    from streamline_svc.patches.lora_slot_routes import apply_lora_routes_patch
    apply_lora_routes_patch()

    AceStepHandler._streamline_slot_patch_applied = True
    logger.info("Applied lora_slot_patch: slot-based weight-space LoRA loading active")
