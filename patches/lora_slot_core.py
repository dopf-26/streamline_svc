"""Core weight-space merging utilities for slot-based LoRA loading.

Ported from HOT-Step-9000's advanced_adapter_mixin.py.  Pure helper
functions — no handler state is accessed here.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import tempfile
import time
from typing import Any, Optional

import torch
from loguru import logger

MAX_ADAPTER_SLOTS = 4


# ---------------------------------------------------------------------------
# VRAM diagnostics
# ---------------------------------------------------------------------------

def _log_vram(label: str) -> None:
    """Log current CUDA VRAM usage with a contextual label."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(f"[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")


# ---------------------------------------------------------------------------
# NF4 quantization helpers
# ---------------------------------------------------------------------------

def _dequantize_decoder_nf4(model: Any) -> int:
    """Dequantize all NF4Tensor weights in the decoder back to bfloat16."""
    import torch.nn as nn
    _log_vram("before dequantize")
    logger.info("[NF4 compat] Starting decoder dequantization...")
    count = 0
    errors = 0
    for _name, module in model.decoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        w_type = type(w.data).__name__
        if w_type in ("Tensor", "Parameter"):
            continue
        try:
            deq = w.data.float().to(torch.bfloat16).detach()
            module.weight = nn.Parameter(deq, requires_grad=False)
            count += 1
        except Exception as e:
            logger.error(f"[NF4 compat] Cannot dequantize {_name} (type={w_type}): {e}")
            errors += 1
    logger.info(
        f"[NF4 compat] Dequantized {count} linear layers"
        + (f" ({errors} errors)" if errors else "")
    )
    _log_vram("after dequantize")
    return count


def _requantize_decoder_nf4(model: Any, skip_parts: tuple = ("tokenizer", "detokenizer")) -> int:
    """Re-apply NF4 quantization to the decoder's linear layers after merge."""
    import torch.nn as nn
    try:
        from torchao.dtypes import to_nf4
        from torchao.dtypes.nf4tensor import NF4Tensor
    except ImportError:
        logger.warning("[NF4 compat] torchao not available — cannot re-quantize")
        return 0

    count = 0
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Linear):
            skip = any(part in name.split(".") for part in skip_parts)
            is_already_nf4 = isinstance(module.weight.data, NF4Tensor)
            if not skip and not is_already_nf4:
                module.weight = nn.Parameter(to_nf4(module.weight.data), requires_grad=False)
                count += 1
    if count:
        logger.info(f"[NF4 compat] Re-quantized {count} linear layers after merge")
        _log_vram("after NF4 re-quantize")
    return count


# ---------------------------------------------------------------------------
# Key classification helpers
# ---------------------------------------------------------------------------

def _determine_group(module_name: str) -> str:
    """Return which module group a named module key belongs to."""
    if "cross_attn" in module_name:
        return "cross_attn"
    elif ".attn." in module_name or ".attn_" in module_name:
        return "self_attn"
    elif ".ff." in module_name or ".ff_" in module_name:
        return "mlp"
    elif "condition_embed" in module_name:
        return "cond_embed"
    return ""


def _extract_layer_index(key: str) -> Optional[int]:
    """Extract transformer layer index from a weight key (e.g. 'layers.7.attn' → 7)."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _derive_adapter_name(lora_path: str, safetensors_file: Optional[str]) -> str:
    """Derive a human-readable adapter name from the path."""
    if safetensors_file:
        name = os.path.splitext(os.path.basename(safetensors_file))[0]
    else:
        name = (
            os.path.basename(os.path.dirname(lora_path))
            if not os.path.isdir(lora_path)
            else os.path.basename(lora_path)
        )
    if name in ("adapter", "best", "final", "lokr_weights"):
        parent = (
            os.path.dirname(lora_path)
            if os.path.isdir(lora_path)
            else os.path.dirname(os.path.dirname(lora_path))
        )
        name = os.path.basename(parent)
    return name


# ---------------------------------------------------------------------------
# Bare PEFT detection (ComfyUI / Kohya single-file adapters)
# ---------------------------------------------------------------------------

def _try_prepare_bare_peft_safetensors(safetensors_path: str) -> Optional[str]:
    """Detect a bare PEFT LoRA safetensors and create a temp loading directory.

    Returns the temp dir path, or None if the file is not PEFT-compatible.
    """
    if not os.path.isfile(safetensors_path) or not safetensors_path.lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file as st_save_file
    except ImportError:
        return None

    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            lora_down_keys = [k for k in keys if "lora_down.weight" in k]
            if not lora_down_keys:
                return None
            first_down = sf.get_tensor(lora_down_keys[0])
            rank = first_down.shape[0]
            alpha_keys = [k for k in keys if k.endswith(".alpha")]
            lora_alpha = float(rank)
            if alpha_keys:
                lora_alpha = float(sf.get_tensor(alpha_keys[0]).item())
            use_dora = any("dora_scale" in k for k in keys)
            target_modules: set[str] = set()
            for k in lora_down_keys:
                parts = k.split(".")
                try:
                    idx = parts.index("lora_down")
                    target_modules.add(parts[idx - 1])
                except (ValueError, IndexError):
                    pass
            if not target_modules:
                return None
            converted: dict[str, Any] = {}
            with safe_open(safetensors_path, framework="pt", device="cpu") as sf2:
                for key in sf2.keys():
                    tensor = sf2.get_tensor(key)
                    if key.endswith(".alpha"):
                        continue
                    new_key = (
                        key
                        .replace(".lora_down.weight", ".lora_A.weight")
                        .replace(".lora_up.weight", ".lora_B.weight")
                        .replace(".dora_scale", ".lora_magnitude_vector")
                    )
                    converted[new_key] = tensor
    except Exception as exc:
        logger.debug(f"Bare PEFT detection failed: {exc}")
        return None

    try:
        tmp_dir = tempfile.mkdtemp(prefix="peft_adapter_")
        config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": "",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_to_transform": None,
            "layers_pattern": None,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0,
            "modules_to_save": None,
            "r": rank,
            "revision": None,
            "target_modules": sorted(target_modules),
            "task_type": None,
            "use_dora": use_dora,
        }
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        st_save_file(converted, os.path.join(tmp_dir, "adapter_model.safetensors"))
        logger.info(f"Inferred PEFT config from bare safetensors → {tmp_dir}")
        return tmp_dir
    except Exception as exc:
        logger.warning(f"Failed to create temp PEFT adapter dir: {exc}")
        return None


# ---------------------------------------------------------------------------
# Delta extraction
# ---------------------------------------------------------------------------

def _extract_delta_direct(lora_path: str) -> dict[str, Any] | None:
    """Compute delta tensors directly from a LoRA safetensors file.

    Reads lora_A / lora_B (and optional alpha) directly — no model loading,
    no PeftModel, no merge_and_unload.  Computes:

        delta[base_key] = lora_B @ lora_A  * (alpha / rank)

    Works for standard LoRA and most Kohya/Civitai exports.  Returns None
    when the file uses DoRA magnitude vectors or an unsupported format, so
    the caller can fall back.

    Args:
        lora_path: Path to adapter directory (containing adapter_config.json
            + adapter_model.safetensors) or a bare .safetensors file.

    Returns:
        Dict mapping base-model weight keys to float32 CPU delta tensors,
        or None when direct extraction is not possible.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    # Locate the safetensors file and read the config if present.
    if os.path.isdir(lora_path):
        st_candidates = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
        if not st_candidates:
            return None
        st_file = st_candidates[0]
        cfg_path = os.path.join(lora_path, "adapter_config.json")
    else:
        st_file = lora_path
        cfg_path = os.path.join(os.path.dirname(lora_path), "adapter_config.json")

    # Read config for alpha / rank defaults when available.
    global_alpha: float | None = None
    global_rank: int | None = None
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                _cfg = json.load(f)
            global_alpha = float(_cfg.get("lora_alpha") or 0) or None
            global_rank = int(_cfg.get("r") or 0) or None
        except Exception:
            pass

    try:
        with safe_open(st_file, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
    except Exception as exc:
        logger.debug(f"[direct_delta] Cannot open {st_file}: {exc}")
        return None

    # Reject DoRA — magnitude vector rescaling requires PEFT or manual impl.
    if any("lora_magnitude_vector" in k or "dora_scale" in k for k in keys):
        logger.info("[direct_delta] DoRA adapter detected — falling back to PEFT merge")
        return None

    lora_a_keys = [k for k in keys if "lora_A.weight" in k]
    if not lora_a_keys:
        return None

    delta: dict[str, Any] = {}
    try:
        with safe_open(st_file, framework="pt", device="cpu") as sf:
            # Read all tensors once.
            tensor_cache: dict[str, torch.Tensor] = {}
            for k in keys:
                tensor_cache[k] = sf.get_tensor(k)
    except Exception as exc:
        logger.debug(f"[direct_delta] Failed to read tensors: {exc}")
        return None

    for a_key in lora_a_keys:
        # Reconstruct the corresponding lora_B and base-model keys.
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        alpha_key = a_key.replace("lora_A.weight", "alpha")

        lora_A = tensor_cache.get(a_key)
        lora_B = tensor_cache.get(b_key)
        if lora_A is None or lora_B is None:
            continue

        rank = lora_A.shape[0]
        if alpha_key in tensor_cache:
            alpha = float(tensor_cache[alpha_key].item())
        elif global_alpha is not None:
            alpha = global_alpha
        else:
            alpha = float(rank)  # default: scale = 1.0

        scale = alpha / rank

        # Convert to float32 for safe matrix multiply.
        lora_A = lora_A.float()
        lora_B = lora_B.float()

        # Compute the weight delta: lora_B @ lora_A (standard LoRA formula)
        if lora_B.dim() == 2 and lora_A.dim() == 2:
            diff = (lora_B @ lora_A) * scale
        elif lora_B.dim() == 4 and lora_A.dim() == 4:
            # Conv2d: (out, in/g, kH, kW) shaped
            diff = (lora_B.flatten(1) @ lora_A.flatten(1)).view(lora_B.shape[0], lora_A.shape[1], *lora_B.shape[2:]) * scale
        else:
            logger.debug(f"[direct_delta] Unsupported tensor dims for {a_key}: A={lora_A.shape} B={lora_B.shape}")
            return None  # Unsupported shape — let PEFT handle it

        # Map back to the base model key.
        # PEFT key format: "base_model.model.<...>.lora_A.weight"
        # ACE-Step decoder key format: "layers.<n>.<module>.weight"
        # Strip the PEFT prefix and the ".lora_A.weight" suffix.
        base_key = a_key.replace("lora_A.weight", "weight")
        for prefix in ("base_model.model.", "model."):
            if base_key.startswith(prefix):
                base_key = base_key[len(prefix):]
                break

        if diff.abs().max().item() > 1e-10:
            delta[base_key] = diff

    if not delta:
        return None

    logger.info(f"[direct_delta] Extracted {len(delta)} delta keys from {os.path.basename(st_file)} without PEFT")
    return delta


def extract_adapter_delta(handler: Any, lora_path: str, decoder_is_at_base: bool = False) -> dict[str, Any]:
    """Load adapter, compute delta = adapted − base, restore base decoder.

    Args:
        handler: AceStepHandler instance.
        lora_path: Path to LoRA adapter directory or .safetensors file.
        decoder_is_at_base: When True, the decoder is already at the base
            state so the initial restore can be skipped (saves ~5s).

    Returns:
        Dict with keys: delta, type, safetensors_file.
    """
    if handler._base_decoder is None:
        raise RuntimeError("Base decoder not backed up yet")

    is_peft = False
    lokr_weights_path = None
    _bare_peft_tmp_dir = None
    _direct_st_file: str | None = None  # safetensors file to try direct extraction on

    if os.path.isdir(lora_path):
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            is_peft = True
        st_files = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
        if not is_peft and st_files:
            for sf in st_files:
                tmp = _try_prepare_bare_peft_safetensors(sf)
                if tmp:
                    _bare_peft_tmp_dir = tmp
                    lora_path = tmp
                    is_peft = True
                    break
            if not is_peft:
                lokr_first = [f for f in st_files if "lokr" in os.path.basename(f).lower()]
                lokr_weights_path = lokr_first[0] if lokr_first else st_files[0]
        if is_peft:
            st_files_peft = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
            if st_files_peft:
                _direct_st_file = st_files_peft[0]
    elif lora_path.endswith(".safetensors"):
        parent = os.path.dirname(lora_path)
        if os.path.exists(os.path.join(parent, "adapter_config.json")):
            lora_path = parent
            is_peft = True
            _direct_st_file = lora_path if lora_path.endswith(".safetensors") else None
            # Refresh direct file for PEFT dir case
            st_files2 = sorted(glob.glob(os.path.join(lora_path, "*.safetensors")))
            _direct_st_file = st_files2[0] if st_files2 else None
        else:
            tmp = _try_prepare_bare_peft_safetensors(lora_path)
            if tmp:
                _bare_peft_tmp_dir = tmp
                lokr_weights_path = lora_path
                _direct_st_file = lora_path  # try direct first on the original file
                lora_path = tmp
                is_peft = True
            else:
                lokr_weights_path = lora_path

    # ------------------------------------------------------------------
    # Fast path: compute delta directly from the safetensors file.
    # Bypasses PeftModel entirely — ~100x faster for standard LoRA.
    # ------------------------------------------------------------------
    if is_peft and _direct_st_file:
        direct_delta = _extract_delta_direct(_direct_st_file)
        if direct_delta is not None:
            if _bare_peft_tmp_dir and os.path.isdir(_bare_peft_tmp_dir):
                try:
                    shutil.rmtree(_bare_peft_tmp_dir, ignore_errors=True)
                except Exception:
                    pass
            sf_file = _direct_st_file if not os.path.isdir(_direct_st_file) else None
            return {"delta": direct_delta, "type": "peft_lora", "safetensors_file": sf_file}

    # ------------------------------------------------------------------
    # Slow path: PEFT merge_and_unload or LyCORIS.
    # ------------------------------------------------------------------

    # Reset decoder to base — skip when caller guarantees it's already there.
    if not decoder_is_at_base:
        torch._dynamo.reset()
        handler.model.decoder.load_state_dict(handler._base_decoder, strict=False, assign=True)
        handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
        handler.model.decoder.eval()

    adapter_type = None

    if is_peft:
        from peft import PeftModel

        # Load adapter directly on the current device — moving to CPU is ~10x slower
        # for large decoders and causes apparent stalls with DoRA adapters.
        logger.info("[extract_delta] Loading PEFT adapter on GPU...")
        handler.model.decoder = PeftModel.from_pretrained(
            handler.model.decoder, lora_path, is_trainable=False,
        )
        handler.model.decoder = handler.model.decoder.to(handler.dtype)
        handler.model.decoder.eval()
        logger.info("[extract_delta] Merging and unloading PEFT adapter...")
        handler.model.decoder = handler.model.decoder.merge_and_unload()
        adapter_type = "peft_lora"

    elif lokr_weights_path is not None:
        from acestep.core.generation.handler.lora.lifecycle import _load_lokr_adapter
        try:
            from lycoris import LycorisNetwork
        except ImportError:
            raise ImportError("LyCORIS library not installed")

        original_device = handler.device
        logger.info("[extract_delta] Loading LoKr adapter on GPU...")
        # Keep on GPU — only move to CPU if absolutely necessary for the lycoris API

        lycoris_net = _load_lokr_adapter(handler.model.decoder, lokr_weights_path)

        param_to_key = {
            param.data_ptr(): key
            for key, param in handler.model.decoder.named_parameters()
        }
        adapted_sd: dict[str, Any] = {}
        lokr_delta_keys: set[str] = set()
        for lora_mod in lycoris_net.loras:
            org_module = lora_mod.org_module[0]
            sd_key = param_to_key.get(org_module.weight.data_ptr())
            if sd_key is None:
                continue
            diff = lora_mod.get_weight(org_module.weight.shape)
            scalar_val = lora_mod.scalar
            if scalar_val is not None:
                diff = diff.float() * scalar_val.float()
            else:
                diff = diff.float()
            if diff.abs().max().item() > 1e-8:
                adapted_sd[sd_key] = diff.detach().cpu()
                lokr_delta_keys.add(sd_key)

        logger.info(f"LoKr direct delta: {len(lokr_delta_keys)} keys from {len(lycoris_net.loras)} modules")

        try:
            lycoris_net.restore()
        except Exception:
            pass

        cleaned = 0
        for _name, mod in handler.model.decoder.named_modules():
            wrappers = getattr(mod, "_lycoris_wrappers", None)
            if wrappers:
                orig_fwd = getattr(mod, "_lycoris_original_forward", None)
                if orig_fwd is not None:
                    mod.forward = orig_fwd
                mod.__dict__.pop("_lycoris_wrappers", None)
                mod.__dict__.pop("_lycoris_original_forward", None)
                cleaned += 1
        if cleaned:
            logger.info(f"Cleaned {cleaned} LyCORIS forward wrappers")

        try:
            if hasattr(handler.model.decoder, "_lycoris_net"):
                delattr(handler.model.decoder, "_lycoris_net")
        except Exception:
            pass

        # Move decoder back to target device (stays on GPU but ensure dtype is correct)
        handler.model.decoder = handler.model.decoder.to(original_device).to(handler.dtype)
        handler.model.decoder.eval()

        del lycoris_net
        adapter_type = "lycoris_lokr"
    else:
        raise ValueError(
            "Invalid adapter path. Expected PEFT dir (adapter_config.json) "
            "or LoKr weights (.safetensors)."
        )

    # Compute delta
    delta: dict[str, Any] = {}
    logger.info("[extract_delta] Computing weight delta against base...")
    if adapter_type == "lycoris_lokr":
        # adapted_sd already holds the pre-computed per-key deltas (from LyCORIS)
        delta = {k: v.float() for k, v in adapted_sd.items()}
    elif adapter_type == "peft_lora":
        # Compute delta key-by-key on GPU against the CPU backup.
        # Avoids materialising a full 8 GB CPU copy of the merged decoder.
        _peft_params = dict(handler.model.decoder.named_parameters())
        _peft_bufs = dict(handler.model.decoder.named_buffers())
        _peft_all = {**_peft_params, **_peft_bufs}
        for k, base_v in handler._base_decoder.items():
            if k in _peft_all:
                adp = _peft_all[k].float()
                base = base_v.float().to(adp.device)
                diff = adp - base
                if diff.abs().max().item() > 1e-8:
                    delta[k] = diff.detach().cpu()
                del adp, base, diff
        del _peft_all, _peft_params, _peft_bufs

    # Restore decoder to base state
    handler.model.decoder.load_state_dict(handler._base_decoder, strict=False, assign=True)
    handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
    handler.model.decoder.eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta_mb = sum(v.numel() * 4 for v in delta.values()) / (1024 ** 2)
    logger.info(f"Extracted adapter delta: {len(delta)} keys, {delta_mb:.1f}MB (fp32 on CPU)")

    if _bare_peft_tmp_dir and os.path.isdir(_bare_peft_tmp_dir):
        try:
            shutil.rmtree(_bare_peft_tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return {"delta": delta, "type": adapter_type, "safetensors_file": lokr_weights_path}


# ---------------------------------------------------------------------------
# Weight merging
# ---------------------------------------------------------------------------

def apply_merged_weights(handler: Any) -> None:
    """Load base + Σ(scale_i × delta_i) onto GPU decoder (no group scaling)."""
    if handler._base_decoder is None:
        return

    active_slots = {
        sid: s for sid, s in handler._adapter_slots.items()
        if s["scale"] > 0 and handler.use_lora
    }

    if not active_slots:
        torch._dynamo.reset()
        handler.model.decoder.load_state_dict(handler._base_decoder, strict=False, assign=True)
        handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
        handler.model.decoder.eval()
        if getattr(handler, "quantization", None) == "nf4":
            _requantize_decoder_nf4(handler.model)
        handler._merged_dirty = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    t0 = time.time()
    all_keys: set[str] = set()
    for s in active_slots.values():
        all_keys.update(s["delta"].keys())

    merged: dict[str, Any] = {}
    for k in handler._base_decoder:
        base_val = handler._base_decoder[k]
        if k in all_keys:
            combined = base_val.float()
            for s in active_slots.values():
                if k in s["delta"]:
                    combined = combined + s["scale"] * s["delta"][k]
            merged[k] = combined.to(dtype=base_val.dtype)
        else:
            merged[k] = base_val

    torch._dynamo.reset()
    handler.model.decoder.load_state_dict(merged, strict=False, assign=True)
    handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
    handler.model.decoder.eval()

    if getattr(handler, "quantization", None) == "nf4":
        _requantize_decoder_nf4(handler.model)

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_vram("after merge + cleanup")

    elapsed = time.time() - t0
    slot_desc = ", ".join(f"slot{sid}={s['name']}@{s['scale']:.2f}" for sid, s in active_slots.items())
    logger.info(f"Merged {len(active_slots)} adapter(s) in {elapsed:.1f}s: {slot_desc}")
    handler._merged_dirty = False


def apply_merged_weights_with_groups(handler: Any) -> None:
    """Load base + Σ(scale_i × group_scale × delta_i) onto GPU decoder."""
    if handler._base_decoder is None:
        return

    active_slots = {
        sid: s for sid, s in handler._adapter_slots.items()
        if s["scale"] > 0 and handler.use_lora
    }

    if not active_slots:
        torch._dynamo.reset()
        handler.model.decoder.load_state_dict(handler._base_decoder, strict=False, assign=True)
        handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
        handler.model.decoder.eval()
        if getattr(handler, "quantization", None) == "nf4":
            _requantize_decoder_nf4(handler.model)
        handler._merged_dirty = False
        return

    t0 = time.time()
    all_keys: set[str] = set()
    for s in active_slots.values():
        all_keys.update(s["delta"].keys())

    merged: dict[str, Any] = {}
    for k in handler._base_decoder:
        base_val = handler._base_decoder[k]
        if k in all_keys:
            group = _determine_group(k)
            layer_idx = _extract_layer_index(k)
            combined = base_val.float()
            for s in active_slots.values():
                if k not in s["delta"]:
                    continue
                gs = s.get("group_scales", {})
                if group:
                    g_scale = gs.get(group, 1.0)
                else:
                    vals = [gs.get("self_attn", 1.0), gs.get("cross_attn", 1.0), gs.get("mlp", 1.0), gs.get("cond_embed", 1.0)]
                    g_scale = sum(vals) / len(vals)
                l_scales = s.get("layer_scales", {})
                if layer_idx is not None:
                    l_scale = l_scales.get(layer_idx, 1.0)
                elif l_scales:
                    l_scale = sum(l_scales.values()) / max(len(l_scales), 1)
                else:
                    l_scale = 1.0
                combined = combined + s["scale"] * g_scale * l_scale * s["delta"][k]
            merged[k] = combined.to(dtype=base_val.dtype)
        else:
            merged[k] = base_val

    torch._dynamo.reset()
    handler.model.decoder.load_state_dict(merged, strict=False, assign=True)
    handler.model.decoder = handler.model.decoder.to(handler.device).to(handler.dtype)
    handler.model.decoder.eval()

    if getattr(handler, "quantization", None) == "nf4":
        _requantize_decoder_nf4(handler.model)

    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_vram("after group-merge + cleanup")

    elapsed = time.time() - t0
    logger.info(f"Merged {len(active_slots)} adapter(s) with group scales in {elapsed:.1f}s")
    handler._merged_dirty = False
