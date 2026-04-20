"""MPS/MLX performance patches injected into the ACE-Step API server process.

Applied by launch_api_server.py before the server starts.
All patches are no-ops on non-MPS platforms and fail safely.

Patches applied:
  1. MLX DiT float16 weights  — halves MLX model memory (~4.7 GB -> ~2.35 GB)
  2. PyTorch decoder CPU offload after MLX init  — saves another ~4.7 GB
  3. _load_model_context extended for MPS+MLX reload  — needed by patch 2
  4. Text encoder float16 on MPS  — saves ~600 MB

Set STREAMLINE_MLX_PATCHES=0 to disable all patches for debugging.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from loguru import logger


def _patch_mlx_dit_float16() -> None:
    """Replace float32 weight conversion with float16 in the MLX DiT loader.

    Halves the memory footprint of the MLX DiT model (~4.7 GB -> ~2.35 GB)
    and speeds up Metal matrix operations by 20-40%.
    """
    from acestep.models.mlx import dit_convert

    def _convert_float16(pytorch_model):
        """Convert PyTorch decoder weights to float16 MLX arrays."""
        import mlx.core as mx

        decoder = pytorch_model.decoder
        state_dict = decoder.state_dict()
        weights = []

        for key, value in state_dict.items():
            # float16 instead of the original float32
            np_val = value.detach().cpu().half().numpy()
            new_key = key

            if key.startswith("proj_in.1."):
                new_key = key.replace("proj_in.1.", "proj_in.")
                if new_key.endswith(".weight"):
                    # PT Conv1d: [out, in, K] -> MLX: [out, K, in]
                    np_val = np_val.swapaxes(1, 2)
            elif key.startswith("proj_out.1."):
                new_key = key.replace("proj_out.1.", "proj_out.")
                if new_key.endswith(".weight"):
                    # PT ConvTranspose1d: [in, out, K] -> MLX: [out, K, in]
                    np_val = np_val.transpose(1, 2, 0)
            elif "rotary_emb" in key:
                # Recomputed natively in MLX
                continue

            weights.append((new_key, mx.array(np_val)))

        logger.info(
            "[Streamline-patch] Converted %d DiT params to float16 for MLX.", len(weights)
        )
        return weights

    dit_convert.convert_decoder_weights = _convert_float16
    logger.info(
        "[Streamline-patch] MLX DiT float16 weights patch applied (~50%% memory reduction)."
    )


def _patch_mlx_decoder_offload() -> None:
    """Offload PyTorch model to CPU immediately after the MLX DiT copy is ready.

    On MPS, both the PyTorch model (~4.7 GB) and the MLX copy (~2.35 GB with
    patch 1) would otherwise occupy unified memory simultaneously.  Since all
    diffusion steps run via MLX, the PyTorch model only needs to be resident
    for rare lyric-timestamp / lyric-score operations.  This patch moves it to
    CPU right after MLX init and reloads on demand via the extended
    _load_model_context below.
    """
    from acestep.core.generation.handler import mlx_dit_init as _dit_mod
    from acestep.core.generation.handler import init_service_offload_context as _ctx_mod

    # ---- Patch 2a: offload immediately after successful MLX initialisation ----

    _original_init_mlx_dit = _dit_mod.MlxDitInitMixin._init_mlx_dit

    def _patched_init_mlx_dit(self, compile_model: bool = False) -> bool:
        """Wrap _init_mlx_dit to offload PyTorch model after MLX copy is ready."""
        result = _original_init_mlx_dit(self, compile_model)
        if result and getattr(self, "model", None) is not None:
            logger.info(
                "[Streamline-patch] MLX DiT ready; offloading PyTorch model to CPU "
                "(reclaims ~4.7 GB unified memory)."
            )
            self._recursive_to_device(self.model, "cpu")
            if getattr(self, "silence_latent", None) is not None:
                self.silence_latent = self.silence_latent.to("cpu")
            self._release_system_memory()
        return result

    _dit_mod.MlxDitInitMixin._init_mlx_dit = _patched_init_mlx_dit

    # ---- Patch 2b: reload on demand when the model is needed (lyric ops) ----

    _original_load_model_context = _ctx_mod.InitServiceOffloadContextMixin._load_model_context

    @contextmanager
    def _patched_load_model_context(self, model_name: str):
        """Extended context: reload PyTorch model from CPU when MPS+MLX is active.

        When the MLX DiT decoder is active and the PyTorch model has been
        offloaded to CPU by patch 2a, this context manager temporarily moves
        the model back to MPS for lyric-timestamp / lyric-score operations,
        then re-offloads it on exit.
        """
        if (
            model_name == "model"
            and not self.offload_to_cpu
            and getattr(self, "use_mlx_dit", False)
            and getattr(self, "model", None) is not None
        ):
            try:
                param = next(self.model.parameters())
                model_is_on_cpu = param.device.type == "cpu"
            except StopIteration:
                model_is_on_cpu = False

            if model_is_on_cpu:
                logger.info(
                    "[Streamline-patch] Loading PyTorch model to %s for lyric operation.",
                    self.device,
                )
                self._recursive_to_device(self.model, self.device, self.dtype)
                if getattr(self, "silence_latent", None) is not None:
                    self.silence_latent = (
                        self.silence_latent.to(self.device).to(self.dtype)
                    )
                self._release_system_memory()
                try:
                    yield
                finally:
                    logger.info(
                        "[Streamline-patch] Offloading PyTorch model back to CPU."
                    )
                    self._recursive_to_device(self.model, "cpu")
                    if getattr(self, "silence_latent", None) is not None:
                        self.silence_latent = self.silence_latent.to("cpu")
                    self._release_system_memory()
                return

        # Default path: delegate to the original context manager
        with _original_load_model_context(self, model_name):
            yield

    _ctx_mod.InitServiceOffloadContextMixin._load_model_context = (
        _patched_load_model_context
    )
    logger.info(
        "[Streamline-patch] MLX decoder CPU offload patch applied (~4.7 GB reclaimed after init)."
    )


def _patch_text_encoder_float16() -> None:
    """Cast the text encoder to float16 on MPS after loading.

    Qwen3-Embedding-0.6B is numerically stable in float16 for inference.
    Outputs are upcast to float32 before the MLX diffusion loop anyway
    (via the existing .float().numpy() in diffusion.py), so generation
    quality is unaffected.  Saves ~600 MB of unified memory.
    """
    from acestep.core.generation.handler import init_service_loader_components as _comp_mod

    _original_load_te = (
        _comp_mod.InitServiceLoaderComponentsMixin._load_text_encoder_and_tokenizer
    )

    def _patched_load_text_encoder(self, *, checkpoint_dir: str, device: str) -> str:
        """Wrap text encoder loader to downcast to float16 on MPS."""
        result = _original_load_te(self, checkpoint_dir=checkpoint_dir, device=device)
        if device == "mps" and getattr(self, "text_encoder", None) is not None:
            self.text_encoder = self.text_encoder.to(torch.float16)
            logger.info(
                "[Streamline-patch] Text encoder cast to float16 (~600 MB saved)."
            )
        return result

    _comp_mod.InitServiceLoaderComponentsMixin._load_text_encoder_and_tokenizer = (
        _patched_load_text_encoder
    )
    logger.info("[Streamline-patch] Text encoder float16 patch applied.")


def _patch_mlx_vae_chunk_size() -> None:
    """Reduce MLX VAE decode chunk size to stay within Metal's per-buffer allocation cap.

    The default mlx_chunk=2048 in _mlx_decode_single produces a single intermediate
    tensor of ~15.6 GB per chunk (VAE upsamples latents ~1760x in the time dimension),
    which exceeds Metal's hard per-buffer limit of ~9.5 GB and causes MLX VAE decode
    to fail entirely, triggering the slow PyTorch MPS / CPU fallback paths.

    512-frame chunks produce ~3.9 GB of intermediates each — safely under the limit.
    Override via STREAMLINE_MLX_VAE_CHUNK env var if needed.
    Also calls mx.clear_cache() after each chunk to release Metal buffers promptly.
    """
    import math
    import os

    from acestep.core.generation.handler import mlx_vae_decode_native as _mod

    def _patched_mlx_decode_single(self, z_nlc, decode_fn=None):
        """Tiled MLX VAE decode with Metal-safe chunk size."""
        import mlx.core as mx
        from tqdm import tqdm

        latent_frames = z_nlc.shape[1]
        mlx_chunk = int(os.environ.get("STREAMLINE_MLX_VAE_CHUNK", "512"))
        mlx_overlap = max(8, mlx_chunk // 16)  # scale overlap with chunk size

        # Always use the uncompiled decode fn for chunked paths.
        # mx.compile shape-specializes on first call, so using the compiled
        # version for a 480-frame chunk can still try to allocate buffers
        # sized for the original 2250-frame input.
        if decode_fn is None:
            decode_fn = self._resolve_mlx_decode_fn()
        uncompiled_fn = getattr(self, "mlx_vae", None)
        if uncompiled_fn is not None:
            uncompiled_fn = uncompiled_fn.decode
        else:
            uncompiled_fn = decode_fn

        if latent_frames <= mlx_chunk:
            result = uncompiled_fn(z_nlc)
            mx.eval(result)
            return result

        stride = mlx_chunk - 2 * mlx_overlap
        num_steps = math.ceil(latent_frames / stride)
        decoded_parts = []
        upsample_factor = None

        for idx in tqdm(range(num_steps), desc="MLX VAE decode", disable=self.disable_tqdm):
            core_start = idx * stride
            core_end = min(core_start + stride, latent_frames)
            win_start = max(0, core_start - mlx_overlap)
            win_end = min(latent_frames, core_end + mlx_overlap)

            chunk = z_nlc[:, win_start:win_end, :]
            audio_chunk = uncompiled_fn(chunk)
            mx.eval(audio_chunk)

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[1] / chunk.shape[1]

            trim_start = int(round((core_start - win_start) * upsample_factor))
            trim_end = int(round((win_end - core_end) * upsample_factor))
            audio_len = audio_chunk.shape[1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            decoded_parts.append(audio_chunk[:, trim_start:end_idx, :])
            mx.clear_cache()  # release Metal buffers before next chunk

        return mx.concatenate(decoded_parts, axis=1)

    _mod.MlxVaeDecodeNativeMixin._mlx_decode_single = _patched_mlx_decode_single
    chunk = os.environ.get("STREAMLINE_MLX_VAE_CHUNK", "512")
    logger.info(
        f"[Streamline-patch] MLX VAE chunk size patched (chunk={chunk} frames, "
        "prevents Metal max-buffer overflow; override via STREAMLINE_MLX_VAE_CHUNK)."
    )

def _patch_bypass_gpu_tier() -> None:
    """Override get_gpu_tier to return 'unlimited' on MPS when STREAMLINE_BYPASS_GPU_TIER=1.

    ACE-Step's tier system caps batch sizes and available LM models based on
    detected VRAM.  On Apple Silicon (unified memory, 11.8 GB reported for
    16 GB chips), the system is placed in tier4 which restricts batch size to
    2 and LM models to 0.6B only.

    With MLX patches active, the effective memory footprint is ~4-5 GB, so
    these restrictions are overly conservative.  This patch returns 'unlimited'
    tier on MPS, giving access to all LM models and larger batch sizes while
    preserving all other MPS overrides (no compile_default, no quantization,
    mlx backend, pt_mlx_only backend restriction).
    """
    import os

    if os.environ.get("STREAMLINE_BYPASS_GPU_TIER") != "1":
        return

    from acestep import gpu_config as _gpu_config_mod

    _orig_get_gpu_tier = _gpu_config_mod.get_gpu_tier

    def _unlimited_get_gpu_tier(gpu_memory_gb: float) -> str:
        """Return 'unlimited' tier on MPS regardless of detected memory."""
        return "unlimited"

    _gpu_config_mod.get_gpu_tier = _unlimited_get_gpu_tier
    logger.info(
        "[Streamline-patch] GPU tier bypass active — 'unlimited' tier applied on MPS "
        "(all LM models and batch sizes unlocked)."
    )


def apply_mlx_patches() -> None:
    """Apply all MPS/MLX performance patches.

    No-op on non-MPS platforms.  Each patch fails independently and logs a
    warning rather than crashing the server, so a partial patch is always
    safer than no server at all.

    Set STREAMLINE_MLX_PATCHES=0 to disable all patches (useful for debugging).
    """
    import os

    from acestep.gpu_config import is_mps_platform

    if not is_mps_platform():
        return

    if os.environ.get("STREAMLINE_MLX_PATCHES", "1").strip() == "0":
        logger.info(
            "[Streamline-patch] MLX patches disabled via STREAMLINE_MLX_PATCHES=0."
        )
        return

    logger.info("[Streamline-patch] MPS detected — applying MLX performance patches...")

    patches = [
        ("MLX DiT float16 weights", _patch_mlx_dit_float16),
        ("MLX decoder CPU offload", _patch_mlx_decoder_offload),
        ("text encoder float16", _patch_text_encoder_float16),
        ("MLX VAE chunk size", _patch_mlx_vae_chunk_size),
        ("GPU tier bypass", _patch_bypass_gpu_tier),
    ]
    for name, fn in patches:
        try:
            fn()
        except Exception as exc:
            logger.warning(
                "[Streamline-patch] '{}' patch failed (non-fatal): {}", name, exc
            )

    logger.info("[Streamline-patch] All MLX patches applied successfully.")
