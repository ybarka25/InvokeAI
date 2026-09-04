"""Tiled VACE video upscaler for Wan 2.2.

Bilinear-upscales the source video to the target size, then re-runs VACE denoise on
overlapping spatial x temporal tiles sized to a bounded working resolution, compositing
the results back with a feathered blend. General approach (tile at a bounded working
size, feather-blend adjacent tiles, anchor each new tile to its already-composited
neighbors to avoid seams) is a well-known pattern used by several tiled-upscale tools;
this is an independent implementation built around this fork's own VACE conditioning
and denoise loop, not a port of any of them.

Because each tile is denoised at ``tile_width x tile_height x tile_num_frames``
regardless of the requested ``upscale_width``/``upscale_height``/duration, the
transformer's attention/activation memory is bounded by the tile size, not the
output size -- this is this fork's Wan VACE memory optimization for large outputs,
not just an upscale feature. See the VACE tiled-upscale plan for the full design.

Reuses:
- ``run_wan_vace_denoise_loop`` (wan_vace_denoise.py) for the per-tile denoise loop,
  sharing one long-lived ``_ExpertSwapper`` across every tile so repeated calls hit
  the model cache instead of reloading experts per tile.
- ``encode_control_video_to_vace_condition`` (wan_vace_extension.py) to build each
  tile's VACE condition directly from in-memory cropped pixel tensors, reusing its
  mask input to anchor the temporal crossfade region to the previous segment's
  already-decoded pixels (the same mask convention used by wan_vace_video_encode).
- ``calc_tiles_min_overlap`` (backend/tiles/, InvokeAI's own Multi-Diffusion tile-grid
  utility) for the spatial tile grid. Spatial tiles are denoised in the grid's
  row-major order and composited into a running per-segment canvas *incrementally*,
  tile by tile (not as a single batched pass): each tile after the first has its
  top/left overlap border (vs. an already-processed neighbor) anchored to that
  neighbor's already-composited pixels via the same mask-anchoring trick as the
  temporal crossfade below (hard 0 = inactive/keep-as-is there), and is itself
  pasted into the canvas with a linear-feathered blend over the overlap --
  independent per-tile generation with only a shared control-video crop (no
  neighbor anchoring) was tried first and produced visible ghosting/double-exposure
  seams where adjacent tiles hallucinated different structure at the boundary;
  VACE's control-video guidance alone is not strong enough to keep
  independently-seeded tiles consistent.
- ``plan_temporal_tiles`` / ``crossfade_videos`` (backend/wan/vace_tile_upscale.py).
"""

from contextlib import ExitStack
from typing import Iterable, Optional

import numpy as np
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    VideoField,
    WanConditioningField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField, WanTransformerField
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.invocations.wan_denoise import (
    WanDenoiseInvocation,
    _ExpertSwapper,
    _get_wan_transformer_working_mem_bytes,
    _resolve_variant,
    _validate_spatial_dimensions,
)
from invokeai.app.invocations.wan_vace_denoise import run_wan_vace_denoise_loop
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import iter_video_frames
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.patches.layer_patcher import PatchSpec
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import Tile
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan
from invokeai.backend.wan.extensions.wan_vace_extension import encode_control_video_to_vace_condition
from invokeai.backend.wan.sampling_utils import (
    get_default_latent_channels,
    get_spatial_scale_factor,
    make_noise,
    num_latent_frames_for,
)
from invokeai.backend.wan.vace_tile_upscale import crossfade_videos, plan_temporal_tiles


def _decode_video_to_canvas(context: InvocationContext, video: VideoField, width: int, height: int) -> torch.Tensor:
    """Decode + bilinear-resize a video to ``[T, H, W, 3]`` float32 on CPU, values in [-1, 1]."""
    video_path = context.videos.get_path(video.video_name)
    frames: list[np.ndarray] = []
    for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
        resized = Image.fromarray(np_frame).convert("RGB").resize((width, height), Image.BILINEAR)
        frames.append(np.asarray(resized, dtype=np.float32))
    if not frames:
        raise ValueError(f"Video {video.video_name} decoded to zero frames.")
    pixel = torch.from_numpy(np.stack(frames, axis=0))
    return pixel / 127.5 - 1.0


def _decode_tile_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """VAE-decode denoiser-space latents to ``[T, H, W, 3]`` float32 on CPU, values in [-1, 1]."""
    device = get_effective_device(vae)
    vae_dtype = next(iter(vae.parameters())).dtype
    latents = latents.to(device=device, dtype=vae_dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    latents = latents * latents_std + latents_mean
    decoded = vae.decode(latents, return_dict=False)[0][0]  # [3, T, H, W]
    return decoded.permute(1, 2, 3, 0).contiguous().cpu().float()


def _composite_tile_into_canvas(canvas: torch.Tensor, tile_pixels: torch.Tensor, tile: Tile) -> None:
    """In-place feathered paste of a decoded spatial tile into the running per-segment canvas.

    Blends linearly across the tile's *entire* overlap with already-pasted neighbors
    above/to the left (row-major processing order means top/left neighbors are always
    pasted first) -- not just a thin band clamped to some fixed width. The overlap can be
    much larger than the requested ``spatial_overlap`` (``calc_tiles_min_overlap`` spreads
    any excess across tiles when the tile grid doesn't evenly divide the canvas), and only
    a small border of that overlap is pinned via the VACE control mask (see ``invoke``) --
    the rest of the overlap is independently regenerated content from two different tiles
    that must be faded between over its full width, or a hard seam/ghosting artifact
    appears where the two tiles' independent interpretations meet -- feathering over the
    tile's full overlap while only clamping the *mask* to a small border is what avoids
    that. Applied tile-by-tile as each tile finishes, rather than as a single batched pass
    over all tiles, so that later tiles' spatial anchor can read already-composited
    neighbor content.
    """
    box = tile.coords
    th, tw = box.bottom - box.top, box.right - box.left
    mask = torch.ones((th, tw), dtype=canvas.dtype)

    if tile.overlap.left > 0:
        mask[:, : tile.overlap.left] = torch.linspace(0.0, 1.0, tile.overlap.left, dtype=canvas.dtype)

    if tile.overlap.top > 0:
        top_mask = torch.ones((th, tw), dtype=canvas.dtype)
        top_mask[: tile.overlap.top, :] = torch.linspace(0.0, 1.0, tile.overlap.top, dtype=canvas.dtype).unsqueeze(1)
        mask = torch.minimum(mask, top_mask)

    mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, th, tw, 1], broadcasts over T and channels
    dst = canvas[:, box.top : box.bottom, box.left : box.right, :]
    canvas[:, box.top : box.bottom, box.left : box.right, :] = tile_pixels * mask + dst * (1.0 - mask)


def _match_histogram_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """CDF-based single-channel histogram match of ``source`` onto ``reference`` (both uint8, 2D)."""
    src_values, src_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)
    src_quantiles = np.cumsum(src_counts).astype(np.float64) / source.size
    ref_quantiles = np.cumsum(ref_counts).astype(np.float64) / reference.size
    interp_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    return interp_values[src_indices].reshape(source.shape)


def _color_match_frames(frames: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-frame RGB histogram-match ``frames`` onto ``reference`` (both ``[T, H, W, 3]`` in [-1, 1])."""
    out = torch.empty_like(frames)
    for t in range(frames.shape[0]):
        src = ((frames[t] + 1.0) * 127.5).clamp(0, 255).round().numpy().astype(np.uint8)
        ref = ((reference[t] + 1.0) * 127.5).clamp(0, 255).round().numpy().astype(np.uint8)
        matched = np.stack([_match_histogram_channel(src[..., c], ref[..., c]) for c in range(3)], axis=-1).clip(0, 255)
        out[t] = torch.from_numpy(matched.astype(np.float32)) / 127.5 - 1.0
    return out


def _write_mp4(path, frames: torch.Tensor, fps: int, is_canceled) -> None:
    from invokeai.app.services.session_processor.session_processor_common import CanceledException

    writer = make_mp4_writer(path, fps)
    try:
        for t in range(frames.shape[0]):
            if is_canceled():
                raise CanceledException
            frame = ((frames[t] + 1.0) * 127.5).round().clamp(0, 255).byte().numpy()
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()


@invocation(
    "wan_vace_tile_upscale",
    title="Tiled Upscale (VACE) - Wan 2.2",
    tags=["video", "wan", "vace", "upscale", "tiled"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class WanVaceTileUpscaleInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscale a video by regenerating it tile-by-tile with Wan 2.2 VACE.

    Bilinear-upscales the source video to the target resolution, then re-denoises
    overlapping spatial x temporal tiles at a bounded working resolution
    (``tile_width`` x ``tile_height`` x ``tile_num_frames``), recompositing with a
    feathered blend. Because the transformer only ever runs at tile size, this also
    bounds VRAM/activation memory regardless of the requested output resolution or
    duration -- use it for outputs too large for ``wan_vace_denoise`` in one pass,
    not just for upscaling.
    """

    transformer: WanTransformerField = InputField(
        description="Wan VACE transformer field (e.g. Wan2.2-VACE-Fun-A14B, dual-expert).",
        input=Input.Connection,
        title="Transformer",
    )
    positive_conditioning: WanConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[WanConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="VAE")

    input_video: VideoField = InputField(description="The video to upscale.")
    control_video: Optional[VideoField] = InputField(
        default=None,
        description="Optional separate structural guide video (e.g. a cleaner source). Defaults to input_video itself.",
    )

    upscale_width: int = InputField(default=1280, gt=0, multiple_of=16, description="Target output width.")
    upscale_height: int = InputField(default=720, gt=0, multiple_of=16, description="Target output height.")

    tile_width: int = InputField(
        default=832,
        gt=0,
        multiple_of=16,
        description="Working resolution width per tile (bounds transformer memory).",
    )
    tile_height: int = InputField(
        default=480,
        gt=0,
        multiple_of=16,
        description="Working resolution height per tile (bounds transformer memory).",
    )
    tile_num_frames: int = InputField(
        default=81,
        ge=5,
        description="Frames processed per temporal tile. Must satisfy (n - 1) %% 4 == 0. Bounds transformer "
        "memory the same way tile_width/tile_height do -- videos longer than this are split into "
        "overlapping segments.",
        title="Tile Frame Count",
    )
    spatial_overlap: int = InputField(
        default=64, ge=0, description="Target overlap (px) between adjacent spatial tiles, feathered on blend."
    )
    temporal_crossfade: int = InputField(
        default=8,
        ge=0,
        description="Overlap (frames) between adjacent temporal tiles. Should be a multiple of 4 to align "
        "with Wan's causal-VAE frame grouping (the same alignment needed for clean start/end-frame "
        "anchoring).",
    )

    denoising_strength: float = InputField(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Img2img strength per tile: 1.0 denoises from pure noise (relies solely on the VACE "
        "control branch for fidelity to the source -- prone to per-tile identity drift at tile seams). "
        "Lower values start denoising from the tile's own upscaled-source pixels (VAE-encoded, noised to "
        "the matching schedule step) instead, directly anchoring identity/detail the way a standard "
        "img2img upscaler does; the VACE control branch then only has to add coherence, not invent it.",
    )
    conditioning_scale: float = InputField(default=1.0, ge=0.0, description="Strength of the control-video guidance.")
    guidance_scale: float = InputField(default=5.0, ge=1.0, description="Classifier-free guidance scale.")
    guidance_scale_low_noise: Optional[float] = InputField(
        default=4.0, ge=0.0, description="Optional separate CFG scale for the low-noise expert."
    )
    steps: int = InputField(default=40, gt=0, description="Number of denoising steps per tile.")
    seed: int = InputField(default=0, description="Base randomness seed; each tile derives its own seed from this.")

    color_match: bool = InputField(
        default=False,
        description="Histogram-match each output frame's colors back to the upscaled source, to counter "
        "VAE color drift across tiles/segments.",
    )
    fps: int = InputField(default=16, ge=1, le=120, description="Frames-per-second for the encoded MP4.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> VideoOutput:
        if (self.tile_num_frames - 1) % 4 != 0:
            raise ValueError(
                f"tile_num_frames must satisfy (tile_num_frames - 1) %% 4 == 0 (got {self.tile_num_frames})."
            )

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        variant = _resolve_variant(context, self.transformer)
        if variant not in (WanVariantType.VACE, WanVariantType.VACE_2_1):
            raise ValueError(
                f"wan_vace_tile_upscale requires a VACE transformer. The selected transformer is {variant.value!r}."
            )
        _validate_spatial_dimensions(variant, self.tile_width, self.tile_height)
        spatial_scale = get_spatial_scale_factor(variant)
        latent_channels = get_default_latent_channels(variant)

        context.util.signal_progress("Decoding source video")
        input_frames = _decode_video_to_canvas(context, self.input_video, self.upscale_width, self.upscale_height)
        if self.control_video is not None:
            control_frames = _decode_video_to_canvas(
                context, self.control_video, self.upscale_width, self.upscale_height
            )
        else:
            control_frames = input_frames
        total_frames = input_frames.shape[0]

        spatial_tiles = calc_tiles_min_overlap(
            image_height=self.upscale_height,
            image_width=self.upscale_width,
            tile_height=self.tile_height,
            tile_width=self.tile_width,
            min_overlap=self.spatial_overlap,
        )
        temporal_tiles = plan_temporal_tiles(total_frames, self.tile_num_frames, self.temporal_crossfade)
        context.logger.info(
            f"Wan VACE tiled upscale: {len(temporal_tiles)} temporal tile(s) x {len(spatial_tiles)} spatial "
            f"tile(s) at {self.tile_width}x{self.tile_height}x{self.tile_num_frames}, output "
            f"{self.upscale_width}x{self.upscale_height}x{total_frames}"
        )

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}.")

        tile_encode_mem = (
            estimate_vae_working_memory_wan(
                operation="encode",
                vae=vae_info.model,
                pixel_height=self.tile_height,
                pixel_width=self.tile_width,
                pixel_frames=self.tile_num_frames,
            )
            * 2
        )  # encode_control_video_to_vace_condition always encodes inactive + reactive streams
        tile_decode_mem = estimate_vae_working_memory_wan(
            operation="decode",
            vae=vae_info.model,
            pixel_height=self.tile_height,
            pixel_width=self.tile_width,
            pixel_frames=self.tile_num_frames,
        )
        vae_working_mem = max(tile_encode_mem, tile_decode_mem)

        proxy = WanDenoiseInvocation.model_construct(
            transformer=self.transformer, positive_conditioning=self.positive_conditioning
        )
        scheduler = WanDenoiseInvocation._build_scheduler(proxy, context, device)

        pos_cond = self._load_conditioning(context, self.positive_conditioning, device=device, dtype=inference_dtype)
        neg_cond: Optional[WanConditioningInfo] = None
        if self.negative_conditioning is not None:
            neg_cond = self._load_conditioning(
                context, self.negative_conditioning, device=device, dtype=inference_dtype
            )

        step_callback = self._build_step_callback(context)

        high_model = self.transformer.transformer
        low_model = self.transformer.transformer_low_noise
        high_config = context.models.get_config(high_model)
        low_config = context.models.get_config(low_model) if low_model is not None else None
        max_resident_gb = context.config.get().wan_max_resident_transformer_gb
        working_mem_bytes = _get_wan_transformer_working_mem_bytes(device, max_resident_gb=max_resident_gb)
        max_resident_bytes = int(max_resident_gb * (2**30)) if working_mem_bytes is not None else None

        def high_lora_factory() -> Iterable[PatchSpec]:
            return proxy._lora_iterator(context, self.transformer.loras)

        def low_lora_factory() -> Iterable[PatchSpec]:
            return proxy._lora_iterator(context, self.transformer.loras_low_noise)

        output_segments: list[torch.Tensor] = []
        prev_tail: Optional[torch.Tensor] = None  # last `crossfade` decoded frames of prev segment, [T,H,W,3]

        with ExitStack() as exit_stack:
            _, vae = exit_stack.enter_context(vae_info.model_on_device(working_mem_bytes=vae_working_mem))
            assert isinstance(vae, AutoencoderKLWan)
            vae.disable_tiling()

            swapper = _ExpertSwapper(
                context=context,
                high_model=high_model,
                low_model=low_model,
                inference_dtype=inference_dtype,
                high_lora_factory=high_lora_factory if self.transformer.loras else None,
                low_lora_factory=low_lora_factory if self.transformer.loras_low_noise else None,
                high_is_quantized=high_config.format == ModelFormat.GGUFQuantized,
                low_is_quantized=low_config.format == ModelFormat.GGUFQuantized if low_config is not None else False,
                working_mem_bytes=working_mem_bytes,
                max_resident_model_bytes=max_resident_bytes,
            )
            exit_stack.callback(swapper.close)

            for temporal_idx, ttile in enumerate(temporal_tiles):
                context.util.signal_progress(
                    f"Wan VACE tiled upscale: segment {temporal_idx + 1}/{len(temporal_tiles)}"
                )
                control_segment = control_frames[ttile.start : ttile.start + ttile.length]
                input_segment = input_frames[ttile.start : ttile.start + ttile.length]
                segment_canvas = torch.zeros(
                    (ttile.length, self.upscale_height, self.upscale_width, 3), dtype=torch.float32
                )

                # One noise draw for the whole upscaled canvas, shared across this segment's spatial
                # tiles (sliced per tile below) rather than an independent draw per tile. Adjacent tiles
                # then start denoising from noise that already agrees in their overlap region -- and
                # correlates smoothly beyond it -- instead of two uncorrelated random fields meeting at
                # the seam, which was a major driver of the independent-identity/seam artifacts seen with
                # per-tile noise (mask-based anchoring alone only pins a thin border, not the noise).
                t_lat_segment = num_latent_frames_for(ttile.length)
                segment_noise = make_noise(
                    batch_size=1,
                    latent_channels=latent_channels,
                    height=self.upscale_height,
                    width=self.upscale_width,
                    spatial_scale_factor=spatial_scale,
                    device=device,
                    dtype=torch.float32,
                    seed=self.seed + temporal_idx * 10007,
                    num_latent_frames=t_lat_segment,
                )

                for spatial_idx, stile in enumerate(spatial_tiles):
                    box = stile.coords
                    control_crop = control_segment[:, box.top : box.bottom, box.left : box.right, :].clone()
                    mask_crop = torch.ones(
                        (ttile.length, box.bottom - box.top, box.right - box.left), dtype=torch.float32
                    )
                    # Spatial anchor: border(s) shared with already-processed neighbor tile(s) (row-major
                    # order means top/left neighbors are always composited into segment_canvas already) are
                    # pinned to that already-generated content -- otherwise each tile is denoised independently
                    # from a fresh seed and can hallucinate different structure at the seam (see module
                    # docstring). Clamped to spatial_overlap so the mask border stays a small,
                    # fixed width even when the tile grid's actual overlap is larger (see
                    # _composite_tile_into_canvas's docstring for why the composite blend itself
                    # doesn't use this same clamp).
                    if stile.overlap.left > 0:
                        w = min(stile.overlap.left, self.spatial_overlap)
                        control_crop[:, :, :w, :] = segment_canvas[:, box.top : box.bottom, box.left : box.left + w, :]
                        mask_crop[:, :, :w] = 0.0
                    if stile.overlap.top > 0:
                        h = min(stile.overlap.top, self.spatial_overlap)
                        control_crop[:, :h, :, :] = segment_canvas[:, box.top : box.top + h, box.left : box.right, :]
                        mask_crop[:, :h, :] = 0.0
                    # Temporal anchor (applied after, so it wins on any frame range it overlaps with the
                    # spatial anchor above -- the two can disagree at a tile that's both a spatial and
                    # a temporal seam, and the temporal continuity constraint takes priority there).
                    if ttile.crossfade > 0 and prev_tail is not None:
                        anchor = prev_tail[-ttile.crossfade :, box.top : box.bottom, box.left : box.right, :]
                        control_crop[: ttile.crossfade] = anchor
                        mask_crop[: ttile.crossfade] = 0.0

                    control_pixel = (
                        control_crop.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=inference_dtype)
                    )
                    mask_pixel = mask_crop.unsqueeze(0).unsqueeze(0).to(device=device, dtype=inference_dtype)

                    control_hidden_states = encode_control_video_to_vace_condition(
                        video=control_pixel,
                        vae=vae,
                        device=device,
                        dtype=inference_dtype,
                        mask=mask_pixel,
                    )

                    t_lat = t_lat_segment
                    lat_top = box.top // spatial_scale
                    lat_left = box.left // spatial_scale
                    lat_h = (box.bottom - box.top) // spatial_scale
                    lat_w = (box.right - box.left) // spatial_scale
                    noise_crop = segment_noise[:, :, :, lat_top : lat_top + lat_h, lat_left : lat_left + lat_w]
                    scheduler.set_timesteps(num_inference_steps=self.steps, device=device)

                    start_step_idx = min(
                        int(round((1.0 - self.denoising_strength) * self.steps)), max(self.steps - 1, 0)
                    )
                    if start_step_idx > 0:
                        # img2img: encode this tile's own upscaled-source crop (not the mask-anchored
                        # control_crop, which has its border pixels overwritten by neighbor-canvas
                        # content) and mix it with noise at the strength the resumed schedule step
                        # expects, exactly mirroring wan_denoise.py's image-to-image init-latent path
                        # (``s_0 * noise + (1 - s_0) * init_latents``). This is the actual fidelity
                        # anchor -- the VACE control branch alone was found to be too soft to hold
                        # per-tile identity/detail consistent, however strongly conditioning_scale
                        # pushed it (see module docstring / this node's tuning history).
                        source_crop = input_segment[:, box.top : box.bottom, box.left : box.right, :]
                        source_pixel = (
                            source_crop.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=inference_dtype)
                        )
                        init_latents = vae.encode(source_pixel, return_dict=False)[0].mode()
                        latents_mean = (
                            torch.tensor(vae.config.latents_mean)
                            .view(1, -1, 1, 1, 1)
                            .to(init_latents.device, init_latents.dtype)
                        )
                        latents_std = (
                            torch.tensor(vae.config.latents_std)
                            .view(1, -1, 1, 1, 1)
                            .to(init_latents.device, init_latents.dtype)
                        )
                        init_latents = ((init_latents - latents_mean) / latents_std).to(torch.float32)
                        s_0 = float(scheduler.sigmas[start_step_idx])
                        latents = s_0 * noise_crop.to(torch.float32) + (1.0 - s_0) * init_latents
                    else:
                        latents = noise_crop.clone()

                    latents = run_wan_vace_denoise_loop(
                        context=context,
                        transformer_field=self.transformer,
                        scheduler=scheduler,
                        latents=latents,
                        control_hidden_states=control_hidden_states,
                        pos_cond=pos_cond,
                        neg_cond=neg_cond,
                        guidance_scale=self.guidance_scale,
                        guidance_scale_low_noise=self.guidance_scale_low_noise,
                        conditioning_scale=self.conditioning_scale,
                        device=device,
                        inference_dtype=inference_dtype,
                        step_callback=step_callback,
                        t_lat=t_lat,
                        swapper=swapper,
                        progress_desc=(
                            f"Wan VACE tile upscale (segment {temporal_idx + 1}/{len(temporal_tiles)}, "
                            f"tile {spatial_idx + 1}/{len(spatial_tiles)})"
                        ),
                        start_step_idx=start_step_idx,
                    )

                    tile_pixels = _decode_tile_latents(vae, latents)
                    _composite_tile_into_canvas(segment_canvas, tile_pixels, stile)
                    TorchDevice.empty_cache()

                segment_tensor = segment_canvas
                output_segments.append(segment_tensor)
                # Size the carried-forward tail to what the *next* tile will need to anchor its
                # crossfade region -- not this tile's own crossfade, which can differ (e.g. the
                # final tile is anchored to end exactly on the last frame and may need a larger
                # overlap with its predecessor than `temporal_crossfade` requested).
                next_crossfade = (
                    temporal_tiles[temporal_idx + 1].crossfade if temporal_idx + 1 < len(temporal_tiles) else 0
                )
                prev_tail = segment_tensor[-next_crossfade:] if next_crossfade > 0 else None

        result = output_segments[0]
        for idx in range(1, len(output_segments)):
            result = crossfade_videos(result, output_segments[idx], temporal_tiles[idx].crossfade)

        if self.color_match:
            context.util.signal_progress("Color-matching output frames")
            result = _color_match_frames(result, input_frames)

        return self._encode_and_save(context, result)

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: WanConditioningField,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WanConditioningInfo:
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        assert len(cond_data.conditionings) == 1
        cond_info = cond_data.conditionings[0]
        assert isinstance(cond_info, WanConditioningInfo)
        return cond_info.to(device=device, dtype=dtype)

    def _build_step_callback(self, context: InvocationContext):
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Wan)

        return step_callback

    def _encode_and_save(self, context: InvocationContext, frames: torch.Tensor) -> VideoOutput:
        import tempfile
        from pathlib import Path

        num_frames, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
        duration = num_frames / float(self.fps)

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_wan_tile_upscale_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            context.util.signal_progress(f"Encoding MP4 ({num_frames} frames @ {self.fps} fps)")
            _write_mp4(tmp_path, frames, self.fps, context.util.is_canceled)
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=duration, fps=float(self.fps)
            )
            context.logger.info(f"Saved video: {video_dto.video_name}")
            return VideoOutput.build(video_dto)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
