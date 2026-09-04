"""Control-video (VAE-latent) encoder for Wan 2.2 VACE video-to-video conditioning.

Wan 2.2 VACE (video-to-video) conditions on a control video by VAE-encoding it
into a 96-channel tensor that the VACE control branch (``vace_blocks``)
consumes as ``control_hidden_states`` at every denoise step, alongside the
usual 16-channel noise latents. Unlike I2V's single-frame reference-image
conditioning, this drives the whole clip's structure (e.g. pose/depth/edge
video, or a rough draft to restyle).

An optional region mask enables VACE's masked editing mode (mirrors ComfyUI's
VACE mask input): white = regenerate this region guided by the control video,
black = keep the control video's pixels as-is here, unregenerated. With no
mask, the whole frame is "reactive" (fully guided by the control video), same
as before. See ``backend/wan/extensions/wan_vace_extension.py`` for the
encode math.
"""

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers.models.autoencoders import AutoencoderKLWan
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    VideoField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import WanVaceConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_thumbnails import iter_video_frames
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan
from invokeai.backend.wan.extensions.wan_vace_extension import encode_control_video_to_vace_condition


@invocation(
    "wan_vace_video_encode",
    title="Control Video - Wan 2.2 VACE",
    tags=["video", "conditioning", "wan", "vace", "v2v"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanVaceVideoEncodeInvocation(BaseInvocation):
    """VAE-encode a control video into Wan 2.2 VACE conditioning.

    ``width``/``height``/``num_frames`` must match the values on the paired
    ``wan_vace_denoise`` node. The control video is resized (not cropped) to
    ``width``x``height``; if it has fewer than ``num_frames`` decodable
    frames, the last frame is held to pad it out — if more, the extras are
    dropped.
    """

    video: VideoField = InputField(
        description="Control video providing the structural guide (pose, depth, edges, a rough draft, etc.)."
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="VAE")
    # Must match wan_vace_denoise's width/height. multiple_of=16 for the same
    # reason as wan_video_denoise: transformer patch_size=(1, 2, 2) needs even
    # latent H/W.
    width: int = InputField(
        default=832,
        gt=0,
        multiple_of=16,
        description="Width to resize the control video to (must match denoise width).",
    )
    height: int = InputField(
        default=480,
        gt=0,
        multiple_of=16,
        description="Height to resize the control video to (must match denoise height).",
    )
    num_frames: int = InputField(
        default=81,
        ge=5,
        description="Pixel-frame count to build the condition for (must match denoise num_frames). "
        "Must satisfy (num_frames - 1) %% 4 == 0.",
        title="Number of Frames",
    )
    reference_images: list[ImageField] = InputField(
        default=[],
        description="Optional reference images for subject/character identity conditioning (VACE's "
        "separate reference-image branch, distinct from the control video). Each is resized to "
        "width x height and prepended as one extra latent frame; the paired wan_vace_denoise node "
        "strips these back off the output automatically.",
    )
    mask: Optional[VideoField] = InputField(
        default=None,
        description="Optional region mask video for masked editing. White regenerates that region "
        "guided by the control video; black keeps the control video's pixels as-is there. Resized/"
        "padded the same way as the control video (must decode to the same usable frame count). "
        "Leave unset to guide the whole frame (previous default behavior).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> WanVaceConditioningOutput:
        if (self.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (num_frames - 1) %% 4 == 0 for the Wan VAE's temporal "
                f"compression (got {self.num_frames}). Try 5, 9, 13, ..., 81, 85, ..."
            )

        video_path = context.videos.get_path(self.video.video_name)

        context.util.signal_progress("Decoding VACE control video")
        frames: list[torch.Tensor] = []
        for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
            resized = Image.fromarray(np_frame).convert("RGB").resize((self.width, self.height), Image.LANCZOS)
            frames.append(TF.to_tensor(resized))  # [3, H, W] in [0, 1]
            if len(frames) >= self.num_frames:
                break

        if not frames:
            raise ValueError(f"Control video {self.video.video_name} decoded to zero frames.")
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))

        # [3, T, H, W] -> [1, 3, T, H, W], scaled to [-1, 1] to match the Wan
        # VAE's expected input range.
        pixel = torch.stack(frames, dim=1).unsqueeze(0)
        pixel = pixel * 2.0 - 1.0

        mask_pixel: Optional[torch.Tensor] = None
        if self.mask is not None:
            mask_path = context.videos.get_path(self.mask.video_name)
            context.util.signal_progress("Decoding VACE region mask video")
            mask_frames: list[torch.Tensor] = []
            for np_frame in iter_video_frames(mask_path, is_canceled=context.util.is_canceled):
                resized = Image.fromarray(np_frame).convert("L").resize((self.width, self.height), Image.LANCZOS)
                mask_frames.append(torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0))  # [H, W] in [0, 1]
                if len(mask_frames) >= self.num_frames:
                    break
            if not mask_frames:
                raise ValueError(f"Mask video {self.mask.video_name} decoded to zero frames.")
            if len(mask_frames) < self.num_frames:
                mask_frames.extend([mask_frames[-1]] * (self.num_frames - len(mask_frames)))
            # [T, H, W] -> [1, 1, T, H, W]
            mask_pixel = torch.stack(mask_frames, dim=0).unsqueeze(0).unsqueeze(0)

        ref_pixel_tensors: list[torch.Tensor] = []
        for ref_image_field in self.reference_images:
            ref_pil = context.images.get_pil(ref_image_field.image_name, "RGB").resize(
                (self.width, self.height), Image.LANCZOS
            )
            ref_tensor = TF.to_tensor(ref_pil) * 2.0 - 1.0  # [3, H, W] in [-1, 1]
            ref_pixel_tensors.append(ref_tensor.unsqueeze(0).unsqueeze(2))  # [1, 3, 1, H, W]

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(
                f"VACE control-video encoder requires AutoencoderKLWan, got {type(vae_info.model).__name__}."
            )

        def _full_frame_estimate() -> int:
            estimate = estimate_vae_working_memory_wan(
                operation="encode",
                vae=vae_info.model,
                pixel_height=self.height,
                pixel_width=self.width,
                pixel_frames=self.num_frames,
            )
            # encode_control_video_to_vace_condition() VAE-encodes the control video twice
            # (inactive + reactive, for VACE's mask-based region split -- see
            # wan_vace_extension.py), always, regardless of whether a mask is actually given. The
            # single-encode estimate above only budgets half of that peak. Each reference image
            # adds one more small single-frame encode on top.
            estimate *= 2
            if self.reference_images:
                estimate += len(self.reference_images) * estimate_vae_working_memory_wan(
                    operation="encode",
                    vae=vae_info.model,
                    pixel_height=self.height,
                    pixel_width=self.width,
                    pixel_frames=1,
                )
            return estimate

        estimated_working_memory = _full_frame_estimate()

        # Long/high-res control videos can need a working set no card fits. When the
        # full-frame estimate exceeds the execution device's total VRAM, fall back to
        # spatial tiling (mirrors wan_l2v's decode-side fallback) and budget for the
        # tiled working set instead. The text encoder and other recently-used models
        # commonly still occupy VRAM at this point (the cache keeps them resident when
        # there's room), so this compares against total, not currently-free, VRAM.
        use_tiling = False
        if not getattr(vae_info.config, "cpu_only", None):
            exec_device = TorchDevice.choose_torch_device()
            total_vram: int | None = None
            if exec_device.type == "cuda":
                total_vram = torch.cuda.get_device_properties(exec_device).total_memory
            elif exec_device.type == "xpu":
                total_vram = torch.xpu.get_device_properties(exec_device).total_memory
            if total_vram is not None and estimated_working_memory > 0.9 * total_vram:
                use_tiling = True
                tile_size = int(getattr(vae_info.model, "tile_sample_min_height", 256))
                estimated_working_memory = estimate_vae_working_memory_wan(
                    operation="encode",
                    vae=vae_info.model,
                    pixel_height=self.height,
                    pixel_width=self.width,
                    pixel_frames=self.num_frames,
                    tile_size=tile_size,
                ) * (2 + (1 if self.reference_images else 0))

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoencoderKLWan)
            device = get_effective_device(vae)
            target_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            context.util.signal_progress(f"VAE-encoding VACE control video ({self.num_frames} frames)")
            # See the matching comment in wan_ref_image_encoder.py: frees cached
            # allocator blocks left over from earlier nodes before this encode's
            # activations compete with them for contiguous VRAM.
            TorchDevice.empty_cache()

            if use_tiling:
                context.logger.info("VACE control video encode: using spatial VAE tiling to fit VRAM.")
                vae.enable_tiling()
            else:
                # AutoencoderKLWan is cached and shared with other nodes/pipelines; clear
                # any tiling state left by a prior encode/decode before running untiled.
                vae.disable_tiling()
            try:
                condition = encode_control_video_to_vace_condition(
                    video=pixel.to(device=device),
                    vae=vae,
                    device=device,
                    dtype=target_dtype,
                    reference_images=[ref.to(device=device) for ref in ref_pixel_tensors] or None,
                    mask=mask_pixel.to(device=device) if mask_pixel is not None else None,
                )
            finally:
                if use_tiling:
                    vae.disable_tiling()

        condition = condition.detach().to("cpu")
        TorchDevice.empty_cache()
        name = context.tensors.save(tensor=condition)
        return WanVaceConditioningOutput.build(
            condition_tensor_name=name,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
            num_reference_images=len(self.reference_images),
        )
