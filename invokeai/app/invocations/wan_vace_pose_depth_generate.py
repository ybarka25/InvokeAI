"""Pose+depth-guided VACE video generation for Wan 2.2, with start/end frame anchoring and
an optional seamless loop.

Structurally the temporal half of ``wan_vace_tile_upscale.py`` (same shared-``_ExpertSwapper``
cycling, same ``plan_temporal_tiles``/``crossfade_videos`` segment stitching, same mask-based
neighbor-anchor trick) but with no spatial tiling -- each cycle denoises the full canvas -- and
two additions:

- Start/end anchor images are spliced into the control video as raw pixels at the true first/
  last frame of the whole clip, with the VACE mask forced to 0 (inactive) there, exactly the
  same soft-anchor mechanism already used for inter-cycle continuity.
- If both anchors are given, an optional "loop bridge" cycle is generated from the end image
  back to the start image (VACE mask=0 pixel-anchor at both ends, fully reactive gray in
  between) and folded into the tail via a ComfyUI-style ``loopback_crossfade``: the bridge is
  appended, then the combined video's tail is crossfaded onto its own head and the duplicated
  tail dropped, so the result plays seamlessly under a video-player loop.

Unlike ``wan_vace_tile_upscale.py``, this node starts every cycle from pure noise (no img2img
init latents) -- there is no "real" source video to preserve pixel-for-pixel here, only a
structural/motion guide, so full generation from noise (softly steered by a low
``conditioning_scale`` control branch) is the correct mode, not upscale-style fidelity.
"""

import math
from contextlib import ExitStack
from typing import Iterable, Literal, Optional

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
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan
from invokeai.backend.wan.extensions.wan_vace_extension import (
    encode_control_video_to_vace_condition,
    run_wan_vace_transformer_with_positional_scale,
)
from invokeai.backend.wan.memory_optimization import wan_memory_optimization
from invokeai.backend.wan.nag import apply_nag
from invokeai.backend.wan.sampling_utils import (
    get_default_latent_channels,
    get_spatial_scale_factor,
    make_noise,
    num_latent_frames_for,
)
from invokeai.backend.wan.vace_tile_upscale import crossfade_videos, plan_temporal_tiles

# --- SA-Solver (Stochastic Adams Solver, NeurIPS 2023) -----------------------------------
#
# Ported directly from ComfyUI's ``comfy/k_diffusion/sa_solver.py`` + the ``sample_sa_solver``
# loop in ``comfy/k_diffusion/sampling.py``, since diffusers' packaged ``SASolverScheduler``
# (used via the normal ``scheduler.step()`` path) was found to leave the schedule one step
# short of full denoising with ``use_flow_sigmas``/``final_sigmas_type`` unavailable to fix it
# from the outside (reproduced: pure-noise output; a post-hoc sigma override to force the
# final step to 0 also failed -- CONFIRMED_BLACK_FRAMES -- because ComfyUI's algorithm treats
# "next sigma is exactly 0" as a special terminal case (`x = denoised` directly, no
# extrapolated update), which diffusers' generic multistep ``step()`` doesn't replicate).
# This reimplements the actual predictor-corrector math against our own CFG/expert-swap loop
# instead of going through a diffusers ``SchedulerMixin`` at all.
#
# Reference: https://github.com/scxue/SA-Solver (NeurIPS 2023, arXiv:2309.05019).


def _sa_solver_half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
    """log(alpha_t / sigma_t) for a flow-matching (CONST) model: alpha_t = 1 - sigma."""
    s = sigma.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log((1.0 - s) / s)


def _sa_solver_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float) -> torch.Tensor:
    tau_mul = 1 + tau_t**2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms_factored = t**p - s**p * (-tau_mul * h).exp()
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    return recursive_coeff_mat @ product_terms_factored


def _sa_solver_simple_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    is_corrector_step: bool,
) -> torch.Tensor:
    tau_mul = 1 + tau_t**2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        b_2 = alpha_t * (0.5 * tau_mul * h**2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])


def _sa_solver_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    is_corrector_step: bool,
) -> torch.Tensor:
    num_timesteps = curr_lambdas.shape[0]
    if num_timesteps == 1:
        # Order-1 fallback (predictor's very first step): plain Euler-in-lambda coefficient.
        tau_mul = 1 + tau_t**2
        h = lambda_t - lambda_s
        alpha_t = sigma_next * lambda_t.exp()
        return (alpha_t * (-h * tau_mul).expm1().neg()).unsqueeze(0)
    exp_integral_coeffs = _sa_solver_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_t = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_t, exp_integral_coeffs)
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


def _sa_solver_tau_func(start_sigma: float, end_sigma: float, eta: float = 1.0):
    def tau(sigma) -> float:
        if eta <= 0:
            return 0.0
        s = float(sigma)
        return eta if start_sigma >= s >= end_sigma else 0.0

    return tau


def _flow_shift_sigma(sigma: float, flow_shift: float) -> float:
    """Same warp diffusers' UniPCMultistepScheduler applies for use_flow_sigmas=True."""
    return flow_shift * sigma / (1 + (flow_shift - 1) * sigma)


def _kl_optimal_flow_sigmas(
    num_inference_steps: int, flow_shift: float, num_train_timesteps: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL-optimal sigma schedule (ComfyUI's ``kl_optimal_scheduler``: interpolates in arctan
    space between sigma_max and sigma_min, then maps back with tan -- denser sampling near the
    schedule's ends than a linear/"simple" schedule), adapted to this pipeline's flow-matching
    sigma convention (bounded to [0, 1], warped by ``flow_shift``, terminal sigma forced to 0).
    """
    sigma_max = _flow_shift_sigma(1.0, flow_shift)
    sigma_min = _flow_shift_sigma(1.0 / num_train_timesteps, flow_shift)

    n = num_inference_steps
    adj_idxs = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    sigmas = np.tan(adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max))

    eps = 1e-6
    if abs(sigmas[0] - 1) < eps:
        sigmas[0] -= eps
    timesteps = sigmas * num_train_timesteps
    sigmas = np.concatenate([sigmas, [0.0]])

    return torch.from_numpy(sigmas).to(torch.float32), torch.from_numpy(timesteps).to(torch.int64)


def _run_sa_solver_vace_denoise_loop(
    *,
    context: InvocationContext,
    transformer_field: WanTransformerField,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    control_hidden_states: torch.Tensor,
    pos_cond: WanConditioningInfo,
    neg_cond: Optional[WanConditioningInfo],
    guidance_scale: float,
    guidance_scale_low_noise: Optional[float],
    conditioning_scale: float,
    device: torch.device,
    inference_dtype: torch.dtype,
    step_callback,
    t_lat: int,
    swapper: _ExpertSwapper,
    progress_desc: str,
    num_ref: int = 0,
    reference_conditioning_scale: float = 1.0,
    predictor_order: int = 3,
    corrector_order: int = 4,
    eta: float = 1.0,
    seed: int = 0,
    use_nag: bool = False,
    nag_scale: float = 5.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
) -> torch.Tensor:
    """SA-Solver predictor-corrector loop (see module-level comment above) driving the same
    per-step CFG + dual-expert swap call as ``run_wan_vace_denoise_loop``, in place of a
    diffusers ``scheduler.step()`` call."""
    total_steps = len(sigmas) - 1
    if total_steps <= 0:
        return latents

    # diffusers schedulers commonly move .timesteps to `device` on set_timesteps() but leave
    # .sigmas CPU-resident (it's only used for internal scheduler math) -- move both explicitly
    # so every tensor derived from them (lambdas, b_coeffs, tau/noise scaling) lands on the same
    # device as the model outputs (pred_mat) they get tensordot-ed against.
    sigmas = sigmas.to(device=device, dtype=torch.float32)
    timesteps = timesteps.to(device=device)

    low_model = transformer_field.transformer_low_noise
    num_train_timesteps = 1000
    boundary_timestep = transformer_field.boundary_ratio * num_train_timesteps if low_model is not None else None

    lambdas = _sa_solver_half_log_snr(sigmas)
    start_idx = max(0, int(0.2 * len(sigmas)))
    end_idx = min(len(sigmas) - 1, int(0.8 * len(sigmas)))
    tau_func = _sa_solver_tau_func(float(sigmas[start_idx]), float(sigmas[end_idx]), eta=eta)
    max_used_order = max(predictor_order, corrector_order)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def call_model(x: torch.Tensor, t_val: torch.Tensor, sigma_val: torch.Tensor) -> torch.Tensor:
        if low_model is not None and float(t_val) < float(boundary_timestep):
            active_label = _ExpertSwapper.LOW
            low_cfg = guidance_scale_low_noise
            active_cfg = low_cfg if (low_cfg is not None and low_cfg >= 1.0) else guidance_scale
        else:
            active_label = _ExpertSwapper.HIGH
            active_cfg = guidance_scale
        transformer = swapper.get(active_label)

        p_h, p_w = transformer.config.patch_size[1:]
        tokens_per_frame = (control_hidden_states.shape[-2] // p_h) * (control_hidden_states.shape[-1] // p_w)
        num_reference_tokens_local = num_ref * tokens_per_frame

        latent_model_input = x.to(dtype=inference_dtype)
        timestep = t_val.expand(x.shape[0])
        with wan_memory_optimization(transformer, enabled=True):
            if use_nag:
                # NAG blends pos/neg guidance inside every cross-attention call -- one forward
                # pass, no separate uncond pass (see wan/nag.py's module docstring).
                assert neg_cond is not None  # validated by the invocation before this loop starts
                neg_encoder_hidden_states = transformer.condition_embedder.text_embedder(
                    neg_cond.prompt_embeds.unsqueeze(0).to(dtype=inference_dtype)
                )
                with apply_nag(transformer, neg_encoder_hidden_states, nag_scale, nag_tau, nag_alpha):
                    noise_pred = run_wan_vace_transformer_with_positional_scale(
                        transformer=transformer,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=pos_cond.prompt_embeds.unsqueeze(0),
                        control_hidden_states=control_hidden_states,
                        num_reference_tokens=num_reference_tokens_local,
                        reference_scale=reference_conditioning_scale,
                        control_scale=conditioning_scale,
                        return_dict=False,
                    )[0]
            else:
                noise_pred_cond = run_wan_vace_transformer_with_positional_scale(
                    transformer=transformer,
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=pos_cond.prompt_embeds.unsqueeze(0),
                    control_hidden_states=control_hidden_states,
                    num_reference_tokens=num_reference_tokens_local,
                    reference_scale=reference_conditioning_scale,
                    control_scale=conditioning_scale,
                    return_dict=False,
                )[0]
                if neg_cond is not None and active_cfg != 1.0:
                    noise_pred_uncond = run_wan_vace_transformer_with_positional_scale(
                        transformer=transformer,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=neg_cond.prompt_embeds.unsqueeze(0),
                        control_hidden_states=control_hidden_states,
                        num_reference_tokens=num_reference_tokens_local,
                        reference_scale=reference_conditioning_scale,
                        control_scale=conditioning_scale,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + active_cfg * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
        # Flow-matching (CONST) x0 conversion, matching comfy.model_sampling.CONST.calculate_denoised.
        return x.to(torch.float32) - sigma_val.to(torch.float32) * noise_pred.to(torch.float32)

    x_pred = latents
    x = latents
    h = torch.zeros((), dtype=torch.float32)
    tau_t = 0.0
    noise = 0.0
    pred_list: list[torch.Tensor] = []
    lower_order_to_end = float(sigmas[-1]) == 0.0

    for i in range(total_steps):
        denoised = call_model(x_pred, timesteps[i], sigmas[i])
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or float(sigmas[i + 1]) == 0.0:
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, total_steps - 1 - i)
            corrector_order_used = min(corrector_order_used, total_steps - i)

        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1 : i + 1]
            b_coeffs = _sa_solver_b_coeffs(sigmas[i], curr_lambdas, lambdas[i - 1], lambdas[i], tau_t, True)
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)
            corr_res = torch.tensordot(pred_mat, b_coeffs.to(pred_mat.dtype), dims=([1], [0]))
            x = sigmas[i] / sigmas[i - 1] * torch.exp(-(tau_t**2) * h) * x + corr_res
            if tau_t > 0:
                x = x + noise

        if float(sigmas[i + 1]) == 0.0:
            x = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1 : i + 1]
            b_coeffs = _sa_solver_b_coeffs(sigmas[i + 1], curr_lambdas, lambdas[i], lambdas[i + 1], tau_t, False)
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)
            pred_res = torch.tensordot(pred_mat, b_coeffs.to(pred_mat.dtype), dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * torch.exp(-(tau_t**2) * h) * x + pred_res
            if tau_t > 0:
                noise_sample = torch.randn(x.shape, dtype=x.dtype, generator=generator).to(x.device)
                noise = noise_sample * sigmas[i + 1] * torch.sqrt((-2 * tau_t**2 * h).expm1().neg())
                x_pred = x_pred + noise

        step_callback(
            PipelineIntermediateState(
                step=i + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(timesteps[i].item()),
                latents=x[:, :, (num_ref + t_lat) // 2],
            )
        )

    return x


def _decode_video_pixels(context: InvocationContext, video: VideoField, width: int, height: int) -> torch.Tensor:
    """Decode a video to ``[T, H, W, 3]`` float32 on CPU, values in [-1, 1]. No resize is applied
    other than a defensive one if the video doesn't already match width/height (it should,
    coming out of ``wan_vace_loop_prep`` / the pose+depth blend chain)."""
    video_path = context.videos.get_path(video.video_name)
    frames: list[np.ndarray] = []
    for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
        img = Image.fromarray(np_frame).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        frames.append(np.asarray(img, dtype=np.float32))
    if not frames:
        raise ValueError(f"Video {video.video_name} decoded to zero frames.")
    pixel = torch.from_numpy(np.stack(frames, axis=0))
    return pixel / 127.5 - 1.0


def _decode_image_pixels(context: InvocationContext, image: ImageField, width: int, height: int) -> torch.Tensor:
    """Decode an image to ``[H, W, 3]`` float32 on CPU, values in [-1, 1].

    Aspect-preserving resize, letterboxed onto a white canvas if the image's own aspect ratio
    doesn't match the target canvas -- matches official VACE's ``prepare_source``/diffusers'
    ``WanVACEPipeline`` reference-image handling (both letterbox onto white, never stretch).
    In practice this rarely does any letterboxing here since the caller
    (``wan_vace_loop_prep``) already crops start/end images to the target aspect ratio; this is
    a correctness fallback for whatever is actually wired in, not a no-op the rest of the time.
    """
    pil = context.images.get_pil(image.image_name, "RGB")
    if pil.size != (width, height):
        src_w, src_h = pil.size
        scale = min(width / src_w, height / src_h)
        new_w, new_h = max(round(src_w * scale), 1), max(round(src_h * scale), 1)
        resized = pil.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        pil = canvas
    return TF.to_tensor(pil).permute(1, 2, 0) * 2.0 - 1.0


def _decode_cycle_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """VAE-decode denoiser-space latents to ``[T, H, W, 3]`` float32 on CPU, values in [-1, 1]."""
    device = get_effective_device(vae)
    vae_dtype = next(iter(vae.parameters())).dtype
    latents = latents.to(device=device, dtype=vae_dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
    latents = latents * latents_std + latents_mean
    decoded = vae.decode(latents, return_dict=False)[0][0]  # [3, T, H, W]
    return decoded.permute(1, 2, 3, 0).contiguous().cpu().float()


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
    "wan_vace_pose_depth_generate",
    title="Pose+Depth Video Generation (VACE) - Wan 2.2",
    tags=["video", "wan", "vace", "pose", "depth", "loop"],
    category="latents",
    version="1.8.1",
    classification=Classification.Prototype,
)
class WanVacePoseDepthGenerateInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate a video guided by a pose+depth control video, anchored to start/end images,
    denoised in overlapping temporal cycles, with an optional seamless loop bridge.

    ``control_video``/``width``/``height``/``num_frames`` should come from ``wan_vace_loop_prep``
    (directly, or via the pose/depth extraction + ``vace_control_blend`` chain wired after it) so
    every input already shares one canvas. ``start_image``/``end_image`` should be
    ``wan_vace_loop_prep``'s own (already cropped/resized) outputs, not the original uploads.
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

    control_video: VideoField = InputField(
        description="Composited pose+depth control video (e.g. from vace_control_blend), already at "
        "width x height x num_frames."
    )
    start_image: Optional[ImageField] = InputField(
        default=None,
        description="Reference photo for the subject's identity/appearance (VACE reference_images channel). "
        "By default this is guidance only -- the model still generates frame 0 itself, following the control "
        "video's pose there. Enable pixel_anchor_start_end to instead splice this image in verbatim as the "
        "literal first frame.",
    )
    end_image: Optional[ImageField] = InputField(
        default=None,
        description="Same as start_image but anchoring the end of the clip (or the loop bridge's target, if "
        "build_loop_bridge is enabled).",
    )
    pixel_anchor_start_end: bool = InputField(
        default=False,
        description="If enabled, splice start_image/end_image in verbatim as the output's literal first/last "
        "frame (raw pixels, VACE mask forced inactive there) instead of letting the model generate them "
        "following the control video's own pose/motion there. Has no effect on the loop bridge cycle (when "
        "build_loop_bridge is on), which always pixel-anchors to close the loop.",
    )

    width: int = InputField(default=832, gt=0, multiple_of=16, description="Canvas width (must match control_video).")
    height: int = InputField(default=480, gt=0, multiple_of=16, description="Canvas height (must match control_video).")
    num_frames: int = InputField(default=49, gt=0, description="Total frame count (must match control_video).")

    frames_per_cycle: int = InputField(
        default=49,
        ge=5,
        description="Frames denoised per cycle. Must satisfy (n - 1) %% 4 == 0. Videos longer than this are "
        "split into overlapping cycles, same as tile_num_frames on the tiled upscaler.",
        title="Frames Per Cycle",
    )
    cycle_crossfade: int = InputField(
        default=8, ge=0, description="Overlap (frames) between adjacent cycles, anchored + crossfaded for continuity."
    )

    conditioning_scale: float = InputField(
        default=1.0,
        ge=0.0,
        description="Strength of the control-video branch (pose/depth/whatever was blended into control_video), "
        "applied per VACE layer inside the transformer. Reference VACE workflows keep this at 1.0; camera "
        "stability comes from the control video's own content, not from turning this down. Does not affect "
        "start/end image anchoring -- see reference_conditioning_scale for that.",
    )
    reference_conditioning_scale: float = InputField(
        default=1.0,
        ge=0.0,
        description="Strength of the reference-image anchor (start_image/end_image, VACE's dedicated identity "
        "channel) independently of conditioning_scale's control-video branch -- e.g. raise this to push "
        "appearance/background fidelity to the reference photo(s) harder without also stiffening pose/motion "
        "guidance. Applied position-wise on the VACE residual injection itself (not on the encoded reference "
        "content), so reference and control-video frames keep influencing each other through the VACE blocks' "
        "own attention while still getting independently-weighted contributions to the output.",
    )
    guidance_scale: float = InputField(default=5.0, ge=1.0, description="Classifier-free guidance scale.")
    guidance_scale_low_noise: Optional[float] = InputField(
        default=4.0, ge=0.0, description="Optional separate CFG scale for the low-noise expert."
    )
    steps: int = InputField(default=40, gt=0, description="Number of denoising steps per cycle.")
    seed: int = InputField(default=0, description="Base randomness seed; each cycle derives its own seed from this.")
    flow_shift: Optional[float] = InputField(
        default=16.0,
        gt=0.0,
        description="The flow-matching schedule's exponential shift. Defaults to 16.0, the official VACE "
        "inference scripts' hard-coded value (both the CLI and Gradio demo override the base model's own "
        "generate() default of 5.0/3.0 with 16 specifically for VACE) -- a much higher shift than plain T2V/I2V "
        "spends far more of the trajectory at high noise, which is where structural conditioning (pose/control "
        "video, identity reference) actually gets decided. Leave unset (None) to fall back to the base "
        "checkpoint's plain T2V/I2V default instead.",
    )
    sampler: Literal["unipc", "sa_solver_simple", "sa_solver_kl_optimal"] = InputField(
        default="unipc",
        description="Solver algorithm + sigma schedule. This fork defaults to UniPC (unset scheduler config, "
        "e.g. GGUF checkpoints). Reference ComfyUI pipelines using the same distillation LoRA stack were found "
        'using SA-Solver instead, with either a linear ("simple") or KL-optimal ("kl_optimal", ComfyUI\'s '
        "kl_optimal_scheduler -- denser sampling near both ends of the schedule) sigma schedule.",
    )

    use_nag: bool = InputField(
        default=False,
        description="Use NAG (Normalized Attention Guidance) instead of classifier-free guidance. NAG blends the "
        "positive/negative text conditioning's raw cross-attention output inside every attention block (main "
        "blocks AND VACE blocks) rather than blending two full model forward passes' noise predictions -- so it "
        "still provides real negative-prompt guidance at guidance_scale=1.0, where CFG's uncond pass is normally "
        "skipped entirely. Reference ComfyUI VACE workflows using a 4-step distillation LoRA at cfg=1.0 use this "
        "(the fork has no negative guidance at all at cfg=1.0 otherwise). Requires negative_conditioning to be "
        "set; ignored for the model's I2V image-context tokens if present (falls back to plain attention there).",
    )
    nag_scale: float = InputField(default=5.0, ge=1.0, description="NAG guidance scale. Only used when use_nag is on.")
    nag_tau: float = InputField(
        default=2.5, gt=0.0, description="NAG's guidance-norm clamp threshold. Only used when use_nag is on."
    )
    nag_alpha: float = InputField(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="NAG's blend factor between guided and unguided attention output. Only used when use_nag is on.",
    )

    build_loop_bridge: bool = InputField(
        default=True,
        description="If both start_image and end_image are set, generate an extra bridging cycle from the "
        "end image back to the start image and fold it into the tail so the output video loops seamlessly. "
        "Ignored (no bridge built) if either anchor image is missing.",
    )
    loop_bridge_frames: Optional[int] = InputField(
        default=None,
        ge=0,
        description="Frame count for the loop-bridge cycle. Must satisfy (n - 1) %% 4 == 0. 0 or unset both "
        "default to frames_per_cycle (the workflow editor's number input writes 0 rather than leaving an "
        "optional field empty, so 0 is treated the same as unset here).",
    )

    fps: int = InputField(default=16, ge=1, le=120, description="Frames-per-second for the encoded MP4.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> VideoOutput:
        if (self.frames_per_cycle - 1) % 4 != 0:
            raise ValueError(
                f"frames_per_cycle must satisfy (frames_per_cycle - 1) %% 4 == 0 (got {self.frames_per_cycle})."
            )
        bridge_frames = self.loop_bridge_frames if self.loop_bridge_frames else self.frames_per_cycle
        want_bridge = self.build_loop_bridge and self.start_image is not None and self.end_image is not None
        if want_bridge and (bridge_frames - 1) % 4 != 0:
            raise ValueError(f"loop_bridge_frames must satisfy (n - 1) %% 4 == 0 (got {bridge_frames}).")
        if self.use_nag and self.negative_conditioning is None:
            raise ValueError("use_nag requires negative_conditioning to be set (NAG needs a negative prompt).")

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        variant = _resolve_variant(context, self.transformer)
        if variant not in (WanVariantType.VACE, WanVariantType.VACE_2_1):
            raise ValueError(
                f"wan_vace_pose_depth_generate requires a VACE transformer. The selected transformer is "
                f"{variant.value!r}."
            )
        _validate_spatial_dimensions(variant, self.width, self.height)
        spatial_scale = get_spatial_scale_factor(variant)
        latent_channels = get_default_latent_channels(variant)

        context.util.signal_progress("Decoding control video")
        control_frames = _decode_video_pixels(context, self.control_video, self.width, self.height)
        if control_frames.shape[0] != self.num_frames:
            raise ValueError(
                f"control_video decoded to {control_frames.shape[0]} frames, expected num_frames={self.num_frames}."
            )

        start_pixel = (
            _decode_image_pixels(context, self.start_image, self.width, self.height) if self.start_image else None
        )
        end_pixel = _decode_image_pixels(context, self.end_image, self.width, self.height) if self.end_image else None

        temporal_tiles = plan_temporal_tiles(self.num_frames, self.frames_per_cycle, self.cycle_crossfade)
        context.logger.info(
            f"Wan VACE pose+depth generate: {len(temporal_tiles)} cycle(s) of {self.frames_per_cycle} frames "
            f"at {self.width}x{self.height}, output {self.num_frames} frames"
            + (f" + loop bridge ({bridge_frames} frames)" if want_bridge else "")
        )

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}.")

        vae_working_mem = (
            estimate_vae_working_memory_wan(
                operation="encode",
                vae=vae_info.model,
                pixel_height=self.height,
                pixel_width=self.width,
                pixel_frames=max(self.frames_per_cycle, bridge_frames if want_bridge else 0),
            )
            * 2
        )
        vae_working_mem = max(
            vae_working_mem,
            estimate_vae_working_memory_wan(
                operation="decode",
                vae=vae_info.model,
                pixel_height=self.height,
                pixel_width=self.width,
                pixel_frames=max(self.frames_per_cycle, bridge_frames if want_bridge else 0),
            ),
        )

        proxy = WanDenoiseInvocation.model_construct(
            transformer=self.transformer, positive_conditioning=self.positive_conditioning
        )
        scheduler = WanDenoiseInvocation._build_scheduler(
            proxy,
            context,
            device,
            flow_shift_override=self.flow_shift,
            sampler="sa_solver" if self.sampler.startswith("sa_solver") else "unipc",
        )

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
        prev_tail: Optional[torch.Tensor] = None

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

            def run_cycle(
                control_segment: torch.Tensor,
                length: int,
                crossfade: int,
                cycle_seed: int,
                progress_desc: str,
                anchor_start: Optional[torch.Tensor] = None,
                anchor_end: Optional[torch.Tensor] = None,
                fade_tail: Optional[int] = None,
                reference_pixels: Optional[list[torch.Tensor]] = None,
            ) -> torch.Tensor:
                control_crop = control_segment.clone()
                mask_crop = torch.ones((length, self.height, self.width), dtype=torch.float32)

                if fade_tail and fade_tail > 0:
                    # Ease this cycle's own pose+depth guidance toward neutral gray (0.0 in this
                    # tensor's [-1, 1] convention -- same "no control" value used for the loop
                    # bridge's free middle) over its last frames, still mask=1/reactive. The bridge
                    # that follows has no structural guidance at all in its interior; crossfading
                    # this cycle's tail straight from full pose+depth rigidity into that gave VACE
                    # two incompatible ideas of what should be happening at the seam (real captured
                    # motion vs. free improvisation), producing limb-doubling ghosting. Loosening
                    # the guide here first gives the model room to already be drifting toward
                    # something the bridge can continue, before the pixel crossfade blends them.
                    n = min(fade_tail, length)
                    fade = torch.linspace(1.0, 0.0, n, dtype=control_crop.dtype).view(-1, 1, 1, 1)
                    control_crop[-n:] = control_crop[-n:] * fade

                if crossfade > 0 and prev_tail is not None:
                    anchor = prev_tail[-crossfade:]
                    control_crop[:crossfade] = anchor
                    mask_crop[:crossfade] = 0.0

                if anchor_start is not None:
                    control_crop[0] = anchor_start
                    mask_crop[0] = 0.0
                if anchor_end is not None:
                    control_crop[-1] = anchor_end
                    mask_crop[-1] = 0.0

                control_pixel = control_crop.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=inference_dtype)
                mask_pixel = mask_crop.unsqueeze(0).unsqueeze(0).to(device=device, dtype=inference_dtype)

                # Reference images (VACE's dedicated identity-anchor channel, same mechanism as
                # wan_vace_video_encode's reference_images / the ComfyUI reference implementation's
                # reference_image + ref_as_init_frame combo) are VAE-encoded and prepended as extra
                # leading latent frames -- not spliced into this cycle's own frame 0/-1. That
                # matters: the frame-0/-1 pixel splice above only gives that one frame something to
                # match, so identity/appearance had nothing anchoring it for the rest of the cycle
                # and drifted toward whatever the prompt implied instead. Prepended reference latents
                # are part of the same sequence every frame's attention can see for the whole cycle,
                # so they hold identity across all of it, not just the anchored frame.
                ref_pixel_tensors = None
                num_ref = 0
                if reference_pixels:
                    ref_pixel_tensors = [
                        p.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=inference_dtype)
                        for p in reference_pixels
                    ]
                    num_ref = len(ref_pixel_tensors)

                control_hidden_states = encode_control_video_to_vace_condition(
                    video=control_pixel,
                    vae=vae,
                    device=device,
                    dtype=inference_dtype,
                    mask=mask_pixel,
                    reference_images=ref_pixel_tensors,
                )

                t_lat = num_latent_frames_for(length)
                latents = make_noise(
                    batch_size=1,
                    latent_channels=latent_channels,
                    height=self.height,
                    width=self.width,
                    spatial_scale_factor=spatial_scale,
                    device=device,
                    dtype=torch.float32,
                    seed=cycle_seed,
                    num_latent_frames=t_lat + num_ref,
                )
                if self.sampler.startswith("sa_solver"):
                    # Bypass diffusers' SASolverScheduler.step() entirely (see the module-level
                    # comment above _run_sa_solver_vace_denoise_loop for why).
                    effective_shift = (
                        self.flow_shift
                        if self.flow_shift is not None
                        else (5.0 if variant == WanVariantType.TI2V_5B else 3.0)
                    )
                    if self.sampler == "sa_solver_kl_optimal":
                        sa_sigmas, sa_timesteps = _kl_optimal_flow_sigmas(
                            num_inference_steps=self.steps, flow_shift=effective_shift
                        )
                        sa_sigmas = sa_sigmas.to(device=device)
                        sa_timesteps = sa_timesteps.to(device=device)
                    else:
                        # "sa_solver_simple": only borrow a throwaway UniPC instance to get a
                        # correctly-terminating (ends at sigma=0), linearly-spaced flow-matching
                        # sigma/timestep schedule, since UniPC's own schedule construction is
                        # already verified correct.
                        from diffusers import UniPCMultistepScheduler

                        sigma_source = UniPCMultistepScheduler(
                            num_train_timesteps=1000,
                            solver_order=2,
                            prediction_type="flow_prediction",
                            flow_shift=effective_shift,
                            use_flow_sigmas=True,
                            solver_type="bh2",
                            final_sigmas_type="zero",
                        )
                        sigma_source.set_timesteps(num_inference_steps=self.steps, device=device)
                        sa_sigmas = sigma_source.sigmas
                        sa_timesteps = sigma_source.timesteps
                    latents = _run_sa_solver_vace_denoise_loop(
                        context=context,
                        transformer_field=self.transformer,
                        sigmas=sa_sigmas,
                        timesteps=sa_timesteps,
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
                        num_ref=num_ref,
                        reference_conditioning_scale=self.reference_conditioning_scale,
                        swapper=swapper,
                        progress_desc=progress_desc,
                        seed=cycle_seed,
                        use_nag=self.use_nag,
                        nag_scale=self.nag_scale,
                        nag_tau=self.nag_tau,
                        nag_alpha=self.nag_alpha,
                    )
                else:
                    scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
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
                        reference_conditioning_scale=self.reference_conditioning_scale,
                        device=device,
                        inference_dtype=inference_dtype,
                        step_callback=step_callback,
                        t_lat=t_lat,
                        num_ref=num_ref,
                        swapper=swapper,
                        progress_desc=progress_desc,
                        use_nag=self.use_nag,
                        nag_scale=self.nag_scale,
                        nag_tau=self.nag_tau,
                        nag_alpha=self.nag_alpha,
                    )
                if num_ref:
                    latents = latents[:, :, num_ref:]
                pixels = _decode_cycle_latents(vae, latents)
                TorchDevice.empty_cache()
                return pixels

            # Persistent identity anchor for every cycle (main + bridge), not just the one that
            # happens to own frame 0/-1 -- otherwise only the first/last cycle had any appearance
            # cue and the rest drifted toward whatever the prompt implied instead of the anchor
            # image(s). Order doesn't carry first/last meaning here (that's what anchor_start/
            # anchor_end + the mask=0 splice are for); this is pure "what this looks like" context.
            reference_pixels_list = [p for p in (start_pixel, end_pixel) if p is not None] or None

            for cycle_idx, ttile in enumerate(temporal_tiles):
                context.util.signal_progress(
                    f"Wan VACE pose+depth generate: cycle {cycle_idx + 1}/{len(temporal_tiles)}"
                )
                is_first = ttile.start == 0
                is_last = ttile.start + ttile.length == self.num_frames
                pixels = run_cycle(
                    control_segment=control_frames[ttile.start : ttile.start + ttile.length],
                    length=ttile.length,
                    crossfade=ttile.crossfade,
                    cycle_seed=self.seed + cycle_idx * 10007,
                    progress_desc=f"Wan VACE pose+depth generate (cycle {cycle_idx + 1}/{len(temporal_tiles)})",
                    anchor_start=start_pixel
                    if (self.pixel_anchor_start_end and is_first and start_pixel is not None)
                    else None,
                    anchor_end=end_pixel
                    if (self.pixel_anchor_start_end and is_last and end_pixel is not None)
                    else None,
                    fade_tail=self.cycle_crossfade if (is_last and want_bridge) else None,
                    reference_pixels=reference_pixels_list,
                )
                output_segments.append(pixels)
                next_crossfade = temporal_tiles[cycle_idx + 1].crossfade if cycle_idx + 1 < len(temporal_tiles) else 0
                prev_tail = pixels[-next_crossfade:] if next_crossfade > 0 else None

            result = output_segments[0]
            for idx in range(1, len(output_segments)):
                result = crossfade_videos(result, output_segments[idx], temporal_tiles[idx].crossfade)

            if want_bridge:
                assert start_pixel is not None and end_pixel is not None
                context.util.signal_progress("Wan VACE pose+depth generate: loop bridge cycle")
                neutral = torch.full((bridge_frames, self.height, self.width, 3), 0.0, dtype=torch.float32)
                neutral[0] = end_pixel
                neutral[-1] = start_pixel
                # Anchor the bridge's head to the stitched main video's own tail (mask=0, hard pixel
                # copy), exactly the same mechanism that already makes ordinary inter-cycle joins
                # seamless -- rather than generating the bridge fully independently and relying only
                # on a post-hoc pixel crossfade to paper over two unrelated generations. That
                # independent-then-crossfade approach is what produced the limb-doubling ghosting:
                # crossfade_videos() blends two genuinely different frames pixel-for-pixel whenever
                # they don't already agree, and nothing before this made them agree. anchor_start
                # (below) still wins on frame 0 specifically, pinning it to end_image.
                prev_tail = result[-self.cycle_crossfade :] if self.cycle_crossfade > 0 else None
                bridge_pixels = run_cycle(
                    control_segment=neutral,
                    length=bridge_frames,
                    crossfade=self.cycle_crossfade,
                    cycle_seed=self.seed + len(temporal_tiles) * 10007,
                    progress_desc="Wan VACE pose+depth generate (loop bridge)",
                    anchor_start=end_pixel,
                    anchor_end=start_pixel,
                    reference_pixels=reference_pixels_list,
                )
                combined = crossfade_videos(result, bridge_pixels, self.cycle_crossfade)
                if self.cycle_crossfade > 0:
                    # crossfade_videos(a, b, n) with a and b both exactly n frames long returns just
                    # the n-frame blend (a's non-overlap prefix and b's non-overlap suffix are both
                    # empty), so this is already the closed seam -- no further slicing needed.
                    closed_head = crossfade_videos(
                        combined[-self.cycle_crossfade :], combined[: self.cycle_crossfade], self.cycle_crossfade
                    )
                    combined[: self.cycle_crossfade] = closed_head
                    combined = combined[: -self.cycle_crossfade]
                result = combined

        return self._encode_and_save(context, result)

    def _load_conditioning(
        self, context: InvocationContext, cond_field: WanConditioningField, *, device: torch.device, dtype: torch.dtype
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

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_wan_vace_pose_depth_", suffix=".mp4", delete=False)
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
