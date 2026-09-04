"""Wan 2.2 VACE (video-to-video) denoise invocation.

Sibling to :mod:`wan_video_denoise` for VACE-Fun-A14B: same flow-matching
schedule + dual-expert MoE swap logic, but every step also feeds the
VACE control branch a control-video conditioning tensor (built by
``wan_vace_video_encode``) via ``control_hidden_states`` /
``control_hidden_states_scale``. Unlike I2V's reference-image conditioning,
the control video itself is not concatenated into the noise latents —
``WanVACETransformer3DModel`` routes it through its own ``vace_blocks`` and
adds the result into the main blocks at ``config.vace_layers``.

If the paired ``wan_vace_video_encode`` node was given reference images
(subject/character identity conditioning), its condition tensor carries extra
leading latent frames for them. This node grows the noise latents by the same
count so the shapes line up, denoises jointly with the rest of the video, and
strips those leading frames back off before returning — mirroring
``WanVACEPipeline.__call__``'s own reference-image handling.

Kept as a separate file rather than parameterizing ``WanVideoDenoiseInvocation``
so the working T2V/I2V paths are not risked by the VACE work; the shared bits
(expert swapper, scheduler construction, conditioning loading, LoRA iteration)
live in ``wan_denoise`` and are imported here, same as ``wan_video_denoise``.

No mask-based region editing yet — see the VACE plan's explicitly-out-of-scope
section and ``wan_vace_extension.py``.
"""

from contextlib import ExitStack
from typing import Callable, Iterable, Optional

import torch
from diffusers.models.attention_dispatch import attention_backend
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    WanConditioningField,
    WanVaceConditioningField,
)
from invokeai.app.invocations.model import WanTransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.wan_denoise import (
    WanDenoiseInvocation,
    _ExpertSwapper,
    _get_wan_transformer_working_mem_bytes,
    _resolve_variant,
    _validate_spatial_dimensions,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.patches.layer_patcher import PatchSpec
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.extensions.wan_vace_extension import run_wan_vace_transformer_with_positional_scale
from invokeai.backend.wan.memory_optimization import wan_memory_optimization
from invokeai.backend.wan.nag import apply_nag
from invokeai.backend.wan.sampling_utils import (
    get_default_latent_channels,
    get_spatial_scale_factor,
    make_noise,
    num_latent_frames_for,
)


def run_wan_vace_denoise_loop(
    *,
    context: InvocationContext,
    transformer_field: WanTransformerField,
    scheduler,
    latents: torch.Tensor,
    control_hidden_states: torch.Tensor,
    pos_cond: WanConditioningInfo,
    neg_cond: Optional[WanConditioningInfo],
    guidance_scale: float,
    guidance_scale_low_noise: Optional[float],
    conditioning_scale: float,
    device: torch.device,
    inference_dtype: torch.dtype,
    step_callback: Callable[[PipelineIntermediateState], None],
    t_lat: int,
    num_ref: int = 0,
    reference_conditioning_scale: float = 1.0,
    swapper: Optional[_ExpertSwapper] = None,
    progress_desc: str = "Denoising Wan 2.2 VACE video",
    start_step_idx: int = 0,
    use_nag: bool = False,
    nag_scale: float = 5.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
) -> torch.Tensor:
    """Run the VACE timestep loop (dual-expert MoE swap, CFG, control conditioning).

    Shared by ``WanVaceDenoiseInvocation`` and the tiled VACE upscaler so both drive the
    identical denoise loop. When ``swapper`` is omitted, one is built and closed internally
    (the single-pass node's behavior); the tiled upscaler instead passes one long-lived
    ``_ExpertSwapper`` shared across every tile so repeated tile-local calls hit the model
    cache instead of reloading/unloading experts per tile.

    ``start_step_idx`` resumes the schedule partway through (img2img-style partial denoise)
    instead of starting from the first, highest-noise step. It slices the *local* view of
    ``scheduler.timesteps`` only -- the scheduler's own ``.timesteps``/``.sigmas`` attributes
    are left untouched, so ``scheduler.step()``'s internal ``index_for_timestep`` lookup still
    resolves each yielded ``t`` to its true position in the full schedule and applies the
    correct sigma bracket. Callers combine ``latents`` (init latents + scaled noise, using
    ``scheduler.sigmas[start_step_idx]`` as the mix weight) accordingly before calling this.
    """
    timesteps = scheduler.timesteps[start_step_idx:]
    total_steps = len(timesteps)
    if total_steps <= 0:
        return latents

    high_model = transformer_field.transformer
    low_model = transformer_field.transformer_low_noise
    low_config = context.models.get_config(low_model) if low_model is not None else None
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    boundary_timestep = transformer_field.boundary_ratio * num_train_timesteps if low_model is not None else None

    high_loras = transformer_field.loras
    low_loras = transformer_field.loras_low_noise
    high_config = context.models.get_config(high_model)
    high_is_quantized = high_config.format == ModelFormat.GGUFQuantized
    low_is_quantized = low_config.format == ModelFormat.GGUFQuantized if low_config is not None else False

    proxy = WanDenoiseInvocation.model_construct(transformer=transformer_field)

    def high_lora_factory() -> Iterable[PatchSpec]:
        return proxy._lora_iterator(context, high_loras)

    def low_lora_factory() -> Iterable[PatchSpec]:
        return proxy._lora_iterator(context, low_loras)

    max_resident_gb = context.config.get().wan_max_resident_transformer_gb
    working_mem_bytes = _get_wan_transformer_working_mem_bytes(device, max_resident_gb=max_resident_gb)
    max_resident_bytes = int(max_resident_gb * (2**30)) if working_mem_bytes is not None else None

    with ExitStack() as exit_stack:
        wanted_attention_backend = context.config.get().wan_attention_backend
        if wanted_attention_backend != "native":
            try:
                exit_stack.enter_context(attention_backend(wanted_attention_backend))
            except Exception as exc:
                context.logger.warning(
                    f"{wanted_attention_backend!r} attention backend unavailable, falling back to native SDPA: {exc}"
                )

        owns_swapper = swapper is None
        if owns_swapper:
            swapper = _ExpertSwapper(
                context=context,
                high_model=high_model,
                low_model=low_model,
                inference_dtype=inference_dtype,
                high_lora_factory=high_lora_factory if high_loras else None,
                low_lora_factory=low_lora_factory if low_loras else None,
                high_is_quantized=high_is_quantized,
                low_is_quantized=low_is_quantized,
                working_mem_bytes=working_mem_bytes,
                max_resident_model_bytes=max_resident_bytes,
            )
            exit_stack.callback(swapper.close)

        for step_idx, t in enumerate(tqdm(timesteps, desc=f"{progress_desc} ({t_lat} latent frames)", total=total_steps)):
            if low_model is not None and float(t) < float(boundary_timestep):
                active_label = _ExpertSwapper.LOW
                low_cfg = guidance_scale_low_noise
                active_cfg = low_cfg if (low_cfg is not None and low_cfg >= 1.0) else guidance_scale
            else:
                active_label = _ExpertSwapper.HIGH
                active_cfg = guidance_scale

            transformer = swapper.get(active_label)

            p_h, p_w = transformer.config.patch_size[1:]
            tokens_per_frame = (control_hidden_states.shape[-2] // p_h) * (control_hidden_states.shape[-1] // p_w)
            num_reference_tokens = num_ref * tokens_per_frame

            latent_model_input = latents.to(dtype=inference_dtype)
            timestep = t.expand(latents.shape[0])

            with wan_memory_optimization(transformer, enabled=True):
                if use_nag:
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
                            num_reference_tokens=num_reference_tokens,
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
                        num_reference_tokens=num_reference_tokens,
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
                            num_reference_tokens=num_reference_tokens,
                            reference_scale=reference_conditioning_scale,
                            control_scale=conditioning_scale,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred_uncond + active_cfg * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            step_callback(
                PipelineIntermediateState(
                    step=step_idx + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t.item()),
                    latents=latents[:, :, (num_ref + t_lat) // 2],
                )
            )

    return latents


@invocation(
    "wan_vace_denoise",
    title="Denoise Video (VACE) - Wan 2.2",
    tags=["video", "wan", "vace", "v2v"],
    category="latents",
    version="1.1.1",
    classification=Classification.Prototype,
)
class WanVaceDenoiseInvocation(BaseInvocation):
    """Run the Wan 2.2 VACE denoising loop, guided by a control video.

    The output is a 5D ``[1, C, T_lat, H/8, W/8]`` latent tensor ready for
    :class:`WanLatentsToVideoInvocation` to VAE-decode and encode as MP4.

    Requires a VACE-capable transformer (e.g. Wan2.2-VACE-Fun-A14B). Load it
    the same way as T2V/I2V A14B: dual-expert (high + low noise) via
    ``wan_model_loader``.
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
    vace_condition: WanVaceConditioningField = InputField(
        description=FieldDescriptions.wan_vace_condition,
        input=Input.Connection,
        title="VACE Condition",
    )
    conditioning_scale: float = InputField(
        default=1.0,
        ge=0.0,
        description="Strength of the control-video guidance. 1.0 matches the Wan reference; lower "
        "values loosen adherence to the control video's structure.",
        title="Conditioning Scale",
    )

    guidance_scale: float = InputField(
        default=5.0,
        ge=1.0,
        description="Classifier-free guidance scale. Wan 2.2 video reference uses 5.0 for the "
        "high-noise expert and 4.0 for the low-noise expert.",
        title="Guidance Scale",
    )
    guidance_scale_low_noise: Optional[float] = InputField(
        default=4.0,
        ge=0.0,
        description="Optional separate CFG scale for the low-noise expert (Wan 2.2 A14B only). "
        "Values below 1.0 fall back to the primary 'Guidance Scale'.",
        title="Guidance Scale (Low Noise)",
    )

    # Wan transformer patch_size=(1, 2, 2) × VAE spatial 8x => H/W multiple of 16.
    width: int = InputField(default=832, gt=0, multiple_of=16, description="Width of the generated video.")
    height: int = InputField(default=480, gt=0, multiple_of=16, description="Height of the generated video.")
    num_frames: int = InputField(
        default=81,
        ge=5,
        description="Number of output frames. Must satisfy (num_frames - 1) %% 4 == 0 so the latent "
        "temporal dim divides cleanly. Must match the paired wan_vace_video_encode node.",
        title="Number of Frames",
    )
    steps: int = InputField(default=40, gt=0, description="Number of denoising steps.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        name = context.tensors.save(tensor=latents)
        from invokeai.app.invocations.fields import LatentsField

        return LatentsOutput(
            latents=LatentsField(latents_name=name, seed=self.seed),
            width=self.width,
            height=self.height,
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        if (self.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (num_frames - 1) %% 4 == 0 for the Wan VAE's temporal "
                f"compression (got {self.num_frames}). Try 5, 9, 13, ..., 81, 85, ..."
            )

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        variant = _resolve_variant(context, self.transformer)
        if variant not in (WanVariantType.VACE, WanVariantType.VACE_2_1):
            raise ValueError(
                f"wan_vace_denoise requires a VACE transformer (e.g. Wan2.2-VACE-Fun-A14B or Wan 2.1 "
                f"VACE-14B). The selected transformer is {variant.value!r}. Use wan_video_denoise for "
                f"T2V/I2V instead."
            )
        _validate_spatial_dimensions(variant, self.width, self.height)
        spatial_scale = get_spatial_scale_factor(variant)

        if self.vace_condition.width != self.width or self.vace_condition.height != self.height:
            raise ValueError(
                f"VACE condition dimensions ({self.vace_condition.width}x{self.vace_condition.height}) "
                f"must match denoise dimensions ({self.width}x{self.height})."
            )
        if self.vace_condition.num_frames != self.num_frames:
            raise ValueError(
                f"VACE condition num_frames ({self.vace_condition.num_frames}) must match denoise "
                f"num_frames ({self.num_frames}). Re-run the Control Video - Wan 2.2 VACE node with "
                f"num_frames={self.num_frames}."
            )

        scheduler_builder = WanDenoiseInvocation._build_scheduler
        proxy = WanDenoiseInvocation.model_construct(
            transformer=self.transformer,
            positive_conditioning=self.positive_conditioning,
        )
        scheduler = scheduler_builder(proxy, context, device)

        pos_cond = self._load_conditioning(context, self.positive_conditioning, device=device, dtype=inference_dtype)
        low_cfg_enabled = (
            self.transformer.transformer_low_noise is not None
            and self.guidance_scale_low_noise is not None
            and self.guidance_scale_low_noise >= 1.0
            and self.guidance_scale_low_noise != 1.0
        )
        do_cfg = self.negative_conditioning is not None and (self.guidance_scale != 1.0 or low_cfg_enabled)
        neg_cond: WanConditioningInfo | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_cond = self._load_conditioning(
                context, self.negative_conditioning, device=device, dtype=inference_dtype
            )

        control_hidden_states = context.tensors.load(self.vace_condition.condition_tensor_name).to(
            device=device, dtype=inference_dtype
        )

        scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
        timesteps = scheduler.timesteps
        total_steps = len(timesteps)

        latent_dtype = torch.float32
        latent_channels = get_default_latent_channels(variant)
        t_lat = num_latent_frames_for(self.num_frames)
        num_ref = self.vace_condition.num_reference_images

        latents = make_noise(
            batch_size=1,
            latent_channels=latent_channels,
            height=self.height,
            width=self.width,
            spatial_scale_factor=spatial_scale,
            device=device,
            dtype=latent_dtype,
            seed=self.seed,
            num_latent_frames=t_lat + num_ref,
        )

        if control_hidden_states.shape[-3:] != latents.shape[-3:]:
            raise ValueError(
                f"VACE condition spatial/temporal shape {tuple(control_hidden_states.shape[-3:])} does "
                f"not match the denoise latent shape {tuple(latents.shape[-3:])}. This should not happen "
                "if width/height/num_frames match the encode node — check for a stale cached tensor."
            )

        if total_steps <= 0:
            return latents[:, :, num_ref:] if num_ref else latents

        step_callback = self._build_step_callback(context)

        max_resident_gb = context.config.get().wan_max_resident_transformer_gb
        if _get_wan_transformer_working_mem_bytes(device, max_resident_gb=max_resident_gb) is not None:
            context.logger.info(
                f"Wan memory optimization: targeting about {max_resident_gb:g} GiB of resident transformer "
                "weights when partial loading is available"
            )

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
            num_ref=num_ref,
            progress_desc=f"Denoising Wan 2.2 VACE video ({self.num_frames} frames)",
        )

        # Reference-image latent frames were prepended as identity anchors for
        # the VACE control branch (see wan_vace_extension.py); they aren't part
        # of the generated video, so strip them before returning (mirrors
        # WanVACEPipeline.__call__'s `latents[:, :, num_reference_images:]`).
        if num_ref:
            latents = latents[:, :, num_ref:]
        return latents

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

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Wan)

        return step_callback
