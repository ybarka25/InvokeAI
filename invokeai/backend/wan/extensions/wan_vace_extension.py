"""Wan 2.2 VACE (video-to-video) control-video conditioning.

VACE conditions on a control video by VAE-encoding it into a 96-channel
tensor that ``WanVACETransformer3DModel`` consumes as ``control_hidden_states``:
32 channels of "inactive"/"reactive" latents (the same control video encoded
twice, split by a region mask) plus 64 channels of packed mask.

An optional per-frame region mask enables VACE's masked editing mode (mirrors
ComfyUI's VACE mask input): white/1.0 = "reactive" (this region is generated
fresh, guided by the control video), black/0.0 = "inactive" (this region is
kept as-is from the control video, not regenerated). With no mask (``mask is
None``), every pixel is reactive and none is inactive — the original
full-frame-guidance behavior. Mirrors diffusers'
``WanVACEPipeline.prepare_video_latents`` / ``prepare_masks`` — see
``pipeline_wan_vace.py`` lines 453-457, 514-579, 581-631, 922-924.

Reference images (subject/character identity conditioning) are supported: each
is VAE-encoded as a single frame, paired with an all-zero "reactive" half
(``[ref_as_inactive, zeros_as_reactive]``, mirroring ``prepare_video_latents``
lines 567-578), and prepended as extra leading latent frames — with the
matching mask frames forced to zero (``prepare_masks`` lines 626-629), since a
reference frame is pure identity anchor, not a region to generate. The caller
(``wan_vace_denoise``) is responsible for growing the noise latents by the
same count and stripping them back off the output before VAE-decoding
(mirrors ``pipeline_wan_vace.py`` lines 932-942, 1023-1024).
"""

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

# TEMPORARY diagnostic switch -- prints per-VACE-layer hint norms (reference-token slice vs
# control-video-token slice, raw and after position-wise scaling) to stdout. Toggle off/remove
# once the reference_conditioning_scale numeric-effect question is settled.
_DEBUG_POSITIONAL_SCALE = False

# Wan VACE packs an 8x8 pixel mask block into one channel group per the VAE's
# 8x spatial compression (matches diffusers' `prepare_masks` block-unshuffle).
_WAN_VACE_MASK_CHANNELS = 64
_VAE_SPATIAL_SCALE = 8


def _pack_mask_to_latent_channels(
    mask_pixel: torch.Tensor, t_lat: int, h_lat: int, w_lat: int
) -> torch.Tensor:
    """Block-unshuffles a pixel-space mask into the 64-channel packed latent mask.

    ``mask_pixel`` is ``[T, H, W]`` in ``[0, 1]``, where ``H``/``W`` are multiples of
    ``_VAE_SPATIAL_SCALE`` (guaranteed by the encode node's ``multiple_of=16`` width/height
    fields, since Wan's transformer patch size is 2 -> 2*8=16). Returns ``[64, t_lat, h_lat,
    w_lat]``, matching diffusers' ``prepare_masks`` (an 8x8 pixel block becomes 64 channels of
    one latent pixel, then nearest-exact-resized to the actual latent shape).
    """
    num_frames, height, width = mask_pixel.shape
    new_h, new_w = height // _VAE_SPATIAL_SCALE, width // _VAE_SPATIAL_SCALE
    packed = mask_pixel.view(num_frames, new_h, _VAE_SPATIAL_SCALE, new_w, _VAE_SPATIAL_SCALE)
    packed = packed.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [64, T, new_h, new_w]
    packed = F.interpolate(
        packed.unsqueeze(0), size=(t_lat, h_lat, w_lat), mode="nearest-exact"
    ).squeeze(0)
    return packed


def encode_control_video_to_vace_condition(
    video: torch.Tensor,
    vae: AutoencoderKLWan,
    device: torch.device,
    dtype: torch.dtype,
    reference_images: list[torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the 96-channel VACE control condition tensor.

    ``video`` must already be normalised to [-1, 1], shape ``[1, 3, T, H, W]``.
    ``reference_images``, if given, are each ``[1, 3, 1, H, W]`` in [-1, 1] at
    the same H/W as ``video``. ``mask``, if given, is ``[1, 1, T, H, W]`` in
    ``[0, 1]`` at the same T/H/W as ``video``: 1.0 = regenerate this region
    (reactive), 0.0 = keep the control video as-is here (inactive). ``None``
    means fully reactive everywhere (no masked editing). Returns
    ``[1, 96, len(reference_images) + T_lat, H // 8, W // 8]``.

    To scale the reference-image branch's influence independently of the control-video branch's,
    see ``run_wan_vace_transformer_with_positional_scale`` below -- it scales the two branches'
    contribution to the output *after* they've both been jointly processed by the VACE blocks
    (position-wise on the resulting hint tensor), rather than pre-scaling the reference latents
    here before they're encoded. Pre-scaling here would change what the model reads as the
    reference image's actual appearance (its magnitude in latent space *is* the image), not how
    strongly that appearance gets injected into the output -- the wrong lever for "trust the
    reference more/less" (further explanation in that function's docstring).
    """
    vae_dtype = next(iter(vae.parameters())).dtype
    video = video.to(device=device, dtype=vae_dtype)

    with torch.inference_mode():
        if mask is not None:
            mask_pixel = mask.to(device=device, dtype=vae_dtype)  # [1, 1, T, H, W] in [0, 1]
            mask_binary = (mask_pixel > 0.5).to(dtype=vae_dtype)
            inactive = video * (1.0 - mask_binary)
            reactive = video * mask_binary
        else:
            inactive = torch.zeros_like(video)
            reactive = video

        inactive_latents = vae.encode(inactive, return_dict=False)[0].mode()
        reactive_latents = vae.encode(reactive, return_dict=False)[0].mode()

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(inactive_latents.device, inactive_latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(inactive_latents.device, inactive_latents.dtype)
        )
        inactive_latents = (inactive_latents - latents_mean) / latents_std
        reactive_latents = (reactive_latents - latents_mean) / latents_std

        latents = torch.cat([inactive_latents, reactive_latents], dim=1)  # [1, 32, T_lat, H_lat, W_lat]
        _, _, t_lat_video, h_lat, w_lat = latents.shape

        # Pack the mask to T_lat_video (the control-video's own latent length) BEFORE any
        # reference-frame prepending -- official `vace_encode_masks` / diffusers `prepare_masks` /
        # ComfyUI all interpolate the mask against the video's frame count first, then prepend
        # num_ref zero frames second. Doing it the other way around (interpolating against
        # num_ref + T_lat, as this used to) resamples the mask against the wrong frame count,
        # time-shifting/squashing it and making the anchor/crossfade mask=0 frames land on the
        # wrong latent positions.
        if mask is not None:
            # Continuous (unbinarized) mask, per diffusers' `prepare_masks` -- the packed
            # channels carry the raw [0, 1] mask, only the inactive/reactive video split above
            # is thresholded.
            mask_latent = _pack_mask_to_latent_channels(mask_pixel[0, 0], t_lat_video, h_lat, w_lat)
            mask_latent = mask_latent.unsqueeze(0).to(device=latents.device, dtype=latents.dtype)
        else:
            mask_latent = torch.ones(
                1, _WAN_VACE_MASK_CHANNELS, t_lat_video, h_lat, w_lat, device=latents.device, dtype=latents.dtype
            )

        num_ref = len(reference_images) if reference_images else 0
        if num_ref:
            ref_latents = []
            for ref in reference_images:
                ref = ref.to(device=device, dtype=vae_dtype)  # [1, 3, 1, H, W]
                ref_latent = vae.encode(ref, return_dict=False)[0].mode()  # [1, 16, 1, H_lat, W_lat]
                ref_latent = (ref_latent - latents_mean) / latents_std
                # [ref_as_inactive, zeros_as_reactive]: the reference frame is
                # given as-is (inactive/keep half); nothing is asked of the
                # model to regenerate for it (reactive half is zero).
                ref_latent = torch.cat([ref_latent, torch.zeros_like(ref_latent)], dim=1)  # [1, 32, 1, H_lat, W_lat]
                ref_latents.append(ref_latent)
            latents = torch.cat([*ref_latents, latents], dim=2)  # prepend along the temporal dim
            mask_pad = torch.zeros_like(mask_latent[:, :, :num_ref])
            mask_latent = torch.cat([mask_pad, mask_latent], dim=2)

        conditioning = torch.cat([latents, mask_latent], dim=1)  # [1, 96, T_lat, H_lat, W_lat]

    return conditioning.to(dtype=dtype)


def run_wan_vace_transformer_with_positional_scale(
    transformer: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_hidden_states: torch.Tensor,
    num_reference_tokens: int,
    reference_scale: float,
    control_scale: float,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
):
    """Re-implements ``WanVACETransformer3DModel.forward``'s inference path, except the VACE hint
    is scaled *position-wise* instead of by one scalar-per-layer for the whole tensor: the leading
    ``num_reference_tokens`` positions (from ``encode_control_video_to_vace_condition``'s prepended
    reference-image frames) get ``reference_scale``, every other position (the control-video/pose
    branch, plus any zero padding) gets ``control_scale``.

    Why this exists: diffusers' packaged forward only accepts ``control_hidden_states_scale``, one
    scalar per VACE layer applied uniformly across the *entire* hint tensor at that layer --
    reference and control-video frames are concatenated into one sequence long before that scale
    is applied, so it cannot tell them apart. ComfyUI's own VACE model supports genuinely separate
    control contexts (a list, each with its own strength) for exactly this kind of case, which is
    the ComfyUI-side feature this was modeled after -- but its contexts are also fully independent
    sequences, each only self-/cross-attending within itself. Reusing that literally here (two
    separate calls through vace_blocks) would stop the pose/depth frames' tokens from ever
    attending to the reference tokens inside the VACE blocks' own self-attention, which is exactly
    the mechanism this fork relies on to keep identity legible on frames other than the anchored
    one (see ``encode_control_video_to_vace_condition``'s module docstring). Keeping both branches
    in one sequence (so that attention still happens) and only splitting the *injection strength*
    position-wise at the final residual-add gets independent strength control without that
    regression.

    ``num_reference_tokens`` is ``num_ref * (H_lat // patch_h) * (W_lat // patch_w)`` -- the reference
    frames are prepended along the temporal axis before patchify, and ``patch_embedding``'s
    ``flatten(2)`` walks the patched tensor in (T, H, W) order, so they land at the head of the
    flattened token sequence contiguously.
    """
    batch_size, _num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    vace_layers = transformer.config.vace_layers

    rotary_emb = transformer.rope(hidden_states)

    hidden_states = transformer.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)
    seq_len = hidden_states.size(1)

    control_hidden_states = transformer.vace_patch_embedding(control_hidden_states)
    control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
    if control_hidden_states.size(1) < seq_len:
        pad = control_hidden_states.new_zeros(
            batch_size, seq_len - control_hidden_states.size(1), control_hidden_states.size(2)
        )
        control_hidden_states = torch.cat([control_hidden_states, pad], dim=1)

    position_scale = control_hidden_states.new_full((1, seq_len, 1), control_scale)
    if num_reference_tokens > 0:
        position_scale[:, :num_reference_tokens, :] = reference_scale

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    control_hidden_states_list = []
    for block in transformer.vace_blocks:
        conditioning_states, control_hidden_states = block(
            hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
        )
        control_hidden_states_list.append(conditioning_states)
    control_hidden_states_list = control_hidden_states_list[::-1]

    for i, block in enumerate(transformer.blocks):
        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        if i in vace_layers:
            control_hint = control_hidden_states_list.pop()
            scaled_hint = control_hint.to(hidden_states.device) * position_scale.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            if _DEBUG_POSITIONAL_SCALE and num_reference_tokens > 0:
                with torch.no_grad():
                    ref_raw = control_hint[:, :num_reference_tokens].float().norm()
                    pose_raw = control_hint[:, num_reference_tokens:].float().norm()
                    ref_scaled = scaled_hint[:, :num_reference_tokens].float().norm()
                    pose_scaled = scaled_hint[:, num_reference_tokens:].float().norm()
                    hs_norm = hidden_states.float().norm()
                    with open("O:/InvokeAI-3/vace_debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            f"layer={i} ref_scale={reference_scale} pose_scale={control_scale} "
                            f"hidden_states_norm={hs_norm:.2f} ref_hint(raw={ref_raw:.2f} scaled={ref_scaled:.2f}) "
                            f"pose_hint(raw={pose_raw:.2f} scaled={pose_scaled:.2f})\n"
                        )
            hidden_states = hidden_states + scaled_hint

    shift, scale = (transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = transformer.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
