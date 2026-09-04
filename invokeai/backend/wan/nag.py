"""NAG (Normalized Attention Guidance) for Wan transformer cross-attention.

Reference: "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
(Chen et al.), ComfyUI-NAG (`ChenDarYen/ComfyUI-NAG`), vendored inside the "SuperUltimateVaceTools"
ComfyUI pack the user's own known-working VACE workflow uses at ``cfg=1.0``.

Unlike classifier-free guidance, NAG doesn't blend two full model forward passes' noise
predictions -- it blends the *raw cross-attention output* (post softmax@V, pre the attention
block's output projection) computed against the positive and negative text embeddings, inside
every cross-attention module. That makes it a per-block, in-place replacement for a plain
single-conditioning cross-attention call, not a change to the outer denoise loop's CFG math --
which is why it still works when nothing else in the model provides negative guidance (e.g. a
distilled few-step config running at ``cfg=1.0``, where the classic cond/uncond dual forward pass
is normally skipped entirely to save compute).

Applied to every cross-attention module in the transformer, main blocks AND VACE blocks alike
(ComfyUI's ``VaceWanAttentionBlock`` subclasses the same attention block class the main transformer
uses and gets patched identically -- there is nothing VACE-specific about it).
"""

from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import _get_qkv_projections


class _NagCrossAttnProcessor:
    """Drop-in replacement for a ``WanAttention`` cross-attention module's ``.processor``.

    Computes the raw attention output against both ``encoder_hidden_states`` (positive, passed
    in at call time same as the original processor) and ``negative_encoder_hidden_states`` (fixed
    for the lifetime of this processor instance), then combines them via the NAG formula before
    the block's own output projection (``to_out``) is applied.
    """

    def __init__(
        self,
        negative_encoder_hidden_states: torch.Tensor,
        nag_scale: float,
        nag_tau: float,
        nag_alpha: float,
    ):
        self.negative_encoder_hidden_states = negative_encoder_hidden_states
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    @staticmethod
    def _raw_cross_attn(attn, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Everything WanAttnProcessor does up to (not including) ``attn.to_out``."""
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        out = dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.flatten(2, 3).type_as(query)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        if attn.add_k_proj is not None:
            # I2V's image-context-token splitting isn't handled here -- fall back to plain
            # (unguided) cross-attention rather than silently mis-computing NAG against it.
            z_positive = self._raw_cross_attn(attn, hidden_states, encoder_hidden_states)
            hidden_states = attn.to_out[0](z_positive)
            return attn.to_out[1](hidden_states)

        z_positive = self._raw_cross_attn(attn, hidden_states, encoder_hidden_states)
        z_negative = self._raw_cross_attn(
            attn, hidden_states, self.negative_encoder_hidden_states.to(dtype=hidden_states.dtype)
        )

        scale = self.nag_scale
        z_guidance = z_positive * scale - z_negative * (scale - 1)

        norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True).expand_as(z_positive).clamp(min=1e-8)
        norm_guidance = torch.norm(z_guidance, p=1, dim=-1, keepdim=True).expand_as(z_guidance)
        ratio = norm_guidance / norm_positive
        z_guidance = z_guidance * torch.minimum(ratio, ratio.new_full((), float(self.nag_tau))) / ratio.clamp(min=1e-8)
        z_guidance = z_guidance * self.nag_alpha + z_positive * (1.0 - self.nag_alpha)

        out = attn.to_out[0](z_guidance)
        return attn.to_out[1](out)


@contextmanager
def apply_nag(
    transformer,
    negative_encoder_hidden_states: torch.Tensor,
    nag_scale: float = 5.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
) -> Iterator[None]:
    """Temporarily replace every cross-attention module's processor (main blocks + VACE blocks,
    if present) with a NAG-applying one for the duration of the context.

    ``negative_encoder_hidden_states`` must already be projected to the transformer's inner_dim
    (i.e. already run through ``transformer.condition_embedder.text_embedder``) -- cross-attention's
    ``to_k``/``to_v`` are sized for that projected width, not the raw text encoder's own output
    width. The positive path gets this projection for free inside the normal forward (the
    transformer's own ``condition_embedder`` call projects ``encoder_hidden_states`` before the
    blocks loop even starts); the negative embeddings passed here never go through that same
    forward, so the caller has to do it explicitly first.
    """
    patched: list[tuple] = []
    blocks = list(getattr(transformer, "blocks", []))
    blocks += list(getattr(transformer, "vace_blocks", []))
    for block in blocks:
        attn2 = getattr(block, "attn2", None)
        if attn2 is None:
            continue
        patched.append((attn2, attn2.processor))
        attn2.processor = _NagCrossAttnProcessor(negative_encoder_hidden_states, nag_scale, nag_tau, nag_alpha)
    try:
        yield
    finally:
        for attn2, original_processor in patched:
            attn2.processor = original_processor
