"""Vision-language image captioning, for building a text prompt that describes a reference photo
(e.g. to feed into wan_text_encoder alongside a motion description from a source video -- see the
"Photo Prompt Builder" callable workflow).

Uses Qwen2.5-VL (natively supported by transformers, instruction-following) rather than Florence-2:
Florence-2's trust_remote_code modeling files are unmaintained since ~2024 and incompatible with
this fork's transformers version at multiple points (generation-config attrs, attention dispatch,
tied-weights format).
"""

from typing import Literal

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.vlm_caption.vlm_caption_pipeline import VlmCaptionPipeline
from invokeai.backend.util.devices import TorchDevice

VlmCaptionModelKey = Literal["qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct"]
VLM_CAPTION_MODEL_IDS: dict[VlmCaptionModelKey, str] = {
    "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

DEFAULT_CAPTION_INSTRUCTION = (
    "Describe this image in a single detailed paragraph, as a prompt for an image/video generation model. "
    "Focus on the subject's appearance (face, hair, clothing, body pose), the setting, lighting, and camera "
    "framing. Do not mention that it is a photo, image, or painting -- describe it as if describing a video."
)


def _load_vlm_caption(repo_id: str) -> VlmCaptionPipeline:
    # Loaded directly via from_pretrained (not context.models.load_remote_model) since this fork's
    # downloader doesn't participate in caching multi-file/processor-heavy repos beyond weights --
    # transformers' own HF Hub cache handles that. Acceptable for an occasional captioning utility
    # node (see wan_character_align.py's grounding_dino.py for the pattern this deviates from).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16)
    # Qwen2.5-VL tokenizes images at native resolution by default -- a full-size reference photo
    # (e.g. several thousand px wide) blows up attention memory. Cap total pixels; the model only
    # needs enough detail to describe appearance/pose, not fine texture.
    processor = AutoProcessor.from_pretrained(repo_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model.eval()
    return VlmCaptionPipeline(model, processor)


@invocation(
    "vlm_caption",
    title="VLM Caption",
    tags=["prompt", "caption", "vlm", "vision-language", "qwen"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VlmCaptionInvocation(BaseInvocation):
    """Generates a text caption for an image using a vision-language model, following a free-form instruction."""

    image: ImageField = InputField(description="The image to caption.")
    model: VlmCaptionModelKey = InputField(
        default="qwen2.5-vl-3b-instruct", description="The vision-language model to use."
    )
    instruction: str = InputField(
        default=DEFAULT_CAPTION_INSTRUCTION,
        description="The captioning instruction given to the model.",
    )
    max_new_tokens: int = InputField(default=512, gt=0, description="Maximum tokens to generate.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        image = context.images.get_pil(self.image.image_name, mode="RGB")

        context.util.signal_progress(f"Loading {self.model}")
        pipeline = _load_vlm_caption(VLM_CAPTION_MODEL_IDS[self.model])
        device = TorchDevice.choose_torch_device()
        try:
            pipeline.to(device=device)
            context.util.signal_progress("Captioning image")
            caption = pipeline.caption(image=image, instruction=self.instruction, max_new_tokens=self.max_new_tokens)
        finally:
            pipeline.to(device=torch.device("cpu"))
            del pipeline
            TorchDevice.empty_cache()

        return StringOutput(value=caption)
