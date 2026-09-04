"""Plain text-in/text-out generation using the same vision-language model as vlm_caption/
vlm_video_motion_caption -- used to rewrite a photo's caption with a video's extracted motion
grafted in, instead of a literal (and fragile) string search/replace.
"""

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.invocations.vlm_caption import VLM_CAPTION_MODEL_IDS, VlmCaptionModelKey, _load_vlm_caption
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "vlm_text_complete",
    title="VLM Text Complete",
    tags=["prompt", "text", "vlm", "qwen", "rewrite"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VlmTextCompleteInvocation(BaseInvocation):
    """Runs a text-only prompt through a vision-language model's language backbone (no image/video)."""

    prompt: str = InputField(description="The fully-assembled text prompt to complete.", ui_component=UIComponent.Textarea)
    model: VlmCaptionModelKey = InputField(
        default="qwen2.5-vl-3b-instruct", description="The vision-language model to use."
    )
    max_new_tokens: int = InputField(default=512, gt=0, description="Maximum tokens to generate.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        context.util.signal_progress(f"Loading {self.model}")
        pipeline = _load_vlm_caption(VLM_CAPTION_MODEL_IDS[self.model])
        device = TorchDevice.choose_torch_device()
        try:
            pipeline.to(device=device)
            context.util.signal_progress("Generating text")
            output = pipeline.complete_text(prompt=self.prompt, max_new_tokens=self.max_new_tokens)
        finally:
            pipeline.to(device=torch.device("cpu"))
            del pipeline
            TorchDevice.empty_cache()

        return StringOutput(value=output)
