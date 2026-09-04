"""Video motion captioning via a vision-language model with native video (temporal) understanding
-- for extracting a motion description from a source video to graft onto a reference photo's
caption (see the "Photo Prompt Builder" callable workflow), so the resulting prompt generalizes to
any video's motion rather than a hand-picked verb.
"""

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.invocations.vlm_caption import VLM_CAPTION_MODEL_IDS, VlmCaptionModelKey, _load_vlm_caption
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice

DEFAULT_MOTION_INSTRUCTION = (
    "Describe only the motion and action performed by the main subject in this video clip, in one "
    "short phrase (e.g. 'walking briskly forward', 'spinning around while raising both arms', "
    "'sitting down and leaning back'). Do not describe appearance, clothing, setting, or camera -- "
    "only the physical action/movement."
)


@invocation(
    "vlm_video_motion_caption",
    title="VLM Video Motion Caption",
    tags=["prompt", "caption", "vlm", "vision-language", "qwen", "video", "motion"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VlmVideoMotionCaptionInvocation(BaseInvocation):
    """Describes the motion/action happening in a video clip using a vision-language model with
    native video understanding (temporal frame sampling), not independent per-frame captioning."""

    video: VideoField = InputField(description="The video to describe the motion of.")
    model: VlmCaptionModelKey = InputField(
        default="qwen2.5-vl-3b-instruct", description="The vision-language model to use."
    )
    instruction: str = InputField(
        default=DEFAULT_MOTION_INSTRUCTION,
        description="The captioning instruction given to the model.",
    )
    num_frames: int = InputField(
        default=12, gt=0, description="Number of evenly-spaced frames to sample from the video."
    )
    max_new_tokens: int = InputField(default=256, gt=0, description="Maximum tokens to generate.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        video_path = context.videos.get_path(self.video.video_name)

        context.util.signal_progress(f"Loading {self.model}")
        pipeline = _load_vlm_caption(VLM_CAPTION_MODEL_IDS[self.model])
        device = TorchDevice.choose_torch_device()
        try:
            pipeline.to(device=device)
            context.util.signal_progress("Describing motion")
            caption = pipeline.caption_video(
                video_path=video_path,
                instruction=self.instruction,
                max_new_tokens=self.max_new_tokens,
                num_frames=self.num_frames,
            )
        finally:
            pipeline.to(device=torch.device("cpu"))
            del pipeline
            TorchDevice.empty_cache()

        return StringOutput(value=caption)
