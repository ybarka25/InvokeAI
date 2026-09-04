"""Depth control-video generator for Wan 2.2 VACE video-to-video conditioning.

Applies a Depth Anything model to every frame of an input video, producing a depth-map
control video suitable for `wan_vace_video_encode` (directly, or via `vace_control_layer` +
`vace_control_blend` to combine with other control modalities). The model is loaded once
and reused across all frames rather than per frame.
"""

from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames
from invokeai.backend.image_util.depth_anything.depth_anything_pipeline import DepthAnythingPipeline

DEPTH_ANYTHING_MODEL_SIZES = Literal["large", "base", "small", "small_v2"]
DEPTH_ANYTHING_MODELS = {
    "large": "LiheYoung/depth-anything-large-hf",
    "base": "LiheYoung/depth-anything-base-hf",
    "small": "LiheYoung/depth-anything-small-hf",
    "small_v2": "depth-anything/Depth-Anything-V2-Small-hf",
}


@invocation(
    "video_depth_anything_depth_estimation",
    title="Depth Anything Depth Estimation - Video",
    tags=["controlnet", "depth", "depth anything", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoDepthAnythingDepthEstimationInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a depth-map control video using a Depth Anything model, frame by frame."""

    video: VideoField = InputField(description="The video to process")
    model_size: DEPTH_ANYTHING_MODEL_SIZES = InputField(
        default="small_v2", description="The size of the depth model to use"
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)
        model_url = DEPTH_ANYTHING_MODELS[self.model_size]
        loaded_model = context.models.load_remote_model(model_url, DepthAnythingPipeline.load_model)

        with loaded_model as depth_anything_detector:
            assert isinstance(depth_anything_detector, DepthAnythingPipeline)

            def process(frame):
                return depth_anything_detector.generate_depth(frame)

            tmp_path, width, height, fps, num_frames = process_video_frames(
                context, video_path, process, "Depth estimation"
            )

        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
