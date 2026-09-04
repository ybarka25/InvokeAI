"""HED (softedge) control-video generator for Wan 2.2 VACE video-to-video conditioning.

Applies the HED softedge model to every frame of an input video. The model is loaded once
and reused across all frames rather than per frame.
"""

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames
from invokeai.backend.image_util.hed import ControlNetHED_Apache2, HEDEdgeDetector


@invocation(
    "video_hed_edge_detection",
    title="HED Edge Detection - Video",
    tags=["controlnet", "hed", "softedge", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoHEDEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a softedge control video using the HED model, frame by frame."""

    video: VideoField = InputField(description="The video to process")
    scribble: bool = InputField(default=False, description=FieldDescriptions.scribble_mode)

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)
        loaded_model = context.models.load_remote_model(HEDEdgeDetector.get_model_url(), HEDEdgeDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, ControlNetHED_Apache2)
            hed_processor = HEDEdgeDetector(model)

            def process(frame):
                return hed_processor.run(image=frame, scribble=self.scribble)

            tmp_path, width, height, fps, num_frames = process_video_frames(
                context, video_path, process, "HED edge detection"
            )

        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
