"""MLSD line-segment control-video generator for Wan 2.2 VACE video-to-video conditioning.

Applies MLSD line-segment detection to every frame of an input video. The model is loaded
once and reused across all frames rather than per frame.
"""

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames
from invokeai.backend.image_util.mlsd import MLSDDetector
from invokeai.backend.image_util.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large


@invocation(
    "video_mlsd_detection",
    title="MLSD Detection - Video",
    tags=["controlnet", "mlsd", "edge", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoMLSDDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a line-segment control video using MLSD, frame by frame."""

    video: VideoField = InputField(description="The video to process")
    score_threshold: float = InputField(
        default=0.1, ge=0, description="The threshold used to score points when determining line segments"
    )
    distance_threshold: float = InputField(
        default=20.0,
        ge=0,
        description="Threshold for including a line segment - lines shorter than this distance will be discarded",
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)
        loaded_model = context.models.load_remote_model(MLSDDetector.get_model_url(), MLSDDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, MobileV2_MLSD_Large)
            detector = MLSDDetector(model)

            def process(frame):
                return detector.run(frame, self.score_threshold, self.distance_threshold)

            tmp_path, width, height, fps, num_frames = process_video_frames(
                context, video_path, process, "MLSD detection"
            )

        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
