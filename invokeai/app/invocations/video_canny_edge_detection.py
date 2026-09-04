"""Canny edge control-video generator for Wan 2.2 VACE video-to-video conditioning.

Applies cv2's Canny edge detector to every frame of an input video, producing an edge-map
control video suitable for `wan_vace_video_encode` (directly, or via `vace_control_layer` +
`vace_control_blend` to combine with other control modalities).
"""

import cv2

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames
from invokeai.backend.image_util.util import cv2_to_pil, pil_to_cv2


@invocation(
    "video_canny_edge_detection",
    title="Canny Edge Detection - Video",
    tags=["controlnet", "canny", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoCannyEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an edge-map control video using cv2's Canny algorithm, frame by frame."""

    video: VideoField = InputField(description="The video to process")
    low_threshold: int = InputField(
        default=100, ge=0, le=255, description="The low threshold of the Canny pixel gradient (0-255)"
    )
    high_threshold: int = InputField(
        default=200, ge=0, le=255, description="The high threshold of the Canny pixel gradient (0-255)"
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)

        def process(frame):
            edge_map = cv2.Canny(pil_to_cv2(frame), self.low_threshold, self.high_threshold)
            return cv2_to_pil(edge_map)

        tmp_path, width, height, fps, num_frames = process_video_frames(
            context, video_path, process, "Canny edge detection"
        )
        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
