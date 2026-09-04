"""Optical-flow control-video generator for Wan 2.2 VACE video-to-video conditioning.

Encodes per-frame motion (camera movement and moving elements in the frame) as an HSV
color-wheel flow map, using OpenCV's Farneback dense optical flow between consecutive
frames -- no model download required. Useful as a control layer (alone or blended with
Canny/Depth/etc. via `vace_control_layer` + `vace_control_blend`) when the motion itself,
not just per-frame structure, needs to be guided.

Frames are processed strictly in decode order via a closure holding the previous frame, so
this relies on `process_video_frames` calling the processor once per frame, in sequence.
"""

import cv2
import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames


def _flow_to_color(flow: np.ndarray, magnitude_scale: float) -> np.ndarray:
    """Converts a dense optical-flow field (H, W, 2) to an HSV-wheel RGB visualization."""
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # hue: direction (OpenCV hue range is 0-180)
    hsv[..., 1] = 255  # full saturation
    hsv[..., 2] = np.clip(magnitude * magnitude_scale, 0, 255).astype(np.uint8)  # value: speed
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


@invocation(
    "video_optical_flow_detection",
    title="Optical Flow (Motion) - Video",
    tags=["controlnet", "optical-flow", "motion", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoOpticalFlowDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a motion control video via dense optical flow (Farneback), frame by frame.

    Encodes the direction of motion as hue and its speed as brightness, so camera pans and
    moving subjects both show up as colored motion vectors. The first output frame has no
    prior frame to compare against, so it's emitted as flat black (zero motion).
    """

    video: VideoField = InputField(description="The video to process")
    magnitude_scale: float = InputField(
        default=15.0,
        ge=0.1,
        description="Multiplier applied to flow magnitude before clamping to the 0-255 brightness range. "
        "Raise for subtle/slow motion, lower if fast motion is clipping to solid white.",
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)

        prev_gray: list[np.ndarray | None] = [None]

        def process(frame: Image.Image) -> Image.Image:
            gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
            if prev_gray[0] is None:
                out = np.zeros((*gray.shape, 3), dtype=np.uint8)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray[0], gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                out = _flow_to_color(flow, self.magnitude_scale)
            prev_gray[0] = gray
            return Image.fromarray(out)

        tmp_path, width, height, fps, num_frames = process_video_frames(
            context, video_path, process, "Optical flow detection"
        )
        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
