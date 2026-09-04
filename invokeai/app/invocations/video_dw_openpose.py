"""Pose control-video generator for Wan 2.2 VACE video-to-video conditioning.

Applies DWPose to every frame of an input video, producing a pose-skeleton control video
suitable for `wan_vace_video_encode` (directly, or via `vace_control_layer` +
`vace_control_blend` to combine with other control modalities). The ONNX sessions are
loaded once and reused across all frames rather than per frame.
"""

import onnxruntime as ort
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_frame_processing import process_video_frames
from invokeai.backend.image_util.dw_openpose import DWOpenposeDetector

# VACE's own pose annotator resizes so the short side is 1024px (rounded to a multiple of 64)
# before detecting/drawing, then resizes the result back down. Detecting/drawing at native
# resolution (e.g. 480p) instead draws the same absolute stickwidth=4/radius=4 lines VACE would
# draw on a much larger canvas, so at typical output resolutions the skeleton ends up
# proportionally thicker and aliased rather than the thin, anti-aliased lines the model was
# trained to recognize.
_POSE_DETECTION_SHORT_SIDE = 1024
_POSE_DETECTION_MULTIPLE = 64


@invocation(
    "video_dw_openpose_detection",
    title="DW Openpose Detection - Video",
    tags=["controlnet", "dwpose", "openpose", "video", "vace"],
    category="controlnet_preprocessors",
    version="1.2.0",
    classification=Classification.Prototype,
)
class VideoDWOpenposeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an openpose control video from a video using DWPose, frame by frame."""

    video: VideoField = InputField(description="The video to process")
    draw_body: bool = InputField(default=True)
    draw_face: bool = InputField(
        default=True,
        description="VACE's own pose annotator (PoseBodyFaceAnnotator) draws body+face by "
        "default -- leaving this off emits a sparser control signal than what the model was "
        "trained against (no head orientation/gaze cue).",
    )
    draw_hands: bool = InputField(default=False)

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video_path = context.videos.get_path(self.video.video_name)

        onnx_det_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_det())
        onnx_pose_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_pose())
        loaded_session_det = context.models.load_local_model(
            onnx_det_path, DWOpenposeDetector.create_onnx_inference_session
        )
        loaded_session_pose = context.models.load_local_model(
            onnx_pose_path, DWOpenposeDetector.create_onnx_inference_session
        )

        with loaded_session_det as session_det, loaded_session_pose as session_pose:
            assert isinstance(session_det, ort.InferenceSession)
            assert isinstance(session_pose, ort.InferenceSession)
            detector = DWOpenposeDetector(session_det=session_det, session_pose=session_pose)

            def process(frame):
                original_size = frame.size  # (W, H)
                src_w, src_h = original_size
                scale = _POSE_DETECTION_SHORT_SIDE / min(src_w, src_h)
                w = max(
                    round(src_w * scale / _POSE_DETECTION_MULTIPLE) * _POSE_DETECTION_MULTIPLE, _POSE_DETECTION_MULTIPLE
                )
                h = max(
                    round(src_h * scale / _POSE_DETECTION_MULTIPLE) * _POSE_DETECTION_MULTIPLE, _POSE_DETECTION_MULTIPLE
                )
                upsized = frame.resize((w, h), Image.LANCZOS)
                drawn = detector.run(
                    upsized, draw_face=self.draw_face, draw_hands=self.draw_hands, draw_body=self.draw_body
                )
                return drawn.resize(original_size, Image.LANCZOS)

            tmp_path, width, height, fps, num_frames = process_video_frames(
                context, video_path, process, "Pose detection"
            )

        try:
            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        return VideoOutput.build(video_dto, num_frames=num_frames)
