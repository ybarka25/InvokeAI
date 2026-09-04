"""Prep node for ``wan_vace_pose_depth_generate``.

Trims a source video, resolves one output aspect ratio (from the video or from the start
anchor image), and center-crops everything on the detected person so the video and the
start/end anchor images all land on one consistent canvas before pose/depth extraction.

The video always uses a single, fixed crop window computed once from a representative frame
-- not recomputed per frame -- so the apparent camera framing doesn't drift across the clip.
Each anchor image gets its own independently computed person-centered crop, since they are
unrelated shots and only need to share the final aspect ratio, not an absolute crop box.
"""

import tempfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import ImageField, InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import WanVaceLoopPrepOutput
from invokeai.app.invocations.wan_ideal_dimensions import (
    WAN_I2V_PIXEL_MULTIPLE,
    WAN_TARGET_RESOLUTION_LABELS,
    WAN_TARGET_RESOLUTION_PX,
    WanRounding,
    WanTargetResolution,
    _scale_and_snap,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video
from invokeai.backend.image_util.dw_openpose import DWOpenposeDetector

_BBoxT = tuple[float, float, float, float]


def _person_bbox(np_image: np.ndarray, detector: DWOpenposeDetector, score_threshold: float = 0.3) -> Optional[_BBoxT]:
    """Returns (left, top, right, bottom) in pixels for the highest-confidence keypoints, or
    None if nothing scores above ``score_threshold`` (falls back to a plain center crop)."""
    keypoints, scores = detector.pose_estimation(np_image)
    visible = scores > score_threshold
    if not visible.any():
        return None
    pts = keypoints[visible]
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _centered_crop_box(image_w: int, image_h: int, bbox: Optional[_BBoxT], target_ratio: float) -> tuple[int, int, int, int]:
    """Expands ``bbox`` (with margin) to ``target_ratio`` (w/h), centered, clamped to the image."""
    if bbox is None:
        cx, cy, bw, bh = image_w / 2.0, image_h / 2.0, float(image_w), float(image_h)
    else:
        left, top, right, bottom = bbox
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        bw, bh = max(right - left, 1.0), max(bottom - top, 1.0)

    # Margin around the raw keypoint bbox so the crop isn't skin-tight on the person.
    margin = 1.6
    bw = max(bw * margin, image_w * 0.1)
    bh = max(bh * margin, image_h * 0.1)

    if bw / bh > target_ratio:
        bh = bw / target_ratio
    else:
        bw = bh * target_ratio

    # Shrink (preserving ratio) if the box doesn't fit in the source image.
    scale = min(image_w / bw, image_h / bh, 1.0)
    bw *= scale
    bh *= scale

    left = max(0.0, min(cx - bw / 2.0, image_w - bw))
    top = max(0.0, min(cy - bh / 2.0, image_h - bh))
    return int(round(left)), int(round(top)), int(round(left + bw)), int(round(top + bh))


@invocation(
    "wan_vace_loop_prep",
    title="VACE Loop Prep - Wan 2.2",
    tags=["video", "wan", "vace", "pose", "depth", "prep"],
    category="video",
    version="1.2.0",
    classification=Classification.Prototype,
)
class WanVaceLoopPrepInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Trims/crops/resizes a source video (and optional start/end anchor images) onto one
    consistent canvas, ready for DWPose/Depth extraction and ``wan_vace_pose_depth_generate``.
    """

    video: VideoField = InputField(description="Source video to trim/crop/resize.")
    start_image: Optional[ImageField] = InputField(default=None, description="Optional start anchor image.")
    end_image: Optional[ImageField] = InputField(default=None, description="Optional end anchor image.")

    num_source_frames: int = InputField(default=49, ge=1, description="Number of frames to keep from the source video.")
    source_start_frame_index: int = InputField(
        default=0, ge=0, description="Frame index in the source video to start trimming from."
    )

    ratio_source: Literal["video", "start_image"] = InputField(
        default="video",
        description="Which input's aspect ratio to keep. 'start_image' requires start_image to be set; the "
        "video is then cropped (person-centered) to match it instead of the other way around.",
    )
    target_resolution: WanTargetResolution = InputField(
        default="720p",
        description="Short-side resolution preset (same convention as Wan Ideal Dimensions).",
        ui_choice_labels=WAN_TARGET_RESOLUTION_LABELS,
    )
    rounding: WanRounding = InputField(
        default="nearest", description="How to snap the computed resolution to the 16px Wan pixel grid."
    )

    def invoke(self, context: InvocationContext) -> WanVaceLoopPrepOutput:
        if self.ratio_source == "start_image" and self.start_image is None:
            raise ValueError("ratio_source='start_image' requires a start_image input.")

        video_path = context.videos.get_path(self.video.video_name)
        src_w, src_h, _duration, fps = probe_video(video_path)
        fps = fps or 16.0

        frames: list[np.ndarray] = []
        for i, np_frame in enumerate(iter_video_frames(video_path, is_canceled=context.util.is_canceled)):
            if i < self.source_start_frame_index:
                continue
            frames.append(np_frame)
            if len(frames) >= self.num_source_frames:
                break
        if len(frames) < self.num_source_frames:
            raise ValueError(
                f"Video {self.video.video_name} only yielded {len(frames)} frame(s) from index "
                f"{self.source_start_frame_index}; requested num_source_frames={self.num_source_frames}."
            )

        start_pil = context.images.get_pil(self.start_image.image_name, "RGB") if self.start_image is not None else None
        end_pil = context.images.get_pil(self.end_image.image_name, "RGB") if self.end_image is not None else None

        onnx_det_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_det())
        onnx_pose_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_pose())
        loaded_det = context.models.load_local_model(onnx_det_path, DWOpenposeDetector.create_onnx_inference_session)
        loaded_pose = context.models.load_local_model(onnx_pose_path, DWOpenposeDetector.create_onnx_inference_session)

        with loaded_det as session_det, loaded_pose as session_pose:
            assert isinstance(session_det, ort.InferenceSession)
            assert isinstance(session_pose, ort.InferenceSession)
            detector = DWOpenposeDetector(session_det=session_det, session_pose=session_pose)

            if self.ratio_source == "start_image":
                assert start_pil is not None
                target_w, target_h = start_pil.size
                cropped_start = start_pil
                video_bbox = _person_bbox(frames[0], detector)
                video_crop = _centered_crop_box(src_w, src_h, video_bbox, target_w / target_h)
                cropped_frames = [np.array(Image.fromarray(f).crop(video_crop)) for f in frames]
            else:
                target_w, target_h = src_w, src_h
                cropped_start = None
                cropped_frames = frames

            if self.ratio_source != "start_image" and start_pil is not None:
                start_bbox = _person_bbox(np.array(start_pil), detector)
                start_crop = _centered_crop_box(start_pil.width, start_pil.height, start_bbox, target_w / target_h)
                cropped_start = start_pil.crop(start_crop)

            cropped_end = None
            if end_pil is not None:
                end_bbox = _person_bbox(np.array(end_pil), detector)
                end_crop = _centered_crop_box(end_pil.width, end_pil.height, end_bbox, target_w / target_h)
                cropped_end = end_pil.crop(end_crop)

        out_w, out_h = _scale_and_snap(
            target_w,
            target_h,
            WAN_TARGET_RESOLUTION_PX[self.target_resolution],
            self.rounding,
            multiple=WAN_I2V_PIXEL_MULTIPLE,
        )

        resized_frames = [np.array(Image.fromarray(f).resize((out_w, out_h), Image.LANCZOS)) for f in cropped_frames]
        resized_start = cropped_start.resize((out_w, out_h), Image.LANCZOS) if cropped_start is not None else None
        resized_end = cropped_end.resize((out_w, out_h), Image.LANCZOS) if cropped_end is not None else None

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_vace_loop_prep_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            writer = make_mp4_writer(tmp_path, fps, crf=15)
            try:
                for frame in resized_frames:
                    writer.append_data(np.ascontiguousarray(frame))
            finally:
                writer.close()
            video_dto = context.videos.save(
                source_path=tmp_path, width=out_w, height=out_h, duration=len(resized_frames) / fps, fps=fps
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        start_field = None
        start_video_field = None
        if resized_start is not None:
            start_dto = context.images.save(image=resized_start)
            start_field = ImageField(image_name=start_dto.image_name)

            # Depth (unlike pose) also encodes the scene/background layout, not just the body --
            # extracting it from the source video pulls in that video's own viewpoint and pulls
            # the generation away from the reference image's composition. Holding start_image
            # static for the full clip and running depth on *this* instead keeps the viewpoint/
            # scene anchored to the reference image while pose (still from the real video) is
            # the only thing carrying motion.
            start_frame_np = np.ascontiguousarray(np.array(resized_start))
            tmp_start = tempfile.NamedTemporaryFile(prefix="invokeai_vace_loop_prep_start_", suffix=".mp4", delete=False)
            tmp_start.close()
            tmp_start_path = Path(tmp_start.name)
            try:
                writer = make_mp4_writer(tmp_start_path, fps, crf=15)
                try:
                    for _ in range(len(resized_frames)):
                        writer.append_data(start_frame_np)
                finally:
                    writer.close()
                start_video_dto = context.videos.save(
                    source_path=tmp_start_path,
                    width=out_w,
                    height=out_h,
                    duration=len(resized_frames) / fps,
                    fps=fps,
                )
                start_video_field = VideoField(video_name=start_video_dto.video_name)
            finally:
                tmp_start_path.unlink(missing_ok=True)

        end_field = None
        if resized_end is not None:
            end_dto = context.images.save(image=resized_end)
            end_field = ImageField(image_name=end_dto.image_name)

        return WanVaceLoopPrepOutput(
            video=VideoField(video_name=video_dto.video_name),
            start_image=start_field,
            end_image=end_field,
            start_image_video=start_video_field,
            width=out_w,
            height=out_h,
            num_frames=len(resized_frames),
        )
