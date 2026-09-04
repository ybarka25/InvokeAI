"""Align a photo and a video's first frame so their detected character shares the same
on-screen position and scale.

Meant to run before ``wan_vace_loop_prep``: feed its ``aligned_photo``/``aligned_video``
outputs into that node's ``start_image``/``video`` inputs. Without this, a start-image anchor
and the pose/depth control video can disagree wildly on where the character is and how big
they are on screen -- the model then gets an identity anchor at one scale and a motion guide
at another, which was one of this session's suspects for weak identity transfer.

Uses Grounding DINO (already implemented in this fork, ``app/invocations/grounding_dino.py``)
for person detection -- no mask needed, just a bounding box per image.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import pipeline
from transformers.pipelines import ZeroShotObjectDetectionPipeline

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import ImageField, InputField, VideoField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import CharacterAlignOutput
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
from invokeai.backend.image_util.grounding_dino.detection_result import DetectionResult
from invokeai.backend.image_util.grounding_dino.grounding_dino_pipeline import GroundingDinoPipeline

GroundingDinoModelKey = Literal["grounding-dino-tiny", "grounding-dino-base"]
GROUNDING_DINO_MODEL_IDS: dict[GroundingDinoModelKey, str] = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}


def _load_grounding_dino(model_path: Path) -> GroundingDinoPipeline:
    grounding_dino_pipeline = pipeline(model=str(model_path), task="zero-shot-object-detection", local_files_only=True)
    assert isinstance(grounding_dino_pipeline, ZeroShotObjectDetectionPipeline)
    return GroundingDinoPipeline(grounding_dino_pipeline)


def _detect_best_box(
    context: InvocationContext,
    image: Image.Image,
    prompt: str,
    threshold: float,
    model: GroundingDinoModelKey,
) -> DetectionResult:
    label = prompt if prompt.endswith(".") else prompt + "."
    with context.models.load_remote_model(
        source=GROUNDING_DINO_MODEL_IDS[model], loader=_load_grounding_dino
    ) as detector:
        assert isinstance(detector, GroundingDinoPipeline)
        detections = detector.detect(image=image, candidate_labels=[label], threshold=threshold)
    if not detections:
        raise ValueError(
            f"Grounding DINO found no match for {label!r} (threshold={threshold}) in a "
            f"{image.width}x{image.height} image. Lower detection_threshold or adjust detect_prompt."
        )
    return max(detections, key=lambda d: d.score)


def _crop_box_for_match(
    a_w: int,
    a_h: int,
    a_box: DetectionResult,
    r_w: int,
    r_h: int,
    r_box: DetectionResult,
    scale_reference: Literal["height", "width"],
) -> tuple[float, float, float, float]:
    """Returns (left, top, right, bottom) on ``A`` (in ``A``'s own pixel coords, may extend
    past its bounds) such that resizing that crop to R's aspect ratio places A's detected
    character at the same relative position and the same relative size as R's character is
    within R -- not just centered on it.
    """
    a_cx = (a_box.box.xmin + a_box.box.xmax) / 2.0
    a_cy = (a_box.box.ymin + a_box.box.ymax) / 2.0
    r_cx = (r_box.box.xmin + r_box.box.xmax) / 2.0
    r_cy = (r_box.box.ymin + r_box.box.ymax) / 2.0

    if scale_reference == "height":
        a_size = max(a_box.box.ymax - a_box.box.ymin, 1.0)
        r_size = max(r_box.box.ymax - r_box.box.ymin, 1.0)
        crop_h = a_size * r_h / r_size
        crop_w = crop_h * r_w / r_h
    else:
        a_size = max(a_box.box.xmax - a_box.box.xmin, 1.0)
        r_size = max(r_box.box.xmax - r_box.box.xmin, 1.0)
        crop_w = a_size * r_w / r_size
        crop_h = crop_w * r_h / r_w

    left = a_cx - (r_cx / r_w) * crop_w
    top = a_cy - (r_cy / r_h) * crop_h
    return left, top, left + crop_w, top + crop_h


def _crop_with_padding(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crops ``image`` to ``box`` (left, top, right, bottom), padding with black wherever the
    box extends past the source image's bounds -- preserves the exact position/scale computed
    by ``_crop_box_for_match`` instead of clamping (which would silently break the match)."""
    left, top, right, bottom = (int(round(v)) for v in box)
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.width)
    pad_bottom = max(0, bottom - image.height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top
    return image.crop((left, top, right, bottom))


@invocation(
    "wan_character_align",
    title="Character Align (Photo <-> Video)",
    tags=["image", "video", "grounding-dino", "align", "crop"],
    category="video",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanCharacterAlignInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Detects the character in a photo and in a video's first frame (Grounding DINO), then
    crops/pads/resizes whichever side isn't the ratio source so both characters land at the
    same on-screen position and the same scale.
    """

    photo: ImageField = InputField(description="Reference photo of the character.")
    video: VideoField = InputField(description="Source video; only its first frame is used for detection.")

    ratio_source: Literal["photo", "video"] = InputField(
        default="photo",
        description="Which side's aspect ratio (and resolution) is kept as-is. The other side is "
        "cropped/padded/resized to match its character's on-screen position and scale.",
    )
    detect_prompt: str = InputField(
        default="person.", description="Grounding DINO text prompt for the character (lowercase, end with '.')."
    )
    detection_model: GroundingDinoModelKey = InputField(
        default="grounding-dino-tiny", description="Grounding DINO model size."
    )
    detection_threshold: float = InputField(
        default=0.3, ge=0.0, le=1.0, description="Grounding DINO detection threshold."
    )
    scale_reference: Literal["height", "width"] = InputField(
        default="height",
        description="Which bounding-box dimension is matched for zoom level. Height suits full-body/portrait shots.",
    )
    target_resolution: WanTargetResolution = InputField(
        default="720p",
        description="Short-side resolution preset for the shared output canvas.",
        ui_choice_labels=WAN_TARGET_RESOLUTION_LABELS,
    )
    rounding: WanRounding = InputField(
        default="nearest", description="How to snap the resolution to the 16px Wan pixel grid."
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> CharacterAlignOutput:
        photo_pil = context.images.get_pil(self.photo.image_name, mode="RGB")

        video_path = context.videos.get_path(self.video.video_name)
        frames: list[np.ndarray] = []
        for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
            frames.append(np_frame)
            break
        if not frames:
            raise ValueError(f"Video {self.video.video_name} decoded to zero frames.")
        first_frame_pil = Image.fromarray(frames[0]).convert("RGB")

        photo_box = _detect_best_box(
            context, photo_pil, self.detect_prompt, self.detection_threshold, self.detection_model
        )
        video_box = _detect_best_box(
            context, first_frame_pil, self.detect_prompt, self.detection_threshold, self.detection_model
        )

        if self.ratio_source == "photo":
            r_w, r_h, r_box = photo_pil.width, photo_pil.height, photo_box
            a_w, a_h, a_box = first_frame_pil.width, first_frame_pil.height, video_box
        else:
            r_w, r_h, r_box = first_frame_pil.width, first_frame_pil.height, video_box
            a_w, a_h, a_box = photo_pil.width, photo_pil.height, photo_box

        out_w, out_h = _scale_and_snap(
            r_w, r_h, WAN_TARGET_RESOLUTION_PX[self.target_resolution], self.rounding, multiple=WAN_I2V_PIXEL_MULTIPLE
        )
        crop_box = _crop_box_for_match(a_w, a_h, a_box, r_w, r_h, r_box, self.scale_reference)

        if self.ratio_source == "photo":
            aligned_photo_pil = photo_pil.resize((out_w, out_h), Image.LANCZOS)
            video_crop_box: Optional[tuple[float, float, float, float]] = crop_box  # applied to every frame below
        else:
            aligned_photo_pil = _crop_with_padding(photo_pil, crop_box).resize((out_w, out_h), Image.LANCZOS)
            video_crop_box = None

        _vw, _vh, _duration, probed_fps = probe_video(video_path)
        fps = probed_fps or 16.0

        # Re-decode the full video (the first pass above only needed frame 0 for detection).
        aligned_frames: list[np.ndarray] = []
        for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
            frame_pil = Image.fromarray(np_frame).convert("RGB")
            if video_crop_box is not None:
                frame_pil = _crop_with_padding(frame_pil, video_crop_box)
            frame_pil = frame_pil.resize((out_w, out_h), Image.LANCZOS)
            aligned_frames.append(np.asarray(frame_pil, dtype=np.uint8))

        import tempfile

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_character_align_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            writer = make_mp4_writer(tmp_path, fps)
            try:
                for frame in aligned_frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=out_w,
                height=out_h,
                duration=len(aligned_frames) / fps,
                fps=fps,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        photo_dto = context.images.save(image=aligned_photo_pil)

        return CharacterAlignOutput(
            aligned_photo=ImageField(image_name=photo_dto.image_name),
            aligned_video=VideoField(video_name=video_dto.video_name),
            width=out_w,
            height=out_h,
        )
