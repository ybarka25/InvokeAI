"""Composites several VACE control-video layers (e.g. canny + depth + pose) into one control video.

The video-decode subsystem only allows one concurrent streaming decode at a time
(``_VIDEO_STREAM_SLOTS`` in ``video_thumbnails.py`` is a capacity-1 semaphore), so layers
cannot be streamed interleaved the way ``video_concat`` streams a single video -- opening a
second layer's decoder while the first is still open deadlocks waiting for that slot. Instead,
each layer is fully decoded to memory one at a time (respecting the 1-slot limit), then blended
in memory. Peak memory is O(num_layers * clip_length) frames, not O(num_layers).
"""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VaceControlLayerField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video


@invocation(
    "vace_control_blend",
    title="Blend Control Video Layers - VACE",
    tags=["video", "conditioning", "wan", "vace", "v2v", "compositing"],
    category="conditioning",
    version="1.2.0",
    classification=Classification.Prototype,
)
class VaceControlBlendInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Composites several control-video layers into one control video for `wan_vace_video_encode`.

    Layers are painted in ascending `order` (ties keep their original wiring position, so a
    duplicate/omitted order still degrades to a stable, close-to-intended result rather than
    an arbitrary shuffle). Each layer is composited over the accumulator with a screen blend
    (`1 - (1-acc)*(1-frame)`), interpolated toward the accumulator by `1 - strength`: a
    strength-1.0 layer fully screens in, strength-0.0 leaves the accumulator untouched. Screen
    blending never darkens what's already there (unlike a straight opacity blend, which dims a
    bright layer -- e.g. a depth map -- wherever a darker layer -- e.g. a mostly-black pose
    skeleton frame -- is composited over it, pushing the combined image out of the brightness
    range either control type was trained on individually).

    All layers must share pixel dimensions. A layer with fewer decoded frames than the
    longest one has its last frame held to pad it out, matching `wan_vace_video_encode`'s
    own padding convention.
    """

    layers: list[VaceControlLayerField] = InputField(
        min_length=1,
        description="Control-video layers to composite. Order/strength come from each vace_control_layer node.",
    )
    fps: Optional[int] = InputField(
        default=None, ge=1, le=120, description="Output frame rate. Defaults to the first layer's fps."
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        if not self.layers:
            raise ValueError("vace_control_blend requires at least one layer.")

        # Stable sort: ties keep their original list position (the order layers were wired
        # in), so a duplicate/omitted `order` degrades gracefully instead of shuffling.
        layers = [layer for _, layer in sorted(enumerate(self.layers), key=lambda pair: (pair[1].order, pair[0]))]

        paths: list[Path] = [context.videos.get_path(layer.video.video_name) for layer in layers]
        probes = [probe_video(p) for p in paths]
        dims = {(w, h) for (w, h, _, _) in probes}
        if len(dims) > 1:
            raise ValueError(
                f"All control-video layers must share the same dimensions. Got: {sorted(dims)}. "
                "Resize the source videos to a single resolution before blending."
            )
        width, height, _, _ = probes[0]
        if width % 2 or height % 2:
            raise ValueError(f"Control-video layers are {width}x{height}; H.264 encoding requires even dimensions.")
        output_fps = float(self.fps) if self.fps is not None else next((fps for (_, _, _, fps) in probes if fps), 16.0)

        context.util.signal_progress(f"Blending {len(layers)} VACE control layer(s)")

        # Decode one layer fully at a time -- see the module docstring for why these can't be
        # streamed interleaved.
        per_layer_frames: list[list[np.ndarray]] = []
        for i, path in enumerate(paths):
            layer_frames = [
                np.ascontiguousarray(frame) for frame in iter_video_frames(path, is_canceled=context.util.is_canceled)
            ]
            if not layer_frames:
                raise ValueError(f"Control-video layer {i} decoded to zero frames.")
            per_layer_frames.append(layer_frames)
        max_frame_count = max(len(lf) for lf in per_layer_frames)

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_vace_blend_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            writer = make_mp4_writer(tmp_path, output_fps, crf=15)
            num_frames = 0
            try:
                for i in range(max_frame_count):
                    if context.util.is_canceled is not None and context.util.is_canceled():
                        raise CanceledException
                    acc = np.zeros((height, width, 3), dtype=np.float32)
                    for layer_frames, layer in zip(per_layer_frames, layers, strict=False):
                        frame = layer_frames[i] if i < len(layer_frames) else layer_frames[-1]
                        strength = float(layer.strength)
                        frame_f = frame.astype(np.float32)
                        screened = 255.0 - (255.0 - acc) * (255.0 - frame_f) / 255.0
                        acc = acc * (1.0 - strength) + screened * strength
                    writer.append_data(np.clip(acc, 0, 255).astype(np.uint8))
                    num_frames += 1
            finally:
                writer.close()

            if num_frames == 0:
                raise ValueError("Blending produced zero output frames.")

            video_dto = context.videos.save(
                source_path=tmp_path, width=width, height=height, duration=num_frames / output_fps, fps=output_fps
            )
            return VideoOutput.build(video_dto, num_frames=num_frames)
        finally:
            tmp_path.unlink(missing_ok=True)
