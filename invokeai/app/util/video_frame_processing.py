"""Shared per-frame video processing for VACE control-video preprocessor nodes.

Streams frames from the input video through a caller-supplied PIL-to-PIL processor and
encodes the result to a temp MP4, mirroring video_concat's stream-through-encoder pattern
so peak memory stays O(1) frames regardless of clip length. Callers load their detector
model once and pass a closure over it, rather than this helper reloading per frame.
"""

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video

# Every caller of process_video_frames is, per this module's purpose, a VACE control-video
# preprocessor -- its output gets VAE-encoded and fed back into VACE, not watched directly. The
# default libx264 CRF (~23) visibly smears/desaturates thin, saturated lines (pose skeletons)
# before the VAE ever sees them; a low CRF here costs some disk space on an intermediate file for
# meaningfully better control fidelity.
_CONTROL_VIDEO_CRF = 15


def process_video_frames(
    context: InvocationContext,
    video_path: Path,
    processor: Callable[[Image.Image], Image.Image],
    progress_label: str,
) -> tuple[Path, int, int, float, int]:
    """Applies `processor` to every decoded frame of `video_path`.

    Returns (tmp_mp4_path, width, height, fps, num_frames). The caller is responsible for
    handing tmp_mp4_path to context.videos.save(...) and deleting it afterward.
    """
    width, height, _duration, fps = probe_video(video_path)
    if width % 2 or height % 2:
        raise ValueError(
            f"Input video is {width}x{height}; H.264 encoding requires even dimensions. "
            "Re-encode or crop the source to even width and height first."
        )
    if not fps or fps <= 0:
        fps = 16.0

    tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_proc_", suffix=".mp4", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    writer = make_mp4_writer(tmp_path, fps, crf=_CONTROL_VIDEO_CRF)
    num_frames = 0
    try:
        for np_frame in iter_video_frames(video_path, is_canceled=context.util.is_canceled):
            if num_frames % 8 == 0:
                context.util.signal_progress(f"{progress_label} (frame {num_frames})")
            pil_frame = Image.fromarray(np_frame).convert("RGB")
            out_frame = processor(pil_frame).convert("RGB")
            if out_frame.size != (width, height):
                out_frame = out_frame.resize((width, height), Image.LANCZOS)
            writer.append_data(np.array(out_frame))
            num_frames += 1
    finally:
        writer.close()

    if num_frames == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Video {video_path} decoded to zero frames.")

    return tmp_path, width, height, fps, num_frames
