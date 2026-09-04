"""Shared MP4 writer construction for video-producing invocations.

All video nodes encode through this helper so the encoder settings stay in one
place. libx264 + yuv420p (imageio's defaults for the FFMPEG plugin) give
broadly-compatible browser playback. ``macro_block_size=1`` is load-bearing:
imageio's default of 16 makes ffmpeg silently *rescale* frames to the next
multiple of 16 (e.g. 1920x1080 -> 1920x1088), which desynchronizes the encoded
file from the dimensions recorded in the video DTO and breaks same-dimension
checks downstream (e.g. concatenating a trimmed clip with its source).

yuv420p requires even dimensions; callers validate that before encoding.
"""

from pathlib import Path
from typing import Optional

import imageio.v2 as iio2


def make_mp4_writer(path: Path | str, fps: float, crf: Optional[int] = None):
    """Returns an imageio FFMPEG writer that preserves frame dimensions exactly.

    ``crf`` (libx264's constant-rate-factor, lower = higher quality/larger file, typical range
    0-51) overrides the default (~23, "medium" quality) when given. Control-video intermediates
    that get VAE-encoded and fed back into VACE (pose skeletons, depth maps) are a bad case for
    the default: thin, fully-saturated lines are exactly what 4:2:0 chroma subsampling + default
    quantization smears/desaturates worst, and that damage happens before the VAE ever sees the
    frame. Final deliverable videos have no such downstream consumer and don't need it.
    """
    output_params = ["-crf", str(crf)] if crf is not None else None
    return iio2.get_writer(
        str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264", macro_block_size=1, output_params=output_params
    )
