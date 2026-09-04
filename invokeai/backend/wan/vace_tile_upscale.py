"""Pure helpers for tiled Wan VACE video upscaling: temporal segment planning and crossfade blending.

Spatial tiling reuses InvokeAI's existing Multi-Diffusion tile-grid/blend utilities
(``backend/tiles/tiles.py``) directly -- only the temporal dimension (which those utilities
don't model) needs new helpers here.
"""

from dataclasses import dataclass

import torch


@dataclass
class TemporalTile:
    """One overlapping segment of a longer video, in pixel-frame units."""

    start: int
    length: int
    crossfade: int
    """Overlap (frames) with the previous tile; 0 for the first tile."""


def plan_temporal_tiles(total_frames: int, tile_frames: int, crossfade_frames: int) -> list[TemporalTile]:
    """Split ``total_frames`` into overlapping ``tile_frames``-long segments.

    Every segment has exactly ``tile_frames`` length (always a valid Wan clip length, since
    callers are expected to pass a ``tile_frames`` that already satisfies the causal-VAE
    ``(n - 1) % 4 == 0`` constraint) except when the whole video fits in one tile. The final
    segment is anchored to end exactly on the last frame rather than shrunk, so no segment
    ever needs a different (possibly invalid) length.
    """
    if total_frames <= tile_frames:
        return [TemporalTile(start=0, length=total_frames, crossfade=0)]

    assert 0 <= crossfade_frames < tile_frames, "crossfade_frames must be in [0, tile_frames)"
    step = tile_frames - crossfade_frames

    starts = list(range(0, total_frames - tile_frames, step))
    last_start = total_frames - tile_frames
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    tiles: list[TemporalTile] = []
    prev_end = 0
    for i, start in enumerate(starts):
        crossfade = 0 if i == 0 else max(0, prev_end - start)
        tiles.append(TemporalTile(start=start, length=tile_frames, crossfade=crossfade))
        prev_end = start + tile_frames
    return tiles


def crossfade_videos(a: torch.Tensor, b: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Linearly crossfade the tail ``num_frames`` of ``a`` into the head ``num_frames`` of ``b``.

    ``a``/``b`` are ``[T, H, W, C]`` pixel tensors in the same value range. Returns their
    concatenation with the overlapping region replaced by a linear alpha blend, so the result
    has ``a.shape[0] + b.shape[0] - num_frames`` frames.
    """
    if num_frames <= 0:
        return torch.cat([a, b], dim=0)
    a_tail = a[-num_frames:]
    b_head = b[:num_frames]
    alpha = torch.linspace(0.0, 1.0, num_frames, device=a.device, dtype=a.dtype).view(-1, 1, 1, 1)
    blended = a_tail * (1.0 - alpha) + b_head * alpha
    return torch.cat([a[:-num_frames], blended, b[num_frames:]], dim=0)
