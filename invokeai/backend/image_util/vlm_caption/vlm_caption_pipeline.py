from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import PreTrainedModel, ProcessorMixin

from invokeai.backend.raw_model import RawModel


def _extract_video_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """Grab ``num_frames`` evenly-spaced frames from ``video_path`` by seeking directly to each
    target frame index, instead of decoding the whole file sequentially.

    transformers' own opencv video loader (``read_video_opencv``) calls ``video.read()`` in a loop
    over *every* frame up to the last sampled index and only keeps the ones it wants -- for a long
    clip (e.g. 30+ seconds at 30fps) that's ~1000 sequential decodes just to keep 16 frames, which
    is exactly the kind of unbounded work we don't want to hand off. Seeking via
    ``CAP_PROP_POS_FRAMES`` decodes only the frames actually needed.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not read frame count from video: {video_path}")
        count = min(num_frames, total)
        indices = [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]

        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        if not frames:
            raise ValueError(f"Could not decode any frames from video: {video_path}")
        return frames
    finally:
        cap.release()


class VlmCaptionPipeline(RawModel):
    """A wrapper around a Qwen2.5-VL model + processor for instruction-following image/video
    captioning and plain text generation (mirrors GroundingDinoPipeline's role for Grounding DINO).
    """

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin):
        self._model = model
        self._processor = processor

    def _generate(self, content: list[dict], processor_kwargs: dict, max_new_tokens: int) -> str:
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        messages = [{"role": "user", "content": content}]
        text_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text_prompt], return_tensors="pt", **processor_kwargs)
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
            for k, v in inputs.items()
        }

        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    def caption(self, image: Image.Image, instruction: str, max_new_tokens: int = 512) -> str:
        """Caption ``image`` following a free-form natural-language ``instruction``."""
        content = [{"type": "image"}, {"type": "text", "text": instruction}]
        return self._generate(content, {"images": [image]}, max_new_tokens)

    def caption_video(
        self, video_path: Path, instruction: str, max_new_tokens: int = 256, num_frames: int = 12
    ) -> str:
        """Describe ``video_path`` (an on-disk video file) following a free-form ``instruction``.

        Extracts ``num_frames`` evenly-spaced frames ourselves (seeking directly to each one,
        rather than handing the whole file to transformers' video loader, which decodes every
        frame sequentially up to the last sampled index -- unbounded work for a long clip, and the
        source of a real OOM: a 34s 1080x1920 video attempted a 100+ GiB allocation). Qwen2.5-VL
        still gets genuine temporal context across these frames (its video position encoding
        applies to any frame sequence), just without decoding frames we'd throw away anyway.
        """
        frames = _extract_video_frames(video_path, num_frames)
        content = [{"type": "video"}, {"type": "text", "text": instruction}]
        return self._generate(
            content,
            {
                "videos": [frames],
                "do_sample_frames": False,
                # Motion description only needs coarse detail, not full-res frames.
                "min_pixels": 128 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            },
            max_new_tokens,
        )

    def complete_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Plain text-in/text-out generation (no image/video), e.g. for rewriting a caption."""
        content = [{"type": "text", "text": prompt}]
        return self._generate(content, {}, max_new_tokens)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self._model.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self._model)
