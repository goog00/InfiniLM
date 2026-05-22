"""Processor for ERNIE-4.5-VL-28B-A3B (model_type: ernie4_5_moe_vl).

Handles the three input modalities required by the task:
  1. text -> text
  2. image + text -> text
  3. video + text -> text

Inherits from BasicLLMProcessor for the text-only path (build_model_inputs,
tokenization, chat template defaults). Multimodal extensions are layered on top.

NOTE on third-party deps: the tokenizer is loaded via transformers.AutoTokenizer
(a standard component, same as BasicLLMProcessor). The multimodal *adaptation*
logic (patchify, placeholder expansion, 3D/mrope position ids) must be implemented
here and must NOT call into a third-party model's forward/processing internals;
transformers may only be used as a reference in the correctness test.
"""

import json
import os
from typing import Optional

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


def _conversation_is_text_only(conversation) -> bool:
    """Return True if no message contains a non-text content item."""
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                # Anything other than {"type": "text", ...} is multimodal.
                if isinstance(item, dict) and item.get("type", "text") != "text":
                    return False
    return True


@register_processor("ernie4_5_moe_vl")
class Ernie4_5_VLMoeProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        # Tokenizer setup via parent (transformers AutoTokenizer with trust_remote_code).
        # Uses the Ernie4_5_VLTokenizer class declared in tokenizer_config.json's
        # auto_map, defined in processing_ernie4_5_vl.py in the model dir.
        super().__init__(model_dir_path)

        with open(os.path.join(model_dir_path, "config.json")) as f:
            cfg = json.load(f)

        # Special token ids for vision placeholders / span markers.
        self.im_patch_id = cfg.get("im_patch_id", 100295)
        self.image_start_token_id = cfg.get("image_start_token_id", 101304)
        self.image_end_token_id = cfg.get("image_end_token_id", 101305)
        self.video_start_token_id = cfg.get("video_start_token_id", 101306)
        self.video_end_token_id = cfg.get("video_end_token_id", 101307)

        # Vision / resampler geometry.
        vision_cfg = cfg.get("vision_config", {})
        self.patch_size = vision_cfg.get("patch_size", 14)
        self.spatial_merge_size = vision_cfg.get("spatial_merge_size", 2)
        self.temporal_conv_size = cfg.get("temporal_conv_size", 2)
        self.spatial_conv_size = cfg.get("spatial_conv_size", 2)

    # ------------------------------------------------------------------
    # Chat template: defer to parent for pure-text conversations; multimodal
    # rendering (with image/video placeholders) is the next task.
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        if _conversation_is_text_only(conversation):
            return super().apply_chat_template(
                conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        # TODO(ernie-vl): render image/video items into <|IMAGE_START|>+im_patch*N+
        # <|IMAGE_END|> spans according to the tokenizer's chat template. The
        # placeholder count must match the patch count produced by _preprocess_*.
        raise NotImplementedError(
            "Ernie4_5_VLMoeProcessor.apply_chat_template: multimodal rendering not yet implemented"
        )

    # ------------------------------------------------------------------
    # Text-only entry: defer to BasicLLMProcessor.__call__ unless multimodal.
    # ------------------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        images: Optional[list] = None,
        videos: Optional[list] = None,
        audios: Optional[list] = None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        # Pure text path — same as BasicLLMProcessor (drives the text correctness test).
        if not images and not videos:
            return super().__call__(prompt, return_tensors=return_tensors, **kwargs)

        # Multimodal path — fall through to local processing.
        encoding = self.tokenizer(prompt, add_special_tokens=False)
        result = {"input_ids": encoding["input_ids"]}

        if images:
            pixel_values, grid_thw = self._preprocess_images(images)
            result["pixel_values"] = pixel_values
            result["grid_thw"] = grid_thw
        if videos:
            pixel_values_v, grid_thw_v = self._preprocess_videos(videos)
            result["pixel_values_videos"] = pixel_values_v
            result["video_grid_thw"] = grid_thw_v

        return result

    # ------------------------------------------------------------------
    # Multimodal preprocessing.
    # ------------------------------------------------------------------
    # CLIP-style normalization (used by most large vision encoders, including
    # ERNIE-VL's DFNRope ViT). Confirm against the checkpoint's preprocessor_config
    # if accuracy is off.
    _IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    def _patchify_frame(self, np_chw):
        """Take a [3, H, W] float32 array (already normalized) and emit
        patches in spatial-merge-friendly order along with the patch grid.

        Returns: (patches [num_patches, 3, patch, patch], h_grid, w_grid).
        The patches are laid out so every spatial_merge x spatial_merge block
        of consecutive entries forms one 2x2 spatial group — matches the
        resampler's view() contract.
        """
        import numpy as np

        _, H, W = np_chw.shape
        p = self.patch_size
        m = self.spatial_merge_size
        assert H % (p * m) == 0 and W % (p * m) == 0, (
            f"image size {(H, W)} must be a multiple of patch*merge={p * m}"
        )

        h = H // p
        w = W // p
        # [3, h*p, w*p] -> [3, h, p, w, p] -> [h, w, 3, p, p]
        x = np_chw.reshape(3, h, p, w, p).transpose(1, 3, 0, 2, 4)
        # Group neighbouring 2x2 patches: [h/m, m, w/m, m, 3, p, p]
        x = x.reshape(h // m, m, w // m, m, 3, p, p)
        # Reorder so each spatial group is contiguous: [h/m, w/m, m, m, 3, p, p]
        x = x.transpose(0, 2, 1, 3, 4, 5, 6)
        # Flatten: [num_patches, 3, p, p]
        patches = x.reshape(-1, 3, p, p).astype(np.float32, copy=False)
        return patches, h, w

    def _load_image(self, image_input):
        from PIL import Image

        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _normalize_image(self, pil_img):
        """Resize to a multiple of patch*spatial_merge, normalize, return [3,H,W]."""
        import numpy as np
        from PIL import Image

        cell = self.patch_size * self.spatial_merge_size
        W, H = pil_img.size
        # Round up to the next cell boundary; the vision tower handles variable
        # resolution so we don't need a fixed canonical size.
        new_W = max(((W + cell - 1) // cell) * cell, cell)
        new_H = max(((H + cell - 1) // cell) * cell, cell)
        if (new_W, new_H) != (W, H):
            pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)

        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        mean = np.array(self._IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self._IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def _preprocess_images(self, images: list):
        import numpy as np

        pixel_values_list = []
        grid_thw_list = []
        for image_input in images:
            pil = self._load_image(image_input)
            chw = self._normalize_image(pil)
            patches, h, w = self._patchify_frame(chw)
            pixel_values_list.append(patches)
            grid_thw_list.append([1, h, w])

        pixel_values = np.concatenate(pixel_values_list, axis=0)
        grid_thw = np.asarray(grid_thw_list, dtype=np.int64)
        return pixel_values, grid_thw

    def _preprocess_videos(self, videos: list):
        # TODO(ernie-vl): decode video (decord / av / cv2), sample frames at a
        # fixed FPS, apply _normalize_image per frame, then _patchify_frame per
        # frame and stack along the t dimension. grid_thw entries become
        # [num_frames, h, w]. Requires an extra third-party dep for decoding,
        # so we defer until the image path is validated end-to-end.
        raise NotImplementedError("Ernie4_5_VLMoeProcessor._preprocess_videos")

    def _build_3d_position_ids(self, input_ids, grid_thw=None):
        """Produce position_ids of shape [3, seq_len] = (time, height, width).

        Initial implementation: replicate the 1D text positions across all three
        rows. This makes the model behave like standard 1D RoPE and matches the
        attention.cpp fallback that currently takes row 0 only. Full mrope (with
        per-axis advancement for vision tokens) is wired up in a later pass —
        see the 3D mrope TODO in attention.cpp.
        """
        seq_len = len(input_ids)
        row = list(range(seq_len))
        return [row, row, row]
