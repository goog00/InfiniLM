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

        # Model compute dtype (pixel_values must match the vision weights).
        self._config_dtype = cfg.get("torch_dtype", "bfloat16")

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

    # Literal markers kept in the rendered prompt string so __call__ can splice the
    # exact number of patch-placeholder tokens per media item. They must not collide
    # with real text; __call__ splits the prompt on them.
    IMAGE_SENTINEL = "<|__ernie_vl_image__|>"
    VIDEO_SENTINEL = "<|__ernie_vl_video__|>"

    # ------------------------------------------------------------------
    # Chat template: pure text defers to parent. For multimodal we render each
    # media item to a sentinel marker (role structure still from the tokenizer
    # template) and let __call__ replace each sentinel with image_start +
    # im_patch * N + image_end token ids. Returns a STRING (tokenize is ignored
    # for the multimodal path; the LLM pipeline tokenizes via __call__).
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

        normalized = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    t = item.get("type", "text")
                    if t == "text":
                        parts.append(item.get("text", ""))
                    elif t in ("image", "image_url"):
                        parts.append(self.IMAGE_SENTINEL)
                    elif t in ("video", "video_url"):
                        parts.append(self.VIDEO_SENTINEL)
                    else:
                        raise ValueError(f"Unsupported content item type: {t}")
                normalized.append({"role": message["role"], "content": "".join(parts)})
            else:
                normalized.append(message)

        # tokenize=False: keep sentinels as literal text; __call__ re-tokenizes and
        # splices placeholder token ids.
        return self.tokenizer.apply_chat_template(
            conversation=normalized,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
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

        if videos:
            # Video preprocessing is not yet wired (see _preprocess_videos).
            raise NotImplementedError("Ernie4_5_VLMoeProcessor: video input not yet supported")

        # Image path: preprocess to patches + grid, then assemble input_ids by
        # splicing image_start + im_patch * N + image_end at each sentinel.
        pixel_values, grid_thw, merged_counts = self._preprocess_images(images)
        input_ids = self._assemble_input_ids(prompt, merged_counts)

        return {
            "input_ids": self._wrap_input_ids(input_ids, return_tensors),
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
        }

    def _assemble_input_ids(self, prompt, merged_counts):
        """Tokenize text segments and splice image placeholder spans.

        For image i, insert image_start + im_patch * merged_counts[i] + image_end at
        the sentinel position. merged_counts[i] equals the number of vision tokens the
        tower emits, so the C++ merge_vision_embeddings scatters them 1:1 onto the
        im_patch positions.
        VERIFY: confirm ERNIE wraps the patch run with image_start/end (vs no wrap)
        and that segment-wise tokenization matches whole-string tokenization at the
        sentinel boundaries.
        """
        segments = prompt.split(self.IMAGE_SENTINEL)
        if len(segments) - 1 != len(merged_counts):
            raise ValueError(
                f"image sentinel count {len(segments) - 1} != image count {len(merged_counts)}"
            )
        ids = []
        for i, seg in enumerate(segments):
            if seg:
                ids.extend(self.tokenizer(seg, add_special_tokens=False)["input_ids"])
            if i < len(merged_counts):
                ids.append(self.image_start_token_id)
                ids.extend([self.im_patch_id] * merged_counts[i])
                ids.append(self.image_end_token_id)
        return ids

    @staticmethod
    def _wrap_input_ids(ids, return_tensors):
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids], dtype=torch.long)
        return ids

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
        merged_counts = []
        block = self.spatial_conv_size * self.spatial_conv_size
        for image_input in images:
            pil = self._load_image(image_input)
            chw = self._normalize_image(pil)
            patches, h, w = self._patchify_frame(chw)
            pixel_values_list.append(patches)
            grid_thw_list.append([1, h, w])
            # Vision tokens after the resampler's spatial merge (t==1 image).
            merged_counts.append((h * w) // block)

        pixel_values = np.concatenate(pixel_values_list, axis=0)
        grid_thw = np.asarray(grid_thw_list, dtype=np.int64)
        return pixel_values, grid_thw, merged_counts

    def _preprocess_videos(self, videos: list):
        # TODO(ernie-vl): decode video (decord / av / cv2), sample frames at a
        # fixed FPS, apply _normalize_image per frame, then _patchify_frame per
        # frame and stack along the t dimension. grid_thw entries become
        # [num_frames, h, w]. Requires an extra third-party dep for decoding,
        # so we defer until the image path is validated end-to-end.
        raise NotImplementedError("Ernie4_5_VLMoeProcessor._preprocess_videos")

    def _build_3d_position_ids(self, input_ids, grid_thw):
        """Qwen2-VL-style get_rope_index -> [3][seq] = (time, height, width).

        Text tokens advance sequentially on all three axes. Each im_patch run gets
        2D (height,width) grid positions offset by the running start; the next text
        token resumes from max(position)+1. The merged grid uses spatial_conv_size
        (the resampler's spatial merge): hh=h//s, ww=w//s.
        VERIFY: match HF modeling_ernie4_5_vl.get_rope_index (axis order + the
        position offset accounting after each image span).
        """
        s = self.spatial_conv_size
        grids = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
        tpos, hpos, wpos = [], [], []
        st = 0
        img_idx = 0
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i] == self.im_patch_id:
                t, h, w = grids[img_idx]
                img_idx += 1
                hh, ww = int(h) // s, int(w) // s
                count = int(t) * hh * ww
                for idx in range(count):
                    ti = idx // (hh * ww)
                    rem = idx % (hh * ww)
                    hi = rem // ww
                    wi = rem % ww
                    tpos.append(st + ti)
                    hpos.append(st + hi)
                    wpos.append(st + wi)
                st = st + max(int(t), hh, ww)
                i += count
            else:
                tpos.append(st)
                hpos.append(st)
                wpos.append(st)
                st += 1
                i += 1
        return [tpos, hpos, wpos]

    def build_model_inputs(self, scheduler_output, temperature=1.0, top_p=0.8, top_k=1):
        """Inject multimodal tensors + 3D mrope positions onto the base text inputs.

        Multimodal data lives on the request's processed_inputs (from __call__).
        pixel_values/tgt_sizes are sent only during prefill (vision is cached after);
        position_ids are replaced with the [3, seq] mrope layout for the tokens
        computed this step. Text-only requests fall through to the base unchanged.
        """
        base = super().build_model_inputs(scheduler_output, temperature, top_p, top_k)

        reqs = getattr(scheduler_output, "scheduled_requests", None)
        if not reqs:
            return base
        req = reqs[0]
        pi = getattr(req, "processed_inputs", None)
        if not pi or pi.get("pixel_values") is None:
            return base

        import infinicore
        import numpy as np

        grid_thw = np.asarray(pi["grid_thw"]).astype(np.int64)
        pos3d = self._build_3d_position_ids(req.get_all_token_ids(), grid_thw)

        if getattr(scheduler_output, "is_prefill", True):
            prefix = getattr(scheduler_output, "prefix_hit_len", 0) or 0
            end = len(req.get_input_tokens())
            pos_slice = [row[prefix:end] for row in pos3d]

            # Flatten patches to [num_patches, C*p*p]; the C++ patch_embed views it
            # back. Build in model dtype so it matches the vision weights.
            pv = np.ascontiguousarray(pi["pixel_values"]).astype(np.float32)
            pv2d = pv.reshape(pv.shape[0], -1)
            base["pixel_values"] = infinicore.from_list(pv2d.tolist(), dtype=self._infini_dtype())
            base["tgt_sizes"] = infinicore.from_list(grid_thw.tolist(), dtype=infinicore.int64)
        else:
            pos = req.get_total_length() - 1
            pos_slice = [[row[pos]] for row in pos3d]

        base["position_ids"] = infinicore.from_list(pos_slice, dtype=infinicore.int64)
        return base

    def _infini_dtype(self):
        """Model compute dtype for pixel_values (must match the vision weights)."""
        import infinicore

        name = (getattr(self, "_config_dtype", "bfloat16") or "bfloat16").lower()
        return {
            "bfloat16": infinicore.bfloat16,
            "float16": infinicore.float16,
            "float32": infinicore.float32,
        }.get(name, infinicore.bfloat16)
