"""Inference correctness test for ERNIE-4.5-VL-28B-A3B.

Covers the three required input modalities and compares InfiniLM output token
sequences against HuggingFace transformers (the reference) under greedy decoding.

Usage:
    python test/models/ernie4_5_moe_vl/test_correctness.py \
        --model /path/to/ERNIE-4.5-VL-28B-A3B-Thinking \
        --device nvidia \
        --image test/assets/demo.jpg \
        --video test/assets/demo.mp4

The HF reference path is gated behind --with-reference (needs the model loadable
by transformers). Without it, the script just runs InfiniLM and prints outputs.
Per the task rules, transformers is used ONLY as a test reference here; the
adapted model/processor code does not depend on it for inference.
"""

import argparse
import ctypes
import os


def _disable_maca_device_heap():
    """Set MACA device malloc heap size to 0 before any GPU allocation.

    On MetaX C500 (64 GB), the model weights consume nearly all VRAM, leaving
    too little for MACA to create its default 8 MB kernel-side heap.  Setting
    the limit to 0 disables the heap entirely; model inference does not use
    device-side malloc so this is safe.
    """
    for libname in ("libmcruntime.so", "libhcruntime.so"):
        try:
            lib = ctypes.CDLL(libname)
            # mcLimitMallocHeapSize / hcLimitMallocHeapSize = 2 (same as cudaLimitMallocHeapSize)
            ret = lib.mcDeviceSetLimit(2, ctypes.c_size_t(0))
            if ret == 0:
                print(f"[INFO] {libname}: mcDeviceSetLimit(MallocHeapSize, 0) OK")
                return
            # fallback: try hc variant
            ret2 = lib.hcDeviceSetLimit(2, ctypes.c_size_t(0))
            if ret2 == 0:
                print(f"[INFO] {libname}: hcDeviceSetLimit(MallocHeapSize, 0) OK")
                return
        except OSError:
            continue
    print("[WARN] could not set device malloc heap size to 0 (library not found)")


_disable_maca_device_heap()


def build_conversation(text, image=None, video=None):
    content = []
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": image}})
    if video is not None:
        content.append({"type": "video_url", "video_url": {"url": video}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def run_infinilm(model_path, device, conversation, max_new_tokens, ignore_eos=False):
    from infinilm.llm.llm import LLM
    from infinilm.llm.sampling_params import SamplingParams

    model = LLM(
        model_path=os.path.expanduser(model_path),
        device=device,
        tensor_parallel_size=1,
        cache_type="static",
        max_batch_size=1,
        max_tokens=max_new_tokens,
        max_cache_len=1024,  # 64 MB KV cache; default 4096 (224 MB) exceeds free VRAM on C500
        temperature=1.0,
        top_k=1,  # greedy
        top_p=1.0,
    )

    # Print diagnostic info about EOS token IDs.
    eos_ids = model.engine.eos_token_ids
    print(f"[DEBUG] eos_token_ids from config: {eos_ids}")

    # Apply chat template and show prompt tokens for debugging.
    prompt_str = model.engine.apply_chat_template(conversation, add_generation_prompt=True)
    prompt_tokens = model.engine.tokenize(prompt_str)
    print(f"[DEBUG] prompt (first 200 chars): {repr(prompt_str[:200])}")
    print(f"[DEBUG] prompt token count: {len(prompt_tokens)}")
    print(f"[DEBUG] first 20 prompt tokens: {prompt_tokens[:20]}")
    print(f"[DEBUG] last 10 prompt tokens: {prompt_tokens[-10:]}")

    # Show decoded form of last few tokens to check chat template boundary.
    try:
        last_tok_decoded = model.engine.tokenizer.convert_ids_to_tokens(prompt_tokens[-10:])
        print(f"[DEBUG] last 10 tokens decoded: {last_tok_decoded}")
    except Exception as e:
        print(f"[DEBUG] could not decode last tokens: {e}")

    # Show what token 2 decodes to in this tokenizer.
    try:
        tok2_str = model.engine.detokenize([2])
        print(f"[DEBUG] token_id=2 decodes to: {repr(tok2_str)}")
        tok1_str = model.engine.detokenize([1])
        print(f"[DEBUG] token_id=1 decodes to: {repr(tok1_str)}")
        tok0_str = model.engine.detokenize([0])
        print(f"[DEBUG] token_id=0 decodes to: {repr(tok0_str)}")
    except Exception as e:
        print(f"[DEBUG] could not decode special tokens: {e}")

    sp = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
    )
    outputs = model.chat(messages=[conversation], sampling_params=sp)
    return outputs


def run_reference(model_path, conversation, max_new_tokens):
    # TODO(ernie-vl): load with transformers AutoModelForCausalLM + AutoProcessor
    # (trust_remote_code=True), run greedy generate on the same inputs, return the
    # generated token ids. Reference only — not used by the adapted inference path.
    raise NotImplementedError("HF reference path not wired up yet")


def compare(infinilm_ids, reference_ids):
    # Primary metric: exact token-sequence match. Secondary (fallback): semantic
    # equivalence on decoded text. See task §4.
    if infinilm_ids == reference_ids:
        print("[PASS] exact token match")
        return True
    n = min(len(infinilm_ids), len(reference_ids))
    first_div = next((i for i in range(n) if infinilm_ids[i] != reference_ids[i]), n)
    print(f"[FAIL] diverged at token {first_div}")
    return False


CASES = [
    ("text",  dict(text="用一句话介绍你自己。")),
    ("image", dict(text="描述这张图片。", image="IMAGE_PATH")),
    ("video", dict(text="描述这段视频的内容。", video="VIDEO_PATH")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--with-reference", action="store_true")
    ap.add_argument("--cases", default="text,image,video")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Ignore EOS during generation to see what tokens follow (debug mode).")
    args = ap.parse_args()

    selected = set(args.cases.split(","))
    for name, kw in CASES:
        if name not in selected:
            continue
        if name == "image":
            if not args.image:
                print("[SKIP] image case: no --image provided")
                continue
            kw["image"] = args.image
        if name == "video":
            if not args.video:
                print("[SKIP] video case: no --video provided")
                continue
            kw["video"] = args.video

        print(f"\n===== case: {name} =====")
        conversation = build_conversation(**kw)
        infinilm_out = run_infinilm(
            args.model, args.device, conversation, args.max_new_tokens,
            ignore_eos=args.ignore_eos,
        )
        print(f"[InfiniLM] {infinilm_out}")

        if args.with_reference:
            ref_out = run_reference(args.model, conversation, args.max_new_tokens)
            compare(infinilm_out, ref_out)


if __name__ == "__main__":
    main()
