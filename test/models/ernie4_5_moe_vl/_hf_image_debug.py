"""TEMPORARY HF reference dump for the ERNIE-4.5-VL image path (delete before submit).

Runs the HF model on the SAME image+prompt as test_correctness.py --cases image
and dumps every comparison checkpoint so we can diff against InfiniLM's ERNIE_DBG:

  inputs : input_ids, grid_thw, pixel_values stats, get_rope_index position_ids
  source : get_rope_index / rope source (verify _build_3d_position_ids + build_mrope_)
  vision : model.visual(...) output stats (== vl.vision_embeds)
  merged : inputs_embeds after vision scatter (== vl.merged)
  L0/L1  : backbone decoder layer hidden_states (== L0/L1 stream)
  logits : top-5 of last-token logits (== [LOGIT DEBUG])

Usage:
  python test/models/ernie4_5_moe_vl/_hf_image_debug.py \
      --model $MODEL --image /tmp/test.jpg --text "描述这张图片。"
"""
import argparse
import inspect

import numpy as np
import torch


def stats(tag, t):
    if t is None:
        print(f"[HFDBG] {tag:18s} None")
        return
    x = t.detach().float().cpu().reshape(-1)
    n = x.numel()
    head = " ".join(f"{v:.4f}" for v in x[:3].tolist())
    tail = " ".join(f"{v:.4f}" for v in x[-3:].tolist())
    print(f"[HFDBG] {tag:18s} numel={n} shape={tuple(t.shape)} "
          f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.6f} "
          f"absmax={x.abs().max():.4f} nan={int(torch.isnan(x).sum())} "
          f"inf={int(torch.isinf(x).sum())} | head={head} tail={tail}")


def try_getsource(obj, name):
    try:
        print(f"\n===== source: {name} =====")
        print(inspect.getsource(obj))
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] could not get source for {name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default="描述这张图片。")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image_url": args.image},
            {"type": "text", "text": args.text},
        ],
    }]
    # The HF processor message schema may differ from InfiniLM's; try both shapes.
    try:
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] apply_chat_template(image_url) failed: {e}; retry HF schema")
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        conversation[0]["content"][0] = {"type": "image", "image": img}
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")

    print("[HFDBG] input keys:", list(inputs.keys()))
    ids = inputs["input_ids"][0].tolist()
    print(f"[HFDBG] input_ids ({len(ids)}): {ids}")
    for k in ("grid_thw", "image_grid_thw", "tgt_sizes"):
        if k in inputs:
            print(f"[HFDBG] {k} = {inputs[k].tolist() if hasattr(inputs[k], 'tolist') else inputs[k]}")
    for k in ("pixel_values", "images", "pixel_values_images"):
        if k in inputs and inputs[k] is not None:
            stats(f"in.{k}", inputs[k])

    # Source of position / rope helpers (verify our processor + build_mrope_).
    for cand in (model, getattr(model, "model", None)):
        if cand is not None and hasattr(cand, "get_rope_index"):
            try_getsource(cand.get_rope_index, "get_rope_index")
            break
    # rope application (verify GPT-J interleave + 3D axis allocation)
    for modname in ("apply_rotary_3d", "apply_rotary", "rotary_emb"):
        for cand in (model, getattr(model, "model", None)):
            obj = getattr(cand, modname, None) if cand is not None else None
            if obj is not None:
                try_getsource(type(obj) if not callable(obj) else obj, modname)
                break

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Print HF's own get_rope_index output to diff against our _build_3d_position_ids.
    try:
        owner = model if hasattr(model, "get_rope_index") else getattr(model, "model", None)
        gri = owner.get_rope_index
        import inspect as _ins
        sig = _ins.signature(gri)
        kw = {}
        if "input_ids" in sig.parameters:
            kw["input_ids"] = inputs["input_ids"]
        for gk in ("image_grid_thw", "grid_thw"):
            if gk in sig.parameters and gk in inputs:
                kw[gk] = inputs[gk]
        if "attention_mask" in sig.parameters:
            kw["attention_mask"] = inputs.get("attention_mask")
        res = gri(**kw)
        pos = res[0] if isinstance(res, tuple) else res
        p = pos.detach().cpu()
        # pos shape commonly [3, bs, seq]; squeeze batch.
        if p.dim() == 3:
            p = p[:, 0, :]
        print(f"[HFDBG] get_rope_index pos shape={tuple(pos.shape)}")
        for ax, nm in enumerate(("time", "height", "width")):
            row = p[ax].tolist() if p.dim() == 2 else p.tolist()
            print(f"[HFDBG] pos.{nm}: {row}")
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] get_rope_index call failed: {e}")

    # Bypass SDPA (no kernel) like the text path: route every attn through eager.
    nlayer = 0
    backbone = model.model if hasattr(model, "model") else model
    layers = None
    for path in ("layers", "language_model.layers", "model.layers"):
        obj = backbone
        ok = True
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                ok = False
                break
        if ok:
            layers = obj
            break
    if layers is not None:
        nlayer = len(layers)
        for lyr in layers:
            attn = getattr(lyr, "self_attn", None)
            if attn is not None and hasattr(attn, "core_attn"):
                attn.attn_func = attn.core_attn
        print(f"[HFDBG] patched core_attn on {nlayer} layers")

    # Hooks: vision output, layer 0/1 output.
    captures = {}

    def mk_hook(name):
        def hook(_m, _i, o):
            captures[name] = o[0] if isinstance(o, tuple) else o
        return hook

    for vname in ("visual", "vision_model", "vision_tower"):
        vis = getattr(model, vname, None) or getattr(backbone, vname, None)
        if vis is not None:
            vis.register_forward_hook(mk_hook("vision"))
            print(f"[HFDBG] hooked vision module: {vname}")
            break
    if layers is not None:
        layers[0].register_forward_hook(mk_hook("L0"))
        if nlayer > 1:
            layers[1].register_forward_hook(mk_hook("L1"))

    # generate (proven for the text reference) builds position_ids/token_type_ids
    # internally; max_new_tokens=1 + output_scores gives the first-step logits.
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=1, do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=True)

    stats("vision", captures.get("vision"))
    stats("L0 stream", captures.get("L0"))
    stats("L1 stream", captures.get("L1"))

    logits = gen.scores[0][0].float().cpu()
    top = torch.topk(logits, 5)
    print("[HFDBG] logits min={:.3f} max={:.3f}".format(logits.min(), logits.max()))
    print("[HFDBG] top5:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
