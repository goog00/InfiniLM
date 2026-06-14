"""TEMPORARY HF reference dump for the ERNIE-4.5-VL image path (delete before submit).

Two goals in one run:
  (A) DUMP HF SOURCE (ground truth to verify our port): processor.__call__,
      model.get_rope_index, rope application. Always prints, even if forward fails.
  (B) RUN HF FORWARD on the same image+prompt and dump comparison checkpoints
      (vision output, L0/L1 hidden states, top-5 logits) to diff vs ERNIE_DBG.

Usage:
  python test/models/ernie4_5_moe_vl/_hf_image_debug.py \
      --model $MODEL --image /tmp/test.jpg --text "描述这张图片。"
"""
import argparse
import inspect

import torch


def stats(tag, t):
    if t is None:
        print(f"[HFDBG] {tag:18s} None")
        return
    x = t.detach().float().cpu().reshape(-1)
    head = " ".join(f"{v:.4f}" for v in x[:3].tolist())
    tail = " ".join(f"{v:.4f}" for v in x[-3:].tolist())
    print(f"[HFDBG] {tag:18s} numel={x.numel()} shape={tuple(t.shape)} "
          f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.6f} "
          f"absmax={x.abs().max():.4f} nan={int(torch.isnan(x).sum())} "
          f"inf={int(torch.isinf(x).sum())} | head={head} tail={tail}")


def dump_source(name, obj):
    print(f"\n===== source: {name} =====")
    try:
        print(inspect.getsource(obj))
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] no source for {name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default="描述这张图片。")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print("[HFDBG] processor type:", type(processor))
    print("[HFDBG] processor public attrs:", [m for m in dir(processor) if not m.startswith("_")])
    print("[HFDBG] tokenizer.chat_template present:", bool(getattr(tokenizer, "chat_template", None)))

    # (A) Dump processor + tokenizer entry points so we know the API + can verify our port.
    dump_source("processor.__call__", type(processor).__call__)
    for meth in ("apply_chat_template", "process", "preprocess", "_add_special_tokens"):
        obj = getattr(processor, meth, None)
        if obj is not None:
            dump_source(f"processor.{meth}", obj)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    backbone = model.model if hasattr(model, "model") else model

    # get_rope_index + rope source (verify _build_3d_position_ids + build_mrope_).
    for owner_name, owner in (("model", model), ("backbone", backbone)):
        if hasattr(owner, "get_rope_index"):
            dump_source(f"{owner_name}.get_rope_index", owner.get_rope_index)
            break

    # (B) Build inputs. processor has no chat template; render via tokenizer then
    # feed text+image to the processor. Try the common transformers patterns.
    img = Image.open(args.image).convert("RGB")
    rendered = None
    try:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.text}]}]
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] tokenizer.apply_chat_template(list) failed: {e}")
        rendered = f"User: {args.text}\nAssistant:"
    print(f"[HFDBG] rendered prompt = {rendered!r}")

    inputs = None
    for desc, fn in (
        ("processor(text=,images=)", lambda: processor(text=rendered, images=[img], return_tensors="pt")),
        ("processor(text=,images=img)", lambda: processor(text=rendered, images=img, return_tensors="pt")),
        ("processor(rendered,[img])", lambda: processor(rendered, [img], return_tensors="pt")),
        ("processor(images=,text=list)", lambda: processor(text=[rendered], images=[img], return_tensors="pt")),
    ):
        try:
            inputs = fn()
            print(f"[HFDBG] inputs built via {desc}; keys={list(inputs.keys())}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[HFDBG] {desc} failed: {e}")
    if inputs is None:
        print("[HFDBG] could not build processor inputs; see source dumps above. Aborting forward.")
        return

    ids = inputs["input_ids"][0].tolist()
    print(f"[HFDBG] input_ids ({len(ids)}): {ids}")
    for k in ("grid_thw", "image_grid_thw", "tgt_sizes"):
        if k in inputs:
            print(f"[HFDBG] {k} = {inputs[k].tolist() if hasattr(inputs[k], 'tolist') else inputs[k]}")
    for k in ("pixel_values", "images", "pixel_values_images"):
        if k in inputs and inputs[k] is not None and hasattr(inputs[k], "shape"):
            stats(f"in.{k}", inputs[k])

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # Call HF get_rope_index to diff against our _build_3d_position_ids.
    try:
        owner = model if hasattr(model, "get_rope_index") else backbone
        sig = inspect.signature(owner.get_rope_index)
        kw = {}
        if "input_ids" in sig.parameters:
            kw["input_ids"] = inputs["input_ids"]
        for gk in ("image_grid_thw", "grid_thw"):
            if gk in sig.parameters and gk in inputs:
                kw[gk] = inputs[gk]
        if "attention_mask" in sig.parameters and "attention_mask" in inputs:
            kw["attention_mask"] = inputs["attention_mask"]
        res = owner.get_rope_index(**kw)
        pos = res[0] if isinstance(res, tuple) else res
        p = pos.detach().cpu()
        if p.dim() == 3:  # [3, bs, seq] -> [3, seq]
            p = p[:, 0, :]
        print(f"[HFDBG] get_rope_index pos shape={tuple(pos.shape)}")
        for ax, nm in enumerate(("time", "height", "width")):
            print(f"[HFDBG] pos.{nm}: {p[ax].tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] get_rope_index call failed: {e}")

    # Bypass SDPA like the text path.
    layers = getattr(backbone, "layers", None)
    nlayer = len(layers) if layers is not None else 0
    if layers is not None:
        for lyr in layers:
            attn = getattr(lyr, "self_attn", None)
            if attn is not None and hasattr(attn, "core_attn"):
                attn.attn_func = attn.core_attn
        print(f"[HFDBG] patched core_attn on {nlayer} layers")

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

    try:
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, num_beams=1,
                return_dict_in_generate=True, output_scores=True)
        logits = gen.scores[0][0].float().cpu()
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] generate failed ({e}); trying plain forward")
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1].float().cpu()

    stats("vision", captures.get("vision"))
    stats("L0 stream", captures.get("L0"))
    stats("L1 stream", captures.get("L1"))
    top = torch.topk(logits, 5)
    print("[HFDBG] logits min={:.3f} max={:.3f}".format(logits.min(), logits.max()))
    print("[HFDBG] top5:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
