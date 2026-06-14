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

    # Position-id logic lives in the processor (it returns position_ids directly).
    for meth in ("_add_image", "_add_text", "_pack_outputs", "_add_video"):
        obj = getattr(processor, meth, None)
        if obj is not None:
            dump_source(f"processor.{meth}", obj)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        print("[HFDBG] image_processor:", type(ip).__name__,
              "mean=", getattr(ip, "image_mean", None),
              "std=", getattr(ip, "image_std", None),
              "rescale=", getattr(ip, "rescale_factor", None))

    # (B) Build inputs via processor(text=rendered, images=[img]).
    img = Image.open(args.image).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.text}]}]
    rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    print(f"[HFDBG] rendered prompt = {rendered!r}")
    inputs = processor(text=rendered, images=[img], return_tensors="pt")
    print(f"[HFDBG] inputs keys={list(inputs.keys())}")

    ids = inputs["input_ids"][0].tolist()
    print(f"[HFDBG] input_ids ({len(ids)}): {ids}")
    if "grid_thw" in inputs:
        print(f"[HFDBG] grid_thw = {inputs['grid_thw'].tolist()}")
    # Position ids + token_type ids: ground truth for our processor.
    for key in ("position_ids", "token_type_ids", "image_type_ids"):
        if key in inputs and inputs[key] is not None:
            t = inputs[key]
            sq = t
            while hasattr(sq, "dim") and sq.dim() > 2 and sq.shape[0] == 1:
                sq = sq[0]
            print(f"[HFDBG] {key} shape={tuple(t.shape)} ->")
            if key == "position_ids" and hasattr(sq, "dim") and sq.dim() == 2 and sq.shape[0] == 3:
                for ax, nm in enumerate(("time", "height", "width")):
                    print(f"[HFDBG]   pos.{nm}: {sq[ax].tolist()}")
            else:
                print(f"[HFDBG]   {sq.tolist()}")
    if "images" in inputs and hasattr(inputs["images"], "shape"):
        stats("in.images(raw)", inputs["images"])

    # Normalize images: processor returns uint8 [N, C*p*p] (channel-major), but
    # vision_forward asserts bf16 normalized. Match our processor (CLIP mean/std).
    imgs = inputs["images"]
    if imgs.dtype == torch.uint8 or imgs.float().max() > 10:
        mean = torch.tensor(getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073]))
        std = torch.tensor(getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711]))
        resc = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
        N = imgs.shape[0]
        x = imgs.float().view(N, 3, -1)
        x = (x * resc - mean.view(1, 3, 1)) / std.view(1, 3, 1)
        inputs["images"] = x.view(N, -1).to(torch.bfloat16)
        stats("in.images(norm)", inputs["images"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    backbone = model.model if hasattr(model, "model") else model

    # vision_forward + conditional-generation forward source (resampler + scatter
    # live in the latter); verify our vision tower + merger port.
    for meth in ("vision_forward",):
        obj = getattr(model, meth, None)
        if obj is not None:
            dump_source(f"model.{meth}", obj)
    dump_source("type(model).forward", type(model).forward)
    print("[HFDBG] model children:", [n for n, _ in model.named_children()])

    # MoE block: layer 1 is the first MoE layer; the vision-token routing
    # (token_type=1 -> vision experts/gate) diverges there (L0 matches, L1 doesn't).
    _bb = model.model if hasattr(model, "model") else model
    _lys = getattr(_bb, "layers", None)
    if _lys is not None and len(_lys) > 1:
        moe = getattr(_lys[1], "mlp", None)
        if moe is not None:
            dump_source("moe_block.forward", type(moe).forward)
            print("[HFDBG] moe children:", [n for n, _ in moe.named_children()])
            for sub in ("gate", "experts", "shared_experts", "moe_statics"):
                so = getattr(moe, sub, None)
                if so is not None:
                    n = len(so) if hasattr(so, "__len__") else ""
                    print(f"[HFDBG] moe.{sub}: {type(so).__name__} len={n}")
            g = getattr(moe, "gate", None)
            if g is not None:
                dump_source("moe_gate.forward", type(g).forward)
                for gm in ("get_gate_weight", "_cal_gate_logits_and_dispatch",
                           "topk_gating", "top1_gating", "gating", "_priority"):
                    gmo = getattr(g, gm, None)
                    if gmo is not None and callable(gmo):
                        dump_source(f"moe_gate.{gm}", gmo)
                for gp in ("weight", "weight_1", "bias"):
                    gw = getattr(g, gp, None)
                    if gw is not None and hasattr(gw, "shape"):
                        print(f"[HFDBG] moe.gate.{gp} shape={tuple(gw.shape)}")
                print("[HFDBG] gate flags:",
                      {k: getattr(g, k, None) for k in
                       ("use_token_type_bias", "norm_min", "moe_k", "top_k",
                        "num_experts", "num_experts_tensor", "config_topk", "use_correction_bias")})
            for dm in ("fused_gate_logits_process_fused", "fused_gate_logits_process",
                       "combine_expert_output", "gate_distribution"):
                dmo = getattr(moe, dm, None)
                if dmo is not None and callable(dmo):
                    dump_source(f"moe.{dm}", dmo)
            ms = getattr(moe, "moe_statics", None)
            if ms is not None:
                eb = getattr(ms, "e_score_correction_bias", None)
                if eb is not None and hasattr(eb, "shape"):
                    print(f"[HFDBG] moe_statics.e_score_correction_bias shape={tuple(eb.shape)}")
            se = getattr(moe, "shared_experts", None)
            if se is not None:
                print(f"[HFDBG] shared_experts: {se}")

    # The block divergence is inside the ViT; dump vision_model + one block + rope.
    vm = getattr(model, "vision_model", None)
    if vm is not None:
        dump_source("vision_model.forward", type(vm).forward)
        print("[HFDBG] vision_model children:", [n for n, _ in vm.named_children()])
        for m in ("rot_pos_emb",):
            mo = getattr(vm, m, None)
            if mo is not None:
                dump_source(f"vision_model.{m}", mo)
        vblocks = getattr(vm, "blocks", None) or getattr(vm, "layers", None)
        if vblocks is not None and len(vblocks) > 0:
            dump_source("vision_block.forward", type(vblocks[0]).forward)
            attn0 = getattr(vblocks[0], "attn", None) or getattr(vblocks[0], "attention", None)
            if attn0 is not None:
                dump_source("vision_attn.forward", type(attn0).forward)
        for rn in ("rotary_pos_emb", "rot_pos_emb", "rope", "rotary_emb"):
            ro = getattr(vm, rn, None)
            if ro is not None:
                dump_source(f"vision.{rn}", type(ro).forward if hasattr(type(ro), "forward") else ro)
                break

    # Resampler/merger source (verify our Ernie4_5_VLResampler; note HF ran
    # temporal_linear even for a single image t=1 -- check how).
    rsm = getattr(model, "resampler_model", None) or getattr(backbone, "resampler_model", None)
    if rsm is not None:
        dump_source("resampler_model.forward", type(rsm).forward)
        for m in ("spatial_conv_reshape",):
            mo = getattr(rsm, m, None)
            if mo is not None:
                dump_source(f"resampler.{m}", mo)
        print("[HFDBG] resampler children:", [n for n, _ in rsm.named_children()])
        an = getattr(rsm, "after_norm", None)
        if an is not None:
            print("[HFDBG] after_norm eps:", getattr(an, "eps", getattr(an, "variance_epsilon", None)))
        for sub in ("spatial_linear", "temporal_linear", "mlp", "after_norm"):
            so = getattr(rsm, sub, None)
            if so is not None:
                print(f"[HFDBG] resampler.{sub} = {so}")

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # The model asserts token_type_ids length == seq+1 (it shifts internally for the
    # next-token slot). Pad one text(0) column at the end.
    tt = inputs.get("token_type_ids")
    if tt is not None and tt.shape[1] == inputs["input_ids"].shape[1]:
        pad = torch.zeros((tt.shape[0], 1), dtype=tt.dtype, device=tt.device)
        inputs["token_type_ids"] = torch.cat([tt, pad], dim=1)
        print(f"[HFDBG] padded token_type_ids -> {tuple(inputs['token_type_ids'].shape)}")

    captures = {}

    def mk_hook(name):
        def hook(_m, _i, o):
            captures[name] = o[0] if isinstance(o, tuple) else o
        return hook

    # Capture the POST-merge vision embeddings (vision_forward return), which is
    # what gets scattered into the text sequence == our vl.vision_embeds [256,2560].
    if hasattr(model, "vision_forward"):
        _orig_vf = model.vision_forward

        def _vf_wrap(*a, **k):
            r = _orig_vf(*a, **k)
            captures["vision_forward_out"] = r[0] if isinstance(r, tuple) else r
            return r

        model.vision_forward = _vf_wrap
        print("[HFDBG] wrapped vision_forward")

    # Bypass SDPA like the text path.
    layers = getattr(backbone, "layers", None)
    nlayer = len(layers) if layers is not None else 0
    if layers is not None:
        for lyr in layers:
            attn = getattr(lyr, "self_attn", None)
            if attn is not None and hasattr(attn, "core_attn"):
                attn.attn_func = attn.core_attn
        print(f"[HFDBG] patched core_attn on {nlayer} layers")

    for vname in ("visual", "vision_model", "vision_tower"):
        vis = getattr(model, vname, None) or getattr(backbone, vname, None)
        if vis is not None:
            vis.register_forward_hook(mk_hook("vision_model_raw"))
            print(f"[HFDBG] hooked raw ViT module: {vname}")
            break
    # patch_embed (compare our vis.patch_embed [1024,1280]) + any resampler/merger.
    import re
    for name, mod in model.named_modules():
        low = name.lower()
        if low.endswith("patch_embed"):
            mod.register_forward_hook(mk_hook("patch_embed"))
            print(f"[HFDBG] hooked patch_embed: {name}")
        if any(k in low for k in ("resampler", "merger", "mlp_ar", "variable_resolution")) \
                and low.count(".") <= 2:
            mod.register_forward_hook(mk_hook("resampler:" + name))
            print(f"[HFDBG] hooked resampler candidate: {name}")
        # First/last ViT block to localize where the explosion starts.
        if re.search(r"vision_model\.(blocks|layers)\.0$", name):
            mod.register_forward_hook(mk_hook("vblock0"))
            print(f"[HFDBG] hooked vblock0: {name}")
        if re.search(r"vision_model\.(blocks|layers)\.(31|30|\d+)$", name):
            # keep updating -> ends on the highest index = last block
            mod.register_forward_hook(mk_hook("vblock_last"))
    if layers is not None:
        layers[0].register_forward_hook(mk_hook("L0"))
        if nlayer > 1:
            layers[1].register_forward_hook(mk_hook("L1"))
            # Layer-1 MoE internals: input (post input_layernorm), shared-expert
            # output, full MoE output -> compare to our moe.input / moe.shared.
            moe1 = getattr(layers[1], "mlp", None)
            if moe1 is not None:
                moe1.register_forward_pre_hook(
                    lambda m, i: captures.__setitem__("moe1_in", i[0]))
                moe1.register_forward_hook(
                    lambda m, i, o: captures.__setitem__(
                        "moe1_out", o[0] if isinstance(o, tuple) else o))
                se1 = getattr(moe1, "shared_experts", None)
                if se1 is not None:
                    se1.register_forward_hook(mk_hook("moe1_shared"))

    # Plain single forward (no generate -> avoids the remote code's legacy-cache
    # past_key_values[0][0] assumption). use_cache=False keeps it a pure prefill.
    logits = None
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float().cpu()
    except Exception as e:  # noqa: BLE001
        print(f"[HFDBG] forward failed: {e}")

    # vision runs before the backbone, so these are captured even if the backbone
    # crashes. vision_model_raw/vision_forward_out are [1024,1280] (pre-merge ViT);
    # resampler:* outputs are post-merge [256,2560] == our vl.vision_embeds.
    stats("patch_embed", captures.get("patch_embed"))
    stats("vblock0", captures.get("vblock0"))
    stats("vblock_last", captures.get("vblock_last"))
    stats("vision_model_raw", captures.get("vision_model_raw"))
    stats("vision_forward_out", captures.get("vision_forward_out"))
    for key in sorted(captures):
        if key.startswith("resampler:"):
            stats(key, captures[key])
    stats("L0 stream", captures.get("L0"))
    stats("moe1_in", captures.get("moe1_in"))
    stats("moe1_shared", captures.get("moe1_shared"))
    stats("moe1_out", captures.get("moe1_out"))
    stats("L1 stream", captures.get("L1"))
    if logits is not None:
        top = torch.topk(logits, 5)
        print("[HFDBG] logits min={:.3f} max={:.3f}".format(logits.min(), logits.max()))
        print("[HFDBG] top5:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
