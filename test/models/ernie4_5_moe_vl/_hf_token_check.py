"""TEMPORARY: teacher-forcing token-level check of our ERNIE-4.5-VL image
generation against HF. DELETE BEFORE SUBMISSION.

Feeds (prompt + our generated token_ids) into HF in one plain forward
(use_cache=False) and checks HF's greedy argmax at each step equals our next
token. For greedy decoding this is equivalent to "HF greedy would reproduce
our sequence exactly" (induction over steps).

Usage:
  python test/models/ernie4_5_moe_vl/_hf_token_check.py \
      --model $MODEL --image /tmp/test.jpg --text "描述这张图片。" \
      --gen-ids 3843,1510,1386,...
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", default="描述这张图片。")
    ap.add_argument("--gen-ids", required=True, help="comma-separated generated token ids")
    args = ap.parse_args()
    gen_ids = [int(x) for x in args.gen_ids.split(",") if x.strip() != ""]

    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ip = getattr(processor, "image_processor", None)

    img = Image.open(args.image).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.text}]}]
    rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=rendered, images=[img], return_tensors="pt")

    prompt_ids = inputs["input_ids"][0].tolist()
    S = len(prompt_ids)
    print(f"[CHK] HF prompt len={S} first8={prompt_ids[:8]} last8={prompt_ids[-8:]}")
    print(f"[CHK] gen tokens ({len(gen_ids)}): {gen_ids}")

    # Normalize images (CLIP mean/std), matching our processor.
    imgs = inputs["images"]
    if imgs.dtype == torch.uint8 or imgs.float().max() > 10:
        mean = torch.tensor(getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073]))
        std = torch.tensor(getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711]))
        resc = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
        N = imgs.shape[0]
        x = imgs.float().view(N, 3, -1)
        x = (x * resc - mean.view(1, 3, 1)) / std.view(1, 3, 1)
        inputs["images"] = x.view(N, -1).to(torch.bfloat16)

    G = len(gen_ids)
    inputs["input_ids"] = torch.tensor([prompt_ids + gen_ids], dtype=torch.long)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones((1, S + G), dtype=inputs["attention_mask"].dtype)

    # Extend 3D position_ids: generated text tokens continue all 3 axes from max+1.
    # The processor returns [batch, seq, 3] (3 = time/height/width per token); also
    # tolerate the transposed [3, seq] layout.
    pos = inputs["position_ids"]
    bdims = pos.dim() - 2
    p = pos
    for _ in range(bdims):
        p = p[0]            # strip leading batch dim(s) -> 2D
    maxp = int(p.max().item())
    if tuple(p.shape) == (S, 3):          # [seq, axes]
        ext = torch.arange(maxp + 1, maxp + 1 + G).view(G, 1).repeat(1, 3).to(p.dtype)
        p_full = torch.cat([p, ext], dim=0)              # [S+G, 3]
        inputs["position_ids"] = p_full.view(*([1] * bdims), S + G, 3)
    elif tuple(p.shape) == (3, S):        # [axes, seq]
        ext = torch.arange(maxp + 1, maxp + 1 + G).view(1, G).repeat(3, 1).to(p.dtype)
        p_full = torch.cat([p, ext], dim=1)              # [3, S+G]
        inputs["position_ids"] = p_full.view(*([1] * bdims), 3, S + G)
    else:
        raise AssertionError(f"unexpected position_ids shape {tuple(pos.shape)}")
    print(f"[CHK] prompt max pos={maxp}; gen positions {maxp + 1}..{maxp + G} (3 axes equal)")

    # token_type_ids: gen tokens are text(0); the model wants len == seq + 1.
    tt = inputs.get("token_type_ids")
    if tt is not None:
        base = tt[0].tolist()[:S]
        vis_pos = [i for i, v in enumerate(base) if v != 0]
        nvision = len(vis_pos)
        if vis_pos:
            lo, hi = vis_pos[0], vis_pos[-1]
            gaps = [vis_pos[i] for i in range(1, len(vis_pos)) if vis_pos[i] != vis_pos[i - 1] + 1]
            print(f"[CHK] HF vision span: count={nvision} idx=[{lo}..{hi}] contiguous={not gaps} "
                  f"ids: before={prompt_ids[lo-1] if lo>0 else None} first={prompt_ids[lo]} "
                  f"last={prompt_ids[hi]} after={prompt_ids[hi+1] if hi+1<S else None}")
        inputs["token_type_ids"] = torch.tensor([base + [0] * G + [0]], dtype=tt.dtype)
        print(f"[CHK] token_type_ids len={S + G + 1} prompt_nvision={nvision} (our impl marks 256)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    backbone = model.model if hasattr(model, "model") else model

    # Bypass SDPA flash (no MetaX kernel) -> eager core_attn, like the text path.
    layers = getattr(backbone, "layers", None)
    if layers is not None:
        for lyr in layers:
            attn = getattr(lyr, "self_attn", None)
            if attn is not None and hasattr(attn, "core_attn"):
                attn.attn_func = attn.core_attn

    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0].float().cpu()       # [S+G, vocab]

    # Teacher-forcing: logits at position (S-1+j) predict gen token j.
    mism = 0
    for j in range(G):
        pos_idx = S - 1 + j
        hf_tok = int(logits[pos_idx].argmax())
        ours = gen_ids[j]
        if hf_tok != ours:
            mism += 1
            t5 = torch.topk(logits[pos_idx], 5)
            print(f"[CHK] MISMATCH @gen{j}: ours={ours} hf_argmax={hf_tok} "
                  f"ours_logit={float(logits[pos_idx][ours]):.2f} "
                  f"hf_top5={[(int(i), round(float(v), 2)) for v, i in zip(t5.values, t5.indices)]}")
    print(f"[CHK] RESULT: {G - mism}/{G} match HF greedy argmax"
          + ("  -> EXACT MATCH" if mism == 0 else f"  ({mism} mismatches)"))


if __name__ == "__main__":
    main()
