# DeepSeek MLA tests

This folder contains correctness + performance tests for the DeepSeek-style MLA module.

## What it measures

- **Prefill (small batch):** 4 requests with `seqlens=[64,128,256,256]` and `pastlens=[512,0,0,256]`. Same shapes run for 100 rounds, report average latency per round.
- **Decode (large batch):** 16 requests with `seqlen=1` and `pastlens=[50,100,200,400] * 4`. Run sequentially for 100 steps with growing cache; each output is used as next input; report average latency per generated token.

## Run (NVIDIA)

```bash
python -m test.models.deepseek_mla.mla_test --nvidia --model_path /path/to/DeepSeek-R1
```

## Notes

- Current script uses the in-repo PyTorch reference MLA from `mla_test1.py`.
- The `build_mla_under_test()` hook in `mla_test.py` is the integration point to call your optimized InfiniLM MLA implementation once available.
