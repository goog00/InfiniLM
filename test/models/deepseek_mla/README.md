# DeepSeek MLA tests

This folder contains correctness + performance tests for the DeepSeek-style MLA module.

## What it measures


## Run (NVIDIA)

```bash
 python -m test.models.deepseek_mla.mla_test --nvidia --model_path /path/to/DeepSeek-R1

# Alternatively (must run from repo root)
 python test/models/deepseek_mla/mla_test.py --nvidia --model_path /path/to/DeepSeek-R1
```

## Notes
- The `build_mla_under_test()` hook in `mla_test.py` is the integration point to call your optimized InfiniLM MLA implementation once available.
