import os
import sys
import time
import torch
import torch.nn.functional as F

# When invoked as `python test/.../mla_test.py` (not `-m`),
# Python won't treat `test` as an importable top-level package.
# Ensure repo root is on sys.path for consistent imports.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep consistent with existing attention tests
WARMUPS = 10
RUNS = 100

PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}
DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek MLA correctness + perf test")
    parser.add_argument("--model_path", action="store", help="Path to DeepSeek-R1/DeepSeek-V3 style model")

    parser.add_argument("--cpu", action="store_true", help="Run cpu test")
    parser.add_argument("--nvidia", action="store_true", help="Run nvidia test")
    parser.add_argument("--metax", action="store_true", help="Run metax test")
    parser.add_argument("--moore", action="store_true", help="Run moore test")
    parser.add_argument("--iluvatar", action="store_true", help="Run iluvatar test")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_correctness", action="store_true", help="Skip correctness checks")
    parser.add_argument("--no_perf", action="store_true", help="Skip performance benchmarks")
    return parser.parse_args()


def torch_synchronize(_device: str):
    if _device == "cuda":
        torch.cuda.synchronize()
    elif _device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(_device: str):
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "musa":
        torch.musa.empty_cache()


def resolve_device(args) -> str:
    if args.cpu:
        return "cpu"
    if args.nvidia:
        return "cuda"
    if args.metax:
        return "cuda"
    if args.moore:
        import torch_musa  # noqa: F401

        return "musa"
    if args.iluvatar:
        return "cuda"

    print(
        "Usage: python test/models/deepseek_mla/mla_test.py [--cpu|--nvidia|--metax|--moore|--iluvatar] --model_path=..."
    )
    sys.exit(1)


def resolve_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------------------
# Reference MLA (PyTorch) + shape contracts
# --------------------------------------------------------------------------------------


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


def get_freqs_cis(seq_len, dim, device, start_pos=0):
    freqs = torch.arange(0, dim, 2, device=device).float() / dim
    freqs = 1.0 / (10000 ** freqs)
    t = torch.arange(start_pos, start_pos + seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def mla_reference_forward(mla_ref, hidden_states, *, kv_cache, pe_cache, start_pos):
    """A thin wrapper around `mla_ref` to standardize cache usage.

    Returns: (out, new_kv_cache, new_pe_cache)
    """
    bs, seq_len, _ = hidden_states.shape

    # rotary only applies to rope_dim
    cos, sin = get_freqs_cis(seq_len, mla_ref.d_rope, hidden_states.device, start_pos=start_pos)
    freqs_cis = (cos, sin)

    out, new_kv, new_pe = mla_ref(
        hidden_states,
        past_kv_cache=kv_cache,
        past_pe_cache=pe_cache,
        freqs_cis=freqs_cis,
    )
    return out, new_kv, new_pe


# --------------------------------------------------------------------------------------
# Under-test implementation (InfiniLM)
# --------------------------------------------------------------------------------------


def maybe_import_infinilm():
    # Keep same approach as mla_test1.py: add repo root then import scripts binding
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        from scripts.libinfinicore_infer.deepseek_v3 import DeepSeekV3Model

        return DeepSeekV3Model
    except Exception as e:
        print(f"Warning: InfiniLM DeepSeekV3Model not available: {e}")
        return None


def build_mla_under_test(*, model_path: str, device: str, dtype):
    """Placeholder: build MLA module under test.

    TODO: replace with your real MLA/engine wrapper once MLA kernel is integrated.
    """
    DeepSeekV3Model = maybe_import_infinilm()
    if DeepSeekV3Model is None:
        return None

    # NOTE: We intentionally do not instantiate the full model here yet.
    # The final implementation should expose a callable attention op with (hidden, kv_cache, pe_cache, start_pos).
    return None


# --------------------------------------------------------------------------------------
# Input generation for the two scenarios
# --------------------------------------------------------------------------------------


def generate_mla_inputs(testcase, *, hidden_size: int, device: str, dtype):
    bs = 1
    req_list = []
    for seq_len, past_len in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.rand((bs, seq_len, hidden_size), device=device, dtype=dtype)

        # kv_cache + pe_cache are random-initialized by requirement
        # Here we store *latent kv* cache for the reference module based on its config (r_kv)
        # and *pe cache* as rope part (d_rope).
        req_list.append(
            {
                "hidden_states": hidden_states,
                "past_len": past_len,
            }
        )
    return req_list


# --------------------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------------------


def compare_tensors(a: torch.Tensor, b: torch.Tensor, *, name: str, atol: float, rtol: float):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    abs_max = (a - b).abs().max().item()
    abs_mean = (a - b).abs().mean().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"[check] {name}: allclose={ok} max_abs={abs_max:.6g} mean_abs={abs_mean:.6g} (atol={atol}, rtol={rtol})")
    return ok


def run_correctness(mla_ref, mla_uut, device: str, dtype):
    # For BF16, start with tolerant thresholds; tighten if needed.
    atol, rtol = (3e-2, 1e-2) if dtype in (torch.bfloat16, torch.float16) else (1e-4, 1e-4)

    print("\n" + "*" * 110)
    print("Correctness: PREFILL scenario")
    print("*" * 110)

    req_list = generate_mla_inputs(PREFILL_TESTCASES, hidden_size=mla_ref.d, device=device, dtype=dtype)

    all_ok = True
    for idx, req in enumerate(req_list):
        hidden_states = req["hidden_states"]
        past_len = req["past_len"]

        # Random init caches
        kv_cache = torch.rand((1, past_len, mla_ref.r_kv), device=device, dtype=dtype)
        pe_cache = torch.rand((1, past_len, mla_ref.d_rope), device=device, dtype=dtype)

        ref_out, _, _ = mla_reference_forward(mla_ref, hidden_states, kv_cache=kv_cache, pe_cache=pe_cache, start_pos=past_len)

        if mla_uut is None:
            print("[skip] Under-test MLA not wired yet; correctness only runs reference.")
            continue

        uut_out = mla_uut(hidden_states, kv_cache=kv_cache, pe_cache=pe_cache, start_pos=past_len)
        all_ok &= compare_tensors(uut_out, ref_out, name=f"prefill_req{idx}", atol=atol, rtol=rtol)

    if mla_uut is not None and not all_ok:
        raise SystemExit("Correctness check failed")


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------


def benchmark_prefill(mla_callable, mla_ref, device: str, dtype):
    print("\n" + "*" * 110)
    print(f"Perf: PREFILL (WARMUPS={WARMUPS}, RUNS={RUNS})")
    print("*" * 110)

    req_list = generate_mla_inputs(PREFILL_TESTCASES, hidden_size=mla_ref.d, device=device, dtype=dtype)

    # Build per-request cache tensors once; crop/reset by slicing for reference.
    kv_caches = []
    pe_caches = []
    for req in req_list:
        past_len = req["past_len"]
        kv_caches.append(torch.rand((1, past_len, mla_ref.r_kv), device=device, dtype=dtype))
        pe_caches.append(torch.rand((1, past_len, mla_ref.d_rope), device=device, dtype=dtype))

    if mla_callable is None:
        print("[skip] Under-test MLA not wired yet; perf uses reference implementation.")
        mla_callable = lambda hidden_states, kv_cache, pe_cache, start_pos: mla_reference_forward(
            mla_ref, hidden_states, kv_cache=kv_cache, pe_cache=pe_cache, start_pos=start_pos
        )[0]

    # warmup
    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            _ = mla_callable(req["hidden_states"], kv_cache=kv_caches[i], pe_cache=pe_caches[i], start_pos=req["past_len"])

    torch_synchronize(device)

    time_consuming = 0.0
    for _ in range(RUNS):
        torch_synchronize(device)
        start = time.time()

        for i, req in enumerate(req_list):
            _ = mla_callable(req["hidden_states"], kv_cache=kv_caches[i], pe_cache=pe_caches[i], start_pos=req["past_len"])

        torch_synchronize(device)
        end = time.time()
        time_consuming += end - start

    avg_ms = (time_consuming * 1000.0) / RUNS
    print(f"PREFILL avg latency/round: {avg_ms:.3f} ms (1 round = 4 req)")


def benchmark_decode(mla_callable, mla_ref, device: str, dtype):
    print("\n" + "*" * 110)
    print(f"Perf: DECODE (WARMUPS={WARMUPS}, RUNS={RUNS})")
    print("*" * 110)

    req_list = generate_mla_inputs(DECODE_TESTCASES, hidden_size=mla_ref.d, device=device, dtype=dtype)

    # Initialize growing caches
    kv_caches = []
    pe_caches = []
    start_pos = []
    for req in req_list:
        past_len = req["past_len"]
        kv_caches.append(torch.rand((1, past_len, mla_ref.r_kv), device=device, dtype=dtype))
        pe_caches.append(torch.rand((1, past_len, mla_ref.d_rope), device=device, dtype=dtype))
        start_pos.append(past_len)

    if mla_callable is None:
        print("[skip] Under-test MLA not wired yet; perf uses reference implementation.")
        mla_callable = lambda hidden_states, kv_cache, pe_cache, start_pos: mla_reference_forward(
            mla_ref, hidden_states, kv_cache=kv_cache, pe_cache=pe_cache, start_pos=start_pos
        )[0]

    # warmup (do not grow caches; just run a few)
    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            _ = mla_callable(req["hidden_states"], kv_cache=kv_caches[i], pe_cache=pe_caches[i], start_pos=start_pos[i])

    # reset input tokens to something deterministic-ish
    for i in range(len(req_list)):
        req_list[i]["hidden_states"] = torch.rand_like(req_list[i]["hidden_states"])

    torch_synchronize(device)
    start = time.time()

    total_tokens = 0
    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            out = mla_callable(req["hidden_states"], kv_cache=kv_caches[i], pe_cache=pe_caches[i], start_pos=start_pos[i])

            # output becomes next token input; cache grows by 1
            req["hidden_states"] = out
            start_pos[i] += 1
            total_tokens += 1

    torch_synchronize(device)
    end = time.time()

    total_time = end - start
    ms_per_token = (total_time * 1000.0) / total_tokens
    tok_per_s = total_tokens / total_time

    print(f"DECODE total_tokens={total_tokens}, total_time={total_time:.4f}s")
    print(f"DECODE avg latency/token: {ms_per_token:.4f} ms")
    print(f"DECODE throughput: {tok_per_s:.2f} tok/s")


def main():
    args = get_args()
    device = resolve_device(args)
    dtype = resolve_dtype(args.dtype)

    torch.manual_seed(args.seed)

    # Import your reference MLA from existing mla_test1.py to keep consistent with in-repo prototype
    try:
        from test.models.deepseek_mla.mla_test1 import DeepSeekV3Config, DeepSeekV3MLA
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Failed to import reference MLA from test.models.deepseek_mla.mla_test1: {e}. "
            "Run from repo root or use `python -m test.models.deepseek_mla.mla_test ...`."
        )

    cfg = DeepSeekV3Config(args.model_path)
    cfg.device = device
    cfg.dtype = dtype

    mla_ref = DeepSeekV3MLA(cfg).to(device=device, dtype=dtype)
    mla_ref.eval()

    # Under-test MLA: wire up later
    mla_uut = build_mla_under_test(model_path=args.model_path, device=device, dtype=dtype)

    if not args.no_correctness:
        run_correctness(mla_ref, mla_uut, device=device, dtype=dtype)

    if not args.no_perf:
        benchmark_prefill(mla_uut, mla_ref, device=device, dtype=dtype)
        benchmark_decode(mla_uut, mla_ref, device=device, dtype=dtype)

    torch_empty_cache(device)


if __name__ == "__main__":
    main()
