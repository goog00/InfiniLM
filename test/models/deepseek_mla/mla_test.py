import os
import sys
import time
import torch
import torch.nn.functional as F

def _maybe_add_repo_root_to_syspath():
    """Make `test.models...` importable for both `python .../mla_test.py` and `python -m ...`.

    Under schedulers (e.g. `srun`) the working directory can differ; this function ensures
    we still find the InfiniLM repo root.
    """

    def looks_like_repo_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, "pyproject.toml")) and os.path.exists(
            os.path.join(path, "test")
        )

    candidates = []
    # 1) Directory derived from this file location
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    # 2) Current working directory (common when running from repo root)
    candidates.append(os.path.abspath(os.getcwd()))
    # 3) Walk up from cwd a few levels
    cur = os.path.abspath(os.getcwd())
    for _ in range(6):
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    for cand in candidates:
        if looks_like_repo_root(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
            return


_maybe_add_repo_root_to_syspath()


# --------------------------------------------------------------------------------------
# Self-contained PyTorch reference MLA (for correctness baseline)
# --------------------------------------------------------------------------------------


class DeepSeekV3Config:
    """Minimal config needed by the reference MLA.

    Defaults match the prototype previously used in this repo; override from config.json when available.
    """

    def __init__(self, model_path: str | None = None):
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 16
        self.rope_dim = 64
        self.nope_dim = 128
        self.q_lora_rank = 128
        self.kv_lora_rank = 128
        self.qk_head_dim = self.rope_dim + self.nope_dim
        self.v_head_dim = 128
        self.rms_norm_eps = 1e-6

        if model_path:
            self.load_from_json(model_path)

    def load_from_json(self, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"Warning: config.json not found at {config_path}; using defaults.")
            return
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.hidden_size = cfg.get("hidden_size", self.hidden_size)
        self.num_heads = cfg.get("num_attention_heads", self.num_heads)
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_kv_heads)
        self.rope_dim = cfg.get("rope_dim", self.rope_dim)
        self.nope_dim = cfg.get("nope_dim", self.nope_dim)
        self.q_lora_rank = cfg.get("q_lora_rank", self.q_lora_rank)
        self.kv_lora_rank = cfg.get("kv_lora_rank", self.kv_lora_rank)
        self.v_head_dim = cfg.get("v_head_dim", self.v_head_dim)
        self.rms_norm_eps = cfg.get("rms_norm_eps", self.rms_norm_eps)
        self.qk_head_dim = self.rope_dim + self.nope_dim


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (xf * self.weight.float()).to(dtype)


class DeepSeekV3MLA(torch.nn.Module):
    """Reference MLA with kv_cache + pe_cache.

    This is for correctness/perf baseline only (not optimized).
    Cache shapes:
    - kv_cache: (bs, past_len, r_kv)
    - pe_cache: (bs, past_len, d_rope)
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.d = config.hidden_size
        self.nh = config.num_heads
        self.d_rope = config.rope_dim
        self.d_nope = config.nope_dim
        self.r_q = config.q_lora_rank
        self.r_kv = config.kv_lora_rank
        self.d_qk = config.qk_head_dim
        self.d_v = config.v_head_dim

        self.mla_norm = RMSNorm(self.d, config.rms_norm_eps)
        self.q_a_norm = RMSNorm(self.r_q, config.rms_norm_eps)
        self.kv_a_norm = RMSNorm(self.r_kv, config.rms_norm_eps)

        self.q_a_proj = torch.nn.Linear(self.d, self.r_q, bias=False)
        self.q_b_proj = torch.nn.Linear(self.r_q, self.nh * self.d_qk, bias=False)
        self.kv_a_proj = torch.nn.Linear(self.d, self.r_kv + self.d_rope, bias=False)
        self.kv_b_proj = torch.nn.Linear(self.r_kv, self.nh * (self.d_nope + self.d_v), bias=False)
        self.o_proj = torch.nn.Linear(self.nh * self.d_v, self.d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv_cache: torch.Tensor | None,
        past_pe_cache: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        batch_size, seq_len, _ = x.shape
        norm_x = self.mla_norm(x)

        q_a = self.q_a_proj(norm_x)
        q_a = self.q_a_norm(q_a)
        q = self.q_b_proj(q_a).view(batch_size, seq_len, self.nh, self.d_qk)

        q_nope = q[..., : self.d_nope]
        q_rope = q[..., self.d_nope :]
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q_rope = apply_rotary_emb(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv_a = self.kv_a_proj(norm_x)
        kv_pass = kv_a[..., : self.r_kv]
        k_rot = kv_a[..., self.r_kv :]
        kv_pass = self.kv_a_norm(kv_pass)
        if freqs_cis is not None:
            cos, sin = freqs_cis
            k_rot = apply_rotary_emb(k_rot.unsqueeze(2), cos, sin).squeeze(2)

        if past_kv_cache is not None:
            kv_pass = torch.cat([past_kv_cache, kv_pass], dim=1)
        if past_pe_cache is not None:
            k_rot = torch.cat([past_pe_cache, k_rot], dim=1)

        total_seq_len = kv_pass.shape[1]
        current_kv_cache = kv_pass
        current_pe_cache = k_rot

        kv_b = self.kv_b_proj(kv_pass).view(batch_size, total_seq_len, self.nh, self.d_nope + self.d_v)
        k_nope = kv_b[..., : self.d_nope]
        v = kv_b[..., self.d_nope :]

        k_rot_expanded = k_rot.unsqueeze(2).expand(-1, -1, self.nh, -1)
        k = torch.cat([k_nope, k_rot_expanded], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_qk**0.5)
        if seq_len > 1:
            mask = torch.triu(
                torch.ones(seq_len, total_seq_len, device=x.device), diagonal=1 + (total_seq_len - seq_len)
            ).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.nh * self.d_v)
        out = self.o_proj(out)
        return out, current_kv_cache, current_pe_cache

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

    cfg = DeepSeekV3Config(args.model_path)
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
