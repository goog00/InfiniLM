import time
import torch
import argparse
import sys

WARMUPS = 10
RUNS = 100

PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256]}
DECODE_TESTCASES = {"seqlens": [1 for _ in range(16)]}


def get_args():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 MoE test scaffold")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--use_c_api", action="store_true", help="Use C API implementation via scripts/deepseek.py")
    parser.add_argument("--model_path", type=str, default=None, help="Path to DeepSeek-V3 model directory (required if --use_c_api)")
    return parser.parse_args()


def torch_synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


class ProxyMoE(torch.nn.Module):
    """A small proxy MoE block for benchmarking and correctness scaffold.

    This is NOT a drop-in DeepSeek-V3 implementation. It mimics shared expert
    + top-k routing + per-expert FFN to provide a baseline harness.
    """

    def __init__(self, hidden_dim=2048, di_moe=4096, nexperts=256, topk=8, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.di_moe = di_moe
        self.nexperts = nexperts
        self.topk = topk
        self.device = device

        self.shared_up = torch.nn.Linear(hidden_dim, di_moe).to(device=device, dtype=dtype)
        self.shared_down = torch.nn.Linear(di_moe, hidden_dim).to(device=device, dtype=dtype)

        # gate projection
        self.gate = torch.nn.Linear(hidden_dim, nexperts).to(device=device, dtype=dtype)

        # small set of experts (for proxy we create fewer experts to save memory)
        self.nlocal = min(16, nexperts)
        self.experts_up = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, di_moe).to(device=device, dtype=dtype) for _ in range(self.nlocal)])
        self.experts_down = torch.nn.ModuleList([torch.nn.Linear(di_moe, hidden_dim).to(device=device, dtype=dtype) for _ in range(self.nlocal)])

    def forward(self, x):
        # x: (1, total_seqlen, hidden_dim)
        batch, seqlen, _ = x.shape
        tokens = x.view(-1, self.hidden_dim)  # (T, D)

        # shared expert
        shared = self.shared_down(torch.nn.functional.silu(self.shared_up(tokens)))

        # gating
        logits = self.gate(tokens)
        topv, topi = torch.topk(logits, k=self.topk, dim=1)
        weights = torch.softmax(topv, dim=1)

        # naive per-token per-expert compute (proxy of current C++ serial strategy)
        router_sum = torch.zeros_like(tokens)
        for t in range(tokens.shape[0]):
            for k in range(self.topk):
                expert_id = int(topi[t, k].item()) % self.nlocal
                w = float(weights[t, k].item())
                tmp = self.experts_down[expert_id](torch.nn.functional.silu(self.experts_up[expert_id](tokens[t : t + 1])))
                router_sum[t : t + 1] += w * tmp

        out = tokens + shared + router_sum
        return out.view(batch, seqlen, -1)


def run_benchmark(model, testcase, device, dtype, warmups, runs):
    total_seqlen = sum(testcase["seqlens"]) if isinstance(testcase, dict) else testcase
    inp = torch.rand((1, total_seqlen, model.hidden_dim), dtype=dtype, device=(device if device == "cuda" else "cpu"))

    # warmup
    for _ in range(warmups):
        model(inp)
    torch_synchronize(device)

    start = time.time()
    for _ in range(runs):
        model(inp)
    torch_synchronize(device)
    end = time.time()

    total_time = end - start
    avg_ms = total_time * 1000.0 / runs
    total_tokens = total_seqlen * runs
    tps = total_tokens / total_time
    print(f"WARMUPS={warmups} RUNS={runs} avg latency: {avg_ms:.2f} ms  throughput: {tps:.2f} tok/s")


if __name__ == "__main__":
    args = get_args()

    device = args.device
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]

    print("DeepSeek-V3 MoE test scaffold")
    print(f"Device: {device}, dtype: {args.dtype}, hidden_dim: {args.hidden_dim}")

    if args.use_c_api:
        if args.model_path is None:
            print("--model_path is required when --use_c_api is set")
            sys.exit(1)
        # Ensure scripts dir is importable
        import os

        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        scripts_dir = os.path.normpath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from deepseek import DeepSeekV3ForCauslLM
        from libinfinicore_infer import DeviceType

        device_type = DeviceType.DEVICE_TYPE_NVIDIA if args.device == "cuda" else DeviceType.DEVICE_TYPE_CPU
        model = DeepSeekV3ForCauslLM(args.model_path, device=device_type, ndev=1)

        # build synthetic tasks for prefill scenario
        from infer_task import InferTask, KVCache

        def build_tasks_from_seqlens(seqlens):
            tasks = []
            for i, sl in enumerate(seqlens):
                tokens = [0] * sl
                t = InferTask(i, tokens, model.max_context_len(), 1.0, 1, 1.0, model.eos_token_id)
                t.bind_kvcache(KVCache(model))
                tasks.append(t)
            return tasks

        print("\n== Prefill scenario (C API) ==")
        tasks = build_tasks_from_seqlens([64, 128, 256, 256])
        # warmup
        for _ in range(args.warmups):
            model.batch_infer_one_round(tasks)
        import time
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.runs):
            model.batch_infer_one_round(tasks)
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print(f"avg latency: {total_time * 1000.0 / args.runs:.2f} ms")

        print("\n== Decode scenario (C API) ==")
        tasks = build_tasks_from_seqlens([1] * 16)
        for _ in range(args.warmups):
            model.batch_infer_one_round(tasks)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.runs):
            model.batch_infer_one_round(tasks)
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print(f"avg latency: {total_time * 1000.0 / args.runs:.2f} ms")

        model.destroy_model_instance()
    else:
        # build proxy model
        model = ProxyMoE(hidden_dim=args.hidden_dim, device=device, dtype=dtype)

        print("\n== Prefill scenario ==")
        run_benchmark(model, PREFILL_TESTCASES, device, dtype, args.warmups, args.runs)

        print("\n== Decode scenario ==")
        run_benchmark(model, DECODE_TESTCASES, device, dtype, args.warmups, args.runs)

        print("\nNote: This is a proxy harness. To compare against the native DeepSeek-V3 C++ implementation, run with --use_c_api --model_path=<model_dir>")
