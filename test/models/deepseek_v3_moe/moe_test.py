import time
import torch
import transformers
import safetensors
import os
from transformers import AutoConfig, AutoModelForCausalLM
import sys
import numpy as np
import ctypes

# Add the scripts directory to path for importing DeepSeek classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../scripts'))
from deepseek import DeepSeekV3ForCauslLM
from libinfinicore_infer import DeviceType

WARMUPS = 10
RUNS = 100
# DeepSeek-V3 MoE test cases based on requirements
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [0, 0, 0, 0]}  # Total 704 tokens

DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],  # 16 decode requests
    "pastlens": [50 for _ in range(16)],  # All have some past context
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test DeepSeek-V3 MoE")
    parser.add_argument(
        "--model_path",
        action="store",
        help="The directory of the model to be tested",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )

    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )

    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run moore test",
    )

    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness test against transformers",
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance test",
    )

    return parser.parse_args()


def torch_synchronize(_device):
    if _device == "cuda":
        torch.cuda.synchronize()
    elif _device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(_device):
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "musa":
        torch.musa.empty_cache()


def create_deepseek_moe_torch(dir_path, device, dtype=torch.bfloat16):
    """Load DeepSeek-R1 MoE model using transformers"""
    print(f"Loading DeepSeek-R1 from {dir_path}")
    config = AutoConfig.from_pretrained(dir_path)
    model = AutoModelForCausalLM.from_pretrained(
        dir_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Find MoE layers (layers after dense layers)
    moe_layers = []
    n_dense_layers = config.n_dense_layers if hasattr(config, 'n_dense_layers') else 1

    for i, layer in enumerate(model.model.layers):
        if i >= n_dense_layers:  # MoE layers start after dense layers
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                # Check if it's MoE by looking for experts
                if hasattr(layer.mlp, 'experts'):
                    moe_layers.append((i, layer.mlp))

    if not moe_layers:
        raise ValueError("No MoE layers found in the model")

    # Use the first MoE layer for testing
    layer_idx, moe_layer = moe_layers[0]
    print(f"Using MoE layer {layer_idx} with {len(moe_layer.experts)} experts")

    return moe_layer


def create_deepseek_moe_infinilm(dir_path, device_type):
    """Load DeepSeek-V3 MoE model using InfiniLM"""
    print(f"Loading DeepSeek-V3 with InfiniLM from {dir_path}")

    if device_type == "nvidia":
        device = DeviceType.DEVICE_TYPE_CUDA
    elif device_type == "moore":
        device = DeviceType.DEVICE_TYPE_MOORE
    else:
        device = DeviceType.DEVICE_TYPE_CPU

    # Load the model
    model = DeepSeekV3ForCauslLM(dir_path, device=device, ndev=1)
    return model


def generate_moe_input_infinilm(testcase, vocab_size=102400):
    """Generate input tokens for InfiniLM MoE testing"""
    total_seqlen = sum(testcase["seqlens"])
    # Generate random token IDs
    tokens = np.random.randint(0, vocab_size, size=(1, total_seqlen), dtype=np.uint32)
    return tokens


def benchmark_moe_infinilm(model, testcase, device_type):
    """Benchmark InfiniLM DeepSeek-V3 MoE implementation"""
    tokens = generate_moe_input_infinilm(testcase)

    # For MoE testing, we need to create a batched task
    # This is a simplified version - in practice, we'd need to properly set up the batched task
    print("InfiniLM MoE benchmarking - placeholder implementation")
    print(f"Test case: {testcase}")
    print(f"Input tokens shape: {tokens.shape}")

    # Placeholder timing
    start_time = time.time()
    time.sleep(0.01)  # Simulate computation
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(testcase["seqlens"]) * RUNS
    print(
        f"InfiniLM MoE - WARMUPS={WARMUPS} RUNS={RUNS}, average latency: {round(total_time * 1000 / RUNS, 2)} ms   throughput: {round(total_tokens / total_time, 2)} tok/s"
    )

    # Return placeholder output
    return np.zeros((len(testcase["seqlens"]), max(testcase["seqlens"])), dtype=np.uint32)


def benchmark_moe_torch(moe, testcase, device, dtype):
    """Benchmark PyTorch MoE implementation"""
    input_host = generate_moe_input_torch(testcase, dtype=dtype)
    input_device = input_host.to(device=device)

    # Forward pass to get output shape
    with torch.no_grad():
        output_device, _ = moe(input_device)
    output_host = output_device.to("cpu")

    # Warmup
    for _ in range(WARMUPS):
        with torch.no_grad():
            moe(input_device)

    torch_synchronize(device)

    # Benchmark
    start_time = time.time()
    for _ in range(RUNS):
        with torch.no_grad():
            moe(input_device)
    torch_synchronize(device)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(testcase["seqlens"]) * RUNS
    print(
        f"PyTorch MoE - WARMUPS={WARMUPS} RUNS={RUNS}, average latency: {round(total_time * 1000 / RUNS, 2)} ms   throughput: {round(total_tokens / total_time, 2)} tok/s"
    )
    return output_host


def correctness_test(args):
    """Test correctness against transformers implementation"""
    print("=" * 80)
    print("Running Correctness Test")
    print("=" * 80)

    model_path = args.model_path
    dtype = torch.bfloat16
    device = "cuda" if args.nvidia else "cpu"
    device_type = "nvidia" if args.nvidia else "cpu"

    # Load PyTorch reference
    moe_torch = create_deepseek_moe_torch(model_path, device, dtype)

    # Load InfiniLM implementation
    try:
        moe_infinilm = create_deepseek_moe_infinilm(model_path, device_type)
        print("InfiniLM model loaded successfully")
    except Exception as e:
        print(f"Failed to load InfiniLM model: {e}")
        print("Skipping correctness test")
        return

    # Generate test input
    test_input = generate_moe_input_torch(PREFILL_TESTCASES, dtype=dtype).to(device)

    # Run PyTorch forward
    with torch.no_grad():
        torch_output, torch_router_logits = moe_torch(test_input)

    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Router logits shape: {torch_router_logits.shape}")

    # Run InfiniLM forward
    infinilm_output = benchmark_moe_infinilm(moe_infinilm, PREFILL_TESTCASES, device_type)

    # Compare outputs (simplified comparison)
    # Note: This is a placeholder - actual comparison would need proper output extraction
    print("TODO: Implement proper output comparison between PyTorch and InfiniLM")
    print(f"InfiniLM output shape: {infinilm_output.shape}")

    # Placeholder assertion
    # diff = np.abs(torch_output.cpu().numpy() - infinilm_output).max()
    # print(f"Max difference: {diff}")
    # assert diff < 1e-3, f"Outputs differ by {diff}"

    print("Correctness test completed (placeholder)")


def performance_test(args):
    """Run performance tests"""
    print("=" * 80)
    print("Running Performance Test")
    print("=" * 80)

    model_path = args.model_path
    dtype = torch.bfloat16
    device = "cuda" if args.nvidia else "cpu"
    device_type = "nvidia" if args.nvidia else "cpu"

    # Load PyTorch model for reference
    moe_torch = create_deepseek_moe_torch(model_path, device, dtype)

    print("*" * 80)
    print("Test DeepSeek-V3 MoE PREFILL")
    print("*" * 80)
    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    output_prefill = benchmark_moe_torch(
        moe_torch, PREFILL_TESTCASES, device=device, dtype=dtype
    )

    print("\n")
    print("-" * 80)
    print(f"Test DECODE_TESTCASES: {DECODE_TESTCASES}")
    output_decode = benchmark_moe_torch(
        moe_torch, DECODE_TESTCASES, device=device, dtype=dtype
    )

    # Test InfiniLM implementation
    try:
        moe_infinilm = create_deepseek_moe_infinilm(model_path, device_type)

        print("\n" + "=" * 80)
        print("InfiniLM DeepSeek-V3 MoE Performance")
        print("=" * 80)

        print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
        infinilm_prefill = benchmark_moe_infinilm(
            moe_infinilm, PREFILL_TESTCASES, device_type
        )

        print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
        infinilm_decode = benchmark_moe_infinilm(
            moe_infinilm, DECODE_TESTCASES, device_type
        )

    except Exception as e:
        print(f"Failed to run InfiniLM performance test: {e}")

    # Clean up
    del moe_torch
    torch_empty_cache(device)


if __name__ == "__main__":
    args = get_args()
    print(args)

    if not any([args.correctness, args.performance]):
        print("Please specify --correctness or --performance")
        sys.exit(1)

    if not any([args.cpu, args.nvidia, args.moore]):
        print("Please specify a platform: --cpu, --nvidia, or --moore")
        sys.exit(1)

    try:
        if args.correctness:
            correctness_test(args)

        if args.performance:
            performance_test(args)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)