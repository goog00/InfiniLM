"""
MLA (Multi-head Latent Attention) Test Script.

This script tests the MLA implementation for DeepSeek-R1 model, including:
1. Correctness validation (comparing absorb mode vs naive mode)
2. Performance benchmarking for prefill and decode scenarios

Test scenarios:
- Small batch prefill: 4 requests, seq_len=[64,128,256,256], past_len=[512,0,0,256]
- Large batch decode: 16 requests, seq_len=1, past_len=[50*4, 100*4, 200*4, 400*4]

Usage:
    python test/models/deepseek_mla/mla_test.py --nvidia [--model_path=<path>]
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional

# Add the current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    MLAConfig, 
    torch_synchronize, 
    torch_empty_cache,
    precompute_freqs_cis,
    compare_tensors,
    print_comparison_result,
)
from mla_reference import (
    MLA,
    MLAWithNaiveCache,
    create_mla_from_config,
    copy_weights,
)


# ============================================================================
# Test Configuration
# ============================================================================
WARMUPS = 10
RUNS = 100

PREFILL_TESTCASES = {
    "seqlens": [64, 128, 256, 256],
    "pastlens": [512, 0, 0, 256],
}

DECODE_TESTCASES = {
    "seqlens": [1] * 16,
    "pastlens": [50] * 4 + [100] * 4 + [200] * 4 + [400] * 4,
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MLA Module")
    parser.add_argument(
        "--model_path",
        action="store",
        help="The directory of the model to be tested (optional for reference test)",
    )
    parser.add_argument("--cpu", action="store_true", help="Run cpu test")
    parser.add_argument("--nvidia", action="store_true", help="Run nvidia test")
    parser.add_argument("--metax", action="store_true", help="Run metax test")
    parser.add_argument("--moore", action="store_true", help="Run moore test")
    parser.add_argument("--iluvatar", action="store_true", help="Run iluvatar test")
    parser.add_argument(
        "--skip_correctness", 
        action="store_true", 
        help="Skip correctness test"
    )
    parser.add_argument(
        "--skip_performance", 
        action="store_true", 
        help="Skip performance test"
    )
    parser.add_argument(
        "--use_small_config",
        action="store_true",
        help="Use smaller config for quick testing (default: full DeepSeek-R1 config)",
    )
    return parser.parse_args()


def get_device(args) -> str:
    """Determine device based on arguments."""
    if args.cpu:
        return "cpu"
    elif args.nvidia or args.metax or args.iluvatar:
        return "cuda"
    elif args.moore:
        import torch_musa
        return "musa"
    else:
        return "cuda"  # default


def get_mla_config(use_small: bool = False) -> MLAConfig:
    """
    Get MLA configuration.
    
    Args:
        use_small: If True, use smaller config for quick testing
    
    Returns:
        MLAConfig instance
    """
    if use_small:
        # Smaller config for quick testing
        return MLAConfig(
            dim=1024,
            n_heads=16,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            max_seq_len=4096,
            original_seq_len=4096,
        )
    else:
        # Full DeepSeek-R1 config
        return MLAConfig(
            dim=7168,
            n_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            max_seq_len=4096 * 4,
            original_seq_len=4096,
            rope_theta=10000.0,
            rope_factor=40.0,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
        )


# ============================================================================
# Correctness Test
# ============================================================================
def test_correctness(
    config: MLAConfig,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """
    Test correctness by comparing absorb mode with naive mode.
    
    Both modes should produce identical outputs given the same inputs and weights.
    
    Args:
        config: MLA configuration
        device: Target device
        dtype: Data type
    
    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST: Comparing Absorb Mode vs Naive Mode")
    print("=" * 80)
    
    # Create both models
    model_absorb = create_mla_from_config(config, device, dtype, use_absorb=True)
    model_naive = create_mla_from_config(config, device, dtype, use_absorb=False)
    
    # Copy weights from absorb to naive to ensure identical weights
    copy_weights(model_absorb, model_naive)
    
    # Precompute freqs_cis
    freqs_cis = precompute_freqs_cis(config, device)
    
    all_passed = True
    
    # Test cases - only test start_pos=0 for correctness (cache comparison is complex)
    test_cases = [
        {"batch_size": 1, "seq_len": 64, "start_pos": 0, "name": "single_prefill"},
        {"batch_size": 1, "seq_len": 128, "start_pos": 0, "name": "longer_prefill"},
        {"batch_size": 4, "seq_len": 1, "start_pos": 0, "name": "decode_no_cache"},
        {"batch_size": 2, "seq_len": 32, "start_pos": 0, "name": "batch_prefill"},
    ]
    
    for tc in test_cases:
        batch_size = tc["batch_size"]
        seq_len = tc["seq_len"]
        start_pos = tc["start_pos"]
        name = tc["name"]
        
        print(f"\n--- Test: {name} (batch={batch_size}, seq={seq_len}, start_pos={start_pos}) ---")
        
        # Initialize caches
        max_seq_len = start_pos + seq_len + 100
        model_absorb.init_cache(batch_size, max_seq_len, device, dtype)
        model_naive.init_cache(batch_size, max_seq_len, device, dtype)
        
        # Set same random seed for reproducibility
        torch.manual_seed(42)
        
        # Random input
        x = torch.randn(batch_size, seq_len, config.dim, device=device, dtype=dtype)
        
        # Get freqs for this sequence
        freqs = freqs_cis[start_pos:start_pos + seq_len]
        
        # Create causal mask for prefill (only needed when seq_len > 1)
        mask = None
        if seq_len > 1:
            total_len = start_pos + seq_len
            # Mask shape should be [seq_len, total_len] for attention scores [batch, seq, head, total_len]
            mask = torch.full((seq_len, total_len), float("-inf"), device=device)
            # Only allow attending to positions <= current position
            for i in range(seq_len):
                mask[i, :start_pos + i + 1] = 0
        
        # Forward pass
        with torch.no_grad():
            out_absorb = model_absorb(x, start_pos, freqs, mask)
            out_naive = model_naive(x, start_pos, freqs, mask)
        
        # Compare outputs
        is_close, stats = compare_tensors(out_absorb, out_naive, f"output_{name}", rtol=1e-2, atol=1e-2)
        print_comparison_result(stats)
        
        if not is_close:
            print(f"  ❌ FAILED: {name}")
            all_passed = False
        else:
            print(f"  ✓ PASSED: {name}")
    
    # Clean up
    del model_absorb, model_naive
    torch_empty_cache(device)
    
    print("\n" + "-" * 80)
    if all_passed:
        print("✓ All correctness tests PASSED!")
    else:
        print("❌ Some correctness tests FAILED!")
    print("-" * 80)
    
    return all_passed


# ============================================================================
# Performance Test - Prefill
# ============================================================================
def generate_prefill_inputs(
    config: MLAConfig,
    testcase: Dict,
    device: str,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """Generate inputs for prefill test."""
    req_list = []
    
    for seq_len, past_len in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.randn(
            1, seq_len, config.dim, device=device, dtype=dtype
        )
        req = {
            "hidden_states": hidden_states,
            "seq_len": seq_len,
            "past_len": past_len,
        }
        req_list.append(req)
    
    return req_list


def benchmark_mla_prefill(
    model: nn.Module,
    config: MLAConfig,
    testcase: Dict,
    device: str,
    dtype: torch.dtype,
    freqs_cis: torch.Tensor,
) -> Tuple[List[torch.Tensor], float]:
    """
    Benchmark MLA prefill performance.
    
    Args:
        model: MLA model
        config: MLA configuration
        testcase: Test case configuration
        device: Target device
        dtype: Data type
        freqs_cis: Precomputed rotary embeddings
    
    Returns:
        Tuple of (output_list, average_latency_ms)
    """
    req_list = generate_prefill_inputs(config, testcase, device, dtype)
    max_total_len = max(r["seq_len"] + r["past_len"] for r in req_list) + 100
    
    # Initial forward pass to collect outputs
    output_list = []
    for req in req_list:
        hidden_states = req["hidden_states"]
        seq_len = req["seq_len"]
        past_len = req["past_len"]
        
        # Initialize cache
        model.init_cache(1, max_total_len, device, dtype)
        
        # Initialize past cache with random values if past_len > 0
        if past_len > 0:
            if hasattr(model, 'kv_cache'):
                model.kv_cache[:, :past_len] = torch.randn(
                    1, past_len, config.kv_lora_rank, device=device, dtype=dtype
                )
                model.pe_cache[:, :past_len] = torch.randn(
                    1, past_len, config.qk_rope_head_dim, device=device, dtype=dtype
                )
            else:
                model.k_cache[:, :past_len] = torch.randn(
                    1, past_len, config.n_heads, config.qk_head_dim, device=device, dtype=dtype
                )
                model.v_cache[:, :past_len] = torch.randn(
                    1, past_len, config.n_heads, config.v_head_dim, device=device, dtype=dtype
                )
        
        # Get freqs
        freqs = freqs_cis[past_len:past_len + seq_len]
        
        # Create mask
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, past_len + seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=past_len + 1)
        
        with torch.no_grad():
            output = model(hidden_states, past_len, freqs, mask)
        
        output_list.append(output.cpu())
    
    torch_synchronize(device)
    
    # Warmup
    for _ in range(WARMUPS):
        for req in req_list:
            hidden_states = req["hidden_states"]
            seq_len = req["seq_len"]
            past_len = req["past_len"]
            
            model.init_cache(1, max_total_len, device, dtype)
            if past_len > 0:
                if hasattr(model, 'kv_cache'):
                    model.kv_cache[:, :past_len].normal_()
                    model.pe_cache[:, :past_len].normal_()
                else:
                    model.k_cache[:, :past_len].normal_()
                    model.v_cache[:, :past_len].normal_()
            
            freqs = freqs_cis[past_len:past_len + seq_len]
            mask = None
            if seq_len > 1:
                mask = torch.full((seq_len, past_len + seq_len), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=past_len + 1)
            
            with torch.no_grad():
                _ = model(hidden_states, past_len, freqs, mask)
    
    torch_synchronize(device)
    
    # Benchmark
    total_time = 0.0
    for _ in range(RUNS):
        # Re-initialize inputs for each run
        for req in req_list:
            hidden_states = req["hidden_states"]
            seq_len = req["seq_len"]
            past_len = req["past_len"]
            
            model.init_cache(1, max_total_len, device, dtype)
            if past_len > 0:
                if hasattr(model, 'kv_cache'):
                    model.kv_cache[:, :past_len].normal_()
                    model.pe_cache[:, :past_len].normal_()
                else:
                    model.k_cache[:, :past_len].normal_()
                    model.v_cache[:, :past_len].normal_()
        
        torch_synchronize(device)
        start_time = time.time()
        
        for req in req_list:
            hidden_states = req["hidden_states"]
            seq_len = req["seq_len"]
            past_len = req["past_len"]
            
            freqs = freqs_cis[past_len:past_len + seq_len]
            mask = None
            if seq_len > 1:
                mask = torch.full((seq_len, past_len + seq_len), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=past_len + 1)
            
            with torch.no_grad():
                _ = model(hidden_states, past_len, freqs, mask)
        
        torch_synchronize(device)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_latency_ms = (total_time / RUNS) * 1000
    
    return output_list, avg_latency_ms


# ============================================================================
# Performance Test - Decode
# ============================================================================
def generate_decode_inputs(
    config: MLAConfig,
    testcase: Dict,
    device: str,
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """Generate inputs for decode test."""
    req_list = []
    
    for seq_len, past_len in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.randn(
            1, seq_len, config.dim, device=device, dtype=dtype
        )
        req = {
            "hidden_states": hidden_states,
            "seq_len": seq_len,
            "past_len": past_len,
            "current_pos": past_len,  # Will be updated during decode
        }
        req_list.append(req)
    
    return req_list


def benchmark_mla_decode(
    model: nn.Module,
    config: MLAConfig,
    testcase: Dict,
    device: str,
    dtype: torch.dtype,
    freqs_cis: torch.Tensor,
) -> Tuple[List[torch.Tensor], float]:
    """
    Benchmark MLA decode performance.
    
    Decode runs sequentially for each request, with cache growing each iteration.
    
    Args:
        model: MLA model
        config: MLA configuration
        testcase: Test case configuration
        device: Target device
        dtype: Data type
        freqs_cis: Precomputed rotary embeddings
    
    Returns:
        Tuple of (output_list, throughput_tok_per_sec)
    """
    req_list = generate_decode_inputs(config, testcase, device, dtype)
    max_past_len = max(testcase["pastlens"])
    max_total_len = max_past_len + RUNS + 100
    
    # Initialize all caches
    caches = []
    for i, req in enumerate(req_list):
        past_len = req["past_len"]
        
        # Create cache storage
        if hasattr(model, 'kv_cache'):
            kv_cache = torch.randn(
                1, max_total_len, config.kv_lora_rank, device=device, dtype=dtype
            )
            pe_cache = torch.randn(
                1, max_total_len, config.qk_rope_head_dim, device=device, dtype=dtype
            )
            caches.append({"kv_cache": kv_cache, "pe_cache": pe_cache, "pos": past_len})
        else:
            k_cache = torch.randn(
                1, max_total_len, config.n_heads, config.qk_head_dim, device=device, dtype=dtype
            )
            v_cache = torch.randn(
                1, max_total_len, config.n_heads, config.v_head_dim, device=device, dtype=dtype
            )
            caches.append({"k_cache": k_cache, "v_cache": v_cache, "pos": past_len})
    
    # Initial forward pass
    output_list = []
    for i, req in enumerate(req_list):
        hidden_states = req["hidden_states"]
        pos = caches[i]["pos"]
        
        # Set model cache
        if hasattr(model, 'kv_cache'):
            model.kv_cache = caches[i]["kv_cache"]
            model.pe_cache = caches[i]["pe_cache"]
        else:
            model.k_cache = caches[i]["k_cache"]
            model.v_cache = caches[i]["v_cache"]
        
        freqs = freqs_cis[pos:pos + 1]
        
        with torch.no_grad():
            output = model(hidden_states, pos, freqs, mask=None)
        
        output_list.append(output.cpu())
    
    torch_synchronize(device)
    
    # Reset positions
    for i, req in enumerate(req_list):
        caches[i]["pos"] = req["past_len"]
    
    # Warmup
    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            hidden_states = req["hidden_states"]
            pos = caches[i]["pos"]
            
            if hasattr(model, 'kv_cache'):
                model.kv_cache = caches[i]["kv_cache"]
                model.pe_cache = caches[i]["pe_cache"]
            else:
                model.k_cache = caches[i]["k_cache"]
                model.v_cache = caches[i]["v_cache"]
            
            freqs = freqs_cis[pos:pos + 1]
            
            with torch.no_grad():
                output = model(hidden_states, pos, freqs, mask=None)
            
            # Update for next iteration (simulate decode)
            caches[i]["pos"] += 1
            req["hidden_states"] = output  # Output becomes next input
    
    torch_synchronize(device)
    
    # Reset positions and regenerate inputs for benchmark
    for i, req in enumerate(req_list):
        caches[i]["pos"] = req["past_len"]
        req["hidden_states"] = torch.randn(1, 1, config.dim, device=device, dtype=dtype)
    
    # Benchmark: Sequential decode for all requests over RUNS iterations
    torch_synchronize(device)
    start_time = time.time()
    
    total_tokens = 0
    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            hidden_states = req["hidden_states"]
            pos = caches[i]["pos"]
            
            if hasattr(model, 'kv_cache'):
                model.kv_cache = caches[i]["kv_cache"]
                model.pe_cache = caches[i]["pe_cache"]
            else:
                model.k_cache = caches[i]["k_cache"]
                model.v_cache = caches[i]["v_cache"]
            
            freqs = freqs_cis[pos:pos + 1]
            
            with torch.no_grad():
                output = model(hidden_states, pos, freqs, mask=None)
            
            # Update for next iteration
            caches[i]["pos"] += 1
            req["hidden_states"] = output
            total_tokens += 1
    
    torch_synchronize(device)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = total_tokens / total_time
    
    return output_list, throughput


# ============================================================================
# Main Test Function
# ============================================================================
def run_performance_tests(
    config: MLAConfig,
    device: str,
    dtype: torch.dtype,
    use_absorb: bool = True,
) -> Dict[str, Any]:
    """
    Run performance benchmarks.
    
    Args:
        config: MLA configuration
        device: Target device
        dtype: Data type
        use_absorb: Use absorb mode (True) or naive mode (False)
    
    Returns:
        Dictionary of benchmark results
    """
    mode_name = "Absorb" if use_absorb else "Naive"
    print(f"\n" + "=" * 80)
    print(f"PERFORMANCE TEST: {mode_name} Mode")
    print("=" * 80)
    
    # Create model
    model = create_mla_from_config(config, device, dtype, use_absorb=use_absorb)
    model.eval()
    
    # Precompute freqs_cis
    freqs_cis = precompute_freqs_cis(config, device)
    
    results = {}
    
    # ----- Prefill Benchmark -----
    print(f"\n--- Prefill Benchmark ---")
    print(f"Test Case: {PREFILL_TESTCASES}")
    
    _, prefill_latency = benchmark_mla_prefill(
        model, config, PREFILL_TESTCASES, device, dtype, freqs_cis
    )
    
    results["prefill_latency_ms"] = prefill_latency
    print(f"\n  WARMUPS={WARMUPS}, RUNS={RUNS}")
    print(f"  Average TTFT (Time To First Token): {prefill_latency:.2f} ms")
    
    # ----- Decode Benchmark -----
    print(f"\n--- Decode Benchmark ---")
    print(f"Test Case: {DECODE_TESTCASES}")
    
    _, decode_throughput = benchmark_mla_decode(
        model, config, DECODE_TESTCASES, device, dtype, freqs_cis
    )
    
    results["decode_throughput_tok_s"] = decode_throughput
    print(f"\n  WARMUPS={WARMUPS}, RUNS={RUNS}")
    print(f"  Average Throughput: {decode_throughput:.2f} tok/s")
    
    # Clean up
    del model
    torch_empty_cache(device)
    
    return results


def main():
    """Main entry point."""
    args = get_args()
    
    # Validate arguments
    if not any([args.cpu, args.nvidia, args.metax, args.moore, args.iluvatar]):
        print(
            "Usage: python test/models/deepseek_mla/mla_test.py "
            "[--cpu | --nvidia | --metax | --moore | --iluvatar] "
            "[--model_path=<path>] [--use_small_config] [--skip_correctness] [--skip_performance]"
        )
        sys.exit(1)
    
    device = get_device(args)
    dtype = torch.bfloat16
    config = get_mla_config(use_small=args.use_small_config)
    
    print("\n" + "*" * 80)
    print("DeepSeek MLA (Multi-head Latent Attention) Test")
    print("*" * 80)
    print(f"\nDevice: {device}")
    print(f"Data Type: {dtype}")
    print(f"\nMLA Configuration:")
    print(f"  dim: {config.dim}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  q_lora_rank: {config.q_lora_rank}")
    print(f"  kv_lora_rank: {config.kv_lora_rank}")
    print(f"  qk_nope_head_dim: {config.qk_nope_head_dim}")
    print(f"  qk_rope_head_dim: {config.qk_rope_head_dim}")
    print(f"  v_head_dim: {config.v_head_dim}")
    print(f"  qk_head_dim (total): {config.qk_head_dim}")
    
    # ----- Correctness Test -----
    if not args.skip_correctness:
        correctness_passed = test_correctness(config, device, dtype)
        if not correctness_passed:
            print("\n⚠️  Correctness tests failed! Performance results may not be meaningful.")
    
    # ----- Performance Tests -----
    if not args.skip_performance:
        # Test Absorb mode (required implementation)
        results_absorb = run_performance_tests(config, device, dtype, use_absorb=True)
        
        # Optionally test Naive mode for comparison
        print("\n" + "-" * 80)
        print("Comparing with Naive mode (for reference)...")
        results_naive = run_performance_tests(config, device, dtype, use_absorb=False)
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nPrefill Latency (lower is better):")
        print(f"  Absorb Mode: {results_absorb['prefill_latency_ms']:.2f} ms")
        print(f"  Naive Mode:  {results_naive['prefill_latency_ms']:.2f} ms")
        
        print(f"\nDecode Throughput (higher is better):")
        print(f"  Absorb Mode: {results_absorb['decode_throughput_tok_s']:.2f} tok/s")
        print(f"  Naive Mode:  {results_naive['decode_throughput_tok_s']:.2f} tok/s")
        
        # Memory comparison
        if hasattr(torch.cuda, 'max_memory_allocated'):
            print(f"\nPeak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # Cache memory comparison
        print(f"\nCache Memory Comparison (per token per layer, BF16):")
        absorb_cache_bytes = (config.kv_lora_rank + config.qk_rope_head_dim) * 2  # kv_cache + pe_cache
        naive_cache_bytes = config.n_heads * (config.qk_head_dim + config.v_head_dim) * 2  # k_cache + v_cache
        print(f"  Absorb Mode: {absorb_cache_bytes} bytes ({config.kv_lora_rank} + {config.qk_rope_head_dim} dims)")
        print(f"  Naive Mode:  {naive_cache_bytes} bytes ({config.n_heads} * ({config.qk_head_dim} + {config.v_head_dim}) dims)")
        print(f"  Compression Ratio: {naive_cache_bytes / absorb_cache_bytes:.1f}x")
        
        # Theoretical speedup analysis
        print(f"\n📊 Performance Analysis:")
        if results_absorb['prefill_latency_ms'] < results_naive['prefill_latency_ms']:
            prefill_speedup = results_naive['prefill_latency_ms'] / results_absorb['prefill_latency_ms']
            print(f"  ✓ Prefill: Absorb is {prefill_speedup:.2f}x faster")
        else:
            prefill_slowdown = results_absorb['prefill_latency_ms'] / results_naive['prefill_latency_ms']
            print(f"  ⚠ Prefill: Absorb is {prefill_slowdown:.2f}x slower (expected in PyTorch, fused kernels needed)")
        
        if results_absorb['decode_throughput_tok_s'] > results_naive['decode_throughput_tok_s']:
            decode_speedup = results_absorb['decode_throughput_tok_s'] / results_naive['decode_throughput_tok_s']
            print(f"  ✓ Decode: Absorb is {decode_speedup:.2f}x faster")
        else:
            decode_slowdown = results_naive['decode_throughput_tok_s'] / results_absorb['decode_throughput_tok_s']
            print(f"  ⚠ Decode: Absorb is {decode_slowdown:.2f}x slower (expected in PyTorch, fused kernels needed)")
        
        print(f"\n💡 Note: Absorb mode benefits from:")
        print(f"   1. {naive_cache_bytes / absorb_cache_bytes:.1f}x smaller KV cache (critical for long contexts)")
        print(f"   2. Fused CUDA kernels (not available in pure PyTorch)")
        print(f"   3. Lower memory bandwidth for cache reads")
    
    print("\n" + "*" * 80)
    print("Test Complete!")
    print("*" * 80)


if __name__ == "__main__":
    main()
