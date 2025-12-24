# DeepSeek-V3 MoE Test

This directory contains test scripts for the DeepSeek-V3 MoE (Mixture of Experts) implementation in InfiniLM.

## Test Scenarios

### Performance Test
- **Small Batch Prefill**: 4 requests with lengths [64, 128, 256, 256] (total 704 tokens)
- **Large Batch Decode**: 16 requests with length 1 each (total 16 tokens)

Both scenarios run 100 rounds and measure average latency and throughput.

### Correctness Test
Compares InfiniLM MoE implementation against the reference PyTorch/transformers implementation.

## Usage

### Prerequisites
- DeepSeek-R1 model weights downloaded to a local directory
- transformers library installed
- InfiniLM built with DeepSeek-V3 support

### Running Tests

```bash
# Performance test on NVIDIA GPU
python test/models/deepseek_v3_moe/moe_test.py --model_path /path/to/deepseek-r1 --nvidia --performance

# Correctness test
python test/models/deepseek_v3_moe/moe_test.py --model_path /path/to/deepseek-r1 --nvidia --correctness

# Moore platform (optional)
python test/models/deepseek_v3_moe/moe_test.py --model_path /path/to/deepseek-r1 --moore --performance
```

### Test Output

The script will output:
- PyTorch reference performance metrics
- InfiniLM implementation performance metrics
- Correctness comparison results (when applicable)

## Implementation Notes

- Uses the same test cases as Qwen3 MoE but adapted for DeepSeek-V3 architecture
- Supports FP8 quantization as required
- Compares against transformers library implementation for correctness validation
- Measures latency and throughput for both prefill and decode scenarios