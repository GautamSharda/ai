"""
Benchmark Qwen3-0.6B on ARC-AGI 2025 evaluation set.
This script measures inference time on a sample of problems and extrapolates to the full set.
"""
import json
import time
import sys
import os

# Add the qwen3 directory to the path so we can import from qwen3_0.6b.py
sys.path.insert(0, '/ai_network_volume/ai/qwen/qwen3')
os.chdir('/ai_network_volume/ai/qwen/qwen3')  # Change dir so relative paths work

# Import using importlib since filename has dots
import importlib.util
spec = importlib.util.spec_from_file_location("qwen3", "/ai_network_volume/ai/qwen/qwen3/qwen3_0.6b.py")
qwen3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3_module)
qwen3_tg = qwen3_module.qwen3_tg

# Load the ARC-AGI 2025 evaluation set
eval_path = '/ai_network_volume/ai/arc-agi/arc-agi-2025/arc-agi_evaluation_challenges.json'
with open(eval_path, 'r') as f:
    eval_data = json.load(f)

# Get first 5 problems
problem_ids = list(eval_data.keys())[:5]
print(f"Benchmarking on {len(problem_ids)} problems: {problem_ids}\n")

# Run inference on each problem
times = []
for idx, problem_id in enumerate(problem_ids, 1):
    problem = eval_data[problem_id]

    # Create a prompt from the problem (simple format)
    # Note: This is a basic prompt format - actual ARC-AGI solving would need more sophisticated prompting
    train_examples = problem.get('train', [])
    test_input = problem.get('test', [{}])[0].get('input', [])

    prompt = f"Given input grid: {test_input[:3]}... predict output."  # Abbreviated prompt

    # Measure inference time
    print(f"[{idx}/{len(problem_ids)}] Processing problem {problem_id}...")
    start_time = time.time()

    try:
        output = qwen3_tg(prompt, max_new_tokens=10)  # Small token count for benchmark
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  ✓ Completed in {elapsed:.2f}s")
        print(f"  Output preview: {output[:80]}...")
    except Exception as e:
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  ✗ Error after {elapsed:.2f}s: {str(e)[:100]}")

    print()

# Calculate statistics
if times:
    avg_time = sum(times) / len(times)
    total_sample_time = sum(times)
    total_problems = len(eval_data)

    # Extrapolate to full dataset
    estimated_total_time = avg_time * total_problems

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Sample size: {len(times)} problems")
    print(f"Total sample time: {total_sample_time:.2f}s ({total_sample_time/60:.2f} min)")
    print(f"Average time per problem: {avg_time:.2f}s")
    print(f"\nFull evaluation set: {total_problems} problems")
    print(f"Estimated total time: {estimated_total_time:.2f}s")
    print(f"                     = {estimated_total_time/60:.2f} minutes")
    print(f"                     = {estimated_total_time/3600:.2f} hours")
    print("=" * 60)
else:
    print("No successful runs to benchmark.")