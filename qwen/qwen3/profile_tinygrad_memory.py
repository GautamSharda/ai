"""
Tinygrad CUDA Memory Profiler - patches tinygrad to track memory allocations
"""

import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Track memory allocations
allocations = []
memory_timeline = []

# Patch tinygrad CUDA allocator
original_cuda_alloc = None

def track_memory(label="checkpoint"):
    """Record memory snapshot"""
    if memory_timeline:
        total_allocated = sum(a['size'] for a in allocations if a['freed'] is None)
    else:
        total_allocated = 0

    memory_timeline.append({
        'label': label,
        'timestamp': time.time(),
        'allocated_gb': total_allocated / 1024**3,
        'num_tensors': len([a for a in allocations if a['freed'] is None])
    })

    print(f"\n{'='*80}")
    print(f"Memory @ {label}")
    print(f"{'='*80}")
    print(f"  Allocated: {total_allocated / 1024**3:.3f} GB")
    print(f"  Active Tensors: {len([a for a in allocations if a['freed'] is None])}")
    print(f"  Total Allocations: {len(allocations)}")
    print(f"{'='*80}\n")

def patched_alloc(self, size, options=None):
    """Patched CUDA alloc that tracks memory"""
    result = original_cuda_alloc(self, size, options)

    allocations.append({
        'size': size,
        'timestamp': time.time(),
        'freed': None,
        'ptr': id(result) if result else None
    })

    total = sum(a['size'] for a in allocations if a['freed'] is None)
    if total / 1024**3 > 15:  # Alert if over 15GB
        print(f"  WARNING: {total / 1024**3:.2f} GB allocated!")

    return result

def plot_memory():
    """Visualize memory usage"""
    if not memory_timeline:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [m['label'] for m in memory_timeline]
    allocated = [m['allocated_gb'] for m in memory_timeline]
    x = range(len(labels))

    ax.plot(x, allocated, 'o-', linewidth=2, markersize=8, color='#E63946')
    ax.axhline(y=22, color='red', linestyle='--', linewidth=2, label='GPU Limit (22 GB)')
    ax.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Tinygrad CUDA Memory Usage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    for i, (a, m) in enumerate(zip(allocated, memory_timeline)):
        ax.text(i, a, f"{a:.1f}GB\n({m['num_tensors']}T)", fontsize=8, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/ai_network_volume/ai/qwen/qwen3/tinygrad_memory.png', dpi=150)
    print(f"\nMemory plot saved to: tinygrad_memory.png\n")

if __name__ == "__main__":
    print("="*80)
    print("TINYGRAD CUDA MEMORY PROFILER")
    print("="*80)

    # Import and patch tinygrad BEFORE loading the model
    import tinygrad.runtime.ops_cuda as cuda_ops
    if hasattr(cuda_ops, 'CUDAAllocator'):
        original_cuda_alloc = cuda_ops.CUDAAllocator._alloc
        cuda_ops.CUDAAllocator._alloc = patched_alloc
        print("✓ Tinygrad CUDA allocator patched\n")
    else:
        print("WARNING: Could not patch CUDA allocator\n")

    track_memory("start")

    try:
        # Import tinygrad components
        import importlib.util
        spec = importlib.util.spec_from_file_location("qwen3", "/ai_network_volume/ai/qwen/qwen3/qwen3_0.6b.py")
        qwen3_module = importlib.util.module_from_spec(spec)

        track_memory("before_module_load")
        spec.loader.exec_module(qwen3_module)
        track_memory("after_module_load")

        # Now run the actual test
        import json

        eval_challenges_path = '/ai_network_volume/ai/arc-agi/arc-agi-2025/arc-agi_evaluation_challenges.json'
        with open(eval_challenges_path, 'r') as f:
            eval_challenges = json.load(f)

        problem_id = list(eval_challenges.keys())[0]
        problem = eval_challenges[problem_id]

        def create_prompt(problem):
            train_examples = problem.get('train', [])
            test_input = problem.get('test', [{}])[0].get('input', [])
            prompt_parts = ["You are given training examples showing input-output pairs. Learn the pattern and predict the output for the test input.\n"]
            for i, example in enumerate(train_examples, 1):
                prompt_parts.append(f"\nTraining Example {i}:")
                prompt_parts.append(f"Input: {example['input']}")
                prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append(f"\nTest Input:")
            prompt_parts.append(f"{test_input}")
            prompt_parts.append(f"\nPredict the output:")
            return "\n".join(prompt_parts)

        prompt = create_prompt(problem)
        print(f"Problem: {problem_id}")
        print(f"Prompt length: {len(prompt)} chars\n")

        track_memory("before_inference")

        # Run with instrumentation
        qwen3_tg = qwen3_module.qwen3_tg

        # Monkey-patch to add checkpoints during generation
        original_qwen3 = qwen3_tg

        def instrumented_qwen3(prompt, max_new_tokens=5):
            import tinygrad
            from safetensors.torch import load_file
            import json

            track_memory("loading_weights")
            ckpt = load_file("Qwen3-0.6B-Base/model.safetensors")
            track_memory("weights_loaded")

            # Call original but with tracking... actually just inline it with tracking
            # This is getting complex, let's just call it and track periodically
            result = original_qwen3(prompt, max_new_tokens)
            return result

        response = instrumented_qwen3(prompt, max_new_tokens=20)

        track_memory("after_inference")
        print(f"\nResponse: {response}\n")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            track_memory("OOM_ERROR")
            print(f"\n{'='*80}")
            print("OUT OF MEMORY ERROR!")
            print(f"{'='*80}")
            print(f"Error: {e}\n")

            # Show what was allocated
            total = sum(a['size'] for a in allocations if a['freed'] is None)
            print(f"Total allocated at OOM: {total / 1024**3:.2f} GB")
            print(f"Number of active tensors: {len([a for a in allocations if a['freed'] is None])}")

            # Show largest allocations
            active = sorted([a for a in allocations if a['freed'] is None],
                          key=lambda x: x['size'], reverse=True)[:10]
            print(f"\nTop 10 largest allocations:")
            for i, a in enumerate(active, 1):
                print(f"  {i}. {a['size'] / 1024**3:.3f} GB")
        raise

    finally:
        track_memory("end")
        plot_memory()

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total allocations: {len(allocations)}")
        active_count = len([a for a in allocations if a['freed'] is None])
        print(f"Active tensors: {active_count}")
        if active_count > 0:
            total_active = sum(a['size'] for a in allocations if a['freed'] is None)
            print(f"Total active memory: {total_active / 1024**3:.2f} GB")
        print(f"{'='*80}\n")
