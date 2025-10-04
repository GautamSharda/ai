"""
CUDA Memory Profiler - wraps test_single_problem.py to visualize memory usage
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys

# Memory tracking
memory_snapshots = []

def log_memory(label):
    """Log current CUDA memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3

        memory_snapshots.append({
            'label': label,
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'timestamp': time.time()
        })

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n{'='*80}")
        print(f"Memory @ {label}")
        print(f"{'='*80}")
        print(f"  Allocated:     {allocated:.3f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved:      {reserved:.3f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Max Allocated: {max_allocated:.3f} GB")
        print(f"  Total GPU:     {total_memory:.3f} GB")
        print(f"{'='*80}\n")

def plot_memory_usage():
    """Create visualization of memory usage"""
    if not memory_snapshots:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    labels = [s['label'] for s in memory_snapshots]
    allocated = [s['allocated'] for s in memory_snapshots]
    reserved = [s['reserved'] for s in memory_snapshots]
    max_alloc = [s['max_allocated'] for s in memory_snapshots]

    x = range(len(labels))
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Plot 1: Memory over time
    ax1.plot(x, allocated, 'o-', label='Allocated', linewidth=2, markersize=6, color='#2E86AB')
    ax1.plot(x, reserved, 's-', label='Reserved', linewidth=2, markersize=6, color='#A23B72')
    ax1.axhline(y=total_memory, color='r', linestyle='--', linewidth=2, label=f'Total GPU ({total_memory:.1f} GB)')
    ax1.set_xlabel('Operation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
    ax1.set_title('CUDA Memory Usage Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    # Plot 2: Memory deltas
    deltas = [0] + [allocated[i] - allocated[i-1] for i in range(1, len(allocated))]
    colors = ['green' if d <= 0 else 'red' for d in deltas]
    ax2.bar(x, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Operation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory Change (GB)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Allocation Changes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0, color='black', linewidth=0.8)

    for i, d in enumerate(deltas):
        if abs(d) > 0.01:
            ax2.text(i, d, f'{d:+.3f}', fontsize=8, ha='center', va='bottom' if d > 0 else 'top')

    plt.tight_layout()
    plt.savefig('/ai_network_volume/ai/qwen/qwen3/memory_profile.png', dpi=150, bbox_inches='tight')
    print(f"\nMemory profile saved to: memory_profile.png\n")

if __name__ == "__main__":
    print("="*80)
    print("CUDA MEMORY PROFILER")
    print("="*80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.empty_cache()

    log_memory("start")

    try:
        # Actually run the original test_single_problem.py
        with open('/ai_network_volume/ai/qwen/qwen3/test_single_problem.py', 'r') as f:
            test_code = f.read()

        # Execute it in a namespace we can monitor
        exec(test_code, {'__name__': '__main__', 'log_memory': log_memory})

    except Exception as e:
        log_memory("error")
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        log_memory("end")
        plot_memory_usage()

        if torch.cuda.is_available():
            print(f"\n{'='*80}")
            print("MEMORY SUMMARY")
            print(f"{'='*80}")
            print(torch.cuda.memory_summary(0))
