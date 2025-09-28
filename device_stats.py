#!/usr/bin/env python3

import os
import subprocess
import shutil
from pathlib import Path


def get_cpu_model():
    """Get CPU model from /proc/cpuinfo"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    return "Unknown"


def bytes_to_gb(bytes_val):
    """Convert bytes to GB with 1 decimal place"""
    return round(bytes_val / 1073741824, 1)


def mib_to_gb(mib_val):
    """Convert MiB to GB with 1 decimal place"""
    return round(mib_val / 1024, 1)


def get_memory_info():
    """Get memory usage in a clean format"""
    if not shutil.which('free'):
        return "0|0|0|0"

    try:
        result = subprocess.run(['free', '-b'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            # Parse the memory line (second line)
            mem_line = lines[1].split()
            if len(mem_line) >= 7:
                total_bytes = int(mem_line[1])
                used_bytes = int(mem_line[2])
                available_bytes = int(mem_line[6])  # available column

                total_gb = total_bytes / 1073741824
                used_gb = used_bytes / 1073741824
                free_gb = available_bytes / 1073741824
                usage_pct = (used_gb / total_gb) * 100

                return f"{total_gb:.1f}|{used_gb:.1f}|{free_gb:.1f}|{usage_pct:.1f}"
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass

    return "0|0|0|0"


def main():
    print("==================== RUNPOD GPU RESOURCES ====================")
    print()

    # GPU Information
    print("🖥️  GPU:")
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)

            gpu_lines = result.stdout.strip().split('\n')
            if gpu_lines and gpu_lines[0]:
                # Process all GPUs
                total_vram = 0
                total_used = 0
                gpu_models = set()

                for i, gpu_line in enumerate(gpu_lines):
                    parts = [part.strip() for part in gpu_line.split(',')]
                    if len(parts) >= 4:
                        gpu_name, gpu_total, gpu_used, gpu_free = parts[:4]

                        gpu_total = int(gpu_total)
                        gpu_used = int(gpu_used)
                        gpu_free = int(gpu_free)

                        total_vram += gpu_total
                        total_used += gpu_used
                        gpu_models.add(gpu_name)

                        gpu_total_gb = mib_to_gb(gpu_total)
                        gpu_used_gb = mib_to_gb(gpu_used)
                        gpu_usage_pct = (gpu_used / gpu_total) * 100

                        if i == 0:
                            print(f"   Model: {gpu_name} (×{len(gpu_lines)})")
                        print(f"   GPU {i}: {gpu_total_gb} GB total | {gpu_used_gb} GB used ({gpu_usage_pct:.1f}%) | {mib_to_gb(gpu_free)} GB free")

                # Show total across all GPUs
                if len(gpu_lines) > 1:
                    total_vram_gb = mib_to_gb(total_vram)
                    total_used_gb = mib_to_gb(total_used)
                    total_usage_pct = (total_used / total_vram) * 100
                    print(f"   Total: {total_vram_gb} GB total | {total_used_gb} GB used ({total_usage_pct:.1f}%)")
            else:
                raise ValueError("No GPU data")
        except (subprocess.CalledProcessError, ValueError):
            runpod_gpu_name = os.environ.get('RUNPOD_GPU_NAME', '')
            if runpod_gpu_name:
                gpu_name = runpod_gpu_name.replace('+', ' ')
                print(f"   Model: {gpu_name}")
                print("   VRAM:  Unable to read (nvidia-smi not accessible)")
            else:
                print("   No GPU detected")
    else:
        runpod_gpu_name = os.environ.get('RUNPOD_GPU_NAME', '')
        if runpod_gpu_name:
            gpu_name = runpod_gpu_name.replace('+', ' ')
            print(f"   Model: {gpu_name}")
            print("   VRAM:  Unable to read (nvidia-smi not available)")
        else:
            print("   No GPU detected")

    print()

    # CPU Information
    print("🔲  CPU:")
    cpu_model = get_cpu_model()
    print(f"   Model: {cpu_model}")

    # Get actual usable cores
    cgroup_cpu_max = Path('/sys/fs/cgroup/cpu.max')
    if cgroup_cpu_max.exists():
        try:
            with open(cgroup_cpu_max, 'r') as f:
                cpu_max_line = f.read().strip()
                parts = cpu_max_line.split()
                if len(parts) >= 2:
                    quota, period = parts[0], parts[1]
                    if quota == "max":
                        actual_cores = "Unlimited"
                    elif quota and period and period != "0":
                        actual_cores = f"{int(quota) / int(period):.1f}"
                    else:
                        actual_cores = get_nproc()
                else:
                    actual_cores = get_nproc()
        except (FileNotFoundError, PermissionError, ValueError):
            actual_cores = get_nproc()
    else:
        actual_cores = get_nproc()

    print(f"   Cores: {actual_cores} cores allocated")

    print()

    # RAM Information
    print("💾  RAM:")
    mem_info = get_memory_info()
    total_gb, used_gb, free_gb, usage_pct = mem_info.split('|')
    print(f"   Total: {total_gb} GB | Used: {used_gb} GB ({usage_pct}%) | Free: {free_gb} GB")

    print()

    # Storage Information
    print("💿  STORAGE:")

    # Root filesystem
    try:
        result = subprocess.run(['df', '-BG', '--output=size,used,avail,pcent', '/'],
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            df_line = lines[-1].split()
            if len(df_line) >= 4:
                total = df_line[0].rstrip('G')
                used = df_line[1].rstrip('G')
                avail = df_line[2].rstrip('G')
                pct = df_line[3].rstrip('%')
                print(f"   Total: {total} GB | Used: {used} GB ({pct}%) | Free: {avail} GB")
            else:
                raise ValueError("Insufficient storage data")
        else:
            raise ValueError("No storage data")
    except (subprocess.CalledProcessError, ValueError):
        print("   Unable to read storage information")

    # Workspace usage
    workspace_path = os.environ.get('RUNPOD_WORKSPACE_PATH', '/workspace')
    if os.path.isdir(workspace_path):
        try:
            result = subprocess.run(['du', '-sb', workspace_path],
                                  capture_output=True, text=True, check=True)
            workspace_usage_bytes = int(result.stdout.split()[0])
            workspace_usage_gb = bytes_to_gb(workspace_usage_bytes)
            print(f"   Workspace: {workspace_usage_gb} GB used in {workspace_path}")
        except (subprocess.CalledProcessError, ValueError, IndexError):
            pass

    print()
    print("==============================================================")


def get_nproc():
    """Get number of processors"""
    try:
        result = subprocess.run(['nproc', '--all'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


if __name__ == "__main__":
    main()