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
    """Convert bytes to GB (decimal gigabytes) with 1 decimal place"""
    return round(bytes_val / 1000000000, 1)


def mib_to_gb(mib_val):
    """Convert MiB to GB (decimal gigabytes) with 1 decimal place"""
    return round(mib_val * 1048576 / 1000000000, 1)


def get_memory_info():
    """Get container memory usage in a clean format"""
    try:
        # Try to get container memory limit first
        container_limit = None
        try:
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                limit_str = f.read().strip()
                if limit_str != "max":
                    container_limit = int(limit_str)
        except (FileNotFoundError, PermissionError, ValueError):
            try:
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    container_limit = int(f.read().strip())
            except (FileNotFoundError, PermissionError, ValueError):
                pass

        # Get current container memory usage
        container_used = None
        if container_limit:
            try:
                with open('/sys/fs/cgroup/memory.current', 'r') as f:
                    container_used = int(f.read().strip())
            except (FileNotFoundError, PermissionError, ValueError):
                try:
                    with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                        container_used = int(f.read().strip())
                except (FileNotFoundError, PermissionError, ValueError):
                    pass

        if container_limit and container_used is not None:
            # Use actual container memory usage
            total_bytes = container_limit
            used_bytes = container_used
            free_bytes = total_bytes - used_bytes
        else:
            # Fall back to host memory info
            if not shutil.which('free'):
                return "0|0|0|0"

            result = subprocess.run(['free', '-b'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_line = lines[1].split()
                if len(mem_line) >= 7:
                    total_bytes = int(mem_line[1])
                    used_bytes = int(mem_line[2])
                    free_bytes = int(mem_line[6])
                else:
                    return "0|0|0|0"
            else:
                return "0|0|0|0"

        total_gb = total_bytes / 1000000000
        used_gb = used_bytes / 1000000000
        free_gb = free_bytes / 1000000000
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

    # Get physical vs logical core info
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
        physical_cores = None
        threads_per_core = None
        for line in result.stdout.split('\n'):
            if 'Core(s) per socket:' in line:
                physical_cores = int(line.split(':')[1].strip())
            elif 'Thread(s) per core:' in line:
                threads_per_core = int(line.split(':')[1].strip())

        if physical_cores and threads_per_core:
            logical_cores = physical_cores * threads_per_core
            print(f"   Cores: {physical_cores} physical ({logical_cores} logical with hyperthreading)")
        else:
            print(f"   Cores: {actual_cores} cores allocated")
    except (subprocess.CalledProcessError, ValueError, IndexError):
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
                total = int(df_line[0].rstrip('G'))
                used = int(df_line[1].rstrip('G'))
                pct = df_line[3].rstrip('%')
                free = total - used
                print(f"   Container: {total} GB | Used: {used} GB ({pct}%) | Free: {free} GB")
            else:
                raise ValueError("Insufficient storage data")
        else:
            raise ValueError("No storage data")
    except (subprocess.CalledProcessError, ValueError):
        print("   Unable to read storage information")

    # Runpod Volume (100GB allocated)
    if os.path.isdir('/ai_network_volume'):
        try:
            result = subprocess.run(['du', '-sb', '/ai_network_volume'],
                                  capture_output=True, text=True, check=True)
            ai_volume_usage_bytes = int(result.stdout.split()[0])
            ai_volume_usage_gb = bytes_to_gb(ai_volume_usage_bytes)
            usage_pct = (ai_volume_usage_gb / 100) * 100
            free_gb = 100 - ai_volume_usage_gb
            print(f"   Runpod Volume (/ai_network_volume): 100 GB | Used: {ai_volume_usage_gb} GB ({usage_pct:.1f}%) | Free: {free_gb:.1f} GB")
        except (subprocess.CalledProcessError, ValueError, IndexError):
            print("   Runpod Volume: Unable to read usage")

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