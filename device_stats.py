#!/usr/bin/env python3

import os
import sys
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


def get_directory_sizes(path, max_depth=1):
    """Get sizes of subdirectories in a path (using GB for consistency)"""
    try:
        result = subprocess.run(['du', '--block-size=1G', f'--max-depth={max_depth}', path],
                              capture_output=True, text=True, check=True, timeout=30)
        lines = result.stdout.strip().split('\n')
        sizes = []
        for line in lines[:-1]:  # Skip the total line
            parts = line.split('\t', 1)
            if len(parts) == 2:
                size_gb_str, dir_path = parts
                try:
                    size_gb = int(size_gb_str)
                    if size_gb >= 1:  # Only show directories >= 1 GB
                        sizes.append((size_gb, f"{size_gb} GB", dir_path))
                except ValueError:
                    pass
        # Sort by size in GB, descending
        sizes.sort(key=lambda x: x[0], reverse=True)
        return [(size, path) for _, size, path in sizes[:5]]  # Top 5
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return []


def get_top_processes(sort_by='mem', limit=5):
    """Get top processes by CPU or memory usage"""
    try:
        sort_flag = '-%mem' if sort_by == 'mem' else '-%cpu'
        result = subprocess.run(['ps', 'aux', f'--sort={sort_flag}'],
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            processes = []
            for line in lines[1:limit+1]:  # Skip header, get top N
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    user, pid, cpu, mem, vsz, rss = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    command = parts[10][:60]  # Truncate long commands
                    if sort_by == 'mem':
                        processes.append(f"      {command}: {int(rss)//1024} MB ({mem}%)")
                    else:
                        processes.append(f"      {command}: {cpu}% CPU")
            return processes
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return []


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

    # Use container memory limits (cgroup)
    try:
        # Try cgroup v2 first, then v1
        try:
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                limit_str = f.read().strip()
                container_limit = int(limit_str) if limit_str != "max" else None
            # For v2, read from memory.stat
            with open('/sys/fs/cgroup/memory.stat', 'r') as f:
                stat_lines = f.read().strip().split('\n')
                rss = 0
                cache = 0
                for line in stat_lines:
                    if line.startswith('anon '):
                        rss = int(line.split()[1])
                    elif line.startswith('file '):
                        cache = int(line.split()[1])
                container_used_rss = rss
                container_cache = cache
        except (FileNotFoundError, ValueError):
            # Try cgroup v1
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                container_limit = int(f.read().strip())
            with open('/sys/fs/cgroup/memory/memory.stat', 'r') as f:
                stat_lines = f.read().strip().split('\n')
                rss = 0
                cache = 0
                for line in stat_lines:
                    if line.startswith('total_rss '):
                        rss = int(line.split()[1])
                    elif line.startswith('total_cache '):
                        cache = int(line.split()[1])
                container_used_rss = rss
                container_cache = cache

        if container_limit:
            total_gb = container_limit / 1000000000
            used_gb = container_used_rss / 1000000000
            cache_gb = container_cache / 1000000000
            free_gb = total_gb - used_gb - cache_gb
            usage_pct = (used_gb / total_gb) * 100

            print(f"   Total: {total_gb:.1f} GB | Used: {used_gb:.1f} GB ({usage_pct:.1f}%) | Cache: {cache_gb:.1f} GB | Free: {free_gb:.1f} GB")

            # Show top RAM consumers
            ps_result = subprocess.run(['ps', 'aux', '--sort=-%mem'],
                                     capture_output=True, text=True)
            ps_lines = ps_result.stdout.strip().split('\n')[1:]
            top_procs = []

            for line in ps_lines[:5]:
                parts = line.split(None, 10)
                if len(parts) >= 6:
                    rss_kb = int(parts[5])
                    rss_mb = rss_kb / 1024
                    cmd = parts[10][:60] if len(parts) >= 11 else parts[0]
                    top_procs.append((rss_mb, cmd))

            if top_procs:
                shown_total_mb = sum(x[0] for x in top_procs)
                print("   Top processes:")
                for rss_mb, cmd in top_procs:
                    print(f"      {cmd}: {rss_mb:.0f} MB")
                print(f"   Top 5 subtotal: {shown_total_mb/1000:.1f} GB")
        else:
            raise ValueError("No container limit")
    except (FileNotFoundError, PermissionError, ValueError):
        # Fallback to old method
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

                # Show proportional breakdown that adds up to actual usage
                try:
                    result = subprocess.run(['du', '--block-size=1', '--max-depth=1', '--exclude=/ai_network_volume', '--exclude=/workspace', '/'],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
                    lines = result.stdout.strip().split('\n')
                    container_dirs = []
                    total_bytes = 0

                    for line in lines[:-1]:  # Skip total
                        parts = line.split('\t', 1)
                        if len(parts) == 2 and parts[1] != '/':
                            try:
                                size_bytes = int(parts[0])
                                container_dirs.append((size_bytes, parts[1]))
                                total_bytes += size_bytes
                            except ValueError:
                                pass

                    if container_dirs and total_bytes > 0:
                        # Scale to match df's reported usage
                        actual_used_bytes = used * 1000000000  # Convert GB to bytes
                        scale_factor = actual_used_bytes / total_bytes

                        scaled_dirs = []
                        for size_bytes, path in container_dirs:
                            scaled_gb = (size_bytes * scale_factor) / 1000000000
                            if scaled_gb >= 0.5:  # Show if >= 0.5 GB
                                scaled_dirs.append((scaled_gb, path))

                        scaled_dirs.sort(key=lambda x: x[0], reverse=True)

                        print("      Top directories (estimated):")
                        for size_gb, path in scaled_dirs[:5]:
                            print(f"         {path}: {size_gb:.1f} GB")

                            # Show subdirectories for this directory
                            try:
                                sub_result = subprocess.run(['du', '--block-size=1', '--max-depth=1', path],
                                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
                                sub_lines = sub_result.stdout.strip().split('\n')
                                sub_dirs = []
                                sub_total = 0

                                for sub_line in sub_lines[:-1]:  # Skip total
                                    sub_parts = sub_line.split('\t', 1)
                                    if len(sub_parts) == 2 and sub_parts[1] != path:
                                        try:
                                            sub_bytes = int(sub_parts[0])
                                            sub_dirs.append((sub_bytes, sub_parts[1]))
                                            sub_total += sub_bytes
                                        except ValueError:
                                            pass

                                if sub_dirs and sub_total > 0:
                                    # Scale subdirs proportionally
                                    sub_scale = (size_gb * 1000000000) / sub_total
                                    sub_scaled = []
                                    for sub_bytes, sub_path in sub_dirs:
                                        sub_gb = (sub_bytes * sub_scale) / 1000000000
                                        if sub_gb >= 0.5:
                                            sub_scaled.append((sub_gb, sub_path))

                                    sub_scaled.sort(key=lambda x: x[0], reverse=True)
                                    for sub_gb, sub_path in sub_scaled[:3]:
                                        print(f"            ├─ {sub_path}: {sub_gb:.1f} GB")
                            except:
                                pass

                        shown_total = sum(x[0] for x in scaled_dirs[:5])
                        other = used - shown_total
                        if other > 0.5:
                            print(f"         (other): {other:.1f} GB")
                        print(f"      Total: {used} GB")
                except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
                    pass
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

            # Show top storage consumers in network volume (scaled to match df)
            # Only show top-level directories to avoid double-counting
            try:
                result = subprocess.run(['du', '--block-size=1', '--max-depth=1', '/ai_network_volume'],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
                lines = result.stdout.strip().split('\n')
                vol_dirs = []
                total_bytes = 0

                for line in lines[:-1]:  # Skip total
                    parts = line.split('\t', 1)
                    if len(parts) == 2 and parts[1] != '/ai_network_volume':
                        try:
                            size_bytes = int(parts[0])
                            vol_dirs.append((size_bytes, parts[1]))
                            total_bytes += size_bytes
                        except ValueError:
                            pass

                if vol_dirs and total_bytes > 0:
                    # Scale to match df's reported usage
                    scale_factor = ai_volume_usage_bytes / total_bytes

                    scaled_dirs = []
                    for size_bytes, path in vol_dirs:
                        scaled_gb = (size_bytes * scale_factor) / 1000000000
                        if scaled_gb >= 0.1:  # Show if >= 100 MB
                            scaled_dirs.append((scaled_gb, path))

                    scaled_dirs.sort(key=lambda x: x[0], reverse=True)

                    print("      Top directories (estimated):")
                    for size_gb, path in scaled_dirs[:5]:
                        print(f"         {path}: {size_gb:.1f} GB")

                        # Show subdirectories for this directory
                        try:
                            sub_result = subprocess.run(['du', '--block-size=1', '--max-depth=1', path],
                                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
                            sub_lines = sub_result.stdout.strip().split('\n')
                            sub_dirs = []
                            sub_total = 0

                            for sub_line in sub_lines[:-1]:  # Skip total
                                sub_parts = sub_line.split('\t', 1)
                                if len(sub_parts) == 2 and sub_parts[1] != path:
                                    try:
                                        sub_bytes = int(sub_parts[0])
                                        sub_dirs.append((sub_bytes, sub_parts[1]))
                                        sub_total += sub_bytes
                                    except ValueError:
                                        pass

                            if sub_dirs and sub_total > 0:
                                # Scale subdirs proportionally
                                sub_scale = (size_gb * 1000000000) / sub_total
                                sub_scaled = []
                                for sub_bytes, sub_path in sub_dirs:
                                    sub_gb = (sub_bytes * sub_scale) / 1000000000
                                    if sub_gb >= 0.1:
                                        sub_scaled.append((sub_gb, sub_path))

                                sub_scaled.sort(key=lambda x: x[0], reverse=True)
                                for sub_gb, sub_path in sub_scaled[:3]:
                                    print(f"            ├─ {sub_path}: {sub_gb:.1f} GB")
                        except:
                            pass

                    shown_total = sum(x[0] for x in scaled_dirs[:5])
                    other = ai_volume_usage_gb - shown_total
                    if other > 0.1:
                        print(f"         (other): {other:.1f} GB")
                    print(f"      Total: {ai_volume_usage_gb:.1f} GB")
            except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
                pass
        except (subprocess.CalledProcessError, ValueError, IndexError):
            print("   Runpod Volume: Unable to read usage")

    print()
    print("==============================================================")

    # Show directory breakdown if path argument provided
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if os.path.isdir(target_path):
            print()
            print(f"==================== DIRECTORY BREAKDOWN: {target_path} ====================")
            print()

            # Determine which filesystem this path is on and get scale factor
            scale_factor = 1.0
            estimated_total_gb = 0

            # Check if it's on the container filesystem or network volume
            if target_path.startswith('/ai_network_volume'):
                # For network volume, get the actual usage to estimate scale
                try:
                    result = subprocess.run(['du', '-sb', '/ai_network_volume'],
                                          capture_output=True, text=True, check=True)
                    ai_volume_usage_bytes = int(result.stdout.split()[0])
                    ai_volume_usage_gb = ai_volume_usage_bytes / 1000000000

                    # Get du total for all subdirs
                    result = subprocess.run(['du', '--block-size=1', '-s', '/ai_network_volume'],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    du_total_bytes = int(result.stdout.split()[0])
                    scale_factor = ai_volume_usage_bytes / du_total_bytes if du_total_bytes > 0 else 1.0
                except:
                    pass
            else:
                # For container filesystem, use df to get the scale factor
                try:
                    # Get df total
                    result = subprocess.run(['df', '-BG', '--output=used', '/'],
                                          capture_output=True, text=True, check=True)
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        df_used_gb = int(lines[-1].rstrip('G'))

                        # Get du total for root
                        result = subprocess.run(['du', '--block-size=1', '-s', '--exclude=/ai_network_volume', '--exclude=/workspace', '/'],
                                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10)
                        du_total_bytes = int(result.stdout.split()[0])
                        du_total_gb = du_total_bytes / 1000000000

                        scale_factor = (df_used_gb * 1000000000) / du_total_bytes if du_total_bytes > 0 else 1.0
                except:
                    pass

            # Get total size and apply scale factor
            try:
                result = subprocess.run(['du', '--block-size=1', '-s', target_path],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
                total_bytes = int(result.stdout.split()[0])
                estimated_total_gb = (total_bytes * scale_factor) / 1000000000
                print(f"Estimated disk usage: {estimated_total_gb:.1f} GB")
                print()
            except:
                total_bytes = 0
                estimated_total_gb = 0

            # Get subdirectory breakdown with scaling
            try:
                result = subprocess.run(['du', '--block-size=1', '--max-depth=1', target_path],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
                lines = result.stdout.strip().split('\n')
                subdirs = []

                for line in lines[:-1]:  # Skip total
                    parts = line.split('\t', 1)
                    if len(parts) == 2 and parts[1] != target_path:
                        try:
                            size_bytes = int(parts[0])
                            scaled_bytes = size_bytes * scale_factor
                            size_gb = scaled_bytes / 1000000000
                            if size_gb >= 0.01:  # Show if >= 10 MB
                                if size_gb >= 1:
                                    size_display = f"{size_gb:.1f} GB"
                                else:
                                    size_mb = scaled_bytes / 1000000
                                    size_display = f"{size_mb:.0f} MB"
                                subdirs.append((size_gb, size_display, parts[1]))
                        except ValueError:
                            pass

                if subdirs:
                    subdirs.sort(key=lambda x: x[0], reverse=True)
                    print("Subdirectories:")
                    for _, size, path in subdirs[:20]:
                        print(f"   {path}: {size}")

                    shown_total = sum(x[0] for x in subdirs[:20])
                    if len(subdirs) > 20:
                        other = estimated_total_gb - shown_total
                        if other > 0.01:
                            print(f"   (other): {other:.1f} GB")
                    print()
                    print(f"Total: {estimated_total_gb:.1f} GB")
            except Exception as e:
                print(f"Error analyzing directory: {e}")

            print()
            print("==============================================================")
        else:
            print()
            print(f"Error: {target_path} is not a valid directory")


def get_nproc():
    """Get number of processors"""
    try:
        result = subprocess.run(['nproc', '--all'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


if __name__ == "__main__":
    main()