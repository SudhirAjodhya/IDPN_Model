"""
check_cuda.py
=============
Quick diagnostic script to verify CUDA availability, GPU device name,
VRAM allocation, and basic tensor computation.

Usage:
    python scripts/diagnostics/check_cuda.py
"""

import time
import torch


def run_test():
    print("=" * 60)
    print("Initializing Hardware & CUDA Test...")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
        print(f"GPU Detected: {gpu_name}\n")
    else:
        device = torch.device("cpu")
        print("No CUDA device found. Falling back to CPU\n")

    try:
        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            print(f"- Memory currently allocated: {mem_alloc:.2f} MB\n")

            print("Generating tensors and moving to GPU...")
            size = 5000
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            print(f"Computing {size}x{size} matrix multiplication...")
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()

            print("-" * 60)
            print(f"Computation successful! Time taken: {end_time - start_time:.4f} seconds")
        else:
            print("Running CPU sanity check...")
            a = torch.randn(1000, 1000)
            b = torch.randn(1000, 1000)
            c = torch.matmul(a, b)
            print("CPU computation successful!")

    except torch.cuda.OutOfMemoryError:
        print("ERROR: Out of GPU Memory! Try reducing the 'size' variable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_test()
