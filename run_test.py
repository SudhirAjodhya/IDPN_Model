import torch
import time
import sys

def run_test():
    print("="*60)
    print("Initializing Hardware Test...")
    print("="*60)

    # Check if device can use CUDA
    if torch.cuda.is_available():
        # Get hardware device
        print("Checking CUDA compatible hardware device...")

        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)

        print(f"GPU Detected: {gpu_name} \n")    
    else:
        device = torch.device("cpu")
        print("No CUDA device found. Falling back to CPU\n")


    try:
        if device.type == 'cuda':    

            # Check allocated memory
            print("Checking allocated for allocated memory...")
            mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            print(f"- Memory currently allocated: {mem_alloc:.2f} MB \n")

            # Matrix Muplication test
            print("Generating tensors and moving to GPU...")
            size = 5000
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            print(f"Computing {size}x{size} matrix multiplication...")
            start_time = time.time()
    
            # Standard PyTorch ops are asynchronous, so we synchronize to get an accurate time
            c = torch.matmul(a, b)

            if device.type == 'cude':
                torch.cuda.synchronize() 
    
            end_time = time.time()
            print("-"*60)
            print(f"Computation successful! Time taken: {end_time - start_time:.4f} seconds")
    except torch.cuda.OutOfMemoryError:
        print("ERROR: Out of GPU Memory! Try reducing the 'size' variable.")
    except Exception as e:
        print("An unexpected error occurred: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_test()
    else:
        print("Error: CUDA is still not available.")