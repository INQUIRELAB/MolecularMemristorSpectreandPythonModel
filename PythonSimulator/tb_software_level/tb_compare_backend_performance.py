import sys
import os
import time
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator
import numpy as np

def run_single_benchmark(size, backend_type, device_type):
    """
    Instantiates a crossbar matrix and times updateCrossbarWeights.
    Returns: (elapsed_time, throughput_cols_per_sec, hybrid_dist)
    """
    sim = MolmemSimulator(device=device_type, backend=backend_type)
    sim.detailedPrint = False
    sim.addCrossbarMatrix(
        rows=size,
        cols=size,
        weightBits=14,
        inputBits=14,
        outputBits=14,
        detailedPrint=False,
        architecture='1T1R',
        force_device_init=False,
        UpdatesPer="Column"
    )
    sim.programmer.calibrate()
    sim.isCalibrated = True
    
    # Deterministic reproducible target weights
    rng = np.random.RandomState(42 + size)
    target_weights = rng.uniform(0.15, 0.85, size=(size, size))
    
    # Garbage collection before timing
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    t0 = time.perf_counter()
    sim.updateCrossbarWeights(target_weights, detailedPrint=False, fineTune=False, recordHistory=False)
    elapsed = time.perf_counter() - t0
    
    throughput = float(size) / max(1e-6, elapsed)
    
    hybrid_dist = None
    if backend_type == 'hybrid':
        dist = getattr(MolmemSimulator, '_last_hybrid_distribution', None)
        if dist is not None and len(dist) == 3:
            hybrid_dist = dist  # (cpu_cols, gpu_cols, total_cols)
            
    del sim
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
        
    return elapsed, throughput, hybrid_dist

def main():
    sizes = [64, 128, 256, 512, 1024]
    results = {}
    
    print("\n=========================================================================================================")
    print("                      MOLMEM HETEROGENEOUS BACKEND COMPARISON BENCHMARK")
    print("=========================================================================================================")
    print(f"Matrix Configurations to Test : {', '.join([f'{s}x{s}' for s in sizes])}")
    print("Backends Evaluated            : CPU (Numba) vs. GPU (ROCm/CUDA) vs. Hybrid (Heterogeneous Co-Execution)")
    print("=========================================================================================================\n")
    
    for size in sizes:
        print(f"[*] Benchmarking Crossbar Size {size}x{size} ({size * size:,} devices)...")
        results[size] = {}
        
        # 1. CPU Benchmark
        sys.stdout.write("    >> [1/3] CPU (Numba / Multi-Core)       ... ")
        sys.stdout.flush()
        t_cpu, tp_cpu, _ = run_single_benchmark(size, backend_type='numba', device_type='cpu')
        print(f"Done in {t_cpu:6.2f}s ({tp_cpu:6.1f} cols/s)")
        results[size]['cpu'] = (t_cpu, tp_cpu)
        
        # 2. GPU Benchmark
        sys.stdout.write("    >> [2/3] GPU (Native ROCm / CUDA)       ... ")
        sys.stdout.flush()
        t_gpu, tp_gpu, _ = run_single_benchmark(size, backend_type='torch', device_type='cuda')
        print(f"Done in {t_gpu:6.2f}s ({tp_gpu:6.1f} cols/s)")
        results[size]['gpu'] = (t_gpu, tp_gpu)
        
        # 3. Hybrid Benchmark
        sys.stdout.write("    >> [3/3] Hybrid (CPU + GPU Co-Execution)... ")
        sys.stdout.flush()
        t_hyb, tp_hyb, dist = run_single_benchmark(size, backend_type='hybrid', device_type='cuda')
        
        if dist is not None and dist[2] > 0:
            cpu_c, gpu_c, tot_c = dist
            dist_str = f"CPU: {cpu_c} cols ({cpu_c/tot_c*100:.1f}%) | GPU: {gpu_c} cols ({gpu_c/tot_c*100:.1f}%)"
        else:
            dist_str = "N/A"
            
        print(f"Done in {t_hyb:6.2f}s ({tp_hyb:6.1f} cols/s) [{dist_str}]")
        results[size]['hybrid'] = (t_hyb, tp_hyb, dist_str)
        print()

    # Final Comparison Table
    print("\n" + "=" * 120)
    print(f"{'MOLMEM PERFORMANCE & WORKLOAD DISTRIBUTION SUMMARY':^120}")
    print("=" * 120)
    header = f"{'Size':<9} | {'CPU Time (Thpt)':<18} | {'GPU Time (Thpt)':<18} | {'Hybrid Time (Thpt)':<19} | {'Speedup/CPU':<11} | {'Speedup/GPU':<11} | {'Hybrid Workload Split':<20}"
    print(header)
    print("-" * 120)
    
    for size in sizes:
        t_cpu, tp_cpu = results[size]['cpu']
        t_gpu, tp_gpu = results[size]['gpu']
        t_hyb, tp_hyb, dist_str = results[size]['hybrid']
        
        speedup_cpu = t_cpu / max(1e-6, t_hyb)
        speedup_gpu = t_gpu / max(1e-6, t_hyb)
        
        cpu_str = f"{t_cpu:5.2f}s ({tp_cpu:5.1f} c/s)"
        gpu_str = f"{t_gpu:5.2f}s ({tp_gpu:5.1f} c/s)"
        hyb_str = f"{t_hyb:5.2f}s ({tp_hyb:5.1f} c/s)"
        
        size_str = f"{size}x{size}"
        sp_cpu_str = f"{speedup_cpu:5.2f}x"
        sp_gpu_str = f"{speedup_gpu:5.2f}x"
        
        row = f"{size_str:<9} | {cpu_str:<18} | {gpu_str:<18} | {hyb_str:<19} | {sp_cpu_str:<11} | {sp_gpu_str:<11} | {dist_str}"
        print(row)
        
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()
