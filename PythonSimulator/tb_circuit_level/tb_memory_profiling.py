import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator

import time
import psutil
import threading
import gc
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 1. Natural Standard Initialization (Identical to tb_dynamic.py)
# This lets Molmem naturally run its full JIT compilation, system notifications,
# and hardware crossover calibration once upfront before any profiling tables are rendered.
sim_init = MolmemSimulator()
sim_init.addCrossbarMatrix(device_type="default", rows=2, cols=2)
sim_init.addDcSource(row=1, voltage=0.5)
sim_init.addDcSource(col=1, voltage=0.0)
sim_init.run(tEnd=1e-6, min_dt=1e-12, tol=1e-4)
sim_init.programWeights(np.random.uniform(0.1, 0.9, (2, 2)), parallel=True, detailedPrint=False)
del sim_init


class MemoryPeakTracker:
    """Lightweight, 100% thread-safe background peak memory sampler."""
    def __init__(self, pid=None):
        self.process = psutil.Process(pid or os.getpid())
        self.peak_rss = 0.0
        self.base_rss = 0.0
        self.running = False
        self.thread = None

    def _sample(self):
        while self.running:
            try:
                rss = self.process.memory_info().rss / (1024.0 * 1024.0)
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(0.001)

    def __enter__(self):
        self.base_rss = self.process.memory_info().rss / (1024.0 * 1024.0)
        self.peak_rss = self.base_rss
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.05)
        try:
            rss = self.process.memory_info().rss / (1024.0 * 1024.0)
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass


def get_cpu_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def get_gpu_memory_mb():
    if HAS_TORCH and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        max_alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_driver = (total_mem - free_mem) / (1024.0 * 1024.0)
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_alloc': max_alloc,
            'used_driver': used_driver,
            'free': free_mem / (1024.0 * 1024.0),
            'total': total_mem / (1024.0 * 1024.0)
        }
    return {'allocated': 0.0, 'reserved': 0.0, 'max_alloc': 0.0, 'used_driver': 0.0, 'free': 0.0, 'total': 0.0}


def get_dict_resizing_bytes(n):
    """Calculates CPython dictionary hash-table bucket memory for crossbar lookup tables."""
    if n <= 0:
        return 0
    cap = 1 << max(3, int(np.ceil(np.log2(max(8, 1.5 * n)))))
    return cap * 24 * 3  # crossbarDevices + nodalMatrix + devices lookup tables


def calculate_theoretical_gpu_bytes(rows, cols, is_device_basis=False, num_streams=4):
    num_devices = rows * cols
    if is_device_basis:
        base_vram = 3.497 * 1024 * 1024
        pinned_dma_bytes = num_devices * 128.0  # 13 states + 3 pulses DMA pinned memory (128 B/dev)
        unit_bytes = 12.58 * 1024
        total_gpu_bytes = base_vram + num_devices * unit_bytes + pinned_dma_bytes
    else:
        base_vram = 0.972 * 1024 * 1024
        pwl_schedule_bytes = cols * max(rows, cols) * (1.02 * 1024)  # PWL pulse waveform buffers (1.02 KB/line)
        nodal_bytes = cols * ((rows + 2) ** 2) * 8.0
        state_pwl_bytes = num_devices * (10.28 * 1024)
        total_gpu_bytes = base_vram + state_pwl_bytes + nodal_bytes + pwl_schedule_bytes

    return total_gpu_bytes, total_gpu_bytes


def calculate_theoretical_cpu_bytes(rows, cols, threads=None, is_device_basis=False):
    num_devices = rows * cols
    dict_heap_bytes = get_dict_resizing_bytes(num_devices)
    if is_device_basis:
        base_ram = 0.001 * 1024 * 1024
        unit_bytes = 1.61 * 1024
        total_cpu_bytes = base_ram + num_devices * unit_bytes + dict_heap_bytes
    else:
        base_ram = 0.000 * 1024 * 1024
        active_threads = threads if threads is not None else min(os.cpu_count() or 16, max(1, cols))
        thread_workspace = active_threads * (0.10 * 1024 + ((rows + 2) ** 2) * 8 + (rows + 2) * 64)
        unit_bytes = 1.59 * 1024
        total_cpu_bytes = base_ram + num_devices * unit_bytes + thread_workspace + dict_heap_bytes

    return total_cpu_bytes


def calculate_theoretical_gpu_host_bytes(rows, cols, is_device_basis=False):
    num_devices = rows * cols
    dict_heap_bytes = get_dict_resizing_bytes(num_devices)

    if is_device_basis:
        base_host_ram = 0.022 * 1024 * 1024
        staging_bytes = num_devices * 4080.5  # Host heap objects, DMA pinned buffers, and readback arrays
        total_host_bytes = base_host_ram + staging_bytes + dict_heap_bytes
    else:
        base_host_ram = 2.370 * 1024 * 1024
        staging_bytes = num_devices * 1039.4
        total_host_bytes = base_host_ram + staging_bytes + dict_heap_bytes

    return total_host_bytes


def run_cpu_benchmarks(sizes, basis="Column"):
    is_device_basis = (basis == "Device")
    basis_label = "Device Basis (Isolated Single-Cell)" if is_device_basis else "Column Basis (Parallel Floating Column)"
    print("\n" + "="*128)
    print(f"               CPU CROSSBAR MEMORY & PERFORMANCE BENCHMARK - {basis_label.upper()}")
    print("="*128)
    print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. RAM':^12} | {'Base RAM':^11} | {'Peak RAM':^11} | {'Delta RAM':^11} | {'Time (ms)':^11} | {'RAM Err':^9}")
    print("-" * 128)

    cpu_results = []
    for r, c in sizes:
        num_devs = r * c
        active_threads = min(os.cpu_count() or 16, max(1, c))
        est_bytes = calculate_theoretical_cpu_bytes(r, c, threads=active_threads, is_device_basis=is_device_basis)
        est_ram_mb = est_bytes / (1024.0 * 1024.0)

        gc.collect()
        t0 = time.perf_counter()

        with MemoryPeakTracker() as tracker:
            sim = MolmemSimulator(device='cpu')
            sim.addCrossbarMatrix(device_type="default", rows=r, cols=c, weightBits=8, inputBits=8, outputBits=8, detailedPrint=False, UpdatesPer=basis)
            np.random.seed(42)
            weights = np.random.uniform(0.1, 0.9, size=(r, c))
            sim.programWeights(weights, parallel=True, detailedPrint=False)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        base_ram = tracker.base_rss
        peak_ram = tracker.peak_rss
        delta_ram = max(0.0, peak_ram - base_ram)

        ram_ratio_str = f"{delta_ram / max(1e-4, est_ram_mb):.2f}x"

        print(f"{f'{r}x{c}':^12} | {num_devs:^9} | {est_ram_mb:^9.3f} MB | {base_ram:^8.1f} MB | {peak_ram:^8.1f} MB | {delta_ram:^8.2f} MB | {t_elapsed:^9.2f}ms | {ram_ratio_str:^9}")

        cpu_results.append({
            'size': f"{r}x{c}",
            'devs': num_devs,
            'threads': active_threads,
            'est_mb': est_ram_mb,
            'base_ram': base_ram,
            'peak_ram': peak_ram,
            'delta_ram': delta_ram,
            'time_ms': t_elapsed
        })
        del sim
        gc.collect()
    return cpu_results


def run_gpu_benchmarks(sizes, basis="Column"):
    if not HAS_TORCH or not torch.cuda.is_available():
        print(f"\n[!] GPU / CUDA not available on this system. Skipping GPU ({basis}) benchmarks.")
        return []

    is_device_basis = (basis == "Device")
    basis_label = "Device Basis (Batched 1x1 Cells)" if is_device_basis else "Column Basis (Full Crossbar Network)"
    print("\n" + "="*145)
    print(f"               GPU CROSSBAR MEMORY & PERFORMANCE BENCHMARK - {basis_label.upper()}")
    print("="*145)
    print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. VRAM':^12} | {'Peak VRAM':^11} | {'Est. RAM':^11} | {'Delta RAM':^11} | {'Driver VRAM':^13} | {'Time (ms)':^11} | {'VRAM Err':^9} | {'RAM Err':^9}")
    print("-" * 145)

    gpu_results = []
    for r, c in sizes:
        num_devs = r * c
        _, est_bytes_overhead = calculate_theoretical_gpu_bytes(r, c, is_device_basis=is_device_basis)
        est_vram_mb = est_bytes_overhead / (1024.0 * 1024.0)
        est_host_bytes = calculate_theoretical_gpu_host_bytes(r, c, is_device_basis=is_device_basis)
        est_host_mb = est_host_bytes / (1024.0 * 1024.0)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()

        with MemoryPeakTracker() as host_tracker:
            sim = MolmemSimulator(device='cuda')
            sim.addCrossbarMatrix(device_type="default", rows=r, cols=c, weightBits=8, inputBits=8, outputBits=8, detailedPrint=False, UpdatesPer=basis)
            np.random.seed(42)
            weights = np.random.uniform(0.1, 0.9, size=(r, c))
            sim.programWeights(weights, parallel=True, detailedPrint=False)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        cur_mem = get_gpu_memory_mb()
        peak_alloc = cur_mem['max_alloc']
        alloc_vram = cur_mem['allocated']
        driver_vram = cur_mem['used_driver']

        base_ram = host_tracker.base_rss
        peak_ram = host_tracker.peak_rss
        delta_ram = max(0.0, peak_ram - base_ram)

        vram_ratio_str = f"{peak_alloc / max(1e-4, est_vram_mb):.2f}x"
        ram_ratio_str = f"{delta_ram / max(1e-4, est_host_mb):.2f}x"

        print(f"{f'{r}x{c}':^12} | {num_devs:^9} | {est_vram_mb:^9.2f} MB | {peak_alloc:^8.2f} MB | {est_host_mb:^8.2f} MB | {delta_ram:^8.2f} MB | {driver_vram:^10.1f} MB | {t_elapsed:^9.2f}ms | {vram_ratio_str:^9} | {ram_ratio_str:^9}")

        gpu_results.append({
            'size': f"{r}x{c}",
            'devs': num_devs,
            'est_mb': est_vram_mb,
            'est_host_mb': est_host_mb,
            'alloc_vram': alloc_vram,
            'peak_vram': peak_alloc,
            'driver_vram': driver_vram,
            'base_ram': base_ram,
            'peak_ram': peak_ram,
            'delta_ram': delta_ram,
            'time_ms': t_elapsed
        })
        del sim
        torch.cuda.empty_cache()
        gc.collect()
    return gpu_results


def fit_memory_scalars(cpu_col_res, gpu_col_res, cpu_dev_res, gpu_dev_res):
    """Dynamically optimizes memory scalar coefficients via non-linear regression with multi-scale loss."""
    fitted_params = {}

    # 1. Fit GPU Column
    if gpu_col_res:
        N_vals = np.array([item['devs'] for item in gpu_col_res], dtype=np.float64)
        R_vals = np.array([int(item['size'].split('x')[0]) for item in gpu_col_res], dtype=np.float64)
        C_vals = np.array([int(item['size'].split('x')[1]) for item in gpu_col_res], dtype=np.float64)
        Y_obs = np.array([item['peak_vram'] for item in gpu_col_res], dtype=np.float64)
        nodal_mb = C_vals * ((R_vals + 2.0) ** 2) * 8.0 / (1024.0 * 1024.0)
        pwl_mb = C_vals * np.maximum(R_vals, C_vals) * (1.02 * 1024) / (1024.0 * 1024.0)
        dict_mb = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_vals])

        def loss_gpu_col(p):
            pred = p[0] + N_vals * p[1] + nodal_mb + pwl_mb
            return np.sum(((pred - Y_obs) / np.maximum(1.0, Y_obs)) ** 2)

        try:
            from scipy.optimize import minimize
            res = minimize(loss_gpu_col, [0.972, 10.28 / 1024.0], bounds=[(0.01, 5.0), (0.001, 2.0)])
            p_gpu_col = res.x
        except Exception:
            p_gpu_col = [0.972, 10.28 / 1024.0]

        fitted_params['gpu_col'] = {
            'base_mb': p_gpu_col[0],
            'dev_kb': p_gpu_col[1] * 1024.0
        }
        for item in gpu_col_res:
            r = int(item['size'].split('x')[0])
            c = int(item['size'].split('x')[1])
            nodal = c * ((r + 2) ** 2) * 8 / (1024.0 * 1024.0)
            pwl = c * max(r, c) * (1.02 * 1024) / (1024.0 * 1024.0)
            item['fit_mb'] = p_gpu_col[0] + item['devs'] * p_gpu_col[1] + nodal + pwl
            item['fit_ratio'] = item['peak_vram'] / max(1e-4, item['fit_mb'])

        # GPU Column Host RAM Fit (Fit against 16x16+ / devs >= 256 to eliminate initial CUDA driver context setup noise)
        fit_items_gpu_col_host = [item for item in gpu_col_res if item['devs'] >= 256]
        if not fit_items_gpu_col_host:
            fit_items_gpu_col_host = gpu_col_res

        N_host_vals_c = np.array([item['devs'] for item in fit_items_gpu_col_host], dtype=np.float64)
        R_host_vals_c = np.array([int(item['size'].split('x')[0]) for item in fit_items_gpu_col_host], dtype=np.float64)
        C_host_vals_c = np.array([int(item['size'].split('x')[1]) for item in fit_items_gpu_col_host], dtype=np.float64)
        Y_host_col = np.array([item['delta_ram'] for item in fit_items_gpu_col_host], dtype=np.float64)
        nodal_mb_c = C_host_vals_c * ((R_host_vals_c + 2.0) ** 2) * 8.0 / (1024.0 * 1024.0)
        pwl_mb_c = C_host_vals_c * np.maximum(R_host_vals_c, C_host_vals_c) * (1.02 * 1024) / (1024.0 * 1024.0)
        dict_mb_c = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_host_vals_c])

        def loss_gpu_host_col(p):
            pred = p[0] + N_host_vals_c * p[1] + dict_mb_c + pwl_mb_c + nodal_mb_c
            return np.sum(((pred - Y_host_col) / np.maximum(1.0, Y_host_col)) ** 2)
        try:
            from scipy.optimize import minimize
            res_hc = minimize(loss_gpu_host_col, [2.370, 1039.4 / (1024.0 * 1024.0)], bounds=[(0.01, 20.0), (0.0001, 0.10)])
            p_gpu_host_col = res_hc.x
        except Exception:
            p_gpu_host_col = [2.370, 1039.4 / (1024.0 * 1024.0)]

        fitted_params['gpu_col_host'] = {
            'base_mb': p_gpu_host_col[0],
            'dev_b': p_gpu_host_col[1] * 1024.0 * 1024.0
        }
        for item in gpu_col_res:
            dict_m = get_dict_resizing_bytes(item['devs']) / (1024.0 * 1024.0)
            item['fit_host_mb'] = p_gpu_host_col[0] + item['devs'] * p_gpu_host_col[1] + dict_m
            item['fit_host_ratio'] = item['delta_ram'] / max(1e-4, item['fit_host_mb'])

    # 2. Fit GPU Device
    if gpu_dev_res:
        N_vals = np.array([item['devs'] for item in gpu_dev_res], dtype=np.float64)
        Y_obs = np.array([item['peak_vram'] for item in gpu_dev_res], dtype=np.float64)
        pinned_mb = N_vals * 128.0 / (1024.0 * 1024.0)
        dict_mb = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_vals])

        def loss_gpu_dev(p):
            growth = 1.0 + np.maximum(0.0, (N_vals - 4096.0) / 12288.0) * p[2]
            pred = p[0] + N_vals * (p[1] * growth) + pinned_mb
            return np.sum(((pred - Y_obs) / np.maximum(1.0, Y_obs)) ** 2)

        try:
            from scipy.optimize import minimize
            res = minimize(loss_gpu_dev, [3.497, 12.58 / 1024.0, 0.0], bounds=[(0.1, 30.0), (0.001, 1.0), (0.0, 10.0)])
            p_gpu_dev = res.x
        except Exception:
            p_gpu_dev = [3.497, 12.58 / 1024.0, 0.0]

        fitted_params['gpu_dev'] = {
            'base_mb': p_gpu_dev[0],
            'dev_kb': p_gpu_dev[1] * 1024.0,
            'growth': p_gpu_dev[2]
        }
        for item in gpu_dev_res:
            growth = 1.0 + max(0.0, (item['devs'] - 4096.0) / 12288.0) * p_gpu_dev[2]
            pinned = item['devs'] * 128.0 / (1024.0 * 1024.0)
            item['fit_mb'] = p_gpu_dev[0] + item['devs'] * (p_gpu_dev[1] * growth) + pinned
            item['fit_ratio'] = item['peak_vram'] / max(1e-4, item['fit_mb'])

        # GPU Device Host RAM Fit (Fit against 16x16+ / devs >= 256 to eliminate initial CUDA driver context setup noise)
        fit_items_gpu_dev_host = [item for item in gpu_dev_res if item['devs'] >= 256]
        if not fit_items_gpu_dev_host:
            fit_items_gpu_dev_host = gpu_dev_res

        N_host_vals_d = np.array([item['devs'] for item in fit_items_gpu_dev_host], dtype=np.float64)
        Y_host_dev = np.array([item['delta_ram'] for item in fit_items_gpu_dev_host], dtype=np.float64)
        dict_mb_d = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_host_vals_d])

        def loss_gpu_host_dev(p):
            pred = p[0] + N_host_vals_d * p[1] + dict_mb_d
            return np.sum(((pred - Y_host_dev) / np.maximum(1.0, Y_host_dev)) ** 2)
        try:
            from scipy.optimize import minimize
            res_hd = minimize(loss_gpu_host_dev, [0.022, 4080.5 / (1024.0 * 1024.0)], bounds=[(0.001, 20.0), (0.0001, 0.10)])
            p_gpu_host_dev = res_hd.x
        except Exception:
            p_gpu_host_dev = [0.022, 4080.5 / (1024.0 * 1024.0)]

        fitted_params['gpu_dev_host'] = {
            'base_mb': p_gpu_host_dev[0],
            'dev_b': p_gpu_host_dev[1] * 1024.0 * 1024.0
        }
        for item in gpu_dev_res:
            dict_m = get_dict_resizing_bytes(item['devs']) / (1024.0 * 1024.0)
            item['fit_host_mb'] = p_gpu_host_dev[0] + item['devs'] * p_gpu_host_dev[1] + dict_m
            item['fit_host_ratio'] = item['delta_ram'] / max(1e-4, item['fit_host_mb'])

    # 3. Fit CPU Column
    if cpu_col_res:
        # Filter to ignore <= 32x32 (devs <= 1024) to eliminate OS page-table allocation noise
        fit_items = [item for item in cpu_col_res if item['devs'] > 1024]
        if not fit_items:
            fit_items = cpu_col_res

        N_vals = np.array([item['devs'] for item in fit_items], dtype=np.float64)
        R_vals = np.array([int(item['size'].split('x')[0]) for item in fit_items], dtype=np.float64)
        C_vals = np.array([int(item['size'].split('x')[1]) for item in fit_items], dtype=np.float64)
        T_vals = np.array([item.get('threads', min(16, max(1, c))) for c in C_vals], dtype=np.float64)
        Y_obs = np.array([item['delta_ram'] for item in fit_items], dtype=np.float64)
        dict_mb = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_vals])
        pwl_mb = C_vals * np.maximum(R_vals, C_vals) * (1.02 * 1024) / (1024.0 * 1024.0)

        def loss_cpu_col(p):
            t_work = T_vals * (p[2] + ((R_vals + 2.0) ** 2) * 8.0 / (1024.0 * 1024.0) + (R_vals + 2.0) * p[3])
            unit_cost = p[1] + np.minimum(p[4], (N_vals / 65536.0) * p[4])
            pred = p[0] + N_vals * unit_cost + t_work + dict_mb
            return np.sum(((pred - Y_obs) / np.maximum(1.0, Y_obs)) ** 2)

        try:
            from scipy.optimize import minimize
            res = minimize(loss_cpu_col, [0.000, 1.59 / 1024.0, 0.10 / 1024.0, 64.0 / (1024.0 * 1024.0), 0.0],
                           bounds=[(0.0, 2.0), (0.0001, 0.05), (0.0001, 0.5), (0.0, 0.01), (0.0, 0.01)])
            p_cpu_col = res.x
        except Exception:
            p_cpu_col = [0.000, 1.59 / 1024.0, 0.10 / 1024.0, 64.0 / (1024.0 * 1024.0), 0.0]

        fitted_params['cpu_col'] = {
            'base_mb': p_cpu_col[0],
            'dev_kb': (p_cpu_col[1] + p_cpu_col[4]) * 1024.0,
            'thread_kb': p_cpu_col[2] * 1024.0,
            'thread_row_b': p_cpu_col[3] * 1024.0 * 1024.0
        }
        for item in cpu_col_res:
            r = int(item['size'].split('x')[0])
            c = int(item['size'].split('x')[1])
            dict_m = get_dict_resizing_bytes(item['devs']) / (1024.0 * 1024.0)
            t_cnt = item.get('threads', min(16, max(1, c)))
            t_work = t_cnt * (p_cpu_col[2] + ((r + 2.0) ** 2) * 8.0 / (1024.0 * 1024.0) + (r + 2.0) * p_cpu_col[3])
            unit_cost = p_cpu_col[1] + min(p_cpu_col[4], (item['devs'] / 65536.0) * p_cpu_col[4])
            item['fit_mb'] = p_cpu_col[0] + item['devs'] * unit_cost + t_work + dict_m
            item['fit_ratio'] = item['delta_ram'] / max(1e-4, item['fit_mb'])

    # 4. Fit CPU Device
    if cpu_dev_res:
        # Filter to ignore <= 32x32 (devs <= 1024) to eliminate OS page-table allocation noise
        fit_items_dev = [item for item in cpu_dev_res if item['devs'] > 1024]
        if not fit_items_dev:
            fit_items_dev = cpu_dev_res

        N_vals = np.array([item['devs'] for item in fit_items_dev], dtype=np.float64)
        Y_obs = np.array([item['delta_ram'] for item in fit_items_dev], dtype=np.float64)
        dict_mb = np.array([get_dict_resizing_bytes(n) / (1024.0 * 1024.0) for n in N_vals])

        def loss_cpu_dev(p):
            unit = p[1] + np.minimum(p[2], (N_vals / 16384.0) * p[2])
            pred = p[0] + N_vals * unit + dict_mb
            return np.sum(((pred - Y_obs) / np.maximum(1.0, Y_obs)) ** 2)

        try:
            from scipy.optimize import minimize
            res = minimize(loss_cpu_dev, [0.001, 1.61 / 1024.0, 0.0 / 1024.0],
                           bounds=[(0.0001, 5.0), (0.0001, 0.01), (0.0, 0.01)])
            p_cpu_dev = res.x
        except Exception:
            p_cpu_dev = [0.001, 1.61 / 1024.0, 0.0 / 1024.0]

        fitted_params['cpu_dev'] = {
            'base_mb': p_cpu_dev[0],
            'dev_kb': p_cpu_dev[1] * 1024.0,
            'exp_kb': p_cpu_dev[2] * 1024.0
        }
        for item in cpu_dev_res:
            dict_m = get_dict_resizing_bytes(item['devs']) / (1024.0 * 1024.0)
            unit = p_cpu_dev[1] + min(p_cpu_dev[2], (item['devs'] / 16384.0) * p_cpu_dev[2])
            item['fit_mb'] = p_cpu_dev[0] + item['devs'] * unit + dict_m
            item['fit_ratio'] = item['delta_ram'] / max(1e-4, item['fit_mb'])

    return fitted_params


if __name__ == "__main__":
    gpu_test_sizes = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128)
    ]

    cpu_test_sizes = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512)
    ]

    # --- 1. Column Basis Benchmarks ---
    cpu_col_res = run_cpu_benchmarks(cpu_test_sizes, basis="Column")
    gpu_col_res = run_gpu_benchmarks(gpu_test_sizes, basis="Column")

    # --- 2. Device Basis Benchmarks ---
    cpu_dev_res = run_cpu_benchmarks(cpu_test_sizes, basis="Device")
    gpu_dev_res = run_gpu_benchmarks(gpu_test_sizes, basis="Device")

    # --- 3. Dynamic Scalar Regression Optimizer ---
    calibrated_scalars = fit_memory_scalars(cpu_col_res, gpu_col_res, cpu_dev_res, gpu_dev_res)

    # -----------------------------------------------------------------------------------------
    # FINAL SUMMARY REPORTS: 4 DEDICATED TABLES (COLUMN & DEVICE FOR CPU & GPU)
    # -----------------------------------------------------------------------------------------

    # Table 1: CPU Column Summary
    print("\n" + "="*128)
    print("         TABLE 1: CPU CROSSBAR MEMORY & ESTIMATION SUMMARY (Column Basis - Parallel Floating Column)")
    print("="*128)
    print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. RAM':^12} | {'Base RAM':^11} | {'Peak RAM':^11} | {'Delta RAM':^11} | {'Time (ms)':^11} | {'Pre-Fit Err':^11} | {'Post-Fit Err':^12}")
    print("-" * 128)
    for c_item in cpu_col_res:
        pre_ratio = f"{c_item['delta_ram'] / max(1e-4, c_item['est_mb']):.2f}x"
        post_ratio = f"{c_item['fit_ratio']:.2f}x"
        print(f"{c_item['size']:^12} | {c_item['devs']:^9} | {c_item['est_mb']:^9.3f} MB | {c_item['base_ram']:^8.1f} MB | {c_item['peak_ram']:^8.1f} MB | {c_item['delta_ram']:^8.2f} MB | {c_item['time_ms']:^9.2f}ms | {pre_ratio:^11} | {post_ratio:^12}")
    print("="*128 + "\n")

    # Table 2: GPU Column Summary (Dual VRAM + Host System RAM)
    if gpu_col_res:
        print("="*160)
        print("         TABLE 2: GPU CROSSBAR MEMORY & ESTIMATION SUMMARY (Column Basis - Full Crossbar Network)")
        print("="*160)
        print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. VRAM':^12} | {'Peak VRAM':^11} | {'VRAM Err':^10} | {'Est. Host RAM':^15} | {'Delta Host RAM':^16} | {'Host RAM Err':^14} | {'Time (ms)':^11}")
        print("-" * 160)
        for g_item in gpu_col_res:
            vram_pre = f"{g_item['peak_vram'] / max(1e-4, g_item['est_mb']):.2f}x"
            vram_post = f"{g_item['fit_ratio']:.2f}x"
            vram_err = f"{vram_pre} / {vram_post}"
            host_pre = f"{g_item['delta_ram'] / max(1e-4, g_item['est_host_mb']):.2f}x"
            host_post = f"{g_item['fit_host_ratio']:.2f}x"
            host_err = f"{host_pre} / {host_post}"
            print(f"{g_item['size']:^12} | {g_item['devs']:^9} | {g_item['est_mb']:^9.3f} MB | {g_item['peak_vram']:^8.2f} MB | {vram_err:^10} | {g_item['est_host_mb']:^12.3f} MB | {g_item['delta_ram']:^13.2f} MB | {host_err:^14} | {g_item['time_ms']:^9.2f}ms")
        print("="*160 + "\n")

    # Table 3: CPU Device Summary
    print("="*128)
    print("         TABLE 3: CPU CROSSBAR MEMORY & ESTIMATION SUMMARY (Device Basis - Isolated Single-Cell)")
    print("="*128)
    print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. RAM':^12} | {'Base RAM':^11} | {'Peak RAM':^11} | {'Delta RAM':^11} | {'Time (ms)':^11} | {'Pre-Fit Err':^11} | {'Post-Fit Err':^12}")
    print("-" * 128)
    for c_item in cpu_dev_res:
        pre_ratio = f"{c_item['delta_ram'] / max(1e-4, c_item['est_mb']):.2f}x"
        post_ratio = f"{c_item['fit_ratio']:.2f}x"
        print(f"{c_item['size']:^12} | {c_item['devs']:^9} | {c_item['est_mb']:^9.3f} MB | {c_item['base_ram']:^8.1f} MB | {c_item['peak_ram']:^8.1f} MB | {c_item['delta_ram']:^8.2f} MB | {c_item['time_ms']:^9.2f}ms | {pre_ratio:^11} | {post_ratio:^12}")
    print("="*128 + "\n")

    # Table 4: GPU Device Summary (Dual VRAM + Host System RAM)
    if gpu_dev_res:
        print("="*160)
        print("         TABLE 4: GPU CROSSBAR MEMORY & ESTIMATION SUMMARY (Device Basis - Batched 1x1 Cells)")
        print("="*160)
        print(f"{'Matrix Size':^12} | {'Devices':^9} | {'Est. VRAM':^12} | {'Peak VRAM':^11} | {'VRAM Err':^10} | {'Est. Host RAM':^15} | {'Delta Host RAM':^16} | {'Host RAM Err':^14} | {'Time (ms)':^11}")
        print("-" * 160)
        for g_item in gpu_dev_res:
            vram_pre = f"{g_item['peak_vram'] / max(1e-4, g_item['est_mb']):.2f}x"
            vram_post = f"{g_item['fit_ratio']:.2f}x"
            vram_err = f"{vram_pre} / {vram_post}"
            host_pre = f"{g_item['delta_ram'] / max(1e-4, g_item['est_host_mb']):.2f}x"
            host_post = f"{g_item['fit_host_ratio']:.2f}x"
            host_err = f"{host_pre} / {host_post}"
            print(f"{g_item['size']:^12} | {g_item['devs']:^9} | {g_item['est_mb']:^9.3f} MB | {g_item['peak_vram']:^8.2f} MB | {vram_err:^10} | {g_item['est_host_mb']:^12.3f} MB | {g_item['delta_ram']:^13.2f} MB | {host_err:^14} | {g_item['time_ms']:^9.2f}ms")
        print("="*160 + "\n")

    # -----------------------------------------------------------------------------------------
    # FINAL CALIBRATION REPORT: OPTIMAL SCALAR PARAMETERS
    # -----------------------------------------------------------------------------------------
    print("="*145)
    print("                          AUTO-CALIBRATED HARDWARE MEMORY SCALARS & REGRESSION REPORT")
    print("="*145)
    if 'gpu_col' in calibrated_scalars:
        gc_p = calibrated_scalars['gpu_col']
        print(f"  > GPU Column Basis (VRAM) : Base = {gc_p['base_mb']:.3f} MB | State+PWL = {gc_p['dev_kb']:.2f} KB/dev | Nodal Matrix = 8.00 B/float | PWL Buffers = 1.02 KB/line")
    if 'gpu_col_host' in calibrated_scalars:
        gch_p = calibrated_scalars['gpu_col_host']
        print(f"  > GPU Column Basis (Host) : Base = {gch_p['base_mb']:.3f} MB | Staging Payload = {gch_p['dev_b']:.1f} B/dev | Dict Resizing = Power-of-2")
    if 'gpu_dev' in calibrated_scalars:
        gd_p = calibrated_scalars['gpu_dev']
        print(f"  > GPU Device Basis (VRAM) : Base = {gd_p['base_mb']:.3f} MB | Linear Item = {gd_p['dev_kb']:.2f} KB/dev | Super-Growth = {gd_p['growth']:.3f}x | Pinned DMA = 128.0 B/dev")
    if 'gpu_dev_host' in calibrated_scalars:
        gdh_p = calibrated_scalars['gpu_dev_host']
        print(f"  > GPU Device Basis (Host) : Base = {gdh_p['base_mb']:.3f} MB | Staging Payload = {gdh_p['dev_b']:.1f} B/dev | Dict Resizing = Power-of-2")
    if 'cpu_col' in calibrated_scalars:
        cc_p = calibrated_scalars['cpu_col']
        print(f"  > CPU Column Basis (RAM)  : Base = {cc_p['base_mb']:.3f} MB | Device Data = {cc_p['dev_kb']:.2f} KB/dev | Thread Workspace = {cc_p['thread_kb']:.2f} KB/thread | Dict Resizing = Power-of-2")
    if 'cpu_dev' in calibrated_scalars:
        cd_p = calibrated_scalars['cpu_dev']
        print(f"  > CPU Device Basis (RAM)  : Base = {cd_p['base_mb']:.3f} MB | Device Data = {cd_p['dev_kb']:.2f} KB/dev | Page Expansion = {cd_p['exp_kb']:.2f} KB/dev | Dict Resizing = Power-of-2")
    print("="*145 + "\n")
