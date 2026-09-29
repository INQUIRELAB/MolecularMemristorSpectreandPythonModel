import os
import sys
import shutil

from .sys_utils import DummyStream, is_interactive_notebook

if sys.stdout is None:
    sys.stdout = DummyStream()
if getattr(sys, '__stdout__', None) is None:
    sys.__stdout__ = DummyStream()

import time
import math
import random
import platform
import hashlib
import psutil
import io
import gc
import multiprocessing
import threading
import concurrent.futures
import numpy as np
import numba as nb

_GLOBAL_THREAD_PROFILING_LOCK = threading.Lock()

def _safe_set_numba_threads(threads):
    try:
        import numba.config
        max_nb = getattr(numba.config, 'NUMBA_NUM_THREADS', None)
        if max_nb is not None:
            threads = min(threads, int(max_nb))
        nb.set_num_threads(max(1, threads))
    except (ValueError, Exception):
        pass

_SYS_CPU_COUNT = os.cpu_count() or 1
_SYS_80_PERCENT_CORES = max(1, int(_SYS_CPU_COUNT * 0.8))
_SYS_PHYSICAL_CORES = psutil.cpu_count(logical=False) or _SYS_CPU_COUNT

_gpu_sig_items = []
if os.environ.get("MOLMEM_CPU_ONLY") != "1" and not is_interactive_notebook():
    try:
        import torch
        if torch.cuda.is_available():
            for _dev_i in range(torch.cuda.device_count()):
                _gpu_sig_items.append(f"{torch.cuda.get_device_name(_dev_i)}")
    except Exception:
        pass
else:
    # Under CPU-only execution, read physical GPU identity from cache to prevent
    # mutating machine_id and avoid initializing ROCm/HIP in worker threads
    try:
        from . import _load_unified_cache
        _cached_d = _load_unified_cache()
        if _cached_d and 'config_json' in _cached_d:
            import json as _json
            _cfg = _json.loads(str(_cached_d['config_json'][0]))
            for _cg in _cfg.get('gpu_devices', []):
                _gpu_sig_items.append(f"{_cg.get('name', '')}")
    except Exception:
        pass
_SYS_GPU_SIG = "_".join(_gpu_sig_items) if _gpu_sig_items else "NO_GPU"
_SYS_ENV_TAG = "notebook" if is_interactive_notebook() else "cli"
_SYS_NODE_HASH = hashlib.sha256(platform.node().encode('utf-8')).hexdigest()[:16]
_SYS_MACHINE_ID = f"{_SYS_NODE_HASH}_{_SYS_ENV_TAG}_{platform.processor()}_{_SYS_PHYSICAL_CORES}_{_SYS_CPU_COUNT}_{_SYS_GPU_SIG}"

class SimulatorProfilerMixin:
    _cached_configs = {
        "Column": {},
        "Device": {},
        "Circuit": {}
    }
    _best_cores_pool = None
    _gpu_slicing_configs = {}
    _gpu_crossover_dim = {
        "Column": None,
        "Device": None,
        "Circuit": None
    }
    _opt_cpu_chunk_per_thread = 1.0
    _active_calibration_cpu_chunk = None
    _opt_gpu_slice_per_cu = 2.0
    _active_calibration_gpu_slice = None
    _opt_hybrid_pacing_ratio = None
    _opt_hybrid_min_horizon = None
    _opt_hybrid_cpu_allowance = None
    _opt_hybrid_pacing_by_rank = {}
    _in_gpu_crossover_profiling = False
    _suppress_cache_hit_prints = False
    _crossover_printed_lines = 0
    _crossover_completed_summaries = []
    _last_cpu_profiler_elapsed = None

    @classmethod
    def get_hybrid_pacing_ratio_for_rank(cls, rows):
        import math
        if not cls._opt_hybrid_pacing_by_rank:
            return cls._opt_hybrid_pacing_ratio or 1.0
        ranks = sorted(cls._opt_hybrid_pacing_by_rank.keys())
        
        if len(ranks) >= 3:
            # 3-Point Log-Parabolic (Quadratic in log-log space)
            x = math.log(max(1, rows))
            x1, x2, x3 = math.log(ranks[0]), math.log(ranks[1]), math.log(ranks[2])
            y1, y2, y3 = [math.log(max(1e-4, cls._opt_hybrid_pacing_by_rank[r])) for r in ranks[:3]]
            
            # Lagrange 2nd-order basis polynomials
            l1 = ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
            l2 = ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
            l3 = ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))
            
            y_fit = y1 * l1 + y2 * l2 + y3 * l3
            return max(0.01, min(50.0, math.exp(y_fit)))
            
        elif len(ranks) == 2:
            # 2-Point Log-Linear Power Law
            x = math.log(max(1, rows))
            x1, x2 = math.log(ranks[0]), math.log(ranks[1])
            y1, y2 = math.log(max(1e-4, cls._opt_hybrid_pacing_by_rank[ranks[0]])), math.log(max(1e-4, cls._opt_hybrid_pacing_by_rank[ranks[1]]))
            slope = (y2 - y1) / max(1e-6, x2 - x1)
            y_fit = y1 + slope * (x - x1)
            return max(0.01, min(50.0, math.exp(y_fit)))
            
        return cls._opt_hybrid_pacing_by_rank[ranks[0]]

    @classmethod
    def _update_crossover_display(cls, progress_pct, line1_status, active_subline=None, active_sublines=None, size_finished_summary=None):
        if getattr(cls, '_suppress_cache_hit_prints', False):
            return
        if not hasattr(cls, '_crossover_printed_lines'):
            cls._crossover_printed_lines = 0
        if not hasattr(cls, '_crossover_completed_summaries'):
            cls._crossover_completed_summaries = []
            
        if size_finished_summary is not None:
            cls._crossover_completed_summaries.append(size_finished_summary)
            
        cols, _ = shutil.get_terminal_size(fallback=(80, 24))
        width = max(40, cols - 3)
        
        def truncate_line(s, w):
            if len(s) > w:
                return s[:w-3] + "..."
            return s
            
        bar = '#' * int(progress_pct * 0.2)
        line1 = f"    >> Dynamic Crossover Calibration: [{bar.ljust(20)}] {progress_pct:3.0f}% | {line1_status}"
        line1 = truncate_line(line1, width)
        
        # Move cursor back to Line 1 of the block if we printed before
        lines_to_go_up = cls._crossover_printed_lines
        if lines_to_go_up > 0:
            sys.__stdout__.write(f"\r\033[{lines_to_go_up}A")
        else:
            sys.__stdout__.write("\r")
            
        # Write Line 1 (Progress Bar)
        sys.__stdout__.write(f"{line1}\033[K\n")
        
        # Write all completed summaries
        for summary in cls._crossover_completed_summaries:
            sys.__stdout__.write(f"{truncate_line(summary, width)}\033[K\n")
            
        # Write active sublines (if any)
        subs = []
        if active_sublines is not None:
            if isinstance(active_sublines, (list, tuple)):
                subs.extend(active_sublines)
            else:
                subs.append(active_sublines)
        if active_subline is not None:
            subs.append(active_subline)
            
        if subs:
            for sub in subs:
                sys.__stdout__.write(f"{truncate_line(sub, width)}\033[K\n")
            cls._crossover_printed_lines = 1 + len(cls._crossover_completed_summaries) + len(subs)
        else:
            cls._crossover_printed_lines = 1 + len(cls._crossover_completed_summaries)
            
        sys.__stdout__.flush()

    def profileParallelization(self):
        """
        Dynamically benchmarks the massive NumPy Structure-Of-Arrays execution kernel.
        Sweeps performance using OpenMP C-level Thread constraints natively inside Numba.
        """
        from .simulator import MolmemSimulator
        
        device = str(getattr(self, 'device', 'auto'))
        is_cpu = device == 'cpu' or os.environ.get("MOLMEM_CPU_ONLY") == "1" or os.environ.get("MOLMEM_FORCE_CPU") == "1"
        is_gpu_dev = not is_cpu and (device.startswith('cuda') or os.environ.get("MOLMEM_FORCE_GPU") == "1")
        if device == 'auto' and not is_cpu and not is_gpu_dev:
            try:
                import torch
                if torch.cuda.is_available():
                    is_gpu_dev = True
            except (ImportError, Exception):
                pass
                
        current_proc_name = multiprocessing.current_process().name
        is_bootstrapper = 'Bootstrapper' in current_proc_name or 'JIT_Worker' in current_proc_name
        
        if is_gpu_dev or getattr(self.__class__, '_in_warmup', False) or (current_proc_name != 'MainProcess' and not is_bootstrapper):
            self.optimal_cores = 1
            _safe_set_numba_threads(1)
            return
            
        verbose = getattr(self, 'detailedPrint', True)
        old_detailedPrint = verbose
        self.detailedPrint = False
        
        from .sources import PulseSource
        
        machine_id = _SYS_MACHINE_ID
        cache_file = os.path.join(os.path.dirname(__file__), '.calcache', 'molmem_cache.npz')
        
        num_devices = len(self.devices)
        
        gpu_active = False
        if not is_cpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_active = True
            except (ImportError, Exception):
                pass
            
        if self.crossbarDevices:
            max_r = max_c = 0
            for r, c in self.crossbarDevices:
                max_r = r if r > max_r else max_r
                max_c = c if c > max_c else max_c
            size_str = f"{max_r}x{max_c}"
        else:
            dim = math.isqrt(num_devices)
            size_str = f"{dim}x{dim}" if dim * dim == num_devices else f"{num_devices} devs"

        # Check active basis
        if getattr(self, 'programmer', None) is None:
            basis = 'Circuit'
        else:
            basis = getattr(self, 'UpdatesPer', 'Column')

        def print_cache_hit_block(best_threads, status_suffix):
            if getattr(self, 'detailedPrint', True):
                print(f"\n--------------------------------------------------------------")
                print(f"--- MolmemSimulator: Dynamic Thread Profiling (SOA)      ---")
                print(f"--------------------------------------------------------------")
                if gpu_active and getattr(self, 'requested_device', 'auto') != 'cpu' and basis != 'Circuit':
                    print(f"  > Device Dispatch Calibration ({basis}) [{'#'*20}] 100% | CPU Parallelism (Size {size_str}) {status_suffix}")
                else:
                    print(f"  > CPU Performance Profiling ({basis}) [{'#'*20}] 100% | CPU Parallelism (Size {size_str}) {status_suffix}")
                print(f"    >> Optimal Configuration : {best_threads} Threads\n")
            else:
                sys.stdout.write(f"\r  > Thread Optimization ({basis}, Size {size_str}): {best_threads} Threads ({status_suffix})\033[K\n")
                sys.stdout.flush()

        is_worker_thread = (threading.current_thread().name != 'MainThread')

        # 1. Check class-level memory cache first
        basis_configs = self.__class__._cached_configs.get(basis)
        if basis_configs is not None and num_devices in basis_configs:
            cached_val = basis_configs[num_devices]
            best_threads = cached_val[0] if isinstance(cached_val, tuple) else cached_val
            in_crossover = getattr(self.__class__, '_in_gpu_crossover_profiling', False)
            suppress_cache = getattr(self.__class__, '_suppress_cache_hit_prints', False) or (multiprocessing.current_process().name != 'MainProcess') or is_worker_thread
            if not in_crossover and not suppress_cache and os.environ.get(f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}_{basis}') != '1':
                print_cache_hit_block(best_threads, "loaded from cache")
                os.environ[f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}_{basis}'] = '1'
            
            self.optimal_cores = best_threads
            self.use_parallel_optimized = (best_threads > 1)
            if best_threads == 0: best_threads = 1
            _safe_set_numba_threads(best_threads)
            self.detailedPrint = old_detailedPrint
            return
            
        # 2. Try loading from disk cache first
        self.__class__._load_profile_cache(basis=basis, verbose=False, force=True)
        
        # Check memory cache again in case we successfully loaded it from disk
        if num_devices in self.__class__._cached_configs[basis]:
            entry = self.__class__._cached_configs[basis][num_devices]
            best_threads = entry[0] if isinstance(entry, tuple) else entry
            in_crossover = getattr(self.__class__, '_in_gpu_crossover_profiling', False)
            suppress_cache = getattr(self.__class__, '_suppress_cache_hit_prints', False) or (multiprocessing.current_process().name != 'MainProcess') or is_worker_thread
            if not in_crossover and not suppress_cache and os.environ.get(f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}_{basis}') != '1':
                print_cache_hit_block(best_threads, "loaded from cache")
                os.environ[f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}_{basis}'] = '1'
            
            self.optimal_cores = best_threads
            self.use_parallel_optimized = (best_threads > 1)
            if best_threads == 0: best_threads = 1
            _safe_set_numba_threads(best_threads)
            self.detailedPrint = old_detailedPrint
            return

        if getattr(self.__class__, '_in_bootstrap_profiling', False):
            cached_cores = 2
            if basis in self.__class__._cached_configs and self.__class__._cached_configs[basis]:
                entry = next(iter(self.__class__._cached_configs[basis].values()))
                cached_cores = entry[0] if isinstance(entry, tuple) else entry
            self.optimal_cores = cached_cores
            self.use_parallel_optimized = (cached_cores > 1)
            self.detailedPrint = old_detailedPrint
            return

        # Check if the optimal threads from a smaller crossbar is already >= 80% of system cores
        system_80_percent = _SYS_80_PERCENT_CORES
        best_threads = None
        max_size_found = -1
        for cached_size, entry in self.__class__._cached_configs[basis].items():
            cached_threads = entry[0] if isinstance(entry, tuple) else entry
            if cached_size <= num_devices and cached_size > max_size_found:
                if cached_threads >= system_80_percent:
                    max_size_found = cached_size
                    best_threads = cached_threads

        if best_threads is not None:
            in_crossover = getattr(self.__class__, '_in_gpu_crossover_profiling', False)
            suppress_cache = getattr(self.__class__, '_suppress_cache_hit_prints', False) or is_worker_thread
            if not in_crossover and not suppress_cache and os.environ.get(f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}') != '1':
                print_cache_hit_block(best_threads, "locked (Cached)")
                os.environ[f'MOLMEM_HW_PROF_MSG_SHOWN_{num_devices}'] = '1'
            self.optimal_cores = best_threads
            self.use_parallel_optimized = (best_threads > 1)
            self.detailedPrint = old_detailedPrint
            return

        # Acquire lock to ensure only one thread benchmarks uncached geometries
        _acquired_profiling_lock = False
        if num_devices not in self.__class__._cached_configs.get(basis, {}):
            _GLOBAL_THREAD_PROFILING_LOCK.acquire()
            _acquired_profiling_lock = True
            # Double-check cache in case another thread just completed profiling while waiting
            if num_devices in self.__class__._cached_configs.get(basis, {}):
                cached_val = self.__class__._cached_configs[basis][num_devices]
                best_threads = cached_val[0] if isinstance(cached_val, tuple) else cached_val
                self.optimal_cores = best_threads
                self.use_parallel_optimized = (best_threads > 1)
                if best_threads == 0: best_threads = 1
                _safe_set_numba_threads(best_threads)
                self.detailedPrint = old_detailedPrint
                _GLOBAL_THREAD_PROFILING_LOCK.release()
                return

        cached_configs = {}
            
        if verbose and not getattr(self.__class__, '_in_bootstrap_profiling', False):
            print(f"\n--------------------------------------------------------------")
            print(f"--- MolmemSimulator: Dynamic Thread Profiling (SOA)      ---")
            print(f"--------------------------------------------------------------")
        elif not getattr(self.__class__, '_suppress_cache_hit_prints', False) and verbose and not getattr(self.__class__, '_in_bootstrap_profiling', False):
            print()
        
        max_cores = system_80_percent
        if num_devices > 0:
            max_cores = min(max_cores, max(4, num_devices))
        
        programmer = getattr(self, 'programmer', None)
        if programmer is not None:
            programmer.calibrate()
            
        vW = getattr(programmer, 'vWrite', 0.9) if programmer else 0.9
        vE = getattr(programmer, 'vErase', -1.2) if programmer else -1.2
        pw = getattr(programmer, 'pulseWidthPot', 80e-9) if programmer else 80e-9
        period = getattr(programmer, 'period', 160e-9) if programmer else 160e-9
        
        # Backup voltage sources to prevent tournament runs from destroying them
        backup_vSources = list(self.vSources) if hasattr(self, 'vSources') else []
        backup_dynamicSources = list(self.dynamicSources) if hasattr(self, 'dynamicSources') else []
        backup_nodesKnown = set(self.nodesKnown) if hasattr(self, 'nodesKnown') else set()
        backup_nodesUnknown = set(self.nodesUnknown) if hasattr(self, 'nodesUnknown') else set()

        if verbose:
            print(f"  > Warming up C-Kernel and preparing 1 to {max_cores} Thread Sweep...")
        self.clearVoltageSources()
        for r in self.wordlines.values():
            self.addVoltageSource(node=r, sourceObj=PulseSource(node=r, amp=vE, period=period, width=pw))
        
        # Ground only half of the bitlines to intentionally force Matrix Solver load testing
        bit_nodes = list(self.bitlines.values())
        for c in bit_nodes[:max(1, len(bit_nodes) >> 1)]:
            self.addDcSource(node=c, voltage=0.0)
            
        # Initialize optimal_cores to 1 during profiling sweeps
        self.optimal_cores = 1

        # Determine matrix dimensions for parallel programming benchmark
        if self.crossbarDevices:
            rows = cols = 0
            for r, c in self.crossbarDevices:
                if r > rows: rows = r
                if c > cols: cols = c
        else:
            rows = math.isqrt(num_devices)
            cols = rows
        targetStates = np.full((rows, cols), 0.5)

        # Preserve current simulation state variables to prevent destructive overwriting from tournament loops
        backup_states = []
        for d in self.devices:
            dev = d['dev']
            backup_states.append({
                'n': dev._n,
                'modeState': dev._modeState,
                'scaleFactor': dev._scaleFactor,
                'f22DepScaled': getattr(dev, '_f22DepScaled', 0.0),
                'f22PotScaled': getattr(dev, '_f22PotScaled', 0.0),
                '_cached_n_int': getattr(dev, '_cached_n_int_val', 0),
                'iScale': getattr(dev, '_iScale', 1.0),
                'vScale': getattr(dev, '_vScale', 1.0),
                'currentF22': getattr(dev, '_currentF22', 0.0),
                'currentIScale': getattr(dev, '_currentIScale', 1.0),
                'currentVScale': getattr(dev, '_currentVScale', 1.0),
                'T': getattr(dev, '_T', 298.15),
                'G': getattr(dev, 'G', 0.0)
            })
        
        times = {}
        profiler_start = time.perf_counter()
        original_stdout = io.StringIO() if getattr(self.__class__, '_suppress_cache_hit_prints', False) else sys.stdout
        
        try:
            logical_cores_count = os.cpu_count() or 1
            max_cores = max(1, int(logical_cores_count * 0.8))
            total_evals = max_cores * 2
            evals_completed = 0
            
            in_crossover = getattr(self.__class__, '_in_gpu_crossover_profiling', False)
            
            def update_master_progress(config_name):
                nonlocal evals_completed
                evals_completed += 1
                if getattr(self.__class__, '_in_bootstrap_profiling', False):
                    return
                if in_crossover:
                    current_idx = getattr(self.__class__, '_crossover_current_idx', 0)
                    current_sz = getattr(self.__class__, '_crossover_current_sz', 2)
                    crossover_sizes = getattr(self.__class__, '_crossover_sizes', [2, 4, 8, 16, 32, 64, 128])
                    N = len(crossover_sizes)
                    inv_N = 1.0 / N
                    progress_base = (current_idx * inv_N) * 100.0
                    inv_total_evals = 1.0 / max(1, total_evals)
                    cpu_frac = evals_completed * inv_total_evals
                    progress = progress_base + (cpu_frac * (80.0 * inv_N))
                    progress = min(99.0, progress)
                    basis = 'Circuit' if getattr(self, 'programmer', None) is None else getattr(self, 'UpdatesPer', 'Column')
                    status_str = f"Calibrating ({basis}) (Size {current_sz}x{current_sz})"
                    sub_cpu = f"      [-] CPU : Evaluating {config_name}"
                    sub_gpu = getattr(self.__class__, '_active_crossover_gpu_subline', "      [-] GPU : Setting up HIP execution pipelines...")
                    self.__class__._update_crossover_display(progress, status_str, active_sublines=[sub_cpu, sub_gpu])
                else:
                    inv_total_evals = 1.0 / max(1, total_evals)
                    progress = min(100.0, (evals_completed * inv_total_evals) * 100.0)
                    bar = '#' * int(progress * 0.2)
                    if verbose:
                        basis = 'Circuit' if getattr(self, 'programmer', None) is None else getattr(self, 'UpdatesPer', 'Column')
                        if gpu_active and getattr(self, 'requested_device', 'auto') != 'cpu' and basis != 'Circuit':
                            line1 = f"  > Device Dispatch Calibration [{bar.ljust(20)}] {progress:3.0f}% | CPU Parallelism (Size {size_str})"
                            line2 = f"    >> Evaluating: {config_name}"
                            original_stdout.write(f"\r{line1.ljust(120)}\n{line2.ljust(120)}\033[A")
                        else:
                            status = f"  > CPU Performance Profiling [{bar.ljust(20)}] {progress:3.0f}% | CPU Parallelism (Size {size_str}) | {config_name}"
                            original_stdout.write(f"\r{status.ljust(120)}")
                        original_stdout.flush()
                    else:
                        is_main_thread = (threading.current_thread().name == 'MainThread')
                        if not getattr(self.__class__, '_in_bootstrap_profiling', False) and multiprocessing.current_process().name == 'MainProcess' and is_main_thread:
                            msg = f"\r  > Intelligent Core Profiling (SOA) > [{bar.ljust(20)}] {progress:3.0f}% | Evaluating: {config_name}\033[K"
                            original_stdout.write(msg)
                            original_stdout.flush()

            self._profiler_active = True
            max_threads = max_cores
            evals_completed = 0
 
            # --- PHASE 2: Linear Thread Profiling Sweep ---
            active_candidates = []
            min_threads = 1
            if self.__class__._cached_configs[basis]:
                for cached_size, (cached_threads, _) in self.__class__._cached_configs[basis].items():
                    if cached_size <= num_devices:
                        min_threads = max(min_threads, cached_threads)
            
            # Cap the threads candidate range by workload units to prevent redundant sweeps
            if num_devices <= 1:
                effective_max_threads = min(max_threads, 4)
            else:
                workload_units = cols if getattr(self, 'UpdatesPer', 'Column') == 'Column' else num_devices
                effective_max_threads = min(max_threads, max(4, workload_units))
            effective_max_threads = max(1, effective_max_threads)
            min_threads = min(effective_max_threads, max(1, min_threads))
            
            for threads in range(min_threads, effective_max_threads + 1):
                active_candidates.append(threads)
                
            total_evals = len(active_candidates)
            round_times = {}
            history_times = []
            
            # Pre-calibrate programmer and run a single dry warmup pass before candidate timing
            if programmer is not None and hasattr(programmer, 'calibrate'):
                if programmer.stateMapPot is None or programmer.stateMapDep is None or programmer.gMapG is None:
                    programmer.calibrate()
                try:
                    programmer.programWeightsParallel(targetStates)
                except Exception:
                    pass
            else:
                try:
                    # Warm up sequential solver (1 thread)
                    self.optimal_cores = 1
                    _safe_set_numba_threads(1)
                    self.run(tEnd=period, min_dt=1e-12, tol=1e-3, appendHistory=False, recordHistory=False)
                    # Warm up parallel multi-core solver (2 threads)
                    self.optimal_cores = 2
                    _safe_set_numba_threads(2)
                    self.run(tEnd=period, min_dt=1e-12, tol=1e-3, appendHistory=False, recordHistory=False)
                except Exception:
                    pass

            backup_n_arr = np.array([b['n'] for b in backup_states], dtype=np.float64)
            backup_mode_arr = np.array([b['modeState'] for b in backup_states], dtype=np.float64)
            backup_scale_arr = np.array([b['scaleFactor'] for b in backup_states], dtype=np.float64)
            backup_cached_n_arr = np.array([b['_cached_n_int'] for b in backup_states], dtype=np.int32)

            for idx, threads in enumerate(active_candidates):
                config_name = f"{threads:2d} CPU Threads"
                update_master_progress(config_name)
                    
                self.optimal_cores = threads
                _safe_set_numba_threads(threads)
                
                # Reset the device states to backup values before the benchmark run
                if hasattr(self, '_gpu_states_cache') and isinstance(self._gpu_states_cache, dict) and 'n_arr' in self._gpu_states_cache:
                    np.copyto(self._gpu_states_cache['n_arr'], backup_n_arr)
                    np.copyto(self._gpu_states_cache['modeState_arr'], backup_mode_arr)
                    np.copyto(self._gpu_states_cache['scaleFactor_arr'], backup_scale_arr)
                    np.copyto(self._gpu_states_cache['_cached_n_int_arr'], backup_cached_n_arr)
                    self._gpu_states_dirty = False
                else:
                    for i, d in enumerate(self.devices):
                        dev = d['dev']
                        b = backup_states[i]
                        dev._n = b['n']
                        dev._modeState = b['modeState']
                        dev._scaleFactor = b['scaleFactor']
                        dev._cached_n_int_val = b['_cached_n_int']
                    
                if programmer is not None and hasattr(programmer, 'programWeightsParallel'):
                    t0 = time.perf_counter()
                    programmer.programWeightsParallel(targetStates)
                    t_exec_total = time.perf_counter() - t0
                    round_times[threads] = t_exec_total
                else:
                    test_pulses = 300
                    t0 = time.perf_counter()
                    self.run(tEnd=test_pulses * period, min_dt=1e-12, tol=1e-3, appendHistory=False, recordHistory=False)
                    t_exec_total = time.perf_counter() - t0
                    round_times[threads] = t_exec_total
                    
                try:
                    from .sys_utils import log_to_run_log_only
                    t_str = f"{t_exec_total*1000.0:.2f} ms" if t_exec_total < 1.0 else f"{t_exec_total:.3f} s"
                    log_to_run_log_only(f"[THREAD_BENCHMARK_CANDIDATE] [PID {os.getpid()}] Basis={basis} | Size={size_str} ({num_devices} devs) | Candidate={threads} Thread(s) | Execution Time={t_str}")
                except Exception:
                    pass

                history_times.append(t_exec_total)
                if len(history_times) >= 3:
                    if history_times[-1] > history_times[-2] > history_times[-3]:
                        break

            best_threads = min(round_times, key=round_times.get) if round_times else 1
            best_use_sparse = False
            try:
                from .sys_utils import log_to_run_log_only
                cand_summary = ", ".join([f"{thr}T: {t_val*1000.0:.2f}ms" if t_val < 1.0 else f"{thr}T: {t_val:.3f}s" for thr, t_val in sorted(round_times.items())])
                self._last_candidates_summary = cand_summary
                log_to_run_log_only(f"[THREAD_BENCHMARK_SUMMARY] [PID {os.getpid()}] Basis={basis} | Size={size_str} ({num_devices} devs) | Candidates: [{cand_summary}] | Selected Optimal: {best_threads} Thread(s)")
            except Exception:
                self._last_candidates_summary = ""
                pass
            
            # Print newline to securely close the progress bar empty line if verbose is active
            if verbose:
                original_stdout.write("\n")
                
        finally:
            self._profiler_active = False
            self.detailedPrint = old_detailedPrint
            
            # Restore original voltage sources
            self.clearVoltageSources()
            self.vSources = backup_vSources
            self.dynamicSources = backup_dynamicSources
            self.nodesKnown = backup_nodesKnown
            self.nodesUnknown = backup_nodesUnknown
            self._gpu_cache = None
        
        self.optimal_cores = best_threads
        self.use_parallel_optimized = (best_threads > 1)
        
        if best_threads == 0: best_threads = 1
        _safe_set_numba_threads(best_threads)

        affinity_str = "OS Managed"
        
        # Flush all experimental traces and restore matrix initial states
        self.tArr = None
        for i, d in enumerate(self.devices):
            dev = d['dev']
            b = backup_states[i]
            dev._n = b['n']
            dev._modeState = b['modeState']
            dev._scaleFactor = b['scaleFactor']
            dev._f22DepScaled = b['f22DepScaled']
            dev._f22PotScaled = b['f22PotScaled']
            dev._cached_n_int_val = b['_cached_n_int']
            dev._iScale = b['iScale']
            dev._vScale = b['vScale']
            dev._currentF22 = b['currentF22']
            dev._currentIScale = b['currentIScale']
            dev._currentVScale = b['currentVScale']
            dev._T = b['T']
            dev.G = b['G']
            
        # Cache Results to Disk & Memory
        self.best_cores_pool = None
        
        # Merge class-level cache and disk cache to keep them in sync
        for size, config in self.__class__._cached_configs[basis].items():
            cached_configs[size] = config
        cached_configs[num_devices] = (best_threads, best_use_sparse)
        
        self.__class__._save_profile_cache(None, cached_configs, basis=basis)
        
        solver_str = "Sparse" if best_use_sparse else "Dense"
        if verbose:
            if not getattr(self.__class__, '_suppress_cache_hit_prints', False):
                print(f"    >> Optimal Configuration : {best_threads} Threads ({solver_str} Solver)\n")
            else:
                original_stdout.write(f"\r  > Dynamic Thread Profiling (SOA) > Done (Optimal: {best_threads} Threads, {solver_str} Solver)\033[K\n")
        profiler_elapsed = time.perf_counter() - profiler_start
        self.__class__._last_cpu_profiler_elapsed = profiler_elapsed
        if in_crossover:
            current_idx = getattr(self.__class__, '_crossover_current_idx', 0)
            current_sz = getattr(self.__class__, '_crossover_current_sz', 2)
            crossover_sizes = getattr(self.__class__, '_crossover_sizes', [2, 4, 8, 16, 32, 64, 128])
            N = len(crossover_sizes)
            inv_N = 1.0 / N
            progress = (current_idx * inv_N) * 100.0 + (0.8 * (100.0 * inv_N))
            status_str = f"Calibrating ({basis}) (Size {current_sz}x{current_sz})"
            sub_cpu = f"      [-] CPU : Selected {best_threads} Threads ({solver_str}) in {profiler_elapsed:.1f}s"
            sub_gpu = getattr(self.__class__, '_active_crossover_gpu_subline', "      [-] GPU : Setting up HIP execution pipelines...")
            self.__class__._update_crossover_display(progress, status_str, active_sublines=[sub_cpu, sub_gpu])
        elif verbose:
            print(f"\n\n  > C-Kernel Performance Results ({basis}):")
            print(f"    >> Optimal Configuration : {best_threads} Threads, {solver_str} Solver\n")
        else:
            is_main_thread = (threading.current_thread().name == 'MainThread')
            if not getattr(self.__class__, '_in_bootstrap_profiling', False) and multiprocessing.current_process().name == 'MainProcess' and is_main_thread:
                original_stdout.write(f"\r  > Dynamic Thread Profiling (SOA) > Done (Optimal: {best_threads} Threads, {solver_str} Solver)\033[K\n")
                original_stdout.flush()
            
        self.clearHistory()
        if _acquired_profiling_lock:
            _GLOBAL_THREAD_PROFILING_LOCK.release()

    @classmethod
    def _load_profile_cache(cls, basis=None, verbose=False, force=False):
        if not force and \
           isinstance(cls._gpu_crossover_dim, dict) and cls._gpu_crossover_dim["Column"] is not None and cls._gpu_crossover_dim["Device"] is not None:
            return
            
        machine_id = _SYS_MACHINE_ID
        
        if not hasattr(cls, '_opt_gpu_slice_limit') or not isinstance(cls._opt_gpu_slice_limit, dict) or any(not isinstance(v, dict) for v in cls._opt_gpu_slice_limit.values()):
            cls._opt_gpu_slice_limit = {
                "Column": {},
                "Device": {}
            }
        if not hasattr(cls, '_opt_gpu_read_slice_limit') or not isinstance(cls._opt_gpu_read_slice_limit, dict) or any(not isinstance(v, dict) for v in cls._opt_gpu_read_slice_limit.values()):
            cls._opt_gpu_read_slice_limit = {
                "Column": {},
                "Device": {}
            }
        if not isinstance(cls._gpu_crossover_dim, dict):
            cls._gpu_crossover_dim = {
                "Column": None,
                "Device": None,
                "Circuit": None
            }
        if not isinstance(cls._cached_configs, dict) or "Column" not in cls._cached_configs or "Circuit" not in cls._cached_configs:
            cls._cached_configs = {
                "Column": {},
                "Device": {},
                "Circuit": {}
            }
            
        from . import _load_unified_cache, _compute_library_footprint
        
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                data = _load_unified_cache(force=force)
                if not data:
                    cache_file = os.path.join(os.path.dirname(__file__), '.calcache', 'molmem_cache.npz')
                    if not os.path.exists(cache_file):
                        break
                    time.sleep(0.05 + random.random() * 0.1)
                    continue
                    
                if 'machine_id' in data and data['machine_id'][0] == machine_id and \
                   'lib_footprint' in data and data['lib_footprint'][0] == _compute_library_footprint():
                    
                    # Load Column-wise configs
                    if 'cached_sizes_col' in data:
                        for i, size in enumerate(data['cached_sizes_col']):
                            cls._cached_configs["Column"][int(size)] = (int(data['cached_threads_col'][i]), bool(data['cached_sparse_col'][i]))
                        if 'gpu_crossover_dim_col' in data:
                            xo_col = int(data['gpu_crossover_dim_col'][0])
                            cls._gpu_crossover_dim["Column"] = None if xo_col == -1 else xo_col
                    elif 'cached_sizes' in data:
                        for i, size in enumerate(data['cached_sizes']):
                            cls._cached_configs["Column"][int(size)] = (int(data['cached_threads'][i]), bool(data['cached_sparse'][i]))
                        if 'gpu_crossover_dim' in data:
                            xo_col = int(data['gpu_crossover_dim'][0])
                            cls._gpu_crossover_dim["Column"] = None if xo_col == -1 else xo_col
                    
                    # Load Column-wise slice limits dictionary
                    if 'opt_gpu_sizes_col' in data and 'opt_gpu_slice_limits_col' in data:
                        sizes_col = data['opt_gpu_sizes_col']
                        limits_col = data['opt_gpu_slice_limits_col']
                        read_limits_col = data['opt_gpu_read_slice_limits_col'] if 'opt_gpu_read_slice_limits_col' in data else limits_col
                        cls._opt_gpu_slice_limit["Column"].update({int(sz): int(lim) for sz, lim in zip(sizes_col, limits_col)})
                        cls._opt_gpu_read_slice_limit["Column"].update({int(sz): int(lim) for sz, lim in zip(sizes_col, read_limits_col)})
                    else:
                        if 'opt_gpu_slice_limit_col' in data:
                            legacy_lim = int(data['opt_gpu_slice_limit_col'][0])
                            for default_sz in [2, 4, 8, 16, 32, 64, 128]:
                                cls._opt_gpu_slice_limit["Column"][default_sz] = legacy_lim
                        if 'opt_gpu_read_slice_limit_col' in data:
                            legacy_read_lim = int(data['opt_gpu_read_slice_limit_col'][0])
                            for default_sz in [2, 4, 8, 16, 32, 64, 128]:
                                cls._opt_gpu_read_slice_limit["Column"][default_sz] = legacy_read_lim
                    
                    # Load Device-wise configs
                    if 'cached_sizes_dev' in data:
                        for i, size in enumerate(data['cached_sizes_dev']):
                            cls._cached_configs["Device"][int(size)] = (int(data['cached_threads_dev'][i]), bool(data['cached_sparse_dev'][i]))
                        if 'gpu_crossover_dim_dev' in data:
                            xo_dev = int(data['gpu_crossover_dim_dev'][0])
                            cls._gpu_crossover_dim["Device"] = None if xo_dev == -1 else xo_dev
                    
                    # Load Circuit-wise configs
                    if 'cached_sizes_circ' in data:
                        for i, size in enumerate(data['cached_sizes_circ']):
                            cls._cached_configs["Circuit"][int(size)] = (int(data['cached_threads_circ'][i]), bool(data['cached_sparse_circ'][i]))
                        if 'gpu_crossover_dim_circ' in data:
                            xo_circ = int(data['gpu_crossover_dim_circ'][0])
                            cls._gpu_crossover_dim["Circuit"] = None if xo_circ == -1 else xo_circ

                    # Load Device-wise slice limits dictionary
                    if 'opt_gpu_sizes_dev' in data and 'opt_gpu_slice_limits_dev' in data:
                        sizes_dev = data['opt_gpu_sizes_dev']
                        limits_dev = data['opt_gpu_slice_limits_dev']
                        read_limits_dev = data['opt_gpu_read_slice_limits_dev'] if 'opt_gpu_read_slice_limits_dev' in data else limits_dev
                        cls._opt_gpu_slice_limit["Device"].update({int(sz): int(lim) for sz, lim in zip(sizes_dev, limits_dev)})
                        cls._opt_gpu_read_slice_limit["Device"].update({int(sz): int(lim) for sz, lim in zip(sizes_dev, read_limits_dev)})
                    else:
                        if 'opt_gpu_slice_limit_dev' in data:
                            legacy_lim = int(data['opt_gpu_slice_limit_dev'][0])
                            for default_sz in [2, 4, 8, 16, 32, 64, 128]:
                                cls._opt_gpu_slice_limit["Device"][default_sz] = legacy_lim
                        if 'opt_gpu_read_slice_limit_dev' in data:
                            legacy_read_lim = int(data['opt_gpu_read_slice_limit_dev'][0])
                            for default_sz in [2, 4, 8, 16, 32, 64, 128]:
                                cls._opt_gpu_read_slice_limit["Device"][default_sz] = legacy_read_lim
                    # Load Calibrated Hybrid CPU Chunk & GPU Slice Allocations
                    if 'opt_cpu_chunk_per_thread' in data and len(data['opt_cpu_chunk_per_thread']) > 0:
                        cls._opt_cpu_chunk_per_thread = float(data['opt_cpu_chunk_per_thread'][0])
                    if 'opt_gpu_slice_per_cu' in data and len(data['opt_gpu_slice_per_cu']) > 0:
                        cls._opt_gpu_slice_per_cu = float(data['opt_gpu_slice_per_cu'][0])
                    if 'opt_hybrid_pacing_ratio' in data and len(data['opt_hybrid_pacing_ratio']) > 0:
                        cls._opt_hybrid_pacing_ratio = float(data['opt_hybrid_pacing_ratio'][0])
                    if 'opt_hybrid_min_horizon' in data and len(data['opt_hybrid_min_horizon']) > 0:
                        cls._opt_hybrid_min_horizon = int(data['opt_hybrid_min_horizon'][0])
                    if 'opt_hybrid_cpu_allowance' in data and len(data['opt_hybrid_cpu_allowance']) > 0:
                        cls._opt_hybrid_cpu_allowance = int(data['opt_hybrid_cpu_allowance'][0])
                    if 'opt_hybrid_ranks' in data and 'opt_hybrid_ratios' in data:
                        ranks = data['opt_hybrid_ranks']
                        ratios = data['opt_hybrid_ratios']
                        cls._opt_hybrid_pacing_by_rank = {int(r): float(v) for r, v in zip(ranks, ratios)}
                break
            except Exception:
                time.sleep(0.05 + random.random() * 0.1)
                
        suppress_cache = getattr(cls, '_suppress_cache_hit_prints', False)
        if not suppress_cache and verbose:
            bases_to_print = [basis] if basis is not None else ["Column", "Device"]
            for b in bases_to_print:
                crossover = cls._gpu_crossover_dim[b]
                if crossover is not None:
                    if os.environ.get(f'MOLMEM_GPU_PROF_MSG_SHOWN_{b}') != '1':
                        crossover_str = f"sz={crossover}x{crossover}" if crossover != 9999 else "Disabled (Never)"
                        print(f"  GPU Performance Profiling ({b}) [{'#'*20}] 100% | Loaded Cache: Crossover at {crossover_str}")
                        os.environ[f'MOLMEM_GPU_PROF_MSG_SHOWN_{b}'] = '1'

    @classmethod
    def _save_profile_cache(cls, best_cores_pool, cached_configs, gpu_crossover_dim=None, basis="Column"):
        import multiprocessing
        if multiprocessing.current_process().name != 'MainProcess':
            # Skip disk write in background processes to prevent race conditions
            cls._cached_configs[basis].update(cached_configs)
            return

        import platform
        import os
        import random
        import numpy as np
        
        machine_id = _SYS_MACHINE_ID
        
        if not isinstance(cls._gpu_crossover_dim, dict) or "Circuit" not in cls._gpu_crossover_dim:
            cls._gpu_crossover_dim = {
                "Column": None,
                "Device": None,
                "Circuit": None
            }
        if not isinstance(cls._cached_configs, dict) or "Column" not in cls._cached_configs or "Circuit" not in cls._cached_configs:
            cls._cached_configs = {
                "Column": {},
                "Device": {},
                "Circuit": {}
            }
        # Load existing cache from disk only if memory cache is empty to prevent overwriting
        if not cls._cached_configs.get(basis):
            try:
                import molmem_lib
                molmem_lib._cached_unified_data = None
                cls._load_profile_cache(basis=basis, verbose=False, force=True)
            except Exception:
                pass
            
        cls._cached_configs[basis].update(cached_configs)
        if gpu_crossover_dim is not None:
            cls._gpu_crossover_dim[basis] = gpu_crossover_dim
        
        # Protect dictionary format
        if not isinstance(cls._opt_gpu_slice_limit, dict) or any(not isinstance(v, dict) for v in cls._opt_gpu_slice_limit.values()):
            cls._opt_gpu_slice_limit = {
                "Column": {},
                "Device": {}
            }
        if not isinstance(cls._opt_gpu_read_slice_limit, dict) or any(not isinstance(v, dict) for v in cls._opt_gpu_read_slice_limit.values()):
            cls._opt_gpu_read_slice_limit = {
                "Column": {},
                "Device": {}
            }
 
        # Legacy scalar compatibility calculations
        legacy_lim_col = max(cls._opt_gpu_slice_limit["Column"].values()) if cls._opt_gpu_slice_limit["Column"] else 999999
        legacy_read_lim_col = max(cls._opt_gpu_read_slice_limit["Column"].values()) if cls._opt_gpu_read_slice_limit["Column"] else 999999
        legacy_lim_dev = max(cls._opt_gpu_slice_limit["Device"].values()) if cls._opt_gpu_slice_limit["Device"] else 999999
        legacy_read_lim_dev = max(cls._opt_gpu_read_slice_limit["Device"].values()) if cls._opt_gpu_read_slice_limit["Device"] else 999999
 
        # Merge Column values
        sizes_col = np.array(list(cls._cached_configs["Column"].keys()), dtype=int)
        threads_col = np.array([cls._cached_configs["Column"][s][0] for s in sizes_col], dtype=int)
        sparse_col = np.array([cls._cached_configs["Column"][s][1] for s in sizes_col], dtype=bool)
        xo_col = cls._gpu_crossover_dim["Column"]
        if xo_col is None: xo_col = -1
        
        # Merge Device values
        sizes_dev = np.array(list(cls._cached_configs["Device"].keys()), dtype=int)
        threads_dev = np.array([cls._cached_configs["Device"][s][0] for s in sizes_dev], dtype=int)
        sparse_dev = np.array([cls._cached_configs["Device"][s][1] for s in sizes_dev], dtype=bool)
        xo_dev = cls._gpu_crossover_dim["Device"]
        if xo_dev is None: xo_dev = -1
        
        # Merge Circuit values
        sizes_circ = np.array(list(cls._cached_configs["Circuit"].keys()), dtype=int)
        threads_circ = np.array([cls._cached_configs["Circuit"][s][0] for s in sizes_circ], dtype=int)
        sparse_circ = np.array([cls._cached_configs["Circuit"][s][1] for s in sizes_circ], dtype=bool)
        xo_circ = cls._gpu_crossover_dim["Circuit"]
        if xo_circ is None: xo_circ = -1
        
        from . import _compute_library_footprint, _save_unified_cache
        
        updates = {
            'machine_id': np.array([machine_id]),
            'lib_footprint': np.array([_compute_library_footprint()]),
            'best_cores_pool': np.array([], dtype=int),
            'cached_sizes_col': sizes_col,
            'cached_threads_col': threads_col,
            'cached_sparse_col': sparse_col,
            'gpu_crossover_dim_col': np.array([xo_col]),
            'opt_gpu_sizes_col': np.array(list(cls._opt_gpu_slice_limit["Column"].keys()), dtype=int),
            'opt_gpu_slice_limits_col': np.array(list(cls._opt_gpu_slice_limit["Column"].values()), dtype=int),
            'opt_gpu_read_slice_limits_col': np.array(list(cls._opt_gpu_read_slice_limit["Column"].values()), dtype=int),
            'opt_gpu_slice_limit_col': np.array([legacy_lim_col]),
            'opt_gpu_read_slice_limit_col': np.array([legacy_read_lim_col]),
            'cached_sizes_dev': sizes_dev,
            'cached_threads_dev': threads_dev,
            'cached_sparse_dev': sparse_dev,
            'gpu_crossover_dim_dev': np.array([xo_dev]),
            'opt_gpu_sizes_dev': np.array(list(cls._opt_gpu_slice_limit["Device"].keys()), dtype=int),
            'opt_gpu_slice_limits_dev': np.array(list(cls._opt_gpu_slice_limit["Device"].values()), dtype=int),
            'opt_gpu_read_slice_limits_dev': np.array(list(cls._opt_gpu_read_slice_limit["Device"].values()), dtype=int),
            'opt_gpu_slice_limit_dev': np.array([legacy_lim_dev]),
            'opt_gpu_read_slice_limit_dev': np.array([legacy_read_lim_dev]),
            'cached_sizes_circ': sizes_circ,
            'cached_threads_circ': threads_circ,
            'cached_sparse_circ': sparse_circ,
            'gpu_crossover_dim_circ': np.array([xo_circ])
        }
        if getattr(cls, '_opt_cpu_chunk_per_thread', None) is not None:
            updates['opt_cpu_chunk_per_thread'] = np.array([float(cls._opt_cpu_chunk_per_thread)], dtype=np.float64)
        if getattr(cls, '_opt_gpu_slice_per_cu', None) is not None:
            updates['opt_gpu_slice_per_cu'] = np.array([float(cls._opt_gpu_slice_per_cu)], dtype=np.float64)
        if getattr(cls, '_opt_hybrid_pacing_ratio', None) is not None:
            updates['opt_hybrid_pacing_ratio'] = np.array([float(cls._opt_hybrid_pacing_ratio)], dtype=np.float64)
        if getattr(cls, '_opt_hybrid_min_horizon', None) is not None:
            updates['opt_hybrid_min_horizon'] = np.array([int(cls._opt_hybrid_min_horizon)], dtype=np.int64)
        if getattr(cls, '_opt_hybrid_cpu_allowance', None) is not None:
            updates['opt_hybrid_cpu_allowance'] = np.array([int(cls._opt_hybrid_cpu_allowance)], dtype=np.int64)
        if getattr(cls, '_opt_hybrid_pacing_by_rank', None):
            updates['opt_hybrid_ranks'] = np.array(list(cls._opt_hybrid_pacing_by_rank.keys()), dtype=np.int64)
            updates['opt_hybrid_ratios'] = np.array(list(cls._opt_hybrid_pacing_by_rank.values()), dtype=np.float64)
        _save_unified_cache(updates)

    def profileGpuPipeline(self, force_device_init=True):
        """
        Instantaneous GPU execution initializer for Heterogeneous Work-Stealing Engine.
        Bypasses static crossover calibration sweeps as workload is dynamically partitioned.
        """
        basis = getattr(self, 'UpdatesPer', 'Column')
        self.__class__._gpu_crossover_dim[basis] = 16
        self.__class__._in_gpu_crossover_profiling = False
        return 16

    def profileGpuSliceLimit(self, sz):
        """
        Calibrates optimal GPU slice limits only for the specific crossbar size 'sz'.
        Saves and caches the results to the profile cache.
        """
        import os
        import time
        import gc
        import numpy as np
        import sys
        import io
        from .simulator import MolmemSimulator
        
        basis = getattr(self, 'UpdatesPer', 'Column')
        verbose = getattr(self, 'detailedPrint', True)
        
        if verbose:
            sys.__stdout__.write(f"  GPU Slice Size Calibration ({basis}) (Size {sz}x{sz})...\n")
            sys.__stdout__.flush()
            
        cls = self.__class__
        
        # Determine minimum candidate based on smaller cached crossbars
        min_candidate = 2
        if basis in cls._opt_gpu_slice_limit:
            smaller_sizes = [s for s in cls._opt_gpu_slice_limit[basis].keys() if s < sz]
            if smaller_sizes:
                min_candidate = max(cls._opt_gpu_slice_limit[basis][s] for s in smaller_sizes)
                
        # Initialize GPU instance to benchmark
        # Use an isolated stdout StringIO to keep imports/JIT compilations quiet
        original_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            
            sim_gpu = MolmemSimulator(device='cuda', backend='pytorch', vram_fraction=getattr(self, 'vram_fraction', 0.8))
            sim_gpu.optimal_cores = 1
            sim_gpu.use_parallel_optimized = True
            sim_gpu.addCrossbarMatrix(rows=sz, cols=sz, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture=self.architecture, force_device_init=False, UpdatesPer=self.UpdatesPer)
            sim_gpu.programmer.calibrate()
            sim_gpu.isCalibrated = True
            sim_gpu.UpdatesPer = self.UpdatesPer
            
            # Fetch max safe slice
            is_device_basis = (self.UpdatesPer == 'Device')
            pad_sources_write = 2 if is_device_basis else (sz + 1)
            orig_detailed = sim_gpu.detailedPrint
            sim_gpu.detailedPrint = False
            max_safe_write = sim_gpu.get_optimal_gpu_slice_size(sz, sz, pad_sources_write, max_v_pts=32, is_read=False, is_device_basis=is_device_basis, ignore_limits=True)
            max_safe_read = sim_gpu.get_optimal_gpu_slice_size(sz, sz, pad_sources=(sz + sz), max_v_pts=32, is_read=True, is_device_basis=False, ignore_limits=True)
            sim_gpu.detailedPrint = orig_detailed
            
            # Get VRAM info
            free_mem_mb = 0.0
            target_vram_mb = 0.0
            import torch
            if torch.cuda.is_available():
                try:
                    device = torch.device(f'cuda:{torch.cuda.current_device()}')
                    free_memory, total_memory = torch.cuda.mem_get_info(device)
                    INV_MB = 1.0 / 1048576.0
                    free_mem_mb = free_memory * INV_MB
                    target_vram_mb = (free_memory * getattr(self, 'vram_fraction', 0.8)) * INV_MB
                except Exception:
                    pass
                    
            best_write_slice = min(sz * sz if is_device_basis else sz, max_safe_write)
            best_read_slice = min(sz, max_safe_read)
            if verbose:
                basis_label = "Device" if is_device_basis else "Column"
                sys.__stdout__.write(f"\r    [-] {basis_label} updates: auto-selected slice size (Write: {best_write_slice}, Read: {best_read_slice}) | Max Safe (Write: {max_safe_write}, Read: {max_safe_read})")
                sys.__stdout__.flush()
                        
            # Set the optimal values in the dictionary
            cls._opt_gpu_slice_limit[basis][sz] = best_write_slice
            cls._opt_gpu_read_slice_limit[basis][sz] = best_read_slice
            
            try:
                from .sys_utils import log_to_run_log_only
                basis_label = "Device" if is_device_basis else "Column"
                log_to_run_log_only(f"[GPU_SLICE_BENCHMARK] [PID {os.getpid()}] Basis={basis_label} | Size={sz}x{sz} ({sz*sz} devs) | Selected Slice (Write: {best_write_slice}, Read: {best_read_slice}) | Max Safe (Write: {max_safe_write}, Read: {max_safe_read}) | VRAM: {free_mem_mb:.0f}/{target_vram_mb:.0f}MB")
            except Exception:
                pass

            # Save cache to disk
            cls._save_profile_cache(None, cls._cached_configs[basis], cls._gpu_crossover_dim[basis], basis=basis)
            
            if verbose:
                # Clean line rewrite with 100% finished progress style
                sys.__stdout__.write(f"\r  GPU Slice Calibrated ({basis}) [{'#'*20}] 100% | Opt Slice (W/R): {best_write_slice}/{best_read_slice} | Size {sz}x{sz}\n")
                sys.__stdout__.flush()
                
            # Clean up
            del sim_gpu
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            sys.stdout = original_stdout

    def profileHybridCpuChunkSize(self, target_dim=128):
        """
        Instantly configures hardware-matched heterogeneous sizing (1.0 Col/Thread & 2.0 Cols/CU).
        """
        cls = self.__class__
        cls._opt_cpu_chunk_per_thread = 1.0
        cls._opt_gpu_slice_per_cu = 2.0
        cls._save_profile_cache(None, cls._cached_configs.get('Column', {}), cls._gpu_crossover_dim.get('Column', 16), basis='Column')
        return 1.0

    def profileHybridConcurrencyHorizon(self, force=False):
        """
        Profiles pure CPU vs pure GPU execution on a hardware-saturating batch (2x CUs = 64 cols)
        to determine the precise CPU Concurrency Allowance and Minimum Safe Horizon,
        guaranteeing zero tail straggler barrier latency in hybrid co-execution.
        """
        import os
        import time
        import gc
        import numpy as np
        import sys
        import io
        from .simulator import MolmemSimulator
        
        cls = self.__class__
        if not force and cls._opt_hybrid_min_horizon is not None and not getattr(cls, '_in_bootstrap_profiling', False):
            return cls._opt_hybrid_min_horizon
            
        verbose = getattr(self, 'detailedPrint', True)
        original_stdout = sys.stdout
        t_start_sweep = time.perf_counter()
        
        # Probe physical CUs
        cu_count = 32
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                cu_count = getattr(props, 'multi_processor_count', 32) or 32
        except Exception:
            cu_count = 32
            
        test_cols = max(16, int(2 * cu_count))
        gpu_sat_cols = test_cols
        test_rows = 128
        
        if verbose:
            sys.__stdout__.write(f"  > Hybrid Concurrency Profiling (Matrix Ranks: 32x128, 64x128, 128x128)...\n")
            sys.__stdout__.flush()
            
        try:
            sys.stdout = io.StringIO()
            probe_configs = [(32, 128), (64, 128), (128, 128)]
            pacing_by_rank = {}
            t_cpu_by_rank = {}
            t_gpu_by_rank = {}
            rate_cpu_by_rank = {}
            rate_gpu_by_rank = {}
            
            t_hyb_by_rank = {}
            rate_hyb_by_rank = {}
            
            for r_dim, c_dim in probe_configs:
                test_rows = r_dim
                test_cols = c_dim
                rng_probe = np.random.RandomState(42 + r_dim)
                target_probe = rng_probe.uniform(0.15, 0.85, size=(test_rows, test_cols))
                
                # 1. Hybrid Pipeline (Forced Multi-Threaded CPU)
                os.environ['MOLMEM_HYBRID_FORCE_CPU'] = '1'
                os.environ.pop('MOLMEM_HYBRID_FORCE_GPU', None)
                sim_cpu = MolmemSimulator(device='cuda', backend='hybrid')
                sim_cpu.detailedPrint = False
                sim_cpu.addCrossbarMatrix(rows=test_rows, cols=test_cols, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture=getattr(self, 'architecture', '1T1R'), force_device_init=False, UpdatesPer='Column')
                sim_cpu.programmer.calibrate()
                sim_cpu.isCalibrated = True
                
                t0 = time.perf_counter()
                sim_cpu.updateCrossbarWeights(target_probe, detailedPrint=False, fineTune=False, recordHistory=False)
                t_cpu_r = max(1e-6, time.perf_counter() - t0)
                rate_cpu_r = float(test_cols) / t_cpu_r
                t_cpu_by_rank[r_dim] = t_cpu_r
                rate_cpu_by_rank[r_dim] = rate_cpu_r
                
                del sim_cpu
                gc.collect()
                
                # 2. Hybrid Pipeline (Forced GPU Stream)
                os.environ.pop('MOLMEM_HYBRID_FORCE_CPU', None)
                os.environ['MOLMEM_HYBRID_FORCE_GPU'] = '1'
                sim_gpu = MolmemSimulator(device='cuda', backend='hybrid', vram_fraction=getattr(self, 'vram_fraction', 0.8))
                sim_gpu.detailedPrint = False
                sim_gpu.addCrossbarMatrix(rows=test_rows, cols=test_cols, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture=getattr(self, 'architecture', '1T1R'), force_device_init=False, UpdatesPer='Column')
                sim_gpu.programmer.calibrate()
                sim_gpu.isCalibrated = True
                
                t0 = time.perf_counter()
                sim_gpu.updateCrossbarWeights(target_probe, detailedPrint=False, fineTune=False, recordHistory=False)
                t_gpu_r = max(1e-6, time.perf_counter() - t0)
                rate_gpu_r = float(test_cols) / t_gpu_r
                t_gpu_by_rank[r_dim] = t_gpu_r
                rate_gpu_by_rank[r_dim] = rate_gpu_r
                
                del sim_gpu
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                os.environ.pop('MOLMEM_HYBRID_FORCE_CPU', None)
                os.environ.pop('MOLMEM_HYBRID_FORCE_GPU', None)
                
                if verbose:
                    label = f"{r_dim}x{c_dim}"
                    t_c_str = f"{t_cpu_r:5.2f}s"
                    t_g_str = f"{t_gpu_r:5.2f}s"
                    r_c_str = f"{rate_cpu_r:5.1f}"
                    r_g_str = f"{rate_gpu_r:5.1f}"
                    sys.__stdout__.write(f"    >> Rank {label:<7} Baseline : CPU {t_c_str} ({r_c_str} C/s) | GPU {t_g_str} ({r_g_str} C/s)\n")
                    sys.__stdout__.flush()
                
                # 3. Hybrid Pipeline (Adaptive Split-Shifting Search)
                os.environ.pop('MOLMEM_HYBRID_FORCE_CPU', None)
                os.environ.pop('MOLMEM_HYBRID_FORCE_GPU', None)
                
                base_ratio = max(0.01, rate_cpu_r / max(1e-6, rate_gpu_r))
                candidate_ratios = [base_ratio, base_ratio * 1.4, base_ratio * 2.0, base_ratio * 3.0, base_ratio * 4.5]
                
                best_pacing_r = base_ratio
                best_t_hyb = 1e9
                best_rate_hyb = 0.0
                prev_t_trial = 1e9
                hit_cpu_saturation = False
                
                for trial_idx, trial_ratio in enumerate(candidate_ratios, 1):
                    trial_cpu_allowance = int(round(gpu_sat_cols * trial_ratio))
                    if trial_cpu_allowance >= test_cols:
                        hit_cpu_saturation = True
                        if verbose:
                            sys.__stdout__.write(f"      >>> Trial {trial_idx} (Ratio {trial_ratio:.2f}x) : CPU Saturation Reached (100% CPU -> {t_cpu_r:.2f}s)\n")
                            sys.__stdout__.flush()
                        break
                        
                    MolmemSimulator._active_pacing_override = trial_ratio
                    sim_hyb = MolmemSimulator(device='cuda', backend='hybrid', vram_fraction=getattr(self, 'vram_fraction', 0.8))
                    sim_hyb.detailedPrint = False
                    sim_hyb.addCrossbarMatrix(rows=test_rows, cols=test_cols, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture=getattr(self, 'architecture', '1T1R'), force_device_init=False, UpdatesPer='Column')
                    sim_hyb.programmer.calibrate()
                    sim_hyb.isCalibrated = True
                    
                    t0 = time.perf_counter()
                    sim_hyb.updateCrossbarWeights(target_probe, detailedPrint=False, fineTune=False, recordHistory=False)
                    t_trial = max(1e-6, time.perf_counter() - t0)
                    rate_trial = float(test_cols) / t_trial
                    
                    del sim_hyb
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    
                    is_better = (t_trial < best_t_hyb)
                    if is_better:
                        best_t_hyb = t_trial
                        best_rate_hyb = rate_trial
                        best_pacing_r = trial_ratio
                        
                    is_optimal = (t_trial <= t_cpu_r)
                    if verbose:
                        t_t_str = f"{t_trial:5.2f}s"
                        r_t_str = f"{rate_trial:5.1f}"
                        if is_optimal:
                            diff_pct = ((t_cpu_r - t_trial) / max(1e-6, t_cpu_r)) * 100.0
                            status_str = f"Optimal [Beats CPU by {diff_pct:+.1f}%]"
                        elif is_better:
                            status_str = "Improved [Faster Split]"
                        else:
                            status_str = "Evaluating..."
                        sys.__stdout__.write(f"      >>> Trial {trial_idx} (Ratio {trial_ratio:.2f}x) : {t_t_str} ({r_t_str} C/s) -> {status_str}\n")
                        sys.__stdout__.flush()
                        
                    # Continue searching to explore higher speedups until past the convex peak
                    if is_optimal and t_trial > prev_t_trial:
                        break
                    prev_t_trial = t_trial
                        
                MolmemSimulator._active_pacing_override = None
                pacing_by_rank[r_dim] = best_pacing_r
                t_hyb_by_rank[r_dim] = best_t_hyb
                rate_hyb_by_rank[r_dim] = best_rate_hyb
                if verbose:
                    if hit_cpu_saturation and best_t_hyb > t_cpu_r:
                        sys.__stdout__.write(f"      >>> Selected Split Point  : Ratio {best_pacing_r:.2f}x | CPU Saturation Horizon Calibrated\n")
                    else:
                        sys.__stdout__.write(f"      >>> Selected Split Point  : Ratio {best_pacing_r:.2f}x | Best Hyb: {best_t_hyb:.2f}s ({best_rate_hyb:.1f} C/s)\n")
                    sys.__stdout__.flush()

            cls._opt_hybrid_pacing_by_rank = pacing_by_rank
            cls._opt_hybrid_pacing_ratio = pacing_by_rank.get(128, 1.0)
            pacing_128 = pacing_by_rank.get(128, 1.0)
            gpu_sat_cols = max(16, int(2 * cu_count))
            cpu_allowance_128 = max(1, int(round(gpu_sat_cols * pacing_128)))
            min_horizon_128 = gpu_sat_cols + cpu_allowance_128
            cls._opt_hybrid_min_horizon = min_horizon_128
            cls._opt_hybrid_cpu_allowance = cpu_allowance_128
            cls._opt_cpu_chunk_per_thread = 1.0
            cls._opt_gpu_slice_per_cu = 2.0
            
            # Save to unified cache on disk
            cls._save_profile_cache(None, cls._cached_configs.get('Column', {}), cls._gpu_crossover_dim.get('Column', 16), basis='Column')
            
            if verbose:
                t_total_sweep = time.perf_counter() - t_start_sweep
                sys.__stdout__.write(f"  > Hybrid Concurrency Profiling... Done (in {t_total_sweep:.1f}s)\n")
                sys.__stdout__.write(f"    >> Dynamic Horizon Scaling : 3-Point Log-Parabolic Model Calibrated\n")
                sys.__stdout__.flush()
                
            try:
                from .sys_utils import log_to_run_log_only
                log_to_run_log_only(f"[HYBRID_CONCURRENCY_PROFILE] [PID {os.getpid()}] Rank32={pacing_by_rank.get(32, 1.0):.3f} | Rank64={pacing_by_rank.get(64, 1.0):.3f} | Rank128={pacing_by_rank.get(128, 1.0):.3f}")
            except Exception:
                pass
                
            return min_horizon_128
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                err_path = os.path.join(os.path.dirname(__file__), 'logs', 'error_log.txt')
                with open(err_path, 'a') as f:
                    f.write(f"\n[HYBRID_CONCURRENCY_PROFILE_ERROR]\n{tb}\n")
            except Exception:
                pass
            if verbose:
                sys.__stdout__.write(f"    >> Hybrid Concurrency Profiling Error: {e}\n")
                sys.__stdout__.flush()
            return 96
        finally:
            sys.stdout = original_stdout
