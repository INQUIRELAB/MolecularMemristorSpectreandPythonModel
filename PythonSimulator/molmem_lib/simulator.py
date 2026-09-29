
import multiprocessing
import os
import weakref
import sys
import ctypes
import math
import gc
# Maintain relaxed GC thresholds globally across simulations
gc.set_threshold(100000, 50, 50)

_GLOBAL_SCHEDULE_CACHE = {}
_GLOBAL_SOURCE_METADATA_CACHE = {}
_GLOBAL_NODE_MAP_CACHE = {}

from .sys_utils import (
    SimulatorSystemMixin, handle_oom, DummyStream,
    is_worker_process, is_interactive_notebook
)
from .profiler import SimulatorProfilerMixin
from .history import SimulatorHistoryMixin, LazyHistoryDict, LazyDeviceHistoryDict, ThreadPoolPingPongManager

is_mp_child = is_worker_process()

try:
    if os.environ.get("MOLMEM_CPU_ONLY") != "1" and not is_mp_child and not is_interactive_notebook():
        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
except ImportError:
    pass

import numpy as np
import time
import concurrent.futures
import numba as nb
import warnings
try:
    from numba.core.errors import NumbaWarning
    warnings.simplefilter('ignore', category=NumbaWarning)
except ImportError:
    pass
from .device import MolMemristor
from .sources import BaseSource, DcSource, PulseSource, PwlSource, BSource
from .device_jit import jit_step_predictor_corrector_array, jit_commit_state_array, jit_getIGeq
from .quantization import WeightQuantizer, OutputQuantizer, InputQuantizer

# High-resolution performance counter frequency for Windows step-level profiling
qpf = None
if sys.platform == 'win32':
    try:
        qpf = ctypes.windll.kernel32.QueryPerformanceFrequency
        qpf.argtypes = [ctypes.c_void_p]
        qpf.restype = ctypes.c_int32
    except Exception:
        pass

from .solver import (
    jit_reset_device_states,
    jit_eval_igeq,
    jit_assemble_and_solve_dense,
    jit_freeze_1t1r_vdevs,
    jit_eval_sources,
    jit_eval_dynamic_sources,
    jit_run_loop,
    jit_cpu_batch_pass_input
)



class _DummyCallable:
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name): return self
    def __setattr__(self, name, value): pass
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __repr__(self): return "0.0"
    def __add__(self, other): return other
    def __radd__(self, other): return other
    def __sub__(self, other): return -other
    def __rsub__(self, other): return other
    def __mul__(self, other): return 0.0
    def __rmul__(self, other): return 0.0
    def __truediv__(self, other): return 0.0
    def __rtruediv__(self, other): return 0.0
    def __lt__(self, other): return True
    def __le__(self, other): return True
    def __gt__(self, other): return False
    def __ge__(self, other): return False
    def __eq__(self, other): return False
    def __ne__(self, other): return True
    def __abs__(self): return 0.0

class _DummyDevice:
    def __init__(self):
        self.params = {}
    def __getattr__(self, name): return _DummyCallable()
    def __setattr__(self, name, value): pass

class _SafeDict(dict):
    def __missing__(self, key): return 0

class _SafeHistoryDict(dict):
    def __missing__(self, key): return np.zeros(1)
    def __contains__(self, key): return True


def _resolve_torch_device(dev_setting):
    """Safely resolves and validates a PyTorch device string against hardware availability and device ordinals."""
    import torch
    from .sys_utils import is_gpu_ready
    if dev_setting == 'auto':
        if is_gpu_ready():
            sel_gpu = os.environ.get('MOLMEM_SELECTED_GPU', '0')
            return torch.device(f'cuda:{sel_gpu}')
        return torch.device('cpu')
    dev_str = str(dev_setting)
    if dev_str.startswith('cuda') or dev_str == 'gpu':
        if not is_gpu_ready(dev_str):
            return torch.device('cpu')
        if ':' in dev_str:
            try:
                dev_idx = int(dev_str.split(':')[1])
                if dev_idx >= torch.cuda.device_count() or dev_idx < 0:
                    dev_str = f'cuda:{torch.cuda.current_device()}'
            except (ValueError, IndexError):
                dev_str = f'cuda:{torch.cuda.current_device()}'
        else:
            dev_str = f'cuda:{torch.cuda.current_device()}'
    return torch.device(dev_str)


class MolmemSimulator(SimulatorSystemMixin, SimulatorProfilerMixin, SimulatorHistoryMixin):
    """
    Non-linear Transient Simulator using Iterative Nodal Analysis (Newton-Raphson).
    """
    _opt_gpu_slice_limit = {"Column": {}, "Device": {}}
    _opt_gpu_read_slice_limit = {"Column": {}, "Device": {}}
    _global_read_stream = None
    _global_read_copy_stream = None

    def __init__(self, instanceName="crossbarArray", device='auto', backend='auto', vram_fraction=0.8):
        if is_mp_child and device == 'auto':
            device = 'cpu'
            backend = 'numba'

        if is_mp_child and ('cuda' in str(device) or backend in ('torch', 'pytorch')):
            self._is_noop = True
            self.instanceName = instanceName
            self.device = device
            self.backend = 'numba'
            self.gpu_slice_limit = None
            self.gpu_read_slice_limit = None
            self.vram_fraction = vram_fraction
            self.force_device_init = False
            self.devices = [{'dev': _DummyDevice()}]
            self.nodesUnknown = set()
            self.nodesKnown = set([0])
            self.vSources = []
            self.dynamicSources = []
            self.wordlines = {}
            self.bitlines = {}
            self.crossbarDevices = _SafeDict()
            self.tArr = np.zeros(1)
            self.vHistory = _SafeHistoryDict()
            self.iHistory = _SafeHistoryDict()
            self.stateHistory = _SafeHistoryDict()
            self.f22History = _SafeHistoryDict()
            self.gHistory = _SafeHistoryDict()
            self.tempHistory = _SafeHistoryDict()
            return

        self.instanceName = instanceName
        try:
            from . import ensure_startup_banner
            ensure_startup_banner()
        except Exception:
            pass
        
        # Override to cuda if forced by command-line flag or environment (strictly bypassed in notebooks)
        _in_nb = is_interactive_notebook()
        if _in_nb or os.environ.get("MOLMEM_CPU_ONLY") == "1":
            device = 'cpu'
            backend = 'numba'

        if device == 'auto' and os.environ.get("MOLMEM_FORCE_GPU") == "1" and not _in_nb:
            device = 'cuda'
            
        self.device = device
        self.requested_device = device
        
        # Auto-resolve device if 'auto'
        resolved_device = device
        if _in_nb or os.environ.get("MOLMEM_CPU_ONLY") == "1" or os.environ.get("MOLMEM_FORCE_CPU") == "1":
            resolved_device = 'cpu'
            self.device = 'cpu'
            self.backend = 'numba'
        elif os.environ.get("MOLMEM_FORCE_GPU") == "1" and self.requested_device != 'cpu':
            from .sys_utils import is_gpu_ready
            if is_gpu_ready():
                selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                resolved_device = f'cuda:{selected_gpu}'
                self.device = resolved_device
                self.backend = 'pytorch'
            else:
                resolved_device = 'cpu'
                self.device = 'cpu'
                self.backend = 'numba'
        elif resolved_device == 'auto':
            # In auto mode, normal transient simulations default strictly to CPU (Numba JIT).
            # GPU/Hybrid acceleration is activated for parallel crossbar operations (weights, batches, tiled) in addCrossbarMatrix().
            resolved_device = 'cpu'
            self.device = 'cpu'
            self.backend = 'numba'
        elif resolved_device in ('cuda', 'gpu'):
            from .sys_utils import is_gpu_ready
            if is_gpu_ready():
                selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                resolved_device = f'cuda:{selected_gpu}'
                self.device = resolved_device
            else:
                resolved_device = 'cpu'
                self.device = 'cpu'
                self.backend = 'numba'
        else:
            self.device = resolved_device

        if str(resolved_device).startswith('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    dev_idx = int(str(resolved_device).split(':')[1]) if ':' in str(resolved_device) else 0
                    if getattr(torch.version, 'hip', None) is None:
                        try:
                            torch.cuda.set_per_process_memory_fraction(float(vram_fraction), device=dev_idx)
                        except Exception:
                            pass
                    if not getattr(MolmemSimulator, '_vram_prewarmed', False):
                        try:
                            free_mem, _ = torch.cuda.mem_get_info(dev_idx)
                            prewarm_bytes = int(free_mem * float(vram_fraction) * 0.70)
                            if prewarm_bytes > 0:
                                _warmup_tensor = torch.empty(prewarm_bytes // 8, dtype=torch.float64, device=f'cuda:{dev_idx}')
                                del _warmup_tensor
                            MolmemSimulator._vram_prewarmed = True
                        except Exception:
                            pass
            except Exception:
                pass
        
        # Resolve backend: CPU is strictly Numba JIT; GPU uses compiled C++/HIP
        if not str(resolved_device).startswith('cuda'):
            self.backend = 'numba'
        elif backend == 'auto':
            self.backend = 'pytorch'
        else:
            self.backend = backend
        self.gpu_slice_limit = None
        self.gpu_read_slice_limit = None
        self.vram_fraction = vram_fraction
        self._write_back_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="MolmemStateWriter")
        self._write_back_futures = []
        
        # Register simulator instance for Ctrl+C cleanup
        try:
            import molmem_lib
            molmem_lib.register_simulator(self)
        except Exception:
            pass
            
        self.force_device_init = True
        self._in_sync_back = False
        self._gpu_states_in_sync = True
        self._gpu_states_dirty = False
        self._cached_hw_tensors = None
        self._cached_hw_arrays = None
        
        self.devices = []       # List of dicts: {'top': idx, 'bottom': idx, 'dev': obj}
        self.nodesUnknown = set()
        self.nodesKnown = set([0])  # Node 0 is always Ground (0V)
        self.vSources = []     # List of Source objects from sources.py
        self.dynamicSources = [] # List of Norton Equivalent High-Z Sources
        self._sources_cached = False
        self._cached_vCurrent_base = None
        self._cached_all_sources = None
        self._cached_src_signatures = None
        self._cached_fast_v_sources_idx = None
        self._cached_pwl_t_matrix = None
        self._cached_pwl_v_matrix = None
        self._cached_pwl_mask = None
        self._cached_other_v_sources = None
        
        # Crossbar Abstraction Mapping
        self.wordlines = {}  # row -> node_id
        self.bitlines = {}   # col -> node_id
        self.crossbarDevices = {} # (row, col) -> dev_idx
        
        # Simulation Results
        self.tArr = None
        self.vHistory = {}
        self.iHistory = {}      # dev_idx -> array
        self.stateHistory = {}  # dev_idx -> array (n)
        self.f22History = {}    # dev_idx -> array (state)
        self.gHistory = {}      # dev_idx -> array (Siemens)
        self.tempHistory = {}   # dev_idx -> array (Kelvin)
        
        # ThreadPoolExecutor cache for massive crossbar arrays
        self.target_cores = max(1, int((os.cpu_count() or 1) * 0.8))
        self.executor = None
        
        # configurable bleed resistor (10 GigaOhms default) for High-Z stability
        self.leakage_g = 1e-10
        self.detailedPrint = True
        self._gpu_cache = None
        self.compile_backend = False
        if os.environ.get("MOLMEM_FORCE_JIT") != "1" and os.environ.get("MOLMEM_CPU_ONLY") != "1" and str(getattr(self, 'device', 'cpu')).lower() != 'cpu':
            try:
                from .torch_simulator import check_has_cpp_extension
                self.compile_backend = check_has_cpp_extension()
            except Exception:
                pass
        self._gpu_states_dirty = True
        self._gpu_states_in_sync = True
        self.lazy_sync = True
        self.is_crossbar = False
        self._is_noop = False

        # Register atexit cleanup to prevent Tcl/Tk thread deadlocks on exit
        import atexit
        self._atexit_cleanup_ref = self._atexit_cleanup
        atexit.register(self._atexit_cleanup_ref)

    def _atexit_cleanup(self):
        try:
            if hasattr(self, '_write_back_executor') and self._write_back_executor is not None:
                self._write_back_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            if hasattr(self, 'executor') and self.executor is not None:
                self.executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.clearHistory(force_gc=True)
        except Exception:
            pass

    def __del__(self):
        # Unregister atexit cleanup to avoid memory leaks
        try:
            import atexit
            if hasattr(self, '_atexit_cleanup_ref'):
                atexit.unregister(self._atexit_cleanup_ref)
        except Exception:
            pass

        if hasattr(self, '_write_back_executor') and self._write_back_executor is not None:
            try:
                self._write_back_executor.shutdown(wait=False)
            except Exception:
                pass
            self._write_back_executor = None

        if hasattr(self, 'executor') and self.executor is not None:
            try:
                from .sys_utils import log_to_run_log_only
                log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Terminated ThreadPoolExecutor: MolmemSimulator_Parallel | Workers={getattr(self, 'active_executor_cores', 'unknown')} | Platform=CPU")
            except Exception:
                pass
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass
            self.executor = None

        try:
            self.clearHistory(force_gc=True)
        except Exception:
            pass

        # Release internal GPU cache tensor structures
        if hasattr(self, '_gpu_cache'):
            self._gpu_cache = None
        if hasattr(self, 'programmer') and self.programmer is not None:
            if hasattr(self.programmer, '_device_sims'):
                self.programmer._device_sims = None

        # Trigger physical VRAM release if using PyTorch backend
        if getattr(self, 'backend', 'numba') in ['torch', 'pytorch']:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


    def _ensure_disk_history_store(self, num_nodes, num_devices, initial_capacity, recordHistory, recordStepTiming=False, batch_size=1):
        from .history import DiskHistoryStore
        from .sys_utils import resolve_record_history_flags
        v_flag, i_flag, state_flag, f22_flag, g_flag, temp_flag, _ = resolve_record_history_flags(recordHistory)
        
        batch_size = max(1, int(batch_size))
        store = getattr(self, '_disk_history_store', None)
        if store is not None and getattr(store, 'batch_size', 1) != batch_size:
            try:
                store.cleanup()
            except Exception:
                pass
            store = None
            self._disk_history_store = None

        if store is None:
            sim_id = getattr(self, 'instanceName', 'sim') + '_' + str(id(self))
            store = DiskHistoryStore(
                sim_id=sim_id,
                num_nodes=num_nodes,
                num_devices=num_devices,
                initial_capacity=initial_capacity,
                v_flag=v_flag,
                i_flag=i_flag,
                state_flag=state_flag,
                f22_flag=f22_flag,
                g_flag=g_flag,
                temp_flag=temp_flag,
                step_time_flag=recordStepTiming,
                batch_size=batch_size
            )
            self._disk_history_store = store
        else:
            store.ensure_capacity(initial_capacity)
            
        self._tArr_full = store.tArr
        self._stepTimeHistory_full = store.stepTime
        self._vHistory_full = store.vHist
        self._iHistory_full = store.iHist
        self._stateHistory_full = store.stateHist
        self._f22History_full = store.f22Hist
        self._gHistory_full = store.gHist
        self._tempHistory_full = store.tempHist
        self._hist_capacity_total = store.capacity
        return store

    def clearInternalCaches(self, force_gc=False):
        """
        Forcefully purge all dynamically generated topological caching bounds
        and Memory-Pool Trace histories from active RAM, returning the engine to baseline.
        """
        if hasattr(self, '_disk_history_store') and self._disk_history_store is not None:
            try:
                self._disk_history_store.cleanup()
                self._disk_history_store = None
            except Exception:
                pass

        cache_targets = [
            '_tArr_full', '_vHistory_full', '_iHistory_full', 
            '_stateHistory_full', '_f22History_full', '_gHistory_full', '_tempHistory_full',
            '_hist_active_len', '_hist_capacity_total'
        ]
        
        for attr in cache_targets:
            if hasattr(self, attr):
                delattr(self, attr)
                
        self.tArr = None
        self.vHistory = {}
        self.iHistory = {}
        self.stateHistory = {}
        self.f22History = {}
        self.gHistory = {}
        self.tempHistory = {}
        self._gpu_cache = None
        
        if hasattr(self, 'programmer') and self.programmer is not None:
            if hasattr(self.programmer, '_device_sims'):
                self.programmer._device_sims = None
        
        if force_gc:
            import gc
            gc.collect()
        
        # Free CUDA/ROCm memory cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def clearHistory(self, force_gc=False):
        """
        Clears simulation history from RAM and disk without destroying the matrix topological caches.
        """
        cache_targets = [
            '_tArr_full', '_stepTimeHistory_full', '_vHistory_full', '_iHistory_full', 
            '_stateHistory_full', '_f22History_full', '_gHistory_full', '_tempHistory_full',
            '_hist_active_len', '_hist_capacity_total'
        ]
        
        for attr in cache_targets:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
                
        self.tArr = None
        self.stepTimeHistory = None
        self.vHistory = {}
        self.iHistory = {}
        self.stateHistory = {}
        self.f22History = {}
        self.gHistory = {}
        self.tempHistory = {}
        
        if hasattr(self, '_disk_history_store') and self._disk_history_store is not None:
            try:
                self._disk_history_store.cleanup()
                self._disk_history_store = None
            except Exception:
                pass
        
        # Free CUDA/ROCm memory cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def sync_to_cpu(self):
        """
        Synchronizes device states from GPU/PyTorch cache back to host CPU objects.
        """
        if not getattr(self, '_gpu_states_in_sync', True):
            cache = getattr(self, '_gpu_states_cache', None)
            if cache is None:
                cache = getattr(self, '_gpu_cache', None)
            if cache is not None:
                self._in_sync_back = True
                try:
                    import numpy as np
                    
                    if isinstance(cache, dict):
                        n_arr = cache.get('n_arr')
                        gEq_arr = cache.get('gEq_arr')
                        modeState_arr = cache.get('modeState_arr')
                        scaleFactor_arr = cache.get('scaleFactor_arr')
                        f22DepScaled_arr = cache.get('f22DepScaled_arr')
                        f22PotScaled_arr = cache.get('f22PotScaled_arr')
                        _cached_n_int_arr = cache.get('_cached_n_int_arr')
                        iScale_arr = cache.get('iScale_arr')
                        vScale_arr = cache.get('vScale_arr')
                        currentF22_arr = cache.get('currentF22_arr')
                        currentIScale_arr = cache.get('currentIScale_arr')
                        currentVScale_arr = cache.get('currentVScale_arr')
                        T_arr = cache.get('T_arr')
                    else:
                        n_arr, modeState_arr, scaleFactor_arr, T_arr, currentF22_arr, _cached_n_int_arr, iScale_arr, vScale_arr, currentIScale_arr, currentVScale_arr, f22DepScaled_arr, f22PotScaled_arr, gEq_arr = cache

                    def _to_np_flat(x):
                        if x is None:
                            return None
                        if hasattr(x, 'cpu'):
                            arr = x.cpu().numpy()
                        else:
                            arr = np.asarray(x)
                        return arr.ravel()

                    gEq_host = _to_np_flat(gEq_arr)
                    n_host = _to_np_flat(n_arr)
                    if gEq_host is None and n_host is not None:
                        gEq_host = np.zeros_like(n_host)
                        
                    modeState_host = _to_np_flat(modeState_arr)
                    scaleFactor_host = _to_np_flat(scaleFactor_arr)
                    f22DepScaled_host = _to_np_flat(f22DepScaled_arr)
                    f22PotScaled_host = _to_np_flat(f22PotScaled_arr)
                    _cached_n_int_host = _to_np_flat(_cached_n_int_arr)
                    iScale_host = _to_np_flat(iScale_arr)
                    vScale_host = _to_np_flat(vScale_arr)
                    currentF22_host = _to_np_flat(currentF22_arr)
                    currentIScale_host = _to_np_flat(currentIScale_arr)
                    currentVScale_host = _to_np_flat(currentVScale_arr)
                    T_host = _to_np_flat(T_arr)
                    
                    if not hasattr(self, '_device_objs_flat') or self._device_objs_flat is None or len(self._device_objs_flat) != len(self.devices):
                        self._device_objs_flat = [d['dev'] for d in self.devices]
                    
                    devs = self._device_objs_flat
                    num_devs = len(devs)
                    if num_devs > 0:
                        d0 = devs[0]
                        is_uniform = (n_host is not None and len(n_host) == num_devs and np.all(n_host == n_host[0]) and
                                      T_host is not None and len(T_host) == num_devs and np.all(T_host == T_host[0]))
                        if is_uniform:
                            n0 = n_host[0]
                            t0 = T_host[0]
                            g0 = gEq_host[0] if gEq_host is not None else None
                            m0 = modeState_host[0] if modeState_host is not None else None
                            sc0 = scaleFactor_host[0] if scaleFactor_host is not None else None
                            fd0 = f22DepScaled_host[0] if f22DepScaled_host is not None else None
                            fp0 = f22PotScaled_host[0] if f22PotScaled_host is not None else None
                            val0 = _cached_n_int_host[0] if _cached_n_int_host is not None else None
                            c0 = (int(val0) if (np.isfinite(val0) and -2147483648 <= val0 <= 2147483647) else -1) if val0 is not None else None
                            is0 = iScale_host[0] if iScale_host is not None else None
                            vs0 = vScale_host[0] if vScale_host is not None else None
                            cf0 = currentF22_host[0] if currentF22_host is not None else None
                            cis0 = currentIScale_host[0] if currentIScale_host is not None else None
                            cvs0 = currentVScale_host[0] if currentVScale_host is not None else None
                            
                            if not (d0._n == n0 and d0._T == t0 and (g0 is None or getattr(d0, '_G', None) == g0) and
                                    (m0 is None or d0._modeState == m0) and (sc0 is None or d0._scaleFactor == sc0)):
                                for dev in devs:
                                    if g0 is not None: dev._G = g0
                                    dev._n = n0
                                    if m0 is not None: dev._modeState = m0
                                    if sc0 is not None: dev._scaleFactor = sc0
                                    if fd0 is not None: dev._f22DepScaled = fd0
                                    if fp0 is not None: dev._f22PotScaled = fp0
                                    if c0 is not None: dev._cached_n_int_val = c0
                                    if is0 is not None: dev._iScale = is0
                                    if vs0 is not None: dev._vScale = vs0
                                    if cf0 is not None: dev._currentF22 = cf0
                                    if cis0 is not None: dev._currentIScale = cis0
                                    if cvs0 is not None: dev._currentVScale = cvs0
                                    dev._T = t0
                        else:
                            len_g = len(gEq_host) if gEq_host is not None else 0
                            len_n = len(n_host) if n_host is not None else 0
                            len_m = len(modeState_host) if modeState_host is not None else 0
                            len_sc = len(scaleFactor_host) if scaleFactor_host is not None else 0
                            len_fd = len(f22DepScaled_host) if f22DepScaled_host is not None else 0
                            len_fp = len(f22PotScaled_host) if f22PotScaled_host is not None else 0
                            len_c = len(_cached_n_int_host) if _cached_n_int_host is not None else 0
                            len_is = len(iScale_host) if iScale_host is not None else 0
                            len_vs = len(vScale_host) if vScale_host is not None else 0
                            len_cf = len(currentF22_host) if currentF22_host is not None else 0
                            len_cis = len(currentIScale_host) if currentIScale_host is not None else 0
                            len_cvs = len(currentVScale_host) if currentVScale_host is not None else 0
                            len_t = len(T_host) if T_host is not None else 0
                            def _sync_slice(s, e):
                                for idx in range(s, e):
                                    dev = devs[idx]
                                    if idx < len_g: dev._G = gEq_host[idx]
                                    if idx < len_n: dev._n = n_host[idx]
                                    if idx < len_m: dev._modeState = modeState_host[idx]
                                    if idx < len_sc: dev._scaleFactor = scaleFactor_host[idx]
                                    if idx < len_fd: dev._f22DepScaled = f22DepScaled_host[idx]
                                    if idx < len_fp: dev._f22PotScaled = f22PotScaled_host[idx]
                                    if idx < len_c:
                                        val = _cached_n_int_host[idx]
                                        dev._cached_n_int_val = int(val) if (np.isfinite(val) and -2147483648 <= val <= 2147483647) else -1
                                    if idx < len_is: dev._iScale = iScale_host[idx]
                                    if idx < len_vs: dev._vScale = vScale_host[idx]
                                    if idx < len_cf: dev._currentF22 = currentF22_host[idx]
                                    if idx < len_cis: dev._currentIScale = currentIScale_host[idx]
                                    if idx < len_cvs: dev._currentVScale = currentVScale_host[idx]
                                    if idx < len_t: dev._T = T_host[idx]

                            import os
                            system_cap = max(1, int((os.cpu_count() or 1) * 0.8))
                            num_workers = min(system_cap, max(1, num_devs // 512))
                            if num_workers > 1:
                                chunk_sz = (num_devs + num_workers - 1) // num_workers
                                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                                    futures = []
                                    for w in range(num_workers):
                                        s = w * chunk_sz
                                        e = min(s + chunk_sz, num_devs)
                                        if s < e:
                                            futures.append(executor.submit(_sync_slice, s, e))
                                    for f in futures:
                                        f.result()
                            else:
                                _sync_slice(0, num_devs)

                finally:
                    self._gpu_states_in_sync = True
                    self._in_sync_back = False


    def resetDeviceStates(self, n=0.0, modeState=0.0, scaleFactor=1.0, T=298.15):
        """
        Instantly resets all device states to baseline/default values.
        Utilizes high-performance JIT-compiled loops for CPU and vectorized transfers.
        """
        if hasattr(self, '_write_back_futures') and self._write_back_futures:
            concurrent.futures.wait(self._write_back_futures)
            self._write_back_futures.clear()

        num_devices = len(self.devices)
        if num_devices == 0:
            return
            
        if not hasattr(self, '_device_objs_flat') or self._device_objs_flat is None or len(self._device_objs_flat) != num_devices:
            self._device_objs_flat = [d['dev'] for d in self.devices]
        devs = self._device_objs_flat
        
        n_arr = np.empty(num_devices, dtype=np.float64)
        modeState_arr = np.empty(num_devices, dtype=np.float64)
        scaleFactor_arr = np.empty(num_devices, dtype=np.float64)
        T_arr = np.empty(num_devices, dtype=np.float64)
        currentF22_arr = np.empty(num_devices, dtype=np.float64)
        _cached_n_int_arr = np.empty(num_devices, dtype=np.int32)
        
        jit_reset_device_states(n_arr, modeState_arr, scaleFactor_arr, T_arr, currentF22_arr, _cached_n_int_arr, n, modeState, scaleFactor, T)
        
        for i, dev in enumerate(devs):
            dev._n = n_arr[i]
            dev._modeState = modeState_arr[i]
            dev._scaleFactor = scaleFactor_arr[i]
            dev._T = T_arr[i]
            dev._currentF22 = currentF22_arr[i]
            dev._cached_n_int_val = int(_cached_n_int_arr[i])
            
        if hasattr(self, '_soa_cDischarge_arr') and self._soa_cDischarge_arr is not None:
            self._soa_cDischarge_arr.fill(0.0)
            
        self._gpu_states_dirty = True
        self._gpu_states_in_sync = True
        self._invalidate_hardware_tensor_cache()

    def relaxDeviceTemperatures(self, T_amb=298.15):
        """
        Relaxes all device temperatures back to ambient baseline (T_amb = 298.15 K),
        representing physical thermal equilibration into the bulk substrate during idle periods
        (such as ADC readback intervals or between API calls). Conductance states (n, modeState,
        scaleFactor) are strictly preserved.
        """
        if hasattr(self, '_write_back_futures') and self._write_back_futures:
            concurrent.futures.wait(self._write_back_futures)
            self._write_back_futures.clear()

        num_devices = len(self.devices)
        if num_devices == 0:
            return

        if not hasattr(self, '_device_objs_flat') or self._device_objs_flat is None or len(self._device_objs_flat) != num_devices:
            self._device_objs_flat = [d['dev'] for d in self.devices]
        devs = self._device_objs_flat

        t_val = float(T_amb)
        for dev in devs:
            dev._T = t_val

        cache = getattr(self, '_gpu_states_cache', None)
        if cache is not None and isinstance(cache, dict):
            if 'T_arr' in cache and cache['T_arr'] is not None:
                t_arr = cache['T_arr']
                if hasattr(t_arr, 'fill_'):
                    t_arr.fill_(t_val)
                elif hasattr(t_arr, 'fill'):
                    t_arr.fill(t_val)
            if 'device_states_soa' in cache and cache['device_states_soa'] is not None:
                soa = cache['device_states_soa']
                if hasattr(soa, 'shape') and len(soa.shape) >= 2:
                    try:
                        if soa.shape[0] > 11:
                            if hasattr(soa[11], 'fill_'):
                                soa[11].fill_(t_val)
                            elif hasattr(soa[11], 'fill'):
                                soa[11].fill(t_val)
                        elif len(soa.shape) == 3 and soa.shape[1] > 11:
                            if hasattr(soa[:, 11, :], 'fill_'):
                                soa[:, 11, :].fill_(t_val)
                            elif hasattr(soa[:, 11, :], 'fill'):
                                soa[:, 11, :].fill(t_val)
                    except Exception:
                        pass

    def addMemristor(self, topNode, bottomNode, deviceInstance):
        """
        Connects a device between topNode and bottomNode.
        Current flows from topNode to bottomNode when V(top) > V(bottom).
        """
        self.devices.append({'top': topNode, 'bottom': bottomNode, 'dev': deviceInstance})
        if hasattr(deviceInstance, '_sim'):
            deviceInstance._sim = weakref.ref(self)
        self._registerNode(topNode)
        self._registerNode(bottomNode)
        
    def addCrossbarMatrix(self, matrix=None, rows=None, cols=None, deviceClass=None, device_type="default", weightBits=None, inputBits=None, outputBits=None, vWrite=0.9, vErase=-0.75, max_vWrite=5.0, max_pw=320e-9, detailedPrint=None, architecture='1T1R', UpdatesPer="Column", force_device_init=True, multiThread=None, write_noise_intensity=0.0, read_noise_intensity=0.0, d2d_variation_intensity=0.0, ignore_safety_cap=False, **kwargs):
        if getattr(self, '_is_noop', False):
            return
        """
        Utility to explicitly instantiate an abstract crossbar matrix.
        Provide 'rows' and 'cols' to auto-generate a full dense array.
        Provide 'matrix' as a list of (row, col) tuples to build a custom/sparse array.
        Nodes are automatically generated so the user only interacts with Wordlines (Rows) and Bitlines (Cols).
        
        Optional Machine Learning API:
        Passing `weightBits`, `inputBits`, and `outputBits` will automatically 
        initialize the hardware inference backend for use with `updateCrossbarWeights()` and `passInput()`.
        
        detailedPrint=False: Compact single-line outputs for profiler, calibration, and threading.
        """
        self._gpu_cache = None
        self._soa_cached = False
        if multiThread is not None:
            self.multiThread = bool(multiThread)
        if detailedPrint is not None:
            self.detailedPrint = detailedPrint
        self.force_device_init = force_device_init
        self.idleStateRow = np.nan
        self.idleStateCol = np.nan
        self.leakage_g = 1e-10
        self.vRead = 0.1
        self.architecture = architecture
        self.UpdatesPer = UpdatesPer
        self.write_noise_intensity = float(write_noise_intensity)
        self.read_noise_intensity = float(read_noise_intensity)
        self.d2d_variation_intensity = float(d2d_variation_intensity)
        basis = getattr(self, 'UpdatesPer', 'Column')
        self.gpu_slice_limit = 999999
        self.gpu_read_slice_limit = 999999
        self.is_crossbar = True
        
        if self.architecture == '1T1R':
            self.idleStateCol = np.nan
        
        MolmemSimulator.warmup_compilers(verbose=self.detailedPrint)
        
        if matrix is None and (rows is None or cols is None):
            raise ValueError("Must provide either a custom 'matrix' pattern or both 'rows' and 'cols'.")
            
        if matrix is None:
            matrix = [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)]
            
        # Determine unique rows and cols in a single pass to auto-generate node IDs
        uniqueRows = set()
        uniqueCols = set()
        for r, c in matrix:
            uniqueRows.add(r)
            uniqueCols.add(c)
        
        # Start node IDs progressively
        maxNode = max(self.nodesKnown.union(self.nodesUnknown)) if (self.nodesKnown or self.nodesUnknown) else 0
        
        for r in uniqueRows:
            if r not in self.wordlines:
                maxNode += 1
                self.wordlines[r] = maxNode
                if maxNode not in self.nodesKnown and maxNode not in self.nodesUnknown:
                    self.nodesUnknown.add(maxNode)
        for c in uniqueCols:
            if c not in self.bitlines:
                maxNode += 1
                self.bitlines[c] = maxNode
                if maxNode not in self.nodesKnown and maxNode not in self.nodesUnknown:
                    self.nodesUnknown.add(maxNode)
                    
        if deviceClass is None:
            deviceClass = MolMemristor
            
        _fast_cache = None
        if deviceClass is MolMemristor:
            dummy = MolMemristor(device_type=device_type, force_init=force_device_init)
            if isinstance(device_type, str):
                cache_key = device_type.lower()
            elif isinstance(device_type, dict):
                cache_key = tuple((k, v) for k, v in sorted(device_type.items()) if isinstance(v, (int, float, str, bool, type(None))))
            else:
                cache_key = id(device_type)
            params = dummy.params
            lookups = dummy.lookups
            vals = MolMemristor._initial_vals_cache.get(cache_key)
            if vals is None:
                vals = {
                    '_cached_n_int': dummy._cached_n_int_val,
                    'iScale': dummy._iScale,
                    'vScale': dummy._vScale,
                    'f22PotScaled': dummy._f22PotScaled,
                    'f22DepScaled': dummy._f22DepScaled,
                    'currentF22': dummy._currentF22,
                    'currentIScale': dummy._currentIScale,
                    'currentVScale': dummy._currentVScale
                }
            const_tuple = (
                dummy.vSmooth, dummy.nSmooth, dummy.kappa, dummy.alpha, dummy.gScale, dummy.vTh,
                dummy.vRefPos, dummy.vRefNeg, dummy.rateAsymmetry, dummy.nMax, dummy.kDischarge, dummy.rC,
                dummy.E_a, dummy.R_th, dummy.tau_th, dummy.gamma, dummy.beta_alpha, dummy.beta_s,
                dummy.rTopBlend, dummy.fDischarge, dummy.cDischarge, dummy.rBottomBlend, dummy.rLatency,
                dummy.k_overdrive, dummy.safe_max_vwrite, dummy.f22StartVal, dummy.f22PeakVal, dummy.f22EndVal,
                dummy.f22PotScaleInv, dummy.f22DepScaleInv, dummy.vBypass, dummy.f22Floor,
                dummy._y_i, dummy._y_v, dummy._y_fp, dummy._y_fd
            )
            init_vals_tuple = (
                dummy._n, dummy._modeState, dummy._cached_n_int_val, dummy._iScale, dummy._vScale,
                dummy._f22PotScaled, dummy._f22DepScaled, dummy._currentF22, dummy._currentIScale, dummy._currentVScale
            )
            _fast_cache = (params, lookups, dummy.dataDir, const_tuple, init_vals_tuple)

        self_ref = weakref.ref(self)
        self_devices_append = self.devices.append
        self_crossbarDevices = self.crossbarDevices
        self_wordlines = self.wordlines
        self_bitlines = self.bitlines
        devIdx = len(self.devices)

        if deviceClass is MolMemristor:
            for r, c in matrix:
                nTop = self_wordlines[r]
                nBot = self_bitlines[c]
                dev = MolMemristor(device_type=device_type, force_init=force_device_init, _fast_cache=_fast_cache)
                self_devices_append({'top': nTop, 'bottom': nBot, 'dev': dev})
                dev._sim = self_ref
                self_crossbarDevices[(r, c)] = devIdx
                devIdx += 1
        else:
            for r, c in matrix:
                nTop = self_wordlines[r]
                nBot = self_bitlines[c]
                try:
                    dev = deviceClass(device_type=device_type, force_init=force_device_init)
                except TypeError:
                    try:
                        dev = deviceClass(device_type=device_type)
                    except TypeError:
                        dev = deviceClass()
                self_devices_append({'top': nTop, 'bottom': nBot, 'dev': dev})
                if hasattr(dev, '_sim'):
                    dev._sim = self_ref
                self_crossbarDevices[(r, c)] = devIdx
                devIdx += 1

        # Apply Empirical D2D Spatial Mismatch if enabled
        if self.d2d_variation_intensity > 0.0:
            from .noise import NoiseEngine
            d2d_deltas = NoiseEngine.sample_d2d_factors(len(matrix), intensity=self.d2d_variation_intensity)
            for i_m, (r, c) in enumerate(matrix):
                dev_idx = self_crossbarDevices[(r, c)]
                d = self.devices[dev_idx]['dev']
                if hasattr(d, 'gScale'):
                    d.gScale = d.gScale * (1.0 + float(d2d_deltas[i_m]))
                    if hasattr(d, '_forceUpdateConductanceParams'):
                        d._forceUpdateConductanceParams()
            self._all_dev_params_same = (len(matrix) <= 1)

        self._dev_top_nodes_arr = np.array([d['top'] for d in self.devices], dtype=np.int32)
        self._dev_bot_nodes_arr = np.array([d['bottom'] for d in self.devices], dtype=np.int32)
        self._dev_objs_flat = [d['dev'] for d in self.devices]
        self._cached_cpu_topology = None
        self._cached_torch_topology = None

        # Initialize ML backend if bit-depths were provided
        if weightBits is not None and inputBits is not None and outputBits is not None:
            from .programmer import CrossbarProgrammer
            
            self.weightBits = weightBits
            self.inputBits = inputBits
            self.outputBits = outputBits
            
            self.wq = WeightQuantizer(targetBits=weightBits, maxStates=16520)
            self.inQ = InputQuantizer(inputBits=inputBits)
            self.adc = OutputQuantizer(adcBits=outputBits)
            # Auto-clamp max_vWrite to device hardware safety limit if configured
            # Fallback: if safe_max_vwrite is 0, 0.0, or None, use the default or passed API value
            first_dev_obj = self.devices[0]['dev'] if len(self.devices) > 0 else None
            safe_cap = getattr(first_dev_obj, 'safe_max_vwrite', None)
            if safe_cap is None and hasattr(first_dev_obj, 'params'):
                safe_cap = first_dev_obj.params.get('safe_max_vwrite', None)
            
            effective_max_vWrite = float(max_vWrite)
            try:
                safe_cap_float = float(safe_cap) if safe_cap is not None else 0.0
            except (ValueError, TypeError):
                safe_cap_float = 0.0

            if not ignore_safety_cap and safe_cap_float > 0.0:
                if effective_max_vWrite > safe_cap_float + 1e-6:
                    if getattr(self, 'detailedPrint', True) and not getattr(self, '_is_calibration_helper', False):
                        print(f"[MolmemSimulator] Notice: Requested max_vWrite ({effective_max_vWrite:.2f}V) exceeds device safe limit ({safe_cap_float:.2f}V). Auto-clamping to {safe_cap_float:.2f}V.")
                    effective_max_vWrite = safe_cap_float

            self.programmer = CrossbarProgrammer(self, vWrite=vWrite, vErase=vErase, pulseWidthPot=80e-9, pulseWidthDep=60e-9, period=160e-9, max_vWrite=effective_max_vWrite, max_pw=max_pw)
            self.isCalibrated = False
            # Auto-link calibrated optimal pulse maps from cache so the programmer immediately knows pulse counts
            if not getattr(self, '_is_calibration_helper', False) and not getattr(self.__class__, '_in_warmup', False):
                try:
                    self.programmer.calibrate()
                    self.isCalibrated = True
                except Exception:
                    self.isCalibrated = False
            
        # A circuit style run is one where the high-level ML API (weightBits, inputBits, outputBits) is not fully specified.
        is_ml_backend = (weightBits is not None and inputBits is not None and outputBits is not None)

        # Resolve device and backend dynamically if device='auto'
        device = getattr(self, 'requested_device', getattr(self, 'device', 'auto'))
        backend = getattr(self, 'backend', 'auto')
        
        if os.environ.get("MOLMEM_FORCE_CPU") == "1":
            device = 'cpu'
            self.device = 'cpu'
            self.backend = 'numba'
        elif os.environ.get("MOLMEM_FORCE_GPU") == "1" and getattr(self, 'requested_device', None) != 'cpu' and getattr(self, 'device', None) != 'cpu':
            from .sys_utils import is_gpu_ready
            if is_gpu_ready():
                selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                device = f'cuda:{selected_gpu}'
                self.device = device
                self.backend = 'pytorch'
            else:
                self.device = 'cpu'
                self.backend = 'numba'
        elif device in ['auto', 'hybrid']:
            if not is_ml_backend:
                self.device = 'cpu'
                self.backend = 'numba'
            else:
                from .sys_utils import is_gpu_ready
                if is_gpu_ready():
                    selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                    self.device = f'cuda:{selected_gpu}'
                    self.backend = 'hybrid'
                else:
                    self.device = 'cpu'
                    self.backend = 'numba'
        else:
            # Respect user explicit device
            if str(self.device).startswith('cuda'):
                self.backend = 'pytorch' if backend == 'auto' else backend
            elif self.device == 'cpu':
                self.backend = 'numba' if backend == 'auto' else backend
                
        # Resolve dynamic GPU slice limit
        if str(self.device).startswith('cuda') and is_ml_backend:
            cur_dim = max(len(uniqueRows), len(uniqueCols))
            basis = getattr(self, 'UpdatesPer', 'Column')
            
            # Load cache
            MolmemSimulator._load_profile_cache(basis=basis)
            
            # Initialize dictionaries for the active basis if they are missing
            if not isinstance(MolmemSimulator._opt_gpu_slice_limit, dict):
                MolmemSimulator._opt_gpu_slice_limit = {"Column": {}, "Device": {}}
            if not isinstance(MolmemSimulator._opt_gpu_read_slice_limit, dict):
                MolmemSimulator._opt_gpu_read_slice_limit = {"Column": {}, "Device": {}}
                
            if basis not in MolmemSimulator._opt_gpu_slice_limit:
                MolmemSimulator._opt_gpu_slice_limit[basis] = {}
            if basis not in MolmemSimulator._opt_gpu_read_slice_limit:
                MolmemSimulator._opt_gpu_read_slice_limit[basis] = {}
                
            cur_dim_key = int(cur_dim)
            
            # If the current dimension has not been calibrated/cached, calibrate it now
            if not hasattr(MolmemSimulator, '_calibrating_slice_sizes'):
                MolmemSimulator._calibrating_slice_sizes = set()
                
            if cur_dim_key not in MolmemSimulator._opt_gpu_slice_limit[basis] and cur_dim_key not in MolmemSimulator._calibrating_slice_sizes:
                import multiprocessing
                if multiprocessing.current_process().name == 'MainProcess':
                    MolmemSimulator._calibrating_slice_sizes.add(cur_dim_key)
                    try:
                        self.profileGpuSliceLimit(cur_dim_key)
                    finally:
                        MolmemSimulator._calibrating_slice_sizes.discard(cur_dim_key)
                        self._gpu_states_cache = None
                        self._gpu_states_in_sync = True
                        self._gpu_states_dirty = True
            
            # Assign limits from cache
            self.gpu_slice_limit = MolmemSimulator._opt_gpu_slice_limit[basis].get(cur_dim_key, 999999)
            if self.gpu_slice_limit is None:
                self.gpu_slice_limit = 999999
                
            self.gpu_read_slice_limit = MolmemSimulator._opt_gpu_read_slice_limit[basis].get(cur_dim_key, 999999)
            if self.gpu_read_slice_limit is None:
                self.gpu_read_slice_limit = 999999
            
            # Dedicated print message when the slice cache is loaded
            suppress_cache = getattr(MolmemSimulator, '_suppress_cache_hit_prints', False)
            if self.detailedPrint and not suppress_cache:
                msg_key = f"MOLMEM_GPU_SLICE_MSG_SHOWN_{basis}_{cur_dim_key}"
                if os.environ.get(msg_key) != '1':
                    print(f"  GPU Slice Cache Loaded ({basis}) [{'#'*20}] 100% | Size {cur_dim_key}x{cur_dim_key} -> Slice Limit: {self.gpu_slice_limit}, Read Limit: {self.gpu_read_slice_limit}")
                    os.environ[msg_key] = '1'
                

        import multiprocessing
        is_subproc = (multiprocessing.current_process().name != 'MainProcess') or os.environ.get("MOLMEM_IN_SUBPROCESS") == "1" or os.environ.get("MOLMEM_MP_CHILD") == "1" or os.environ.get("MOLMEM_SILENT_WARMUP") == "1"
        is_profiling = getattr(self, '_is_calibration_helper', False) or getattr(self, '_in_warmup', False) or getattr(self.__class__, '_in_warmup', False) or getattr(MolmemSimulator, '_in_warmup', False)
        if not getattr(self, '_is_noop', False) and not is_profiling and not is_subproc:
            if self.detailedPrint and not getattr(MolmemSimulator, '_full_init_printed', False) and not getattr(MolmemSimulator, '_hardware_status_printed', False):
                r_cnt = len(uniqueRows)
                c_cnt = len(uniqueCols)
                dev_cnt = len(self.devices)
                arch_str = getattr(self, 'architecture', '1R')
                dev_str = "CPU"
                device_name = str(getattr(self, 'device', 'cpu')).lower()
                is_gpu = 'cuda' in device_name or 'hip' in device_name
                if is_gpu:
                    gpu_name = "ROCm/HIP" if 'hip' in device_name or getattr(self, 'compile_backend', False) else "CUDA"
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_name = torch.cuda.get_device_name(int(device_name.split(':')[1]) if ':' in device_name else 0)
                    except Exception:
                        pass
                    dev_dispatch_str = f"Complete (Dispatched to GPU: {gpu_name})"
                    exec_backend_str = f"GPU ({gpu_name})"
                else:
                    target_cores = getattr(self, 'optimal_cores', None) or getattr(self, 'target_cores', None) or max(1, int((os.cpu_count() or 16) * 0.8))
                    dev_dispatch_str = f"Complete (Dispatched to CPU | {target_cores} Worker Threads Active)"
                    exec_backend_str = f"CPU ({target_cores} Threads)"
                
                # Check JIT precompile cache status
                import glob
                pycache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
                has_nbc = len(glob.glob(os.path.join(pycache_dir, '*.nbc'))) > 0
                jit_status = "Complete (Bytecode Cache Linked)" if has_nbc else "Complete (Foundation JIT Compiled)"
                
                # Check hardware calibration status
                cal_status = "Complete (Quantization & Pulse Maps Ready)" if is_ml_backend else "Complete (Standard Circuit Baseline)"
                
                # Check thread profiling bootstrapping status
                try:
                    from .profiler import SimulatorProfilerMixin
                    has_cached_threads = (4 in SimulatorProfilerMixin._cached_configs.get('Circuit', {}) or 
                                          4 in SimulatorProfilerMixin._cached_configs.get('Column', {}) or
                                          4 in SimulatorProfilerMixin._cached_configs.get('Device', {}))
                except Exception:
                    has_cached_threads = False
                thread_status = "Complete (Bootstrapped & Loaded from Cache)" if has_cached_threads else "Complete (Optimal Configs Bound)"

                banner_line = "-" * 62
                mid_line = f"--- MolmemSimulator: Crossbar Setup & Hardware Status ".ljust(59) + "---"
                print(f"\n{banner_line}\n{mid_line}\n{banner_line}")
                print(f"  > JIT Pre-compilation    : {jit_status}")
                print(f"  > Hardware Calibration   : {cal_status}")
                print(f"  > Thread Profiling (SOA) : {thread_status}")
                print(f"  > Device Dispatch        : {dev_dispatch_str}\n")
                MolmemSimulator._hardware_status_printed = True
            


    def _resolveNode(self, node=None, row=None, col=None):
        if getattr(self, '_is_noop', False):
            return 0
        """Helper to resolve a node ID from either direct node, row, or col."""
        if node is not None:
            return node
        elif row is not None:
            wl = self.wordlines.get(row)
            if wl is None:
                raise ValueError(f"Row {row} does not exist in the crossbar.")
            return wl
        elif col is not None:
            bl = self.bitlines.get(col)
            if bl is None:
                raise ValueError(f"Col {col} does not exist in the crossbar.")
            return bl
        raise ValueError("Must specify exactly one of: node, row, or col.")
            
    def _resolveDeviceIndex(self, topNode=None, bottomNode=None, row=None, col=None):
        if getattr(self, '_is_noop', False):
            return 0
        """Helper to resolve a device index from either (topNode, bottomNode) or (row, col)."""
        if type(topNode) is int and bottomNode is None and row is None and col is None:
            if 0 <= topNode < len(self.devices):
                return topNode
        if len(self.devices) == 1 and topNode is None and bottomNode is None and row is None and col is None:
            return 0
            
        hasNodes = topNode is not None and bottomNode is not None
        hasCoords = row is not None and col is not None
        
        if hasCoords and not hasNodes:
            dev_idx = self.crossbarDevices.get((row, col))
            if dev_idx is None:
                raise ValueError(f"No crossbar device found at Row {row}, Col {col}.")
            return dev_idx
        elif hasNodes and not hasCoords:
            if not hasattr(self, '_cached_node_pair_to_dev_idx') or self._cached_node_pair_to_dev_idx is None or len(self._cached_node_pair_to_dev_idx) != len(self.devices):
                self._cached_node_pair_to_dev_idx = {(d['top'], d['bottom']): idx for idx, d in enumerate(self.devices)}
            dev_idx = self._cached_node_pair_to_dev_idx.get((topNode, bottomNode))
            if dev_idx is not None:
                return dev_idx
            raise ValueError(f"No device found connected from node {topNode} to {bottomNode}.")
        else:
            raise ValueError("Must specify exactly one pair: (topNode, bottomNode) or (row, col).")

    def clearVoltageSources(self):
        """Removes all applied voltage/current sources and resets known nodes."""
        if getattr(self, '_is_noop', False):
            return
        if not self.vSources and not self.dynamicSources:
            return
        self._gpu_cache = None
        self._invalidate_hardware_tensor_cache()
        self._cached_torch_topology = None
        self._cached_cpu_topology = None
        self._cached_cpu_topology_key = None
        self.circuit_topology_modified = True
        self._soa_cached = False
        self._sources_cached = False
        self._cached_inference_sources = None
        # Un-mark all currently forced nodes (except Node 0 Ground)
        for src in self.vSources:
            if src.node is not None and src.node != 0:
                self.nodesKnown.discard(src.node)
                self.nodesUnknown.add(src.node)
        for src in self.dynamicSources:
            if src.node is not None and src.node != 0:
                self.nodesKnown.discard(src.node)
                self.nodesUnknown.add(src.node)
        self.vSources.clear()
        self.dynamicSources.clear()

    def addVoltageSource(self, node=None, sourceObj=None, row=None, col=None):
        """
        Defines an ideal voltage source referenced to Ground (node 0) using a Source object.
        """
        if self._is_noop:
            return
        self._gpu_cache = None
        self._cached_torch_topology = None
        self._cached_cpu_topology = None
        self._cached_cpu_topology_key = None
        self.circuit_topology_modified = True
        self._soa_cached = False
        self._sources_cached = False
        self._cached_inference_sources = None
        targetNode = node if (node is not None and row is None and col is None) else self._resolveNode(node, row, col)
        
        # Override the node property on the source object to bind it to our resolved ID
        sourceObj.node = targetNode
        self.vSources.append(sourceObj)
        
        self.nodesKnown.add(targetNode)
        self.nodesUnknown.discard(targetNode)
            
    def addDynamicSource(self, node=None, sourceObj=None, row=None, col=None):
        """
        Defines a non-ideal Norton Equivalent source that dynamically swaps to High-Z (floating) when Source.getVoltage(t) returns np.nan.
        """
        if self._is_noop:
            return
        self._gpu_cache = None
        self._cached_torch_topology = None
        self._cached_cpu_topology = None
        self._cached_cpu_topology_key = None
        self.circuit_topology_modified = True
        self._soa_cached = False
        self._sources_cached = False
        self._cached_inference_sources = None
        targetNode = node if (node is not None and row is None and col is None) else self._resolveNode(node, row, col)
        sourceObj.node = targetNode
        self.dynamicSources.append(sourceObj)
        if targetNode not in self.nodesKnown:
            self.nodesUnknown.add(targetNode)
            
    def addDcSource(self, node=None, voltage=0.0, row=None, col=None):
        if self._is_noop:
            return
        """
        Convenience method to add a constant DC voltage source (or Ground if 0.0V).
        """
        self.addVoltageSource(node=node, row=row, col=col, sourceObj=DcSource(None, voltage))
        
    def addPulseSource(self, node=None, amplitude=0.0, period=0.0, dutyCycle=0.5, delay=0.0, numPulses=None, row=None, col=None):
        if self._is_noop:
            return
        """
        Convenience method to add a dynamic pulse source using the Event Queue scheduler.
        """
        width = period * dutyCycle
        self.addVoltageSource(node=node, row=row, col=col, sourceObj=PulseSource(None, amplitude, period, width, delay, max_pulses=numPulses))
        
    def addPwlSource(self, node=None, t_points=(), v_points=(), row=None, col=None):
        if self._is_noop:
            return
        self.addVoltageSource(node=node, row=row, col=col, sourceObj=PwlSource(None, t_points, v_points))

    def addBSource(self, node=None, vFunc=None, sources=(), edge_points=(), row=None, col=None, source_intervals=None):
        if self._is_noop:
            return
        self.addVoltageSource(node=node, row=row, col=col, sourceObj=BSource(None, vFunc, sources, edge_points, source_intervals))
            
    def _registerNode(self, node):
        if node not in self.nodesKnown and node not in self.nodesUnknown:
            self.nodesUnknown.add(node)

    def _convert_bsources_to_pwl(self, sources, tEnd):
        if not any(isinstance(src, BSource) for src in sources):
            return list(sources)
        converted = []
        for src in sources:
            if isinstance(src, BSource):
                cached = getattr(src, '_cached_pwl_conv', None)
                if cached is not None and cached[0] == tEnd:
                    converted.append(cached[1])
                    continue

                ed = src.get_all_edges(0.0, tEnd)
                if isinstance(ed, np.ndarray):
                    schedule_arr = np.concatenate(([0.0], ed, [tEnd])) if len(ed) > 0 else np.array([0.0, tEnd], dtype=np.float64)
                else:
                    schedule_arr = np.array([0.0] + list(ed) + [tEnd], dtype=np.float64)
                t_points = np.unique(schedule_arr)

                # If no discrete edge triggers exist and no dependent sources were provided,
                # this BSource represents a continuous function (e.g. sinusoidal sweep, chirp).
                # Uniformly sample the continuous callable across [0, tEnd] to preserve waveform fidelity:
                if len(t_points) <= 2 and not getattr(src, 'sources', None) and not getattr(src, 'edge_points', None):
                    t_points = np.linspace(0.0, tEnd, 2001, dtype=np.float64)
                
                v_points = src.getVoltage(t_points)
                
                # Compress consecutive duplicate voltages for Zero-Order Hold (ZOH) representation safely
                if len(t_points) > 2:
                    keep = np.ones(len(t_points), dtype=bool)
                    # Keep point if either its left neighbor or right neighbor differs (preserves transition edges)
                    diff_max = np.maximum(np.abs(v_points[1:-1] - v_points[:-2]), np.abs(v_points[1:-1] - v_points[2:]))
                    keep[1:-1] = (diff_max > 1e-9)
                    t_points = t_points[keep]
                    v_points = v_points[keep]
                    
                pwl = PwlSource(node=src.node, t_points=t_points, v_points=v_points)
                src._cached_pwl_conv = (tEnd, pwl)
                converted.append(pwl)
            else:
                converted.append(src)
        return converted

    def _serialize_sources(self, sources, mapping_dict):
        if getattr(self, '_sources_cached', False) and getattr(self, '_last_serialized_sources', None) is not None:
            if getattr(self, '_last_serialized_sources_ref', None) is sources and getattr(self, '_last_serialized_mapping_ref', None) is mapping_dict:
                self._last_serialized_sources[7].fill(0)
                return self._last_serialized_sources

        src_signatures = tuple(
            src.get_signature() if hasattr(src, 'get_signature') else (src.__class__.__name__, id(src))
            for src in sources
        )
        node_map_tuple = tuple(mapping_dict.items()) if isinstance(mapping_dict, dict) else ()
        cache_key = ('_serialize_sources', src_signatures, node_map_tuple)
        cached = _GLOBAL_SOURCE_METADATA_CACHE.get(cache_key)
        if cached is not None:
            cached[7].fill(0)
            self._last_serialized_sources = cached
            self._last_serialized_sources_ref = sources
            self._last_serialized_mapping_ref = mapping_dict
            return cached

        num_sources = len(sources)
        v_types = np.zeros(num_sources, dtype=np.int32)
        v_nodes = np.zeros(num_sources, dtype=np.int32)
        v_dc_voltages = np.zeros(num_sources, dtype=np.float64)
        v_pulse_params = np.zeros((num_sources, 6), dtype=np.float64)
        v_pulse_params[:, 1] = 1.0  # Default period to 1.0 to prevent division by zero in unused JIT branches
        v_pulse_params[:, 4] = 1.0  # Default reciprocal period
        v_pulse_params[:, 5] = -1.0 # Default max_pulses to -1.0 (no limit)
        
        pwl_pts = [len(src.t_points) for src in sources if isinstance(src, PwlSource)]
        max_pwl_pts = max(pwl_pts) if pwl_pts else 1
        
        v_pwl_t_matrix = np.full((num_sources, max_pwl_pts), -1.0, dtype=np.float64)
        v_pwl_v_matrix = np.zeros((num_sources, max_pwl_pts), dtype=np.float64)
        v_pwl_mask = np.zeros(num_sources, dtype=bool)
        v_pwl_cache_idx = np.zeros(num_sources, dtype=np.int32)
        
        for i, src in enumerate(sources):
            v_nodes[i] = mapping_dict.get(src.node, -1)
            if isinstance(src, DcSource):
                v_types[i] = 0
                v_dc_voltages[i] = src.voltage
            elif isinstance(src, PulseSource):
                v_types[i] = 1
                v_pulse_params[i, :] = (src.amp, src.period, src.width, src.delay, src.inv_period, src.max_pulses if src.max_pulses is not None else -1.0)
            elif isinstance(src, PwlSource):
                v_types[i] = 2
                pts = len(src.t_points)
                v_pwl_t_matrix[i, :pts] = src.t_points
                v_pwl_v_matrix[i, :pts] = np.nan_to_num(src.v_points, nan=1e20, copy=False)
                v_pwl_mask[i] = True
            else:
                raise TypeError(f"Unknown source type: {type(src)}")
                
        res = (v_types, v_nodes, v_dc_voltages, v_pulse_params, 
               v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx)
        _GLOBAL_SOURCE_METADATA_CACHE[cache_key] = res
        self._last_serialized_sources = res
        self._last_serialized_sources_ref = sources
        self._last_serialized_mapping_ref = mapping_dict
        return res
            
    def _ensure_parallelization_profiled(self):
        if (not str(self.device).startswith('cuda') and 
            not getattr(self, 'use_parallel_optimized', False) and 
            not hasattr(self, 'optimal_cores')):
            import multiprocessing
            if multiprocessing.current_process().name == 'MainProcess' and not getattr(self.__class__, '_in_warmup', False):
                self.profileParallelization()


    def run(self, tEnd, min_dt=1e-12, tol=1e-3, maxIter=100, title="Hardware Simulation", appendHistory=False, recordHistory=True, recordStepTiming=False, stream=None, sync_states_back=True):
        if getattr(self, '_is_noop', False):
            return
            
        if not getattr(self.__class__, '_in_warmup', False) and not getattr(self, '_is_calibration_helper', False):
            from .sys_utils import ensure_bootstrap_profiled
            ensure_bootstrap_profiled(verbose=getattr(self, 'detailedPrint', True))
        self._ensure_parallelization_profiled()
            
        # Safeguard: Prevent child processes from executing simulations during multiprocessing imports on Windows
        import multiprocessing
        import os
        current_proc_name = multiprocessing.current_process().name
        is_bootstrapper = 'Bootstrapper' in current_proc_name or 'JIT_Worker' in current_proc_name
        if current_proc_name != 'MainProcess' and not is_bootstrapper:
            if os.environ.get('MOLMEM_ACTIVE_WORKER_TASK') != '1':
                import inspect
                frame = inspect.currentframe()
                is_top_level_import = False
                try:
                    caller = frame.f_back
                    while caller:
                        if caller.f_code.co_name == '<module>' and caller.f_globals.get('__name__') != '__main__':
                            is_top_level_import = True
                            break
                        caller = caller.f_back
                finally:
                    del frame
                if is_top_level_import:
                    return

        if hasattr(self, '_write_back_futures') and self._write_back_futures:
            concurrent.futures.wait(self._write_back_futures)
            self._write_back_futures.clear()
        # Check device: CUDA/GPU launches C++/HIP extension solver, CPU launches Numba dense solver
        if os.environ.get("MOLMEM_FORCE_CPU") == "1":
            device = 'cpu'
            self.device = 'cpu'
            is_gpu = False
        elif os.environ.get("MOLMEM_FORCE_GPU") == "1" and getattr(self, 'requested_device', None) != 'cpu' and getattr(self, 'device', None) != 'cpu':
            selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
            device = f'cuda:{selected_gpu}'
            self.device = device
            is_gpu = True
        else:
            req_dev = getattr(self, 'requested_device', None)
            device = getattr(self, 'device', 'cpu')
            backend = getattr(self, 'backend', 'numba')
            if req_dev == 'auto' and backend not in ['hybrid', 'pytorch', 'torch']:
                # Standard circuit transient simulations in auto mode: strictly CPU (Numba JIT)
                device = 'cpu'
                self.device = 'cpu'
                self.backend = 'numba'
                is_gpu = False
            elif backend == 'hybrid':
                # Heterogeneous Hybrid mode: GPU accelerated solver with CPU pacing
                from .sys_utils import is_gpu_ready
                if is_gpu_ready():
                    selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                    device = f'cuda:{selected_gpu}'
                    self.device = device
                    is_gpu = True
                else:
                    device = 'cpu'
                    self.device = 'cpu'
                    self.backend = 'numba'
                    is_gpu = False
            else:
                is_gpu = str(device).startswith('cuda') or device in ('gpu', 'cuda')

        # Enforce tolerance floor to prevent solver stalls/oscillations
        min_tol = 1e-9
        if tol < min_tol:
            tol = min_tol

        pid = os.getpid()
        if is_gpu:
            try:
                from .sys_utils import log_to_run_log_only
                if not getattr(self.__class__, '_in_warmup', False):
                    log_to_run_log_only(f"[GPU_EXEC] [PID {pid}] Launching GPU simulation: Device={self.device} | Backend={getattr(self, 'backend', 'pytorch')}")
            except Exception:
                pass

            from .torch_simulator import run_torch_simulation
            try:
                run_torch_simulation(self, tEnd, min_dt=min_dt, tol=tol, maxIter=maxIter, title=title, appendHistory=appendHistory, recordHistory=recordHistory, stream=stream, sync_states_back=sync_states_back)
                return
            except (KeyboardInterrupt, RuntimeError) as e:
                e_str = str(e)
                if isinstance(e, KeyboardInterrupt) or "KeyboardInterrupt" in e_str:
                    print("\n[!] Simulation interrupted by user. Terminating process...")
                    import os
                    os.kill(os.getpid(), 9)
                raise e

        try:
            from .sys_utils import log_to_run_log_only
            threads = getattr(self, 'optimal_cores', 1)
            solver_name = "Dense"
            log_to_run_log_only(f"[CPU_EXEC] [PID {pid}] Launching CPU simulation: Solver={solver_name} | Threads={threads} | Core Affinity=OS Managed")
        except Exception:
            pass

        try:
            return self._run_impl(tEnd, min_dt=min_dt, tol=tol, maxIter=maxIter, title=title, appendHistory=appendHistory, recordHistory=recordHistory, recordStepTiming=recordStepTiming, sync_states_back=sync_states_back)
        except (KeyboardInterrupt, RuntimeError) as e:
            e_str = str(e)
            if isinstance(e, KeyboardInterrupt) or "KeyboardInterrupt" in e_str:
                print("\n[!] Simulation interrupted by user. Terminating process...")
                try:
                    sys.exit(130)
                except SystemExit:
                    raise
            raise e

    @handle_oom
    def _run_impl(self, tEnd, min_dt=1e-12, tol=1e-3, maxIter=100, title="Hardware Simulation", appendHistory=False, recordHistory=True, recordStepTiming=False, sync_states_back=True):
        if getattr(self, '_is_noop', False):
            return 0
            
        from .sys_utils import resolve_record_history_flags
        _, _, _, _, _, _, bitmask = resolve_record_history_flags(recordHistory)
        recordHistory = bitmask
        """
        Runs the transient simulation using a Predictor-Corrector variable time-stepper.
        If appendHistory=False: runs from t=0 to tEnd (absolute).
        If appendHistory=True: runs from the current clock to tEnd (absolute), maintaining timeline continuity.
        dt dynamically scales up to the next Event Queue Breakpoint to maintain accuracy within 'tol'.
        """
        unkNodes = sorted(self.nodesUnknown)
        nodeToIdx = {n: i for i, n in enumerate(unkNodes)}
        numNodes = len(unkNodes)
        num_devices = len(self.devices)
        
        # Pre-flight host RAM check against calibrated theoretical memory footprint
        if num_devices > 0:
            try:
                import psutil
                rows = getattr(self, 'rows', int(np.sqrt(num_devices)))
                cols = getattr(self, 'cols', int(np.ceil(num_devices / max(1, rows))))
                updates_per_dev = getattr(self, 'UpdatesPer', 'Column') == 'Device'
                if self.backend in ['torch', 'pytorch', 'hybrid']:
                    est_ram_bytes = self.estimate_gpu_host_ram_bytes(rows, cols, is_device_basis=updates_per_dev)
                else:
                    est_ram_bytes = self.estimate_cpu_memory_bytes(rows, cols, is_device_basis=updates_per_dev)
                avail_ram_bytes = psutil.virtual_memory().available
                if est_ram_bytes > avail_ram_bytes * 0.95:
                    backend_label = "GPU/Hybrid" if self.backend in ['torch', 'pytorch', 'hybrid'] else "CPU"
                    raise MemoryError(
                        f"Insufficient System RAM: Simulating {rows}x{cols} crossbar ({num_devices} devices) on {backend_label} "
                        f"requires estimated {est_ram_bytes / (1024.0**3):.2f} GB Host RAM, but only "
                        f"{avail_ram_bytes / (1024.0**3):.2f} GB is currently available."
                    )
            except MemoryError:
                raise
            except Exception:
                pass

        target_cores = getattr(self, 'target_cores', getattr(self.__class__, 'target_cores', max(1, int((os.cpu_count() or 1) * 0.8))))
        
        if getattr(self, '_force_parallel', None) is not None:
            use_parallel = self._force_parallel
        elif getattr(self, 'use_parallel_optimized', None) is not None:
            use_parallel = self.use_parallel_optimized
            target_cores = getattr(self, 'optimal_cores', target_cores)
        else:
            # Fallback when ML API wasn't triggered
            use_parallel = (num_devices >= 128)
            
        if use_parallel:
            # Dynamically restart ThreadPoolExecutor if the user or profiler changes the optimal max_workers
            if self.executor is not None and getattr(self, 'active_executor_cores', None) != target_cores:
                try:
                    from .sys_utils import log_to_run_log_only
                    log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Terminated ThreadPoolExecutor: MolmemSimulator_Parallel | Workers={self.active_executor_cores} | Platform=CPU")
                except Exception:
                    pass
                self.executor.shutdown(wait=False)
                self.executor = None
            if self.executor is None:
                try:
                    from .sys_utils import log_to_run_log_only
                    log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Initialized ThreadPoolExecutor: MolmemSimulator_Parallel | Workers={target_cores} | Platform=CPU")
                except Exception:
                    pass
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=target_cores)
                self.active_executor_cores = target_cores
                
        # Critical execution parameter: Enforce active core threadcaps natively via Numba to prevent
        # uncontrolled CPU saturation on hardware-level simulations that lack the software-level profiler hooks.
        # Setup History Lists
        if not appendHistory or not hasattr(self, 'tArr') or self.tArr is None:
            self.iHistory = {}
            self.stateHistory = {}
            self.f22History = {}
            self.gHistory = {}
            self.vHistory = {}
            self.tArr = None
            self._hist_active_len = 0
            
        # Setup Dynamic Norton Sources
        num_dynamic_sources = len(self.dynamicSources)
        dynamic_node_idx = np.array([nodeToIdx.get(src.node, -1) for src in self.dynamicSources], dtype=np.int32)
        dynamic_G_arr = np.zeros(num_dynamic_sources, dtype=np.float64)
        dynamic_I_arr = np.zeros(num_dynamic_sources, dtype=np.float64)
        
        # Optimization 2: JIT Vectorized Dynamic PWL Tracking
        max_dyn_pwl_pts = max([len(src.t_points) for src in self.dynamicSources if hasattr(src, 't_points')] + [1])
        dyn_pwl_t_matrix = np.full((num_dynamic_sources, max_dyn_pwl_pts), -1.0)
        dyn_pwl_v_matrix = np.zeros((num_dynamic_sources, max_dyn_pwl_pts))
        dyn_pwl_mask = np.zeros(num_dynamic_sources, dtype=bool)
        
        dyn_cache_idx = np.zeros(num_dynamic_sources, dtype=np.int32)
        
        self.dev_to_dyn_src = np.full(num_devices, -1, dtype=np.int32)
        if num_dynamic_sources > 0:
            node_to_dyn_idx = {src.node: d_idx for d_idx, src in enumerate(self.dynamicSources)}
            for i, d in enumerate(self.devices):
                d_idx = node_to_dyn_idx.get(d['bottom'])
                if d_idx is not None:
                    self.dev_to_dyn_src[i] = d_idx
        
        for i, src in enumerate(self.dynamicSources):
            if hasattr(src, 't_points'):
                pts = len(src.t_points)
                dyn_pwl_t_matrix[i, :pts] = src.t_points
                dyn_pwl_v_matrix[i, :pts] = np.where(np.isnan(src.v_points), 1e20, src.v_points)
                dyn_pwl_mask[i] = True
                
        other_dyn_sources = [(i, src.getVoltage) for i, src in enumerate(self.dynamicSources) if not dyn_pwl_mask[i]]
            
        # Optimization 1: Pre-allocate Dense Math Arrays to prevent OS allocation overhead
        if numNodes > 0:
            gMat = np.zeros((numNodes, numNodes))
            iRes = np.zeros(numNodes)
        
        # Optimization 2: Flatten vCurrent dictionary to a NumPy array mapped by nodeToIdx
        sorted_unk = sorted(self.nodesUnknown)
        sorted_known = sorted(self.nodesKnown - {0})
        node_cache_key = (len(sorted_unk), len(sorted_known), tuple(sorted_unk), tuple(sorted_known))
        cached_node_entry = _GLOBAL_NODE_MAP_CACHE.get(node_cache_key)
        if cached_node_entry is not None:
            all_nodes_ordered, fullNodeToIdx = cached_node_entry
        else:
            all_nodes_ordered = sorted_unk + sorted_known
            fullNodeToIdx = {n: i for i, n in enumerate(all_nodes_ordered)}
            _GLOBAL_NODE_MAP_CACHE[node_cache_key] = (all_nodes_ordered, fullNodeToIdx)
        
        self._cached_all_nodes_ordered = all_nodes_ordered
        self._cached_fullNodeToIdx = fullNodeToIdx
        
        if getattr(self, '_sources_cached', False) and getattr(self, '_cached_vCurrent_base', None) is not None:
            vCurrent_arr = self._cached_vCurrent_base.copy()
        else:
            dc_sig = tuple((src.node, float(src.voltage)) for src in self.vSources if isinstance(src, DcSource))
            vcurrent_cache_key = (node_cache_key, dc_sig)
            cached_vcurrent = _GLOBAL_SOURCE_METADATA_CACHE.get(vcurrent_cache_key)
            if cached_vcurrent is not None:
                self._cached_vCurrent_base = cached_vcurrent
                vCurrent_arr = cached_vcurrent.copy()
            else:
                vCurrent_base = np.zeros(len(all_nodes_ordered))
                for src in self.vSources:
                    if isinstance(src, DcSource):
                        vCurrent_base[fullNodeToIdx[src.node]] = src.voltage
                _GLOBAL_SOURCE_METADATA_CACHE[vcurrent_cache_key] = vCurrent_base
                self._cached_vCurrent_base = vCurrent_base
                vCurrent_arr = vCurrent_base.copy()
        
        # Check cache validity for static arrays
        is_cached = getattr(self, '_soa_cached', False) and len(self._soa_cached_devices) == num_devices
        
        if is_cached:
            dev_top_idx = self._soa_dev_top_idx
            dev_bot_idx = self._soa_dev_bot_idx
            dev_top_full_idx = self._soa_dev_top_full_idx
            dev_bot_full_idx = self._soa_dev_bot_full_idx
            
            vSmooth_arr = self._soa_vSmooth_arr
            nSmooth_arr = self._soa_nSmooth_arr
            kappa_arr = self._soa_kappa_arr
            alpha_arr = self._soa_alpha_arr
            vTh_arr = self._soa_vTh_arr
            vBypass_arr = self._soa_vBypass_arr
            vRefPos_arr = self._soa_vRefPos_arr
            vRefNeg_arr = self._soa_vRefNeg_arr
            nMax_arr = self._soa_nMax_arr
            kDischarge_arr = self._soa_kDischarge_arr
            gScale_arr = self._soa_gScale_arr
            rC_arr = self._soa_rC_arr
            f22StartVal_arr = self._soa_f22StartVal_arr
            f22PeakVal_arr = self._soa_f22PeakVal_arr
            f22EndVal_arr = self._soa_f22EndVal_arr
            f22PotScaleInv_arr = self._soa_f22PotScaleInv_arr
            f22DepScaleInv_arr = self._soa_f22DepScaleInv_arr
            E_a_arr = self._soa_E_a_arr
            R_th_arr = self._soa_R_th_arr
            tau_th_arr = self._soa_tau_th_arr
            gamma_arr = self._soa_gamma_arr
            beta_alpha_arr = self._soa_beta_alpha_arr
            beta_s_arr = self._soa_beta_s_arr
            rateAsymmetry_arr = self._soa_rateAsymmetry_arr
            rTopBlend_arr = self._soa_rTopBlend_arr
            fDischarge_arr = self._soa_fDischarge_arr
            cDischarge_arr = self._soa_cDischarge_arr
            rBottomBlend_arr = self._soa_rBottomBlend_arr
            rLatency_arr = self._soa_rLatency_arr
            k_overdrive_arr = self._soa_k_overdrive_arr
            
            y_i = self._soa_y_i
            y_v = self._soa_y_v
            y_fp = self._soa_y_fp
            y_fd = self._soa_y_fd
        else:
            topo_key = (len(sorted_unk), len(sorted_known), tuple(all_nodes_ordered), num_devices)
            cached_topos = getattr(self, '_cached_cpu_topologies', None)
            if cached_topos is None:
                cached_topos = {}
                self._cached_cpu_topologies = cached_topos
                
            cached_topo = cached_topos.get(topo_key)
            if cached_topo is not None:
                dev_top_idx, dev_bot_idx, dev_top_full_idx, dev_bot_full_idx = cached_topo
            else:
                if num_devices == 1:
                    d0 = self.devices[0]
                    top = d0['top']
                    bot = d0['bottom']
                    dev_top_idx = np.array([nodeToIdx.get(top, -1)], dtype=np.int32)
                    dev_bot_idx = np.array([nodeToIdx.get(bot, -1)], dtype=np.int32)
                    dev_top_full_idx = np.array([fullNodeToIdx[top]], dtype=np.int32)
                    dev_bot_full_idx = np.array([fullNodeToIdx[bot]], dtype=np.int32)
                else:
                    top_arr = getattr(self, '_dev_top_nodes_arr', None)
                    bot_arr = getattr(self, '_dev_bot_nodes_arr', None)
                    if top_arr is None or len(top_arr) != num_devices:
                        top_arr = np.array([d['top'] for d in self.devices], dtype=np.int32)
                        bot_arr = np.array([d['bottom'] for d in self.devices], dtype=np.int32)
                        self._dev_top_nodes_arr = top_arr
                        self._dev_bot_nodes_arr = bot_arr
                    
                    max_node_val = max(max(all_nodes_ordered) if all_nodes_ordered else 0, int(top_arr.max() if len(top_arr) else 0), int(bot_arr.max() if len(bot_arr) else 0))
                    lut_nodeToIdx = np.full(max_node_val + 1, -1, dtype=np.int32)
                    if nodeToIdx:
                        n_keys = np.fromiter(nodeToIdx.keys(), dtype=np.int32, count=len(nodeToIdx))
                        n_vals = np.fromiter(nodeToIdx.values(), dtype=np.int32, count=len(nodeToIdx))
                        valid_mask = (n_keys >= 0) & (n_keys <= max_node_val)
                        lut_nodeToIdx[n_keys[valid_mask]] = n_vals[valid_mask]

                    lut_fullNodeToIdx = np.full(max_node_val + 1, -1, dtype=np.int32)
                    if fullNodeToIdx:
                        fn_keys = np.fromiter(fullNodeToIdx.keys(), dtype=np.int32, count=len(fullNodeToIdx))
                        fn_vals = np.fromiter(fullNodeToIdx.values(), dtype=np.int32, count=len(fullNodeToIdx))
                        valid_fn_mask = (fn_keys >= 0) & (fn_keys <= max_node_val)
                        lut_fullNodeToIdx[fn_keys[valid_fn_mask]] = fn_vals[valid_fn_mask]
                    
                    dev_top_idx = lut_nodeToIdx[top_arr]
                    dev_bot_idx = lut_nodeToIdx[bot_arr]
                    dev_top_full_idx = lut_fullNodeToIdx[top_arr]
                    dev_bot_full_idx = lut_fullNodeToIdx[bot_arr]
                
                cached_topos[topo_key] = (dev_top_idx, dev_bot_idx, dev_top_full_idx, dev_bot_full_idx)
                self._cached_cpu_topology = (dev_top_idx, dev_bot_idx, dev_top_full_idx, dev_bot_full_idx)
                self._cached_cpu_topology_key = topo_key
                self.circuit_topology_modified = False
            
            if num_devices > 0:
                first_dev = self.devices[0]['dev']
                all_same = getattr(self, '_all_dev_params_same', None)
                if all_same is None:
                    all_same = True
                    first_params = first_dev.params
                    for d in self.devices:
                        if d['dev'].params is not first_params:
                            all_same = False
                            break
                    self._all_dev_params_same = all_same
                
                if all_same:
                    vSmooth_arr = np.full(num_devices, first_dev.vSmooth, dtype=np.float64)
                    nSmooth_arr = np.full(num_devices, first_dev.nSmooth, dtype=np.float64)
                    kappa_arr = np.full(num_devices, first_dev.kappa, dtype=np.float64)
                    alpha_arr = np.full(num_devices, first_dev.alpha, dtype=np.float64)
                    vTh_arr = np.full(num_devices, first_dev.vTh, dtype=np.float64)
                    vBypass_arr = np.full(num_devices, first_dev.vBypass, dtype=np.float64)
                    vRefPos_arr = np.full(num_devices, first_dev.vRefPos, dtype=np.float64)
                    vRefNeg_arr = np.full(num_devices, first_dev.vRefNeg, dtype=np.float64)
                    nMax_arr = np.full(num_devices, first_dev.nMax, dtype=np.float64)
                    kDischarge_arr = np.full(num_devices, first_dev.kDischarge, dtype=np.float64)
                    gScale_arr = np.full(num_devices, first_dev.gScale, dtype=np.float64)
                    rC_arr = np.full(num_devices, first_dev.rC, dtype=np.float64)
                    f22StartVal_arr = np.full(num_devices, first_dev.f22StartVal, dtype=np.float64)
                    f22PeakVal_arr = np.full(num_devices, first_dev.f22PeakVal, dtype=np.float64)
                    f22EndVal_arr = np.full(num_devices, first_dev.f22EndVal, dtype=np.float64)
                    f22PotScaleInv_arr = np.full(num_devices, first_dev.f22PotScaleInv, dtype=np.float64)
                    f22DepScaleInv_arr = np.full(num_devices, first_dev.f22DepScaleInv, dtype=np.float64)
                    E_a_arr = np.full(num_devices, first_dev.E_a, dtype=np.float64)
                    R_th_arr = np.full(num_devices, first_dev.R_th, dtype=np.float64)
                    tau_th_arr = np.full(num_devices, first_dev.tau_th, dtype=np.float64)
                    gamma_arr = np.full(num_devices, first_dev.gamma, dtype=np.float64)
                    beta_alpha_arr = np.full(num_devices, first_dev.beta_alpha, dtype=np.float64)
                    beta_s_arr = np.full(num_devices, first_dev.beta_s, dtype=np.float64)
                    rateAsymmetry_arr = np.full(num_devices, first_dev.rateAsymmetry, dtype=np.float64)
                    rTopBlend_arr = np.full(num_devices, getattr(first_dev, 'rTopBlend', 0.050000), dtype=np.float64)
                    fDischarge_arr = np.full(num_devices, first_dev.fDischarge, dtype=np.float64)
                    cDischarge_arr = np.full(num_devices, getattr(first_dev, 'cDischarge', 0.0), dtype=np.float64)
                    rBottomBlend_arr = np.full(num_devices, getattr(first_dev, 'rBottomBlend', 0.714488), dtype=np.float64)
                    rLatency_arr = np.full(num_devices, getattr(first_dev, 'rLatency', 0.878506), dtype=np.float64)
                    k_overdrive_arr = np.full(num_devices, getattr(first_dev, 'k_overdrive', 7.296635), dtype=np.float64)
                else:
                    dev_objs = [d['dev'] for d in self.devices]
                    vSmooth_arr = np.ascontiguousarray([d.vSmooth for d in dev_objs], dtype=np.float64)
                    nSmooth_arr = np.ascontiguousarray([d.nSmooth for d in dev_objs], dtype=np.float64)
                    kappa_arr = np.ascontiguousarray([d.kappa for d in dev_objs], dtype=np.float64)
                    alpha_arr = np.ascontiguousarray([d.alpha for d in dev_objs], dtype=np.float64)
                    vTh_arr = np.ascontiguousarray([d.vTh for d in dev_objs], dtype=np.float64)
                    vBypass_arr = np.ascontiguousarray([d.vBypass for d in dev_objs], dtype=np.float64)
                    vRefPos_arr = np.ascontiguousarray([d.vRefPos for d in dev_objs], dtype=np.float64)
                    vRefNeg_arr = np.ascontiguousarray([d.vRefNeg for d in dev_objs], dtype=np.float64)
                    nMax_arr = np.ascontiguousarray([d.nMax for d in dev_objs], dtype=np.float64)
                    kDischarge_arr = np.ascontiguousarray([d.kDischarge for d in dev_objs], dtype=np.float64)
                    gScale_arr = np.ascontiguousarray([d.gScale for d in dev_objs], dtype=np.float64)
                    rC_arr = np.ascontiguousarray([d.rC for d in dev_objs], dtype=np.float64)
                    f22StartVal_arr = np.ascontiguousarray([d.f22StartVal for d in dev_objs], dtype=np.float64)
                    f22PeakVal_arr = np.ascontiguousarray([d.f22PeakVal for d in dev_objs], dtype=np.float64)
                    f22EndVal_arr = np.ascontiguousarray([d.f22EndVal for d in dev_objs], dtype=np.float64)
                    f22PotScaleInv_arr = np.ascontiguousarray([d.f22PotScaleInv for d in dev_objs], dtype=np.float64)
                    f22DepScaleInv_arr = np.ascontiguousarray([d.f22DepScaleInv for d in dev_objs], dtype=np.float64)
                    E_a_arr = np.ascontiguousarray([d.E_a for d in dev_objs], dtype=np.float64)
                    R_th_arr = np.ascontiguousarray([d.R_th for d in dev_objs], dtype=np.float64)
                    tau_th_arr = np.ascontiguousarray([d.tau_th for d in dev_objs], dtype=np.float64)
                    gamma_arr = np.ascontiguousarray([d.gamma for d in dev_objs], dtype=np.float64)
                    beta_alpha_arr = np.ascontiguousarray([d.beta_alpha for d in dev_objs], dtype=np.float64)
                    beta_s_arr = np.ascontiguousarray([d.beta_s for d in dev_objs], dtype=np.float64)
                    rateAsymmetry_arr = np.ascontiguousarray([d.rateAsymmetry for d in dev_objs], dtype=np.float64)
                    rTopBlend_arr = np.ascontiguousarray([getattr(d, 'rTopBlend', 0.050000) for d in dev_objs], dtype=np.float64)
                    fDischarge_arr = np.ascontiguousarray([d.fDischarge for d in dev_objs], dtype=np.float64)
                    cDischarge_arr = np.ascontiguousarray([getattr(d, 'cDischarge', 0.0) for d in dev_objs], dtype=np.float64)
                    rBottomBlend_arr = np.ascontiguousarray([getattr(d, 'rBottomBlend', 0.714488) for d in dev_objs], dtype=np.float64)
                    rLatency_arr = np.ascontiguousarray([getattr(d, 'rLatency', 0.878506) for d in dev_objs], dtype=np.float64)
                    k_overdrive_arr = np.ascontiguousarray([getattr(d, 'k_overdrive', 7.296635) for d in dev_objs], dtype=np.float64)
                
                dev0 = self.devices[0]['dev']
                y_i = dev0._y_i
                y_v = dev0._y_v
                y_fp = dev0._y_fp
                y_fd = dev0._y_fd
                
                # Cache them
                self._soa_vSmooth_arr = vSmooth_arr
                self._soa_nSmooth_arr = nSmooth_arr
                self._soa_kappa_arr = kappa_arr
                self._soa_alpha_arr = alpha_arr
                self._soa_vTh_arr = vTh_arr
                self._soa_vBypass_arr = vBypass_arr
                self._soa_vRefPos_arr = vRefPos_arr
                self._soa_vRefNeg_arr = vRefNeg_arr
                self._soa_nMax_arr = nMax_arr
                self._soa_kDischarge_arr = kDischarge_arr
                self._soa_gScale_arr = gScale_arr
                self._soa_rC_arr = rC_arr
                self._soa_f22StartVal_arr = f22StartVal_arr
                self._soa_f22PeakVal_arr = f22PeakVal_arr
                self._soa_f22EndVal_arr = f22EndVal_arr
                self._soa_f22PotScaleInv_arr = f22PotScaleInv_arr
                self._soa_f22DepScaleInv_arr = f22DepScaleInv_arr
                self._soa_E_a_arr = E_a_arr
                self._soa_R_th_arr = R_th_arr
                self._soa_tau_th_arr = tau_th_arr
                self._soa_gamma_arr = gamma_arr
                self._soa_beta_alpha_arr = beta_alpha_arr
                self._soa_beta_s_arr = beta_s_arr
                self._soa_rateAsymmetry_arr = rateAsymmetry_arr
                self._soa_rTopBlend_arr = rTopBlend_arr
                self._soa_fDischarge_arr = fDischarge_arr
                self._soa_cDischarge_arr = cDischarge_arr
                self._soa_rBottomBlend_arr = rBottomBlend_arr
                self._soa_rLatency_arr = rLatency_arr
                self._soa_k_overdrive_arr = k_overdrive_arr
                self._soa_y_i = y_i
                self._soa_y_v = y_v
                self._soa_y_fp = y_fp
                self._soa_y_fd = y_fd
                # Cache topology indices
                self._soa_dev_top_idx = dev_top_idx
                self._soa_dev_bot_idx = dev_bot_idx
                self._soa_dev_top_full_idx = dev_top_full_idx
                self._soa_dev_bot_full_idx = dev_bot_full_idx
                self._soa_cached_devices = list(self.devices)
                self._soa_cached = True
            else:
                self._soa_dev_top_idx = dev_top_idx
                self._soa_dev_bot_idx = dev_bot_idx
                self._soa_dev_top_full_idx = dev_top_full_idx
                self._soa_dev_bot_full_idx = dev_bot_full_idx
                y_i=y_v=y_fp=y_fd = np.zeros(1)
                
        gEq_arr = np.zeros(num_devices)
        iDev_arr = np.zeros(num_devices)

        # Optimization 3: Master SOA (Structure-Of-Arrays) Architecture
        if num_devices > 0:
            n_arr = np.zeros(num_devices, dtype=np.float64)
            modeState_arr = np.zeros(num_devices, dtype=np.float64)
            scaleFactor_arr = np.zeros(num_devices, dtype=np.float64)
            f22DepScaled_arr = np.zeros(num_devices, dtype=np.float64)
            f22PotScaled_arr = np.zeros(num_devices, dtype=np.float64)
            _cached_n_int_arr = np.zeros(num_devices, dtype=np.int32)
            iScale_arr = np.zeros(num_devices, dtype=np.float64)
            vScale_arr = np.zeros(num_devices, dtype=np.float64)
            currentF22_arr = np.zeros(num_devices, dtype=np.float64)
            currentIScale_arr = np.zeros(num_devices, dtype=np.float64)
            currentVScale_arr = np.zeros(num_devices, dtype=np.float64)
            T_arr = np.zeros(num_devices, dtype=np.float64)
            
            self.sync_to_cpu()
            if num_devices == 1:
                dev = self.devices[0]['dev']
                n_arr[0] = dev._n
                modeState_arr[0] = dev._modeState
                scaleFactor_arr[0] = dev._scaleFactor
                f22DepScaled_arr[0] = dev._f22DepScaled
                f22PotScaled_arr[0] = dev._f22PotScaled
                val = dev._cached_n_int_val
                if type(val) is int:
                    _cached_n_int_arr[0] = val
                elif isinstance(val, (int, np.integer)):
                    _cached_n_int_arr[0] = int(val)
                elif isinstance(val, (float, np.floating)) and math.isfinite(val):
                    _cached_n_int_arr[0] = int(val)
                else:
                    _cached_n_int_arr[0] = -1
                iScale_arr[0] = dev._iScale
                vScale_arr[0] = dev._vScale
                currentF22_arr[0] = dev._currentF22
                currentIScale_arr[0] = dev._currentIScale
                currentVScale_arr[0] = dev._currentVScale
                T_arr[0] = dev._T
            else:
                cache = getattr(self, '_gpu_states_cache', None)
                if self.backend in ['torch', 'pytorch', 'hybrid'] and cache is not None and not getattr(self, '_gpu_states_dirty', False) and isinstance(cache, dict) and 'n_arr' in cache and len(cache['n_arr']) == num_devices:
                    np.copyto(n_arr, cache['n_arr'])
                    np.copyto(modeState_arr, cache['modeState_arr'])
                    np.copyto(scaleFactor_arr, cache['scaleFactor_arr'])
                    np.copyto(f22DepScaled_arr, cache['f22DepScaled_arr'])
                    np.copyto(f22PotScaled_arr, cache['f22PotScaled_arr'])
                    np.copyto(_cached_n_int_arr, cache['_cached_n_int_arr'])
                    np.copyto(iScale_arr, cache['iScale_arr'])
                    np.copyto(vScale_arr, cache['vScale_arr'])
                    np.copyto(currentF22_arr, cache['currentF22_arr'])
                    np.copyto(currentIScale_arr, cache['currentIScale_arr'])
                    np.copyto(currentVScale_arr, cache['currentVScale_arr'])
                    np.copyto(T_arr, cache['T_arr'])
                else:
                    if not hasattr(self, '_dev_objs_flat') or self._dev_objs_flat is None or len(self._dev_objs_flat) != num_devices:
                        self._dev_objs_flat = [d['dev'] for d in self.devices]
                    dev_objs = self._dev_objs_flat
                    
                    for i, dev in enumerate(dev_objs):
                        n_arr[i] = dev._n
                        modeState_arr[i] = dev._modeState
                        scaleFactor_arr[i] = dev._scaleFactor
                        f22DepScaled_arr[i] = dev._f22DepScaled
                        f22PotScaled_arr[i] = dev._f22PotScaled
                        _cached_n_int_arr[i] = dev._cached_n_int_val
                        iScale_arr[i] = dev._iScale
                        vScale_arr[i] = dev._vScale
                        currentF22_arr[i] = dev._currentF22
                        currentIScale_arr[i] = dev._currentIScale
                        currentVScale_arr[i] = dev._currentVScale
                        T_arr[i] = dev._T
                        
                    self._gpu_states_cache = {
                        'n_arr': n_arr.copy(),
                        'modeState_arr': modeState_arr.copy(),
                        'scaleFactor_arr': scaleFactor_arr.copy(),
                        'T_arr': T_arr.copy(),
                        'currentF22_arr': currentF22_arr.copy(),
                        '_cached_n_int_arr': _cached_n_int_arr.copy(),
                        'iScale_arr': iScale_arr.copy(),
                        'vScale_arr': vScale_arr.copy(),
                        'currentIScale_arr': currentIScale_arr.copy(),
                        'currentVScale_arr': currentVScale_arr.copy(),
                        'f22DepScaled_arr': f22DepScaled_arr.copy(),
                        'f22PotScaled_arr': f22PotScaled_arr.copy(),
                    }
                    self._gpu_states_dirty = False
                    self._gpu_states_in_sync = True
            
            next_states = np.zeros((num_devices, 4), dtype=np.float64)
            err_arr = np.zeros(num_devices, dtype=np.float64)

        hist_idx = 0
        
        # Optimization 5: PWL Fast-path batch processing inside Numba
        v_sources = self.vSources
        num_v_sources = len(v_sources)
        if getattr(self, '_sources_cached', False) and getattr(self, '_cached_src_signatures', None) is not None:
            all_sources = self._cached_all_sources
            src_signatures = self._cached_src_signatures
            fast_v_sources_idx = self._cached_fast_v_sources_idx
            pwl_t_matrix = self._cached_pwl_t_matrix
            pwl_v_matrix = self._cached_pwl_v_matrix
            pwl_mask = self._cached_pwl_mask
            other_v_sources = self._cached_other_v_sources
        else:
            if num_v_sources == 0:
                fast_v_sources_idx = np.empty(0, dtype=np.int32)
                pwl_t_matrix = np.full((0, 1), -1.0, dtype=np.float64)
                pwl_v_matrix = np.zeros((0, 1), dtype=np.float64)
                pwl_mask = np.zeros(0, dtype=bool)
            all_sources = v_sources + self.dynamicSources
            src_signatures = tuple(
                src.get_signature() if hasattr(src, 'get_signature') else (src.__class__.__name__, id(src))
                for src in all_sources
            )
            
            src_meta_key = (src_signatures, tuple(fast_v_idx_tuple if (fast_v_idx_tuple := tuple(fullNodeToIdx[s.node] for s in v_sources)) else ()))
            cached_src_meta = _GLOBAL_SOURCE_METADATA_CACHE.get(src_meta_key)
            if cached_src_meta is not None:
                fast_v_sources_idx, pwl_t_matrix, pwl_v_matrix, pwl_mask, other_v_sources = cached_src_meta
            else:
                if num_v_sources == 0:
                    fast_v_sources_idx = np.empty(0, dtype=np.int32)
                    pwl_t_matrix = np.empty((0, 1), dtype=np.float64)
                    pwl_v_matrix = np.empty((0, 1), dtype=np.float64)
                    pwl_mask = np.empty(0, dtype=bool)
                    other_v_sources = []
                elif num_v_sources == 1:
                    src = v_sources[0]
                    fast_v_sources_idx = np.array([fullNodeToIdx[src.node]], dtype=np.int32)
                    t_pts = getattr(src, 't_points', None)
                    if t_pts is not None:
                        pts = len(t_pts)
                        pwl_t_matrix = np.empty((1, pts), dtype=np.float64)
                        pwl_t_matrix[0, :] = t_pts
                        pwl_v_matrix = np.empty((1, pts), dtype=np.float64)
                        pwl_v_matrix[0, :] = src.v_points
                        pwl_mask = np.ones(1, dtype=bool)
                        other_v_sources = []
                    else:
                        pwl_t_matrix = np.full((1, 1), -1.0, dtype=np.float64)
                        pwl_v_matrix = np.zeros((1, 1), dtype=np.float64)
                        pwl_mask = np.zeros(1, dtype=bool)
                        other_v_sources = [] if isinstance(src, DcSource) else [(0, src.getVoltage)]
                else:
                    fast_v_sources_idx = np.array([fullNodeToIdx[src.node] for src in v_sources], dtype=np.int32)
                    pwl_sources_info = [(i, src, t_pts) for i, src in enumerate(v_sources) if (t_pts := getattr(src, 't_points', None)) is not None]
                    if not pwl_sources_info:
                        pwl_t_matrix = np.full((num_v_sources, 1), -1.0, dtype=np.float64)
                        pwl_v_matrix = np.zeros((num_v_sources, 1), dtype=np.float64)
                        pwl_mask = np.zeros(num_v_sources, dtype=bool)
                    else:
                        max_pwl_pts = max(len(t) for _, _, t in pwl_sources_info)
                        pwl_t_matrix = np.full((num_v_sources, max_pwl_pts), -1.0, dtype=np.float64)
                        pwl_v_matrix = np.zeros((num_v_sources, max_pwl_pts), dtype=np.float64)
                        pwl_mask = np.zeros(num_v_sources, dtype=bool)
                        for i, src, t_pts in pwl_sources_info:
                            pts = len(t_pts)
                            pwl_t_matrix[i, :pts] = t_pts
                            pwl_v_matrix[i, :pts] = src.v_points
                            pwl_mask[i] = True
                    other_v_sources = [(i, src.getVoltage) for i, src in enumerate(v_sources) if not pwl_mask[i] and not isinstance(src, DcSource)]
                _GLOBAL_SOURCE_METADATA_CACHE[src_meta_key] = (fast_v_sources_idx, pwl_t_matrix, pwl_v_matrix, pwl_mask, other_v_sources)
            
            self._cached_all_sources = all_sources
            self._cached_src_signatures = src_signatures
            self._cached_fast_v_sources_idx = fast_v_sources_idx
            self._cached_pwl_t_matrix = pwl_t_matrix
            self._cached_pwl_v_matrix = pwl_v_matrix
            self._cached_pwl_mask = pwl_mask
            self._cached_other_v_sources = other_v_sources
            self._sources_cached = True
        
        pwl_cache_idx = np.zeros(num_v_sources, dtype=np.int32)
        
        # Optimization 6: Input-Driven Pre-Analyzed Fixed-Grid Architecture
        cache_key = (float(tEnd), len(all_sources), src_signatures)
        cached_info = _GLOBAL_SCHEDULE_CACHE.get(cache_key)
        if cached_info is not None:
            unique_edges, master_schedule, is_subsample = cached_info
        else:
            edge_arrays = [np.array([0.0, tEnd], dtype=np.float64)]
            for src in all_sources:
                if isinstance(src, BaseSource):
                    ed = src.get_all_edges(0.0, tEnd)
                    if isinstance(ed, np.ndarray) and len(ed) > 0:
                        edge_arrays.append(ed)
                    elif isinstance(ed, (list, tuple)) and len(ed) > 0:
                        edge_arrays.append(np.array(ed, dtype=np.float64))
            
            raw_edges = edge_arrays[0] if len(edge_arrays) == 1 else np.concatenate(edge_arrays)
            unique_edges = np.unique(raw_edges)
            
            sub_dt = 10e-9  # Fixed 10ns microstepping during active pulses
            inv_sub_dt = 1.0 / sub_dt
            sub_dt_threshold = sub_dt * 1.5

            if len(unique_edges) > 1:
                t_starts = unique_edges[:-1]
                t_ends = unique_edges[1:]
                dt_segs = t_ends - t_starts
                t_centers = t_starts + 0.5 * dt_segs
                
                # Vectorized evaluation across all sources
                is_active = np.zeros(len(t_centers), dtype=bool)
                for src in all_sources:
                    try:
                        v_test = src.getVoltage(t_centers)
                        if isinstance(v_test, np.ndarray):
                            is_active |= (np.abs(v_test) > 1e-6)
                        elif abs(v_test) > 1e-6:
                            is_active[:] = True
                            break
                    except Exception:
                        pass

                from .solver import jit_build_full_master_schedule
                master_schedule, is_subsample = jit_build_full_master_schedule(
                    unique_edges, is_active, sub_dt_threshold, inv_sub_dt
                )
            else:
                master_schedule = unique_edges.copy()
                is_subsample = np.zeros(len(master_schedule), dtype=bool)

            _GLOBAL_SCHEDULE_CACHE[cache_key] = (unique_edges, master_schedule, is_subsample)

        self._sources_cached = True

        # Optimization 12: Dynamic Footprint Initialization Shrinking Memory Demands
        is_cal_helper = getattr(self, '_is_calibration_helper', False)
        if is_cal_helper:
            cal_cap = max(20000, len(master_schedule) * 10)
            tArr_buffers = [np.zeros(cal_cap, dtype=np.float64), np.zeros(cal_cap, dtype=np.float64)]
            stepTimeHist_buffers = [np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)]
            vHist_buffers = [np.zeros((len(all_nodes_ordered), 2), dtype=np.float64, order='F'), np.zeros((len(all_nodes_ordered), 2), dtype=np.float64, order='F')]
            iHist_buffers = [np.zeros((num_devices, 2), dtype=np.float64, order='F'), np.zeros((num_devices, 2), dtype=np.float64, order='F')]
            stateHist_buffers = [np.zeros((num_devices, cal_cap), dtype=np.float64, order='F'), np.zeros((num_devices, cal_cap), dtype=np.float64, order='F')]
            f22Hist_buffers = [np.zeros((num_devices, 2), dtype=np.float64, order='F'), np.zeros((num_devices, 2), dtype=np.float64, order='F')]
            gHist_buffers = [np.zeros((num_devices, 2), dtype=np.float64, order='F'), np.zeros((num_devices, 2), dtype=np.float64, order='F')]
            tempHist_buffers = [np.zeros((num_devices, 2), dtype=np.float64, order='F'), np.zeros((num_devices, 2), dtype=np.float64, order='F')]
        elif recordHistory:
            from .sys_utils import get_cpu_cache_budget_mb
            _, ping_pong_mb, _ = get_cpu_cache_budget_mb()
            active_threads = max(1, getattr(self, 'optimal_cores', 1))
            target_bytes = (ping_pong_mb * 1024 * 1024) / active_threads
            bytes_per_step = max(8, (1 + (len(all_nodes_ordered) if (recordHistory & 1) else 0) + (num_devices * 5 if num_devices > 0 else 0)) * 8)
            expected_steps = max(len(master_schedule) * 50 + 2000, int(len(master_schedule) * 1.5))
            target_steps = int(target_bytes / bytes_per_step)
            hist_capacity = min(max(1000, target_steps), expected_steps) if expected_steps >= 1000 else max(10, expected_steps)
            
            # Double-Buffering Ping-Pong Buffer Sets (Set 0 and Set 1) from Thread-Local Pool
            worker_id = getattr(self, '_worker_id', 0)
            worker_bufs = ThreadPoolPingPongManager.get_instance().get_worker_buffers(
                worker_id=worker_id,
                num_nodes=len(all_nodes_ordered),
                num_devices=num_devices,
                hist_capacity=hist_capacity,
                recordHistory=recordHistory,
                recordStepTiming=recordStepTiming
            )
            tArr_buffers = worker_bufs['tArr_buffers']
            vHist_buffers = worker_bufs['vHist_buffers']
            stepTimeHist_buffers = worker_bufs['stepTimeHist_buffers']
            iHist_buffers = worker_bufs['iHist_buffers']
            stateHist_buffers = worker_bufs['stateHist_buffers']
            f22Hist_buffers = worker_bufs['f22Hist_buffers']
            gHist_buffers = worker_bufs['gHist_buffers']
            tempHist_buffers = worker_bufs['tempHist_buffers']
        else:
            hist_capacity = 2
            worker_id = getattr(self, '_worker_id', 0)
            worker_bufs = ThreadPoolPingPongManager.get_instance().get_worker_buffers(
                worker_id=worker_id,
                num_nodes=len(all_nodes_ordered),
                num_devices=num_devices,
                hist_capacity=hist_capacity,
                recordHistory=recordHistory,
                recordStepTiming=recordStepTiming
            )
            tArr_buffers = worker_bufs['tArr_buffers']
            vHist_buffers = worker_bufs['vHist_buffers']
            stepTimeHist_buffers = worker_bufs['stepTimeHist_buffers']
            iHist_buffers = worker_bufs['iHist_buffers']
            stateHist_buffers = worker_bufs['stateHist_buffers']
            f22Hist_buffers = worker_bufs['f22Hist_buffers']
            gHist_buffers = worker_bufs['gHist_buffers']
            tempHist_buffers = worker_bufs['tempHist_buffers']


        # Load QPF frequency value
        if sys.platform == 'win32' and qpf is not None:
            freq = ctypes.c_int64(0)
            qpf(ctypes.byref(freq))
            freq_val = float(freq.value) if freq.value > 0 else 1.0
        else:
            freq_val = 1.0
        buf_idx = 0
        schedule_idx = 0
        
        detailed = getattr(self, 'detailedPrint', True)
        verbose = True
        is_profiling = (getattr(self.__class__, '_in_bootstrap_profiling', False) or 
                        getattr(self.__class__, '_in_gpu_crossover_profiling', False) or
                        getattr(self.__class__, '_in_parallel_profiling', False) or
                        getattr(self.__class__, '_in_warmup', False) or
                        getattr(self, '_is_calibration_helper', False) or
                        getattr(self, '_profiler_active', False) or
                        multiprocessing.current_process().name != 'MainProcess')
        if is_profiling:
            verbose = False
            detailed = False
        if verbose and detailed and title:
            # Resolve dimensions/layout of the crossbar
            if hasattr(self, 'crossbarDevices') and self.crossbarDevices:
                max_r = 0
                max_c = 0
                for r, c in self.crossbarDevices.keys():
                    if r > max_r: max_r = r
                    if c > max_c: max_c = c
                layout_str = f" ({max_r}x{max_c})"
            else:
                layout_str = ""
                
            # Resolve hardware binding
            binding_str = "Cores OS Managed"
                
            # Resolve backend / device name
            dev_str = "CPU"
            device_name = str(getattr(self, 'device', 'cpu')).lower()
            if 'cuda' in device_name or 'hip' in device_name:
                dev_str = "GPU (ROCm)" if 'hip' in device_name or getattr(self, 'compile_backend', False) else "GPU (CUDA)"
                binding_str = "GPU Device Bound"
                
            banner_text = f"MolmemSimulator: {title}"
            banner_line = "-" * 62
            mid_line = f"--- {banner_text} ".ljust(59) + "---"
            
            print(f"\n{banner_line}\n{mid_line}\n{banner_line}")
            print(f"  > Target Array          : {num_devices} Devices{layout_str}")
            print(f"  > Solver Topology       : Dense Solver (Nodes: {numNodes} Unknown, {len(self.nodesKnown)} Known)")
            print(f"  > Executed Backend      : {dev_str} ({target_cores} Threads)\n")
        tStart = time.time()
        
        t = 0.0
        # When appending, continue from the current simulation clock
        if appendHistory and self.tArr is not None and len(self.tArr) > 0:
            t = self.tArr[-1]
        
        is_1t1r = getattr(self, 'architecture', '1R') == '1T1R'
        leakage_g = getattr(self, 'leakage_g', 1e-10)
        print_interval = tEnd * 0.01
        
        # Initialize default dt
        dt = 100e-9
        last_print_t = t

        # Monolithic JIT Loop Optimization Integration
        # If num_devices is 0, initialize all state/parameters arrays to empty to ensure Numba JIT typing correctness
        if num_devices == 0:
            n_arr = np.zeros(0, dtype=np.float64)
            modeState_arr = np.zeros(0, dtype=np.float64)
            scaleFactor_arr = np.zeros(0, dtype=np.float64)
            f22DepScaled_arr = np.zeros(0, dtype=np.float64)
            f22PotScaled_arr = np.zeros(0, dtype=np.float64)
            _cached_n_int_arr = np.zeros(0, dtype=np.int32)
            iScale_arr = np.zeros(0, dtype=np.float64)
            vScale_arr = np.zeros(0, dtype=np.float64)
            currentF22_arr = np.zeros(0, dtype=np.float64)
            currentIScale_arr = np.zeros(0, dtype=np.float64)
            currentVScale_arr = np.zeros(0, dtype=np.float64)
            T_arr = np.zeros(0, dtype=np.float64)
            next_states = np.zeros((0, 4), dtype=np.float64)
            err_arr = np.zeros(0, dtype=np.float64)
            
            vSmooth_arr = np.zeros(0, dtype=np.float64)
            nSmooth_arr = np.zeros(0, dtype=np.float64)
            kappa_arr = np.zeros(0, dtype=np.float64)
            alpha_arr = np.zeros(0, dtype=np.float64)
            vTh_arr = np.zeros(0, dtype=np.float64)
            vBypass_arr = np.zeros(0, dtype=np.float64)
            vRefPos_arr = np.zeros(0, dtype=np.float64)
            vRefNeg_arr = np.zeros(0, dtype=np.float64)
            nMax_arr = np.zeros(0, dtype=np.float64)
            kDischarge_arr = np.zeros(0, dtype=np.float64)
            gScale_arr = np.zeros(0, dtype=np.float64)
            rC_arr = np.zeros(0, dtype=np.float64)
            f22StartVal_arr = np.zeros(0, dtype=np.float64)
            f22PeakVal_arr = np.zeros(0, dtype=np.float64)
            f22EndVal_arr = np.zeros(0, dtype=np.float64)
            E_a_arr = np.zeros(0, dtype=np.float64)
            R_th_arr = np.zeros(0, dtype=np.float64)
            tau_th_arr = np.zeros(0, dtype=np.float64)
            gamma_arr = np.zeros(0, dtype=np.float64)
            beta_alpha_arr = np.zeros(0, dtype=np.float64)
            beta_s_arr = np.zeros(0, dtype=np.float64)
            
            y_i = np.zeros(1, dtype=np.float64)
            y_v = np.zeros(1, dtype=np.float64)
            y_fp = np.zeros(1, dtype=np.float64)
            y_fd = np.zeros(1, dtype=np.float64)
            
            iHist_buffer = np.zeros((0, hist_capacity), order='F')
            stateHist_buffer = np.zeros((0, hist_capacity), order='F')
            f22Hist_buffer = np.zeros((0, hist_capacity), order='F')
            gHist_buffer = np.zeros((0, hist_capacity), order='F')
            tempHist_buffer = np.zeros((0, hist_capacity), order='F')

        # Convert BSource callables into PwlSource for JIT loop processing
        sim_vSources = self._convert_bsources_to_pwl(self.vSources, tEnd)
        sim_dynamicSources = self._convert_bsources_to_pwl(self.dynamicSources, tEnd)
        
        # Serialize voltage sources
        (v_types, v_nodes, v_dc_voltages, v_pulse_params, 
         v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx) = self._serialize_sources(sim_vSources, fullNodeToIdx)
         
        # Serialize dynamic sources
        (dyn_types, dyn_nodes, dyn_dc_voltages, dyn_pulse_params, 
         dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask, dyn_pwl_cache_idx) = self._serialize_sources(sim_dynamicSources, nodeToIdx)
        
        # Use serialized nodes for dynamic solver lookup indices
        dynamic_node_idx = dyn_nodes
        
        # Initialize dynamic G and I arrays
        dynamic_G_arr = np.zeros(len(sim_dynamicSources), dtype=np.float64)
        dynamic_I_arr = np.zeros(len(sim_dynamicSources), dtype=np.float64)
        
        # Recompute dev_to_dyn_src using the new simulation sources list to keep mappings aligned
        self.dev_to_dyn_src = np.full(num_devices, -1, dtype=np.int32)
        if len(sim_dynamicSources) > 0:
            node_to_dyn_idx = {src.node: d_idx for d_idx, src in enumerate(sim_dynamicSources)}
            for i, d in enumerate(self.devices):
                d_idx = node_to_dyn_idx.get(d['bottom'])
                if d_idx is not None:
                    self.dev_to_dyn_src[i] = d_idx

        # Pre-allocate vDevs buffer for zero heap allocation
        vDevs_buffer = np.zeros(num_devices, dtype=np.float64)
        
        # Pre-initialize disk history store with conservative initial capacity to prevent re-initialization during simulation
        if recordHistory and not is_cal_helper:
            self._ensure_disk_history_store(
                num_nodes=len(all_nodes_ordered),
                num_devices=num_devices,
                initial_capacity=20000,
                recordHistory=recordHistory,
                recordStepTiming=recordStepTiming
            )

        # Routing configurations based on autotuner cores decision
        use_parallel = getattr(self, 'optimal_cores', 1) > 1
        t_start = t
        hist_idx_start = 0
        _buf_futures = [None, None]
        while True:
            # Ensure the previous background write to this buffer slot has completed before reusing it
            if _buf_futures[buf_idx] is not None and not _buf_futures[buf_idx].done():
                _buf_futures[buf_idx].result()

            tArr_buffer = tArr_buffers[buf_idx]
            vHist_buffer = vHist_buffers[buf_idx]
            stepTimeHist_buffer = stepTimeHist_buffers[buf_idx]
            if num_devices > 0:
                iHist_buffer = iHist_buffers[buf_idx]
                stateHist_buffer = stateHist_buffers[buf_idx]
                f22Hist_buffer = f22Hist_buffers[buf_idx]
                gHist_buffer = gHist_buffers[buf_idx]
                tempHist_buffer = tempHist_buffers[buf_idx]
            else:
                iHist_buffer = stateHist_buffer = f22Hist_buffer = gHist_buffer = tempHist_buffer = None

            enable_progress_flag = (not getattr(self, '_is_calibration_helper', False) and not getattr(self, '_in_warmup', False) and not getattr(self.__class__, '_in_warmup', False) and self.detailedPrint)
            hist_res, t_end_reached = jit_run_loop(
                tEnd, min_dt, tol, maxIter,
                numNodes, num_devices,
                dev_top_idx, dev_bot_idx, dev_top_full_idx, dev_bot_full_idx,
                v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx,
                dyn_types, dyn_dc_voltages, dyn_pulse_params, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask, dyn_pwl_cache_idx,
                dynamic_node_idx, dynamic_G_arr, dynamic_I_arr,
                n_arr, modeState_arr, scaleFactor_arr, T_arr,
                vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
                vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
                f22DepScaled_arr, f22PotScaled_arr,
                _cached_n_int_arr, iScale_arr, vScale_arr,
                currentF22_arr, currentIScale_arr, currentVScale_arr,
                gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
                E_a_arr, R_th_arr, tau_th_arr, gamma_arr, beta_alpha_arr, beta_s_arr, rateAsymmetry_arr,
                rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
                y_i, y_v, y_fp, y_fd,
                tArr_buffer, vHist_buffer, iHist_buffer, stateHist_buffer, f22Hist_buffer, gHist_buffer, tempHist_buffer, stepTimeHist_buffer, freq_val,
                vCurrent_arr, gEq_arr, iDev_arr, next_states, err_arr, rC_arr, vDevs_buffer,
                master_schedule, is_subsample, recordHistory, leakage_g, is_1t1r, self.dev_to_dyn_src, t_start, hist_idx_start,
                recordStepTiming, enable_progress_flag
            )
            if hist_res == -9999999:
                raise RuntimeError("Early Stop: JIT simulation exceeded step limit.")
            elif hist_res >= 0:
                hist_idx = hist_res
                break
            else:
                completed_steps = -hist_res
                if is_cal_helper:
                    host_start = getattr(self, '_hist_active_len', 0)
                    host_end = host_start + completed_steps
                    if not hasattr(self, '_cal_tArr_list'):
                        self._cal_tArr_list = []
                        self._cal_state_list = []
                    self._cal_tArr_list.append(tArr_buffer[:completed_steps].copy())
                    self._cal_state_list.append(stateHist_buffer[0, :completed_steps].copy())
                    self._hist_active_len = host_end
                elif recordHistory:
                    host_start = getattr(self, '_hist_active_len', 0)
                    host_end = host_start + completed_steps
                    needed_cap = max(20000, host_end)
                    store = self._ensure_disk_history_store(
                        num_nodes=len(all_nodes_ordered),
                        num_devices=num_devices,
                        initial_capacity=needed_cap,
                        recordHistory=recordHistory,
                        recordStepTiming=recordStepTiming
                    )
                    store.ensure_capacity(host_end)
                    future = store.write_chunk_async(
                        host_start=host_start,
                        completed_steps=completed_steps,
                        recordHistory=recordHistory,
                        recordStepTiming=recordStepTiming,
                        tArr=tArr_buffer,
                        vHist=vHist_buffer,
                        iHist=iHist_buffer,
                        stateHist=stateHist_buffer,
                        f22Hist=f22Hist_buffer,
                        gHist=gHist_buffer,
                        tempHist=tempHist_buffer,
                        stepTimeHist=stepTimeHist_buffer
                    )
                    _buf_futures[buf_idx] = future
                            
                    self._tArr_full = store.tArr
                    self._stepTimeHistory_full = store.stepTime
                    self._vHistory_full = store.vHist
                    self._iHistory_full = store.iHist
                    self._stateHistory_full = store.stateHist
                    self._f22History_full = store.f22Hist
                    self._gHistory_full = store.gHist
                    self._tempHistory_full = store.tempHist
                    self._hist_capacity_total = store.capacity
                    self._hist_active_len = host_end

                if detailed and verbose and not getattr(self, '_in_warmup', False) and not getattr(self.__class__, '_in_warmup', False) and title:
                    t_elapsed_cur = time.time() - tStart
                    total_cur_steps = getattr(self, '_hist_active_len', completed_steps)
                    steps_sec = total_cur_steps / max(1e-6, t_elapsed_cur)
                    progress = min(99.0, (t_end_reached / tEnd) * 100.0) if tEnd > 0 else 0.0
                    bar = '#' * int(progress / 5)
                    from .sys_utils import get_progress_suffix
                    suffix = get_progress_suffix(self)
                    status = f"Progress ({suffix}): [{bar.ljust(20)}] {progress:3.0f}% | {total_cur_steps} steps | {steps_sec:.0f} steps/sec"
                    import shutil
                    term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
                    width = max(20, min(150, term_cols - 3))
                    if len(status) > width:
                        status = status[:width-3] + "..."
                    print(f"\r  {status.ljust(width)}", end="", flush=True)

                hist_idx_start = -1
                t_start = t_end_reached
                buf_idx = 1 - buf_idx
        
        # Ensure all in-flight asynchronous history writes finish before post-processing
        if not is_cal_helper and recordHistory and hasattr(self, '_disk_history_store') and self._disk_history_store is not None:
            self._disk_history_store.wait_for_writes()

        
            
        # --- End of Simulation Back-Sync ---
        if num_devices > 0:
            if not np.all(np.isfinite(gEq_arr)) or not np.all(np.isfinite(n_arr)):
                bad_idx = np.where(~np.isfinite(gEq_arr) | ~np.isfinite(n_arr))[0]
                first_bad = int(bad_idx[0]) if len(bad_idx) > 0 else 0
                raise FloatingPointError(
                    f"Numerical Divergence detected during ODE transient integration: Crossbar device index {first_bad} "
                    f"reached a non-finite state (Conductance={gEq_arr[first_bad]}, State n={n_arr[first_bad]}) at simulation time t={t_end_reached:.3e}s."
                )
            self.last_iDev = iDev_arr.copy()
            if sync_states_back:
                num_devs = len(self.devices)
                if not hasattr(self, '_dev_objs_flat') or self._dev_objs_flat is None or len(self._dev_objs_flat) != num_devs:
                    self._dev_objs_flat = [d['dev'] for d in self.devices]
                dev_objs = self._dev_objs_flat
                
                if num_devs == 1:
                    dev = dev_objs[0]
                    dev._G = gEq_arr[0]
                    dev._n = n_arr[0]
                    dev._modeState = modeState_arr[0]
                    dev._scaleFactor = scaleFactor_arr[0]
                    dev._f22DepScaled = f22DepScaled_arr[0]
                    dev._f22PotScaled = f22PotScaled_host[0] if 'f22PotScaled_host' in locals() else f22PotScaled_arr[0]
                    dev._cached_n_int_val = _cached_n_int_arr[0]
                    dev._iScale = iScale_arr[0]
                    dev._vScale = vScale_arr[0]
                    dev._currentF22 = currentF22_arr[0]
                    dev._currentIScale = currentIScale_arr[0]
                    dev._currentVScale = currentVScale_arr[0]
                    dev._T = T_arr[0]
                else:
                    pot_scaled_src = f22PotScaled_host if 'f22PotScaled_host' in locals() else f22PotScaled_arr
                    if getattr(self, 'lazy_sync', True):
                        self._gpu_states_cache = {
                            'n_arr': n_arr.copy(),
                            'modeState_arr': modeState_arr.copy(),
                            'scaleFactor_arr': scaleFactor_arr.copy(),
                            'T_arr': T_arr.copy(),
                            'currentF22_arr': currentF22_arr.copy(),
                            '_cached_n_int_arr': _cached_n_int_arr.copy(),
                            'iScale_arr': iScale_arr.copy(),
                            'vScale_arr': vScale_arr.copy(),
                            'currentIScale_arr': currentIScale_arr.copy(),
                            'currentVScale_arr': currentVScale_arr.copy(),
                            'f22DepScaled_arr': f22DepScaled_arr.copy(),
                            'f22PotScaled_arr': pot_scaled_src.copy(),
                            'gEq_arr': gEq_arr.copy()
                        }
                        self._gpu_states_in_sync = False
                        self._gpu_states_dirty = False
                    elif (n_arr[0] == n_arr[-1] and T_arr[0] == T_arr[-1] and gEq_arr[0] == gEq_arr[-1] and 
                        modeState_arr[0] == modeState_arr[-1] and np.all(n_arr == n_arr[0])):
                        g0 = float(gEq_arr[0])
                        n0 = float(n_arr[0])
                        m0 = float(modeState_arr[0])
                        sc0 = float(scaleFactor_arr[0])
                        fd0 = float(f22DepScaled_arr[0])
                        fp0 = float(pot_scaled_src[0])
                        c0 = _cached_n_int_arr[0]
                        is0 = float(iScale_arr[0])
                        vs0 = float(vScale_arr[0])
                        cf0 = float(currentF22_arr[0])
                        cis0 = float(currentIScale_arr[0])
                        cvs0 = float(currentVScale_arr[0])
                        t0 = float(T_arr[0])
                        for dev in dev_objs:
                            dev._G = g0
                            dev._n = n0
                            dev._modeState = m0
                            dev._scaleFactor = sc0
                            dev._f22DepScaled = fd0
                            dev._f22PotScaled = fp0
                            dev._cached_n_int_val = c0
                            dev._iScale = is0
                            dev._vScale = vs0
                            dev._currentF22 = cf0
                            dev._currentIScale = cis0
                            dev._currentVScale = cvs0
                            dev._T = t0
                    else:
                        for i, dev in enumerate(dev_objs):
                            dev._G = gEq_arr[i]
                            dev._n = n_arr[i]
                            dev._modeState = modeState_arr[i]
                            dev._scaleFactor = scaleFactor_arr[i]
                            dev._f22DepScaled = f22DepScaled_arr[i]
                            dev._f22PotScaled = pot_scaled_src[i]
                            dev._cached_n_int_val = _cached_n_int_arr[i]
                            dev._iScale = iScale_arr[i]
                            dev._vScale = vScale_arr[i]
                            dev._currentF22 = currentF22_arr[i]
                            dev._currentIScale = currentIScale_arr[i]
                            dev._currentVScale = currentVScale_arr[i]
                            dev._T = T_arr[i]
                self._gpu_states_in_sync = True
                self._gpu_states_dirty = False
                
        if is_cal_helper:
            if hasattr(self, '_cal_tArr_list') and len(self._cal_tArr_list) > 0:
                self._cal_tArr_list.append(tArr_buffer[:hist_idx].copy())
                self._cal_state_list.append(stateHist_buffer[0, :hist_idx].copy())
                self.tArr = np.concatenate(self._cal_tArr_list)
                self.stateHistory = {0: np.concatenate(self._cal_state_list)}
                try:
                    del self._cal_tArr_list
                    del self._cal_state_list
                except Exception:
                    pass
            else:
                self.tArr = tArr_buffer[:hist_idx].copy()
                self.stateHistory = {0: stateHist_buffer[0, :hist_idx].copy()}
            self.stepTimeHistory = {}
            self.vHistory = {}
            self.iHistory = {}
            self.f22History = {}
            self.gHistory = {}
            self.tempHistory = {}
            self._hist_active_len = len(self.tArr)
        elif recordHistory:
            host_start = getattr(self, '_hist_active_len', 0)
            host_end = host_start + hist_idx
            needed_cap = max(2000, host_end)
            store = self._ensure_disk_history_store(
                num_nodes=len(all_nodes_ordered),
                num_devices=num_devices,
                initial_capacity=needed_cap,
                recordHistory=recordHistory,
                recordStepTiming=recordStepTiming
            )
            store.ensure_capacity(host_end)

            # Append logic natively mutating the raw memory blocks
            start = host_start
            end = host_end
            store.write_chunk_async(
                host_start=start,
                completed_steps=hist_idx,
                recordHistory=recordHistory,
                recordStepTiming=recordStepTiming,
                tArr=tArr_buffer,
                vHist=vHist_buffer,
                iHist=iHist_buffer,
                stateHist=stateHist_buffer,
                f22Hist=f22Hist_buffer,
                gHist=gHist_buffer,
                tempHist=tempHist_buffer,
                stepTimeHist=stepTimeHist_buffer
            )
            store.wait_for_writes()
                
            self._tArr_full = store.tArr
            self._stepTimeHistory_full = store.stepTime
            self._vHistory_full = store.vHist
            self._iHistory_full = store.iHist
            self._stateHistory_full = store.stateHist
            self._f22History_full = store.f22Hist
            self._gHistory_full = store.gHist
            self._tempHistory_full = store.tempHist
            self._hist_capacity_total = store.capacity
            self._hist_active_len = end
            
            # Publish the sliced memory views natively into the public API space (0 allocation cost)
            self.tArr = self._tArr_full[:end]
            self.stepTimeHistory = self._stepTimeHistory_full[:end] if recordStepTiming else {}
            self.vHistory = LazyHistoryDict(self._vHistory_full, all_nodes_ordered, end) if (recordHistory & 1) else {}
            if num_devices > 0:
                self.iHistory = LazyDeviceHistoryDict(self._iHistory_full, num_devices, end) if (recordHistory & 2) else {}
                self.stateHistory = LazyDeviceHistoryDict(self._stateHistory_full, num_devices, end) if (recordHistory & 4) else {}
                self.f22History = LazyDeviceHistoryDict(self._f22History_full, num_devices, end) if (recordHistory & 8) else {}
                self.gHistory = LazyDeviceHistoryDict(self._gHistory_full, num_devices, end) if (recordHistory & 16) else {}
                self.tempHistory = LazyDeviceHistoryDict(self._tempHistory_full, num_devices, end) if (recordHistory & 32) else {}
        else:
            self.tArr = None
            self.stepTimeHistory = {}
            self.vHistory = {}
            self.iHistory = {}
            self.stateHistory = {}
            self.f22History = {}
            self.gHistory = {}
            self.tempHistory = {}
            self._hist_active_len = 0

                
        # Persist the ThreadPoolExecutor across cycles; it will be destroyed in __del__
        pass
                
        if verbose and not getattr(self, '_in_warmup', False) and not getattr(self.__class__, '_in_warmup', False) and title:
            tElapsed = time.time() - tStart
            total_steps = getattr(self, '_hist_active_len', hist_idx) if recordHistory else hist_idx
            stepsPerSec = total_steps / tElapsed if tElapsed > 0 else 0
            if detailed:
                from .sys_utils import get_progress_suffix
                suffix = get_progress_suffix(self)
                final_status = f"Progress ({suffix}): [{'#' * 20}] 100% | Done in {tElapsed:.2f}s ({total_steps} steps, {stepsPerSec:.0f} steps/sec)"
                import shutil
                term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
                width = max(20, min(150, term_cols - 3))
                if len(final_status) > width:
                    final_status = final_status[:width-3] + "..."
                print(f"\r  {final_status.ljust(width)}\n")
            else:
                import shutil
                term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
                width = max(20, min(150, term_cols - 3))
                final_status = f"  > {title}: Done (in {tElapsed:.2f}s | {total_steps} steps)"
                if len(final_status) > width:
                    final_status = final_status[:width-3] + "..."
                print(f"\r{final_status.ljust(width)}\n")

    # --- Data Extraction Data ---



    # --- High-Level Machine Learning Inference API ---

    @handle_oom
    def programWeights(self, targetWeightsNormalized, parallel=True, detailedPrint=False, **kwargs):
        """
        Convenience alias for weight programming. Auto-initializes quantizers and
        CrossbarProgrammer if addCrossbarMatrix was instantiated without bitdepths.
        """
        if not hasattr(self, 'programmer') or not hasattr(self, 'wq'):
            from .quantization import WeightQuantizer, InputQuantizer, OutputQuantizer
            from .programmer import CrossbarProgrammer
            self.weightBits = 8
            self.inputBits = 8
            self.outputBits = 8
            self.wq = WeightQuantizer(targetBits=8, maxStates=16520)
            self.inQ = InputQuantizer(inputBits=8)
            self.adc = OutputQuantizer(adcBits=8)
            self.programmer = CrossbarProgrammer(self, pulseWidthPot=80e-9, pulseWidthDep=60e-9, period=160e-9)
            self.isCalibrated = False
        return self.updateCrossbarWeights(targetWeightsNormalized, detailedPrint=detailedPrint, **kwargs)

    @handle_oom
    def updateCrossbarWeights(self, targetWeightsNormalized, detailedPrint=None, prefix="", fineTune=True, recordHistory=False, silentUpdate=False):
        """
        Two-pass closed-loop programming of a normalized [0.0, 1.0] weight matrix.
        Pass 1: Initial programming pulse schedule based on pre-read conductance.
        Pass 2: Correction cycle that physically re-reads the array and applies
                 residual pulses to compensate for V/2 Inhibit drift.
        All weight reporting derives strictly from physical ADC reads. No peeking.
        
        detailedPrint=False: Single progress bar with compact summary. No headers.
        fineTune=False:      Optionally bypass Pass 2 for extreme execution speed.
        recordHistory=False: Automatically purge transient pulse history upon completion to prevent RAM creep during multple updates.
        """
        import gc
        was_gc_enabled = gc.isenabled()
        if was_gc_enabled:
            gc.disable()
        try:
            return self._updateCrossbarWeights_impl(targetWeightsNormalized, detailedPrint=detailedPrint, prefix=prefix, fineTune=fineTune, recordHistory=recordHistory, silentUpdate=silentUpdate)
        finally:
            self._invalidate_hardware_tensor_cache()
            if was_gc_enabled:
                gc.enable()

    def _updateCrossbarWeights_impl(self, targetWeightsNormalized, detailedPrint=None, prefix="", fineTune=True, recordHistory=False, silentUpdate=False):
        if detailedPrint is None:
            detailedPrint = getattr(self, 'detailedPrint', True)
        self.detailedPrint = detailedPrint
        if not hasattr(self, 'programmer'):
            raise RuntimeError("Must instantiate addCrossbarMatrix with quantization bits before updating weights.")
            
        if not self.isCalibrated:
            self.programmer.calibrate()
            self.isCalibrated = True
            
        rows = len(self.wordlines)
        cols = len(self.bitlines)
        if hasattr(self, 'wq') and self.wq is not None:
            targetStates = self.wq.mapToStates(targetWeightsNormalized)
        elif hasattr(self, 'programmer') and self.programmer is not None:
            max_s = float(getattr(self.programmer, 'maxLinearState', 16520.0))
            targetStates = np.clip(targetWeightsNormalized * max_s, 0.0, max_s)
        else:
            targetStates = np.clip(targetWeightsNormalized * 16520.0, 0.0, 16520.0)
        
        self._cumulative_hybrid_distribution = [0, 0, 0]
        self.relaxDeviceTemperatures()
        
        if not detailedPrint:
            # --- Compact Mode: Single progress bar, all internal output suppressed ---
            import shutil
            term_cols = shutil.get_terminal_size().columns
            width = max(20, min(150, term_cols - 3))
            
            def write_status(msg):
                if len(msg) > width:
                    msg = msg[:width-3] + "..."
                old_stdout.write(msg.ljust(width))
                old_stdout.flush()
                
            total_steps = 6  # programWeights, run, read, programWeights, run, read
            step = 0
            
            pfx = f"  Programming {rows}x{cols} " if not prefix else f"  {prefix} "
            
            old_stdout = sys.stdout
            if old_stdout is None:
                old_stdout = DummyStream()
            prog_start = time.perf_counter()
            try:
                # Pass 1
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*3}{' '*17}]  17% | Pass 1: Calculating Pulses...")
                
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*6}{' '*14}]  33% | Pass 1: Applying Parallel Updates...")
                self.programmer.programWeightsParallel(targetStates, isCorrectionPass=False, recordHistory=recordHistory)
                
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*10}{' '*10}]  50% | Pass 1: Reading Conductance...")
                gMatrix, _ = self.readCrossbarMatrix(detailedPrint=False, purpose=None)
                self.relaxDeviceTemperatures()
                
                # Pass 2
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*13}{' '*7}]  67% | Pass 2: Calculating Corrections...")
                
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*16}{' '*4}]  83% | Pass 2: Applying Parallel Corrections...")
                self.programmer.programWeightsParallel(targetStates, isCorrectionPass=True, tolerance=0.0001 if fineTune else 0.005, recordHistory=recordHistory)
                
                if not silentUpdate:
                    write_status(f"\r{pfx}[{'#'*20}] 100% | Verifying Matrix...")
                gMatrixFinal, _ = self.readCrossbarMatrix(detailedPrint=False, purpose=None)
                
                # Final compact summary
                elapsed = time.perf_counter() - prog_start
                if prefix:
                    # External integrations (TF) don't want trailing newlines or matrix dumps
                    if not silentUpdate:
                        write_status(f"\r{pfx}[{'#'*20}] 100% | Programmed in {elapsed:.3f}s")
                else:
                    if not silentUpdate:
                        wFinalNorm = self._conductanceToWeights(gMatrixFinal)
                        wStr = np.array2string(np.round(wFinalNorm, 2), separator=', ', max_line_width=200, suppress_small=True).replace('\n', '')
                        write_status(f"\r{pfx}[{'#'*20}] 100% | HW: {wStr} | {elapsed:.3f}s")
                        old_stdout.write("\n\n")
                        old_stdout.flush()
                
            finally:
                self.relaxDeviceTemperatures()
                if not recordHistory:
                    self.clearHistory()
                self.clearVoltageSources()
            return
        
        # --- Detailed Mode: Full verbose output ---
        print(f"\n--------------------------------------------------------------")
        print(f"--- MolmemSimulator: Updating {rows}x{cols} Weights".ljust(61) + "---")
        print(f"--------------------------------------------------------------")
        
        # --- Pass 1: Initial Programming Cycle ---
        print("Executing 100% physically accurate parallel column updates ...")
        self.programmer.programWeightsParallel(targetStates, isCorrectionPass=False, recordHistory=recordHistory)
        print(f"Pass 1 complete.")
        
        # Physical conductance read to assess accuracy after Pass 1
        gMatrix, _ = self.readCrossbarMatrix(detailedPrint=False, purpose=None)
        self.relaxDeviceTemperatures()
        
        # --- Pass 2: Correction Cycle ---
        print("Executing parallel correction cycle ...")
        self.programmer.programWeightsParallel(targetStates, isCorrectionPass=True, tolerance=0.0001 if fineTune else 0.005, recordHistory=recordHistory)
        print(f"Pass 2 complete.")
        
        # Final physical conductance read to report the true hardware state
        gMatrixFinal, _ = self.readCrossbarMatrix(detailedPrint=False, purpose=None)
        self.relaxDeviceTemperatures()
        
        wFinalNorm = self._conductanceToWeights(gMatrixFinal)
        wFinalNormStr = np.array2string(np.round(wFinalNorm, 4), separator=', ', suppress_small=True)
        
        print(f"\nProgramming cycle physically synchronized.")
        print(f"Ideal Weights:\n{targetWeightsNormalized}")
        print(f"\nProgrammed Weights (Hardware ADC):\n{wFinalNormStr}\n")
        
        if not recordHistory:
            self.clearHistory()
        self.clearVoltageSources()
        self._invalidate_hardware_tensor_cache()

    def _invalidate_hardware_tensor_cache(self):
        """Invalidates pre-staged GPU/NumPy hardware tensors when physical device states change."""
        self._cached_hw_tensors = None
        self._cached_hw_arrays = None

    def _get_hardware_tensor_cache(self, device=None, dtype=None, is_gpu=False):
        """
        Retrieves pre-staged physical device matrices (currentIScale, currentVScale, currentF22, rC),
        completely eliminating per-inference Python device scans, NumPy array reallocations,
        and redundant host-to-device PCIe transfers.
        """
        if getattr(self, '_gpu_states_dirty', False):
            self._invalidate_hardware_tensor_cache()

        rows = len(self.wordlines)
        cols = len(self.bitlines)
        if not hasattr(self, '_cached_crossbar_dev_indices') or self._cached_crossbar_dev_indices is None or len(self._cached_crossbar_dev_indices) != rows * cols:
            self._cached_crossbar_dev_indices = [self.crossbarDevices[(r+1, c+1)] for r in range(rows) for c in range(cols)]
        dev_indices = self._cached_crossbar_dev_indices

        if is_gpu:
            cache = getattr(self, '_cached_hw_tensors', None)
            if cache is not None and cache.get('device') == device and cache.get('dtype') == dtype:
                return cache['iScale'], cache['vScale'], cache['f22'], cache['rC']

            self.sync_to_cpu()
            currentIScale_np = np.array([self.devices[idx]['dev']._currentIScale for idx in dev_indices], dtype=np.float64).reshape(rows, cols)
            currentVScale_np = np.array([self.devices[idx]['dev']._currentVScale for idx in dev_indices], dtype=np.float64).reshape(rows, cols)
            currentF22_np = np.array([self.devices[idx]['dev']._currentF22 for idx in dev_indices], dtype=np.float64).reshape(rows, cols)
            rC_np = np.array([self.devices[idx]['dev'].rC for idx in dev_indices], dtype=np.float64).reshape(rows, cols)

            import torch
            iScale_t = torch.from_numpy(currentIScale_np).to(device=device, dtype=dtype)
            vScale_t = torch.from_numpy(currentVScale_np).to(device=device, dtype=dtype)
            f22_t = torch.from_numpy(currentF22_np).to(device=device, dtype=dtype)
            rC_t = torch.from_numpy(rC_np).to(device=device, dtype=dtype)

            self._cached_hw_tensors = {
                'device': device,
                'dtype': dtype,
                'iScale': iScale_t,
                'vScale': vScale_t,
                'f22': f22_t,
                'rC': rC_t
            }
            return iScale_t, vScale_t, f22_t, rC_t
        else:
            cache = getattr(self, '_cached_hw_arrays', None)
            if cache is not None:
                return cache['iScale'], cache['vScale'], cache['f22'], cache['rC']

            self.sync_to_cpu()
            currentIScale_arr = np.array([self.devices[idx]['dev']._currentIScale for idx in dev_indices], dtype=np.float64)
            vScale_arr = np.array([self.devices[idx]['dev']._currentVScale for idx in dev_indices], dtype=np.float64)
            currentF22_arr = np.array([self.devices[idx]['dev']._currentF22 for idx in dev_indices], dtype=np.float64)
            rC_arr = np.array([self.devices[idx]['dev'].rC for idx in dev_indices], dtype=np.float64)

            self._cached_hw_arrays = {
                'iScale': currentIScale_arr,
                'vScale': vScale_arr,
                'f22': currentF22_arr,
                'rC': rC_arr
            }
            return currentIScale_arr, vScale_arr, currentF22_arr, rC_arr

    def _conductanceToWeights(self, gMatrix):
        """
        Maps a physically measured Conductance matrix back to normalized [0, 1] weights
        using the programmer's calibrated conductance bounds.
        """
        if gMatrix is None:
            return np.zeros((len(self.wordlines), len(self.bitlines)), dtype=np.float64)
        rows, cols = gMatrix.shape
        if rows == 0 or cols == 0 or getattr(self, '_is_noop', False):
            return np.zeros((rows, cols), dtype=np.float64)

        if self.programmer.gMapG is None or self.programmer.gMapN is None:
            self.programmer.calibrate()

        if self.programmer.gMapG is None or self.programmer.gMapN is None:
            return np.zeros((rows, cols), dtype=np.float64)

        idx_g = np.clip(np.searchsorted(self.programmer.gMapG, np.asarray(gMatrix, dtype=np.float64)), 0, len(self.programmer.gMapG) - 1)
        measured_n = self.programmer.gMapN[idx_g]
        max_s = float(getattr(self.programmer, 'maxLinearState', 16520.0))
        if hasattr(self, 'wq') and hasattr(self.wq, 'maxStates'):
            max_s = float(self.wq.maxStates)
        min_s = float(getattr(self.wq, 'minStates', 0.0)) if hasattr(self, 'wq') else 0.0
        wNorm = (measured_n - min_s) / max(1.0, (max_s - min_s))
        return np.clip(wNorm, 0.0, 1.0)

    def _get_linear_max_conductance(self):
        """
        Returns the calibrated conductance corresponding to full-scale linear weight (W = 1.0, maxLinearState).
        Used for ADC current reference scaling during matrix-vector inference.
        """
        if getattr(self, 'programmer', None) is not None:
            if self.programmer.gMapG is None or self.programmer.gMapN is None:
                self.programmer.calibrate()
            if self.programmer.gMapG is not None and self.programmer.gMapN is not None:
                max_s = float(getattr(self.programmer, 'maxLinearState', 16520.0))
                if hasattr(self, 'wq') and hasattr(self.wq, 'maxStates'):
                    max_s = float(self.wq.maxStates)
                idx_s = int(np.clip(np.searchsorted(self.programmer.gMapN, max_s), 0, len(self.programmer.gMapG) - 1))
                return float(self.programmer.gMapG[idx_s])
        return 4e-4

    def saveStates(self):
        """
        Creates a deep, backend-agnostic snapshot of all device internal physical states
        (supporting CPU NumPy, PyTorch GPU tensors, and lazy synchronization).
        """
        self.sync_to_cpu()
        is_dirty = getattr(self, '_gpu_states_dirty', True)
        snapshot = {
            'backend': self.backend,
            '_gpu_states_dirty': is_dirty,
            '_gpu_states_in_sync': getattr(self, '_gpu_states_in_sync', True),
            '_cached_hw_tensors': getattr(self, '_cached_hw_tensors', None),
            '_cached_hw_arrays': getattr(self, '_cached_hw_arrays', None),
        }
        if getattr(self, '_gpu_cache', None) is not None:
            cache = self._gpu_cache
            if 'device_states_soa' in cache and cache['device_states_soa'] is not None:
                snapshot['gpu_cache'] = {
                    'device_states_soa': cache['device_states_soa'].clone(),
                    'gEq_arr': cache['gEq_arr'].clone() if 'gEq_arr' in cache and cache['gEq_arr'] is not None else None
                }
        if getattr(self, '_gpu_states_cache', None) is not None:
            snapshot['gpu_states_cache'] = {
                name: (self._gpu_states_cache[name].clone() if hasattr(self._gpu_states_cache[name], 'clone') else self._gpu_states_cache[name].copy())
                for name in self._gpu_states_cache
                if self._gpu_states_cache.get(name) is not None
            }
        if len(self.devices) > 0:
            dev_objs = getattr(self, '_dev_objs_flat', None)
            if dev_objs is None or len(dev_objs) != len(self.devices):
                dev_objs = [d['dev'] for d in self.devices]
                self._dev_objs_flat = dev_objs
            num_devs = len(dev_objs)
            state_buf = np.empty((13, num_devs), dtype=np.float64)
            for i in range(num_devs):
                dev = dev_objs[i]
                state_buf[0, i] = dev._n
                state_buf[1, i] = dev._modeState
                state_buf[2, i] = dev._scaleFactor
                state_buf[3, i] = dev._f22DepScaled
                state_buf[4, i] = dev._f22PotScaled
                state_buf[5, i] = dev._cached_n_int_val
                state_buf[6, i] = dev._iScale
                state_buf[7, i] = dev._vScale
                state_buf[8, i] = dev._currentF22
                state_buf[9, i] = dev._currentIScale
                state_buf[10, i] = dev._currentVScale
                state_buf[11, i] = dev._T
                state_buf[12, i] = getattr(dev, '_G', np.nan)
            snapshot['cpu_dev_states'] = state_buf
        return snapshot

    def restoreStates(self, snapshot):
        """
        Restores device internal physical states from a snapshot previously created by saveStates().
        """
        if not snapshot or not isinstance(snapshot, dict):
            return
        self._gpu_states_dirty = True
        self._gpu_states_in_sync = snapshot.get('_gpu_states_in_sync', True)
        if '_cached_hw_tensors' in snapshot:
            self._cached_hw_tensors = snapshot['_cached_hw_tensors']
        else:
            self._cached_hw_tensors = None

        if '_cached_hw_arrays' in snapshot:
            self._cached_hw_arrays = snapshot['_cached_hw_arrays']
        else:
            self._cached_hw_arrays = None
        
        if 'gpu_cache' in snapshot and getattr(self, '_gpu_cache', None) is not None:
            cache = self._gpu_cache
            for k, v in snapshot['gpu_cache'].items():
                if v is not None and k in cache:
                    if hasattr(cache[k], 'shape') and hasattr(v, 'shape') and cache[k].shape == v.shape:
                        if hasattr(cache[k], 'copy_'):
                            cache[k].copy_(v)
                        elif hasattr(cache[k], 'copy'):
                            np.copyto(cache[k], v)
                    else:
                        cache[k] = v.clone() if hasattr(v, 'clone') else (v.copy() if hasattr(v, 'copy') else v)
        if 'gpu_states_cache' in snapshot and snapshot['gpu_states_cache'] is not None:
            self._gpu_states_cache = {
                name: (snapshot['gpu_states_cache'][name].clone() if hasattr(snapshot['gpu_states_cache'][name], 'clone') else snapshot['gpu_states_cache'][name].copy())
                for name in snapshot['gpu_states_cache']
            }
            if self.backend in ['torch', 'pytorch', 'hybrid']:
                self._gpu_states_dirty = False
        if 'cpu_dev_states' in snapshot and snapshot['cpu_dev_states'] is not None:
            cpu_states = snapshot['cpu_dev_states']
            if isinstance(cpu_states, np.ndarray) and cpu_states.ndim == 2:
                dev_objs = getattr(self, '_dev_objs_flat', None)
                if dev_objs is None or len(dev_objs) != len(self.devices):
                    dev_objs = [d['dev'] for d in self.devices]
                    self._dev_objs_flat = dev_objs
                num_devs = len(dev_objs)
                if num_devs > 0:
                    is_uniform = (cpu_states.shape[1] == num_devs and 
                                  np.all(cpu_states[0, :] == cpu_states[0, 0]) and 
                                  np.all(cpu_states[11, :] == cpu_states[11, 0]))
                    if is_uniform:
                        n0 = cpu_states[0, 0]
                        m0 = cpu_states[1, 0]
                        sc0 = cpu_states[2, 0]
                        fd0 = cpu_states[3, 0]
                        fp0 = cpu_states[4, 0]
                        c0 = int(cpu_states[5, 0])
                        is0 = cpu_states[6, 0]
                        vs0 = cpu_states[7, 0]
                        cf0 = cpu_states[8, 0]
                        cis0 = cpu_states[9, 0]
                        cvs0 = cpu_states[10, 0]
                        t0 = cpu_states[11, 0]
                        g0 = cpu_states[12, 0]
                        has_g0 = not np.isnan(g0)
                        
                        for dev in dev_objs:
                            dev._n = n0
                            dev._modeState = m0
                            dev._scaleFactor = sc0
                            dev._f22DepScaled = fd0
                            dev._f22PotScaled = fp0
                            dev._cached_n_int_val = c0
                            dev._iScale = is0
                            dev._vScale = vs0
                            dev._currentF22 = cf0
                            dev._currentIScale = cis0
                            dev._currentVScale = cvs0
                            dev._T = t0
                            if has_g0:
                                dev._G = g0
                    else:
                        limit = min(num_devs, cpu_states.shape[1])
                        for i in range(limit):
                            dev = dev_objs[i]
                            dev._n = cpu_states[0, i]
                            dev._modeState = cpu_states[1, i]
                            dev._scaleFactor = cpu_states[2, i]
                            dev._f22DepScaled = cpu_states[3, i]
                            dev._f22PotScaled = cpu_states[4, i]
                            dev._cached_n_int_val = int(cpu_states[5, i])
                            dev._iScale = cpu_states[6, i]
                            dev._vScale = cpu_states[7, i]
                            dev._currentF22 = cpu_states[8, i]
                            dev._currentIScale = cpu_states[9, i]
                            dev._currentVScale = cpu_states[10, i]
                            dev._T = cpu_states[11, i]
                            g_val = cpu_states[12, i]
                            if not np.isnan(g_val):
                                dev._G = g_val
            elif isinstance(cpu_states, (list, tuple)):
                dev_objs = getattr(self, '_dev_objs_flat', None)
                if dev_objs is None or len(dev_objs) != len(self.devices):
                    dev_objs = [d['dev'] for d in self.devices]
                    self._dev_objs_flat = dev_objs
                for i, dev in enumerate(dev_objs):
                    if i < len(cpu_states):
                        (dev._n, dev._modeState, dev._scaleFactor, dev._f22DepScaled,
                         dev._f22PotScaled, dev._cached_n_int_val, dev._iScale,
                         dev._vScale, dev._currentF22, dev._currentIScale,
                         dev._currentVScale, dev._T, dev_G) = cpu_states[i]
                        if dev_G is not None:
                            dev._G = dev_G

    @staticmethod
    def get_dict_resizing_bytes(n):
        """Calculates CPython dictionary hash-table bucket memory for crossbar lookup tables."""
        if n <= 0:
            return 0
        cap = 1 << max(3, int(np.ceil(np.log2(max(8, 1.5 * n)))))
        return cap * 24 * 3  # crossbarDevices + nodalMatrix + devices lookup tables

    def estimate_gpu_memory_bytes(self, rows, cols, is_device_basis=False):
        """Calculates calibrated theoretical GPU VRAM footprint for crossbar simulations."""
        num_devices = rows * cols
        if is_device_basis:
            base_vram = 3.528 * 1024 * 1024
            pinned_dma_bytes = num_devices * 128.0  # 13 states + 3 pulses DMA pinned memory (128 B/dev)
            unit_bytes = 12.58 * 1024
            total_gpu_bytes = base_vram + num_devices * unit_bytes + pinned_dma_bytes
        else:
            base_vram = 1.052 * 1024 * 1024
            pwl_schedule_bytes = cols * max(rows, cols) * (1.02 * 1024)  # PWL pulse waveform buffers (1.02 KB/line)
            nodal_bytes = cols * ((rows + 2) ** 2) * 8.0
            state_pwl_bytes = num_devices * (5.24 * 1024)
            total_gpu_bytes = base_vram + state_pwl_bytes + nodal_bytes + pwl_schedule_bytes

        return total_gpu_bytes

    def estimate_cpu_memory_bytes(self, rows, cols, threads=None, is_device_basis=False):
        """Calculates calibrated theoretical CPU RAM footprint for crossbar simulations."""
        num_devices = rows * cols
        dict_heap_bytes = self.get_dict_resizing_bytes(num_devices)
        if is_device_basis:
            base_ram = 0.000 * 1024 * 1024
            unit_bytes = 1.22 * 1024
            total_cpu_bytes = base_ram + num_devices * unit_bytes + dict_heap_bytes
        else:
            base_ram = 0.000 * 1024 * 1024
            active_threads = threads if threads is not None else getattr(self, 'optimal_cores', None) or getattr(self, 'active_executor_cores', None) or min(os.cpu_count() or 16, max(1, cols))
            thread_workspace = active_threads * (0.10 * 1024 + ((rows + 2) ** 2) * 8 + (rows + 2) * 64)
            unit_bytes = 1.09 * 1024
            total_cpu_bytes = base_ram + num_devices * unit_bytes + thread_workspace + dict_heap_bytes

        return total_cpu_bytes

    def estimate_gpu_host_ram_bytes(self, rows, cols, is_device_basis=False):
        """Calculates theoretical Host System RAM footprint during GPU crossbar simulation."""
        num_devices = rows * cols
        dict_heap_bytes = self.get_dict_resizing_bytes(num_devices)

        if is_device_basis:
            base_host_ram = 0.001 * 1024 * 1024
            staging_bytes = num_devices * 1149.3  # Host heap objects, DMA pinned buffers, and readback arrays
            total_host_bytes = base_host_ram + staging_bytes + dict_heap_bytes
        else:
            base_host_ram = 9.572 * 1024 * 1024
            staging_bytes = num_devices * 104.9
            total_host_bytes = base_host_ram + staging_bytes + dict_heap_bytes

        return total_host_bytes

    def get_optimal_gpu_slice_size(self, rows, cols, pad_sources, max_v_pts, is_read=False, is_device_basis=False, ignore_limits=False):
        """
        Calculates the optimal slice size to use up to available VRAM fraction,
        capped by the locked-in maximum slice limits from crossover profiling.
        """
        import torch
        dev_str = str(self.device)
        is_gpu = dev_str == 'auto' or dev_str == 'cuda' or dev_str.startswith('cuda:')
        if is_gpu and torch.cuda.is_available():
            try:
                if dev_str.startswith('cuda:') and len(dev_str) > 5:
                    device = torch.device(dev_str)
                else:
                    device = torch.device(f'cuda:{torch.cuda.current_device()}')
                free_driver, total_memory = torch.cuda.mem_get_info(device)
                cached_unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                free_memory = free_driver + max(0, cached_unused)
            except Exception:
                return 64
        else:
            return 64
            
        num_devices = rows * cols
        if is_device_basis:
            base_overhead = 3.528 * 1024 * 1024
            bytes_per_item_with_overhead = (12.58 * 1024) + 128.0
        else:
            base_overhead = 1.052 * 1024 * 1024
            # Column slice scaling: state variables, quadratic nodal matrix, and PWL pulse timeline buffers
            nodal_bytes_per_col = ((rows + 2) ** 2) * 8
            state_bytes_per_col = rows * (5.24 * 1024)
            pwl_bytes_per_col = max(rows, cols) * (1.02 * 1024)
            bytes_per_item_with_overhead = state_bytes_per_col + nodal_bytes_per_col + pwl_bytes_per_col
        
        allowed_vram = max(0.0, (free_memory - base_overhead)) * getattr(self, 'vram_fraction', 0.8) * 0.95
        raw_items = int(allowed_vram // max(1.0, bytes_per_item_with_overhead))
        if raw_items > 1:
            p2 = 1 << (raw_items.bit_length() - 1)
            vram_limit = max(1, p2)
        else:
            vram_limit = max(1, raw_items)
        
        optimal_slice = vram_limit
        
        if is_gpu:
            try:
                props = torch.cuda.get_device_properties(device)
                cu_count = getattr(props, 'multi_processor_count', 32) or 32
                threads_per_cu = getattr(props, 'max_threads_per_multi_processor', 1024) or 1024
                self._max_gpu_device_batch = int(cu_count * threads_per_cu)
                max_cu_slice = max(8, int(2 * cu_count))
                optimal_slice = min(optimal_slice, max_cu_slice)
            except Exception:
                pass
        
        # Cap by the loaded maximum slice limits
        if not ignore_limits:
            limit = getattr(self, 'gpu_read_slice_limit', 999999) if is_read else getattr(self, 'gpu_slice_limit', 999999)
            optimal_slice = min(optimal_slice, limit)
        
        if os.environ.get("MOLMEM_DEBUG_VRAM") == "1":
            print(f"  [DYNAMIC VRAM PROBING] Free VRAM: {free_memory / (1024**2):.1f} MB | Est. slice footprint: {bytes_per_item_with_overhead / 1024:.2f} KB/item | Calculated slice limit: {optimal_slice}")
            
        return optimal_slice

    @handle_oom
    def readCrossbarMatrix(self, detailedPrint=True, purpose="Diagnostic Matrix State Read", returnAnalog=False):
        if hasattr(self, '_write_back_futures') and self._write_back_futures:
            concurrent.futures.wait(self._write_back_futures)
            self._write_back_futures.clear()

        if not hasattr(self, 'adc'):
            raise RuntimeError("Must instantiate addCrossbarMatrix with quantization bits before reading matrix output.")
            
        vRead = getattr(self, 'vRead', 0.1)
        rows = len(self.wordlines)
        cols = len(self.bitlines)
        
        if getattr(self, '_is_noop', False) or rows == 0 or cols == 0:
            return np.zeros((rows, cols)), np.zeros((rows, cols), dtype=np.int32)
        
        iMatrixAnalog = np.zeros((rows, cols))
        
        if purpose:
            print(f"\n--------------------------------------------------------------")
            print(f"--- MolmemSimulator: {purpose}".ljust(61) + "---")
            print(f"--------------------------------------------------------------")
            
        if detailedPrint:
            pass
            
        # Synchronize CPU physical device states
        self.sync_to_cpu()
        if not hasattr(self, '_read_device_indices') or self._read_device_indices.shape != (rows, cols):
            read_device_indices = np.empty((rows, cols), dtype=np.int64)
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    read_device_indices[r - 1, c - 1] = self.crossbarDevices[(r, c)]
            self._read_device_indices = read_device_indices

        dev_indices_flat = self._read_device_indices.ravel()
        currentIScale_np = np.array([self.devices[idx]['dev']._currentIScale for idx in dev_indices_flat], dtype=np.float64).reshape(rows, cols)
        currentVScale_np = np.array([self.devices[idx]['dev']._currentVScale for idx in dev_indices_flat], dtype=np.float64).reshape(rows, cols)
        currentF22_np = np.array([self.devices[idx]['dev']._currentF22 for idx in dev_indices_flat], dtype=np.float64).reshape(rows, cols)
        rC_np = np.array([self.devices[idx]['dev'].rC for idx in dev_indices_flat], dtype=np.float64).reshape(rows, cols)

        if self.backend in ['torch', 'pytorch', 'hybrid']:
            # GPU Vectorized Instant Cell Current Evaluation (Zero Launch Latency)
            import torch
            dev_str = getattr(self, 'device', 'auto')
            device = _resolve_torch_device(dev_str)
            dtype = torch.float64

            currentIScale_t = torch.from_numpy(currentIScale_np).to(device=device, dtype=dtype)
            currentVScale_t = torch.from_numpy(currentVScale_np).to(device=device, dtype=dtype)
            currentF22_t = torch.from_numpy(currentF22_np).to(device=device, dtype=dtype)
            rC_t = torch.from_numpy(rC_np).to(device=device, dtype=dtype)

            vS = currentVScale_t + 1e-12
            inv_vS = 1.0 / vS
            i_prod = currentIScale_t * currentF22_t
            g_base = i_prod * inv_vS
            rSeries = 2.0 * rC_t
            v = vRead * inv_vS
            coeff = rSeries * g_base

            y = v / (coeff + 1e-20)
            asinh_y = torch.asinh(y)
            x = torch.where(coeff <= 1e-18, v, torch.clamp(v, max=asinh_y))
            for _ in range(6):
                sinh_x = torch.sinh(x)
                cosh_x = torch.cosh(x)
                fx = x + coeff * sinh_x - v
                fpx = 1.0 + coeff * cosh_x
                dx = fx / fpx
                x = x - dx

            i_cell_t = torch.where(currentF22_t == 0.0, torch.zeros_like(i_prod), i_prod * torch.sinh(x))
            iMatrixAnalog = i_cell_t.cpu().numpy()
        else:
            # CPU Compiled Device Cell Current Evaluation
            from .device_jit import jit_getIGeq
            for r in range(rows):
                for c in range(cols):
                    iMatrixAnalog[r, c], _ = jit_getIGeq(
                        vRead,
                        float(currentIScale_np[r, c]),
                        float(currentVScale_np[r, c]),
                        float(currentF22_np[r, c]),
                        float(rC_np[r, c])
                    )
            
        # Inject Empirical Read Noise into analog currents prior to ADC quantization
        read_intensity = getattr(self, 'read_noise_intensity', 0.0)
        if read_intensity > 0.0:
            from .noise import NoiseEngine
            g_min_val = float(self.programmer.gMapG[0]) if (getattr(self, 'programmer', None) is not None and getattr(self.programmer, 'gMapG', None) is not None) else 0.0
            g_max_val = float(self.programmer.gMapG[-1]) if (getattr(self, 'programmer', None) is not None and getattr(self.programmer, 'gMapG', None) is not None) else 4e-4
            iMatrixAnalog = NoiseEngine.inject_read_noise_cpu(iMatrixAnalog, vRead, g_min_val, g_max_val, intensity=read_intensity)

        # After physically sweeping the entire array, quantize the aggregated current map using physical hardware range
        gMax = float(self.programmer.gMapG[-1]) if (getattr(self, 'programmer', None) is not None and getattr(self.programmer, 'gMapG', None) is not None) else self._get_linear_max_conductance()
        iMax_single = float(vRead * gMax)
        self.adc.setRange(0.0, iMax_single)
        iMatrixBits = self.adc.quantizeCurrents(iMatrixAnalog, returnBits=True)
        iMatrixQuantizedAnalog = self.adc.quantizeCurrents(iMatrixAnalog, returnBits=False)
        
        # Post-Process the quantized data to derive Conductance (G = I_quant / V)
        inv_vRead = 1.0 / vRead if vRead != 0.0 else 0.0
        gMatrixQuantized = iMatrixQuantizedAnalog * inv_vRead
        
        if detailedPrint:
            print(f"Read Conductances (S):\n{gMatrixQuantized}")
            print(f"\nStatic Conductance ADC Output Bits:\n{iMatrixBits}\n")
            
        # Cleanly disconnect all active diagnostic DC sources so programmed matrices can map unhindered
        self.clearVoltageSources()
        
        if returnAnalog:
            gMatrixAnalog = iMatrixAnalog * inv_vRead
            return gMatrixAnalog, iMatrixBits

        return gMatrixQuantized, iMatrixBits

    @handle_oom
    def passInput(self, xNormalized, detailedPrint=True, appendHistory=False, stream=None, return_torch=False):
        """
        Passes a normalized Input Vector through the programmed crossbar array.
        Maps inputs to pulse widths, simulates the analog propagation, 
        digitizes instantaneous column currents, and integrates them.

        Args:
            xNormalized: Input vector or 2D batch of input vectors (NumPy array or PyTorch Tensor).
            detailedPrint: Whether to print verbose progress.
            appendHistory: Whether to append to transient waveform history.
            stream: Optional PyTorch CUDA/HIP Stream to execute on.
            return_torch: If True, returns the output directly as a PyTorch Tensor on GPU VRAM
                          (bypassing CPU synchronization and host DMA transfers).
        """
        if hasattr(self, '_write_back_futures') and self._write_back_futures:
            concurrent.futures.wait(self._write_back_futures)
            self._write_back_futures.clear()

        if not hasattr(self, 'inQ'):
            raise RuntimeError("Must instantiate addCrossbarMatrix with quantization bits before passing input.")
            
        rows = len(self.wordlines)
        cols = len(self.bitlines)
        vRead = getattr(self, 'vRead', 0.1)
        
        # Check if the input is a 2D batch
        try:
            import torch
            if isinstance(xNormalized, torch.Tensor):
                xNormalized_np = xNormalized.detach().cpu().numpy().astype(np.float32)
            else:
                xNormalized_np = np.asarray(xNormalized, dtype=np.float32)
        except ImportError:
            xNormalized_np = np.asarray(xNormalized, dtype=np.float32)

        flatten_output_1d = False
        if xNormalized_np.ndim == 1 and not appendHistory:
            if len(xNormalized_np) != rows:
                raise ValueError(f"Input vector size ({len(xNormalized_np)}) must match number of rows ({rows}).")
            xNormalized_np = xNormalized_np.reshape(1, rows)
            flatten_output_1d = True

        if xNormalized_np.ndim == 2:
            if appendHistory:
                raise ValueError("History logging (appendHistory=True) is not supported when passing inputs in batch parallel mode.")
                
            batch_size, input_rows = xNormalized_np.shape
            if input_rows != rows:
                raise ValueError(f"Input vector size ({input_rows}) must match number of rows ({rows}).")
                
            if detailedPrint:
                print(f"\n--------------------------------------------------------------")
                print(f"--- MolmemSimulator: Passing Batch of Inputs ({batch_size} vectors) ---")
                print(f"--------------------------------------------------------------")
                
            # GPU Backend Batched Pathway
            dev_str = getattr(self, 'device', 'auto')
            from .sys_utils import is_gpu_ready
            if dev_str == 'auto':
                if is_gpu_ready():
                    selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                    self.device = f'cuda:{selected_gpu}'
                    dev_str = self.device
                    is_gpu = True
                else:
                    self.device = 'cpu'
                    self.backend = 'numba'
                    dev_str = 'cpu'
                    is_gpu = False
            else:
                is_gpu = str(dev_str).startswith('cuda') or dev_str in ('gpu', 'cuda')

            if self.backend in ['torch', 'pytorch', 'hybrid'] or is_gpu:
                import torch
                import contextlib
                from .torch_simulator import run_torch_simulation, torch_integrate_read_currents, to_device_async
                
                device = _resolve_torch_device(dev_str)
                dtype = torch.float64
                cuda_dev_context = torch.cuda.device(device) if (torch.cuda.is_available() and device.type == 'cuda') else contextlib.nullcontext()
                
                # Quantize all inputs in the batch to identify zero-input vectors
                pulseWidths_batch = self.inQ.quantizeInputs(xNormalized_np)
                is_zero_input = pulseWidths_batch.max(axis=1) == 0.0
                
                # If all inputs in the batch are zero, bypass entirely
                if is_zero_input.all():
                    if detailedPrint:
                        print("All batch inputs are zero. Bypassing simulation entirely.")
                    if return_torch:
                        res_t = torch.zeros((batch_size, cols), dtype=torch.float32, device=device)
                        return res_t[0] if flatten_output_1d else res_t
                    res_np = np.zeros((batch_size, cols), dtype=np.float32)
                    return res_np[0] if flatten_output_1d else res_np
                
                # Direct Fused GPU Parallel Batch Inference Solver (Zero Launch Latency / Zero ODE overhead)
                currentIScale_tensor, currentVScale_tensor, currentF22_tensor, rC_tensor = self._get_hardware_tensor_cache(device=device, dtype=dtype, is_gpu=True)
                pulse_widths_tensor = to_device_async(torch.from_numpy(pulseWidths_batch), device).to(dtype=dtype)

                g_linear_max = self._get_linear_max_conductance()
                iMax_single = float(vRead * g_linear_max)
                adc_iMin = 0.0
                adc_iMax = iMax_single * rows

                from .torch_simulator import torch_gpu_batch_pass_input
                yNormalized_tensor = torch_gpu_batch_pass_input(
                    pulse_widths_batch=pulse_widths_tensor,
                    currentIScale_tensor=currentIScale_tensor,
                    currentVScale_tensor=currentVScale_tensor,
                    currentF22_tensor=currentF22_tensor,
                    rC_tensor=rC_tensor,
                    vRead=float(vRead),
                    rows=rows,
                    cols=cols,
                    adc_iMin=float(adc_iMin),
                    adc_iMax=float(adc_iMax),
                    numLevels=self.adc.numLevels,
                    maxPulseWidth=float(self.inQ.maxPulseWidth)
                )

                if getattr(self, 'read_noise_intensity', 0.0) > 0.0:
                    from .noise import NoiseEngine
                    yNormalized_tensor = NoiseEngine.inject_read_noise_torch(yNormalized_tensor, 1.0, 0.0, 1.0, intensity=self.read_noise_intensity)

                if return_torch:
                    return yNormalized_tensor[0] if flatten_output_1d else yNormalized_tensor
                res_np = yNormalized_tensor.cpu().numpy()
                return res_np[0] if flatten_output_1d else res_np
                
            # Quantize all inputs in the batch to identify zero-input vectors
            pulseWidths_batch = self.inQ.quantizeInputs(xNormalized_np)
            is_zero_input = pulseWidths_batch.max(axis=1) == 0.0
            
            # If all inputs in the batch are zero, bypass entirely
            if is_zero_input.all():
                if detailedPrint:
                    print("All batch inputs are zero. Bypassing simulation entirely.")
                res = np.zeros((batch_size, cols), dtype=np.float32)
                try:
                    import torch
                    if return_torch:
                        res_t = torch.from_numpy(res)
                        return res_t[0] if flatten_output_1d else res_t
                    return res[0] if flatten_output_1d else res
                except ImportError:
                    return res[0] if flatten_output_1d else res
                
            # Native Compiled Numba/TBB Parallel Batch Execution (Zero GIL Contention)
            currentIScale_arr, vScale_arr, currentF22_arr, rC_arr = self._get_hardware_tensor_cache(is_gpu=False)

            pulse_widths_2d = self.inQ.quantizeInputs(xNormalized_np)
            if not isinstance(pulse_widths_2d, np.ndarray) or pulse_widths_2d.dtype != np.float64:
                pulse_widths_2d = np.ascontiguousarray(pulse_widths_2d, dtype=np.float64)

            g_linear_max = self._get_linear_max_conductance()
            iMax_single = float(vRead * g_linear_max)
            adc_iMin = 0.0
            adc_iMax = iMax_single * rows
            max_adc_bits = float(self.adc.numLevels - 1)
            maxPulseWidth = float(self.inQ.maxPulseWidth)

            results = np.empty((batch_size, cols), dtype=np.float32)
            system_cap = max(1, int((os.cpu_count() or 1) * 0.8))
            nb_threads = min(batch_size, system_cap)
            import numba as nb
            prev_threads = nb.get_num_threads()
            try:
                nb.set_num_threads(nb_threads)
                jit_cpu_batch_pass_input(
                    pulse_widths_2d,
                    currentIScale_arr,
                    vScale_arr,
                    currentF22_arr,
                    rC_arr,
                    float(vRead),
                    rows,
                    cols,
                    float(adc_iMin),
                    float(adc_iMax),
                    max_adc_bits,
                    maxPulseWidth,
                    results
                )
            finally:
                nb.set_num_threads(prev_threads)

            if getattr(self, 'read_noise_intensity', 0.0) > 0.0:
                from .noise import NoiseEngine
                results = NoiseEngine.inject_read_noise_cpu(results, 1.0, 0.0, 1.0, intensity=self.read_noise_intensity)

            try:
                import torch
                if return_torch:
                    res_t = torch.from_numpy(results)
                    return res_t[0] if flatten_output_1d else res_t
                return results[0] if flatten_output_1d else results
            except ImportError:
                return results[0] if flatten_output_1d else results

        # 1D single-vector fallback path starts here
        if len(xNormalized_np) != rows:
            raise ValueError(f"Input vector size ({len(xNormalized_np)}) must match number of rows ({rows}).")
            
        if detailedPrint:
            print(f"\n--------------------------------------------------------------")
            print(f"--- MolmemSimulator: Passing Single Input Vector ---")
            print(f"--------------------------------------------------------------")
            
        # 1. Quantize Inputs
        pulseWidths = self.inQ.quantizeInputs(xNormalized_np)
        
        # Check if all inputs are zero
        if np.max(pulseWidths) == 0.0:
            if detailedPrint:
                print("All inputs are zero. Bypassing simulation entirely.")
            y_res = np.zeros(cols, dtype=np.float32)
            try:
                import torch
                return torch.from_numpy(y_res) if return_torch else y_res
            except ImportError:
                return y_res
            
        # 2. Reset simulator sources for the read cycle
        self.clearVoltageSources()
        for c in range(cols):
            self.addDcSource(col=c+1, voltage=0.0)
            
        maxT = float(np.max(pulseWidths)) if len(pulseWidths) > 0 else 0.0
        v_points_const = np.array([0.0, vRead, vRead, 0.0], dtype=np.float64)
        for r in range(rows):
            width = float(pulseWidths[r])
            if width > 0.0:
                tPoints = np.array([0.0, 1e-11, width, width + 1e-11], dtype=np.float64)
                self.addVoltageSource(row=r+1, sourceObj=PwlSource(node=None, t_points=tPoints, v_points=v_points_const))
            else:
                self.addDcSource(row=r+1, voltage=0.0)
                
        # 3. Add a tail buffer for the Physics Integrator to settle
        tTail = 1e-9
        tEnd = maxT + tTail
        if tEnd == tTail:
            tEnd = 1e-9
                
        # 3. Simulate (appendHistory=False prevents RAM accumulation in loops)
        start_idx = self._hist_active_len if appendHistory else 0
        
        if detailedPrint:
            print(f"Executing {tEnd*1e9:.0f} ns Physical Inference Simulation cycle...")
            
        self.run(
            tEnd=tEnd, min_dt=1e-12, tol=1e-3, 
            title="Inference Pass" if detailedPrint else None, 
            appendHistory=appendHistory, 
            recordHistory=['i'] if not appendHistory else True
        )
        self._ensure_history_synced()
        
        # 4. Read Columns, Quantize Instantaneous Current, and Integrate (Q_bits = I_bits * t)
        tHist = self.getTime()[start_idx:]
        
        # Extract 2D array of all mapped crossbar devices: shape (rows*cols, time_steps)
        if not hasattr(self, '_cached_crossbar_dev_indices') or self._cached_crossbar_dev_indices is None or len(self._cached_crossbar_dev_indices) != rows * cols:
            self._cached_crossbar_dev_indices = [self.crossbarDevices[(r+1, c+1)] for r in range(rows) for c in range(cols)]
        dev_indices = self._cached_crossbar_dev_indices
        
        # Build 3D array scaling natively and cleanly avoiding iterative aggregation loops
        if rows == 1 and cols == 1:
            if start_idx == 0 and hasattr(self.iHistory, '_full_array') and hasattr(self.iHistory, '_active_len'):
                colAnalogCurrentsMatrix = self.iHistory._full_array[dev_indices[0]:dev_indices[0]+1, :self.iHistory._active_len]
            else:
                colAnalogCurrentsMatrix = self.iHistory[dev_indices[0]][start_idx:][np.newaxis, :]
        elif rows == 1:
            if start_idx == 0 and hasattr(self.iHistory, '_full_array') and hasattr(self.iHistory, '_active_len'):
                colAnalogCurrentsMatrix = self.iHistory._full_array[dev_indices, :self.iHistory._active_len]
            else:
                colAnalogCurrentsMatrix = np.array([self.iHistory[idx][start_idx:] for idx in dev_indices])
        else:
            if start_idx == 0 and hasattr(self.iHistory, '_full_array') and hasattr(self.iHistory, '_active_len'):
                i_hist_matrix_flat = self.iHistory._full_array[dev_indices, :self.iHistory._active_len]
            else:
                i_hist_matrix_flat = np.array([self.iHistory[idx][start_idx:] for idx in dev_indices])
            i_hist_matrix = i_hist_matrix_flat.reshape(rows, cols, -1)
            colAnalogCurrentsMatrix = np.sum(i_hist_matrix, axis=0)
        g_linear_max = self._get_linear_max_conductance()
        iMax_single = float(vRead * g_linear_max)
        
        self.adc.setRange(0.0, iMax_single * rows)
        
        # Execute global quantization simultaneously natively over Numpy backend bounds
        colIHistDigitalMatrix = self.adc.quantizeCurrents(colAnalogCurrentsMatrix, returnBits=True)
        
        # 6. Normalize the integrated physical Yield to match abstract software Math (Y = X * W)
        max_adc_bits = self.adc.numLevels - 1
        inv_max_integral = float(rows) / (max_adc_bits * self.inQ.maxPulseWidth)
        
        # Mathematically integrate identical digital bit streams globally in fused trapezoidal pass
        if colIHistDigitalMatrix.shape[1] > 1 and len(tHist) > 1:
            dt_steps = np.diff(tHist)
            y_sum = colIHistDigitalMatrix[:, 1:] + colIHistDigitalMatrix[:, :-1]
            yNormalized = np.sum(y_sum * dt_steps, axis=1) * (0.5 * inv_max_integral)
        else:
            yNormalized = np.zeros(cols, dtype=np.float32)
            
        if getattr(self, 'read_noise_intensity', 0.0) > 0.0:
            from .noise import NoiseEngine
            yNormalized = NoiseEngine.inject_read_noise_cpu(yNormalized, 1.0, 0.0, 1.0, intensity=self.read_noise_intensity)
            
        if os.environ.get('MOLMEM_DEBUG_GPU', '0') == '1':
            print(f"[DEBUG_1D_INT] colIHist shape={colIHistDigitalMatrix.shape} | len(tHist)={len(tHist)} | yNormalized={yNormalized}")
            
        if return_torch:
            try:
                import torch
                return torch.from_numpy(yNormalized)
            except ImportError:
                pass
        return yNormalized


import queue
import os
import atexit
import threading

_GLOBAL_PARALLEL_CROSSBAR_EXECUTOR = None
_GLOBAL_CPU_CROSSBAR_LOCK = threading.Lock()

def _cleanup_parallel_crossbar_executor():
    global _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR
    if _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR is not None:
        try:
            _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR.shutdown(wait=False)
        except Exception:
            pass
        _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR = None

atexit.register(_cleanup_parallel_crossbar_executor)

def updateCrossbarsParallel(simulators, weight_matrices, detailedPrint=True, prefix="", fineTune=True, recordHistory=False):
    """
    Simultaneously dispatches `updateCrossbarWeights` across an arbitrary list of disjoint 
    MolmemSimulator objects, strictly utilizing dynamic thread-resource pooling.
    """
    if len(simulators) != len(weight_matrices):
        raise ValueError(f"Mismatch: Received {len(simulators)} simulators but {len(weight_matrices)} weight matrices.")
        
    if len(simulators) == 0:
        return

    # Validate weight matrix dimensions and finiteness upfront on main thread
    for i, (sim, w) in enumerate(zip(simulators, weight_matrices)):
        w_arr = np.asarray(w)
        exp_shape = (len(getattr(sim, 'wordlines', {})), len(getattr(sim, 'bitlines', {})))
        if w_arr.shape != exp_shape:
            raise ValueError(f"Weight matrix shape mismatch for simulator {i} ({getattr(sim, 'instanceName', f'sim_{i}')}): Expected {exp_shape}, got {w_arr.shape}")
        if not np.all(np.isfinite(w_arr)):
            raise ValueError(f"Weight matrix for simulator {i} ({getattr(sim, 'instanceName', f'sim_{i}')}) contains NaN or Infinite values.")

    try:
        from . import ensure_startup_banner
        ensure_startup_banner()
    except Exception:
        pass

    original_vram_fractions = []
    try:
        # 1. Sync Hardware Calibrations (Prevent Thread Death)
        for sim in simulators:
            prog = getattr(sim, 'programmer', None)
            if prog is not None:
                if prog.stateMapPot is None or prog.stateMapDep is None or prog.gMapG is None:
                    if detailedPrint:
                        print(f"\n[+] Global Parallel Dispatch: Enforcing synchronous calibration for {sim.instanceName} limits...")
                    prog.calibrate()

        # Synchronously pre-profile thread parallelization on the main thread for the first array
        # to ensure all worker threads immediately hit memory cache and prevent concurrent sweep collisions
        if simulators:
            first_sim = simulators[0]
            if hasattr(first_sim, 'profileParallelization'):
                orig_dp = getattr(first_sim, 'detailedPrint', True)
                first_sim.detailedPrint = detailedPrint
                try:
                    first_sim.profileParallelization()
                finally:
                    first_sim.detailedPrint = orig_dp

            # Propagate optimal_cores to remaining simulators of matching geometry & backend
            first_opt = getattr(first_sim, 'optimal_cores', None)
            first_use_par = getattr(first_sim, 'use_parallel_optimized', False)
            if first_opt is not None:
                for s in simulators[1:]:
                    if not hasattr(s, 'optimal_cores'):
                        if (len(getattr(s, 'wordlines', {})) == len(getattr(first_sim, 'wordlines', {})) and
                            len(getattr(s, 'bitlines', {})) == len(getattr(first_sim, 'bitlines', {})) and
                            getattr(s, 'backend', None) == getattr(first_sim, 'backend', None)):
                            s.optimal_cores = first_opt
                            s.use_parallel_optimized = first_use_par

        # Dynamically scale VRAM fraction budget and thread allocation proportionally based on profiled metrics
        sizes = [max(1, len(getattr(sim, 'wordlines', {})) * len(getattr(sim, 'bitlines', {}))) for sim in simulators]
        total_size = sum(sizes)
        inv_total_size = 1.0 / total_size if total_size > 0 else 1.0

        K = len(simulators)
        total_cpu_budget = max(1, int((os.cpu_count() or 32) * 0.8))
        per_sim_cpu_budget = max(1, total_cpu_budget // K)
        
        # Query GPU multi-processor and wavefront properties
        cu_count = 32
        threads_per_cu = 1024
        if not is_interactive_notebook() and os.environ.get("MOLMEM_CPU_ONLY") != "1":
            try:
                import torch
                if torch.cuda.is_available():
                    selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                    if selected_gpu >= torch.cuda.device_count():
                        selected_gpu = 0
                    props = torch.cuda.get_device_properties(selected_gpu)
                    cu_count = getattr(props, 'multi_processor_count', 32) or 32
                    threads_per_cu = getattr(props, 'max_threads_per_multi_processor', 1024) or 1024
            except Exception:
                pass
            
        per_sim_cu_budget = max(1, cu_count // K)
        per_sim_max_gpu_devs = int(per_sim_cu_budget * threads_per_cu)

        for sim, size in zip(simulators, sizes):
            orig_frac = getattr(sim, 'vram_fraction', 0.8)
            original_vram_fractions.append(orig_frac)
            sim.vram_fraction = orig_frac * (size * inv_total_size)
            opt = getattr(sim, 'optimal_cores', per_sim_cpu_budget) or per_sim_cpu_budget
            sim._thread_budget = max(1, min(int(opt), per_sim_cpu_budget))
            sim._cu_budget = per_sim_cu_budget
            sim._max_gpu_device_batch = per_sim_max_gpu_devs

        # Pre-allocate dedicated CUDA streams on the main thread to eliminate driver contention during concurrent dispatch
        from .sys_utils import is_gpu_ready
        gpu_ready = is_gpu_ready()
        if not is_interactive_notebook() and os.environ.get("MOLMEM_CPU_ONLY") != "1" and gpu_ready:
            try:
                import torch
                if torch.cuda.is_available():
                    for sim in simulators:
                        use_streams = (sim.backend in ['torch', 'pytorch', 'hybrid', 'auto'] and 
                                       (sim.device == 'auto' or str(sim.device).startswith('cuda')))
                        if use_streams:
                            sim_dev_str = getattr(sim, 'device', 'auto')
                            if sim_dev_str == 'auto':
                                selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                                target_device = torch.device(f'cuda:{selected_gpu}')
                            else:
                                target_device = torch.device(sim_dev_str) if isinstance(sim_dev_str, str) else sim_dev_str
                            if not hasattr(sim, '_cuda_stream') or sim._cuda_stream is None or getattr(sim._cuda_stream, 'device', None) != target_device:
                                sim._cuda_stream = torch.cuda.Stream(device=target_device)
            except Exception:
                pass

        # 2. Concurrent Execution Pool across all independent arrays (persistent to eliminate OS thread churn)
        global _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR
        outer_workers = K
        if _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR is None or getattr(_GLOBAL_PARALLEL_CROSSBAR_EXECUTOR, '_max_workers', None) != outer_workers:
            if _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR is not None:
                try:
                    _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR.shutdown(wait=False)
                except Exception:
                    pass
            _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=outer_workers,
                thread_name_prefix="CrossbarParallelWorker"
            )
        executor = _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR
        
        def _update_single(sim, weights, sim_pfx):
            use_streams = (gpu_ready and sim.backend in ['torch', 'pytorch', 'hybrid', 'auto'] and 
                           (sim.device == 'auto' or str(sim.device).startswith('cuda')))
            if use_streams:
                import torch
                if torch.cuda.is_available():
                    sim_dev_str = getattr(sim, 'device', 'auto')
                    if sim_dev_str == 'auto':
                        selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
                        target_device = torch.device(f'cuda:{selected_gpu}')
                    else:
                        target_device = torch.device(sim_dev_str) if isinstance(sim_dev_str, str) else sim_dev_str
                    if not hasattr(sim, '_cuda_stream') or sim._cuda_stream is None or getattr(sim._cuda_stream, 'device', None) != target_device:
                        sim._cuda_stream = torch.cuda.Stream(device=target_device)
                    with torch.cuda.device(target_device):
                        with torch.cuda.stream(sim._cuda_stream):
                            sim.updateCrossbarWeights(
                                targetWeightsNormalized=weights, 
                                detailedPrint=detailedPrint, 
                                prefix=sim_pfx, 
                                fineTune=fineTune, 
                                recordHistory=recordHistory,
                                silentUpdate=not detailedPrint
                            )
                            if hasattr(sim, '_cuda_stream') and sim._cuda_stream is not None:
                                sim._cuda_stream.synchronize()
                            return 1
            threading_layer = "unknown"
            try:
                from .sys_utils import get_numba_threading_layer
                threading_layer = get_numba_threading_layer()
            except Exception:
                pass

            if threading_layer != "tbb":
                with _GLOBAL_CPU_CROSSBAR_LOCK:
                    sim.updateCrossbarWeights(
                        targetWeightsNormalized=weights, 
                        detailedPrint=detailedPrint, 
                        prefix=sim_pfx, 
                        fineTune=fineTune, 
                        recordHistory=recordHistory,
                        silentUpdate=not detailedPrint
                    )
            else:
                sim.updateCrossbarWeights(
                    targetWeightsNormalized=weights, 
                    detailedPrint=detailedPrint, 
                    prefix=sim_pfx, 
                    fineTune=fineTune, 
                    recordHistory=recordHistory,
                    silentUpdate=not detailedPrint
                )
            return 1


        futures = []
        for i, (sim, weights) in enumerate(zip(simulators, weight_matrices)):
            sim_pfx = f"[Parallel {i}] {prefix}" if prefix else f"[{sim.instanceName}]"
            futures.append(executor.submit(_update_single, sim, weights, sim_pfx))
            
        total_sims = len(simulators)
        completed = 0
        
        old_stdout = sys.__stdout__
        if old_stdout is None:
            old_stdout = DummyStream()
        prog_start = time.perf_counter()
        
        from .sys_utils import isMolmemLibSilenced
        is_silenced = isMolmemLibSilenced()
        
        if not detailedPrint and not is_silenced:
            pfx = f"[Parallel-Dispatch] " if not prefix else f"  {prefix} "
            old_stdout.write(f"\r  {pfx}[{' '*20}]   0% | Submitting independent arrays... ".ljust(95))
            old_stdout.flush()
        
        
        try:
            for f in concurrent.futures.as_completed(futures):
                completed += f.result()
        except KeyboardInterrupt:
            if not detailedPrint and not is_silenced:
                old_stdout.write("\n[!] Parallel programming interrupted by user. Terminating process...\n")
                old_stdout.flush()
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            _GLOBAL_PARALLEL_CROSSBAR_EXECUTOR = None
            os.kill(os.getpid(), 9)
                
        if not detailedPrint and not is_silenced:
            elapsed = time.perf_counter() - prog_start
            pfx = f"[Parallel-Dispatch] " if not prefix else f"  {prefix} "
            old_stdout.write(f"\r  {pfx}[{'#'*20}] 100% | Synchronized in {elapsed:.3f}s ".ljust(95))
            old_stdout.write("\n")
            old_stdout.flush()
            
        # Keep persistent executor alive across batches to prevent OS thread creation churn
    finally:
        # Restore original VRAM fractions
        for sim, orig_frac in zip(simulators, original_vram_fractions):
            sim.vram_fraction = orig_frac
