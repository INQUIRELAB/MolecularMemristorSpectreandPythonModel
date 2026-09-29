import os
import sys

# Prevent Intel OpenMP runtime aborts (Error #15 duplicate libiomp5md.dll initialization across PyTorch/MKL/Numba)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress Qt C++ runtime teardown warnings (e.g. QThreadStorage out-of-order destruction during exit)
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

# Ensure Linux dynamic linker can resolve libtbb.so for Numba TBB threading layer
if sys.platform != 'win32' and os.environ.get('_MOLMEM_TBB_LD_REEXEC') != '1' and not any(arg == '-c' for arg in sys.argv):
    _workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _candidate_lib_dirs = [
        os.path.join(sys.prefix, 'lib'),
        os.path.join(_workspace_root, 'molmem-env', 'lib'),
        os.path.join(_workspace_root, 'PythonSimulator', 'molmem-env', 'lib')
    ]
    _target_lib_dir = None
    for _d in _candidate_lib_dirs:
        if os.path.isfile(os.path.join(_d, 'libtbb.so.12')) or os.path.isfile(os.path.join(_d, 'libtbb.so')):
            _target_lib_dir = _d
            break
            
    if _target_lib_dir:
        _curr_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if _target_lib_dir not in _curr_ld.split(os.pathsep):
            os.environ['_MOLMEM_TBB_LD_REEXEC'] = '1'
            os.environ['LD_LIBRARY_PATH'] = _target_lib_dir + (os.pathsep + _curr_ld if _curr_ld else '')
            os.execv(sys.executable, [sys.executable] + sys.argv)

import warnings

__version__ = "1.0"

# Suppress PyTorch's redundant hardware incompatibility warning for devices > SM_90
# (Molmem_Lib banner already cleanly flags these devices as unavailable).
# Uses (?s) to ensure the regex matches across newlines since PyTorch formats the warning as a multiline string starting with \n.
warnings.filterwarnings("ignore", category=UserWarning, message=r"(?s).*is not compatible with the current PyTorch installation.*")
warnings.filterwarnings("ignore", message=r"(?s).*is not compatible with the current PyTorch installation.*")
warnings.filterwarnings("ignore", message=r"[\s\S]*is not compatible with the current PyTorch installation[\s\S]*")

# Suppress PyTorch TorchScript deprecation warnings (torch.jit.load is deprecated. Please switch to torch.export.)
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*`?torch\.jit\.load`? is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*torch\.jit\._serialization.*")
warnings.filterwarnings("ignore", message=r".*`?torch\.jit\.load`? is deprecated.*")




# =====================================================================
# GLOBAL CONFIGURATION: Logging Control
# Set ENABLE_LOGGING = True to persist all run, error, import, and profile logs.
# Default is False (zero disk footprint unless -logging CLI flag is supplied).
# =====================================================================
ENABLE_LOGGING = False

# Intercept CLI logging flags dynamically (-logging, --logging, -log, --log)
for _log_flag in ['-logging', '--logging', '-log', '--log']:
    if _log_flag in sys.argv:
        ENABLE_LOGGING = True
        os.environ["MOLMEM_ENABLE_LOGGING"] = "1"
        try:
            sys.argv.remove(_log_flag)
        except Exception:
            pass

# Intercept CLI SSD profiling flags dynamically (-profile-ssd, --profile-ssd, -ssd-profile, --ssd-profile)
for _ssd_prof_flag in ['-profile-ssd', '--profile-ssd', '-ssd-profile', '--ssd-profile']:
    if _ssd_prof_flag in sys.argv:
        ENABLE_LOGGING = True
        os.environ["MOLMEM_ENABLE_LOGGING"] = "1"
        os.environ["MOLMEM_PROFILE_SSD"] = "1"
        try:
            sys.argv.remove(_ssd_prof_flag)
        except Exception:
            pass

# Intercept CLI profiling flags dynamically (-profile, --profile, -profiler, --profiler, -profile-lines, --profile-lines)
for _prof_flag in ['-profile', '--profile', '-profiler', '--profiler', '-profile-lines', '--profile-lines']:
    if _prof_flag in sys.argv:
        ENABLE_LOGGING = True
        os.environ["MOLMEM_ENABLE_LOGGING"] = "1"
        os.environ["MOLMEM_PROFILE_LINES"] = "1"
        os.environ["MOLMEM_PROFILE_SSD"] = "1"
        try:
            sys.argv.remove(_prof_flag)
        except Exception:
            pass

# Intercept CLI CPU override flags dynamically (-cpu, --cpu, -force-cpu, --force-cpu, --cpu-only)
for _cpu_flag in ['-cpu', '--cpu', '-force-cpu', '--force-cpu', '-cpu-only', '--cpu-only']:
    if _cpu_flag in sys.argv:
        os.environ["MOLMEM_FORCE_CPU"] = "1"
        os.environ["MOLMEM_CPU_ONLY"] = "1"
        try:
            sys.argv.remove(_cpu_flag)
        except Exception:
            pass

# Intercept CLI GPU override flags dynamically (-gpu, --gpu, -force-gpu, --force-gpu)
for _gpu_flag in ['-gpu', '--gpu', '-force-gpu', '--force-gpu']:
    if _gpu_flag in sys.argv:
        os.environ["MOLMEM_FORCE_GPU"] = "1"
        try:
            sys.argv.remove(_gpu_flag)
        except Exception:
            pass

# Intercept CLI GPU debug flags dynamically (-debug-gpu, --debug-gpu, -gpu-debug, --gpu-debug)
for _debug_gpu_flag in ['-debug-gpu', '--debug-gpu', '-gpu-debug', '--gpu-debug']:
    if _debug_gpu_flag in sys.argv:
        os.environ["MOLMEM_DEBUG_GPU"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"
        try:
            sys.argv.remove(_debug_gpu_flag)
        except Exception:
            pass

# Intercept CLI silent flags dynamically (-silent, --silent, -quiet, --quiet)
for _silent_flag in ['-silent', '--silent', '-quiet', '--quiet']:
    if _silent_flag in sys.argv:
        os.environ["MOLMEM_SILENT"] = "1"
        try:
            sys.argv.remove(_silent_flag)
        except Exception:
            pass

# Intercept CLI offline flags dynamically (-offline, --offline)
for _offline_flag in ['-offline', '--offline']:
    if _offline_flag in sys.argv:
        os.environ["MOLMEM_OFFLINE"] = "1"
        os.environ["UV_OFFLINE"] = "1"
        try:
            sys.argv.remove(_offline_flag)
        except Exception:
            pass

# Intercept CLI check-updates flags dynamically (--update-pkgs, -update-pkgs, --check-updates, -check-updates)
for _update_flag in ['--update-pkgs', '-update-pkgs', '--check-updates', '-check-updates']:
    if _update_flag in sys.argv:
        os.environ["MOLMEM_CHECK_UPDATES"] = "1"
        try:
            sys.argv.remove(_update_flag)
        except Exception:
            pass

if os.environ.get("MOLMEM_ENABLE_LOGGING") == "1":
    ENABLE_LOGGING = True

# Ensure AMD libdrm discovers the ASIC ID table on Linux distros (avoids (null) missing file errors from libdrm_amdgpu)
if sys.platform != "win32" and "AMDGPU_ASIC_ID_TABLE_PATHS" not in os.environ:
    for _drm_candidate in ["/usr/share/libdrm", "/usr/local/share/libdrm", "/opt/rocm/share/libdrm"]:
        if os.path.exists(os.path.join(_drm_candidate, "amdgpu.ids")):
            os.environ["AMDGPU_ASIC_ID_TABLE_PATHS"] = _drm_candidate
            break

from .sys_utils import (
    DummyStream, _fast_rmtree, _find_rocm_path, _find_cuda_path, _find_vcvars64, 
    _find_msvc_bin, _find_windows_sdk_bin, is_worker_process, 
    configure_default_threading, get_numba_threading_layer, is_logging_enabled, get_log_dir,
    get_temp_history_dir, cleanup_temp_history_dir,
    resolve_and_install_caller_dependencies,
    silenceMolmemLib, isMolmemLibSilenced, silence_molmem_lib, is_molmem_lib_silenced,
    unsilence_molmem_lib, unsilenceMolmemLib, is_interactive_notebook,
    is_gpu_ready, clear_gpu_ready_cache, MolmemSilence
)

# In interactive notebooks (Jupyter, Colab, Kaggle), deny GPU usage and bypass GPU logic completely.
# This prevents driver conflicts, memory fragmentation, and overwhelming configuration requirements.
if is_interactive_notebook():
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ["MOLMEM_RUN_MODE"] = "CPU"

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')
if _CSRC_DIR not in sys.path:
    sys.path.insert(0, _CSRC_DIR)

if sys.platform == 'win32':
    _rocm_p = _find_rocm_path()
    if _rocm_p:
        _rocm_bin = os.path.join(_rocm_p, 'bin')
        if os.path.isdir(_rocm_bin):
            try:
                os.add_dll_directory(_rocm_bin)
            except Exception:
                pass
    _cuda_p = _find_cuda_path()
    if _cuda_p:
        _cuda_bin = os.path.join(_cuda_p, 'bin')
        if os.path.isdir(_cuda_bin):
            try:
                os.add_dll_directory(_cuda_bin)
            except Exception:
                pass
    _env_lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(_env_lib_bin):
        try:
            os.add_dll_directory(_env_lib_bin)
        except Exception:
            pass
        if _env_lib_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _env_lib_bin + os.pathsep + os.environ.get("PATH", "")


if sys.stdout is None:
    sys.stdout = DummyStream()
if getattr(sys, '__stdout__', None) is None:
    sys.__stdout__ = DummyStream()

import threading
import json
import datetime
import multiprocessing

is_mp_child = is_worker_process()

_redirected = False

def _get_process_display_name():
    try:
        proc_name = multiprocessing.current_process().name
        role = os.environ.get('MOLMEM_WORKER_ROLE')
        if not role:
            if os.environ.get("MOLMEM_IS_CALIBRATION_WORKER") == "1":
                role = "Numba calibration worker"
            elif "line_profiler.py" in " ".join(sys.argv):
                role = "Line Profiler Generator"
            elif proc_name == 'MainProcess':
                if os.environ.get('MOLMEM_REDIRECTED') == '1':
                    role = "Main Simulation Process (ROCm Env)"
                else:
                    role = "Main Simulation Process"
        if role:
            return f"{proc_name} | Role: {role}"
        return proc_name
    except Exception:
        return "Process"

_proc_name_cache = _get_process_display_name()
_import_records = []
_import_lock = threading.Lock()

class TorchImportTracer:
    _local = threading.local()

    def find_spec(self, fullname, path, target=None):
        if not is_logging_enabled():
            return None
        is_critical = fullname == "torch" or fullname.startswith("torch.") or fullname == "molmem_lib" or fullname.startswith("molmem_lib.")
        if not is_critical:
            return None
            
        if getattr(TorchImportTracer._local, "active", False):
            return None
        TorchImportTracer._local.active = True
        
        try:
            pid = os.getpid()
            ppid = os.getppid() if hasattr(os, 'getppid') else 0
            proc_name = _proc_name_cache
                
            # Get caller frame details using raw code objects (zero imports, zero disk I/O)
            caller_file = "Unknown"
            caller_line = 0
            caller_name = "Unknown"
            try:
                frame = sys._getframe()
                while frame:
                    f_code = frame.f_code
                    f_name = f_code.co_filename
                    if "importlib" not in f_name and "__init__.py" not in f_name:
                        caller_file = f_name
                        caller_line = frame.f_lineno
                        caller_name = f_code.co_name
                        break
                    frame = frame.f_back
            except Exception:
                pass
                
            now_str = datetime.datetime.now().isoformat()
            record = (now_str, pid, ppid, proc_name, fullname, caller_file, caller_line, caller_name)
            with _import_lock:
                _import_records.append(record)
        except Exception:
            pass
        finally:
            TorchImportTracer._local.active = False
            
        return None

def _flush_import_records():
    if not is_logging_enabled() or not _import_records:
        return
    try:
        log_dir = get_log_dir(auto_create=True)
        import_log_file = os.path.join(log_dir, "import_log.txt")
        with _import_lock:
            records = list(_import_records)
            _import_records.clear()
        if records:
            with open(import_log_file, "a", encoding="utf-8") as f:
                for rec in records:
                    now_str, pid, ppid, proc_name, fullname, caller_file, caller_line, caller_name = rec
                    f.write(f"[{now_str}] [PID {pid}] [Parent PID {ppid}] [{proc_name}] "
                            f"Imported: {fullname} | Called from {caller_file}:{caller_line} in {caller_name}\n")
    except Exception:
        pass

import signal
import weakref

_active_simulators = weakref.WeakSet()

def register_simulator(sim):
    try:
        _active_simulators.add(sim)
    except Exception:
        pass

import atexit
atexit.register(_flush_import_records)

def _cleanup_all_temp_history():
    try:
        for sim in list(_active_simulators):
            try:
                if hasattr(sim, 'clearHistory'):
                    sim.clearHistory()
                elif hasattr(sim, '_disk_history_store') and sim._disk_history_store is not None:
                    sim._disk_history_store.cleanup()
                    sim._disk_history_store = None
            except Exception:
                pass
        import gc
        gc.collect()
        cleanup_temp_history_dir(force_all=True)
    except Exception:
        pass
atexit.register(_cleanup_all_temp_history)

# Configure Matplotlib backend safely across platforms and environments
try:
    import matplotlib
    from .sys_utils import is_interactive_notebook
    if is_interactive_notebook():
        try:
            from IPython import get_ipython
            _ipy = get_ipython()
            if _ipy is not None:
                try:
                    _ipy.enable_matplotlib('inline')
                except Exception:
                    _ipy.run_line_magic('matplotlib', 'inline')
            else:
                matplotlib.use('module://matplotlib_inline.backend_inline')
        except Exception:
            try:
                matplotlib.use('module://matplotlib_inline.backend_inline')
            except Exception:
                try:
                    matplotlib.use('inline')
                except Exception:
                    pass
    else:
        if sys.platform == 'win32':
            matplotlib.use('TkAgg')
        elif sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            matplotlib.use('Agg')
except Exception:
    pass

# Global list to track active plotter child processes for cleanup at exit
_active_plot_processes = []

# Uniformly configure environment variables, threading budgets, and Numba workqueue layer
configure_default_threading()

# Clear the logs and sweep orphaned temp files at the start of every new run
if not is_worker_process():
    try:
        cleanup_temp_history_dir(force_all=True, allow_current_pid=True)
    except Exception:
        pass

if is_logging_enabled() and not is_mp_child and os.environ.get("MOLMEM_REDIRECTED") != "1":
    try:
        log_dir = get_log_dir(auto_create=True)
        
        # 1. Reset Error Log
        _err_log = os.path.join(log_dir, "error_log.txt")
        with open(_err_log, "w", encoding="utf-8") as _f:
            _f.write("==================================================\n")
            _f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
            _f.write("==================================================\n")
            _f.write("Status          : Running...\n")
            _f.write("==================================================\n\n")
            
        # 2. Reset Import Log
        _imp_log = os.path.join(log_dir, "import_log.txt")
        if os.path.exists(_imp_log):
            with open(_imp_log, "w", encoding="utf-8") as _f:
                _f.write("")
                
        # 3. Reset Profile Logs
        for _file in os.listdir(log_dir):
            if _file.startswith("profile_log") or _file.startswith(".temp_profile_") or _file == ".aggregate_profile.pkl":
                try:
                    os.remove(os.path.join(log_dir, _file))
                except Exception:
                    pass
    except Exception:
        pass

_sigint_in_progress = False

def _sigint_handler(signum, frame):
    global _sigint_in_progress
    if _sigint_in_progress:
        return
    _sigint_in_progress = True
    
    sys.stderr.write("\n[!] Ctrl+C detected. Shutting down active thread pools and child processes...\n")
    sys.stderr.flush()
    
    for sim in list(_active_simulators):
        try:
            for attr_name in ['_write_back_executor', 'executor']:
                if hasattr(sim, attr_name):
                    exec_obj = getattr(sim, attr_name)
                    if exec_obj is not None:
                        exec_obj.shutdown(wait=False)
            if hasattr(sim, 'clearHistory'):
                sim.clearHistory()
        except Exception:
            pass

    try:
        from .sys_utils import kill_child_processes
        kill_child_processes()
    except Exception:
        pass

    try:
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.synchronize(i)
                    except Exception:
                        pass
                torch.cuda.synchronize()
                sys.stderr.write("[+] GPU synchronized.\n")
                sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[-] GPU synchronization failed during SIGINT: {e}\n")
        sys.stderr.flush()
    os._exit(1)

if not is_interactive_notebook():
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except ValueError:
        pass

    if sys.platform == 'win32':
        try:
            import ctypes
            BOOL = ctypes.c_long
            DWORD = ctypes.c_ulong
            PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(BOOL, DWORD)
            
            def win32_ctrl_handler(ctrl_type):
                if ctrl_type in (0, 1):
                    _sigint_handler(signal.SIGINT, None)
                    return 1
                return 0
                
            global _win32_ctrl_handler_ref
            _win32_ctrl_handler_ref = PHANDLER_ROUTINE(win32_ctrl_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_win32_ctrl_handler_ref, 1)
        except Exception:
            pass

# Force instant clean exit on normal termination to bypass PyTorch/ROCm driver shutdown deadlocks on Windows
import atexit
_should_force_exit = False

def _force_clean_exit():
    if _redirected:
        return
    try:
        from .sys_utils import is_interactive_notebook
        if is_interactive_notebook():
            return
    except Exception:
        pass
    try:
        from .sys_utils import kill_child_processes
        kill_child_processes()
    except Exception:
        pass

    if sys.platform == 'win32':
        is_gpu_initialized = False
        try:
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch is not None and hasattr(torch, "cuda") and torch.cuda is not None:
                    if torch.cuda.is_initialized():
                        is_gpu_initialized = True
        except Exception:
            pass
            
        if is_gpu_initialized or os.environ.get("MOLMEM_FORCE_GPU") == "1":
            try:
                import ctypes
                k = ctypes.WinDLL('kernel32')
                k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                k.TerminateProcess(k.GetCurrentProcess(), 0)
            except Exception:
                pass
            os._exit(0)

def register_clean_exit():
    global _should_force_exit
    try:
        # Check if PyTorch GPU context is loaded to target only GPU runs on Windows
        if sys.platform == 'win32' and "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch is not None and hasattr(torch, "cuda") and torch.cuda is not None:
                if torch.cuda.is_available():
                    _should_force_exit = True
    except Exception:
        pass
    try:
        atexit.unregister(_force_clean_exit)
    except Exception:
        pass
    atexit.register(_force_clean_exit)

register_clean_exit()

# Force TensorFlow environment variables to disable oneDNN and limit threadpools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import multiprocessing
import shutil

_cache_lock = threading.Lock()
_cached_unified_data = None

def _load_unified_cache(force=False):
    """
    Safely reads the unified molmem_cache.npz file, returning a dictionary of all arrays.
    Handles read contention under concurrency with retry backoffs.
    """
    global _cached_unified_data
    if not force and _cached_unified_data is not None:
        return _cached_unified_data
        
    import numpy as np
    import time
    import random
    
    cache_file = os.path.join(os.path.dirname(__file__), '.calcache', 'molmem_cache.npz')
    if not os.path.exists(cache_file):
        return {}
        
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with np.load(cache_file) as loaded:
                # Copy data out of the loaded npz file context in C to avoid lazy evaluation issues
                data = dict(loaded)
            _cached_unified_data = data
            return data
        except Exception:
            time.sleep(0.02 + random.random() * 0.05)
            
    # If unpickling / reading completely fails after all retries, the file is corrupted
    try:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            add_system_notification(
                event="Corrupted Hardware Cache Recovered",
                action="Purged invalid .calcache/molmem_cache.npz binary stream",
                reason="File corruption or truncated write detected on disk; triggered fresh self-healing calibration"
            )
    except Exception:
        pass
    return {}

_MACHINE_SPECIFIC_CACHE_KEYS = {
    'config_json', 'machine_id', 'lib_footprint', 'best_cores_pool',
    'cached_sizes_col', 'cached_threads_col', 'cached_sparse_col', 'gpu_crossover_dim_col',
    'opt_gpu_sizes_col', 'opt_gpu_slice_limits_col', 'opt_gpu_read_slice_limits_col',
    'opt_gpu_slice_limit_col', 'opt_gpu_read_slice_limit_col',
    'cached_sizes_dev', 'cached_threads_dev', 'cached_sparse_dev', 'gpu_crossover_dim_dev',
    'opt_gpu_sizes_dev', 'opt_gpu_slice_limits_dev', 'opt_gpu_read_slice_limits_dev',
    'opt_gpu_slice_limit_dev', 'opt_gpu_read_slice_limit_dev',
    'cached_sizes_circ', 'cached_threads_circ', 'cached_sparse_circ', 'gpu_crossover_dim_circ',
    'opt_cpu_chunk_per_thread', 'opt_gpu_slice_per_cu',
    'opt_hybrid_pacing_ratio', 'opt_hybrid_min_horizon', 'opt_hybrid_cpu_allowance',
    'opt_hybrid_ranks', 'opt_hybrid_ratios'
}

def _save_unified_cache(updates, purge_keys=None):
    """
    Saves updates to the unified molmem_cache.npz file using a read-merge-write sequence.
    Guarantees thread-safe and process-safe atomic file replacement.
    Optionally evicts purge_keys before merging to ensure clean machine transitions.
    """
    import numpy as np
    import time
    import random
    global _cached_unified_data
    
    cache_dir = os.path.join(os.path.dirname(__file__), '.calcache')
    cache_file = os.path.join(cache_dir, 'molmem_cache.npz')
    lock_dir = os.path.join(cache_dir, 'molmem_cache.lock')
    
    with _cache_lock:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except (OSError, PermissionError):
            return
        
        # 1. Acquire cross-process lock directory atomically at the OS level
        lock_acquired = False
        max_lock_attempts = 150  # Up to ~7.5 seconds of total backoff
        for lock_attempt in range(max_lock_attempts):
            try:
                os.mkdir(lock_dir)
                lock_acquired = True
                break
            except OSError:
                # Check for stale lock (older than 10 seconds)
                try:
                    mtime = os.stat(lock_dir).st_mtime
                    if time.time() - mtime > 10.0:
                        try:
                            os.rmdir(lock_dir)
                            # Retry immediately after clearing stale lock
                            continue
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.02 + random.random() * 0.03)
                
        tmp_file = os.path.join(cache_dir, f"molmem_cache.tmp.{os.getpid()}.{threading.get_ident()}.npz")
        
        try:
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    # Read-merge-write
                    existing_data = {}
                    if os.path.exists(cache_file):
                        try:
                            with np.load(cache_file) as loaded:
                                existing_data = dict(loaded)
                        except Exception:
                            time.sleep(0.02 + random.random() * 0.05)
                            continue
                    
                    # Evict any keys requested for purging (e.g. on host signature mismatch)
                    if purge_keys:
                        for pk in purge_keys:
                            existing_data.pop(pk, None)

                    # Merge the updates
                    merged = {**existing_data, **updates}
                    
                    # Write to tmp file
                    np.savez_compressed(tmp_file, **merged)
                    
                    if os.path.exists(tmp_file):
                        os.replace(tmp_file, cache_file)
                        _cached_unified_data = merged
                        
                        # Update programmer class cache in sync if present
                        try:
                            from .programmer import CrossbarProgrammer
                            CrossbarProgrammer._cached_data = merged
                        except ImportError:
                            pass
                    break
                except Exception:
                    try:
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                    except Exception:
                        pass
                    time.sleep(0.02 + random.random() * 0.05)
        finally:
            # Release lock directory
            if lock_acquired:
                try:
                    os.rmdir(lock_dir)
                except Exception:
                    pass

_system_notifications = []

def add_system_notification(event, action, reason):
    global _system_notifications
    _system_notifications.append({
        "event": event,
        "action": action,
        "reason": reason
    })

def _flush_system_notifications():
    global _system_notifications
    if not _system_notifications:
        return
        
    cols, _ = shutil.get_terminal_size(fallback=(105, 24))
    w = max(60, min(105, cols - 2))
    border = "=" * w
    
    import textwrap
    print(f"\n{border}")
    print("MOLMEM SYSTEM NOTIFICATIONS".center(w))
    print(border)
    for idx, notif in enumerate(_system_notifications, 1):
        ev_lines = textwrap.wrap(f"Event  : {notif['event']}", width=w - 10)
        print(f"  [{idx}] {ev_lines[0]}")
        for extra in ev_lines[1:]:
            print(f"        {extra}")
        act_lines = textwrap.wrap(f"Action : {notif['action']}", width=w - 10)
        for act in act_lines:
            print(f"      {act}")
        rsn_lines = textwrap.wrap(f"Reason : {notif['reason']}", width=w - 10)
        for rsn in rsn_lines:
            print(f"      {rsn}")
        if idx < len(_system_notifications):
            print("  " + "-" * (w - 4))
    print(f"{border}\n")
    _system_notifications.clear()

def clear_cache(deep=False):
    """
    Globally purges all saved Hardware Matrix Calibration profiles and 
    Profiler Config caches off of the local hard-drive. Forces the simulator
    to run fresh hardware re-profiles and re-calibrations on the next launch.

    If deep=True (or via CLI --clean-dist / --deep-clean-cache / --sanitize-dist):
      Performs an exhaustive zero-imprint sanitization for safe library distribution:
      - Purges intermediate C++/HIP compiler build artifacts (csrc/build, *.egg-info)
        to remove local usernames and hardcoded filesystem paths.
      - Removes compiled native binaries (*.pyd, *.so) and build hashes (.csrc_hash, .csrc_arch).
      - Resets/truncates runtime execution logs and profiles in molmem_lib/logs/.
      - Purges all __pycache__ bytecode directories across the library tree.
    """
    global _cached_unified_data
    _cached_unified_data = None

    try:
        clear_gpu_ready_cache()
    except Exception:
        pass
    try:
        from .torch_simulator import clear_compiled_archs_cache
        clear_compiled_archs_cache()
    except Exception:
        pass
    
    try:
        from .programmer import CrossbarProgrammer
        CrossbarProgrammer._cached_data = None
        CrossbarProgrammer._resolved_caches = {}
    except Exception:
        pass
        
    try:
        from .profiler import SimulatorProfilerMixin
        SimulatorProfilerMixin._cached_configs = {
            "Column": {},
            "Device": {},
            "Circuit": {}
        }
        SimulatorProfilerMixin._gpu_slicing_configs = {}
        SimulatorProfilerMixin._gpu_crossover_dim = {
            "Column": None,
            "Device": None,
            "Circuit": None
        }
        SimulatorProfilerMixin._opt_cpu_chunk_per_thread = None
        SimulatorProfilerMixin._opt_gpu_slice_per_cu = None
        SimulatorProfilerMixin._opt_hybrid_pacing_ratio = None
        SimulatorProfilerMixin._opt_hybrid_min_horizon = None
        SimulatorProfilerMixin._opt_hybrid_cpu_allowance = None
        SimulatorProfilerMixin._opt_hybrid_pacing_by_rank = {}
    except Exception:
        pass
        
    try:
        from .simulator import _GLOBAL_SCHEDULE_CACHE, _GLOBAL_SOURCE_METADATA_CACHE, _GLOBAL_NODE_MAP_CACHE
        _GLOBAL_SCHEDULE_CACHE.clear()
        _GLOBAL_SOURCE_METADATA_CACHE.clear()
        _GLOBAL_NODE_MAP_CACHE.clear()
    except Exception:
        pass
    
    # Check the actual library installation directory's cache folder
    lib_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), '.calcache'))
    lib_dir = os.path.abspath(os.path.dirname(__file__))
    if lib_cache.startswith(lib_dir) and os.path.basename(lib_cache) == ".calcache" and os.path.exists(lib_cache):
        try:
            from .sys_utils import _fast_rmtree
            _fast_rmtree(lib_cache)
        except Exception:
            shutil.rmtree(lib_cache, ignore_errors=True)
            
    # Purge Numba JIT compiled bytecode (.nbc and .nbi) from __pycache__
    lib_pycache = os.path.join(os.path.dirname(__file__), '__pycache__')
    if os.path.exists(lib_pycache):
        try:
            import glob
            for nbc_file in glob.glob(os.path.join(lib_pycache, '*.nbc')) + glob.glob(os.path.join(lib_pycache, '*.nbi')):
                try:
                    os.remove(nbc_file)
                except Exception:
                    pass
        except Exception:
            pass

    if deep:
        # 1. Purge intermediate C++/HIP compiler build artifacts and egg metadata
        csrc_dir = os.path.abspath(os.path.join(lib_dir, 'csrc'))
        if os.path.isdir(csrc_dir):
            for sub_name in ['build', 'molmem_cuda_solver.egg-info']:
                sub_target = os.path.join(csrc_dir, sub_name)
                if os.path.isdir(sub_target):
                    try:
                        from .sys_utils import _fast_rmtree
                        _fast_rmtree(sub_target)
                    except Exception:
                        shutil.rmtree(sub_target, ignore_errors=True)
            
            # Remove compiler architecture target and MD5 hash files
            for hash_name in ['.csrc_hash', '.csrc_arch']:
                hash_path = os.path.join(csrc_dir, hash_name)
                if os.path.exists(hash_path):
                    try:
                        if sys.platform == 'win32':
                            try:
                                import ctypes
                                ctypes.windll.kernel32.SetFileAttributesW(str(hash_path), 0x80)
                            except Exception:
                                pass
                        os.remove(hash_path)
                    except Exception:
                        pass
                        
            # Remove any compiled binaries (*.pyd, *.so, *.lib, *.exp)
            for search_d in [csrc_dir, lib_dir]:
                if os.path.isdir(search_d):
                    for fname in os.listdir(search_d):
                        if fname.startswith('molmem_cuda_solver') and (fname.endswith('.pyd') or fname.endswith('.so') or fname.endswith('.lib') or fname.endswith('.exp')):
                            try:
                                os.remove(os.path.join(search_d, fname))
                            except Exception:
                                pass

        # 2. Reset / sanitize runtime logs to clean state
        logs_dir = os.path.abspath(os.path.join(lib_dir, 'logs'))
        if os.path.isdir(logs_dir):
            for sub_name in ['simulation_profiles', 'gpu_profile']:
                sub_target = os.path.join(logs_dir, sub_name)
                if os.path.isdir(sub_target):
                    try:
                        from .sys_utils import _fast_rmtree
                        _fast_rmtree(sub_target)
                    except Exception:
                        shutil.rmtree(sub_target, ignore_errors=True)
            for log_file in os.listdir(logs_dir):
                log_fpath = os.path.join(logs_dir, log_file)
                if os.path.isfile(log_fpath) and (log_file.endswith('.log') or log_file.endswith('.txt')):
                    try:
                        with open(log_fpath, 'w', encoding='utf-8') as f:
                            f.truncate(0)
                    except Exception:
                        pass

        # 3. Purge all __pycache__ directories across the library tree
        for root_p, dirnames, _ in os.walk(lib_dir, topdown=False):
            for d in dirnames:
                if d == '__pycache__':
                    try:
                        shutil.rmtree(os.path.join(root_p, d), ignore_errors=True)
                    except Exception:
                        pass

    # Log cache reset event safely without desynchronizing open file handles
    try:
        from .sys_utils import is_logging_enabled, log_to_run_log_only
        if is_logging_enabled():
            msg = "[CACHE_RESET] Deep distribution sanitization completed via clear_cache(deep=True)" if deep else "[CACHE_RESET] Run log reset via clear_cache()"
            log_to_run_log_only(msg)
    except Exception:
        pass

    add_system_notification(
        event="Hardware Calibration Cache Reset Requested" if not deep else "Deep Distribution Cache Sanitization Completed",
        action="Purged all hardware matrix calibration profiles and JIT bytecode from .calcache and __pycache__" if not deep else "Purged .calcache, intermediate build artifacts, compiled extension binaries, build hashes, logs, and all bytecode caches",
        reason="Manual user request or code modification requires fresh hardware re-calibration" if not deep else "Sanitized library distribution for zero-imprint host privacy"
    )

def sanitize_distribution():
    """
    Globally purges all hardware calibration profiles, intermediate C++ build artifacts,
    compiled extension binaries, build hashes, logs, and bytecode caches across
    the library to guarantee zero host-identifying imprints before distribution.
    """
    return clear_cache(deep=True)

# --- Environment Validation Check ---

def _summarize_import_log():
    if _redirected:
        return
    _flush_import_records()
    import os
    import re
    import datetime
    from .sys_utils import is_logging_enabled, get_log_dir
    if not is_logging_enabled():
        return
    
    log_dir = get_log_dir(auto_create=False)
    log_file = os.path.join(log_dir, "import_log.txt")
    summary_file = os.path.join(log_dir, "import_log_summary.txt")
    
    if not os.path.exists(log_file):
        return
        
    pid_data = {}
    line_re = re.compile(
        r'^\[.*?\]\s+\[PID\s+(\d+)\]\s+\[Parent\s+PID\s+(\d+)\]\s+\[(.*?)\]\s+Imported:\s+([^\s|]+)'
    )
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                m = line_re.match(line)
                if m:
                    pid = m.group(1)
                    ppid = m.group(2)
                    proc_name = m.group(3)
                    module = m.group(4)
                    
                    if module == "torch" or module.startswith("torch."):
                        norm_mod = "torch"
                    elif module.startswith("molmem_lib."):
                        parts = module.split(".")
                        norm_mod = f"molmem_lib.{parts[1]}"
                    else:
                        norm_mod = module.split(".")[0]
                        
                    key = (pid, ppid, proc_name)
                    if key not in pid_data:
                        pid_data[key] = set()
                    pid_data[key].add(norm_mod)
    except Exception:
        return
        
    try:
        with open(summary_file, "w", encoding="utf-8") as sf:
            sf.write("==================================================\n")
            sf.write("              MOLMEM IMPORT LOG SUMMARY             \n")
            sf.write("==================================================\n")
            sf.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
            sf.write("==================================================\n\n")
            
            sorted_keys = sorted(pid_data.keys(), key=lambda k: (k[2], int(k[0])))
            for key in sorted_keys:
                pid, ppid, proc_name = key
                if " | Role: " in proc_name:
                    name_part, role_part = proc_name.split(" | Role: ", 1)
                    display_name = f"{name_part} ({role_part})"
                else:
                    display_name = proc_name
                sf.write(f"[Process: {display_name}] [PID {pid}] [Parent PID {ppid}]\n")
                sorted_imports = sorted(pid_data[key])
                for imp in sorted_imports:
                    sf.write(f"  - {imp}\n")
                sf.write("\n")
    except Exception:
        pass

if not is_mp_child:
    import atexit
    atexit.register(_summarize_import_log)

if is_mp_child:
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
        except Exception:
            pass
    else:
        import signal
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass

def _print_system_info_banner(gpus):
    import platform
    import shutil
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        cpu_phys = psutil.cpu_count(logical=False)
    except Exception:
        ram_gb = "N/A"
        cpu_phys = "N/A"
        
    cols, _ = shutil.get_terminal_size(fallback=(90, 24))
    w = max(60, cols - 2)
    border = "=" * w
    gpu_str = ", ".join(gpus) if gpus else "None Detected (CPU Execution Mode)"
    
    sys.stdout.write(f"\n{border}\n")
    sys.stdout.write(f"  MOLMEM PROVISIONING ENGINE - SYSTEM DIAGNOSTICS\n")
    sys.stdout.write(f"{border}\n")
    sys.stdout.write(f"  OS Platform       : {platform.system()} {platform.release()} ({platform.machine()})\n")
    sys.stdout.write(f"  Python Version    : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
    sys.stdout.write(f"  CPU Cores         : {os.cpu_count() or 1} Logical | {cpu_phys} Physical\n")
    sys.stdout.write(f"  System RAM        : {ram_gb} GB\n")
    sys.stdout.write(f"  Detected GPUs     : {gpu_str}\n")
    sys.stdout.write(f"  Active Python     : {sys.executable}\n")
    sys.stdout.write(f"{border}\n\n")
    sys.stdout.flush()

def _detect_installed_cuda_version(cuda_p, search_path):
    detected_cuda_ver = None
    if cuda_p:
        import re
        v_json = os.path.join(cuda_p, "version.json")
        if os.path.isfile(v_json):
            try:
                import json
                with open(v_json, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                    detected_cuda_ver = v_data.get("cuda", {}).get("version", None)
            except Exception:
                pass
        if not detected_cuda_ver:
            v_txt = os.path.join(cuda_p, "version.txt")
            if os.path.isfile(v_txt):
                try:
                    with open(v_txt, "r", encoding="utf-8") as f:
                        m = re.search(r"(\d+\.\d+)", f.read())
                        if m:
                            detected_cuda_ver = m.group(1)
                except Exception:
                    pass
        if not detected_cuda_ver:
            m = re.search(r"[vV](\d+\.\d+)", cuda_p)
            if m:
                detected_cuda_ver = m.group(1)

    if not detected_cuda_ver:
        try:
            import subprocess, re, shutil
            nvcc_cmd = shutil.which("nvcc", path=search_path) or (os.path.join(cuda_p, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc") if cuda_p else None)
            if nvcc_cmd and os.path.exists(nvcc_cmd):
                out = subprocess.check_output([nvcc_cmd, "--version"], stderr=subprocess.DEVNULL, timeout=2.0, text=True)
                m = re.search(r"release (\d+\.\d+)", out)
                if m:
                    detected_cuda_ver = m.group(1)
        except Exception:
            pass

    return detected_cuda_ver

def _get_missing_compiler_items(has_amd, has_nvidia):
    import shutil
    venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "molmem-env", "Scripts" if sys.platform == "win32" else "bin"))
    search_path = os.environ.get("PATH", "")
    if os.path.exists(venv_bin):
        search_path = venv_bin + os.pathsep + search_path

    rocm_p = _find_rocm_path() if has_amd else None
    cuda_p = _find_cuda_path() if has_nvidia else None
    if cuda_p:
        c_bin = os.path.join(cuda_p, "bin")
        if os.path.exists(c_bin) and c_bin not in search_path:
            search_path = c_bin + os.pathsep + search_path
            if c_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = c_bin + os.pathsep + os.environ.get("PATH", "")

    has_hipcc = (shutil.which("hipcc", path=search_path) is not None) or (
        rocm_p is not None and (
            os.path.exists(os.path.join(rocm_p, "bin", "hipcc.exe")) or 
            os.path.exists(os.path.join(rocm_p, "bin", "hipcc.bat")) or
            os.path.exists(os.path.join(rocm_p, "bin", "hipcc"))
        )
    )
    has_nvcc = (shutil.which("nvcc", path=search_path) is not None) or (
        cuda_p is not None and (
            os.path.exists(os.path.join(cuda_p, "bin", "nvcc")) or
            os.path.exists(os.path.join(cuda_p, "bin", "nvcc.exe"))
        )
    )
    has_cl = (shutil.which("cl", path=search_path) is not None) or (_find_vcvars64() is not None)
    has_gcc = shutil.which("gcc", path=search_path) is not None or shutil.which("g++", path=search_path) is not None
    has_ninja = shutil.which("ninja", path=search_path) is not None
    has_hip_env = any(os.environ.get(v) and os.path.exists(os.environ.get(v, "")) for v in ["HIP_PATH", "ROCM_PATH", "HIP_DIR"]) or (rocm_p is not None and os.path.exists(rocm_p))
    
    missing_items = []
    advice = []
    
    if sys.platform == "win32":
        if not has_cl:
            missing_items.append("MSVC C++ Compiler (cl.exe)")
            advice.append("Install Visual Studio 2022 Build Tools with 'Desktop development with C++' workload.")
        if has_amd and (not has_hipcc or not has_hip_env):
            missing_items.append("AMD ROCm SDK for Windows (v7.x)")
            advice.append("Install AMD ROCm SDK for Windows from amd.com (provides hipcc compiler & sets HIP_PATH).")
        if has_nvidia:
            expected_cuda = "13.3"
            if not has_nvcc:
                missing_items.append("NVIDIA CUDA Toolkit (v13.3 - v13.x)")
                advice.append("Install NVIDIA CUDA Toolkit (v13.3 through v13.x, < 14.0) from developer.nvidia.com.")
            else:
                detected_ver = _detect_installed_cuda_version(cuda_p, search_path)
                if detected_ver:
                    try:
                        maj, min_ver = [int(x) if x.isdigit() else 0 for x in detected_ver.split(".")[:2]]
                        if (maj, min_ver) < (13, 3):
                            missing_items.append(f"NVIDIA CUDA Toolkit v13.3 - v13.x (Detected v{detected_ver}, but requires v13.3 or newer)")
                            advice.append("Install NVIDIA CUDA Toolkit v13.3 - v13.x from developer.nvidia.com or point CUDA_PATH to CUDA v13.3.")
                        elif maj >= 14:
                            missing_items.append(f"Supported NVIDIA CUDA Toolkit (Detected v{detected_ver}, but CUDA 14+ is unsupported; requires v13.3 - v13.x)")
                            advice.append("Install NVIDIA CUDA Toolkit v13.3 - v13.x from developer.nvidia.com or point CUDA_PATH to CUDA v13.x.")
                    except Exception:
                        pass
        if not has_ninja:
            missing_items.append("Ninja Build Accelerator (ninja.exe)")
            advice.append("Install ninja for fast parallel extension builds.")
    else:
        if sys.platform == 'darwin':
            if not has_gcc and shutil.which("clang") is None and shutil.which("clang++") is None:
                missing_items.append("Apple Clang / GCC Compiler")
                advice.append("Install Xcode Command Line Tools via 'xcode-select --install' or Homebrew.")
            if has_nvidia and not has_nvcc:
                missing_items.append("NVIDIA CUDA Compiler (nvcc)")
                advice.append("CUDA is not natively supported on modern macOS (Apple Silicon). Use CPU/Numba JIT mode.")
        else:
            if not has_gcc:
                missing_items.append("GCC/G++ Compiler")
                advice.append("Run 'sudo apt install build-essential' (Ubuntu/Debian) or equivalent package manager command.")
            if has_amd and not has_hipcc:
                missing_items.append("AMD ROCm HIP Compiler (hipcc)")
                advice.append("Run 'sudo apt install rocm-dev' (Linux ROCm package).")
            if has_nvidia:
                expected_cuda = "13.3"
                if not has_nvcc:
                    missing_items.append("NVIDIA CUDA Compiler (nvcc v13.3 - v13.x)")
                    advice.append("Run 'sudo apt update && sudo apt install -y nvidia-cuda-toolkit' (or export PATH=/usr/local/cuda/bin:$PATH)")
                else:
                    detected_ver = _detect_installed_cuda_version(cuda_p, search_path)
                    if detected_ver:
                        try:
                            maj, min_ver = [int(x) if x.isdigit() else 0 for x in detected_ver.split(".")[:2]]
                            if (maj, min_ver) < (13, 3):
                                missing_items.append(f"NVIDIA CUDA Toolkit v13.3 - v13.x (Detected v{detected_ver}, but requires v13.3 or newer)")
                                advice.append("Install NVIDIA CUDA Toolkit v13.3 - v13.x (or export PATH=/usr/local/cuda/bin:$PATH).")
                            elif maj >= 14:
                                missing_items.append(f"Supported NVIDIA CUDA Toolkit (Detected v{detected_ver}, but CUDA 14+ is unsupported; requires v13.3 - v13.x)")
                                advice.append("Install NVIDIA CUDA Toolkit v13.3 - v13.x (or export PATH=/usr/local/cuda/bin:$PATH).")
                        except Exception:
                            pass

    return missing_items, advice

def _check_compiler_toolchain(has_amd, has_nvidia, quiet=False):
    if not has_amd and not has_nvidia:
        if not quiet:
            sys.stdout.write(f"  [+] CPU Execution Mode active (No GPU detected). Numba JIT compiler ready.\n\n")
            sys.stdout.flush()
        return True

    missing_items, advice = _get_missing_compiler_items(has_amd, has_nvidia)

    if missing_items:
        if not quiet:
            import shutil
            cols, _ = shutil.get_terminal_size(fallback=(90, 24))
            w = max(60, cols - 2)
            border = "=" * w
            sys.stdout.write(f"\n{border}\n")
            sys.stdout.write(f"  MOLMEM DIAGNOSTIC: C++/HIP Extension Toolchain Requirements\n")
            sys.stdout.write(f"{border}\n")
            sys.stdout.write(f"  Note: PyTorch runtime support is loaded, but native extension compilation toolchains are missing:\n")
            for item in missing_items:
                sys.stdout.write(f"    - Missing: {item}\n")
            sys.stdout.write(f"\n  To enable native C++ extension compilation on {sys.platform}:\n")
            for adv in advice:
                sys.stdout.write(f"    * {adv}\n")
            sys.stdout.write(f"{border}\n")
            sys.stdout.write(f"  The simulator will persist using CPU/Numba JIT execution mode.\n")
            sys.stdout.write(f"{border}\n\n")
            sys.stdout.flush()
        return False
    return True

def _check_selective_package_updates(venv_python):
    import os
    import sys
    import shutil
    import subprocess
    import json
    import urllib.request

    # Strict network isolation: only check PyPI if explicitly requested via CLI flag or env var
    if os.environ.get("MOLMEM_CHECK_UPDATES") != "1":
        return
    if os.environ.get("MOLMEM_OFFLINE") == "1" or os.environ.get("UV_OFFLINE") == "1":
        return
    if os.environ.get("MOLMEM_ENV_MANAGED") == "1" or os.environ.get("MOLMEM_NO_PROVISION") == "1":
        return

    targets = {
        'numpy': '<2.5.0',
        'scipy': None,
        'numba': None,
        'tbb': None,
        'tensorflow': None,
        'matplotlib': None,
        'psutil': None,
        'ninja': None
    }
    
    target_names = list(targets.keys())
    check_code = f"import json, importlib.metadata; targets = {target_names}; print(json.dumps({{p: (importlib.metadata.version(p) if True else None) for p in targets if not [None for _ in [0] if p not in [d.metadata['Name'].lower().replace('-', '_') for d in importlib.metadata.distributions() if 'Name' in d.metadata] and not any(p == d.name.lower().replace('-', '_') for d in importlib.metadata.distributions())]}}))"
    
    installed_map = {}
    try:
        query_script = "import json, importlib.metadata; res = {};\nfor p in " + str(target_names) + ":\n    try: res[p] = importlib.metadata.version(p)\n    except Exception: res[p] = None\nprint(json.dumps(res))"
        res = subprocess.run([venv_python, "-c", query_script], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            installed_map = json.loads(res.stdout.strip())
    except Exception:
        installed_map = {}

    updates_needed = []
    
    for pkg, spec in targets.items():
        current = installed_map.get(pkg)
        if not current:
            updates_needed.append(pkg)
            continue
            
        try:
            url = f"https://pypi.org/pypi/{pkg}/json"
            req = urllib.request.Request(url, headers={'User-Agent': 'MolmemProvisioner'})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                latest = data.get('info', {}).get('version')
                if not latest:
                    continue
                    
                if pkg == 'numpy':
                    l_parts = [int(x) for x in latest.split('.')[:2] if x.isdigit()]
                    if l_parts and l_parts[0] == 2 and l_parts[1] >= 5:
                        continue
                    if current != latest and l_parts < [2, 5]:
                        updates_needed.append(f"numpy>={current},<2.5.0")
                else:
                    if current != latest and not pkg.startswith("rocm"):
                        updates_needed.append(pkg)
        except Exception:
            pass
            
    if updates_needed:
        sys.stdout.write(f"[*] Missing/Outdated Packages Detected: {', '.join(updates_needed)}\n")
        sys.stdout.write(f"[*] Applying package installations via uv...\n")
        sys.stdout.flush()
        uv_bin = shutil.which("uv")
        if uv_bin:
            cmd = [uv_bin, "pip", "install", "--python", venv_python] + updates_needed
        else:
            cmd = [venv_python, "-m", "pip", "install"] + updates_needed
        try:
            subprocess.run(cmd, check=True)
            sys.stdout.write(f"[+] Packages installed successfully.\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"[!] Package install warning: {e}\n")
            sys.stdout.flush()
def _sanitize_linux_rocm_execstack(target_python=None):
    """
    On modern Linux distributions (Ubuntu 24.04+ / glibc 2.38+), dynamic loading (dlopen)
    of shared libraries requesting an executable memory stack is strictly rejected for security.
    Certain pre-compiled PyTorch ROCm wheels ship with PT_GNU_STACK flagged as RWE (executable),
    raising: 'ImportError: cannot enable executable stack as shared object requires: Invalid argument'.
    This utility transparently detects and clears the executable bit (PF_X) from affected libraries
    in the environment, ensuring reliable PyTorch/ROCm execution without weakening OS security.
    """
    if not sys.platform.startswith('linux'):
        return

    import struct
    pt_gnu_stack = 0x6474e551

    py_exec = target_python or sys.executable
    search_dirs = []
    try:
        venv_root = os.path.dirname(os.path.dirname(os.path.abspath(py_exec)))
        for cand in [
            os.path.join(venv_root, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
            os.path.join(venv_root, "lib", "python3.12", "site-packages"),
        ]:
            if os.path.isdir(cand) and cand not in search_dirs:
                search_dirs.append(cand)
    except Exception:
        pass

    for p in sys.path:
        if p and "site-packages" in p and os.path.isdir(p) and p not in search_dirs:
            search_dirs.append(p)

    target_subpaths = ["torch/lib", "torchvision.libs", "triton/backends/amd/lib"]

    def _clear_exec_bit(filepath):
        try:
            with open(filepath, 'r+b') as f:
                magic = f.read(4)
                if magic != b'\x7fELF':
                    return False
                ei_class = f.read(1)[0]
                if ei_class != 2:
                    return False
                f.seek(32)
                e_phoff = struct.unpack('<Q', f.read(8))[0]
                f.seek(54)
                e_phentsize = struct.unpack('<H', f.read(2))[0]
                e_phnum = struct.unpack('<H', f.read(2))[0]
                for i in range(e_phnum):
                    offset = e_phoff + i * e_phentsize
                    f.seek(offset)
                    p_type = struct.unpack('<I', f.read(4))[0]
                    if p_type == pt_gnu_stack:
                        p_flags = struct.unpack('<I', f.read(4))[0]
                        if p_flags & 1:  # PF_X is set
                            f.seek(offset + 4)
                            f.write(struct.pack('<I', p_flags & ~1))
                            return True
        except Exception:
            pass
        return False

    for sdir in search_dirs:
        for sub in target_subpaths:
            tdir = os.path.join(sdir, sub)
            if os.path.isdir(tdir):
                try:
                    for fname in os.listdir(tdir):
                        if fname.endswith('.so') or '.so.' in fname:
                            fpath = os.path.join(tdir, fname)
                            if os.path.isfile(fpath):
                                _clear_exec_bit(fpath)
                except Exception:
                    pass


def _sanitize_windows_rocm_cli(target_python=None):
    """
    On Windows, AMD ROCm SDK console script entry points (offload-arch, hipcc, etc.)
    use os.execv(full_path, [str(full_path)] + sys.argv[1:]) in their _cli.py trampoline.
    Under the Windows C runtime (_wexecv), argv[0] is not enclosed in double quotes.
    If the virtual environment or user profile path contains spaces (e.g. 'C:\\Users\\Username With Spaces\\...'),
    LLVM/Clang binary argument parsers split on the whitespace, throwing:
    "Username: Unknown command line argument 'With Spaces\\...'"
    This utility inspects rocm_sdk* packages in site-packages and transparently patches _cli.py
    to use subprocess.call with sys.exit, ensuring paths containing whitespace are safely quoted.
    """
    if sys.platform != "win32":
        return

    py_exec = target_python or sys.executable
    search_dirs = []
    try:
        venv_root = os.path.dirname(os.path.dirname(os.path.abspath(py_exec)))
        for cand in [
            os.path.join(venv_root, "Lib", "site-packages"),
            os.path.join(venv_root, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
        ]:
            if os.path.isdir(cand) and cand not in search_dirs:
                search_dirs.append(cand)
    except Exception:
        pass

    for p in sys.path:
        if p and "site-packages" in p and os.path.isdir(p) and p not in search_dirs:
            search_dirs.append(p)

    target_pkgs = ["rocm_sdk_core", "rocm_sdk_devel", "rocm_sdk_libraries_custom"]
    for sdir in search_dirs:
        for pkg in target_pkgs:
            cli_path = os.path.join(sdir, pkg, "_cli.py")
            if os.path.isfile(cli_path):
                try:
                    with open(cli_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    old_snippet = "os.execv(full_path, [str(full_path)] + sys.argv[1:])"
                    if old_snippet in content and "subprocess.call" not in content:
                        new_snippet = (
                            "if sys.platform == \"win32\":\n"
                            "        import subprocess\n"
                            "        sys.exit(subprocess.call([str(full_path)] + sys.argv[1:]))\n"
                            "    os.execv(full_path, [str(full_path)] + sys.argv[1:])"
                        )
                        new_content = content.replace(old_snippet, new_snippet)
                        with open(cli_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                except Exception:
                    pass


def _sanitize_windows_torch_cpp_extension(target_python=None):
    """
    On Windows with AMD ROCm / HIP, PyTorch's torch/utils/cpp_extension.py has a bug:
    It replaces spaces with backslashes in pp_opts (include directories):
        pp_opts = ["-I{}".format(s[2:].replace(" ", "\\\\")) if s.startswith('-I') else s for s in pp_opts]
    and passes those backslash-escaped paths to MSVC cl.exe (cflags).
    Because MSVC cl.exe does not recognize backslash as an escape for space (it treats it as a directory
    separator), cl.exe fails to find headers (e.g. 'torch/extension.h') whenever the path contains spaces
    (such as 'C:\\Users\\Username With Spaces\\...').
    This utility inspects torch/utils/cpp_extension.py and ensures that MSVC cl.exe receives valid,
    un-mangled include paths while hipcc receives properly escaped paths.
    """
    if sys.platform != "win32":
        return

    py_exec = target_python or sys.executable
    search_dirs = []
    try:
        venv_root = os.path.dirname(os.path.dirname(os.path.abspath(py_exec)))
        for cand in [
            os.path.join(venv_root, "Lib", "site-packages"),
            os.path.join(venv_root, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
        ]:
            if os.path.isdir(cand) and cand not in search_dirs:
                search_dirs.append(cand)
    except Exception:
        pass

    for p in sys.path:
        if p and "site-packages" in p and os.path.isdir(p) and p not in search_dirs:
            search_dirs.append(p)

    for sdir in search_dirs:
        ext_path = os.path.join(sdir, "torch", "utils", "cpp_extension.py")
        if os.path.isfile(ext_path):
            try:
                with open(ext_path, "r", encoding="utf-8") as f:
                    content = f.read()
                old_code = (
                    "            # Replace space with \\ when using hipcc (hipcc passes includes to clang without \"\"s so clang sees space in include paths as new argument)\n"
                    "            if IS_HIP_EXTENSION:\n"
                    "                pp_opts = [\"-I{}\".format(s[2:].replace(\" \", \"\\\\\")) if s.startswith('-I') else s for s in pp_opts]"
                )
                new_code = (
                    "            # Replace space with \\ when using hipcc (hipcc passes includes to clang without \"\"s so clang sees space in include paths as new argument)\n"
                    "            hip_pp_opts = pp_opts\n"
                    "            if IS_HIP_EXTENSION:\n"
                    "                hip_pp_opts = [\"-I{}\".format(s[2:].replace(\" \", \"\\\\\")) if s.startswith('-I') else s for s in pp_opts]"
                )
                old_call = "cuda_cflags.extend(pp_opts)"
                new_call = "cuda_cflags.extend(hip_pp_opts)"
                if old_code in content and old_call in content:
                    content = content.replace(old_code, new_code, 1)
                    content = content.replace(old_call, new_call, 1)
                    with open(ext_path, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception:
                pass


def _detect_physical_gpus():
    import os
    if is_interactive_notebook() or os.environ.get("MOLMEM_CPU_ONLY") == "1":
        return []
    import subprocess
    import shutil
    ps_bin = shutil.which("powershell") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    gpus = []
    if sys.platform == "win32":
        # 1. Try clean PowerShell Get-CimInstance list call
        try:
            cmd = [ps_bin, "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5.0)
            for line in out.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    gpus.append(line)
        except Exception:
            pass

        # 2. Try PowerShell Get-WmiObject fallback
        if not gpus:
            try:
                cmd = [ps_bin, "-NoProfile", "-Command", "(Get-WmiObject Win32_VideoController).Name"]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5.0)
                for line in out.splitlines():
                    line = line.strip()
                    if line:
                        gpus.append(line)
            except Exception:
                pass

        # 3. Try Windows Registry query for Display Adapters
        if not gpus:
            try:
                cmd = ["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}", "/v", "DriverDesc", "/s"]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5.0)
                for line in out.splitlines():
                    if "DriverDesc" in line:
                        val = line.split("REG_SZ")[-1].strip()
                        if val and val not in gpus:
                            gpus.append(val)
            except Exception:
                pass

        # 4. Try nvidia-smi CLI
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True, timeout=5.0)
            for line in out.splitlines():
                if "GPU" in line:
                    gpus.append(line.strip())
        except Exception:
            pass

        # 5. Check ROCm SDK environment or directory
        if not gpus and (os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH") or os.path.exists(r"C:\Program Files\AMD\ROCm")):
            gpus.append("AMD Radeon Graphics (ROCm Hardware Accelerated)")

        # 6. Legacy wmic fallback
        if not gpus:
            try:
                out = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "Name"], stderr=subprocess.DEVNULL, text=True, timeout=5.0)
                for line in out.splitlines():
                    line = line.strip()
                    if line and line.lower() != "name":
                        gpus.append(line)
            except Exception:
                pass
    elif sys.platform == 'darwin':
        try:
            out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL, text=True, timeout=5.0)
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    chip = line.split(":", 1)[-1].strip()
                    if chip and chip not in gpus:
                        gpus.append(chip)
        except Exception:
            pass
        if not gpus:
            try:
                import platform
                if "arm" in platform.machine().lower():
                    gpus.append("Apple Silicon GPU (Metal/MPS Acceleration)")
            except Exception:
                pass
    else:
        # 1. Try lspci
        try:
            out = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL, text=True)
            for line in out.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    gpus.append(line.split(":", 2)[-1].strip())
        except Exception:
            pass

        # 2. Try Linux sysfs DRM device vendor inspection (available on all Linux distributions)
        if not gpus:
            try:
                import glob
                for vendor_file in glob.glob("/sys/class/drm/card*/device/vendor"):
                    with open(vendor_file, "r") as f:
                        vid = f.read().strip().lower()
                    if vid in ["0x10de", "0x1002", "0x8086"]:
                        dev_name_file = os.path.join(os.path.dirname(vendor_file), "uevent")
                        vendor_map = {"0x10de": "NVIDIA Graphics Device", "0x1002": "AMD Radeon Graphics", "0x8086": "Intel Graphics"}
                        dev_desc = vendor_map.get(vid, "Graphics Adapter")
                        if os.path.exists(dev_name_file):
                            try:
                                with open(dev_name_file, "r") as f_u:
                                    for l in f_u:
                                        if l.startswith("PCI_ID="):
                                            dev_desc += f" (PCI {l.split('=', 1)[1].strip()})"
                                            break
                            except Exception:
                                pass
                        if dev_desc not in gpus:
                            gpus.append(dev_desc)
            except Exception:
                pass

        # 3. Try command line GPU management tools
        if not gpus:
            for cli_tool, flag in [("nvidia-smi", "-L"), ("rocminfo", None), ("rocm-smi", "--showid")]:
                try:
                    cmd = [cli_tool] + ([flag] if flag else [])
                    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
                    for line in out.splitlines():
                        line = line.strip()
                        if "GPU" in line or "gfx" in line.lower():
                            if line not in gpus:
                                gpus.append(line)
                except Exception:
                    pass

        # 4. Try PyTorch initialized devices if already imported
        if not gpus and 'torch' in sys.modules:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpus.append(torch.cuda.get_device_name(i))
            except Exception:
                pass
    return gpus

def _is_dedicated_gpu(gpu_name):
    name_lower = gpu_name.lower()
    legacy_intel = ["intel(r) hd graphics", "intel hd graphics", "intel(r) uhd graphics", "intel uhd graphics", "iris", "intel graphics"]
    amd_integrated = ["radeon 610m", "radeon 660m", "radeon 680m", "radeon 740m", "radeon 760m", "radeon 780m", "radeon 880m", "radeon 890m", "vega 3", "vega 6", "vega 7", "vega 8", "vega 10", "vega 11", "apu", "amd radeon(tm) graphics", "radeon graphics", "gfx1036"]
    if any(k in name_lower for k in legacy_intel):
        return False
    if any(k in name_lower for k in amd_integrated):
        return False
    return True

def _terminate_venv_processes(venv_path):
    if not venv_path or not os.path.exists(venv_path):
        return
    try:
        import psutil
        norm_venv = os.path.normpath(venv_path).lower()
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                exe = proc.info.get('exe')
                if exe and norm_venv in os.path.normpath(exe).lower() and proc.info['pid'] != current_pid:
                    proc.kill()
            except Exception:
                pass
    except Exception:
        pass

def _resolve_uv_command(workspace_root):
    import subprocess
    import shutil
    import os
    import sys

    # 1. Search PATH and standard user installation directories
    candidate_bins = [
        shutil.which("uv"),
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
        os.path.join(workspace_root, "uv"),
        os.path.join(workspace_root, "uv.exe"),
    ]
    for cand in candidate_bins:
        if cand and os.path.isfile(cand):
            try:
                res = subprocess.run([cand, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if res.returncode == 0:
                    return [cand]
            except Exception:
                pass

    # 2. On Unix/Linux/WSL/macOS: install standalone uv via official Astral installer (zero pip, zero system impact)
    if sys.platform != "win32":
        if shutil.which("curl") and shutil.which("sh"):
            try:
                subprocess.run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            except Exception:
                pass
        elif shutil.which("wget") and shutil.which("sh"):
            try:
                subprocess.run(["sh", "-c", "wget -qO- https://astral.sh/uv/install.sh | sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
            except Exception:
                pass

        # Re-check user bins after installer
        for cand in [os.path.expanduser("~/.local/bin/uv"), os.path.expanduser("~/.cargo/bin/uv"), shutil.which("uv")]:
            if cand and os.path.isfile(cand):
                try:
                    res = subprocess.run([cand, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                    if res.returncode == 0:
                        return [cand]
                except Exception:
                    pass

        # If curl/wget was missing, download standalone uv release tarball via urllib into workspace
        try:
            import platform, urllib.request, tarfile
            arch = platform.machine().lower()
            arch_map = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}
            target_arch = arch_map.get(arch, 'x86_64')
            tar_url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target_arch}-unknown-linux-musl.tar.gz"
            local_tar = os.path.join(workspace_root, "uv.tar.gz")
            urllib.request.urlretrieve(tar_url, local_tar)
            with tarfile.open(local_tar, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/uv") or member.name == "uv":
                        member.name = "uv"
                        tar.extract(member, path=workspace_root)
                        break
            if os.path.exists(local_tar):
                os.remove(local_tar)
            extracted_uv = os.path.join(workspace_root, "uv")
            if os.path.isfile(extracted_uv):
                os.chmod(extracted_uv, 0o755)
                if subprocess.run([extracted_uv, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0:
                    return [extracted_uv]
        except Exception:
            pass

    # 3. On Windows: fallback to standard pip install (Windows does not have PEP 668 externally managed environment)
    if sys.platform == "win32":
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            for cand in [shutil.which("uv"), os.path.join(sys.prefix, "Scripts", "uv.exe")]:
                if cand and os.path.isfile(cand):
                    if subprocess.run([cand, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0:
                        return [cand]
            if subprocess.run([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0:
                return [sys.executable, "-m", "uv"]
        except Exception:
            pass

    return None

def _run_provisioning_engine():
    import sys
    import os
    import subprocess
    import shutil

    if os.environ.get("MOLMEM_REDIRECTED") == "1" or os.environ.get("MOLMEM_NO_PROVISION") == "1" or os.environ.get("MOLMEM_ENV_MANAGED") == "1":
        return

    # Detect interactive Jupyter/Colab notebook environments
    from .sys_utils import is_interactive_notebook
    is_nb = is_interactive_notebook()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_path = os.path.join(workspace_root, "molmem-env")
    
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    venv_python = os.path.join(venv_path, "Scripts" if sys.platform == "win32" else "bin", f"python{exe_suffix}")

    is_in_venv = os.path.exists(venv_python) and os.path.abspath(sys.executable) == os.path.abspath(venv_python)
    incomplete_sentinel = os.path.join(venv_path, ".incomplete")

    # Verify if venv python binary is executable on this system (detects copied/broken venvs)
    is_venv_valid = False
    if os.path.exists(venv_python):
        try:
            res = subprocess.run([venv_python, "-c", "import sys"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            is_venv_valid = (res.returncode == 0)
        except Exception:
            is_venv_valid = False

    is_first_run = not is_venv_valid

    force_fresh_env = '--fresh-env' in sys.argv
    force_fresh_download = '--fresh-download' in sys.argv or '--redownload-env' in sys.argv
    
    for flag_name in ['--fresh-env', '--fresh-download', '--redownload-env']:
        if flag_name in sys.argv:
            try:
                sys.argv.remove(flag_name)
            except Exception:
                pass

    # If running in an interactive notebook or cloud container, skip auto-recreation
    if is_nb and not force_fresh_env and not force_fresh_download:
        return

    should_recreate = force_fresh_env or force_fresh_download

    if not is_in_venv:
        if not should_recreate and is_venv_valid:
            if os.environ.get("MOLMEM_ENV_MANAGED") != "1" and os.environ.get("MOLMEM_NO_PROVISION") != "1":
                try:
                    _check_selective_package_updates(venv_python)
                    resolve_and_install_caller_dependencies(venv_python)
                except SystemExit:
                    raise
                except Exception as e:
                    sys.stderr.write(f"[!] Package updater warning: {e}\n")

            if not isMolmemLibSilenced():
                sys.stdout.write("[*] Activating 'molmem-env' environment...\n")
                sys.stdout.flush()
            os.environ["MOLMEM_REDIRECTED"] = "1"
            env = os.environ.copy()
            local_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            existing_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = local_parent + os.pathsep + existing_path if existing_path else local_parent
            venv_lib = os.path.join(venv_path, "lib")
            if os.path.isdir(venv_lib):
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                if venv_lib not in existing_ld.split(os.pathsep):
                    env["LD_LIBRARY_PATH"] = venv_lib + (os.pathsep + existing_ld if existing_ld else "")

            cmd = [venv_python] + sys.argv
            ret = subprocess.call(cmd, env=env)
            exit_code = ret if (-2147483648 <= ret <= 2147483647) else (ret & 0xFF)
            os._exit(exit_code)

        gpus = _detect_physical_gpus()
        _print_system_info_banner(gpus)

        # Mark environment incomplete until all phases succeed
        try:
            os.makedirs(venv_path, exist_ok=True)
            with open(incomplete_sentinel, "w") as f:
                f.write("1")
        except Exception:
            pass

        active_phases = []
        if should_recreate:
            active_phases.append("Environment Purge & Wheel Reset")
        if not is_venv_valid or should_recreate:
            active_phases.append("Python 3.12 Virtual Environment Provisioning")
        active_phases.append("Core Dependencies Installation")
        active_phases.append("GPU Dependencies & Runtime Setup")
        active_phases.append("Caller Dependencies & Extension Compilation")

        total_phases = len(active_phases)
        phase_idx = 1

        uv_cmd = _resolve_uv_command(workspace_root)

        if should_recreate:
            sys.stdout.write(f"[Phase {phase_idx} of {total_phases}] Environment Purge & Wheel Reset...\n")
            sys.stdout.write(f"  [-] Wiping legacy environment folder: {venv_path}\n")
            sys.stdout.flush()
            _terminate_venv_processes(venv_path)
            _fast_rmtree(venv_path)
            
            if force_fresh_download:
                sys.stdout.write(f"  [-] Purging uv package & wheel cache (forcing PyPI network re-download)...\n")
                sys.stdout.flush()
                try:
                    subprocess.run(["uv", "cache", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            phase_idx += 1

        if not is_venv_valid or should_recreate:
            sys.stdout.write(f"[Phase {phase_idx} of {total_phases}] Provisioning clean Python 3.12 virtual environment (molmem-env)...\n")
            sys.stdout.flush()

            venv_created = False
            if uv_cmd:
                try:
                    subprocess.run(uv_cmd + ["venv", "--python", "3.12", "--seed", "--allow-existing", venv_path], check=True)
                    venv_created = True
                except Exception:
                    pass
                if not venv_created:
                    try:
                        subprocess.run(uv_cmd + ["venv", "--allow-existing", venv_path], check=True)
                        venv_created = True
                    except Exception:
                        pass

            if not venv_created:
                try:
                    subprocess.run([sys.executable, "-m", "venv", "--clear", venv_path], check=True)
                    venv_created = True
                except Exception as venv_err:
                    sys.stderr.write("\n" + "=" * 80 + "\n")
                    sys.stderr.write(" [!] MOLMEM ONBOARDING FAILURE: COULD NOT INITIALIZE VIRTUAL ENVIRONMENT\n")
                    sys.stderr.write("=" * 80 + "\n")
                    sys.stderr.write(" Both 'uv' and Python's built-in 'venv' failed to create the virtual environment.\n")
                    sys.stderr.write(f" Error detail: {venv_err}\n\n")
                    sys.stderr.write(" Recommendation for Debian/Ubuntu/WSL:\n")
                    sys.stderr.write("   curl -LsSf https://astral.sh/uv/install.sh | sh\n")
                    sys.stderr.write("   OR: sudo apt update && sudo apt install -y python3-venv python3-pip\n")
                    sys.stderr.write("=" * 80 + "\n\n")
                    sys.stderr.flush()
                    sys.exit(1)
            
            sys.stdout.write(f"  [+] Phase {phase_idx} Complete: Python 3.12 environment ready.\n\n")
            sys.stdout.flush()
            phase_idx += 1

            # Mark environment incomplete after recreate
            try:
                with open(incomplete_sentinel, "w") as f:
                    f.write("1")
            except Exception:
                pass

        cache_flag = ["--no-cache"] if force_fresh_download else []
        uv_pip_base = (uv_cmd + ["pip", "install", "--python", venv_python]) if uv_cmd else [venv_python, "-m", "pip", "install"]

        sys.stdout.write(f"[Phase {phase_idx} of {total_phases}] Installing Core Dependencies via uv (numpy<2.5, scipy, numba, tbb, tensorflow, matplotlib, psutil, ninja)...\n")
        sys.stdout.flush()
        core_cmd = uv_pip_base + cache_flag + ["numpy>=1.26.0,<2.5.0", "scipy", "numba", "tbb", "tensorflow", "matplotlib", "psutil", "ninja"]
        res = subprocess.run(core_cmd, check=False)
        if res.returncode != 0:
            sys.stdout.write("  [!] Full bundle install failed; falling back to essential simulation dependencies (numpy, scipy, numba, matplotlib, psutil)...\n")
            sys.stdout.flush()
            essential_cmd = uv_pip_base + cache_flag + ["numpy>=1.26.0,<2.5.0", "scipy", "numba", "matplotlib", "psutil", "ninja"]
            subprocess.run(essential_cmd, check=True)
            for opt_pkg in ["tbb", "tensorflow", "ninja"]:
                subprocess.run(uv_pip_base + cache_flag + [opt_pkg], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        phase_idx += 1

        sys.stdout.write(f"[Phase {phase_idx} of {total_phases}] Auto-detecting GPU Hardware & Installing PyTorch Wheels via uv...\n")
        sys.stdout.flush()

        has_nvidia = any("nvidia" in g.lower() and _is_dedicated_gpu(g) for g in gpus)
        has_amd = any(any(v in g.lower() for v in ["amd", "radeon", "advanced micro devices"]) and _is_dedicated_gpu(g) for g in gpus)

        if has_nvidia:
            gpu_cmd = uv_pip_base + cache_flag + ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
            subprocess.run(gpu_cmd, check=False)
        elif has_amd:
            if sys.platform == "win32":
                py_ver = "cp312"  # Always target molmem-env Python 3.12
                sdk_cmd = uv_pip_base + cache_flag + [
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"
                ]
                subprocess.run(sdk_cmd, check=False)
                torch_cmd = uv_pip_base + cache_flag + [
                    f"https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1+rocm7.2.1-{py_ver}-{py_ver}-win_amd64.whl",
                    f"https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1+rocm7.2.1-{py_ver}-{py_ver}-win_amd64.whl",
                    f"https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1+rocm7.2.1-{py_ver}-{py_ver}-win_amd64.whl"
                ]
                res = subprocess.run(torch_cmd, check=False)
                if res.returncode != 0:
                    # Fallback to standard PyTorch if custom ROCm wheel fails
                    fallback_cmd = uv_pip_base + cache_flag + ["torch", "torchvision", "torchaudio"]
                    subprocess.run(fallback_cmd, check=False)
            else:
                gpu_cmd = uv_pip_base + cache_flag + ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/rocm7.2"]
                subprocess.run(gpu_cmd, check=False)
        else:
            cpu_index = ["--index-url", "https://download.pytorch.org/whl/cpu"] if sys.platform != "win32" else []
            cpu_cmd = uv_pip_base + cache_flag + ["torch", "torchvision", "torchaudio"] + cpu_index
            subprocess.run(cpu_cmd, check=False)

        # Ensure Linux ROCm shared libraries have non-executable stacks for Ubuntu/glibc compliance
        _sanitize_linux_rocm_execstack(venv_python)
        # Ensure Windows ROCm SDK CLI trampolines safely quote executable paths with spaces
        _sanitize_windows_rocm_cli(venv_python)
        _sanitize_windows_torch_cpp_extension(venv_python)

        phase_idx += 1

        # Resolve any caller script third-party dependencies via uv
        try:
            resolve_and_install_caller_dependencies(venv_python)
        except SystemExit:
            raise
        except Exception as e:
            sys.stderr.write(f"[!] Caller dependency resolution warning: {e}\n")

        sys.stdout.write(f"[Phase {phase_idx} of {total_phases}] Inspecting C++/HIP Extension Compiler Toolchains...\n")
        sys.stdout.flush()
        _check_compiler_toolchain(has_amd, has_nvidia)

        # Clear incomplete sentinel upon successful provisioning completion
        try:
            if os.path.exists(incomplete_sentinel):
                os.remove(incomplete_sentinel)
        except Exception:
            pass

        sys.stdout.write(f"[+] Provisioning Complete! Relaunching in 'molmem-env'...\n\n")
        sys.stdout.flush()

        os.environ["MOLMEM_REDIRECTED"] = "1"
        env = os.environ.copy()
        local_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = local_parent + os.pathsep + existing_path if existing_path else local_parent

        cmd = [venv_python] + sys.argv
        ret = subprocess.call(cmd, env=env)
        exit_code = (ret & 0xFF) if ret is not None else 0
        os._exit(exit_code)
    else:
        try:
            _sanitize_linux_rocm_execstack(sys.executable)
            _sanitize_windows_rocm_cli(sys.executable)
            _sanitize_windows_torch_cpp_extension(sys.executable)
            if os.environ.get("MOLMEM_ENV_MANAGED") != "1" and os.environ.get("MOLMEM_NO_PROVISION") != "1":
                _check_selective_package_updates(sys.executable)
                resolve_and_install_caller_dependencies(sys.executable)
        except SystemExit:
            raise
        except Exception as e:
            sys.stderr.write(f"[!] Package updater warning: {e}\n")

def _check_early_redirect():
    _sanitize_linux_rocm_execstack()
    _sanitize_windows_rocm_cli()
    _sanitize_windows_torch_cpp_extension()
    _run_provisioning_engine()

if not is_mp_child:
    try:
        _check_early_redirect()
    except Exception as e:
        sys.stderr.write(f"[!] Provisioning Engine Error: {e}\n")
        sys.stderr.flush()
        _ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _vpy = os.path.join(_ws, "molmem-env", "Scripts" if sys.platform == "win32" else "bin", "python" + (".exe" if sys.platform == "win32" else ""))
        if not (os.path.exists(_vpy) and os.path.abspath(sys.executable) == os.path.abspath(_vpy)):
            sys.exit(1)
        
    try:
        from .sys_utils import init_cli_logger
        init_cli_logger()
    except Exception:
        pass

    # Intercept --clear-cache and distribution sanitization arguments from command-line dynamically
    deep_clean_requested = any(f in sys.argv for f in ['--clean-dist', '--deep-clean-cache', '--sanitize-dist'])
    standard_clean_requested = '--clear-cache' in sys.argv

    if deep_clean_requested or standard_clean_requested:
        try:
            import multiprocessing
            is_main = (multiprocessing.current_process().name == 'MainProcess' and not is_mp_child)
            if is_main:
                clear_cache(deep=deep_clean_requested)
            for f in ['--clean-dist', '--deep-clean-cache', '--sanitize-dist', '--clear-cache']:
                while f in sys.argv:
                    sys.argv.remove(f)
        except Exception:
            pass
            
    # Intercept GPU JIT override flags from command-line dynamically
    for flag in ['--gpu-jit', '--gpujit']:
        if flag in sys.argv:
            try:
                os.environ["MOLMEM_FORCE_GPU"] = "1"
                os.environ["MOLMEM_FORCE_JIT"] = "1"
                sys.argv.remove(flag)
            except Exception:
                pass
                
    # GPU debug and blocking execution (off by default; opt-in via --debug-gpu / --gpu-debug)
    debug_gpu_requested = False
    for flag in ['--debug-gpu', '--gpu-debug']:
        if flag in sys.argv:
            debug_gpu_requested = True
            try:
                sys.argv.remove(flag)
            except Exception:
                pass
    if debug_gpu_requested:
        os.environ["MOLMEM_DEBUG_GPU"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.setdefault("MOLMEM_DEBUG_GPU", "0")

    # Intercept GPU override flags from command-line dynamically
    for flag in ['--gpu', '--force-gpu']:
        if flag in sys.argv:
            try:
                os.environ["MOLMEM_FORCE_GPU"] = "1"
                sys.argv.remove(flag)
            except Exception:
                pass

    # Intercept CPU override flags from command-line dynamically
    for flag in ['--cpu', '-cpu', '--force-cpu', '--cpu-only']:
        if flag in sys.argv:
            try:
                os.environ["MOLMEM_FORCE_CPU"] = "1"
                os.environ["MOLMEM_CPU_ONLY"] = "1"
                sys.argv.remove(flag)
            except Exception:
                pass

    # Intercept explicit compiler opt-out flags from command-line dynamically
    for flag in ['--no-compile', '--skip-compile']:
        if flag in sys.argv:
            try:
                os.environ["MOLMEM_NO_COMPILE"] = "1"
                sys.argv.remove(flag)
            except Exception:
                pass
    if not is_mp_child and os.environ.get("MOLMEM_FORCE_GPU") == "1" and os.environ.get("MOLMEM_CPU_ONLY") != "1":
        try:
            import torch
            if not torch.cuda.is_available():
                from .sys_utils import handle_fatal_hardware_error
                handle_fatal_hardware_error(
                    "GPU/ROCm execution target was requested, but PyTorch reports that CUDA/HIP is unavailable.",
                    error_type="GPU UNAVAILABLE ERROR"
                )
            try:
                num_gpus = torch.cuda.device_count()
            except Exception as query_err:
                from .sys_utils import handle_fatal_hardware_error
                handle_fatal_hardware_error(
                    f"Failed to get device count: {query_err}",
                    error_type="Failed to get device count",
                    exc=query_err
                )
            if num_gpus <= 0:
                from .sys_utils import handle_fatal_hardware_error
                handle_fatal_hardware_error(
                    "GPU/ROCm execution target was requested, but PyTorch reports a device count of 0.",
                    error_type="GPU DEVICE COUNT ZERO ERROR"
                )
        except Exception as e:
            if "Failed to get device count" in str(e):
                err_type = "Failed to get device count"
            else:
                err_type = "GPU DRIVER INITIALIZATION ERROR"
            from .sys_utils import handle_fatal_hardware_error
            handle_fatal_hardware_error(
                f"GPU/ROCm execution target was requested, but PyTorch failed to initialize: {e}",
                error_type=err_type,
                exc=e
            )

    # --- Line-by-Line Profiler Startup & Child Subprocess Tracking ---
    _global_line_profiler = None
    _disable_profile_hook = False

    def start_line_profiler():
        """
        Programmatically starts line-by-line execution profiling across all threads and modules.
        Logs will be automatically aggregated and generated at process exit or upon calling stop_line_profiler().
        """
        global _global_line_profiler
        os.environ["MOLMEM_PROFILE_LINES"] = "1"
        if _global_line_profiler is None:
            from .line_profiler import LineProfiler
            _global_line_profiler = LineProfiler()
        _global_line_profiler.start()

    def stop_line_profiler(generate_report=True):
        """
        Programmatically stops line-by-line execution profiling and optionally generates the profile logs.
        """
        global _global_line_profiler
        if _global_line_profiler is not None:
            _global_line_profiler.stop()
            if generate_report:
                _cleanup_line_profiler()

    def _cleanup_line_profiler():
        import sys
        import os
        import glob
        import pickle
        if is_mp_child:
            return
        global _global_line_profiler, _disable_profile_hook
        if _disable_profile_hook:
            return
        try:
            if _global_line_profiler is not None:
                _global_line_profiler.stop()
                data = _global_line_profiler.get_data()
            else:
                data = {}
                
            lib_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = get_log_dir(auto_create=True)

            # Aggregate data from all child workers (.temp_profile_*.pkl)
            for temp_pkl in glob.glob(os.path.join(log_dir, ".temp_profile_*.pkl")):
                try:
                    with open(temp_pkl, 'rb') as f:
                        c_raw = pickle.load(f)
                    for k, v in c_raw.items():
                        if isinstance(k, list):
                            k = tuple(k)
                        if k in data:
                            data[k][0] += v[0]
                            data[k][1] += v[1]
                        else:
                            data[k] = list(v)
                    os.remove(temp_pkl)
                except Exception:
                    pass
                
            # Load existing aggregate profile data if available
            agg_path = os.path.join(log_dir, ".aggregate_profile.pkl")
            agg_data = {}
            if os.path.exists(agg_path):
                try:
                    with open(agg_path, 'rb') as f:
                        raw_agg = pickle.load(f)
                    for k, v in raw_agg.items():
                        if isinstance(k, list):
                            k = tuple(k)
                        filename = k[0]
                        if filename:
                            base = os.path.basename(filename).lower()
                            if base in ("line_profiler.py", "profiler.py", "sys_utils.py"):
                                continue
                        agg_data[k] = v
                except Exception:
                    pass
                    
            # Merge current run data
            profile_data = {}
            for k, v in data.items():
                filename = k[0]
                if filename:
                    base = os.path.basename(filename).lower()
                    if base in ("line_profiler.py", "profiler.py", "sys_utils.py"):
                        continue
                profile_data[k] = v
                
            for k, v in profile_data.items():
                if k in agg_data:
                    agg_data[k][0] += v[0]  # hits
                    agg_data[k][1] += v[1]  # total_time
                else:
                    agg_data[k] = v
                    
            # Save aggregate profile data
            try:
                with open(agg_path, 'wb') as f:
                    pickle.dump(agg_data, f)
            except Exception:
                pass
                
            if agg_data:
                sys.stdout.write("[*] Processing and generating line execution profile log...\n")
                sys.stdout.flush()
                from .line_profiler import run_profiler_logger
                run_profiler_logger(agg_data, lib_dir)
            try:
                from .sys_utils import dump_last_100_lines_log
                dump_last_100_lines_log()
            except Exception:
                pass
        except Exception as e:
            try:
                from .sys_utils import log_exception_file
                log_exception_file(f"Line Profiler cleanup error: {e}")
            except Exception:
                pass

    if not is_mp_child and os.environ.get("MOLMEM_PROFILE_LINES") == "1":
        try:
            main_module = sys.modules.get('__main__')
            is_mp_spawn = bool(main_module and getattr(main_module, '__name__', '') == '__mp_main__')
            if not is_mp_spawn and os.environ.get("MOLMEM_REDIRECTED") != "1":
                try:
                    import glob
                    log_dir = get_log_dir(auto_create=True)
                    agg_path = os.path.join(log_dir, ".aggregate_profile.pkl")
                    if os.path.exists(agg_path):
                        os.remove(agg_path)
                    db_path = os.path.join(log_dir, "trace_debug.txt")
                    if os.path.exists(db_path):
                        os.remove(db_path)
                    
                    for stranded_pkl in glob.glob(os.path.join(log_dir, ".temp_profile_*.pkl")):
                        try:
                            os.remove(stranded_pkl)
                        except Exception:
                            pass
                    for log_name in ["profile_log.txt", "profile_log_by_hits.txt", "profile_log_by_avg.txt", "profile_log_summary.txt"]:
                        try:
                            p = os.path.join(log_dir, log_name)
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
                    sim_profiles_dir = os.path.join(log_dir, "simulation_profiles")
                    if os.path.exists(sim_profiles_dir):
                        for f_old in glob.glob(os.path.join(sim_profiles_dir, "*")):
                            try:
                                os.remove(f_old)
                            except Exception:
                                pass
                except Exception:
                    pass

            from .line_profiler import LineProfiler
            _global_line_profiler = LineProfiler()
            _global_line_profiler.start()

            import atexit
            atexit.register(_cleanup_line_profiler)
        except Exception:
            pass

    if not is_mp_child and os.environ.get("MOLMEM_PROFILE_SSD") == "1" and os.environ.get("MOLMEM_PROFILE_LINES") != "1":
        def _cleanup_ssd_only_profiler():
            try:
                from .history import generate_ssd_pipeline_report
                log_dir = get_log_dir(auto_create=True)
                sim_dir = os.path.join(log_dir, "simulation_profiles")
                generate_ssd_pipeline_report(sim_dir)
                sys.stdout.write("[*] SSD Pipeline Execution Report generated: logs/simulation_profiles/ssd_pipeline_summary.txt\n")
                sys.stdout.flush()
            except Exception:
                pass
        import atexit
        atexit.register(_cleanup_ssd_only_profiler)

    # Child multiprocessing worker line profiler attachment
    if is_mp_child and os.environ.get("MOLMEM_PROFILE_LINES") == "1":
        try:
            from .line_profiler import LineProfiler
            _child_line_profiler = LineProfiler()
            _child_line_profiler.start()

            def _cleanup_child_line_profiler():
                global _child_line_profiler
                try:
                    if _child_line_profiler is not None:
                        _child_line_profiler.stop()
                        c_data = _child_line_profiler.get_data()
                        if c_data:
                            import pickle
                            import uuid
                            log_dir = get_log_dir(auto_create=True)
                            out_path = os.path.join(log_dir, f".temp_profile_{os.getpid()}_{uuid.uuid4().hex[:8]}.pkl")
                            with open(out_path, 'wb') as f:
                                pickle.dump(c_data, f)
                except Exception:
                    pass

            import atexit
            atexit.register(_cleanup_child_line_profiler)
            try:
                import multiprocessing.util as mp_util
                mp_util.Finalize(None, _cleanup_child_line_profiler, exitpriority=0)
            except Exception:
                pass
        except Exception:
            pass

    def _onboard_compiler_extension():
        import sys
        import os
        import subprocess
        import shutil
        import json
        import hashlib
        import time

        if os.environ.get("MOLMEM_NO_COMPILE") == "1" or os.environ.get("MOLMEM_SKIP_COMPILE") == "1":
            return
            
        # Check if user declined compilation previously
        compile_declined = False
        try:
            data = _load_unified_cache()
            if 'config_json' in data:
                config = json.loads(str(data['config_json'][0]))
                if config.get("extension_compile_declined") is True:
                    compile_declined = True
        except Exception:
            pass
            
        if compile_declined:
            return
            
        try:
            import torch
            if not torch.cuda.is_available():
                return
        except ImportError:
            return

        lib_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.join(lib_dir, "csrc")
        if csrc_dir not in sys.path:
            sys.path.insert(0, csrc_dir)
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

        # Ensure current python environment's bin/Scripts is in PATH so tools like ninja are discoverable
        venv_bin = os.path.dirname(sys.executable)
        if os.path.isdir(venv_bin) and venv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

        if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
            for d in [csrc_dir, lib_dir]:
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
            r_p = _find_rocm_path()
            if r_p and os.path.isdir(os.path.join(r_p, 'bin')):
                try:
                    os.add_dll_directory(os.path.join(r_p, 'bin'))
                except Exception:
                    pass
            c_p = _find_cuda_path()
            if c_p and os.path.isdir(os.path.join(c_p, 'bin')):
                try:
                    os.add_dll_directory(os.path.join(c_p, 'bin'))
                except Exception:
                    pass

        def _get_csrc_combined_hash(csrc_d):
            files = ["gpu_solver.hip", "gpu_solver_binding.cpp", "c10_stub.cpp", "setup.py"]
            hasher = hashlib.md5()
            for f_name in files:
                f_path = os.path.join(csrc_d, f_name)
                if os.path.exists(f_path):
                    with open(f_path, "rb") as f:
                        hasher.update(f.read())
            try:
                import torch
                hasher.update(str(torch.__version__).encode('utf-8'))
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        hasher.update(torch.cuda.get_device_name(i).encode('utf-8'))
            except Exception:
                pass
            return hasher.hexdigest()

        current_hash = ""
        try:
            current_hash = _get_csrc_combined_hash(csrc_dir)
        except Exception:
            pass

        hash_file = os.path.join(csrc_dir, ".csrc_hash")
        cached_hash = ""
        if os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as f:
                    cached_hash = f.read().strip()
            except Exception:
                pass

        # Check if binary already exists in lib_dir or csrc_dir
        has_binary = False
        for check_d in [lib_dir, csrc_dir]:
            if os.path.exists(check_d):
                for f in os.listdir(check_d):
                    if f.startswith("molmem_cuda_solver") and (f.endswith(".pyd") or f.endswith(".so")):
                        has_binary = True
                        break
            if has_binary:
                break

        # Only skip if hash is non-empty, matches current hash, binary exists, and is verified for the active GPU architecture
        if cached_hash != "" and cached_hash == current_hash and has_binary:
            try:
                from .torch_simulator import check_has_cpp_extension
                if check_has_cpp_extension():
                    return
            except Exception:
                pass

        is_rocm = getattr(torch.version, 'hip', None) is not None
        gpu_type = "AMD ROCm" if is_rocm else "NVIDIA CUDA"

        # Check toolchains
        if not _check_compiler_toolchain(has_amd=is_rocm, has_nvidia=(not is_rocm), quiet=True):
            return

        vs_bat = _find_vcvars64()
        has_compiler = False
        if is_rocm:
            rocm_p = _find_rocm_path()
            if rocm_p and os.path.isdir(os.path.join(rocm_p, "bin")):
                r_bin = os.path.join(rocm_p, "bin")
                if r_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = r_bin + os.pathsep + os.environ.get("PATH", "")
            hipcc_path = shutil.which("hipcc")
            if not hipcc_path and rocm_p:
                for cand in [os.path.join(rocm_p, "bin", "hipcc.exe"), os.path.join(rocm_p, "bin", "hipcc.bat"), os.path.join(rocm_p, "bin", "hipcc")]:
                    if os.path.exists(cand):
                        hipcc_path = cand
                        break
            if hipcc_path and (vs_bat is not None or sys.platform != 'win32'):
                has_compiler = True
        else:
            cuda_p = _find_cuda_path()
            if cuda_p:
                c_bin = os.path.join(cuda_p, "bin")
                if os.path.exists(c_bin) and c_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = c_bin + os.pathsep + os.environ.get("PATH", "")
            nvcc_path = shutil.which("nvcc")
            if not nvcc_path and cuda_p:
                for cand in [os.path.join(cuda_p, "bin", "nvcc"), os.path.join(cuda_p, "bin", "nvcc.exe")]:
                    if os.path.exists(cand):
                        nvcc_path = cand
                        break
            if nvcc_path and (vs_bat is not None or sys.platform != 'win32'):
                has_compiler = True

        if not has_compiler:
            return

        # Execute single compilation pass
        import multiprocessing
        if "MAX_JOBS" not in os.environ:
            os.environ["MAX_JOBS"] = str(min(multiprocessing.cpu_count(), 61))

        compile_lock_path = os.path.join(csrc_dir, ".compile.lock")
        lock_acquired = False
        try:
            if os.path.exists(compile_lock_path):
                try:
                    if time.time() - os.path.getmtime(compile_lock_path) > 30.0:
                        os.remove(compile_lock_path)
                except Exception:
                    pass

            t_lock_start = time.time()
            while time.time() - t_lock_start < 60:
                try:
                    lock_fd = os.open(compile_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(lock_fd)
                    lock_acquired = True
                    break
                except OSError:
                    time.sleep(0.5)

            sys.stdout.write(
                f"\n=========================================================================================================\n"
                f"MOLMEM JIT: Building {gpu_type} C++ GPU Extension (one-time compile)...\n"
                f"=========================================================================================================\n"
            )
            sys.stdout.flush()

            temp_bat = None
            if sys.platform == 'win32':
                temp_bat = os.path.join(csrc_dir, "onboard_compile.bat")
                msvc_bin = _find_msvc_bin()
                sdk_bin = _find_windows_sdk_bin()
                try:
                    with open(temp_bat, "w") as f:
                        f.write('@echo off\n')
                        if vs_bat and vs_bat != "CL_READY":
                            f.write(f'call "{vs_bat}"\n')
                        if msvc_bin:
                            f.write(f'set "PATH={msvc_bin};%PATH%"\n')
                        if sdk_bin:
                            f.write(f'set "PATH={sdk_bin};%PATH%"\n')
                        if is_rocm:
                            r_path = _find_rocm_path()
                            f.write(f'set "HIP_PATH={r_path}"\n')
                            f.write(f'set "ROCM_HOME={r_path}"\n')
                            f.write(f'set "ROCM_PATH={r_path}"\n')
                            f.write(f'set "HIP_DIR={r_path}"\n')
                            f.write(f'set "CUDA_HOME={r_path}"\n')
                            f.write(f'set "CUDA_PATH={r_path}"\n')
                            f.write(f'set "PATH={os.path.join(r_path, "bin")};%PATH%"\n')
                        else:
                            c_path = _find_cuda_path()
                            if c_path:
                                f.write(f'set "CUDA_HOME={c_path}"\n')
                                f.write(f'set "CUDA_PATH={c_path}"\n')
                                f.write(f'set "PATH={os.path.join(c_path, "bin")};%PATH%"\n')
                        if "MAX_JOBS" in os.environ:
                            f.write(f'set "MAX_JOBS={os.environ["MAX_JOBS"]}"\n')
                        f.write('set DISTUTILS_USE_SDK=1\n')
                        f.write(f'"{sys.executable}" setup.py build_ext --inplace\n')
                except Exception:
                    pass
                cmd = ["cmd.exe", "/c", "onboard_compile.bat"]
            else:
                cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

            build_temp = os.path.join(csrc_dir, "build")
            if os.path.isdir(build_temp):
                try:
                    shutil.rmtree(build_temp, ignore_errors=True)
                except Exception:
                    pass

            # Self-healing: Purge any stale binaries or leftover CUDA source
            # so an incompatible architecture binary never lingers if build fails
            for target_d in [csrc_dir, lib_dir]:
                if os.path.isdir(target_d):
                    try:
                        for f_name in os.listdir(target_d):
                            if f_name.startswith("molmem_cuda_solver") and (f_name.endswith(".pyd") or f_name.endswith(".so")):
                                try:
                                    os.remove(os.path.join(target_d, f_name))
                                except Exception:
                                    pass
                    except Exception:
                        pass
            cu_solver_file = os.path.join(csrc_dir, "gpu_solver.cu")
            if os.path.exists(cu_solver_file):
                try:
                    os.remove(cu_solver_file)
                except Exception:
                    pass

            res = subprocess.run(cmd, cwd=csrc_dir, capture_output=True, text=True)
            if temp_bat and os.path.exists(temp_bat):
                try:
                    os.remove(temp_bat)
                except Exception:
                    pass

            if res.returncode == 0:
                try:
                    for f_name in os.listdir(csrc_dir):
                        if f_name.startswith("molmem_cuda_solver") and (f_name.endswith(".pyd") or f_name.endswith(".so")):
                            src_pyd = os.path.join(csrc_dir, f_name)
                            dst_lib_pyd = os.path.join(lib_dir, f_name)
                            shutil.copy2(src_pyd, dst_lib_pyd)
                except Exception:
                    pass
                try:
                    import sysconfig
                    site_packages = sysconfig.get_path('purelib')
                    if site_packages and os.path.exists(site_packages):
                        for f_name in os.listdir(csrc_dir):
                            if f_name.startswith("molmem_cuda_solver") and (f_name.endswith(".pyd") or f_name.endswith(".so")):
                                src_pyd = os.path.join(csrc_dir, f_name)
                                dst_pyd = os.path.join(site_packages, f_name)
                                shutil.copy2(src_pyd, dst_pyd)
                except Exception:
                    pass

                try:
                    if sys.platform == 'win32' and os.path.exists(hash_file):
                        try:
                            import ctypes
                            ctypes.windll.kernel32.SetFileAttributesW(str(hash_file), 0x80)
                        except Exception:
                            pass
                    with open(hash_file, "w") as f:
                        f.write(current_hash)
                except Exception:
                    pass

                try:
                    import importlib
                    if 'molmem_cuda_solver' in sys.modules:
                        importlib.reload(sys.modules['molmem_cuda_solver'])
                    else:
                        import molmem_cuda_solver
                except Exception:
                    pass

                sys.stdout.write(
                    f"=========================================================================================================\n"
                    f"[+] SUCCESS: {gpu_type} C++ Extension compiled and ready!\n"
                    f"=========================================================================================================\n\n"
                )
                sys.stdout.flush()
            else:
                try:
                    if res.stderr:
                        with open(os.path.join(csrc_dir, "compile_error.log"), "w") as f:
                            f.write(res.stdout or "")
                            f.write("\n--- STDERR ---\n")
                            f.write(res.stderr)
                except Exception:
                    pass

                sys.stderr.write(
                    f"\n[!] C++ Extension compilation failed. Falling back to PyTorch JIT mode.\n"
                    f"     To inspect compiler errors, run:\n"
                    f"       cd \"{csrc_dir}\" && \"{sys.executable}\" setup.py build_ext --inplace\n\n"
                )
                sys.stderr.flush()
        finally:
            # Auto-clean intermediate compiler build directory to ensure zero residual host paths
            for clean_sub in ["build", "molmem_cuda_solver.egg-info"]:
                clean_target = os.path.join(csrc_dir, clean_sub)
                if os.path.isdir(clean_target):
                    try:
                        from .sys_utils import _fast_rmtree
                        _fast_rmtree(clean_target)
                    except Exception:
                        shutil.rmtree(clean_target, ignore_errors=True)
            if lock_acquired and os.path.exists(compile_lock_path):
                try:
                    os.remove(compile_lock_path)
                except Exception:
                    pass

# --- Windows/Unix OS-Level Ctrl-C Handlers to bypass JIT lockups ---
# Preserves standard catchable KeyboardInterrupt in Jupyter/Colab notebooks to prevent terminating the kernel process.
_is_nb_interrupt = False
try:
    from .sys_utils import is_interactive_notebook
    _is_nb_interrupt = is_interactive_notebook()
except Exception:
    pass

if not is_mp_child and not _is_nb_interrupt:
    if sys.platform == 'win32':
        _win_ctrl_handler_c = None
        def install_windows_ctrl_handler():
            def _handler(ctrl_type):
                if ctrl_type in (0, 1, 2, 5, 6):
                    # Cleanly terminate any active plotter child processes
                    global _active_plot_processes
                    for proc in _active_plot_processes:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    sys.__stdout__.write("\n\n[!] Windows OS-Level KeyboardInterrupt detected (Ctrl-C). Forcing shutdown...\n")
                    sys.__stdout__.flush()
                    try:
                        from .sys_utils import dump_last_100_lines_log
                        dump_last_100_lines_log()
                    except Exception:
                        pass
                    try:
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                        kernel32.TerminateProcess.restype = ctypes.c_int
                        kernel32.GetCurrentProcess.argtypes = []
                        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                        res = kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 1)
                        if res == 0:
                            os._exit(1)
                    except Exception:
                        os._exit(1)
                return True
            try:
                import ctypes
                global _win_ctrl_handler_c
                _win_ctrl_handler_c = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(_handler)
                ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_ctrl_handler_c, True)
                
                # Enable ANSI/VT100 escape sequences on Windows consoles
                h_stdout = ctypes.windll.kernel32.GetStdHandle(-11)
                h_stderr = ctypes.windll.kernel32.GetStdHandle(-12)
                for h in [h_stdout, h_stderr]:
                    if h and h != -1:
                        mode = ctypes.c_ulong()
                        if ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                            ctypes.windll.kernel32.SetConsoleMode(h, mode.value | 0x0004)
            except Exception:
                pass
        install_windows_ctrl_handler()
    else:
        def install_unix_ctrl_handler():
            import signal
            def _handler(sig, frame):
                sys.__stdout__.write("\n\n[!] Unix OS-Level KeyboardInterrupt detected (Ctrl-C). Forcing shutdown...\n")
                sys.__stdout__.flush()
                try:
                    from .sys_utils import dump_last_100_lines_log
                    dump_last_100_lines_log()
                except Exception:
                    pass
                os._exit(1)
            try:
                signal.signal(signal.SIGINT, _handler)
            except Exception:
                pass
        install_unix_ctrl_handler()

# All submodules (simulator, quantization, devices, plotter, tf_wrapper, programmer, tiled_simulator) are lazy loaded via __getattr__
_LAZY_EXPORTS = {
    "MolmemSimulator": ".simulator",
    "updateCrossbarsParallel": ".simulator",
    "WeightQuantizer": ".quantization",
    "InputQuantizer": ".quantization",
    "OutputQuantizer": ".quantization",
    "MolMemristor": ".device",
    "BaseSource": ".sources",
    "DcSource": ".sources",
    "PulseSource": ".sources",
    "PwlSource": ".sources",
    "BSource": ".sources",
    "get_device_config": ".devices",
    "MolmemPlotter": ".plotter",
    "CrossbarDense": ".tf_wrapper",
    "EpochTracker": ".tf_wrapper",
    "suppress_c_stderr": ".sys_utils",
    "silenceMolmemLib": ".sys_utils",
    "MolmemSilence": ".sys_utils",
    "isMolmemLibSilenced": ".sys_utils",
    "silence_molmem_lib": ".sys_utils",
    "is_molmem_lib_silenced": ".sys_utils",
    "CrossbarProgrammer": ".programmer",
    "MolmemTiledSimulator": ".tiled_simulator",
    "NoiseEngine": ".noise",
}

def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_path = _LAZY_EXPORTS[name]
        # For tf_wrapper elements, provide safe placeholders in child worker processes
        # so that multiprocessing spawn re-importing the main script does not trigger an ImportError.
        if name in {"CrossbarDense", "EpochTracker"}:
            import multiprocessing
            if multiprocessing.current_process().name != 'MainProcess':
                class _ChildProcessTfPlaceholder:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __call__(self, *args, **kwargs):
                        return self
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                val = _ChildProcessTfPlaceholder
                globals()[name] = val
                return val
                
        import importlib
        module = importlib.import_module(module_path, __name__)
        val = getattr(module, name)
        globals()[name] = val
        return val
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(set(globals()).union(_LAZY_EXPORTS))



_cached_footprint = None

def _compute_library_footprint():
    """
    Computes a cryptographic SHA256 footprint of the entire `molmem_lib`
    source codebase physically resident on the hard-drive. Used for cache versioning
    to auto-purge stale hardware mappings when native simulator internals shift.
    """
    global _cached_footprint
    if _cached_footprint is not None:
        return _cached_footprint
        
    import hashlib
    from glob import glob
    hasher = hashlib.sha256()
    
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = sorted(glob(os.path.join(lib_dir, "*.py")))
    
    for py_file in source_files:
        if os.path.basename(py_file) == "devices.py":
            continue
        try:
            with open(py_file, 'rb') as f:
                content = f.read().replace(b'\r\n', b'\n')
                hasher.update(content)
        except Exception:
            pass
            
    _cached_footprint = hasher.hexdigest()
    return _cached_footprint

def get_cpu_sockets_info():
    import sys
    import os
    import json
    import subprocess
    
    sockets_info = []
    
    if sys.platform == 'win32':
        try:
            ps_bin = shutil.which("powershell") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            cmd = [ps_bin, "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object NumberOfCores, MaxClockSpeed, L3CacheSize | ConvertTo-Json"]
            res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
            if res.strip():
                data = json.loads(res)
                if isinstance(data, dict):
                    data = [data]
                for idx, item in enumerate(data):
                    cores_count = item.get('NumberOfCores') or 1
                    clock_mhz = item.get('MaxClockSpeed') or 1000
                    l3_kb = item.get('L3CacheSize') or 0
                    
                    sockets_info.append({
                        'index': idx,
                        'cores': [],
                        'cores_count': cores_count,
                        'clock_mhz': clock_mhz,
                        'l3_kb': l3_kb
                    })
        except Exception:
            pass
            
        try:
            import ctypes
            from ctypes import wintypes
            RelationProcessorPackage = 3

            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", ctypes.c_int),
                    ("Reserved", ctypes.c_uint64 * 2)
                ]

            kernel32 = ctypes.windll.kernel32
            cbBuffer = wintypes.DWORD(0)
            kernel32.GetLogicalProcessorInformation(None, ctypes.byref(cbBuffer))
            if cbBuffer.value > 0:
                num_elements = cbBuffer.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * num_elements)()
                if kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(cbBuffer)):
                    pkg_idx = 0
                    for info in buffer:
                        if info.Relationship == RelationProcessorPackage:
                            mask = info.ProcessorMask
                            cores = []
                            bit = 0
                            temp_mask = mask
                            while temp_mask > 0:
                                if temp_mask & 1:
                                    cores.append(bit)
                                temp_mask >>= 1
                                bit += 1
                            
                            if pkg_idx < len(sockets_info):
                                sockets_info[pkg_idx]['cores'] = cores
                            else:
                                sockets_info.append({
                                    'index': pkg_idx,
                                    'cores': cores,
                                    'cores_count': len(cores),
                                    'clock_mhz': 1000,
                                    'l3_kb': 0
                                })
                            pkg_idx += 1
        except Exception:
            pass
            
    elif sys.platform == 'darwin':
        try:
            num_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu'], stderr=subprocess.DEVNULL).strip())
            num_logical = int(subprocess.check_output(['sysctl', '-n', 'hw.logicalcpu'], stderr=subprocess.DEVNULL).strip())
            clock_mhz = 2400
            try:
                clock_hz = int(subprocess.check_output(['sysctl', '-n', 'hw.cpufrequency'], stderr=subprocess.DEVNULL).strip())
                if clock_hz > 0:
                    clock_mhz = clock_hz // 1_000_000
            except Exception:
                pass
            l3_kb = 0
            try:
                l3_kb = int(subprocess.check_output(['sysctl', '-n', 'hw.l3cachesize'], stderr=subprocess.DEVNULL).strip()) // 1024
            except Exception:
                pass
            sockets_info.append({
                'index': 0,
                'cores': list(range(num_logical)),
                'cores_count': num_cores,
                'clock_mhz': clock_mhz,
                'l3_kb': l3_kb
            })
        except Exception:
            pass
    elif sys.platform.startswith('linux'):
        from collections import defaultdict
        try:
            sockets = defaultdict(list)
            for cpu_dir in os.listdir('/sys/devices/system/cpu/'):
                if cpu_dir.startswith('cpu') and cpu_dir[3:].isdigit():
                    cpu_id = int(cpu_dir[3:])
                    pkg_path = f'/sys/devices/system/cpu/{cpu_dir}/topology/physical_package_id'
                    if os.path.exists(pkg_path):
                        with open(pkg_path, 'r') as f:
                            pkg_id = int(f.read().strip())
                            sockets[pkg_id].append(cpu_id)
            
            if sockets:
                for idx, (pkg_id, cores) in enumerate(sorted(sockets.items())):
                    clock_mhz = 1000
                    for cpu_id in cores:
                        freq_path = f'/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq'
                        if os.path.exists(freq_path):
                            try:
                                with open(freq_path, 'r') as f:
                                    freq_val = int(f.read().strip()) // 1000
                                    clock_mhz = max(clock_mhz, freq_val)
                                    break
                            except Exception:
                                pass
                                
                    l3_kb = 0
                    for cpu_id in cores:
                        cache_path = f'/sys/devices/system/cpu/cpu{cpu_id}/cache/index3/size'
                        if os.path.exists(cache_path):
                            try:
                                with open(cache_path, 'r') as f:
                                    size_str = f.read().strip()
                                    if size_str.endswith('K') or size_str.endswith('k'):
                                        l3_kb = int(size_str[:-1])
                                    elif size_str.endswith('M') or size_str.endswith('m'):
                                        l3_kb = int(size_str[:-1]) * 1024
                                    elif size_str.isdigit():
                                        l3_kb = int(size_str) // 1024
                                    break
                            except Exception:
                                pass
                                
                    sockets_info.append({
                        'index': idx,
                        'cores': sorted(cores),
                        'cores_count': len(cores),
                        'clock_mhz': clock_mhz,
                        'l3_kb': l3_kb
                    })
        except Exception:
            pass
            
    if not sockets_info:
        import psutil
        logical_cores = list(range(psutil.cpu_count(logical=True) or 1))
        clock_mhz = 1000
        try:
            freq = psutil.cpu_freq()
            if freq and getattr(freq, 'max', 0) > 0:
                clock_mhz = int(freq.max)
        except Exception:
            pass
        sockets_info.append({
            'index': 0,
            'cores': logical_cores,
            'cores_count': len(logical_cores),
            'clock_mhz': clock_mhz,
            'l3_kb': 0
        })
        
    for s in sockets_info:
        cores_val = len(s['cores']) if s['cores'] else s['cores_count']
        clock_ghz = s['clock_mhz'] / 1000.0
        l3_mb = s['l3_kb'] / 1024.0 if s['l3_kb'] > 0 else 1.0
        s['score'] = cores_val * clock_ghz * l3_mb
        
    return sockets_info

def get_gpu_devices_info():
    gpu_info = []
    if is_interactive_notebook() or os.environ.get("MOLMEM_CPU_ONLY") == "1":
        return gpu_info
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            is_dedicated = globals().get('_is_dedicated_gpu', lambda name: True)
            is_rocm = getattr(torch.version, 'hip', None) is not None
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                dedicated = is_dedicated(name)
                props = torch.cuda.get_device_properties(i)
                raw_vram = props.total_memory / (1024 ** 3)
                sm_count = getattr(props, 'multi_processor_count', 1)
                major = getattr(props, 'major', 0)
                minor = getattr(props, 'minor', 0)
                
                # Integrated GPUs share system RAM (e.g. 45GB GTT), not dedicated GDDR/HBM.
                # Adjust score so dedicated discrete GPUs are prioritized:
                vram = raw_vram
                if not dedicated:
                    effective_vram = min(raw_vram, 2.0)
                    score = effective_vram * sm_count * 0.25
                else:
                    score = vram * sm_count

                if is_rocm:
                    arch_name = None
                    try:
                        from .torch_simulator import get_active_gpu_arch
                        arch_name = get_active_gpu_arch(i)
                    except Exception:
                        pass
                    if not arch_name:
                        gcn = getattr(props, 'gcnArchName', '')
                        arch_name = gcn.split(':')[0].strip().lower() if gcn else ""

                    if sys.platform == 'win32':
                        win_rocm_supported = ["gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]
                        if arch_name and arch_name not in win_rocm_supported and not any(arch_name.startswith(p) for p in ["gfx103", "gfx11", "gfx12"]):
                            supported = False
                            unsupported_reason = f"ROCm on Windows does not support {arch_name} (requires RDNA2+ gfx1030-gfx1201)"
                        else:
                            supported = True
                            unsupported_reason = ""
                    else:
                        supported = True
                        unsupported_reason = ""
                else:
                    supported = (major <= 9)
                    unsupported_reason = f"SM_{major}{minor} exceeds max SM_90 for current Molmem_Lib version" if not supported else ""
                gpu_info.append({
                    'index': i,
                    'name': name,
                    'vram': vram,
                    'raw_vram': raw_vram,
                    'is_dedicated': dedicated,
                    'sm_count': sm_count,
                    'major': major,
                    'minor': minor,
                    'score': score,
                    'supported': supported,
                    'unsupported_reason': unsupported_reason,
                    'is_rocm': is_rocm
                })
    except Exception:
        pass
    return gpu_info

def _query_cpu_caches():
    import sys
    import os
    import json
    import subprocess
    import shutil
    
    l1_size = None
    l2_size = None
    l3_size = None
    
    if sys.platform == 'win32':
        try:
            ps_bin = shutil.which("powershell") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            cmd = [ps_bin, "-NoProfile", "-Command", "Get-CimInstance Win32_CacheMemory | Select-Object InstalledSize, Level | ConvertTo-Json"]
            res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
            if res.strip():
                data = json.loads(res)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    level = item.get('Level')
                    size = item.get('InstalledSize')
                    if level == 3:
                        l1_size = size
                    elif level == 4:
                        l2_size = size
                    elif level == 5:
                        l3_size = size
        except Exception:
            pass
            
        if l2_size is None and l3_size is None:
            try:
                res = subprocess.check_output("wmic cpu get L2CacheSize, L3CacheSize /value", shell=True, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
                for line in res.splitlines():
                    if '=' in line:
                        k, v = line.split('=', 1)
                        if k.strip() == "L2CacheSize":
                            l2_size = int(v.strip())
                        elif k.strip() == "L3CacheSize":
                            l3_size = int(v.strip())
            except Exception:
                pass
    elif sys.platform == 'darwin':
        try:
            for sysctl_name, target in [('hw.l1dcachesize', 1), ('hw.l2cachesize', 2), ('hw.l3cachesize', 3)]:
                try:
                    val_bytes = int(subprocess.check_output(['sysctl', '-n', sysctl_name], stderr=subprocess.DEVNULL).strip())
                    val_kb = val_bytes // 1024
                    if target == 1:
                        l1_size = val_kb
                    elif target == 2:
                        l2_size = val_kb
                    elif target == 3:
                        l3_size = val_kb
                except Exception:
                    pass
        except Exception:
            pass
    elif sys.platform.startswith('linux'):
        try:
            import glob
            l1_val = 0
            for path in glob.glob('/sys/devices/system/cpu/cpu0/cache/index*'):
                try:
                    with open(os.path.join(path, 'level'), 'r') as f:
                        level = int(f.read().strip())
                    with open(os.path.join(path, 'size'), 'r') as f:
                        size_str = f.read().strip()
                    
                    val_kb = 0
                    if size_str.endswith('K') or size_str.endswith('k'):
                        val_kb = int(size_str[:-1])
                    elif size_str.endswith('M') or size_str.endswith('m'):
                        val_kb = int(size_str[:-1]) * 1024
                    elif size_str.isdigit():
                        val_kb = int(size_str) // 1024
                        
                    if level == 1:
                        l1_val += val_kb
                    elif level == 2:
                        l2_size = val_kb
                    elif level == 3:
                        l3_size = val_kb
                except Exception:
                    pass
            if l1_val > 0:
                l1_size = l1_val
        except Exception:
            pass
            
    def format_cache_size(size_kb):
        if size_kb is None or size_kb <= 0:
            return None
        if size_kb >= 1024:
            if size_kb % 1024 == 0:
                return f"{int(size_kb // 1024)} MB"
            else:
                return f"{size_kb / 1024:.2f} MB"
        return f"{size_kb} KB"
        
    parts = []
    f_l1 = format_cache_size(l1_size)
    f_l2 = format_cache_size(l2_size)
    f_l3 = format_cache_size(l3_size)
    
    if f_l1: parts.append(f"L1: {f_l1}")
    if f_l2: parts.append(f"L2: {f_l2}")
    if f_l3: parts.append(f"L3: {f_l3}")
    
    return " | ".join(parts) if parts else None

_hardware_config = {}

def _initialize_hardware_targets():
    import os
    import sys
    import json
    global _hardware_config
    
    if is_mp_child:
        return
        
    import platform
    import hashlib
    from .sys_utils import is_interactive_notebook
    anonymized_node = hashlib.sha256(platform.node().encode('utf-8')).hexdigest()[:16]
    is_cpu_mode = (os.environ.get("MOLMEM_CPU_ONLY") == "1" or is_interactive_notebook())
    gpu_devices = get_gpu_devices_info()
    cpu_sockets = get_cpu_sockets_info()
    gpu_sig = ",".join(f"{d['name']}_{d['vram']:.1f}" for d in gpu_devices) if gpu_devices else "NO_GPU"
    env_mode = "notebook" if is_interactive_notebook() else "cli"
    current_host_sig = f"{anonymized_node}:{sys.platform}:{env_mode}:{os.cpu_count()}:{gpu_sig}"
    config_loaded = False
    _purge_machine_cache_on_mismatch = False
    try:
        data = _load_unified_cache()
        if 'config_json' in data:
            candidate_config = json.loads(str(data['config_json'][0]))
            required_keys = {
                "selected_cpu", "selected_gpu", "cpu_name", "cpu_cores_info",
                "gpu_detected", "gpu_name", "gpu_vram", "cpu_sockets", "gpu_devices", "gpu_type"
            }
            if required_keys.issubset(candidate_config.keys()):
                cand_sig = candidate_config.get("host_signature", "")
                cand_parts = cand_sig.split(":")
                is_same_machine = (
                    len(cand_parts) >= 4 and
                    cand_parts[0] == anonymized_node and
                    cand_parts[1] == sys.platform and
                    cand_parts[3] == str(os.cpu_count())
                )
                if is_same_machine:
                    if is_cpu_mode:
                        # Same physical machine running in CPU-override mode:
                        # Adopt canonical machine hardware config in memory without purging or corrupting disk cache
                        _hardware_config = dict(candidate_config)
                        _hardware_config['selected_gpu'] = 0
                        _hardware_config['gpu_detected'] = False
                        config_loaded = True
                    else:
                        # Normal / GPU execution on same machine: verify GPU devices match
                        cached_gpus = candidate_config.get("gpu_devices", [])
                        if len(cached_gpus) == len(gpu_devices) and all(
                            cg.get('name') == dg.get('name') and cg.get('supported') == dg.get('supported')
                            for cg, dg in zip(cached_gpus, gpu_devices)
                        ):
                            _hardware_config = candidate_config
                            config_loaded = True
                        else:
                            # Physical GPU on this machine actually changed
                            _purge_machine_cache_on_mismatch = True
                else:
                    # Genuinely different physical host or OS container (e.g. Google Colab / Kaggle cloud VM)
                    if not is_cpu_mode:
                        _purge_machine_cache_on_mismatch = True
        elif not is_cpu_mode:
            # molmem_cache.npz is missing or has no config_json (fresh clone or pre-cleared cache).
            # Verify whether existing .csrc_arch or binaries conflict with the active GPU platform.
            csrc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'csrc'))
            arch_file = os.path.join(csrc_dir, '.csrc_arch')
            if os.path.isfile(arch_file):
                try:
                    with open(arch_file, 'r', encoding='utf-8') as _af:
                        _content = _af.read().strip()
                    if _content:
                        _raw = [x.strip().lower() for x in _content.replace(";", " ").replace(",", " ").split()]
                        _is_rocm_now = any(d.get('is_rocm', False) for d in gpu_devices) if gpu_devices else False
                        _has_gfx = any(x.startswith("gfx") for x in _raw)
                        _has_cuda = any(not x.startswith("gfx") for x in _raw)
                        if (_is_rocm_now and not _has_gfx and _has_cuda) or (not _is_rocm_now and _has_gfx and not _has_cuda):
                            _purge_machine_cache_on_mismatch = True
                except Exception:
                    pass
    except Exception:
        pass
            
    if not config_loaded:
        if _purge_machine_cache_on_mismatch and not is_cpu_mode:
            # Hardware/environment transition detected: safely purge cache, incompatible native binaries, and architecture hashes
            try:
                clear_cache(deep=True)
            except Exception:
                pass

        import psutil
        import platform
        
        num_cpus = len(cpu_sockets)
        num_gpus = len(gpu_devices)
        
        selected_cpu_idx = 0
        selected_gpu_idx = 0
        
        best_socket_idx = 0
        all_cpu_equal = True
        if num_cpus > 0:
            max_cpu_score = -1.0
            first_cpu_score = cpu_sockets[0]['score']
            for s in cpu_sockets:
                score = s.get('score', 0.0)
                if abs(score - first_cpu_score) > 1e-5:
                    all_cpu_equal = False
                if score > max_cpu_score:
                    max_cpu_score = score
                    best_socket_idx = s['index']
            selected_cpu_idx = best_socket_idx
            
        best_gpu_idx = 0
        all_gpu_equal = True
        supported_gpus = [dev for dev in gpu_devices if dev.get('supported', True)]
        selected_gpu_idx = None
        if supported_gpus:
            # Sort by dedicated discrete status first, then by score
            supported_gpus_sorted = sorted(
                supported_gpus,
                key=lambda d: (1 if d.get('is_dedicated', True) else 0, d.get('score', 0.0)),
                reverse=True
            )
            best_gpu_idx = supported_gpus_sorted[0]['index']
            first_gpu_score = supported_gpus[0].get('score', 0.0)
            for dev in supported_gpus:
                score = dev.get('score', 0.0)
                if abs(score - first_gpu_score) > 1e-5:
                    all_gpu_equal = False
            selected_gpu_idx = best_gpu_idx

        is_cpu_only = ('--cpu' in sys.argv or '-cpu' in sys.argv or os.environ.get('MOLMEM_RUN_MODE') == 'CPU')
        is_non_interactive = (
            not sys.stdin.isatty() or
            '--non-interactive' in sys.argv or
            '-y' in sys.argv or
            '--yes' in sys.argv or
            '--quick' in sys.argv or
            os.environ.get('MOLMEM_SELECTED_GPU') is not None or
            os.environ.get('CI') == '1'
        )

        if (num_cpus > 1 or num_gpus > 1) and not is_cpu_only and not is_non_interactive:
            print("=" * 105)
            print("  MULTIPLE HARDWARE TARGETS DETECTED")
            print("=" * 105)
            
            if num_gpus > 1:
                print("  Available Graphics Devices (GPUs):")
                valid_indices = []
                for dev in gpu_devices:
                    is_supp = dev.get('supported', True)
                    score = dev.get('score', 0.0)
                    maj = dev.get('major', 0)
                    min_ver = dev.get('minor', 0)
                    cu_label = "CUs" if dev.get('is_rocm', False) else "SMs/CUs"
                    tag = " [Dedicated]" if dev.get('is_dedicated', True) else " [Integrated / Shared Memory]"
                    if is_supp:
                        valid_indices.append(dev['index'])
                        if dev['index'] == best_gpu_idx:
                            rec_label = f" (Recommended - highest performance score: {score:.1f})"
                            if all_gpu_equal:
                                rec_label = " (Recommended - primary GPU)"
                        else:
                            rec_label = ""
                        print(f"    [{dev['index']}] {dev['name']}{tag} | VRAM: {dev['vram']:.2f} GB | {cu_label}: {dev.get('sm_count', 1)}{rec_label}")
                    else:
                        reason = dev.get('unsupported_reason') or f"SM_{maj}{min_ver} exceeds max SM_90 for current Molmem_Lib version"
                        print(f"    [{dev['index']}] {dev['name']}{tag} | VRAM: {dev['vram']:.2f} GB | {cu_label}: {dev.get('sm_count', 1)} [Unavailable: {reason}]")
                
                if valid_indices:
                    while True:
                        try:
                            choice = input(f"  ==> Select GPU device index {valid_indices} (default {best_gpu_idx}): ").strip()
                            if not choice:
                                selected_gpu_idx = best_gpu_idx
                                break
                            val = int(choice)
                            if val in valid_indices:
                                selected_gpu_idx = val
                                break
                            else:
                                print(f"    [!] Invalid index. Must be one of {valid_indices} (supported GPUs only).")
                        except Exception:
                            print("    [!] Invalid input. Please enter a valid integer.")
                print()
                
            if num_cpus > 1:
                print("  Available CPU Packages (Sockets):")
                for s in cpu_sockets:
                    n_cores = len(s['cores']) if s['cores'] else s['cores_count']
                    clock_ghz = s['clock_mhz'] / 1000.0
                    l3_mb = s['l3_kb'] / 1024.0
                    l3_str = f"{int(l3_mb) if l3_mb.is_integer() else l3_mb:.1f} MB" if l3_mb > 0 else "Unknown"
                    score = s.get('score', 0.0)
                    
                    if s['index'] == best_socket_idx:
                        rec_label = f" (Recommended - highest performance score: {score:.1f})"
                        if all_cpu_equal:
                            rec_label = " (Recommended - primary NUMA node for localized cache & memory bandwidth)"
                    else:
                        rec_label = ""
                    print(f"    [{s['index']}] CPU Package {s['index']} ({n_cores} Cores @ {clock_ghz:.2f} GHz | L3 Cache: {l3_str}){rec_label}")
                
                while True:
                    try:
                        choice = input(f"  ==> Select CPU socket index [0-{num_cpus - 1}] (default {best_socket_idx}): ").strip()
                        if not choice:
                            selected_cpu_idx = best_socket_idx
                            break
                        selected_cpu_idx = int(choice)
                        if 0 <= selected_cpu_idx < num_cpus:
                            break
                        else:
                            print(f"    [!] Invalid index. Must be between 0 and {num_cpus - 1}.")
                    except Exception:
                        print("    [!] Invalid input. Please enter a valid integer.")
                print()
                
            print("  [+] Hardware preferences selected.")
            print("=" * 105)
            print()
            
        cpu_name = "Unknown CPU"
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            except Exception:
                cpu_name = platform.processor()
        else:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_name = line.split(':', 1)[1].strip()
                            break
            except Exception:
                cpu_name = platform.processor()
        if not cpu_name:
            cpu_name = platform.processor() or "Generic CPU"
            
        try:
            freq = psutil.cpu_freq()
            if freq and getattr(freq, 'max', 0) > 0:
                max_freq_ghz = freq.max / 1000.0
                if f"{max_freq_ghz:.2f}" not in cpu_name and f"{max_freq_ghz:.1f}" not in cpu_name:
                    cpu_name = f"{cpu_name} @ {max_freq_ghz:.2f} GHz"
        except Exception:
            pass
            
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        
        gpu_detected = False
        gpu_name = ""
        gpu_vram = 0.0
        gpu_sm_count = 0
        
        chosen_dev = None
        if selected_gpu_idx is not None:
            for dev in supported_gpus:
                if dev['index'] == selected_gpu_idx:
                    chosen_dev = dev
                    break
                
        if chosen_dev is not None:
            gpu_detected = True
            gpu_name = chosen_dev['name']
            gpu_vram = chosen_dev['vram']
            gpu_sm_count = chosen_dev.get('sm_count', 0)
            
        if selected_cpu_idx < num_cpus:
            cores_to_bind = cpu_sockets[selected_cpu_idx]['cores']
            num_cores_bound = len(cores_to_bind)
            cpu_cores_info = f"Cores : {num_cores_bound} Bound (Socket {selected_cpu_idx}) | RAM : {total_ram:.2f} GB"
        else:
            cpu_cores_info = f"Cores : {physical_cores} Physical, {logical_cores} Logical | RAM : {total_ram:.2f} GB"
            
        cpu_cache_info = _query_cpu_caches()
        
        gpu_type = "NVIDIA CUDA"
        try:
            import torch
            if getattr(torch.version, 'hip', None) is not None:
                gpu_type = "AMD ROCm"
        except ImportError:
            pass
            
        _hardware_config = {
            'host_signature': current_host_sig,
            'selected_cpu': selected_cpu_idx,
            'selected_gpu': selected_gpu_idx,
            'cpu_name': cpu_name,
            'cpu_cores_info': cpu_cores_info,
            'cpu_cache_info': cpu_cache_info,
            'gpu_detected': gpu_detected,
            'gpu_name': gpu_name,
            'gpu_vram': gpu_vram,
            'gpu_sm_count': gpu_sm_count,
            'cpu_sockets': cpu_sockets,
            'gpu_devices': gpu_devices,
            'gpu_type': gpu_type
        }
        
        try:
            import numpy as np
            # Do not persist a temporary CPU-override configuration over the canonical machine hardware profile on disk
            if not is_cpu_mode:
                purge = _MACHINE_SPECIFIC_CACHE_KEYS if _purge_machine_cache_on_mismatch else None
                _save_unified_cache({'config_json': np.array([json.dumps(_hardware_config)])}, purge_keys=purge)
        except Exception:
            pass
            
    # Apply settings
    selected_cpu_idx = _hardware_config['selected_cpu']
    selected_gpu_idx = _hardware_config['selected_gpu']
    gpu_detected = _hardware_config.get('gpu_detected', False)
    
    if gpu_detected and selected_gpu_idx is not None:
        os.environ['MOLMEM_SELECTED_GPU'] = str(selected_gpu_idx)
    else:
        os.environ.pop('MOLMEM_SELECTED_GPU', None)
    os.environ['MOLMEM_SELECTED_CPU'] = str(selected_cpu_idx)
    if gpu_detected and selected_gpu_idx is not None and os.environ.get("MOLMEM_CPU_ONLY") != "1" and not is_interactive_notebook():
        try:
            import torch
            if torch.cuda.is_available() and selected_gpu_idx < torch.cuda.device_count():
                torch.cuda.set_device(selected_gpu_idx)
        except Exception:
            pass

def _print_notebook_banner():
    """Prints a concise, single notebook-focused startup banner."""
    if (os.environ.get('MOLMEM_BANNER_SHOWN') == '1' or 
        os.environ.get('MOLMEM_SILENT_WARMUP') == '1' or
        is_worker_process()):
        return
    os.environ['MOLMEM_BANNER_SHOWN'] = '1'
    global _banner_printed
    _banner_printed = True

    target_stream = sys.stdout
    if hasattr(target_stream, '_target') and getattr(target_stream, '_target', None) is not None:
        target_stream = target_stream._target

    banner_lines = [
        "=" * 88,
        f"  MolmemSimulator v{__version__} | Molecular Memristor Simulation Platform",
        "  Publisher : Logan Larsh (logan.c.larsh-1@ou.edu) | License : MIT",
        "  Platform  : Interactive Notebook (GPU bypassed; CPU-only Numba JIT solver active)",
        "  Silence   : Library output silenced by default to preserve clean notebook cells.",
        "  Unsilence : To view verbose simulation logs, run: molmem_lib.unsilence_molmem_lib()",
        "=" * 88,
        ""
    ]
    try:
        target_stream.write("\n".join(banner_lines) + "\n")
        target_stream.flush()
    except Exception:
        print("\n".join(banner_lines))

def _print_startup_banner():
    import os
    import sys
    from .sys_utils import is_interactive_notebook
    global _hardware_config, _banner_printed
    
    if (os.environ.get('MOLMEM_SILENT') == '1' or 
        os.environ.get('MOLMEM_SILENT_WARMUP') == '1' or
        os.environ.get('MOLMEM_BANNER_SHOWN') == '1' or
        is_worker_process()):
        return False
        
    if any(arg == '-c' for arg in sys.argv):
        return False

    if is_interactive_notebook():
        _print_notebook_banner()
        return True

    if not _hardware_config:
        return False
        
    os.environ['MOLMEM_BANNER_SHOWN'] = '1'
    _banner_printed = True
        
    cpu_name = _hardware_config.get('cpu_name', 'Unknown CPU')
    if "GHz" not in cpu_name and "@" not in cpu_name:
        try:
            import psutil
            freq = psutil.cpu_freq()
            if freq and getattr(freq, 'max', 0) > 0:
                max_freq_ghz = freq.max / 1000.0
                cpu_name = f"{cpu_name} @ {max_freq_ghz:.2f} GHz"
        except Exception:
            pass
    cpu_cores_info = _hardware_config.get('cpu_cores_info', '')
    cpu_cache_info = _hardware_config.get('cpu_cache_info')
    if not cpu_cache_info:
        cpu_cache_info = _query_cpu_caches()
        _hardware_config['cpu_cache_info'] = cpu_cache_info
        
    gpu_detected = _hardware_config.get('gpu_detected', False)
    gpu_name = _hardware_config.get('gpu_name', '')
    gpu_vram = _hardware_config.get('gpu_vram', 0.0)
    selected_gpu = _hardware_config.get('selected_gpu', 0)
    selected_cpu = _hardware_config.get('selected_cpu', 0)
    cpu_sockets = _hardware_config.get('cpu_sockets', [])
    gpu_devices = _hardware_config.get('gpu_devices', [])
    gpu_sm_count = _hardware_config.get('gpu_sm_count')
    if not gpu_sm_count and gpu_detected:
        for dev in gpu_devices:
            if dev.get('index') == selected_gpu:
                gpu_sm_count = dev.get('sm_count')
                break
    
    _flush_system_notifications()
    print("=" * 105)
    print("              MOLECULAR MEMRISTOR SIMULATION PLATFORM (MOLMEM)")
    print("=" * 105)
    print(f"  Developer : Logan Larsh (logan.c.larsh-1@ou.edu) | License : MIT")
    print(f"  Version   : {__version__}")
    amd_supp = "AMD ROCm RDNA 2 – RDNA 4 (gfx1030 – gfx1201)" if sys.platform == "win32" else "AMD ROCm RDNA 1 – 4 / CDNA 1 – 3 (gfx906 – gfx1201)"
    print(f"  Supported : NVIDIA SM 7.5 – SM 9.0 (Turing, Ampere, Ada Lovelace, Hopper; Max SM_90)")
    print(f"              {amd_supp}")
    print()
    print(f"  Host CPU  : {cpu_name}")
    if len(cpu_sockets) > 1:
        for s in cpu_sockets:
            idx = s['index']
            n_cores = len(s['cores']) if s['cores'] else s['cores_count']
            clock_ghz = s['clock_mhz'] / 1000.0
            l3_mb = s['l3_kb'] / 1024.0
            l3_str = f"{int(l3_mb) if l3_mb.is_integer() else l3_mb:.1f} MB" if l3_mb > 0 else "Unknown"
            active_tag = " [Active]" if idx == selected_cpu else ""
            print(f"              Socket {idx} : {n_cores} Cores @ {clock_ghz:.2f} GHz | L3 Cache: {l3_str}{active_tag}")
        ram_part = cpu_cores_info.split('|')[-1].strip() if '|' in cpu_cores_info else cpu_cores_info
        print(f"              {ram_part}")
    else:
        print(f"              {cpu_cores_info}")
    if cpu_cache_info:
        print(f"              Cache : {cpu_cache_info}")
    print()
    gpu_type = _hardware_config.get('gpu_type', 'NVIDIA CUDA')
    if gpu_detected:
        print(f"  Graphics  : {gpu_type} GPU Acceleration Enabled")
        if len(gpu_devices) > 1:
            for dev in gpu_devices:
                idx = dev['index']
                name = dev['name']
                vram = dev['vram']
                sm_count = dev.get('sm_count', 0)
                is_rocm_dev = dev.get('is_rocm', False) or gpu_type == "AMD ROCm"
                cu_label = "CUs" if is_rocm_dev else "SMs/CUs"
                sm_str = f" | {cu_label} : {sm_count}" if sm_count else ""
                is_supp = dev.get('supported', True)
                if not is_supp:
                    reason = dev.get('unsupported_reason') or f"SM_{dev.get('major', 0)}{dev.get('minor', 0)} exceeds max SM_90 for current Molmem_Lib version"
                    active_tag = f" [Unavailable: {reason}]"
                elif idx == selected_gpu:
                    active_tag = " [Active]"
                else:
                    active_tag = ""
                print(f"              Device {idx} : {name} | VRAM : {vram:.2f} GB{sm_str}{active_tag}")
        else:
            is_rocm_dev = gpu_type == "AMD ROCm"
            cu_label = "CUs" if is_rocm_dev else "SMs/CUs"
            sm_str = f" | {cu_label} : {gpu_sm_count}" if gpu_sm_count else ""
            print(f"              Device {selected_gpu} : {gpu_name} | VRAM : {gpu_vram:.2f} GB{sm_str}")
    else:
        # Check if there were unsupported physical GPUs to provide actionable diagnostic feedback
        unsupp_gpus = [d for d in gpu_devices if not d.get('supported', True)]
        if unsupp_gpus:
            first_unsupp = unsupp_gpus[0]
            reason = first_unsupp.get('unsupported_reason') or f"SM_{first_unsupp.get('major', 0)}{first_unsupp.get('minor', 0)} exceeds max SM_90"
            print(f"  Graphics  : Device {first_unsupp.get('index', 0)} : {first_unsupp.get('name', 'GPU')} [Unavailable: {reason}]")
            print(f"              (Defaulting to CPU-only mode utilizing Numba JIT loops)")
        else:
            print(f"  Graphics  : No hardware acceleration compatible GPU detected.")
            print(f"              (Running in CPU-only mode utilizing Numba JIT loops)")
    print()
    print(f"  Cache     : Run molmem_lib.clear_cache() (--clear-cache) or molmem_lib.sanitize_distribution() (--clean-dist).")
    print(f"  Silence   : Suppress library output via molmem_lib.silenceMolmemLib() (or CLI flag --silent).")
    
    # Print compilation info
    has_ext = False
    try:
        from .torch_simulator import HAS_CPP_EXTENSION
        has_ext = HAS_CPP_EXTENSION
    except Exception:
        try:
            import molmem_cuda_solver
            has_ext = True
        except ImportError:
            pass
        
    is_rocm = False
    has_torch_cuda = False
    try:
        import torch
        is_rocm = getattr(torch.version, 'hip', None) is not None
        has_torch_cuda = torch.cuda.is_available()
    except Exception:
        pass

    phys_gpus = _detect_physical_gpus()
    phys_str = " ".join(phys_gpus).lower() if phys_gpus else ""

    if is_rocm:
        gpu_type = "AMD ROCm"
        is_amd = True
        is_nvidia = False
    elif has_torch_cuda or any(k in phys_str for k in ["nvidia", "geforce", "quadro", "rtx"]):
        gpu_type = "NVIDIA CUDA"
        is_nvidia = True
        is_amd = False
    elif any(k in phys_str for k in ["amd", "ati", "radeon", "navi", "raphael"]):
        gpu_type = "AMD ROCm"
        is_amd = True
        is_nvidia = False
    else:
        gpu_type = "Native"
        is_amd = False
        is_nvidia = False

    has_ext = False
    disabled_reason = None
    active_arch = None
    try:
        from . import torch_simulator
        if hasattr(torch_simulator, 'check_has_cpp_extension'):
            has_ext = torch_simulator.check_has_cpp_extension(device_idx=selected_gpu)
        else:
            has_ext = getattr(torch_simulator, 'HAS_CPP_EXTENSION', False)
        disabled_reason = getattr(torch_simulator, 'CPP_EXTENSION_DISABLED_REASON', None)
        active_arch = getattr(torch_simulator, 'ACTIVE_GPU_ARCH', None)
    except Exception:
        has_ext = False
        disabled_reason = "PyTorch not installed (running CPU execution mode)"
        active_arch = None

    print(f"  Extension : {gpu_type} C++ Extension | Compiled & Ready: {has_ext}")
    if not has_ext and (gpu_detected or phys_gpus):
        is_nb = is_interactive_notebook()
        is_colab = ('COLAB_GPU' in os.environ) or ('google.colab' in sys.modules)

        if is_nb:
            py_ver = f"Python {sys.version_info.major}.{sys.version_info.minor}"
            if is_colab:
                print(f"               Status : Cloud notebook ({py_ver}); pre-compiled native C++/HIP binary not bundled.")
                print(f"               Action : To enable GPU acceleration, select 'Runtime' -> 'Change runtime type' -> 'T4 GPU'.")
                print(f"                        Simulator persists seamlessly using PyTorch tensor kernels / Numba CPU solver.")
            else:
                workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                has_molmem_env = os.path.exists(os.path.join(workspace_root, "molmem-env"))
                kernel_target = "'molmem-env'" if has_molmem_env else "a GPU-enabled environment"
                print(f"               Status : Active Jupyter kernel ({py_ver}) does not have compiled GPU extension.")
                print(f"               Action : To use GPU acceleration, switch Jupyter kernel to {kernel_target} ('Kernel' -> 'Change Kernel...').")
                print(f"                        Otherwise, no action required: simulator persists using high-performance CPU / Numba JIT solver.")
        missing, advice = _get_missing_compiler_items(has_amd=is_amd, has_nvidia=is_nvidia)

        if missing:
            print(f"               Status : Native C++ extension not built (Missing: {', '.join(missing)})")
            if advice:
                for idx, adv in enumerate(advice):
                    prefix = "Action : " if idx == 0 else "         "
                    print(f"               {prefix}{adv}")
            else:
                print(f"               Action : Install required build tools to enable native C++/HIP kernel compilation.")
            print(f"                        Simulator persists using PyTorch {gpu_type} JIT / Numba fallback.")
        elif disabled_reason:
            print(f"               Status : {disabled_reason}")
            arch_str = active_arch if active_arch else "active GPU"
            sdk_name = "NVIDIA CUDA Toolkit" if is_nvidia else "AMD ROCm SDK"
            compile_cmd = "molmem_lib/csrc/build_ext.bat" if sys.platform == "win32" else "molmem_lib/csrc/compile.sh"
            print(f"               Action : Recompile extension targeting {arch_str} via {sdk_name} (run: {compile_cmd}).")
            print(f"                        Simulator persists using high-performance CPU / Numba JIT solver.")
        else:
            print(f"               Status : Extension source ready to auto-compile on next run.")
    print("=" * 105)
    print()
    return True

def _compute_per_file_hashes():
    import hashlib, glob
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = sorted(glob.glob(os.path.join(lib_dir, "*.py")))
    hashes = {}
    for py_file in source_files:
        basename = os.path.basename(py_file)
        try:
            with open(py_file, 'rb') as f:
                hashes[basename] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
    return hashes

def _check_modular_source_hashes():
    """
    Computes per-module SHA-256 cryptographic footprints for all source files in molmem_lib.
    Selectively invalidates only the specific compiled bytecode (.nbc) and calibration profiles
    corresponding to the modified files, leaving unaffected compiled kernels 100% cached.
    """
    import json, glob
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(lib_dir, '.calcache')
    os.makedirs(cache_dir, exist_ok=True)
    hash_file = os.path.join(cache_dir, 'modular_source_hashes.json')
    lib_pycache = os.path.join(lib_dir, '__pycache__')
    
    current_hashes = _compute_per_file_hashes()
    stored_hashes = {}
    if os.path.isfile(hash_file):
        try:
            with open(hash_file, 'r') as f:
                stored_hashes = json.load(f)
        except Exception:
            stored_hashes = {}
            
    # Find modified or newly added files
    modified_files = set()
    for fname, hval in current_hashes.items():
        if stored_hashes.get(fname) != hval:
            modified_files.add(fname)
            
    if not modified_files:
        return # Zero modifications! 100% cache hit!
        
    # Determine affected modules and invalidate selectively
    invalidated_targets = []
    
    # 1. Device Physics / Arrhenius ODE
    if 'device.py' in modified_files or 'device_jit.py' in modified_files:
        invalidated_targets.append("Device ODE & Spline Physics")
        if os.path.exists(lib_pycache):
            for pattern in ['device_jit*.nbc', 'device_jit*.nbi', 'device*.nbc', 'device*.nbi']:
                for f in glob.glob(os.path.join(lib_pycache, pattern)):
                    try: os.remove(f)
                    except Exception: pass
        for cal_f in glob.glob(os.path.join(cache_dir, 'grid_calibration_*.npz')):
            try: os.remove(cal_f)
            except Exception: pass
            
    # 2. Solver & Nodal Integration Loops
    if 'solver.py' in modified_files:
        invalidated_targets.append("Nodal Solver & Assembly Loops")
        if os.path.exists(lib_pycache):
            for pattern in ['solver*.nbc', 'solver*.nbi']:
                for f in glob.glob(os.path.join(lib_pycache, pattern)):
                    try: os.remove(f)
                    except Exception: pass
                    
    # 3. 3D Pulse Map Inverse Programmer
    if 'programmer_solver.py' in modified_files or 'programmer.py' in modified_files:
        invalidated_targets.append("3D Pulse Map Inverse Solver")
        if os.path.exists(lib_pycache):
            for pattern in ['programmer_solver*.nbc', 'programmer_solver*.nbi', 'programmer*.nbc', 'programmer*.nbi']:
                for f in glob.glob(os.path.join(lib_pycache, pattern)):
                    try: os.remove(f)
                    except Exception: pass
        torch_cache = os.path.join(cache_dir, 'torchscript_pwl.pt')
        if os.path.exists(torch_cache):
            try: os.remove(torch_cache)
            except Exception: pass
        for cal_f in glob.glob(os.path.join(cache_dir, 'grid_calibration_*.npz')):
            try: os.remove(cal_f)
            except Exception: pass
            
    # 4. Quantization
    if 'quantization.py' in modified_files:
        invalidated_targets.append("Quantization Mapping Tables")
        
    # 5. Profiler
    if 'profiler.py' in modified_files:
        invalidated_targets.append("Hardware Affinity Thread Profiler")
        for prof_f in glob.glob(os.path.join(cache_dir, '*affinity*.json')) + glob.glob(os.path.join(cache_dir, '*profiler*.npz')):
            try: os.remove(prof_f)
            except Exception: pass

    # Save updated modular hashes
    try:
        with open(hash_file, 'w') as f:
            json.dump(current_hashes, f, indent=2)
    except Exception:
        pass
        
    if invalidated_targets:
        add_system_notification(
            event=f"Source Modified: {', '.join(sorted(modified_files))}",
            action=f"Selectively invalidated targets: {', '.join(invalidated_targets)}",
            reason="Preserves unaffected pre-compiled bytecode while guaranteeing 100% precision for modified modules"
        )

_banner_printed = False

def ensure_startup_banner():
    """Prints the startup banner once unless the library has been silenced."""
    global _banner_printed
    if _banner_printed or is_mp_child or isMolmemLibSilenced() or os.environ.get('MOLMEM_BANNER_SHOWN') == '1':
        return
    if is_interactive_notebook():
        _banner_printed = True
        _print_notebook_banner()
    else:
        if _print_startup_banner():
            _banner_printed = True

def show_startup_banner():
    """Explicitly displays the Molmem simulation platform banner."""
    global _banner_printed
    _banner_printed = True
    if is_interactive_notebook():
        _print_notebook_banner()
    else:
        _print_startup_banner()

if not is_mp_child:
    _check_modular_source_hashes()
    _initialize_hardware_targets()
    _onboard_compiler_extension()
    # In interactive notebooks, print the single concise startup banner and engage default silence immediately
    if is_interactive_notebook():
        _print_notebook_banner()
        silenceMolmemLib(True)
    elif not isMolmemLibSilenced() and os.environ.get('MOLMEM_SILENT') != '1':
        _print_startup_banner()

if is_logging_enabled() and not any(isinstance(f, TorchImportTracer) for f in sys.meta_path):
    sys.meta_path.insert(0, TorchImportTracer())

