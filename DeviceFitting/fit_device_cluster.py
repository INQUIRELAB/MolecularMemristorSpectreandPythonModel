import os
import sys

# Multiprocessing serialization compatibility:
# Alias '__reloaded__' in sys.modules so worker processes spawned on Windows can always resolve functions
if "__main__" in sys.modules:
    sys.modules["__reloaded__"] = sys.modules["__main__"]

# Elevate Linux file descriptor limit (RLIMIT_NOFILE) to prevent Errno 24 with high worker counts
if sys.platform != "win32":
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if (hard != -1 and hard != getattr(resource, 'RLIM_INFINITY', -1)) else 1048576
        target = max(soft, min(target, 1048576))
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

# Limit CPU threading for background libraries to prevent thread over-subscription deadlocks in multiprocessing pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# Add PythonSimulator to sys.path and import molmem_lib first so environment and libraries auto-configure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PythonSimulator')))
from molmem_lib import MolmemSimulator
from molmem_lib.sources import PulseSource
from molmem_lib.devices import default_device

import signal
# Prevent molmem_lib's sys_utils from treating normal POSIX SIGTERM as a native hardware crash
try:
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
except Exception:
    pass

import warnings
# Silence SciPy solver deprecation warnings to keep stdout clean for status bar drawing
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Windows Console VT100 / ANSI in-place rendering & Instant Ctrl+C handler
_WIN32_CTRL_HANDLER_REF = None
GLOBAL_DISPATCHER_REF = None
GLOBAL_BEACON_REF = None

def init_windows_console():
    """
    Initializes Windows console:
    1. Enables ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004) and ENABLE_PROCESSED_OUTPUT (0x0001)
       so ANSI escape sequences (\033[...]) operate in-place without scrolling.
    2. Disables ENABLE_QUICK_EDIT_MODE (0x0040) on stdin so mouse/trackpad clicks
       do not freeze console execution or hijack Ctrl+C into clipboard copy.
    3. Registers a native Win32 ConsoleCtrlHandler so Ctrl+C immediately terminates the process.
    """
    if sys.platform != "win32":
        return
    # Strictly bypass in any spawned multiprocessing worker
    if any(arg == '-c' for arg in sys.argv) or any(s in " ".join(sys.argv) for s in ('spawn_main', 'pipe_handle', 'parent_sentinel', '--multiprocessing-fork')):
        return
    import multiprocessing
    if hasattr(multiprocessing, 'parent_process') and multiprocessing.parent_process() is not None:
        return
    if multiprocessing.current_process().name != 'MainProcess':
        return

    try:
        os.system('')  # Triggers cmd.exe/powershell VT100 initialization
    except Exception:
        pass

    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32

        # 1. Enable VT100 on stdout and CONOUT$
        for h_id in (-11,):  # STD_OUTPUT_HANDLE
            h_out = kernel32.GetStdHandle(h_id)
            out_mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_out, ctypes.byref(out_mode)):
                kernel32.SetConsoleMode(h_out, out_mode.value | 0x0004 | 0x0001 | 0x0002)

        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        OPEN_EXISTING = 3
        h_conout = kernel32.CreateFileW("CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
        if h_conout != -1 and h_conout != 0:
            conout_mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_conout, ctypes.byref(conout_mode)):
                kernel32.SetConsoleMode(h_conout, conout_mode.value | 0x0004 | 0x0001 | 0x0002)
            kernel32.CloseHandle(h_conout)

        # 2. Disable QuickEdit on stdin and CONIN$
        for h_id in (-10,):  # STD_INPUT_HANDLE
            h_in = kernel32.GetStdHandle(h_id)
            in_mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_in, ctypes.byref(in_mode)):
                new_in_mode = (in_mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_in, new_in_mode)

        h_conin = kernel32.CreateFileW("CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
        if h_conin != -1 and h_conin != 0:
            conin_mode = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_conin, ctypes.byref(conin_mode)):
                new_in_mode = (conin_mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_conin, new_in_mode)
            kernel32.CloseHandle(h_conin)

        # 3. Native Win32 ConsoleCtrlHandler
        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        def _win32_ctrl_handler(ctrl_type):
            if ctrl_type in (0, 1, 2):  # CTRL_C, CTRL_BREAK, CTRL_CLOSE
                try:
                    if 'GLOBAL_NODE_TABLE' in globals():
                        GLOBAL_NODE_TABLE.clear()
                    if 'GLOBAL_LOBBY_TABLE' in globals():
                        GLOBAL_LOBBY_TABLE.clear()
                    if 'GLOBAL_CLUSTER_TABLE' in globals():
                        GLOBAL_CLUSTER_TABLE.clear()
                except Exception:
                    pass
                sys.stdout.write("\n[Shutdown] Process terminated by user (Ctrl+C). Exiting cleanly.\n")
                sys.stdout.flush()
                # Stop UDP beaconing and gracefully notify all cluster nodes over TCP
                global GLOBAL_BEACON_REF, GLOBAL_DISPATCHER_REF
                if GLOBAL_BEACON_REF is not None:
                    try:
                        GLOBAL_BEACON_REF.stop()
                    except Exception:
                        pass
                if GLOBAL_DISPATCHER_REF is not None:
                    try:
                        GLOBAL_DISPATCHER_REF.shutdown_cluster()
                    except Exception:
                        pass
                # Clean up any active child processes
                try:
                    import multiprocessing
                    for child in multiprocessing.active_children():
                        try:
                            child.terminate()
                        except Exception:
                            pass
                except Exception:
                    pass
                os._exit(0)
                return True
            return False

        global _WIN32_CTRL_HANDLER_REF
        _WIN32_CTRL_HANDLER_REF = HandlerRoutine(_win32_ctrl_handler)
        kernel32.SetConsoleCtrlHandler(_WIN32_CTRL_HANDLER_REF, True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from scipy.optimize import minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy._lib._util import MapWrapper
from scipy.interpolate import PchipInterpolator

import socket
import select
import struct
import zlib
import pickle
import threading
import time
import math
import argparse
import collections
import queue
import shutil
import hashlib
import subprocess
import io
import zipfile
import json
import datetime

try:
    import psutil
except Exception:
    psutil = None

class GlobalCpuSampler:
    """
    Thread-safe, throttled CPU utilization tracker with continuous-time EMA smoothing.
    
    Prevents sub-timer sampling artifacts on Windows (where kernel timers tick at 15.625 ms)
    and eliminates thread-ID isolation bugs in psutil.cpu_percent():
    - Throttles OS kernel counter reads to a minimum interval (0.5s).
    - Rapid queries (e.g. 500 Hz inside local_feeder dispatch loops) return cached EMA instantly.
    - Computes global (user + system) / total CPU deltas directly from psutil.cpu_times().
    - Smooths readings with a continuous-time low-pass filter (tau = 1.5s).
    """
    def __init__(self, sample_interval=0.5, tau=1.5):
        self.sample_interval = sample_interval
        self.tau = tau
        self.lock = threading.Lock()
        self.last_sample_t = time.time()
        self.last_cpu_times = None
        self.ema_cpu = 0.0
        self.initialized = False
        if psutil is not None:
            try:
                self.last_cpu_times = psutil.cpu_times()
            except Exception:
                pass

    def get_util(self):
        if psutil is None:
            return 0.0
        now = time.time()
        # Fast non-blocking check: if within sample_interval, return cached value immediately
        if self.initialized and (now - self.last_sample_t) < self.sample_interval:
            return float(self.ema_cpu)

        with self.lock:
            # Double check inside lock
            if self.initialized and (now - self.last_sample_t) < self.sample_interval:
                return float(self.ema_cpu)
            try:
                cur_times = psutil.cpu_times()
                dt = now - self.last_sample_t
                self.last_sample_t = now

                if self.last_cpu_times is None:
                    self.last_cpu_times = cur_times
                    return float(self.ema_cpu)

                d_user = cur_times.user - self.last_cpu_times.user
                d_system = cur_times.system - self.last_cpu_times.system
                d_idle = cur_times.idle - self.last_cpu_times.idle
                self.last_cpu_times = cur_times

                d_total = d_user + d_system + d_idle
                if d_total <= 0:
                    raw = self.ema_cpu
                else:
                    raw = ((d_user + d_system) / d_total) * 100.0

                if not self.initialized:
                    self.ema_cpu = raw
                    self.initialized = True
                else:
                    alpha = 1.0 - math.exp(-max(0.1, dt) / self.tau)
                    self.ema_cpu = alpha * raw + (1.0 - alpha) * self.ema_cpu

                return float(self.ema_cpu)
            except Exception:
                return float(self.ema_cpu)

GLOBAL_CPU_SAMPLER = GlobalCpuSampler(sample_interval=0.5, tau=1.5)

def get_system_cpu_util():
    """Non-blocking query of global system CPU percentage (cached / EMA-smoothed)."""
    return GLOBAL_CPU_SAMPLER.get_util()

class CpuGovernor:
    """
    Anti-hunting CPU Utilization Governor targeting a 95% CPU utilization setpoint.
    Dynamically scales active worker concurrency to prevent thermal throttling and OS thrashing:
    - Slew-rate limited: adjusts concurrency by at most +/- 1 worker per second.
    - Deadband [92.0%, 98.0%]: eliminates micro-chatter when near 95%.
    - Zero pool churn: throttles active in-flight task limit rather than destroying/recreating pools.
    """
    def __init__(self, total_cores, initial_workers, target_util=95.0, deadband=3.0):
        self.total_cores = total_cores
        self.max_workers = max(1, min(initial_workers, 61))
        self.min_workers = 1
        self.active_workers = self.max_workers
        self.target_util = target_util
        self.deadband = deadband  # [92.0%, 98.0%]
        self.last_step_t = time.time()
        self.current_cpu_util = 0.0

    @property
    def cap_ratio(self):
        """Ratio of dynamic CPU-capped capacity to baseline capacity: active_workers / max_workers."""
        if self.max_workers <= 0:
            return 1.0
        return float(self.active_workers) / float(self.max_workers)

    def update(self, cpu_util=None):
        now = time.time()
        if cpu_util is None:
            cpu_util = get_system_cpu_util()
        self.current_cpu_util = cpu_util

        # Slew-rate limit: allow at most 1 adjustment every 1.0s
        if (now - self.last_step_t) < 1.0:
            return self.active_workers

        # Deadband check around target 80.0%
        if cpu_util > (self.target_util + self.deadband):
            # Over-utilized (> 84%): shed 1 worker to eliminate thrashing
            if self.active_workers > self.min_workers:
                self.active_workers -= 1
                self.last_step_t = now
        elif cpu_util < (self.target_util - self.deadband):
            # Under-utilized (< 76%): absorb available headroom toward 80%
            if self.active_workers < self.max_workers:
                self.active_workers += 1
                self.last_step_t = now
        else:
            # Inside deadband: zero step timer so transitions out of deadband are prompt
            self.last_step_t = now - 0.5

        return self.active_workers

def is_multiprocessing_worker():
    """
    Returns True if the current process is a background multiprocessing child worker
    spawned by multiprocessing.Pool, avoiding expensive import-time file hashing and seals.
    """
    import multiprocessing
    if hasattr(multiprocessing, 'parent_process') and multiprocessing.parent_process() is not None:
        return True
    if multiprocessing.current_process().name != 'MainProcess':
        return True
    arg_str = " ".join(sys.argv)
    return any(s in arg_str for s in ('spawn_main', 'pipe_handle', 'parent_sentinel', '--multiprocessing-fork'))

def compute_script_version_hash(force=False):
    """
    Computes a deterministic cryptographic fingerprint of fit_device_cluster.py
    with normalized line endings (\r\n -> \n) to validate version parity across
    heterogeneous operating systems (Windows & Linux).
    Excludes the user-tunable configuration and hyperparameter block.
    """
    if not force and is_multiprocessing_worker():
        return "WORKER"
    try:
        script_path = os.path.abspath(__file__)
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().replace("\r\n", "\n")
            
        start_marker = "# >>> " + "CONFIGURATION & HYPERPARAMETERS (EXEMPT FROM CODE HASH) >>>"
        end_marker = "# <<< " + "END CONFIGURATION & HYPERPARAMETERS <<<"
        
        idx_start = content.find(start_marker)
        idx_end = content.find(end_marker)
        
        if idx_start != -1 and idx_end != -1 and idx_start < idx_end:
            content = content[:idx_start] + "\n" + content[idx_end + len(end_marker):]
            
        normalized = content.strip().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:16]
    except Exception:
        return "UNKNOWN"

def check_core_bundle_alignment(bundle_path, lib_dir):
    """
    Checks if molmem_core.bin exists and its attestation seal validates
    against the current lib_dir footprint. Returns True if aligned, False otherwise.
    """
    if not os.path.isfile(bundle_path):
        return False
    try:
        with open(bundle_path, "rb") as f:
            raw_bytes = f.read(48)
        if len(raw_bytes) < 48:
            return False
        import glob
        magic = raw_bytes[:8]
        if magic != b"MOLMEM\x03\x00":
            return False
        magic, version, num_sections, salt, wrapped_dek, auth_token, reserved = struct.unpack(
            "<8sIIQQQQ", raw_bytes[:48]
        )
        if version != 3:
            return False
            
        hasher = hashlib.sha256()
        source_files = sorted(glob.glob(os.path.join(lib_dir, "*.py")))
        for py_file in source_files:
            if os.path.basename(py_file) == "devices.py":
                continue
            try:
                with open(py_file, 'rb') as f:
                    content = f.read().replace(b'\r\n', b'\n')
                    hasher.update(content)
            except Exception:
                pass
        footprint = hasher.hexdigest()
        
        digest = hashlib.sha256(footprint.encode('ascii')).digest()
        kek = struct.unpack("<Q", digest[:8])[0]
        
        mask64 = 0xFFFFFFFFFFFFFFFF
        salt_scramble = (salt * 6364136223846793005 + 1442695040888963407) & mask64
        dek = (wrapped_dek ^ kek ^ salt_scramble) & mask64
        
        expected_token = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek)).digest()[:8])[0]
        if expected_token == auth_token:
            return True

        # Canonical footprint fallback (aligns with sys_utils.py)
        CANONICAL_FOOTPRINT = "2ab6592f9b19e9c23e46c7350c95ab6a2c01f117ce3bc3d4f30ef8ea1337050d"
        kek_digest_canon = hashlib.sha256(CANONICAL_FOOTPRINT.encode('ascii')).digest()
        kek_canon = struct.unpack("<Q", kek_digest_canon[:8])[0]
        dek_canon = (wrapped_dek ^ kek_canon ^ salt_scramble) & mask64
        token_canon = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek_canon)).digest()[:8])[0]
        return token_canon == auth_token
    except Exception:
        return False

def is_cluster_host_environment():
    """
    Returns True only if the current execution context is explicitly an authoritative
    Cluster Host coordinator. Compute nodes, worker processes, and node daemon sessions
    must NEVER attempt to re-seal or re-compile molmem_core.bin locally with ephemeral random salts.
    """
    if is_multiprocessing_worker():
        return False
    if os.environ.get("MOLMEM_CLUSTER_ROLE") in ("node", "worker", "slave"):
        return False
    for i, arg in enumerate(sys.argv):
        if arg == "--role" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1].lower()
            if val in ("node", "worker", "slave"):
                return False
            if val in ("host", "master"):
                return True
        elif arg.startswith("--role="):
            val = arg.split("=", 1)[1].lower()
            if val in ("node", "worker", "slave"):
                return False
            if val in ("host", "master"):
                return True
        elif arg == "--connect" or arg.startswith("--connect="):
            return False
    return ("--role" in sys.argv and any(r in sys.argv for r in ("host", "master"))) or os.environ.get("MOLMEM_CLUSTER_ROLE") in ("host", "master")


def ensure_molmem_core_sealed(molmem_lib_dir=None, force=False):
    """
    Ensures that molmem_core.bin in molmem_lib/data is sealed with the exact
    codebase integrity attestation key matching the current molmem_lib source.
    If mismatched, outdated, or forced, automatically invokes DevicePhysicsCompiler
    to re-encrypt and re-seal the binary asset bundle before hashing and OTA packaging.
    Strictly disabled on compute nodes to preserve authoritative master binary hashes.
    """
    if is_multiprocessing_worker():
        return True
    if not is_cluster_host_environment():
        if molmem_lib_dir is None:
            try:
                import molmem_lib
                molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
            except Exception:
                molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
        bundle_path = os.path.join(molmem_lib_dir, "data", "molmem_core.bin") if molmem_lib_dir else ""
        return os.path.isfile(bundle_path)

    if molmem_lib_dir is None:
        try:
            import molmem_lib
            molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
        except Exception:
            molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
            
    if not os.path.exists(molmem_lib_dir):
        print(f"[Attestation Error] Could not find molmem_lib directory at: {molmem_lib_dir}", flush=True)
        return False
        
    data_dir = os.path.join(molmem_lib_dir, "data")
    bundle_path = os.path.join(data_dir, "molmem_core.bin")
    
    # If already aligned and not forced, nothing to do
    if not force and check_core_bundle_alignment(bundle_path, molmem_lib_dir):
        return True
        
    # Check if compiler is available (Master node environment)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compiler_dir = os.path.join(workspace_root, "DevicePhysicsCompiler")
    compiler_script = os.path.join(compiler_dir, "compile_physics_data.py")
    
    if not os.path.isfile(compiler_script):
        if not os.path.isfile(bundle_path):
            print(
                "\n"
                "========================================================================================\n"
                " [!] MOLMEM CLUSTER ERROR: Physics core bundle 'molmem_core.bin' was not found\n"
                "     in 'PythonSimulator/molmem_lib/data', and 'DevicePhysicsCompiler' is not present\n"
                "     in this installation to compile it.\n"
                "     Please provide a valid 'molmem_core.bin' asset file.\n"
                "========================================================================================\n",
                flush=True
            )
            return False
        else:
            print(
                "\n"
                "========================================================================================\n"
                " [!] MOLMEM CLUSTER WARNING: 'molmem_core.bin' failed cryptographic attestation\n"
                "     against the current 'molmem_lib' source files, and 'DevicePhysicsCompiler' is not\n"
                "     present in this environment to re-seal it.\n"
                "     Please restore the original unmodified 'molmem_lib' files or obtain an updated bundle.\n"
                "========================================================================================\n",
                flush=True
            )
            return False
        
    print("[Attestation Master] Re-sealing molmem_core.bin to align with current molmem_lib codebase...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("compile_physics_data", compiler_script)
        if spec and spec.loader:
            compiler_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiler_mod)
            success = compiler_mod.compile_all(source_dir=compiler_dir, target_dir=data_dir)
            if success:
                print("[Attestation Master] molmem_core.bin successfully re-sealed and verified bit-exact.")
                return True
    except Exception as e:
        print(f"[Attestation Warning] In-process compilation error: {e}. Falling back to subprocess...")
        
    try:
        res = subprocess.run([sys.executable, compiler_script], cwd=compiler_dir, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print("[Attestation Master] molmem_core.bin successfully re-sealed via subprocess.")
            return True
        else:
            print(f"[Attestation Error] Compiler exited with code {res.returncode}: {res.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[Attestation Error] Could not run compiler: {e}")
        return False

def compute_molmem_lib_hash(molmem_lib_dir=None, force=False):
    """
    Computes a deterministic cryptographic fingerprint across all source files,
    GPU extensions (csrc), lookup tables, and the sealed physics binary (molmem_core.bin)
    in molmem_lib. Normalizes line endings (\r\n -> \n) and sorted relative paths.
    Excludes platform-specific compiled binaries (.pyd, .so), bytecode (__pycache__, .pyc),
    and local compiler caches (.csrc_hash, .csrc_arch).
    """
    if not force and is_multiprocessing_worker():
        return "WORKER"
    if molmem_lib_dir is None:
        try:
            import molmem_lib
            molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
        except Exception:
            molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
            
    if not os.path.exists(molmem_lib_dir):
        return "UNKNOWN_LIB"
        
    # Ensure molmem_core.bin is sealed and aligned before hashing if compiler is present
    try:
        ensure_molmem_core_sealed(molmem_lib_dir)
    except Exception:
        pass

    hasher = hashlib.sha256()
    try:
        file_list = []
        text_exts = (".py", ".pwl", ".json", ".hip", ".cpp", ".h", ".sh", ".bat")
        binary_exts = (".bin",)
        
        for root, dirs, files in os.walk(molmem_lib_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".calcache", ".temp_history", ".git", "build", "molmem_cuda_solver.egg-info")]
            for f in files:
                if f.startswith(".csrc_"):
                    continue
                if f.endswith(text_exts) or f.endswith(binary_exts):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, molmem_lib_dir).replace("\\", "/")
                    file_list.append((rel_path, full_path, f.endswith(text_exts)))
                    
        file_list.sort(key=lambda x: x[0])
        for rel_path, full_path, is_text in file_list:
            hasher.update(rel_path.encode("utf-8"))
            with open(full_path, "rb") as fh:
                raw = fh.read()
                if is_text:
                    raw = raw.replace(b"\r\n", b"\n")
                hasher.update(raw)
        return hasher.hexdigest()[:16]
    except Exception:
        return "UNKNOWN_LIB"

def package_molmem_lib_payload(molmem_lib_dir=None):
    """
    Packages all Python sources, GPU extensions (csrc), lookup tables, and the
    cryptographically sealed molmem_core.bin into a compressed in-memory zip archive.
    Strictly excludes platform-specific binaries (.pyd, .so) to guarantee OS neutrality.
    """
    if molmem_lib_dir is None:
        try:
            import molmem_lib
            molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
        except Exception:
            molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
            
    if not os.path.exists(molmem_lib_dir):
        return None
        
    # Guarantee binary seal alignment before packaging
    ensure_molmem_core_sealed(molmem_lib_dir)

    text_exts = (".py", ".pwl", ".json", ".hip", ".cpp", ".h", ".sh", ".bat")
    binary_exts = (".bin",)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(molmem_lib_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".calcache", ".temp_history", ".git", "build", "molmem_cuda_solver.egg-info")]
            for f in files:
                if f.startswith(".csrc_"):
                    continue
                if f.endswith(text_exts) or f.endswith(binary_exts):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, molmem_lib_dir).replace("\\", "/")
                    with open(full_path, "rb") as fh:
                        data = fh.read()
                    if f.endswith(text_exts):
                        data = data.replace(b"\r\n", b"\n")
                    zf.writestr(rel_path, data)
    return buf.getvalue()

def scavenge_ota_backups(backup_dir=None, max_keep=3):
    """
    Scavenges pre-OTA backup files and directories on compute nodes,
    ensuring no more than `max_keep` (default: 3) most recent versions are retained.
    Prunes versions older than 3 OTAs strictly and exclusively for:
      1. fit_device_cluster_pre_ota_<timestamp>.py
      2. molmem_lib_pre_ota_<timestamp> (directory)
    Positively ignores and preserves all other backups (e.g. fit_device_subset,
    sys_utils, HardwareTestTB, manual backups, or any other timestamped archives).
    """
    if backup_dir is None:
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backup_dir = os.path.join(workspace_root, "_backups")
    if not os.path.isdir(backup_dir):
        return

    import re
    # Strict regex whitelisting for ONLY clustered fitting pre-OTA artifacts:
    # 1. fit_device_cluster_pre_ota_<timestamp>.py (regular file)
    # 2. molmem_lib_pre_ota_<timestamp> (directory)
    cluster_script_pattern = re.compile(r"^fit_device_cluster_pre_ota_\d+\.py$")
    molmem_lib_pattern = re.compile(r"^molmem_lib_pre_ota_\d+$")

    try:
        categories = {
            "fit_device_cluster": [],
            "molmem_lib": []
        }

        for name in os.listdir(backup_dir):
            full_path = os.path.join(backup_dir, name)
            
            # Explicitly verify exact filename pattern and file/dir type
            if cluster_script_pattern.match(name) and os.path.isfile(full_path):
                category = "fit_device_cluster"
            elif molmem_lib_pattern.match(name) and os.path.isdir(full_path):
                category = "molmem_lib"
            else:
                # Strictly bypass every other backup file, script, or directory
                continue

            try:
                mtime = os.path.getmtime(full_path)
            except Exception:
                mtime = 0.0
            categories[category].append((mtime, full_path))

        for category, items in categories.items():
            if len(items) <= max_keep:
                continue
            # Sort descending by mtime (newest first)
            items.sort(key=lambda x: x[0], reverse=True)
            # Prune only versions older than max_keep
            to_prune = items[max_keep:]
            for _, item_path in to_prune:
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
    except Exception:
        pass

def apply_molmem_lib_payload(payload_bytes, molmem_lib_dir=None):
    """
    Backs up the existing molmem_lib directory on the compute node, unpacks the fresh
    zip payload delivered by Master (including sealed molmem_core.bin and csrc sources),
    purges __pycache__, and invalidates .csrc_hash so local GPU compilation triggers on boot.
    """
    if not payload_bytes:
        return False
        
    if molmem_lib_dir is None:
        try:
            import molmem_lib
            molmem_lib_dir = os.path.dirname(os.path.abspath(molmem_lib.__file__))
        except Exception:
            molmem_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PythonSimulator", "molmem_lib")
            
    try:
        backup_parent = os.path.join(os.path.dirname(os.path.dirname(molmem_lib_dir)), "_backups")
        os.makedirs(backup_parent, exist_ok=True)
        backup_dest = os.path.join(backup_parent, f"molmem_lib_pre_ota_{int(time.time())}")
        try:
            shutil.copytree(molmem_lib_dir, backup_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".calcache", ".temp_history"))
        except Exception:
            pass
        scavenge_ota_backups(backup_dir=backup_parent, max_keep=3)

        buf = io.BytesIO(payload_bytes)
        with zipfile.ZipFile(buf, mode="r") as zf:
            payload_extracted = set()
            for zip_info in zf.infolist():
                extracted_path = os.path.join(molmem_lib_dir, zip_info.filename.replace("/", os.sep))
                payload_extracted.add(os.path.abspath(extracted_path))
                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                with zf.open(zip_info) as src, open(extracted_path, "wb") as dst:
                    dst.write(src.read())

            # Prune any stale/orphan source or binary files in molmem_lib that are not in master's payload
            for root, dirs, files in os.walk(molmem_lib_dir):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", ".calcache", ".temp_history", ".git", "build", "molmem_cuda_solver.egg-info", "logs")]
                for f in files:
                    if f.startswith(".csrc_"):
                        continue
                    if f.endswith((".py", ".bin", ".pwl", ".json", ".hip", ".cpp", ".h", ".sh", ".bat")):
                        full_f = os.path.abspath(os.path.join(root, f))
                        if full_f not in payload_extracted:
                            try:
                                os.remove(full_f)
                            except Exception:
                                pass

        # Purge stale bytecode caches
        for root, dirs, files in os.walk(molmem_lib_dir):
            if os.path.basename(root) == "__pycache__":
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                        
        # Invalidate GPU extension hash so local toolchain compiles fresh binary if present
        csrc_hash_file = os.path.join(molmem_lib_dir, "csrc", ".csrc_hash")
        if os.path.exists(csrc_hash_file):
            try:
                os.remove(csrc_hash_file)
            except Exception:
                pass
                
        return True
    except Exception as e:
        print(f"[OTA Error] Failed applying molmem_lib payload: {e}")
        return False

SCRIPT_CODE_HASH = compute_script_version_hash()
MOLMEM_LIB_HASH = compute_molmem_lib_hash()

# >>> CONFIGURATION & HYPERPARAMETERS (EXEMPT FROM CODE HASH) >>>
# =============================================================================
# OPTIMIZATION HYPERPARAMETERS & CONFIGURATION (EASY TUNING)
# =============================================================================

# --- 1. Differential Evolution (DE) Hyperparameters ---
POP_SIZE = 30000                    # Population size multiplier for DE
MAX_ITER = 50                      # Maximum DE generations (Phase 1)
TOL = 0.0                          # DE convergence tolerance
DE_LHS_BLEND = 0.1                 # Blending ratio: 10% Latin Hypercube, 90% Random uniform
DE_ELITE_RATIO = 0.10              # Top elite fraction (10%) carried over unchanged
DE_EXPLORER_RATIO = 0.10           # Explorer fraction (10%) injected with randomized traits
DE_MIN_CLUSTER_DIST = 0.008        # Spatial exclusion distance against archive (de-clustering)
REFINEMENT_SWEEP_STEPS = 10000      # Coordinate refinement steps around final DE candidate
DITHER_DECAY_START = 1.0           # Initial dither mutation ceiling (f_max)
DITHER_DECAY_FLOOR = 0.10          # Minimum dither mutation floor
CENTER_W_BASE_INIT = 0.60          # Curvature-steering baseline weight (floor raised to 0.50 to give linear tail equal weight)

# --- 2. Multi-Basin Polish (Phase 2 L-BFGS-B) Hyperparameters ---
LBFGS_MAX_ITER = 40                # Maximum L-BFGS-B iterations per basin
LBFGS_TOL = 5e-4                   # L-BFGS-B gradient tolerance
LBFGS_MAX_BASINS = 10               # Number of diverse elite basins to polish
JACOBIAN_EPSILON = 1e-4            # Central-difference step size ratio (42 evaluations / step)

# --- 3. Objective Function & Cost Formulation ---
USE_RAW_EMPIRICAL_RANGE = True     # Retain raw unscaled empirical bounds
USE_NORMALIZED_MSE = True          # Normalize MSE by empirical data variance
GSCALE_NOMINAL = 6.30              # Empirical baseline
GSCALE_PRIOR_WEIGHT = 0.02         # Soft Bayesian/Tikhonov prior weight penalizing gScale deviation from nominal baseline

# --- 4. Simulation & Pulse Train Timing ---
DEFAULT_T_PERIOD = 160e-9          # Pulse period (160 ns)
DEFAULT_T_WIDTH_POT = 80e-9        # Potentiation pulse width (80 ns)
DEFAULT_T_WIDTH_DEP = 60e-9        # Depression pulse width (60 ns)
DEFAULT_V_POT = 0.90               # Potentiation pulse amplitude (0.90 V)
DEFAULT_V_DEP = -0.75              # Depression pulse amplitude (-0.75 V)
DEFAULT_V_POT_122 = 1.22           # Saturation pulse amplitude (1.22 V)
SIM_TOL = 1e-3                     # ODE integration relative tolerance

# --- 5. Network & Cluster Configuration ---
DEFAULT_BEACON_PORT = 48898        # UDP broadcast port for auto-discovery
DEFAULT_TCP_PORT = 48899           # TCP port for streaming candidate batches
BEACON_TIMEOUT = 1.0               # Seconds to probe for existing Master before electing
DEFAULT_WAN_HOST = os.environ.get("MOLMEM_WAN_HOST", "computecluster.larshindustries.com")  # Cloudflare Tunnel / unproxied DNS hostname
TARGET_CPU_UTIL = 87.0             # Target CPU utilization setpoint (default 95.0%)
CPU_GOVERNOR_DEADBAND = 3.0        # Governor deadband (+/- 3.0%, i.e. [92.0%, 98.0%])
COMM_CORES_FLOOR = 2               # Minimum communication cores on Master when remote nodes are active (default 2)

# --- 6. 20 Parameter Space: Names, Log-Scale Masks, and Default Search Bounds ---
PARAM_NAMES = [
    "kappa", "alpha", "vTh", "gScale", "rC", 
    "vSmooth", "nSmooth", "kDischarge",
    "E_a", "R_th", "tau_th", "gamma", "beta_alpha", "beta_s",
    "rate_asymmetry", "fDischarge", "rLatency", "rBottomBlend", "k_overdrive"
]

LOG_SCALE_PARAMS = [
    True,   # 0: kappa (1e5 to 1e8)
    False,  # 1: alpha
    False,  # 2: vTh
    False,  # 3: gScale
    False,  # 4: rC
    False,  # 5: vSmooth
    True,   # 6: nSmooth (10.0 to 1500.0)
    True,   # 7: kDischarge (0.1 to 800.0: apex rapid drop boost amplitude)
    False,  # 8: E_a
    True,   # 9: R_th (50.0 to 1e5)
    True,   # 10: tau_th (1e-9 to 3e-7)
    False,  # 11: gamma
    False,  # 12: beta_alpha
    False,  # 13: beta_s
    False,  # 14: rate_asymmetry (0.05 to 0.99)
    False,  # 15: fDischarge (dynamic knee transition: 0.10 to 0.70)
    False,  # 16: rLatency (shelf latency throttle factor: 0.0 to 0.99)
    False,  # 17: rBottomBlend (saturation recovery base landing boundary scalar: 0.10 to 1.90)
    False   # 18: k_overdrive (voltage memory overdrive latency gating scalar: 0.0 to 10.0)
]

DEFAULT_PARAMETER_BOUNDS = [
    (1e6, 3e7),          # 0: kappa
    (0.1, 10.0),         # 1: alpha
    (0.38, 0.85),        # 2: vTh (widened from 0.50 to 0.38 to accommodate Ru_azo baseline and relieve boundary pinning)
    (5.2, 6.9),          # 3: gScale (physically constrained to Ru_azo 7.13 mS transconductance)
    (0.001, 0.05),       # 4: rC (capped to prevent unphysical high-gScale compression)
    (0.001, 0.30),       # 5: vSmooth
    (10.0, 1500.0),      # 6: nSmooth (floor lowered from 50.0 to 10.0 to relieve shoulder pinning)
    (0.1, 800.0),        # 7: kDischarge (apex rapid drop boost amplitude, widened to 800.0)
    (0.15, 0.60),        # 8: E_a
    (50.0, 10000),       # 9: R_th (floor lowered from 100.0 to 50.0 to relieve lower boundary pressure)
    (1e-9, 3e-7),        # 10: tau_th (widened ceiling from 1e-7 to 3e-7 per physical thermal relaxation range)
    (-0.9, 0),           # 11: gamma
    (0.0, 5.0),          # 12: beta_alpha
    (-5.0, 10.0),        # 13: beta_s (widened from 5.0 to 10.0 for 1.22V high-voltage saturation shoulder rounding)
    (0.05, 0.99),        # 14: rate_asymmetry
    (0.10, 0.70),        # 15: fDischarge (Option A relaxation curvature: p_decay = 1/fDischarge, 0.10 to 0.70)
    (0.0, 0.99),         # 16: rLatency (shelf latency throttle factor: widened from 0.95 to 0.99)
    (0.10, 1.00),        # 17: rBottomBlend (saturation recovery base landing boundary scalar: 0.10 to 1.00)
    (0.0, 10.0)          # 18: k_overdrive (voltage memory overdrive latency gating scalar: 0.0 to 10.0)
]
# <<< END CONFIGURATION & HYPERPARAMETERS <<<

def to_physical_params(x):
    x_phys = np.copy(x)
    for idx, is_log in enumerate(LOG_SCALE_PARAMS):
        if is_log:
            x_phys[idx] = 10**x[idx]
    return x_phys

def to_optimizer_params(x_phys):
    x_opt = np.copy(x_phys)
    for idx, is_log in enumerate(LOG_SCALE_PARAMS):
        if is_log:
            x_opt[idx] = np.log10(x_phys[idx])
    return x_opt

def get_dict_params(x_opt, base_params):
    x_phys = to_physical_params(x_opt)
    d = base_params.copy()
    d["kappa"] = x_phys[0]
    d["alpha"] = x_phys[1]
    d["vTh"] = x_phys[2]
    d["vRefPos"] = 0.9
    d["vRefNeg"] = 0.75
    d["gScale"] = x_phys[3]
    d["rC"] = x_phys[4]
    d["vSmooth"] = x_phys[5]
    d["nSmooth"] = x_phys[6]
    d["kDischarge"] = x_phys[7]
    d["f22StartVal"] = base_params["f22StartVal"]
    d["f22PeakVal"] = base_params["f22PeakVal"]
    d["f22EndVal"] = base_params["f22EndVal"]
    d["E_a"] = x_phys[8]
    d["R_th"] = x_phys[9]
    d["tau_th"] = x_phys[10]
    d["gamma"] = x_phys[11]
    d["beta_alpha"] = x_phys[12]
    d["beta_s"] = x_phys[13]
    d["rate_asymmetry"] = x_phys[14]
    if len(x_phys) >= 19:
        d["fDischarge"] = x_phys[15]
        d["rLatency"] = x_phys[16]
        d["rBottomBlend"] = x_phys[17]
        d["k_overdrive"] = x_phys[18]
    elif len(x_phys) >= 18:
        d["fDischarge"] = x_phys[15]
        d["rLatency"] = x_phys[16]
        d["rBottomBlend"] = x_phys[17]
        d["k_overdrive"] = base_params.get("k_overdrive", 2.85)
    elif len(x_phys) >= 17:
        d["fDischarge"] = x_phys[15]
        d["rLatency"] = x_phys[16]
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
        d["k_overdrive"] = base_params.get("k_overdrive", 2.85)
    elif len(x_phys) >= 16:
        d["rLatency"] = x_phys[15]
        d["fDischarge"] = base_params.get("fDischarge", 0.35)
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
        d["k_overdrive"] = base_params.get("k_overdrive", 2.85)
    else:
        d["rLatency"] = 0.5
        d["fDischarge"] = base_params.get("fDischarge", 0.35)
        d["rBottomBlend"] = base_params.get("rBottomBlend", 1.0)
        d["k_overdrive"] = base_params.get("k_overdrive", 2.85)
    return d

BEST_CANDIDATE_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cluster_best_candidate_cache.json")

def format_params_summary(x_opt):
    """
    Extracts and formats key physical parameters into a compact single-line summary
    for consistent display across Host prompt, Host telemetry, and Node footers.
    """
    try:
        x_phys = to_physical_params(x_opt)
        kp = x_phys[0]
        al = x_phys[1]
        vTh = x_phys[2]
        kDisch = x_phys[7]
        E_a = x_phys[8]
        R_th = x_phys[9]
        gamma = x_phys[11]
        if len(x_phys) >= 20:
            uKnee = x_phys[16]
            rLat = x_phys[17]
            rBot = x_phys[18]
            kOd = x_phys[19]
            return f"kp={kp:.2e} al={al:.2f} vTh={vTh:.2f} kDisch={kDisch:.2f} uKn={uKnee:.2f} rLat={rLat:.2f} rBot={rBot:.2f} kOd={kOd:.2f} Ea={E_a:.2f} Rth={R_th:.1f} g={gamma:.2f}"
        elif len(x_phys) >= 19:
            uKnee = x_phys[16]
            rLat = x_phys[17]
            rBot = x_phys[18]
        elif len(x_phys) >= 18:
            uKnee = x_phys[16]
            rLat = x_phys[17]
            rBot = 1.0
        elif len(x_phys) >= 17:
            uKnee = 0.35
            rLat = x_phys[16]
            rBot = 1.0
        else:
            uKnee = 0.35
            rLat = 0.5
            rBot = 1.0
        return f"kp={kp:.2e} al={al:.2f} vTh={vTh:.2f} kDisch={kDisch:.2f} uKn={uKnee:.2f} rLat={rLat:.2f} rBot={rBot:.2f} Ea={E_a:.2f} Rth={R_th:.1f} g={gamma:.2f}"
    except Exception:
        return ""

def save_node_best_candidate(candidate):
    """
    Saves a candidate dictionary to local disk cache.
    Keys by '{device}_{selection}' as well as '_last'.
    Safely ignores worse candidates to prevent regression.
    """
    if not candidate or not isinstance(candidate, dict):
        return
    try:
        new_cost = float(candidate.get("cost", float('inf')))
        if not np.isfinite(new_cost):
            return
        cache = {}
        if os.path.exists(BEST_CANDIDATE_CACHE_FILE):
            try:
                with open(BEST_CANDIDATE_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        dev = str(candidate.get("device", "unknown"))
        sel = str(candidate.get("selection", "all"))
        key = f"{dev}_{sel}"

        # Guard against regression: if an existing entry has lower cost, do not overwrite it
        if key in cache and isinstance(cache[key], dict) and "cost" in cache[key]:
            try:
                old_cost = float(cache[key]["cost"])
                if new_cost >= old_cost:
                    if "_last" not in cache:
                        cache["_last"] = cache[key]
                        with open(BEST_CANDIDATE_CACHE_FILE, "w", encoding="utf-8") as f:
                            json.dump(cache, f, indent=2)
                    return
            except Exception:
                pass

        cache[key] = candidate
        # Only update _last if candidate is at least as good as current _last, or if _last missing
        update_last = True
        if "_last" in cache and isinstance(cache["_last"], dict) and "cost" in cache["_last"]:
            try:
                old_last_cost = float(cache["_last"]["cost"])
                if new_cost > old_last_cost:
                    update_last = False
            except Exception:
                pass
        if update_last:
            cache["_last"] = candidate

        with open(BEST_CANDIDATE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def load_node_best_candidate(device=None, selection=None):
    """
    Loads candidate dictionary from local disk cache.
    Matches '{device}_{selection}' first, falls back to '_last'.
    """
    if not os.path.exists(BEST_CANDIDATE_CACHE_FILE):
        return None
    try:
        with open(BEST_CANDIDATE_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if device and selection:
            key = f"{device}_{selection}"
            if key in cache and isinstance(cache[key], dict) and "cost" in cache[key]:
                return cache[key]
        if "_last" in cache and isinstance(cache["_last"], dict) and "cost" in cache["_last"]:
            return cache["_last"]
        for k, v in cache.items():
            if isinstance(v, dict) and "cost" in v:
                return v
    except Exception:
        pass
    return None

def load_all_node_candidates():
    """
    Loads all distinct valid candidate dictionaries from local disk cache.
    Safely handles legacy formats, corrupted JSON, or missing files.
    """
    if not os.path.exists(BEST_CANDIDATE_CACHE_FILE):
        return []
    candidates = []
    try:
        with open(BEST_CANDIDATE_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if not isinstance(cache, dict):
            return []
        seen_keys = set()
        for k, v in cache.items():
            if isinstance(v, dict) and "cost" in v and "params" in v:
                try:
                    c = float(v["cost"])
                    if np.isfinite(c) and c > 0:
                        dev = str(v.get("device", "unknown"))
                        sel = str(v.get("selection", "all"))
                        key_tuple = (dev, sel, round(c, 8))
                        if key_tuple not in seen_keys:
                            seen_keys.add(key_tuple)
                            candidates.append(v)
                except Exception:
                    continue
    except Exception:
        pass
    return candidates

def format_node_candidate_footer(candidate):
    """
    Returns lines for the persistent candidate display footer.
    """
    if not candidate or not isinstance(candidate, dict) or "cost" not in candidate:
        return [
            "=" * 134,
            " [+] Node hasn't cached a best candidate yet",
            "=" * 134
        ]
    cost = candidate.get("cost", 0.0)
    dev = candidate.get("device", "unknown")
    sel = candidate.get("selection", "all")
    gen = candidate.get("gen", 0)
    p_str = candidate.get("params_phys_str", "")
    ts = candidate.get("timestamp", "")
    ts_str = f" | {ts}" if ts else ""
    return [
        "=" * 134,
        f" BEST CACHED CANDIDATE: Cost: {cost:.6f} | [{dev} / {sel}] | Gen {gen}{ts_str}",
        f" Parameters: {p_str}" if p_str else " Parameters: (none)",
        "=" * 134
    ]

# =============================================================================
# Network Protocol & Framing Configuration
# =============================================================================

def get_local_ip():
    """Returns the primary LAN IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def configure_aggressive_keepalive(sock, idle_sec=10, interval_sec=2, count=5):
    """
    Configures low-level TCP keepalive probes and user timeouts so dead links and NAT dropouts
    are detected in ~8-15 seconds instead of the OS default of 2 hours.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if sys.platform == "win32":
            sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, int(idle_sec * 1000), int(interval_sec * 1000)))
        elif sys.platform.startswith("linux"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, int(idle_sec))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, int(interval_sec))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, int(count))
            # Enable TCP_USER_TIMEOUT (8000 ms) on Linux kernel to abort stuck WAN connections promptly
            if hasattr(socket, 'TCP_USER_TIMEOUT'):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 8000)
                except Exception:
                    pass
        elif sys.platform == "darwin":
            TCP_KEEPALIVE = getattr(socket, 'TCP_KEEPALIVE', 0x10)
            sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, int(idle_sec))
    except Exception:
        pass

def send_framed_msg(sock, data):
    """
    Sends a length-prefixed, compressed pickled object over a TCP stream.
    Format: [4-byte Big-Endian Length] + [Zlib-compressed Pickle Data]
    """
    try:
        raw = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=1)
        header = struct.pack('>I', len(raw))
        sock.sendall(header + raw)
        return True
    except (socket.error, ConnectionResetError, BrokenPipeError):
        return False

def recv_framed_msg(sock):
    """
    Receives a length-prefixed, compressed pickled object from a TCP stream.
    Returns None if connection is broken or closed.
    """
    try:
        header = b''
        while len(header) < 4:
            chunk = sock.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        payload_len = struct.unpack('>I', header)[0]
        chunks = []
        bytes_received = 0
        while bytes_received < payload_len:
            chunk = sock.recv(min(65536, payload_len - bytes_received))
            if not chunk:
                return None
            chunks.append(chunk)
            bytes_received += len(chunk)
        raw = b''.join(chunks)
        return pickle.loads(zlib.decompress(raw))
    except (socket.error, ConnectionResetError, BrokenPipeError, zlib.error, pickle.UnpicklingError):
        return None

def discover_host_beacon(beacon_port=DEFAULT_BEACON_PORT, timeout=1.0):
    """
    Listens for UDP broadcast beacons from a Cluster Host node.
    Returns (host_ip, host_tcp_port) if detected within timeout, else None.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout)
        sock.bind(('', beacon_port))
        start_t = time.time()
        while time.time() - start_t < timeout:
            try:
                data, addr = sock.recvfrom(2048)
                if data.startswith(b"MOLMEM_HOST_BEACON:") or data.startswith(b"MOLMEM_MASTER_BEACON:"):
                    parts = data.decode('utf-8', errors='ignore').split(":")
                    if len(parts) >= 3:
                        host_ip = parts[1]
                        host_port = int(parts[2])
                        return host_ip, host_port
            except socket.timeout:
                break
    except Exception:
        pass
    finally:
        sock.close()
    return None

discover_master_beacon = discover_host_beacon

def parse_host_port(target, default_port):
    """Parses a host string that may optionally contain a port (e.g. 'cluster.domain.com:48899')."""
    if not target:
        return "", default_port
    target = str(target).strip()
    if ":" in target:
        parts = target.rsplit(":", 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return parts[0], default_port
    return target, default_port

def probe_tcp_host(host, port, timeout=2.5):
    """
    Probes whether an active Cluster Host is reachable at host:port (e.g. via Cloudflare Tunnel / WAN).
    Returns (True, parsed_host, parsed_port) if reachable, (False, host, port) otherwise.
    Uses a 2.5s timeout to absorb WAN round-trip latency and transient TCP SYN retransmissions.
    """
    if not host:
        return False, "", port
    host, port = parse_host_port(host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True, host, port
    except Exception:
        return False, host, port
    finally:
        try:
            sock.close()
        except Exception:
            pass

probe_tcp_master = probe_tcp_host

class HostBeaconBroadcaster(threading.Thread):
    """Periodically broadcasts UDP discovery beacons on the local network."""
    def __init__(self, master_ip, tcp_port, beacon_port=DEFAULT_BEACON_PORT):
        super().__init__(daemon=True)
        self.master_ip = master_ip
        self.tcp_port = tcp_port
        self.beacon_port = beacon_port
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            msg_host = f"MOLMEM_HOST_BEACON:{self.master_ip}:{self.tcp_port}".encode('utf-8')
            msg_legacy = f"MOLMEM_MASTER_BEACON:{self.master_ip}:{self.tcp_port}".encode('utf-8')
            while self.running:
                try:
                    sock.sendto(msg_host, ('<broadcast>', self.beacon_port))
                    sock.sendto(msg_host, ('255.255.255.255', self.beacon_port))
                    sock.sendto(msg_legacy, ('<broadcast>', self.beacon_port))
                    sock.sendto(msg_legacy, ('255.255.255.255', self.beacon_port))
                except Exception:
                    pass
                time.sleep(0.5)
        finally:
            sock.close()

    def stop(self):
        self.running = False

MasterBeaconBroadcaster = HostBeaconBroadcaster


# =============================================================================
# Custom Solver with Decaying Mutation (Integrated with Cluster Dispatcher)
# =============================================================================
class DecayingDESolver(DifferentialEvolutionSolver):
    def __init__(self, *args, **kwargs):
        self.cluster_dispatcher = kwargs.pop('cluster_dispatcher', None)
        self.custom_callback = kwargs.pop('callback', None)
        self.current_gen = 0
        self.center_w_base = 0.15
        self.de_bounds = kwargs.get('bounds', None)
        if self.de_bounds is None and len(args) > 1:
            self.de_bounds = args[1]
        super().__init__(*args, callback=None, **kwargs)
        self.callback = self.custom_callback
        self.workers = self.map_with_progress
        self._wrapped_primary_mapper = self.map_with_progress
        self._mapwrapper = MapWrapper(self.map_with_progress)

    def _mutate_many(self, candidates):
        """
        High-performance vectorized mutation strategy replacing SciPy's un-vectorized
        _select_samples loop (which shuffles the full 84,000-element population index
        75,600 times, causing a 38-second inter-generation freeze).
        Completes in ~40 ms across 75,600 candidates while guaranteeing strictly distinct
        sample indices and excluding the candidate itself.
        """
        rng = self.random_number_generator
        S = len(candidates)
        N = self.num_population_members
        cand_arr = np.asarray(candidates)

        # Draw random candidate indices in [0, N - 2] and shift >= candidate to avoid self-selection
        if hasattr(rng, 'integers'):
            samples = rng.integers(0, N - 1, size=(S, 5))
        else:
            samples = rng.randint(0, N - 1, size=(S, 5))
        samples += (samples >= cand_arr[:, None])

        # Enforce distinct indices across the 5 sample slots per candidate
        dup = (
            (samples[:, 0] == samples[:, 1]) | (samples[:, 0] == samples[:, 2]) |
            (samples[:, 0] == samples[:, 3]) | (samples[:, 0] == samples[:, 4]) |
            (samples[:, 1] == samples[:, 2]) | (samples[:, 1] == samples[:, 3]) |
            (samples[:, 1] == samples[:, 4]) | (samples[:, 2] == samples[:, 3]) |
            (samples[:, 2] == samples[:, 4]) | (samples[:, 3] == samples[:, 4])
        )
        while np.any(dup):
            idx = np.where(dup)[0]
            n_rerun = len(idx)
            if hasattr(rng, 'integers'):
                ns = rng.integers(0, N - 1, size=(n_rerun, 5))
            else:
                ns = rng.randint(0, N - 1, size=(n_rerun, 5))
            ns += (ns >= cand_arr[idx, None])
            samples[idx] = ns
            dup[idx] = (
                (ns[:, 0] == ns[:, 1]) | (ns[:, 0] == ns[:, 2]) |
                (ns[:, 0] == ns[:, 3]) | (ns[:, 0] == ns[:, 4]) |
                (ns[:, 1] == ns[:, 2]) | (ns[:, 1] == ns[:, 3]) |
                (ns[:, 1] == ns[:, 4]) | (ns[:, 2] == ns[:, 3]) |
                (ns[:, 2] == ns[:, 4]) | (ns[:, 3] == ns[:, 4])
            )

        trial = np.copy(self.population[candidates])
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidates, samples)
        else:
            bprime = self.mutation_func(samples)

        if hasattr(rng, 'integers'):
            fill_point = rng.integers(0, self.parameter_count, size=S)
        else:
            fill_point = rng.randint(0, self.parameter_count, size=S)

        crossovers = rng.uniform(size=(S, self.parameter_count)) < self.cross_over_probability
        if self.strategy in self._binomial:
            i = np.arange(S)
            crossovers[i, fill_point[i]] = True
            trial = np.where(crossovers, bprime, trial)
            return trial
        elif self.strategy in self._exponential:
            crossovers[..., 0] = True
            for j in range(S):
                i = 0
                init_fill = fill_point[j]
                while i < self.parameter_count and crossovers[j, i]:
                    trial[j, init_fill] = bprime[j, init_fill]
                    init_fill = (init_fill + 1) % self.parameter_count
                    i += 1
            return trial
        return trial

    def _adapt_curvature_steering(self, nit, improved=True):
        if not hasattr(self, 'x') or self.x is None:
            return
            
        if hasattr(self, 'cluster_dispatcher') and self.cluster_dispatcher is not None:
            self.cluster_dispatcher.update_intergen_status(f"Gen {self.current_gen}/{self.maxiter} | Adapting Curvature Steering...")
            
        cost_function(self.x)
        
        active_err_high = []
        active_err_low = []
        
        global GLOBAL_LAST_G_SAT, GLOBAL_LAST_G_NS, GLOBAL_LAST_G_122
        global GLOBAL_COND_SAT, GLOBAL_COND_NS, GLOBAL_COND_122
        global GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_122
        global GLOBAL_FIT_SAT, GLOBAL_FIT_NS, GLOBAL_FIT_122
        
        if GLOBAL_FIT_SAT and GLOBAL_LAST_G_SAT is not None and GLOBAL_NORM_NONLIN_SAT is not None:
            res_sat = np.abs(GLOBAL_LAST_G_SAT - GLOBAL_COND_SAT)
            high_mask = GLOBAL_NORM_NONLIN_SAT >= np.median(GLOBAL_NORM_NONLIN_SAT)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_SAT[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_SAT[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_sat[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_sat[low_mask])) / mean_low)
            
        if GLOBAL_FIT_NS and GLOBAL_LAST_G_NS is not None and GLOBAL_NORM_NONLIN_NS is not None:
            res_ns = np.abs(GLOBAL_LAST_G_NS - GLOBAL_COND_NS)
            high_mask = GLOBAL_NORM_NONLIN_NS >= np.median(GLOBAL_NORM_NONLIN_NS)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_NS[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_NS[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_ns[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_ns[low_mask])) / mean_low)
            
        if GLOBAL_FIT_122 and GLOBAL_LAST_G_122 is not None and GLOBAL_NORM_NONLIN_122 is not None:
            res_122 = np.abs(GLOBAL_LAST_G_122 - GLOBAL_COND_122)
            high_mask = GLOBAL_NORM_NONLIN_122 >= np.median(GLOBAL_NORM_NONLIN_122)
            low_mask = ~high_mask
            mean_high = max(float(np.mean(GLOBAL_COND_122[high_mask])), 1e-6)
            mean_low = max(float(np.mean(GLOBAL_COND_122[low_mask])), 1e-6)
            active_err_high.append(float(np.mean(res_122[high_mask])) / mean_high)
            active_err_low.append(float(np.mean(res_122[low_mask])) / mean_low)
            
        if active_err_high and active_err_low:
            if hasattr(self, 'cluster_dispatcher') and self.cluster_dispatcher is not None:
                self.cluster_dispatcher.update_intergen_status(f"Gen {nit}/{self.maxiter} | Adapting Curvature Steering...")
            err_h = float(np.mean(active_err_high))
            err_l = float(np.mean(active_err_low))
            old_center = self.center_w_base
            step = 0.02 if improved else 0.01
            if err_h > err_l:
                self.center_w_base = max(0.0, round(self.center_w_base - step, 3))
                steer_msg = f"Decreased w_base by {step:.2f} (pulling apex & reverse collapse)"
            else:
                self.center_w_base = min(1.0, round(self.center_w_base + step, 3))
                steer_msg = f"Increased w_base by {step:.2f} (pulling body & tails)"
                
            print_log(
                f"  [Adaptive Steering] Gen {nit}: High-Nonlin Err: {err_h*100:.2f}% vs Low-Nonlin Err: {err_l*100:.2f}%\n"
                f"                      Unified w_base: {old_center:.2f} -> {self.center_w_base:.2f} | {steer_msg}"
            )

    def map_with_progress(self, func, iterable):
        candidates = list(iterable)
        total = len(candidates)
        is_refinement = getattr(self, 'in_refinement', False)
        
        # Pass candidates with the unified dynamically steered w_base across all cluster workers
        w_current = round(self.center_w_base, 3)
        if not is_refinement:
            inputs_to_map = [(cand, w_current) for cand in candidates]
        else:
            inputs_to_map = candidates
            
        if self.cluster_dispatcher is not None:
            return self.cluster_dispatcher.map(
                inputs_to_map, 
                current_gen=self.current_gen, 
                maxiter=self.maxiter, 
                solver=self,
                is_refinement=is_refinement
            )
        else:
            # Fallback to direct local evaluation if no dispatcher
            return [cost_function(item) for item in inputs_to_map]

    def __next__(self):
        if not hasattr(self, 'momentum'):
            self.momentum = np.zeros(self.parameter_count)

        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))
            self._promote_lowest_energy()

        if hasattr(self, 'cluster_dispatcher') and self.cluster_dispatcher is not None:
            self.cluster_dispatcher.update_intergen_status(f"Gen {self.current_gen}/{self.maxiter} | Sorting Elites & Recombining...")
        n_elites = max(1, int(0.10 * self.num_population_members))
        num_explorers = max(1, int(0.10 * self.num_population_members))

        sorted_indices_prev = np.argsort(self.population_energies)
        centroid_prev = np.mean(self.population[sorted_indices_prev[:n_elites]], axis=0)

        sorted_indices = np.argsort(self.population_energies)
        self.population = self.population[sorted_indices]
        self.population_energies = self.population_energies[sorted_indices]
        self.feasible = self.feasible[sorted_indices]
        self.constraint_violation = self.constraint_violation[sorted_indices]

        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0], self.dither[1])

        if self._nfev >= self.maxfun:
            raise StopIteration

        active_indices = np.arange(n_elites, self.num_population_members)
        trial_pop_active = self._mutate_many(active_indices)

        n_random_traits = self.parameter_count // 2
        for i in range(len(trial_pop_active) - num_explorers, len(trial_pop_active)):
            explorer = np.copy(self.population[0])
            rand_traits = self.random_number_generator.choice(
                self.parameter_count, size=n_random_traits, replace=False
            )
            explorer[rand_traits] = self.random_number_generator.uniform(
                0.0, 1.0, size=n_random_traits
            )
            trial_pop_active[i] = explorer

        for i in range(len(trial_pop_active) - num_explorers):
            if self.random_number_generator.uniform(0, 1) < 0.5:
                scale = self.random_number_generator.uniform(0.01, 0.50)
                trial_pop_active[i] += scale * self.momentum

        if not hasattr(self, 'historical_archive'):
            self.historical_archive = []
            
        top_elite_batch = self.population[:min(50, n_elites)]
        for elite_cand in top_elite_batch:
            if len(self.historical_archive) == 0 or np.min(np.linalg.norm(self.historical_archive - elite_cand, axis=1)) > 0.005:
                self.historical_archive.append(np.copy(elite_cand))
                if len(self.historical_archive) > 300:
                    self.historical_archive.pop(0)
                    
        if hasattr(self, 'cluster_dispatcher') and self.cluster_dispatcher is not None:
            self.cluster_dispatcher.update_intergen_status(f"Gen {self.current_gen}/{self.maxiter} | Spatial De-Clustering Archive Check...")
        archive_mat = np.array(self.historical_archive)
        MIN_CLUSTER_DIST = 0.008
        n_declustered = 0
        n_to_check = len(trial_pop_active) - num_explorers
        if len(archive_mat) > 0 and n_to_check > 0:
            diffs = trial_pop_active[:n_to_check, None, :] - archive_mat[None, :, :]
            dist_sq = np.sum(diffs ** 2, axis=2)
            min_dist_idx = np.argmin(dist_sq, axis=1)
            min_dists = np.sqrt(dist_sq[np.arange(n_to_check), min_dist_idx])
            
            crowded_mask = min_dists < MIN_CLUSTER_DIST
            crowded_indices = np.where(crowded_mask)[0]
            n_declustered = len(crowded_indices)
            
            for idx in crowded_indices:
                rand_vec = self.random_number_generator.standard_normal(self.parameter_count)
                norm_v = np.linalg.norm(rand_vec)
                rand_vec /= norm_v if norm_v > 1e-12 else 1.0
                push = MIN_CLUSTER_DIST + self.random_number_generator.uniform(0.005, 0.025)
                trial_pop_active[idx] = trial_pop_active[idx] + push * rand_vec

        self.last_n_declustered = n_declustered

        self._ensure_constraint(trial_pop_active)

        feasible_active, cv_active = self._calculate_population_feasibilities(trial_pop_active)
        trial_energies_active = np.full(len(active_indices), np.inf)

        trial_energies_active[feasible_active] = self._calculate_population_energies(
            trial_pop_active[feasible_active])

        loc = [self._accept_trial(*val) for val in
               zip(trial_energies_active, feasible_active, cv_active,
                   self.population_energies[n_elites:], self.feasible[n_elites:], self.constraint_violation[n_elites:])]
        loc = np.array(loc)

        self.population[n_elites:] = np.where(loc[:, np.newaxis], trial_pop_active, self.population[n_elites:])
        self.population_energies[n_elites:] = np.where(loc, trial_energies_active, self.population_energies[n_elites:])
        self.feasible[n_elites:] = np.where(loc, feasible_active, self.feasible[n_elites:])
        self.constraint_violation[n_elites:] = np.where(loc[:, np.newaxis], cv_active, self.constraint_violation[n_elites:])

        self._promote_lowest_energy()

        sorted_indices_curr = np.argsort(self.population_energies)
        centroid_curr = np.mean(self.population[sorted_indices_curr[:n_elites]], axis=0)
        step_diff = centroid_curr - centroid_prev
        self.momentum = 0.7 * self.momentum + 0.3 * step_diff

        return self.x, self.population_energies[0]

    def solve(self):
        if np.all(np.isinf(self.population_energies)):
            update_dashboard(phase=1, current=0, total=self.maxiter, cost=np.inf, best_xk=self.x, dither=[0.01, 1.0], w_base=self.center_w_base)
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(self.population[self.feasible]))
            self._promote_lowest_energy()
            
            # Fire callback for Generation 0 (initial population exploration completion)
            if self.callback is not None and hasattr(self, 'x') and self.x is not None:
                try:
                    self.callback(self.x, nit=0)
                except TypeError:
                    self.callback(self.x)
                except Exception as ce:
                    print_log(f"  [Callback Warning] Exception in Gen 0 callback: {ce}")

        nit = 0
        status_message = "Optimization terminated successfully."
        warning_flag = False

        best_cost = np.inf
        no_improvement_count = 0
        decay_nit = 1

        for nit in range(1, self.maxiter + 1):
            self.current_gen = nit
            if self.maxiter > 1:
                fraction = (decay_nit - 1) / (self.maxiter - 1)
            else:
                fraction = 0.0
            
            f_min = 0.01
            f_max = max(0.30, 1.0 - (1.0 - 0.05) * fraction)
            self.dither = [f_min, f_max]
            
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                break
                
            current_cost = self.population_energies[0]
            improved = False
            if current_cost < best_cost - 1e-5:
                best_cost = current_cost
                no_improvement_count = 0
                improved = True
                
                if REFINEMENT_SWEEP_STEPS > 0:
                    print_log(f"  [DE] Generation {nit}/{self.maxiter} | New Best Cost: {current_cost:.6f} | Running coordinate refinement sweep ({self.parameter_count * REFINEMENT_SWEEP_STEPS} points across cluster)...")
                    
                    x_best_opt = np.zeros(self.parameter_count)
                    for col_idx in range(self.parameter_count):
                        lo, hi = self.de_bounds[col_idx]
                        x_best_opt[col_idx] = lo + self.population[0, col_idx] * (hi - lo)
                        
                    total_refine_samples = self.parameter_count * REFINEMENT_SWEEP_STEPS
                    x_refine_samples = np.zeros((total_refine_samples, self.parameter_count))
                    for i in range(total_refine_samples):
                        x_refine_samples[i, :] = x_best_opt.copy()
                        
                    for col_idx in range(self.parameter_count):
                        lo, hi = self.de_bounds[col_idx]
                        grid = np.linspace(lo, hi, REFINEMENT_SWEEP_STEPS)
                        start_row = col_idx * REFINEMENT_SWEEP_STEPS
                        x_refine_samples[start_row : start_row + REFINEMENT_SWEEP_STEPS, col_idx] = grid
                        
                    self.in_refinement = True
                    refine_costs = self.workers(self.func, x_refine_samples)
                    self.in_refinement = False
                    refine_costs = np.array(refine_costs)
                    
                    best_refine_idx = np.argmin(refine_costs)
                    cost_refine = refine_costs[best_refine_idx]
                    
                    if cost_refine < current_cost - 1e-5:
                        x_refine_best = x_refine_samples[best_refine_idx]
                        u_refine_best = np.zeros(self.parameter_count)
                        for col_idx in range(self.parameter_count):
                            lo, hi = self.de_bounds[col_idx]
                            u_refine_best[col_idx] = (x_refine_best[col_idx] - lo) / (hi - lo)
                            
                        self.population[0] = u_refine_best
                        self.population_energies[0] = cost_refine
                        self._promote_lowest_energy()
                        print_log(f"  [Refinement] Successfully polished cost from {current_cost:.6f} to {cost_refine:.6f}!")
                        current_cost = cost_refine
                        best_cost = cost_refine
                    else:
                        print_log(f"  [Refinement] No improvement found (retained cost: {current_cost:.6f}).")
            else:
                no_improvement_count += 1

            lo_k, hi_k = self.de_bounds[7]
            k_pop_phys = 10.0 ** (lo_k + self.population[:, 7] * (hi_k - lo_k))
            avg_k = float(np.mean(k_pop_phys))
            med_k = float(np.median(k_pop_phys))
            best_k = float(to_physical_params(self.x)[7]) if hasattr(self, 'x') and self.x is not None else 0.0

            mom_norm = np.linalg.norm(self.momentum) if hasattr(self, 'momentum') else 0.0
            n_decluster = getattr(self, 'last_n_declustered', 0)
            print_log(f"[DE] Generation {nit}/{self.maxiter} | Best Cost: {current_cost:.6f} | w_base: {self.center_w_base:.2f} | kDischarge: avg={avg_k:.2f} (med={med_k:.2f}, best={best_k:.2f}) | de-clustered: {n_decluster} | dither: [{f_min:.4f}, {f_max:.4f} (0.30)] | momentum norm: {mom_norm:.4f}")
            update_dashboard(phase=1, current=nit, total=self.maxiter, cost=current_cost, best_xk=self.x, dither=self.dither, w_base=self.center_w_base, avg_k=avg_k)
            
            if nit >= 2 and hasattr(self, 'x') and self.x is not None:
                try:
                    self._adapt_curvature_steering(nit, improved=improved)
                except Exception:
                    pass
            
            if self.callback is not None and hasattr(self, 'x') and self.x is not None:
                try:
                    self.callback(self.x, nit=nit)
                except TypeError:
                    self.callback(self.x)
                except Exception as ce:
                    print_log(f"  [Callback Warning] Exception in Gen {nit} callback: {ce}")
                    import traceback
                    traceback.print_exc()
                
            # Proactive inter-generation transient history sweep: cleans up any orphaned binary files
            # from crashed or dropped worker processes without waiting for total run completion
            try:
                from molmem_lib.sys_utils import cleanup_temp_history_dir
                cleanup_temp_history_dir(force_all=True)
            except Exception:
                pass

            if improved:
                decay_nit = min(self.maxiter, decay_nit + 1)
            else:
                if f_max > 0.30:
                    jump_steps = no_improvement_count
                    decay_nit = min(self.maxiter, decay_nit + jump_steps)
                    print_log(f"  [DE] Stagnation detected ({no_improvement_count} gens). Jumping decay ahead by {jump_steps} iterations to reduce dither.")
                else:
                    if no_improvement_count >= 3:
                        print_log(f"  [DE] Early stopping triggered: no cost improvement in {no_improvement_count} generations at the dither floor (0.30).")
                        break
                    decay_nit = min(self.maxiter, decay_nit + 1)

            if self.converged():
                break
        else:
            warning_flag = True

        return self._result(nit=nit, message=status_message, warning_flag=warning_flag)


# =============================================================================
# Local Minimization Tracker
# =============================================================================
class LocalMinimizationTracker:
    def __init__(self, start_cost, best_overall_cost_ref, on_new_best_callback, start_x, basin_idx, total_basins):
        self.best_cost = start_cost
        self.best_x = np.copy(start_x)
        self.no_improvement_count = 0
        self.best_overall_cost_ref = best_overall_cost_ref
        self.on_new_best_callback = on_new_best_callback
        self.basin_idx = basin_idx
        self.total_basins = total_basins

    def __call__(self, xk):
        cost = cost_function(xk)
        update_dashboard(phase=2, current=self.basin_idx, total=self.total_basins, cost=self.best_overall_cost_ref[0], best_xk=xk)
        
        improvement = self.best_cost - cost
        if cost < self.best_cost - 1e-5:
            is_substantial = True
            if cost > self.best_overall_cost_ref[0]:
                gap = cost - self.best_overall_cost_ref[0]
                min_required = max(1e-3, 0.05 * gap)
                if improvement < min_required:
                    is_substantial = False
                    
            self.best_cost = cost
            self.best_x = np.copy(xk)
            
            if is_substantial:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                print_log(f"  [L-BFGS-B] Basin {self.basin_idx}/{self.total_basins} | Cost: {cost:.6f} (Stagnant: improvement {improvement:.6f} < required {min_required:.6f}) | Count: {self.no_improvement_count}/3")
            
            if cost < self.best_overall_cost_ref[0]:
                self.best_overall_cost_ref[0] = cost
                self.on_new_best_callback(xk)
        else:
            self.no_improvement_count += 1
            print_log(f"  [L-BFGS-B] Basin {self.basin_idx}/{self.total_basins} | Iteration cost: {cost:.6f} (No improvement count: {self.no_improvement_count}/3)")
            
        if self.no_improvement_count >= 3:
            print_log(f"  [L-BFGS-B] Early stopping triggered: stagnant convergence in basin.")
            raise StopIteration("Stagnant convergence in basin")


# =============================================================================
# Global variables for cost function evaluation
# =============================================================================
GLOBAL_COND_NS = None
GLOBAL_PULSES_NS = None
GLOBAL_COND_SAT = None
GLOBAL_PULSES_SAT = None
GLOBAL_COND_122 = None
GLOBAL_PULSES_122 = None
GLOBAL_BASE_PARAMS = None
GLOBAL_T_PERIOD = None
GLOBAL_T_WIDTH = None
GLOBAL_VPOT = None
GLOBAL_VDEP = None

GLOBAL_SIM_NS = None
GLOBAL_SIM_SAT = None
GLOBAL_SIM_122 = None
GLOBAL_VPOT_122 = None
GLOBAL_VDEP_122 = None

GLOBAL_DIFF_NS = None
GLOBAL_DIFF_SAT = None
GLOBAL_DIFF_122 = None

GLOBAL_LOG_NS = None
GLOBAL_LOG_SAT = None
GLOBAL_LOG_122 = None

GLOBAL_VAR_NS = 1.0
GLOBAL_VAR_DIFF_NS = 1.0
GLOBAL_VAR_LOG_NS = 1.0

GLOBAL_VAR_SAT = 1.0
GLOBAL_VAR_DIFF_SAT = 1.0
GLOBAL_VAR_LOG_SAT = 1.0

GLOBAL_VAR_122 = 1.0
GLOBAL_VAR_DIFF_122 = 1.0
GLOBAL_VAR_LOG_122 = 1.0

GLOBAL_SIM_TOL_VAL = None
GLOBAL_BEST_COST_VAL = None
GLOBAL_LAST_X = None
GLOBAL_LAST_COST = None

GLOBAL_FIT_NS = True
GLOBAL_FIT_SAT = True
GLOBAL_FIT_122 = True

GLOBAL_WEIGHTS_NS = None
GLOBAL_WEIGHTS_SAT = None
GLOBAL_WEIGHTS_122 = None
GLOBAL_SUM_WEIGHTS_NS = 1.0
GLOBAL_SUM_WEIGHTS_SAT = 1.0
GLOBAL_SUM_WEIGHTS_122 = 1.0
GLOBAL_NORM_NONLIN_NS = None
GLOBAL_NORM_NONLIN_SAT = None
GLOBAL_NORM_NONLIN_122 = None
GLOBAL_LAST_G_NS = None
GLOBAL_LAST_G_SAT = None
GLOBAL_LAST_G_122 = None

def compute_normalized_nonlinearity(x_vals, y_vals, raw_x=None, raw_y=None, window_fraction=0.04):
    """
    Computes normalized second-order curvature / chord deviation (0.0 to 1.0).
    If raw sparse empirical coordinates are provided, curvature is computed on the raw
    points and interpolated onto the dense curve to ensure the full S-curve knees, shoulders,
    and apex receive strong gradient emphasis without being suppressed by dense grid spacing.
    If only dense points are provided, evaluates chord deviation across a physical window
    (default 4% of curve width).
    """
    if raw_x is not None and raw_y is not None and len(raw_x) < len(x_vals) and len(raw_x) > 2:
        s_idx = np.argsort(raw_x)
        rx = np.asarray(raw_x, dtype=np.float64)[s_idx]
        ry = np.asarray(raw_y, dtype=np.float64)[s_idx]
        raw_nonlin = compute_normalized_nonlinearity(rx, ry)
        return np.interp(x_vals, rx, raw_nonlin)

    n = len(y_vals)
    if n <= 2:
        return np.zeros(n, dtype=np.float64)
    
    step = max(1, int(n * window_fraction / 2.0))
    d_nonlin = np.zeros(n, dtype=np.float64)
    for i in range(n):
        i0 = max(0, i - step)
        i2 = min(n - 1, i + step)
        if i2 - i0 < 2:
            continue
        x0, x1, x2 = float(x_vals[i0]), float(x_vals[i]), float(x_vals[i2])
        y0, y1, y2 = float(y_vals[i0]), float(y_vals[i]), float(y_vals[i2])
        dx = x2 - x0
        if abs(dx) > 1e-12:
            y_chord = y0 + (y2 - y0) * ((x1 - x0) / dx)
            d_nonlin[i] = abs(y1 - y_chord)
        else:
            d_nonlin[i] = 0.0
            
    max_d = np.max(d_nonlin)
    if max_d > 1e-12:
        return d_nonlin / max_d
    return np.zeros(n, dtype=np.float64)

def compute_nonlinearity_weights(x_vals, y_vals, w_base=0.50, raw_x=None, raw_y=None):
    norm_d = compute_normalized_nonlinearity(x_vals, y_vals, raw_x=raw_x, raw_y=raw_y)
    return w_base + (1.0 - w_base) * norm_d

def create_pchip_dense_dataset(pulses, cond, num_points=1000):
    """
    Interpolate empirical data points using monotonic PCHIP (matching compare_fit.py)
    and resample into exactly num_points uniformly spaced points.
    """
    p_vals = np.asarray(pulses, dtype=np.float64)
    c_vals = np.asarray(cond, dtype=np.float64)
    
    sort_idx = np.argsort(p_vals)
    p_sorted = p_vals[sort_idx]
    c_sorted = c_vals[sort_idx]
    
    u_idx = [0]
    for i in range(1, len(p_sorted)):
        if p_sorted[i] > p_sorted[u_idx[-1]] + 1e-5:
            u_idx.append(i)
            
    p_unique = p_sorted[u_idx]
    c_unique = c_sorted[u_idx]
    
    pchip = PchipInterpolator(p_unique, c_unique)
    p_dense = np.linspace(p_unique[0], p_unique[-1], num_points)
    c_dense = pchip(p_dense)
    return p_dense, c_dense

def init_worker(cond_ns, pulses_ns, cond_sat, pulses_sat, cond_122, pulses_122, 
                base_params, t_period, t_width, vpot, vdep, tol_val=None, best_cost_val=None,
                fit_ns=True, fit_sat=True, fit_122=True):
    try:
        import signal, os, multiprocessing
        if multiprocessing.current_process().name != 'MainProcess':
            signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, lambda sig, frame: os._exit(0))
            os.environ["MOLMEM_ACTIVE_WORKER_TASK"] = "1"

            if 'concurrent.futures.thread' in sys.modules:
                try:
                    sys.modules['concurrent.futures.thread']._threads_queues.clear()
                except Exception:
                    pass

        try:
            import numba
            numba.set_num_threads(1)
        except Exception:
            pass
        torch_mod = sys.modules.get('torch')
        if torch_mod is not None:
            try:
                torch_mod.set_num_threads(1)
            except Exception:
                pass

        # Capture compact raw empirical points before expanding into dense 1,000-point monotonic PCHIP curve
        raw_p_ns, raw_c_ns = (pulses_ns, cond_ns) if (cond_ns is not None and len(cond_ns) != 1000) else (None, None)
        raw_p_sat, raw_c_sat = (pulses_sat, cond_sat) if (cond_sat is not None and len(cond_sat) != 1000) else (None, None)
        raw_p_122, raw_c_122 = (pulses_122, cond_122) if (cond_122 is not None and len(cond_122) != 1000) else (None, None)

        # Expand compact raw empirical points into dense 1,000-point monotonic PCHIP curve in worker memory
        if fit_ns and cond_ns is not None and len(cond_ns) != 1000:
            pulses_ns, cond_ns = create_pchip_dense_dataset(pulses_ns, cond_ns, num_points=1000)
        if fit_sat and cond_sat is not None and len(cond_sat) != 1000:
            pulses_sat, cond_sat = create_pchip_dense_dataset(pulses_sat, cond_sat, num_points=1000)
        if fit_122 and cond_122 is not None and len(cond_122) != 1000:
            pulses_122, cond_122 = create_pchip_dense_dataset(pulses_122, cond_122, num_points=1000)

        global GLOBAL_COND_NS, GLOBAL_PULSES_NS, GLOBAL_COND_SAT, GLOBAL_PULSES_SAT
        global GLOBAL_COND_122, GLOBAL_PULSES_122, GLOBAL_BASE_PARAMS
        global GLOBAL_T_PERIOD, GLOBAL_T_WIDTH, GLOBAL_VPOT, GLOBAL_VDEP
        global GLOBAL_SIM_NS, GLOBAL_SIM_SAT, GLOBAL_SIM_122
        global GLOBAL_VPOT_122, GLOBAL_VDEP_122
        global GLOBAL_DIFF_NS, GLOBAL_DIFF_SAT, GLOBAL_DIFF_122
        global GLOBAL_LOG_NS, GLOBAL_LOG_SAT, GLOBAL_LOG_122
        global GLOBAL_SIM_TOL_VAL, GLOBAL_BEST_COST_VAL
        global GLOBAL_VAR_NS, GLOBAL_VAR_DIFF_NS, GLOBAL_VAR_LOG_NS
        global GLOBAL_VAR_SAT, GLOBAL_VAR_DIFF_SAT, GLOBAL_VAR_LOG_SAT
        global GLOBAL_VAR_122, GLOBAL_VAR_DIFF_122, GLOBAL_VAR_LOG_122
        global GLOBAL_FIT_NS, GLOBAL_FIT_SAT, GLOBAL_FIT_122
        global GLOBAL_WEIGHTS_NS, GLOBAL_WEIGHTS_SAT, GLOBAL_WEIGHTS_122
        global GLOBAL_SUM_WEIGHTS_NS, GLOBAL_SUM_WEIGHTS_SAT, GLOBAL_SUM_WEIGHTS_122
        global GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_122
        
        GLOBAL_COND_NS = cond_ns
        GLOBAL_PULSES_NS = pulses_ns
        GLOBAL_COND_SAT = cond_sat
        GLOBAL_PULSES_SAT = pulses_sat
        GLOBAL_COND_122 = cond_122
        GLOBAL_PULSES_122 = pulses_122
        GLOBAL_BASE_PARAMS = base_params
        GLOBAL_T_PERIOD = t_period
        GLOBAL_T_WIDTH = t_width
        GLOBAL_VPOT = vpot
        GLOBAL_VDEP = vdep
        GLOBAL_SIM_TOL_VAL = tol_val
        GLOBAL_BEST_COST_VAL = best_cost_val
        GLOBAL_FIT_NS = fit_ns
        GLOBAL_FIT_SAT = fit_sat
        GLOBAL_FIT_122 = fit_122

        if fit_ns:
            GLOBAL_DIFF_NS = np.diff(cond_ns)
            GLOBAL_LOG_NS = np.log10(np.clip(cond_ns, 5e-5, None))
            GLOBAL_VAR_NS = max(np.var(cond_ns), 1e-6)
            GLOBAL_VAR_DIFF_NS = max(np.var(GLOBAL_DIFF_NS), 1e-9)
            GLOBAL_VAR_LOG_NS = max(np.var(GLOBAL_LOG_NS), 1e-6)
            GLOBAL_NORM_NONLIN_NS = compute_normalized_nonlinearity(pulses_ns, cond_ns, raw_x=raw_p_ns, raw_y=raw_c_ns)
            GLOBAL_WEIGHTS_NS = compute_nonlinearity_weights(pulses_ns, cond_ns, w_base=0.50, raw_x=raw_p_ns, raw_y=raw_c_ns)
            GLOBAL_SUM_WEIGHTS_NS = max(float(np.sum(GLOBAL_WEIGHTS_NS)), 1e-12)
        else:
            GLOBAL_DIFF_NS = None
            GLOBAL_LOG_NS = None
            GLOBAL_VAR_NS = 1.0
            GLOBAL_VAR_DIFF_NS = 1.0
            GLOBAL_VAR_LOG_NS = 1.0
            GLOBAL_NORM_NONLIN_NS = None
            GLOBAL_WEIGHTS_NS = None
            GLOBAL_SUM_WEIGHTS_NS = 1.0

        if fit_sat:
            GLOBAL_DIFF_SAT = np.diff(cond_sat)
            GLOBAL_LOG_SAT = np.log10(np.clip(cond_sat, 5e-5, None))
            GLOBAL_VAR_SAT = max(np.var(cond_sat), 1e-6)
            GLOBAL_VAR_DIFF_SAT = max(np.var(GLOBAL_DIFF_SAT), 1e-9)
            GLOBAL_VAR_LOG_SAT = max(np.var(GLOBAL_LOG_SAT), 1e-6)
            GLOBAL_NORM_NONLIN_SAT = compute_normalized_nonlinearity(pulses_sat, cond_sat, raw_x=raw_p_sat, raw_y=raw_c_sat)
            GLOBAL_WEIGHTS_SAT = compute_nonlinearity_weights(pulses_sat, cond_sat, w_base=0.50, raw_x=raw_p_sat, raw_y=raw_c_sat)
            GLOBAL_SUM_WEIGHTS_SAT = max(float(np.sum(GLOBAL_WEIGHTS_SAT)), 1e-12)
        else:
            GLOBAL_DIFF_SAT = None
            GLOBAL_LOG_SAT = None
            GLOBAL_VAR_SAT = 1.0
            GLOBAL_VAR_DIFF_SAT = 1.0
            GLOBAL_VAR_LOG_SAT = 1.0
            GLOBAL_NORM_NONLIN_SAT = None
            GLOBAL_WEIGHTS_SAT = None
            GLOBAL_SUM_WEIGHTS_SAT = 1.0

        if fit_122:
            GLOBAL_DIFF_122 = np.diff(cond_122)
            GLOBAL_LOG_122 = np.log10(np.clip(cond_122, 5e-5, None))
            GLOBAL_VAR_122 = max(np.var(cond_122), 1e-6)
            GLOBAL_VAR_DIFF_122 = max(np.var(GLOBAL_DIFF_122), 1e-9)
            GLOBAL_VAR_LOG_122 = max(np.var(GLOBAL_LOG_122), 1e-6)
            GLOBAL_NORM_NONLIN_122 = compute_normalized_nonlinearity(pulses_122, cond_122, raw_x=raw_p_122, raw_y=raw_c_122)
            GLOBAL_WEIGHTS_122 = compute_nonlinearity_weights(pulses_122, cond_122, w_base=0.50, raw_x=raw_p_122, raw_y=raw_c_122)
            GLOBAL_SUM_WEIGHTS_122 = max(float(np.sum(GLOBAL_WEIGHTS_122)), 1e-12)
        else:
            GLOBAL_DIFF_122 = None
            GLOBAL_LOG_122 = None
            GLOBAL_VAR_122 = 1.0
            GLOBAL_VAR_DIFF_122 = 1.0
            GLOBAL_VAR_LOG_122 = 1.0
            GLOBAL_NORM_NONLIN_122 = None
            GLOBAL_WEIGHTS_122 = None
            GLOBAL_SUM_WEIGHTS_122 = 1.0

        # 1. Non-saturating Sweep Simulator
        if fit_ns:
            GLOBAL_SIM_NS = MolmemSimulator(device='cpu', backend='numba')
            GLOBAL_SIM_NS.detailedPrint = False
            GLOBAL_SIM_NS.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
            
            t_switch_ns = (np.max(pulses_ns) / 2.0) * t_period
            def v_pulse_train_ns(t, sources):
                if isinstance(t, np.ndarray):
                    out = np.zeros_like(t)
                    mask = t < t_switch_ns
                    if np.any(mask):
                        out[mask] = sources[0].getVoltage(t[mask])
                    if np.any(~mask):
                        out[~mask] = sources[1].getVoltage(t[~mask])
                    return out
                return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
            edge_points_ns = [t_switch_ns]
            GLOBAL_SIM_NS.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns)
            GLOBAL_SIM_NS.addDcSource(col=1, voltage=0.0)
        else:
            GLOBAL_SIM_NS = None

        # 2. Saturating Sweep Simulator
        if fit_sat:
            GLOBAL_SIM_SAT = MolmemSimulator(device='cpu', backend='numba')
            GLOBAL_SIM_SAT.detailedPrint = False
            GLOBAL_SIM_SAT.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
            
            t_switch_sat = (np.max(pulses_sat) / 2.0) * t_period
            def v_pulse_train_sat(t, sources):
                if isinstance(t, np.ndarray):
                    out = np.zeros_like(t)
                    mask = t < t_switch_sat
                    if np.any(mask):
                        out[mask] = sources[0].getVoltage(t[mask])
                    if np.any(~mask):
                        out[~mask] = sources[1].getVoltage(t[~mask])
                    return out
                return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
            edge_points_sat = [t_switch_sat]
            GLOBAL_SIM_SAT.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat)
            GLOBAL_SIM_SAT.addDcSource(col=1, voltage=0.0)
        else:
            GLOBAL_SIM_SAT = None

        # 3. 1.22V Saturating Sweep Simulator
        if fit_122:
            GLOBAL_SIM_122 = MolmemSimulator(device='cpu', backend='numba')
            GLOBAL_SIM_122.detailedPrint = False
            GLOBAL_SIM_122.addCrossbarMatrix(device_type=base_params, multiThread=False, rows=1, cols=1)
            
            t_switch_122 = (np.max(pulses_122) / 2.0) * t_period
            GLOBAL_VPOT_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
            GLOBAL_VDEP_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
            def v_pulse_train_122(t, sources):
                if isinstance(t, np.ndarray):
                    out = np.zeros_like(t)
                    mask = t < t_switch_122
                    if np.any(mask):
                        out[mask] = sources[0].getVoltage(t[mask])
                    if np.any(~mask):
                        out[~mask] = sources[1].getVoltage(t[~mask])
                    return out
                return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
            edge_points_122 = [t_switch_122]
            GLOBAL_SIM_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[GLOBAL_VPOT_122, GLOBAL_VDEP_122], edge_points=edge_points_122)
            GLOBAL_SIM_122.addDcSource(col=1, voltage=0.0)
        else:
            GLOBAL_SIM_122 = None
    except Exception as e:
        import traceback
        sys.stderr.write(f"\n[FATAL ERROR IN WORKER PID {os.getpid()} init_worker]: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

init_worker.__module__ = "__main__"

def update_simulator_candidate(sim, candidate):
    dev = sim.devices[0]['dev']
    dev.vSmooth = candidate["vSmooth"]
    dev.nSmooth = candidate["nSmooth"]
    dev.kappa = candidate["kappa"]
    dev.alpha = candidate["alpha"]
    dev.gScale = candidate["gScale"]
    dev.vTh = candidate["vTh"]
    dev.vRefPos = candidate["vRefPos"]
    dev.vRefNeg = candidate["vRefNeg"]
    dev.kDischarge = candidate["kDischarge"]
    dev.rC = candidate["rC"]
    dev.E_a = candidate.get("E_a", 0.0)
    dev.R_th = candidate.get("R_th", 0.0)
    dev.tau_th = candidate.get("tau_th", 1e-9)
    dev.gamma = candidate.get("gamma", 0.0)
    dev.beta_alpha = candidate.get("beta_alpha", 0.0)
    dev.beta_s = candidate.get("beta_s", 0.0)
    dev.rateAsymmetry = candidate.get("rate_asymmetry", 1.0)
    dev.rTopBlend = candidate.get("rTopBlend", 0.05)
    dev.rLatency = candidate.get("rLatency", 0.5)
    dev.fDischarge = candidate.get("fDischarge", 0.25)
    dev.rBottomBlend = candidate.get("rBottomBlend", 0.05)
    dev.k_overdrive = candidate.get("k_overdrive", 2.85)
    dev.f22StartVal = candidate["f22StartVal"]
    dev.f22PeakVal = candidate["f22PeakVal"]
    dev.f22EndVal = candidate["f22EndVal"]
    dev.f22PotScaleInv = 1.0 / (candidate["f22PeakVal"] - candidate["f22StartVal"])
    dev.f22DepScaleInv = 1.0 / (candidate["f22PeakVal"] - candidate["f22EndVal"])
    
    vref_max = max(abs(dev.vRefPos), abs(dev.vRefNeg))
    denom = max(1e-6, vref_max - dev.vTh + 1e-6)
    ratio = 1e-11 / (1e-8 * max(1e-6, dev.kappa))
    term1 = ratio ** (1.0 / max(0.1, dev.alpha))
    arg = (denom / max(1e-6, dev.vSmooth)) * term1
    if arg > 1e-20:
        vBypass_val = dev.vTh + dev.vSmooth * np.log(arg)
        dev.vBypass = max(0.0, min(dev.vTh - 3.0 * dev.vSmooth, vBypass_val))
    else:
        dev.vBypass = 0.0
        
    if hasattr(sim, '_soa_vSmooth_arr'):
        sim._soa_vSmooth_arr[0] = dev.vSmooth
        sim._soa_nSmooth_arr[0] = dev.nSmooth
        sim._soa_kappa_arr[0] = dev.kappa
        sim._soa_alpha_arr[0] = dev.alpha
        sim._soa_vTh_arr[0] = dev.vTh
        sim._soa_vBypass_arr[0] = dev.vBypass
        sim._soa_vRefPos_arr[0] = dev.vRefPos
        sim._soa_vRefNeg_arr[0] = dev.vRefNeg
        sim._soa_nMax_arr[0] = dev.nMax
        sim._soa_kDischarge_arr[0] = dev.kDischarge
        sim._soa_gScale_arr[0] = dev.gScale
        sim._soa_rC_arr[0] = dev.rC
        sim._soa_f22StartVal_arr[0] = dev.f22StartVal
        sim._soa_f22PeakVal_arr[0] = dev.f22PeakVal
        sim._soa_f22EndVal_arr[0] = dev.f22EndVal
        sim._soa_f22PotScaleInv_arr[0] = dev.f22PotScaleInv
        sim._soa_f22DepScaleInv_arr[0] = dev.f22DepScaleInv
        sim._soa_E_a_arr[0] = dev.E_a
        sim._soa_R_th_arr[0] = dev.R_th
        sim._soa_tau_th_arr[0] = dev.tau_th
        sim._soa_gamma_arr[0] = dev.gamma
        sim._soa_beta_alpha_arr[0] = dev.beta_alpha
        sim._soa_beta_s_arr[0] = dev.beta_s
        sim._soa_rateAsymmetry_arr[0] = dev.rateAsymmetry
        if hasattr(sim, '_soa_rTopBlend_arr'):
            sim._soa_rTopBlend_arr[0] = dev.rTopBlend
        if hasattr(sim, '_soa_fDischarge_arr'):
            sim._soa_fDischarge_arr[0] = dev.fDischarge
        if hasattr(sim, '_soa_rBottomBlend_arr'):
            sim._soa_rBottomBlend_arr[0] = dev.rBottomBlend
        if hasattr(sim, '_soa_rLatency_arr'):
            sim._soa_rLatency_arr[0] = dev.rLatency
        if hasattr(sim, '_soa_k_overdrive_arr'):
            sim._soa_k_overdrive_arr[0] = dev.k_overdrive
    
    if hasattr(sim, '_cached_phys_tensor'):
        delattr(sim, '_cached_phys_tensor')


# Parameter Space Scaling & Optimization Constants are defined in the top configuration block.

def save_preset_txt(x, base_params, txt_path, func_name, device_name, data_dir):
    final_params = get_dict_params(x, base_params)
    n_init_best = 0.0
    preset_str = f"""def {func_name}(data_dir=None):
    \"\"\"
    Returns optimized parameters and PWL paths for the {device_name} device (Clustered Multi-Sweep Fit).
    \"\"\"
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        
    return {{
        "vSmooth": {final_params['vSmooth']:.6f},
        "nSmooth": {final_params['nSmooth']:.2f},
        "kappa": {final_params['kappa']:.6e},
        "alpha": {final_params['alpha']:.6f},
        "gScale": {final_params['gScale']:.6f},
        "vTh": {final_params['vTh']:.6f},
        "vRefPos": {final_params['vRefPos']:.6f},
        "vRefNeg": {final_params['vRefNeg']:.6f},
        "nMax": {final_params['nMax']},
        "nInit": {n_init_best:.6f},
        "kDischarge": {final_params['kDischarge']:.2f},
        "rC": {final_params['rC']:.6f},
        "E_a": {final_params['E_a']:.6f},
        "R_th": {final_params['R_th']:.2f},
        "tau_th": {final_params['tau_th']:.3e},
        "gamma": {final_params['gamma']:.6f},
        "beta_alpha": {final_params['beta_alpha']:.6f},
        "beta_s": {final_params['beta_s']:.6f},
        "rate_asymmetry": {final_params['rate_asymmetry']:.6f},
        "rTopBlend": {final_params.get('rTopBlend', 0.05):.6f},
        "fDischarge": {final_params.get('fDischarge', 0.25):.6f},
        "rBottomBlend": {final_params.get('rBottomBlend', 0.05):.6f},
        "rLatency": {final_params.get('rLatency', 0.5):.6f},
        "k_overdrive": {final_params.get('k_overdrive', 2.85):.6f},
        "safe_max_vwrite": 0.00,
        "f22StartVal": {final_params['f22StartVal']:.8f},
        "f22PeakVal": {final_params['f22PeakVal']:.8f},
        "f22EndVal": {final_params['f22EndVal']:.8f},
        # Lookup table files
        "iScale_file": os.path.join(data_dir, "I_scale.pwl"),
        "vScale_file": os.path.join(data_dir, "V_scale.pwl"),
        "f22PotMap_file": os.path.join(data_dir, "f22_pot_map.pwl"),
        "f22DepMap_file": os.path.join(data_dir, "f22_dep_map.pwl")
    }}
"""
    with open(txt_path, "w") as f:
        f.write(preset_str)


# =============================================================================
# =============================================================================
# Cost Function Formulation (Point-to-point Normalized MSE)
# =============================================================================
_PRINTED_SIM_EXCEPTION = False

def cost_function(cand_input):
    cohort_w_base = None
    if isinstance(cand_input, tuple) and len(cand_input) == 2 and isinstance(cand_input[1], (int, float)):
        x, cohort_w_base = cand_input
    else:
        x = cand_input

    global GLOBAL_COND_NS, GLOBAL_PULSES_NS, GLOBAL_COND_SAT, GLOBAL_PULSES_SAT
    global GLOBAL_COND_122, GLOBAL_PULSES_122, GLOBAL_BASE_PARAMS
    global GLOBAL_T_PERIOD, GLOBAL_T_WIDTH, GLOBAL_VPOT, GLOBAL_VDEP
    global GLOBAL_SIM_NS, GLOBAL_SIM_SAT, GLOBAL_SIM_122
    global GLOBAL_DIFF_NS, GLOBAL_DIFF_SAT, GLOBAL_DIFF_122
    global GLOBAL_VAR_DIFF_NS, GLOBAL_VAR_DIFF_SAT, GLOBAL_VAR_DIFF_122
    global GLOBAL_VAR_NS, GLOBAL_VAR_SAT, GLOBAL_VAR_122
    global GLOBAL_LOG_NS, GLOBAL_LOG_SAT, GLOBAL_LOG_122
    global GLOBAL_SIM_TOL_VAL, GLOBAL_BEST_COST_VAL
    global GLOBAL_FIT_NS, GLOBAL_FIT_SAT, GLOBAL_FIT_122
    global GLOBAL_WEIGHTS_NS, GLOBAL_WEIGHTS_SAT, GLOBAL_WEIGHTS_122
    global GLOBAL_SUM_WEIGHTS_NS, GLOBAL_SUM_WEIGHTS_SAT, GLOBAL_SUM_WEIGHTS_122
    global GLOBAL_NORM_NONLIN_NS, GLOBAL_NORM_NONLIN_SAT, GLOBAL_NORM_NONLIN_122
    global GLOBAL_LAST_G_NS, GLOBAL_LAST_G_SAT, GLOBAL_LAST_G_122
    global _PRINTED_SIM_EXCEPTION
    
    candidate = get_dict_params(x, GLOBAL_BASE_PARAMS)
    sim_tol = GLOBAL_SIM_TOL_VAL.value if GLOBAL_SIM_TOL_VAL is not None else 1e-3
    
    # 1. Non-saturating sweep
    if GLOBAL_FIT_NS:
        update_simulator_candidate(GLOBAL_SIM_NS, candidate)
        GLOBAL_SIM_NS.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_NS.devices[0]['dev']._forceUpdateConductanceParams()
        t_stop_ns = np.max(GLOBAL_PULSES_NS) * GLOBAL_T_PERIOD
        try:
            GLOBAL_SIM_NS.run(tEnd=t_stop_ns, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_ns = GLOBAL_SIM_NS.getTime()
            g_sim_ns = GLOBAL_SIM_NS.getConductance(row=1, col=1)
            g_sim_ns_ms = g_sim_ns * 1e3
            sim_pulses_ns = t_arr_ns / GLOBAL_T_PERIOD
            g_sim_ns_interp = np.interp(GLOBAL_PULSES_NS - 0.05, sim_pulses_ns, g_sim_ns_ms)
            GLOBAL_LAST_G_NS = np.copy(g_sim_ns_interp)
            
            err_sq_ns = (g_sim_ns_interp - GLOBAL_COND_NS) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_NS is not None:
                w_ns = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_NS
                sum_w_ns = max(float(np.sum(w_ns)), 1e-12)
            else:
                w_ns = GLOBAL_WEIGHTS_NS
                sum_w_ns = GLOBAL_SUM_WEIGHTS_NS

            if USE_NORMALIZED_MSE:
                cost_ns = np.sum(w_ns * err_sq_ns) / (sum_w_ns * GLOBAL_VAR_NS)
            else:
                cost_ns = np.sum(w_ns * err_sq_ns) / sum_w_ns

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_NS is not None:
                cost_deriv_ns = np.mean((np.diff(g_sim_ns_interp) - GLOBAL_DIFF_NS) ** 2) / GLOBAL_VAR_DIFF_NS
                cost_endpoints_ns = ((g_sim_ns_interp[0] - GLOBAL_COND_NS[0]) ** 2 + 3.0 * (g_sim_ns_interp[-1] - GLOBAL_COND_NS[-1]) ** 2) / GLOBAL_VAR_NS
                cost_peak_ns = ((np.max(g_sim_ns_interp) - np.max(GLOBAL_COND_NS)) ** 2) / GLOBAL_VAR_NS
                cost_ns += 0.25 * cost_deriv_ns + 1.50 * cost_endpoints_ns + 0.50 * cost_peak_ns
        except Exception as e:
            if not _PRINTED_SIM_EXCEPTION:
                _PRINTED_SIM_EXCEPTION = True
                print(f"\n[EXCEPTION IN cost_function (NS sweep)]: {e}", flush=True)
                import traceback
                traceback.print_exc()
            cost_ns = 1e6
    else:
        cost_ns = 0.0

    if GLOBAL_BEST_COST_VAL is not None and cost_ns > GLOBAL_BEST_COST_VAL.value:
        return cost_ns

    # 2. Saturating sweep
    if GLOBAL_FIT_SAT:
        update_simulator_candidate(GLOBAL_SIM_SAT, candidate)
        GLOBAL_SIM_SAT.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_SAT.devices[0]['dev']._forceUpdateConductanceParams()
        t_stop_sat = np.max(GLOBAL_PULSES_SAT) * GLOBAL_T_PERIOD
        try:
            GLOBAL_SIM_SAT.run(tEnd=t_stop_sat, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_sat = GLOBAL_SIM_SAT.getTime()
            g_sim_sat = GLOBAL_SIM_SAT.getConductance(row=1, col=1)
            g_sim_sat_ms = g_sim_sat * 1e3
            sim_pulses_sat = t_arr_sat / GLOBAL_T_PERIOD
            g_sim_sat_interp = np.interp(GLOBAL_PULSES_SAT - 0.05, sim_pulses_sat, g_sim_sat_ms)
            GLOBAL_LAST_G_SAT = np.copy(g_sim_sat_interp)
            
            err_sq_sat = (g_sim_sat_interp - GLOBAL_COND_SAT) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_SAT is not None:
                w_sat = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_SAT
                sum_w_sat = max(float(np.sum(w_sat)), 1e-12)
            else:
                w_sat = GLOBAL_WEIGHTS_SAT
                sum_w_sat = GLOBAL_SUM_WEIGHTS_SAT

            if USE_NORMALIZED_MSE:
                cost_sat = np.sum(w_sat * err_sq_sat) / (sum_w_sat * GLOBAL_VAR_SAT)
            else:
                cost_sat = np.sum(w_sat * err_sq_sat) / sum_w_sat

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_SAT is not None:
                cost_deriv_sat = np.mean((np.diff(g_sim_sat_interp) - GLOBAL_DIFF_SAT) ** 2) / GLOBAL_VAR_DIFF_SAT
                cost_endpoints_sat = ((g_sim_sat_interp[0] - GLOBAL_COND_SAT[0]) ** 2 + 3.0 * (g_sim_sat_interp[-1] - GLOBAL_COND_SAT[-1]) ** 2) / GLOBAL_VAR_SAT
                cost_peak_sat = ((np.max(g_sim_sat_interp) - np.max(GLOBAL_COND_SAT)) ** 2) / GLOBAL_VAR_SAT
                cost_sat += 0.25 * cost_deriv_sat + 1.50 * cost_endpoints_sat + 0.50 * cost_peak_sat

            # Saturation barrier penalty: disincentivize optimizer from staying below n_base (16520.0)
            n_arr_sat = GLOBAL_SIM_SAT.getState(row=1, col=1)
            if n_arr_sat is not None and len(n_arr_sat) > 0:
                n_peak_sat = float(np.max(n_arr_sat))
                if n_peak_sat < 16520.0:
                    shortfall = (16520.0 - n_peak_sat) / 16520.0
                    cost_sat += 5.0 * (shortfall ** 2)
        except Exception as e:
            if not _PRINTED_SIM_EXCEPTION:
                _PRINTED_SIM_EXCEPTION = True
                print(f"\n[EXCEPTION IN cost_function (SAT sweep)]: {e}", flush=True)
                import traceback
                traceback.print_exc()
            cost_sat = 1e6
    else:
        cost_sat = 0.0

    cost_ns_sat = cost_ns + cost_sat
    if GLOBAL_BEST_COST_VAL is not None and cost_ns_sat > GLOBAL_BEST_COST_VAL.value:
        return cost_ns_sat

    # 3. 1.22V Saturating sweep
    if GLOBAL_FIT_122:
        update_simulator_candidate(GLOBAL_SIM_122, candidate)
        GLOBAL_SIM_122.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
        GLOBAL_SIM_122.devices[0]['dev']._forceUpdateConductanceParams()
        t_stop_122 = np.max(GLOBAL_PULSES_122) * GLOBAL_T_PERIOD
        try:
            GLOBAL_SIM_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=sim_tol, title=None, recordHistory=['g', 'state', 'f22'])
            t_arr_122 = GLOBAL_SIM_122.getTime()
            g_sim_122 = GLOBAL_SIM_122.getConductance(row=1, col=1)
            g_sim_122_ms = g_sim_122 * 1e3
            sim_pulses_122 = t_arr_122 / GLOBAL_T_PERIOD
            g_sim_122_interp = np.interp(GLOBAL_PULSES_122 - 0.05, sim_pulses_122, g_sim_122_ms)
            GLOBAL_LAST_G_122 = np.copy(g_sim_122_interp)
            
            err_sq_122 = (g_sim_122_interp - GLOBAL_COND_122) ** 2
            if cohort_w_base is not None and GLOBAL_NORM_NONLIN_122 is not None:
                w_122 = cohort_w_base + (1.0 - cohort_w_base) * GLOBAL_NORM_NONLIN_122
                sum_w_122 = max(float(np.sum(w_122)), 1e-12)
            else:
                w_122 = GLOBAL_WEIGHTS_122
                sum_w_122 = GLOBAL_SUM_WEIGHTS_122

            if USE_NORMALIZED_MSE:
                cost_122 = np.sum(w_122 * err_sq_122) / (sum_w_122 * GLOBAL_VAR_122)
            else:
                cost_122 = np.sum(w_122 * err_sq_122) / sum_w_122

            # Derivative / slope match + endpoint boundary pinning + apex peak match
            if GLOBAL_DIFF_122 is not None:
                cost_deriv_122 = np.mean((np.diff(g_sim_122_interp) - GLOBAL_DIFF_122) ** 2) / GLOBAL_VAR_DIFF_122
                cost_endpoints_122 = ((g_sim_122_interp[0] - GLOBAL_COND_122[0]) ** 2 + 3.0 * (g_sim_122_interp[-1] - GLOBAL_COND_122[-1]) ** 2) / GLOBAL_VAR_122
                cost_peak_122 = ((np.max(g_sim_122_interp) - np.max(GLOBAL_COND_122)) ** 2) / GLOBAL_VAR_122
                cost_122 += 0.25 * cost_deriv_122 + 1.50 * cost_endpoints_122 + 0.50 * cost_peak_122
        except Exception as e:
            if not _PRINTED_SIM_EXCEPTION:
                _PRINTED_SIM_EXCEPTION = True
                print(f"\n[EXCEPTION IN cost_function (122 sweep)]: {e}", flush=True)
                import traceback
                traceback.print_exc()
            cost_122 = 1e6
    else:
        cost_122 = 0.0
        
    combined_cost = cost_ns_sat + cost_122
    
    # Soft Bayesian prior on gScale to break collinearity with kappa and avoid high-gScale drift
    if GSCALE_PRIOR_WEIGHT > 0.0 and GSCALE_NOMINAL > 0.0:
        g_scale_cand = float(candidate.get("gScale", GSCALE_NOMINAL))
        g_scale_dev = (g_scale_cand - GSCALE_NOMINAL) / GSCALE_NOMINAL
        cost_prior_gscale = GSCALE_PRIOR_WEIGHT * (g_scale_dev ** 2)
        combined_cost += cost_prior_gscale
    
    if GLOBAL_BEST_COST_VAL is not None:
        with GLOBAL_BEST_COST_VAL.get_lock():
            if combined_cost < GLOBAL_BEST_COST_VAL.value:
                GLOBAL_BEST_COST_VAL.value = combined_cost
                
    global GLOBAL_LAST_X, GLOBAL_LAST_COST
    GLOBAL_LAST_X = np.copy(x)
    GLOBAL_LAST_COST = combined_cost
    if GLOBAL_FIT_NS and 'g_sim_ns_interp' in locals():
        GLOBAL_LAST_G_NS = np.copy(g_sim_ns_interp)
    if GLOBAL_FIT_SAT and 'g_sim_sat_interp' in locals():
        GLOBAL_LAST_G_SAT = np.copy(g_sim_sat_interp)
    if GLOBAL_FIT_122 and 'g_sim_122_interp' in locals():
        GLOBAL_LAST_G_122 = np.copy(g_sim_122_interp)
    
    return combined_cost

cost_function.__module__ = "__main__"


# =============================================================================
# Dynamic Cluster Telemetry Console Display Utilities
# =============================================================================
def compact_table_for_terminal(lines, max_lines, local_marker="Node (Local)"):
    """
    Compacts table lines when the cluster has more nodes than the terminal window height.
    Preserves table headers, footers, Host row, and local node row, while collapsing
    excess middle rows into a single summary line so the table never scrolls.
    """
    if len(lines) <= max_lines:
        return lines

    dividers = [i for i, l in enumerate(lines) if l.startswith("---") or l.startswith("===")]
    if len(dividers) >= 3:
        header_end = dividers[1] + 1
        footer_start = dividers[-2]
        
        header = lines[:header_end]
        rows = lines[header_end:footer_start]
        footer = lines[footer_start:]
        
        available_row_slots = max_lines - len(header) - len(footer) - 1
        if available_row_slots < 2:
            return lines[:max_lines - 1] + [lines[-1]]
            
        kept_rows = []
        host_row = None
        local_row = None
        other_rows = []
        
        for r in rows:
            if local_marker in r and local_row is None:
                local_row = r
            elif ("Host" in r or "HOST" in r) and host_row is None:
                host_row = r
            else:
                other_rows.append(r)
                
        if host_row:
            kept_rows.append(host_row)
            available_row_slots -= 1
        if local_row and local_row != host_row:
            kept_rows.append(local_row)
            available_row_slots -= 1
            
        slots_for_others = max(0, available_row_slots)
        kept_rows.extend(other_rows[:slots_for_others])
        hidden_count = len(rows) - len(kept_rows)
        
        if hidden_count > 0:
            compact_line = f"  ... ({hidden_count} additional nodes hidden to fit terminal window height) ..."
            return header + kept_rows + [compact_line] + footer
        else:
            return header + kept_rows + footer
    else:
        return lines[:max_lines - 1] + [lines[-1]]


class DynamicClusterTable:
    """
    Renders an in-place, flicker-free multi-line cluster telemetry table at the bottom of the console
    using atomic single-buffer screen updates without blank frame flashing.
    """
    def __init__(self):
        self.num_lines = 0
        self.lock = threading.Lock()
        self.last_render_t = 0.0

    def clear(self):
        with self.lock:
            if self.num_lines > 0:
                buf = [f"\033[{self.num_lines}F"]
                for _ in range(self.num_lines):
                    buf.append("\033[2K\n")
                buf.append(f"\033[{self.num_lines}F")
                sys.stdout.write("".join(buf))
                sys.stdout.flush()
                self.num_lines = 0

    def render(self, lines, force=False):
        now = time.time()
        with self.lock:
            if not force and self.num_lines > 0 and (now - self.last_render_t) < 0.20:
                return
            self.last_render_t = now
            term_size = shutil.get_terminal_size()
            cols = term_size.columns
            max_lines = max(10, term_size.lines - 2)

            # Compact table lines if total exceeds terminal window height (prevents scrolling on laptops)
            if len(lines) > max_lines:
                lines = compact_table_for_terminal(lines, max_lines)
            
            buf = []
            if self.num_lines > 0:
                buf.append(f"\033[{self.num_lines}F")
                
            for l in lines:
                truncated = l[:cols - 1]
                buf.append(f"\033[2K{truncated}\n")
                
            if self.num_lines > len(lines):
                for _ in range(self.num_lines - len(lines)):
                    buf.append("\033[2K\n")
                buf.append(f"\033[{self.num_lines - len(lines)}F")
                
            sys.stdout.write("".join(buf))
            sys.stdout.flush()
            self.num_lines = len(lines)

GLOBAL_CLUSTER_TABLE = DynamicClusterTable()
GLOBAL_LOBBY_TABLE = DynamicClusterTable()
GLOBAL_NODE_TABLE = DynamicClusterTable()
GLOBAL_ACTIVE_DISPATCHER = None

def customize_table_for_local_node(lines, local_node_id, local_ip, local_hostname):
    """
    Transforms the synchronized Cluster Host table lines for local Node display:
    - Marks '(Local)' on the row corresponding to this machine.
    - Sets Role to 'Node (Local)' for this machine.
    - Sets Host's role to 'Host' (removing '(Local)' from Host).
    """
    out = []
    node_keys = [str(local_node_id).strip(), str(local_ip).strip(), str(local_hostname).strip()]
    bare_node_id = str(local_node_id).replace("node-", "").replace("slave-", "").strip()
    if bare_node_id:
        node_keys.append(bare_node_id)
        parts = bare_node_id.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            node_keys.append(parts[0])

    node_keys = [k for k in node_keys if len(k) >= 3]

    for line in lines:
        new_line = line
        if "Host (Local)" in new_line:
            new_line = new_line.replace("Host (Local)", "Host        ")
        elif "MASTER (Local)" in new_line:
            new_line = new_line.replace("MASTER (Local)", "HOST          ")

        if "Press [ENTER] to lock in cluster and begin optimization..." in new_line:
            new_line = "Press [ENTER] on Host to lock in cluster and begin optimization..."
        elif "Press [ENTER]" in new_line:
            new_line = new_line.replace("Press [ENTER]", "Press [ENTER] on Host")

        lower_line = new_line.lower()
        is_my_row = any(k.lower() in lower_line for k in node_keys)
        if is_my_row:
            if "Node          " in new_line:
                new_line = new_line.replace("Node          ", "Node (Local)  ")
            elif "Node        " in new_line:
                new_line = new_line.replace("Node        ", "Node (Local)")
            elif "Slave       " in new_line:
                new_line = new_line.replace("Slave       ", "Node (Local)")
            elif "Node  " in new_line:
                new_line = new_line.replace("Node  ", "Node (Local)")
            elif "Slave " in new_line:
                new_line = new_line.replace("Slave ", "Node (Local)")
        else:
            if "HOST" not in new_line:
                if "Node (Local)  " in new_line:
                    new_line = new_line.replace("Node (Local)  ", "Node          ")
                elif "Node (Local)" in new_line:
                    new_line = new_line.replace("Node (Local)", "Node        ")
        out.append(new_line)
    return out

def print_log(message):
    GLOBAL_CLUSTER_TABLE.clear()
    sys.stdout.write(message + "\n")
    sys.stdout.flush()
    if GLOBAL_ACTIVE_DISPATCHER is not None and getattr(GLOBAL_ACTIVE_DISPATCHER, 'last_telemetry_lines', None):
        GLOBAL_CLUSTER_TABLE.render(GLOBAL_ACTIVE_DISPATCHER.last_telemetry_lines, force=True)

def print_lobby_log(message):
    GLOBAL_LOBBY_TABLE.clear()
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

def update_dashboard(phase, current, total, cost, best_xk, dither=None, w_base=None, avg_k=None, cluster_info=""):
    pass

def clear_dashboard():
    GLOBAL_CLUSTER_TABLE.clear()


# =============================================================================
# Cluster Host Manager & Dynamic Task Dispatcher
# =============================================================================
class NodeConnection:
    """Represents an active connection to a remote Compute Node."""
    def __init__(self, node_id, conn, addr, hostname, cores, compute_workers, code_hash="", molmem_hash=""):
        self.node_id = node_id
        self.conn = conn
        self.addr = addr
        self.hostname = hostname
        self.cores = cores
        self.initial_workers = compute_workers
        self.compute_workers = compute_workers
        self.code_hash = code_hash
        self.molmem_hash = molmem_hash
        self.in_flight = {}  # task_id -> task_input
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.active = True
        self.ready = False
        self.tasks_completed = 0
        self.gen_completed = 0
        self.total_completed = 0
        self.last_seen = time.time()
        self.last_send_t = time.time()
        self.dropped_time = 0.0
        self.is_congested = False
        self.last_congested_time = 0.0
        self.cpu_util = 0.0

    @property
    def cap_ratio(self):
        base = max(1, getattr(self, 'initial_workers', self.compute_workers))
        return float(self.compute_workers) / float(base)

    @property
    def ip(self):
        return self.addr[0] if (getattr(self, 'addr', None) and len(self.addr) > 0) else ""

SlaveConnection = NodeConnection

class ClusterDispatcher:
    """
    Manages the cluster pool on the Cluster Host.
    - Allocates compute workers dynamically via adaptive PID communication governor.
    - Distributes tasks dynamically maintaining up to 2x each Node's worker capacity in buffer.
    - Recycles in-flight tasks if any Node disconnects.
    - Accepts late-joining Nodes mid-run.
    """
    def __init__(self, master_cores, tcp_port=DEFAULT_TCP_PORT):
        self.master_cores = master_cores
        self.tcp_port = tcp_port
        self.master_ip = get_local_ip()
        self.host_ip = self.master_ip
        self.nodes = {}  # node_id -> NodeConnection
        self.slaves = self.nodes
        self.updating_nodes = {}  # host_key -> dict(node_id, ip, hostname, cores, compute_workers, timestamp, old_hash)
        self.nodes_lock = threading.RLock()
        self.slaves_lock = self.nodes_lock
        self.is_locked = False
        self.is_running = False
        self.setup_payload = None
        self.local_pool = None
        self.master_compute_workers = max(1, master_cores)
        self.server_sock = None
        self.accept_thread = None
        self.local_in_flight = {}
        self.local_gen_completed = 0
        self.local_total_completed = 0
        self.last_telemetry_lines = None
        self._last_telemetry_bcast_t = 0.0
        # Adaptive PID Cluster Communication Governor (Setpoint 0.80)
        self.comm_cores = COMM_CORES_FLOOR
        self.comm_pid_integral = 0.0
        self.comm_pid_last_err = 0.0
        self.comm_pid_last_t = time.time()
        self.comm_hold_start_t = 0.0
        self.comm_step_up_t = 0.0
        self.gen_start_t = 0.0
        self.comm_ratio = 1.0
        # Dynamic 95% CPU Utilization Governor for Host
        self.cpu_governor = CpuGovernor(self.master_cores, max(1, self.master_cores - 1), target_util=TARGET_CPU_UTIL, deadband=CPU_GOVERNOR_DEADBAND)
        self.host_cpu_util = 0.0
        self.pending_setup_logs = []
        self.best_overall_cost = float('inf')
        self.best_overall_candidate = None
        self.last_candidate_bcast_t = 0.0
        self.active_device_name = None
        self.active_selection_str = None
        self.candidate_responses = []
        self.harvested_candidates = []  # Candidates harvested pre-OTA or during handshakes
        global GLOBAL_ACTIVE_DISPATCHER
        GLOBAL_ACTIVE_DISPATCHER = self

    def flush_setup_logs(self):
        with self.nodes_lock:
            if self.pending_setup_logs:
                for msg in self.pending_setup_logs:
                    sys.stdout.write(msg + "\n")
                sys.stdout.flush()
                self.pending_setup_logs.clear()

    def send_to_node(self, node, msg):
        try:
            with node.send_lock:
                node.last_send_t = time.time()
                return send_framed_msg(node.conn, msg)
        except Exception:
            return False

    send_to_slave = send_to_node

    def register_best_candidate(self, candidate_x, cost, gen=0, force_broadcast=False):
        """
        Registers an improved candidate into the cluster optimization pool.
        Updates Host local cache and broadcasts candidate update to all active nodes.
        """
        if cost is None or np.isinf(cost) or np.isnan(cost) or candidate_x is None:
            return
        if cost < self.best_overall_cost:
            self.best_overall_cost = float(cost)
            p_str = format_params_summary(candidate_x)
            cand_dict = {
                "cost": float(cost),
                "params": [float(v) for v in candidate_x],
                "params_phys_str": p_str,
                "gen": int(gen),
                "device": self.active_device_name or "unknown",
                "selection": self.active_selection_str or "all",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.best_overall_candidate = cand_dict
            save_node_best_candidate(cand_dict)
            self.broadcast_best_candidate(cand_dict, force=force_broadcast)

    def broadcast_best_candidate(self, candidate_dict, force=False):
        """
        Broadcasts the current best candidate to all active nodes (throttled to 1.0s unless forced).
        """
        if not candidate_dict or not isinstance(candidate_dict, dict):
            return
        now = time.time()
        if not force and (now - getattr(self, 'last_candidate_bcast_t', 0.0)) < 1.0:
            return
        self.last_candidate_bcast_t = now
        with self.nodes_lock:
            active_nodes = [s for s in self.nodes.values() if s.active]
        for s in active_nodes:
            self.send_to_node(s, {
                "type": "BEST_CANDIDATE_UPDATE",
                "candidate": candidate_dict
            })

    def query_cluster_best_candidate(self, device_name, selection_str):
        """
        Queries connected nodes and local Host cache for the best candidate matching device_name / selection_str.
        Displays metrics in the Host CLI and prompts the user whether to seed it into the initial population.
        Returns the selected candidate dictionary if accepted, else None.
        """
        candidates = []
        # 1. Check Host's own local disk cache
        host_cand = load_node_best_candidate(device_name, selection_str)
        if host_cand and isinstance(host_cand, dict) and "cost" in host_cand:
            candidates.append((host_cand, "Host Local Cache"))

        # 1b. Check candidates harvested from nodes prior to OTA / Handshake
        with self.nodes_lock:
            harvested = list(getattr(self, "harvested_candidates", []))
        for hc in harvested:
            if not isinstance(hc, dict) or "cost" not in hc:
                continue
            c_dev = hc.get("device")
            c_sel = hc.get("selection")
            dev_match = (not c_dev or c_dev == "unknown" or c_dev == device_name)
            sel_match = (not c_sel or c_sel == "all" or c_sel == selection_str)
            if dev_match and sel_match:
                source_label = f"{hc.get('source_node', 'Harvested Node Cache')} (Pre-OTA Harvest)"
                candidates.append((hc, source_label))

        # 1c. Check if a preset text file exists on disk (fallback discovery if not present in cache json)
        fitting_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(fitting_dir, f"{device_name}_preset_devices_py_subset_{selection_str}.txt")
        if os.path.exists(txt_path) and not any("Host Local Cache" in src for _, src in candidates):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read()
                dict_start = txt_content.find("return {")
                dict_end = txt_content.rfind("}")
                if dict_start != -1 and dict_end != -1:
                    dict_str = txt_content[dict_start + 7:dict_end + 1]
                    import re
                    p_dict = {}
                    for line in dict_str.splitlines():
                        m = re.match(r'\s*"([^"]+)":\s*([0-9.eE+-]+),?', line)
                        if m:
                            k, val_str = m.group(1), m.group(2)
                            p_dict[k] = float(val_str)
                    if p_dict and "kappa" in p_dict:
                        preset_p_vals = [
                            p_dict.get("kappa", 1.2e7), p_dict.get("alpha", 3.0), p_dict.get("vTh", 0.6),
                            p_dict.get("gScale", 6.05), p_dict.get("rC", 0.04), p_dict.get("vSmooth", 0.05),
                            p_dict.get("nSmooth", 350.0), min(50.0, max(0.1, p_dict.get("kDischarge", 5.0))),
                            p_dict.get("E_a", 0.25), p_dict.get("R_th", 3000.0), p_dict.get("tau_th", 1.5e-8),
                            p_dict.get("gamma", -0.3), p_dict.get("beta_alpha", 2.0),
                            p_dict.get("beta_s", 2.5), p_dict.get("rate_asymmetry", 0.45),
                            min(0.70, max(0.10, p_dict.get("fDischarge", 0.35))),
                            p_dict.get("rLatency", 0.70),
                            min(1.90, max(0.10, p_dict.get("rBottomBlend", 1.0))),
                            p_dict.get("k_overdrive", 2.85)
                        ]
                        preset_opt = to_optimizer_params(preset_p_vals).tolist()
                        cand_preset = {
                            "cost": 999.0,
                            "params": preset_opt,
                            "params_phys_str": f"Disk Preset: {os.path.basename(txt_path)}",
                            "device": device_name,
                            "selection": selection_str,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(txt_path)))
                        }
                        candidates.append((cand_preset, f"Preset File ({os.path.basename(txt_path)})"))
            except Exception:
                pass

        # 2. Query all active remote nodes
        with self.nodes_lock:
            active_nodes = [s for s in self.nodes.values() if s.active]

        for s in active_nodes:
            self.send_to_node(s, {
                "type": "QUERY_BEST_CANDIDATE",
                "device": device_name,
                "selection": selection_str
            })

        # 3. Collect replies with short non-blocking timeout (up to 1.2s total)
        if active_nodes:
            node_socks = {s.conn: s for s in active_nodes}
            deadline = time.time() + 1.2
            while node_socks and time.time() < deadline:
                rem_t = max(0.05, deadline - time.time())
                try:
                    rlist, _, _ = select.select(list(node_socks.keys()), [], [], rem_t)
                    if not rlist:
                        break
                    for sock in rlist:
                        s = node_socks.get(sock)
                        if s is None:
                            continue
                        msg = recv_framed_msg(sock)
                        if not msg:
                            node_socks.pop(sock, None)
                            continue
                        if msg.get("type") == "LOBBY_HEARTBEAT":
                            if "cpu_util" in msg:
                                s.cpu_util = float(msg["cpu_util"])
                            continue
                        node_socks.pop(sock, None)
                        if msg.get("type") == "QUERY_BEST_CANDIDATE_ACK":
                            c = msg.get("candidate")
                            if c and isinstance(c, dict) and "cost" in c:
                                source_name = f"Node {s.hostname} ({s.addr[0]})"
                                candidates.append((c, source_name))
                except Exception:
                    break

        if not candidates:
            print("\n[+] No cached best candidate found in cluster pool for this dataset.")
            return None

        # Pick candidate with lowest cost
        best_cand, best_source = min(candidates, key=lambda item: item[0].get("cost", float('inf')))
        best_cost = best_cand.get("cost", float('inf'))
        if np.isinf(best_cost):
            return None

        print("\n" + "=" * 100)
        print(f"  CLUSTER BEST CANDIDATE DISCOVERED ({best_source})")
        print("=" * 100)
        print(f"  Device / Selection : [{best_cand.get('device', device_name)} / {best_cand.get('selection', selection_str)}]")
        print(f"  Cost (MSE)         : {best_cost:.6f} | Generation: {best_cand.get('gen', 0)}")
        p_str = best_cand.get("params_phys_str", "")
        if p_str:
            print(f"  Parameters         : {p_str}")
        if best_cand.get("timestamp"):
            print(f"  Timestamp          : {best_cand['timestamp']}")
        print("=" * 100)

        try:
            choice = input("Pull and seed into optimization population? [Y/n]: ").strip().lower()
        except Exception:
            choice = "y"

        if choice in ("", "y", "yes"):
            print("[Seed Baseline] Confirmed. Candidate will be seeded as elite into initial population.")
            self.best_overall_cost = best_cost
            self.best_overall_candidate = best_cand
            save_node_best_candidate(best_cand)
            self.broadcast_best_candidate(best_cand, force=True)
            return best_cand
        else:
            print("[Seed Baseline] Skipped candidate pull. Starting fresh with balanced physical heuristic baseline.")
            return None

    def update_intergen_status(self, status_msg):
        with self.nodes_lock:
            if self.last_telemetry_lines and len(self.last_telemetry_lines) > 2:
                cols = shutil.get_terminal_size().columns
                lines = list(self.last_telemetry_lines)
                lines[1] = f"  CLUSTER TELEMETRY | Phase 1/2: DE | [{status_msg}]"[:cols - 1]
                self.last_telemetry_lines = lines
                GLOBAL_CLUSTER_TABLE.render(lines, force=True)

    def update_comm_governor(self, pending_count):
        """
        Dynamically adjusts communication core allocation based on in-flight fill ratio
        using an anti-hunting PID governor targeting 0.80 (80% buffer fullness).
        Prioritizes cluster throughput by allowing comm_cores to scale up to master_cores - 1
        if needed to prevent remote worker starvation.
        Includes OS socket buffer backpressure sensing to softly prevent artificial core escalation
        when in-flight buffers collapse due to physical internet/WAN bandwidth limits.
        """
        now = time.time()
        any_congested = False
        with self.nodes_lock:
            active_nodes = [s for s in self.nodes.values() if s.active]
            if not active_nodes:
                self.comm_cores = 0
                self.comm_ratio = 1.0
                return

            total_target = sum(max(2, 2 * s.compute_workers) for s in active_nodes)
            total_in_flight = sum(len(s.in_flight) for s in active_nodes)

            # OS Socket Send Buffer Backpressure Sensing (Option 2)
            # Non-blocking query across all active node TCP sockets
            active_conns = [s.conn for s in active_nodes if getattr(s, 'conn', None)]
            if active_conns:
                try:
                    _, wlist, _ = select.select([], active_conns, [], 0)
                    writable_set = set(wlist)
                    for s in active_nodes:
                        if s.conn not in writable_set:
                            s.last_congested_time = now
                            s.is_congested = True
                        else:
                            # 1.0s smoothing window: mark non-congested once buffer drains steadily for 1.0s
                            s.is_congested = ((now - getattr(s, 'last_congested_time', 0.0)) < 1.0)
                except Exception:
                    pass
            # Partition active nodes into uncongested vs congested
            uncongested_nodes = [s for s in active_nodes if not getattr(s, 'is_congested', False)]

        ratio = total_in_flight / max(1, total_target)
        self.comm_ratio = ratio

        # If generation is winding down, clear integral and timers but preserve learned comm_cores
        if pending_count == 0 or total_target == 0:
            self.comm_pid_integral = 0.0
            self.comm_hold_start_t = 0.0
            self.comm_step_up_t = 0.0
            return

        # Initial queue ramp-up grace period: allow 2.0s for initial batches to reach nodes
        # Use learned comm_cores for high-speed initial dispatch, but freeze PID adjustments
        if (now - getattr(self, 'gen_start_t', 0.0)) < 2.0:
            self.comm_hold_start_t = 0.0
            self.comm_step_up_t = 0.0
            return

        dt = max(0.01, min(1.0, now - self.comm_pid_last_t))
        self.comm_pid_last_t = now

        target_ratio = 0.80

        # Calculate escalation error strictly from uncongested nodes.
        # If a node is congested, its buffer deficit is due to network wire capacity,
        # so it does not pull down the escalation ratio or trigger core increases.
        if uncongested_nodes:
            uncongested_target = sum(max(2, 2 * s.compute_workers) for s in uncongested_nodes)
            uncongested_in_flight = sum(len(s.in_flight) for s in uncongested_nodes)
            escalation_ratio = uncongested_in_flight / max(1, uncongested_target)
            raw_error = target_ratio - escalation_ratio
        else:
            # All active nodes are experiencing network backpressure:
            # Do not escalate; allow baseline core to service what the wire can accept.
            escalation_ratio = 1.0
            raw_error = 0.0

        # Deadband [0.75, 0.85] on uncongested nodes to eliminate micro-chatter
        if abs(raw_error) <= 0.05:
            error = 0.0
        else:
            error = raw_error

        # Allow scaling all the way up to master_cores - 1 with configured comm floor
        comm_floor = min(COMM_CORES_FLOOR, self.master_cores - 1) if active_nodes else 0
        max_comm = max(comm_floor, self.master_cores - 1)
        max_extra = max(0, max_comm - comm_floor)

        if max_extra == 0:
            self.comm_cores = comm_floor
            return

        Kp = float(max_extra) / 0.50
        Ki = Kp * 0.25
        Kd = Kp * 0.05

        # Anti-windup: clamp integral accumulator
        self.comm_pid_integral = max(-1.0, min(float(max_extra), self.comm_pid_integral + Ki * error * dt))
        derivative = (error - self.comm_pid_last_err) / dt
        self.comm_pid_last_err = error

        control_signal = Kp * error + self.comm_pid_integral + Kd * derivative
        desired_extra = max(0, min(max_extra, int(math.ceil(control_signal))))

        # Slew-Rate Attack / Responsive Decay with Per-Node Congestion Isolation:
        # Slew-Rate Attack: Step up by at most +1 core per 1.0s of sustained uncongested starvation
        # Responsive Decay: When uncongested nodes are healthy (>= 0.85), step down 1 core every 1.0s (or 0.5s if >= 0.92)
        current_extra = max(0, self.comm_cores - comm_floor)
        if desired_extra > current_extra:
            self.comm_hold_start_t = 0.0
            if self.comm_step_up_t == 0.0:
                self.comm_step_up_t = now
            elif (now - self.comm_step_up_t) >= 1.0:
                self.comm_cores = min(max_comm, self.comm_cores + 1)
                self.comm_step_up_t = now
        elif desired_extra < current_extra:
            self.comm_step_up_t = 0.0
            if not uncongested_nodes or escalation_ratio >= 0.85:
                release_delay = 0.5 if (escalation_ratio >= 0.92 or not uncongested_nodes) else 1.0
                if self.comm_hold_start_t == 0.0:
                    self.comm_hold_start_t = now
                elif (now - self.comm_hold_start_t) >= release_delay:
                    self.comm_cores = max(comm_floor, self.comm_cores - 1)
                    self.comm_hold_start_t = now
            else:
                self.comm_hold_start_t = 0.0
        else:
            self.comm_step_up_t = 0.0
            self.comm_hold_start_t = 0.0

    def get_current_master_workers(self, active_nodes_count=None, active_slaves_count=None):
        """Returns dynamic number of local compute workers on Host based on active connected nodes, PID comm allocation, and 95% CPU governor."""
        self.host_cpu_util = get_system_cpu_util()
        cpu_allowed = self.cpu_governor.update(self.host_cpu_util)
        if active_nodes_count is None:
            active_nodes_count = active_slaves_count
        if active_nodes_count is None:
            with self.nodes_lock:
                active_nodes_count = len([s for s in self.nodes.values() if s.active])
        comm_allowed = max(1, min(self.master_cores - self.comm_cores, 61)) if active_nodes_count else max(1, min(self.master_cores, 61))
        return max(1, min(comm_allowed, cpu_allowed, 61))

    def start_server(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('', self.tcp_port))
        self.server_sock.listen(128)
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        self.lobby_monitor_thread = threading.Thread(target=self._lobby_monitor_loop, daemon=True)
        self.lobby_monitor_thread.start()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        """
        Periodically sends a lightweight HEARTBEAT packet to active nodes every 2.0s
        if no data has been sent recently, preventing idle/inter-generation node timeouts.
        """
        while self.server_sock:
            time.sleep(2.0)
            now = time.time()
            with self.nodes_lock:
                active_nodes = [s for s in self.nodes.values() if s.active]
            for s in active_nodes:
                if (now - getattr(s, 'last_send_t', 0.0)) >= 2.0:
                    self.send_to_node(s, {"type": "HEARTBEAT"})

    def _lobby_monitor_loop(self):
        """
        Monitors connected node sockets and in-flight OTA updates during the lobby phase.
        Detects dropped / disconnected nodes instantly via select() and socket PEEK.
        Dynamically refreshes the lobby table every 0.5s when updates are in flight.
        """
        last_lobby_refresh_t = 0.0
        while not self.is_locked and self.server_sock:
            with self.nodes_lock:
                active_nodes = [s for s in self.nodes.values() if s.active]
                has_updating = bool(self.updating_nodes)

            dropped_any = False
            if active_nodes:
                sock_to_node = {s.conn: s for s in active_nodes}
                try:
                    rlist, _, _ = select.select(list(sock_to_node.keys()), [], [], 0.5)
                    if self.is_locked:
                        break
                    for sock in rlist:
                        node = sock_to_node.get(sock)
                        if node is None:
                            continue
                        try:
                            peek = sock.recv(1, socket.MSG_PEEK)
                            n_pid = getattr(node, 'node_id', '').rsplit('-', 1)[-1] if '-' in getattr(node, 'node_id', '') else ''
                            n_pid_tag = f" | PID: {n_pid}" if n_pid else ""
                            if not peek:
                                with self.nodes_lock:
                                    node.active = False
                                    if getattr(node, 'dropped_time', None) is None:
                                        node.dropped_time = time.time()
                                dropped_any = True
                                print_lobby_log(f"\n[Cluster Lobby Alert] Node disconnected: {node.hostname} ({node.addr[0]}{n_pid_tag}). Dropped from lobby.")
                            else:
                                msg = recv_framed_msg(sock)
                                if msg is None:
                                    with self.nodes_lock:
                                        node.active = False
                                        if getattr(node, 'dropped_time', None) is None:
                                            node.dropped_time = time.time()
                                    dropped_any = True
                                    print_lobby_log(f"\n[Cluster Lobby Alert] Node disconnected: {node.hostname} ({node.addr[0]}{n_pid_tag}). Dropped from lobby.")
                                elif msg.get("type") == "LOBBY_HEARTBEAT":
                                    if "cpu_util" in msg:
                                        with node.lock:
                                            node.cpu_util = float(msg["cpu_util"])
                        except Exception:
                            n_pid = getattr(node, 'node_id', '').rsplit('-', 1)[-1] if '-' in getattr(node, 'node_id', '') else ''
                            n_pid_tag = f" | PID: {n_pid}" if n_pid else ""
                            with self.nodes_lock:
                                node.active = False
                                if getattr(node, 'dropped_time', None) is None:
                                    node.dropped_time = time.time()
                            dropped_any = True
                            print_lobby_log(f"\n[Cluster Lobby Alert] Node disconnected: {node.hostname} ({node.addr[0]}{n_pid_tag}). Dropped from lobby.")
                except Exception:
                    pass
            else:
                time.sleep(0.5)

            if self.is_locked:
                break

            with self.nodes_lock:
                has_recent_dropped = any((not s.active and (time.time() - (getattr(s, 'dropped_time', 0.0) or 0.0)) < 12.0) for s in self.nodes.values())
            now_lobby = time.time()
            if not self.is_locked and (dropped_any or has_updating or has_recent_dropped or (now_lobby - last_lobby_refresh_t >= 1.0)):
                last_lobby_refresh_t = now_lobby
                self.update_lobby_display()

    def _accept_loop(self):
        while True:
            try:
                conn, addr = self.server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                configure_aggressive_keepalive(conn, idle_sec=10, interval_sec=2, count=5)
                conn.settimeout(15.0)
                threading.Thread(target=self._handle_client_handshake, args=(conn, addr), daemon=True).start()
            except Exception:
                break

    def _handle_client_handshake(self, conn, addr):
        msg = recv_framed_msg(conn)
        if not msg or msg.get("type") != "HANDSHAKE_REQUEST":
            conn.close()
            return
            
        node_id = msg.get("node_id", f"node-{addr[0]}-{addr[1]}")
        hostname = msg.get("hostname", addr[0])
        cores = msg.get("cores", 1)
        compute_workers = msg.get("compute_workers", max(1, cores - 1))
        node_hash = msg.get("node_hash") or msg.get("code_hash", "") or msg.get("slave_hash", "")
        node_molmem_hash = msg.get("molmem_hash", "")
        node_cpu_util = float(msg.get("cpu_util", 0.0))
        node_pid = node_id.rsplit("-", 1)[-1] if "-" in node_id else "unknown"

        # Fallback-tolerant Pre-OTA / Handshake Candidate Harvesting:
        # Securely capture cached candidate(s) from node BEFORE evaluating version mismatches or pushing OTA.
        node_candidates = []
        try:
            raw_cands = msg.get("cached_candidates")
            if raw_cands and isinstance(raw_cands, list):
                node_candidates.extend(raw_cands)
            raw_single = msg.get("cached_candidate")
            if raw_single and isinstance(raw_single, dict):
                node_candidates.append(raw_single)
        except Exception:
            pass

        for c in node_candidates:
            if not isinstance(c, dict):
                continue
            try:
                c_cost = float(c.get("cost", float('inf')))
                c_params = c.get("params")
                if not np.isfinite(c_cost) or c_cost <= 0.0 or not c_params:
                    continue
                if not isinstance(c_params, (list, tuple, np.ndarray)) or len(c_params) == 0:
                    continue

                c_dev = str(c.get("device", "unknown"))
                c_sel = str(c.get("selection", "all"))
                c_gen = int(c.get("gen", 0))

                # Register in Host's in-memory harvested candidates
                with self.nodes_lock:
                    if not hasattr(self, "harvested_candidates"):
                        self.harvested_candidates = []
                    # Deduplicate by device, selection, and cost within 1e-9
                    dup = any(
                        h.get("device") == c_dev and h.get("selection") == c_sel and
                        abs(float(h.get("cost", -1.0)) - c_cost) < 1e-9
                        for h in self.harvested_candidates
                    )
                    if not dup:
                        entry = dict(c)
                        entry["source_node"] = f"Node {hostname} ({addr[0]})"
                        entry["hostname"] = hostname
                        entry["ip"] = addr[0]
                        self.harvested_candidates.append(entry)

                # Update Host's local disk cache if candidate has lower cost than existing entry
                host_cached = load_node_best_candidate(c_dev, c_sel)
                host_cost = float(host_cached.get("cost", float('inf'))) if (host_cached and isinstance(host_cached, dict)) else float('inf')
                if c_cost < host_cost:
                    save_node_best_candidate(c)
                    harvest_log = f"[Cluster Auto-Sync] Captured candidate (Cost: {c_cost:.6f} | [{c_dev} / {c_sel}]) from {hostname} ({addr[0]}) prior to OTA/Handshake."
                    if not self.is_locked:
                        print_lobby_log(harvest_log)
                        self.update_lobby_display()
                    elif not self.is_running:
                        print(harvest_log)
            except Exception:
                # Gracefully swallow schema mismatches from legacy nodes
                continue
        
        script_mismatch = (node_hash != SCRIPT_CODE_HASH)
        molmem_mismatch = (node_molmem_hash != MOLMEM_LIB_HASH)
        
        # Protective validation & Auto-OTA Code/Library Synchronization:
        if script_mismatch or molmem_mismatch:
            master_code_payload = None
            molmem_payload = None
            if script_mismatch:
                try:
                    with open(os.path.abspath(__file__), "r", encoding="utf-8") as f:
                        master_code_payload = zlib.compress(f.read().encode("utf-8"))
                except Exception:
                    pass
            if molmem_mismatch:
                try:
                    molmem_payload = package_molmem_lib_payload()
                except Exception:
                    pass

            update_key = f"{hostname}_{addr[0]}"
            with self.nodes_lock:
                self.updating_nodes[update_key] = {
                    "node_id": node_id,
                    "ip": addr[0],
                    "hostname": hostname,
                    "cores": cores,
                    "compute_workers": compute_workers,
                    "old_hash": node_hash or "None",
                    "old_molmem": node_molmem_hash or "None",
                    "timestamp": time.time()
                }

            mismatch_reasons = []
            if script_mismatch:
                mismatch_reasons.append(f"Script (Host={SCRIPT_CODE_HASH} != Node={node_hash or 'None'})")
            if molmem_mismatch:
                mismatch_reasons.append(f"molmem_lib (Host={MOLMEM_LIB_HASH} != Node={node_molmem_hash or 'None'})")

            send_framed_msg(conn, {
                "type": "HANDSHAKE_ACK",
                "status": "AUTO_UPDATE_PAYLOAD",
                "node_id": node_id,
                "master_hash": SCRIPT_CODE_HASH,
                "host_hash": SCRIPT_CODE_HASH,
                "node_hash": node_hash,
                "slave_hash": node_hash,
                "master_molmem_hash": MOLMEM_LIB_HASH,
                "host_molmem_hash": MOLMEM_LIB_HASH,
                "node_molmem_hash": node_molmem_hash,
                "slave_molmem_hash": node_molmem_hash,
                "code_payload": master_code_payload,
                "molmem_payload": molmem_payload,
                "error": f"Version mismatch in: {', '.join(mismatch_reasons)}. Delivering auto-update payload."
            })
            try:
                conn.close()
            except Exception:
                pass
            total_kb = ((len(master_code_payload or b'') + len(molmem_payload or b'')) / 1024.0)
            sync_log = f"[Cluster Auto-Sync] Node {hostname} ({addr[0]}): Pushed OTA update ({', '.join(mismatch_reasons)}) [{total_kb:.1f} KB]. Node is applying update and restarting..."
            if not self.is_locked:
                print_lobby_log(sync_log)
                self.update_lobby_display()
            elif not self.is_running:
                with self.nodes_lock:
                    self.pending_setup_logs.append(sync_log)
            else:
                print_log(sync_log)
            return

        update_key = f"{hostname}_{addr[0]}"
        with self.nodes_lock:
            was_updating = False
            for k in list(self.updating_nodes.keys()):
                if k == update_key or k == node_id or self.updating_nodes[k].get("hostname") == hostname:
                    was_updating = True
                    del self.updating_nodes[k]

        node_conn = NodeConnection(node_id, conn, addr, hostname, cores, compute_workers, code_hash=node_hash, molmem_hash=node_molmem_hash)
        node_conn.cpu_util = node_cpu_util
        
        with self.nodes_lock:
            # Check for existing connection from the same machine (even if PID changed on restart)
            existing_s_id = None
            for s_id, existing_s in list(self.nodes.items()):
                if s_id == node_id or (existing_s.addr[0] == addr[0] and existing_s.hostname == hostname):
                    existing_s_id = s_id
                    break

            is_reconnected = (existing_s_id is not None)
            if is_reconnected:
                old_node = self.nodes[existing_s_id]
                old_pid = getattr(old_node, 'node_id', '').rsplit('-', 1)[-1] if '-' in getattr(old_node, 'node_id', '') else 'unknown'

                # Test if old_node is still genuinely alive and active in lobby
                old_alive = False
                if getattr(old_node, 'active', False) and getattr(old_node, 'conn', None):
                    try:
                        old_node.conn.setblocking(False)
                        peek = old_node.conn.recv(1, socket.MSG_PEEK)
                        old_alive = (peek != b'')
                    except (BlockingIOError, socket.error):
                        old_alive = True
                    except Exception:
                        old_alive = False
                    finally:
                        try:
                            old_node.conn.setblocking(True)
                        except Exception:
                            pass

                # If the existing node is genuinely alive and has a different PID, reject the incoming duplicate
                if old_alive and existing_s_id != node_id:
                    send_framed_msg(conn, {
                        "type": "HANDSHAKE_ACK",
                        "status": "DUPLICATE_REJECTED",
                        "node_id": node_id,
                        "error": f"Active coordinator session (PID {old_pid}) is already established from {hostname}."
                    })
                    try:
                        conn.close()
                    except Exception:
                        pass
                    print_lobby_log(
                        f"\n[Cluster Lobby Notice] Rejected duplicate connection from {hostname} ({addr[0]} | PID {node_pid}): "
                        f"Active session (PID {old_pid}) is already established in lobby."
                    )
                    return

                node_conn.gen_completed = old_node.gen_completed
                node_conn.total_completed = old_node.total_completed
                node_conn.tasks_completed = old_node.tasks_completed
                # Cleanly decommission old ghost socket
                try:
                    old_node.active = False
                    old_node.conn.close()
                except Exception:
                    pass
                if existing_s_id != node_id:
                    del self.nodes[existing_s_id]

                # Rapid PID churn detection (multiple orphan processes on same client machine)
                now_churn = time.time()
                if not hasattr(self, '_churn_history'):
                    self._churn_history = {}
                history = self._churn_history.setdefault(addr[0], [])
                history = [(t, p) for (t, p) in history if (now_churn - t) < 5.0]
                history.append((now_churn, node_pid))
                self._churn_history[addr[0]] = history
                distinct_pids = set(p for _, p in history)
                if len(history) >= 4 and len(distinct_pids) >= 2:
                    pids_formatted = ", ".join(sorted(distinct_pids))
                    print_lobby_log(
                        f"\n[Cluster Lobby Warning] Rapid PID churn from {hostname} ({addr[0]}): "
                        f"Multiple distinct Python processes ({pids_formatted}) are competing for the cluster connection!\n"
                        f"  -> Fix: On {hostname}, run: 'Stop-Process -Name python -Force' (or taskkill /F /IM python.exe) to terminate stale background processes.\n"
                    )

            self.nodes[node_id] = node_conn
            num_nodes = len([s for s in self.nodes.values() if s.active])
            
        # Send Handshake Acknowledgment
        send_framed_msg(conn, {
            "type": "HANDSHAKE_ACK",
            "status": "ACCEPTED",
            "node_id": node_id,
            "master_cores": self.master_cores,
            "code_hash": SCRIPT_CODE_HASH,
            "node_hash": SCRIPT_CODE_HASH,
            "molmem_hash": MOLMEM_LIB_HASH
        })
        
        pid_tag = f" | PID {node_pid}" if node_pid and node_pid != "unknown" else ""
        if was_updating:
            join_msg = f"[Cluster Auto-Sync] Node {hostname} ({addr[0]}{pid_tag}): OTA restart complete! Code verified ({SCRIPT_CODE_HASH} / lib:{MOLMEM_LIB_HASH[:8]}). Successfully joined cluster!"
        else:
            join_str = "reconnected" if is_reconnected else "joined"
            join_msg = f"[Cluster Lobby] Node {join_str}: {hostname} ({addr[0]}{pid_tag}) | Cores: {cores} | Workers: {compute_workers} | Code: {node_hash} / lib:{node_molmem_hash[:8]} (VERIFIED)" if not self.is_locked else f"[Cluster] Node {join_str} mid-run: {hostname} ({addr[0]}{pid_tag}) | Cores: {cores} | Workers: {compute_workers} | Code: {node_hash} / lib:{node_molmem_hash[:8]} (VERIFIED)"

        try:
            conn.settimeout(None)
        except Exception:
            pass

        if not self.is_locked:
            print_lobby_log(join_msg)
            self.update_lobby_display()
        elif not self.is_running:
            with self.nodes_lock:
                self.pending_setup_logs.append(join_msg)
        else:
            print_log(join_msg)

        # If simulation has already started, deliver setup payload immediately so node can join mid-run
        if self.is_running and self.setup_payload is not None:
            self._provision_node(node_conn)
        elif self.is_locked:
            send_framed_msg(node_conn.conn, {"type": "CLUSTER_LOCK"})

    def _provision_node(self, node):
        success = self.send_to_node(node, {
            "type": "SETUP_PAYLOAD",
            "payload": self.setup_payload
        })
        if not success:
            with node.lock:
                node.active = False
                if getattr(node, 'dropped_time', None) is None:
                    node.dropped_time = time.time()
            return

    _provision_slave = _provision_node

    def update_lobby_display(self, locked=False):
        if self.is_locked and not locked:
            return
        with self.nodes_lock:
            now = time.time()
            self.updating_nodes = {k: v for k, v in self.updating_nodes.items() if (now - v.get("timestamp", 0)) < 45.0}

            active_list = [s for s in self.nodes.values() if s.active]
            for s in self.nodes.values():
                if not s.active and getattr(s, 'dropped_time', 0.0) == 0.0:
                    s.dropped_time = now
            dropped_list = [s for s in self.nodes.values() if not s.active and (now - (getattr(s, 'dropped_time', 0.0) or 0.0)) < 10.0]
            num_nodes = len(active_list)
            dedicated_master_comm = 1 if num_nodes > 0 else 0
            master_workers = max(1, min(self.master_cores - dedicated_master_comm, 61))
            total_compute = master_workers + sum(s.compute_workers for s in active_list)
            
            status_tag = " [LOCKED]" if self.is_locked or locked else " (Waiting for Lock-In)"
            headers = ["Node ID", "Role", "IP Address", "Hostname", "Total Cores", "Workers", "CPU%", "Code / Lib Version"]
            self.host_cpu_util = get_system_cpu_util()
            rows = [
                ("HOST", "Host (Local)", get_local_ip(), socket.gethostname(), str(self.master_cores), str(master_workers), f"{self.host_cpu_util:.1f}%", f"{SCRIPT_CODE_HASH} / {MOLMEM_LIB_HASH[:8]} (Host)")
            ]
            # Query OS socket buffer backpressure for active nodes in lobby
            active_conns = [s.conn for s in active_list if getattr(s, 'conn', None)]
            if active_conns:
                try:
                    _, wlist, _ = select.select([], active_conns, [], 0)
                    writable_set = set(wlist)
                    for s in active_list:
                        if s.conn not in writable_set:
                            s.last_congested_time = now
                            s.is_congested = True
                        else:
                            s.is_congested = ((now - getattr(s, 'last_congested_time', 0.0)) < 1.0)
                except Exception:
                    pass

            for s in active_list:
                s_hash = getattr(s, 'code_hash', SCRIPT_CODE_HASH)
                s_lib_hash = getattr(s, 'molmem_hash', MOLMEM_LIB_HASH)
                s_id_str = str(s.node_id).replace("slave-", "node-")
                is_cong = getattr(s, 'is_congested', False)
                w_str = f"{s.compute_workers} (!)" if is_cong else str(s.compute_workers)
                s_cpu_str = f"{getattr(s, 'cpu_util', 0.0):.1f}%"
                rows.append((s_id_str, "Node", str(s.addr[0]), str(s.hostname), str(s.cores), w_str, s_cpu_str, f"{s_hash} / {s_lib_hash[:8]} (Verified)"))
            for u in self.updating_nodes.values():
                u_id = str(u.get("node_id", f"node-{u['ip']}")).replace("slave-", "node-")
                elapsed = int(now - u.get("timestamp", now))
                if elapsed < 3:
                    ota_status = f"[RECEIVING OTA... {elapsed}s]"
                else:
                    ota_status = f"[RESTARTING... {elapsed}s]"
                rows.append((u_id, "UPDATING", str(u["ip"]), str(u["hostname"]), str(u["cores"]), f"({u['compute_workers']})", "-", ota_status))
            for s in dropped_list:
                s_id_str = str(s.node_id).replace("slave-", "node-")
                rows.append((s_id_str, "DROPPED", str(s.addr[0]), str(s.hostname), str(s.cores), "0", "-", "(DISCONNECTED)"))
            
            min_w = [18, 14, 16, 15, 12, 8, 7, 26]
            widths = [max(min_w[i], max(len(headers[i]), max(len(r[i]) for r in rows)) + 2) for i in range(len(headers))]
            fmt = " ".join([f"{{:<{w}}}" for w in widths])
            table_width = max(106, sum(widths) + len(widths) - 1)
            
            lines = []
            lines.append("=" * table_width)
            lines.append(f"  CURRENT CLUSTER INVENTORY{status_tag}")
            lines.append("=" * table_width)
            lines.append(fmt.format(*headers))
            lines.append("-" * table_width)
            for r in rows:
                lines.append(fmt.format(*r))
            lines.append("-" * table_width)
            extras = []
            if self.updating_nodes:
                extras.append(f"{len(self.updating_nodes)} updating")
            if dropped_list:
                extras.append(f"{len(dropped_list)} dropped")
            status_note = f" ({', '.join(extras)})" if extras else ""
            comm_tag = f"{dedicated_master_comm} Host Comm Core(s) (Dynamic PID: 80% Setpoint)" if num_nodes > 0 else "0 Host Comm Cores"
            lines.append(f"Total Cluster Pool: {1 + num_nodes} Active Nodes{status_note} | {total_compute} Compute Workers | {comm_tag}")
            lines.append("=" * table_width)
            if not self.is_locked and not locked:
                lines.append("Press [ENTER] to lock in cluster and begin optimization...")
            
            GLOBAL_LOBBY_TABLE.render(lines)
            if self.is_locked or locked:
                GLOBAL_LOBBY_TABLE.num_lines = 0

            # Broadcast synchronized lobby table to all active nodes
            for s in active_list:
                self.send_to_node(s, {"type": "LOBBY_TABLE", "lines": lines})

    def lock_cluster(self):
        with self.nodes_lock:
            self.is_locked = True
            active_list = [s for s in self.nodes.values() if s.active]
            num_nodes = len(active_list)
            max_seed = max(1, self.master_cores // 2)
            self.comm_cores = max(COMM_CORES_FLOOR, min(num_nodes, max_seed)) if num_nodes > 0 else 0
            self.master_compute_workers = max(1, min(self.master_cores - self.comm_cores, 61))
            for s in active_list:
                self.send_to_node(s, {"type": "CLUSTER_LOCK"})

        if hasattr(self, 'lobby_monitor_thread') and self.lobby_monitor_thread.is_alive():
            try:
                self.lobby_monitor_thread.join(timeout=0.6)
            except Exception:
                pass

    def broadcast_setup_payload(self, payload):
        self.setup_payload = payload
        self.is_running = True
        with self.nodes_lock:
            for s in self.nodes.values():
                if s.active:
                    self._provision_node(s)

    def get_cluster_status_str(self):
        with self.nodes_lock:
            active_nodes = [s for s in self.nodes.values() if s.active]
            total_workers = self.master_compute_workers + sum(s.compute_workers for s in active_nodes)
            return f"Cluster: {1 + len(active_nodes)} Nodes ({total_workers} Workers)"

    def map(self, inputs_to_map, current_gen=0, maxiter=0, solver=None, is_refinement=False, current_x=None, phase_name=""):
        """
        Distributes candidate evaluations dynamically across local pool and all active Nodes.
        Maintains an active buffer of up to 2x each Node's worker pool size.
        Renders a dynamic, in-place multi-line cluster telemetry table during execution.
        Clears the table upon generation completion so prior status prints do not explode history.
        """
        total = len(inputs_to_map)
        results = [None] * total
        pending = collections.deque([(idx, item) for idx, item in enumerate(inputs_to_map)])
        
        task_lock = threading.Lock()
        results_lock = threading.Lock()
        done_event = threading.Event()
        completed_count = [0]
        gen_start_t = time.time()
        self.gen_start_t = gen_start_t
        # Seed comm_cores with active nodes up to half the OS cores, or preserve learned steady state
        with self.nodes_lock:
            active_cnt = len([s for s in self.nodes.values() if s.active])
        comm_floor = COMM_CORES_FLOOR if active_cnt > 0 else 0
        max_comm = max(comm_floor, self.master_cores - 1)
        max_seed = max(comm_floor, min(active_cnt, self.master_cores // 2)) if active_cnt > 0 else 0
        current_comm = getattr(self, 'comm_cores', None)
        if current_comm is None:
            self.comm_cores = max_seed
        else:
            self.comm_cores = max(comm_floor, min(current_comm, max_comm))
        self.comm_step_up_t = 0.0
        self.comm_hold_start_t = 0.0
        
        self.current_generation_epoch = getattr(self, 'current_generation_epoch', 0) + 1
        epoch = self.current_generation_epoch

        with self.nodes_lock:
            self.local_in_flight.clear()
            self.local_gen_completed = 0
            for s in self.nodes.values():
                s.gen_completed = 0
                with s.lock:
                    s.in_flight.clear()
                if s.active:
                    try:
                        self.send_to_node(s, {"type": "RESET_TASKS", "gen": current_gen})
                    except Exception:
                        pass

        update_frame = {
            'last_t': 0.0,
            'best': float('inf')
        }
        
        def refresh_progress(current_completed):
            now = time.time()
            if current_completed == 1 or current_completed == total or (now - update_frame['last_t']) >= 0.15:
                update_frame['last_t'] = now
                elapsed = max(0.01, now - gen_start_t)
                overall_rate = current_completed / elapsed
                pct = int(current_completed / total * 100) if total > 0 else 0
                bar_width = 15
                filled = int(current_completed / total * bar_width) if total > 0 else 0
                bar = "█" * filled + "░" * (bar_width - filled)
                
                f_min, f_max = solver.dither if solver and hasattr(solver, 'dither') else (0.01, 1.0)
                dither_str = f" | dither: [{f_min:.2f}, {f_max:.2f}]"
                best_val = update_frame['best']
                cost_str = f"{best_val:.6f}" if not np.isinf(best_val) else "inf"
                w_base_str = f" | w_base: {solver.center_w_base:.2f}" if solver and hasattr(solver, 'center_w_base') else ""
                
                if solver and hasattr(solver, 'population') and solver.population is not None and len(solver.population) > 0:
                    lo_k, hi_k = solver.de_bounds[7]
                    k_pop_phys = 10.0 ** (lo_k + solver.population[:, 7] * (hi_k - lo_k))
                    avg_k = float(np.mean(k_pop_phys))
                    k_str = f" | kDisch: avg={avg_k:.0f}"
                else:
                    k_str = ""

                active_x = current_x
                if active_x is None and solver and hasattr(solver, 'x') and solver.x is not None:
                    active_x = solver.x

                if active_x is not None:
                    p_str = format_params_summary(active_x)
                else:
                    p_str = "Initializing..."
                    
                refine_tag = " [Refinement]" if is_refinement else ""
                if phase_name:
                    phase_title = phase_name
                elif maxiter > 0:
                    phase_title = f"Phase 1/2: DE | Gen {current_gen}/{maxiter}{refine_tag}"
                else:
                    phase_title = "Evaluation Batch"
                
                with self.nodes_lock:
                    now_prune = time.time()
                    self.updating_nodes = {k: v for k, v in self.updating_nodes.items() if (now_prune - v.get("timestamp", 0)) < 45.0}

                    active_nodes = [s for s in self.nodes.values() if s.active]
                    for s in self.nodes.values():
                        if not s.active and getattr(s, 'dropped_time', 0.0) == 0.0:
                            s.dropped_time = now_prune
                    dropped_nodes = [s for s in self.nodes.values() if not s.active and (now_prune - (getattr(s, 'dropped_time', 0.0) or 0.0)) < 10.0]
                    current_master_workers = self.get_current_master_workers(active_nodes_count=len(active_nodes))
                    total_workers = current_master_workers + sum(s.compute_workers for s in active_nodes)
                    master_in_flight = len(self.local_in_flight)
                    master_completed = self.local_gen_completed
                    master_rate = master_completed / elapsed
                    total_in_flight = master_in_flight + sum(len(s.in_flight) for s in active_nodes)
                    
                    comm_str = f"{self.comm_cores} ({int(self.comm_ratio*100)}%)" if active_nodes else "0"
                    host_cpu_str = f"{self.host_cpu_util:.1f}%"
                    host_ratio = self.cpu_governor.cap_ratio
                    host_in_flight_str = f"{master_in_flight} ({host_ratio:.2f})"
                    host_completed_str = f"{master_completed:,} ({master_rate:.1f}/s)"
                    node_rows = [
                        ("HOST", "Host (Local)", socket.gethostname(), str(current_master_workers), host_cpu_str, comm_str, host_in_flight_str, host_completed_str)
                    ]
                    for s in active_nodes:
                        s_rate = s.gen_completed / elapsed
                        s_id_str = str(s.node_id).replace("slave-", "node-")
                        parts = s_id_str.rsplit("-", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            s_id_str = parts[0]
                        is_cong = getattr(s, 'is_congested', False)
                        s_ratio = s.cap_ratio
                        cong_tag = " (!)" if is_cong else ""
                        s_in_flight = f"{len(s.in_flight)}{cong_tag} ({s_ratio:.2f})"
                        node_cpu_str = f"{getattr(s, 'cpu_util', 0.0):.1f}%"
                        s_completed_str = f"{s.gen_completed:,} ({s_rate:.1f}/s)"
                        is_local_node = (str(s.hostname).lower() == socket.gethostname().lower() or s.ip in (getattr(self, 'master_ip', get_local_ip()), get_local_ip(), "127.0.0.1"))
                        role_str = "Node (Local)" if is_local_node else "Node"
                        node_rows.append((s_id_str, role_str, str(s.hostname), str(s.compute_workers), node_cpu_str, "1", s_in_flight, s_completed_str))
                    for u in self.updating_nodes.values():
                        u_id = str(u.get("node_id", f"node-{u['ip']}")).replace("slave-", "node-")
                        parts = u_id.rsplit("-", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            u_id = parts[0]
                        node_rows.append((u_id, "UPDATING", str(u["hostname"]), f"({u['compute_workers']})", "-", "-", "0 (-)", "OTA RESTART"))
                    for s in dropped_nodes:
                        s_id_str = str(s.node_id).replace("slave-", "node-")
                        parts = s_id_str.rsplit("-", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            s_id_str = parts[0]
                        node_rows.append((s_id_str, "DROPPED", str(s.hostname), "0", "-", "-", "0 (0.00)", f"{s.gen_completed:,} (OFFLINE)"))

                    headers = ["Node ID", "Role", "Hostname", "Workers", "CPU%", "Comm Cores", "In-Flight (Ratio)", "Completed"]
                    min_w = [18, 14, 16, 8, 7, 11, 18, 18]
                    widths = [max(min_w[i], max(len(headers[i]), max(len(r[i]) for r in node_rows)) + 2) for i in range(len(headers))]
                    fmt = " ".join([f"{{:<{w}}}" for w in widths])
                    table_width = max(116, sum(widths) + len(widths) - 1)

                    lines = []
                    lines.append("=" * table_width)
                    lines.append(f"  CLUSTER TELEMETRY | {phase_title} | Gen {current_gen}/{maxiter}{refine_tag} | [{bar}] {pct}% ({current_completed}/{total}) | Elapsed: {elapsed:.1f}s")
                    lines.append(f"  Best Cost: {cost_str}{w_base_str}{dither_str}{k_str}")
                    lines.append(f"  Parameters: {p_str}")
                    lines.append("-" * table_width)
                    lines.append(fmt.format(*headers))
                    lines.append("-" * table_width)
                    for r in node_rows:
                        lines.append(fmt.format(*r))
                    lines.append("-" * table_width)
                    status_extra = f" ({len(dropped_nodes)} Offline)" if dropped_nodes else ""
                    updating_extra = f" | {len(self.updating_nodes)} Updating OTA" if self.updating_nodes else ""
                    congested_count = sum(1 for s in active_nodes if getattr(s, 'is_congested', False))
                    congested_extra = f" | {congested_count} Net-Congested (Escalation Paused)" if congested_count > 0 else ""
                    all_cpus = [self.host_cpu_util] + [s.cpu_util for s in active_nodes if getattr(s, 'cpu_util', 0.0) > 0]
                    avg_cpu_str = f" | Avg CPU: {sum(all_cpus)/len(all_cpus):.1f}%" if all_cpus else ""
                    lines.append(f"Cluster Total: {1 + len(active_nodes)} Active Nodes{status_extra}{updating_extra}{congested_extra} | {total_workers} Active Workers{avg_cpu_str} | In-Flight: {total_in_flight} | Throughput: {overall_rate:.1f} eval/s")
                    lines.append("=" * table_width)

                self.last_telemetry_lines = lines
                GLOBAL_CLUSTER_TABLE.render(lines)

                # Broadcast synchronized telemetry table to all active nodes (throttled to ~5 Hz)
                now_bcast = time.time()
                if (now_bcast - getattr(self, '_last_telemetry_bcast_t', 0.0)) >= 0.20 or current_completed == total:
                    self._last_telemetry_bcast_t = now_bcast
                    for s in active_nodes:
                        self.send_to_node(s, {"type": "TELEMETRY_TABLE", "lines": lines})

        refresh_progress(0)
        
        # Local Master Pool Feeder Thread
        def local_feeder():
            while not done_event.is_set():
                with task_lock:
                    rem_pending = len(pending)
                self.update_comm_governor(rem_pending)
                current_master_limit = max(2 if self.master_cores >= 2 else 1, self.get_current_master_workers())
                with self.nodes_lock:
                    is_saturated = (len(self.local_in_flight) >= current_master_limit)
                if is_saturated:
                    time.sleep(0.002)
                    continue

                task = None
                with task_lock:
                    if pending:
                        task = pending.popleft()
                if task is None:
                    time.sleep(0.005)
                    continue
                    
                task_id, task_input = task
                with self.nodes_lock:
                    self.local_in_flight[task_id] = task_input
                
                def on_local_result(res, t_id=task_id, t_inp=task_input, t_epoch=epoch):
                    if t_epoch != getattr(self, 'current_generation_epoch', 0) or done_event.is_set():
                        return
                    with self.nodes_lock:
                        self.local_in_flight.pop(t_id, None)
                        self.local_gen_completed += 1
                        self.local_total_completed += 1
                    with results_lock:
                        if 0 <= t_id < total and results[t_id] is None:
                            results[t_id] = res
                            completed_count[0] += 1
                            if res < update_frame['best']:
                                update_frame['best'] = res
                                c_cand = t_inp[0] if isinstance(t_inp, (list, tuple)) else t_inp
                                self.register_best_candidate(c_cand, res, gen=current_gen)
                            cnt = completed_count[0]
                        else:
                            return
                    refresh_progress(cnt)
                    if cnt >= total:
                        done_event.set()
                        
                def on_local_error(err, t_id=task_id, t_inp=task_input, t_epoch=epoch):
                    if t_epoch != getattr(self, 'current_generation_epoch', 0) or done_event.is_set():
                        return
                    with self.nodes_lock:
                        self.local_in_flight.pop(t_id, None)
                    with task_lock:
                        if 0 <= t_id < total and results[t_id] is None:
                            pending.appendleft((t_id, t_inp))
                        
                try:
                    self.local_pool.apply_async(
                        cost_function, 
                        (task_input,), 
                        callback=on_local_result, 
                        error_callback=on_local_error
                    )
                except Exception:
                    with self.nodes_lock:
                        self.local_in_flight.pop(task_id, None)
                    with task_lock:
                        pending.appendleft((task_id, task_input))
                    time.sleep(0.01)

        # Dedicated Communication Thread for each Compute Node
        def node_worker_loop(node):
            while not done_event.is_set() and node.active:
                max_buffer = max(2, 2 * node.compute_workers)
                to_send = []
                with task_lock:
                    with node.lock:
                        room = max_buffer - len(node.in_flight)
                        while room > 0 and pending:
                            t = pending.popleft()
                            node.in_flight[t[0]] = t[1]
                            to_send.append(t)
                            room -= 1
                            
                # Check socket write buffer backpressure
                try:
                    _, w_check, _ = select.select([], [node.conn], [], 0)
                    if not w_check:
                        node.last_congested_time = time.time()
                        node.is_congested = True
                except Exception:
                    pass

                if to_send:
                    success = self.send_to_node(node, {
                        "type": "TASK_BATCH",
                        "tasks": to_send
                    })
                    if not success:
                        self._recycle_node_tasks(node, pending, task_lock, total=total, results=results)
                        break

                drained_any = False
                try:
                    # Non-blocking burst-drain of incoming result batches from node
                    while True:
                        r, _, _ = select.select([node.conn], [], [], 0.002)
                        if not r:
                            break
                        msg = recv_framed_msg(node.conn)
                        if msg is None:
                            self._recycle_node_tasks(node, pending, task_lock, total=total, results=results)
                            return
                        if msg.get("type") == "RESULT_BATCH":
                            batch_res = msg.get("results", [])
                            # Ingest live CPU utilization and dynamic worker scale from Compute Node
                            if "cpu_util" in msg:
                                node.cpu_util = float(msg["cpu_util"])
                            if "active_workers" in msg:
                                node.compute_workers = max(1, min(node.cores - 1 if node.cores > 1 else 1, int(msg["active_workers"])))
                            if batch_res:
                                with results_lock:
                                    with node.lock:
                                        for t_id, cost in batch_res:
                                            if not (0 <= t_id < total):
                                                continue
                                            if t_id in node.in_flight:
                                                t_inp = node.in_flight.pop(t_id, None)
                                            elif results[t_id] is None:
                                                t_inp = inputs_to_map[t_id]
                                            else:
                                                continue

                                            if results[t_id] is None:
                                                results[t_id] = cost
                                                completed_count[0] += 1
                                                node.gen_completed += 1
                                                node.total_completed += 1
                                                node.tasks_completed += 1
                                                if cost < update_frame['best']:
                                                    update_frame['best'] = cost
                                                    c_cand = t_inp[0] if isinstance(t_inp, (list, tuple)) else t_inp
                                                    self.register_best_candidate(c_cand, cost, gen=current_gen)
                                        cnt = completed_count[0]
                                refresh_progress(cnt)
                                drained_any = True
                                if cnt >= total:
                                    done_event.set()
                                    return
                            else:
                                drained_any = True
                        elif msg.get("type") == "LOBBY_HEARTBEAT":
                            if "cpu_util" in msg:
                                node.cpu_util = float(msg["cpu_util"])
                            drained_any = True
                except Exception:
                    self._recycle_node_tasks(node, pending, task_lock, total=total, results=results)
                    break
                    
                if not to_send and not drained_any:
                    time.sleep(0.002)

        slave_worker_loop = node_worker_loop

        threads = []
        launched_conns = set()
        t_local = threading.Thread(target=local_feeder, daemon=True)
        t_local.start()
        threads.append(t_local)
        
        with self.nodes_lock:
            for node in self.nodes.values():
                if node.active:
                    launched_conns.add(id(node.conn))
                    t_node = threading.Thread(target=node_worker_loop, args=(node,), daemon=True)
                    t_node.start()
                    threads.append(t_node)
                    
        def late_join_watcher():
            while not done_event.is_set():
                with self.nodes_lock:
                    for s_id, node in self.nodes.items():
                        if node.active and id(node.conn) not in launched_conns:
                            launched_conns.add(id(node.conn))
                            t_new = threading.Thread(target=node_worker_loop, args=(node,), daemon=True)
                            t_new.start()
                            threads.append(t_new)
                            print_log(f"\n[Cluster Mid-Run Join] Node {node.hostname} ({node.addr[0]}) dynamically joined active generation!")
                time.sleep(0.5)

        def cluster_watchdog():
            last_audit = {}  # s_id -> (timestamp, tasks_completed)
            gen_start_time = time.time()
            
            while not done_event.is_set():
                time.sleep(1.0)
                now = time.time()
                with self.nodes_lock:
                    has_recent_dropped = any((not s.active and (now - (getattr(s, 'dropped_time', 0.0) or 0.0)) < 12.0) for s in self.nodes.values())
                if has_recent_dropped:
                    refresh_progress(completed_count[0])

                # 5s startup grace period for workers to spool up initial candidate evaluations
                if (now - gen_start_time) < 5.0:
                    continue
                    
                with self.nodes_lock:
                    active_nodes = [s for s in self.nodes.values() if s.active]
                    
                for node in active_nodes:
                    if not node.active or done_event.is_set():
                        continue
                    s_id = node.node_id
                    with node.lock:
                        curr_in_flight = len(node.in_flight)
                        curr_completed = node.tasks_completed
                        
                    if curr_in_flight == 0:
                        last_audit[s_id] = (now, curr_completed)
                        continue
                        
                    if s_id not in last_audit:
                        last_audit[s_id] = (now, curr_completed)
                        continue
                        
                    last_t, last_done = last_audit[s_id]
                    elapsed = now - last_t
                    if elapsed < 5.0:
                        continue
                        
                    delta_done = curr_completed - last_done
                    last_audit[s_id] = (now, curr_completed)
                    
                    # Stall reclamation: if node has tasks in flight but 0 evaluations completed over 5.0s,
                    # reclaim in-flight tasks to queue without disconnecting the node.
                    if curr_in_flight > 0 and delta_done == 0:
                        with task_lock:
                            with node.lock:
                                n_reclaimed = 0
                                for t_id, t_inp in list(node.in_flight.items()):
                                    if 0 <= t_id < total and results[t_id] is None:
                                        pending.appendleft((t_id, t_inp))
                                        n_reclaimed += 1
                                node.in_flight.clear()
                        if n_reclaimed > 0:
                            print_log(
                                f"\n[Cluster Watchdog] Node {node.hostname} ({node.addr[0]}) stalled (0 evals over {elapsed:.1f}s with {n_reclaimed} in-flight). "
                                f"Reclaimed tasks to queue without disconnecting node."
                            )

        t_watchdog = threading.Thread(target=cluster_watchdog, daemon=True)
        t_watchdog.start()
        threads.append(t_watchdog)

        t_join = threading.Thread(target=late_join_watcher, daemon=True)
        t_join.start()
        threads.append(t_join)

        while not done_event.wait(timeout=0.2):
            pass
        
        # Final refresh at 100% completion; keep table persistent across generations
        refresh_progress(total)
        if solver is not None and hasattr(solver, 'x') and solver.x is not None:
            best_c = solver.population_energies[0] if hasattr(solver, 'population_energies') and len(solver.population_energies) > 0 else update_frame['best']
            if not np.isinf(best_c):
                self.register_best_candidate(solver.x, best_c, gen=current_gen, force_broadcast=True)
        return results

    def _recycle_node_tasks(self, node, pending, task_lock, total=None, results=None):
        with task_lock:
            with node.lock:
                node.active = False
                if getattr(node, 'dropped_time', None) is None:
                    node.dropped_time = time.time()
                try:
                    node.conn.close()
                except Exception:
                    pass
                n_recycled = 0
                if len(node.in_flight) > 0:
                    for t_id, t_inp in list(node.in_flight.items()):
                        if results is not None and total is not None:
                            if 0 <= t_id < total and results[t_id] is None:
                                pending.appendleft((t_id, t_inp))
                                n_recycled += 1
                        else:
                            pending.appendleft((t_id, t_inp))
                            n_recycled += 1
                    node.in_flight.clear()
                if n_recycled > 0:
                    print_log(f"\n[Cluster Alert] Node {node.hostname} ({node.addr[0]}) disconnected! Recycled {n_recycled} in-flight tasks back to queue.")

    _recycle_slave_tasks = _recycle_node_tasks

    def shutdown_cluster(self):
        with self.nodes_lock:
            for node in list(self.nodes.values()):
                if node.active:
                    node.active = False
                    try:
                        send_framed_msg(node.conn, {"type": "SHUTDOWN"})
                    except Exception:
                        pass
                    try:
                        node.conn.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        node.conn.close()
                    except Exception:
                        pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None


def safe_close_pool(pool, timeout=3.0):
    """
    Cleanly shuts down a multiprocessing.Pool across Windows and Linux.
    Safely terminates workers, joins individual worker process handles with a timeout,
    closes internal IPC queue pipes, and forces garbage collection to prevent file descriptor leaks.
    """
    if pool is None:
        return
    try:
        pool.close()
    except Exception:
        pass
    try:
        pool.terminate()
    except Exception:
        pass
    try:
        workers = list(getattr(pool, '_pool', []))
        worker_pids = [p.pid for p in workers if hasattr(p, 'pid') and p.pid]
        if workers:
            per_worker_timeout = max(0.05, timeout / len(workers))
            for p in workers:
                try:
                    p.join(timeout=per_worker_timeout)
                except Exception:
                    pass
                if p.is_alive():
                    # Escalate to hard SIGKILL on Linux to prevent orphaned workers from accumulating
                    try:
                        if hasattr(signal, 'SIGKILL'):
                            os.kill(p.pid, signal.SIGKILL)
                        else:
                            p.terminate()
                    except Exception:
                        pass
        # Sweep temporary history files owned by terminated worker processes
        if worker_pids:
            try:
                from molmem_lib.sys_utils import cleanup_temp_history_dir
                for w_pid in worker_pids:
                    try:
                        cleanup_temp_history_dir(target_pid=w_pid)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    try:
        for th_name in ('_worker_handler', '_task_handler', '_result_handler'):
            th = getattr(pool, th_name, None)
            if th is not None and th.is_alive():
                th.join(timeout=0.2)
    except Exception:
        pass
    for q_attr in ('_inqueue', '_outqueue', '_change_notifier'):
        q = getattr(pool, q_attr, None)
        if q is not None:
            try:
                if hasattr(q, 'close'):
                    q.close()
            except Exception:
                pass
            try:
                reader = getattr(q, '_reader', None)
                if reader is not None and hasattr(reader, 'close'):
                    reader.close()
            except Exception:
                pass
            try:
                writer = getattr(q, '_writer', None)
                if writer is not None and hasattr(writer, 'close'):
                    writer.close()
            except Exception:
                pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    if sys.platform == "win32":
        time.sleep(0.3)


class OtaReloadRequired(Exception):
    """
    Raised when an authoritative OTA update is written to disk on a compute node,
    signaling the node runner to perform an in-process reload without terminating
    the process or dropping back to the shell/PowerShell prompt.
    """
    def __init__(self, master_ip=None, tcp_port=None):
        super().__init__("OTA update applied. In-process reload required.")
        self.master_ip = master_ip
        self.tcp_port = tcp_port


def apply_ota_update_and_restart(ack, master_ip=None, tcp_port=None):
    """
    Overwrites local fit_device_cluster.py and/or molmem_lib with authoritative
    payloads provided by Master, backs up existing files, and hot-reloads the node.
    On Windows, triggers an in-process hot reload (OtaReloadRequired) to keep the original
    process attached in the foreground and preserve Ctrl+C console signal routing.
    On Linux, invokes os.execv to replace the process image in-place.
    """
    code_payload = ack.get("code_payload")
    molmem_payload = ack.get("molmem_payload")
    if not code_payload and not molmem_payload:
        return False
        
    try:
        script_path = os.path.abspath(__file__)
        backup_dir = os.path.join(os.path.dirname(os.path.dirname(script_path)), "_backups")
        os.makedirs(backup_dir, exist_ok=True)
        ts_now = int(time.time())

        updated_items = []
        if code_payload:
            new_code = zlib.decompress(code_payload).decode("utf-8")
            backup_file = os.path.join(backup_dir, f"fit_device_cluster_pre_ota_{ts_now}.py")
            try:
                shutil.copy2(script_path, backup_file)
            except Exception:
                pass
            scavenge_ota_backups(backup_dir=backup_dir, max_keep=3)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            updated_items.append(f"fit_device_cluster.py -> {ack.get('master_hash', 'UPDATED')}")

        if molmem_payload:
            molmem_ok = apply_molmem_lib_payload(molmem_payload)
            if molmem_ok:
                updated_items.append(f"molmem_lib -> {ack.get('master_molmem_hash', 'UPDATED')}")

        cols = shutil.get_terminal_size().columns
        table_width = max(100, min(cols - 2, 120))
        lines = [
            "=" * table_width,
            "  MOLMEM COMPUTE NODE | OVER-THE-AIR (OTA) SYNCHRONIZATION",
            "=" * table_width,
            f"  Cluster Host:    Authoritative update received from Host",
            f"  Updated Modules: {', '.join(updated_items)}",
            f"  Backup Saved:    {backup_dir}",
            "  Action:          Hot-reloading Compute Node in-process...",
            "=" * table_width
        ]
        GLOBAL_NODE_TABLE.render(lines, force=True)
        time.sleep(0.5)
        
        # Cleanly terminate any active background worker processes before reloading
        for child in multiprocessing.active_children():
            try:
                child.terminate()
                child.join(timeout=1.0)
            except Exception:
                pass

        # Clean up pre-OTA warmup temporary history files
        try:
            from molmem_lib.sys_utils import cleanup_temp_history_dir
            cleanup_temp_history_dir(force_all=True, allow_current_pid=True)
        except Exception:
            pass

        # Invalidate any stale bytecode for fit_device_cluster.py
        pycache_dir = os.path.join(os.path.dirname(script_path), "__pycache__")
        if os.path.isdir(pycache_dir):
            for f in os.listdir(pycache_dir):
                if f.startswith("fit_device_cluster") and f.endswith(".pyc"):
                    try:
                        os.remove(os.path.join(pycache_dir, f))
                    except Exception:
                        pass

        os.environ["MOLMEM_CLUSTER_ROLE"] = "node"

        # On POSIX (Linux), os.execv cleanly replaces the process image in-place without PID change
        if sys.platform != "win32":
            restart_args = []
            skip_next = False
            for a in sys.argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if a in ("--role", "-r"):
                    skip_next = True
                    continue
                if a.startswith("--role="):
                    continue
                if a in ("--connect", "-c"):
                    skip_next = True
                    continue
                if a.startswith("--connect="):
                    continue
                restart_args.append(a)

            restart_args.extend(["--role", "node"])
            if master_ip and tcp_port:
                restart_args.extend(["--connect", f"{master_ip}:{tcp_port}"])
            elif master_ip:
                restart_args.extend(["--connect", str(master_ip)])

            py_candidates = []
            if sys.executable and os.path.exists(sys.executable):
                py_candidates.append(sys.executable)
            base_py = getattr(sys, '_base_executable', None)
            if base_py and os.path.exists(base_py) and base_py not in py_candidates:
                py_candidates.append(base_py)
            for cand in [shutil.which("python"), shutil.which("python3")]:
                if cand and cand not in py_candidates:
                    py_candidates.append(cand)
            for py in py_candidates:
                try:
                    os.execv(py, [py, script_path] + restart_args)
                except Exception:
                    continue

        # On Windows, raise OtaReloadRequired to perform an in-process reload.
        # This keeps the original Python process running in the foreground,
        # completely preventing PowerShell from dropping back to the CLI prompt,
        # avoiding trampoline child detachment, and preserving Win32 Ctrl+C handling.
        raise OtaReloadRequired(master_ip=master_ip, tcp_port=tcp_port)
    except OtaReloadRequired:
        raise
    except Exception as e:
        print(f"\n[Auto-Sync Error] Failed to apply OTA update: {e}")
        return False


# =============================================================================
# Compute Node Runtime Engine
# =============================================================================
def render_node_standby_lobby(node_id, total_cores, compute_workers, session_count=1, probe_status="Probing for Cluster Host...", candidate=None):
    cols = shutil.get_terminal_size().columns
    table_width = max(100, min(cols - 2, 120))
    lines = [
        "=" * table_width,
        f"  MOLMEM COMPUTE NODE | STANDBY LOBBY (Session #{session_count})",
        "=" * table_width,
        f"  Local Node ID:   {node_id}",
        f"  Hostname:        {socket.gethostname()} | Local IP: {get_local_ip()}",
        f"  Compute Pool:    {compute_workers} Simulation Workers | 1 Dedicated Comm Core ({total_cores} Total Cores)",
        f"  Code / Lib Hash: {SCRIPT_CODE_HASH} / {MOLMEM_LIB_HASH[:8]} (Verified)",
        f"  Network Status:  {probe_status}",
        "=" * table_width,
        "  (Press Ctrl+C to terminate Compute Node daemon)"
    ]
    if candidate is None:
        candidate = load_node_best_candidate()
    footer = format_node_candidate_footer(candidate)
    GLOBAL_NODE_TABLE.render(lines + footer, force=True)

def acquire_node_daemon_lock():
    """
    Ensures that only ONE compute node coordinator daemon runs concurrently on this machine.
    Prevents orphaned background workers or duplicate processes from competing for cluster slots.
    """
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), "molmem_compute_node.lock")
    try:
        if sys.platform == "win32":
            import msvcrt
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return fd
            except (IOError, OSError):
                try:
                    os.close(fd)
                except Exception:
                    pass
                return None
        else:
            import fcntl
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except (IOError, OSError):
                try:
                    os.close(fd)
                except Exception:
                    pass
                return None
    except Exception:
        return None

def release_node_daemon_lock(lock_fd):
    if lock_fd is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                try:
                    import fcntl
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception:
                    pass
            os.close(lock_fd)
        except Exception:
            pass


def run_node_session(master_ip, tcp_port, total_cores, compute_workers, node_id=None):
    """
    Connects to Cluster Host and participates in a single simulation session until Host shuts down or disconnects.
    Returns True if session completed normally, False if disconnected/failed.
    """
    if node_id is None:
        node_id = f"node-{get_local_ip()}-{os.getpid()}"
    
    def try_connect():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        configure_aggressive_keepalive(s, idle_sec=10, interval_sec=2, count=5)
        s.settimeout(15.0)
        try:
            s.connect((master_ip, tcp_port))

            # Fallback-safe pre-handshake candidate harvesting:
            cached_single = None
            cached_all = []
            try:
                cached_single = load_node_best_candidate()
                cached_all = load_all_node_candidates()
            except Exception:
                pass

            # Ensure compute node network handshake always sends the real cryptographic hashes
            node_code_h = SCRIPT_CODE_HASH
            if not node_code_h or node_code_h == "WORKER":
                node_code_h = compute_script_version_hash(force=True)
                
            node_lib_h = MOLMEM_LIB_HASH
            if not node_lib_h or node_lib_h == "WORKER":
                node_lib_h = compute_molmem_lib_hash(force=True)

            send_framed_msg(s, {
                "type": "HANDSHAKE_REQUEST",
                "node_id": node_id,
                "hostname": socket.gethostname(),
                "cores": total_cores,
                "compute_workers": compute_workers,
                "code_hash": node_code_h,
                "molmem_hash": node_lib_h,
                "cpu_util": round(get_system_cpu_util(), 1),
                "cached_candidate": cached_single,
                "cached_candidates": cached_all
            })
            ack = recv_framed_msg(s)
            s.settimeout(None)
            if not ack or ack.get("type") != "HANDSHAKE_ACK":
                try:
                    s.close()
                except Exception:
                    pass
                return None, "NO_ACK", None
            if ack.get("status") != "ACCEPTED":
                try:
                    s.close()
                except Exception:
                    pass
                return None, ack.get("status", "REJECTED"), ack
            return s, "ACCEPTED", ack
        except Exception:
            try:
                s.close()
            except Exception:
                pass
            raise

    sock = None
    try:
        sock, status, ack = try_connect()
        if sock is None:
            if status == "DUPLICATE_REJECTED":
                err_msg = ack.get("error", "Another active session is already connected.") if ack else "Another active session is already connected."
                print(f"\n[Node Notice] {err_msg} Exiting duplicate process PID {os.getpid()}.\n", flush=True)
                sys.exit(0)
            elif status == "AUTO_UPDATE_PAYLOAD" and ack and (ack.get("code_payload") or ack.get("molmem_payload")):
                apply_ota_update_and_restart(ack, master_ip=master_ip, tcp_port=tcp_port)
                return False
            elif status in ("AUTO_UPDATE_PAYLOAD", "VERSION_MISMATCH"):
                master_h = ack.get("master_hash", "UNKNOWN") if ack else "UNKNOWN"
                cols = shutil.get_terminal_size().columns
                table_width = max(100, min(cols - 2, 120))
                lines = [
                    "=" * table_width,
                    "  [CRITICAL ERROR] CLUSTER CODE VERSION MISMATCH DETECTED",
                    "=" * table_width,
                    f"  Host Code Hash:  {master_h}",
                    f"  Local Node Hash: {SCRIPT_CODE_HASH}",
                    "  This node was REJECTED by Host because the script versions differ.",
                    "  Please copy the latest 'fit_device_cluster.py' from Host to this machine.",
                    "=" * table_width
                ]
                GLOBAL_NODE_TABLE.render(lines, force=True)
                time.sleep(3.0)
            return False
    except OtaReloadRequired:
        raise
    except Exception:
        return False
        
    node_best_candidate = load_node_best_candidate()

    def render_node_screen(screen_lines, force=False):
        footer = format_node_candidate_footer(node_best_candidate)
        GLOBAL_NODE_TABLE.render(screen_lines + footer, force=force)

    render_node_standby_lobby(node_id, total_cores, compute_workers, 1, f"Connected to Cluster Host at {master_ip}:{tcp_port}. Awaiting Host lock-in...", candidate=node_best_candidate)
    
    pool = None
    clean_shutdown = False
    
    result_queue = queue.Queue()
    sender_stop_event = threading.Event()
    connection_lost_event = threading.Event()
    sock_send_lock = threading.Lock()
    sender_thread = None
    node_cpu_governor = CpuGovernor(total_cores, compute_workers, target_util=TARGET_CPU_UTIL, deadband=CPU_GOVERNOR_DEADBAND)

    def result_sender_loop():
        last_heartbeat_t = time.time()
        while not sender_stop_event.is_set():
            batch = []
            try:
                # Wait up to 10ms for at least one completed evaluation
                item = result_queue.get(timeout=0.01)
                batch.append(item)
                # Burst drain up to 64 completed evaluations without blocking
                while len(batch) < 64:
                    try:
                        batch.append(result_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            now = time.time()
            if batch or (now - last_heartbeat_t) >= 1.0:
                curr_cpu = get_system_cpu_util()
                eff_workers = node_cpu_governor.update(curr_cpu)
                last_heartbeat_t = now
                with sock_send_lock:
                    success = send_framed_msg(sock, {
                        "type": "RESULT_BATCH",
                        "results": batch,
                        "cpu_util": round(curr_cpu, 1),
                        "active_workers": eff_workers
                    })
                if not success:
                    connection_lost_event.set()
                    break

    try:
        last_msg_t = time.time()
        last_lobby_heartbeat_t = 0.0
        cluster_locked = False
        while not clean_shutdown and not connection_lost_event.is_set():
            # Send periodic live CPU utilization heartbeat to Host while in lobby standby
            now_lobby_t = time.time()
            if not cluster_locked and pool is None and (now_lobby_t - last_lobby_heartbeat_t) >= 1.0:
                last_lobby_heartbeat_t = now_lobby_t
                curr_cpu = get_system_cpu_util()
                with sock_send_lock:
                    send_framed_msg(sock, {
                        "type": "LOBBY_HEARTBEAT",
                        "cpu_util": round(curr_cpu, 1)
                    })

            try:
                rlist, _, _ = select.select([sock], [], [], 0.5)
            except Exception:
                break
            if not rlist:
                if connection_lost_event.is_set():
                    break
                silence_t = time.time() - last_msg_t
                if silence_t > 6.0:
                    # Non-blocking probe to avoid kernel socket blocking on Linux/Docker bridge
                    conn_alive = True
                    try:
                        sock.setblocking(False)
                        peek = sock.recv(1, socket.MSG_PEEK)
                        if peek == b'':
                            conn_alive = False
                    except (BlockingIOError, socket.error):
                        # Still open in kernel, but if silence persists > 10s during active run, Host is dead
                        if silence_t > 10.0:
                            conn_alive = False
                    except Exception:
                        conn_alive = False
                    finally:
                        try:
                            sock.setblocking(True)
                        except Exception:
                            pass
                    if not conn_alive:
                        break
                continue

            msg = recv_framed_msg(sock)
            if msg is None:
                try:
                    sock.close()
                except Exception:
                    pass
                break
            last_msg_t = time.time()
                
            m_type = msg.get("type")
            if m_type == "HEARTBEAT":
                continue
            
            if m_type == "LOBBY_TABLE":
                lines = msg.get("lines", [])
                if lines:
                    custom = customize_table_for_local_node(lines, node_id, get_local_ip(), socket.gethostname())
                    render_node_screen(custom)

            elif m_type == "TELEMETRY_TABLE":
                lines = msg.get("lines", [])
                if lines:
                    custom = customize_table_for_local_node(lines, node_id, get_local_ip(), socket.gethostname())
                    render_node_screen(custom)

            elif m_type == "CLUSTER_LOCK":
                cluster_locked = True
                cols = shutil.get_terminal_size().columns
                table_width = max(100, min(cols - 2, 120))
                lines = [
                    "=" * table_width,
                    "  MOLMEM COMPUTE NODE | CLUSTER PREPARATION",
                    "=" * table_width,
                    f"  Host:            Connected to Cluster Host at {master_ip}:{tcp_port}",
                    f"  Local Node:      {node_id} | {compute_workers} Simulation Workers",
                    "  Status:          Cluster locked by Host. Spooling up simulation pool...",
                    "=" * table_width
                ]
                render_node_screen(lines, force=True)
                
            elif m_type == "SETUP_PAYLOAD":
                cluster_locked = True
                p = msg.get("payload", {})
                cols = shutil.get_terminal_size().columns
                table_width = max(100, min(cols - 2, 120))
                lines = [
                    "=" * table_width,
                    "  MOLMEM COMPUTE NODE | ACTIVE SESSION INITIALIZATION",
                    "=" * table_width,
                    f"  Host:            Connected to Cluster Host at {master_ip}:{tcp_port}",
                    f"  Local Node:      {node_id} | {compute_workers} Simulation Workers",
                    "  Status:          Compiling JIT kernels & initializing worker pool...",
                    "=" * table_width
                ]
                render_node_screen(lines, force=True)

                # Warm up local JIT compilers on node
                MolmemSimulator.warmup_compilers()
                
                tol_shared = multiprocessing.Value('d', p.get("tol_val", 3e-3))
                best_cost_shared = multiprocessing.Value('d', 1e9)
                
                init_args = (
                    p["cond_ns"], p["pulses_ns"],
                    p["cond_sat"], p["pulses_sat"],
                    p["cond_122"], p["pulses_122"],
                    p["base_params"], p["t_period"], p["t_width"],
                    p["vpot"], p["vdep"],
                    tol_shared, best_cost_shared,
                    p.get("fit_ns", True), p.get("fit_sat", True), p.get("fit_122", True)
                )
                
                safe_close_pool(pool)
                pool = None
                
                # Ensure sender_thread is stopped before creating worker pool so fork is strictly single-threaded
                if sender_thread is not None and sender_thread.is_alive():
                    sender_stop_event.set()
                    try:
                        sender_thread.join(timeout=2.0)
                    except Exception:
                        pass
                    sender_thread = None

                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        break

                init_worker.__module__ = "__main__"
                cost_function.__module__ = "__main__"
                if "__main__" in sys.modules:
                    sys.modules["__reloaded__"] = sys.modules["__main__"]

                pool = multiprocessing.Pool(
                    processes=compute_workers,
                    initializer=init_worker,
                    initargs=init_args
                )
                
                if sender_thread is None or not sender_thread.is_alive():
                    sender_stop_event.clear()
                    sender_thread = threading.Thread(target=result_sender_loop, daemon=True)
                    sender_thread.start()

            elif m_type == "BEST_CANDIDATE_UPDATE":
                cand = msg.get("candidate")
                if cand and isinstance(cand, dict) and "cost" in cand:
                    node_best_candidate = cand
                    save_node_best_candidate(cand)

            elif m_type == "QUERY_BEST_CANDIDATE":
                dev = msg.get("device")
                sel = msg.get("selection")
                cand_to_send = load_node_best_candidate(dev, sel)
                if cand_to_send is None and node_best_candidate is not None:
                    cand_to_send = node_best_candidate
                with sock_send_lock:
                    send_framed_msg(sock, {
                        "type": "QUERY_BEST_CANDIDATE_ACK",
                        "candidate": cand_to_send
                    })
                
            elif m_type == "TASK_BATCH":
                tasks = msg.get("tasks", [])
                if pool is None:
                    continue
                    
                # Stream candidate tasks into local pool asynchronously without blocking the network loop
                for task_id, task_input in tasks:
                    def make_callbacks(tid):
                        def on_task_success(cost):
                            result_queue.put((tid, cost))
                        def on_task_error(err):
                            result_queue.put((tid, 1e9))
                        return on_task_success, on_task_error
                        
                    cb_s, cb_e = make_callbacks(task_id)
                    try:
                        pool.apply_async(cost_function, (task_input,), callback=cb_s, error_callback=cb_e)
                    except Exception:
                        result_queue.put((task_id, 1e9))
                
            elif m_type == "TOL_UPDATE":
                pass
                
            elif m_type == "RESET_TASKS":
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        break
                
            elif m_type == "SHUTDOWN":
                clean_shutdown = True
                break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import traceback
        sys.stderr.write(f"\n[Node Session Error]: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        time.sleep(1.0)
    finally:
        connection_lost_event.set()
        sender_stop_event.set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        if sender_thread is not None:
            try:
                sender_thread.join(timeout=1.0)
            except Exception:
                pass
            sender_thread = None
        safe_close_pool(pool)
        pool = None
        
    return clean_shutdown

run_slave_session = run_node_session

def run_compute_node(master_ip, tcp_port, total_cores, beacon_port=DEFAULT_BEACON_PORT, wan_host=DEFAULT_WAN_HOST, fixed_master=False):
    """
    Executes on a Compute Node machine in persistent standby daemon mode:
    - Reserves 1 communication core for socket streaming.
    - Allocates max(1, total_cores - 1) to its local simulation pool.
    - Connects to Cluster Host, receives SETUP_PAYLOAD, evaluates candidate batches, and streams results.
    - Displays synchronized in-place tables (Lobby, OTA, Runtime Telemetry) so CLI never scrolls.
    - When a run completes or Host disconnects, automatically re-enters Standby Lobby in-place.
    """
    # Scavenge pre-OTA backups older than 3 versions upon initializing as a compute node
    scavenge_ota_backups(max_keep=3)
    try:
        from molmem_lib.sys_utils import cleanup_temp_history_dir
        cleanup_temp_history_dir(force_all=True, allow_current_pid=True)
    except Exception:
        pass

    global SCRIPT_CODE_HASH, MOLMEM_LIB_HASH
    SCRIPT_CODE_HASH = compute_script_version_hash(force=True)
    MOLMEM_LIB_HASH = compute_molmem_lib_hash(force=True)

    comm_cores = 1
    compute_workers = max(1, min(total_cores - comm_cores, 61))
    node_id = f"node-{get_local_ip()}-{os.getpid()}"
    
    if master_ip:
        master_ip, tcp_port = parse_host_port(master_ip, tcp_port)

    daemon_lock = acquire_node_daemon_lock()
    if daemon_lock is None:
        print(f"\n[Node Daemon Notice] Another compute node coordinator is already active on this machine. Exiting cleanly.\n", flush=True)
        return

    session_count = 0
    curr_master_ip = master_ip
    curr_tcp_port = tcp_port
    default_port = tcp_port

    spin_chars = ["|", "/", "-", "\\"]
    spin_idx = 0

    try:
        while True:
            # If no host is designated or if completing a previous session:
            if session_count > 0 or not curr_master_ip:
                discovered = False
                while not discovered:
                    spin = spin_chars[spin_idx % len(spin_chars)]
                    spin_idx += 1
                    status_text = f"[{spin}] Probing LAN (Port {beacon_port})" + (f" / WAN ({wan_host}:{default_port})" if wan_host else "") + "..."
                    render_node_standby_lobby(node_id, total_cores, compute_workers, session_count + 1, status_text)

                    if not fixed_master:
                        # Fast reconnect path: verify if last known Host is still reachable
                        if curr_master_ip:
                            reachable, _, _ = probe_tcp_host(curr_master_ip, curr_tcp_port, timeout=2.5)
                            if reachable:
                                discovered = True
                                break

                        found = discover_host_beacon(beacon_port=beacon_port, timeout=1.0)
                        if found:
                            curr_master_ip, curr_tcp_port = found
                            discovered = True
                            break

                        if wan_host:
                            reachable, w_ip, w_port = probe_tcp_host(wan_host, default_port, timeout=2.5)
                            if reachable:
                                curr_master_ip = w_ip
                                curr_tcp_port = w_port
                                discovered = True
                                break
                    else:
                        reachable, _, _ = probe_tcp_host(curr_master_ip, curr_tcp_port, timeout=2.5)
                        if reachable:
                            discovered = True
                            break

                    # Standby throttle: ensure compute node remains at 0% CPU and prevents TTY buffer overflow while idle
                    time.sleep(1.0)

            session_count += 1
            render_node_standby_lobby(node_id, total_cores, compute_workers, session_count, f"Connecting to Cluster Host at {curr_master_ip}:{curr_tcp_port}...")
            _ = run_node_session(curr_master_ip, curr_tcp_port, total_cores, compute_workers, node_id=node_id)

            GLOBAL_NODE_TABLE.clear()
            render_node_standby_lobby(node_id, total_cores, compute_workers, session_count + 1, "Session concluded. Returning to Standby Lobby...")
            time.sleep(1.0)

    except KeyboardInterrupt:
        GLOBAL_NODE_TABLE.clear()
        print("[Shutdown] Compute Node daemon terminated by user (Ctrl+C). Exiting cleanly.")
        try:
            if hasattr(os, 'killpg') and hasattr(os, 'getpgrp'):
                # Ignore SIGTERM on the parent daemon process so it completes teardown without interruption
                if hasattr(signal, 'SIGTERM'):
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.killpg(os.getpgrp(), signal.SIGTERM)
        except Exception:
            pass
        sys.exit(0)
    finally:
        release_node_daemon_lock(daemon_lock)

run_slave_node = run_compute_node


# =============================================================================
# Helper: Clustered Parallel Jacobian for Multi-Basin L-BFGS-B
# =============================================================================
def make_cluster_parallel_jacobian(cluster_dispatcher, cost_func, bounds, epsilon=1e-4):
    def parallel_jac(x):
        n = len(x)
        perturbed_xs = []
        h_vals = []
        for i in range(n):
            lo, hi = bounds[i]
            h = epsilon * (hi - lo)
            h_vals.append(h)
            
            x_plus = x.copy()
            x_plus[i] += h
            perturbed_xs.append(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= h
            perturbed_xs.append(x_minus)
            
        results = cluster_dispatcher.map(perturbed_xs, current_x=x, phase_name="Phase 2/2: L-BFGS-B (Jacobian)")
        grad = np.zeros(n)
        for i in range(n):
            y_plus = results[2 * i]
            y_minus = results[2 * i + 1]
            grad[i] = (y_plus - y_minus) / (2.0 * h_vals[i])
        return grad
    return parallel_jac


# =============================================================================
# Main Program Execution
# =============================================================================
def main():
    global SCRIPT_CODE_HASH, MOLMEM_LIB_HASH, GLOBAL_DISPATCHER_REF, GLOBAL_BEACON_REF, MolmemSimulator
    parser = argparse.ArgumentParser(description="Clustered Molecular Memristor Parameter Optimization")
    parser.add_argument("--role", choices=["auto", "host", "node", "master", "slave"], default="auto", help="Cluster role: auto, host, or node (legacy aliases: master, slave)")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT, help="TCP port for cluster streaming (default: 48899)")
    parser.add_argument("--beacon-port", type=int, default=DEFAULT_BEACON_PORT, help="UDP port for discovery beacons (default: 48898)")
    parser.add_argument("--connect", type=str, default="", help="Cluster Host IP address to directly join as Compute Node")
    parser.add_argument("--wan-host", type=str, default=DEFAULT_WAN_HOST, help="Cloudflare / WAN hostname or IP for remote cluster routing")
    parser.add_argument("--device", type=str, default="", help="Pre-select device subdirectory (e.g. Ru_azo)")
    parser.add_argument("--sweeps", type=str, default="", help="Pre-select sweeps (1=NS, 2=SAT, 3=122V, 5=ALL)")
    args = parser.parse_args()

    total_cores = os.cpu_count() or 1
    
    # -------------------------------------------------------------------------
    # Pre-Warm Local JIT Compilers & Hardware Systems Initialization
    # -------------------------------------------------------------------------
    print("Warming up local JIT compilers...", flush=True)
    MolmemSimulator.warmup_compilers()
    print("Local compiler warmup complete.\n", flush=True)

    # -------------------------------------------------------------------------
    # Auto-Discovery & Role Self-Election (Dual LAN + WAN Tunnel Staging)
    # -------------------------------------------------------------------------
    role = args.role
    if role in ("master", "host"):
        role = "host"
    elif role in ("slave", "node"):
        role = "node"

    master_ip = args.connect
    wan_host = args.wan_host
    fixed_master = bool(master_ip)
    
    if master_ip:
        master_ip, args.port = parse_host_port(master_ip, args.port)
        
    if role == "auto":
        if master_ip:
            role = "node"
        else:
            print("Broadcasting LAN discovery query to check if a Cluster Host already exists...", flush=True)
            found = discover_host_beacon(beacon_port=args.beacon_port, timeout=BEACON_TIMEOUT)
            if found:
                master_ip, master_tcp_port = found
                role = "node"
                args.port = master_tcp_port
                print(f"[LAN Discovery] Discovered active Cluster Host on local subnet at {master_ip}:{master_tcp_port}. Auto-electing as COMPUTE NODE.\n")
            else:
                reachable, wan_ip, wan_tcp_port = probe_tcp_host(wan_host, args.port, timeout=2.5)
                if reachable:
                    master_ip = wan_ip
                    args.port = wan_tcp_port
                    role = "node"
                    print(f"[WAN Discovery] Discovered active Cluster Host via Cloudflare/WAN tunnel at {master_ip}:{args.port}. Auto-electing as COMPUTE NODE.\n")
                else:
                    role = "host"
                    if wan_host:
                        print(f"No active Cluster Host found on LAN or at WAN tunnel ({wan_host}:{args.port}). Auto-electing as CLUSTER HOST coordinator.\n")
                    else:
                        print("No active Cluster Host found on LAN. Auto-electing as CLUSTER HOST coordinator.\n")

    # -------------------------------------------------------------------------
    # Node Branch (Persistent Standby Daemon)
    # -------------------------------------------------------------------------
    if role == "node":
        os.environ["MOLMEM_CLUSTER_ROLE"] = "node"
        curr_master_ip = master_ip
        curr_port = args.port
        curr_fixed_master = fixed_master
        while True:
            try:
                run_compute_node(curr_master_ip, curr_port, total_cores, beacon_port=args.beacon_port, wan_host=wan_host, fixed_master=curr_fixed_master)
                return
            except OtaReloadRequired as ota_req:
                if ota_req.master_ip:
                    curr_master_ip = ota_req.master_ip
                if ota_req.tcp_port:
                    curr_port = ota_req.tcp_port
                curr_fixed_master = True

                GLOBAL_NODE_TABLE.clear()
                os.environ.pop("MOLMEM_ACTIVE_WORKER_TASK", None)
                print("\n" + "=" * 72, flush=True)
                print("  [OTA Reload] Applying in-process update to simulation engine...", flush=True)
                print("=" * 72 + "\n", flush=True)

                # Purge molmem_lib modules from sys.modules to force fresh import of updated code
                import importlib
                importlib.invalidate_caches()
                for mod_name in list(sys.modules.keys()):
                    if mod_name.startswith("molmem_lib") or mod_name == "molmem_lib":
                        sys.modules.pop(mod_name, None)

                # Recompile and rebind fit_device_cluster.py in-place without re-invoking main()
                script_path = os.path.abspath(__file__)
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        new_source = f.read()
                    compiled = compile(new_source, script_path, "exec")
                    globals()["__name__"] = "__main__"
                    if "__main__" in sys.modules:
                        sys.modules["__reloaded__"] = sys.modules["__main__"]
                    globals()["_IN_OTA_RELOAD"] = True
                    try:
                        exec(compiled, globals())
                    finally:
                        globals().pop("_IN_OTA_RELOAD", None)
                        globals()["__name__"] = "__main__"
                        if "__main__" in sys.modules:
                            sys.modules["__reloaded__"] = sys.modules["__main__"]
                except Exception as ex:
                    print(f"[OTA Reload Error] In-place script recompile error: {ex}", flush=True)

                # Recompute in-memory cryptographic fingerprints from the updated files on disk
                SCRIPT_CODE_HASH = compute_script_version_hash(force=True)
                MOLMEM_LIB_HASH = compute_molmem_lib_hash(force=True)

                # Re-warm updated JIT compilers
                try:
                    import molmem_lib
                    MolmemSimulator = molmem_lib.MolmemSimulator
                    print("Warming up updated JIT compilers...", flush=True)
                    MolmemSimulator.warmup_compilers()
                    print("Local compiler warmup complete.\n", flush=True)
                except Exception:
                    pass

                print(f"[OTA Reload] In-process reload complete (Code: {SCRIPT_CODE_HASH} | Lib: {MOLMEM_LIB_HASH[:8] if MOLMEM_LIB_HASH else 'None'}). Reconnecting to Cluster Host...\n", flush=True)
                time.sleep(1.0)

    # -------------------------------------------------------------------------
    # Host Branch
    # -------------------------------------------------------------------------
    os.environ["MOLMEM_CLUSTER_ROLE"] = "host"
    try:
        if not ensure_molmem_core_sealed():
            sys.exit(1)
        MOLMEM_LIB_HASH = compute_molmem_lib_hash()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[Attestation Warning] Verification error: {e}", flush=True)
    print("\n" + "="*72)
    print("  STAGE: Clustered Device Parameter Optimization (Cluster Host Coordinator)")
    print("="*72 + "\n")
    
    local_ip = get_local_ip()
    print(f"Host Hostname: {socket.gethostname()} | Local IP: {local_ip}")
    print(f"Cluster Protocol Version Hash: {SCRIPT_CODE_HASH}")
    print(f"Binding TCP Cluster Server on port {args.port}...")
    print(f"Broadcasting UDP Discovery Beacons on port {args.beacon_port}...\n")
    
    # Initialize Cluster Dispatcher
    dispatcher = ClusterDispatcher(master_cores=total_cores, tcp_port=args.port)
    GLOBAL_DISPATCHER_REF = dispatcher
    dispatcher.start_server()
    
    # Start background beacon broadcaster
    beacon_broadcaster = HostBeaconBroadcaster(master_ip=local_ip, tcp_port=args.port, beacon_port=args.beacon_port)
    GLOBAL_BEACON_REF = beacon_broadcaster
    beacon_broadcaster.start()

    def handle_host_sigint(signum, frame):
        print("\n[Cluster Host] Interrupted by user (Ctrl+C). Shutting down cluster nodes...")
        try:
            beacon_broadcaster.stop()
        except Exception:
            pass
        try:
            dispatcher.shutdown_cluster()
        except Exception:
            pass
        if dispatcher.local_pool is not None:
            try:
                safe_close_pool(dispatcher.local_pool)
            except Exception:
                pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, handle_host_sigint)
    except Exception:
        pass
    
    # -------------------------------------------------------------------------
    # Interactive Lobby Console Gate
    # -------------------------------------------------------------------------
    print("="*72)
    print("  INTERACTIVE CLUSTER LOBBY")
    print("="*72)
    print("You can launch this script on other machines in your LAN right now.")
    print("They will auto-detect this Cluster Host and join the cluster.")
    if wan_host:
        print(f"WAN Tunnel / External Endpoint: {wan_host}:{args.port}")
    print("Host uses an adaptive PID communication governor (targeting 80% buffer fullness).")
    print("Nodes will reserve 1 core for communication and use the rest for simulation.\n")
    
    dispatcher.update_lobby_display()
    
    try:
        input("")
    except (KeyboardInterrupt, EOFError):
        print("\nAborting cluster setup.")
        beacon_broadcaster.stop()
        dispatcher.shutdown_cluster()
        sys.exit(0)
        
    dispatcher.lock_cluster()
    dispatcher.update_lobby_display(locked=True)
    
    # -------------------------------------------------------------------------
    # Device and Dataset Selection
    # -------------------------------------------------------------------------
    fitting_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(fitting_dir)
    data_dir = os.path.join(project_root, "PythonSimulator", "molmem_lib", "data")
    
    detected_devices = []
    for entry in os.scandir(fitting_dir):
        if entry.is_dir() and entry.name not in ["Figures", "__pycache__", "PlotData", "Screenshots", "TargetVsRawPlots"]:
            has_ns = False
            has_sat = False
            has_122 = False
            ns_file = ""
            sat_file = ""
            sat_122_file = ""
            for file in os.scandir(entry.path):
                if file.is_file() and file.name.lower().endswith(".csv"):
                    if "nonsaturation" in file.name.lower():
                        has_ns = True
                        ns_file = file.path
                    elif "1.22vsaturation" in file.name.lower():
                        has_122 = True
                        sat_122_file = file.path
                    elif "saturation" in file.name.lower():
                        has_sat = True
                        sat_file = file.path
            if has_ns and has_sat and has_122:
                detected_devices.append({
                    "name": entry.name,
                    "ns_file": ns_file,
                    "sat_file": sat_file,
                    "sat_122_file": sat_122_file
                })
                
    detected_devices.sort(key=lambda d: d["name"])
    
    if not detected_devices:
        print("[Error] No valid device subdirectories found.")
        beacon_broadcaster.stop()
        dispatcher.shutdown_cluster()
        sys.exit(1)
        
    if args.device:
        matches = [d for d in detected_devices if d["name"].lower() == args.device.lower()]
        if matches:
            selected_device = matches[0]
        else:
            selected_device = detected_devices[0]
    elif len(detected_devices) == 1:
        selected_device = detected_devices[0]
        print(f"Auto-selecting the only available device: {selected_device['name']}")
    else:
        print("Detected device subdirectories:")
        for idx, dev in enumerate(detected_devices):
            print(f" [{idx + 1}] {dev['name']}")
        try:
            selection = input(f"Select a device to fit (1-{len(detected_devices)}): ").strip()
            sel_idx = int(selection) - 1
            selected_device = detected_devices[sel_idx]
        except Exception:
            selected_device = detected_devices[0]
            
    device_name = selected_device["name"]
    csv_path_ns = selected_device["ns_file"]
    csv_path_sat = selected_device["sat_file"]
    csv_path_122 = selected_device["sat_122_file"]
    print(f"\nSelected device: {device_name}")
    
    fit_ns = True
    fit_sat = True
    fit_122 = True
    selection_str = "ALL"
    
    if args.sweeps:
        ans = args.sweeps
    else:
        print("\nSelect which datasets to include in the optimization:")
        print(" [1] Non-Saturating Sweep only")
        print(" [2] Saturating Sweep only")
        print(" [3] 1.22V Saturating Sweep only")
        print(" [4] Custom combination (comma-separated, e.g. 1,3)")
        print(" [5] All Sweeps (Default)")
        try:
            ans = input("Enter selection (1-5) [5]: ").strip()
        except Exception:
            ans = "5"
            
    if ans == "1":
        fit_ns, fit_sat, fit_122 = True, False, False
        selection_str = "NS"
    elif ans == "2":
        fit_ns, fit_sat, fit_122 = False, True, False
        selection_str = "SAT"
    elif ans == "3":
        fit_ns, fit_sat, fit_122 = False, False, True
        selection_str = "122V"
    elif ans == "4":
        sub_ans = input("Enter indices (e.g. 1,3): ").strip().split(",")
        fit_ns = "1" in sub_ans
        fit_sat = "2" in sub_ans
        fit_122 = "3" in sub_ans
        parts = []
        if fit_ns: parts.append("NS")
        if fit_sat: parts.append("SAT")
        if fit_122: parts.append("122V")
        selection_str = "_".join(parts) if parts else "ALL"
    else:
        fit_ns, fit_sat, fit_122 = True, True, True
        selection_str = "ALL"
        
    print(f"Fitting target behavior: {selection_str}")
    dispatcher.flush_setup_logs()
    dispatcher.active_device_name = device_name
    dispatcher.active_selection_str = selection_str

    seeded_candidate = dispatcher.query_cluster_best_candidate(device_name, selection_str)
    func_name = device_name.replace(" ", "_").replace("-", "_")

    # Load experimental datasets
    df_ns = pd.read_csv(csv_path_ns, comment='#')
    pulses_exp_ns_raw = df_ns["X_value"] * 330.4
    x_ns_min = pulses_exp_ns_raw.min()
    x_ns_max = pulses_exp_ns_raw.max()
    raw_pulses_exp_ns = np.asarray((pulses_exp_ns_raw - x_ns_min) / (x_ns_max - x_ns_min) * 33040.0, dtype=np.float64)
    raw_cond_exp_ns_ms = np.asarray(df_ns["Y_value"], dtype=np.float64)
    
    df_sat = pd.read_csv(csv_path_sat, comment='#')
    pulses_exp_sat_raw = df_sat["X_value"] * 520.0
    x_sat_min = pulses_exp_sat_raw.min()
    x_sat_max = pulses_exp_sat_raw.max()
    raw_pulses_exp_sat = np.asarray((pulses_exp_sat_raw - x_sat_min) / (x_sat_max - x_sat_min) * 52000.0, dtype=np.float64)
    raw_cond_exp_sat_ms = np.asarray(df_sat["Y_value"], dtype=np.float64)
    
    df_122 = pd.read_csv(csv_path_122, comment='#')
    pulses_exp_122_raw = df_122["X_value"]
    x_122_min = pulses_exp_122_raw.min()
    x_122_max = pulses_exp_122_raw.max()
    raw_pulses_exp_122 = np.asarray((pulses_exp_122_raw - x_122_min) / (x_122_max - x_122_min) * 33040.0, dtype=np.float64)
    raw_cond_exp_122_ms = np.asarray(df_122["Y_value"], dtype=np.float64)

    # Resample monotonic PCHIP target curves into 1,000 dense points matching FittedFigures
    pulses_exp_ns, cond_exp_ns_ms = create_pchip_dense_dataset(raw_pulses_exp_ns, raw_cond_exp_ns_ms, num_points=1000)
    pulses_exp_sat, cond_exp_sat_ms = create_pchip_dense_dataset(raw_pulses_exp_sat, raw_cond_exp_sat_ms, num_points=1000)
    pulses_exp_122, cond_exp_122_ms = create_pchip_dense_dataset(raw_pulses_exp_122, raw_cond_exp_122_ms, num_points=1000)

    base_params = default_device(data_dir=data_dir)

    # Save target vs raw comparison plots
    plots_dir = os.path.join(fitting_dir, "TargetVsRawPlots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes[0, 0].scatter(df_ns["X_value"], df_ns["Y_value"], color="#d9b310", alpha=0.6, edgecolors="k", s=25)
    axes[0, 0].set_title("Non-Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].plot(pulses_exp_ns, cond_exp_ns_ms, color="#328cc1", linewidth=2, label="1000-pt PCHIP Target")
    axes[0, 1].scatter(raw_pulses_exp_ns, raw_cond_exp_ns_ms, color="#d9b310", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[0, 1].set_title("Non-Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(frameon=True, fontsize=9)
    axes[1, 0].scatter(df_sat["X_value"], df_sat["Y_value"], color="#ff5a5f", alpha=0.6, edgecolors="k", s=25)
    axes[1, 0].set_title("Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].plot(pulses_exp_sat, cond_exp_sat_ms, color="#0b3c5d", linewidth=2, label="1000-pt PCHIP Target")
    axes[1, 1].scatter(raw_pulses_exp_sat, raw_cond_exp_sat_ms, color="#ff5a5f", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[1, 1].set_title("Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(frameon=True, fontsize=9)
    axes[2, 0].scatter(df_122["X_value"], df_122["Y_value"], color="#10b981", alpha=0.6, edgecolors="k", s=25)
    axes[2, 0].set_title("1.22V Saturating Raw Digitized Data", fontsize=11, fontweight="bold")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].plot(pulses_exp_122, cond_exp_122_ms, color="#1e3a8a", linewidth=2, label="1000-pt PCHIP Target")
    axes[2, 1].scatter(raw_pulses_exp_122, raw_cond_exp_122_ms, color="#10b981", alpha=0.6, edgecolors="k", s=25, label="Raw Mapped")
    axes[2, 1].set_title("1.22V Saturating Target Mapped Data (1000-pt PCHIP)", fontsize=11, fontweight="bold")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].legend(frameon=True, fontsize=9)
    plt.suptitle(f"Raw vs. Transformed Target Data - {device_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    target_vs_raw_plot_path = os.path.join(plots_dir, f"{device_name}_target_vs_raw_subset_{selection_str}.png")
    plt.savefig(target_vs_raw_plot_path, dpi=300)
    plt.close()
    print(f"Saved target vs raw digitized comparison plot to: {target_vs_raw_plot_path}")

    t_period = DEFAULT_T_PERIOD
    t_width_pot = DEFAULT_T_WIDTH_POT
    t_width_dep = DEFAULT_T_WIDTH_DEP
    vpot = PulseSource(node=None, amp=DEFAULT_V_POT, period=t_period, width=t_width_pot)
    vdep = PulseSource(node=None, amp=DEFAULT_V_DEP, period=t_period, width=t_width_dep)
    
    tol_shared = multiprocessing.Value('d', SIM_TOL)
    best_cost_shared = multiprocessing.Value('d', 1e9)
    
    # Initialize Master's own simulation objects (auto-expands compact raw arrays to 1000-pt PCHIP in-memory)
    init_worker(
        raw_cond_exp_ns_ms, raw_pulses_exp_ns, 
        raw_cond_exp_sat_ms, raw_pulses_exp_sat,
        raw_cond_exp_122_ms, raw_pulses_exp_122,
        base_params, t_period, t_width_pot, vpot, vdep,
        tol_shared, best_cost_shared,
        fit_ns, fit_sat, fit_122
    )
    
    bounds = list(DEFAULT_PARAMETER_BOUNDS)
    
    optimizer_bounds = []
    for idx, (lo, hi) in enumerate(bounds):
        if LOG_SCALE_PARAMS[idx]:
            optimizer_bounds.append((np.log10(lo), np.log10(hi)))
        else:
            optimizer_bounds.append((lo, hi))
            
    # Package SETUP_PAYLOAD with compact raw arrays (~1.4 KB total) and distribute to cluster Nodes
    setup_payload = {
        "cond_ns": raw_cond_exp_ns_ms,
        "pulses_ns": raw_pulses_exp_ns,
        "cond_sat": raw_cond_exp_sat_ms,
        "pulses_sat": raw_pulses_exp_sat,
        "cond_122": raw_cond_exp_122_ms,
        "pulses_122": raw_pulses_exp_122,
        "base_params": base_params,
        "t_period": t_period,
        "t_width": t_width_pot,
        "vpot": vpot,
        "vdep": vdep,
        "tol_val": SIM_TOL,
        "fit_ns": fit_ns,
        "fit_sat": fit_sat,
        "fit_122": fit_122
    }
    
    # Initialize Master's local pool on its dedicated local compute workers
    # Using compact raw arrays ensures pickled process_obj < 2.5 KB (well below 4 KB Windows pipe buffer)
    # restoring instantaneous (<0.2s) parallel pool spawning without host thread blocking
    init_args = (
        raw_cond_exp_ns_ms, raw_pulses_exp_ns, 
        raw_cond_exp_sat_ms, raw_pulses_exp_sat,
        raw_cond_exp_122_ms, raw_pulses_exp_122,
        base_params, t_period, t_width_pot, vpot, vdep,
        tol_shared, best_cost_shared,
        fit_ns, fit_sat, fit_122
    )
    
    master_pool_workers = max(1, min(dispatcher.master_cores, 61))
    init_active_workers = dispatcher.get_current_master_workers()
    print(f"\nInitializing Master local compute pool on {master_pool_workers} workers (active compute: {init_active_workers}, dynamic comm cores: {dispatcher.comm_cores})...", flush=True)
    t_pool_start = time.time()
    init_worker.__module__ = "__main__"
    cost_function.__module__ = "__main__"
    if "__main__" in sys.modules:
        sys.modules["__reloaded__"] = sys.modules["__main__"]
    dispatcher.local_pool = multiprocessing.Pool(
        processes=master_pool_workers,
        initializer=init_worker,
        initargs=init_args
    )
    print(f"Master local compute pool initialized successfully in {time.time() - t_pool_start:.2f}s.", flush=True)
    
    # Initialize Host process's own simulation engines for direct evaluations & callbacks
    init_worker(*init_args)
    
    # Generate initial LHS population
    n_params = len(bounds)
    pop_size_total = POP_SIZE * n_params
    print(f"Generating initial Latin Hypercube Sampling (LHS) population ({pop_size_total} candidates)...", flush=True)
    init_pop = np.zeros((pop_size_total, n_params))
    for col_idx, (lo, hi) in enumerate(optimizer_bounds):
        intervals = np.linspace(0.0, 1.0, pop_size_total + 1)
        bin_mins = intervals[:-1]
        bin_maxs = intervals[1:]
        points_lhs = bin_mins + np.random.uniform(0.0, 1.0, size=pop_size_total) * (bin_maxs - bin_mins)
        np.random.shuffle(points_lhs)
        points_rand = np.random.uniform(0.0, 1.0, size=pop_size_total)
        points = DE_LHS_BLEND * points_lhs + (1.0 - DE_LHS_BLEND) * points_rand
        init_pop[:, col_idx] = lo + points * (hi - lo)
        
    txt_path = os.path.join(fitting_dir, f"{device_name}_preset_devices_py_subset_{selection_str}.txt")
    
    if seeded_candidate is not None and "params" in seeded_candidate:
        # -------------------------------------------------------------------------
        # Case A: User confirmed pulling cached candidate or cluster elite
        # -------------------------------------------------------------------------
        try:
            cand_p = np.array(seeded_candidate["params"], dtype=float)
            if len(cand_p) == n_params:
                clamped_seed = np.clip(cand_p, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
                init_pop[0, :] = clamped_seed
                default_opt_vals = clamped_seed
                cost_val = seeded_candidate.get('cost', 0.0)
                print(f"  [Cluster Seed] Injected best candidate (Cost: {cost_val:.6f}) as elite seed into init_pop[0].")
            elif len(cand_p) == 19 and n_params == 20:
                cand_p_20 = np.zeros(20, dtype=float)
                cand_p_20[:19] = cand_p[:19]
                cand_p_20[19] = 2.85  # k_overdrive default
                cand_p = cand_p_20
                clamped_seed = np.clip(cand_p, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
                init_pop[0, :] = clamped_seed
                default_opt_vals = clamped_seed
                cost_val = seeded_candidate.get('cost', 0.0)
                print(f"  [Cluster Seed] Upgraded 19-param candidate to 20-param (k_overdrive=2.85) (Cost: {cost_val:.6f}) into init_pop[0].")
            elif len(cand_p) == 18 and n_params == 20:
                cand_p_20 = np.zeros(20, dtype=float)
                cand_p_20[:18] = cand_p[:18]
                cand_p_20[18] = 1.0   # rBottomBlend default
                cand_p_20[19] = 2.85  # k_overdrive default
                cand_p = cand_p_20
                clamped_seed = np.clip(cand_p, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
                init_pop[0, :] = clamped_seed
                default_opt_vals = clamped_seed
                cost_val = seeded_candidate.get('cost', 0.0)
                print(f"  [Cluster Seed] Upgraded 18-param candidate to 20-param (rBottomBlend=1.0, k_overdrive=2.85) (Cost: {cost_val:.6f}) into init_pop[0].")
            elif len(cand_p) == 18 and n_params == 19:
                cand_p_19 = np.zeros(19, dtype=float)
                cand_p_19[:18] = cand_p[:18]
                cand_p_19[18] = 1.0  # rBottomBlend default
                cand_p = cand_p_19
                clamped_seed = np.clip(cand_p, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
                init_pop[0, :] = clamped_seed
                default_opt_vals = clamped_seed
                cost_val = seeded_candidate.get('cost', 0.0)
                print(f"  [Cluster Seed] Upgraded 18-param candidate to 19-param (rBottomBlend=1.0) (Cost: {cost_val:.6f}) into init_pop[0].")
            elif len(cand_p) == 17 and n_params >= 19:
                cand_p_new = np.zeros(n_params, dtype=float)
                cand_p_new[:16] = cand_p[:16]
                cand_p_new[16] = 0.35  # fDischarge default
                cand_p_new[17] = cand_p[16]  # rLatency
                cand_p_new[18] = 1.0   # rBottomBlend default
                if n_params >= 20:
                    cand_p_new[19] = 2.85
                cand_p = cand_p_new
                clamped_seed = np.clip(cand_p, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
                init_pop[0, :] = clamped_seed
                default_opt_vals = clamped_seed
                cost_val = seeded_candidate.get('cost', 0.0)
                print(f"  [Cluster Seed] Upgraded 17-param candidate to {n_params}-param (fDisch=0.35, rBot=1.0) (Cost: {cost_val:.6f}) into init_pop[0].")
                
                # Load preset metadata into base_params if matching preset file exists
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            txt_content = f.read()
                        dict_start = txt_content.find("return {")
                        dict_end = txt_content.rfind("}")
                        if dict_start != -1 and dict_end != -1:
                            dict_str = txt_content[dict_start + 7:dict_end + 1]
                            import re
                            for line in dict_str.splitlines():
                                m = re.match(r'\s*"([^"]+)":\s*([0-9.eE+-]+),?', line)
                                if m:
                                    k, val_str = m.group(1), m.group(2)
                                    base_params[k] = float(val_str)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [Cluster Seed] Error injecting candidate into population: {e}")

        # Inject tight Gaussian exploration cloud around the seeded best in top 5 slots
        for elite_i in range(1, min(5, pop_size_total)):
            jitter = np.random.normal(0.0, 0.012, size=n_params)
            init_pop[elite_i, :] = np.clip(default_opt_vals + jitter, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])
    else:
        # -------------------------------------------------------------------------
        # Case B: Fresh start (declined cache / previous fit)
        # -------------------------------------------------------------------------
        print("  [Seed Baseline] Fresh initialization confirmed (declined cached/preset parameters).")
        print("  [Seed Baseline] Seeding slot 0 with balanced heuristic middle ground, slot 1 with OG default physics.")
        print("  [Seed Baseline] Population initialized with diverse Latin Hypercube Samples (LHS) for broad global exploration.")
        
        # 1. Balanced physical heuristic baseline (wisdom + interior middle ground across all physics traits)
        fresh_p_vals = [
            3.5e6,    # 0: kappa (bounds: 1e6 to 3e7, log)
            3.2,      # 1: alpha (bounds: 0.1 to 10.0)
            0.50,     # 2: vTh (bounds: 0.38 to 0.85)
            8.0,      # 3: gScale (bounds: 0.5 to 30.0)
            0.05,     # 4: rC (bounds: 0.001 to 0.15)
            0.10,     # 5: vSmooth (bounds: 0.001 to 0.30)
            350.0,    # 6: nSmooth (bounds: 10.0 to 1500.0, log)
            5.0,      # 7: kDischarge (bounds: 0.1 to 800.0, log: apex rapid drop boost amplitude)
            0.25,     # 8: E_a (bounds: 0.15 to 0.60)
            3000.0,   # 9: R_th (bounds: 50.0 to 10000.0, log)
            1.5e-8,   # 10: tau_th (bounds: 1e-9 to 3e-7, log)
            -0.30,    # 11: gamma (bounds: -0.9 to 0.0)
            2.0,      # 12: beta_alpha (bounds: 0.0 to 5.0)
            2.5,      # 13: beta_s (bounds: -5.0 to 10.0)
            0.45,     # 14: rate_asymmetry (bounds: 0.05 to 1.50)
            0.35,     # 15: fDischarge (dynamic knee transition: 0.10 to 0.70)
            0.75,     # 16: rLatency (bounds: 0.0 to 0.99)
            1.00,     # 17: rBottomBlend (bounds: 0.10 to 1.90)
            2.85      # 18: k_overdrive (bounds: 0.0 to 10.0)
        ]
        clamped_fresh = [np.clip(fresh_p_vals[i], bounds[i][0], bounds[i][1]) for i in range(n_params)]
        fresh_opt_vals = to_optimizer_params(clamped_fresh)
        init_pop[0, :] = fresh_opt_vals
        default_opt_vals = fresh_opt_vals

        # 2. OG default heuristical model from devices.py (mapped to safe interior bounds)
        og_p_vals = [
            base_params["kappa"],                                      # 0: 1.2e7
            base_params["alpha"],                                      # 1: 3.0
            base_params["vTh"],                                        # 2: 0.60
            base_params["gScale"],                                     # 3: 6.051677
            base_params["rC"],                                         # 4: 0.04
            base_params["vSmooth"],                                    # 5: 0.05
            max(200.0, base_params["nSmooth"]),                        # 6: 200.0 (safe interior above 150 floor)
            min(5000.0, max(0.5, base_params.get("kDischarge", 100.0))), # 7: 100.0 (apex rapid drop boost amplitude)
            0.20,                                                    # 8: E_a (safe interior above 0.15 floor)
            1000.0,                                                  # 9: R_th (safe interior above 100 floor)
            1.0e-8,                                                  # 10: tau_th (safe interior)
            -0.15,                                                   # 11: gamma (safe interior)
            0.5,                                                     # 12: beta_alpha
            1.0,                                                     # 13: beta_s
            base_params.get("rate_asymmetry", 0.954),                # 14: 0.954
            base_params.get("fDischarge", 0.35),                     # 15: fDischarge
            base_params.get("rLatency", 0.70),                       # 16: rLatency
            base_params.get("rBottomBlend", 1.00),                   # 17: rBottomBlend
            base_params.get("k_overdrive", 2.85)                     # 18: k_overdrive
        ]
        clamped_og = [np.clip(og_p_vals[i], bounds[i][0], bounds[i][1]) for i in range(n_params)]
        og_opt_vals = to_optimizer_params(clamped_og)
        init_pop[1, :] = og_opt_vals

        # 3. Slots 2..4: Exploratory variations around the fresh middle-ground baseline
        for elite_i in range(2, min(5, pop_size_total)):
            jitter = np.random.normal(0.0, 0.015, size=n_params)
            init_pop[elite_i, :] = np.clip(fresh_opt_vals + jitter, [lo for lo, hi in optimizer_bounds], [hi for lo, hi in optimizer_bounds])

    best_written_cost = [1e9]
    try:
        print("  [Seed Baseline] Evaluating initial seed candidate on Host...", flush=True)
        initial_seed_cost = cost_function(default_opt_vals)
        best_written_cost[0] = initial_seed_cost
        print(f"  [Seed Baseline] Seed candidate initial evaluation MSE: {initial_seed_cost:.6f}", flush=True)
    except Exception as e:
        print(f"  [Seed Baseline Warning] Failed to evaluate seed candidate on Host: {e}", flush=True)
    
    figures_dir = os.path.join(fitting_dir, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    def generate_fit_comparison_plot(cand_x, tag="latest"):
        try:
            cand_params = get_dict_params(cand_x, base_params)
            active_sweeps = []
            if fit_ns: active_sweeps.append("NS")
            if fit_sat: active_sweeps.append("SAT")
            if fit_122: active_sweeps.append("122V")
            n_cols = len(active_sweeps)
            if n_cols == 0:
                return

            fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 12))
            ax_cond_list = [axes[0]] if n_cols == 1 else [axes[0, i] for i in range(n_cols)]
            ax_temp_list = [axes[1]] if n_cols == 1 else [axes[1, i] for i in range(n_cols)]
            curr_col = 0

            if fit_ns:
                ax1 = ax_cond_list[curr_col]
                ax4 = ax_temp_list[curr_col]
                curr_col += 1
                sim_ns = MolmemSimulator(device='cpu', backend='numba')
                sim_ns.detailedPrint = False
                candidate_ns = cand_params.copy()
                candidate_ns["nInit"] = 0.0
                sim_ns.addCrossbarMatrix(device_type=candidate_ns, multiThread=False, rows=1, cols=1)
                sim_ns.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_ns.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_ns = np.max(pulses_exp_ns)
                t_switch_ns = (max_p_ns / 2.0) * t_period
                t_stop_ns = max_p_ns * t_period
                def v_pulse_train_ns(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_ns
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
                edge_points_ns = [t_switch_ns]
                sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns)
                sim_ns.addDcSource(col=1, voltage=0.0)
                sim_ns.run(tEnd=t_stop_ns, min_dt=1e-11, tol=1e-3, title=None, recordHistory=['g', 'state', 'f22', 'temp'])
                g_sim_best_ns = sim_ns.getConductance(row=1, col=1) * 1e3
                t_arr_ns = sim_ns.getTime()
                sim_pulses_ns = t_arr_ns / t_period
                T_sim_best_ns = sim_ns.getTemperature(row=1, col=1)

                ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax1.scatter(raw_pulses_exp_ns, raw_cond_exp_ns_ms, color="#d9b310", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax1.plot(pulses_exp_ns, cond_exp_ns_ms, color="#d9b310", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_ns_arr = np.arange(1, max_p_ns + 1)
                g_sim_best_ns_read = np.interp(pulses_dense_ns_arr - 0.05, sim_pulses_ns, g_sim_best_ns)
                ax1.plot(pulses_dense_ns_arr, g_sim_best_ns_read, color="#328cc1", linewidth=3, label="Fitted Simulation Model")
                ax1.set_title(f"Non-Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax1.set_ylabel("Conductance (mS)", fontsize=10)
                ax1.legend(frameon=True, fontsize=9)

                ax4.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax4.plot(np.linspace(1, max_p_ns, len(T_sim_best_ns)), T_sim_best_ns, color="#d9b310", linewidth=2.5, label="Local Temp")
                ax4.set_title("Non-Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax4.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax4.set_ylabel("Temperature (K)", fontsize=10)
                ax4.legend(frameon=True, fontsize=9)

            if fit_sat:
                ax2 = ax_cond_list[curr_col]
                ax5 = ax_temp_list[curr_col]
                curr_col += 1
                sim_sat = MolmemSimulator(device='cpu', backend='numba')
                sim_sat.detailedPrint = False
                candidate_sat = cand_params.copy()
                candidate_sat["nInit"] = 0.0
                sim_sat.addCrossbarMatrix(device_type=candidate_sat, multiThread=False, rows=1, cols=1)
                sim_sat.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_sat.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_sat = np.max(pulses_exp_sat)
                t_switch_sat = (max_p_sat / 2.0) * t_period
                t_stop_sat = max_p_sat * t_period
                def v_pulse_train_sat(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_sat
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
                edge_points_sat = [t_switch_sat]
                sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat)
                sim_sat.addDcSource(col=1, voltage=0.0)
                sim_sat.run(tEnd=t_stop_sat, min_dt=1e-11, tol=1e-3, title=None, recordHistory=['g', 'state', 'f22', 'temp'])
                g_sim_best_sat = sim_sat.getConductance(row=1, col=1) * 1e3
                t_arr_sat = sim_sat.getTime()
                sim_pulses_sat = t_arr_sat / t_period
                T_sim_best_sat = sim_sat.getTemperature(row=1, col=1)

                ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax2.scatter(raw_pulses_exp_sat, raw_cond_exp_sat_ms, color="#ff5a5f", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax2.plot(pulses_exp_sat, cond_exp_sat_ms, color="#ff5a5f", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_sat_arr = np.arange(1, max_p_sat + 1)
                g_sim_best_sat_read = np.interp(pulses_dense_sat_arr - 0.05, sim_pulses_sat, g_sim_best_sat)
                ax2.plot(pulses_dense_sat_arr, g_sim_best_sat_read, color="#0b3c5d", linewidth=3, label="Fitted Simulation Model")
                ax2.set_title(f"Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax2.set_ylabel("Conductance (mS)", fontsize=10)
                ax2.legend(frameon=True, fontsize=9)

                ax5.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax5.plot(np.linspace(1, max_p_sat, len(T_sim_best_sat)), T_sim_best_sat, color="#ff5a5f", linewidth=2.5, label="Local Temp")
                ax5.set_title("Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax5.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax5.set_ylabel("Temperature (K)", fontsize=10)
                ax5.legend(frameon=True, fontsize=9)

            if fit_122:
                ax3 = ax_cond_list[curr_col]
                ax6 = ax_temp_list[curr_col]
                curr_col += 1
                sim_122 = MolmemSimulator(device='cpu', backend='numba')
                sim_122.detailedPrint = False
                candidate_122 = cand_params.copy()
                candidate_122["nInit"] = 0.0
                sim_122.addCrossbarMatrix(device_type=candidate_122, multiThread=False, rows=1, cols=1)
                sim_122.resetDeviceStates(n=0.0, modeState=1.0, scaleFactor=1.0, T=298.15)
                sim_122.devices[0]['dev']._forceUpdateConductanceParams()
                max_p_122 = np.max(pulses_exp_122)
                t_switch_122 = (max_p_122 / 2.0) * t_period
                t_stop_122 = max_p_122 * t_period
                vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
                vdep_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
                def v_pulse_train_122(t, sources):
                    if isinstance(t, np.ndarray):
                        out = np.zeros_like(t)
                        mask = t < t_switch_122
                        if np.any(mask): out[mask] = sources[0].getVoltage(t[mask])
                        if np.any(~mask): out[~mask] = sources[1].getVoltage(t[~mask])
                        return out
                    return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
                edge_points_122 = [t_switch_122]
                sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep_122], edge_points=edge_points_122)
                sim_122.addDcSource(col=1, voltage=0.0)
                sim_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=1e-3, title=None, recordHistory=['g', 'state', 'f22', 'temp'])
                g_sim_best_122 = sim_122.getConductance(row=1, col=1) * 1e3
                t_arr_122 = sim_122.getTime()
                sim_pulses_122 = t_arr_122 / t_period
                T_sim_best_122 = sim_122.getTemperature(row=1, col=1)

                ax3.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax3.scatter(raw_pulses_exp_122, raw_cond_exp_122_ms, color="#10b981", alpha=0.5, edgecolors="k", s=25, label="Empirical Data")
                ax3.plot(pulses_exp_122, cond_exp_122_ms, color="#10b981", linewidth=2.0, linestyle="--", alpha=0.85, label="Empirical Fit (PCHIP)")
                pulses_dense_122_arr = np.arange(1, max_p_122 + 1)
                g_sim_best_122_read = np.interp(pulses_dense_122_arr - 0.05, sim_pulses_122, g_sim_best_122)
                ax3.plot(pulses_dense_122_arr, g_sim_best_122_read, color="#1e3a8a", linewidth=3, label="Fitted Simulation Model")
                ax3.set_title(f"1.22V Saturating Sweep ({device_name}) [{tag}]", fontsize=11, fontweight="bold")
                ax3.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax3.set_ylabel("Conductance (mS)", fontsize=10)
                ax3.legend(frameon=True, fontsize=9)

                ax6.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
                ax6.plot(np.linspace(1, max_p_122, len(T_sim_best_122)), T_sim_best_122, color="#10b981", linewidth=2.5, label="Local Temp")
                ax6.set_title("1.22V Saturating Sweep Local Temp", fontsize=11, fontweight="bold")
                ax6.set_xlabel("Pulse Counts (n)", fontsize=10)
                ax6.set_ylabel("Temperature (K)", fontsize=10)
                ax6.legend(frameon=True, fontsize=9)

            plt.tight_layout()
            out_file = os.path.join(figures_dir, f"{device_name}_fit_comparison_{tag}.png")
            plt.savefig(out_file, dpi=200)
            plt.close(fig)
            print_log(f"  [Live Plot] Saved updated fitting comparison plot to {os.path.basename(out_file)}")
        except Exception as pe:
            print_log(f"  [Live Plot Notice] Could not render live plot: {pe}")

    def de_callback(xk, nit=None, *args, **kwargs):
        cost = cost_function(xk)
        is_new_best = False
        if cost < best_written_cost[0]:
            best_written_cost[0] = cost
            is_new_best = True
            save_preset_txt(xk, base_params, txt_path, func_name, device_name, data_dir)
            print_log(f"  [Incremental Save] Updated best parameters to {os.path.basename(txt_path)} (Cost: {cost:.6f})")
            generate_fit_comparison_plot(xk, tag="latest")
        if nit is not None:
            generate_fit_comparison_plot(xk, tag=f"gen_{nit:03d}")
            try:
                from molmem_lib.sys_utils import cleanup_temp_history_dir
                cleanup_temp_history_dir(force_all=True)
            except Exception:
                pass

    # Render baseline comparison plot for the initial seed model
    generate_fit_comparison_plot(default_opt_vals, tag="initial_baseline")
    generate_fit_comparison_plot(default_opt_vals, tag="latest")

    # Broadcast setup payload to all cluster nodes now that Master is fully initialized
    print("\nBroadcasting simulation setup payload to all cluster nodes...")
    dispatcher.broadcast_setup_payload(setup_payload)

    # -------------------------------------------------------------------------
    # Clustered Differential Evolution Execution
    # -------------------------------------------------------------------------
    print("\nRunning Clustered Differential Evolution (Dynamic Work Streaming & Task Recycling)...")
    solver = DecayingDESolver(
        cost_function,
        bounds=optimizer_bounds,
        args=(),
        popsize=POP_SIZE,
        maxiter=MAX_ITER,
        tol=TOL,
        disp=False,
        updating='deferred',
        init=init_pop,
        mutation=(0.01, 1.0),
        polish=False,
        callback=de_callback,
        cluster_dispatcher=dispatcher
    )
    
    res_de = solver.solve()
    print("\nDifferential Evolution finished. Best candidate parameters:")
    print(to_physical_params(res_de.x))
    try:
        from molmem_lib.sys_utils import cleanup_temp_history_dir
        cleanup_temp_history_dir(force_all=True)
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Clustered Multi-Basin L-BFGS-B Polish
    # -------------------------------------------------------------------------
    print("\nExtracting top unique candidates from Differential Evolution population...")
    unnormalized_population = np.zeros_like(solver.population)
    for col_idx, (lo, hi) in enumerate(optimizer_bounds):
        unnormalized_population[:, col_idx] = lo + solver.population[:, col_idx] * (hi - lo)

    candidates = []
    candidate_energies = []
    seen_norm = []
    best_de_cost = res_de.fun
    
    for idx_p, x_cand_norm in enumerate(solver.population):
        cand_cost = solver.population_energies[idx_p]
        if cand_cost > best_de_cost * 1.15:
            continue
        is_unique = True
        for s_norm in seen_norm:
            if np.max(np.abs(x_cand_norm - s_norm)) < 0.15:
                is_unique = False
                break
        if is_unique:
            candidates.append(unnormalized_population[idx_p])
            candidate_energies.append(cand_cost)
            seen_norm.append(x_cand_norm)
            if len(candidates) >= 5:
                break
                
    print(f"Identified {len(candidates)} unique starting basins for local fine-tuning.")
    
    tol_shared.value = 5e-4
    best_cost_shared.value = 1e9
    print("Switching to tight tolerance (tol=5e-4) for L-BFGS-B parallel Jacobian minimization...")
    
    best_overall_cost_ref = [best_de_cost]
    best_overall_x = res_de.x
    
    cluster_jac = make_cluster_parallel_jacobian(dispatcher, cost_function, bounds=optimizer_bounds)
    
    for idx, (x0_cand, start_energy) in enumerate(zip(candidates, candidate_energies)):
        print_log(f"\n[Basin {idx + 1}/{len(candidates)}] Local minimization starting cost: {start_energy:.6f}...")
        tracker = LocalMinimizationTracker(start_energy, best_overall_cost_ref, de_callback, x0_cand, idx + 1, len(candidates))
        try:
            res_local = minimize(
                cost_function,
                x0=x0_cand,
                args=(),
                bounds=optimizer_bounds,
                method='L-BFGS-B',
                jac=cluster_jac,
                options={'maxiter': LBFGS_MAX_ITER, 'disp': False},
                callback=tracker
            )
            cand_cost = res_local.fun
            best_x_candidate = res_local.x
        except StopIteration:
            cand_cost = tracker.best_cost
            best_x_candidate = tracker.best_x
            print_log(f"[Basin {idx + 1}/{len(candidates)}] Stopped early due to stagnant convergence.")
            
        print_log(f"[Basin {idx + 1}/{len(candidates)}] Tuning complete. Final cost: {cand_cost:.6f}")
        if cand_cost < best_overall_cost_ref[0]:
            best_overall_cost_ref[0] = cand_cost
            best_overall_x = best_x_candidate
            de_callback(best_overall_x)
            
        try:
            from molmem_lib.sys_utils import cleanup_temp_history_dir
            cleanup_temp_history_dir(force_all=True)
        except Exception:
            pass
            
    clear_dashboard()
    best_x = best_overall_x
    best_cost = best_overall_cost_ref[0]
    print(f"\nCluster Optimization Completed. Best cost (MSE) = {best_cost:.6f}")
    
    final_params = get_dict_params(best_x, base_params)
    save_preset_txt(best_x, base_params, txt_path, func_name, device_name, data_dir)
    print(f"Final optimized preset saved to: {txt_path}")

    # -------------------------------------------------------------------------
    # Verification Plots
    # -------------------------------------------------------------------------
    print("Generating final verification plots...")
    generate_fit_comparison_plot(best_x, tag=f"subset_{selection_str}")

    # Clean shutdown
    beacon_broadcaster.stop()
    dispatcher.shutdown_cluster()
    if dispatcher.local_pool is not None:
        safe_close_pool(dispatcher.local_pool)
    print("Master coordinator finished successfully.")


if __name__ == "__main__" and not globals().get("_IN_OTA_RELOAD", False) and not is_multiprocessing_worker():
    init_windows_console()
    main()
