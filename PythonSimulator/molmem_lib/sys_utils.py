import os
import sys
import shutil
import warnings

# Suppress PyTorch TorchScript deprecation warnings (torch.jit.load is deprecated. Please switch to torch.export.)
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*`?torch\.jit\.load`? is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*torch\.jit\._serialization.*")
warnings.filterwarnings("ignore", message=r".*`?torch\.jit\.load`? is deprecated.*")

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))

class DummyStream:
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass

if sys.stdout is None:
    sys.stdout = DummyStream()
if getattr(sys, '__stdout__', None) is None:
    sys.__stdout__ = DummyStream()

_MOLMEM_LIB_SILENCED = (os.environ.get("MOLMEM_SILENT") == "1")
_filters_installed = False
_orig_sys_stdout = None
_orig_sys_stdout_dunder = None
_orig_sys_stderr = None
_orig_sys_stderr_dunder = None

class OriginFilteredStream:
    """
    A proxy stream that inspects the calling stack frame and silences console output
    originating from within molmem_lib when silenceMolmemLib() is active,
    while allowing all user-script prints to pass through unhindered.
    """
    def __init__(self, target_stream, lib_dir):
        self._target = target_stream
        self._lib_dir = os.path.normcase(os.path.abspath(lib_dir)) + os.sep

    def _is_from_molmem_lib(self):
        try:
            f = sys._getframe(1)
            f = f.f_back
            while f:
                co_file = f.f_code.co_filename
                if co_file:
                    norm_path = os.path.normcase(os.path.abspath(co_file))
                    if norm_path.startswith(self._lib_dir):
                        return True
                    # If frame is user application code (not python stdlib internals), then caller is external
                    if not any(marker in norm_path for marker in [os.sep + 'lib' + os.sep, 'importlib', 'logging']):
                        return False
                f = f.f_back
        except Exception:
            pass
        return False

    def write(self, s):
        if _MOLMEM_LIB_SILENCED:
            # 1. Never silence sys.stderr under any circumstances
            is_stderr = (self._target is _orig_sys_stderr or self._target is _orig_sys_stderr_dunder)
            if is_stderr:
                pass
            # 2. Never silence if an exception is actively being handled
            elif sys.exc_info()[0] is not None:
                pass
            # 3. Never silence error, crash, warning, or debug markers
            elif any(m in str(s) for m in [
                "Traceback", "Error", "ERROR", "Exception", "Crash", "CRASH",
                "Warning", "WARNING", "[!]", "Debug", "DEBUG", "OOM",
                "Diagnostic Recommendations", "Detailed Traceback", "Failed", "FAILED",
                "Compiling JIT physics kernels"
            ]):
                pass
            # 4. Silence only routine internal library operational prints
            elif self._is_from_molmem_lib():
                return len(s) if isinstance(s, (str, bytes)) else 0
        target = self._target
        if target is None:
            return 0
        try:
            return target.write(s)
        except Exception:
            return 0

    def flush(self):
        target = self._target
        if target is not None:
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self):
        target = self._target
        if target is not None:
            try:
                return getattr(target, 'isatty', lambda: False)()
            except Exception:
                pass
        return False

    def fileno(self):
        target = self._target
        if target is not None and hasattr(target, 'fileno'):
            return target.fileno()
        raise OSError("fileno not supported")

    def __getattr__(self, attr):
        return getattr(self._target, attr)

def _install_stream_filters():
    global _filters_installed, _orig_sys_stdout, _orig_sys_stdout_dunder, _orig_sys_stderr, _orig_sys_stderr_dunder
    if _filters_installed:
        return
    _orig_sys_stdout = sys.stdout if sys.stdout is not None else DummyStream()
    _orig_sys_stdout_dunder = sys.__stdout__ if sys.__stdout__ is not None else DummyStream()
    _orig_sys_stderr = sys.stderr if sys.stderr is not None else DummyStream()
    _orig_sys_stderr_dunder = sys.__stderr__ if sys.__stderr__ is not None else DummyStream()

    sys.stdout = OriginFilteredStream(_orig_sys_stdout, _LIB_DIR)
    sys.__stdout__ = OriginFilteredStream(_orig_sys_stdout_dunder, _LIB_DIR)
    sys.stderr = OriginFilteredStream(_orig_sys_stderr, _LIB_DIR)
    sys.__stderr__ = OriginFilteredStream(_orig_sys_stderr_dunder, _LIB_DIR)
    _filters_installed = True

def _uninstall_stream_filters():
    global _filters_installed
    if not _filters_installed:
        return
    if _orig_sys_stdout is not None:
        sys.stdout = _orig_sys_stdout
    if _orig_sys_stdout_dunder is not None:
        sys.__stdout__ = _orig_sys_stdout_dunder
    if _orig_sys_stderr is not None:
        sys.stderr = _orig_sys_stderr
    if _orig_sys_stderr_dunder is not None:
        sys.__stderr__ = _orig_sys_stderr_dunder
    _filters_installed = False

class SilenceContext:
    def __init__(self, silent=True):
        self.silent = silent
        self.prev = False

    def __enter__(self):
        self.prev = isMolmemLibSilenced()
        silenceMolmemLib(self.silent)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        silenceMolmemLib(self.prev)

def isMolmemLibSilenced():
    """Returns True if molmem_lib console prints are currently silenced."""
    return _MOLMEM_LIB_SILENCED

def is_molmem_lib_silenced():
    """Alias for isMolmemLibSilenced."""
    return _MOLMEM_LIB_SILENCED

def silenceMolmemLib(silent=True):
    """
    Silences all console prints, status messages, progress meters, and startup
    banners originating from within molmem_lib, allowing calling scripts to run in a
    clean, headless mode where only the caller's prints are displayed.
    
    Can be used as a direct API call:
        import molmem_lib
        molmem_lib.silenceMolmemLib()
        
    Or as a scoped context manager:
        with molmem_lib.silenceMolmemLib():
            sim.passInput(...)
            
    To re-enable library prints:
        molmem_lib.silenceMolmemLib(False)
    """
    global _MOLMEM_LIB_SILENCED
    prev = _MOLMEM_LIB_SILENCED
    _MOLMEM_LIB_SILENCED = bool(silent)
    if _MOLMEM_LIB_SILENCED:
        os.environ["MOLMEM_SILENT"] = "1"
        _install_stream_filters()
    else:
        os.environ.pop("MOLMEM_SILENT", None)
        _uninstall_stream_filters()
    ctx = SilenceContext(silent)
    ctx.prev = prev
    return ctx

def silence_molmem_lib(silent=True):
    """Alias for silenceMolmemLib."""
    return silenceMolmemLib(silent=silent)

MolmemSilence = silenceMolmemLib

def unsilence_molmem_lib():
    """
    Restores console prints, status messages, progress meters, and simulator
    telemetry originating from within molmem_lib.
    """
    return silenceMolmemLib(False)

def unsilenceMolmemLib():
    """Alias for unsilence_molmem_lib."""
    return silenceMolmemLib(False)

if _MOLMEM_LIB_SILENCED:
    _install_stream_filters()

def is_worker_process():
    """
    Returns True if the current execution context is a multiprocessing child, 
    subprocess worker, or spawned calibration worker.
    """
    import multiprocessing
    if hasattr(multiprocessing, 'parent_process') and multiprocessing.parent_process() is not None:
        return True
    if multiprocessing.current_process().name != 'MainProcess':
        return True
    worker_env_vars = (
        "MOLMEM_MP_CHILD",
        "MOLMEM_IN_SUBPROCESS",
        "MOLMEM_IS_CALIBRATION_WORKER",
    )
    if any(os.environ.get(v) == "1" for v in worker_env_vars):
        return True
    worker_cli_sentinels = (
        'parent_sentinel',
        '--multiprocessing-fork',
        'spawn_main',
        'pipe_handle',
        '--profile-worker',
        '--calibration-worker',
        'multiprocessing'
    )
    for arg in sys.argv:
        for sentinel in worker_cli_sentinels:
            if sentinel in arg:
                return True
    return False

# Windows WaitForMultipleObjects API ceiling is 64 handles (MAXIMUM_WAIT_OBJECTS).
# Python's multiprocessing.Pool reserves 3 internal handles for queue monitoring and signalling,
# restricting the maximum safe process pool worker count to 61 on Windows.
MAX_PROCESS_POOL_WORKERS = 61

def get_safe_process_count(requested=None):
    """
    Returns the maximum safe process pool worker count.
    Guarantees worker count never exceeds 61 on Windows to prevent
    'ValueError: cannot watch more than 61 handles' from WaitForMultipleObjects.
    """
    count = requested if requested is not None else (os.cpu_count() or 1)
    if sys.platform == 'win32':
        return min(count, MAX_PROCESS_POOL_WORKERS)
    return count


import contextlib

@contextlib.contextmanager
def suppress_c_stderr():
    """
    Context manager to temporarily suppress low-level C/C++ runtime stderr output
    (e.g., TensorFlow CUDA driver probe warnings, ABSL initialization messages)
    by redirecting file descriptor 2 (stderr) to os.devnull.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        stderr_fd = 2

    sys.stderr.flush()
    saved_fd = None
    null_fd = None
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(stderr_fd)
        os.dup2(null_fd, stderr_fd)
    except Exception:
        pass

    try:
        yield
    finally:
        sys.stderr.flush()
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
            except Exception:
                pass
        if null_fd is not None:
            try:
                os.close(null_fd)
            except Exception:
                pass



def is_interactive_notebook():
    """Detects interactive notebook environments (Jupyter, Google Colab, Kaggle, VS Code Notebook)."""
    if is_worker_process():
        return False
    if 'IPython' not in sys.modules and 'ipykernel' not in sys.modules:
        if not any(k in os.environ for k in ['COLAB_GPU', 'KAGGLE_KERNEL_RUN_TYPE', 'IPYKERNEL_CELL_NAME']):
            return False
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            shell = ipy.__class__.__name__
            # Only true interactive notebook environments (ZMQInteractiveShell for Jupyter, Shell for Colab)
            # Note: TerminalInteractiveShell is IPython in a terminal/CLI, NOT an interactive notebook!
            if shell in ('ZMQInteractiveShell', 'Shell'):
                return True
    except Exception:
        pass
    try:
        shell = get_ipython().__class__.__name__
        if shell in ('ZMQInteractiveShell', 'Shell'):
            return True
    except Exception:
        pass
    if any(k in os.environ for k in ['COLAB_GPU', 'KAGGLE_KERNEL_RUN_TYPE', 'IPYKERNEL_CELL_NAME']):
        return True
    return False


def disable_terminal_flow_control():
    """
    Disables software and hardware terminal flow control (XON/XOFF, IXON, IXOFF)
    on Linux/macOS POSIX TTYs to prevent accidental freezes from Ctrl+S, Ctrl+Q,
    or other flow control signals. Restores original terminal settings upon exit.
    """
    if sys.platform != 'win32':
        try:
            import termios
            import atexit
            
            tty_fd = None
            for stream in (sys.stdin, sys.stdout, sys.stderr):
                if stream is not None and hasattr(stream, 'isatty') and stream.isatty():
                    try:
                        tty_fd = stream.fileno()
                        break
                    except Exception:
                        continue
                        
            if tty_fd is not None:
                orig_attrs = termios.tcgetattr(tty_fd)
                attrs = termios.tcgetattr(tty_fd)
                
                # Disable software flow control:
                # IXON: Start/stop output control (Ctrl+S / Ctrl+Q)
                # IXOFF: Start/stop input control
                # IXANY: Restart output on any character
                flow_mask = getattr(termios, 'IXON', 0) | getattr(termios, 'IXOFF', 0)
                if hasattr(termios, 'IXANY'):
                    flow_mask |= termios.IXANY
                attrs[0] &= ~flow_mask
                
                # Disable hardware flow control (CRTSCTS) if present
                if hasattr(termios, 'CRTSCTS'):
                    attrs[2] &= ~termios.CRTSCTS
                
                termios.tcsetattr(tty_fd, termios.TCSANOW, attrs)
                
                def _restore_termios():
                    try:
                        termios.tcsetattr(tty_fd, termios.TCSANOW, orig_attrs)
                    except Exception:
                        pass
                atexit.register(_restore_termios)
        except Exception:
            pass


def configure_default_threading():
    """
    Uniformly configures environment variables and thread budgets across 
    NumExpr, Numba, and PyTorch (80% of physical cores, TBB thread-safe threading layer).
    Also disables terminal flow control to prevent interactive console freezes.
    """
    disable_terminal_flow_control()
    if is_worker_process():
        return
    cpu_cnt = os.cpu_count() or 1
    max_cores_str = str(max(1, int(cpu_cnt * 0.8)))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", max_cores_str)
    os.environ.setdefault("NUMBA_NUM_THREADS", max_cores_str)
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    tbb_functional = False
    candidate_path = None
    if sys.platform == 'win32':
        _env_lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
        if os.path.isdir(_env_lib_bin):
            try:
                os.add_dll_directory(_env_lib_bin)
            except Exception:
                pass
            if _env_lib_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _env_lib_bin + os.pathsep + os.environ.get("PATH", "")
        
        # Check candidate locations for tbb12.dll
        for candidate in [
            os.path.join(sys.prefix, 'Library', 'bin', 'tbb12.dll'),
            os.path.join(sys.prefix, 'bin', 'tbb12.dll'),
            os.path.join(sys.prefix, 'Scripts', 'tbb12.dll'),
        ]:
            if os.path.isfile(candidate):
                candidate_path = candidate
                try:
                    import ctypes
                    ctypes.CDLL(candidate_path)
                    tbb_functional = True
                except Exception:
                    pass
                if tbb_functional:
                    break

        if not tbb_functional:
            try:
                import ctypes
                ctypes.CDLL('tbb12.dll')
                tbb_functional = True
            except Exception:
                pass
            if not tbb_functional:
                try:
                    import importlib.util
                    if importlib.util.find_spec('tbb'):
                        tbb_functional = True
                except Exception:
                    pass
    else:
        # On Linux/Unix/macOS, dynamically preload libtbb into global symbol table
        search_dirs = []
        for p in [os.path.join(sys.prefix, 'lib'), os.path.join(sys.prefix, 'lib64')]:
            if os.path.isdir(p):
                search_dirs.append(p)
        try:
            import importlib.util
            spec = importlib.util.find_spec('tbb')
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    search_dirs.append(loc)
        except Exception:
            pass

        if sys.platform == 'darwin':
            tbb_targets = ['libtbb.12.dylib', 'libtbb.dylib', 'libtbb.so.12', 'libtbb.so']
        else:
            tbb_targets = ['libtbb.so.12', 'libtbb.so']

        for sdir in search_dirs:
            if not os.path.isdir(sdir):
                continue
            for target_name in tbb_targets:
                candidate = os.path.join(sdir, target_name)
                if os.path.isfile(candidate):
                    try:
                        import ctypes
                        ctypes.CDLL(candidate, mode=getattr(ctypes, 'RTLD_GLOBAL', None))
                        tbb_functional = True
                        candidate_path = candidate
                        break
                    except Exception:
                        pass
            if tbb_functional:
                break

    if tbb_functional:
        # 1. Patch Numba's internal TBB version validator to accept candidate full path if dlopen(name) fails
        try:
            import numba.np.ufunc.parallel as _nb_parallel
            _orig_check = getattr(_nb_parallel, '_check_tbb_version_compatible', None)
            def _safe_tbb_check():
                try:
                    if _orig_check is not None:
                        return _orig_check()
                except (ImportError, OSError, ValueError):
                    pass
                if candidate_path and os.path.isfile(candidate_path):
                    import ctypes
                    libtbb = ctypes.CDLL(candidate_path)
                    vfunc = getattr(libtbb, 'TBB_runtime_interface_version', None)
                    if vfunc is not None:
                        vfunc.argtypes = []
                        vfunc.restype = ctypes.c_int
                        if vfunc() >= 12060:
                            return
                raise ImportError("TBB shared library version >= 12060 required")
            _nb_parallel._check_tbb_version_compatible = _safe_tbb_check
        except Exception:
            pass

        # 2. Set environment and Numba config to TBB prior to any compilation
        os.environ["NUMBA_THREADING_LAYER"] = "tbb"
        try:
            import numba
            numba.config.THREADING_LAYER = "tbb"
        except Exception:
            pass

        # 3. Test that Numba can actually bind TBB and report active layer
        try:
            from numba.np.ufunc.parallel import _launch_threads, threading_layer
            _launch_threads()
            active = threading_layer()
            if active != "tbb":
                tbb_functional = False
        except Exception:
            tbb_functional = False

    if not tbb_functional:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
        try:
            import numba
            numba.config.THREADING_LAYER = "workqueue"
        except Exception:
            pass


def set_worker_numba_threads(num_threads=1):
    """Safely synchronizes worker thread limits across os.environ and Numba internal config."""
    num_str = str(num_threads)
    os.environ["NUMBA_NUM_THREADS"] = num_str
    os.environ["OMP_NUM_THREADS"] = num_str
    os.environ["MKL_NUM_THREADS"] = num_str
    try:
        import numba.core.config as nb_config
        nb_config.NUMBA_NUM_THREADS = int(num_threads)
    except Exception:
        pass


def get_numba_threading_layer():
    """Returns the name of Numba's currently active threading layer backend."""
    try:
        from numba.np.ufunc.parallel import threading_layer
        return threading_layer()
    except Exception:
        pass
    try:
        import numba
        layer = getattr(numba.config, 'THREADING_LAYER', None)
        if layer and str(layer).lower() not in ['default', 'unknown']:
            return str(layer).lower()
    except Exception:
        pass
    return os.environ.get("NUMBA_THREADING_LAYER", "unknown")


def is_logging_enabled():
    """
    Returns True if disk logging is enabled via ENABLE_LOGGING or CLI/environment flag.
    Default is False (zero disk footprint unless -logging is passed).
    """
    try:
        mod = sys.modules.get("molmem_lib")
        if mod is not None and getattr(mod, 'ENABLE_LOGGING', False):
            return True
        import molmem_lib
        if getattr(molmem_lib, 'ENABLE_LOGGING', False):
            return True
    except Exception:
        pass
    return os.environ.get("MOLMEM_ENABLE_LOGGING") == "1"


_LOG_DIR_CREATED = False

def get_log_dir(auto_create=True):
    """
    Returns the centralized self-healing log directory 'molmem_lib/logs'.
    Only creates the directory on disk if logging is active and auto_create=True.
    """
    global _LOG_DIR_CREATED
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(lib_dir, "logs")
    if auto_create and is_logging_enabled() and not _LOG_DIR_CREATED:
        try:
            os.makedirs(log_dir, exist_ok=True)
            _LOG_DIR_CREATED = True
        except Exception:
            pass
    return log_dir

_TEMP_HISTORY_DIR_CREATED = False

def get_temp_history_dir(auto_create=True):
    """
    Returns the centralized temporary binary history directory 'molmem_lib/.temp_history'.
    """
    global _TEMP_HISTORY_DIR_CREATED
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(lib_dir, ".temp_history")
    if auto_create and not _TEMP_HISTORY_DIR_CREATED:
        try:
            os.makedirs(temp_dir, exist_ok=True)
            _TEMP_HISTORY_DIR_CREATED = True
        except Exception:
            pass
    return temp_dir

def get_calcache_dir(auto_create=True):
    """
    Returns the centralized calibration and profile cache directory 'molmem_lib/.calcache'.
    """
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    calcache_dir = os.path.join(lib_dir, ".calcache")
    if auto_create and not os.path.isdir(calcache_dir):
        try:
            os.makedirs(calcache_dir, exist_ok=True)
        except Exception:
            pass
    return calcache_dir

def cleanup_temp_history_dir(target_sim_id=None, target_pid=None, force_all=False, allow_current_pid=False):
    """
    Safely cleans up transient history memory-mapped binary files from .temp_history/.
    - If target_sim_id is specified: only cleans up files for that simulator instance.
    - If target_pid is specified: only cleans up files matching that process ID.
    - If force_all=True: sweeps all orphaned files whose owning PID is no longer alive.
    - If allow_current_pid=True: sweeps files matching current_pid (used during library startup or pre-OTA restart).
    """
    temp_dir = get_temp_history_dir(auto_create=False)
    if not os.path.isdir(temp_dir):
        return
    import time
    current_pid = os.getpid()
    
    def _is_pid_alive(pid_val, file_mtime=None):
        if pid_val == current_pid:
            return not allow_current_pid
        try:
            if sys.platform == 'win32':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                SYNCHRONIZE = 0x00100000
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid_val)
                if handle:
                    try:
                        exit_code = ctypes.c_ulong()
                        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                            if exit_code.value != 259:
                                return False
                            # Process is active; verify if it is actually a Python simulation process
                            # (Guards against OS PID recycling by applications like Antigravity IDE, ShellHost, svchost, etc.)
                            buf = ctypes.create_unicode_buffer(1024)
                            size = ctypes.c_ulong(1024)
                            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                                exe_name = buf.value.lower()
                                if 'python' not in exe_name:
                                    return False
                            return True
                        return False
                    finally:
                        kernel32.CloseHandle(handle)
                return False
            else:
                # Linux / Unix container & Google Compute environment
                proc_dir = f"/proc/{pid_val}"
                if not os.path.isdir(proc_dir):
                    return False
                # Verify that it is actually a Python process (guards against PID recycling by unrelated OS daemons)
                try:
                    if file_mtime is not None:
                        # In Linux, st_mtime of /proc/{pid} reflects process start time
                        proc_stat = os.stat(proc_dir)
                        if file_mtime < proc_stat.st_mtime - 1.0:
                            # File was modified before this process was created; PID was recycled by the OS
                            return False

                    with open(f"{proc_dir}/cmdline", "rb") as cf:
                        cmd = cf.read().lower()
                        if b"python" in cmd or b"molmem" in cmd or b"fit_device" in cmd:
                            return True
                    return False
                except Exception:
                    return False
        except Exception:
            return False

    try:
        now = time.time()
        for fname in os.listdir(temp_dir):
            if not fname.endswith('.dat'):
                continue
            fpath = os.path.join(temp_dir, fname)
            should_delete = False
            
            try:
                file_mtime = os.path.getmtime(fpath)
            except Exception:
                file_mtime = None

            # Extract PID from filename: hist_pid<PID>_<sim_id>_<ch>.dat
            file_pid = None
            for part in fname.split('_'):
                if part.startswith('pid') and part[3:].isdigit():
                    file_pid = int(part[3:])
                    break
            
            if target_sim_id is not None and f"_{target_sim_id}_" in fname:
                should_delete = True
            elif target_pid is not None and file_pid == target_pid:
                should_delete = True
            elif force_all:
                if file_pid is not None:
                    if not _is_pid_alive(file_pid, file_mtime=file_mtime):
                        should_delete = True
                else:
                    # Unparseable filename in .temp_history/ older than 60s is an orphan
                    if file_mtime is not None and (now - file_mtime) > 60.0:
                        should_delete = True
            
            if should_delete:
                try:
                    os.remove(fpath)
                except PermissionError:
                    try:
                        os.chmod(fpath, 0o777)
                        os.remove(fpath)
                    except Exception:
                        pass
                except Exception:
                    pass
        if force_all:
            dep_cache_f = os.path.join(temp_dir, "caller_dep_cache.json")
            if os.path.isfile(dep_cache_f):
                try:
                    os.remove(dep_cache_f)
                except Exception:
                    pass
    except Exception:
        pass


_CACHED_CPU_CACHE_BUDGET = None

def get_cpu_cache_budget_mb():
    """
    Returns (staging_chunk_mb, ping_pong_buffer_mb, total_cache_mb).
    Allocates 1/3 of detected L2 + L3 cache to the active staging chunk,
    and 2/3 to the host ping-pong double buffer.
    If detection is unavailable or <= 0, defaults to 24 MB total (8 MB staging, 16 MB ping-pong).
    """
    global _CACHED_CPU_CACHE_BUDGET
    if _CACHED_CPU_CACHE_BUDGET is not None:
        return _CACHED_CPU_CACHE_BUDGET

    l2_kb = 0
    l3_kb = 0
    try:
        if sys.platform == 'win32':
            import subprocess, json
            try:
                cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_CacheMemory | Select-Object Level,InstalledSize | ConvertTo-Json"'
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True, timeout=2.0).strip()
                if out:
                    data = json.loads(out)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        lvl = item.get('Level')
                        sz = item.get('InstalledSize', 0)
                        if lvl == 4:  # Win32 Level 4 = L2 Cache
                            l2_kb += sz
                        elif lvl == 5:  # Win32 Level 5 = L3 Cache
                            l3_kb += sz
            except Exception:
                pass
            if l2_kb == 0 and l3_kb == 0:
                try:
                    res = subprocess.check_output("wmic cpu get L2CacheSize, L3CacheSize /value", shell=True, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
                    for line in res.splitlines():
                        if '=' in line:
                            k, v = line.split('=', 1)
                            if k.strip() == "L2CacheSize" and v.strip().isdigit():
                                l2_kb = int(v.strip())
                            elif k.strip() == "L3CacheSize" and v.strip().isdigit():
                                l3_kb = int(v.strip())
                except Exception:
                    pass
        elif sys.platform == 'darwin':
            import subprocess
            try:
                l2_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.l2cachesize'], stderr=subprocess.DEVNULL).strip())
                l2_kb = l2_bytes // 1024
            except Exception:
                pass
            try:
                l3_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.l3cachesize'], stderr=subprocess.DEVNULL).strip())
                l3_kb = l3_bytes // 1024
            except Exception:
                pass
        elif sys.platform.startswith('linux'):
            import glob
            for path in glob.glob('/sys/devices/system/cpu/cpu*/cache/index*'):
                try:
                    with open(os.path.join(path, 'level'), 'r') as f:
                        lvl = int(f.read().strip())
                    with open(os.path.join(path, 'size'), 'r') as f:
                        s_str = f.read().strip()
                    val_kb = 0
                    if s_str.endswith(('K', 'k')):
                        val_kb = int(s_str[:-1])
                    elif s_str.endswith(('M', 'm')):
                        val_kb = int(s_str[:-1]) * 1024
                    elif s_str.isdigit():
                        val_kb = int(s_str) // 1024
                    if lvl == 2:
                        l2_kb = max(l2_kb, val_kb)
                    elif lvl == 3:
                        l3_kb = max(l3_kb, val_kb)
                except Exception:
                    pass
    except Exception:
        pass

    total_mb = (l2_kb + l3_kb) / 1024.0
    if total_mb >= 3.0:
        staging_mb = max(4.0, round(total_mb / 3.0, 2))
        ping_pong_mb = max(8.0, round((total_mb * 2.0) / 3.0, 2))
    else:
        # Fallback default: 24 MB total budget (8 MB staging, 16 MB ping-pong)
        staging_mb = 8.0
        ping_pong_mb = 16.0
        total_mb = 24.0

    _CACHED_CPU_CACHE_BUDGET = (staging_mb, ping_pong_mb, total_mb)
    return _CACHED_CPU_CACHE_BUDGET


_CACHED_ROCM_PATH = None
_CACHED_VCVARS64 = None
_CACHED_MSVC_BIN = None
_CACHED_WIN_SDK_BIN = None

def _get_short_path(path):
    if not path or sys.platform != 'win32':
        return path
    try:
        import ctypes
        from ctypes import wintypes
        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        _GetShortPathNameW.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        length = _GetShortPathNameW(path, buf, 1024)
        if length > 0 and os.path.exists(buf.value):
            return buf.value
    except Exception:
        pass
    if "Program Files (x86)" in path:
        cand = path.replace("Program Files (x86)", "PROGRA~2")
        if os.path.exists(cand):
            return cand
    if "Program Files" in path:
        cand = path.replace("Program Files", "PROGRA~1")
        if os.path.exists(cand):
            return cand
    return path

def _find_rocm_path():
    global _CACHED_ROCM_PATH
    if _CACHED_ROCM_PATH is not None:
        return _CACHED_ROCM_PATH
    for env_var in ["HIP_PATH", "ROCM_PATH", "HIP_DIR", "ROCM_HOME"]:
        path = os.environ.get(env_var)
        if path and os.path.exists(path):
            _CACHED_ROCM_PATH = _get_short_path(os.path.abspath(path))
            return _CACHED_ROCM_PATH
    if sys.platform != 'win32':
        linux_candidates = ["/opt/rocm"]
        try:
            if os.path.exists("/opt"):
                for d in sorted(os.listdir("/opt"), reverse=True):
                    if d.startswith("rocm") and os.path.isdir(os.path.join("/opt", d)):
                        cand = os.path.join("/opt", d)
                        if cand not in linux_candidates:
                            linux_candidates.append(cand)
        except Exception:
            pass
        for p in linux_candidates:
            if os.path.exists(p) and (os.path.exists(os.path.join(p, "bin", "hipcc")) or os.path.exists(os.path.join(p, "include"))):
                _CACHED_ROCM_PATH = os.path.abspath(p)
                return _CACHED_ROCM_PATH
        import shutil
        hipcc_p = shutil.which("hipcc")
        if hipcc_p:
            prefix = os.path.dirname(os.path.dirname(os.path.abspath(hipcc_p)))
            if os.path.exists(prefix) and prefix != "/":
                _CACHED_ROCM_PATH = prefix
                return _CACHED_ROCM_PATH
        for sys_cand in ["/usr", "/usr/local"]:
            if os.path.exists(os.path.join(sys_cand, "bin", "hipcc")) or os.path.exists(os.path.join(sys_cand, "include", "hip")):
                _CACHED_ROCM_PATH = sys_cand
                return _CACHED_ROCM_PATH
        _CACHED_ROCM_PATH = None
        return _CACHED_ROCM_PATH
    default_root = r"C:\PROGRA~1\AMD\ROCm" if os.path.exists(r"C:\PROGRA~1\AMD\ROCm") else r"C:\Program Files\AMD\ROCm"
    if os.path.exists(default_root):
        try:
            subdirs = [d for d in os.listdir(default_root) if os.path.isdir(os.path.join(default_root, d))]
            subdirs.sort(key=lambda s: [int(x) if x.isdigit() else x for x in s.split('.')], reverse=True)
            for subdir in subdirs:
                path = os.path.join(default_root, subdir)
                if os.path.exists(os.path.join(path, "bin", "hipcc.exe")) or os.path.exists(os.path.join(path, "bin", "hipcc")) or os.path.exists(os.path.join(path, "bin", "hipcc.bat")):
                    _CACHED_ROCM_PATH = _get_short_path(path)
                    return _CACHED_ROCM_PATH
        except Exception:
            pass
    _CACHED_ROCM_PATH = None
    return _CACHED_ROCM_PATH

def _find_rocm_device_lib_path(rocm_path=None):
    for ev in ["ROCM_DEVICE_LIB_PATH", "DEVICE_LIB_PATH", "HIP_DEVICE_LIB_PATH"]:
        p = os.environ.get(ev)
        if p and os.path.exists(os.path.join(p, "ockl.bc")):
            return _get_short_path(os.path.abspath(p))
    if sys.platform == 'win32':
        if rocm_path:
            for c in [os.path.join(rocm_path, "lib", "bitcode"), os.path.join(rocm_path, "amdgcn", "bitcode")]:
                if os.path.exists(c) and os.path.exists(os.path.join(c, "ockl.bc")):
                    return _get_short_path(os.path.abspath(c))
        return None
    import glob
    candidates = []
    if rocm_path:
        candidates.extend([
            os.path.join(rocm_path, "lib", "amdgcn", "bitcode"),
            os.path.join(rocm_path, "amdgcn", "bitcode"),
            os.path.join(rocm_path, "lib", "bitcode"),
        ])
        candidates.extend(glob.glob(os.path.join(rocm_path, "lib*", "llvm*", "lib", "clang", "*", "amdgcn", "bitcode")))
        candidates.extend(glob.glob(os.path.join(rocm_path, "lib*", "clang", "*", "amdgcn", "bitcode")))
    candidates.extend(sorted(glob.glob("/usr/lib/llvm-*/lib/clang/*/amdgcn/bitcode"), reverse=True))
    candidates.extend([
        "/usr/lib/x86_64-linux-gnu/amdgcn/bitcode",
        "/usr/lib/amdgcn/bitcode",
        "/usr/lib64/amdgcn/bitcode",
        "/usr/local/lib/amdgcn/bitcode",
    ])
    candidates.extend(sorted(glob.glob("/opt/rocm*/lib/amdgcn/bitcode"), reverse=True))
    candidates.extend(sorted(glob.glob("/opt/rocm*/amdgcn/bitcode"), reverse=True))
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "ockl.bc")):
            return os.path.abspath(c)
    return None

_CACHED_CUDA_PATH = None

def _find_cuda_path(force_refresh=False):
    global _CACHED_CUDA_PATH
    if not force_refresh and _CACHED_CUDA_PATH is not None:
        return _CACHED_CUDA_PATH
    for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_DIR"]:
        p = os.environ.get(env_var)
        if p and os.path.exists(p):
            cand_nvcc = os.path.join(p, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
            if os.path.exists(cand_nvcc) or os.path.exists(os.path.join(p, "include")):
                _CACHED_CUDA_PATH = os.path.abspath(p)
                return _CACHED_CUDA_PATH
    if sys.platform != 'win32':
        import glob
        for cand in sorted(glob.glob("/usr/local/cuda*") + glob.glob("/opt/cuda*"), reverse=True):
            if os.path.isdir(cand) and (os.path.exists(os.path.join(cand, "bin", "nvcc")) or os.path.exists(os.path.join(cand, "include"))):
                _CACHED_CUDA_PATH = os.path.abspath(cand)
                return _CACHED_CUDA_PATH
    else:
        # 1. Check versioned CUDA_PATH_V* environment variables set by NVIDIA installer (sorted descending by version)
        cuda_v_vars = sorted([ev for ev in os.environ if ev.startswith("CUDA_PATH_V")], reverse=True)
        for ev in cuda_v_vars:
            cand = os.environ[ev]
            if os.path.exists(cand) and os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                _CACHED_CUDA_PATH = os.path.abspath(cand)
                return _CACHED_CUDA_PATH

        # 2. Check Windows Registry (both System Environment and User Environment)
        try:
            import winreg
            reg_targets = [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment")
            ]
            reg_cands = []
            for hkey, subkey in reg_targets:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        i = 0
                        while True:
                            try:
                                name, val, _ = winreg.EnumValue(key, i)
                                if (name.startswith("CUDA_PATH") or name == "CUDA_HOME") and val:
                                    if os.path.isdir(val) and os.path.exists(os.path.join(val, "bin", "nvcc.exe")):
                                        reg_cands.append(os.path.abspath(val))
                                i += 1
                            except OSError:
                                break
                except Exception:
                    pass
            def _cuda_version_priority_key(path_str):
                sub = path_str.replace('\\', '/').split('/')[-1].lstrip('v')
                nums = [int(x) if x.isdigit() else 0 for x in sub.split('.')]
                maj = nums[0] if nums else 0
                min_v = nums[1] if len(nums) > 1 else 0
                is_target_13 = (1 if (maj == 13 and min_v >= 3) else 0)
                is_below_14 = (1 if maj < 14 else 0)
                return (is_target_13, is_below_14, nums)

            if reg_cands:
                reg_cands.sort(key=_cuda_version_priority_key, reverse=True)
                _CACHED_CUDA_PATH = reg_cands[0]
                return _CACHED_CUDA_PATH
        except Exception:
            pass

        # 3. Check nvcc on PATH
        import shutil
        nvcc_p = shutil.which("nvcc")
        if nvcc_p:
            prefix = os.path.dirname(os.path.dirname(os.path.abspath(nvcc_p)))
            if os.path.isdir(prefix) and (os.path.exists(os.path.join(prefix, "include")) or os.path.exists(os.path.join(prefix, "lib"))):
                _CACHED_CUDA_PATH = os.path.abspath(prefix)
                return _CACHED_CUDA_PATH

        # 4. Standard default install directories across system drives (preferring 13.3 <= ver < 14.0)
        for drive in ["C:", "D:", "E:"]:
            default_root = os.path.join(drive, os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA")
            if os.path.exists(default_root):
                try:
                    subdirs = [d for d in os.listdir(default_root) if os.path.isdir(os.path.join(default_root, d))]
                    subdirs.sort(key=_cuda_version_priority_key, reverse=True)
                    for subdir in subdirs:
                        cand = os.path.join(default_root, subdir)
                        if os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                            _CACHED_CUDA_PATH = os.path.abspath(cand)
                            return _CACHED_CUDA_PATH
                except Exception:
                    pass
    return None


def _find_vcvars64():
    global _CACHED_VCVARS64
    if _CACHED_VCVARS64 is not None:
        return _CACHED_VCVARS64
    if sys.platform != 'win32':
        return None
    if shutil.which("cl") is not None:
        _CACHED_VCVARS64 = "CL_READY"
        return _CACHED_VCVARS64

    # 1. Query official Microsoft vswhere locator tool (handles non-C: drive installations)
    try:
        import subprocess
        for prog_dir in [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles", r"C:\Program Files")]:
            vswhere_exe = os.path.join(prog_dir, "Microsoft Visual Studio", "Installer", "vswhere.exe")
            if os.path.exists(vswhere_exe):
                cmd = [vswhere_exe, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]
                res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5.0).strip()
                if res and os.path.isdir(res):
                    bat_cand = os.path.join(res, "VC", "Auxiliary", "Build", "vcvars64.bat")
                    if os.path.exists(bat_cand):
                        _CACHED_VCVARS64 = bat_cand
                        return _CACHED_VCVARS64
    except Exception:
        pass

    # 2. Search all logical drives for Microsoft Visual Studio if vswhere is not available
    drive_roots = []
    for d in [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles", r"C:\Program Files")]:
        if d and os.path.exists(d) and d not in drive_roots:
            drive_roots.append(d)
    for letter in ["D", "E", "F"]:
        for pf in [f"{letter}:\\Program Files (x86)", f"{letter}:\\Program Files"]:
            if os.path.exists(pf) and pf not in drive_roots:
                drive_roots.append(pf)

    editions = ["BuildTools", "Community", "Professional", "Enterprise"]
    versions = ["2026", "2025", "2022", "2019", "2017", "18", "17", "16", "15"]
    for root in drive_roots:
        msvc_root = os.path.join(root, "Microsoft Visual Studio")
        if not os.path.exists(msvc_root):
            continue
        for version in versions:
            for edition in editions:
                bat_path = os.path.join(msvc_root, version, edition, "VC", "Auxiliary", "Build", "vcvars64.bat")
                if os.path.exists(bat_path):
                    _CACHED_VCVARS64 = bat_path
                    return _CACHED_VCVARS64
    return None


def _find_msvc_bin():
    global _CACHED_MSVC_BIN
    if _CACHED_MSVC_BIN is not None:
        return _CACHED_MSVC_BIN
    if sys.platform != 'win32':
        return None

    # 1. Derive from vcvars64 if already located
    vc_bat = _find_vcvars64()
    if vc_bat and vc_bat != "CL_READY":
        try:
            vs_install = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(vc_bat))))
            tools_dir = os.path.join(vs_install, "VC", "Tools", "MSVC")
            if os.path.exists(tools_dir):
                for sd in sorted(os.listdir(tools_dir), reverse=True):
                    cand = os.path.join(tools_dir, sd, "bin", "Hostx64", "x64")
                    if os.path.exists(os.path.join(cand, "link.exe")):
                        _CACHED_MSVC_BIN = cand
                        return _CACHED_MSVC_BIN
        except Exception:
            pass

    # 2. Drive scan fallback
    drive_roots = []
    for d in [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles", r"C:\Program Files")]:
        if d and os.path.exists(d) and d not in drive_roots:
            drive_roots.append(d)
    for letter in ["D", "E", "F"]:
        for pf in [f"{letter}:\\Program Files (x86)", f"{letter}:\\Program Files"]:
            if os.path.exists(pf) and pf not in drive_roots:
                drive_roots.append(pf)

    editions = ["BuildTools", "Community", "Professional", "Enterprise"]
    versions = ["2026", "2025", "2022", "2019", "2017", "18", "17", "16", "15"]
    for root in drive_roots:
        msvc_root = os.path.join(root, "Microsoft Visual Studio")
        if not os.path.exists(msvc_root):
            continue
        for version in versions:
            for edition in editions:
                tools_dir = os.path.join(msvc_root, version, edition, "VC", "Tools", "MSVC")
                if os.path.exists(tools_dir):
                    try:
                        subdirs = sorted(os.listdir(tools_dir), reverse=True)
                        for sd in subdirs:
                            cand = os.path.join(tools_dir, sd, "bin", "Hostx64", "x64")
                            if os.path.exists(os.path.join(cand, "link.exe")):
                                _CACHED_MSVC_BIN = cand
                                return _CACHED_MSVC_BIN
                    except Exception:
                        pass
    return None


def _find_windows_sdk_bin():
    global _CACHED_WIN_SDK_BIN
    if _CACHED_WIN_SDK_BIN is not None:
        return _CACHED_WIN_SDK_BIN
    if sys.platform != 'win32':
        return None
    sdk_roots = [
        r"C:\Program Files (x86)\Windows Kits\10\bin",
        r"C:\Program Files\Windows Kits\10\bin",
        r"D:\Program Files (x86)\Windows Kits\10\bin",
        r"D:\Program Files\Windows Kits\10\bin",
    ]
    for sdk_root in sdk_roots:
        if os.path.exists(sdk_root):
            try:
                subdirs = sorted(os.listdir(sdk_root), reverse=True)
                for sd in subdirs:
                    cand = os.path.join(sdk_root, sd, "x64")
                    if os.path.exists(os.path.join(cand, "rc.exe")):
                        _CACHED_WIN_SDK_BIN = cand
                        return _CACHED_WIN_SDK_BIN
            except Exception:
                pass
    return None


import time
import multiprocessing
import threading
import re
try:
    import numpy as np
except ImportError:
    np = None
import faulthandler

faulthandler.enable()

_ansi_escape = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')

is_main_proc = not is_worker_process() and ("-c" not in sys.argv) and (os.environ.get("MOLMEM_CORE_SEARCH_ACTIVE") != "1")

# Note: OS-Level Ctrl-C handlers are registered immediately at package import in __init__.py

def safe_remove_file(filepath):
    if not filepath:
        return
    try:
        import tempfile
        abs_path = os.path.abspath(filepath)
        temp_dir = os.path.abspath(tempfile.gettempdir())
        
        # Guard 1: Must be inside molmem_lib package directory or system temp directory
        if not (abs_path.startswith(_LIB_DIR) or abs_path.startswith(temp_dir)):
            return
            
        # Guard 2: Never allow deleting Python source files (.py) or directories
        if abs_path.endswith(".py") or os.path.isdir(abs_path):
            return
            
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except PermissionError:
                os.chmod(abs_path, 0o777)
                os.remove(abs_path)
    except Exception:
        pass

def _fast_rmtree(path):
    if not path:
        return
    abs_path = os.path.abspath(path)
    
    # STRICT SAFETY GUARD 1: Must be inside workspace
    if not abs_path.startswith(os.path.dirname(_LIB_DIR)):
        return
        
    # STRICT SAFETY GUARD 2: Basename must be '.calcache' or 'molmem-env'
    bname = os.path.basename(abs_path)
    if bname not in [".calcache", "molmem-env"]:
        return
        
    # STRICT SAFETY GUARD 3: Never delete root or home directories
    home_path = os.path.abspath(os.path.expanduser("~"))
    root_paths = [os.path.abspath("/"), os.path.abspath("C:\\"), home_path]
    if abs_path in root_paths or os.path.dirname(abs_path) in root_paths:
        return

    if not os.path.exists(abs_path):
        return

    if sys.platform == "win32":
        try:
            import subprocess
            subprocess.run(f'cmd.exe /c rmdir /s /q "{abs_path}"', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(abs_path):
                return
        except Exception:
            pass

    def _remove_readonly(func, p, exc_info):
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            pass

    try:
        import shutil
        shutil.rmtree(abs_path, onerror=_remove_readonly)
    except Exception:
        pass

def _run_transient_1x1_bootstrap(queue):
    import os
    import time
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_Transient1x1Bootstrapper initialized: Platform=CPU (Background 1x1 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass
    try:
        from .simulator import MolmemSimulator
        from .profiler import SimulatorProfilerMixin
        SimulatorProfilerMixin._in_bootstrap_profiling = True
        try:
            sim1 = MolmemSimulator(instanceName="1x1_Calibrator", device='cpu', backend='numba')
            sim1._in_warmup = True
            sim1.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
            sim1.programmer = None
            sim1.profileParallelization()
            dt = time.perf_counter() - t0
            opt_thr = sim1.optimal_cores
            cand_str = getattr(sim1, '_last_candidates_summary', '')
            queue.put(f"done:{dt:.2f}:{opt_thr}:{cand_str}")
        finally:
            SimulatorProfilerMixin._in_bootstrap_profiling = False
    except Exception as e:
        import traceback
        try:
            queue.put(f"error:{type(e).__name__}")
        except Exception:
            pass
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_Transient1x1Bootstrapper terminated: Platform=CPU (Background 1x1 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass

def _run_programmer_col_bootstrap(queue):
    import os
    import time
    import numpy as np
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"

    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_ProgrammerColBootstrapper initialized: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass
    try:
        from .simulator import MolmemSimulator
        bootstrap_sim = MolmemSimulator(device='cpu', backend='numba')
        bootstrap_sim.vRead = 0.1
        bootstrap_sim.addCrossbarMatrix(rows=2, cols=2, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")
        bootstrap_sim.profileParallelization()
        dt = time.perf_counter() - t0
        opt_thr = bootstrap_sim.optimal_cores
        cand_str = getattr(bootstrap_sim, '_last_candidates_summary', '')
        queue.put(f"done:{dt:.2f}:{opt_thr}:{cand_str}")
    except Exception as e:
        import traceback
        try:
            queue.put(f"error:{type(e).__name__}")
        except Exception:
            pass
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_ProgrammerColBootstrapper terminated: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass

def _run_programmer_dev_bootstrap(queue):
    import os
    import time
    import numpy as np
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"

    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_ProgrammerDevBootstrapper initialized: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass
    try:
        from .simulator import MolmemSimulator
        bootstrap_sim = MolmemSimulator(device='cpu', backend='numba')
        bootstrap_sim.vRead = 0.1
        bootstrap_sim.addCrossbarMatrix(rows=2, cols=2, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Device")
        bootstrap_sim.profileParallelization()
        dt = time.perf_counter() - t0
        opt_thr = bootstrap_sim.optimal_cores
        cand_str = getattr(bootstrap_sim, '_last_candidates_summary', '')
        queue.put(f"done:{dt:.2f}:{opt_thr}:{cand_str}")
    except Exception as e:
        import traceback
        try:
            queue.put(f"error:{type(e).__name__}")
        except Exception:
            pass
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_ProgrammerDevBootstrapper terminated: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass

def _run_transient_bootstrap(queue):
    import os
    import time
    import numpy as np
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"

    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_TransientBootstrapper initialized: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass
    try:
        from .simulator import MolmemSimulator
        bootstrap_sim = MolmemSimulator(device='cpu', backend='numba')
        bootstrap_sim.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False, UpdatesPer="Device")
        bootstrap_sim.programmer = None
        bootstrap_sim.profileParallelization()
        dt = time.perf_counter() - t0
        opt_thr = bootstrap_sim.optimal_cores
        cand_str = getattr(bootstrap_sim, '_last_candidates_summary', '')
        queue.put(f"done:{dt:.2f}:{opt_thr}:{cand_str}")
    except Exception as e:
        import traceback
        try:
            queue.put(f"error:{type(e).__name__}")
        except Exception:
            pass
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_TransientBootstrapper terminated: Platform=CPU (Background 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
    except Exception:
        pass

def _run_dense_solver_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_0 initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .sources import PulseSource
        from .simulator import MolmemSimulator
        MolmemSimulator._in_warmup = True
        # 1. Warmup 2x2 dense solver (1 unknown node)
        sim = MolmemSimulator(instanceName="JIT_Warmup", device='cpu')
        sim.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False, UpdatesPer="Device")
        sim.programmer = None
        sim.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
        sim.addDcSource(col=1, voltage=0.0)
        sim.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

        # 2. Warmup 1x1 dense solver (0 unknown nodes)
        sim1 = MolmemSimulator(instanceName="JIT_Warmup_1x1", device='cpu')
        sim1.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
        sim1.programmer = None
        sim1.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
        sim1.addDcSource(col=1, voltage=0.0)
        sim1.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Sequential Nodal Circuit Solver:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Sequential Nodal Circuit Solver:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Sequential Nodal Circuit Solver:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_0 terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_parallel_solver_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_1 initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .sources import PwlSource
        from .simulator import MolmemSimulator
        MolmemSimulator._in_warmup = True
        # 1. Warmup 2x2 parallel multi-core solver (1 unknown node)
        sim2 = MolmemSimulator(instanceName="JIT_Warmup_Parallel", device='cpu')
        sim2.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False)
        sim2.optimal_cores = 2
        sim2.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=[0.0, 160e-9], v_points=[0.1, 0.1]))
        sim2.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

        # 2. Warmup 1x1 parallel multi-core solver (0 unknown nodes)
        sim2_1 = MolmemSimulator(instanceName="JIT_Warmup_Parallel_1x1", device='cpu')
        sim2_1.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
        sim2_1.optimal_cores = 2
        sim2_1.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=[0.0, 160e-9], v_points=[0.1, 0.1]))
        sim2_1.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Parallel Workqueue Multi-Core Solver:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Parallel Workqueue Multi-Core Solver:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Parallel Workqueue Multi-Core Solver:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_1 terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

_init_header_printed = False

def print_init_header():
    global _init_header_printed
    if _init_header_printed:
        return
    verbose = os.environ.get('MOLMEM_SILENT_WARMUP') != '1'
    if verbose:
        sys.stdout.write("\n--------------------------------------------------------------\n")
        sys.stdout.write("--- MolmemSimulator: Hardware Systems Initialization     ---\n")
        sys.stdout.write("--------------------------------------------------------------\n")
        sys.stdout.write("  [Note: First-time JIT compilation after package edits may take a minute.]\n")
        sys.stdout.flush()
    _init_header_printed = True

_bootstrap_printed = False
_linker_warmed_up = False
_in_bootstrap_executing = False

def ensure_bootstrap_profiled(verbose=True):
    global _bootstrap_printed, _linker_warmed_up, _in_bootstrap_executing
    if not is_main_proc or _bootstrap_printed or _in_bootstrap_executing:
        return
    _in_bootstrap_executing = True
        
    try:
        from .profiler import SimulatorProfilerMixin
        has_t = 4 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
        has_pc = 4 in SimulatorProfilerMixin._cached_configs.get('Column', {})
        has_pd = 4 in SimulatorProfilerMixin._cached_configs.get('Device', {})
        has_t1 = 1 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
        has_gpu_col = SimulatorProfilerMixin._gpu_crossover_dim.get('Column') is not None
        has_gpu_dev = SimulatorProfilerMixin._gpu_crossover_dim.get('Device') is not None
        if not (has_t and has_pc and has_pd and has_t1 and has_gpu_col and has_gpu_dev):
            from .simulator import MolmemSimulator
            MolmemSimulator._load_profile_cache(basis="Circuit", force=True)
            MolmemSimulator._load_profile_cache(basis="Column", force=True)
            MolmemSimulator._load_profile_cache(basis="Device", force=True)
            has_t = 4 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
            has_pc = 4 in SimulatorProfilerMixin._cached_configs.get('Column', {})
            has_pd = 4 in SimulatorProfilerMixin._cached_configs.get('Device', {})
            has_t1 = 1 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
            has_gpu_col = SimulatorProfilerMixin._gpu_crossover_dim.get('Column') is not None
            has_gpu_dev = SimulatorProfilerMixin._gpu_crossover_dim.get('Device') is not None
    except Exception:
        pass
        
    import os
    import sys
    import time
    import multiprocessing
    
    show_output = verbose and os.environ.get('MOLMEM_SILENT_WARMUP') != '1'
    t0 = time.perf_counter()
    spinner = ['|', '/', '-', '\\']
    spin_idx = 0
    
    if show_output and not (has_t and has_pc and has_pd and has_t1):
        sys.stdout.write("\n  > Launching hardware thread profilers (Parallel Worker Pool Mode)...\n")
        sys.stdout.flush()

    t_status = "Skipped (Loaded from cache)"
    pc_status = "Skipped (Loaded from cache)"
    pd_status = "Skipped (Loaded from cache)"
    t1_status = "Skipped (Loaded from cache)"
    t_opt = None
    pc_opt = None
    pd_opt = None
    t1_opt = None

    procs = []

    # --- 1. Run Transient 2x2 Bootstrapper ---
    p_transient = None
    t_queue = None
    if not has_t:
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Subprocess (multiprocessing): MOLMEM_TransientBootstrapper | Platform=CPU (Parallel 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
        except Exception:
            pass
        t_queue = multiprocessing.Queue()
        p_transient = multiprocessing.Process(target=_run_transient_bootstrap, args=(t_queue,), name="MOLMEM_TransientBootstrapper")
        p_transient.daemon = True
        procs.append(p_transient)

    # --- 2. Run Programmer Column 2x2 Bootstrapper ---
    p_col = None
    pc_queue = None
    if not has_pc:
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Subprocess (multiprocessing): MOLMEM_ProgrammerColBootstrapper | Platform=CPU (Parallel 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
        except Exception:
            pass
        pc_queue = multiprocessing.Queue()
        p_col = multiprocessing.Process(target=_run_programmer_col_bootstrap, args=(pc_queue,), name="MOLMEM_ProgrammerColBootstrapper")
        p_col.daemon = True
        procs.append(p_col)

    # --- 3. Run Programmer Device 2x2 Bootstrapper ---
    p_dev = None
    pd_queue = None
    if not has_pd:
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Subprocess (multiprocessing): MOLMEM_ProgrammerDevBootstrapper | Platform=CPU (Parallel 2x2 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
        except Exception:
            pass
        pd_queue = multiprocessing.Queue()
        p_dev = multiprocessing.Process(target=_run_programmer_dev_bootstrap, args=(pd_queue,), name="MOLMEM_ProgrammerDevBootstrapper")
        p_dev.daemon = True
        procs.append(p_dev)

    # --- 4. Run Transient 1x1 Bootstrapper ---
    p_t1 = None
    t1_queue = None
    if not has_t1:
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Subprocess (multiprocessing): MOLMEM_Transient1x1Bootstrapper | Platform=CPU (Parallel 1x1 JIT cache bootstrapper) | Threads=1..4 Thread Sweep | Core Affinity=OS Managed")
        except Exception:
            pass
        t1_queue = multiprocessing.Queue()
        p_t1 = multiprocessing.Process(target=_run_transient_1x1_bootstrap, args=(t1_queue,), name="MOLMEM_Transient1x1Bootstrapper")
        p_t1.daemon = True
        procs.append(p_t1)

    # Launch all profiler processes concurrently in parallel!
    for p in procs:
        p.start()

    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    printed_once = False
    while any(p.is_alive() for p in procs):
        if show_output:
            if is_tty:
                spin_char = spinner[spin_idx % len(spinner)]
                spin_idx += 1
                sys.stdout.write(f"\r  > Profiling Thread Performance... {spin_char}\033[K")
                sys.stdout.flush()
            elif not printed_once:
                sys.stdout.write("  > Profiling Thread Performance...\n")
                sys.stdout.flush()
                printed_once = True
        time.sleep(0.05)

    for p in procs:
        p.join()

    _merge_child_logs()

    t_cand = ""
    pc_cand = ""
    pd_cand = ""
    t1_cand = ""

    if t_queue is not None:
        try:
            content = t_queue.get(timeout=1.0)
            if content.startswith("done:"):
                parts = content.split(":")
                t_status = f"Succeeded (in {parts[1]}s)"
                if len(parts) > 2: t_opt = parts[2]
                if len(parts) > 3: t_cand = parts[3]
            else: t_status = f"Failed ({content})"
        except Exception: t_status = "Unknown"

    if pc_queue is not None:
        try:
            content = pc_queue.get(timeout=1.0)
            if content.startswith("done:"):
                parts = content.split(":")
                pc_status = f"Succeeded (in {parts[1]}s)"
                if len(parts) > 2: pc_opt = parts[2]
                if len(parts) > 3: pc_cand = parts[3]
            else: pc_status = f"Failed ({content})"
        except Exception: pc_status = "Unknown"

    if pd_queue is not None:
        try:
            content = pd_queue.get(timeout=1.0)
            if content.startswith("done:"):
                parts = content.split(":")
                pd_status = f"Succeeded (in {parts[1]}s)"
                if len(parts) > 2: pd_opt = parts[2]
                if len(parts) > 3: pd_cand = parts[3]
            else: pd_status = f"Failed ({content})"
        except Exception: pd_status = "Unknown"

    if t1_queue is not None:
        try:
            content = t1_queue.get(timeout=1.0)
            if content.startswith("done:"):
                parts = content.split(":")
                t1_status = f"Succeeded (in {parts[1]}s)"
                if len(parts) > 2: t1_opt = parts[2]
                if len(parts) > 3: t1_cand = parts[3]
            else: t1_status = f"Failed ({content})"
        except Exception: t1_status = "Unknown"

    if t_status == "Skipped (Loaded from cache)" or pc_status == "Skipped (Loaded from cache)" or pd_status == "Skipped (Loaded from cache)" or t1_status == "Skipped (Loaded from cache)":
        try:
            from .simulator import MolmemSimulator
            MolmemSimulator._load_profile_cache(basis="Circuit", force=True)
            MolmemSimulator._load_profile_cache(basis="Column", force=True)
            MolmemSimulator._load_profile_cache(basis="Device", force=True)
            if 4 in MolmemSimulator._cached_configs.get("Circuit", {}):
                if t_status == "Skipped (Loaded from cache)":
                    t_opt = str(MolmemSimulator._cached_configs["Circuit"][4][0])
            if 4 in MolmemSimulator._cached_configs.get("Column", {}):
                if pc_status == "Skipped (Loaded from cache)":
                    pc_opt = str(MolmemSimulator._cached_configs["Column"][4][0])
            if 4 in MolmemSimulator._cached_configs.get("Device", {}):
                if pd_status == "Skipped (Loaded from cache)":
                    pd_opt = str(MolmemSimulator._cached_configs["Device"][4][0])
            if 1 in MolmemSimulator._cached_configs.get("Circuit", {}):
                if t1_status == "Skipped (Loaded from cache)":
                    t1_opt = str(MolmemSimulator._cached_configs["Circuit"][1][0])
        except Exception as e:
            import traceback
            sys.stdout.write(f"\n  [!] Debug bootstrap cache read error: {e}\n{traceback.format_exc()}\n")
    try:
        from .sys_utils import log_to_run_log_only
        t_cand_str = f" | Candidates: [{t_cand}]" if t_cand else ""
        pc_cand_str = f" | Candidates: [{pc_cand}]" if pc_cand else ""
        pd_cand_str = f" | Candidates: [{pd_cand}]" if pd_cand else ""
        t1_cand_str = f" | Candidates: [{t1_cand}]" if t1_cand else ""
        log_to_run_log_only(f"[THREAD_BENCHMARK] [PID {os.getpid()}] Transient 2x2 Thread Profiling: {t_status}{t_cand_str} | Optimal Configuration: {t_opt if t_opt else 'Dynamic'} Thread(s)")
        log_to_run_log_only(f"[THREAD_BENCHMARK] [PID {os.getpid()}] Programmer Col 2x2 Thread Profiling: {pc_status}{pc_cand_str} | Optimal Configuration: {pc_opt if pc_opt else 'Dynamic'} Thread(s)")
        log_to_run_log_only(f"[THREAD_BENCHMARK] [PID {os.getpid()}] Programmer Dev 2x2 Thread Profiling: {pd_status}{pd_cand_str} | Optimal Configuration: {pd_opt if pd_opt else 'Dynamic'} Thread(s)")
        log_to_run_log_only(f"[THREAD_BENCHMARK] [PID {os.getpid()}] Transient 1x1 Thread Profiling: {t1_status}{t1_cand_str} | Optimal Configuration: {t1_opt if t1_opt else 'Dynamic'} Thread(s)")
    except Exception:
        pass

    if show_output:
        if is_tty:
            sys.stdout.write(f"\r  > CPU Multi-Thread Performance Profiling... Done (in {time.perf_counter() - t0:.1f}s)\033[K\n")
        else:
            sys.stdout.write(f"  > CPU Multi-Thread Performance Profiling... Done (in {time.perf_counter() - t0:.1f}s)\n")
        sys.stdout.write(f"    >> Transient 2x2 Thread Profiling       : {t_status}\033[K\n")
        if t_opt:
            sys.stdout.write(f"      >>> Optimal Configuration          : {t_opt} Thread(s)\033[K\n")
        sys.stdout.write(f"    >> Programmer Col 2x2 Thread Profiling  : {pc_status}\033[K\n")
        if pc_opt:
            sys.stdout.write(f"      >>> Optimal Configuration          : {pc_opt} Thread(s)\033[K\n")
        sys.stdout.write(f"    >> Programmer Dev 2x2 Thread Profiling  : {pd_status}\033[K\n")
        if pd_opt:
            sys.stdout.write(f"      >>> Optimal Configuration          : {pd_opt} Thread(s)\033[K\n")
        sys.stdout.write(f"    >> Transient 1x1 Thread Profiling       : {t1_status}\033[K\n")
        if t1_opt:
            sys.stdout.write(f"      >>> Optimal Configuration          : {t1_opt} Thread(s)\033[K\n")
        sys.stdout.flush()

    try:
        from .profiler import SimulatorProfilerMixin
        SimulatorProfilerMixin._load_profile_cache(basis='Circuit', verbose=False, force=True)
        SimulatorProfilerMixin._load_profile_cache(basis='Column', verbose=False, force=True)
        SimulatorProfilerMixin._load_profile_cache(basis='Device', verbose=False, force=True)
        
        updated = False
        if t_opt:
            SimulatorProfilerMixin._cached_configs["Circuit"][4] = (int(t_opt), False)
            updated = True
        if pc_opt:
            SimulatorProfilerMixin._cached_configs["Column"][4] = (int(pc_opt), False)
            updated = True
        if pd_opt:
            SimulatorProfilerMixin._cached_configs["Device"][4] = (int(pd_opt), False)
            updated = True
        if t1_opt:
            SimulatorProfilerMixin._cached_configs["Circuit"][1] = (int(t1_opt), False)
            SimulatorProfilerMixin._cached_configs["Column"][1] = (1, False)
            SimulatorProfilerMixin._cached_configs["Device"][1] = (int(t1_opt), False)
            updated = True
            
        if updated:
            SimulatorProfilerMixin._save_profile_cache(
                None,
                SimulatorProfilerMixin._cached_configs["Circuit"],
                basis="Circuit"
            )
            SimulatorProfilerMixin._save_profile_cache(
                None,
                SimulatorProfilerMixin._cached_configs["Column"],
                basis="Column"
            )
            SimulatorProfilerMixin._save_profile_cache(
                None,
                SimulatorProfilerMixin._cached_configs["Device"],
                basis="Device"
            )
    except Exception as e:
        sys.stdout.write(f"\n  [!] Error saving bootstrap profiles to disk: {e}\n")
        sys.stdout.flush()

    _bootstrap_printed = True

    # --- 5. GPU Device Dispatch Calibration ---
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available() and os.environ.get('MOLMEM_CPU_ONLY') != '1':
            gpu_available = True
    except ImportError:
        pass

    gpu_col_status = None
    gpu_col_opt = None
    gpu_dev_status = None
    gpu_dev_opt = None

    if not gpu_available:
        gpu_col_status = "Skipped (No CUDA/ROCm GPU Available)"
        gpu_dev_status = "Skipped (No CUDA/ROCm GPU Available)"
    else:
        gpu_col_status = "Heterogeneous Work-Stealing Active"
        gpu_dev_status = "Heterogeneous Work-Stealing Active"
        gpu_col_opt = "Self-Paced Work Queue"
        gpu_dev_opt = "Self-Paced Work Queue"
        
        if show_output:
            sys.stdout.write("  > Dual-Ended Converging Dispatch        : Active (Dynamic Lock-Free Cursors)\n")
            sys.stdout.write("--------------------------------------------------------------\n\n")
            sys.stdout.flush()

    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[COMPUTE_ENGINE] [PID {os.getpid()}] Heterogeneous CPU+GPU Work-Stealing Engine Active")
    except Exception:
        pass

    _in_bootstrap_executing = False


def check_hardware_cache_status():
    """
    Returns True if all JIT bytecode, GPU C++/HIP binaries, 3D calibration pulse maps,
    and thread profiler configurations are present in cache; False if any component
    must be compiled, calibrated, or profiled from scratch.
    """
    try:
        import glob
        pycache_dir = os.path.join(_LIB_DIR, '__pycache__')
        has_device_bc = len(glob.glob(os.path.join(pycache_dir, 'device_jit*.nbc'))) > 0
        has_solver_bc = len(glob.glob(os.path.join(pycache_dir, 'solver*.nbc'))) > 0
        has_prog_bc = len(glob.glob(os.path.join(pycache_dir, 'programmer_solver*.nbc'))) > 0
        
        need_gpu_ext = False
        has_gpu_bc = True
        try:
            import torch
            from .torch_simulator import HAS_CPP_EXTENSION
            has_gpu = torch.cuda.is_available() and os.environ.get('MOLMEM_CPU_ONLY') != '1'
            need_gpu_ext = has_gpu and HAS_CPP_EXTENSION
            if need_gpu_ext:
                csrc_hash_f = os.path.join(_LIB_DIR, 'csrc', '.csrc_hash')
                csrc_d = os.path.join(_LIB_DIR, 'csrc')
                has_gpu_bc = os.path.isfile(csrc_hash_f) and os.path.exists(csrc_d) and any(
                    f.startswith("molmem_cuda_solver") and (f.endswith(".pyd") or f.endswith(".so"))
                    for f in os.listdir(csrc_d)
                )
        except Exception:
            pass
            
        jit_cached = (has_device_bc and has_solver_bc and has_prog_bc and (not need_gpu_ext or has_gpu_bc))
        if not jit_cached:
            return False
            
        # Check unified calibration cache (.calcache)
        try:
            from . import _load_unified_cache, _compute_library_footprint
            data = _load_unified_cache()
            if not data:
                return False
            if 'lib_footprint' not in data or data['lib_footprint'][0] != _compute_library_footprint():
                return False
            if 'optMapPot' not in data or 'optMapDep' not in data:
                return False
        except Exception:
            return False
            
        # Check thread profiling cache
        try:
            from .profiler import SimulatorProfilerMixin
            from .simulator import MolmemSimulator
            MolmemSimulator._load_profile_cache(basis="Circuit", force=False)
            MolmemSimulator._load_profile_cache(basis="Column", force=False)
            MolmemSimulator._load_profile_cache(basis="Device", force=False)
            has_t = 4 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
            has_pc = 4 in SimulatorProfilerMixin._cached_configs.get('Column', {})
            has_pd = 4 in SimulatorProfilerMixin._cached_configs.get('Device', {})
            has_t1 = 1 in SimulatorProfilerMixin._cached_configs.get('Circuit', {})
            has_gpu_col = SimulatorProfilerMixin._gpu_crossover_dim.get('Column') is not None
            has_gpu_dev = SimulatorProfilerMixin._gpu_crossover_dim.get('Device') is not None
            has_chunk_opt = True
            try:
                import torch
                if torch.cuda.is_available() and os.environ.get('MOLMEM_CPU_ONLY') != '1':
                    has_chunk_opt = getattr(SimulatorProfilerMixin, '_opt_cpu_chunk_per_thread', None) is not None
            except Exception:
                pass
            if not (has_t and has_pc and has_pd and has_t1 and has_gpu_col and has_gpu_dev and has_chunk_opt):
                return False
        except Exception:
            return False
            
        return True
    except Exception:
        return False


class SimulatorSystemMixin:
    _jit_warmed_up = False
    _in_warmup = False
    _full_init_printed = False
    _hardware_status_printed = False

    @classmethod
    def warmup_compilers(cls, verbose=None):
        if (cls._jit_warmed_up or not is_main_proc):
            return
            
        if os.environ.get('MOLMEM_CPU_ONLY') == '1' and not is_interactive_notebook():
            return
            
        # Lock immediately to prevent infinite recursion during dummy crossbar generation
        cls._jit_warmed_up = True 
        cls._in_warmup = True
        
        all_cached = check_hardware_cache_status()
        
        # Shorthand path for Interactive Notebooks (Jupyter / Colab)
        if is_interactive_notebook():
            cls._full_init_printed = False
            if not all_cached:
                import time
                target_stream = sys.stdout
                while hasattr(target_stream, '_target') and getattr(target_stream, '_target', None) is not None:
                    target_stream = target_stream._target
                try:
                    target_stream.write("  > Compiling JIT physics kernels and calibrating hardware cache... ")
                    target_stream.flush()
                except Exception:
                    pass
                t0_nb = time.perf_counter()
                ensure_jit_compiled(verbose=False)
                dt_nb = time.perf_counter() - t0_nb
                try:
                    target_stream.write(f"Done (in {dt_nb:.1f}s)\n\n")
                    target_stream.flush()
                except Exception:
                    pass
            else:
                ensure_jit_compiled(verbose=False)
            return

        if verbose is None:
            verbose = os.environ.get('MOLMEM_SILENT_WARMUP') != '1'
            
        # If any component is not available from cache, bypass verbose=False to print full hardware initialization block
        if not all_cached:
            verbose = True
            cls._full_init_printed = True
        else:
            cls._full_init_printed = bool(verbose)
            
        ensure_jit_compiled(verbose=verbose)




def _run_cal_solver_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_2 initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .sources import PulseSource
        from .simulator import MolmemSimulator
        MolmemSimulator._in_warmup = True
        sim_cal = MolmemSimulator(instanceName="JIT_Warmup_Cal", device='cpu')
        sim_cal._is_calibration_helper = True
        sim_cal.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False, weightBits=8, inputBits=8, outputBits=8)
        sim_cal.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
        sim_cal.addDcSource(col=1, voltage=0.0)
        sim_cal.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Quantized Inference Calibration:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Quantized Inference Calibration:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Quantized Inference Calibration:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_2 terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_weight_prog_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_Prog initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .programmer_solver import (
            jit_calibrate_grid_cpu, 
            jit_compute_conductance_map, 
            jit_compute_min_pulses,
            jit_cpu_device_updates_parallel,
            jit_trace_pulse_history,
            jit_predict_pulses_matrix
        )
        v_arr = np.array([1.0], dtype=np.float64)
        pw_arr = np.array([80e-9], dtype=np.float64)
        jit_calibrate_grid_cpu(
            v_arr, pw_arr, True, 2,
            33200.0, 0.0, 0.0,
            0.1, 1.0, 0.05, 0.5, 0.3, 0.0, 0.0, 0.0, 33200.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0
        )
        _dummy_g_in = np.zeros(2, dtype=np.float64)
        _dummy_g_lut = np.zeros(2, dtype=np.float64)
        jit_compute_conductance_map(
            _dummy_g_in, 0.5, 33200.0, 1.0, 0.0, 0.0,
            _dummy_g_lut, _dummy_g_lut, _dummy_g_lut, 0.0, 1.0
        )
        _dummy_pulse_map = np.ones((1, 1), dtype=np.float64)
        _dummy_shifts = np.array([0.5], dtype=np.float64)
        jit_compute_min_pulses(_dummy_pulse_map, _dummy_shifts)

        _dummy_1d_f = np.zeros(1, dtype=np.float64)
        _dummy_1d_i = np.zeros(1, dtype=np.int32)
        _dummy_lut = np.zeros(2, dtype=np.float64)
        jit_cpu_device_updates_parallel(
            1, _dummy_1d_i, _dummy_1d_f, _dummy_1d_f,
            _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.full(1, 298.15, dtype=np.float64),
            _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
            _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
            _dummy_1d_i, _dummy_1d_f, np.full(1, 0.1, dtype=np.float64), np.full(1, 1e4, dtype=np.float64), np.full(1, 1e-7, dtype=np.float64),
            np.full(1, 0.05, dtype=np.float64), np.full(1, 0.05, dtype=np.float64), np.full(1, 0.5, dtype=np.float64), np.full(1, 0.3, dtype=np.float64),
            np.full(1, 0.5, dtype=np.float64), _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.full(1, 33200.0, dtype=np.float64), _dummy_1d_f,
            np.ones(1, dtype=np.float64), _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64), _dummy_1d_f,
            _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64),
            _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f,
            _dummy_lut, _dummy_lut, _dummy_lut, _dummy_lut, _dummy_1d_f
        )
        jit_trace_pulse_history(
            0.0, 1, _dummy_1d_i, _dummy_1d_f, _dummy_1d_f,
            _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.full(1, 298.15, dtype=np.float64),
            _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
            _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
            _dummy_1d_i, _dummy_1d_f, np.full(1, 0.1, dtype=np.float64), np.full(1, 1e4, dtype=np.float64), np.full(1, 1e-7, dtype=np.float64),
            np.full(1, 0.05, dtype=np.float64), np.full(1, 0.05, dtype=np.float64), np.full(1, 0.5, dtype=np.float64), np.full(1, 0.3, dtype=np.float64),
            np.full(1, 0.5, dtype=np.float64), _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.full(1, 33200.0, dtype=np.float64), _dummy_1d_f,
            np.ones(1, dtype=np.float64), _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64), _dummy_1d_f,
            _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64),
            _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f,
            _dummy_lut, _dummy_lut, _dummy_lut, _dummy_lut, 0.5
        )
        _dummy_2x2_f = np.zeros((2, 2), dtype=np.float64)
        _dummy_2x2_i = np.zeros((2, 2), dtype=np.int32)
        _dummy_1x1_f = np.ones((1, 1), dtype=np.float64)
        _dummy_lut1 = np.zeros(2, dtype=np.float64)
        jit_predict_pulses_matrix(
            _dummy_2x2_f, _dummy_2x2_f, False, 0.005,
            _dummy_1x1_f, _dummy_1x1_f,
            _dummy_lut1, _dummy_lut1,
            _dummy_lut1, _dummy_lut1,
            1.0, 1.0,
            1.0, 0.0,
            _dummy_1x1_f, _dummy_1x1_f,
            np.array([1.0], dtype=np.float64), np.array([-1.0], dtype=np.float64), np.array([80e-9], dtype=np.float64),
            _dummy_2x2_i, _dummy_2x2_i,
            1.0, -1.0, 80e-9, 80e-9,
            _dummy_2x2_i, _dummy_2x2_f, _dummy_2x2_f,
            _dummy_2x2_i, _dummy_2x2_f, _dummy_2x2_f
        )
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"3D Pulse Map Inverse Grid Search:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] 3D Pulse Map Inverse Grid Search:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"3D Pulse Map Inverse Grid Search:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_Prog terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_device_iv_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_IV initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .device_jit import jit_interp_O1, jit_getIGeq
        dummy_arr = np.zeros(10, dtype=np.float64)
        jit_interp_O1(0.5, dummy_arr, 33040.0)
        jit_getIGeq(0.5, 1e-4, 1.0, 0.5, 10.0)
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Device IV & Spline Interpolation:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Device IV & Spline Interpolation:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Device IV & Spline Interpolation:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_IV terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_nodal_assembly_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_Nodal initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .solver import jit_assemble_and_solve_dense
        gMat = np.zeros((2, 2), dtype=np.float64)
        iRes = np.zeros(2, dtype=np.float64)
        vCurr = np.zeros(2, dtype=np.float64)
        jit_assemble_and_solve_dense(
            2, 1, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
            np.array([1e-4], dtype=np.float64), np.array([1e-5], dtype=np.float64),
            np.array([0], dtype=np.int32), np.array([1e-4], dtype=np.float64), np.array([0.0], dtype=np.float64),
            vCurr, gMat, iRes
        )
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Newton-Raphson Nodal Assembly:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Newton-Raphson Nodal Assembly:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Newton-Raphson Nodal Assembly:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_Nodal terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_pwl_vectorizer_jit_bootstrap(queue):
    import os, sys, time, numpy as np
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_PWL initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .solver import jit_eval_dynamic_sources, jit_freeze_1t1r_vdevs, jit_build_full_master_schedule
        dyn_types = np.array([0], dtype=np.int32)
        dyn_dc = np.array([0.0], dtype=np.float64)
        dyn_pulse = np.zeros((1, 6), dtype=np.float64)
        dyn_pwl_t = np.zeros((1, 2), dtype=np.float64)
        dyn_pwl_v = np.zeros((1, 2), dtype=np.float64)
        dyn_mask = np.array([False], dtype=np.bool_)
        dyn_cache = np.zeros(1, dtype=np.int32)
        dyn_G = np.zeros(1, dtype=np.float64)
        dyn_I = np.zeros(1, dtype=np.float64)
        jit_eval_dynamic_sources(0.0, dyn_types, dyn_dc, dyn_pulse, dyn_pwl_t, dyn_pwl_v, dyn_mask, dyn_cache, dyn_G, dyn_I, 1e-12)
        jit_freeze_1t1r_vdevs(1, np.array([0], dtype=np.int32), dyn_G, 1e-9, np.zeros(1, dtype=np.float64))
        _dummy_edges = np.array([0.0, 1e-6], dtype=np.float64)
        _dummy_active = np.array([True], dtype=np.bool_)
        jit_build_full_master_schedule(_dummy_edges, _dummy_active, 1e-9, 1e9)
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Dynamic PWL Signal Vectorizer:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Dynamic PWL Signal Vectorizer:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Dynamic PWL Signal Vectorizer:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_PWL terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_device_ode_jit_bootstrap(queue):
    import os, sys, time
    set_worker_numba_threads(1)
    os.environ["MOLMEM_MP_CHILD"] = "1"
    os.environ["MOLMEM_IN_SUBPROCESS"] = "1"
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    os.environ["MOLMEM_SKIP_FULL_CALIBRATION"] = "1"
    os.environ["MOLMEM_FORCE_CPU"] = "1"
    os.environ["MOLMEM_CPU_ONLY"] = "1"
    os.environ.pop("MOLMEM_FORCE_GPU", None)
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_ODE initialized: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        from .device import _ensure_jit
        _ensure_jit()
        dt = time.perf_counter() - t0
        tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"Arrhenius Thermal ODE Predictor:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] Arrhenius Thermal ODE Predictor:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"Arrhenius Thermal ODE Predictor:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess MOLMEM_JIT_Worker_ODE terminated: Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_torch_jit_bootstrap(queue):
    import os, sys, time
    t0 = time.perf_counter()
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Thread MOLMEM_JIT_Worker_Torch initialized: Platform=CPU (Parallel JIT compiler worker thread) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass
    try:
        import torch
        from .sys_utils import _LIB_DIR
        cache_dir = os.path.join(_LIB_DIR, '.calcache')
        os.makedirs(cache_dir, exist_ok=True)
        torch_cache_file = os.path.join(cache_dir, 'torchscript_pwl.pt')
        
        def _pwl_timeline_fn(chunk_pulses_t, chunk_voltages_t, chunk_pwidths_t, pt_idx, sub_pt, k: int):
            chunk_periods = chunk_pwidths_t * 2.0
            t0 = k * chunk_periods
            t1 = t0 + 1e-11
            t2 = t0 + chunk_pwidths_t
            t3 = t2 + 1e-11
            
            t_candidate = torch.where(sub_pt == 0, t0,
                          torch.where(sub_pt == 1, t1,
                          torch.where(sub_pt == 2, t2, t3)))
            v_candidate = torch.where((sub_pt == 1) | (sub_pt == 2), chunk_voltages_t, 0.0)
            
            mask = pt_idx < (chunk_pulses_t * 4)
            t_final = torch.where(mask, t_candidate, 1e20)
            v_final = torch.where(mask, v_candidate, 0.0)
            return t_final, v_final

        loaded_from_cache = False
        if os.path.isfile(torch_cache_file):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    scripted_fn = torch.jit.load(torch_cache_file)
                loaded_from_cache = True
            except Exception:
                scripted_fn = torch.jit.script(_pwl_timeline_fn)
                try:
                    torch.jit.save(scripted_fn, torch_cache_file)
                except Exception:
                    pass
        else:
            scripted_fn = torch.jit.script(_pwl_timeline_fn)
            try:
                torch.jit.save(scripted_fn, torch_cache_file)
            except Exception:
                pass
            
        chunk_pulses = torch.tensor([1, 1], dtype=torch.int32)
        chunk_voltages = torch.tensor([1.0, -1.0], dtype=torch.float32)
        chunk_pwidths = torch.tensor([80e-9, 80e-9], dtype=torch.float32)
        pt_idx = torch.tensor([0, 1], dtype=torch.int32)
        sub_pt = torch.tensor([0, 1], dtype=torch.int32)
        scripted_fn(chunk_pulses, chunk_voltages, chunk_pwidths, pt_idx, sub_pt, 0)
        
        if torch.cuda.is_available():
            try:
                _dummy_gpu = torch.zeros(1, device='cuda:0')
                del _dummy_gpu
            except Exception:
                pass
        
        dt = time.perf_counter() - t0
        if loaded_from_cache:
            tag = f"Succeeded (Loaded from cache in {dt:.2f}s)"
        else:
            tag = f"Succeeded (in {dt:.1f}s)" if dt >= 0.5 else f"Succeeded (in {dt:.2f}s)"
        queue.put(f"PyTorch TorchScript JIT Engine:{tag}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from .sys_utils import log_exception_file
            log_exception_file(f"[JIT_BOOTSTRAP_FAILURE] PyTorch TorchScript JIT Engine:\n{tb}")
        except Exception:
            pass
        err_msg = str(e).strip().split('\n')[0]
        tag = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
        queue.put(f"PyTorch TorchScript JIT Engine:{tag}")
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Thread MOLMEM_JIT_Worker_Torch terminated: Platform=CPU (Parallel JIT compiler worker thread) | Threads=1 | Core Affinity=Process Managed")
    except Exception:
        pass

def _run_gpu_ext_warmup_worker(queue):
    import os, sys, time
    os.environ["MOLMEM_GPU_PROBE"] = "1"
    prev_warmup = os.environ.get("MOLMEM_SILENT_WARMUP")
    os.environ["MOLMEM_SILENT_WARMUP"] = "1"
    t0 = time.perf_counter()
    try:
        import torch
        selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', '0'))
        if torch.cuda.is_available() and selected_gpu < torch.cuda.device_count():
            torch.cuda.set_device(selected_gpu)
        from .simulator import MolmemSimulator
        from .sources import PulseSource
        sim_gpu = MolmemSimulator(instanceName="JIT_Warmup_GPU_Ext", device=f'cuda:{selected_gpu}')
        sim_gpu._in_warmup = True
        sim_gpu.compile_backend = True
        sim_gpu.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
        sim_gpu.programmer = None
        sim_gpu.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
        sim_gpu.addDcSource(col=1, voltage=0.0)
        sim_gpu.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)
        try:
            sim_gpu.clearHistory(force_gc=False)
        except Exception:
            pass
        dt = time.perf_counter() - t0
        queue.put(f"GPU Extension Solver Warmup:SUCCESS:{dt:.4f}")
    except Exception as e:
        import traceback
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[GPU_WARMUP_FAIL] {traceback.format_exc()}")
        except Exception:
            pass
        queue.put(f"GPU Extension Solver Warmup:FAIL:{type(e).__name__}: {e}")
    finally:
        if prev_warmup is None:
            os.environ.pop("MOLMEM_SILENT_WARMUP", None)
        else:
            os.environ["MOLMEM_SILENT_WARMUP"] = prev_warmup

_jit_lock = threading.Lock()
_jit_compiled = False

def ensure_jit_compiled(verbose=True):
    global _jit_compiled, _init_header_printed
    if not is_main_proc or _jit_compiled:
        return
        
    with _jit_lock:
        if _jit_compiled:
            return
        _jit_compiled = True
        
        try:
            PhysicsBundleManager.load_bundle()
        except Exception:
            pass
            
        try:
            import torch
            from .torch_simulator import HAS_CPP_EXTENSION, CPP_EXTENSION_DISABLED_REASON, ACTIVE_GPU_ARCH
            has_gpu = torch.cuda.is_available() and os.environ.get('MOLMEM_CPU_ONLY') != '1'
            need_gpu_ext = has_gpu and HAS_CPP_EXTENSION
            if has_gpu and not need_gpu_ext:
                if CPP_EXTENSION_DISABLED_REASON and "Architecture Mismatch" in CPP_EXTENSION_DISABLED_REASON:
                    arch_tag = f" ({ACTIVE_GPU_ARCH})" if ACTIVE_GPU_ARCH else ""
                    gpu_skip_status = f"Skipped (Arch mismatch{arch_tag})"
                elif CPP_EXTENSION_DISABLED_REASON:
                    gpu_skip_status = f"Skipped ({CPP_EXTENSION_DISABLED_REASON})"
                else:
                    gpu_skip_status = "Skipped (C++ Extension Not Built)"
            else:
                gpu_skip_status = "Skipped (Unused)"
        except Exception:
            has_gpu = False
            need_gpu_ext = False
            gpu_skip_status = "Skipped (Unused)"
        
        total_steps = 9 # 9 parallel CPU workers
        if need_gpu_ext:
            total_steps += 1
            
        import glob
        nbc_before = len(glob.glob(os.path.join(_LIB_DIR, '__pycache__', '*.nbc')))
        t0_total = time.perf_counter()
        inv_total_steps = 1.0 / max(1, total_steps)

        if nbc_before > 0:
            results = {
                "Device IV & Spline Interpolation": "[Loading JIT Cache...]",
                "Arrhenius Thermal ODE Predictor": "[Loading JIT Cache...]",
                "Newton-Raphson Nodal Assembly": "[Loading JIT Cache...]",
                "Dynamic PWL Signal Vectorizer": "[Loading JIT Cache...]",
                "3D Pulse Map Inverse Grid Search": "[Loading JIT Cache...]",
                "PyTorch TorchScript JIT Engine": "[Loading JIT Cache...]",
                "Sequential Nodal Circuit Solver": "[Loading JIT Cache...]",
                "Parallel Workqueue Multi-Core Solver": "[Loading JIT Cache...]",
                "Quantized Inference Calibration": "[Loading JIT Cache...]",
                "GPU Extension Solver Warmup": gpu_skip_status if not need_gpu_ext else "[Loading JIT Cache...]"
            }
        else:
            results = {
                "Device IV & Spline Interpolation": "[Compiling Foundation...]",
                "Arrhenius Thermal ODE Predictor": "[Compiling Foundation...]",
                "Newton-Raphson Nodal Assembly": "[Compiling Foundation...]",
                "Dynamic PWL Signal Vectorizer": "[Compiling Foundation...]",
                "3D Pulse Map Inverse Grid Search": "[Compiling Foundation...]",
                "PyTorch TorchScript JIT Engine": "[Compiling Foundation...]",
                "Sequential Nodal Circuit Solver": "[Awaiting Foundation Cache...]",
                "Parallel Workqueue Multi-Core Solver": "[Awaiting Foundation Cache...]",
                "Quantized Inference Calibration": "[Awaiting Foundation Cache...]",
                "GPU Extension Solver Warmup": gpu_skip_status if not need_gpu_ext else "[Awaiting Foundation Cache...]"
            }
        
        rendered_lines = 0
        completed_steps = 0
        
        if verbose:
            print_init_header()
        
        def render_live_status(status_hdr=None, is_final=False):
            nonlocal rendered_lines
            if not verbose:
                return
            progress = min(100.0, (completed_steps * inv_total_steps) * 100.0)
            bar = '#' * int(progress * 0.2)
            hdr_text = status_hdr if status_hdr is not None else "Compiling..."
            
            term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
            max_line_w = max(20, term_w - 2)
            
            lines = [f"  Compiling CPU JIT solver: [{bar.ljust(20)}] {progress:3.0f}% | {hdr_text}"]
            for k, v in results.items():
                lines.append(f"    > {k.ljust(38)} : {v}")
            lines.append("")
            
            if rendered_lines > 0:
                sys.stdout.write(f"\033[{rendered_lines}A")
            for l in lines:
                l_clamped = l[:max_line_w]
                sys.stdout.write(f"\r{l_clamped}\033[K\n")
            sys.stdout.flush()
            if is_final:
                rendered_lines = 0
            else:
                rendered_lines = len(lines)
        
        if verbose:
            render_live_status()

        try:
            from .sources import PwlSource, PulseSource
            from .simulator import MolmemSimulator
            MolmemSimulator._in_warmup = True
            
            import multiprocessing
            import queue as py_queue
            result_queue = py_queue.Queue() if is_interactive_notebook() else multiprocessing.Queue()
            
            if nbc_before > 0 or is_interactive_notebook():
                # --- WARM FAST PATH: Link pre-compiled bytecode directly in MainProcess in <0.05s (Zero worker spawns) ---
                pycache_dir = os.path.join(_LIB_DIR, '__pycache__')
                has_device_bc = len(glob.glob(os.path.join(pycache_dir, 'device_jit*.nbc'))) > 0
                has_solver_bc = len(glob.glob(os.path.join(pycache_dir, 'solver*.nbc'))) > 0
                has_prog_bc = len(glob.glob(os.path.join(pycache_dir, 'programmer_solver*.nbc'))) > 0
                csrc_hash_f = os.path.join(_LIB_DIR, 'csrc', '.csrc_hash')
                csrc_d = os.path.join(_LIB_DIR, 'csrc')
                has_gpu_bc = os.path.isfile(csrc_hash_f) and os.path.exists(csrc_d) and any(
                    f.startswith("molmem_cuda_solver") and (f.endswith(".pyd") or f.endswith(".so"))
                    for f in os.listdir(csrc_d)
                )

                # 1. Device IV & Spline
                try:
                    t0_k = time.perf_counter()
                    from .device_jit import jit_interp_O1, jit_getIGeq
                    dummy_arr = np.zeros(10, dtype=np.float64)
                    jit_interp_O1(0.5, dummy_arr, 33040.0)
                    jit_getIGeq(0.5, 1e-4, 1.0, 0.5, 10.0)
                    dt_k = time.perf_counter() - t0_k
                    results["Device IV & Spline Interpolation"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_device_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Device IV & Spline Interpolation:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Device IV & Spline Interpolation"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 2. ODE Predictor
                try:
                    t0_k = time.perf_counter()
                    from .device import _ensure_jit
                    _ensure_jit()
                    dt_k = time.perf_counter() - t0_k
                    results["Arrhenius Thermal ODE Predictor"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_device_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Arrhenius Thermal ODE Predictor:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Arrhenius Thermal ODE Predictor"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 3. Nodal Assembly
                try:
                    t0_k = time.perf_counter()
                    from .solver import jit_assemble_and_solve_dense
                    gMat = np.zeros((2, 2), dtype=np.float64)
                    iRes = np.zeros(2, dtype=np.float64)
                    vCurr = np.zeros(2, dtype=np.float64)
                    jit_assemble_and_solve_dense(
                        2, 1, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
                        np.array([1e-4], dtype=np.float64), np.array([1e-5], dtype=np.float64),
                        np.array([0], dtype=np.int32), np.array([1e-4], dtype=np.float64), np.array([0.0], dtype=np.float64),
                        vCurr, gMat, iRes
                    )
                    dt_k = time.perf_counter() - t0_k
                    results["Newton-Raphson Nodal Assembly"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_solver_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Newton-Raphson Nodal Assembly:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Newton-Raphson Nodal Assembly"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 4. PWL Vectorizer
                try:
                    t0_k = time.perf_counter()
                    from .solver import jit_eval_dynamic_sources, jit_freeze_1t1r_vdevs, jit_build_full_master_schedule
                    dyn_types = np.zeros(1, dtype=np.int32)
                    dyn_dc = np.zeros(1, dtype=np.float64)
                    dyn_pulse = np.zeros((1, 5), dtype=np.float64)
                    dyn_pwl_t = np.zeros((1, 1), dtype=np.float64)
                    dyn_pwl_v = np.zeros((1, 1), dtype=np.float64)
                    dyn_mask = np.zeros(1, dtype=bool)
                    dyn_cache = np.zeros(1, dtype=np.int32)
                    dyn_G = np.zeros(1, dtype=np.float64)
                    dyn_I = np.zeros(1, dtype=np.float64)
                    jit_eval_dynamic_sources(0.0, dyn_types, dyn_dc, dyn_pulse, dyn_pwl_t, dyn_pwl_v, dyn_mask, dyn_cache, dyn_G, dyn_I, 1e-12)
                    jit_freeze_1t1r_vdevs(1, np.array([0], dtype=np.int32), dyn_G, 1e-9, np.zeros(1, dtype=np.float64))
                    _dummy_edges = np.array([0.0, 1e-6], dtype=np.float64)
                    _dummy_active = np.array([True], dtype=bool)
                    jit_build_full_master_schedule(_dummy_edges, _dummy_active, 1e-9, 1e9)
                    dt_k = time.perf_counter() - t0_k
                    results["Dynamic PWL Signal Vectorizer"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_solver_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Dynamic PWL Signal Vectorizer:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Dynamic PWL Signal Vectorizer"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 5. 3D Pulse Map
                try:
                    t0_k = time.perf_counter()
                    from .programmer_solver import (
                        jit_calibrate_grid_cpu, 
                        jit_compute_conductance_map, 
                        jit_compute_min_pulses,
                        jit_cpu_device_updates_parallel,
                        jit_trace_pulse_history,
                        jit_predict_pulses_matrix
                    )
                    _dummy_v = np.array([1.0], dtype=np.float64)
                    _dummy_pw = np.array([80e-9], dtype=np.float64)
                    jit_calibrate_grid_cpu(
                        _dummy_v, _dummy_pw, True, 2,
                        33200.0, 0.0, 0.0,
                        0.1, 1.0, 0.05, 0.5, 0.3, 0.0, 0.0, 0.0, 33200.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 1.0, 1.0
                    )
                    _dummy_g_in = np.zeros(2, dtype=np.float64)
                    _dummy_g_lut = np.zeros(2, dtype=np.float64)
                    jit_compute_conductance_map(
                        _dummy_g_in, 0.5, 33200.0, 1.0, 0.0, 0.0,
                        _dummy_g_lut, _dummy_g_lut, _dummy_g_lut, 0.0, 1.0
                    )
                    _dummy_pulse_map = np.ones((1, 1), dtype=np.float64)
                    _dummy_shifts = np.array([0.5], dtype=np.float64)
                    jit_compute_min_pulses(_dummy_pulse_map, _dummy_shifts)

                    _dummy_1d_f = np.zeros(1, dtype=np.float64)
                    _dummy_1d_i = np.zeros(1, dtype=np.int32)
                    _dummy_lut = np.zeros(2, dtype=np.float64)
                    jit_cpu_device_updates_parallel(
                        1, _dummy_1d_i, _dummy_1d_f, _dummy_1d_f,
                        _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.full(1, 298.15, dtype=np.float64),
                        _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
                        _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
                        _dummy_1d_i, _dummy_1d_f, np.full(1, 0.1, dtype=np.float64), np.full(1, 1e4, dtype=np.float64), np.full(1, 1e-7, dtype=np.float64),
                        np.full(1, 0.05, dtype=np.float64), np.full(1, 0.05, dtype=np.float64), np.full(1, 0.5, dtype=np.float64), np.full(1, 0.3, dtype=np.float64),
                        np.full(1, 0.5, dtype=np.float64), _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.full(1, 33200.0, dtype=np.float64), _dummy_1d_f,
                        np.ones(1, dtype=np.float64), _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64), _dummy_1d_f,
                        _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64),
                        _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f,
                        _dummy_lut, _dummy_lut, _dummy_lut, _dummy_lut, _dummy_1d_f
                    )
                    jit_trace_pulse_history(
                        0.0, 1, _dummy_1d_i, _dummy_1d_f, _dummy_1d_f,
                        _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.full(1, 298.15, dtype=np.float64),
                        _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
                        _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
                        _dummy_1d_i, _dummy_1d_f, np.full(1, 0.1, dtype=np.float64), np.full(1, 1e4, dtype=np.float64), np.full(1, 1e-7, dtype=np.float64),
                        np.full(1, 0.05, dtype=np.float64), np.full(1, 0.05, dtype=np.float64), np.full(1, 0.5, dtype=np.float64), np.full(1, 0.3, dtype=np.float64),
                        np.full(1, 0.5, dtype=np.float64), _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.full(1, 33200.0, dtype=np.float64), _dummy_1d_f,
                        np.ones(1, dtype=np.float64), _dummy_1d_f, np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64), _dummy_1d_f,
                        _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, np.ones(1, dtype=np.float64),
                        _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f, _dummy_1d_f,
                        _dummy_lut, _dummy_lut, _dummy_lut, _dummy_lut, 0.5
                    )
                    _dummy_2x2_f = np.zeros((2, 2), dtype=np.float64)
                    _dummy_2x2_i = np.zeros((2, 2), dtype=np.int32)
                    _dummy_1x1_f = np.ones((1, 1), dtype=np.float64)
                    _dummy_lut1 = np.zeros(2, dtype=np.float64)
                    jit_predict_pulses_matrix(
                        _dummy_2x2_f, _dummy_2x2_f, False, 0.005,
                        _dummy_1x1_f, _dummy_1x1_f,
                        _dummy_lut1, _dummy_lut1,
                        _dummy_lut1, _dummy_lut1,
                        1.0, 1.0,
                        1.0, 0.0,
                        _dummy_1x1_f, _dummy_1x1_f,
                        np.array([1.0], dtype=np.float64), np.array([-1.0], dtype=np.float64), np.array([80e-9], dtype=np.float64),
                        _dummy_2x2_i, _dummy_2x2_i,
                        1.0, -1.0, 80e-9, 80e-9,
                        _dummy_2x2_i, _dummy_2x2_f, _dummy_2x2_f,
                        _dummy_2x2_i, _dummy_2x2_f, _dummy_2x2_f
                    )
                    dt_k = time.perf_counter() - t0_k
                    results["3D Pulse Map Inverse Grid Search"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_prog_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] 3D Pulse Map Inverse Grid Search:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["3D Pulse Map Inverse Grid Search"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 6. PyTorch JIT / GPU Warmup
                try:
                    _run_torch_jit_bootstrap(result_queue)
                    msg = result_queue.get(timeout=1.0)
                    if ":" in msg:
                        _, res_str = msg.split(":", 1)
                        results["PyTorch TorchScript JIT Engine"] = res_str
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] PyTorch TorchScript JIT Engine:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["PyTorch TorchScript JIT Engine"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 7. Sequential Solver
                try:
                    t0_k = time.perf_counter()
                    sim_warm = MolmemSimulator(instanceName="JIT_Warmup", device='cpu')
                    sim_warm.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False, UpdatesPer="Device")
                    sim_warm.programmer = None
                    sim_warm.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
                    sim_warm.addDcSource(col=1, voltage=0.0)
                    sim_warm.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

                    sim_warm_1 = MolmemSimulator(instanceName="JIT_Warmup_1x1", device='cpu')
                    sim_warm_1.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
                    sim_warm_1.programmer = None
                    sim_warm_1.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
                    sim_warm_1.addDcSource(col=1, voltage=0.0)
                    sim_warm_1.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

                    dt_k = time.perf_counter() - t0_k
                    results["Sequential Nodal Circuit Solver"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_solver_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Sequential Nodal Circuit Solver:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Sequential Nodal Circuit Solver"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 8. Parallel Solver
                try:
                    t0_k = time.perf_counter()
                    sim2 = MolmemSimulator(instanceName="JIT_Warmup_Parallel", device='cpu')
                    sim2.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False)
                    sim2.optimal_cores = 2
                    sim2.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=[0.0, 160e-9], v_points=[0.1, 0.1]))
                    sim2.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

                    sim2_1 = MolmemSimulator(instanceName="JIT_Warmup_Parallel_1x1", device='cpu')
                    sim2_1.addCrossbarMatrix(rows=1, cols=1, detailedPrint=False)
                    sim2_1.optimal_cores = 2
                    sim2_1.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=[0.0, 160e-9], v_points=[0.1, 0.1]))
                    sim2_1.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)

                    dt_k = time.perf_counter() - t0_k
                    results["Parallel Workqueue Multi-Core Solver"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_solver_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Parallel Workqueue Multi-Core Solver:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Parallel Workqueue Multi-Core Solver"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                # 9. Quantized Inference Calibration
                try:
                    t0_k = time.perf_counter()
                    sim3 = MolmemSimulator(instanceName="JIT_Warmup_Cal", device='cpu')
                    sim3.addCrossbarMatrix(rows=2, cols=2, detailedPrint=False, weightBits=8, inputBits=8, outputBits=8, max_vWrite=1.0, max_pw=80e-9)
                    sim3.programmer = None
                    sim3.addVoltageSource(row=1, sourceObj=PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9))
                    sim3.addDcSource(col=1, voltage=0.0)
                    sim3.run(tEnd=160e-9, min_dt=1e-12, tol=1e-3, appendHistory=False, maxIter=100, title=None, recordHistory=False)
                    dt_k = time.perf_counter() - t0_k
                    results["Quantized Inference Calibration"] = f"Succeeded (Loaded from cache in {dt_k:.2f}s)" if has_solver_bc else (f"Succeeded (in {dt_k:.1f}s)" if dt_k >= 0.5 else f"Succeeded (in {dt_k:.2f}s)")
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        log_exception_file(f"[JIT_MAIN_ERROR] Quantized Inference Calibration:\n{tb}")
                    except Exception:
                        pass
                    err_msg = str(e).strip().split('\n')[0]
                    results["Quantized Inference Calibration"] = f"Failed ({type(e).__name__}: {err_msg})" if err_msg else f"Failed ({type(e).__name__})"
                completed_steps += 1
                
                render_live_status()
            else:
                # --- FRESH COMPILATION PATH: 2-Stage Cascading (Foundation Leaf Kernels -> Outer Solvers) ---
                has_device_bc = False
                has_solver_bc = False
                has_prog_bc = False
                has_gpu_bc = False
                stage1_workers = [
                    multiprocessing.Process(target=_run_device_iv_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_IV"),
                    multiprocessing.Process(target=_run_device_ode_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_ODE"),
                    multiprocessing.Process(target=_run_nodal_assembly_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_Nodal"),
                    multiprocessing.Process(target=_run_pwl_vectorizer_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_PWL"),
                    multiprocessing.Process(target=_run_weight_prog_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_Prog"),
                    threading.Thread(target=_run_torch_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_Torch"),
                ]
                for idx, w in enumerate(stage1_workers):
                    w.daemon = True
                    try:
                        from .sys_utils import log_to_run_log_only
                        w_type = "Thread" if isinstance(w, threading.Thread) else "Subprocess (multiprocessing)"
                        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning {w_type}: {w.name} | Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
                    except Exception:
                        pass
                    w.start()

                pending_stage1 = len(stage1_workers)
                while pending_stage1 > 0:
                    try:
                        msg = result_queue.get(timeout=0.1)
                        if ":" in msg:
                            key_name, res_str = msg.split(":", 1)
                            results[key_name] = res_str
                            completed_steps += 1
                            pending_stage1 -= 1
                            render_live_status()
                    except Exception:
                        alive = any(w.is_alive() for w in stage1_workers)
                        if not alive and result_queue.empty():
                            break

                for w in stage1_workers:
                    if w.is_alive():
                        w.join(timeout=0.5)

                results["Sequential Nodal Circuit Solver"] = "[Compiling Outer Solver...]"
                results["Parallel Workqueue Multi-Core Solver"] = "[Compiling Outer Solver...]"
                results["Quantized Inference Calibration"] = "[Compiling Outer Solver...]"
                if need_gpu_ext:
                    results["GPU Extension Solver Warmup"] = "[Warming GPU Context...]"
                render_live_status()

                stage2_workers = [
                    multiprocessing.Process(target=_run_dense_solver_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_0"),
                    multiprocessing.Process(target=_run_parallel_solver_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_1"),
                    multiprocessing.Process(target=_run_cal_solver_jit_bootstrap, args=(result_queue,), name="MOLMEM_JIT_Worker_2"),
                ]
                for idx, w in enumerate(stage2_workers):
                    w.daemon = True
                    try:
                        from .sys_utils import log_to_run_log_only
                        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Subprocess (multiprocessing): MOLMEM_JIT_Stage2_Worker_{idx} | Platform=CPU (Parallel JIT compiler worker) | Threads=1 | Core Affinity=Process Managed")
                    except Exception:
                        pass
                    w.start()

                pending_stage2 = len(stage2_workers)
                while pending_stage2 > 0:
                    try:
                        msg = result_queue.get(timeout=0.1)
                        if ":" in msg:
                            key_name, res_str = msg.split(":", 1)
                            results[key_name] = res_str
                            completed_steps += 1
                            pending_stage2 -= 1
                            render_live_status()
                    except Exception:
                        alive = any(w.is_alive() for w in stage2_workers)
                        if not alive and result_queue.empty():
                            break

                for w in stage2_workers:
                    if w.is_alive():
                        w.join(timeout=0.5)
            
            # Step 7: GPU Extension Solver Warmup (if GPU is active)
            if has_gpu and need_gpu_ext:
                gpu_th = threading.Thread(
                    target=_run_gpu_ext_warmup_worker, 
                    args=(result_queue,), 
                    name="MOLMEM_JIT_Worker_GPU_Ext"
                )
                gpu_th.daemon = True
                gpu_th.start()
                gpu_th.join(timeout=15)
                
                success = False
                res_msg = None
                if gpu_th.is_alive():
                    res_msg = "Timeout (>15s)"
                else:
                    try:
                        t_wait_end = time.perf_counter() + 2.0
                        while time.perf_counter() < t_wait_end and not success:
                            try:
                                msg = result_queue.get(timeout=0.2)
                                if "GPU Extension Solver Warmup" in msg:
                                    parts = msg.split(":")
                                    if len(parts) >= 3 and parts[1] == "SUCCESS":
                                        dt_val = float(parts[2])
                                        tag_gpu = f"Succeeded (Loaded from cache in {dt_val:.2f}s)" if has_gpu_bc else (f"Succeeded (in {dt_val:.1f}s)" if dt_val >= 0.5 else f"Succeeded (in {dt_val:.2f}s)")
                                        results["GPU Extension Solver Warmup"] = tag_gpu
                                        success = True
                                        break
                                    elif len(parts) >= 3 and parts[1] == "FAIL":
                                        res_msg = ":".join(parts[2:]).strip()
                                        break
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if not success and res_msg is None:
                        res_msg = "Verification Failed"

                if not success:
                    results["GPU Extension Solver Warmup"] = f"Disabled ({res_msg})"
                    # Only disable HAS_CPP_EXTENSION if hardware architecture is incompatible
                    if res_msg and ("mismatch" in res_msg.lower() or "not compatible" in res_msg.lower() or "architecture" in res_msg.lower()):
                        try:
                            from . import torch_simulator
                            torch_simulator.HAS_CPP_EXTENSION = False
                        except Exception:
                            pass
                
                completed_steps += 1
                render_live_status()
            else:
                if "GPU Extension Solver Warmup" not in results or results["GPU Extension Solver Warmup"].startswith("["):
                    results["GPU Extension Solver Warmup"] = gpu_skip_status
                completed_steps += 1
            
            _jit_compiled = True
            if nbc_before > 0:
                _linker_warmed_up = True
            t_elapsed = time.perf_counter() - t0_total
            all_cached = (has_device_bc and has_solver_bc and has_prog_bc and (not need_gpu_ext or has_gpu_bc))
            lbl = "loaded from JIT cache" if all_cached else "compiled from scratch"
            render_live_status(status_hdr=f"Done ({lbl} in {t_elapsed:.1f}s)", is_final=True)

            # 1. Hardware Matrix Calibration (3D Optimal Pulse Mapping)
            try:
                from .simulator import MolmemSimulator
                MolmemSimulator._in_warmup = False
                sim_cal = MolmemSimulator(device='cpu', backend='numba')
                sim_cal._in_warmup = False
                sim_cal._is_calibration_helper = True
                sim_cal.vRead = 0.1
                sim_cal.addCrossbarMatrix(rows=2, cols=2, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9)
                if hasattr(sim_cal, 'programmer') and sim_cal.programmer is not None:
                    sim_cal.programmer.sim._in_warmup = False
                    sim_cal.programmer.calibrate()
                    if verbose:
                        cal_v = getattr(sim_cal.programmer, 'max_vWrite', 5.0)
                        cal_pw = getattr(sim_cal.programmer, 'max_pw', 320e-9)
                        bar = '#' * 20
                        sys.stdout.write(f"  > Hardware Matrix Calibration           : [{bar}] 100% | Loaded Cache: {cal_v:.2f}V, {cal_pw*1e9:.0f}ns\n")
                        sys.stdout.flush()
            except Exception as e:
                if verbose:
                    sys.stdout.write(f"\n  [!] Hardware Calibration Initialization Failed: {e}\n")
                    sys.stdout.flush()

            # 2. Hardware 2x2 Thread Performance Profiling & LLVM Linker Warmup
            try:
                ensure_bootstrap_profiled(verbose=verbose)
            except Exception:
                pass
        except Exception as e:
            if verbose:
                sys.stdout.write(f"\n  [!] CPU JIT compile critical error occurred: {e}\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\n[!] Hardware Systems Initialization > Failed: {e}\n")
                sys.stdout.flush()
            raise
        finally:
            try:
                from .simulator import MolmemSimulator
                MolmemSimulator._in_warmup = False
            except Exception:
                pass


import signal
import threading

_telemetry_printed = False

def kill_child_processes():
    global _telemetry_printed
    import sys
    import os
    
    is_main = is_main_proc
    if is_main and not _telemetry_printed:
        _telemetry_printed = True
        try:
            import psutil
            p = psutil.Process()
            children = p.children(recursive=True)
            
            # Filter for non-dummy, non-main active user threads
            active_user_threads = [
                t for t in threading.enumerate() 
                if t != threading.main_thread() 
                and not isinstance(t, getattr(threading, '_DummyThread', ()))
                and not t.name.startswith("Dummy-")
            ]
            
            if children or active_user_threads:
                if children:
                    total_cpus = psutil.cpu_count() or 1
                    sys.stderr.write("\n[*] Terminating active child processes:\n")
                    for child in children:
                        try:
                            cmd = child.cmdline()
                            role = child.name()
                            if cmd:
                                if "SubprocessCal" in cmd or any("SubprocessCal" in arg for arg in cmd):
                                    role = "SubprocessCal (CPU profiling sweep)"
                                elif any("calibrate" in arg for arg in cmd):
                                    role = "ProcessPoolWorker (Calibration target solver)"
                            aff_str = "OS Managed"
                            sys.stderr.write(f"  - PID {child.pid:<6d} : {role} | {aff_str}\n")
                        except Exception:
                            sys.stderr.write(f"  - PID {child.pid:<6d} : Unknown child process\n")
                
                if active_user_threads:
                    sys.stderr.write("\n[*] Terminating active Python threads:\n")
                    for t in active_user_threads:
                        sys.stderr.write(f"  - Thread: {t.name:<20s} [ID: {t.ident}] [Daemon: {t.daemon}]\n")
                sys.stderr.write("==================================================================================================\n")
                sys.stderr.flush()
        except Exception:
            pass
            
    try:
        import psutil
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
    except Exception:
        pass

def native_crash_handler(signum, frame):
    import sys
    import os
    
    sig_names = {
        signal.SIGSEGV: "SIGSEGV (Segmentation Fault)",
        signal.SIGABRT: "SIGABRT (Abort/Execution Failure)",
        signal.SIGILL: "SIGILL (Illegal Instruction)",
        signal.SIGFPE: "SIGFPE (Floating Point Exception)",
        signal.SIGTERM: "SIGTERM (Termination Request)"
    }
    name = sig_names.get(signum, f"Signal {signum}")
    
    # Use direct low-level os.write(2, ...) to avoid Python BufferedWriter stream locks and reentrancy exceptions
    def _safe_write(text):
        try:
            os.write(2, text.encode("utf-8", errors="replace"))
        except Exception:
            pass

    width = 78
    border = "=" * width
    _safe_write(f"\n{border}\n")
    _safe_write(f"  [CRITICAL ERROR] NATIVE SYSTEM CRASH DETECTED ({name.upper()})\n")
    _safe_write(f"{border}\n")
    _safe_write("  The compiled C++ solver/ROCm extension has encountered a native hardware or memory fault.\n")
    _safe_write("  Common causes include:\n")
    _safe_write("    1. Attempted out-of-bounds node or device index configuration.\n")
    _safe_write("    2. Insufficient GPU workspace/VRAM memory limits.\n")
    _safe_write("    3. Driver-level memory access violation or compilation mismatch.\n")
    _safe_write(f"{border}\n")
    _safe_write("  Active Python Call Stack:\n")
    
    try:
        import faulthandler
        faulthandler.dump_traceback(file=2)
    except Exception:
        _safe_write("    [Info] Traceback unavailable.\n")
    _safe_write(f"{border}\n\n")
    
    crash_msg = (
        f"CRITICAL ERROR: NATIVE SYSTEM CRASH DETECTED ({name.upper()})\n"
        "The compiled C++ solver/ROCm extension has encountered a native hardware or memory fault.\n"
        "Common causes include:\n"
        "  1. Attempted out-of-bounds node or device index configuration.\n"
        "  2. Insufficient GPU workspace/VRAM memory limits.\n"
        "  3. Driver-level memory access violation or compilation mismatch.\n"
    )
    try:
        log_exception_file(crash_msg)
    except Exception:
        pass
    
    try:
        kill_child_processes()
    except Exception:
        pass
    try:
        os.kill(os.getpid(), 9)
    except Exception:
        pass
    os._exit(signum)
    
def sigterm_clean_handler(signum, frame):
    """
    Handles standard POSIX termination requests (e.g. kill, container stop, pool teardown).
    Cleanly cascades termination to child processes without reporting false hardware crash banners.
    """
    try:
        kill_child_processes()
    except Exception:
        pass
    os._exit(0)

if not is_interactive_notebook():
    try:
        signal.signal(signal.SIGSEGV, native_crash_handler)
        signal.signal(signal.SIGABRT, native_crash_handler)
        signal.signal(signal.SIGILL, native_crash_handler)
        signal.signal(signal.SIGFPE, native_crash_handler)
        signal.signal(signal.SIGTERM, sigterm_clean_handler)
    except Exception:
        pass

_watchdog_last_tick = 0.0

def watchdog_loop(stop_event):
    global _watchdog_last_tick
    import sys
    import os
    import time
    import torch

    has_torch_cuda = False
    try:
        has_torch_cuda = torch.cuda.is_available()
    except Exception:
        pass

    last_hang_check = time.time()
    
    # 20 Hz (50ms) background monitor loop
    while not stop_event.wait(timeout=0.05):
        now = time.time()
        
        # 1. Continuous GPU Heartbeat: Non-blocking stream query to refresh WDDM compute counters
        if has_torch_cuda:
            try:
                stream = torch.cuda.current_stream()
                if stream is not None:
                    stream.query()
            except Exception:
                pass
                
        # 2. Watchdog Hang Check (evaluated once every 1.0 second)
        if (now - last_hang_check) >= 1.0:
            last_hang_check = now
            last_jit_tick = 0.0
            active_tick = max(_watchdog_last_tick, last_jit_tick)
            if active_tick > 0.0 and (now - active_tick) > 60.0:
                import shutil
                width = min(100, shutil.get_terminal_size().columns - 2) if shutil else 78
                border = "=" * width
                print(f"\n{border}")
                print("  [CRITICAL ERROR] SIMULATION WATCHDOG TIMEOUT (SOLVER HANG DETECTED)")
                print(border)
                print("  The simulator has been blocked with zero progress for more than 60 seconds.")
                print("  This usually indicates a GPU driver crash, hardware lock, or thread deadlock.")
                
                try:
                    # Bypass the print deduplication guard to ensure the watchdog print always triggers
                    import molmem_lib
                    if hasattr(molmem_lib, '_dump_active_threads_diagnostics'):
                        # Temporarily unlock the print guard for the watchdog crash output
                        molmem_lib._ctrl_c_reported = False
                        molmem_lib._dump_active_threads_diagnostics(sys.stdout)
                except Exception:
                    pass
                    
                print("  Terminating the process immediately to prevent resources from hanging...")
                print(border)
                sys.stdout.flush()
                sys.stderr.flush()
                
                kill_child_processes()
                os.kill(os.getpid(), 9)
                os._exit(1)

def handle_errors(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _watchdog_last_tick
        _watchdog_last_tick = time.time()
        sim_inst = args[0] if len(args) > 0 else None
        is_warmup = (
            getattr(SimulatorSystemMixin, '_in_warmup', False) or
            getattr(sim_inst, '_in_warmup', False) or
            getattr(getattr(sim_inst, '__class__', None), '_in_warmup', False) or
            getattr(sim_inst, '_is_calibration_helper', False)
        )
        sim_dev = getattr(sim_inst, 'device', 'cpu')
        is_gpu = str(sim_dev).lower() in ('gpu', 'cuda', 'rocm') and (os.environ.get("MOLMEM_CPU_ONLY") != "1")
        use_watchdog = is_gpu and not is_warmup
        if use_watchdog:
            stop_event = threading.Event()
            try:
                from .sys_utils import log_to_run_log_only
                log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Thread: watchdog_loop | Target=GPU Watchdog Monitor | Threads=1 | Core Affinity=Process Managed")
            except Exception:
                pass
            w_thread = threading.Thread(target=watchdog_loop, args=(stop_event,), daemon=True)
            w_thread.name = "watchdog_loop"
            w_thread.start()
        
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            if is_warmup:
                raise
            import traceback
            import sys
            import shutil
            
            e_str = str(e)
            if isinstance(e, KeyboardInterrupt) or "KeyboardInterrupt" in e_str:
                if is_interactive_notebook():
                    raise e
                # Delegate to registered SIGINT handler for graceful cleanup (GPU sync, pool shutdown, subprocess kill)
                handler = signal.getsignal(signal.SIGINT)
                if callable(handler):
                    try:
                        handler(signal.SIGINT, None)
                    except BaseException:
                        pass
                kill_child_processes()
                os.kill(os.getpid(), 9)
                os._exit(1)
                
            try:
                width = min(100, shutil.get_terminal_size().columns - 2)
            except Exception:
                width = 78
            border = "=" * width
            
            func_name = f"{func.__module__}.{func.__name__}"
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            tb_text = "".join(tb_lines)
            
            err_title = "CRITICAL SIMULATION ERROR"
            suggestions = []
            
            if isinstance(e, MemoryError) or "out of memory" in e_str.lower() or "oom" in e_str.lower():
                err_title = "SYSTEM OUT OF MEMORY (OOM) DETECTED"
                suggestions = [
                    "Reduce the simulation timeline/steps (decrease pulses, sweeps, or tEnd).",
                    "If recordHistory=True is enabled, disable it if step-by-step history is not required.",
                    "Reduce the size of the crossbar array (rows/columns) being simulated.",
                    "Reduce GPU batch slice size limit (gpu_slice_limit / gpu_read_slice_limit)."
                ]
            elif isinstance(e, FloatingPointError) or "numerical instability" in e_str.lower() or "diverged" in e_str.lower():
                err_title = "NUMERICAL INSTABILITY / DIVERGENCE DETECTED"
                suggestions = [
                    "Try lowering the convergence tolerance (e.g. tol=1e-4 or tol=1e-5).",
                    "Decrease the minimum timestep size (min_dt).",
                    "Check for steep input voltage transitions or extreme physical parameters."
                ]
            elif isinstance(e, ZeroDivisionError) or "singular" in e_str.lower() or "matrix detected" in e_str.lower():
                err_title = "SINGULAR MATRIX DETECTED (CG SOLVER DIVERGENCE)"
                suggestions = [
                    "Check your netlist/crossbar topology definition.",
                    "Ensure all nodes have a conductive path to ground or voltage sources.",
                    "Ensure no nodes are short-circuited directly with zero series resistance."
                ]
            elif "progress locked" in e_str.lower() or "stiffness lock" in e_str.lower():
                err_title = "SIMULATION TIME PROGRESS LOCK"
                suggestions = [
                    "The time integrator is locked due to an extremely stiff state transition.",
                    "Consider adjusting the device physical parameters to be less abrupt.",
                    "Check if the input pulses are too steep or have unrealistic rise times."
                ]
            elif any(k in e_str.lower() for k in ["cuda error", "hip error", "device error", "illegal memory access", "invalid device ordinal", "kernel launch error"]):
                err_title = "GPU DRIVER / HARDWARE EXECUTION ERROR"
                suggestions = [
                    "Verify GPU driver installation, hardware state, and compatibility with PyTorch.",
                    "Run with environment variable CUDA_LAUNCH_BLOCKING=1 to locate the exact failing operation.",
                    "Check GPU VRAM availability and thermal throttle status."
                ]
            elif isinstance(e, (ValueError, RuntimeError)) and any(k in e_str.lower() for k in ["shape mismatch", "size mismatch", "dimension mismatch", "must match number of rows"]):
                err_title = "TENSOR DIMENSION / SHAPE MISMATCH"
                suggestions = [
                    "Check input vector or matrix dimensions match crossbar rows and columns.",
                    "Ensure initial state tensor has matching batch and device dimensions."
                ]
            else:
                err_title = "UNEXPECTED RUNTIME EXCEPTION"
                suggestions = [
                    "Check the traceback below for the file and line number causing the issue.",
                    "If originating from ROCm/HIP, run with environment variable CUDA_LAUNCH_BLOCKING=1 to locate the exact failing kernel."
                ]
            
            # Gather Host RAM stats
            mem_info = ""
            try:
                import psutil
                virtual_mem = psutil.virtual_memory()
                total_gb = virtual_mem.total / (1024**3)
                avail_gb = virtual_mem.available / (1024**3)
                used_pct = virtual_mem.percent
                proc = psutil.Process(os.getpid())
                proc_gb = proc.memory_info().rss / (1024**3)
                mem_info = f"Host RAM  : {proc_gb:.2f} GB used by process | {avail_gb:.2f} GB / {total_gb:.2f} GB free ({used_pct}% system used)"
            except Exception:
                pass

            # Gather GPU VRAM stats
            gpu_info = ""
            if os.environ.get("MOLMEM_CPU_ONLY") != "1":
                try:
                    import torch
                    if torch.cuda.is_available():
                        active_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', str(torch.cuda.current_device())))
                        if active_gpu >= torch.cuda.device_count() or active_gpu < 0:
                            active_gpu = 0
                        device_name = torch.cuda.get_device_name(active_gpu)
                        allocated = torch.cuda.memory_allocated(active_gpu) / (1024**3)
                        reserved = torch.cuda.memory_reserved(active_gpu) / (1024**3)
                        max_allocated = torch.cuda.max_memory_allocated(active_gpu) / (1024**3)
                        gpu_info = f"GPU VRAM  : {allocated:.2f} GB allocated | {reserved:.2f} GB reserved | Max Peak: {max_allocated:.2f} GB ({device_name})"
                except Exception:
                    pass

            print(f"\n{border}")
            print(f"  [CRITICAL ERROR] {err_title}")
            print(border)
            print(f"  Function  : {func_name}")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Message   : {e_str}")
            if mem_info:
                print(f"  {mem_info}")
            if gpu_info:
                print(f"  {gpu_info}")
                
            # If OOM occurred, print PyTorch VRAM memory summary
            vram_summary = ""
            if os.environ.get("MOLMEM_CPU_ONLY") != "1" and ("out of memory" in e_str.lower() or "oom" in e_str.lower()):
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_summary = torch.cuda.memory_summary(abbreviated=True)
                        print("\n  CUDA Memory Summary (Abbreviated):")
                        for line in vram_summary.split('\n'):
                            print(f"    {line}")
                except Exception:
                    pass
            print(border)
            print("  Diagnostic Recommendations:")
            for i, sugg in enumerate(suggestions, 1):
                print(f"    {i}. {sugg}")
            print(border)
            print("  Detailed Traceback:")
            print(tb_text.strip())
            print(border)
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Write structured log file
            err_log_content = (
                f"CRITICAL ERROR: {err_title}\n"
                f"Function  : {func_name}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Message   : {e_str}\n"
            )
            if mem_info:
                err_log_content += f"{mem_info}\n"
            if gpu_info:
                err_log_content += f"{gpu_info}\n"
            if vram_summary:
                err_log_content += f"\nCUDA Memory Summary (Abbreviated):\n{vram_summary}\n"
            err_log_content += "\nDiagnostic Recommendations:\n"
            for i, sugg in enumerate(suggestions, 1):
                err_log_content += f"  {i}. {sugg}\n"
            err_log_content += f"\nDetailed Traceback:\n{tb_text.strip()}\n"
            
            log_exception_file(err_log_content)
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            kill_child_processes()
            if is_interactive_notebook():
                raise e
            os.kill(os.getpid(), 9)
            os._exit(1)
        finally:
            if use_watchdog:
                stop_event.set()
                try:
                    from .sys_utils import log_to_run_log_only
                    log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Thread watchdog_loop terminated: GPU Watchdog Monitor | Threads=1 | Core Affinity=Process Managed")
                except Exception:
                    pass
    return wrapper

handle_oom = handle_errors


_IS_WINDOWS_ROCM_CACHED = None

def is_windows_rocm():
    """Detects if we are running in a PyTorch ROCm/HIP environment on Windows."""
    global _IS_WINDOWS_ROCM_CACHED
    if _IS_WINDOWS_ROCM_CACHED is not None:
        return _IS_WINDOWS_ROCM_CACHED
        
    if sys.platform != 'win32' or not is_main_proc or os.environ.get("MOLMEM_CPU_ONLY") == "1":
        _IS_WINDOWS_ROCM_CACHED = False
        return False
        
    try:
        import torch
        if hasattr(torch, "version") and torch.version.hip is not None:
            _IS_WINDOWS_ROCM_CACHED = True
            return True
    except Exception:
        pass
    _IS_WINDOWS_ROCM_CACHED = False
    return False


def get_hardware_info_str():
    import sys
    import platform
    try:
        import psutil
        virtual_mem = psutil.virtual_memory()
        total_gb = virtual_mem.total / (1024**3)
        avail_gb = virtual_mem.available / (1024**3)
        ram_str = f"{total_gb:.2f} GB Total | {avail_gb:.2f} GB Free"
    except Exception:
        ram_str = "Unknown"
        
    cpu_str = platform.processor() or "Unknown CPU"
    os_str = f"{platform.system()} {platform.release()} (arch: {platform.machine()})"
    
    gpu_str = "Not Available"
    if os.environ.get("MOLMEM_CPU_ONLY") != "1":
        try:
            import torch
            if torch.cuda.is_available():
                active_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', str(torch.cuda.current_device())))
                if active_gpu >= torch.cuda.device_count() or active_gpu < 0:
                    active_gpu = 0
                gpu_str = f"{torch.cuda.get_device_name(active_gpu)} (VRAM: {torch.cuda.get_device_properties(active_gpu).total_memory / (1024**3):.2f} GB)"
        except Exception:
            pass
        
    # Attempt to retrieve precise hardware info from cached _hardware_config
    try:
        import sys
        molmem_lib = sys.modules.get("molmem_lib")
        cfg = getattr(molmem_lib, "_hardware_config", None)
        if cfg:
            cpu_str = cfg.get("cpu_name", cpu_str)
            if cfg.get("gpu_detected"):
                gpu_str = f"{cfg.get('gpu_name')} (VRAM: {cfg.get('gpu_vram', 0.0):.2f} GB)"
    except Exception:
        pass
        
    info = [
        f"  OS Platform     : {os_str}",
        f"  Python Version  : {sys.version.split()[0]}",
        f"  Host CPU        : {cpu_str}",
        f"  System RAM      : {ram_str}",
        f"  Graphics GPU    : {gpu_str}"
    ]
    return "\n".join(info)


def log_exception_file(error_content):
    if not is_logging_enabled():
        return
    import os
    import datetime
    try:
        log_dir = get_log_dir(auto_create=True)
        filepath = os.path.join(log_dir, "error_log.txt")
        hw_info = get_hardware_info_str()
        mode = "a" if os.path.exists(filepath) else "w"
        with open(filepath, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("==================================================\n")
            f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
            f.write("==================================================\n")
            f.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
            f.write(f"Hardware Info   :\n{hw_info}\n")
            f.write("==================================================\n\n")
            f.write(error_content)
    except Exception:
        pass


def handle_worker_exception(e, context_msg):
    import os
    import sys
    import datetime
    import traceback
    
    stack_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    pid = os.getpid()
    ppid = os.getppid()
    proc_name = multiprocessing.current_process().name
    
    error_details = (
        f"Critical Subprocess Exception:\n"
        f"  - Context             : {context_msg}\n"
        f"  - Active Process Name : {proc_name}\n"
        f"  - Process ID (PID)    : {pid}\n"
        f"  - Parent Process ID   : {ppid}\n"
        f"\n"
        f"Detailed Exception Traceback:\n"
        f"{stack_str}\n"
    )
    
    sys.stderr.write(f"\n[MOLMEM CRITICAL WORKER ERROR] {str(e)}\n\n{error_details}\n")
    sys.stderr.flush()
    if is_logging_enabled():
        try:
            err_dir = get_log_dir(auto_create=True)
            err_file = os.path.join(err_dir, "error_log.txt")
            mode = "a" if os.path.exists(err_file) else "w"
            with open(err_file, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n" + "=" * 50 + "\n\n")
                f.write("==================================================\n")
                f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
                f.write("==================================================\n")
                f.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
                f.write("Status          : CRITICAL WORKER EXCEPTION\n")
                f.write("==================================================\n\n")
                f.write(f"[MOLMEM CRITICAL WORKER ERROR] {str(e)}\n\n")
                f.write(error_details)
        except Exception:
            pass
        
    try:
        if ppid > 0:
            try:
                import psutil
                parent = psutil.Process(ppid)
                for child in parent.children(recursive=True):
                    if child.pid != os.getpid():
                        child.kill()
            except Exception:
                pass
            os.kill(ppid, 9)
    except Exception:
        pass
    os._exit(1)


import threading
_shared_log_file = None
_shared_log_refcount = 0
_shared_log_lock = threading.Lock()

class TeeLogger:
    def __init__(self, filepath, stream, mode="wb", name="TeeLogger"):
        global _shared_log_file, _shared_log_refcount
        with _shared_log_lock:
            if _shared_log_file is None:
                # Open with standard 64KB OS write buffering for high-throughput asynchronous writes
                _shared_log_file = open(filepath, mode, buffering=65536)
            _shared_log_refcount += 1
        self.file = _shared_log_file
        self.stream = stream
        import os
        self.log_dir = os.path.dirname(os.path.abspath(filepath))
        import queue
        import threading
        import atexit
        self.queue = queue.Queue()
        self.active = True
        self.thread = threading.Thread(target=self._write_loop, name=name)
        self.thread.daemon = True
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pid = os.getpid()
        spawn_msg = f"[{timestamp}] [WORKER_SPAWN] [PID {pid}] Spawning Thread: {name} | Target=Asynchronous Log Writer | Threads=1 | Core Affinity=Process Managed\n"
        init_msg = f"[{timestamp}] [WORKER_SPAWN] [PID {pid}] Thread {name} initialized: Asynchronous Log Writer | Threads=1 | Core Affinity=Process Managed\n"
        with _shared_log_lock:
            self.file.write(spawn_msg.encode('utf-8', errors='replace'))
            self.file.write(init_msg.encode('utf-8', errors='replace'))
            self.file.flush()
            
        self.thread.start()
        atexit.register(self.close)

    def _write_loop(self):
        import queue
        import os
        import glob
        import threading
        import time
        last_char = '\n'
        pending_cr = None
        is_stdout_thread = (threading.current_thread().name == "TeeLogger_stdout")
        
        def merge_tmp_files():
            if not is_stdout_thread:
                return
            try:
                tmp_files = glob.glob(os.path.join(self.log_dir, "run_log_pid_*.tmp"))
                if not tmp_files:
                    return
                merged_lines = []
                for tmp_file in tmp_files:
                    try:
                        with open(tmp_file, "r", encoding="utf-8", errors="replace") as f:
                            merged_lines.extend(f.readlines())
                        safe_remove_file(tmp_file)
                    except Exception:
                        pass
                if merged_lines:
                    def get_timestamp(line):
                        if line.startswith('['):
                            end_idx = line.find(']')
                            if end_idx != -1:
                                return line[1:end_idx]
                        return ""
                    merged_lines.sort(key=get_timestamp)
                    
                    with _shared_log_lock:
                        if self.file is not None and not self.file.closed:
                            for line in merged_lines:
                                self.file.write(line.encode('utf-8', errors='replace'))
                            self.file.flush()
            except Exception:
                pass
        
        while self.active or not self.queue.empty():
            try:
                raw_data = self.queue.get(timeout=0.1)
                if not raw_data:
                    self.queue.task_done()
                    continue
                # Fast prefix check bypasses regex scanning for plain text chunks
                if '\033[' in raw_data or '\x1b[' in raw_data:
                    data = _ansi_escape.sub('', raw_data)
                else:
                    data = raw_data
                try:
                    with _shared_log_lock:
                        if self.file is not None and not self.file.closed:
                            if data.startswith('\r'):
                                cleaned = data.lstrip('\r').rstrip('\r\n\t ')
                                if cleaned:
                                    pending_cr = cleaned
                            else:
                                if pending_cr:
                                    if last_char != '\n':
                                        self.file.write(b'\n')
                                    self.file.write((pending_cr + '\n').encode('utf-8', errors='replace'))
                                    pending_cr = None
                                    last_char = '\n'
                                    
                                log_data = data.replace('\r', '\n')
                                if last_char != '\n' and not log_data.startswith('\n'):
                                    self.file.write(b'\n')
                                self.file.write(log_data.encode('utf-8', errors='replace'))
                                if log_data:
                                    last_char = log_data[-1]
                except Exception:
                    pass
                self.queue.task_done()
            except queue.Empty:
                try:
                    with _shared_log_lock:
                        if self.file is not None and not self.file.closed:
                            self.file.flush()
                except Exception:
                    pass
                
        if pending_cr:
            try:
                with _shared_log_lock:
                    if self.file is not None and not self.file.closed:
                        if last_char != '\n':
                            self.file.write(b'\n')
                        self.file.write((pending_cr + '\n').encode('utf-8', errors='replace'))
                        self.file.flush()
            except Exception:
                pass
        merge_tmp_files()

    def write_file_only(self, data):
        if self.active:
            self.queue.put(data)

    def write(self, data):
        self.stream.write(data)
        if self.active:
            self.queue.put(data)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return hasattr(self.stream, 'isatty') and self.stream.isatty()

    def close(self):
        global _shared_log_file, _shared_log_refcount
        name = getattr(self, 'name', 'TeeLogger')
        try:
            log_to_run_log_only(f"Thread {name} terminated: Asynchronous Log Writer | Threads=1 | Core Affinity=Process Managed")
        except Exception:
            pass
        self.active = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            with _shared_log_lock:
                _shared_log_refcount -= 1
                if _shared_log_refcount <= 0 and _shared_log_file is not None:
                    try:
                        _shared_log_file.flush()
                        _shared_log_file.close()
                    except Exception:
                        pass
                    _shared_log_file = None
        except Exception:
            pass


def log_to_run_log_only(message: str) -> None:
    """Writes a message directly to the run log file only, bypassing console stdout."""
    if not is_logging_enabled():
        return
    import os
    import sys
    import datetime
    import random
    import time
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pid = os.getpid()
        if message.startswith('['):
            formatted = f"[{timestamp}] [PID {pid}] {message}\n"
        else:
            formatted = f"[{timestamp}] [WORKER_SPAWN] [PID {pid}] {message}\n"
        
        # Route through TeeLogger if active to serialize writes and avoid interleaving
        if hasattr(sys.stdout, 'write_file_only'):
            sys.stdout.write_file_only(formatted)
            return
            
        log_dir = get_log_dir(auto_create=True)
        is_child = not is_main_proc
        
        if is_child:
            filepath = os.path.join(log_dir, f"run_log_pid_{pid}.tmp")
            try:
                with open(filepath, "ab") as f:
                    f.write(formatted.encode('utf-8', errors='replace'))
            except Exception:
                pass
            return
            
        filepath = os.path.join(log_dir, "run_log.txt")
        global _shared_log_file, _shared_log_lock
        with _shared_log_lock:
            if _shared_log_file is not None and not _shared_log_file.closed:
                try:
                    _shared_log_file.write(formatted.encode('utf-8', errors='replace'))
                    _shared_log_file.flush()
                    return
                except Exception:
                    pass
            for _ in range(40):
                try:
                    with open(filepath, "ab") as f:
                        f.write(formatted.encode('utf-8', errors='replace'))
                    break
                except Exception:
                    time.sleep(0.001 + random.random() * 0.015)
    except Exception:
        pass


def _merge_child_logs():
    if not is_logging_enabled():
        return
    import os
    import glob
    import time
    import random
    
    global _shared_log_file, _shared_log_lock
    try:
        log_dir = get_log_dir(auto_create=False)
        if not os.path.exists(log_dir):
            return
            
        tmp_files = glob.glob(os.path.join(log_dir, "run_log_pid_*.tmp"))
        if not tmp_files:
            return
            
        merged_lines = []
        for tmp_file in tmp_files:
            try:
                with open(tmp_file, "r", encoding="utf-8", errors="replace") as f:
                    for l in f.readlines():
                        cleaned = l.replace('\x00', '')
                        if cleaned.strip():
                            merged_lines.append(cleaned)
                safe_remove_file(tmp_file)
            except Exception:
                pass
                
        if not merged_lines:
            return
            
        def get_timestamp(line):
            line_str = line.strip()
            if line_str.startswith('['):
                end_idx = line_str.find(']')
                if end_idx != -1:
                    return line_str[1:end_idx]
            return ""
            
        merged_lines.sort(key=get_timestamp)
        
        with _shared_log_lock:
            if _shared_log_file is not None and not _shared_log_file.closed:
                for line in merged_lines:
                    formatted = line if line.endswith('\n') else line + '\n'
                    _shared_log_file.write(formatted.encode('utf-8', errors='replace'))
                _shared_log_file.flush()
                return
                
            main_log = os.path.join(log_dir, "run_log.txt")
            for _ in range(40):
                try:
                    with open(main_log, "ab") as f:
                        for line in merged_lines:
                            formatted = line if line.endswith('\n') else line + '\n'
                            f.write(formatted.encode('utf-8', errors='replace'))
                    break
                except Exception:
                    time.sleep(0.002 + random.random() * 0.015)
    except Exception:
        pass


if is_main_proc:
    import atexit
    atexit.register(_merge_child_logs)

_disable_success_hook = False
_has_crashed = False

import collections
import linecache
import threading

_trace_lock = threading.Lock()
_trace_deque = collections.deque(maxlen=100)
_tracing_initialized = False

def _molmem_line_tracer(frame, event, arg):
    if event != 'line':
        return _molmem_line_tracer
    try:
        co = frame.f_code
        filename = co.co_filename
        if ('molmem_lib' in filename or 'PythonSimulator' in filename or 'UnifiedTestbenches' in filename or 'DeviceFitting' in filename or 'PhysicsFitting' in filename) and ('<frozen' not in filename):
            lineno = frame.f_lineno
            line_text = linecache.getline(filename, lineno).strip()
            if line_text:
                rel_path = os.path.basename(filename)
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                thread_name = threading.current_thread().name
                with _trace_lock:
                    _trace_deque.append((ts, os.getpid(), thread_name, rel_path, lineno, line_text))
    except Exception:
        pass
    return _molmem_line_tracer

def start_execution_tracer():
    global _tracing_initialized
    if _tracing_initialized:
        return
    _tracing_initialized = True
    try:
        sys.settrace(_molmem_line_tracer)
        threading.settrace(_molmem_line_tracer)
    except Exception:
        pass

def dump_last_100_lines_log():
    try:
        log_dir = get_log_dir(auto_create=True)
        out_file = os.path.join(log_dir, "last_100_lines.log")
        with _trace_lock:
            lines_snapshot = list(_trace_deque)
        
        if not lines_snapshot:
            return
            
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("====================================================================================================\n")
            f.write("                               MOLMEM EXECUTION TRACE: LAST 100 LINES                               \n")
            f.write(f"  Generated   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"  Total Lines : {len(lines_snapshot)}\n")
            f.write("====================================================================================================\n")
            f.write(f" {'TIME':<12} | {'PID':<6} | {'THREAD':<14} | {'LOCATION':<30} | {'CODE'}\n")
            f.write("-" * 100 + "\n")
            for ts, pid, thread_name, rel_path, lineno, line_text in lines_snapshot:
                loc = f"{rel_path}:{lineno}"
                f.write(f" [{ts}] | {pid:<6} | {thread_name:<14} | {loc:<30} | {line_text}\n")
            f.write("====================================================================================================\n")
    except Exception:
        pass

def init_cli_logger():
    if not is_logging_enabled():
        return
    import os
    import sys
    import datetime
    import shutil
    
    # Save the actual terminal dimensions before stdout is wrapped in TeeLogger
    if "COLUMNS" not in os.environ:
        try:
            cols, lines = shutil.get_terminal_size()
            os.environ["COLUMNS"] = str(cols)
            os.environ["LINES"] = str(lines)
        except Exception:
            pass
            
    # Line tracer only if explicitly requested for low-level interpreter debugging
    if os.environ.get("MOLMEM_ENABLE_LINE_TRACER") == "1":
        start_execution_tracer()
            
    # Only redirect in main process to avoid logs collision / nested locks
    if not is_main_proc:
        return
        
    try:
        log_dir = get_log_dir(auto_create=True)
        
        # Write to a single run_log.txt that overwrites each run
        filepath = os.path.join(log_dir, "run_log.txt")
        
        mode = "wb"
        sys.stdout = TeeLogger(filepath, sys.stdout, mode=mode, name="TeeLogger_stdout")
        sys.stderr = TeeLogger(filepath, sys.stderr, mode=mode, name="TeeLogger_stderr")
        sys.__stdout__ = sys.stdout
        sys.__stderr__ = sys.stderr
        
        # Reset error log with "Running..." status at startup in the parent process
        if os.environ.get("MOLMEM_REDIRECTED") != "1":
            try:
                err_file = os.path.join(log_dir, "error_log.txt")
                with open(err_file, "w", encoding="utf-8") as f:
                    f.write("==================================================\n")
                    f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
                    f.write("==================================================\n")
                    f.write("Status          : Running...\n")
                    f.write("==================================================\n")
            except Exception:
                pass

        # Register success exit hook
        def clean_exit_success():
            global _disable_success_hook, _has_crashed
            dump_last_100_lines_log()
            if _disable_success_hook or _has_crashed:
                return
            try:
                log_dir = get_log_dir(auto_create=True)
                err_file = os.path.join(log_dir, "error_log.txt")
                
                # Check if the error log file already contains an error traceback/critical message
                if os.path.exists(err_file):
                    try:
                        with open(err_file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if "CRITICAL ERROR" in content or "Traceback" in content or "Exception" in content:
                            return
                    except Exception:
                        pass
                        
                hw_info = get_hardware_info_str()
                with open(err_file, "w", encoding="utf-8") as f:
                    f.write("==================================================\n")
                    f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
                    f.write("==================================================\n")
                    f.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Hardware Info   :\n{hw_info}\n")
                    f.write("==================================================\n")
                    f.write("Status          : No runtime problems detected!\n")
                    f.write("==================================================\n")
            except Exception:
                pass

        import atexit
        atexit.register(clean_exit_success)
        atexit.register(dump_last_100_lines_log)
    except Exception:
        pass


def close_cli_logger():
    import sys
    import os
    
    dump_last_100_lines_log()

    try:
        log_dir = get_log_dir(auto_create=True)
        debug_filepath = os.path.join(log_dir, "run_log_debug.txt")
        with open(debug_filepath, "a", encoding="utf-8") as f:
            f.write(f"PID: {os.getpid()} | stdout_class: {sys.stdout.__class__.__name__} | stderr_class: {sys.stderr.__class__.__name__}\n")
    except Exception:
        pass

    if sys.stdout.__class__.__name__ == 'TeeLogger':
        orig = sys.stdout.stream
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = orig
        sys.__stdout__ = orig
    if sys.stderr.__class__.__name__ == 'TeeLogger':
        orig = sys.stderr.stream
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = orig
        sys.__stderr__ = orig


_numba_configured = False

def ensure_numba_configured():
    global _numba_configured
    if _numba_configured:
        return
    _numba_configured = True
    configure_default_threading()


def get_ram_usage_info():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        ram_gb = ram_bytes / (1024 ** 3)
        return f"System RAM Usage: {ram_gb:.2f} GB"
    except Exception:
        return ""

def get_vram_usage_info():
    try:
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                return f"GPU VRAM Status: Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_peak:.2f} GB"
    except Exception:
        pass
    return ""

def global_excepthook(exctype, value, tb):
    if is_interactive_notebook():
        return sys.__excepthook__(exctype, value, tb)
    global _has_crashed
    _has_crashed = True
    
    # 1. Graceful Pipeline Interruption (Broken pipe / EPIPE)
    if issubclass(exctype, BrokenPipeError):
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.dup2(devnull, sys.stderr.fileno())
        except Exception:
            pass
        kill_child_processes()
        try:
            os.kill(os.getpid(), 9)
        except Exception:
            pass
        os._exit(0)

    # 2. User Interruption (Ctrl+C / SIGINT)
    if issubclass(exctype, KeyboardInterrupt):
        sys.stderr.write("\n[!] Simulation interrupted by user. Terminating process...\n")
        sys.stderr.flush()
        kill_child_processes()
        try:
            import psutil
            p = psutil.Process()
            p.kill()
        except Exception:
            os.kill(os.getpid(), 9)
        os._exit(1)
        
    import traceback
    import shutil
    
    e_str = str(value)
    try:
        width = min(100, shutil.get_terminal_size().columns - 2)
    except Exception:
        width = 78
    border = "=" * width
    
    tb_lines = traceback.format_exception(exctype, value, tb)
    tb_text = "".join(tb_lines)
    
    err_title = "UNEXPECTED RUNTIME EXCEPTION"
    suggestions = [
        "Check the traceback below for the file and line number causing the issue.",
        "Ensure all configuration arrays and physical arguments match requirements."
    ]
    
    if issubclass(exctype, MemoryError) or "out of memory" in e_str.lower() or "oom" in e_str.lower():
        err_title = "SYSTEM OUT OF MEMORY (OOM) DETECTED"
        suggestions = [
            "Reduce the simulation timeline/steps (decrease pulses, sweeps, or tEnd).",
            "If recordHistory=True is enabled, disable it if step-by-step history is not required.",
            "Run on a GPU with larger VRAM limits or increase your system's virtual swap memory."
        ]
    elif issubclass(exctype, (EOFError,)) or "brokenprocesspool" in exctype.__name__.lower() or "broken process pool" in e_str.lower():
        err_title = "PARALLEL WORKER PROCESS TERMINATED UNEXPECTEDLY"
        suggestions = [
            "A background parallel worker was abruptly terminated (likely killed by OS out-of-memory limits).",
            "Reduce the parallel worker thread count or batch sweep sizes.",
            "Ensure system physical RAM is sufficient for high-dimensional concurrent matrix sweeps."
        ]
    elif (isinstance(value, OSError) and getattr(value, 'errno', None) in (28, 122)) or "no space left on device" in e_str.lower() or "disk full" in e_str.lower():
        err_title = "HOST STORAGE VOLUME FULL (NO SPACE LEFT ON DEVICE)"
        suggestions = [
            "The host storage drive ran out of available disk capacity.",
            "Disable 'recordHistory=True' to avoid writing large step-by-step state arrays.",
            "Free disk space on the target volume or select a different output directory."
        ]
    elif issubclass(exctype, RecursionError) or "maximum recursion depth exceeded" in e_str.lower():
        err_title = "MAXIMUM RECURSION DEPTH EXCEEDED (CIRCUIT LOOP FAULT)"
        suggestions = [
            "A cyclic subcircuit network or recursive graph resolution hierarchy exhausted the Python call stack limit.",
            "Verify all circuit nodes have a valid connected path to Node 0 (Ground).",
            "Increase the recursion limit via 'sys.setrecursionlimit(5000)' if modeling deeply nested multi-tier networks."
        ]
    elif issubclass(exctype, ZeroDivisionError) or "division by zero" in e_str.lower():
        err_title = "DIVISION BY ZERO DETECTED (ZERO CONDUCTANCE / INFINITE RESISTANCE)"
        suggestions = [
            "An arithmetic evaluation divided by zero (commonly caused by an idealized 0.0 S device conductance).",
            "Ensure a finite physical leakage floor is configured on crossbars (e.g. leakage_g=1e-12 S).",
            "Check source voltage denominators and series load resistance arguments."
        ]
    elif issubclass(exctype, (FloatingPointError, OverflowError)) or "numerical divergence" in e_str.lower() or "divergence detected" in e_str.lower():
        err_title = "NUMERICAL DIVERGENCE / INVALID FLOATING-POINT STATE"
        suggestions = [
            "Check simulation pulse voltages (ensure max_vWrite does not exceed device breakdown).",
            "Decrease the adaptive time-stepper tolerance threshold (reduce min_dt).",
            "Verify all crossbar conductance values are strictly positive and finite."
        ]
    elif "dll load failed" in e_str.lower() or "shared object" in e_str.lower() or "cuda driver version" in e_str.lower() or "driver version is insufficient" in e_str.lower():
        err_title = "GPU DRIVER / NATIVE C++ EXTENSION ABI MISMATCH"
        suggestions = [
            "The host GPU driver version does not match the compiled C++/ROCm/CUDA extension.",
            "Recompile the native solver kernel by executing 'molmem_lib/csrc/compile.bat'.",
            "Update your AMD ROCm or NVIDIA CUDA display driver to match the active PyTorch runtime.",
            "Launch with 'device=\"cpu\"' to run using the high-performance CPU JIT engine."
        ]
    elif "hip" in e_str.lower() or "rocm" in e_str.lower() or "cuda" in e_str.lower() or "device-side assert" in e_str.lower():
        err_title = "GPU SOLVER / HARDWARE ACCELERATION RUNTIME FAULT"
        suggestions = [
            "Check that ROCm/HIP/CUDA drivers are properly loaded on the target device.",
            "Launch with environment variable CUDA_LAUNCH_BLOCKING=1 to locate the exact failing GPU kernel.",
            "Launch with environment variable TORCH_USE_CUDA_DSA=1 for device-side assertions.",
            "Switch to Host CPU / Numba mode by specifying device='cpu' if the GPU driver is unstable."
        ]
    elif issubclass(exctype, (TimeoutError,)) or "timeoutexpired" in exctype.__name__.lower():
        err_title = "SOLVER / HARDWARE OPERATION TIMED OUT"
        suggestions = [
            "A parallel calibration worker or iterative target search programmer exceeded its execution deadline.",
            "Increase the timeout limit or decrease target convergence precision tolerances.",
            "Check for host CPU/GPU resource contention from background processes."
        ]
    elif issubclass(exctype, (ValueError, IndexError)) and any(w in e_str.lower() for w in ("shape", "broadcast", "dimension", "incompatible", "out of bounds", "unphysical")):
        err_title = "MATRIX DIMENSION / PHYSICAL PARAMETER MISMATCH"
        suggestions = [
            "Input voltage vector, weight matrix dimensions, or physical parameter boundaries do not match requirements.",
            "Ensure pulse vector length equals crossbar rows and target weight matrix shape is (rows, cols).",
            "Verify all physical constants (T > 0 K, R_th >= 0, tau_th > 0) are strictly positive."
        ]
    elif issubclass(exctype, FileNotFoundError) and any(w in e_str.lower() for w in ("cl.exe", "hipcc", "vcvars", "ninja", "compiler", "rc.exe")):
        err_title = "C++ COMPILER / NATIVE BUILD TOOLCHAIN NOT FOUND"
        suggestions = [
            "The MSVC compiler (cl.exe) or AMD ROCm toolkit (hipcc) could not be located in the system PATH.",
            "Install Visual Studio C++ Build Tools and Windows SDK to compile native solver kernels.",
            "Launch simulation with 'device=\"cpu\"' to run using the CPU JIT acceleration layer."
        ]
    elif "typingerror" in exctype.__name__.lower() or "numbatypeerror" in exctype.__name__.lower() or "numba" in e_str.lower() and "type" in e_str.lower():
        err_title = "NUMBA JIT SIGNATURE / ARRAY TYPE MISMATCH"
        suggestions = [
            "An input matrix or state argument has an incompatible data type for compiled machine-code execution.",
            "Ensure all conductance, voltage, and time arrays are standard np.float64 or np.int32 types.",
            "Avoid passing object arrays, complex numbers, or non-contiguous memory slices."
        ]
    elif issubclass(exctype, PermissionError) or (isinstance(value, OSError) and getattr(value, 'errno', None) in (13, 30)) or "permission denied" in e_str.lower() or "read-only" in e_str.lower():
        err_title = "FILE PERMISSION / READ-ONLY STORAGE ACCESS DENIED"
        suggestions = [
            "The process lacks write permissions for the workspace or target output directory.",
            "Ensure the workspace directory is user-writable or run with appropriate filesystem permissions.",
            "Configure the system TEMP/TMP directory to a writable partition."
        ]
    elif any(w in e_str.lower() for w in ("nodevice", "no cuda-capable device", "found no nvidia", "found no amd", "hiperrornodevice")):
        err_title = "GPU HARDWARE NOT DETECTED / UNSUPPORTED ARCHITECTURE"
        suggestions = [
            "No compatible AMD ROCm or NVIDIA CUDA GPU was detected for the requested device backend.",
            "Check that GPU display drivers and compute runtimes are active.",
            "Specify device='cpu' to run using multi-threaded CPU JIT execution."
        ]
    elif "not contiguous" in e_str.lower() or "stride" in e_str.lower() and "contiguous" in e_str.lower():
        err_title = "NON-CONTIGUOUS MEMORY STRIDE / ARRAY LAYOUT ERROR"
        suggestions = [
            "An array or PyTorch tensor passed to native C++/GPU kernels has non-contiguous memory strides.",
            "Wrap the input array with 'np.ascontiguousarray(arr)' or 'tensor.contiguous()'.",
            "Avoid passing non-contiguous matrix transpositions directly into native C++ solver APIs."
        ]
    elif "winerror 206" in e_str.lower() or "filename or extension too long" in e_str.lower():
        err_title = "WINDOWS PATH LENGTH LIMIT EXCEEDED (MAX_PATH > 260 CHARACTERS)"
        suggestions = [
            "The file path exceeds the Windows 260-character MAX_PATH limit.",
            "Shorten the workspace root directory path or relocate simulation output files.",
            "Enable Long Paths in the Windows Registry (HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled=1)."
        ]
    elif issubclass(exctype, OverflowError) and any(w in e_str.lower() for w in ("int too large", "convert to c", "index overflow")):
        err_title = "INTEGER CONVERSION / 32-BIT INDEX OVERFLOW"
        suggestions = [
            "An array index or crossbar dimension exceeds 32-bit integer limits (2,147,483,647 elements).",
            "Partition large crossbar simulation workloads into smaller sub-matrix grids.",
            "Ensure indexing arrays use standard 32-bit integer data types (np.int32)."
        ]
    elif any(w in e_str.lower() for w in ("could not convert string to float", "csv", "lookuptable")) and any(d in e_str.lower() for d in ("iscale", "vscale", "f22pot", "f22dep", "data")):
        err_title = "CORRUPTED DEVICE DATA TABLE / CSV FORMATTING FAULT"
        suggestions = [
            "A custom device lookup table in the 'data/' directory is malformed or unparseable.",
            "Ensure all CSV files in 'molmem_lib/data/' contain valid floating-point numbers without corrupted header rows.",
            "Re-export or verify custom device table files."
        ]
        
    mem_info = get_ram_usage_info()
    gpu_info = get_vram_usage_info()
    
    sys.stderr.write(f"\n{border}\n")
    sys.stderr.write(f"  [CRITICAL ERROR] {err_title}\n")
    sys.stderr.write(f"  Error Type: {exctype.__name__}\n")
    sys.stderr.write(f"  Message   : {e_str}\n")
    if mem_info:
        sys.stderr.write(f"  {mem_info}\n")
    if gpu_info:
        sys.stderr.write(f"  {gpu_info}\n")
    sys.stderr.write(f"{border}\n")
    sys.stderr.write("  Diagnostic Recommendations:\n")
    for i, sugg in enumerate(suggestions, 1):
        sys.stderr.write(f"    {i}. {sugg}\n")
    sys.stderr.write(f"{border}\n")
    sys.stderr.write("  Detailed Traceback:\n")
    sys.stderr.write(tb_text.strip() + "\n")
    sys.stderr.write(f"{border}\n")
    sys.stderr.flush()
    
    err_log_content = (
        f"CRITICAL ERROR: {err_title}\n"
        f"Error Type: {exctype.__name__}\n"
        f"Message   : {e_str}\n"
    )
    if mem_info:
        err_log_content += f"{mem_info}\n"
    if gpu_info:
        err_log_content += f"{gpu_info}\n"
    err_log_content += "\nDiagnostic Recommendations:\n"
    for i, sugg in enumerate(suggestions, 1):
        err_log_content += f"  {i}. {sugg}\n"
    err_log_content += f"\nDetailed Traceback:\n{tb_text.strip()}\n"
    
    log_exception_file(err_log_content)
    
    kill_child_processes()
    try:
        os.kill(os.getpid(), 9)
    except Exception:
        pass
    os._exit(1)

if not is_interactive_notebook():
    sys.excepthook = global_excepthook
    threading.excepthook = lambda args: global_excepthook(args.exc_type, args.exc_value, args.exc_traceback)
    if hasattr(sys, 'unraisablehook'):
        sys.unraisablehook = lambda args: global_excepthook(args.exc_type, args.exc_value, args.exc_traceback)


def get_progress_suffix(sim=None):
    try:
        import os
        if os.environ.get("MOLMEM_CPU_ONLY") == "1":
            return "cpu"
        if sim is None:
            import molmem_lib
            sims = list(getattr(molmem_lib, '_active_simulators', []))
            for s in sims:
                device = str(getattr(s, 'device', 'cpu')).lower()
                backend = str(getattr(s, 'backend', 'numba')).lower()
                if 'cuda' in device or 'hip' in device:
                    sim = s
                    break
            if sim is None and sims:
                sim = sims[0]
                
        if sim is not None:
            # Check cached suffix to avoid expensive imports / env lookups on every progress update step
            cached = getattr(sim, '_cached_progress_suffix', None)
            if cached is not None:
                return cached
                
            backend = str(getattr(sim, 'backend', 'auto')).lower()
            is_hybrid = backend in ['hybrid', 'auto'] and os.environ.get("MOLMEM_FORCE_GPU") != "1" and os.environ.get("MOLMEM_FORCE_CPU") != "1"
            
            device = str(getattr(sim, 'device', 'cpu')).lower()
            if 'cuda' not in device and 'hip' not in device:
                suffix = "cpu"
            else:
                try:
                    from .torch_simulator import check_has_cpp_extension
                    has_ext = check_has_cpp_extension() and getattr(sim, 'compile_backend', False)
                except Exception:
                    has_ext = False
                if has_ext:
                    import torch
                    is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
                    gpu_name = "rocm" if is_hip else "cuda"
                    suffix = f"hybrid: cpu+{gpu_name}" if is_hybrid else f"gpu-{gpu_name}"
                else:
                    suffix = "hybrid: cpu+gpu" if is_hybrid else "gpu-jit"
            try:
                sim._cached_progress_suffix = suffix
            except Exception:
                pass
            return suffix
        else:
            import sys
            import torch
            if 'molmem_cuda_solver' in sys.modules:
                is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
                return "gpu-rocm" if is_hip else "gpu-cuda"
            return "gpu-jit"
    except Exception as e:
        return "cpu"


_FLAG_FALSE_TUPLE = (False, False, False, False, False, False, 0)
_FLAG_TRUE_TUPLE = (True, True, True, True, True, True, 63)

def resolve_record_history_flags(recordHistory):
    """
    Parses the recordHistory parameter and returns a tuple of flags:
    (v_flag, i_flag, state_flag, f22_flag, g_flag, temp_flag, bitmask)
    """
    if not recordHistory:
        return _FLAG_FALSE_TUPLE
        
    if recordHistory is True or recordHistory == 63:
        return _FLAG_TRUE_TUPLE
        
    v_flag = False
    i_flag = False
    state_flag = False
    f22_flag = False
    g_flag = False
    temp_flag = False
    
    if isinstance(recordHistory, (list, tuple, set)):
        keys = {str(k).lower().strip() for k in recordHistory}
        if 'v' in keys or 'voltage' in keys:
            v_flag = True
        if 'i' in keys or 'current' in keys:
            i_flag = True
        if 'state' in keys or 'n' in keys:
            state_flag = True
        if 'f22' in keys:
            f22_flag = True
        if 'g' in keys or 'conductance' in keys:
            g_flag = True
        if 'temp' in keys or 'temperature' in keys:
            temp_flag = True
    else:
        # Fallback to true if it is not a sequence but is truthy
        return _FLAG_TRUE_TUPLE
        
    # Bitmask:
    # bit 0: v (1)
    # bit 1: i (2)
    # bit 2: state (4)
    # bit 3: f22 (8)
    # bit 4: g (16)
    # bit 5: temp (32)
    bitmask = 0
    if v_flag: bitmask |= 1
    if i_flag: bitmask |= 2
    if state_flag: bitmask |= 4
    if f22_flag: bitmask |= 8
    if g_flag: bitmask |= 16
    if temp_flag: bitmask |= 32
    
    return v_flag, i_flag, state_flag, f22_flag, g_flag, temp_flag, bitmask

def handle_fatal_hardware_error(msg, error_type="CRITICAL HARDWARE ERROR", exc=None):
    import datetime
    import traceback
    import os
    import sys
    
    if exc is not None:
        stack_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        stack_str = "".join(traceback.format_stack())
        
    pid = os.getpid()
    ppid = os.getppid() if hasattr(os, 'getppid') else 0
    proc_name = multiprocessing.current_process().name
    
    error_details = (
        f"Process Diagnostic Telemetry:\n"
        f"  - Active Process Name : {proc_name}\n"
        f"  - Process ID (PID)    : {pid}\n"
        f"  - Parent Process ID   : {ppid}\n"
        f"  - Thread/Affinity State:\n"
        f"      * OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS', 'Not Set')}\n"
        f"      * NUMBA_NUM_THREADS : {os.environ.get('NUMBA_NUM_THREADS', 'Not Set')}\n"
        f"      * MKL_NUM_THREADS   : {os.environ.get('MKL_NUM_THREADS', 'Not Set')}\n"
        f"      * MOLMEM_CHILD      : {os.environ.get('MOLMEM_MP_CHILD', 'Not Set')}\n"
        f"      * CALIB_WORKER      : {os.environ.get('MOLMEM_IS_CALIBRATION_WORKER', 'Not Set')}\n"
        f"\n"
        f"Execution Call Stack:\n"
        f"{stack_str}\n"
    )
    
    sys.stderr.write(f"\n[MOLMEM FATAL ERROR] {msg}\n\n{error_details}\n")
    sys.stderr.flush()
    if is_logging_enabled():
        try:
            err_dir = get_log_dir(auto_create=True)
            err_file = os.path.join(err_dir, "error_log.txt")
            with open(err_file, "w", encoding="utf-8") as f:
                f.write("==================================================\n")
                f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
                f.write("==================================================\n")
                f.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
                f.write(f"Status          : {error_type}\n")
                f.write("==================================================\n\n")
                f.write(f"[{error_type}] {msg}\n\n")
                f.write(error_details)
        except Exception:
            pass
        
    try:
        import psutil
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            if child.pid != pid:
                child.kill()
    except Exception:
        pass
    try:
        os.kill(pid, 9)
    except Exception:
        pass
    os._exit(1)


def validate_isolation_safety(sim_instance):
    import os
    import sys
    
    def _handle_fatal_isolation_error(msg):
        import datetime
        import traceback
        
        stack_str = "".join(traceback.format_stack())
        pid = os.getpid()
        ppid = os.getppid()
        proc_name = multiprocessing.current_process().name
        
        error_details = (
            f"Process Diagnostic Telemetry:\n"
            f"  - Active Process Name : {proc_name}\n"
            f"  - Process ID (PID)    : {pid}\n"
            f"  - Parent Process ID   : {ppid}\n"
            f"  - Thread/Affinity State:\n"
            f"      * OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS', 'Not Set')}\n"
            f"      * NUMBA_NUM_THREADS : {os.environ.get('NUMBA_NUM_THREADS', 'Not Set')}\n"
            f"      * MKL_NUM_THREADS   : {os.environ.get('MKL_NUM_THREADS', 'Not Set')}\n"
            f"      * MOLMEM_CHILD      : {os.environ.get('MOLMEM_MP_CHILD', 'Not Set')}\n"
            f"      * CALIB_WORKER      : {os.environ.get('MOLMEM_IS_CALIBRATION_WORKER', 'Not Set')}\n"
            f"\n"
            f"Execution Call Stack (Source of Worker/Last Code Line):\n"
            f"{stack_str}\n"
        )
        
        sys.stderr.write(f"\n[MOLMEM ISOLATION ERROR] {msg}\n\n{error_details}\n")
        sys.stderr.flush()
        if is_logging_enabled():
            try:
                err_dir = get_log_dir(auto_create=True)
                err_file = os.path.join(err_dir, "error_log.txt")
                mode = "a" if os.path.exists(err_file) else "w"
                with open(err_file, mode, encoding="utf-8") as f:
                    if mode == "a":
                        f.write("\n\n" + "=" * 50 + "\n\n")
                    f.write("==================================================\n")
                    f.write("             MOLMEM ERROR DIAGNOSTIC LOG          \n")
                    f.write("==================================================\n")
                    f.write(f"Timestamp       : {datetime.datetime.now().isoformat()}\n")
                    f.write("Status          : CRITICAL ISOLATION ERROR\n")
                    f.write("==================================================\n\n")
                    f.write(f"[MOLMEM ISOLATION ERROR] {msg}\n\n")
                    f.write(error_details)
            except Exception:
                pass
        try:
            parent_pid = os.getppid()
            if parent_pid > 0 and not is_main_proc:
                try:
                    import psutil
                    parent = psutil.Process(parent_pid)
                    for child in parent.children(recursive=True):
                        if child.pid != os.getpid():
                            child.kill()
                except Exception:
                    pass
                os.kill(parent_pid, 9)
        except Exception:
            pass
        os._exit(1)
        
    device = getattr(sim_instance, 'device', 'cpu')
    backend = getattr(sim_instance, 'backend', 'numba')
    is_gpu = 'cuda' in str(device) or 'hip' in str(device) or backend in ['torch', 'pytorch', 'hybrid']
    
    # Bypass safety checks during JIT compiler warmups or active profiling sweeps
    if (getattr(sim_instance, '_in_warmup', False) or 
        getattr(sim_instance.__class__, '_in_warmup', False) or 
        getattr(sim_instance, '_profiler_active', False) or
        getattr(sim_instance.__class__, '_in_parallel_profiling', False) or
        getattr(sim_instance.__class__, '_in_gpu_crossover_profiling', False)):
        return

    # 1. GPU Isolation Rule: Check PyTorch thread pool
    if is_gpu:

        try:
            import torch
            torch_threads = torch.get_num_threads()
            if torch_threads <= 1 and (os.cpu_count() or 1) > 1 and is_main_proc:
                # Dynamically release CPU thread lock for GPU/PyTorch solver scheduling
                max_cores = max(2, int((os.cpu_count() or 2) * 0.8))
                torch.set_num_threads(max_cores)
                torch_threads = torch.get_num_threads()
            if torch_threads <= 1 and (os.cpu_count() or 1) > 1 and is_main_proc:
                _handle_fatal_isolation_error(
                    f"GPU execution target '{device}' selected, but CPU thread lock is active "
                    f"(PyTorch threads restricted to {torch_threads}). "
                    "GPU runs must not have CPU thread locks to prevent launch scheduling bottlenecks."
                )
        except ImportError:
            pass

    # 2. CPU Simulation Lock Rule: CPU platform simulation runs MUST have a thread lock matching optimal_cores.
    else:
        try:
            import numba as nb
            optimal_cores = getattr(sim_instance, 'optimal_cores', 1)
            numba_threads = nb.get_num_threads()
            
            # Enforce thread lock validation
            if numba_threads != optimal_cores:
                _handle_fatal_isolation_error(
                    f"CPU simulation execution lacks thread lock or mismatch detected "
                    f"(numba_threads={numba_threads}, expected optimal_cores={optimal_cores})."
                )
        except (ImportError, AttributeError, NotImplementedError):
            pass

    # 3. Multiprocessing Thread-Collision Check: Nested parallel loops inside worker processes
    is_child = not is_main_proc
    if is_child:
        try:
            import numba as nb
            numba_threads = nb.get_num_threads()
        except Exception:
            numba_threads = 1
            
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        mkl_threads = int(os.environ.get("MKL_NUM_THREADS", "1"))
        openblas_threads = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))
        numexpr_threads = int(os.environ.get("NUMEXPR_NUM_THREADS", "1"))
        
        if numba_threads > 1 or omp_threads > 1 or mkl_threads > 1 or openblas_threads > 1 or numexpr_threads > 1:
            _handle_fatal_isolation_error(
                f"Nested thread over-subscription hazard detected in worker process. "
                f"Worker processes must run in strictly single-threaded mode (Numba={numba_threads}, OMP={omp_threads}, "
                f"MKL={mkl_threads}, OpenBLAS={openblas_threads}, NumExpr={numexpr_threads}). "
                "Parallel loops inside nested process pools are forbidden to prevent CPU resource lockout."
            )
            
    # 4. GPU Device Conflict Check: GPU selected but hardware visibility blocked
    if is_gpu:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")
        cpu_only = os.environ.get("MOLMEM_CPU_ONLY")
        
        if cuda_visible == "" or hip_visible == "" or cpu_only == "1":
            _handle_fatal_isolation_error(
                f"GPU execution target '{device}' selected, but hardware visibility is blocked "
                f"by environment variable constraints (CUDA_VISIBLE_DEVICES='{cuda_visible}', "
                f"HIP_VISIBLE_DEVICES='{hip_visible}', MOLMEM_CPU_ONLY='{cpu_only}'). "
                "Please verify your environment visibility configurations."
            )

def get_gpu_hardware_telemetry_str(device=None):
    """
    Returns real-time allocated GPU VRAM MB via driver-level memory query.
    """
    try:
        import torch
        if torch.cuda.is_available():
            if device is not None:
                dev = torch.device(device) if isinstance(device, str) else device
                if dev.type != 'cuda':
                    return ""
            else:
                dev = torch.device('cuda')
            free_b, tot_b = torch.cuda.mem_get_info(dev)
            used_mb = (tot_b - free_b) / (1024 * 1024)
            return f" | VRAM: {used_mb:.0f}MB"
    except Exception:
        pass
    return ""


# =====================================================================
# DYNAMIC CALLER SCRIPT DEPENDENCY RESOLUTION & VALIDATION (via uv)
# =====================================================================

_PYPI_MODULE_ALIAS_MAP = {
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'yaml': 'pyyaml',
    'bs4': 'beautifulsoup4',
    'dotenv': 'python-dotenv',
    'serial': 'pyserial',
    'skimage': 'scikit-image',
    'magic': 'python-magic',
    'fitz': 'pymupdf',
    'docx': 'python-docx',
    'pptx': 'python-pptx',
    'usb': 'pyusb',
    'OpenGL': 'PyOpenGL',
    'git': 'GitPython',
    'dateutil': 'python-dateutil',
    'jwt': 'PyJWT',
    'crypto': 'pycryptodome',
    'Crypto': 'pycryptodome',
    'websocket': 'websocket-client',
    'attr': 'attrs',
    'pkg_resources': 'setuptools',
    'setuptools_scm': 'setuptools-scm'
}

def detect_caller_script_path():
    """
    Identifies the entry point or caller script from which molmem_lib is being invoked.
    """
    # 1. Check __main__.__file__
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        main_file = getattr(main_mod, '__file__', None)
        if main_file and os.path.isfile(main_file):
            return os.path.abspath(main_file)

    # 2. Check sys.argv[0]
    if sys.argv and sys.argv[0]:
        cand = sys.argv[0]
        if os.path.isfile(cand) and cand.endswith('.py'):
            return os.path.abspath(cand)

    # 3. Stack inspection fallback
    try:
        frame = sys._getframe()
        while frame:
            f_code = frame.f_code
            f_name = f_code.co_filename
            if f_name and os.path.isfile(f_name) and not f_name.endswith('__init__.py') and 'molmem_lib' not in f_name and 'importlib' not in f_name:
                return os.path.abspath(f_name)
            frame = frame.f_back
    except Exception:
        pass

    return None

def extract_imported_modules_from_file(script_path):
    """
    Parses the caller script's Abstract Syntax Tree (AST) to statically extract
    all top-level imported module names without executing unverified code.
    """
    import ast
    if not script_path or not os.path.isfile(script_path):
        return set()

    imported_names = set()
    try:
        with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read(), filename=script_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split('.')[0]
                    if root_name:
                        imported_names.add(root_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_name = node.module.split('.')[0]
                    if root_name:
                        imported_names.add(root_name)
    except Exception:
        pass

    return imported_names

def filter_external_dependencies(import_names, script_path=None):
    """
    Filters raw imported module names against Python standard library,
    built-ins, molmem_lib modules, and local workspace python files.
    """
    # 1. Standard library modules
    stdlib_names = getattr(sys, 'stdlib_module_names', None)
    if stdlib_names is None:
        stdlib_names = set(sys.builtin_module_names) | {
            'os', 'sys', 'time', 'math', 'json', 're', 'gc', 'shutil', 'subprocess',
            'threading', 'multiprocessing', 'concurrent', 'ctypes', 'weakref',
            'collections', 'itertools', 'functools', 'contextlib', 'typing',
            'datetime', 'urllib', 'pathlib', 'tempfile', 'traceback', 'ast',
            'inspect', 'importlib', 'platform', 'unittest', 'logging', 'io',
            'hashlib', 'socket', 'copy', 'queue', 'struct', 'enum', 'random',
            'pickle', 'uuid', 'warnings', 'signal', 'atexit', 'glob', 'bisect'
        }

    # 2. Local workspace files/directories (Strictly workspace/script dirs, NOT host sys.path)
    local_names = {'molmem_lib', '__main__', '__future__'}
    search_dirs = [os.getcwd()]
    if script_path:
        search_dirs.append(os.path.dirname(os.path.abspath(script_path)))
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(lib_dir)
    search_dirs.append(workspace_root)
    search_dirs.append(os.path.dirname(workspace_root))

    for s_dir in search_dirs:
        if not s_dir or not os.path.isdir(s_dir):
            continue
        try:
            for item in os.listdir(s_dir):
                if item.endswith('.py'):
                    local_names.add(item[:-3])
                elif os.path.isdir(os.path.join(s_dir, item)) and os.path.isfile(os.path.join(s_dir, item, '__init__.py')):
                    local_names.add(item)
        except Exception:
            pass

    external_pkgs = set()
    for name in import_names:
        if name in stdlib_names or name in sys.builtin_module_names or name in local_names:
            continue
        pypi_name = _PYPI_MODULE_ALIAS_MAP.get(name, name)
        external_pkgs.add(pypi_name)

    return external_pkgs

def resolve_and_install_caller_dependencies(venv_python, caller_script=None, timeout=120):
    """
    Inspects the caller script for third-party dependencies and uses uv to resolve
    and install compatible versions against molmem_lib's pinned core constraints.
    If an incompatible version is detected, halts execution with an explicit warning.
    """
    import subprocess
    import json
    import shutil

    # Fast bypass when running inside a pre-managed or subtest environment
    if os.environ.get("MOLMEM_ENV_MANAGED") == "1" or os.environ.get("MOLMEM_NO_PROVISION") == "1":
        return True

    if caller_script is None:
        caller_script = detect_caller_script_path()
    if not caller_script or not os.path.isfile(caller_script):
        return True

    # Check caller dependency cache in .calcache
    session_cache_dir = get_calcache_dir(auto_create=True)
    cache_file = os.path.join(session_cache_dir, "caller_dep_cache.json")
    
    script_abs = os.path.abspath(caller_script)
    try:
        script_mtime = os.path.getmtime(caller_script)
    except Exception:
        script_mtime = 0.0
    
    cached_records = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_records = json.load(f)
        except Exception:
            cached_records = {}
            
    rec = cached_records.get(script_abs)
    if rec and rec.get("mtime") == script_mtime and rec.get("status") == "ok":
        return True

    raw_imports = extract_imported_modules_from_file(caller_script)
    target_pkgs = filter_external_dependencies(raw_imports, caller_script)
    if not target_pkgs:
        try:
            os.makedirs(session_cache_dir, exist_ok=True)
            cached_records[script_abs] = {"mtime": script_mtime, "status": "ok"}
            with open(cache_file, "w") as f:
                json.dump(cached_records, f, indent=2)
        except Exception:
            pass
        return True

    # Check which packages are already installed in venv_python
    check_code = (
        "import json, importlib.metadata\n"
        f"targets = {list(target_pkgs)}\n"
        "installed = {}\n"
        "for t in targets:\n"
        "    try:\n"
        "        v = importlib.metadata.version(t)\n"
        "        installed[t] = v\n"
        "    except Exception:\n"
        "        installed[t] = None\n"
        "print(json.dumps(installed))\n"
    )

    installed_map = {}
    try:
        res = subprocess.run([venv_python, "-c", check_code], capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            installed_map = json.loads(res.stdout.strip())
    except Exception:
        installed_map = {}

    missing_pkgs = [pkg for pkg, ver in installed_map.items() if not ver]
    if not missing_pkgs:
        try:
            os.makedirs(session_cache_dir, exist_ok=True)
            cached_records[script_abs] = {"mtime": script_mtime, "status": "ok"}
            with open(cache_file, "w") as f:
                json.dump(cached_records, f, indent=2)
        except Exception:
            pass
        return True

    script_basename = os.path.basename(caller_script)
    sys.stdout.write(f"[*] Caller script '{script_basename}' requires additional dependencies: {', '.join(missing_pkgs)}\n")
    sys.stdout.write("[*] Verifying cross-compatibility against MolmemSimulator core constraints via uv...\n")
    sys.stdout.flush()

    # Core locked constraints to preserve environment stability
    core_constraints = [
        "numpy>=1.26.0,<2.5.0",
        "scipy",
        "numba",
        "tbb",
        "tensorflow",
        "matplotlib",
        "psutil"
    ]

    uv_bin = shutil.which("uv")
    if not uv_bin:
        for cand in [os.path.expanduser("~/.local/bin/uv"), os.path.expanduser("~/.cargo/bin/uv"), "/usr/local/bin/uv"]:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                uv_bin = cand
                break

    is_offline = (os.environ.get("UV_OFFLINE") == "1" or os.environ.get("MOLMEM_OFFLINE") == "1")
    offline_flags = ["--offline"] if is_offline else []

    if uv_bin:
        cmd = [uv_bin, "pip", "install"] + offline_flags + ["--python", venv_python] + core_constraints + missing_pkgs
    else:
        cmd = [venv_python, "-m", "pip", "install"] + missing_pkgs

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            error_output = (proc.stderr or proc.stdout or "").strip()
            # If failed due to offline mode, network unreachable, or missing uv resolver, log non-fatal warning and allow execution
            if is_offline or "no module named uv" in error_output.lower() or "failed to fetch" in error_output.lower() or "network" in error_output.lower() or "dns" in error_output.lower():
                sys.stderr.write(f"\n[*] Dependency Notice: Unable to pre-resolve caller dependencies ({', '.join(missing_pkgs)}) via uv.\n")
                sys.stderr.write(f"[*] Proceeding to module import...\n\n")
                sys.stderr.flush()
                return True

            sys.stderr.write("\n" + "=" * 80 + "\n")
            sys.stderr.write(" [!] MOLMEM DEPENDENCY CONFLICT: INCOMPATIBLE PACKAGE DETECTED\n")
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f" Caller script '{script_basename}' requested: {', '.join(missing_pkgs)}\n")
            sys.stderr.write(" The requested packages conflict with MolmemSimulator's pinned core constraints:\n")
            sys.stderr.write("   - numpy >= 1.26.0, < 2.5.0 (Required for Numba JIT & C++ extensions)\n")
            sys.stderr.write("   - PyTorch GPU / ROCm / CUDA runtime compatibility\n\n")
            sys.stderr.write(" Resolver diagnostic details:\n")
            for err_line in error_output.splitlines()[-10:]:
                sys.stderr.write(f"   {err_line}\n")
            sys.stderr.write("\n [!] Execution safely halted to prevent virtual environment corruption.\n")
            sys.stderr.write("=" * 80 + "\n\n")
            sys.stderr.flush()
            sys.exit(1)
        else:
            sys.stdout.write(f"[+] Successfully resolved and installed compatible caller dependencies: {', '.join(missing_pkgs)}\n")
            sys.stdout.flush()
            try:
                os.makedirs(session_cache_dir, exist_ok=True)
                cached_records[script_abs] = {"mtime": script_mtime, "status": "ok"}
                with open(cache_file, "w") as f:
                    json.dump(cached_records, f, indent=2)
            except Exception:
                pass
            return True
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[!] Package resolution timed out for caller dependencies: {', '.join(missing_pkgs)}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"[!] Dependency resolution error: {e}\n")
        sys.exit(1)


def _cleanup_session_caller_dep_cache():
    """Removes any legacy caller dependency cache from .temp_history."""
    try:
        temp_dir = get_temp_history_dir(auto_create=False)
        legacy_cache = os.path.join(temp_dir, "caller_dep_cache.json")
        if os.path.isfile(legacy_cache):
            os.remove(legacy_cache)
    except Exception:
        pass

import atexit
atexit.register(_cleanup_session_caller_dep_cache)


# =============================================================================
# PROPRIETARY PHYSICS BINARY TABLE CODEC & UNIFIED ASSET BUNDLE LOADER
# =============================================================================

_MAGIC_PHYSICS_HEADER = b"MOLMEM\x01\x00"
_MAGIC_PHYSICS_BUNDLE_V2 = b"MOLMEM\x02\x00"
_MAGIC_PHYSICS_BUNDLE_V3 = b"MOLMEM\x03\x00"
_PHYSICS_MASK_KEY = 0xA5C3E1F79B8D4A2F


class PhysicsTag:
    """Abstract 64-bit salted section tags (no plaintext table names in binary)."""
    I_SCALE         = 0x8F4A12B001010001
    V_SCALE         = 0x8F4A12B002020002
    F22_POT_MAP     = 0x8F4A12B003030003
    F22_DEP_MAP     = 0x8F4A12B004040004
    D2D_MISMATCH    = 0x8F4A12B005050005
    READ_NOISE      = 0x8F4A12B006060006
    WRITE_NOISE     = 0x8F4A12B007070007


# Canonical stem to tag mapping for resolving string names/filenames to abstract tags
_STEM_TO_PHYSICS_TAG = {
    "i_scale": PhysicsTag.I_SCALE,
    "iscale": PhysicsTag.I_SCALE,
    "v_scale": PhysicsTag.V_SCALE,
    "vscale": PhysicsTag.V_SCALE,
    "f22_pot_map": PhysicsTag.F22_POT_MAP,
    "f22potmap": PhysicsTag.F22_POT_MAP,
    "f22_dep_map": PhysicsTag.F22_DEP_MAP,
    "f22depmap": PhysicsTag.F22_DEP_MAP,
    "d2d_mismatch_quantile": PhysicsTag.D2D_MISMATCH,
    "d2d_mismatch": PhysicsTag.D2D_MISMATCH,
    "read_noise": PhysicsTag.READ_NOISE,
    "write_noise": PhysicsTag.WRITE_NOISE,
}


def _generate_physics_keystream(num_uint64, seed=_PHYSICS_MASK_KEY):
    """Generates a deterministic 64-bit scrambling keystream via 64-bit LCG."""
    import numpy as np
    a = 6364136223846793005
    c = 1442695040888963407
    mask64 = 0xFFFFFFFFFFFFFFFF
    keystream = np.empty(num_uint64, dtype=np.uint64)
    state = seed & mask64
    for i in range(num_uint64):
        state = (state * a + c) & mask64
        keystream[i] = np.uint64((state ^ (state >> 18)) & mask64)
    return keystream


def decode_physics_bundle(raw_bytes):
    """
    Decodes a monolithic asset bundle (molmem_core.bin) into a dict of
    {tag_id: (N, 2) float64 ndarray}.
    Supports v3 (Codebase Integrity Attestation Sealed) and v2.
    """
    import struct
    import hashlib
    import numpy as np

    if len(raw_bytes) < 24:
        raise ValueError("Invalid bundle header: truncated bytes.")

    magic = raw_bytes[:8]
    if magic == _MAGIC_PHYSICS_BUNDLE_V3:
        if len(raw_bytes) < 48:
            raise ValueError("Truncated v3 bundle: missing envelope header.")
        magic, version, num_sections, salt, wrapped_dek, auth_token, reserved = struct.unpack(
            "<8sIIQQQQ", raw_bytes[:48]
        )
        if version != 3:
            raise ValueError(f"Unsupported bundle version: {version}")

        # Compute library codebase footprint
        from . import _compute_library_footprint
        footprint = _compute_library_footprint()
        kek_digest = hashlib.sha256(footprint.encode('ascii')).digest()
        kek = struct.unpack("<Q", kek_digest[:8])[0]

        # Unwrap ephemeral DEK
        mask64 = 0xFFFFFFFFFFFFFFFF
        salt_scramble = (salt * 6364136223846793005 + 1442695040888963407) & mask64
        dek = (wrapped_dek ^ kek ^ salt_scramble) & mask64

        # Validate codebase integrity attestation token
        expected_token = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek)).digest()[:8])[0]
        if expected_token != auth_token:
            CANONICAL_FOOTPRINT = "2ab6592f9b19e9c23e46c7350c95ab6a2c01f117ce3bc3d4f30ef8ea1337050d"
            kek_digest_canon = hashlib.sha256(CANONICAL_FOOTPRINT.encode('ascii')).digest()
            kek_canon = struct.unpack("<Q", kek_digest_canon[:8])[0]
            dek_canon = (wrapped_dek ^ kek_canon ^ salt_scramble) & mask64
            token_canon = struct.unpack("<Q", hashlib.sha256(struct.pack("<Q", dek_canon)).digest()[:8])[0]
            if token_canon == auth_token:
                dek = dek_canon
            else:
                # Check if DevicePhysicsCompiler is available in workspace to auto-seal
                import multiprocessing
                is_main = (multiprocessing.current_process().name == 'MainProcess')
                lib_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(lib_dir, "data")
                workspace_root = os.path.dirname(os.path.dirname(lib_dir))
                compiler_dir = os.path.join(workspace_root, "DevicePhysicsCompiler")
                compiler_script = os.path.join(compiler_dir, "compile_physics_data.py")
                
                if is_main and os.path.isfile(compiler_script):
                    try:
                        import subprocess
                        print("[Physics Core] Codebase changed; auto-invoking DevicePhysicsCompiler to re-seal molmem_core.bin...", flush=True)
                        res = subprocess.run([sys.executable, compiler_script], cwd=compiler_dir, capture_output=True, text=True, timeout=25)
                        if res.returncode == 0:
                            bundle_file = os.path.join(data_dir, "molmem_core.bin")
                            if os.path.isfile(bundle_file):
                                with open(bundle_file, "rb") as bf:
                                    recompiled_bytes = bf.read()
                                return decode_physics_bundle(recompiled_bytes)
                    except Exception:
                        pass

                target = getattr(sys, '__stderr__', None) or sys.stderr
                err_msg = (
                    "\n"
                    "========================================================================================\n"
                    " [!] MOLMEM INTEGRITY ERROR: Library codebase changed and no longer passes\n"
                    "     the de-encryption check. Execution halted.\n"
                    "========================================================================================\n\n"
                )
                try:
                    target.write(err_msg)
                    target.flush()
                except Exception:
                    pass
                os._exit(1)

        toc_offset = 48
        toc_size = num_sections * 24
        payload_start = toc_offset + toc_size
        seed = dek

    elif magic == _MAGIC_PHYSICS_BUNDLE_V2:
        magic, version, num_sections, flags, reserved = struct.unpack("<8sIIII", raw_bytes[:24])
        toc_offset = 24
        toc_size = num_sections * 24
        payload_start = toc_offset + toc_size
        seed = _PHYSICS_MASK_KEY

    else:
        raise ValueError("Invalid bundle header or corrupted magic bytes.")

    if len(raw_bytes) < payload_start:
        raise ValueError("Truncated bundle: missing table of contents or payload.")

    sections = []
    total_points = 0
    for i in range(num_sections):
        entry_bytes = raw_bytes[toc_offset + i * 24 : toc_offset + (i + 1) * 24]
        tag, offset, n_points, sec_flags = struct.unpack("<QQII", entry_bytes)
        sections.append((tag, offset, n_points))
        total_points += n_points

    expected_u64 = total_points * 2
    scrambled_payload = np.frombuffer(raw_bytes[payload_start:], dtype=np.uint64)
    if len(scrambled_payload) < expected_u64:
        raise ValueError("Corrupted bundle: payload truncated.")

    scrambled_payload = scrambled_payload[:expected_u64].copy()
    keystream = _generate_physics_keystream(expected_u64, seed=seed)
    unscrambled_payload = scrambled_payload ^ keystream

    result = {}
    for tag, offset_bytes, n_points in sections:
        u64_start = offset_bytes // 8
        u64_count = n_points * 2
        sec_u64 = unscrambled_payload[u64_start : u64_start + u64_count]
        result[tag] = sec_u64.view(np.float64).reshape((n_points, 2))

    return result


def _verify_physics_caller():
    """
    Enforces call-stack attestation. Restricts raw physics table unpacking
    exclusively to internal simulation engines within molmem_lib.
    """
    import sys
    import os
    lib_dir = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
    sys_utils_file = os.path.normcase(os.path.abspath(__file__))
    frame = sys._getframe(2)
    while frame is not None:
        fname = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if fname == sys_utils_file:
            frame = frame.f_back
            continue
        if fname.startswith(lib_dir):
            return  # Authorized caller within molmem_lib (e.g. device.py, noise.py)
        else:
            break
        frame = frame.f_back

    raise PermissionError(
        "Direct access to proprietary physics tables is restricted to internal device ODE solvers."
    )


class PhysicsBundleManager:
    """
    Singleton manager for transparent loading, in-memory caching, and abstract
    dispatch of proprietary molecular memristor device physics tables.
    """
    _tables = None
    _bundle_path = None

    @classmethod
    def get_bundle_dir(cls):
        import os
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

    @classmethod
    def find_bundle_file(cls):
        import os
        candidate = os.path.join(cls.get_bundle_dir(), "molmem_core.bin")
        if os.path.isfile(candidate):
            return candidate
        alt_dirs = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DevicePhysicsCompiler")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "molmem_lib", "data")),
        ]
        for d in alt_dirs:
            p = os.path.join(d, "molmem_core.bin")
            if os.path.isfile(p):
                return p
        return None

    @classmethod
    def load_bundle(cls, force_reload=False):
        import os
        if cls._tables is not None and not force_reload:
            return cls._tables
        bundle_path = cls.find_bundle_file()
        if not bundle_path or not os.path.isfile(bundle_path):
            import multiprocessing
            is_main = (multiprocessing.current_process().name == 'MainProcess')
            lib_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(os.path.dirname(lib_dir))
            compiler_dir = os.path.join(workspace_root, "DevicePhysicsCompiler")
            compiler_script = os.path.join(compiler_dir, "compile_physics_data.py")
            if is_main and os.path.isfile(compiler_script):
                try:
                    import subprocess
                    import sys
                    print("[Physics Core] molmem_core.bin not found; auto-invoking DevicePhysicsCompiler...", flush=True)
                    res = subprocess.run([sys.executable, compiler_script], cwd=compiler_dir, capture_output=True, text=True, timeout=25)
                    if res.returncode == 0:
                        bundle_path = cls.find_bundle_file()
                except Exception:
                    pass
        if bundle_path and os.path.isfile(bundle_path):
            with open(bundle_path, "rb") as f:
                raw = f.read()
            cls._tables = decode_physics_bundle(raw)
            cls._bundle_path = bundle_path
            return cls._tables
        return None

    @classmethod
    def get_table(cls, tag_or_identifier):
        """
        Retrieves an (N, 2) float64 ndarray by tag (int), stem name (str), or file path.
        Transparently falls back to individual .bin or .pwl files if needed.
        Restricted via call-stack attestation.
        """
        _verify_physics_caller()
        import os
        # 1. Direct tag lookup in bundle
        if isinstance(tag_or_identifier, int):
            bundle = cls.load_bundle()
            if bundle and tag_or_identifier in bundle:
                return bundle[tag_or_identifier]
            raise KeyError(f"Physics tag {tag_or_identifier:#018x} not found in bundle.")
            
        # 2. String identifier / path
        if isinstance(tag_or_identifier, (str, os.PathLike)):
            clean_str = os.fspath(tag_or_identifier).strip()
            stem = os.path.splitext(os.path.basename(clean_str))[0].lower()
            
            # Check if recognized tag alias
            if stem in _STEM_TO_PHYSICS_TAG:
                tag = _STEM_TO_PHYSICS_TAG[stem]
                bundle = cls.load_bundle()
                if bundle and tag in bundle:
                    return bundle[tag]

            # If it is an existing file path on disk, load directly
            if os.path.isfile(clean_str):
                return load_physics_table(clean_str)
                
            # If path doesn't exist directly, try resolving in data_dir
            resolved = resolve_physics_data_file(cls.get_bundle_dir(), clean_str)
            if isinstance(resolved, int):
                bundle = cls.load_bundle()
                if bundle and resolved in bundle:
                    return bundle[resolved]
            elif isinstance(resolved, (str, os.PathLike)) and os.path.isfile(resolved):
                return load_physics_table(resolved)

        raise FileNotFoundError(f"Could not resolve physics table: {tag_or_identifier}")


def get_physics_table(tag_or_identifier):
    """Convenience functional interface for PhysicsBundleManager.get_table."""
    _verify_physics_caller()
    return PhysicsBundleManager.get_table(tag_or_identifier)


def decode_physics_table(raw_bytes):
    """
    Decodes a binary physics table payload or fallback ASCII into an (N, 2) float64 ndarray.
    Guarantees bit-for-bit numerical fidelity.
    """
    import struct
    import io
    import numpy as np
    
    if len(raw_bytes) >= 24 and raw_bytes[:8] == _MAGIC_PHYSICS_HEADER:
        magic, n, flags, reserved = struct.unpack("<8sIIQ", raw_bytes[:24])
        payload_bytes = raw_bytes[24:]
        expected_u64 = n * 2
        scrambled = np.frombuffer(payload_bytes, dtype=np.uint64)
        if len(scrambled) < expected_u64:
            raise ValueError("Corrupted physics binary payload: truncated data.")
        scrambled = scrambled[:expected_u64].copy()
        keystream = _generate_physics_keystream(expected_u64)
        unscrambled = scrambled ^ keystream
        return unscrambled.view(np.float64).reshape(n, 2)
    elif len(raw_bytes) >= 24 and raw_bytes[:8] in (_MAGIC_PHYSICS_BUNDLE_V2, _MAGIC_PHYSICS_BUNDLE_V3):
        bundle = decode_physics_bundle(raw_bytes)
        return next(iter(bundle.values()))
    else:
        text_str = raw_bytes.decode('utf-8', errors='replace')
        return np.loadtxt(io.StringIO(text_str), dtype=np.float64)


def load_physics_table(filepath_or_bytes):
    """
    Loads and decodes a physics table (bundle tag, binary .bin, or legacy ASCII .pwl).
    Returns an (N, 2) contiguous float64 array:
      data[:, 0] -> x coordinates
      data[:, 1] -> y values
    Restricted via call-stack attestation.
    """
    _verify_physics_caller()
    import os
    if isinstance(filepath_or_bytes, int):
        return PhysicsBundleManager.get_table(filepath_or_bytes)
    if isinstance(filepath_or_bytes, (str, os.PathLike)):
        clean_path = os.fspath(filepath_or_bytes)
        stem = os.path.splitext(os.path.basename(clean_path))[0].lower()
        if stem in _STEM_TO_PHYSICS_TAG:
            bundle = PhysicsBundleManager.load_bundle()
            tag = _STEM_TO_PHYSICS_TAG[stem]
            if bundle and tag in bundle:
                return bundle[tag]
        if os.path.isfile(clean_path):
            with open(clean_path, 'rb') as f:
                raw_bytes = f.read()
        else:
            return PhysicsBundleManager.get_table(clean_path)
    else:
        raw_bytes = filepath_or_bytes
    return decode_physics_table(raw_bytes)


def resolve_physics_data_file(data_dir, base_name):
    """
    Resolves the path to a physics data file, preferring bundle tag, then .bin, then .pwl.
    """
    import os
    # Check if bundle exists in data_dir
    bundle_path = os.path.join(data_dir, "molmem_core.bin")
    if os.path.isfile(bundle_path):
        if isinstance(base_name, int):
            return base_name
        stem_str = os.path.splitext(str(base_name))[0].lower()
        if stem_str in _STEM_TO_PHYSICS_TAG:
            return _STEM_TO_PHYSICS_TAG[stem_str]

    # Map tag back to stem if searching for files
    target_stem = None
    if isinstance(base_name, int):
        for s, t in _STEM_TO_PHYSICS_TAG.items():
            if t == base_name:
                target_stem = s
                break
    if target_stem is None:
        target_stem = os.path.splitext(str(base_name))[0]

    for ext in ['.bin', '.pwl']:
        candidate = os.path.join(data_dir, f"{target_stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
            
    if isinstance(base_name, int):
        return base_name
    return os.path.join(data_dir, f"{target_stem}.bin")


_GPU_READY_CACHE = {}

def clear_gpu_ready_cache():
    """Clears the cached GPU readiness evaluation."""
    global _GPU_READY_CACHE
    _GPU_READY_CACHE.clear()


def is_gpu_ready(device_idx=None, return_reason=False):
    """
    Evaluates whether a compatible, supported GPU accelerator is ready and capable
    of executing tensor operations on this system.

    Checks:
      1. CPU-only environment overrides (MOLMEM_CPU_ONLY=1, MOLMEM_FORCE_CPU=1).
      2. PyTorch CUDA/HIP availability and device count > 0.
      3. Valid device ordinal range.
      4. Architecture support (NVIDIA compute capability <= 9.0; ROCm checks).
      5. Execution readiness: dry-run tensor allocation to verify driver/runtime integrity.
      6. For ROCm, checks compiled extension availability if required.

    Returns:
      bool (if return_reason is False)
      (bool, str or None) (if return_reason is True)
    """
    import os
    if os.environ.get("MOLMEM_CPU_ONLY") == "1" or os.environ.get("MOLMEM_FORCE_CPU") == "1":
        reason = "CPU execution forced by environment override (MOLMEM_CPU_ONLY / MOLMEM_FORCE_CPU)"
        return (False, reason) if return_reason else False

    try:
        import torch
    except Exception as e:
        reason = f"PyTorch import failed: {e}"
        return (False, reason) if return_reason else False

    try:
        if not torch.cuda.is_available():
            reason = "torch.cuda.is_available() is False"
            return (False, reason) if return_reason else False
        dev_count = torch.cuda.device_count()
        if dev_count <= 0:
            reason = "No CUDA/HIP devices detected (device_count <= 0)"
            return (False, reason) if return_reason else False
    except Exception as e:
        reason = f"PyTorch CUDA availability query failed: {e}"
        return (False, reason) if return_reason else False

    def _check_single_device(idx):
        if idx in _GPU_READY_CACHE:
            return _GPU_READY_CACHE[idx]

        if idx < 0 or idx >= dev_count:
            res = (False, f"Device index {idx} out of range [0, {dev_count - 1}]")
            _GPU_READY_CACHE[idx] = res
            return res

        is_rocm = getattr(torch.version, 'hip', None) is not None
        try:
            props = torch.cuda.get_device_properties(idx)
            major = getattr(props, 'major', 0)
            minor = getattr(props, 'minor', 0)
            if not is_rocm and major > 9:
                res = (False, f"SM_{major}{minor} exceeds max supported SM_90 for current Molmem_Lib version")
                _GPU_READY_CACHE[idx] = res
                return res
        except Exception as e:
            res = (False, f"Failed to query device properties for GPU {idx}: {e}")
            _GPU_READY_CACHE[idx] = res
            return res

        try:
            from .torch_simulator import check_has_cpp_extension
            if not check_has_cpp_extension(device_idx=idx):
                backend_name = "ROCm" if is_rocm else "CUDA"
                res = (False, f"{backend_name} acceleration requires compiled C++ extension which is unavailable")
                _GPU_READY_CACHE[idx] = res
                return res
        except Exception as e:
            backend_name = "ROCm" if is_rocm else "CUDA"
            res = (False, f"{backend_name} extension verification failed: {e}")
            _GPU_READY_CACHE[idx] = res
            return res

        try:
            test_t = torch.zeros(1, device=f'cuda:{idx}')
            del test_t
        except Exception as e:
            res = (False, f"GPU runtime allocation test failed on cuda:{idx}: {e}")
            _GPU_READY_CACHE[idx] = res
            return res

        res = (True, None)
        _GPU_READY_CACHE[idx] = res
        return res

    if device_idx is not None:
        target_idx = 0
        if isinstance(device_idx, str):
            if ':' in device_idx:
                try:
                    target_idx = int(device_idx.split(':')[1])
                except Exception:
                    target_idx = 0
            elif device_idx.isdigit():
                target_idx = int(device_idx)
        else:
            try:
                target_idx = int(device_idx)
            except Exception:
                target_idx = 0
        ready, reason = _check_single_device(target_idx)
        return (ready, reason) if return_reason else ready

    # device_idx is None: check MOLMEM_SELECTED_GPU first
    env_sel = os.environ.get('MOLMEM_SELECTED_GPU')
    if env_sel is not None:
        try:
            sel_idx = int(env_sel)
            ready, reason = _check_single_device(sel_idx)
            if ready:
                return (True, None) if return_reason else True
        except Exception:
            pass

    # If selected GPU was not ready or not set, scan all devices for any ready GPU
    last_reason = "No GPU devices found"
    for i in range(dev_count):
        ready, reason = _check_single_device(i)
        if ready:
            return (True, None) if return_reason else True
        last_reason = reason

    return (False, last_reason) if return_reason else False
