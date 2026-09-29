import time
import os
import sys
import math
import contextlib
import concurrent.futures
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r"(?s).*is not compatible with the current PyTorch installation.*")
warnings.filterwarnings("ignore", message=r"(?s).*is not compatible with the current PyTorch installation.*")
warnings.filterwarnings("ignore", message=r"[\s\S]*is not compatible with the current PyTorch installation[\s\S]*")

# Suppress PyTorch TorchScript deprecation warnings (torch.jit.load is deprecated. Please switch to torch.export.)
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*`?torch\.jit\.load`? is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*torch\.jit\._serialization.*")
warnings.filterwarnings("ignore", message=r".*`?torch\.jit\.load`? is deprecated.*")

import torch
import numpy as np
from typing import Tuple
from .history import LazyHistoryDict, LazyDeviceHistoryDict
from .sources import DcSource, PwlSource, BSource, BaseSource, PulseSource
from .sys_utils import DummyStream, _find_rocm_path, _find_cuda_path

if sys.stdout is None:
    sys.stdout = DummyStream()

HAS_CPP_EXTENSION = False
CPP_EXTENSION_DISABLED_REASON = None
CPP_EXTENSION_COMPILED_ARCHS = []
ACTIVE_GPU_ARCH = None
molmem_cuda_solver = None
_CSRC_DIR = os.path.join(os.path.dirname(__file__), 'csrc')
_compiled_loop_fn = None
_compiled_loop_path = os.path.join(_CSRC_DIR, "torch_transient_loop.pt")

def _query_rocm_device_arch(device_idx=0):
    """
    Directly query AMD ROCm driver / kernel interfaces for the target GFX architecture,
    avoiding brittle marketing name matching.
    """
    import shutil
    # 1. Linux sysfs KFD topology (AMDGPU kernel driver exposes gfx_target_version per GPU node)
    if sys.platform != 'win32':
        kfd_nodes = "/sys/class/kfd/kfd/topology/nodes"
        if os.path.isdir(kfd_nodes):
            try:
                gpu_count = 0
                for node in sorted(os.listdir(kfd_nodes), key=lambda x: int(x) if x.isdigit() else 999):
                    prop_path = os.path.join(kfd_nodes, node, "properties")
                    if os.path.isfile(prop_path):
                        with open(prop_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        props_map = {}
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                props_map[parts[0]] = parts[1]
                        gpu_id = int(props_map.get("gpu_id", 0))
                        if gpu_id > 0:
                            if gpu_count == device_idx:
                                gfx_ver = int(props_map.get("gfx_target_version", 0))
                                if gfx_ver > 0:
                                    maj = gfx_ver // 10000
                                    min_v = (gfx_ver % 10000) // 100
                                    step = gfx_ver % 100
                                    if step > 0:
                                        return f"gfx{maj}{min_v:02d}{step:01d}"
                                    else:
                                        return f"gfx{maj}{min_v:02d}"
                            gpu_count += 1
            except Exception:
                pass

    # 2. Query rocminfo if available
    try:
        import subprocess
        rocm_info_path = shutil.which("rocminfo")
        if not rocm_info_path:
            for cand in ["/opt/rocm/bin/rocminfo", r"C:\Program Files\AMD\ROCm\bin\rocminfo.exe"]:
                if os.path.exists(cand):
                    rocm_info_path = cand
                    break
        if rocm_info_path:
            out = subprocess.check_output([rocm_info_path], stderr=subprocess.DEVNULL, timeout=5).decode("utf-8", errors="ignore")
            agents_gfx = []
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Name:") and "gfx" in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith("gfx"):
                            agents_gfx.append(p.lower())
                            break
            if device_idx < len(agents_gfx):
                return agents_gfx[device_idx]
    except Exception:
        pass

    return None

def get_active_gpu_arch(device_idx=None):
    """Retrieve the hardware architecture target of the active GPU (e.g. 'gfx1150', 'sm_89')."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if device_idx is None:
            gpu_env = os.environ.get('MOLMEM_SELECTED_GPU')
            if gpu_env is not None:
                try:
                    device_idx = int(gpu_env)
                except Exception:
                    device_idx = None
            if device_idx is None:
                try:
                    device_idx = torch.cuda.current_device()
                except Exception:
                    device_idx = 0

        is_rocm = getattr(torch.version, 'hip', None) is not None
        if not is_rocm:
            # NVIDIA CUDA: If the candidate GPU exceeds max supported capability (SM_90), scan for a valid supported GPU
            try:
                major, _ = torch.cuda.get_device_capability(device_idx)
                if major > 9:
                    for d_i in range(torch.cuda.device_count()):
                        d_maj, _ = torch.cuda.get_device_capability(d_i)
                        if d_maj <= 9:
                            device_idx = d_i
                            break
            except Exception:
                pass
            major, minor = torch.cuda.get_device_capability(device_idx)
            return f"sm_{major}{minor}"
        else:
            # AMD ROCm / HIP: Query architecture target from driver/runtime
            props = torch.cuda.get_device_properties(device_idx)
            gcn = getattr(props, 'gcnArchName', None)
            if gcn:
                return gcn.split(':')[0].strip().lower()

            # Driver/system topology fallback (without marketing name heuristics)
            arch = _query_rocm_device_arch(device_idx)
            if arch:
                return arch

            # Direct GFX IP fallback from PyTorch HIP properties
            major = getattr(props, 'major', 0)
            minor = getattr(props, 'minor', 0)
            if major > 0:
                return f"gfx{major}{minor:02d}" if minor >= 10 else f"gfx{major}{minor}0"
    except Exception:
        pass
    return None

_CACHED_COMPILED_ARCHS = None

def clear_compiled_archs_cache():
    """Invalidates the in-memory cache of compiled extension architectures and status."""
    global _CACHED_COMPILED_ARCHS, HAS_CPP_EXTENSION, CPP_EXTENSION_DISABLED_REASON, ACTIVE_GPU_ARCH
    _CACHED_COMPILED_ARCHS = None
    HAS_CPP_EXTENSION = False
    CPP_EXTENSION_DISABLED_REASON = None
    ACTIVE_GPU_ARCH = None

def get_compiled_extension_archs():
    """Retrieve list of GPU architectures targeted when the native C++ extension was compiled."""
    global _CACHED_COMPILED_ARCHS
    if _CACHED_COMPILED_ARCHS is not None:
        return _CACHED_COMPILED_ARCHS

    is_rocm = False
    try:
        import torch
        is_rocm = getattr(torch.version, 'hip', None) is not None
    except Exception:
        pass

    arch_file = os.path.join(_CSRC_DIR, ".csrc_arch")
    if os.path.isfile(arch_file):
        try:
            with open(arch_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    raw = [x.strip().lower() for x in content.replace(";", " ").replace(",", " ").split()]
                    res = [x for x in raw if x]
                    if res:
                        # Check platform consistency: if running on ROCm but .csrc_arch has CUDA targets (or vice versa),
                        # the .csrc_arch file is stale from a previous cross-machine/cross-platform build.
                        has_gfx = any(x.startswith("gfx") for x in res)
                        has_cuda = any(not x.startswith("gfx") for x in res)
                        if (is_rocm and not has_gfx and has_cuda) or (not is_rocm and has_gfx and not has_cuda):
                            try:
                                os.remove(arch_file)
                            except Exception:
                                pass
                        else:
                            _CACHED_COMPILED_ARCHS = res
                            return _CACHED_COMPILED_ARCHS
        except Exception:
            pass

    candidates = [
        "gfx906", "gfx908", "gfx90a", "gfx942",
        "gfx1010", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1035", "gfx1036",
        "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
        "gfx1200", "gfx1201",
        "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"
    ]
    found = []
    lib_dir = os.path.dirname(_CSRC_DIR)
    for search_dir in [_CSRC_DIR, lib_dir]:
        if os.path.isdir(search_dir):
            for fname in os.listdir(search_dir):
                if fname.startswith("molmem_cuda_solver") and (fname.endswith(".pyd") or fname.endswith(".so")):
                    fpath = os.path.join(search_dir, fname)
                    try:
                        with open(fpath, "rb") as f:
                            bin_data = f.read()
                            for c in candidates:
                                if c.encode("ascii") in bin_data:
                                    found.append(c)
                    except Exception:
                        pass
                    if found:
                        break
            if found:
                break
    if found:
        _CACHED_COMPILED_ARCHS = sorted(list(set(found)))
        return _CACHED_COMPILED_ARCHS
    _CACHED_COMPILED_ARCHS = []
    return _CACHED_COMPILED_ARCHS

def _is_gpu_arch_compatible(active_arch, compiled_archs):
    """
    Evaluates whether the active hardware architecture matches any of the targets
    compiled into the native C++/HIP extension. Handles:
      - ROCm GFX strings (e.g., 'gfx1100', 'gfx1200'), fully exempt from SM constraints
      - CUDA compute notations ('sm_86', 'compute_86', '8.6', '8.6+ptx') with strict <= SM_90 ceiling
    """
    if not active_arch or not compiled_archs:
        return True

    act = str(active_arch).strip().lower()

    # 1. AMD ROCm GFX target handling (completely exempt from NVIDIA SM checks)
    if act.startswith("gfx"):
        for c in compiled_archs:
            c_clean = str(c).strip().lower()
            if act == c_clean or act == c_clean.split('+')[0] or act == c_clean.split(':')[0]:
                return True
        return False

    # 2. NVIDIA CUDA Architecture Check: Enforce strict SM_90 ceiling
    import re
    m_act = re.match(r"(?:sm_|compute_)?(\d+)(?:\.(\d+))?$", act)
    if m_act:
        act_num_str = m_act.group(1)
        if m_act.group(2) is not None:
            act_major = int(m_act.group(1))
            act_minor = int(m_act.group(2))
        elif len(act_num_str) >= 2:
            act_major = int(act_num_str[:-1])
            act_minor = int(act_num_str[-1])
        else:
            act_major = int(act_num_str)
            act_minor = 0

        # Strict protection: Any NVIDIA architecture exceeding SM_90 (e.g. Blackwell SM_100, RTX 50-series SM_120) is unsupported
        if act_major > 9:
            return False

        for c in compiled_archs:
            c_str = str(c).strip().lower()
            if c_str.startswith("gfx"):
                continue
            has_ptx = "+ptx" in c_str
            c_base = c_str.replace("+ptx", "").strip()
            m_comp = re.match(r"(?:sm_|compute_)?(\d+)(?:\.(\d+))?$", c_base)
            if m_comp:
                if m_comp.group(2) is not None:
                    c_major = int(m_comp.group(1))
                    c_minor = int(m_comp.group(2))
                elif len(m_comp.group(1)) >= 2:
                    c_major = int(m_comp.group(1)[:-1])
                    c_minor = int(m_comp.group(1)[-1])
                else:
                    c_major = int(m_comp.group(1))
                    c_minor = 0

                # Strict protection: compiled targets exceeding SM_90 are also rejected
                if c_major > 9:
                    continue

                # Exact architecture match (e.g. 8.6 matches sm_86)
                if act_major == c_major and act_minor == c_minor:
                    return True
                # Forward compatibility via PTX JIT (capped at max supported SM_90 for Molmem)
                if has_ptx and (act_major, act_minor) >= (c_major, c_minor) and act_major <= 9:
                    return True

    return False

def check_has_cpp_extension(device_idx=None):
    global HAS_CPP_EXTENSION, CPP_EXTENSION_DISABLED_REASON, CPP_EXTENSION_COMPILED_ARCHS, ACTIVE_GPU_ARCH, molmem_cuda_solver
    active_arch = get_active_gpu_arch(device_idx)
    ACTIVE_GPU_ARCH = active_arch
    compiled_archs = get_compiled_extension_archs()
    CPP_EXTENSION_COMPILED_ARCHS = compiled_archs

    if HAS_CPP_EXTENSION and molmem_cuda_solver is not None:
        if active_arch and compiled_archs:
            if not _is_gpu_arch_compatible(active_arch, compiled_archs):
                HAS_CPP_EXTENSION = False
                CPP_EXTENSION_DISABLED_REASON = (
                    f"Architecture Mismatch (Active GPU is {active_arch}, but extension only targets {', '.join(compiled_archs)})"
                )
                return False
        return True

    # Import main package to check background compiler subprocess
    import molmem_lib
    import multiprocessing
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=False)
    except (RuntimeError, ValueError):
        pass
    is_coordinator = (multiprocessing.current_process().name == 'MainProcess')
    if is_coordinator and hasattr(molmem_lib, "_wait_for_bg_extension_compile"):
        try:
            molmem_lib._wait_for_bg_extension_compile(verbose=True)
        except Exception:
            pass

    # Upfront hardware architecture verification
    if active_arch and compiled_archs:
        is_matched = _is_gpu_arch_compatible(active_arch, compiled_archs)
        if not is_matched:
            HAS_CPP_EXTENSION = False
            CPP_EXTENSION_DISABLED_REASON = (
                f"Architecture Mismatch (Active GPU is {active_arch}, but extension only targets {', '.join(compiled_archs)})"
            )
            return False

    try:
        csrc_dir = _CSRC_DIR
        lib_dir = os.path.dirname(_CSRC_DIR)
        if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
            for d in [csrc_dir, lib_dir]:
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
            rocm_path = _find_rocm_path()
            if rocm_path:
                rocm_bin = os.path.join(rocm_path, 'bin')
                if os.path.isdir(rocm_bin):
                    try:
                        os.add_dll_directory(rocm_bin)
                    except Exception:
                        pass
            cuda_path = _find_cuda_path()
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, 'bin')
                if os.path.isdir(cuda_bin):
                    try:
                        os.add_dll_directory(cuda_bin)
                    except Exception:
                        pass
        if csrc_dir not in sys.path:
            sys.path.insert(0, csrc_dir)
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        import molmem_cuda_solver as solver
        molmem_cuda_solver = solver
        HAS_CPP_EXTENSION = True
        CPP_EXTENSION_DISABLED_REASON = None
    except Exception as e:
        HAS_CPP_EXTENSION = False
        if isinstance(e, (ModuleNotFoundError, ImportError)) and "molmem_cuda_solver" in str(e):
            CPP_EXTENSION_DISABLED_REASON = None
        else:
            CPP_EXTENSION_DISABLED_REASON = f"Import Error: {e}"
        # Self-healing: if an existing binary is incompatible or linked against missing dynamic libraries
        # (e.g. AMD ROCm .so on an NVIDIA system or vice versa), evict the corrupt/incompatible binary
        # and invalidate the compilation hash to trigger a clean native rebuild on next run.
        try:
            err_str = str(e).lower()
            if any(k in err_str for k in ["cannot open shared object file", "dll load failed", "undefined symbol", "not found", "libamdhip", "libcudart"]):
                for target_d in [csrc_dir, lib_dir]:
                    if os.path.isdir(target_d):
                        for f_name in os.listdir(target_d):
                            if f_name.startswith("molmem_cuda_solver") and (f_name.endswith(".pyd") or f_name.endswith(".so")):
                                try:
                                    os.remove(os.path.join(target_d, f_name))
                                except Exception:
                                    pass
                hash_file = os.path.join(csrc_dir, ".csrc_hash")
                if os.path.exists(hash_file):
                    try:
                        os.remove(hash_file)
                    except Exception:
                        pass
        except Exception:
            pass
    return HAS_CPP_EXTENSION

# Perform an initial quick check without waiting (lazy load)
try:
    check_has_cpp_extension()
except Exception:
    pass

import threading
_GPU_KERNEL_LAUNCH_LOCK = threading.Lock()

_GLOBAL_GPU_STREAM_POOL = {}

def get_gpu_stream_bundle(device, stream=None):
    """Retrieve or allocate process-wide cached CUDA/HIP streams and events for a device and active compute stream."""
    stream_id = getattr(stream, 'cuda_stream', 0) if stream is not None else 0
    pool_key = f"{device}_{stream_id}"
    if pool_key not in _GLOBAL_GPU_STREAM_POOL:
        if torch.cuda.is_available() and device.type == 'cuda':
            with torch.cuda.device(device):
                try:
                    copy_st = torch.cuda.Stream(device, priority=0)
                except Exception:
                    copy_st = torch.cuda.Stream(device)
                _GLOBAL_GPU_STREAM_POOL[pool_key] = {
                    'copy_stream': copy_st,
                    'chunk_event': torch.cuda.Event(),
                    'copy_event_slot_a': torch.cuda.Event(),
                    'copy_event_slot_b': torch.cuda.Event()
                }
        else:
            _GLOBAL_GPU_STREAM_POOL[pool_key] = {
                'copy_stream': None,
                'chunk_event': None,
                'copy_event_slot_a': None,
                'copy_event_slot_b': None
            }
    return _GLOBAL_GPU_STREAM_POOL[pool_key]

def _cleanup_global_gpu_streams():
    """Flushes and synchronizes all active GPU streams before process termination."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

import atexit
atexit.register(_cleanup_global_gpu_streams)


def torch_integrate_read_currents(
    iHist_buffer: torch.Tensor,
    tArr_buffer: torch.Tensor,
    dev_indices: torch.Tensor,
    rows: int,
    cols: int,
    iMax_single: float,
    numLevels: int,
    maxPulseWidth: float,
    hist_idx: int
) -> torch.Tensor:
    if hist_idx <= 1:
        batch_sz = tArr_buffer.shape[0] if (tArr_buffer.ndim == 2 and tArr_buffer.shape[0] != hist_idx) else (tArr_buffer.shape[1] if tArr_buffer.ndim == 2 else 1)
        return torch.zeros((batch_sz, cols), dtype=torch.float32, device=tArr_buffer.device)
        
    if iHist_buffer.ndim == 3 and tArr_buffer.ndim == 2:
        if iHist_buffer.shape[0] != tArr_buffer.shape[0] and iHist_buffer.shape[2] == tArr_buffer.shape[0]:
            iHist_buffer = iHist_buffer.permute(2, 0, 1).contiguous()
            tArr_buffer = tArr_buffer.permute(1, 0).contiguous()
        else:
            iHist_buffer = iHist_buffer.contiguous()
            tArr_buffer = tArr_buffer.contiguous()
        
    i_dev_hist = torch.index_select(iHist_buffer, 1, dev_indices)[:, :, :hist_idx]
    levels_minus_1 = float(numLevels - 1)
    inv_denom = 1.0 / (maxPulseWidth * levels_minus_1)
    if rows == 1:
        colAnalogCurrentsMatrix = i_dev_hist.view(iHist_buffer.shape[0], cols, hist_idx)
        scale = levels_minus_1 / (iMax_single + 1e-12)
        inv_max_integral = 0.5 * inv_denom
    else:
        i_hist_matrix = i_dev_hist.view(iHist_buffer.shape[0], rows, cols, hist_idx)
        colAnalogCurrentsMatrix = torch.sum(i_hist_matrix, dim=1)
        scale = levels_minus_1 / (iMax_single * rows + 1e-12)
        inv_max_integral = (0.5 * rows) * inv_denom
        
    colIHistDigitalMatrix = torch.clamp(colAnalogCurrentsMatrix * scale, 0.0, levels_minus_1).round_()
    
    dt_raw = tArr_buffer[:, 1:hist_idx] - tArr_buffer[:, :hist_idx - 1]
    dt_steps = torch.where((dt_raw > 0.0) & (dt_raw <= maxPulseWidth * 2.0), dt_raw, torch.zeros_like(dt_raw))
    y_sum = colIHistDigitalMatrix[:, :, 1:hist_idx] + colIHistDigitalMatrix[:, :, :hist_idx - 1]
    yIntegratedBits = torch.sum(y_sum * dt_steps.unsqueeze(1), dim=2)
    
    yNormalized = yIntegratedBits * inv_max_integral
    if os.environ.get('MOLMEM_DEBUG_GPU', '0') == '1':
        print(f"[DEBUG_INT] hist_idx={hist_idx} | colAnalog max={colAnalogCurrentsMatrix.max().item():.4e} | dt_steps min={dt_steps.min().item():.4e} max={dt_steps.max().item():.4e} | yIntegratedBits[0]={yIntegratedBits[0].cpu().numpy()} | inv_max_integral={inv_max_integral}")
    return yNormalized

def torch_gpu_batch_pass_input(
    pulse_widths_batch: torch.Tensor,
    currentIScale_tensor: torch.Tensor,
    currentVScale_tensor: torch.Tensor,
    currentF22_tensor: torch.Tensor,
    rC_tensor: torch.Tensor,
    vRead: float,
    rows: int,
    cols: int,
    adc_iMin: float,
    adc_iMax: float,
    numLevels: int,
    maxPulseWidth: float
) -> torch.Tensor:
    """
    Direct Fused GPU Parallel Batch Inference Solver (Zero Launch Latency / Zero ODE overhead).
    Evaluates all B batch items across GPU tensor cores in a single fused pass.
    """
    device = pulse_widths_batch.device
    dtype = currentIScale_tensor.dtype
    batch_size = pulse_widths_batch.shape[0]

    # 1. Precompute steady-state cell currents under vRead
    vS = currentVScale_tensor + 1e-12
    inv_vS = 1.0 / vS
    i_prod = currentIScale_tensor * currentF22_tensor
    g_base = i_prod * inv_vS
    rSeries = 2.0 * rC_tensor
    v = vRead * inv_vS
    coeff = rSeries * g_base

    # Solve series contact resistance Newton-Raphson: x + coeff * sinh(x) = v
    y = v / (coeff + 1e-20)
    asinh_y = torch.asinh(y)
    x = torch.where(coeff <= 1e-18, v, torch.clamp(v, max=asinh_y))
    
    # 6 Newton-Raphson vectorized steps across all crossbar cells in parallel
    for _ in range(6):
        sinh_x = torch.sinh(x)
        cosh_x = torch.cosh(x)
        fx = x + coeff * sinh_x - v
        fpx = 1.0 + coeff * cosh_x
        dx = fx / fpx
        x = x - dx

    sinh_final = torch.sinh(x)
    i_cell = torch.where(currentF22_tensor == 0.0, torch.zeros_like(i_prod), i_prod * sinh_final)

    # 2. Vectorized PWM intervals across the entire batch
    sorted_pw, _ = torch.sort(pulse_widths_batch, dim=1)
    edges = torch.cat([torch.zeros((batch_size, 1), dtype=dtype, device=device), sorted_pw], dim=1)
    dt_k = edges[:, 1:] - edges[:, :-1]
    t_mid = edges[:, 1:]

    # Broadcasted active mask: pulse_widths_batch is (B, R_rows, 1), t_mid is (B, 1, R_intervals)
    active_mask = (pulse_widths_batch.unsqueeze(2) >= (t_mid.unsqueeze(1) - 1e-15)).to(dtype)

    # Column current for each interval: i_col = sum_r (active_mask[b, r, k] * i_cell[r, c])
    i_col = torch.einsum('brk,rc->bkc', active_mask, i_cell)

    # 3. ADC Quantization
    levels_minus_1 = float(numLevels - 1)
    adc_range = adc_iMax - adc_iMin if adc_iMax > adc_iMin else 1e-12
    inv_adc_range = 1.0 / adc_range
    i_clamped = torch.clamp(i_col, adc_iMin, adc_iMax)
    d_bits = torch.round((i_clamped - adc_iMin) * (inv_adc_range * levels_minus_1))

    # 4. Temporal Charge Integration and Normalization
    q_sum = torch.sum(d_bits * dt_k.unsqueeze(2), dim=1)
    inv_max_integral = float(rows) / (levels_minus_1 * maxPulseWidth)
    yNormalized = (q_sum * inv_max_integral).to(torch.float32)

    return yNormalized

def to_device_async(tensor, device):
    if device.type == 'cuda' and not tensor.is_cuda and not tensor.is_pinned() and tensor.numel() >= 1024:
        try:
            return tensor.pin_memory().to(device, non_blocking=True)
        except Exception:
            return tensor.to(device, non_blocking=True)
    return tensor.to(device, non_blocking=True)

class StreamProxyDict(dict):
    _STATIC_KEYS = frozenset({'device', 'num_devices', 'num_v_sources', 'num_dyn_sources', 'batch_size', 'master_schedule_cpu'})

    def __init__(self, target_dict, suffix):
        super().__init__()
        self.target_dict = target_dict
        self.suffix = suffix
        self._no_suffix = (not suffix)
        
    def __getitem__(self, key):
        if self._no_suffix or key in self._STATIC_KEYS:
            return self.target_dict[key]
        return self.target_dict[key + self.suffix]
        
    def __setitem__(self, key, value):
        if self._no_suffix or key in self._STATIC_KEYS:
            self.target_dict[key] = value
        else:
            self.target_dict[key + self.suffix] = value
            
    def __contains__(self, key):
        if self._no_suffix or key in self._STATIC_KEYS:
            return key in self.target_dict
        return (key + self.suffix) in self.target_dict
        
    def get(self, key, default=None):
        if self._no_suffix or key in self._STATIC_KEYS:
            return self.target_dict.get(key, default)
        return self.target_dict.get(key + self.suffix, default)
        
    def pop(self, key, default=None):
        if self._no_suffix or key in self._STATIC_KEYS:
            return self.target_dict.pop(key, default)
        return self.target_dict.pop(key + self.suffix, default)
        
    def update(self, other):
        if not other:
            return
        static_keys = self._STATIC_KEYS
        suffix = self.suffix
        self.target_dict.update({(k if k in static_keys else k + suffix): v for k, v in other.items()})

from .sys_utils import handle_oom

@handle_oom
def run_torch_simulation(
    sim, tEnd, min_dt=1e-12, tol=1e-3, maxIter=100, title=None, appendHistory=False, recordHistory=False,
    batch_size=1, batched_v_pwl_t=None, batched_v_pwl_v=None,
    batched_dyn_pwl_t=None, batched_dyn_pwl_v=None, initial_state_data=None, pwl_mask_tensor=None, stream=None,
    sync_readback=None, sync_states_back=True, sync_stream=None
):
    """
    Main entry point for PyTorch-based transient simulation with native 3D batching.
    """
    from .sys_utils import resolve_record_history_flags, validate_isolation_safety
    validate_isolation_safety(sim)
    v_flag, i_flag, state_flag, f22_flag, g_flag, temp_flag, bitmask = resolve_record_history_flags(recordHistory)
    record_hist_bool = bool(recordHistory) if isinstance(recordHistory, bool) else (bitmask != 0)
    check_has_cpp_extension()
    if sync_stream is None:
        sync_stream = (os.environ.get('MOLMEM_DEBUG_GPU', '0') == '1') or (stream is None)
    if sync_readback is None:
        t_start_calc = 0.0
        if appendHistory and getattr(sim, 'tArr', None) is not None and len(sim.tArr) > 0:
            t_start_calc = sim.tArr[-1]
        if batch_size > 1:
            sync_readback = False
        elif os.name == 'nt' and (tEnd - t_start_calc) > 1e-4:
            sync_readback = True
        else:
            sync_readback = record_hist_bool
    from .sys_utils import is_gpu_ready
    torch_device_str = getattr(sim, 'device', 'cuda')
    if torch_device_str == 'auto':
        torch_device_str = 'cuda' if is_gpu_ready() else 'cpu'
    if torch_device_str.startswith('cuda') or torch_device_str == 'gpu':
        if not is_gpu_ready(torch_device_str):
            torch_device_str = 'cpu'
        elif ':' in torch_device_str:
            try:
                dev_idx = int(torch_device_str.split(':')[1])
                if not is_gpu_ready(dev_idx):
                    torch_device_str = 'cpu'
                else:
                    torch_device_str = f'cuda:{dev_idx}'
            except (ValueError, IndexError):
                torch_device_str = 'cpu'
        else:
            default_gpu = os.environ.get('MOLMEM_SELECTED_GPU')
            if default_gpu is not None and is_gpu_ready(default_gpu):
                torch_device_str = f'cuda:{default_gpu}'
            elif is_gpu_ready(0):
                torch_device_str = 'cuda:0'
            else:
                torch_device_str = 'cpu'
    device = torch.device(torch_device_str)
    if device.type == 'cuda':
        dev_idx = device.index if device.index is not None else 0
        check_has_cpp_extension(device_idx=dev_idx)
        if HAS_CPP_EXTENSION and os.environ.get("MOLMEM_FORCE_JIT") != "1":
            sim.compile_backend = True
    dtype = torch.float64
    
    numNodes = len(sim.nodesUnknown)
    num_devices = len(sim.devices)
    
    detailed = getattr(sim, 'detailedPrint', True)
    verbose = detailed and getattr(sim, 'verbose', True)
    import multiprocessing
    from .simulator import MolmemSimulator
    is_profiling = (getattr(MolmemSimulator, '_in_bootstrap_profiling', False) or 
                    getattr(MolmemSimulator, '_in_gpu_crossover_profiling', False) or
                    getattr(MolmemSimulator, '_in_parallel_profiling', False) or
                    getattr(sim, '_in_warmup', False) or
                    getattr(sim.__class__, '_in_warmup', False) or
                    getattr(sim, '_is_calibration_helper', False) or
                    multiprocessing.current_process().name != 'MainProcess')
    if is_profiling or title is None:
        verbose = False
        detailed = False
        
    backend_desc = "C++ Compiled" if (HAS_CPP_EXTENSION and getattr(sim, 'compile_backend', False)) else "JIT Fallback"
    prec_str = "FP64"
    if verbose and detailed and title:
        # Resolve dimensions/layout of the crossbar
        if hasattr(sim, 'wordlines') and hasattr(sim, 'bitlines') and sim.wordlines and sim.bitlines:
            layout_str = f" ({len(sim.wordlines)}x{len(sim.bitlines)})"
        elif hasattr(sim, 'crossbarDevices') and sim.crossbarDevices:
            rows = max(r for r, _ in sim.crossbarDevices)
            cols = max(c for _, c in sim.crossbarDevices)
            layout_str = f" ({rows}x{cols})"
        else:
            layout_str = ""

        # Resolve executed backend and precision description
        is_hybrid = getattr(sim, 'backend', 'auto') in ['hybrid', 'auto'] and os.environ.get("MOLMEM_FORCE_GPU") != "1" and os.environ.get("MOLMEM_FORCE_CPU") != "1"
        is_rocm = getattr(getattr(torch, 'version', None), 'hip', None) is not None or 'rocm' in str(device).lower()
        gpu_type = "ROCm" if is_rocm else "CUDA"
        if is_hybrid:
            dev_desc = f"Heterogeneous Hybrid (CPU + GPU {gpu_type})"
        else:
            dev_desc = f"GPU ({gpu_type})"
        
        banner_text = f"MolmemSimulator: {title}"
        banner_line = "-" * 62
        mid_line = f"--- {banner_text} ".ljust(59) + "---"
        
        print(f"\n{banner_line}\n{mid_line}\n{banner_line}")
        print(f"  > Target Array          : {num_devices} Devices{layout_str}")
        print(f"  > Solver Topology       : Dense Solver (Nodes: {numNodes} Unknown, {len(sim.nodesKnown)} Known)")
        print(f"  > Executed Backend      : {dev_desc} ({backend_desc} | {prec_str} | Batch Size: {batch_size})\n")
        
    try:
        from .sys_utils import log_to_run_log_only
        backend_full = f"{gpu_type} C++ Extension" if (HAS_CPP_EXTENSION and getattr(sim, 'compile_backend', False)) else "TorchScript JIT Compiler"
        prec_full = "FP64 (Double Precision)"
        stream_id = str(stream.cuda_stream) if (stream is not None and hasattr(stream, 'cuda_stream')) else "0 (Default/NULL Stream)"
        log_to_run_log_only(
            f"[GPU_EXEC] Submitting simulation execution batch to GPU device | Device: {device} | Backend: {backend_full} | Precision: {prec_full} | Batch Size: {batch_size} | CUDA/HIP Stream: {stream_id}"
        )
    except Exception:
        pass
    num_v_sources = len(sim.vSources)
    num_dyn_sources = len(sim.dynamicSources)
    
    is_1t1r = getattr(sim, 'architecture', '1R') == '1T1R'
    leakage_g = getattr(sim, 'leakage_g', 1e-10)

    stream_key = f"_stream_{stream.cuda_stream}" if (stream is not None and hasattr(stream, 'cuda_stream') and stream.cuda_stream != 0) else ""
    raw_cache = getattr(sim, '_gpu_cache', None)
    
    # Invalidate cache completely if configuration properties changed (e.g. batch_size, num_devices)
    config_match = (raw_cache is not None and 
                    raw_cache.get('device') == device and 
                    raw_cache.get('num_devices') == num_devices and
                    raw_cache.get('num_v_sources') == num_v_sources and
                    raw_cache.get('num_dyn_sources') == num_dyn_sources and
                    raw_cache.get('batch_size', 1) == batch_size)
                    
    if raw_cache is not None and not config_match:
        try:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"[GPU_CACHE_INVALIDATE] Cache invalidated | OldConfig={raw_cache.get('num_devices')} devs, Batch={raw_cache.get('batch_size')} -> NewConfig={num_devices} devs, Batch={batch_size}")
        except Exception:
            pass
        sim._gpu_cache = None
        raw_cache = None
        
    if raw_cache is not None:
        # Prune cache if active stream count exceeds Compute Unit parallelism limit
        cached_streams = set()
        for key in list(raw_cache.keys()):
            if '_stream_' in key:
                stream_id = key.split('_stream_')[-1]
                cached_streams.add(stream_id)
        max_streams = torch.cuda.get_device_properties(device).multi_processor_count if device.type == 'cuda' else 32
        if len(cached_streams) > max_streams:
            current_stream_id = str(stream.cuda_stream) if (stream is not None and hasattr(stream, 'cuda_stream')) else ""
            streams_to_remove = [sid for sid in cached_streams if sid != current_stream_id]
            streams_to_remove = streams_to_remove[:len(streams_to_remove)//2 + 1]
            for sid in streams_to_remove:
                suffix = f"_stream_{sid}"
                for key in list(raw_cache.keys()):
                    if key.endswith(suffix):
                        del raw_cache[key]
            torch.cuda.empty_cache()

    gpu_cache_hit = False
    
    if (raw_cache is not None and config_match and ('device_states_soa' + stream_key) in raw_cache):
        gpu_cache_hit = True

    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"[GPU_CACHE_LOOKUP] Status={'HIT' if gpu_cache_hit else 'MISS'} | StreamKey='{stream_key}' | Devices={num_devices} | Batch={batch_size}")
    except Exception:
        pass

    if raw_cache is not None:
        cache = raw_cache if not stream_key else StreamProxyDict(raw_cache, stream_key)
    else:
        cache = None


    static_vCurrent = None
    static_currentIScale = None
    static_currentVScale = None
    static_currentF22 = None
    static_dynamic_G = None
    static_dynamic_I = None
    static_gMat = None
    static_iRes = None
    static_iDev = None
    static_gEq = None
    static_deltaV = None

    if gpu_cache_hit:
        dev_top_idx = cache['dev_top_idx']
        dev_bot_idx = cache['dev_bot_idx']
        dev_top_full_idx = cache['dev_top_full_idx']
        dev_bot_full_idx = cache['dev_bot_full_idx']
        all_nodes_ordered = cache.get('all_nodes_ordered', None)
        
        gMat = cache['gMat']
        iRes = cache['iRes']
        vCurrent_arr = cache['vCurrent_arr']
        
        gpu_vSources = cache['gpu_vSources']
        v_nodes_list = cache['v_nodes_list']
        pwl_mask_list = cache['pwl_mask_list']
        v_pwl_t_matrix = cache['v_pwl_t_matrix']
        v_pwl_v_matrix = cache['v_pwl_v_matrix']
        v_nodes_gpu = cache['v_nodes_gpu']
        v_pwl_mask_gpu = cache['v_pwl_mask_gpu']
        
        gpu_dynSources = cache['gpu_dynSources']
        dyn_nodes_list = cache['dyn_nodes_list']
        dyn_pwl_mask_list = cache['dyn_pwl_mask_list']
        dyn_pwl_t_matrix = cache['dyn_pwl_t_matrix']
        dyn_pwl_v_matrix = cache['dyn_pwl_v_matrix']
        dyn_nodes_gpu = cache['dyn_nodes_gpu']
        dyn_pwl_mask_gpu = cache['dyn_pwl_mask_gpu']
        
        dynamic_G_arr = cache['dynamic_G_arr']
        dynamic_I_arr = cache['dynamic_I_arr']
        dev_to_dyn_src = cache['dev_to_dyn_src']
        dev_both_idx = cache['dev_both_idx']
        
        if batched_v_pwl_t is not None:
            if v_pwl_t_matrix.shape != batched_v_pwl_t.shape or v_pwl_t_matrix.dtype != batched_v_pwl_t.dtype:
                v_pwl_t_matrix = batched_v_pwl_t.to(torch.float64)
                cache['v_pwl_t_matrix'] = v_pwl_t_matrix
            else:
                v_pwl_t_matrix.copy_(batched_v_pwl_t)
            if v_pwl_v_matrix.shape != batched_v_pwl_v.shape or v_pwl_v_matrix.dtype != batched_v_pwl_v.dtype:
                v_pwl_v_matrix = batched_v_pwl_v.to(torch.float64)
                cache['v_pwl_v_matrix'] = v_pwl_v_matrix
            else:
                v_pwl_v_matrix.copy_(batched_v_pwl_v)
        if batched_dyn_pwl_t is not None:
            if dyn_pwl_t_matrix.shape != batched_dyn_pwl_t.shape or dyn_pwl_t_matrix.dtype != batched_dyn_pwl_t.dtype:
                dyn_pwl_t_matrix = batched_dyn_pwl_t.to(torch.float64)
                cache['dyn_pwl_t_matrix'] = dyn_pwl_t_matrix
            else:
                dyn_pwl_t_matrix.copy_(batched_dyn_pwl_t)
            if dyn_pwl_v_matrix.shape != batched_dyn_pwl_v.shape or dyn_pwl_v_matrix.dtype != batched_dyn_pwl_v.dtype:
                dyn_pwl_v_matrix = batched_dyn_pwl_v.to(torch.float64)
                cache['dyn_pwl_v_matrix'] = dyn_pwl_v_matrix
            else:
                dyn_pwl_v_matrix.copy_(batched_dyn_pwl_v)
            
        top_mask = cache.get('top_mask')
        bot_mask = cache.get('bot_mask')
        both_mask = cache.get('both_mask')
        dyn_mask = cache.get('dyn_mask')
        
        top_indices = cache.get('top_indices')
        if top_indices is None and top_mask is not None:
            top_indices = torch.where(top_mask.to('cpu'))[0].to(device)
        bot_indices = cache.get('bot_indices')
        if bot_indices is None and bot_mask is not None:
            bot_indices = torch.where(bot_mask.to('cpu'))[0].to(device)
        both_indices = cache.get('both_indices')
        if both_indices is None and both_mask is not None:
            both_indices = torch.where(both_mask.to('cpu'))[0].to(device)
        dyn_indices = cache.get('dyn_indices')
        if dyn_indices is None and dyn_mask is not None:
            dyn_indices = torch.where(dyn_mask.to('cpu'))[0].to(device)
            
        d_nodes = cache['d_nodes']
        gMat_row_indices = cache['gMat_row_indices']
        gMat_col_indices = cache['gMat_col_indices']
        iRes_indices = cache['iRes_indices']
        
        if num_devices > 0:
            device_states_soa = cache['device_states_soa']
            device_params_soa = cache['device_params_soa']
            vSmooth_arr = cache['vSmooth_arr']
            nSmooth_arr = cache['nSmooth_arr']
            kappa_arr = cache['kappa_arr']
            alpha_arr = cache['alpha_arr']
            vTh_arr = cache['vTh_arr']
            vRefPos_arr = cache['vRefPos_arr']
            vRefNeg_arr = cache['vRefNeg_arr']
            nMax_arr = cache['nMax_arr']
            kDischarge_arr = cache['kDischarge_arr']
            gScale_arr = cache['gScale_arr']
            rC_arr = cache['rC_arr']
            f22StartVal_arr = cache['f22StartVal_arr']
            f22PeakVal_arr = cache['f22PeakVal_arr']
            f22EndVal_arr = cache['f22EndVal_arr']
            f22PotScaleInv_arr = cache['f22PotScaleInv_arr']
            f22DepScaleInv_arr = cache['f22DepScaleInv_arr']
            E_a_arr = cache['E_a_arr']
            R_th_arr = cache['R_th_arr']
            tau_th_arr = cache['tau_th_arr']
            gamma_arr = cache['gamma_arr']
            beta_alpha_arr = cache['beta_alpha_arr']
            beta_s_arr = cache['beta_s_arr']
            vBypass_arr = cache['vBypass_arr']
            
            y_i = cache['y_i']
            y_v = cache['y_v']
            y_fp = cache['y_fp']
            y_fd = cache['y_fd']
            
            n_arr = cache['n_arr']
            modeState_arr = cache['modeState_arr']
            scaleFactor_arr = cache['scaleFactor_arr']
            f22DepScaled_arr = cache['f22DepScaled_arr']
            f22PotScaled_arr = cache['f22PotScaled_arr']
            _cached_n_int_arr = cache['_cached_n_int_arr']
            iScale_arr = cache['iScale_arr']
            vScale_arr = cache['vScale_arr']
            currentF22_arr = cache['currentF22_arr']
            currentIScale_arr = cache['currentIScale_arr']
            currentVScale_arr = cache['currentVScale_arr']
            T_arr = cache['T_arr']
            
            iDev_arr = cache['iDev_arr']
            gEq_arr = cache['gEq_arr']
            
            iRes.zero_()
            vCurrent_arr.zero_()
            dynamic_G_arr.zero_()
            dynamic_I_arr.zero_()
            iDev_arr.zero_()
            gEq_arr.zero_()
            
            if getattr(sim, '_gpu_states_dirty', True) or initial_state_data is not None:
                if initial_state_data is not None:
                    if isinstance(initial_state_data, torch.Tensor):
                        state_tensor = to_device_async(initial_state_data, device)
                    else:
                        state_tensor = to_device_async(torch.from_numpy(initial_state_data), device)
                    if state_tensor.ndim == 2:
                        state_tensor_exp = state_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                    elif state_tensor.shape[0] in (11, 12) and state_tensor.shape[1] == batch_size:
                        state_tensor_exp = state_tensor.permute(1, 0, 2)
                    else:
                        state_tensor_exp = state_tensor
                    if state_tensor_exp.shape[1] >= 12:
                        device_states_soa[:, 0:12, :].copy_(state_tensor_exp[:, 0:12, :])
                    else:
                        device_states_soa[:, 0:5, :].copy_(state_tensor_exp[:, 0:5, :])
                        device_states_soa[:, 6:12, :].copy_(state_tensor_exp[:, 5:11, :])
                        device_states_soa[:, 5, :].fill_(-1.0)
                else:
                    state_data = np.empty((11, num_devices), dtype=np.float64)
                    sanitized_cached_n = None
                    if getattr(sim, 'lazy_sync', True) and not getattr(sim, '_gpu_states_in_sync', True) and getattr(sim, '_gpu_states_cache', None) is not None:
                        cache_st = sim._gpu_states_cache
                        if isinstance(cache_st, dict) and 'n_arr' in cache_st and (cache_st['n_arr'].shape[-1] == num_devices if hasattr(cache_st['n_arr'], 'shape') else len(cache_st['n_arr']) == num_devices):
                            def _extract_host_1d(t):
                                if isinstance(t, torch.Tensor):
                                    arr = t.detach().cpu().numpy()
                                    return arr.ravel()
                                arr = np.asarray(t)
                                return arr.ravel()
                            state_data[0, :] = _extract_host_1d(cache_st['n_arr'])
                            state_data[1, :] = _extract_host_1d(cache_st['modeState_arr'])
                            state_data[2, :] = _extract_host_1d(cache_st['scaleFactor_arr'])
                            state_data[3, :] = _extract_host_1d(cache_st['f22DepScaled_arr'])
                            state_data[4, :] = _extract_host_1d(cache_st['f22PotScaled_arr'])
                            state_data[5, :] = _extract_host_1d(cache_st['iScale_arr'])
                            state_data[6, :] = _extract_host_1d(cache_st['vScale_arr'])
                            state_data[7, :] = _extract_host_1d(cache_st['currentF22_arr'])
                            state_data[8, :] = _extract_host_1d(cache_st['currentIScale_arr'])
                            state_data[9, :] = _extract_host_1d(cache_st['currentVScale_arr'])
                            state_data[10, :] = _extract_host_1d(cache_st['T_arr'])
                            sanitized_cached_n = np.asarray(_extract_host_1d(cache_st['_cached_n_int_arr']), dtype=np.int32)
                    
                    if sanitized_cached_n is None:
                        if not hasattr(sim, '_device_objs_flat') or sim._device_objs_flat is None or len(sim._device_objs_flat) != num_devices:
                            sim._device_objs_flat = [d['dev'] for d in sim.devices]
                        devs = sim._device_objs_flat
                        sim.sync_to_cpu()
                        if num_devices == 1:
                            d = devs[0]
                            state_data[0, 0] = d._n
                            state_data[1, 0] = d._modeState
                            state_data[2, 0] = d._scaleFactor
                            state_data[3, 0] = d._f22DepScaled
                            state_data[4, 0] = d._f22PotScaled
                            state_data[5, 0] = d._iScale
                            state_data[6, 0] = d._vScale
                            state_data[7, 0] = d._currentF22
                            state_data[8, 0] = d._currentIScale
                            state_data[9, 0] = d._currentVScale
                            state_data[10, 0] = d._T
                            sanitized_cached_n = np.array([d._cached_n_int_val], dtype=np.int32)
                        else:
                            d0 = devs[0]
                            d0_n = d0._n
                            d0_mode = d0._modeState
                            d0_scale = d0._scaleFactor
                            d0_fd = d0._f22DepScaled
                            d0_fp = d0._f22PotScaled
                            d0_is = d0._iScale
                            d0_vs = d0._vScale
                            d0_f22 = d0._currentF22
                            d0_cis = d0._currentIScale
                            d0_cvs = d0._currentVScale
                            d0_T = d0._T
                            col0 = np.array([
                                d0_n, d0_mode, d0_scale, d0_fd,
                                d0_fp, d0_is, d0_vs, d0_f22,
                                d0_cis, d0_cvs, d0_T
                            ], dtype=np.float64)
                            c_val0 = d0._cached_n_int_val
                            
                            is_homogeneous = True
                            dl = devs[-1]
                            if (dl._n != d0_n or dl._modeState != d0_mode or 
                                dl._scaleFactor != d0_scale or dl._T != d0_T or
                                dl._currentF22 != d0_f22 or dl._f22PotScaled != d0_fp):
                                is_homogeneous = False
                            else:
                                for d in devs[1:-1]:
                                    if (d._n != d0_n or d._modeState != d0_mode or 
                                        d._scaleFactor != d0_scale or d._T != d0_T or
                                        d._currentF22 != d0_f22 or d._f22PotScaled != d0_fp):
                                        is_homogeneous = False
                                        break
                            
                            if is_homogeneous:
                                state_data[:] = col0[:, None]
                                sanitized_cached_n = np.full(num_devices, c_val0, dtype=np.int32)
                            else:
                                sanitized_cached_n = np.empty(num_devices, dtype=np.int32)
                                def _pack_slice_sim(s, e):
                                    for idx in range(s, e):
                                        d = devs[idx]
                                        state_data[0, idx] = d._n
                                        state_data[1, idx] = d._modeState
                                        state_data[2, idx] = d._scaleFactor
                                        state_data[3, idx] = d._f22DepScaled
                                        state_data[4, idx] = d._f22PotScaled
                                        state_data[5, idx] = d._iScale
                                        state_data[6, idx] = d._vScale
                                        state_data[7, idx] = d._currentF22
                                        state_data[8, idx] = d._currentIScale
                                        state_data[9, idx] = d._currentVScale
                                        state_data[10, idx] = d._T
                                        sanitized_cached_n[idx] = d._cached_n_int_val

                                system_cap = max(1, int((os.cpu_count() or 1) * 0.8))
                                num_workers = min(system_cap, max(1, num_devices // 512))
                                if num_workers > 1:
                                    chunk_sz = (num_devices + num_workers - 1) // num_workers
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                                        futures = []
                                        for w in range(num_workers):
                                            s = w * chunk_sz
                                            e = min(s + chunk_sz, num_devices)
                                            if s < e:
                                                futures.append(executor.submit(_pack_slice_sim, s, e))
                                        for f in futures:
                                            f.result()
                                else:
                                    _pack_slice_sim(0, num_devices)

                    state_tensor = to_device_async(torch.from_numpy(state_data), device)
                    expanded = state_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                    device_states_soa[:, 0:5, :].copy_(expanded[:, 0:5, :])
                    device_states_soa[:, 6:12, :].copy_(expanded[:, 5:11, :])
                    
                    if isinstance(sanitized_cached_n, torch.Tensor):
                        _cached_n_int_arr.copy_(sanitized_cached_n.unsqueeze(0).expand(batch_size, -1))
                    else:
                        _cached_n_int_arr.copy_(to_device_async(torch.from_numpy(np.asarray(sanitized_cached_n, dtype=np.int32)), device).unsqueeze(0).expand(batch_size, -1))
            
            if len(sim._non_pwl_v_indices) > 0:
                v_vals = np.array([gpu_vSources[i].voltage for i in sim._non_pwl_v_indices], dtype=np.float64)
                vCurrent_arr[:, sim._non_pwl_v_nodes] = to_device_async(torch.from_numpy(v_vals), device).unsqueeze(0)
            if len(sim._non_pwl_dyn_indices) > 0:
                dyn_voltages = np.array([gpu_dynSources[i].voltage for i in sim._non_pwl_dyn_indices], dtype=np.float64)
                nan_mask = np.isnan(dyn_voltages) | (dyn_voltages > 1e19)
                g_vals = np.where(nan_mask, leakage_g, 1e5).astype(np.float64)
                i_vals = np.where(nan_mask, 0.0, dyn_voltages * 1e5).astype(np.float64)
                dynamic_G_arr[:, sim._non_pwl_dyn_indices] = to_device_async(torch.from_numpy(g_vals), device).unsqueeze(0)
                dynamic_I_arr[:, sim._non_pwl_dyn_indices] = to_device_async(torch.from_numpy(i_vals), device).unsqueeze(0)
    else:
        topo_cache = getattr(sim, '_cached_torch_topology', None)
        sorted_known = tuple(sorted(sim.nodesKnown, key=lambda x: str(x)))
        sorted_unk = tuple(sorted(sim.nodesUnknown, key=lambda x: str(x)))
        if (topo_cache is not None and 
            topo_cache.get('num_devices') == num_devices and 
            topo_cache.get('num_nodes') == numNodes and 
            topo_cache.get('device') == device and
            topo_cache.get('known_nodes') == sorted_known and
            topo_cache.get('unk_nodes') == sorted_unk):
            unkNodes = topo_cache['unkNodes']
            nodeToIdx = topo_cache['nodeToIdx']
            all_nodes_ordered = topo_cache['all_nodes_ordered']
            fullNodeToIdx = topo_cache['fullNodeToIdx']
            dev_top_idx = topo_cache['dev_top_idx']
            dev_bot_idx = topo_cache['dev_bot_idx']
            dev_top_full_idx = topo_cache['dev_top_full_idx']
            dev_bot_full_idx = topo_cache['dev_bot_full_idx']
        else:
            unkNodes = list(sorted_unk)
            nodeToIdx = {n: i for i, n in enumerate(unkNodes)}
            all_nodes_ordered = unkNodes + list(sorted_known)
            fullNodeToIdx = {n: i for i, n in enumerate(all_nodes_ordered)}
            
            if num_devices == 1:
                d0 = sim.devices[0]
                top = d0['top']
                bot = d0['bottom']
                dev_top_idx = torch.tensor([nodeToIdx.get(top, -1)], dtype=torch.int32, device=device)
                dev_bot_idx = torch.tensor([nodeToIdx.get(bot, -1)], dtype=torch.int32, device=device)
                dev_top_full_idx = torch.tensor([fullNodeToIdx[top]], dtype=torch.int32, device=device)
                dev_bot_full_idx = torch.tensor([fullNodeToIdx[bot]], dtype=torch.int32, device=device)
            else:
                top_arr = getattr(sim, '_dev_top_nodes_arr', None)
                bot_arr = getattr(sim, '_dev_bot_nodes_arr', None)
                if top_arr is None or len(top_arr) != num_devices:
                    top_arr = np.array([d['top'] for d in sim.devices], dtype=np.int32)
                    bot_arr = np.array([d['bottom'] for d in sim.devices], dtype=np.int32)
                    sim._dev_top_nodes_arr = top_arr
                    sim._dev_bot_nodes_arr = bot_arr
                
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
                
                arr_top = lut_nodeToIdx[top_arr]
                arr_bot = lut_nodeToIdx[bot_arr]
                arr_top_full = lut_fullNodeToIdx[top_arr]
                arr_bot_full = lut_fullNodeToIdx[bot_arr]
                
                dev_top_idx = torch.from_numpy(arr_top).to(device, non_blocking=True)
                dev_bot_idx = torch.from_numpy(arr_bot).to(device, non_blocking=True)
                dev_top_full_idx = torch.from_numpy(arr_top_full).to(device, non_blocking=True)
                dev_bot_full_idx = torch.from_numpy(arr_bot_full).to(device, non_blocking=True)
            
            sim._cached_torch_topology = {
                'num_devices': num_devices,
                'num_nodes': numNodes,
                'device': device,
                'known_nodes': sorted_known,
                'unk_nodes': sorted_unk,
                'unkNodes': unkNodes,
                'nodeToIdx': nodeToIdx,
                'all_nodes_ordered': all_nodes_ordered,
                'fullNodeToIdx': fullNodeToIdx,
                'dev_top_idx': dev_top_idx,
                'dev_bot_idx': dev_bot_idx,
                'dev_top_full_idx': dev_top_full_idx,
                'dev_bot_full_idx': dev_bot_full_idx,
            }
        
        gMat = torch.empty(0, dtype=dtype, device=device)
        iRes = torch.zeros((batch_size, numNodes), dtype=dtype, device=device)
        vCurrent_arr = torch.zeros((batch_size, len(all_nodes_ordered)), dtype=dtype, device=device)
        
        gpu_vSources = []
        v_nodes_list = []
        pwl_mask_list = []
        longest_v_pts = 2
        if len(sim.vSources) > 0:
            for src in sim.vSources:
                if isinstance(src, DcSource):
                    gpu_src = src
                    is_pwl = False
                elif isinstance(src, PwlSource):
                    gpu_src = src
                    is_pwl = True
                elif isinstance(src, BSource):
                    gpu_src = sim._convert_bsources_to_pwl([src], tEnd)[0]
                    is_pwl = isinstance(gpu_src, PwlSource)
                elif isinstance(src, BaseSource):
                    schedule = [0.0, *src.get_all_edges(0.0, tEnd), tEnd]
                    t_points = np.unique(schedule)
                    v_points = src.getVoltage(t_points)
                    gpu_src = PwlSource(node=src.node, t_points=t_points, v_points=v_points)
                    is_pwl = True
                else:
                    gpu_src = src
                    is_pwl = isinstance(src, PwlSource)
                
                gpu_vSources.append(gpu_src)
                v_nodes_list.append(fullNodeToIdx[gpu_src.node])
                pwl_mask_list.append(is_pwl)
                if is_pwl:
                    longest_v_pts = max(longest_v_pts, len(gpu_src.t_points))
        num_v_sources = len(gpu_vSources)
        
        # 1. Discover all voltage and dynamic sources to compute unified max_pts
        gpu_dynSources = []
        dyn_nodes_list = []
        dyn_pwl_mask_list = []
        longest_dyn_pts = 2
        if len(sim.dynamicSources) > 0:
            for src in sim.dynamicSources:
                if isinstance(src, DcSource):
                    gpu_src = src
                    is_pwl = False
                elif isinstance(src, PwlSource):
                    gpu_src = src
                    is_pwl = True
                elif isinstance(src, BSource):
                    gpu_src = sim._convert_bsources_to_pwl([src], tEnd)[0]
                    is_pwl = isinstance(gpu_src, PwlSource)
                elif isinstance(src, BaseSource):
                    schedule = [0.0, *src.get_all_edges(0.0, tEnd), tEnd]
                    t_points = np.unique(schedule)
                    v_points = src.getVoltage(t_points)
                    gpu_src = PwlSource(node=src.node, t_points=t_points, v_points=v_points)
                    is_pwl = True
                else:
                    gpu_src = src
                    is_pwl = isinstance(src, PwlSource)
                    
                gpu_dynSources.append(gpu_src)
                dyn_nodes_list.append(nodeToIdx.get(gpu_src.node, -1))
                dyn_pwl_mask_list.append(is_pwl)
                if is_pwl:
                    longest_dyn_pts = max(longest_dyn_pts, len(gpu_src.t_points))
        num_dyn_sources = len(gpu_dynSources)

        max_pts = max(32, max(longest_v_pts, longest_dyn_pts))
        pad_sources = max(2, len(sim.vSources), len(sim.dynamicSources))
        
        if batched_v_pwl_t is not None:
            v_pwl_t_matrix = batched_v_pwl_t if batched_v_pwl_t.dtype == torch.float64 else batched_v_pwl_t.to(torch.float64)
            v_pwl_v_matrix = batched_v_pwl_v if batched_v_pwl_v.dtype == torch.float64 else batched_v_pwl_v.to(torch.float64)
        else:
            v_pwl_t_host = np.full((pad_sources, max_pts), 1e20, dtype=np.float64)
            v_pwl_v_host = np.zeros((pad_sources, max_pts), dtype=np.float64)
            for i, src in enumerate(gpu_vSources):
                if isinstance(src, PwlSource):
                    pts = len(src.t_points)
                    v_pwl_t_host[i, :pts] = src.t_points
                    v_pwl_v_host[i, :pts] = np.where(np.isnan(src.v_points), 1e20, src.v_points)
            v_pwl_t_matrix_2d = to_device_async(torch.from_numpy(v_pwl_t_host), device)
            v_pwl_v_matrix_2d = to_device_async(torch.from_numpy(v_pwl_v_host), device)
            v_pwl_t_matrix = v_pwl_t_matrix_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            v_pwl_v_matrix = v_pwl_v_matrix_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            
        v_nodes_padded = list(v_nodes_list)
        pwl_mask_padded = list(pwl_mask_list)
        if len(v_nodes_padded) < pad_sources:
            v_nodes_padded.extend([-1] * (pad_sources - len(v_nodes_padded)))
            pwl_mask_padded.extend([False] * (pad_sources - len(pwl_mask_padded)))
            
        v_nodes_gpu = torch.tensor(v_nodes_padded, dtype=torch.int32, device=device)
        if pwl_mask_tensor is not None:
            v_pwl_mask_gpu = pwl_mask_tensor
        else:
            v_pwl_mask_gpu = torch.tensor(pwl_mask_padded, dtype=torch.int32, device=device)
        
        sim._non_pwl_v_indices = [i for i, mask in enumerate(pwl_mask_list) if not mask]
        sim._non_pwl_v_nodes = [v_nodes_list[i] for i in sim._non_pwl_v_indices]
        if len(sim._non_pwl_v_indices) > 0:
            v_vals = np.array([gpu_vSources[i].voltage for i in sim._non_pwl_v_indices], dtype=np.float64)
            vCurrent_arr[:, sim._non_pwl_v_nodes] = to_device_async(torch.from_numpy(v_vals), device).unsqueeze(0)
                
        if batched_dyn_pwl_t is not None:
            dyn_pwl_t_matrix = batched_dyn_pwl_t
            dyn_pwl_v_matrix = batched_dyn_pwl_v
        else:
            dyn_pwl_t_host = np.full((pad_sources, max_pts), 1e20, dtype=np.float64)
            dyn_pwl_v_host = np.zeros((pad_sources, max_pts), dtype=np.float64)
            for i, src in enumerate(gpu_dynSources):
                if isinstance(src, PwlSource):
                    pts = len(src.t_points)
                    dyn_pwl_t_host[i, :pts] = src.t_points
                    dyn_pwl_v_host[i, :pts] = np.where(np.isnan(src.v_points), 1e20, src.v_points)
            dyn_pwl_t_matrix_2d = to_device_async(torch.from_numpy(dyn_pwl_t_host), device)
            dyn_pwl_v_matrix_2d = to_device_async(torch.from_numpy(dyn_pwl_v_host), device)
            dyn_pwl_t_matrix = dyn_pwl_t_matrix_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            dyn_pwl_v_matrix = dyn_pwl_v_matrix_2d.unsqueeze(0).expand(batch_size, -1, -1).clone()
            
        dyn_nodes_padded = list(dyn_nodes_list)
        dyn_pwl_mask_padded = list(dyn_pwl_mask_list)
        if len(dyn_nodes_padded) < pad_sources:
            dyn_nodes_padded.extend([-1] * (pad_sources - len(dyn_nodes_padded)))
            dyn_pwl_mask_padded.extend([False] * (pad_sources - len(dyn_pwl_mask_padded)))
            
        dyn_nodes_gpu = torch.tensor(dyn_nodes_padded, dtype=torch.int32, device=device)
        dyn_pwl_mask_gpu = torch.tensor(dyn_pwl_mask_padded, dtype=torch.int32, device=device)
            
        dynamic_G_arr = torch.zeros((batch_size, pad_sources), dtype=dtype, device=device)
        dynamic_I_arr = torch.zeros((batch_size, pad_sources), dtype=dtype, device=device)
        
        sim._non_pwl_dyn_indices = [i for i, mask in enumerate(dyn_pwl_mask_list) if not mask]
        if len(sim._non_pwl_dyn_indices) > 0:
            dyn_voltages = np.array([gpu_dynSources[i].voltage for i in sim._non_pwl_dyn_indices], dtype=np.float64)
            nan_mask = np.isnan(dyn_voltages) | (dyn_voltages > 1e19)
            g_vals = np.where(nan_mask, leakage_g, 1e5).astype(np.float64)
            i_vals = np.where(nan_mask, 0.0, dyn_voltages * 1e5).astype(np.float64)
            dynamic_G_arr[:, sim._non_pwl_dyn_indices] = to_device_async(torch.from_numpy(g_vals), device).unsqueeze(0)
            dynamic_I_arr[:, sim._non_pwl_dyn_indices] = to_device_async(torch.from_numpy(i_vals), device).unsqueeze(0)
                    
        if len(gpu_dynSources) == 0:
            dev_to_dyn_src = torch.full((num_devices,), -1, dtype=torch.int32, device=device)
        else:
            dev_to_dyn_src_host = np.full((num_devices,), -1, dtype=np.int32)
            node_to_dyn_idx = {src.node: d_idx for d_idx, src in enumerate(gpu_dynSources)}
            for i, d in enumerate(sim.devices):
                bot_node = d['bottom']
                if bot_node in node_to_dyn_idx:
                    dev_to_dyn_src_host[i] = node_to_dyn_idx[bot_node]
            dev_to_dyn_src = to_device_async(torch.from_numpy(dev_to_dyn_src_host), device)
                    
        if num_devices > 0:
            cached = getattr(sim, '_cached_phys_tensor', None)
            expected_dtype = torch.float64
            if (cached is not None and cached.shape[0] == 29 and cached.shape[1] == num_devices and 
                cached.device == device and cached.dtype == expected_dtype and 
                float(cached[0, 0]) == sim.devices[0]['dev'].vSmooth):
                phys_tensor = cached
            else:
                phys_data = np.empty((30, num_devices), dtype=np.float64)
                dev0 = sim.devices[0]['dev']
                col0 = np.array([
                    dev0.vSmooth, dev0.nSmooth, dev0.kappa, dev0.alpha, dev0.vTh,
                    dev0.vRefPos, dev0.vRefNeg, dev0.nMax, dev0.kDischarge, dev0.gScale,
                    dev0.rC, dev0.f22StartVal, dev0.f22PeakVal, dev0.f22EndVal,
                    dev0.f22PotScaleInv, dev0.f22DepScaleInv, dev0.E_a, dev0.R_th,
                    dev0.tau_th, dev0.gamma, dev0.beta_alpha,
                    dev0.beta_s, dev0.vBypass, dev0.rateAsymmetry,
                    getattr(dev0, 'rTopBlend', 0.050000), getattr(dev0, 'fDischarge', 0.374848), getattr(dev0, 'cDischarge', 0.0),
                    getattr(dev0, 'rBottomBlend', 0.714488), getattr(dev0, 'rLatency', 0.878506),
                    getattr(dev0, 'k_overdrive', 7.296635)
                ], dtype=np.float64)
                
                is_homogeneous = getattr(sim, '_all_dev_params_same', None)
                if is_homogeneous is None:
                    is_homogeneous = True
                    if num_devices > 1:
                        for d in sim.devices[1:]:
                            dev = d['dev']
                            if dev is not dev0 and dev.params is not dev0.params:
                                if (dev.vSmooth != dev0.vSmooth or dev.kappa != dev0.kappa or
                                    dev.alpha != dev0.alpha or dev.vTh != dev0.vTh or
                                    dev.gScale != dev0.gScale or dev.E_a != dev0.E_a or
                                    dev.R_th != dev0.R_th or dev.gamma != dev0.gamma or
                                    dev.rateAsymmetry != dev0.rateAsymmetry or
                                    getattr(dev, 'rTopBlend', 0.050000) != getattr(dev0, 'rTopBlend', 0.050000) or
                                    getattr(dev, 'fDischarge', 0.374848) != getattr(dev0, 'fDischarge', 0.374848) or
                                    getattr(dev, 'cDischarge', 0.0) != getattr(dev0, 'cDischarge', 0.0) or
                                    getattr(dev, 'rBottomBlend', 0.714488) != getattr(dev0, 'rBottomBlend', 0.714488) or
                                    getattr(dev, 'rLatency', 0.878506) != getattr(dev0, 'rLatency', 0.878506) or
                                    getattr(dev, 'k_overdrive', 7.296635) != getattr(dev0, 'k_overdrive', 7.296635)):
                                    is_homogeneous = False
                                    break
                    sim._all_dev_params_same = is_homogeneous
                
                if is_homogeneous:
                    phys_data[:] = col0[:, None]
                else:
                    for idx, d in enumerate(sim.devices):
                        dev = d['dev']
                        phys_data[0, idx] = dev.vSmooth
                        phys_data[1, idx] = dev.nSmooth
                        phys_data[2, idx] = dev.kappa
                        phys_data[3, idx] = dev.alpha
                        phys_data[4, idx] = dev.vTh
                        phys_data[5, idx] = dev.vRefPos
                        phys_data[6, idx] = dev.vRefNeg
                        phys_data[7, idx] = dev.nMax
                        phys_data[8, idx] = dev.kDischarge
                        phys_data[9, idx] = dev.gScale
                        phys_data[10, idx] = dev.rC
                        phys_data[11, idx] = dev.f22StartVal
                        phys_data[12, idx] = dev.f22PeakVal
                        phys_data[13, idx] = dev.f22EndVal
                        phys_data[14, idx] = dev.f22PotScaleInv
                        phys_data[15, idx] = dev.f22DepScaleInv
                        phys_data[16, idx] = dev.E_a
                        phys_data[17, idx] = dev.R_th
                        phys_data[18, idx] = dev.tau_th
                        phys_data[19, idx] = dev.gamma
                        phys_data[20, idx] = dev.beta_alpha
                        phys_data[21, idx] = dev.beta_s
                        phys_data[22, idx] = dev.vBypass
                        phys_data[23, idx] = dev.rateAsymmetry
                        phys_data[24, idx] = getattr(dev, 'rTopBlend', 0.050000)
                        phys_data[25, idx] = getattr(dev, 'fDischarge', 0.374848)
                        phys_data[26, idx] = getattr(dev, 'cDischarge', 0.0)
                        phys_data[27, idx] = getattr(dev, 'rBottomBlend', 0.714488)
                        phys_data[28, idx] = getattr(dev, 'rLatency', 0.878506)
                        phys_data[29, idx] = getattr(dev, 'k_overdrive', 7.296635)
                phys_tensor = to_device_async(torch.from_numpy(phys_data), device)
                sim._cached_phys_tensor = phys_tensor
            vSmooth_arr = phys_tensor[0]
            nSmooth_arr = phys_tensor[1]
            kappa_arr = phys_tensor[2]
            alpha_arr = phys_tensor[3]
            vTh_arr = phys_tensor[4]
            vRefPos_arr = phys_tensor[5]
            vRefNeg_arr = phys_tensor[6]
            nMax_arr = phys_tensor[7]
            kDischarge_arr = phys_tensor[8]
            gScale_arr = phys_tensor[9]
            rC_arr = phys_tensor[10]
            f22StartVal_arr = phys_tensor[11]
            f22PeakVal_arr = phys_tensor[12]
            f22EndVal_arr = phys_tensor[13]
            f22PotScaleInv_arr = phys_tensor[14]
            f22DepScaleInv_arr = phys_tensor[15]
            E_a_arr = phys_tensor[16]
            R_th_arr = phys_tensor[17]
            tau_th_arr = phys_tensor[18]
            gamma_arr = phys_tensor[19]
            beta_alpha_arr = phys_tensor[20]
            beta_s_arr = phys_tensor[21]
            vBypass_arr = phys_tensor[22]
            rateAsymmetry_arr = phys_tensor[23]
            rTopBlend_arr = phys_tensor[24]
            fDischarge_arr = phys_tensor[25]
            cDischarge_arr = phys_tensor[26]
            rBottomBlend_arr = phys_tensor[27]
            rLatency_arr = phys_tensor[28]
            k_overdrive_arr = phys_tensor[29]
            
            cached_lookups = getattr(sim, '_cached_lookup_tensors', None)
            if (cached_lookups is not None and cached_lookups[0].device == device and cached_lookups[0].dtype == dtype):
                y_i, y_v, y_fp, y_fd = cached_lookups
            else:
                dev0 = sim.devices[0]['dev']
                y_i = torch.tensor(dev0._y_i, dtype=dtype, device=device)
                y_v = torch.tensor(dev0._y_v, dtype=dtype, device=device)
                y_fp = torch.tensor(dev0._y_fp, dtype=dtype, device=device)
                y_fd = torch.tensor(dev0._y_fd, dtype=dtype, device=device)
                sim._cached_lookup_tensors = (y_i, y_v, y_fp, y_fd)
            
            if num_devices > 0:
                device_states_soa = torch.zeros((batch_size, 12, num_devices), dtype=dtype, device=device)
                device_params_soa = phys_tensor
                
                n_arr = device_states_soa[:, 0]
                modeState_arr = device_states_soa[:, 1]
                scaleFactor_arr = device_states_soa[:, 2]
                f22DepScaled_arr = device_states_soa[:, 3]
                f22PotScaled_arr = device_states_soa[:, 4]
                _cached_n_int_arr = device_states_soa[:, 5]
                iScale_arr = device_states_soa[:, 6]
                vScale_arr = device_states_soa[:, 7]
                currentF22_arr = device_states_soa[:, 8]
                currentIScale_arr = device_states_soa[:, 9]
                currentVScale_arr = device_states_soa[:, 10]
                T_arr = device_states_soa[:, 11]
            else:
                device_states_soa = torch.empty((batch_size, 12, 0), dtype=dtype, device=device)
                device_params_soa = torch.empty((30, 0), dtype=dtype, device=device)
                
                n_arr = torch.empty(0, dtype=dtype, device=device)
                modeState_arr = torch.empty(0, dtype=dtype, device=device)
                scaleFactor_arr = torch.empty(0, dtype=dtype, device=device)
                f22DepScaled_arr = torch.empty(0, dtype=dtype, device=device)
                f22PotScaled_arr = torch.empty(0, dtype=dtype, device=device)
                _cached_n_int_arr = torch.empty(0, dtype=torch.int32, device=device)
                iScale_arr = torch.empty(0, dtype=dtype, device=device)
                vScale_arr = torch.empty(0, dtype=dtype, device=device)
                currentF22_arr = torch.empty(0, dtype=dtype, device=device)
                currentIScale_arr = torch.empty(0, dtype=dtype, device=device)
                currentVScale_arr = torch.empty(0, dtype=dtype, device=device)
                T_arr = torch.empty(0, dtype=dtype, device=device)
            
            if verbose:
                c_exists = cache is not None
                g_clean = not getattr(sim, '_gpu_states_dirty', True)
                n_in_c = 'n_arr' in cache if c_exists else False
                s_match = cache['n_arr'].shape[1] == num_devices if (c_exists and n_in_c) else False
                pass
            
            if initial_state_data is not None:
                if isinstance(initial_state_data, torch.Tensor):
                    state_tensor = to_device_async(initial_state_data, device)
                else:
                    state_tensor = to_device_async(torch.from_numpy(initial_state_data), device)
                if state_tensor.ndim == 2:
                    state_tensor_exp = state_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                elif state_tensor.shape[0] in (11, 12) and state_tensor.shape[1] == batch_size:
                    state_tensor_exp = state_tensor.permute(1, 0, 2)
                elif state_tensor.shape[0] == batch_size:
                    state_tensor_exp = state_tensor
                else:
                    state_tensor_exp = state_tensor[0:1].expand(batch_size, -1, -1)
                if state_tensor_exp.shape[1] >= 12:
                    device_states_soa[:, 0:12, :].copy_(state_tensor_exp[:, 0:12, :])
                else:
                    device_states_soa[:, 0:5, :].copy_(state_tensor_exp[:, 0:5, :])
                    device_states_soa[:, 6:12, :].copy_(state_tensor_exp[:, 5:11, :])
                
                if not hasattr(sim, '_device_objs_flat') or sim._device_objs_flat is None or len(sim._device_objs_flat) != num_devices:
                    sim._device_objs_flat = [d['dev'] for d in sim.devices]
                devs = sim._device_objs_flat
                sanitized_cached_n = [int(d._cached_n_int) if (isinstance(d._cached_n_int, (int, np.integer, float, np.floating)) and np.isfinite(d._cached_n_int) and -2147483648 <= d._cached_n_int <= 2147483647) else -1 for d in devs]
                _cached_n_int_arr.copy_(to_device_async(torch.from_numpy(np.array(sanitized_cached_n, dtype=np.int32)), device).unsqueeze(0).expand(batch_size, -1))
            elif (getattr(sim, '_gpu_states_cache', None) is not None and
                  not getattr(sim, '_gpu_states_dirty', True)):
                if 'device_states_soa' in sim._gpu_states_cache:
                    src_d_soa = sim._gpu_states_cache['device_states_soa']
                    if isinstance(src_d_soa, np.ndarray):
                        src_d_soa = torch.as_tensor(src_d_soa, device=device, dtype=dtype)
                    if src_d_soa.ndim == 2:
                        src_d_soa = src_d_soa.unsqueeze(0)
                    if src_d_soa.shape[0] == 1:
                        device_states_soa.copy_(src_d_soa.expand(batch_size, -1, -1))
                    elif src_d_soa.shape[0] == batch_size:
                        device_states_soa.copy_(src_d_soa)
                    else:
                        device_states_soa.copy_(src_d_soa[0:1].expand(batch_size, -1, -1))
                    if '_cached_n_int_arr' in sim._gpu_states_cache:
                        src_c_int = sim._gpu_states_cache['_cached_n_int_arr']
                        if isinstance(src_c_int, np.ndarray):
                            src_c_int = torch.as_tensor(src_c_int, device=device, dtype=torch.int32)
                        if src_c_int.ndim == 1:
                            src_c_int = src_c_int.unsqueeze(0)
                        if src_c_int.shape[0] == 1:
                            _cached_n_int_arr.copy_(src_c_int.expand(batch_size, -1))
                        elif src_c_int.shape[0] == batch_size:
                            _cached_n_int_arr.copy_(src_c_int)
                        else:
                            _cached_n_int_arr.copy_(src_c_int[0:1].expand(batch_size, -1))
                else:
                    def inherit_from_states_cache(new_tensor, name):
                        src_tensor = sim._gpu_states_cache[name]
                        if isinstance(src_tensor, np.ndarray):
                            src_tensor = torch.as_tensor(src_tensor, device=new_tensor.device, dtype=new_tensor.dtype)
                        if src_tensor.ndim == 1:
                            src_tensor = src_tensor.unsqueeze(0)
                        if src_tensor.shape[0] == 1:
                            new_tensor.copy_(src_tensor.expand(batch_size, -1))
                        elif src_tensor.shape[0] == batch_size:
                            new_tensor.copy_(src_tensor)
                        else:
                            new_tensor.copy_(src_tensor[0:1].expand(batch_size, -1))
                    inherit_from_states_cache(n_arr, 'n_arr')
                    inherit_from_states_cache(modeState_arr, 'modeState_arr')
                    inherit_from_states_cache(scaleFactor_arr, 'scaleFactor_arr')
                    inherit_from_states_cache(f22DepScaled_arr, 'f22DepScaled_arr')
                    inherit_from_states_cache(f22PotScaled_arr, 'f22PotScaled_arr')
                    inherit_from_states_cache(iScale_arr, 'iScale_arr')
                    inherit_from_states_cache(vScale_arr, 'vScale_arr')
                    inherit_from_states_cache(currentF22_arr, 'currentF22_arr')
                    inherit_from_states_cache(currentIScale_arr, 'currentIScale_arr')
                    inherit_from_states_cache(currentVScale_arr, 'currentVScale_arr')
                    inherit_from_states_cache(T_arr, 'T_arr')
                    inherit_from_states_cache(_cached_n_int_arr, '_cached_n_int_arr')
                if verbose:
                    pass
            elif (cache is not None and not getattr(sim, '_gpu_states_dirty', True) and
                  'n_arr' in cache and cache['n_arr'].shape[1] == num_devices):
                def inherit_tensor(new_tensor, name):
                    src_tensor = cache[name]
                    if isinstance(src_tensor, np.ndarray):
                        src_tensor = torch.as_tensor(src_tensor, device=new_tensor.device, dtype=new_tensor.dtype)
                    if src_tensor.ndim == 1:
                        src_tensor = src_tensor.unsqueeze(0)
                    if src_tensor.shape[0] == 1:
                        new_tensor.copy_(src_tensor.expand(batch_size, -1))
                    elif src_tensor.shape[0] == batch_size:
                        new_tensor.copy_(src_tensor)
                    else:
                        new_tensor.copy_(src_tensor[0:1].expand(batch_size, -1))
                inherit_tensor(n_arr, 'n_arr')
                inherit_tensor(modeState_arr, 'modeState_arr')
                inherit_tensor(scaleFactor_arr, 'scaleFactor_arr')
                inherit_tensor(f22DepScaled_arr, 'f22DepScaled_arr')
                inherit_tensor(f22PotScaled_arr, 'f22PotScaled_arr')
                inherit_tensor(iScale_arr, 'iScale_arr')
                inherit_tensor(vScale_arr, 'vScale_arr')
                inherit_tensor(currentF22_arr, 'currentF22_arr')
                inherit_tensor(currentIScale_arr, 'currentIScale_arr')
                inherit_tensor(currentVScale_arr, 'currentVScale_arr')
                inherit_tensor(T_arr, 'T_arr')
                inherit_tensor(_cached_n_int_arr, '_cached_n_int_arr')
                if verbose:
                    pass
            else:
                state_data = np.empty((11, num_devices), dtype=np.float64)
                cached_n_arr = np.empty(num_devices, dtype=np.int32)
                if not hasattr(sim, '_device_objs_flat') or sim._device_objs_flat is None or len(sim._device_objs_flat) != num_devices:
                    sim._device_objs_flat = [d['dev'] for d in sim.devices]
                devs = sim._device_objs_flat
                sim.sync_to_cpu()
                states_cache = getattr(sim, '_gpu_states_cache', None)
                if states_cache is not None and not getattr(sim, '_gpu_states_dirty', False) and isinstance(states_cache, dict) and 'n_arr' in states_cache and (states_cache['n_arr'].shape[-1] == num_devices if hasattr(states_cache['n_arr'], 'shape') else len(states_cache['n_arr']) == num_devices):
                    state_data[0, :] = states_cache['n_arr']
                    state_data[1, :] = states_cache['modeState_arr']
                    state_data[2, :] = states_cache['scaleFactor_arr']
                    state_data[3, :] = states_cache['f22DepScaled_arr']
                    state_data[4, :] = states_cache['f22PotScaled_arr']
                    state_data[5, :] = states_cache['iScale_arr']
                    state_data[6, :] = states_cache['vScale_arr']
                    state_data[7, :] = states_cache['currentF22_arr']
                    state_data[8, :] = states_cache['currentIScale_arr']
                    state_data[9, :] = states_cache['currentVScale_arr']
                    state_data[10, :] = states_cache['T_arr']
                    np.copyto(cached_n_arr, states_cache['_cached_n_int_arr'])
                elif num_devices == 1:
                    d = devs[0]
                    state_data[0, 0] = d._n
                    state_data[1, 0] = d._modeState
                    state_data[2, 0] = d._scaleFactor
                    state_data[3, 0] = d._f22DepScaled
                    state_data[4, 0] = d._f22PotScaled
                    state_data[5, 0] = d._iScale
                    state_data[6, 0] = d._vScale
                    state_data[7, 0] = d._currentF22
                    state_data[8, 0] = d._currentIScale
                    state_data[9, 0] = d._currentVScale
                    state_data[10, 0] = d._T
                    cached_n_arr[0] = d._cached_n_int_val
                else:
                    d0 = devs[0]
                    d0_n = d0._n
                    d0_mode = d0._modeState
                    d0_scale = d0._scaleFactor
                    d0_fd = d0._f22DepScaled
                    d0_fp = d0._f22PotScaled
                    d0_is = d0._iScale
                    d0_vs = d0._vScale
                    d0_f22 = d0._currentF22
                    d0_cis = d0._currentIScale
                    d0_cvs = d0._currentVScale
                    d0_T = d0._T
                    col0 = np.array([
                        d0_n, d0_mode, d0_scale, d0_fd,
                        d0_fp, d0_is, d0_vs, d0_f22,
                        d0_cis, d0_cvs, d0_T
                    ], dtype=np.float64)
                    c_val0 = d0._cached_n_int_val
                    
                    is_homogeneous = True
                    dl = devs[-1]
                    if (dl._n != d0_n or dl._modeState != d0_mode or 
                        dl._scaleFactor != d0_scale or dl._T != d0_T or
                        dl._currentF22 != d0_f22 or dl._f22PotScaled != d0_fp):
                        is_homogeneous = False
                    else:
                        for d in devs[1:-1]:
                            if (d._n != d0_n or d._modeState != d0_mode or 
                                d._scaleFactor != d0_scale or d._T != d0_T or
                                d._currentF22 != d0_f22 or d._f22PotScaled != d0_fp):
                                is_homogeneous = False
                                break
                    
                    if is_homogeneous:
                        state_data[:] = col0[:, None]
                        cached_n_arr.fill(c_val0)
                    else:
                        def _pack_slice(s, e):
                            for idx in range(s, e):
                                d = devs[idx]
                                state_data[0, idx] = d._n
                                state_data[1, idx] = d._modeState
                                state_data[2, idx] = d._scaleFactor
                                state_data[3, idx] = d._f22DepScaled
                                state_data[4, idx] = d._f22PotScaled
                                state_data[5, idx] = d._iScale
                                state_data[6, idx] = d._vScale
                                state_data[7, idx] = d._currentF22
                                state_data[8, idx] = d._currentIScale
                                state_data[9, idx] = d._currentVScale
                                state_data[10, idx] = d._T
                                cached_n_arr[idx] = d._cached_n_int_val

                        system_cap = max(1, int((os.cpu_count() or 1) * 0.8))
                        num_workers = min(system_cap, max(1, num_devices // 512))
                        if num_workers > 1:
                            chunk_sz = (num_devices + num_workers - 1) // num_workers
                            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                                futures = []
                                for w in range(num_workers):
                                    s = w * chunk_sz
                                    e = min(s + chunk_sz, num_devices)
                                    if s < e:
                                        futures.append(executor.submit(_pack_slice, s, e))
                                for f in futures:
                                    f.result()
                        else:
                            _pack_slice(0, num_devices)
                        
                        sim._gpu_states_cache = {
                            'n_arr': state_data[0, :].copy(),
                            'modeState_arr': state_data[1, :].copy(),
                            'scaleFactor_arr': state_data[2, :].copy(),
                            'f22DepScaled_arr': state_data[3, :].copy(),
                            'f22PotScaled_arr': state_data[4, :].copy(),
                            'iScale_arr': state_data[5, :].copy(),
                            'vScale_arr': state_data[6, :].copy(),
                            'currentF22_arr': state_data[7, :].copy(),
                            'currentIScale_arr': state_data[8, :].copy(),
                            'currentVScale_arr': state_data[9, :].copy(),
                            'T_arr': state_data[10, :].copy(),
                            '_cached_n_int_arr': cached_n_arr.copy(),
                        }
                        sim._gpu_states_dirty = False
                        sim._gpu_states_in_sync = True

                if not np.all(np.isfinite(state_data)):
                    raise ValueError("[MOLMEM ERROR] Initial device states contain NaNs or Infs! Please check your state variables (n, T, etc.) initialization.")
                state_tensor = to_device_async(torch.from_numpy(state_data), device)
                state_tensor_exp = state_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                
                device_states_soa[:, 0:5, :].copy_(state_tensor_exp[:, 0:5, :])
                device_states_soa[:, 6:12, :].copy_(state_tensor_exp[:, 5:11, :])
                
                _cached_n_int_arr.copy_(to_device_async(torch.from_numpy(cached_n_arr), device).unsqueeze(0).expand(batch_size, -1))
                if verbose:
                    pass
            
            iDev_arr = torch.zeros((batch_size, num_devices), dtype=dtype, device=device)
            gEq_arr = torch.zeros((batch_size, num_devices), dtype=dtype, device=device)
        else:
            vSmooth_arr = torch.empty(0, dtype=dtype, device=device)
            nSmooth_arr = torch.empty(0, dtype=dtype, device=device)
            kappa_arr = torch.empty(0, dtype=dtype, device=device)
            alpha_arr = torch.empty(0, dtype=dtype, device=device)
            vTh_arr = torch.empty(0, dtype=dtype, device=device)
            vRefPos_arr = torch.empty(0, dtype=dtype, device=device)
            vRefNeg_arr = torch.empty(0, dtype=dtype, device=device)
            nMax_arr = torch.empty(0, dtype=dtype, device=device)
            kDischarge_arr = torch.empty(0, dtype=dtype, device=device)
            gScale_arr = torch.empty(0, dtype=dtype, device=device)
            rC_arr = torch.empty(0, dtype=dtype, device=device)
            f22StartVal_arr = torch.empty(0, dtype=dtype, device=device)
            f22PeakVal_arr = torch.empty(0, dtype=dtype, device=device)
            f22EndVal_arr = torch.empty(0, dtype=dtype, device=device)
            f22PotScaleInv_arr = torch.empty(0, dtype=dtype, device=device)
            f22DepScaleInv_arr = torch.empty(0, dtype=dtype, device=device)
            E_a_arr = torch.empty(0, dtype=dtype, device=device)
            R_th_arr = torch.empty(0, dtype=dtype, device=device)
            tau_th_arr = torch.empty(0, dtype=dtype, device=device)
            gamma_arr = torch.empty(0, dtype=dtype, device=device)
            beta_alpha_arr = torch.empty(0, dtype=dtype, device=device)
            beta_s_arr = torch.empty(0, dtype=dtype, device=device)
            vBypass_arr = torch.empty(0, dtype=dtype, device=device)
            rateAsymmetry_arr = torch.empty(0, dtype=dtype, device=device)
            rTopBlend_arr = torch.empty(0, dtype=dtype, device=device)
            fDischarge_arr = torch.empty(0, dtype=dtype, device=device)
            cDischarge_arr = torch.empty(0, dtype=dtype, device=device)
            rBottomBlend_arr = torch.empty(0, dtype=dtype, device=device)
            rLatency_arr = torch.empty(0, dtype=dtype, device=device)
            k_overdrive_arr = torch.empty(0, dtype=dtype, device=device)
            
            y_i = torch.empty(0, dtype=dtype, device=device)
            y_v = torch.empty(0, dtype=dtype, device=device)
            y_fp = torch.empty(0, dtype=dtype, device=device)
            y_fd = torch.empty(0, dtype=dtype, device=device)
            
            device_states_soa = torch.empty((batch_size, 12, 0), dtype=dtype, device=device)
            device_params_soa = torch.empty((31, 0), dtype=dtype, device=device)
            
            n_arr = torch.empty(0, dtype=dtype, device=device)
            modeState_arr = torch.empty(0, dtype=dtype, device=device)
            scaleFactor_arr = torch.empty(0, dtype=dtype, device=device)
            f22DepScaled_arr = torch.empty(0, dtype=dtype, device=device)
            f22PotScaled_arr = torch.empty(0, dtype=dtype, device=device)
            _cached_n_int_arr = torch.empty(0, dtype=torch.int32, device=device)
            iScale_arr = torch.empty(0, dtype=dtype, device=device)
            vScale_arr = torch.empty(0, dtype=dtype, device=device)
            currentF22_arr = torch.empty(0, dtype=dtype, device=device)
            currentIScale_arr = torch.empty(0, dtype=dtype, device=device)
            currentVScale_arr = torch.empty(0, dtype=dtype, device=device)
            T_arr = torch.empty(0, dtype=dtype, device=device)
            iDev_arr = torch.empty(0, dtype=dtype, device=device)
            gEq_arr = torch.empty(0, dtype=dtype, device=device)
        
        dev_top_idx_cpu = dev_top_idx.to('cpu')
        dev_bot_idx_cpu = dev_bot_idx.to('cpu')
        dyn_nodes_cpu = dyn_nodes_gpu.to('cpu')

        top_mask = (dev_top_idx_cpu >= 0).numpy()
        bot_mask = (dev_bot_idx_cpu >= 0).numpy()
        both_mask = top_mask & bot_mask
        dyn_mask = (dyn_nodes_cpu >= 0).numpy()
        
        top_indices_np = np.where(top_mask)[0].astype(np.int32)
        bot_indices_np = np.where(bot_mask)[0].astype(np.int32)
        both_indices_np = np.where(both_mask)[0].astype(np.int32)
        dyn_indices_np = np.where(dyn_mask)[0].astype(np.int32)
        
        dev_both_idx_np = np.full(num_devices, -1, dtype=np.int32)
        dev_both_idx_np[both_indices_np] = np.arange(len(both_indices_np), dtype=np.int32)
        
        top_cpu_np = dev_top_idx_cpu.numpy().astype(np.int32)
        bot_cpu_np = dev_bot_idx_cpu.numpy().astype(np.int32)
        dyn_cpu_np = dyn_nodes_cpu.numpy().astype(np.int32)
        
        t_idx_np = top_cpu_np[top_mask]
        b_idx_np = bot_cpu_np[bot_mask]
        t_idx_both_np = top_cpu_np[both_mask]
        b_idx_both_np = bot_cpu_np[both_mask]
        d_nodes_np = dyn_cpu_np[dyn_mask]
        
        gMat_row_np = np.concatenate([t_idx_np, b_idx_np, t_idx_both_np, b_idx_both_np, d_nodes_np])
        gMat_col_np = np.concatenate([t_idx_np, b_idx_np, b_idx_both_np, t_idx_both_np, d_nodes_np])
        iRes_np = np.concatenate([t_idx_np, b_idx_np, d_nodes_np])
        
        top_indices = torch.from_numpy(top_indices_np).to(device)
        bot_indices = torch.from_numpy(bot_indices_np).to(device)
        both_indices = torch.from_numpy(both_indices_np).to(device)
        dyn_indices = torch.from_numpy(dyn_indices_np).to(device)
        dev_both_idx = torch.from_numpy(dev_both_idx_np).to(device)
        d_nodes = torch.from_numpy(d_nodes_np).to(device)
        
        gMat_row_indices = torch.from_numpy(gMat_row_np).to(device)
        gMat_col_indices = torch.from_numpy(gMat_col_np).to(device)
        iRes_indices = torch.from_numpy(iRes_np).to(device)
        
        v_pwl_mask_gpu_int = v_pwl_mask_gpu
        dyn_pwl_mask_gpu_int = dyn_pwl_mask_gpu
        
        new_cache = {
            'device': device,
            'num_devices': num_devices,
            'num_v_sources': num_v_sources,
            'num_dyn_sources': num_dyn_sources,
            'batch_size': batch_size,
            
            'dev_top_idx': dev_top_idx,
            'dev_bot_idx': dev_bot_idx,
            'dev_top_full_idx': dev_top_full_idx,
            'dev_bot_full_idx': dev_bot_full_idx,
            'dev_both_idx': dev_both_idx,
            
            'gMat': gMat,
            'iRes': iRes,
            'vCurrent_arr': vCurrent_arr,
            
            'gpu_vSources': gpu_vSources,
            'v_nodes_list': v_nodes_list,
            'pwl_mask_list': pwl_mask_list,
            'v_pwl_t_matrix': v_pwl_t_matrix,
            'v_pwl_v_matrix': v_pwl_v_matrix,
            'v_nodes_gpu': v_nodes_gpu,
            'v_pwl_mask_gpu': v_pwl_mask_gpu,
            'v_pwl_mask_gpu_int': v_pwl_mask_gpu_int,
            
            'gpu_dynSources': gpu_dynSources,
            'dyn_nodes_list': dyn_nodes_list,
            'dyn_pwl_mask_list': dyn_pwl_mask_list,
            'dyn_pwl_t_matrix': dyn_pwl_t_matrix,
            'dyn_pwl_v_matrix': dyn_pwl_v_matrix,
            'dyn_nodes_gpu': dyn_nodes_gpu,
            'dyn_pwl_mask_gpu': dyn_pwl_mask_gpu,
            'dyn_pwl_mask_gpu_int': dyn_pwl_mask_gpu_int,
            
            'dynamic_G_arr': dynamic_G_arr,
            'dynamic_I_arr': dynamic_I_arr,
            'dev_to_dyn_src': dev_to_dyn_src,
            
            'top_mask': top_mask,
            'bot_mask': bot_mask,
            'both_mask': both_mask,
            'dyn_mask': dyn_mask,
            'top_indices': top_indices,
            'bot_indices': bot_indices,
            'both_indices': both_indices,
            'dyn_indices': dyn_indices,
            'd_nodes': d_nodes,
            'gMat_row_indices': gMat_row_indices,
            'gMat_col_indices': gMat_col_indices,
            'iRes_indices': iRes_indices,
            'all_nodes_ordered': all_nodes_ordered,
        }
        if num_devices > 0:
            new_cache.update({
                'device_states_soa': device_states_soa,
                'device_params_soa': device_params_soa,
                'vSmooth_arr': vSmooth_arr,
                'nSmooth_arr': nSmooth_arr,
                'kappa_arr': kappa_arr,
                'alpha_arr': alpha_arr,
                'vTh_arr': vTh_arr,
                'vRefPos_arr': vRefPos_arr,
                'vRefNeg_arr': vRefNeg_arr,
                'nMax_arr': nMax_arr,
                'kDischarge_arr': kDischarge_arr,
                'gScale_arr': gScale_arr,
                'rC_arr': rC_arr,
                'f22StartVal_arr': f22StartVal_arr,
                'f22PeakVal_arr': f22PeakVal_arr,
                'f22EndVal_arr': f22EndVal_arr,
                'f22PotScaleInv_arr': f22PotScaleInv_arr,
                'f22DepScaleInv_arr': f22DepScaleInv_arr,
                'E_a_arr': E_a_arr,
                'R_th_arr': R_th_arr,
                'tau_th_arr': tau_th_arr,
                'gamma_arr': gamma_arr,
                'beta_alpha_arr': beta_alpha_arr,
                'beta_s_arr': beta_s_arr,
                'vBypass_arr': vBypass_arr,
                
                'y_i': y_i,
                'y_v': y_v,
                'y_fp': y_fp,
                'y_fd': y_fd,
                
                'n_arr': n_arr,
                'modeState_arr': modeState_arr,
                'scaleFactor_arr': scaleFactor_arr,
                'f22DepScaled_arr': f22DepScaled_arr,
                'f22PotScaled_arr': f22PotScaled_arr,
                '_cached_n_int_arr': _cached_n_int_arr,
                'iScale_arr': iScale_arr,
                'vScale_arr': vScale_arr,
                'currentF22_arr': currentF22_arr,
                'currentIScale_arr': currentIScale_arr,
                'currentVScale_arr': currentVScale_arr,
                'T_arr': T_arr,
                
                'iDev_arr': iDev_arr,
                'gEq_arr': gEq_arr,
            })
            
        if cache is None:
            raw_cache = {
                'device': device,
                'num_devices': num_devices,
                'num_v_sources': num_v_sources,
                'num_dyn_sources': num_dyn_sources,
                'batch_size': batch_size
            }
            cache = raw_cache if not stream_key else StreamProxyDict(raw_cache, stream_key)
        cache.update(new_cache)
        sim._gpu_cache = raw_cache if not stream_key else getattr(cache, 'target_dict', cache)

    t_start = 0.0
    compile_backend = getattr(sim, 'compile_backend', False)
    if appendHistory and sim.tArr is not None and len(sim.tArr) > 0:
        t_start = sim.tArr[-1]
    raw_cache_sched = getattr(sim, '_gpu_cache', None)
    cache = (raw_cache_sched if not stream_key else StreamProxyDict(raw_cache_sched, stream_key)) if raw_cache_sched is not None else None
    schedule_cached = False
    all_srcs = (gpu_vSources + gpu_dynSources) if (len(gpu_vSources) > 0 or len(gpu_dynSources) > 0) else (sim.vSources + sim.dynamicSources)
    src_signatures = tuple(
        src.get_signature() if hasattr(src, 'get_signature') else (src.__class__.__name__, id(src))
        for src in all_srcs
    )
    if (cache is not None and 'master_schedule_gpu' in cache and 
        cache.get('last_tEnd') == float(tEnd) and
        cache.get('last_src_signatures') == src_signatures and
        batched_v_pwl_t is None and batched_dyn_pwl_t is None):
        master_schedule_gpu = cache['master_schedule_gpu']
        master_schedule = cache['master_schedule_cpu']
        schedule_cached = True
        
    if not schedule_cached:
        cache_key = (float(tEnd), len(all_srcs), src_signatures)
        from .simulator import _GLOBAL_SCHEDULE_CACHE
        cached_info = _GLOBAL_SCHEDULE_CACHE.get(cache_key)
        if cached_info is not None and batched_v_pwl_t is None and batched_dyn_pwl_t is None:
            unique_edges_arr, master_schedule, is_subsample = cached_info
        else:
            edge_arrays = [np.array([0.0, tEnd], dtype=np.float64)]
            for src in all_srcs:
                if hasattr(src, 't_points'):
                    edge_arrays.append(src.t_points)
                elif hasattr(src, 'get_all_edges'):
                    ed = src.get_all_edges(0.0, tEnd)
                    if isinstance(ed, np.ndarray) and len(ed) > 0:
                        edge_arrays.append(ed)
                    elif isinstance(ed, (list, tuple)) and len(ed) > 0:
                        edge_arrays.append(np.array(ed, dtype=np.float64))
                        
            if batched_v_pwl_t is not None:
                v_edges_cpu = batched_v_pwl_t.to('cpu')
                v_edges = v_edges_cpu[v_edges_cpu < 1e19].numpy()
                if len(v_edges) > 0:
                    edge_arrays.append(v_edges)
            if batched_dyn_pwl_t is not None:
                dyn_edges_cpu = batched_dyn_pwl_t.to('cpu')
                dyn_edges = dyn_edges_cpu[dyn_edges_cpu < 1e19].numpy()
                if len(dyn_edges) > 0:
                    edge_arrays.append(dyn_edges)
                    
            raw_edges = edge_arrays[0] if len(edge_arrays) == 1 else np.concatenate(edge_arrays)
            unique_edges_arr = np.unique(raw_edges)
            unique_edges_arr = unique_edges_arr[(unique_edges_arr >= 0.0) & (unique_edges_arr <= tEnd)]
            
            sub_dt = 10e-9
            inv_sub_dt = 1.0 / sub_dt
            sub_dt_threshold = sub_dt * 1.5
            if len(unique_edges_arr) > 1:
                t_starts = unique_edges_arr[:-1]
                t_ends = unique_edges_arr[1:]
                dt_segs = t_ends - t_starts
                t_centers = t_starts + 0.5 * dt_segs
                
                is_active_mask = np.zeros(len(t_centers), dtype=bool)
                if batched_v_pwl_v is not None or batched_v_pwl_t is not None:
                    is_active_mask.fill(True)
                elif len(all_srcs) > 0:
                    for src in all_srcs:
                        if hasattr(src, 'getVoltage'):
                            try:
                                v_res = src.getVoltage(t_centers)
                                if isinstance(v_res, np.ndarray):
                                    is_active_mask |= (np.abs(v_res) > 1e-6)
                                else:
                                    is_active_mask |= (abs(float(v_res)) > 1e-6)
                            except Exception:
                                is_active_mask.fill(True)
                                break
                
                from .solver import jit_build_full_master_schedule
                master_schedule, is_subsample = jit_build_full_master_schedule(
                    unique_edges_arr, is_active_mask, sub_dt_threshold, inv_sub_dt
                )
            else:
                master_schedule = unique_edges_arr.copy()
                is_subsample = np.zeros(len(master_schedule), dtype=bool)

            if batched_v_pwl_t is None and batched_dyn_pwl_t is None:
                _GLOBAL_SCHEDULE_CACHE[cache_key] = (unique_edges_arr, master_schedule, is_subsample)

        master_schedule_gpu = torch.tensor(master_schedule, dtype=dtype, device=device)
        is_subsample_gpu = torch.tensor(is_subsample.astype(np.int32), dtype=torch.int32, device=device)
        if cache is not None:
            cache['master_schedule_gpu'] = master_schedule_gpu
            cache['is_subsample_gpu'] = is_subsample_gpu
            cache['master_schedule_cpu'] = master_schedule
            cache['is_subsample_cpu'] = is_subsample
            cache['last_tEnd'] = float(tEnd)
            cache['last_src_signatures'] = src_signatures
    else:
        is_subsample_gpu = cache['is_subsample_gpu']

    if recordHistory:
        from .sys_utils import get_cpu_cache_budget_mb
        staging_mb, ping_pong_mb, _ = get_cpu_cache_budget_mb()
        expected_steps = max(20_000, min(1_000_000, len(master_schedule) * 50 + 5000))
        target_bytes = staging_mb * 1024 * 1024
        num_channels = 1 + (len(all_nodes_ordered) if v_flag else 0) + (num_devices * (int(i_flag) + int(state_flag) + int(f22_flag) + int(g_flag) + int(temp_flag)))
        bytes_per_step = max(8, num_channels * 8 * batch_size)
        target_steps = int(target_bytes / bytes_per_step)
        if not sync_readback and batch_size > 1:
            # Single-shot batched solve in GPU VRAM: ensure capacity fits the entire master schedule
            hist_capacity = max(10, min(expected_steps, max(len(master_schedule) + 500, target_steps)))
        else:
            hist_capacity = max(10, min(target_steps, expected_steps))
    else:
        hist_capacity = 2
    
    if HAS_CPP_EXTENSION and compile_backend:
        if (cache is not None and 'tArr_buffer' in cache and 
            cache['tArr_buffer'].size(0) == batch_size and cache['tArr_buffer'].size(1) == hist_capacity and 
            cache['tArr_buffer'].dtype == dtype):
            tArr_buffer = cache['tArr_buffer'].zero_()
            vHist_buffer = cache['vHist_buffer'].zero_()
            iHist_buffer = cache['iHist_buffer'].zero_()
            stateHist_buffer = cache['stateHist_buffer'].zero_()
            f22Hist_buffer = cache['f22Hist_buffer'].zero_()
            gHist_buffer = cache['gHist_buffer'].zero_()
            tempHist_buffer = cache['tempHist_buffer'].zero_()
        else:
            if 'all_nodes_ordered' not in locals() or all_nodes_ordered is None:
                all_nodes_ordered = cache.get('all_nodes_ordered', None) if cache is not None else None
                if all_nodes_ordered is None:
                    topo = getattr(sim, '_cached_torch_topology', None)
                    if topo is not None and 'all_nodes_ordered' in topo:
                        all_nodes_ordered = topo['all_nodes_ordered']
                    else:
                        all_nodes_ordered = list(sim.nodesUnknown) + list(sim.nodesKnown)
            num_all_nodes = len(all_nodes_ordered)
            tArr_buffer = torch.zeros((batch_size, hist_capacity), dtype=dtype, device=device)
            vHist_buffer = torch.zeros((batch_size, num_all_nodes, hist_capacity), dtype=dtype, device=device)
            if num_devices > 0:
                iHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
                stateHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
                f22Hist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
                gHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
                tempHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
            else:
                iHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
                stateHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
                f22Hist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
                gHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
                tempHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
            if cache is not None:
                cache['tArr_buffer'] = tArr_buffer
                cache['vHist_buffer'] = vHist_buffer
                cache['iHist_buffer'] = iHist_buffer
                cache['stateHist_buffer'] = stateHist_buffer
                cache['f22Hist_buffer'] = f22Hist_buffer
                cache['gHist_buffer'] = gHist_buffer
                cache['tempHist_buffer'] = tempHist_buffer
            try:
                from .sys_utils import log_to_run_log_only
                hist_mb = (hist_capacity * batch_size * 8 * (1 + (len(all_nodes_ordered) if v_flag else 0) + (num_devices if i_flag else 0) + (num_devices if state_flag else 0) + (num_devices if f22_flag else 0) + (num_devices if g_flag else 0) + (num_devices if temp_flag else 0))) / (1024 * 1024)
                log_to_run_log_only(f"[GPU_ALLOC] History Buffers Allocated | HistCapacity={hist_capacity} | Batch={batch_size} | VRAM_MB={hist_mb:.1f}")
            except Exception:
                pass
    else:
        tArr_buffer = torch.zeros((batch_size, hist_capacity), dtype=dtype, device=device)
        vHist_buffer = torch.zeros((batch_size, len(all_nodes_ordered), hist_capacity), dtype=dtype, device=device)
        if num_devices > 0:
            iHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
            stateHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
            f22Hist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
            gHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
            tempHist_buffer = torch.zeros((batch_size, num_devices, hist_capacity), dtype=dtype, device=device)
        else:
            iHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
            stateHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
            f22Hist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
            gHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
            tempHist_buffer = torch.empty((batch_size, 0, 0), dtype=dtype, device=device)
        
    cached_sched = getattr(sim, '_cached_master_schedule_gpu', None)
    cached_sched_list = getattr(sim, '_cached_master_schedule_list', None)
    sched_equal = False
    if cached_sched is not None and cached_sched_list is not None and cached_sched.device == device and cached_sched.dtype == dtype:
        if isinstance(master_schedule, np.ndarray) and isinstance(cached_sched_list, np.ndarray):
            sched_equal = (cached_sched_list.shape == master_schedule.shape and np.array_equal(cached_sched_list, master_schedule))
        elif hasattr(cached_sched_list, '__len__') and hasattr(master_schedule, '__len__') and len(cached_sched_list) == len(master_schedule):
            sched_equal = (cached_sched_list == master_schedule)
    if sched_equal:
        master_schedule_gpu = cached_sched
    else:
        master_schedule_gpu = torch.tensor(master_schedule, dtype=dtype, device=device)
        sim._cached_master_schedule_gpu = master_schedule_gpu
        sim._cached_master_schedule_list = master_schedule
    
    # is_profiling and verbose resolved at entry
    is_helper = not getattr(sim, 'force_device_init', True)
    is_compiling = os.environ.get('MOLMEM_GPU_JIT_WARMED_UP') != '1'
    if HAS_CPP_EXTENSION and compile_backend:
        is_compiling = False
        
    # JIT compilation message is printed inline dynamically inside compiling blocks.
        
    tStartBench = time.time()

    import contextlib
    if device.type == 'cuda' and torch.cuda.is_available():
        cm = contextlib.ExitStack()
        cm.enter_context(torch.cuda.device(device))
        if stream is not None and getattr(stream, 'cuda_stream', 0) != 0:
            cm.enter_context(torch.cuda.stream(stream))
    else:
        cm = contextlib.nullcontext()
    
    loop_args = (
        t_start, tEnd, min_dt, tol, maxIter, hist_capacity,
        master_schedule_gpu,
        device_states_soa, device_params_soa,
        y_i, y_v, y_fp, y_fd,
        numNodes, num_devices, dev_top_full_idx, dev_bot_full_idx,
        dev_top_idx, dev_bot_idx,
        gMat_row_indices, gMat_col_indices, iRes_indices,
        top_indices, bot_indices, both_indices,
        d_nodes, dyn_indices, dev_to_dyn_src,
        v_nodes_gpu, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask_gpu,
        dyn_nodes_gpu, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask_gpu,
        is_1t1r, record_hist_bool, leakage_g,
        gMat, iRes, vCurrent_arr,
        dynamic_G_arr, dynamic_I_arr,
        iDev_arr, gEq_arr,
        tArr_buffer, vHist_buffer, iHist_buffer,
        stateHist_buffer, f22Hist_buffer, gHist_buffer,
        tempHist_buffer,
        num_v_sources,
        num_dyn_sources,
        static_vCurrent,
        static_currentIScale,
        static_currentVScale,
        static_currentF22,
        static_dynamic_G,
        static_dynamic_I,
        static_gMat,
        static_iRes,
        static_iDev,
        static_gEq,
        static_deltaV,
        (verbose or is_compiling) and not is_helper and not getattr(sim, '_in_warmup', False) and not getattr(sim.__class__, '_in_warmup', False),
        os.environ.get("MOLMEM_DEBUG_GPU") == "1"
    )
    
    t_start_loop = time.time()
    with cm:
        if HAS_CPP_EXTENSION and compile_backend:
            raw_cache_solv = getattr(sim, '_gpu_cache', None)
            cache = (raw_cache_solv if not stream_key else StreamProxyDict(raw_cache_solv, stream_key)) if raw_cache_solv is not None else None
            
            # 1. Hot-cache and reuse device_states_soa_coalesced
            if cache is not None and 'device_states_soa_coalesced' in cache and cache['device_states_soa_coalesced'].shape == (12, num_devices, batch_size):
                device_states_soa_coalesced = cache['device_states_soa_coalesced']
                device_states_soa_coalesced.copy_(device_states_soa.permute(1, 2, 0))
            else:
                if cache is not None:
                    cache.pop('device_states_soa_coalesced', None)
                device_states_soa_coalesced = device_states_soa.permute(1, 2, 0).contiguous()
                if cache is not None:
                    cache['device_states_soa_coalesced'] = device_states_soa_coalesced
            
            # 2. Hot-cache and reuse device_params_soa_transposed (kept original name to minimize changes)
            if cache is not None and 'device_params_soa_transposed' in cache and cache['device_params_soa_transposed'].shape == (30, num_devices):
                device_params_soa_transposed = cache['device_params_soa_transposed']
            else:
                if cache is not None:
                    cache.pop('device_params_soa_transposed', None)
                device_params_soa_transposed = device_params_soa.contiguous()
                if cache is not None:
                    cache['device_params_soa_transposed'] = device_params_soa_transposed
            
            # 3. Hot-cache and reuse workspaces
            batch_size = device_states_soa_coalesced.size(2)
            use_global_workspace = True
            if use_global_workspace:
                num_gMat_vals = gMat_col_indices.size(0)
                cg_dtype = dtype
                pad_sources = max(num_v_sources, num_dyn_sources)
                if pad_sources < 2:
                    pad_sources = 2
                cg_size = 6 * numNodes * batch_size
                t_elements = num_gMat_vals + 6 * numNodes + 8 * (num_v_sources + pad_sources) + 24 * num_devices
                t_size = t_elements * batch_size
                
                if (cache is not None and 'workspace_cg' in cache and 
                    cache['workspace_cg'].numel() >= cg_size and cache['workspace_cg'].dtype == cg_dtype and 
                    'workspace_t' in cache and cache['workspace_t'].numel() >= t_size and cache['workspace_t'].dtype == dtype):
                    workspace_cg = cache['workspace_cg'][:cg_size].zero_()
                    workspace_t = cache['workspace_t'][:t_size].zero_()
                else:
                    if cache is not None:
                        cache.pop('workspace_cg', None)
                        cache.pop('workspace_t', None)
                    workspace_cg = torch.zeros(cg_size, dtype=cg_dtype, device=device)
                    workspace_t = torch.zeros(t_size, dtype=dtype, device=device)
                    if cache is not None:
                        cache['workspace_cg'] = workspace_cg
                        cache['workspace_t'] = workspace_t
            else:
                workspace_cg = torch.empty(0, dtype=dtype, device=device)
                workspace_t = torch.empty(0, dtype=dtype, device=device)

            # 4. Hot-cache and reuse dev_both_idx
            if cache is not None and 'dev_both_idx' in cache and cache['dev_both_idx'].size(0) == num_devices:
                dev_both_idx = cache['dev_both_idx']
            else:
                dev_both_idx = torch.full((num_devices,), -1, dtype=torch.int32, device=device)
                dev_both_idx[both_indices] = torch.arange(both_indices.size(0), dtype=torch.int32, device=device)
                if cache is not None:
                    cache['dev_both_idx'] = dev_both_idx
            
            # 5. Hot-cache and reuse out_tensor (with sufficient slots for granular cycle telemetry and per-batch hist_idx)
            needed_out_slots = max(128, 32 + batch_size)
            if cache is not None and 'out_tensor' in cache and cache['out_tensor'].numel() >= needed_out_slots:
                out_tensor = cache['out_tensor']
            else:
                out_tensor = torch.zeros(needed_out_slots, dtype=dtype, device=device)
                if cache is not None:
                    cache['out_tensor'] = out_tensor
            out_tensor.zero_()
            
            # 6. Retrieve pre-cast inputs from cache
            v_pwl_mask_gpu_int = v_pwl_mask_gpu
            dyn_pwl_mask_gpu_int = dyn_pwl_mask_gpu
            
            master_schedule_gpu_typed = master_schedule_gpu.to(dtype)
            v_pwl_t_matrix_typed = v_pwl_t_matrix.to(dtype)
            dyn_pwl_t_matrix_typed = dyn_pwl_t_matrix.to(dtype)
            
            if torch.cuda.is_available() and device.type == 'cuda':
                if stream is not None and getattr(stream, 'device', None) == device:
                    stream_ptr = stream.cuda_stream
                else:
                    stream_ptr = torch.cuda.current_stream(device).cuda_stream
            else:
                stream_ptr = 0


            underflow_warned_py = False

            # Execute through the compiled high-performance templated C++/HIP extension solver (Zero Launch Latency)
            t_chunk_start = t_start
            # Dynamic initial probe slice (~100 timesteps / max 10% of total span) to measure empirical speed without cold-start lag
            t_chunk_step = max(min_dt * 100.0, min(1e-5, (tEnd - t_start) * 0.1))
            hist_idx = 0
            t_final = t_chunk_start
            first_chunk = True
            
            # Pre-initialize host disk history and fixed pinned transfer buffer
            if sync_readback and recordHistory:
                has_history = appendHistory and getattr(sim, '_hist_active_len', 0) > 0
                if not has_history:
                    sim._hist_active_len = 0
                    store = sim._ensure_disk_history_store(
                        num_nodes=len(all_nodes_ordered),
                        num_devices=num_devices,
                        initial_capacity=max(20000, hist_capacity),
                        recordHistory=recordHistory,
                        recordStepTiming=False,
                        batch_size=batch_size
                    )
                else:
                    store = sim._ensure_disk_history_store(
                        num_nodes=len(all_nodes_ordered),
                        num_devices=num_devices,
                        initial_capacity=getattr(sim, '_hist_active_len', 0) + hist_capacity,
                        recordHistory=recordHistory,
                        recordStepTiming=False,
                        batch_size=batch_size
                    )
                
                # Allocate fixed-size pinned transfer buffer (scaled to ping_pong_mb)
                from .sys_utils import get_cpu_cache_budget_mb
                _, ping_pong_mb, _ = get_cpu_cache_budget_mb()
                ping_pong_bytes = ping_pong_mb * 1024 * 1024
                bytes_per_step = max(8, (1 + (len(all_nodes_ordered) if v_flag else 0) + (num_devices * 5 if num_devices > 0 else 0)) * 8 * batch_size)
                transfer_capacity = max(hist_capacity, min(hist_capacity * 2, int(ping_pong_bytes / bytes_per_step)))
                if torch.cuda.is_available() and device.type == 'cuda':
                    try:
                        if batch_size > 1:
                            sim._pinned_tArr = torch.empty((batch_size, transfer_capacity), dtype=dtype, pin_memory=True)
                            if v_flag:
                                sim._pinned_vHist = torch.empty((batch_size, len(all_nodes_ordered), transfer_capacity), dtype=dtype, pin_memory=True)
                            if num_devices > 0:
                                if i_flag:
                                    sim._pinned_iHist = torch.empty((batch_size, num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if state_flag:
                                    sim._pinned_stateHist = torch.empty((batch_size, num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if f22_flag:
                                    sim._pinned_f22Hist = torch.empty((batch_size, num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if g_flag:
                                    sim._pinned_gHist = torch.empty((batch_size, num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if temp_flag:
                                    sim._pinned_tempHist = torch.empty((batch_size, num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                        else:
                            sim._pinned_tArr = torch.empty(transfer_capacity, dtype=dtype, pin_memory=True)
                            if v_flag:
                                sim._pinned_vHist = torch.empty((len(all_nodes_ordered), transfer_capacity), dtype=dtype, pin_memory=True)
                            if num_devices > 0:
                                if i_flag:
                                    sim._pinned_iHist = torch.empty((num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if state_flag:
                                    sim._pinned_stateHist = torch.empty((num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if f22_flag:
                                    sim._pinned_f22Hist = torch.empty((num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if g_flag:
                                    sim._pinned_gHist = torch.empty((num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                                if temp_flag:
                                    sim._pinned_tempHist = torch.empty((num_devices, transfer_capacity), dtype=dtype, pin_memory=True)
                    except Exception:
                        pass

            # Retrieve dedicated background PCIe DMA transfer stream and hardware event fences from global pool
            bundle = get_gpu_stream_bundle(device, stream)
            copy_stream = cache.get('copy_stream', bundle['copy_stream']) if cache is not None else bundle['copy_stream']
            chunk_event = cache.get('chunk_event', bundle['chunk_event']) if cache is not None else bundle['chunk_event']
            copy_event_slot_a = cache.get('copy_event_slot_a', bundle['copy_event_slot_a']) if cache is not None else bundle['copy_event_slot_a']
            copy_event_slot_b = cache.get('copy_event_slot_b', bundle['copy_event_slot_b']) if cache is not None else bundle['copy_event_slot_b']

            cuda_dev_context = torch.cuda.device(device) if (torch.cuda.is_available() and device.type == 'cuda') else contextlib.nullcontext()

            if v_pwl_v_matrix.dtype != torch.float64:
                v_pwl_v_matrix = v_pwl_v_matrix.to(torch.float64)
            if dyn_pwl_v_matrix.dtype != torch.float64:
                dyn_pwl_v_matrix = dyn_pwl_v_matrix.to(torch.float64)

            if not sync_readback:
                try:
                    from .sys_utils import log_to_run_log_only
                    pulses_str = ""
                    if len(sim.vSources) > 0:
                        from .sources import PulseSource
                        p_counts = [int(np.ceil((tEnd - t_start) / s.period)) for s in sim.vSources if isinstance(s, PulseSource) and s.period > 0]
                        if p_counts:
                            pulses_str = f" | Pulses={max(p_counts)}"
                    log_to_run_log_only(f"[GPU_SOLVE_DISPATCH] Non-streaming solve | tSpan=[{t_start:.6e}, {tEnd:.6e}] | Nodes={numNodes} | Devices={num_devices} | Batch={batch_size}{pulses_str}")
                except Exception:
                    pass
                with cuda_dev_context:
                    try:
                        with _GPU_KERNEL_LAUNCH_LOCK:
                            molmem_cuda_solver.run_gpu_simulation_double(
                                t_start, tEnd, min_dt, tol, maxIter, hist_capacity,
                                master_schedule_gpu_typed,
                                is_subsample_gpu,
                                device_states_soa_coalesced,
                                device_params_soa_transposed,
                                numNodes, num_devices,
                                dev_top_full_idx, dev_bot_full_idx,
                                dev_top_idx, dev_bot_idx,
                                gMat_row_indices, gMat_col_indices, iRes_indices,
                                top_indices, bot_indices, both_indices, dev_both_idx,
                                d_nodes, dyn_indices, dev_to_dyn_src,
                                v_nodes_gpu, v_pwl_t_matrix_typed, v_pwl_v_matrix, v_pwl_mask_gpu_int,
                                dyn_nodes_gpu, dyn_pwl_t_matrix_typed, dyn_pwl_v_matrix, dyn_pwl_mask_gpu_int,
                                is_1t1r, record_hist_bool, leakage_g,
                                num_v_sources, num_dyn_sources,
                                tArr_buffer, vHist_buffer, iHist_buffer,
                                stateHist_buffer, f22Hist_buffer, gHist_buffer,
                                tempHist_buffer,
                                gMat_col_indices.size(0),
                                v_pwl_v_matrix.size(2),
                                batch_size,
                                y_i,
                                y_v,
                                y_fp,
                                y_fd,
                                y_i.size(0),
                                workspace_cg,
                                workspace_t,
                                out_tensor,
                                stream_ptr
                            )
                    except (torch.OutOfMemoryError, RuntimeError) as oom_e:
                        if "out of memory" in str(oom_e).lower() or "hip out of memory" in str(oom_e).lower():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        raise oom_e
                t_final = tEnd
                if sync_stream:
                    if stream is not None and getattr(stream, 'cuda_stream', 0) != 0:
                        stream.synchronize()
                    elif torch.cuda.is_available() and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    out_cpu = out_tensor.detach().cpu().numpy()
                    if batch_size > 1 and len(out_cpu) >= 32 + batch_size:
                        hist_idx = int(np.max(out_cpu[32 : 32 + batch_size]))
                    else:
                        hist_idx = int(out_cpu[0])
                    if HAS_CPP_EXTENSION and compile_backend and num_devices > 0:
                        device_states_soa.copy_(device_states_soa_coalesced.permute(2, 0, 1))
                        _cached_n_int_arr.copy_(device_states_soa_coalesced[5].permute(1, 0).to(torch.int32))
                    if os.environ.get('MOLMEM_DEBUG_GPU', '0') == '1':
                        ov = out_cpu
                        print(f"[DEBUG_GPU_KERNEL] steps={int(ov[0])} | currentF22={ov[16]:.4e} | currentIScale={ov[17]:.4e} | currentVScale={ov[18]:.4e} | iDev={ov[19]:.4e} | n={ov[20]:.1f} | mode={ov[21]:.4f} | scale={ov[22]:.4f} | vTop={ov[23]:.4f} | vBot={ov[24]:.4f} | topIdx={int(ov[25])} | botIdx={int(ov[26])}")
                    try:
                        from .sys_utils import log_to_run_log_only
                        log_to_run_log_only(f"[GPU_SOLVE_COMPLETE] Non-streaming solve finished | TotalSteps={hist_idx} | tFinal={t_final:.6e}")
                    except Exception:
                        pass
                else:
                    hist_idx = 0
                    if HAS_CPP_EXTENSION and compile_backend and num_devices > 0:
                        device_states_soa.copy_(device_states_soa_coalesced.permute(2, 0, 1))
                        _cached_n_int_arr.copy_(device_states_soa_coalesced[5].permute(1, 0).to(torch.int32))
                
            chunk_idx = 0
            gpu_hist_offset = 0
            half_capacity = max(10, hist_capacity // 2)
            try:
                while sync_readback and t_chunk_start < tEnd:
                    if verbose and not is_helper:
                        import shutil
                        cols = shutil.get_terminal_size().columns
                        width = max(20, min(150, cols - 3))
                        t_elapsed_cur = time.time() - tStartBench
                        total_cur_steps = getattr(sim, '_hist_active_len', hist_idx) if recordHistory else hist_idx
                        steps_sec = total_cur_steps / max(1e-6, t_elapsed_cur)
                        progress = (t_chunk_start / tEnd) * 100
                        bar = '#' * int(progress / 5)
                        from .sys_utils import get_progress_suffix, get_gpu_hardware_telemetry_str
                        suffix = get_progress_suffix(sim)
                        gpu_telemetry_str = get_gpu_hardware_telemetry_str(device)
                        if detailed:
                            status = f"Progress ({suffix}): [{bar.ljust(20)}] {progress:3.0f}% | {total_cur_steps} steps | {steps_sec:.0f} steps/sec{gpu_telemetry_str}"
                            if len(status) > width:
                                status = status[:width-3] + "..."
                            print(f"\r  {status.ljust(width)}", end="", flush=True)
                        else:
                            status = f"  > {title}: [{bar.ljust(20)}] {progress:3.0f}% | dt: {dt_val*1e9:.3f} ns"
                            if len(status) > width:
                                status = status[:width-3] + "..."
                            print(f"\r{status.ljust(width)}", end="", flush=True)
                    
                    active_stream = stream if stream is not None else (torch.cuda.current_stream(device) if device.type == 'cuda' else None)
                    active_stream_ptr = active_stream.cuda_stream if (active_stream is not None and hasattr(active_stream, 'cuda_stream')) else 0
                    
                    gpu_hist_start = 0 if chunk_idx == 0 else 1

                    t_chunk_end = min(t_chunk_start + t_chunk_step, tEnd)
                    cuda_dev_ctx = torch.cuda.device(device) if (device.type == 'cuda' and torch.cuda.is_available()) else contextlib.nullcontext()
                    stream_ctx = torch.cuda.stream(active_stream) if (active_stream is not None and getattr(active_stream, 'cuda_stream', 0) != 0) else contextlib.nullcontext()
                    with cuda_dev_ctx:
                        with stream_ctx:
                            out_tensor[0] = 0.0
                            if v_pwl_v_matrix.dtype != torch.float64:
                                v_pwl_v_matrix = v_pwl_v_matrix.to(torch.float64)
                            if dyn_pwl_v_matrix.dtype != torch.float64:
                                dyn_pwl_v_matrix = dyn_pwl_v_matrix.to(torch.float64)
                            t0_wall = time.time()
                            try:
                                with _GPU_KERNEL_LAUNCH_LOCK:
                                    molmem_cuda_solver.run_gpu_simulation_double(
                                        t_chunk_start, t_chunk_end, min_dt, tol, maxIter, hist_capacity,
                                        master_schedule_gpu_typed,
                                        is_subsample_gpu,
                                        device_states_soa_coalesced,
                                        device_params_soa_transposed,
                                        numNodes, num_devices,
                                        dev_top_full_idx, dev_bot_full_idx,
                                        dev_top_idx, dev_bot_idx,
                                        gMat_row_indices, gMat_col_indices, iRes_indices,
                                        top_indices, bot_indices, both_indices, dev_both_idx,
                                        d_nodes, dyn_indices, dev_to_dyn_src,
                                        v_nodes_gpu, v_pwl_t_matrix_typed, v_pwl_v_matrix, v_pwl_mask_gpu_int,
                                        dyn_nodes_gpu, dyn_pwl_t_matrix_typed, dyn_pwl_v_matrix, dyn_pwl_mask_gpu_int,
                                        is_1t1r, record_hist_bool, leakage_g,
                                        num_v_sources, num_dyn_sources,
                                        tArr_buffer, vHist_buffer, iHist_buffer,
                                        stateHist_buffer, f22Hist_buffer, gHist_buffer,
                                        tempHist_buffer,
                                        gMat_col_indices.size(0),
                                        v_pwl_v_matrix.size(2),
                                        batch_size,
                                        y_i,
                                        y_v,
                                        y_fp,
                                        y_fd,
                                        y_i.size(0),
                                        workspace_cg,
                                        workspace_t,
                                        out_tensor,
                                        active_stream_ptr
                                    )
                                if active_stream is not None and chunk_event is not None:
                                    chunk_event.record(active_stream)
                                if sync_stream:
                                    if stream is not None and getattr(stream, 'cuda_stream', 0) != 0:
                                        stream.synchronize()
                                    elif torch.cuda.is_available() and device.type == 'cuda':
                                        torch.cuda.synchronize(device)
                            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower():
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    raise MemoryError(
                                        f"[MOLMEM GPU ERROR] Device ran out of memory (VRAM) during simulation execution: {e}\n"
                                        "Please consider reducing batch size, reducing history capacity, or running on CPU."
                                    ) from e
                                raise e
                    out_cpu = out_tensor.detach().cpu().numpy()
                    gpu_hist_end = int(out_cpu[0])
                    t_final = float(out_cpu[1])
                    underflow_flag = float(out_cpu[2])
                    
                    try:
                        from .sys_utils import log_to_run_log_only
                        steps_in_chunk = gpu_hist_end - gpu_hist_start
                        cg_iters = int(out_cpu[18]) if len(out_cpu) >= 19 else 0
                        log_to_run_log_only(f"[GPU_CHUNK_EXEC] Chunk {chunk_idx+1} | tSpan=[{t_chunk_start:.6e}, {t_chunk_end:.6e}] | Steps={steps_in_chunk} | tFinal={t_final:.6e} | CGIters={cg_iters} | KernelMs={(time.time()-t0_wall)*1000.0:.2f}")
                    except Exception:
                        pass
                    
                    try:
                        from .line_profiler import record_gpu_kernel_telemetry
                        if len(out_cpu) >= 20:
                            total_cyc = float(out_cpu[3])
                            if total_cyc > 0:
                                record_gpu_kernel_telemetry({
                                    'total_cycles': total_cyc,
                                    'pwl_cycles': float(out_cpu[4]),
                                    'assembly_cycles': float(out_cpu[5]),
                                    'cg_cycles': float(out_cpu[6]),
                                    'ode_cycles': float(out_cpu[7]),
                                    'hist_cycles': float(out_cpu[8]),
                                    'lut_cycles': float(out_cpu[9]),
                                    'thermal_cycles': float(out_cpu[10]),
                                    'rate_cycles': float(out_cpu[11]),
                                    'boundary_cycles': float(out_cpu[12]),
                                    'mode_cycles': float(out_cpu[13]),
                                    'pred_full_cycles': float(out_cpu[14]),
                                    'pred_half_cycles': float(out_cpu[15]),
                                    'lte_err_cycles': float(out_cpu[16]),
                                    'state_commit_cycles': float(out_cpu[17]),
                                    'cg_iters': int(out_cpu[18]),
                                    'steps': int(out_cpu[19]),
                                    'wall_time': (time.time() - t0_wall) if 't0_wall' in locals() else 0.0,
                                    'batch_size': batch_size,
                                    'num_devices': num_devices,
                                    'num_nodes': numNodes,
                                })
                    except Exception:
                        pass
                    if underflow_flag > 0.5 and not underflow_warned_py:
                        if verbose and not is_helper:
                            print("\r" + " " * width + "\r", end="", flush=True)
                            print("[MOLMEM] Warning: Timestep underflow detected (minimum resolution floor reached). Force-accepting steps to maintain stability. Consider loosening the tolerance 'tol' or running in double precision (dtype=torch.float64).")
                        underflow_warned_py = True
                        
                    if not math.isfinite(t_final) or not math.isfinite(out_tensor[0].item()):
                        raise FloatingPointError(
                            f"[MOLMEM ERROR] Numerical instability detected in compiled C++/HIP solver at t = {t_final:.6e} s. "
                            "One or more state variables or node voltages have diverged to NaN or Inf."
                        )
                    
                    # Asynchronous DMA copy to pinned Host RAM on dedicated transfer stream and flush to DiskHistoryStore
                    if recordHistory:
                        gpu_hist_start = 0 if chunk_idx == 0 else 1
                        chunk_steps = max(0, gpu_hist_end - gpu_hist_start)
                        if chunk_steps > 0:
                            host_start = sim._hist_active_len
                            host_end = host_start + chunk_steps
                            store = sim._ensure_disk_history_store(
                                num_nodes=len(all_nodes_ordered),
                                num_devices=num_devices,
                                initial_capacity=host_end,
                                recordHistory=recordHistory,
                                recordStepTiming=False,
                                batch_size=batch_size
                            )
                            store.ensure_capacity(host_end)
                            store.wait_for_writes()
                            
                            if getattr(sim, '_pinned_tArr', None) is not None and copy_stream is not None:
                                t0_dma = time.perf_counter()
                                with torch.cuda.stream(copy_stream):
                                    if chunk_event is not None:
                                        copy_stream.wait_event(chunk_event)
                                    if batch_size > 1:
                                        sim._pinned_tArr[:, :chunk_steps].copy_(tArr_buffer[:, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                        if v_flag:
                                            sim._pinned_vHist[:, :, :chunk_steps].copy_(vHist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                        if num_devices > 0:
                                            if i_flag:
                                                sim._pinned_iHist[:, :, :chunk_steps].copy_(iHist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if state_flag:
                                                sim._pinned_stateHist[:, :, :chunk_steps].copy_(stateHist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if f22_flag:
                                                sim._pinned_f22Hist[:, :, :chunk_steps].copy_(f22Hist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if g_flag:
                                                sim._pinned_gHist[:, :, :chunk_steps].copy_(gHist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if temp_flag:
                                                sim._pinned_tempHist[:, :, :chunk_steps].copy_(tempHist_buffer[:, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                    else:
                                        sim._pinned_tArr[:chunk_steps].copy_(tArr_buffer[0, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                        if v_flag:
                                            sim._pinned_vHist[:, :chunk_steps].copy_(vHist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                        if num_devices > 0:
                                            if i_flag:
                                                sim._pinned_iHist[:, :chunk_steps].copy_(iHist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if state_flag:
                                                sim._pinned_stateHist[:, :chunk_steps].copy_(stateHist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if f22_flag:
                                                sim._pinned_f22Hist[:, :chunk_steps].copy_(f22Hist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if g_flag:
                                                sim._pinned_gHist[:, :chunk_steps].copy_(gHist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                            if temp_flag:
                                                sim._pinned_tempHist[:, :chunk_steps].copy_(tempHist_buffer[0, :, gpu_hist_start:gpu_hist_end], non_blocking=True)
                                copy_stream.synchronize()
                                dma_dur = time.perf_counter() - t0_dma
                                dma_bytes = (1 + (len(all_nodes_ordered) if v_flag else 0) + (num_devices * 5 if num_devices > 0 else 0)) * chunk_steps * 8 * batch_size
                                try:
                                    from .history import SSDPipelineTelemetry
                                    SSDPipelineTelemetry.get_instance().record_dma(dma_dur, dma_bytes)
                                except Exception:
                                    pass
                                store.write_chunk_async(
                                    host_start=host_start,
                                    completed_steps=chunk_steps,
                                    recordHistory=recordHistory,
                                    recordStepTiming=False,
                                    tArr=sim._pinned_tArr.numpy(),
                                    vHist=sim._pinned_vHist.numpy() if v_flag else None,
                                    iHist=sim._pinned_iHist.numpy() if (num_devices > 0 and i_flag) else None,
                                    stateHist=sim._pinned_stateHist.numpy() if (num_devices > 0 and state_flag) else None,
                                    f22Hist=sim._pinned_f22Hist.numpy() if (num_devices > 0 and f22_flag) else None,
                                    gHist=sim._pinned_gHist.numpy() if (num_devices > 0 and g_flag) else None,
                                    tempHist=sim._pinned_tempHist.numpy() if (num_devices > 0 and temp_flag) else None
                                )
                            else:
                                if batch_size > 1:
                                    store.write_chunk_async(
                                        host_start=host_start,
                                        completed_steps=chunk_steps,
                                        recordHistory=recordHistory,
                                        recordStepTiming=False,
                                        tArr=tArr_buffer[:, gpu_hist_start:gpu_hist_end].cpu().numpy(),
                                        vHist=vHist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if v_flag else None,
                                        iHist=iHist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and i_flag) else None,
                                        stateHist=stateHist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and state_flag) else None,
                                        f22Hist=f22Hist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and f22_flag) else None,
                                        gHist=gHist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and g_flag) else None,
                                        tempHist=tempHist_buffer[:, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and temp_flag) else None
                                    )
                                else:
                                    store.write_chunk_async(
                                        host_start=host_start,
                                        completed_steps=chunk_steps,
                                        recordHistory=recordHistory,
                                        recordStepTiming=False,
                                        tArr=tArr_buffer[0, gpu_hist_start:gpu_hist_end].cpu().numpy(),
                                        vHist=vHist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if v_flag else None,
                                        iHist=iHist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and i_flag) else None,
                                        stateHist=stateHist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and state_flag) else None,
                                        f22Hist=f22Hist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and f22_flag) else None,
                                        gHist=gHist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and g_flag) else None,
                                        tempHist=tempHist_buffer[0, :, gpu_hist_start:gpu_hist_end].cpu().numpy() if (num_devices > 0 and temp_flag) else None
                                    )
                            
                            sim._tArr_full = store.tArr
                            sim._stepTimeHistory_full = store.stepTime
                            sim._vHistory_full = store.vHist
                            sim._iHistory_full = store.iHist
                            sim._stateHistory_full = store.stateHist
                            sim._f22History_full = store.f22Hist
                            sim._gHistory_full = store.gHist
                            sim._tempHistory_full = store.tempHist
                            sim._hist_capacity_total = store.capacity
                            sim._hist_active_len = host_end

                    dt_wall = time.time() - t0_wall
                    t_simulated = t_final - t_chunk_start
                    if dt_wall > 0.0001 and t_simulated > 0:
                        speed = t_simulated / dt_wall
                        t_remaining_physical = tEnd - t_final
                        projected_remaining_wall = t_remaining_physical / speed if speed > 1e-15 else float('inf')
                        # Target 0.9s wall-time chunk steps to maximize continuous execution while staying under Windows WDDM 2.0s TDR limit
                        if projected_remaining_wall < 0.5:
                            # Remainder fits well within single safe budget, dispatch full remaining duration
                            t_chunk_step = t_remaining_physical
                        else:
                            target_dt_wall = 0.9
                            t_chunk_step = max(1e-6, min(speed * target_dt_wall, t_remaining_physical))
                        
                    chunk_idx += 1
                    if t_final <= t_chunk_start + 1e-18:
                        break
                    t_chunk_start = t_final
            finally:
                if copy_stream is not None:
                    try:
                        copy_stream.synchronize()
                    except Exception:
                        pass
                if 'store' in locals() and store is not None:
                    try:
                        store.wait_for_writes()
                    except Exception:
                        pass
            
            if not math.isfinite(t_final):
                raise FloatingPointError(
                    f"[MOLMEM ERROR] Numerical instability detected in compiled C++/HIP solver at t = {t_final:.6e} s. "
                    "One or more state variables or node voltages have diverged to NaN or Inf."
                )
            
            if t_final < tEnd - max(1e-9, tEnd * 1e-5):
                sys.stderr.write(f"\n[Warning] GPU simulation stopped early at t = {t_final:.6e} s (requested tEnd = {tEnd:.6e} s).\n")
                sys.stderr.write("This occurs if the timestep reaches the minimum resolution floor or step rejection limit.\n")
                sys.stderr.flush()
            
            # Explicit stream synchronization before copying transient outputs to host-visible variables
            active_stream = stream if stream is not None else (torch.cuda.current_stream(device) if device.type == 'cuda' else None)
            if active_stream is not None:
                active_stream.synchronize()
                
            if recordHistory and getattr(sim, '_disk_history_store', None) is not None:
                sim._disk_history_store.wait_for_writes()
                
            # Write back updated states to original in-place cache tensor
            device_states_soa.copy_(device_states_soa_coalesced.permute(2, 0, 1))
            if cache is not None:
                cache['n_arr'].copy_(device_states_soa[:, 0, :])
                cache['modeState_arr'].copy_(device_states_soa[:, 1, :])
                cache['scaleFactor_arr'].copy_(device_states_soa[:, 2, :])
                cache['f22DepScaled_arr'].copy_(device_states_soa[:, 3, :])
                cache['f22PotScaled_arr'].copy_(device_states_soa[:, 4, :])
                cache['_cached_n_int_arr'].copy_(device_states_soa[:, 5, :])
                cache['iScale_arr'].copy_(device_states_soa[:, 6, :])
                cache['vScale_arr'].copy_(device_states_soa[:, 7, :])
                cache['currentF22_arr'].copy_(device_states_soa[:, 8, :])
                cache['currentIScale_arr'].copy_(device_states_soa[:, 9, :])
                cache['currentVScale_arr'].copy_(device_states_soa[:, 10, :])
                cache['T_arr'].copy_(device_states_soa[:, 11, :])
            
            if cache is not None and num_devices > 0:
                if workspace_t is not None and workspace_t.numel() > 0:
                    pad_srcs = max(num_v_sources, num_dyn_sources)
                    if pad_srcs < 2:
                        pad_srcs = 2
                    offset_base = num_gMat_vals + 4 * numNodes + num_v_sources + pad_srcs
                    flat_iDev = workspace_t[(offset_base + 18 * num_devices) * batch_size : (offset_base + 19 * num_devices) * batch_size]
                    flat_gEq = workspace_t[(offset_base + 19 * num_devices) * batch_size : (offset_base + 20 * num_devices) * batch_size]
                    cache['iDev_arr'].copy_(flat_iDev.view(batch_size, num_devices))
                    cache['gEq_arr'].copy_(flat_gEq.view(batch_size, num_devices))
                elif i_flag and hist_idx > 0 and iHist_buffer.size(-1) >= hist_idx:
                    cache['iDev_arr'].copy_(iHist_buffer[:, :, hist_idx - 1])
                    if g_flag and gHist_buffer.size(-1) >= hist_idx:
                        cache['gEq_arr'].copy_(gHist_buffer[:, :, hist_idx - 1])
        else:
            global _compiled_loop_fn
            if _compiled_loop_fn is None:
                # Under Windows/ROCm without Triton, attempt to load pre-compiled TorchScript loop if available
                if os.path.exists(_compiled_loop_path):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            _compiled_loop_fn = torch.jit.load(_compiled_loop_path)
                    except Exception:
                        pass
                
                if _compiled_loop_fn is None:
                    reason = CPP_EXTENSION_DISABLED_REASON or "Molmem C++ ROCm/CUDA extension is not compiled or active"
                    raise RuntimeError(
                        f"[MOLMEM ERROR] GPU transient simulation requires the compiled Molmem C++ / HIP extension solver.\n"
                        f"Status: {reason}.\n"
                        f"Please build the C++ extension (via csrc_setup.py) or configure the simulator with device='cpu'."
                    )
            
            if sync_readback and (tEnd - t_start) > 1e-12:
                loop_args_list = list(loop_args)
                t_chunk_start = t_start
                num_chunks = 20
                t_chunk_step = (tEnd - t_start) / num_chunks
                if t_chunk_step < 1e-12:
                    t_chunk_step = tEnd - t_start
                
                hist_idx = 0
                t_final = t_start
                detailed = getattr(sim, 'detailedPrint', True)
                
                if verbose and not is_helper:
                    import shutil
                    cols = shutil.get_terminal_size().columns
                    width = max(20, min(150, cols - 3))
                    bar = ' ' * 20
                    from .sys_utils import get_progress_suffix
                    suffix = get_progress_suffix(sim)
                    if detailed:
                        status = f"Progress ({suffix}): [{bar}]   0% | 0 steps | 0 steps/sec"
                        if len(status) > width:
                            status = status[:width-3] + "..."
                        sys.stdout.write(f"\r  {status.ljust(width)}")
                    else:
                        status = f"  > {title}: [{bar}]   0%"
                        if len(status) > width:
                            status = status[:width-3] + "..."
                        sys.stdout.write(f"\r{status.ljust(width)}")
                    sys.stdout.flush()
                
                while t_chunk_start < tEnd:
                    t_chunk_end = min(t_chunk_start + t_chunk_step, tEnd)
                    loop_args_list[0] = t_chunk_start
                    loop_args_list[1] = t_chunk_end
                    
                    chunk_hist_idx, t_final_chunk = _compiled_loop_fn(*(tuple(loop_args_list) + (hist_idx,)))
                    hist_idx = chunk_hist_idx
                    t_final = float(t_final_chunk.min().item())
                    
                    if verbose and not is_helper:
                        import shutil
                        cols = shutil.get_terminal_size().columns
                        width = max(20, min(150, cols - 3))
                        t_elapsed_cur = time.time() - tStartBench
                        steps_sec = hist_idx / max(1e-6, t_elapsed_cur)
                        progress = ((t_final - t_start) / (tEnd - t_start)) * 100.0
                        bar = '#' * int(progress / 5)
                        from .sys_utils import get_progress_suffix
                        suffix = get_progress_suffix(sim)
                        if detailed:
                            status = f"Progress ({suffix}): [{bar.ljust(20)}] {progress:3.0f}% | {hist_idx} steps | {steps_sec:.0f} steps/sec"
                            if len(status) > width:
                                status = status[:width-3] + "..."
                            sys.stdout.write(f"\r  {status.ljust(width)}")
                        else:
                            status = f"  > {title}: [{bar.ljust(20)}] {progress:3.0f}%"
                            if len(status) > width:
                                status = status[:width-3] + "..."
                            sys.stdout.write(f"\r{status.ljust(width)}")
                        sys.stdout.flush()
                        
                    if t_final <= t_chunk_start + 1e-18:
                        break
                    t_chunk_start = t_final
                
                if verbose and not is_helper:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            else:
                hist_idx, t_final_tensor = _compiled_loop_fn(*(loop_args + (0,)))
                t_final = float(t_final_tensor.min().item())

    # 4. Numerical Instability check (NaN/Inf Detection for JIT fallback)
    if num_devices > 0 and (not HAS_CPP_EXTENSION or not compile_backend):
        if torch.isnan(cache['device_states_soa']).any() or torch.isinf(cache['device_states_soa']).any() or \
           torch.isnan(cache['vCurrent_arr']).any() or torch.isinf(cache['vCurrent_arr']).any():
            raise FloatingPointError(
                "Numerical Instability Detected: The transient solver has diverged, producing NaN or Inf "
                "in node voltages or device states. Suggestions:\n"
                "  1. Try lowering the convergence tolerance (e.g. tol=1e-4 or tol=1e-5).\n"
                "  2. Decrease the minimum timestep size (min_dt).\n"
                "  3. Check for steep input voltage transitions or extreme physical parameters."
            )

    t_end_loop = time.time()
    
    os.environ['MOLMEM_GPU_JIT_WARMED_UP'] = '1'
    
    if HAS_CPP_EXTENSION and compile_backend and 'device_states_soa_coalesced' in locals() and device_states_soa_coalesced is not None and device_states_soa.numel() > 0:
        device_states_soa.copy_(device_states_soa_coalesced.permute(2, 0, 1))
    
    # Back-sync states to sim.devices (on host) - LAZY SYNC
    sim._gpu_states_in_sync = False
    sim._gpu_states_dirty = False
    if num_devices > 0 and not getattr(sim, 'skip_readback', False):
        if HAS_CPP_EXTENSION and compile_backend and 'workspace_t' in locals() and workspace_t is not None and workspace_t.numel() > 0:
            pad_srcs = max(num_v_sources, num_dyn_sources)
            if pad_srcs < 2:
                pad_srcs = 2
            offset_iDev = (num_gMat_vals + 4 * numNodes + num_v_sources + pad_srcs + 18 * num_devices) * batch_size
            dest_iDev_tensor = workspace_t[offset_iDev : offset_iDev + batch_size * num_devices].view(batch_size, num_devices)
            if batch_size == 1:
                sim.last_iDev = dest_iDev_tensor[0].cpu().numpy()
            else:
                sim.last_iDev = dest_iDev_tensor.cpu().numpy()
        elif cache is not None and isinstance(cache, dict) and 'iDev_arr' in cache and cache['iDev_arr'] is not None:
            if batch_size == 1:
                sim.last_iDev = cache['iDev_arr'][0].cpu().numpy()
            else:
                sim.last_iDev = cache['iDev_arr'].cpu().numpy()
        elif 'iDev_arr' in locals() and iDev_arr is not None:
            if batch_size == 1:
                sim.last_iDev = iDev_arr[0].cpu().numpy()
            else:
                sim.last_iDev = iDev_arr.cpu().numpy()
    if num_devices > 0 and not getattr(sim, 'lazy_sync', True):
        sim.sync_to_cpu()
            
    # Write history if batch_size == 1 (standard single simulation backward-compatibility)
    if recordHistory and batch_size == 1:
        if not sync_readback:
            has_history = appendHistory and getattr(sim, '_hist_active_len', 0) > 0
            if not has_history:
                sim._hist_active_len = 0
                if all_nodes_ordered is None:
                    topo = getattr(sim, '_cached_torch_topology', None)
                    if topo is not None and 'all_nodes_ordered' in topo:
                        all_nodes_ordered = topo['all_nodes_ordered']
                    elif hasattr(sim, 'nodesUnknown') and hasattr(sim, 'nodesKnown'):
                        all_nodes_ordered = list(sim.nodesUnknown) + list(sim.nodesKnown)
                    else:
                        all_nodes_ordered = ['gnd']
                store = sim._ensure_disk_history_store(
                    num_nodes=len(all_nodes_ordered),
                    num_devices=num_devices,
                    initial_capacity=max(2000, hist_idx * 2),
                    recordHistory=recordHistory,
                    recordStepTiming=False
                )
            else:
                store = sim._ensure_disk_history_store(
                    num_nodes=len(all_nodes_ordered),
                    num_devices=num_devices,
                    initial_capacity=getattr(sim, '_hist_active_len', 0) + hist_idx,
                    recordHistory=recordHistory,
                    recordStepTiming=False
                )
                
            start = getattr(sim, '_hist_active_len', 0)
            end = start + hist_idx
            store.ensure_capacity(end)
            
            store.tArr[start:end] = tArr_buffer[0, :hist_idx].cpu().numpy()
            if v_flag:
                store.vHist[:, start:end] = vHist_buffer[0, :, :hist_idx].cpu().numpy()
            
            if num_devices > 0:
                if i_flag:
                    store.iHist[:, start:end] = iHist_buffer[0, :, :hist_idx].cpu().numpy()
                if state_flag:
                    store.stateHist[:, start:end] = stateHist_buffer[0, :, :hist_idx].cpu().numpy()
                if f22_flag:
                    store.f22Hist[:, start:end] = f22Hist_buffer[0, :, :hist_idx].cpu().numpy()
                if g_flag:
                    store.gHist[:, start:end] = gHist_buffer[0, :, :hist_idx].cpu().numpy()
                if temp_flag:
                    store.tempHist[:, start:end] = tempHist_buffer[0, :, :hist_idx].cpu().numpy()
                
            sim._tArr_full = store.tArr
            sim._stepTimeHistory_full = store.stepTime
            sim._vHistory_full = store.vHist
            sim._iHistory_full = store.iHist
            sim._stateHistory_full = store.stateHist
            sim._f22History_full = store.f22Hist
            sim._gHistory_full = store.gHist
            sim._tempHistory_full = store.tempHist
            sim._hist_capacity_total = store.capacity
            sim._hist_active_len = end
            
        end = getattr(sim, '_hist_active_len', hist_idx)
        if hasattr(sim, '_tArr_full') and sim._tArr_full is not None:
            sim.tArr = sim._tArr_full[:end]
            sim.stepTimeHistory = sim._stepTimeHistory_full[:end]
            sim.vHistory = LazyHistoryDict(sim._vHistory_full, all_nodes_ordered, end) if v_flag else {}
            
            if num_devices > 0:
                if HAS_CPP_EXTENSION and compile_backend:
                    _cached_n_int_arr.copy_(device_states_soa_coalesced[5].permute(1, 0).to(torch.int32))
                sim.iHistory = LazyDeviceHistoryDict(sim._iHistory_full, num_devices, end) if i_flag else {}
                sim.stateHistory = LazyDeviceHistoryDict(sim._stateHistory_full, num_devices, end) if state_flag else {}
                sim.f22History = LazyDeviceHistoryDict(sim._f22History_full, num_devices, end) if f22_flag else {}
                sim.gHistory = LazyDeviceHistoryDict(sim._gHistory_full, num_devices, end) if g_flag else {}
                sim.tempHistory = LazyDeviceHistoryDict(sim._tempHistory_full, num_devices, end) if temp_flag else {}
            
    sim._gpu_states_dirty = False
    sim._gpu_states_in_sync = False
    
    if num_devices > 0:
        if HAS_CPP_EXTENSION and compile_backend:
            device_states_soa.copy_(device_states_soa_coalesced.permute(2, 0, 1))
            _cached_n_int_arr.copy_(device_states_soa_coalesced[5].permute(1, 0).to(torch.int32))
        d_soa_cloned = device_states_soa[0].unsqueeze(0).clone() if batch_size > 1 else device_states_soa.clone()
        c_int_cloned = _cached_n_int_arr[0].unsqueeze(0).clone() if batch_size > 1 else _cached_n_int_arr.clone()
        sim._gpu_states_cache = {
            'device_states_soa': d_soa_cloned,
            '_cached_n_int_arr': c_int_cloned,
            'n_arr': d_soa_cloned[:, 0, :],
            'modeState_arr': d_soa_cloned[:, 1, :],
            'scaleFactor_arr': d_soa_cloned[:, 2, :],
            'f22DepScaled_arr': d_soa_cloned[:, 3, :],
            'f22PotScaled_arr': d_soa_cloned[:, 4, :],
            'iScale_arr': d_soa_cloned[:, 6, :],
            'vScale_arr': d_soa_cloned[:, 7, :],
            'currentF22_arr': d_soa_cloned[:, 8, :],
            'currentIScale_arr': d_soa_cloned[:, 9, :],
            'currentVScale_arr': d_soa_cloned[:, 10, :],
            'T_arr': d_soa_cloned[:, 11, :],
        }
            
    tElapsed = time.time() - tStartBench
    total_reported_steps = getattr(sim, '_hist_active_len', hist_idx) if recordHistory else hist_idx
    if (verbose or is_compiling) and not is_helper and not getattr(sim, '_in_warmup', False) and not getattr(sim.__class__, '_in_warmup', False):
        if detailed:
            from .sys_utils import get_progress_suffix, get_gpu_hardware_telemetry_str
            suffix = get_progress_suffix(sim)
            prefix = f"Compiling ({suffix}):" if is_compiling else f"Progress ({suffix}):"
            steps_sec = total_reported_steps / max(1e-6, tElapsed)
            gpu_telemetry_str = get_gpu_hardware_telemetry_str(device)
            print(f"\r  {prefix} [{'#' * 20}] 100% | Done in {tElapsed:.2f}s ({total_reported_steps} steps, {steps_sec:.0f} steps/sec{gpu_telemetry_str})")
        else:
            import shutil
            cols = shutil.get_terminal_size().columns
            width = max(20, min(150, cols - 3))
            status = f"  > {title}: Done (in {tElapsed:.2f}s | {total_reported_steps} steps)"
            if len(status) > width:
                status = status[:width-3] + "..."
            print(f"\r{status.ljust(width)}")
        
    final_hist_idx = (getattr(sim, '_hist_active_len', hist_idx) if (recordHistory and sync_readback) else hist_idx)
    return final_hist_idx, tArr_buffer, vHist_buffer, iHist_buffer, n_arr, modeState_arr, scaleFactor_arr, f22DepScaled_arr, f22PotScaled_arr, _cached_n_int_arr, iScale_arr, vScale_arr, currentF22_arr, currentIScale_arr, currentVScale_arr, T_arr
