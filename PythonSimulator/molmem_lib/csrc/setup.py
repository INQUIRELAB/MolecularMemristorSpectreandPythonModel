from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension
import os
import torch
import multiprocessing

# Permit compiling with newer CUDA versions (e.g. CUDA 13.3 with PyTorch 12.1) without PyTorch throwing a hard RuntimeError
try:
    _orig_check_cuda_version = torch.utils.cpp_extension._check_cuda_version
    def _lenient_check_cuda_version(*args, **kwargs):
        try:
            _orig_check_cuda_version(*args, **kwargs)
        except RuntimeError as _e:
            import warnings
            warnings.warn(f"Permitting CUDA forward compatibility: {str(_e).strip()}")
    torch.utils.cpp_extension._check_cuda_version = _lenient_check_cuda_version
except Exception:
    pass

# Enable parallel compilation for setuptools / PyTorch extension build
if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = str(min(multiprocessing.cpu_count(), 61))

# Detect whether we are compiling for AMD ROCm or NVIDIA CUDA
is_rocm = False
if hasattr(torch, "version") and torch.version.hip is not None:
    is_rocm = True

# Set broad architecture target defaults if not already defined in environment
if is_rocm:
    if os.name == 'nt':
        # Windows ROCm targets: RDNA2 (gfx1030), RDNA3/APU (gfx1100, gfx1101, gfx1102, gfx1103), RDNA3.5 (gfx1150, gfx1151), and RDNA4 (gfx1200, gfx1201)
        # Always override on Windows since ROCm Windows compiler does not support CDNA/gfx940 targets.
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
    elif "PYTORCH_ROCM_ARCH" not in os.environ:
        # Linux ROCm targets: Vega (gfx906), CDNA1/2/3 (gfx908, gfx90a, gfx942), RDNA1 (gfx1010, gfx1012), RDNA2 (gfx1030, gfx1031, gfx1032, gfx1035, gfx1036), RDNA3/APU (gfx1100, gfx1101, gfx1102, gfx1103), RDNA3.5 (gfx1150, gfx1151), and RDNA4 (gfx1200, gfx1201)
        base_archs = ["gfx906", "gfx908", "gfx90a", "gfx942", "gfx1010", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1035", "gfx1036", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]
        try:
            import subprocess, shutil
            info_bin = shutil.which("rocminfo") or shutil.which("rocm_agent_enumerator")
            if info_bin:
                out = subprocess.check_output([info_bin], stderr=subprocess.DEVNULL, text=True, timeout=3)
                for word in out.split():
                    w = word.strip().lower()
                    if w.startswith("gfx") and len(w) in (6, 7) and w not in base_archs:
                        base_archs.append(w)
        except Exception:
            pass
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(base_archs)
else:
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        # Check if CUDA 13 or newer is used (CUDA 13 dropped offline compilation support for compute_60 and compute_70)
        is_cuda13_or_newer = False
        try:
            import subprocess, shutil, re
            nvcc_candidate = shutil.which("nvcc")
            if not nvcc_candidate and os.name == "nt":
                for c_drive in ["C:", "D:", "E:"]:
                    c_root = os.path.join(c_drive, os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA")
                    if os.path.exists(c_root):
                        sub_c = sorted([d for d in os.listdir(c_root) if os.path.isdir(os.path.join(c_root, d))], reverse=True)
                        for sc in sub_c:
                            test_bin = os.path.join(c_root, sc, "bin", "nvcc.exe")
                            if os.path.exists(test_bin):
                                nvcc_candidate = test_bin
                                break
                    if nvcc_candidate:
                        break
            if nvcc_candidate:
                nvcc_out = subprocess.check_output([nvcc_candidate, "--version"], stderr=subprocess.DEVNULL, text=True, timeout=2)
                m = re.search(r"release (\d+)\.", nvcc_out)
                if m and int(m.group(1)) >= 13:
                    is_cuda13_or_newer = True
        except Exception:
            pass

        # Dynamically test candidate architectures against PyTorch's internal table to prevent ValueError
        if is_cuda13_or_newer:
            candidate_archs = ["7.5", "8.0", "8.6", "8.7", "8.9", "9.0", "10.0", "12.0"]
        else:
            candidate_archs = ["6.0", "7.0", "7.5", "8.0", "8.6", "8.7", "8.9", "9.0", "10.0", "12.0"]
        supported_archs = []
        try:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            for arch in candidate_archs:
                old_val = os.environ.get("TORCH_CUDA_ARCH_LIST")
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
                try:
                    _get_cuda_arch_flags()
                    supported_archs.append(arch)
                except ValueError:
                    pass
                finally:
                    if old_val is not None:
                        os.environ["TORCH_CUDA_ARCH_LIST"] = old_val
                    else:
                        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        except Exception:
            supported_archs = ["7.5", "8.0", "8.6", "8.9", "9.0"]

        if supported_archs:
            # Append +PTX to the highest architecture for forward compatibility (e.g. Blackwell RTX 5090)
            supported_archs[-1] = supported_archs[-1] + "+PTX"
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(supported_archs)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"

# Write target architecture metadata to .csrc_arch so runtime can verify compatibility
try:
    _target_archs = os.environ.get("PYTORCH_ROCM_ARCH" if is_rocm else "TORCH_CUDA_ARCH_LIST", "")
    if _target_archs:
        arch_file = os.path.join(os.path.dirname(__file__), ".csrc_arch")
        if os.name == 'nt' and os.path.exists(arch_file):
            try:
                import ctypes
                ctypes.windll.kernel32.SetFileAttributesW(str(arch_file), 0x80)
            except Exception:
                pass
        with open(arch_file, "w", encoding="utf-8") as _f:
            _f.write(_target_archs.strip() + "\n")
except Exception:
    pass

def _get_short_path(path):
    if not path or os.name != 'nt':
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
    for env_var in ["HIP_PATH", "ROCM_PATH", "HIP_DIR", "ROCM_HOME"]:
        path = os.environ.get(env_var)
        if path and os.path.exists(path):
            return _get_short_path(os.path.abspath(path))
    if os.name != 'nt':
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
                return os.path.abspath(p)
        import shutil
        hipcc_p = shutil.which("hipcc")
        if hipcc_p:
            prefix = os.path.dirname(os.path.dirname(os.path.abspath(hipcc_p)))
            if os.path.exists(prefix) and prefix != "/":
                return prefix
        for sys_cand in ["/usr", "/usr/local"]:
            if os.path.exists(os.path.join(sys_cand, "bin", "hipcc")) or os.path.exists(os.path.join(sys_cand, "include", "hip")):
                return sys_cand
        return None
    default_root = r"C:\PROGRA~1\AMD\ROCm" if os.path.exists(r"C:\PROGRA~1\AMD\ROCm") else r"C:\Program Files\AMD\ROCm"
    if os.path.exists(default_root):
        try:
            subdirs = [d for d in os.listdir(default_root) if os.path.isdir(os.path.join(default_root, d))]
            subdirs.sort(key=lambda s: [int(x) if x.isdigit() else x for x in s.split('.')], reverse=True)
            for subdir in subdirs:
                path = os.path.join(default_root, subdir)
                if os.path.exists(os.path.join(path, "bin", "hipcc.exe")) or os.path.exists(os.path.join(path, "bin", "hipcc")):
                    return _get_short_path(path)
        except Exception:
            pass
    return None

def _find_rocm_device_lib_path(rocm_path=None):
    for ev in ["ROCM_DEVICE_LIB_PATH", "DEVICE_LIB_PATH", "HIP_DEVICE_LIB_PATH"]:
        p = os.environ.get(ev)
        if p and os.path.exists(os.path.join(p, "ockl.bc")):
            return _get_short_path(os.path.abspath(p))
    if os.name == 'nt':
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

include_dirs = []
if is_rocm:
    # Target AMD ROCm include folders dynamically based on OS platform
    rocm_path = _find_rocm_path()
    if rocm_path and os.path.exists(rocm_path):
        rocm_path = _get_short_path(rocm_path)
        for ev in ["ROCM_HOME", "ROCM_PATH", "HIP_PATH", "HIP_DIR"]:
            os.environ[ev] = rocm_path
        rocm_include = os.path.join(rocm_path, "include").replace("\\", "/")
        if os.path.exists(rocm_include) and rocm_include not in include_dirs:
            include_dirs.append(rocm_include)
        hip_include = os.path.join(rocm_path, "include", "hip").replace("\\", "/")
        if os.path.exists(hip_include) and hip_include not in include_dirs:
            include_dirs.append(hip_include)
    nvcc_args = ['-O3', '-gline-tables-only', '-nohipwrapperinc', '-D_HAS_CMATH_INTRINSICS=0', '-ffast-math', '-fcuda-approx-transcendentals']
    dev_lib_path = _find_rocm_device_lib_path(rocm_path)
    if dev_lib_path:
        dev_lib_path = _get_short_path(dev_lib_path)
        nvcc_args.append(f'--rocm-device-lib-path={dev_lib_path}')
        os.environ["ROCM_DEVICE_LIB_PATH"] = dev_lib_path
    solver_src = 'gpu_solver.hip'
else:
    # Target standard NVIDIA CUDA compiler flags
    # PyTorch BuildExtension strictly requires .cu extension to invoke nvcc.
    # If passed a .hip file, setuptools passes it to gcc, which ignores it without building .o.
    solver_src = 'gpu_solver.cu'
    try:
        import shutil
        this_dir = os.path.dirname(os.path.abspath(__file__))
        hip_p = os.path.join(this_dir, 'gpu_solver.hip')
        cu_p = os.path.join(this_dir, 'gpu_solver.cu')
        if os.path.exists(hip_p):
            shutil.copyfile(hip_p, cu_p)
    except Exception:
        pass
    def _find_cuda_home_path():
        for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_DIR"]:
            p = os.environ.get(env_var)
            if p and os.path.exists(p):
                cand_nvcc = os.path.join(p, "bin", "nvcc.exe" if os.name == 'nt' else "nvcc")
                if os.path.exists(cand_nvcc) or os.path.exists(os.path.join(p, "include")):
                    return _get_short_path(os.path.abspath(p))
        if os.name == 'nt':
            # Check versioned CUDA_PATH_V* in environment (sorted descending to prefer latest like v13.3)
            cuda_v_vars = sorted([ev for ev in os.environ if ev.startswith("CUDA_PATH_V")], reverse=True)
            for ev in cuda_v_vars:
                cand = os.environ[ev]
                if os.path.exists(cand) and os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                    return _get_short_path(os.path.abspath(cand))
            # Check Windows Registry (both System Environment and User Environment)
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
                    return _get_short_path(reg_cands[0])
            except Exception:
                pass
            # Check nvcc on PATH
            import shutil
            nvcc_p = shutil.which("nvcc")
            if nvcc_p:
                prefix = os.path.dirname(os.path.dirname(os.path.abspath(nvcc_p)))
                if os.path.isdir(prefix) and (os.path.exists(os.path.join(prefix, "include")) or os.path.exists(os.path.join(prefix, "lib"))):
                    return _get_short_path(os.path.abspath(prefix))
            # Check standard NVIDIA GPU Computing Toolkit install directories across drives (preferring 13.3 <= ver < 14.0)
            for drive in ["C:", "D:", "E:"]:
                default_root = os.path.join(drive, os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA")
                if os.path.exists(default_root):
                    try:
                        subdirs = [d for d in os.listdir(default_root) if os.path.isdir(os.path.join(default_root, d))]
                        subdirs.sort(key=_cuda_version_priority_key, reverse=True)
                        for subdir in subdirs:
                            cand = os.path.join(default_root, subdir)
                            if os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                                return _get_short_path(os.path.abspath(cand))
                    except Exception:
                        pass
        else:
            import glob
            for cand in sorted(glob.glob("/usr/local/cuda*") + glob.glob("/opt/cuda*"), reverse=True):
                if os.path.isdir(cand) and (os.path.exists(os.path.join(cand, "bin", "nvcc")) or os.path.exists(os.path.join(cand, "include"))):
                    return os.path.abspath(cand)
        return None

    cuda_home = _find_cuda_home_path()
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        cuda_inc = os.path.join(cuda_home, "include").replace("\\", "/")
        if os.path.isdir(cuda_inc) and cuda_inc not in include_dirs:
            include_dirs.append(cuda_inc)
    nvcc_args = ['-O3', '-lineinfo', '-use_fast_math', '-diag-suppress', '177']

cxx_args = ['/O2', '/MP'] if os.name == 'nt' else ['-O3']

try:
    ext_sources = ['gpu_solver_binding.cpp', solver_src]
    if is_rocm:
        ext_sources.append('c10_stub.cpp')

    setup(
        name='molmem_cuda_solver',
        ext_modules=[
            CUDAExtension(
                name='molmem_cuda_solver',
                sources=ext_sources,
                include_dirs=include_dirs,
                extra_compile_args={
                    'cxx': cxx_args,
                    'nvcc': nvcc_args
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    
    # Auto-write source hashes locally if the build succeeded (produced a binary)
    import hashlib
    files = ["gpu_solver.hip", "gpu_solver_binding.cpp", "c10_stub.cpp", "setup.py"]
    hasher = hashlib.md5()
    for f_name in files:
        if os.path.exists(f_name):
            with open(f_name, "rb") as f:
                hasher.update(f.read())
    try:
        import torch
        hasher.update(str(torch.__version__).encode('utf-8'))
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                hasher.update(torch.cuda.get_device_name(i).encode('utf-8'))
    except Exception:
        pass
    current_hash = hasher.hexdigest()
    
    csrc_dir = os.path.dirname(os.path.abspath(__file__))
    has_binary = False
    for file in os.listdir(csrc_dir):
        if file.startswith("molmem_cuda_solver") and (file.endswith(".pyd") or file.endswith(".so")):
            has_binary = True
            break
                
    if has_binary:
        hash_file = os.path.join(csrc_dir, ".csrc_hash")
        try:
            if os.name == 'nt' and os.path.exists(hash_file):
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetFileAttributesW(str(hash_file), 0x80)
                except Exception:
                    pass
            with open(hash_file, "w") as f:
                f.write(current_hash)
            print(f"[MOLMEM JIT] C++ Extension hash updated locally in setup.py: {current_hash}")
        except Exception:
            pass

        # Auto-clean intermediate compiler build directory to ensure zero residual host paths
        try:
            import shutil
            for clean_sub in ["build", "molmem_cuda_solver.egg-info"]:
                clean_target = os.path.join(csrc_dir, clean_sub)
                if os.path.isdir(clean_target):
                    shutil.rmtree(clean_target, ignore_errors=True)
        except Exception:
            pass
except Exception as e:
    raise e
