"""
Universal Empirical Noise Engine for MolmemSimulator.
Provides physically accurate statistical noise channels extracted from experimental hardware:
1. Device-to-Device (D2D) Spatial Mismatch via empirical Inverse-CDF quantile mapping.
2. Temporal Read Noise (Johnson-Nyquist thermal floor + Hooge 1/f flicker noise).
3. Cycle-to-Cycle Write Noise (Langevin ionic hopping programming variance).
"""

import os
import threading
import numpy as np

_PWL_DIR_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "noise")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PhysicsFitting", "Noise", "generated_pwl")),
    os.path.abspath("PhysicsFitting/Noise/generated_pwl"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Noise", "generated_pwl")),
    os.path.abspath("Noise/generated_pwl"),
]

def _find_pwl_file(filename):
    base, ext = os.path.splitext(filename)
    candidates_to_try = [base + ".bin", base + ".pwl"] if ext in (".bin", ".pwl") else [filename, base + ".bin", base + ".pwl"]
    for directory in _PWL_DIR_CANDIDATES:
        for fname in candidates_to_try:
            candidate = os.path.join(directory, fname)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"Could not locate noise data file '{filename}' in candidate directories: {_PWL_DIR_CANDIDATES}")

def _load_pwl(filepath):
    from .sys_utils import load_physics_table
    data = load_physics_table(filepath)
    return np.ascontiguousarray(data[:, 0], dtype=np.float64), np.ascontiguousarray(data[:, 1], dtype=np.float64)


class NoiseEngine:
    """
    Singleton noise manager providing high-performance CPU (NumPy) and GPU (PyTorch)
    statistical noise injection with zero-overhead bypass when intensity is 0.0.
    """
    _initialized = False
    _d2d_u = None
    _d2d_dev = None
    _read_g = None
    _read_sigma = None
    _write_g = None
    _write_sigma = None
    _torch_read_cache = {}
    _lock = threading.Lock()

    @classmethod
    def _ensure_loaded(cls):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            from .sys_utils import load_physics_table, PhysicsTag
            try:
                d2d_data = load_physics_table(PhysicsTag.D2D_MISMATCH)
                read_data = load_physics_table(PhysicsTag.READ_NOISE)
                write_data = load_physics_table(PhysicsTag.WRITE_NOISE)
                cls._d2d_u, cls._d2d_dev = d2d_data[:, 0], d2d_data[:, 1]
                cls._read_g, cls._read_sigma = read_data[:, 0], read_data[:, 1]
                cls._write_g, cls._write_sigma = write_data[:, 0], write_data[:, 1]
            except Exception:
                d2d_path = _find_pwl_file("d2d_mismatch_quantile.pwl")
                read_path = _find_pwl_file("read_noise.pwl")
                write_path = _find_pwl_file("write_noise.pwl")

                cls._d2d_u, cls._d2d_dev = _load_pwl(d2d_path)
                cls._read_g, cls._read_sigma = _load_pwl(read_path)
                cls._write_g, cls._write_sigma = _load_pwl(write_path)
            cls._initialized = True

    @classmethod
    def sample_d2d_factors(cls, shape, intensity=0.0):
        """
        Samples relative device parameter perturbations from empirical inverse-CDF quantile table.
        Returns array of shape `shape` with relative perturbation values (intensity * delta).
        New parameter P = P_nom * (1.0 + perturbation).
        """
        if intensity <= 0.0:
            return np.zeros(shape, dtype=np.float64)
        cls._ensure_loaded()
        u = np.random.uniform(0.0, 1.0, size=shape)
        delta = np.interp(u, cls._d2d_u, cls._d2d_dev)
        return float(intensity) * delta

    @classmethod
    def inject_read_noise_cpu(cls, i_analog, v_read, g_min, g_max, intensity=0.0):
        """
        Injects state-dependent read noise into analog currents (NumPy array).
        Captures thermal floor at low conductance and flicker noise scaling at high conductance.
        """
        if intensity <= 0.0:
            return i_analog
        cls._ensure_loaded()
        i_arr = np.asarray(i_analog, dtype=np.float64)
        v_eff = max(1e-12, abs(float(v_read)))
        g_cell = np.abs(i_arr) / v_eff
        g_range = max(1e-18, float(g_max - g_min))
        g_norm = np.clip((g_cell - float(g_min)) / g_range, 0.0, 1.0)
        rel_sigma = float(intensity) * np.interp(g_norm, cls._read_g, cls._read_sigma)
        g_floor = max(float(g_min), 1e-6)
        i_scale = np.maximum(np.abs(i_arr), v_eff * g_floor)
        sigma_i = rel_sigma * i_scale
        noise = np.random.normal(0.0, sigma_i, size=i_arr.shape)
        return i_arr + noise

    @classmethod
    def inject_read_noise_torch(cls, i_cell_tensor, v_read, g_min, g_max, intensity=0.0):
        """
        Injects state-dependent read noise into PyTorch tensor on GPU/CPU without host transfers.
        """
        if intensity <= 0.0:
            return i_cell_tensor
        cls._ensure_loaded()
        import torch

        device = i_cell_tensor.device
        dtype = i_cell_tensor.dtype
        cache_key = (device, dtype)

        if cache_key not in cls._torch_read_cache:
            with cls._lock:
                if cache_key not in cls._torch_read_cache:
                    g_t = torch.from_numpy(cls._read_g).to(device=device, dtype=dtype)
                    s_t = torch.from_numpy(cls._read_sigma).to(device=device, dtype=dtype)
                    cls._torch_read_cache[cache_key] = (g_t, s_t)
        g_t, s_t = cls._torch_read_cache[cache_key]

        v_eff = max(1e-12, abs(float(v_read)))
        g_cell = torch.abs(i_cell_tensor) / v_eff
        g_range = max(1e-18, float(g_max - g_min))
        g_norm = torch.clamp((g_cell - float(g_min)) / g_range, 0.0, 1.0)

        # 1D Piecewise-Linear Interpolation in PyTorch
        idx = torch.bucketize(g_norm, g_t)
        idx_lo = torch.clamp(idx - 1, 0, len(g_t) - 2)
        idx_hi = idx_lo + 1
        x_lo = g_t[idx_lo]
        x_hi = g_t[idx_hi]
        f_lo = s_t[idx_lo]
        f_hi = s_t[idx_hi]
        weight = (g_norm - x_lo) / torch.clamp(x_hi - x_lo, min=1e-15)
        rel_sigma = torch.clamp(f_lo + weight * (f_hi - f_lo), min=0.0)

        g_floor = max(float(g_min), 1e-6)
        i_floor_t = torch.tensor(v_eff * g_floor, device=device, dtype=dtype)
        i_scale = torch.maximum(torch.abs(i_cell_tensor), i_floor_t)
        sigma_i = (float(intensity) * rel_sigma) * i_scale
        noise = torch.randn_like(i_cell_tensor) * sigma_i
        return i_cell_tensor + noise

    @classmethod
    def inject_write_noise(cls, delta_state, state_norm, intensity=0.0):
        """
        Injects cycle-to-cycle Langevin write stochasticity into state changes.
        """
        if intensity <= 0.0:
            return delta_state
        cls._ensure_loaded()
        s_norm_arr = np.clip(np.asarray(state_norm, dtype=np.float64), 0.0, 1.0)
        rel_sigma = float(intensity) * np.interp(s_norm_arr, cls._write_g, cls._write_sigma)
        noise = np.random.normal(0.0, rel_sigma, size=np.shape(delta_state))
        return delta_state * (1.0 + noise)
