import numpy as np
import os
import shutil
import weakref
import sys
import math

from .sys_utils import DummyStream, is_worker_process

if sys.stdout is None:
    sys.stdout = DummyStream()
if getattr(sys, '__stdout__', None) is None:
    sys.__stdout__ = DummyStream()

import time
import concurrent.futures
import multiprocessing
import threading

_jit_pulse_predict_lock = threading.Lock()

is_mp_child = is_worker_process()
is_coordinator = not is_mp_child

if is_mp_child:
    class DummyTorch:
        class jit:
            @staticmethod
            def script(fn):
                return fn
    torch = DummyTorch()
else:
    import torch
from .sys_utils import handle_oom, ensure_bootstrap_profiled, log_to_run_log_only, kill_child_processes
from .sources import PwlSource, PulseSource
from .programmer_solver import jit_cpu_device_updates_parallel, jit_predict_pulses_matrix
from .device import MolMemristor

def _generate_pwl_timeline(chunk_pulses_t, chunk_voltages_t, chunk_pwidths_t, pt_idx, sub_pt, k):
    chunk_periods = chunk_pwidths_t * 2.0
    t0 = k * chunk_periods
    t1 = t0 + 1e-11
    t2 = t0 + chunk_pwidths_t
    t3 = t2 + 1e-11
    
    t_candidate = torch.where(sub_pt == 0, t0,
                  torch.where(sub_pt == 1, t1,
                  torch.where(sub_pt == 2, t2, t3)))
    zero_v = torch.zeros(1, dtype=chunk_voltages_t.dtype, device=chunk_voltages_t.device)
    v_candidate = torch.where((sub_pt == 1) | (sub_pt == 2), chunk_voltages_t, zero_v)
    
    mask = pt_idx < (chunk_pulses_t.to(torch.int64) * 4)
    t_final = torch.where(mask, t_candidate, 1e20)
    v_final = torch.where(mask, v_candidate, zero_v)
    return t_final, v_final


def _safe_pin_numpy(arr, dtype=None):
    t = torch.from_numpy(arr) if dtype is None else torch.from_numpy(arr).to(dtype=dtype)
    if torch.cuda.is_available():
        try:
            return t.pin_memory()
        except Exception:
            return t
    return t


def _safe_pinned_empty(shape, dtype, is_cuda):
    if is_cuda:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except Exception:
            return torch.empty(shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype)


def _bg_write_back_device(stacked_host_tensor, chunk_devs, copy_back_event):
    if copy_back_event is not None:
        copy_back_event.synchronize()
    stacked_host = stacked_host_tensor.numpy()
    n_host = stacked_host[0, :, 0]
    modeState_host = stacked_host[1, :, 0]
    scaleFactor_host = stacked_host[2, :, 0]
    f22DepScaled_host = stacked_host[3, :, 0]
    f22PotScaled_host = stacked_host[4, :, 0]
    cDischarge_host = stacked_host[5, :, 0]
    iScale_host = stacked_host[6, :, 0]
    vScale_host = stacked_host[7, :, 0]
    currentF22_host = stacked_host[8, :, 0]
    currentIScale_host = stacked_host[9, :, 0]
    currentVScale_host = stacked_host[10, :, 0]
    T_host = stacked_host[11, :, 0]
    gEq_host = stacked_host[12, :, 0]
    for idx, main_dev in enumerate(chunk_devs):
        main_dev._n = n_host[idx]
        main_dev._modeState = modeState_host[idx]
        main_dev._scaleFactor = scaleFactor_host[idx]
        main_dev._f22DepScaled = f22DepScaled_host[idx]
        main_dev._f22PotScaled = f22PotScaled_host[idx]
        c_val = cDischarge_host[idx]
        main_dev.cDischarge = float(c_val) if math.isfinite(c_val) else getattr(main_dev, 'cDischarge', 0.0)
        main_dev._cached_n_int_val = int(main_dev._n) if (math.isfinite(main_dev._n) and -2147483648 <= main_dev._n <= 2147483647) else -1
        main_dev._currentF22 = currentF22_host[idx]
        main_dev._currentIScale = currentIScale_host[idx]
        main_dev._currentVScale = currentVScale_host[idx]
        main_dev._iScale = iScale_host[idx]
        main_dev._vScale = vScale_host[idx]
        main_dev._T = T_host[idx]
        main_dev._G = gEq_host[idx]


def _bg_write_back_column(stacked_host_tensor, chunk_devs_col, copy_back_event, rows):
    if copy_back_event is not None:
        copy_back_event.synchronize()
    stacked_host = stacked_host_tensor.numpy()
    n_host = stacked_host[0].ravel()
    modeState_host = stacked_host[1].ravel()
    scaleFactor_host = stacked_host[2].ravel()
    f22DepScaled_host = stacked_host[3].ravel()
    f22PotScaled_host = stacked_host[4].ravel()
    cDischarge_host = stacked_host[5].ravel()
    iScale_host = stacked_host[6].ravel()
    vScale_host = stacked_host[7].ravel()
    currentF22_host = stacked_host[8].ravel()
    currentIScale_host = stacked_host[9].ravel()
    currentVScale_host = stacked_host[10].ravel()
    T_host = stacked_host[11].ravel()
    gEq_host = stacked_host[12].ravel()
    
    for idx, main_dev in enumerate(chunk_devs_col):
        main_dev._n = n_host[idx]
        main_dev._modeState = modeState_host[idx]
        main_dev._scaleFactor = scaleFactor_host[idx]
        main_dev._f22DepScaled = f22DepScaled_host[idx]
        main_dev._f22PotScaled = f22PotScaled_host[idx]
        c_val = cDischarge_host[idx]
        main_dev.cDischarge = float(c_val) if math.isfinite(c_val) else getattr(main_dev, 'cDischarge', 0.0)
        main_dev._cached_n_int_val = int(main_dev._n) if (math.isfinite(main_dev._n) and -2147483648 <= main_dev._n <= 2147483647) else -1
        main_dev._currentF22 = currentF22_host[idx]
        main_dev._currentIScale = currentIScale_host[idx]
        main_dev._currentVScale = currentVScale_host[idx]
        main_dev._iScale = iScale_host[idx]
        main_dev._vScale = vScale_host[idx]
        main_dev._T = T_host[idx]
        main_dev._G = gEq_host[idx]


def _init_crossover_worker(*args, **kwargs):
    import os
    import sys
    import re
    import multiprocessing
    
    os.environ['MOLMEM_IS_CALIBRATION_WORKER'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HIP_VISIBLE_DEVICES'] = ''
    os.environ['MOLMEM_CPU_ONLY'] = '1'
    os.environ['MOLMEM_FORCE_GPU'] = '0'
    
    # Enforce strict single-thread environment limits for nested workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    try:
        import numba.core.config as nb_config
        nb_config.NUMBA_NUM_THREADS = 1
    except Exception:
        pass
    try:
        import numba as nb
        nb.set_num_threads(1)
    except Exception:
        pass
        
    try:
        import psutil
        
        worker_seq_idx = None
        if len(args) >= 2:
            counter = args[0]
            lock = args[1]
            if hasattr(counter, 'value') and hasattr(lock, 'acquire'):
                with lock:
                    worker_seq_idx = counter.value
                    counter.value += 1
                    
        if worker_seq_idx is not None:
            idx = worker_seq_idx
        else:
            proc_name = multiprocessing.current_process().name
            match = re.search(r'\d+$', proc_name)
            if match:
                idx = int(match.group(0))
            else:
                idx = os.getpid()
            
        affinity_str = "OS Managed"
    except Exception:
        affinity_str = "OS Managed"
        idx = -1
        
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"Worker {idx} initialized: Platform=CPU (Numba calibration worker) | Threads=1 | Core Affinity={affinity_str}")
        
        import multiprocessing.util as mp_util
        def _exit_worker():
            try:
                log_to_run_log_only(f"Worker {idx} terminated: Platform=CPU (Numba calibration worker)")
            except Exception:
                pass
        mp_util.Finalize(None, _exit_worker, exitpriority=0)
    except Exception:
        pass


class CrossbarProgrammer:
    """
    Automates the complex pulse-train scheduling required to program 
    a physical crossbar matrix to arbitrary target states. Includes physically
    accurate V/2 Inhibit Schemes.
    """
    _cached_data = None  # Class-level cache to share parsed npz data across programmer instances in the same process
    _resolved_caches = {}
    
    
    @property
    def sim(self):
        sim_obj = self._sim_ref()
        if sim_obj is None:
            raise ReferenceError("Simulator object has been garbage collected.")
        return sim_obj

    def __init__(self, simulator, vWrite=0.9, vErase=-0.75, pulseWidthPot=80e-9, pulseWidthDep=60e-9, period=160e-9, max_vWrite=5.0, max_pw=320e-9):
        self._sim_ref = weakref.ref(simulator)
        self.vWrite = vWrite
        self.vErase = vErase
        self.pulseWidthPot = pulseWidthPot
        self.pulseWidthDep = pulseWidthDep
        self.period = period
        self.max_vWrite = max_vWrite
        self.max_pw = max_pw
        
        self.stateMapPot = None  # Lookup table for Potentiation (Write) pulses -> state n
        self.stateMapDep = None  # Lookup table for Depression (Erase) pulses -> state n
        self.gMapN = None        # Pre-calculated N states for conductance mapping
        self.gMapG = None        # Lookup table for N -> Conductance G at 0.5V
        self._dev_cache = None
        self._dev_cache_key = None
        
    def _get_device_lists(self, rows, cols):
        cache_key = (rows, cols)
        if not hasattr(self, '_dev_cache') or self._dev_cache is None or getattr(self, '_dev_cache_key', None) != cache_key:
            flat_devs_device = []
            dev_grid = np.empty((rows, cols), dtype=object)
            sim_devices = self.sim.devices
            cb_devs = self.sim.crossbarDevices
            for r in range(rows):
                for c in range(cols):
                    dev_idx = cb_devs[(r + 1, c + 1)]
                    dev = sim_devices[dev_idx]['dev']
                    flat_devs_device.append(dev)
                    dev_grid[r, c] = dev
            
            self._dev_cache = {
                'flat_device': flat_devs_device,
                'grid': dev_grid
            }
            self._dev_cache_key = cache_key
        return self._dev_cache

    @handle_oom
    def calibrate(self, maxPulses=None, maxLinearState=None):
        """
        Runs an isolated, ultra-fast calibration simulation to build a mapping
        of precisely how many pulses yield a target memristor state (n).
        """
        if getattr(self.sim, '_is_noop', False):
            return

        verbose = getattr(self.sim, 'detailedPrint', True)
        
        # Get first device configuration params to validate cache validity
        import json, os, sys
        first_dev_params = {}
        dev_nmax = 33040.0
        if len(self.sim.devices) > 0:
            dev_entry = self.sim.devices[0]
            dev_obj = dev_entry.get('dev') if isinstance(dev_entry, dict) else getattr(dev_entry, 'dev', None)
            params_candidate = getattr(dev_obj, 'params', {}) if dev_obj is not None else {}
            if isinstance(params_candidate, dict):
                first_dev_params = params_candidate
            if dev_obj is not None and hasattr(dev_obj, 'nMax'):
                dev_nmax = float(dev_obj.nMax)
        try:
            params_str = json.dumps(first_dev_params, sort_keys=True)
        except Exception:
            params_str = "{}"
            
        default_linear_limit = getattr(getattr(self.sim, 'wq', None), 'maxStates', None)
        if default_linear_limit is None:
            default_linear_limit = float(getattr(dev_obj, 'nLinearMax', 0.5 * dev_nmax))

        if maxLinearState is None:
            maxLinearState = float(default_linear_limit)
        if maxPulses is None:
            maxPulses = int(default_linear_limit)
        self.maxPulses = maxPulses
        self.maxLinearState = maxLinearState
        
        # Check resolved memory cache first
        cache_key = (self.vWrite, self.vErase, self.pulseWidthPot, self.pulseWidthDep, self.period, self.max_vWrite, self.max_pw, maxPulses, maxLinearState, self.sim.vRead, params_str)
        if cache_key in CrossbarProgrammer._resolved_caches:
            # Move to end to track LRU order in dict
            cached = CrossbarProgrammer._resolved_caches.pop(cache_key)
            CrossbarProgrammer._resolved_caches[cache_key] = cached
            self.stateMapPot = cached['stateMapPot']
            self.stateMapDep = cached['stateMapDep']
            self.gMapN = cached['gMapN']
            self.gMapG = cached['gMapG']
            self.vWriteRange = cached['vWriteRange']
            self.pwRange = cached['pwRange']
            self.optMapPot = cached['optMapPot']
            self.vEraseRange = cached['vEraseRange']
            self.optMapDep = cached['optMapDep']
            self.minPulsesMapPot_coarse = cached['minPulsesMapPot_coarse']
            self.minPulsesMapPot_fine = cached['minPulsesMapPot_fine']
            self.minPulsesMapDep_coarse = cached['minPulsesMapDep_coarse']
            self.minPulsesMapDep_fine = cached['minPulsesMapDep_fine']
            self.maxPulses = int(cached.get('maxPulses', maxPulses))
            self.maxLinearState = float(cached.get('maxLinearState', maxLinearState))
            
            if hasattr(self.sim, 'wq'):
                self.sim.wq.minStates = 0.0
                self.sim.wq.maxStates = float(self.maxLinearState)
            self.sim.isCalibrated = True
            size_key = f"{len(self.sim.wordlines)}_{len(self.sim.bitlines)}"
            if verbose and not getattr(self.sim, '_is_calibration_helper', False) and os.environ.get(f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}') != '1':
                print(f"    >> Hardware Calibration: [{'#'*20}] 100% | Loaded Cache: {self.max_vWrite:.2f}V, {self.max_pw*1e9:.0f}ns")
                os.environ[f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}'] = '1'
            return
            
        cache_dir = os.path.join(os.path.dirname(__file__), '.calcache')
        
        # 1. Try class-level memory cache first
        data = None
        if CrossbarProgrammer._cached_data is not None:
            data = CrossbarProgrammer._cached_data
            
        # 2. Try loading from disk cache with retries if not in memory cache
        if data is None:
            from . import _load_unified_cache
            data = _load_unified_cache()
            if data:
                # Keep class cache in sync
                CrossbarProgrammer._cached_data = data

        if data is not None:
            cache_valid = True
            for param_key, current_val in (
                ('vWrite', self.vWrite),
                ('vErase', self.vErase),
                ('pulseWidthPot', self.pulseWidthPot),
                ('pulseWidthDep', self.pulseWidthDep),
                ('period', self.period),
                ('max_vWrite', self.max_vWrite),
                ('max_pw', self.max_pw),
                ('maxPulses', maxPulses),
                ('maxLinearState', maxLinearState),
                ('vRead', self.sim.vRead)
            ):
                if param_key not in data:
                    cache_valid = False
                    break
                cached_val = float(data[param_key][0])
                if abs(cached_val - current_val) > 1e-5 * (abs(current_val) if abs(current_val) > 1e-12 else 1e-12):
                    cache_valid = False
                    break
                    
            from . import _compute_library_footprint
            if 'lib_footprint' not in data or data['lib_footprint'][0] != _compute_library_footprint():
                cache_valid = False
                
            if 'device_params' not in data or data['device_params'][0] != params_str:
                cache_valid = False
                    
            if not cache_valid:
                if os.environ.get('MOLMEM_SKIP_FULL_CALIBRATION') == '1':
                    self.vWriteRange = np.linspace(0.5, self.max_vWrite, 10)
                    self.pwRange = np.linspace(10e-9, self.max_pw, 10)
                    self.vEraseRange = -self.vWriteRange
                    self.optMapPot = np.zeros((10, 10))
                    self.optMapDep = np.zeros((10, 10))
                    self.stateMapPot = np.linspace(0, 33200, 100)
                    self.stateMapDep = np.linspace(0, 33200, 100)
                    self.gMapN = np.linspace(0, 33200, 100)
                    self.gMapG = np.linspace(1e-6, 1e-3, 100)
                    self.minPulsesMapPot_coarse = np.ones(100, dtype=int)
                    self.minPulsesMapPot_fine = np.ones(100, dtype=int)
                    self.minPulsesMapDep_coarse = np.ones(100, dtype=int)
                    self.minPulsesMapDep_fine = np.ones(100, dtype=int)
                    self.sim.isCalibrated = True
                    return
                if verbose: print("\n  > Cache footprint mismatch or outdated profile. Recalibrating hardware parameters...")
            else:
                self.stateMapPot = data['stateMapPot']
                self.stateMapDep = data['stateMapDep']
                self.gMapN = data['gMapN']
                self.gMapG = data['gMapG']
                if 'maxLinearState' in data:
                    self.maxLinearState = float(data['maxLinearState'][0])
                # Check grid bounds for incremental loading
                incremental_update = False
                if 'optMapPot' in data:
                    old_vWriteRange = data['vWriteRange']
                    old_pwRange = data['pwRange']
                    old_optMapPot = data['optMapPot']
                    old_optMapDep = data['optMapDep']
                    old_max_vWrite = data['max_vWrite'][0] if 'max_vWrite' in data else self.max_vWrite
                    old_max_pw = data['max_pw'][0] if 'max_pw' in data else self.max_pw
                    
                    if 'max_vWrite' not in data or 'max_pw' not in data:
                        incremental_update = True
                    elif old_max_vWrite < self.max_vWrite - 1e-5 or old_max_pw < self.max_pw - 1e-5:
                        incremental_update = True
                        
                    if not incremental_update:
                        size_key = f"{len(self.sim.wordlines)}_{len(self.sim.bitlines)}"
                        if old_max_vWrite > self.max_vWrite + 1e-5 or old_max_pw > self.max_pw + 1e-5:
                            if verbose and os.environ.get(f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}') != '1':
                                print(f"    >> Hardware Calibration: [{'#'*20}] 100% | Extracted Sub-Cache: {self.max_vWrite:.2f}V, {self.max_pw*1e9:.0f}ns")
                                os.environ[f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}'] = '1'
                            
                            v_val = old_vWriteRange <= self.max_vWrite + 1e-4
                            pw_val = old_pwRange <= self.max_pw + 1e-12
                            
                            self.vWriteRange = old_vWriteRange[v_val]
                            self.pwRange = old_pwRange[pw_val]
                            
                            ix = np.ix_(v_val, pw_val)
                            self.optMapPot = old_optMapPot[ix]
                            self.vEraseRange = data['vEraseRange'][v_val]
                            self.optMapDep = old_optMapDep[ix]
                            
                            len_pot = len(self.stateMapPot)
                            shifts_pot = np.linspace(0.0, 1.0, len_pot)
                            len_dep = len(self.stateMapDep)
                            shifts_dep = np.linspace(0.0, 1.0, len_dep)
                            try:
                                from .programmer_solver import jit_compute_min_pulses
                                self.minPulsesMapPot_coarse, self.minPulsesMapPot_fine = jit_compute_min_pulses(
                                    self.optMapPot, shifts_pot, coarse_tol=0.005, fine_tol=0.0001
                                )
                                self.minPulsesMapDep_coarse, self.minPulsesMapDep_fine = jit_compute_min_pulses(
                                    self.optMapDep, shifts_dep, coarse_tol=0.005, fine_tol=0.0001
                                )
                            except Exception:
                                scaled_shifts_pot = np.ceil(shifts_pot * len_pot).astype(int, copy=False)
                                shifts_pot_3d = shifts_pot[:, None, None]
                                p_pot = np.round(self.optMapPot[None, :, :] * shifts_pot_3d)
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    err_pot = np.abs(p_pot / self.optMapPot[None, :, :] - shifts_pot_3d)
                                
                                c_p_c = np.min(np.where((self.optMapPot[None, :, :] > 0) & (err_pot < 0.005) & (p_pot > 0), p_pot, np.inf), axis=(1, 2))
                                inf_p_c = np.isinf(c_p_c)
                                c_p_c[inf_p_c] = scaled_shifts_pot[inf_p_c]
                                self.minPulsesMapPot_coarse = c_p_c
                                
                                c_p_f = np.min(np.where((self.optMapPot[None, :, :] > 0) & (err_pot < 0.0001) & (p_pot > 0), p_pot, np.inf), axis=(1, 2))
                                inf_p_f = np.isinf(c_p_f)
                                c_p_f[inf_p_f] = scaled_shifts_pot[inf_p_f]
                                self.minPulsesMapPot_fine = c_p_f

                                scaled_shifts_dep = np.ceil(shifts_dep * len_dep).astype(int, copy=False)
                                shifts_dep_3d = shifts_dep[:, None, None]
                                p_dep = np.round(self.optMapDep[None, :, :] * shifts_dep_3d)
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    err_dep = np.abs(p_dep / self.optMapDep[None, :, :] - shifts_dep_3d)
                                    
                                c_d_c = np.min(np.where((self.optMapDep[None, :, :] > 0) & (err_dep < 0.005) & (p_dep > 0), p_dep, np.inf), axis=(1, 2))
                                inf_d_c = np.isinf(c_d_c)
                                c_d_c[inf_d_c] = scaled_shifts_dep[inf_d_c]
                                self.minPulsesMapDep_coarse = c_d_c
                                
                                c_d_f = np.min(np.where((self.optMapDep[None, :, :] > 0) & (err_dep < 0.0001) & (p_dep > 0), p_dep, np.inf), axis=(1, 2))
                                inf_d_f = np.isinf(c_d_f)
                                c_d_f[inf_d_f] = scaled_shifts_dep[inf_d_f]
                                self.minPulsesMapDep_fine = c_d_f
                            
                        else:
                            v_val = old_vWriteRange <= self.max_vWrite + 1e-4
                            pw_val = old_pwRange <= self.max_pw + 1e-12
                            self.vWriteRange = old_vWriteRange[v_val]
                            self.pwRange = old_pwRange[pw_val]
                            ix = np.ix_(v_val, pw_val)
                            self.optMapPot = old_optMapPot[ix]
                            self.vEraseRange = data['vEraseRange'][v_val]
                            self.optMapDep = old_optMapDep[ix]
                            
                            if (len(self.pwRange) != len(old_pwRange) or len(self.vWriteRange) != len(old_vWriteRange)):
                                len_pot = len(self.stateMapPot)
                                shifts_pot = np.linspace(0.0, 1.0, len_pot)
                                len_dep = len(self.stateMapDep)
                                shifts_dep = np.linspace(0.0, 1.0, len_dep)
                                try:
                                    from .programmer_solver import jit_compute_min_pulses
                                    self.minPulsesMapPot_coarse, self.minPulsesMapPot_fine = jit_compute_min_pulses(
                                        self.optMapPot, shifts_pot, coarse_tol=0.005, fine_tol=0.0001
                                    )
                                    self.minPulsesMapDep_coarse, self.minPulsesMapDep_fine = jit_compute_min_pulses(
                                        self.optMapDep, shifts_dep, coarse_tol=0.005, fine_tol=0.0001
                                    )
                                except Exception:
                                    pass
                            elif 'minPulsesMapPot_coarse' in data:
                                self.minPulsesMapPot_coarse = data['minPulsesMapPot_coarse']
                                self.minPulsesMapDep_coarse = data['minPulsesMapDep_coarse']
                                self.minPulsesMapPot_fine = data['minPulsesMapPot_fine']
                                self.minPulsesMapDep_fine = data['minPulsesMapDep_fine']
                                
                            if verbose and os.environ.get(f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}') != '1':
                                print(f"    >> Hardware Calibration: [{'#'*20}] 100% | Loaded Cache: {self.max_vWrite:.2f}V, {self.max_pw*1e9:.0f}ns")
                                os.environ[f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}'] = '1'
                            
                        if hasattr(self.sim, 'wq'):
                            self.sim.wq.minStates = 0.0
                            self.sim.wq.maxStates = float(self.maxLinearState)
                            
                        # Store in class resolved caches (evict oldest if size >= 16)
                        if len(CrossbarProgrammer._resolved_caches) >= 16:
                            oldest = next(iter(CrossbarProgrammer._resolved_caches))
                            CrossbarProgrammer._resolved_caches.pop(oldest, None)
                        CrossbarProgrammer._resolved_caches[cache_key] = {
                            'stateMapPot': self.stateMapPot,
                            'stateMapDep': self.stateMapDep,
                            'gMapN': self.gMapN,
                            'gMapG': self.gMapG,
                            'vWriteRange': self.vWriteRange,
                            'pwRange': self.pwRange,
                            'optMapPot': self.optMapPot,
                            'vEraseRange': self.vEraseRange,
                            'optMapDep': self.optMapDep,
                            'minPulsesMapPot_coarse': self.minPulsesMapPot_coarse,
                            'minPulsesMapPot_fine': self.minPulsesMapPot_fine,
                            'minPulsesMapDep_coarse': self.minPulsesMapDep_coarse,
                            'minPulsesMapDep_fine': self.minPulsesMapDep_fine,
                            'maxPulses': self.maxPulses,
                            'maxLinearState': self.maxLinearState,
                        }
                        self.sim.isCalibrated = True
                        return
                    else:
                        if self.sim.detailedPrint:
                            print(f"  > Merging cached {old_max_vWrite:.2f}V / {old_max_pw*1e9:.0f}ns calibration map into current grid...")
                        else:
                            print(f"  > Merging cached {old_max_vWrite:.2f}V / {old_max_pw*1e9:.0f}ns calibration map into current grid...")
                else:
                    size_key = f"{len(self.sim.wordlines)}_{len(self.sim.bitlines)}"
                    if verbose and os.environ.get(f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}') != '1':
                        print(f"    >> Hardware Calibration: [{'#'*20}] 100% | Loaded Cache: {self.max_vWrite:.2f}V, {self.max_pw*1e9:.0f}ns")
                        os.environ[f'MOLMEM_CAL_PROF_MSG_SHOWN_{size_key}'] = '1'
                    if verbose: print("  > Cache missing 3D Pulse Map. Rebuilding calibration...")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        from .simulator import MolmemSimulator
        calSim = MolmemSimulator(device='cpu', backend='numba')
        calSim._is_calibration_helper = True
        calSim.optimal_cores = 1
        calSim.addCrossbarMatrix(device_type=first_dev_params, rows=1, cols=1, detailedPrint=False)
        
        cal_period = max(self.period, self.max_pw * 2.0)
        pulseWrite = PulseSource(node=None, amp=self.max_vWrite, period=cal_period, width=self.max_pw)
        calSim.addVoltageSource(row=1, sourceObj=pulseWrite)
        calSim.addDcSource(col=1, voltage=0.0)
        
        tEnd = maxPulses * cal_period
        calSim.run(tEnd=tEnd, min_dt=1e-12, tol=1e-3, title="Write Calibration Sweep", recordHistory=['state'])
        tHist = calSim.tArr
        nHist = calSim.stateHistory[0]
        
        dense_samples_count = max(20000, maxPulses)
        if len(tHist) > 1:
            tSamples = np.linspace(float(tHist[0]), float(tHist[-1]), dense_samples_count)
            idx_pot = np.clip(np.searchsorted(tHist, tSamples), 0, len(nHist) - 1)
            self.stateMapPot = np.clip(nHist[idx_pot], 0, maxLinearState)
        else:
            self.stateMapPot = np.linspace(0.0, maxLinearState, dense_samples_count)
        
        calSim.devices[0]['dev']._n = float(self.stateMapPot.max())
        calSim.devices[0]['dev']._modeState = 0.0
        calSim.devices[0]['dev']._T = 298.15
        calSim.clearVoltageSources()
        pulseErase = PulseSource(node=None, amp=-self.max_vWrite, period=cal_period, width=self.max_pw)
        calSim.addVoltageSource(row=1, sourceObj=pulseErase)
        calSim.addDcSource(col=1, voltage=0.0)
        calSim.tArr = None 
        calSim.run(tEnd=tEnd, min_dt=1e-12, tol=1e-3, title="Erase Calibration Sweep", recordHistory=['state'])
        tHistErase = calSim.tArr
        nHistErase = calSim.stateHistory[0]
        
        if len(tHistErase) > 1:
            tSamplesErase = np.linspace(float(tHistErase[0]), float(tHistErase[-1]), dense_samples_count)
            idx_dep = np.clip(np.searchsorted(tHistErase, tSamplesErase), 0, len(nHistErase) - 1)
            self.stateMapDep = np.clip(nHistErase[idx_dep], 0, maxLinearState)
        else:
            self.stateMapDep = np.linspace(maxLinearState, 0.0, dense_samples_count)
        
        # Dynamically guarantee mapping resolution matches or exceeds hardware bit precision
        map_points = max(20000, maxPulses)
        if hasattr(self.sim, 'wq') and hasattr(self.sim.wq, 'weightBits'):
            map_points = max(map_points, (1 << self.sim.wq.weightBits) + 1000)
            
        dev = calSim.devices[0]['dev']
        max_cal_n = float(getattr(dev, 'nMax', 33040.0))
        self.gMapN = np.linspace(0.0, max_cal_n, map_points)
        v_cal_read = self.sim.vRead
        if abs(v_cal_read) < 1e-12:
            v_cal_read = 0.5
        try:
            from .programmer_solver import jit_compute_conductance_map
            self.gMapG = jit_compute_conductance_map(
                self.gMapN, float(v_cal_read), float(dev.nMax), float(dev.gScale), 
                float(dev.gamma), float(dev.rC),
                dev._y_i, dev._y_v, dev._y_fp, float(dev.f22StartVal), float(dev.f22PotScaleInv)
            )
        except Exception:
            inv_v_cal_read = 1.0 / v_cal_read
            self.gMapG = np.zeros(map_points)
            for i, n_val in enumerate(self.gMapN):
                dev._n = n_val
                dev._forceUpdateConductanceParams()
                currentI, _ = dev.getIGeq(v_cal_read)
                self.gMapG[i] = currentI * inv_v_cal_read
            
        try:
            calSim.clearHistory(force_gc=False)
        except Exception:
            pass
            
        if verbose: 
            print(f"\n  > Calibrating 3D Optimal Pulse Map for Crossbar Architecture...")
            print(f"    >> Voltage Limit: {self.max_vWrite:.2f}V | Pulse-Width Limit: {self.max_pw*1e9:.0f}ns")
            
        self.vWriteRange = np.arange(0.7, self.max_vWrite + 0.025, 0.05)
        self.pwRange = np.arange(40e-9, self.max_pw + 2.5e-9, 5e-9)
        
        self.optMapPot = np.zeros((len(self.vWriteRange), len(self.pwRange)), dtype=np.float64)
        self.vEraseRange = -self.vWriteRange
        self.optMapDep = np.zeros((len(self.vEraseRange), len(self.pwRange)), dtype=np.float64)
        
        # --- SMART RAM CACHE MAPPING ---
        if 'old_optMapPot' in locals():
            v_match = np.isclose(old_vWriteRange[:, None], self.vWriteRange[None, :], atol=1e-3)
            pw_match = np.isclose(old_pwRange[:, None], self.pwRange[None, :], atol=1e-9)
            
            v_old_idx, v_new_idx = np.where(v_match)
            pw_old_idx, pw_new_idx = np.where(pw_match)
            
            old_ix_mesh = np.ix_(v_old_idx, pw_old_idx)
            new_ix_mesh = np.ix_(v_new_idx, pw_new_idx)
            
            self.optMapPot[new_ix_mesh] = old_optMapPot[old_ix_mesh]
            self.optMapDep[new_ix_mesh] = old_optMapDep[old_ix_mesh]
        
        targetMaxN = float(self.maxLinearState)
        targetMinN = 0.0
        chunk_size = 500
        
        old_stdout = sys.stdout
        
        total_traces = len(self.vWriteRange) * len(self.pwRange)
        trace_count = 0
        
        try:
            optMaxPulses = maxPulses
            pot_zeros_v, _ = np.where(self.optMapPot == 0)
            dep_zeros_v, _ = np.where(self.optMapDep == 0)
            num_uncalibrated = len(pot_zeros_v) + len(dep_zeros_v)
                        
            if num_uncalibrated == 0:
                in_crossover = getattr(self.sim.__class__, '_in_gpu_crossover_profiling', False)
                if in_crossover:
                    pass
                else:
                    if verbose:
                        print(f"    >> Hardware Calibration: [{'#'*20}] 100% | Loaded Cache: {self.max_vWrite:.2f}V, {self.max_pw*1e9:.0f}ns")
            else:
                cols = shutil.get_terminal_size().columns
                width = max(20, min(150, cols - 3))
                in_warmup = getattr(self.sim, '_in_warmup', False) or getattr(self.sim.__class__, '_in_warmup', False)
                cal_t0 = time.time()
                
                def write_cal_status(msg):
                    if os.environ.get('MOLMEM_IN_SUBPROCESS') == '1' or os.environ.get('MOLMEM_SILENT_WARMUP') == '1':
                        return
                    if len(msg) > width:
                        msg = msg[:width-3] + "..."
                    ending = "\n" if "Done" in msg else ""
                    msg_text = msg.rstrip("\r\n")
                    sys.stdout.write(f"\r{msg_text}\033[K{ending}")
                    sys.stdout.flush()
                    
                in_crossover = getattr(self.sim.__class__, '_in_gpu_crossover_profiling', False)
                if in_warmup:
                    pass
                elif in_crossover:
                    current_idx = getattr(self.sim.__class__, '_crossover_current_idx', 0)
                    current_sz = getattr(self.sim.__class__, '_crossover_current_sz', 2)
                    crossover_sizes = getattr(self.sim.__class__, '_crossover_sizes', [2, 4, 8, 16, 32, 64, 128])
                    N = len(crossover_sizes)
                    progress_base = (current_idx / N) * 100.0
                    progress = progress_base + (0.1 * (100.0 / N))
                    basis = getattr(self.sim, 'UpdatesPer', 'Column')
                    status_str = f"Calibrating ({basis}) (Size {current_sz}x{current_sz})"
                    subline = f"    [-] Calibration: Dispatching {num_uncalibrated} tasks..."
                    self.sim.__class__._update_crossover_display(progress, status_str, active_subline=subline)
                else:
                    pass
                
                try:
                    from .programmer_solver import jit_calibrate_grid_cpu
                    dev = calSim.devices[0]['dev']
                    optMaxPulses = maxPulses
                    
                    try:
                        import numba as nb
                        import os
                        nb.set_num_threads(os.cpu_count() or 32)
                    except Exception:
                        pass
                    
                    try:
                        from .sys_utils import log_to_run_log_only
                        num_threads = os.cpu_count() or 32
                        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Spawning Hardware Calibration (OpenMP/TBB Thread Pool): MOLMEM_HardwareCalibrator | Platform=CPU (3D Pulse Map Grid Sweep) | Threads={num_threads} | Core Affinity=All Logical Cores")
                    except Exception:
                        pass
                    
                    # Potentiation Sweep (CPU Multi-threaded SIMD across all cores)
                    self.optMapPot = jit_calibrate_grid_cpu(
                        self.vWriteRange, self.pwRange, True, optMaxPulses,
                        targetMaxN, targetMinN, 0.0,
                        dev.vSmooth, dev.nSmooth, dev.kappa, dev.alpha,
                        dev.vTh, dev.vBypass, dev.vRefPos, dev.vRefNeg, dev.nMax, dev.kDischarge,
                        dev.beta_alpha, dev.beta_s,
                        dev.gamma, dev._vScale, dev.rateAsymmetry,
                        getattr(dev, 'rTopBlend', 0.050000), getattr(dev, 'fDischarge', 0.374848), getattr(dev, 'cDischarge', 0.0),
                        getattr(dev, 'rBottomBlend', 0.714488), getattr(dev, 'rLatency', 0.878506), getattr(dev, 'k_overdrive', 7.296635),
                        getattr(dev, 'rC', 0.008033), getattr(dev, 'E_a', 0.155923), getattr(dev, 'R_th', 52.14), getattr(dev, 'tau_th', 1.278e-07),
                        dev._y_i, dev._y_v, dev._y_fp, dev._y_fd,
                        float(dev.gScale), float(dev.f22StartVal), float(dev.f22PotScaleInv), float(dev.f22DepScaleInv), float(dev.f22EndVal)
                    )
                    
                    # Depression Sweep (CPU Multi-threaded SIMD)
                    self.optMapDep = jit_calibrate_grid_cpu(
                        self.vWriteRange, self.pwRange, False, optMaxPulses,
                        targetMaxN, targetMinN, targetMaxN,
                        dev.vSmooth, dev.nSmooth, dev.kappa, dev.alpha,
                        dev.vTh, dev.vBypass, dev.vRefPos, dev.vRefNeg, dev.nMax, dev.kDischarge,
                        dev.beta_alpha, dev.beta_s,
                        dev.gamma, dev._vScale, dev.rateAsymmetry,
                        getattr(dev, 'rTopBlend', 0.050000), getattr(dev, 'fDischarge', 0.374848), getattr(dev, 'cDischarge', 0.0),
                        getattr(dev, 'rBottomBlend', 0.714488), getattr(dev, 'rLatency', 0.878506), getattr(dev, 'k_overdrive', 7.296635),
                        getattr(dev, 'rC', 0.008033), getattr(dev, 'E_a', 0.155923), getattr(dev, 'R_th', 52.14), getattr(dev, 'tau_th', 1.278e-07),
                        dev._y_i, dev._y_v, dev._y_fp, dev._y_fd,
                        float(dev.gScale), float(dev.f22StartVal), float(dev.f22PotScaleInv), float(dev.f22DepScaleInv), float(dev.f22EndVal)
                    )
                    
                    try:
                        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Hardware Calibration (OpenMP/TBB Thread Pool) terminated: Platform=CPU (3D Pulse Map Grid Sweep) | Threads={num_threads} | Core Affinity=OS Managed")
                    except Exception:
                        pass
                    
                    cal_t1 = time.time()
                    if not in_crossover:
                        bar = '#' * 20
                        write_cal_status(f"\r    >> Hardware Calibration: [{bar}] 100% | Done (in {cal_t1 - cal_t0:.1f}s)\n")
                except KeyboardInterrupt:
                    sys.__stdout__.write("\n[!] Simulation interrupted by user. Terminating process...\n")
                    sys.__stdout__.flush()
                    try:
                        kill_child_processes()
                    except Exception:
                        pass
                    os.kill(os.getpid(), 9)
                    os._exit(1)
                except Exception as ex:
                    from .sys_utils import log_to_run_log_only
                    log_to_run_log_only(f"Option B calibration fallback: {ex}")
                            
                if in_crossover:
                    current_idx = getattr(self.sim.__class__, '_crossover_current_idx', 0)
                    current_sz = getattr(self.sim.__class__, '_crossover_current_sz', 2)
                    crossover_sizes = getattr(self.sim.__class__, '_crossover_sizes', [2, 4, 8, 16, 32, 64, 128])
                    N = max(1, len(crossover_sizes))
                    inv_N = 1.0 / N
                    progress_base = (current_idx * inv_N) * 100.0
                    progress = progress_base + (0.7 * (100.0 * inv_N))
                    basis = getattr(self.sim, 'UpdatesPer', 'Column')
                    status_str = f"Calibrating ({basis}) (Size {current_sz}x{current_sz})"
                    subline = "    [+] Device calibration complete!"
                    self.sim.__class__._update_crossover_display(progress, status_str, active_subline=subline)
            
        finally:
            pass
            
        if hasattr(self.sim, 'wq'):
            self.sim.wq.minStates = float(targetMinN)
            self.sim.wq.maxStates = float(targetMaxN)
            
        if verbose: print(f"    >> Validated calibration mappings. Saving cache to disk...")
        
        # --- MONOTONIC DISK MERGE (Never shrink cache file) ---
        final_vWrite = max(self.max_vWrite, old_max_vWrite) if 'old_max_vWrite' in locals() else self.max_vWrite
        final_pw = max(self.max_pw, old_max_pw) if 'old_max_pw' in locals() else self.max_pw
        
        final_vWriteRange = np.arange(0.7, final_vWrite + 0.025, 0.05)
        final_pwRange = np.arange(40e-9, final_pw + 2.5e-9, 5e-9)
        final_optMapPot = np.zeros((len(final_vWriteRange), len(final_pwRange)))
        final_optMapDep = np.zeros((len(final_vWriteRange), len(final_pwRange)))
        
        if 'old_optMapPot' in locals():
            f_v_match = np.isclose(old_vWriteRange[:, None], final_vWriteRange[None, :], atol=1e-3)
            f_pw_match = np.isclose(old_pwRange[:, None], final_pwRange[None, :], atol=1e-9)
            old_v_idx, fin_v_idx = np.where(f_v_match)
            old_pw_idx, fin_pw_idx = np.where(f_pw_match)
            old_ix = np.ix_(old_v_idx, old_pw_idx)
            fin_ix = np.ix_(fin_v_idx, fin_pw_idx)
            final_optMapPot[fin_ix] = old_optMapPot[old_ix]
            final_optMapDep[fin_ix] = old_optMapDep[old_ix]
            
        n_v_match = np.isclose(self.vWriteRange[:, None], final_vWriteRange[None, :], atol=1e-3)
        n_pw_match = np.isclose(self.pwRange[:, None], final_pwRange[None, :], atol=1e-9)
        new_v_idx, nfin_v_idx = np.where(n_v_match)
        new_pw_idx, nfin_pw_idx = np.where(n_pw_match)
        new_ix = np.ix_(new_v_idx, new_pw_idx)
        nfin_ix = np.ix_(nfin_v_idx, nfin_pw_idx)
        final_optMapPot[nfin_ix] = self.optMapPot[new_ix]
        final_optMapDep[nfin_ix] = self.optMapDep[new_ix]
        
        self.vWriteRange = final_vWriteRange
        self.pwRange = final_pwRange
        self.optMapPot = final_optMapPot
        self.optMapDep = final_optMapDep
        self.vEraseRange = -final_vWriteRange
        self.max_vWrite = final_vWrite
        self.max_pw = final_pw
        
        n_pot = len(self.stateMapPot)
        shifts_pot = np.linspace(0.0, 1.0, n_pot)
        
        n_dep = len(self.stateMapDep)
        shifts_dep = np.linspace(0.0, 1.0, n_dep)
        
        try:
            from .programmer_solver import jit_compute_min_pulses
            self.minPulsesMapPot_coarse, self.minPulsesMapPot_fine = jit_compute_min_pulses(
                final_optMapPot, shifts_pot, coarse_tol=0.005, fine_tol=0.0001
            )
            self.minPulsesMapDep_coarse, self.minPulsesMapDep_fine = jit_compute_min_pulses(
                final_optMapDep, shifts_dep, coarse_tol=0.005, fine_tol=0.0001
            )
        except Exception:
            shift_3d_pot = shifts_pot[:, None, None]
            opt_3d_pot = final_optMapPot[None, :, :]
            p_tgt_pot = np.round(opt_3d_pot * shift_3d_pot)
            with np.errstate(divide='ignore', invalid='ignore'):
                err_pot = np.abs(p_tgt_pot / opt_3d_pot - shift_3d_pot)
                
            cst_pot_coarse = np.where((opt_3d_pot > 0.0) & (err_pot < 0.005) & (p_tgt_pot > 0), p_tgt_pot, np.inf)
            min_pot_c = np.min(cst_pot_coarse, axis=(1, 2))
            msk_pot_c = np.isinf(min_pot_c)
            min_pot_c[msk_pot_c] = np.ceil(shifts_pot[msk_pot_c] * n_pot).astype(np.int32, copy=False)
            self.minPulsesMapPot_coarse = min_pot_c
            
            cst_pot_fine = np.where((opt_3d_pot > 0.0) & (err_pot < 0.0001) & (p_tgt_pot > 0), p_tgt_pot, np.inf)
            min_pot_f = np.min(cst_pot_fine, axis=(1, 2))
            msk_pot_f = np.isinf(min_pot_f)
            min_pot_f[msk_pot_f] = np.ceil(shifts_pot[msk_pot_f] * n_pot).astype(np.int32, copy=False)
            self.minPulsesMapPot_fine = min_pot_f

            shift_3d_dep = shifts_dep[:, None, None]
            opt_3d_dep = final_optMapDep[None, :, :]
            p_tgt_dep = np.round(opt_3d_dep * shift_3d_dep)
            with np.errstate(divide='ignore', invalid='ignore'):
                err_dep = np.abs(p_tgt_dep / opt_3d_dep - shift_3d_dep)
                
            cst_dep_coarse = np.where((opt_3d_dep > 0.0) & (err_dep < 0.005) & (p_tgt_dep > 0), p_tgt_dep, np.inf)
            min_dep_c = np.min(cst_dep_coarse, axis=(1, 2))
            msk_dep_c = np.isinf(min_dep_c)
            min_dep_c[msk_dep_c] = np.ceil(shifts_dep[msk_dep_c] * n_dep).astype(np.int32, copy=False)
            self.minPulsesMapDep_coarse = min_dep_c
            
            cst_dep_fine = np.where((opt_3d_dep > 0.0) & (err_dep < 0.0001) & (p_tgt_dep > 0), p_tgt_dep, np.inf)
            min_dep_f = np.min(cst_dep_fine, axis=(1, 2))
            msk_dep_f = np.isinf(min_dep_f)
            min_dep_f[msk_dep_f] = np.ceil(shifts_dep[msk_dep_f] * n_dep).astype(np.int32, copy=False)
            self.minPulsesMapDep_fine = min_dep_f
        
        from . import _compute_library_footprint
        # Write to the unified cache using read-merge-write
        updates = {
            'lib_footprint': np.array([_compute_library_footprint()]),
            'device_params': np.array([params_str]),
            'stateMapPot': self.stateMapPot, 
            'stateMapDep': self.stateMapDep, 
            'gMapN': self.gMapN, 
            'gMapG': self.gMapG,
            'vWriteRange': final_vWriteRange,
            'pwRange': final_pwRange,
            'optMapPot': final_optMapPot,
            'vEraseRange': -final_vWriteRange,
            'optMapDep': final_optMapDep,
            'minPulsesMapPot_coarse': self.minPulsesMapPot_coarse,
            'minPulsesMapDep_coarse': self.minPulsesMapDep_coarse,
            'minPulsesMapPot_fine': self.minPulsesMapPot_fine,
            'minPulsesMapDep_fine': self.minPulsesMapDep_fine,
            'vWrite': np.array([self.vWrite]),
            'vErase': np.array([self.vErase]),
            'pulseWidthPot': np.array([self.pulseWidthPot]),
            'pulseWidthDep': np.array([self.pulseWidthDep]),
            'period': np.array([self.period]),
            'max_vWrite': np.array([final_vWrite]),
            'max_pw': np.array([final_pw]),
            'maxPulses': np.array([maxPulses]),
            'maxLinearState': np.array([maxLinearState]),
            'vRead': np.array([self.sim.vRead])
        }
        from . import _save_unified_cache
        _save_unified_cache(updates)
        CrossbarProgrammer._cached_data = updates
        
        # Store in class resolved caches (evict oldest if size >= 16)
        if len(CrossbarProgrammer._resolved_caches) >= 16:
            oldest = next(iter(CrossbarProgrammer._resolved_caches))
            CrossbarProgrammer._resolved_caches.pop(oldest, None)
        CrossbarProgrammer._resolved_caches[cache_key] = {
            'stateMapPot': self.stateMapPot,
            'stateMapDep': self.stateMapDep,
            'gMapN': self.gMapN,
            'gMapG': self.gMapG,
            'vWriteRange': self.vWriteRange,
            'pwRange': self.pwRange,
            'optMapPot': self.optMapPot,
            'vEraseRange': self.vEraseRange,
            'optMapDep': self.optMapDep,
            'minPulsesMapPot_coarse': self.minPulsesMapPot_coarse,
            'minPulsesMapPot_fine': self.minPulsesMapPot_fine,
            'minPulsesMapDep_coarse': self.minPulsesMapDep_coarse,
            'minPulsesMapDep_fine': self.minPulsesMapDep_fine,
            'maxPulses': maxPulses,
            'maxLinearState': maxLinearState,
        }
        self.maxPulses = maxPulses
        self.maxLinearState = maxLinearState
        self.sim.isCalibrated = True

    def _compute_optimal_pulse_matrices(self, targetStates, isCorrectionPass=False, tolerance=0.005):
        rows, cols = targetStates.shape
        if getattr(self.sim, '_is_noop', False) or rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=np.int32), np.zeros((rows, cols)), np.zeros((rows, cols))

        if self.stateMapPot is None or self.stateMapDep is None or self.gMapG is None:
            self.calibrate()
            
        if self.stateMapPot is None or self.stateMapDep is None or self.gMapG is None:
            return np.zeros((rows, cols), dtype=np.int32), np.zeros((rows, cols)), np.zeros((rows, cols))
            
        gMatrixQuantized, _ = self.sim.readCrossbarMatrix(detailedPrint=False, purpose=None)
            
        rows, cols = targetStates.shape
        
        # --- VECTORIZED OPTIMAL TRACE SEARCH STAGE ---
        state_map_pot = self.stateMapPot
        state_map_dep = self.stateMapDep
        g_map_g = self.gMapG
        g_map_n = self.gMapN
        maxLinearState = float(state_map_pot.max())
        len_gMapG_minus_1 = len(g_map_g) - 1
        len_pot_minus_1 = len(state_map_pot) - 1
        
        # 1. Gather all current states mathematically
        measuredG = np.maximum(0.0, gMatrixQuantized)
        idxG = np.searchsorted(g_map_g, measuredG)
        np.clip(idxG, 0, len_gMapG_minus_1, out=idxG)
        currentN = g_map_n[idxG]
        
        # 2. Extract directions
        is_pot = targetStates >= currentN
        is_dep = ~is_pot

        # 3. Create global Fallbacks (Standard map differences)
        idxTargetPot = np.searchsorted(state_map_pot, targetStates)
        np.clip(idxTargetPot, 0, len_pot_minus_1, out=idxTargetPot)
        idxCurrentPot = np.searchsorted(state_map_pot, currentN)
        np.clip(idxCurrentPot, 0, len_pot_minus_1, out=idxCurrentPot)
        fallback_pulses_pot = np.maximum(0, idxTargetPot - idxCurrentPot)

        if not hasattr(self, '_reversed_stateMapDep') or self._reversed_stateMapDep is None or len(self._reversed_stateMapDep) != len(state_map_dep):
            self._reversed_stateMapDep = state_map_dep[::-1]
            self._reversed_dep_midpoints = (self._reversed_stateMapDep[:-1] + self._reversed_stateMapDep[1:]) * 0.5
        reversed_dep = self._reversed_stateMapDep
        midpoints = self._reversed_dep_midpoints
        len_dep_minus_1 = len(reversed_dep) - 1
        
        def find_nearest_dep_idx(vals):
            idx = np.searchsorted(reversed_dep, vals)
            np.clip(idx, 1, len_dep_minus_1, out=idx)
            use_left = vals < midpoints[idx - 1]
            nearest = np.where(use_left, idx - 1, idx)
            nearest[vals <= reversed_dep[0]] = 0
            nearest[vals >= reversed_dep[-1]] = len_dep_minus_1
            return len_dep_minus_1 - nearest

        idxTargetDep = find_nearest_dep_idx(targetStates)
        idxCurrentDep = find_nearest_dep_idx(currentN)
        fallback_pulses_dep = np.maximum(0, idxTargetDep - idxCurrentDep)
        
        # --- JIT-COMPILED PULSE SEARCH ENGINE ---
        final_pulses_pot = np.zeros((rows, cols), dtype=np.int32)
        final_v_pot = np.zeros((rows, cols), dtype=np.float64)
        final_pw_pot = np.zeros((rows, cols), dtype=np.float64)
        
        final_pulses_dep = np.zeros((rows, cols), dtype=np.int32)
        final_v_dep = np.zeros((rows, cols), dtype=np.float64)
        final_pw_dep = np.zeros((rows, cols), dtype=np.float64)
        
        min_dep = 0.0
        scale_pot_matrix = getattr(self, '_scale_pot_matrix', None)
        if scale_pot_matrix is None:
            full_shift_pot = max(1e-12, maxLinearState)
            full_shift_dep = max(1e-12, full_shift_pot - min_dep)
            scale_pot_matrix = np.ascontiguousarray(full_shift_pot / np.maximum(1e-12, self.optMapPot), dtype=np.float64)
            scale_dep_matrix = np.ascontiguousarray(full_shift_dep / np.maximum(1e-12, self.optMapDep), dtype=np.float64)
            self._scale_pot_matrix = scale_pot_matrix
            self._scale_dep_matrix = scale_dep_matrix
            self._full_shift_pot = full_shift_pot
            self._full_shift_dep = full_shift_dep
        else:
            scale_dep_matrix = self._scale_dep_matrix
            full_shift_pot = self._full_shift_pot
            full_shift_dep = self._full_shift_dep
            
        inv_full_shift_pot = 1.0 / full_shift_pot
        inv_full_shift_dep = 1.0 / full_shift_dep
        map_pot = self.minPulsesMapPot_fine if tolerance <= 0.0001 else self.minPulsesMapPot_coarse
        map_dep = self.minPulsesMapDep_fine if tolerance <= 0.0001 else self.minPulsesMapDep_coarse
        
        target_states_contig = np.ascontiguousarray(targetStates, dtype=np.float64)
        current_n_contig = np.ascontiguousarray(currentN, dtype=np.float64)
        opt_pot_contig = np.ascontiguousarray(self.optMapPot, dtype=np.float64)
        opt_dep_contig = np.ascontiguousarray(self.optMapDep, dtype=np.float64)
        state_pot_contig = np.ascontiguousarray(self.stateMapPot, dtype=np.float64)
        map_pot_contig = np.ascontiguousarray(map_pot, dtype=np.float64)
        map_dep_contig = np.ascontiguousarray(map_dep, dtype=np.float64)
        v_write_range_contig = np.ascontiguousarray(self.vWriteRange, dtype=np.float64)
        v_erase_range_contig = np.ascontiguousarray(self.vEraseRange, dtype=np.float64)
        pw_range_contig = np.ascontiguousarray(self.pwRange, dtype=np.float64)
        fallback_pot_contig = np.ascontiguousarray(fallback_pulses_pot, dtype=np.int32)
        fallback_dep_contig = np.ascontiguousarray(fallback_pulses_dep, dtype=np.int32)
        
        with _jit_pulse_predict_lock:
            jit_predict_pulses_matrix(
                target_states_contig, current_n_contig, bool(isCorrectionPass), float(tolerance),
                opt_pot_contig, opt_dep_contig,
                state_pot_contig, reversed_dep,
                map_pot_contig, map_dep_contig,
                inv_full_shift_pot, inv_full_shift_dep,
                full_shift_pot, min_dep,
                scale_pot_matrix, scale_dep_matrix,
                v_write_range_contig, v_erase_range_contig, pw_range_contig,
                fallback_pot_contig, fallback_dep_contig,
                float(self.vWrite), float(self.vErase), float(self.pulseWidthPot), float(self.pulseWidthDep),
                final_pulses_pot, final_v_pot, final_pw_pot,
                final_pulses_dep, final_v_dep, final_pw_dep
            )

        # --- COMBINE SPACES ---
        pulsesNeeded = np.where(is_pot, final_pulses_pot, final_pulses_dep).astype(np.int32, copy=False)
        max_p_limit = int(getattr(self, 'maxPulses', 33040) or 33040)
        pulsesNeeded = np.clip(pulsesNeeded, 0, max_p_limit)
        voltagesNeeded = np.where(is_pot, final_v_pot, final_v_dep)
        pWidthsNeeded = np.where(is_pot, final_pw_pot, final_pw_dep)
        return pulsesNeeded, voltagesNeeded, pWidthsNeeded

    @handle_oom
    def programWeights(self, targetStates, isCorrectionPass=False, tolerance=0.005):
        """
        Translates a 2D matrix of target states into precise PWL Pulse Streams 
        for every Wordline using the Parallel Floating Column Scheme.
        """
        if getattr(self.sim, '_is_noop', False) or targetStates.size == 0:
            return 0.0
        if hasattr(self.sim, '_ensure_parallelization_profiled'):
            ensure_bootstrap_profiled(verbose=getattr(self.sim, 'detailedPrint', True))
            self.sim._ensure_parallelization_profiled()
            
        verbose = getattr(self.sim, 'detailedPrint', True)
        rows, cols = targetStates.shape
        pulsesNeeded, voltagesNeeded, pWidthsNeeded = self._compute_optimal_pulse_matrices(
            targetStates, isCorrectionPass=isCorrectionPass, tolerance=tolerance
        )
        
        if verbose:
            printedPulses = np.copy(pulsesNeeded)
            printedPulses[voltagesNeeded < 0] *= -1
            print(f"PULSES NEEDED:\n{printedPulses}")
            print(f"VOLTAGES CHOSEN (V):\n{np.round(voltagesNeeded, 3)}")
            print(f"PULSE WIDTHS CHOSEN (ns):\n{np.round(pWidthsNeeded * 1e9, 1)}")
            
        max_p_limit = int(getattr(self, 'maxPulses', 33040) or 33040)
        max_pulses_any = int(min(max(0, pulsesNeeded.max() if pulsesNeeded.size > 0 else 0), max_p_limit))
        if max_pulses_any == 0:
            return 0.0

        # --- VECTORIZED TIMELINE SCHEDULE COMPILER ---
        currentSysTime = self.sim.tArr[-1] if (self.sim.tArr is not None and len(self.sim.tArr) > 0) else 0.0
        currentTime = currentSysTime
        
        idleStateRow = getattr(self.sim, 'idleStateRow', np.nan)
        idleStateCol = getattr(self.sim, 'idleStateCol', np.nan)
        
        updates_per_device = getattr(self.sim, 'UpdatesPer', 'Column') == 'Device'
        
        if isCorrectionPass:
            # Phase 2: Conductance-Delay Heuristic (Most Resistive First, Most Conductive Last)
            col_conductance_sum = targetStates.sum(axis=0)
            col_order = np.argsort(col_conductance_sum)
        else:
            # Phase 1: Pulse-Density Heuristic (Longest Durations First)
            col_pulse_totals = pulsesNeeded.sum(axis=0)
            col_order = np.argsort(col_pulse_totals)[::-1]

        if updates_per_device:
            # Device updates: program cells one-by-one
            active_devices = [
                (r, c)
                for c in col_order
                for r in range(rows)
                if pulsesNeeded[r, c] > 0
            ]
                        
            if len(active_devices) == 0:
                self.sim.clearVoltageSources()
                return currentTime
                
            # Pre-calculate durations for all active devices
            dev_r, dev_c = zip(*active_devices)
            dev_durations = pulsesNeeded[dev_r, dev_c] * (pWidthsNeeded[dev_r, dev_c] * 2.0)
            
            # Pre-calculate start and end times for all active devices using np.cumsum
            ends = currentTime + np.cumsum(dev_durations + 10e-9) - 10e-9
            starts = ends - dev_durations
            
            final_currentTime = ends[-1] + 10e-9
            
            # Pre-group active device indices by row and column in a single O(N) pass
            row_to_dev_indices = [[] for _ in range(rows)]
            col_to_dev_indices = [[] for _ in range(cols)]
            for i, (dr, dc) in enumerate(active_devices):
                row_to_dev_indices[dr].append((i, dc))
                col_to_dev_indices[dc].append(i)
            
            idle_wl_T = np.array([currentTime, final_currentTime])
            idle_wl_V = np.array([idleStateRow, idleStateRow])
            idle_bl_T = np.array([currentTime, final_currentTime])
            idle_bl_V = np.array([idleStateCol, idleStateCol])
            
            # Generate row wordline schedules
            for r in range(rows):
                # Find all active devices in row r
                row_dev_indices = row_to_dev_indices[r]
                
                row_pulses = pulsesNeeded[r]
                row_v = voltagesNeeded[r]
                row_pw = pWidthsNeeded[r]
                if len(row_dev_indices) == 0:
                    wl_T = idle_wl_T
                    wl_V = idle_wl_V
                else:
                    total_pulses = sum(row_pulses[dc] for i, dc in row_dev_indices)
                    alloc_size = 2 + (total_pulses << 2)
                    wl_T = np.empty(alloc_size)
                    wl_V = np.empty(alloc_size)
                    
                    wl_T[0] = currentTime
                    wl_V[0] = idleStateRow
                    
                    ptr = 1
                    for i, dc in row_dev_indices:
                        p_count = row_pulses[dc]
                        colStart = starts[i]
                        v_amp = row_v[dc]
                        pw = row_pw[dc]
                        period = pw * 2.0
                        
                        bases = np.arange(p_count) * period + colStart
                        bases_pw = bases + pw
                        
                        end_ptr = ptr + p_count * 4
                        wl_T[ptr : end_ptr : 4] = bases
                        wl_T[ptr + 1 : end_ptr : 4] = bases
                        wl_T[ptr + 2 : end_ptr : 4] = bases_pw
                        wl_T[ptr + 3 : end_ptr : 4] = bases_pw
                        
                        wl_V[ptr : end_ptr : 4] = idleStateRow
                        wl_V[ptr + 1 : end_ptr : 4] = v_amp
                        wl_V[ptr + 2 : end_ptr : 4] = v_amp
                        wl_V[ptr + 3 : end_ptr : 4] = idleStateRow
                        ptr = end_ptr
                        
                    wl_T[ptr] = final_currentTime
                    wl_V[ptr] = idleStateRow
                    
                self.sim.addDynamicSource(row=r+1, sourceObj=PwlSource(node=None, t_points=wl_T, v_points=wl_V))
                
            # Generate column bitline schedules
            for c in range(cols):
                # Find all active devices in column c
                col_dev_indices = col_to_dev_indices[c]
                
                if len(col_dev_indices) == 0:
                    bl_T = idle_bl_T
                    bl_V = idle_bl_V
                else:
                    n_intervals = len(col_dev_indices)
                    alloc_size = 2 + (n_intervals << 2)
                    bl_T = np.empty(alloc_size)
                    bl_V = np.empty(alloc_size)
                    
                    bl_T[0] = currentTime
                    bl_V[0] = idleStateCol
                    
                    ptr = 1
                    for i in col_dev_indices:
                        colStart = starts[i]
                        colEnd = ends[i]
                        bl_T[ptr] = colStart
                        bl_T[ptr + 1] = colStart
                        bl_T[ptr + 2] = colEnd
                        bl_T[ptr + 3] = colEnd
                        bl_V[ptr] = idleStateCol
                        bl_V[ptr + 1] = 0.0
                        bl_V[ptr + 2] = 0.0
                        bl_V[ptr + 3] = idleStateCol
                        ptr += 4
                        
                    bl_T[ptr] = final_currentTime
                    bl_V[ptr] = idleStateCol
                    
                self.sim.addDynamicSource(col=c+1, sourceObj=PwlSource(node=None, t_points=bl_T, v_points=bl_V))
                
            currentTime = final_currentTime
            
        else:
            # Column updates: program multiple rows per column in parallel
            has_pulses = (pulsesNeeded.max(axis=0) > 0)
            active_cols = [c for c in col_order if has_pulses[c]]
            
            if len(active_cols) == 0:
                self.sim.clearVoltageSources()
                return currentTime
                
            # Pre-calculate duration of each active column: max pulse time in that column
            active_cols_arr = np.array(active_cols, dtype=np.intp)
            col_durations = np.max(pulsesNeeded[:, active_cols_arr] * (pWidthsNeeded[:, active_cols_arr] * 2.0), axis=0)
            
            # Pre-calculate start and end times for each active column using np.cumsum
            ends = currentTime + np.cumsum(col_durations + 10e-9) - 10e-9
            starts = ends - col_durations
            
            final_currentTime = ends[-1] + 10e-9
            
            # Map column indices to their starts and ends
            col_starts = dict(zip(active_cols, starts))
            col_ends = dict(zip(active_cols, ends))
            
            # Pre-group active columns per row in a single pass
            row_to_active_cols = [[c for c in active_cols if pulsesNeeded[r, c] > 0] for r in range(rows)]
            
            idle_wl_T = np.array([currentTime, final_currentTime])
            idle_wl_V = np.array([idleStateRow, idleStateRow])
            idle_bl_T = np.array([currentTime, final_currentTime])
            idle_bl_V = np.array([idleStateCol, idleStateCol])
            
            # Generate row wordline schedules
            for r in range(rows):
                # Find active columns that program row r
                row_active_cols = row_to_active_cols[r]
                row_pulses = pulsesNeeded[r]
                row_v = voltagesNeeded[r]
                row_pw = pWidthsNeeded[r]
                
                if len(row_active_cols) == 0:
                    wl_T = idle_wl_T
                    wl_V = idle_wl_V
                else:
                    total_pulses = sum(row_pulses[c] for c in row_active_cols)
                    alloc_size = 2 + (total_pulses << 2)
                    wl_T = np.empty(alloc_size)
                    wl_V = np.empty(alloc_size)
                    
                    wl_T[0] = currentTime
                    wl_V[0] = idleStateRow
                    
                    ptr = 1
                    for c in row_active_cols:
                        p_count = row_pulses[c]
                        colStart = col_starts[c]
                        v_amp = row_v[c]
                        pw = row_pw[c]
                        period = pw * 2.0
                        
                        bases = np.arange(p_count) * period + colStart
                        bases_pw = bases + pw
                        
                        end_ptr = ptr + (p_count << 2)
                        wl_T[ptr : end_ptr : 4] = bases
                        wl_T[ptr + 1 : end_ptr : 4] = bases
                        wl_T[ptr + 2 : end_ptr : 4] = bases_pw
                        wl_T[ptr + 3 : end_ptr : 4] = bases_pw
                        
                        wl_V[ptr : end_ptr : 4] = idleStateRow
                        wl_V[ptr + 1 : end_ptr : 4] = v_amp
                        wl_V[ptr + 2 : end_ptr : 4] = v_amp
                        wl_V[ptr + 3 : end_ptr : 4] = idleStateRow
                        ptr = end_ptr
                        
                    wl_T[ptr] = final_currentTime
                    wl_V[ptr] = idleStateRow
                    
                self.sim.addDynamicSource(row=r+1, sourceObj=PwlSource(node=None, t_points=wl_T, v_points=wl_V))
                
            # Generate column bitline schedules
            for c in range(cols):
                colStart = col_starts.get(c)
                if colStart is None:
                    bl_T = idle_bl_T
                    bl_V = idle_bl_V
                else:
                    colEnd = col_ends[c]
                    bl_T = np.array([currentTime, colStart, colStart, colEnd, colEnd, final_currentTime])
                    bl_V = np.array([idleStateCol, idleStateCol, 0.0, 0.0, idleStateCol, idleStateCol])
                    
                self.sim.addDynamicSource(col=c+1, sourceObj=PwlSource(node=None, t_points=bl_T, v_points=bl_V))
                
            currentTime = final_currentTime
            
        if verbose:
            print(f"Parallel Async Schedules bound. Total Pulses: {int(pulsesNeeded.sum())}")
        return currentTime

    @handle_oom
    def programWeightsParallel(self, targetStates, isCorrectionPass=False, tolerance=0.005, recordHistory=False):
        if getattr(self.sim, '_is_noop', False) or targetStates.size == 0:
            return 0.0
        if hasattr(self.sim, '_ensure_parallelization_profiled'):
            ensure_bootstrap_profiled(verbose=getattr(self.sim, 'detailedPrint', True))
            self.sim._ensure_parallelization_profiled()
            
        if hasattr(self.sim, '_write_back_futures') and self.sim._write_back_futures:
            concurrent.futures.wait(self.sim._write_back_futures)
            self.sim._write_back_futures.clear()

        basis = getattr(self.sim, 'UpdatesPer', 'Column')
        self.sim._load_profile_cache(basis=basis, force=True)
        rows, cols = targetStates.shape
        num_devices = rows * cols
        if num_devices in self.sim._cached_configs[basis]:
            self.sim.optimal_cores = self.sim._cached_configs[basis][num_devices][0]

        if not isCorrectionPass:
            getattr(self.sim, 'relaxDeviceTemperatures', lambda: None)()

        pulsesNeeded, voltagesNeeded, pWidthsNeeded = self._compute_optimal_pulse_matrices(
            targetStates, isCorrectionPass=isCorrectionPass, tolerance=tolerance
        )
        
        if getattr(self, '_force_fixed_pulses', False):
            pulsesNeeded = np.full((rows, cols), 50, dtype=np.int32)
            voltagesNeeded = np.full((rows, cols), 5.0)
            pWidthsNeeded = np.full((rows, cols), 320e-9)
        
        max_p_limit = int(getattr(self, 'maxPulses', 33040) or 33040)
        max_pulses_any = int(min(max(0, pulsesNeeded.max() if pulsesNeeded.size > 0 else 0), max_p_limit))
        if max_pulses_any == 0:
            return 0.0

        first_dev = self.sim.devices[0]['dev']
        device_type = getattr(first_dev, 'device_type', first_dev.params)
        architecture = getattr(self.sim, 'architecture', '1R')
        leakage_g = getattr(self.sim, 'leakage_g', 1e-10)
        
        updates_per_device = getattr(self.sim, 'UpdatesPer', 'Column') == 'Device'
        
        all_same = getattr(self.sim, '_all_dev_params_same', None)
        if all_same is None:
            all_same = getattr(self, '_all_dev_params_same', None)
        if all_same is None:
            all_same = True
            first_params = first_dev.params
            for dev_item in self.sim.devices:
                d = dev_item['dev'] if isinstance(dev_item, dict) else dev_item
                if getattr(d, 'params', None) is not first_params:
                    all_same = False
                    break
            self._all_dev_params_same = all_same and (getattr(self.sim, 'd2d_variation_intensity', 0.0) <= 0.0)
            self.sim._all_dev_params_same = self._all_dev_params_same
            
        if self.sim.backend in ['torch', 'pytorch', 'hybrid']:
            # GPU/Hybrid Backend Batched Pathway
            import torch
            import math
            import contextlib
            from .torch_simulator import run_torch_simulation, to_device_async
            
            selected_gpu = int(os.environ.get('MOLMEM_SELECTED_GPU', str(torch.cuda.current_device()) if torch.cuda.is_available() else '0'))
            device_str = f'cuda:{selected_gpu}' if self.sim.device in ('auto', 'cuda', 'gpu') else self.sim.device
            if device_str.startswith('cuda') and ':' not in device_str:
                device_str = f'cuda:{selected_gpu}'
            device = torch.device(device_str)
            dtype = torch.float64
            
            # Use dedicated stream only if an explicit non-default stream is set, otherwise None
            if device.type == 'cuda' and torch.cuda.is_available():
                cur_st = torch.cuda.current_stream(device)
                prog_stream = cur_st if getattr(cur_st, 'cuda_stream', 0) != 0 else None
            else:
                prog_stream = None
            
            max_p_limit = int(getattr(self, 'maxPulses', 33040) or 33040)
            max_pulses_any = int(min(max(0, pulsesNeeded.max()), max_p_limit))
            if max_pulses_any == 0:
                return 0.0
                
            max_pts = max(2, max_pulses_any * 4)

            # Pre-flight host RAM check for GPU runs
            try:
                import psutil
                if hasattr(self.sim, 'estimate_gpu_host_ram_bytes'):
                    est_gpu_host_bytes = self.sim.estimate_gpu_host_ram_bytes(rows, cols, is_device_basis=updates_per_device)
                    avail_ram_bytes = psutil.virtual_memory().available
                    if est_gpu_host_bytes > avail_ram_bytes * 0.95:
                        raise MemoryError(
                            f"Insufficient System RAM: GPU programming of {rows}x{cols} crossbar ({num_devices} devices) "
                            f"requires estimated {est_gpu_host_bytes / (1024.0**3):.2f} GB Host RAM, but only "
                            f"{avail_ram_bytes / (1024.0**3):.2f} GB is currently available."
                        )
            except MemoryError:
                raise
            except Exception:
                pass
            
            if updates_per_device:
                original_batch_size = rows * cols
                # Create helper sim with 1 row, 1 col
                if not hasattr(self, '_device_sims') or self._device_sims is None or len(self._device_sims) == 0:
                    localSim = self.sim.__class__(device=self.sim.device, backend=self.sim.backend)
                    localSim.compile_backend = getattr(self.sim, 'compile_backend', True)
                    localSim._is_calibration_helper = True
                    localSim.skip_readback = True
                    localSim.addCrossbarMatrix(
                        device_type=device_type, 
                        rows=1, 
                        cols=1, 
                        detailedPrint=False, 
                        architecture=architecture, 
                        force_device_init=False
                    )
                    localSim.leakage_g = leakage_g
                    self._device_sims = [localSim]
                else:
                    localSim = self._device_sims[0]
                    localSim.skip_readback = True
                
                # Setup template sources in localSim (1 col DC source, 1 row PWL source)
                localSim.clearVoltageSources()
                localSim.addDcSource(col=1, voltage=0.0)
                localSim.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=np.zeros(max_pts), v_points=np.zeros(max_pts)))
                
                pad_sources = max(2, len(localSim.vSources), len(localSim.dynamicSources))
                max_v_pts = 1 << (max(31, max_pts - 1)).bit_length()
                
                chunk_limit = self.sim.get_optimal_gpu_slice_size(rows, cols, pad_sources, max_v_pts, is_device_basis=True)
                
                pulses_flat = pulsesNeeded.ravel()
                voltages_flat = voltagesNeeded.ravel()
                pwidths_flat = pWidthsNeeded.ravel()
                devs = self._get_device_lists(rows, cols)['flat_device']
                
                tEnd_global = float((pulses_flat * pwidths_flat).max()) * 2.0
                
                max_batch_size = 1 << (max(0, min(chunk_limit, original_batch_size) - 1)).bit_length()
                
                # Pre-pin input arrays on host CPU memory to enable async DMA transfers
                pulses_host_pinned = _safe_pin_numpy(pulses_flat)
                voltages_host_pinned = _safe_pin_numpy(voltages_flat, dtype=dtype)
                pwidths_host_pinned = _safe_pin_numpy(pwidths_flat, dtype=torch.float64)
                
                # Lazy Barrier: Ensure any prior background GPU worker has safely finished before allocating new VRAM
                if hasattr(self.sim, '_active_hybrid_gpu_thread') and self.sim._active_hybrid_gpu_thread is not None and self.sim._active_hybrid_gpu_thread.is_alive():
                    self.sim._active_hybrid_gpu_thread.join()
                
                # Timeline parameter caches
                batched_v_pwl_t_cached = torch.full((max_batch_size, pad_sources, max_v_pts), 1e20, dtype=torch.float64, device=device)
                batched_v_pwl_v_cached = torch.zeros((max_batch_size, pad_sources, max_v_pts), dtype=dtype, device=device)
                
                # Pre-calculate constant PWL coordinate layout once prior to the slicing loops on CPU
                pt_idx_cpu = torch.arange(max_v_pts, dtype=torch.int64)
                k_cpu = torch.div(pt_idx_cpu, 4, rounding_mode='floor')
                sub_pt_cpu = pt_idx_cpu % 4
                pt_idx = pt_idx_cpu.to(device=device).view(1, 1, -1)
                k = k_cpu.to(device=device).view(1, 1, -1)
                sub_pt = sub_pt_cpu.to(device=device).view(1, 1, -1)
                if prog_stream is not None:
                    prog_stream.synchronize()
                elif torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                
                # Pinned initial state data (12 channels SoA)
                initial_state_data_tensor = _safe_pinned_empty((12, max_batch_size, 1), dtype=dtype, is_cuda=(device.type == 'cuda'))
                initial_state_data_np = initial_state_data_tensor.numpy()

                stream_context = torch.cuda.stream(prog_stream) if prog_stream is not None else contextlib.nullcontext()
                streams = [prog_stream]

                # Pre-pack entire device slice initial state data ONCE before slicing loop
                flat_states = np.empty((12, original_batch_size), dtype=np.float64)
                cache = getattr(self.sim, '_gpu_states_cache', None)
                if cache is not None and not getattr(self.sim, '_gpu_states_dirty', False) and isinstance(cache, dict) and 'n_arr' in cache and len(cache['n_arr']) >= original_batch_size:
                    def _to_np(arr):
                        return arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
                    flat_states[0] = _to_np(cache['n_arr'])[:original_batch_size]
                    flat_states[1] = _to_np(cache['modeState_arr'])[:original_batch_size]
                    flat_states[2] = _to_np(cache['scaleFactor_arr'])[:original_batch_size]
                    flat_states[3] = _to_np(cache['f22DepScaled_arr'])[:original_batch_size]
                    flat_states[4] = _to_np(cache['f22PotScaled_arr'])[:original_batch_size]
                    flat_states[5] = _to_np(cache.get('cDischarge_arr', np.full(original_batch_size, 0.0)))[:original_batch_size]
                    flat_states[6] = _to_np(cache['iScale_arr'])[:original_batch_size]
                    flat_states[7] = _to_np(cache['vScale_arr'])[:original_batch_size]
                    flat_states[8] = _to_np(cache['currentF22_arr'])[:original_batch_size]
                    flat_states[9] = _to_np(cache['currentIScale_arr'])[:original_batch_size]
                    flat_states[10] = _to_np(cache['currentVScale_arr'])[:original_batch_size]
                    flat_states[11] = _to_np(cache['T_arr'])[:original_batch_size]
                else:
                    for idx_d, d in enumerate(devs[:original_batch_size]):
                        flat_states[0, idx_d] = d._n
                        flat_states[1, idx_d] = d._modeState
                        flat_states[2, idx_d] = d._scaleFactor
                        flat_states[3, idx_d] = d._f22DepScaled
                        flat_states[4, idx_d] = d._f22PotScaled
                        flat_states[5, idx_d] = getattr(d, 'cDischarge', 0.0)
                        flat_states[6, idx_d] = d._iScale
                        flat_states[7, idx_d] = d._vScale
                        flat_states[8, idx_d] = d._currentF22
                        flat_states[9, idx_d] = d._currentIScale
                        flat_states[10, idx_d] = d._currentVScale
                        flat_states[11, idx_d] = d._T

                for chunk_idx, chunk_start in enumerate(range(0, original_batch_size, chunk_limit)):
                    chunk_end = min(chunk_start + chunk_limit, original_batch_size)
                    chunk_len = chunk_end - chunk_start
                    batch_size = 1 << (max(0, chunk_len - 1)).bit_length()
                    
                    chunk_pulses = pulses_flat[chunk_start:chunk_end]
                    chunk_voltages = voltages_flat[chunk_start:chunk_end]
                    chunk_pwidths = pwidths_flat[chunk_start:chunk_end]
                    
                    tEnd_chunk = float(np.max(chunk_pulses * chunk_pwidths * 2.0))
                    if tEnd_chunk <= 0.0:
                        continue
                        
                    batched_v_pwl_t = batched_v_pwl_t_cached[:batch_size]
                    batched_v_pwl_v = batched_v_pwl_v_cached[:batch_size]
                    batched_v_pwl_t.fill_(1e20)
                    batched_v_pwl_v.zero_()
                    
                    # Asynchronously transfer chunk-specific inputs from pinned CPU memory to GPU VRAM
                    chunk_pulses_gpu = pulses_host_pinned[chunk_start:chunk_end].to(device, non_blocking=True)
                    chunk_voltages_gpu = voltages_host_pinned[chunk_start:chunk_end].to(device, non_blocking=True)
                    chunk_pwidths_gpu = pwidths_host_pinned[chunk_start:chunk_end].to(device, non_blocking=True)
                    
                    chunk_pulses_t = chunk_pulses_gpu.unsqueeze(1).unsqueeze(2)
                    chunk_voltages_t = chunk_voltages_gpu.unsqueeze(1).unsqueeze(2)
                    chunk_pwidths_t = chunk_pwidths_gpu.unsqueeze(1).unsqueeze(2)
                    
                    t_final, v_final = _generate_pwl_timeline(
                        chunk_pulses_t, chunk_voltages_t, chunk_pwidths_t,
                        pt_idx, sub_pt, k
                    )
                    
                    batched_v_pwl_t[:chunk_len, 1, :] = t_final.reshape(chunk_len, max_v_pts)
                    batched_v_pwl_v[:chunk_len, 1, :] = v_final.reshape(chunk_len, max_v_pts)
                    
                    # Build initial state data using pre-allocated pinned CPU tensor in 0.001ms
                    initial_state_data = initial_state_data_np[:, :batch_size, :]
                    initial_state_data[:12, :chunk_len, 0] = flat_states[:, chunk_start:chunk_end]
                    chunk_devs = devs[chunk_start:chunk_end]
                    if batch_size > chunk_len:
                        initial_state_data[:, chunk_len:, :] = initial_state_data[:, 0:1, :]
                    
                    cur_stream = streams[chunk_idx % len(streams)]
                    cuda_dev_ctx = torch.cuda.device(device) if (device.type == 'cuda' and torch.cuda.is_available()) else contextlib.nullcontext()
                    stream_ctx = torch.cuda.stream(cur_stream) if (cur_stream is not None and getattr(cur_stream, 'cuda_stream', 0) != 0) else contextlib.nullcontext()
                    with cuda_dev_ctx, stream_ctx:
                        results = run_torch_simulation(
                            localSim, tEnd=tEnd_chunk, min_dt=1e-12, tol=1e-3, maxIter=100, title="Parallel Auto-Programming",
                            appendHistory=False, recordHistory=False, batch_size=batch_size,
                            batched_v_pwl_t=batched_v_pwl_t, batched_v_pwl_v=batched_v_pwl_v,
                            initial_state_data=initial_state_data_tensor[:, :batch_size, :].contiguous(),
                            stream=cur_stream, sync_stream=False
                        )
                        
                        v_read_val = float(getattr(self.sim, 'vRead', 0.1))
                        f22_val = results[12][:chunk_len]
                        i_scale_val = results[13][:chunk_len]
                        v_scale_eff = results[14][:chunk_len]
                        g_core = (i_scale_val * f22_val) / (v_scale_eff + 1e-12)
                        if abs(v_read_val) > 1e-5:
                            ratio = torch.clamp(v_read_val / (v_scale_eff + 1e-12), -50.0, 50.0)
                            g_core = g_core * (torch.sinh(ratio) / (v_read_val / (v_scale_eff + 1e-12)))
                        rC_chunk = torch.tensor([getattr(d, 'rC', 0.0) for d in chunk_devs], dtype=dtype, device=device).unsqueeze(1)
                        g_core_drop = torch.where(rC_chunk == 0.0, g_core, g_core / (1.0 + g_core * (2.0 * rC_chunk)))
                        g_eq_tensor = torch.where(f22_val <= 0.0, torch.zeros_like(g_core), g_core_drop)
                        
                        stacked_gpu = torch.empty((13, chunk_len, 1), dtype=dtype, device=device)
                        for state_idx in range(12):
                            stacked_gpu[state_idx].copy_(results[4 + state_idx][:chunk_len])
                        stacked_gpu[12].copy_(g_eq_tensor)
                        
                        stacked_host_tensor = torch.empty((13, chunk_len, 1), dtype=dtype, pin_memory=(device.type == 'cuda'))
                        stacked_host_tensor.copy_(stacked_gpu, non_blocking=True)
                        
                        if cur_stream is not None and getattr(cur_stream, 'cuda_stream', 0) != 0:
                            cur_stream.synchronize()
                        elif device.type == 'cuda' and torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                    
                    written_bg = False
                    if hasattr(self.sim, '_write_back_executor') and self.sim._write_back_executor is not None:
                        try:
                            fut = self.sim._write_back_executor.submit(_bg_write_back_device, stacked_host_tensor, chunk_devs, None)
                            self.sim._write_back_futures.append(fut)
                            written_bg = True
                        except RuntimeError:
                            # Executor shut down or unavailable; fall back to immediate inline write-back
                            pass
                    if not written_bg:
                        _bg_write_back_device(stacked_host_tensor, chunk_devs, None)

                if device.type == 'cuda' and torch.cuda.is_available():
                    for s in streams:
                        if s is not None:
                            s.synchronize()
                if hasattr(self.sim, '_write_back_futures') and self.sim._write_back_futures:
                    concurrent.futures.wait(self.sim._write_back_futures)
                    self.sim._write_back_futures.clear()
                localSim.clearVoltageSources()
                self.sim._gpu_states_dirty = True
                self.sim._gpu_states_in_sync = True
                self.sim._gpu_states_cache = None
                return tEnd_global
            else:
                original_batch_size = cols
                max_v_pts = 1 << (max(31, max_pts - 1)).bit_length()
                is_device_batched = (architecture == '1T1R')
                if is_device_batched:
                    # 1T1R decoupled sub-circuit: 1 row, 1 col helper for O(1) matrix complexity
                    if not hasattr(self, '_device_sims') or self._device_sims is None or len(self._device_sims) == 0:
                        localSim = self.sim.__class__(device=self.sim.device, backend=self.sim.backend)
                        localSim.compile_backend = getattr(self.sim, 'compile_backend', True)
                        localSim._is_calibration_helper = True
                        localSim.skip_readback = True
                        localSim.addCrossbarMatrix(
                            device_type=device_type, 
                            rows=1, 
                            cols=1, 
                            detailedPrint=False, 
                            architecture=architecture, 
                            force_device_init=False
                        )
                        localSim.leakage_g = leakage_g
                        self._device_sims = [localSim]
                    else:
                        localSim = self._device_sims[0]
                        localSim.skip_readback = True
                    
                    localSim.clearVoltageSources()
                    localSim.addDcSource(col=1, voltage=0.0)
                    localSim.addVoltageSource(row=1, sourceObj=PwlSource(node=None, t_points=np.zeros(max_pts), v_points=np.zeros(max_pts)))
                    pad_sources = 2
                    chunk_limit = self.sim.get_optimal_gpu_slice_size(rows, cols, pad_sources, max_v_pts, is_device_basis=False)
                    # Bound column chunk so total in-flight devices never exceed hardware warp occupancy ceiling
                    props = torch.cuda.get_device_properties(device) if (torch.cuda.is_available() and device.type == 'cuda') else None
                    cu_count = getattr(props, 'multi_processor_count', 32) or 32
                    threads_per_cu = getattr(props, 'max_threads_per_multi_processor', 1024) or 1024
                    max_hw_devs = getattr(self.sim, '_max_gpu_device_batch', int(cu_count * threads_per_cu))
                    chunk_limit = max(1, min(chunk_limit, max_hw_devs // max(1, rows)))
                    max_batch_size = 1 << (max(0, min(chunk_limit * rows, original_batch_size * rows) - 1)).bit_length()
                    
                    # Pre-pin flat input arrays for single-device sub-circuit batching
                    pulses_host_pinned_flat = _safe_pin_numpy(pulsesNeeded.T.ravel())
                    voltages_host_pinned_flat = _safe_pin_numpy(voltagesNeeded.T.ravel(), dtype=dtype)
                    pwidths_host_pinned_flat = _safe_pin_numpy(pWidthsNeeded.T.ravel(), dtype=torch.float64)
                    
                    batched_v_pwl_t_cached = torch.full((max_batch_size, pad_sources, max_v_pts), 1e20, dtype=torch.float64, device=device)
                    batched_v_pwl_v_cached = torch.zeros((max_batch_size, pad_sources, max_v_pts), dtype=dtype, device=device)
                    initial_state_data_tensor = _safe_pinned_empty((12, max_batch_size, 1), dtype=dtype, is_cuda=(device.type == 'cuda'))
                else:
                    # Non-1T1R legacy coupled crossbar
                    if not hasattr(self, '_column_sims') or self._column_sims is None or len(self._column_sims) == 0:
                        localSim = self.sim.__class__(device=self.sim.device, backend=self.sim.backend)
                        localSim.compile_backend = getattr(self.sim, 'compile_backend', True)
                        localSim._is_calibration_helper = True
                        localSim.skip_readback = True
                        localSim.addCrossbarMatrix(
                            device_type=device_type, 
                            rows=rows, 
                            cols=1, 
                            detailedPrint=False, 
                            architecture=architecture, 
                            force_device_init=False
                        )
                        localSim.leakage_g = leakage_g
                        self._column_sims = [localSim]
                    else:
                        localSim = self._column_sims[0]
                        localSim.skip_readback = True
                    
                    localSim.clearVoltageSources()
                    localSim.addDcSource(col=1, voltage=0.0)
                    for r in range(rows):
                        localSim.addVoltageSource(row=r+1, sourceObj=PwlSource(node=None, t_points=np.zeros(max_pts), v_points=np.zeros(max_pts)))
                    pad_sources = max(2, len(localSim.vSources), len(localSim.dynamicSources))
                    chunk_limit = self.sim.get_optimal_gpu_slice_size(rows, cols, pad_sources, max_v_pts, is_device_basis=False)
                    max_batch_size = 1 << (max(0, min(chunk_limit, original_batch_size) - 1)).bit_length()
                    
                    pulses_host_pinned = _safe_pin_numpy(pulsesNeeded)
                    voltages_host_pinned = _safe_pin_numpy(voltagesNeeded, dtype=dtype)
                    pwidths_host_pinned = _safe_pin_numpy(pWidthsNeeded, dtype=torch.float64)
                    
                    batched_v_pwl_t_cached = torch.full((max_batch_size, pad_sources, max_v_pts), 1e20, dtype=torch.float64, device=device)
                    batched_v_pwl_v_cached = torch.zeros((max_batch_size, pad_sources, max_v_pts), dtype=dtype, device=device)
                    initial_state_data_tensor = _safe_pinned_empty((12, max_batch_size, rows), dtype=dtype, is_cuda=(device.type == 'cuda'))
                
                devs_col = self._get_device_lists(rows, cols)['grid'].T.ravel()
                if not updates_per_device:
                    tEnd_global = float(np.sum(np.max(pulsesNeeded * pWidthsNeeded * 2.0, axis=0)))
                else:
                    tEnd_global = float(np.max(pulsesNeeded * pWidthsNeeded * 2.0))
                
                # Pre-calculate constant PWL coordinate layout once prior to the slicing loops on CPU
                pt_idx_cpu = torch.arange(max_v_pts, dtype=torch.int64)
                k_cpu = pt_idx_cpu >> 2
                sub_pt_cpu = pt_idx_cpu & 3
                pt_idx = pt_idx_cpu.to(device=device).view(1, 1, -1)
                k = k_cpu.to(device=device).view(1, 1, -1)
                sub_pt = sub_pt_cpu.to(device=device).view(1, 1, -1)
                prog_stream = getattr(self.sim, '_cuda_stream', None)
                if prog_stream is None and torch.cuda.is_available() and device.type == 'cuda':
                    cur_st = torch.cuda.current_stream(device)
                    if getattr(cur_st, 'cuda_stream', 0) != 0:
                        prog_stream = cur_st
                    else:
                        prog_stream = torch.cuda.Stream(device=device)
                        self.sim._cuda_stream = prog_stream
                if prog_stream is not None:
                    prog_stream.synchronize()
                elif torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                initial_state_data_np = initial_state_data_tensor.numpy()
                streams = [prog_stream]

                # Pre-pack entire column crossbar states ONCE before slicing loop (12 channels SoA)
                total_col_devs = rows * cols
                flat_states_col = np.empty((12, total_col_devs), dtype=np.float64)
                cache = getattr(self.sim, '_gpu_states_cache', None)
                if cache is not None and not getattr(self.sim, '_gpu_states_dirty', False) and isinstance(cache, dict) and 'n_arr' in cache and len(cache['n_arr']) >= total_col_devs:
                    if not hasattr(self, '_col_major_device_indices') or len(self._col_major_device_indices) != total_col_devs:
                        self._col_major_device_indices = np.arange(total_col_devs).reshape(rows, cols).T.ravel()
                    col_idx = self._col_major_device_indices
                    def _to_np(arr):
                        return arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
                    flat_states_col[0] = _to_np(cache['n_arr'])[col_idx]
                    flat_states_col[1] = _to_np(cache['modeState_arr'])[col_idx]
                    flat_states_col[2] = _to_np(cache['scaleFactor_arr'])[col_idx]
                    flat_states_col[3] = _to_np(cache['f22DepScaled_arr'])[col_idx]
                    flat_states_col[4] = _to_np(cache['f22PotScaled_arr'])[col_idx]
                    flat_states_col[5] = _to_np(cache.get('cDischarge_arr', np.full(total_col_devs, 0.0)))[col_idx]
                    flat_states_col[6] = _to_np(cache['iScale_arr'])[col_idx]
                    flat_states_col[7] = _to_np(cache['vScale_arr'])[col_idx]
                    flat_states_col[8] = _to_np(cache['currentF22_arr'])[col_idx]
                    flat_states_col[9] = _to_np(cache['currentIScale_arr'])[col_idx]
                    flat_states_col[10] = _to_np(cache['currentVScale_arr'])[col_idx]
                    flat_states_col[11] = _to_np(cache['T_arr'])[col_idx]
                else:
                    num_devs_col = len(devs_col)
                    flat_states_col = np.empty((12, num_devs_col), dtype=np.float64)
                    def _pack_col_slice(s, e):
                        for idx in range(s, e):
                            d = devs_col[idx]
                            flat_states_col[0, idx] = d._n
                            flat_states_col[1, idx] = d._modeState
                            flat_states_col[2, idx] = d._scaleFactor
                            flat_states_col[3, idx] = d._f22DepScaled
                            flat_states_col[4, idx] = d._f22PotScaled
                            flat_states_col[5, idx] = getattr(d, 'cDischarge', 0.0)
                            flat_states_col[6, idx] = d._iScale
                            flat_states_col[7, idx] = d._vScale
                            flat_states_col[8, idx] = d._currentF22
                            flat_states_col[9, idx] = d._currentIScale
                            flat_states_col[10, idx] = d._currentVScale
                            flat_states_col[11, idx] = d._T

                    system_cap = getattr(self.sim, '_thread_budget', max(1, int((os.cpu_count() or 1) * 0.8)))
                    num_workers = min(system_cap, max(1, num_devs_col // 512))
                    if num_workers > 1:
                        chunk_sz = (num_devs_col + num_workers - 1) // num_workers
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = []
                            for w in range(num_workers):
                                s = w * chunk_sz
                                e = min(s + chunk_sz, num_devs_col)
                                if s < e:
                                    futures.append(executor.submit(_pack_col_slice, s, e))
                            for f in futures:
                                f.result()
                    else:
                        _pack_col_slice(0, num_devs_col)

                # Heterogeneous Work-Stealing Dispatch: GPU Worker Thread + CPU Multi-Core Worker Pool
                from .programmer_solver import HeterogeneousWorkQueue, jit_cpu_device_updates_slice
                
                is_hybrid_mode = (getattr(self.sim, 'backend', 'auto') in ['hybrid', 'auto'] and 
                                  os.environ.get('MOLMEM_FORCE_GPU') != '1' and 
                                  os.environ.get('MOLMEM_FORCE_CPU') != '1')
                
                try:
                    import psutil
                    cpu_phys = psutil.cpu_count(logical=False) or 8
                except Exception:
                    cpu_phys = os.cpu_count() or 8
                    
                opt_threads = getattr(self.sim, 'optimal_cores', None) or cpu_phys
                
                from .simulator import MolmemSimulator
                cpu_chunk_sz = max(1, min(original_batch_size, int(opt_threads)))
                gpu_slice_sz = chunk_limit
                
                queue = HeterogeneousWorkQueue(
                    original_batch_size, rows, 
                    default_gpu_slice=gpu_slice_sz, 
                    default_cpu_chunk=cpu_chunk_sz
                )
                
                total_devs_count = rows * cols
                cpu_pulses_flat = pulsesNeeded.T.ravel()
                cpu_voltages_flat = voltagesNeeded.T.ravel()
                cpu_pwidths_flat = pWidthsNeeded.T.ravel()
                
                cpu_n_arr = flat_states_col[0].copy()
                cpu_modeState_arr = flat_states_col[1].copy()
                cpu_scaleFactor_arr = flat_states_col[2].copy()
                cpu_f22DepScaled_arr = flat_states_col[3].copy()
                cpu_f22PotScaled_arr = flat_states_col[4].copy()
                cDischarge_arr = flat_states_col[5].copy()
                cpu_iScale_arr = flat_states_col[6].copy()
                cpu_vScale_arr = flat_states_col[7].copy()
                cpu_currentF22_arr = flat_states_col[8].copy()
                cpu_currentIScale_arr = flat_states_col[9].copy()
                cpu_currentVScale_arr = flat_states_col[10].copy()
                cpu_T_arr = flat_states_col[11].copy()
                cpu_cached_n_int_arr = np.array([int(d._cached_n_int_val) if isinstance(d._cached_n_int_val, (int, np.integer)) else -1 for d in devs_col], dtype=np.int32)
                cpu_G_arr = np.array([float(getattr(d, '_G', 0.0)) for d in devs_col], dtype=np.float64)
                
                all_same = getattr(self.sim, '_all_dev_params_same', True)
                d0 = devs_col[0]
                if all_same:
                    rC_arr = np.full(total_devs_count, d0.rC, dtype=np.float64)
                    E_a_arr = np.full(total_devs_count, d0.E_a, dtype=np.float64)
                    R_th_arr = np.full(total_devs_count, d0.R_th, dtype=np.float64)
                    tau_th_arr = np.full(total_devs_count, d0.tau_th, dtype=np.float64)
                    vSmooth_arr = np.full(total_devs_count, d0.vSmooth, dtype=np.float64)
                    nSmooth_arr = np.full(total_devs_count, d0.nSmooth, dtype=np.float64)
                    kappa_arr = np.full(total_devs_count, d0.kappa, dtype=np.float64)
                    alpha_arr = np.full(total_devs_count, d0.alpha, dtype=np.float64)
                    vTh_arr = np.full(total_devs_count, d0.vTh, dtype=np.float64)
                    vBypass_arr = np.full(total_devs_count, d0.vBypass, dtype=np.float64)
                    vRefPos_arr = np.full(total_devs_count, d0.vRefPos, dtype=np.float64)
                    vRefNeg_arr = np.full(total_devs_count, d0.vRefNeg, dtype=np.float64)
                    nMax_arr = np.full(total_devs_count, d0.nMax, dtype=np.float64)
                    kDischarge_arr = np.full(total_devs_count, d0.kDischarge, dtype=np.float64)
                    if getattr(self.sim, 'd2d_variation_intensity', 0.0) > 0.0:
                        gScale_arr = np.array([d.gScale for d in devs_col], dtype=np.float64)
                    else:
                        gScale_arr = np.full(total_devs_count, d0.gScale, dtype=np.float64)
                    f22StartVal_arr = np.full(total_devs_count, d0.f22StartVal, dtype=np.float64)
                    f22PotScaleInv_arr = np.full(total_devs_count, d0.f22PotScaleInv, dtype=np.float64)
                    f22DepScaleInv_arr = np.full(total_devs_count, d0.f22DepScaleInv, dtype=np.float64)
                    f22EndVal_arr = np.full(total_devs_count, d0.f22EndVal, dtype=np.float64)
                    beta_alpha_arr = np.full(total_devs_count, d0.beta_alpha, dtype=np.float64)
                    beta_s_arr = np.full(total_devs_count, d0.beta_s, dtype=np.float64)
                    gamma_arr = np.full(total_devs_count, d0.gamma, dtype=np.float64)
                    rateAsymmetry_arr = np.full(total_devs_count, d0.rateAsymmetry, dtype=np.float64)
                    rTopBlend_arr = np.full(total_devs_count, getattr(d0, 'rTopBlend', 0.050000), dtype=np.float64)
                    fDischarge_arr = np.full(total_devs_count, getattr(d0, 'fDischarge', 0.374848), dtype=np.float64)
                    cDischarge_arr = np.full(total_devs_count, getattr(d0, 'cDischarge', 0.0), dtype=np.float64)
                    rBottomBlend_arr = np.full(total_devs_count, getattr(d0, 'rBottomBlend', 0.714488), dtype=np.float64)
                    rLatency_arr = np.full(total_devs_count, getattr(d0, 'rLatency', 0.878506), dtype=np.float64)
                    k_overdrive_arr = np.full(total_devs_count, getattr(d0, 'k_overdrive', 7.296635), dtype=np.float64)
                else:
                    rC_arr = np.array([d.rC for d in devs_col], dtype=np.float64)
                    E_a_arr = np.array([d.E_a for d in devs_col], dtype=np.float64)
                    R_th_arr = np.array([d.R_th for d in devs_col], dtype=np.float64)
                    tau_th_arr = np.array([d.tau_th for d in devs_col], dtype=np.float64)
                    vSmooth_arr = np.array([d.vSmooth for d in devs_col], dtype=np.float64)
                    nSmooth_arr = np.array([d.nSmooth for d in devs_col], dtype=np.float64)
                    kappa_arr = np.array([d.kappa for d in devs_col], dtype=np.float64)
                    alpha_arr = np.array([d.alpha for d in devs_col], dtype=np.float64)
                    vTh_arr = np.array([d.vTh for d in devs_col], dtype=np.float64)
                    vBypass_arr = np.array([d.vBypass for d in devs_col], dtype=np.float64)
                    vRefPos_arr = np.array([d.vRefPos for d in devs_col], dtype=np.float64)
                    vRefNeg_arr = np.array([d.vRefNeg for d in devs_col], dtype=np.float64)
                    nMax_arr = np.array([d.nMax for d in devs_col], dtype=np.float64)
                    kDischarge_arr = np.array([d.kDischarge for d in devs_col], dtype=np.float64)
                    gScale_arr = np.array([d.gScale for d in devs_col], dtype=np.float64)
                    f22StartVal_arr = np.array([d.f22StartVal for d in devs_col], dtype=np.float64)
                    f22PotScaleInv_arr = np.array([d.f22PotScaleInv for d in devs_col], dtype=np.float64)
                    f22DepScaleInv_arr = np.array([d.f22DepScaleInv for d in devs_col], dtype=np.float64)
                    f22EndVal_arr = np.array([d.f22EndVal for d in devs_col], dtype=np.float64)
                    beta_alpha_arr = np.array([d.beta_alpha for d in devs_col], dtype=np.float64)
                    beta_s_arr = np.array([d.beta_s for d in devs_col], dtype=np.float64)
                    gamma_arr = np.array([d.gamma for d in devs_col], dtype=np.float64)
                    rateAsymmetry_arr = np.array([d.rateAsymmetry for d in devs_col], dtype=np.float64)
                    rTopBlend_arr = np.array([getattr(d, 'rTopBlend', 0.050000) for d in devs_col], dtype=np.float64)
                    fDischarge_arr = np.array([getattr(d, 'fDischarge', 0.374848) for d in devs_col], dtype=np.float64)
                    cDischarge_arr = np.array([getattr(d, 'cDischarge', 0.0) for d in devs_col], dtype=np.float64)
                    rBottomBlend_arr = np.array([getattr(d, 'rBottomBlend', 0.714488) for d in devs_col], dtype=np.float64)
                    rLatency_arr = np.array([getattr(d, 'rLatency', 0.878506) for d in devs_col], dtype=np.float64)
                    k_overdrive_arr = np.array([getattr(d, 'k_overdrive', 7.296635) for d in devs_col], dtype=np.float64)
                
                lookups = d0.lookups
                y_i = lookups["iScale"]
                y_v = lookups["vScale"]
                y_fp = lookups["f22PotMap"]
                y_fd = lookups["f22DepMap"]
                v_read_val = float(getattr(self.sim, 'vRead', 0.1))
                
                if recordHistory:
                    trace_n = cpu_n_arr.copy()
                    trace_mode = cpu_modeState_arr.copy()
                    trace_scale = cpu_scaleFactor_arr.copy()
                    trace_T = cpu_T_arr.copy()
                    trace_f22Dep = cpu_f22DepScaled_arr.copy()
                    trace_f22Pot = cpu_f22PotScaled_arr.copy()
                    trace_iScale = cpu_iScale_arr.copy()
                    trace_vScale = cpu_vScale_arr.copy()
                    trace_currentF22 = cpu_currentF22_arr.copy()
                    trace_currentIScale = cpu_currentIScale_arr.copy()
                    trace_currentVScale = cpu_currentVScale_arr.copy()
                    trace_cached_n_int = cpu_cached_n_int_arr.copy()

                gpu_slice_latencies = []
                cpu_chunk_latencies = []
                
                def _run_gpu_worker():
                    try:
                        if device.type == 'cuda' and torch.cuda.is_available():
                            try:
                                torch.cuda.set_device(device)
                            except Exception:
                                pass
                        if is_hybrid_mode:
                            # Dedicated 20% core budget for GPU worker and PyTorch driver staging
                            try:
                                total_sys_cores = os.cpu_count() or 32
                                hybrid_gpu_threads = max(1, total_sys_cores - int(total_sys_cores * 0.8))
                                torch.set_num_threads(hybrid_gpu_threads)
                                
                                # On Windows, pin GPU worker thread to the top 20% hardware cores
                                if sys.platform == 'win32':
                                    import ctypes
                                    cpu_core_count = int(total_sys_cores * 0.8)
                                    gpu_core_count = total_sys_cores - cpu_core_count
                                    gpu_mask = ((1 << gpu_core_count) - 1) << cpu_core_count
                                    ctypes.windll.kernel32.SetThreadAffinityMask(
                                        ctypes.windll.kernel32.GetCurrentThread(),
                                        ctypes.c_ulonglong(gpu_mask)
                                    )
                            except Exception:
                                pass
                                
                        chunk_range = queue.claim_gpu_chunk()
                        while chunk_range is not None:
                            t_s0 = time.perf_counter()
                            chunk_start, chunk_end = chunk_range
                            chunk_len = chunk_end - chunk_start
                            c_start = chunk_start * rows
                            c_end = chunk_end * rows
                            c_len = c_end - c_start
                            chunk_devs_col = devs_col[c_start:c_end]
                            
                            chunk_pulses = pulsesNeeded[:, chunk_start:chunk_end]
                            chunk_voltages = voltagesNeeded[:, chunk_start:chunk_end]
                            chunk_pwidths = pWidthsNeeded[:, chunk_start:chunk_end]
                            
                            tEnd_chunk = float(np.max(chunk_pulses * chunk_pwidths * 2.0))
                            if tEnd_chunk <= 0.0:
                                queue.mark_gpu_completed(chunk_start, chunk_end)
                                chunk_range = queue.claim_gpu_chunk()
                                continue
                                
                            cur_stream = streams[0]
                            stream_ctx = torch.cuda.stream(cur_stream) if (cur_stream is not None and getattr(cur_stream, 'cuda_stream', 0) != 0) else contextlib.nullcontext()
                            with stream_ctx:
                                if is_device_batched:
                                    num_devs = c_len
                                    batch_size = 1 << (max(0, num_devs - 1)).bit_length()
                                    batched_v_pwl_t = batched_v_pwl_t_cached[:batch_size, :, :]
                                    batched_v_pwl_v = batched_v_pwl_v_cached[:batch_size, :, :]
                                    
                                    pulses_dev = pulses_host_pinned_flat[c_start:c_end].to(device, non_blocking=True).view(num_devs, 1, 1)
                                    voltages_dev = voltages_host_pinned_flat[c_start:c_end].to(device, non_blocking=True).view(num_devs, 1, 1)
                                    pwidths_dev = pwidths_host_pinned_flat[c_start:c_end].to(device, non_blocking=True).view(num_devs, 1, 1)
                                    
                                    t_final, v_final = _generate_pwl_timeline(
                                        pulses_dev, voltages_dev, pwidths_dev,
                                        pt_idx, sub_pt, k
                                    )
                                    batched_v_pwl_t[:num_devs, 1:2, :] = t_final
                                    batched_v_pwl_v[:num_devs, 1:2, :] = v_final
                                    if batch_size > num_devs:
                                        batched_v_pwl_t[num_devs:batch_size, 1:2, :] = 1e20
                                        batched_v_pwl_v[num_devs:batch_size, 1:2, :] = 0.0
                                    
                                    initial_state_data = initial_state_data_np[:, :batch_size, :]
                                    initial_state_data[:12, :num_devs, 0] = flat_states_col[:, c_start:c_end]
                                    if batch_size > num_devs:
                                        initial_state_data[:, num_devs:, :] = initial_state_data[:, 0:1, :]
                                        
                                    results = run_torch_simulation(
                                        localSim, tEnd=tEnd_chunk, min_dt=1e-12, tol=1e-3, maxIter=100, title="Parallel Auto-Programming",
                                        appendHistory=False, recordHistory=False, batch_size=batch_size,
                                        batched_v_pwl_t=batched_v_pwl_t, batched_v_pwl_v=batched_v_pwl_v,
                                        initial_state_data=initial_state_data_tensor[:, :batch_size, :].contiguous(),
                                        stream=cur_stream, sync_stream=False
                                    )
                                    f22_val = results[12][:num_devs, 0]
                                    i_scale_val = results[13][:num_devs, 0]
                                    v_scale_eff = results[14][:num_devs, 0]
                                    g_core = (i_scale_val * f22_val) / (v_scale_eff + 1e-12)
                                    if abs(v_read_val) > 1e-5:
                                        ratio = torch.clamp(v_read_val / (v_scale_eff + 1e-12), -50.0, 50.0)
                                        g_core = g_core * (torch.sinh(ratio) / (v_read_val / (v_scale_eff + 1e-12)))
                                    rC_chunk = torch.from_numpy(rC_arr[c_start:c_end]).to(device=device, dtype=dtype)
                                    g_core_drop = torch.where(rC_chunk == 0.0, g_core, g_core / (1.0 + g_core * (2.0 * rC_chunk)))
                                    g_eq_tensor = torch.where(f22_val <= 0.0, torch.zeros_like(g_core), g_core_drop)
                                        
                                    stacked_gpu = torch.empty((13, num_devs), dtype=dtype, device=device)
                                    for state_idx in range(12):
                                        stacked_gpu[state_idx].copy_(results[4 + state_idx][:num_devs, 0])
                                    stacked_gpu[12].copy_(g_eq_tensor)
                                else:
                                    batch_size = 1 << (max(0, chunk_len - 1)).bit_length()
                                    batched_v_pwl_t = batched_v_pwl_t_cached[:batch_size, :, :]
                                    batched_v_pwl_v = batched_v_pwl_v_cached[:batch_size, :, :]
                                    
                                    chunk_pulses_gpu = pulses_host_pinned[:, chunk_start:chunk_end].T.to(device, non_blocking=True)
                                    chunk_voltages_gpu = voltages_host_pinned[:, chunk_start:chunk_end].T.to(device, non_blocking=True)
                                    chunk_pwidths_gpu = pwidths_host_pinned[:, chunk_start:chunk_end].T.to(device, non_blocking=True)
                                    
                                    chunk_pulses_t = chunk_pulses_gpu.unsqueeze(2)
                                    chunk_voltages_t = chunk_voltages_gpu.unsqueeze(2)
                                    chunk_pwidths_t = chunk_pwidths_gpu.unsqueeze(2)
                                    
                                    t_final, v_final = _generate_pwl_timeline(
                                        chunk_pulses_t, chunk_voltages_t, chunk_pwidths_t,
                                        pt_idx, sub_pt, k
                                    )
                                    batched_v_pwl_t[:chunk_len, 1 : 1 + rows, :] = t_final
                                    batched_v_pwl_v[:chunk_len, 1 : 1 + rows, :] = v_final
                                    if batch_size > chunk_len:
                                        batched_v_pwl_t[chunk_len:batch_size, 1 : 1 + rows, :] = 1e20
                                        batched_v_pwl_v[chunk_len:batch_size, 1 : 1 + rows, :] = 0.0
                                    
                                    initial_state_data = initial_state_data_np[:, :batch_size, :]
                                    tgt_view = initial_state_data[:12, :chunk_len, :].reshape(12, -1)
                                    tgt_view[:, :c_len] = flat_states_col[:, c_start:c_end]
                                    if batch_size > chunk_len:
                                        initial_state_data[:, chunk_len:, :] = initial_state_data[:, 0:1, :]
                                    
                                    results = run_torch_simulation(
                                        localSim, tEnd=tEnd_chunk, min_dt=1e-12, tol=1e-3, maxIter=100, title="Parallel Auto-Programming",
                                        appendHistory=False, recordHistory=False, batch_size=batch_size,
                                        batched_v_pwl_t=batched_v_pwl_t, batched_v_pwl_v=batched_v_pwl_v,
                                        initial_state_data=initial_state_data_tensor[:, :batch_size, :].contiguous(),
                                        stream=cur_stream, sync_stream=False
                                    )
                                    f22_val = results[12][:chunk_len]
                                    i_scale_val = results[13][:chunk_len]
                                    v_scale_eff = results[14][:chunk_len]
                                    g_core = (i_scale_val * f22_val) / (v_scale_eff + 1e-12)
                                    if abs(v_read_val) > 1e-5:
                                        ratio = torch.clamp(v_read_val / (v_scale_eff + 1e-12), -50.0, 50.0)
                                        g_core = g_core * (torch.sinh(ratio) / (v_read_val / (v_scale_eff + 1e-12)))
                                    rC_chunk = torch.from_numpy(rC_arr[c_start:c_end].reshape(chunk_len, rows)).to(device=device, dtype=dtype)
                                    g_core_drop = torch.where(rC_chunk == 0.0, g_core, g_core / (1.0 + g_core * (2.0 * rC_chunk)))
                                    g_eq_tensor = torch.where(f22_val <= 0.0, torch.zeros_like(g_core), g_core_drop)
                                    
                                    stacked_gpu = torch.empty((13, chunk_len, rows), dtype=dtype, device=device)
                                    for state_idx in range(12):
                                        stacked_gpu[state_idx].copy_(results[4 + state_idx][:chunk_len])
                                    stacked_gpu[12].copy_(g_eq_tensor)
                                
                                if cur_stream is not None and getattr(cur_stream, 'cuda_stream', 0) != 0:
                                    cur_stream.synchronize()
                                elif device.type == 'cuda' and torch.cuda.is_available():
                                    torch.cuda.synchronize(device)
                            
                            # Fast direct numpy slice writeback into shared state vectors (zero object iteration, zero GIL lock)
                            stacked_host = stacked_gpu.cpu().numpy()
                            cpu_n_arr[c_start:c_end] = stacked_host[0].ravel()
                            cpu_modeState_arr[c_start:c_end] = stacked_host[1].ravel()
                            cpu_scaleFactor_arr[c_start:c_end] = stacked_host[2].ravel()
                            cpu_f22DepScaled_arr[c_start:c_end] = stacked_host[3].ravel()
                            cpu_f22PotScaled_arr[c_start:c_end] = stacked_host[4].ravel()
                            cDischarge_arr[c_start:c_end] = stacked_host[5].ravel()
                            cpu_cached_n_int_arr[c_start:c_end] = np.nan_to_num(stacked_host[0].ravel(), nan=-1.0).astype(np.int32)
                            cpu_iScale_arr[c_start:c_end] = stacked_host[6].ravel()
                            cpu_vScale_arr[c_start:c_end] = stacked_host[7].ravel()
                            cpu_currentF22_arr[c_start:c_end] = stacked_host[8].ravel()
                            cpu_currentIScale_arr[c_start:c_end] = stacked_host[9].ravel()
                            cpu_currentVScale_arr[c_start:c_end] = stacked_host[10].ravel()
                            cpu_T_arr[c_start:c_end] = stacked_host[11].ravel()
                            cpu_G_arr[c_start:c_end] = stacked_host[12].ravel()
                                
                            t_s_done = time.perf_counter() - t_s0
                            gpu_slice_latencies.append(t_s_done)
                            queue.mark_gpu_completed(chunk_start, chunk_end)
                            chunk_range = queue.claim_gpu_chunk()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        from .sys_utils import log_to_run_log_only
                        try:
                            log_to_run_log_only(f"[HYBRID_GPU_WORKER_EXCEPTION] [PID {os.getpid()}] {e}")
                        except Exception:
                            pass
                        if not is_hybrid_mode:
                            raise e

                t_prog_start = time.perf_counter()
                if is_hybrid_mode:
                    import threading
                    
                    gpu_worker_thread = threading.Thread(target=_run_gpu_worker, daemon=True)
                    gpu_worker_thread.start()
                    
                    try:
                        import numba as nb
                        total_sys_cores = getattr(self.sim, '_thread_budget', max(1, int((os.cpu_count() or 32) * 0.8)))
                        hybrid_cpu_threads = max(1, min(int(total_sys_cores), int(opt_threads)))
                        nb.set_num_threads(hybrid_cpu_threads)
                    except Exception:
                        pass
                        
                    # Execute CPU block using Numba's Intel oneTBB multi-core engine (opt_threads)
                    while True:
                        cpu_chunk = queue.claim_cpu_chunk()
                        if cpu_chunk is None:
                            break
                        t_c0 = time.perf_counter()
                        c_start_col, c_end_col = cpu_chunk
                        dev_start = c_start_col * rows
                        dev_end = c_end_col * rows
                        
                        jit_cpu_device_updates_slice(
                            dev_start, dev_end, cpu_pulses_flat, cpu_voltages_flat, cpu_pwidths_flat,
                            cpu_n_arr, cpu_modeState_arr, cpu_scaleFactor_arr, cpu_T_arr,
                            cpu_f22DepScaled_arr, cpu_f22PotScaled_arr, cpu_iScale_arr, cpu_vScale_arr,
                            cpu_currentF22_arr, cpu_currentIScale_arr, cpu_currentVScale_arr,
                            cpu_cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
                            vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
                            vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
                            gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
                            beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
                            rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
                            y_i, y_v, y_fp, y_fd, cpu_G_arr, v_read_val
                        )
                        t_c_done = time.perf_counter() - t_c0
                        cpu_chunk_latencies.append(t_c_done)
                        
                    if queue.is_all_done_by_cpu():
                        gpu_worker_thread.join(timeout=0.01)
                    else:
                        gpu_worker_thread.join()
                    self.sim._active_hybrid_gpu_thread = gpu_worker_thread
                    
                    # Restore default 80% thread budget for standard PyTorch operations
                    try:
                        total_sys_cores = os.cpu_count() or 32
                        torch.set_num_threads(max(2, int(total_sys_cores * 0.8)))
                    except Exception:
                        pass
                    
                    # Parallel Multi-Threaded Bulk Deferred Write-Back to Python device objects (using 80% CPU threads)
                    write_intensity = getattr(self.sim, 'write_noise_intensity', 0.0)
                    if write_intensity > 0.0:
                        from .noise import NoiseEngine
                        old_n_arr = flat_states_col[0]
                        delta_n = cpu_n_arr - old_n_arr
                        n_max_ref = devs_col[0].nMax if hasattr(devs_col[0], 'nMax') else 16520.0
                        n_norm = np.clip(cpu_n_arr / max(1.0, float(n_max_ref)), 0.0, 1.0)
                        delta_n_noisy = NoiseEngine.inject_write_noise(delta_n, n_norm, intensity=write_intensity)
                        cpu_n_arr = np.clip(old_n_arr + delta_n_noisy, 0.0, float(n_max_ref))

                    num_devs_write = len(devs_col)
                    def _write_back_slice(s, e):
                        for d_idx in range(s, e):
                            d = devs_col[d_idx]
                            d._n = cpu_n_arr[d_idx]
                            d._modeState = cpu_modeState_arr[d_idx]
                            d._scaleFactor = cpu_scaleFactor_arr[d_idx]
                            if write_intensity > 0.0:
                                d._forceUpdateConductanceParams()
                            else:
                                d._f22DepScaled = cpu_f22DepScaled_arr[d_idx]
                                d._f22PotScaled = cpu_f22PotScaled_arr[d_idx]
                                d._cached_n_int_val = int(cpu_cached_n_int_arr[d_idx])
                                d.cDischarge = float(cDischarge_arr[d_idx])
                                d._iScale = cpu_iScale_arr[d_idx]
                                d._vScale = cpu_vScale_arr[d_idx]
                                d._currentF22 = cpu_currentF22_arr[d_idx]
                                d._currentIScale = cpu_currentIScale_arr[d_idx]
                                d._currentVScale = cpu_currentVScale_arr[d_idx]
                            d._T = cpu_T_arr[d_idx]
                            d._G = cpu_G_arr[d_idx]

                    system_cap = getattr(self.sim, '_thread_budget', max(1, int((os.cpu_count() or 1) * 0.8)))
                    num_workers = min(system_cap, max(1, num_devs_write // 512))
                    if num_workers > 1:
                        chunk_sz = (num_devs_write + num_workers - 1) // num_workers
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = []
                            for w in range(num_workers):
                                s = w * chunk_sz
                                e = min(s + chunk_sz, num_devs_write)
                                if s < e:
                                    futures.append(executor.submit(_write_back_slice, s, e))
                            for f in futures:
                                f.result()
                    else:
                        _write_back_slice(0, num_devs_write)
                    
                    # Record pacing latencies and workload distribution
                    t_prog_elapsed = time.perf_counter() - t_prog_start
                    total_cols = queue.total_columns
                    gpu_cols = queue.gpu_columns_processed
                    cpu_cols = queue.cpu_columns_processed
                    avg_gpu_slice = float(np.mean(gpu_slice_latencies)) if gpu_slice_latencies else 0.0
                    avg_cpu_chunk = float(np.mean(cpu_chunk_latencies)) if cpu_chunk_latencies else 0.0
                    from .simulator import MolmemSimulator
                    cum_dist = getattr(self.sim, '_cumulative_hybrid_distribution', None)
                    if cum_dist is not None:
                        cum_dist[0] += cpu_cols
                        cum_dist[1] += gpu_cols
                        cum_dist[2] += total_cols
                        MolmemSimulator._last_hybrid_distribution = tuple(cum_dist)
                    else:
                        MolmemSimulator._last_hybrid_distribution = (cpu_cols, gpu_cols, total_cols)
                    MolmemSimulator._last_hybrid_pass_distribution = (cpu_cols, gpu_cols, total_cols)
                    MolmemSimulator._last_hybrid_pacing = (avg_cpu_chunk, avg_gpu_slice, abs(avg_cpu_chunk - avg_gpu_slice))
                    if getattr(self.sim, 'detailedPrint', True) and os.environ.get('MOLMEM_PROGRESS_COMPACT') != '1':
                        gpu_pct = (gpu_cols / total_cols) * 100.0 if total_cols > 0 else 0.0
                        cpu_pct = (cpu_cols / total_cols) * 100.0 if total_cols > 0 else 0.0
                        bar = '#' * 20
                        from .sys_utils import get_progress_suffix
                        suffix = get_progress_suffix(self.sim)
                        item_label = "Columns" if not updates_per_device else "Devices"
                        dist_str = f"GPU: {gpu_pct:.0f}% | CPU: {cpu_pct:.0f}%" if gpu_cols > 0 else f"CPU: {cpu_pct:.0f}% | GPU: {gpu_pct:.0f}%"
                        sys.stdout.write(f"  Progress ({suffix}): [{bar}] 100% | Done in {t_prog_elapsed:.2f}s | {total_cols} {item_label} ({dist_str})\n")
                        sys.stdout.flush()
                else:
                    _run_gpu_worker()
                    t_prog_elapsed = time.perf_counter() - t_prog_start
                    
                    # Single Bulk Deferred Write-Back to Python device objects for pure GPU
                    write_intensity = getattr(self.sim, 'write_noise_intensity', 0.0)
                    if write_intensity > 0.0:
                        from .noise import NoiseEngine
                        old_n_arr = flat_states_col[0]
                        delta_n = cpu_n_arr - old_n_arr
                        n_max_ref = devs_col[0].nMax if hasattr(devs_col[0], 'nMax') else 16520.0
                        n_norm = np.clip(cpu_n_arr / max(1.0, float(n_max_ref)), 0.0, 1.0)
                        delta_n_noisy = NoiseEngine.inject_write_noise(delta_n, n_norm, intensity=write_intensity)
                        cpu_n_arr = np.clip(old_n_arr + delta_n_noisy, 0.0, float(n_max_ref))

                    for d_idx, d in enumerate(devs_col):
                        d._n = cpu_n_arr[d_idx]
                        d._modeState = cpu_modeState_arr[d_idx]
                        d._scaleFactor = cpu_scaleFactor_arr[d_idx]
                        if write_intensity > 0.0:
                            d._forceUpdateConductanceParams()
                        else:
                            d._f22DepScaled = cpu_f22DepScaled_arr[d_idx]
                            d._f22PotScaled = cpu_f22PotScaled_arr[d_idx]
                            d._cached_n_int_val = int(cpu_cached_n_int_arr[d_idx])
                            d.cDischarge = float(cDischarge_arr[d_idx])
                            d._iScale = cpu_iScale_arr[d_idx]
                            d._vScale = cpu_vScale_arr[d_idx]
                            d._currentF22 = cpu_currentF22_arr[d_idx]
                            d._currentIScale = cpu_currentIScale_arr[d_idx]
                            d._currentVScale = cpu_currentVScale_arr[d_idx]
                        d._T = cpu_T_arr[d_idx]
                        d._G = cpu_G_arr[d_idx]
                        
                    if getattr(self.sim, 'detailedPrint', True) and os.environ.get('MOLMEM_PROGRESS_COMPACT') != '1':
                        bar = '#' * 20
                        from .sys_utils import get_progress_suffix
                        suffix = get_progress_suffix(self.sim)
                        item_label = "Columns" if not updates_per_device else "Devices"
                        sys.stdout.write(f"  Progress ({suffix}): [{bar}] 100% | Done in {t_prog_elapsed:.2f}s | {original_batch_size} {item_label}\n")
                        sys.stdout.flush()

                # If CPU completed all columns, skip GPU stream sync so program exits immediately
                if not (is_hybrid_mode and queue.is_all_done_by_cpu()):
                    if device.type == 'cuda' and torch.cuda.is_available():
                        for s in streams:
                            if s is not None:
                                s.synchronize()
                    if hasattr(self.sim, '_write_back_futures') and self.sim._write_back_futures:
                        concurrent.futures.wait(self.sim._write_back_futures)
                        self.sim._write_back_futures.clear()
                localSim.clearVoltageSources()
                self.sim._gpu_states_dirty = True
                self.sim._gpu_states_in_sync = True
                self.sim._gpu_states_cache = None
                
                if recordHistory:
                    from .programmer_solver import jit_trace_pulse_history
                    t_start = 0.0
                    existing_store = getattr(self.sim, '_disk_history_store', None)
                    if existing_store is not None and getattr(existing_store, 'active_len', 0) > 0:
                        t_start = float(existing_store.tArr[existing_store.active_len - 1])
                    elif hasattr(self.sim, 'tArr') and self.sim.tArr is not None and len(self.sim.tArr) > 0:
                        t_start = float(self.sim.tArr[-1])
                    t_out, g_out, n_out, f22_out, temp_out = jit_trace_pulse_history(
                        t_start, total_devs_count, cpu_pulses_flat, cpu_voltages_flat, cpu_pwidths_flat,
                        trace_n, trace_mode, trace_scale, trace_T,
                        trace_f22Dep, trace_f22Pot, trace_iScale, trace_vScale,
                        trace_currentF22, trace_currentIScale, trace_currentVScale,
                        trace_cached_n_int, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
                        vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
                        vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
                        gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
                        beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
                        rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
                        y_i, y_v, y_fp, y_fd, v_read_val,
                        rows, cols, True, (not updates_per_device)
                    )
                    
                    from .history import LazyDeviceHistoryDict, LazyHistoryDict
                    all_nodes_ordered = getattr(self.sim, '_cached_all_nodes_ordered', None) or list(range(len(getattr(self.sim, 'wordlines', {})) + len(getattr(self.sim, 'bitlines', {})) + 1))
                    num_nodes = len(all_nodes_ordered)
                    store = self.sim._ensure_disk_history_store(num_nodes, total_devs_count, initial_capacity=max(2000, len(t_out) + 100), recordHistory=True)
                    
                    skip = 1 if store.active_len > 0 else 0
                    n_pts = len(t_out) - skip
                    store.ensure_capacity(store.active_len + n_pts)
                    curr = store.active_len
                    store.tArr[curr:curr+n_pts] = t_out[skip:]
                    
                    # Remap column-major solver history to row-major crossbar device indices
                    row_major_map = np.array([(k % cols) * rows + (k // cols) for k in range(total_devs_count)], dtype=np.int32)
                    store.gHist[:, curr:curr+n_pts] = g_out[row_major_map, skip:]
                    store.stateHist[:, curr:curr+n_pts] = n_out[row_major_map, skip:]
                    store.f22Hist[:, curr:curr+n_pts] = f22_out[row_major_map, skip:]
                    store.tempHist[:, curr:curr+n_pts] = temp_out[row_major_map, skip:]
                    store.active_len = curr + n_pts
                    
                    end = store.active_len
                    self.sim._tArr_full = store.tArr
                    self.sim.tArr = self.sim._tArr_full[:end]
                    self.sim._gHistory_full = store.gHist
                    self.sim.gHistory = LazyDeviceHistoryDict(self.sim._gHistory_full, total_devs_count, end)
                    self.sim._stateHistory_full = store.stateHist
                    self.sim.stateHistory = LazyDeviceHistoryDict(self.sim._stateHistory_full, total_devs_count, end)
                    self.sim._f22History_full = store.f22Hist
                    self.sim.f22History = LazyDeviceHistoryDict(self.sim._f22History_full, total_devs_count, end)
                    self.sim._vHistory_full = store.vHist
                    self.sim.vHistory = LazyHistoryDict(self.sim._vHistory_full, all_nodes_ordered, end)
                    self.sim._iHistory_full = store.iHist
                    self.sim.iHistory = LazyDeviceHistoryDict(self.sim._iHistory_full, total_devs_count, end)
                    self.sim._tempHistory_full = store.tempHist
                    self.sim.tempHistory = LazyDeviceHistoryDict(self.sim._tempHistory_full, total_devs_count, end)

                return tEnd_global
        num_devices = rows * cols
        
        # Pre-flight host RAM check against calibrated theoretical memory footprint
        try:
            import psutil
            if hasattr(self.sim, 'estimate_cpu_memory_bytes'):
                est_cpu_bytes = self.sim.estimate_cpu_memory_bytes(rows, cols, is_device_basis=updates_per_device)
                avail_ram_bytes = psutil.virtual_memory().available
                if est_cpu_bytes > avail_ram_bytes * 0.95:
                    raise MemoryError(
                        f"Insufficient System RAM: Programming {rows}x{cols} crossbar ({num_devices} devices) "
                        f"requires estimated {est_cpu_bytes / (1024.0**3):.2f} GB RAM, but only "
                        f"{avail_ram_bytes / (1024.0**3):.2f} GB is currently available."
                    )
        except MemoryError:
            raise
        except Exception:
            pass

        # Pack device attributes
        devs_flat = self._get_device_lists(rows, cols)['flat_device']
                
        n_arr = np.empty(num_devices, dtype=np.float64)
        modeState_arr = np.empty(num_devices, dtype=np.float64)
        scaleFactor_arr = np.empty(num_devices, dtype=np.float64)
        f22DepScaled_arr = np.empty(num_devices, dtype=np.float64)
        f22PotScaled_arr = np.empty(num_devices, dtype=np.float64)
        _cached_n_int_arr = np.empty(num_devices, dtype=np.int32)
        iScale_arr = np.empty(num_devices, dtype=np.float64)
        vScale_arr = np.empty(num_devices, dtype=np.float64)
        currentF22_arr = np.empty(num_devices, dtype=np.float64)
        currentIScale_arr = np.empty(num_devices, dtype=np.float64)
        currentVScale_arr = np.empty(num_devices, dtype=np.float64)
        T_arr = np.empty(num_devices, dtype=np.float64)
        G_arr = np.empty(num_devices, dtype=np.float64)
        
        # Check if homogeneous
        if num_devices > 0:
            first_dev = devs_flat[0]
            all_same = getattr(self.sim, '_all_dev_params_same', None)
            if all_same is None:
                all_same = getattr(self, '_all_dev_params_same', None)
            if all_same is None:
                all_same = True
                first_params = first_dev.params
                for dev in devs_flat:
                    if dev.params is not first_params:
                        all_same = False
                        break
                self._all_dev_params_same = all_same and (getattr(self.sim, 'd2d_variation_intensity', 0.0) <= 0.0)
                self.sim._all_dev_params_same = self._all_dev_params_same
            
            if all_same:
                rC_arr = np.full(num_devices, first_dev.rC, dtype=np.float64)
                E_a_arr = np.full(num_devices, first_dev.E_a, dtype=np.float64)
                R_th_arr = np.full(num_devices, first_dev.R_th, dtype=np.float64)
                tau_th_arr = np.full(num_devices, first_dev.tau_th, dtype=np.float64)
                vSmooth_arr = np.full(num_devices, first_dev.vSmooth, dtype=np.float64)
                nSmooth_arr = np.full(num_devices, first_dev.nSmooth, dtype=np.float64)
                kappa_arr = np.full(num_devices, first_dev.kappa, dtype=np.float64)
                alpha_arr = np.full(num_devices, first_dev.alpha, dtype=np.float64)
                vTh_arr = np.full(num_devices, first_dev.vTh, dtype=np.float64)
                vRefPos_arr = np.full(num_devices, first_dev.vRefPos, dtype=np.float64)
                vRefNeg_arr = np.full(num_devices, first_dev.vRefNeg, dtype=np.float64)
                nMax_arr = np.full(num_devices, first_dev.nMax, dtype=np.float64)
                kDischarge_arr = np.full(num_devices, first_dev.kDischarge, dtype=np.float64)
                if getattr(self.sim, 'd2d_variation_intensity', 0.0) > 0.0:
                    gScale_arr = np.array([d.gScale for d in devs_flat], dtype=np.float64)
                else:
                    gScale_arr = np.full(num_devices, first_dev.gScale, dtype=np.float64)
                vBypass_arr = np.full(num_devices, first_dev.vBypass, dtype=np.float64)
                beta_alpha_arr = np.full(num_devices, first_dev.beta_alpha, dtype=np.float64)
                beta_s_arr = np.full(num_devices, first_dev.beta_s, dtype=np.float64)
                gamma_arr = np.full(num_devices, first_dev.gamma, dtype=np.float64)
                rateAsymmetry_arr = np.full(num_devices, first_dev.rateAsymmetry, dtype=np.float64)
                rTopBlend_arr = np.full(num_devices, getattr(first_dev, 'rTopBlend', 0.050000), dtype=np.float64)
                fDischarge_arr = np.full(num_devices, getattr(first_dev, 'fDischarge', 0.374848), dtype=np.float64)
                cDischarge_arr = np.full(num_devices, getattr(first_dev, 'cDischarge', 0.0), dtype=np.float64)
                rBottomBlend_arr = np.full(num_devices, getattr(first_dev, 'rBottomBlend', 0.714488), dtype=np.float64)
                rLatency_arr = np.full(num_devices, getattr(first_dev, 'rLatency', 0.878506), dtype=np.float64)
                k_overdrive_arr = np.full(num_devices, getattr(first_dev, 'k_overdrive', 7.296635), dtype=np.float64)
                f22StartVal_arr = np.full(num_devices, first_dev.f22StartVal, dtype=np.float64)
                f22PotScaleInv_arr = np.full(num_devices, first_dev.f22PotScaleInv, dtype=np.float64)
                f22DepScaleInv_arr = np.full(num_devices, first_dev.f22DepScaleInv, dtype=np.float64)
                f22EndVal_arr = np.full(num_devices, first_dev.f22EndVal, dtype=np.float64)
                
                if num_devices == 1:
                    dev = devs_flat[0]
                    n_arr[0] = dev._n
                    modeState_arr[0] = dev._modeState
                    scaleFactor_arr[0] = dev._scaleFactor
                    f22DepScaled_arr[0] = dev._f22DepScaled
                    f22PotScaled_arr[0] = dev._f22PotScaled
                    _cached_n_int_arr[0] = dev._cached_n_int_val
                    iScale_arr[0] = dev._iScale
                    vScale_arr[0] = dev._vScale
                    currentF22_arr[0] = dev._currentF22
                    currentIScale_arr[0] = dev._currentIScale
                    currentVScale_arr[0] = dev._currentVScale
                    T_arr[0] = dev._T
                else:
                    cache = getattr(self.sim, '_gpu_states_cache', None)
                    if cache is not None and not getattr(self.sim, '_gpu_states_dirty', False) and isinstance(cache, dict) and 'n_arr' in cache and len(cache['n_arr']) == num_devices:
                        np.copyto(n_arr, cache['n_arr'])
                        np.copyto(modeState_arr, cache['modeState_arr'])
                        np.copyto(scaleFactor_arr, cache['scaleFactor_arr'])
                        np.copyto(f22DepScaled_arr, cache['f22DepScaled_arr'])
                        np.copyto(f22PotScaled_arr, cache['f22PotScaled_arr'])
                        np.copyto(_cached_n_int_arr, cache['_cached_n_int_arr'])
                        if 'cDischarge_arr' in cache:
                            np.copyto(cDischarge_arr, cache['cDischarge_arr'])
                        np.copyto(iScale_arr, cache['iScale_arr'])
                        np.copyto(vScale_arr, cache['vScale_arr'])
                        np.copyto(currentF22_arr, cache['currentF22_arr'])
                        np.copyto(currentIScale_arr, cache['currentIScale_arr'])
                        np.copyto(currentVScale_arr, cache['currentVScale_arr'])
                        np.copyto(T_arr, cache['T_arr'])
                    else:
                        d0 = devs_flat[0]
                        d0_n = d0._n
                        d0_mode = d0._modeState
                        d0_scale = d0._scaleFactor
                        d0_T = d0._T
                        d0_f22 = d0._currentF22
                        d0_fp = d0._f22PotScaled
                        d0_fd = d0._f22DepScaled
                        d0_is = d0._iScale
                        d0_vs = d0._vScale
                        d0_cis = d0._currentIScale
                        d0_cvs = d0._currentVScale
                        val0 = d0._cached_n_int_val
                        c_val0 = val0 if type(val0) is int else (int(val0) if isinstance(val0, (int, np.integer)) else -1)
                        
                        is_homogeneous = True
                        dl = devs_flat[-1]
                        if (dl._n != d0_n or dl._modeState != d0_mode or 
                            dl._scaleFactor != d0_scale or dl._T != d0_T or
                            dl._currentF22 != d0_f22 or dl._f22PotScaled != d0_fp):
                            is_homogeneous = False
                        else:
                            for dev in devs_flat[1:-1]:
                                if (dev._n != d0_n or dev._modeState != d0_mode or 
                                    dev._scaleFactor != d0_scale or dev._T != d0_T or
                                    dev._currentF22 != d0_f22 or dev._f22PotScaled != d0_fp):
                                    is_homogeneous = False
                                    break
                        
                        if is_homogeneous:
                            n_arr.fill(d0_n)
                            modeState_arr.fill(d0_mode)
                            scaleFactor_arr.fill(d0_scale)
                            f22DepScaled_arr.fill(d0_fd)
                            f22PotScaled_arr.fill(d0_fp)
                            _cached_n_int_arr.fill(c_val0)
                            iScale_arr.fill(d0_is)
                            vScale_arr.fill(d0_vs)
                            currentF22_arr.fill(d0_f22)
                            currentIScale_arr.fill(d0_cis)
                            currentVScale_arr.fill(d0_cvs)
                            T_arr.fill(d0_T)
                        else:
                            for i, dev in enumerate(devs_flat):
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
                            
                            self.sim._gpu_states_cache = {
                                'n_arr': n_arr.copy(),
                                'modeState_arr': modeState_arr.copy(),
                                'scaleFactor_arr': scaleFactor_arr.copy(),
                                'T_arr': T_arr.copy(),
                                'currentF22_arr': currentF22_arr.copy(),
                                '_cached_n_int_arr': _cached_n_int_arr.copy(),
                                'cDischarge_arr': cDischarge_arr.copy(),
                                'iScale_arr': iScale_arr.copy(),
                                'vScale_arr': vScale_arr.copy(),
                                'currentIScale_arr': currentIScale_arr.copy(),
                                'currentVScale_arr': currentVScale_arr.copy(),
                                'f22DepScaled_arr': f22DepScaled_arr.copy(),
                                'f22PotScaled_arr': f22PotScaled_arr.copy()
                            }
                        self.sim._gpu_states_dirty = False
                        self.sim._gpu_states_in_sync = True
            else:
                rC_arr = np.empty(num_devices, dtype=np.float64)
                E_a_arr = np.empty(num_devices, dtype=np.float64)
                R_th_arr = np.empty(num_devices, dtype=np.float64)
                tau_th_arr = np.empty(num_devices, dtype=np.float64)
                vSmooth_arr = np.empty(num_devices, dtype=np.float64)
                nSmooth_arr = np.empty(num_devices, dtype=np.float64)
                kappa_arr = np.empty(num_devices, dtype=np.float64)
                alpha_arr = np.empty(num_devices, dtype=np.float64)
                vTh_arr = np.empty(num_devices, dtype=np.float64)
                vRefPos_arr = np.empty(num_devices, dtype=np.float64)
                vRefNeg_arr = np.empty(num_devices, dtype=np.float64)
                nMax_arr = np.empty(num_devices, dtype=np.float64)
                kDischarge_arr = np.empty(num_devices, dtype=np.float64)
                gScale_arr = np.empty(num_devices, dtype=np.float64)
                vBypass_arr = np.empty(num_devices, dtype=np.float64)
                beta_alpha_arr = np.empty(num_devices, dtype=np.float64)
                beta_s_arr = np.empty(num_devices, dtype=np.float64)
                gamma_arr = np.empty(num_devices, dtype=np.float64)
                rateAsymmetry_arr = np.empty(num_devices, dtype=np.float64)
                rTopBlend_arr = np.empty(num_devices, dtype=np.float64)
                fDischarge_arr = np.empty(num_devices, dtype=np.float64)
                cDischarge_arr = np.empty(num_devices, dtype=np.float64)
                rBottomBlend_arr = np.empty(num_devices, dtype=np.float64)
                rLatency_arr = np.empty(num_devices, dtype=np.float64)
                k_overdrive_arr = np.empty(num_devices, dtype=np.float64)
                f22StartVal_arr = np.empty(num_devices, dtype=np.float64)
                f22PotScaleInv_arr = np.empty(num_devices, dtype=np.float64)
                f22DepScaleInv_arr = np.empty(num_devices, dtype=np.float64)
                f22EndVal_arr = np.empty(num_devices, dtype=np.float64)
                
                for i, dev in enumerate(devs_flat):
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
                    rC_arr[i] = dev.rC
                    E_a_arr[i] = dev.E_a
                    R_th_arr[i] = dev.R_th
                    tau_th_arr[i] = dev.tau_th
                    vSmooth_arr[i] = dev.vSmooth
                    nSmooth_arr[i] = dev.nSmooth
                    kappa_arr[i] = dev.kappa
                    alpha_arr[i] = dev.alpha
                    vTh_arr[i] = dev.vTh
                    vRefPos_arr[i] = dev.vRefPos
                    vRefNeg_arr[i] = dev.vRefNeg
                    nMax_arr[i] = dev.nMax
                    kDischarge_arr[i] = dev.kDischarge
                    gScale_arr[i] = dev.gScale
                    vBypass_arr[i] = dev.vBypass
                    beta_alpha_arr[i] = dev.beta_alpha
                    beta_s_arr[i] = dev.beta_s
                    gamma_arr[i] = dev.gamma
                    rateAsymmetry_arr[i] = dev.rateAsymmetry
                    rTopBlend_arr[i] = getattr(dev, 'rTopBlend', 0.050000)
                    fDischarge_arr[i] = getattr(dev, 'fDischarge', 0.374848)
                    cDischarge_arr[i] = getattr(dev, 'cDischarge', 0.0)
                    rBottomBlend_arr[i] = getattr(dev, 'rBottomBlend', 0.714488)
                    rLatency_arr[i] = getattr(dev, 'rLatency', 0.878506)
                    k_overdrive_arr[i] = getattr(dev, 'k_overdrive', 7.296635)
                    f22StartVal_arr[i] = dev.f22StartVal
                    f22PotScaleInv_arr[i] = dev.f22PotScaleInv
                    f22DepScaleInv_arr[i] = dev.f22DepScaleInv
                    f22EndVal_arr[i] = dev.f22EndVal
            
        t_cpu_start = time.perf_counter()
        pulses_flat = pulsesNeeded.ravel()
        voltages_flat = voltagesNeeded.ravel()
        pwidths_flat = pWidthsNeeded.ravel()
        
        first_lookups = devs_flat[0].lookups
        y_i = first_lookups["iScale"]
        y_v = first_lookups["vScale"]
        y_fp = first_lookups["f22PotMap"]
        y_fd = first_lookups["f22DepMap"]
        
        if recordHistory:
            trace_n = n_arr.copy()
            trace_mode = modeState_arr.copy()
            trace_scale = scaleFactor_arr.copy()
            trace_T = T_arr.copy()
            trace_f22Dep = f22DepScaled_arr.copy()
            trace_f22Pot = f22PotScaled_arr.copy()
            trace_iScale = iScale_arr.copy()
            trace_vScale = vScale_arr.copy()
            trace_currentF22 = currentF22_arr.copy()
            trace_currentIScale = currentIScale_arr.copy()
            trace_currentVScale = currentVScale_arr.copy()
            trace_cached_n_int = _cached_n_int_arr.copy()
        
        try:
            import numba as nb
            nb.set_num_threads(max(1, getattr(self.sim, 'optimal_cores', 1)))
        except Exception:
            pass
        
        v_read_val = float(getattr(self.sim, 'vRead', 0.1))
        jit_cpu_device_updates_parallel(
            num_devices, pulses_flat, voltages_flat, pwidths_flat,
            n_arr, modeState_arr, scaleFactor_arr, T_arr,
            f22DepScaled_arr, f22PotScaled_arr, iScale_arr, vScale_arr,
            currentF22_arr, currentIScale_arr, currentVScale_arr,
            _cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
            vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
            vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
            gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
            beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
            rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
            y_i, y_v, y_fp, y_fd, G_arr, v_read_val
        )

        write_intensity = getattr(self.sim, 'write_noise_intensity', 0.0)
        if write_intensity > 0.0:
            from .noise import NoiseEngine
            old_n_arr = np.array([d._n for d in devs_flat], dtype=np.float64)
            delta_n = n_arr - old_n_arr
            n_max_ref = devs_flat[0].nMax if hasattr(devs_flat[0], 'nMax') else 16520.0
            n_norm = np.clip(n_arr / max(1.0, float(n_max_ref)), 0.0, 1.0)
            delta_n_noisy = NoiseEngine.inject_write_noise(delta_n, n_norm, intensity=write_intensity)
            n_arr = np.clip(old_n_arr + delta_n_noisy, 0.0, float(n_max_ref))
        
        if num_devices == 1:
            dev = devs_flat[0]
            dev._n = n_arr[0]
            dev._modeState = modeState_arr[0]
            dev._scaleFactor = scaleFactor_arr[0]
            if write_intensity > 0.0:
                dev._forceUpdateConductanceParams()
            else:
                dev._f22DepScaled = f22DepScaled_arr[0]
                dev._f22PotScaled = f22PotScaled_arr[0]
                dev._cached_n_int_val = _cached_n_int_arr[0]
                dev.cDischarge = cDischarge_arr[0]
                dev._iScale = iScale_arr[0]
                dev._vScale = vScale_arr[0]
                dev._currentF22 = currentF22_arr[0]
                dev._currentIScale = currentIScale_arr[0]
                dev._currentVScale = currentVScale_arr[0]
            dev._T = T_arr[0]
            dev._G = G_arr[0]
        else:
            if (write_intensity == 0.0 and n_arr[0] == n_arr[-1] and T_arr[0] == T_arr[-1] and G_arr[0] == G_arr[-1] and 
                modeState_arr[0] == modeState_arr[-1] and np.all(n_arr == n_arr[0])):
                n0 = n_arr[0]
                m0 = modeState_arr[0]
                sc0 = scaleFactor_arr[0]
                fd0 = f22DepScaled_arr[0]
                fp0 = f22PotScaled_arr[0]
                c0 = _cached_n_int_arr[0]
                cd0 = cDischarge_arr[0]
                is0 = iScale_arr[0]
                vs0 = vScale_arr[0]
                cf0 = currentF22_arr[0]
                cis0 = currentIScale_arr[0]
                cvs0 = currentVScale_arr[0]
                t0 = T_arr[0]
                g0 = G_arr[0]
                d0 = devs_flat[0]
                if not (d0._n == n0 and d0._modeState == m0 and d0._scaleFactor == sc0 and 
                        d0._f22DepScaled == fd0 and d0._f22PotScaled == fp0 and 
                        d0._cached_n_int_val == c0 and d0._T == t0 and getattr(d0, '_G', None) == g0):
                    for dev in devs_flat:
                        dev._n = n0
                        dev._modeState = m0
                        dev._scaleFactor = sc0
                        dev._f22DepScaled = fd0
                        dev._f22PotScaled = fp0
                        dev._cached_n_int_val = c0
                        dev.cDischarge = cd0
                        dev._iScale = is0
                        dev._vScale = vs0
                        dev._currentF22 = cf0
                        dev._currentIScale = cis0
                        dev._currentVScale = cvs0
                        dev._T = t0
                        dev._G = g0
            else:
                active_col_mask = (pulsesNeeded > 0).any(axis=0) if 'pulsesNeeded' in locals() and pulsesNeeded is not None else None
                if active_col_mask is not None and not active_col_mask.all():
                    dev_grid = self._get_device_lists(rows, cols)['grid']
                    active_cols = np.where(active_col_mask)[0]
                    for c in active_cols:
                        for r in range(rows):
                            i = r * cols + c
                            dev = dev_grid[r, c]
                            new_n = n_arr[i]
                            if abs(dev._n - new_n) > 1e-12 or abs(dev._T - T_arr[i]) > 1e-4 or dev._modeState != modeState_arr[i]:
                                dev._n = new_n
                                dev._modeState = modeState_arr[i]
                                dev._scaleFactor = scaleFactor_arr[i]
                                if write_intensity > 0.0:
                                    dev._forceUpdateConductanceParams()
                                else:
                                    dev._f22DepScaled = f22DepScaled_arr[i]
                                    dev._f22PotScaled = f22PotScaled_arr[i]
                                    dev._cached_n_int_val = _cached_n_int_arr[i]
                                    dev.cDischarge = cDischarge_arr[i]
                                    dev._iScale = iScale_arr[i]
                                    dev._vScale = vScale_arr[i]
                                    dev._currentF22 = currentF22_arr[i]
                                    dev._currentIScale = currentIScale_arr[i]
                                    dev._currentVScale = currentVScale_arr[i]
                                dev._T = T_arr[i]
                                dev._G = G_arr[i]
                else:
                    for i, dev in enumerate(devs_flat):
                        new_n = n_arr[i]
                        if abs(dev._n - new_n) > 1e-12 or abs(dev._T - T_arr[i]) > 1e-4 or dev._modeState != modeState_arr[i]:
                            dev._n = new_n
                            dev._modeState = modeState_arr[i]
                            dev._scaleFactor = scaleFactor_arr[i]
                            if write_intensity > 0.0:
                                dev._forceUpdateConductanceParams()
                            else:
                                dev._f22DepScaled = f22DepScaled_arr[i]
                                dev._f22PotScaled = f22PotScaled_arr[i]
                                dev._cached_n_int_val = _cached_n_int_arr[i]
                                dev.cDischarge = cDischarge_arr[i]
                                dev._iScale = iScale_arr[i]
                                dev._vScale = vScale_arr[i]
                                dev._currentF22 = currentF22_arr[i]
                                dev._currentIScale = currentIScale_arr[i]
                                dev._currentVScale = currentVScale_arr[i]
                            dev._T = T_arr[i]
                            dev._G = G_arr[i]

            if getattr(self.sim, 'lazy_sync', True):
                self.sim._gpu_states_cache = {
                    'n_arr': n_arr.copy(),
                    'modeState_arr': modeState_arr.copy(),
                    'scaleFactor_arr': scaleFactor_arr.copy(),
                    'T_arr': T_arr.copy(),
                    'currentF22_arr': currentF22_arr.copy(),
                    '_cached_n_int_arr': _cached_n_int_arr.copy(),
                    'cDischarge_arr': cDischarge_arr.copy(),
                    'iScale_arr': iScale_arr.copy(),
                    'vScale_arr': vScale_arr.copy(),
                    'currentIScale_arr': currentIScale_arr.copy(),
                    'currentVScale_arr': currentVScale_arr.copy(),
                    'f22DepScaled_arr': f22DepScaled_arr.copy(),
                    'f22PotScaled_arr': f22PotScaled_arr.copy(),
                    'gEq_arr': G_arr.copy()
                }
                self.sim._gpu_states_in_sync = False
                self.sim._gpu_states_dirty = False
            
        if recordHistory:
            from .programmer_solver import jit_trace_pulse_history
            t_start = float(self.sim.tArr[-1]) if hasattr(self.sim, 'tArr') and self.sim.tArr is not None and len(self.sim.tArr) > 0 else 0.0
            t_out, g_out, n_out, f22_out, temp_out = jit_trace_pulse_history(
                t_start, num_devices, pulses_flat, voltages_flat, pwidths_flat,
                trace_n, trace_mode, trace_scale, trace_T,
                trace_f22Dep, trace_f22Pot, trace_iScale, trace_vScale,
                trace_currentF22, trace_currentIScale, trace_currentVScale,
                trace_cached_n_int, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
                vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
                vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
                gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
                beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
                rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
                y_i, y_v, y_fp, y_fd, v_read_val,
                rows, cols, False, (not updates_per_device)
            )
            
            from .history import LazyDeviceHistoryDict, LazyHistoryDict
            all_nodes_ordered = getattr(self.sim, '_cached_all_nodes_ordered', None) or list(range(len(getattr(self.sim, 'wordlines', {})) + len(getattr(self.sim, 'bitlines', {})) + 1))
            num_nodes = len(all_nodes_ordered)
            store = self.sim._ensure_disk_history_store(num_nodes, num_devices, initial_capacity=max(2000, len(t_out) + 100), recordHistory=True)
            
            skip = 1 if store.active_len > 0 else 0
            n_pts = len(t_out) - skip
            store.ensure_capacity(store.active_len + n_pts)
            curr = store.active_len
            store.tArr[curr:curr+n_pts] = t_out[skip:]
            store.gHist[:, curr:curr+n_pts] = g_out[:, skip:]
            store.stateHist[:, curr:curr+n_pts] = n_out[:, skip:]
            store.f22Hist[:, curr:curr+n_pts] = f22_out[:, skip:]
            store.tempHist[:, curr:curr+n_pts] = temp_out[:, skip:]
            store.active_len = curr + n_pts
            
            end = store.active_len
            self.sim._tArr_full = store.tArr
            self.sim.tArr = self.sim._tArr_full[:end]
            self.sim._gHistory_full = store.gHist
            self.sim.gHistory = LazyDeviceHistoryDict(self.sim._gHistory_full, num_devices, end)
            self.sim._stateHistory_full = store.stateHist
            self.sim.stateHistory = LazyDeviceHistoryDict(self.sim._stateHistory_full, num_devices, end)
            self.sim._f22History_full = store.f22Hist
            self.sim.f22History = LazyDeviceHistoryDict(self.sim._f22History_full, num_devices, end)
            self.sim._vHistory_full = store.vHist
            self.sim.vHistory = LazyHistoryDict(self.sim._vHistory_full, all_nodes_ordered, end)
            self.sim._iHistory_full = store.iHist
            self.sim.iHistory = LazyDeviceHistoryDict(self.sim._iHistory_full, num_devices, end)
            self.sim._tempHistory_full = store.tempHist
            self.sim.tempHistory = LazyDeviceHistoryDict(self.sim._tempHistory_full, num_devices, end)
            
        if getattr(self.sim, 'detailedPrint', True) and os.environ.get('MOLMEM_PROGRESS_COMPACT') != '1':
            t_cpu_elapsed = time.perf_counter() - t_cpu_start
            bar = '#' * 20
            sys.stdout.write(f"  Progress (cpu): [{bar}] 100% | Done in {t_cpu_elapsed:.2f}s | {num_devices} Devices\n")
            sys.stdout.flush()
            
        if not updates_per_device and architecture != '1T1R':
            tEnd = float(np.sum(np.max(pulsesNeeded * pWidthsNeeded * 2.0, axis=0)))
        else:
            tEnd = float((pulses_flat * pwidths_flat).max()) * 2.0
        return tEnd
