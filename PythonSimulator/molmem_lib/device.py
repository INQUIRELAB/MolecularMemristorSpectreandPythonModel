import numpy as np
import os
import math
from .devices import get_device_config
_jit_cache = {}
_fn_jit_interp_O1 = None
_fn_jit_getIGeq = None
_fn_jit_predict_correct = None
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ensure_jit():
    global _fn_jit_interp_O1, _fn_jit_getIGeq, _fn_jit_predict_correct, _jit_cache
    if not _jit_cache:
        from .device_jit import (
            jit_interp_O1,
            jit_getIGeq,
            jit_predict_correct,
            jit_updateState,
            jit_step_predictor_corrector_array,
            jit_commit_state_array
        )
        _fn_jit_interp_O1 = jit_interp_O1
        _fn_jit_getIGeq = jit_getIGeq
        _fn_jit_predict_correct = jit_predict_correct
        _jit_cache.update({
            'jit_interp_O1': jit_interp_O1,
            'jit_getIGeq': jit_getIGeq,
            'jit_predict_correct': jit_predict_correct,
            'jit_updateState': jit_updateState,
            'jit_step_predictor_corrector_array': jit_step_predictor_corrector_array,
            'jit_commit_state_array': jit_commit_state_array
        })

_ensure_jit()


class SynchronizedField:
    __slots__ = ['private_name']
    
    def __init__(self, private_name):
        self.private_name = private_name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        sim_ref = getattr(instance, '_sim', None)
        if sim_ref is not None:
            sim = sim_ref()
            if sim is not None and not getattr(sim, '_in_sync_back', False) and not getattr(sim, '_gpu_states_in_sync', True):
                sim.sync_to_cpu()
        return getattr(instance, self.private_name)

    def __set__(self, instance, value):
        sim_ref = getattr(instance, '_sim', None)
        if sim_ref is not None:
            sim = sim_ref()
            if sim is not None and not getattr(sim, '_in_sync_back', False):
                sim._gpu_states_dirty = True
                sim._gpu_states_in_sync = True
        object.__setattr__(instance, self.private_name, value)


class MolMemristor:
    __slots__ = [
        'params', 'vSmooth', 'nSmooth', 'kappa', 'alpha', 'gScale', 'vTh',
        'vRefPos', 'vRefNeg', 'rateAsymmetry', 'nMax', 'kDischarge', 'rC',
        'f22StartVal', 'f22PeakVal', 'f22EndVal', 'f22PotScaleInv', 'f22DepScaleInv', 'f22Floor',
        '_n', '_modeState', '_scaleFactor', '_currentIScale', '_currentVScale', '_currentF22',
        'dataDir', 'lookups', '_cached_n_int_val', '_iScale', '_vScale', '_f22PotScaled', '_f22DepScaled',
        '_G', 'E_a', 'R_th', 'tau_th', '_T', '_sim', 'gamma', 'beta_alpha', 'beta_s',
        'vBypass', 'device_type', '_init_done',
        'rTopBlend', 'fDischarge', 'cDischarge', 'rBottomBlend', 'rLatency', 'k_overdrive', 'safe_max_vwrite',
        '_y_i', '_y_v', '_y_fp', '_y_fd'
    ]

    n = SynchronizedField('_n')
    modeState = SynchronizedField('_modeState')
    scaleFactor = SynchronizedField('_scaleFactor')
    T = SynchronizedField('_T')
    G = SynchronizedField('_G')
    currentF22 = SynchronizedField('_currentF22')
    currentIScale = SynchronizedField('_currentIScale')
    currentVScale = SynchronizedField('_currentVScale')
    iScale = SynchronizedField('_iScale')
    vScale = SynchronizedField('_vScale')
    f22DepScaled = SynchronizedField('_f22DepScaled')
    f22PotScaled = SynchronizedField('_f22PotScaled')
    _cached_n_int = SynchronizedField('_cached_n_int_val')
    _params_cache = {}
    _lookups_cache_dict = {}
    _initial_vals_cache = {}
    _lookups_cache = {}

    def __init__(self, device_type="default", dataDir=None, force_init=True, _fast_cache=None):
        object.__setattr__(self, '_init_done', False)
        self._sim = None
        self.device_type = device_type
        if _fast_cache is not None:
            if len(_fast_cache) == 5:
                (self.params, self.lookups, self.dataDir, const_tuple, val_tuple) = _fast_cache
                (self.vSmooth, self.nSmooth, self.kappa, self.alpha, self.gScale, self.vTh,
                 self.vRefPos, self.vRefNeg, self.rateAsymmetry, self.nMax, self.kDischarge, self.rC,
                 self.E_a, self.R_th, self.tau_th, self.gamma, self.beta_alpha, self.beta_s,
                 self.rTopBlend, self.fDischarge, self.cDischarge, self.rBottomBlend, self.rLatency,
                 self.k_overdrive, self.safe_max_vwrite, self.f22StartVal, self.f22PeakVal, self.f22EndVal,
                 self.f22PotScaleInv, self.f22DepScaleInv, self.vBypass, self.f22Floor,
                 self._y_i, self._y_v, self._y_fp, self._y_fd) = const_tuple
                (self._n, self._modeState, self._cached_n_int_val, self._iScale, self._vScale,
                 self._f22PotScaled, self._f22DepScaled, self._currentF22, self._currentIScale, self._currentVScale) = val_tuple
                self._scaleFactor = 1.0
                self._T = 298.15
                self._G = 0.0
                self._init_done = True
                return
            self.params, self.lookups, vals = _fast_cache
            self.vSmooth = self.params["vSmooth"]
            self.nSmooth = self.params["nSmooth"]
            self.kappa = self.params["kappa"]
            self.alpha = self.params["alpha"]
            self.gScale = self.params["gScale"]
            self.vTh = self.params["vTh"]
            self.vRefPos = self.params["vRefPos"]
            self.vRefNeg = self.params["vRefNeg"]
            self.rateAsymmetry = self.params.get("rate_asymmetry", 1.0)
            self.nMax = self.params["nMax"]
            self.kDischarge = self.params["kDischarge"]
            self.rC = self.params["rC"]
            self.E_a = self.params.get("E_a", 0.0)
            self.R_th = self.params.get("R_th", 0.0)
            self.tau_th = self.params.get("tau_th", 1e-9)
            self.gamma = self.params.get("gamma", 0.0)
            self.beta_alpha = self.params.get("beta_alpha", 0.0)
            self.beta_s = self.params.get("beta_s", 0.0)
            self.rTopBlend = self.params.get("rTopBlend", 0.050000)
            self.fDischarge = self.params.get("fDischarge", 0.374848)
            self.cDischarge = self.params.get("cDischarge", 0.0)
            self.rBottomBlend = self.params.get("rBottomBlend", 0.714488)
            self.rLatency = self.params.get("rLatency", 0.878506)
            self.k_overdrive = float(self.params.get("k_overdrive", 7.296635))
            self.safe_max_vwrite = self.params.get("safe_max_vwrite", None)
            self.f22StartVal = self.params["f22StartVal"]
            self.f22PeakVal = self.params["f22PeakVal"]
            self.f22EndVal = self.params["f22EndVal"]
            if "f22PotScaleInv" in self.params:
                self.f22PotScaleInv = self.params["f22PotScaleInv"]
                self.f22DepScaleInv = self.params["f22DepScaleInv"]
            else:
                self.f22PotScaleInv = 1.0 / (self.f22PeakVal - self.f22StartVal)
                self.f22DepScaleInv = 1.0 / (self.f22PeakVal - self.f22EndVal)
                self.params["f22PotScaleInv"] = self.f22PotScaleInv
                self.params["f22DepScaleInv"] = self.f22DepScaleInv

            # Pre-calculate vBypass
            if "vBypass" in self.params:
                self.vBypass = self.params["vBypass"]
            else:
                abs_pos = abs(self.vRefPos)
                abs_neg = abs(self.vRefNeg)
                vref_max = abs_pos if abs_pos > abs_neg else abs_neg
                denom = max(1e-6, vref_max - self.vTh + 1e-6)
                ratio = 1e-3 / (self.kappa if self.kappa > 1e-6 else 1e-6)
                term1 = math.pow(ratio, 1.0 / max(0.1, self.alpha))
                arg = (denom / max(1e-6, self.vSmooth)) * term1
                if arg > 1e-20:
                    vBypass_val = self.vTh + self.vSmooth * math.log(arg)
                    limit = self.vTh - 3.0 * self.vSmooth
                    self.vBypass = 0.0 if vBypass_val < 0.0 else (limit if vBypass_val > limit else vBypass_val)
                else:
                    self.vBypass = 0.0
                self.params["vBypass"] = self.vBypass

            self._n = self.params.get("nInit", 0)
            self._modeState = float(self.params.get("modeInit", 1.0 if self._n > 0 else 0.0))
            self.f22Floor = self.params.get("f22Floor", 0.0)
            self._scaleFactor = 1.0
            self._T = 298.15
            self._G = 0.0
            self.dataDir = dataDir if dataDir is not None else DEFAULT_DATA_DIR
            self._y_i = self.lookups["iScale"]
            self._y_v = self.lookups["vScale"]
            self._y_fp = self.lookups["f22PotMap"]
            self._y_fd = self.lookups["f22DepMap"]
            self._cached_n_int_val = vals['_cached_n_int']
            self._iScale = vals['iScale']
            self._vScale = vals['vScale']
            self._f22PotScaled = vals['f22PotScaled']
            self._f22DepScaled = vals['f22DepScaled']
            self._currentF22 = vals['currentF22']
            self._currentIScale = vals['currentIScale']
            self._currentVScale = vals['currentVScale']
            self._init_done = True
            return

        # Build a hashable cache key
        if isinstance(device_type, str):
            cache_key = device_type.lower()
        elif isinstance(device_type, dict):
            cache_key = tuple(
                (k, v) for k, v in sorted(device_type.items())
                if isinstance(v, (int, float, str, bool, type(None)))
            )
        else:
            cache_key = id(device_type)

        self.params = MolMemristor._params_cache.get(cache_key)
        if self.params is None:
            if dataDir is None:
                dataDir = DEFAULT_DATA_DIR
            self.params = get_device_config(device_type, dataDir)
            MolMemristor._params_cache[cache_key] = self.params
            
        # Unroll parameters to instance attributes for fast access
        self.vSmooth = self.params["vSmooth"]
        self.nSmooth = self.params["nSmooth"]
        self.kappa = self.params["kappa"]
        self.alpha = self.params["alpha"]
        self.gScale = self.params["gScale"]
        self.vTh = self.params["vTh"]
        self.vRefPos = self.params["vRefPos"]
        self.vRefNeg = self.params["vRefNeg"]
        self.rateAsymmetry = self.params.get("rate_asymmetry", 1.0)
        self.nMax = self.params["nMax"]
        self.kDischarge = self.params["kDischarge"]
        self.rC = self.params["rC"]
        self.E_a = self.params.get("E_a", 0.0)
        self.R_th = self.params.get("R_th", 0.0)
        self.tau_th = self.params.get("tau_th", 1e-9)
        self.gamma = self.params.get("gamma", 0.0)
        self.beta_alpha = self.params.get("beta_alpha", 0.0)
        self.beta_s = self.params.get("beta_s", 0.0)
        self.rTopBlend = self.params.get("rTopBlend", 0.050000)
        self.fDischarge = self.params.get("fDischarge", 0.374848)
        self.cDischarge = self.params.get("cDischarge", 0.0)
        self.rBottomBlend = self.params.get("rBottomBlend", 0.714488)
        self.rLatency = self.params.get("rLatency", 0.878506)
        self.k_overdrive = float(self.params.get("k_overdrive", 7.296635))
        self.safe_max_vwrite = self.params.get("safe_max_vwrite", None)
        self.f22StartVal = self.params["f22StartVal"]
        self.f22PeakVal = self.params["f22PeakVal"]
        self.f22EndVal = self.params["f22EndVal"]
        if "f22PotScaleInv" in self.params:
            self.f22PotScaleInv = self.params["f22PotScaleInv"]
            self.f22DepScaleInv = self.params["f22DepScaleInv"]
        else:
            self.f22PotScaleInv = 1.0 / (self.f22PeakVal - self.f22StartVal)
            self.f22DepScaleInv = 1.0 / (self.f22PeakVal - self.f22EndVal)
            self.params["f22PotScaleInv"] = self.f22PotScaleInv
            self.params["f22DepScaleInv"] = self.f22DepScaleInv

        # Pre-calculate vBypass
        if "vBypass" in self.params:
            self.vBypass = self.params["vBypass"]
        else:
            abs_pos = abs(self.vRefPos)
            abs_neg = abs(self.vRefNeg)
            vref_max = abs_pos if abs_pos > abs_neg else abs_neg
            denom = max(1e-6, vref_max - self.vTh + 1e-6)
            ratio = 1e-3 / (self.kappa if self.kappa > 1e-6 else 1e-6)
            term1 = math.pow(ratio, 1.0 / max(0.1, self.alpha))
            arg = (denom / max(1e-6, self.vSmooth)) * term1
            if arg > 1e-20:
                vBypass_val = self.vTh + self.vSmooth * math.log(arg)
                limit = self.vTh - 3.0 * self.vSmooth
                self.vBypass = 0.0 if vBypass_val < 0.0 else (limit if vBypass_val > limit else vBypass_val)
            else:
                self.vBypass = 0.0
            self.params["vBypass"] = self.vBypass

        # Initialize State
        self._n = self.params.get("nInit", 0)
        self._modeState = float(self.params.get("modeInit", 1.0 if self._n > 0 else 0.0))
        self._scaleFactor = 1.0
        self._T = 298.15
        self._G = 0.0

        # Validate physical parameter boundaries
        if self._T <= 0.0:
            raise ValueError(f"Unphysical device temperature T={self._T} K (absolute temperature must be strictly > 0 K).")
        if self.R_th < 0.0:
            raise ValueError(f"Unphysical thermal resistance R_th={self.R_th} K/W (thermal resistance cannot be negative).")
        if self.tau_th <= 0.0:
            raise ValueError(f"Unphysical thermal time constant tau_th={self.tau_th} s (time constant must be strictly > 0 s).")
        if self.gScale <= 0.0:
            raise ValueError(f"Unphysical conductance scale gScale={self.gScale} (conductance scale must be strictly positive).")

        # Cached conductance parameters for fast iteration
        self._currentIScale = 0.0
        self._currentVScale = 1.0
        self._currentF22 = 1.0
        self._f22PotScaled = 0.0
        self._f22DepScaled = 0.0

        # Load Lookup Tables
        self.dataDir = dataDir if dataDir is not None else DEFAULT_DATA_DIR
        self.lookups = MolMemristor._lookups_cache_dict.get(cache_key)
        if self.lookups is None:
            self.lookups = {}
            self._loadTables()
            MolMemristor._lookups_cache_dict[cache_key] = self.lookups
        self._y_i = self.lookups["iScale"]
        self._y_v = self.lookups["vScale"]
        self._y_fp = self.lookups["f22PotMap"]
        self._y_fd = self.lookups["f22DepMap"]
        
        # Physical minimum off-state barrier floor from depression table at n=0 (coordinate nMax)
        f22DepRaw_n0 = float(self._y_fd[-1]) if len(self._y_fd) > 0 else 0.0
        floor_val = (1.0 - f22DepRaw_n0 - self.f22EndVal) * self.f22DepScaleInv
        self.f22Floor = 0.0 if floor_val < 0.0 else floor_val
        self.params["f22Floor"] = self.f22Floor
        
        # Calibrate potentiation origin to match physical off-state barrier floor at n=0
        if len(self._y_fp) > 0 and self.f22Floor > 0.0:
            f22PotRaw_n0 = float(self._y_fp[0])
            f22StartVal_cal = (f22PotRaw_n0 - self.f22Floor * self.f22PeakVal) / (1.0 - self.f22Floor)
            self.f22StartVal = f22StartVal_cal
            self.f22PotScaleInv = 1.0 / (self.f22PeakVal - self.f22StartVal)
            self.params["f22StartVal"] = self.f22StartVal
            self.params["f22PotScaleInv"] = self.f22PotScaleInv
        
        # State parameter cache tracking
        vals = MolMemristor._initial_vals_cache.get(cache_key)
        if vals is not None:
            self._cached_n_int_val = vals['_cached_n_int']
            self._iScale = vals['iScale']
            self._vScale = vals['vScale']
            self._f22PotScaled = vals['f22PotScaled']
            self._f22DepScaled = vals['f22DepScaled']
            self._currentF22 = vals['currentF22']
            self._currentIScale = vals['currentIScale']
            self._currentVScale = vals['currentVScale']
        else:
            self._cached_n_int_val = int(self._n)
            self._forceUpdateConductanceParams()
            MolMemristor._initial_vals_cache[cache_key] = {
                '_cached_n_int': self._cached_n_int_val,
                'iScale': self._iScale,
                'vScale': self._vScale,
                'f22PotScaled': self._f22PotScaled,
                'f22DepScaled': self._f22DepScaled,
                'currentF22': self._currentF22,
                'currentIScale': self._currentIScale,
                'currentVScale': self._currentVScale
            }
        self._init_done = True
            
    def _loadTables(self):
        files = {
            "iScale": self.params.get("iScale_file"),
            "vScale": self.params.get("vScale_file"),
            "f22PotMap": self.params.get("f22PotMap_file"),
            "f22DepMap": self.params.get("f22DepMap_file")
        }

        x_uniform = None
        for key, path in files.items():
            if path is None:
                raise ValueError(f"Path for {key} lookup not specified in device parameters.")
            
            cached = MolMemristor._lookups_cache.get(path)
            if cached is not None:
                self.lookups[key] = cached
            else:
                from .sys_utils import load_physics_table
                data = load_physics_table(path)
                if x_uniform is None:
                    n_points = 10000
                    x_uniform = np.linspace(0.0, self.params["nMax"], n_points)
                # Uniformly oversample the PWL tables into an instantly indexable geometry
                interpolated = np.interp(x_uniform, data[:, 0], data[:, 1])
                MolMemristor._lookups_cache[path] = interpolated
                self.lookups[key] = interpolated

    def _refreshLookupCache(self, n_val, sDir, v_app=0.0):
        """Internal call to run jit_interp only when int(n) changes."""
        n_max = self.nMax
        if _fn_jit_interp_O1 is None:
            _ensure_jit()
        if _fn_jit_interp_O1 is not None:
            interp = _fn_jit_interp_O1
            self._iScale = interp(n_val, self._y_i, n_max)
            self._vScale = interp(n_val, self._y_v, n_max)
            f22DepRaw = interp(n_max - n_val, self._y_fd, n_max)
            f22_dep_val = (1.0 - f22DepRaw - self.f22EndVal) * self.f22DepScaleInv
            floor = getattr(self, 'f22Floor', 0.0)
            self._f22DepScaled = floor if f22_dep_val < floor else f22_dep_val
            f22PotRaw = interp(n_val, self._y_fp, n_max)
            f22_pot_val = (f22PotRaw - self.f22StartVal) * self.f22PotScaleInv
            if f22_pot_val > 0.0 and n_val > 16520.0:
                u_sat = (n_val - 16520.0) / 16520.0
                if u_sat > 1.0: u_sat = 1.0
                n_sm_eff = getattr(self, 'nSmooth', 150.0)
                beta_s = getattr(self, 'beta_s', 0.0)
                v_app = abs(v_app)
                v_ref = getattr(self, 'vRefPos', 0.90)
                v_t = getattr(self, 'vTh', 0.68)
                if v_app > v_ref and beta_s != 0.0:
                    d_v = v_ref - v_t + 1e-6
                    del_v = (v_app - v_ref) / (d_v if d_v > 1e-15 else 1e-15)
                    n_sm_eff = n_sm_eff * (1.0 + beta_s * math.tanh(del_v))
                    if n_sm_eff < 1.0: n_sm_eff = 1.0
                    elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                f22_pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
            floor = getattr(self, 'f22Floor', 0.0)
            self._f22PotScaled = floor if f22_pot_val < floor else f22_pot_val
        else:
            idx = int(min(max(0.0, n_val), n_max)) if n_max > 0 else 0
            idx_dep = int(min(max(0.0, n_max - n_val), n_max)) if n_max > 0 else 0
            self._iScale = float(self._y_i[idx]) if len(self._y_i) > idx else 1.0
            self._vScale = float(self._y_v[idx]) if len(self._y_v) > idx else 1.0
            f22DepRaw = float(self._y_fd[idx_dep]) if len(self._y_fd) > idx_dep else 0.0
            f22_dep_val = (1.0 - f22DepRaw - self.f22EndVal) * self.f22DepScaleInv
            floor = getattr(self, 'f22Floor', 0.0)
            self._f22DepScaled = floor if f22_dep_val < floor else f22_dep_val
            f22PotRaw = float(self._y_fp[idx]) if len(self._y_fp) > idx else 0.0
            f22_pot_val = (f22PotRaw - self.f22StartVal) * self.f22PotScaleInv
            if f22_pot_val > 0.0 and n_val > 16520.0:
                u_sat = (n_val - 16520.0) / 16520.0
                if u_sat > 1.0: u_sat = 1.0
                n_sm_eff = getattr(self, 'nSmooth', 150.0)
                beta_s = getattr(self, 'beta_s', 0.0)
                v_app = abs(v_app)
                v_ref = getattr(self, 'vRefPos', 0.90)
                v_t = getattr(self, 'vTh', 0.68)
                if v_app > v_ref and beta_s != 0.0:
                    d_v = v_ref - v_t + 1e-6
                    del_v = (v_app - v_ref) / (d_v if d_v > 1e-15 else 1e-15)
                    n_sm_eff = n_sm_eff * (1.0 + beta_s * math.tanh(del_v))
                    if n_sm_eff < 1.0: n_sm_eff = 1.0
                    elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                f22_pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
            floor = getattr(self, 'f22Floor', 0.0)
            self._f22PotScaled = floor if f22_pot_val < floor else f22_pot_val

    def _forceUpdateConductanceParams(self):
        """Force a recalculation of continuous states immediately."""
        self._cached_n_int_val = int(self._n) if (math.isfinite(self._n) and -2147483648 <= self._n <= 2147483647) else -1
        ms = self._modeState
        sDir = 0.0 if ms < 0.0 else (1.0 if ms > 1.0 else ms)
        self._refreshLookupCache(self._n, sDir)
        
        if sDir > 0.6:
            denom = self._f22DepScaled if self._f22DepScaled > 1e-6 else 1e-6
            self._scaleFactor = min(100.0, max(0.0, self._f22PotScaled / denom))

        f22 = self._f22PotScaled if sDir > 0.5 else (self._f22DepScaled * self._scaleFactor)
        floor = getattr(self, 'f22Floor', 0.0)
        self._currentF22 = floor if f22 < floor else f22
        self._currentIScale = self._iScale * self.gScale
        gamma_eff = self.gamma
        self._currentVScale = (self._vScale / (1.0 + gamma_eff * self._currentF22)) if gamma_eff != 0.0 else self._vScale

    def getIGeq(self, vApplied):
        if _fn_jit_getIGeq is None:
            _ensure_jit()
        return _fn_jit_getIGeq(vApplied, self._currentIScale, self._currentVScale, self._currentF22, self.rC)

    def step_predictor_corrector(self, dt, vApplied):
        """
        Dispatches to the 1-big-step, 2-small-step JIT simulator.
        Returns the (err evaluator, and the N, Mode, Scale, Temp for the ACCEPTED double-step)
        """
        if _fn_jit_predict_correct is None:
            _ensure_jit()
        err, n_d, mode_d, scale_d, T_d = _fn_jit_predict_correct(
            dt, vApplied, self._n, self._modeState, self._scaleFactor, self._T,
            self.vSmooth, self.nSmooth, self.kappa, self.alpha,
            self.vTh, self.vBypass, self.vRefPos, self.vRefNeg, self.nMax, self.kDischarge,
            self._f22DepScaled, self._f22PotScaled,
            self._currentIScale, self._currentVScale, self._currentF22, self.rC,
            self.E_a, self.R_th, self.tau_th,
            self.beta_alpha, self.beta_s, self.gamma, self._vScale,
            self.rateAsymmetry, self.rTopBlend, self.fDischarge, self.cDischarge, self.rBottomBlend, self.rLatency,
            getattr(self, 'k_overdrive', 0.0)
        )
        return err, n_d, mode_d, scale_d, T_d
        
    def commitState(self, n_new, mode_new, scale_new, T_new, vApplied=0.0):
        """
        Commits an ACCEPTED Predictor-Corrector timestep state back into the device
        cache natively. If a step is rejected, nothing happens because state was never overwritten!
        """
        mode_changed = (self._modeState > 0.5) != (mode_new > 0.5)
        n_int = int(n_new) if (math.isfinite(n_new) and -2147483648 <= n_new <= 2147483647) else -1
        state_changed = (n_int != self._cached_n_int_val)
        self._n = n_new
        self._modeState = mode_new
        self._scaleFactor = scale_new
        self._T = T_new
        sDir = 0.0 if mode_new < 0.0 else (1.0 if mode_new > 1.0 else mode_new)
        
        # --- Continuity Scale Factor Cache (Includes inline parameter caching) ---
        if state_changed or mode_changed:
            self._cached_n_int_val = n_int
            self._refreshLookupCache(self._n, sDir, abs(vApplied))

        if sDir > 0.6:
            denom = self._f22DepScaled if self._f22DepScaled > 1e-6 else 1e-6
            self._scaleFactor = min(100.0, max(0.0, self._f22PotScaled / denom))
            scale_new = self._scaleFactor
             
        # Inline the fast dependent variable calculation for the next solver timestep
        f22 = self._f22PotScaled if sDir > 0.5 else (self._f22DepScaled * scale_new)
        floor = getattr(self, 'f22Floor', 0.0)
        self._currentF22 = floor if f22 < floor else f22
        self._currentIScale = self._iScale * self.gScale
        gamma = self.gamma
        if gamma != 0.0:
            self._currentVScale = self._vScale / (1.0 + gamma * self._currentF22)
        else:
            self._currentVScale = self._vScale

        k_od = getattr(self, 'k_overdrive', 0.0)
        od_val = self.cDischarge * k_od
        od_clamp = 1.0 if od_val > 1.0 else (0.0 if od_val < 0.0 else od_val)
        scale_base = self.rBottomBlend if (self.rBottomBlend >= 0.05 and self.rBottomBlend <= 2.5) else 1.0
        n_base = 0.5 * self.nMax * (1.0 - od_clamp * (1.0 - scale_base))
        if self._modeState > 0.5 and self._n > 0.5 * self.nMax:
            d_vref_vth = self.vRefPos - self.vTh
            if vApplied > self.vRefPos and d_vref_vth > 1e-6:
                od_inst = (vApplied - self.vRefPos) / d_vref_vth
                if od_inst > self.cDischarge:
                    self.cDischarge = od_inst
        elif self._n <= n_base:
            self.cDischarge = 0.0
