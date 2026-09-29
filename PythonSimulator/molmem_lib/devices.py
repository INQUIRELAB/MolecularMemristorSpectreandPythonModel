import os
from .sys_utils import resolve_physics_data_file as _resolve_physics_file, PhysicsTag

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_RU_AZO_CONFIG = {
    "vSmooth": 0.103776,
    "nSmooth": 17.06,
    "kappa": 1.172865e+07,
    "alpha": 0.931756,
    "gScale": 6.227303,
    "vTh": 0.466409,
    "vRefPos": 0.900000,
    "vRefNeg": 0.750000,
    "nMax": 33040,
    "nInit": 0.000000,
    "kDischarge": 362.88,
    "rC": 0.008033,
    "E_a": 0.155923,
    "R_th": 52.14,
    "tau_th": 1.278e-07,
    "gamma": -0.012230,
    "beta_alpha": 0.125186,
    "beta_s": -4.382853,
    "rate_asymmetry": 0.985694,
    "rTopBlend": 0.050000,
    "fDischarge": 0.374848,
    "rBottomBlend": 0.714488,
    "rLatency": 0.878506,
    "k_overdrive": 7.296635,
    "safe_max_vwrite": 2.27,
    "f22StartVal": 0.12037256,
    "f22PeakVal": 0.87791974,
    "f22EndVal": 0.13955489,
    # Lookup table files / abstract bundle tags
    "iScale_file": _resolve_physics_file(DEFAULT_DATA_DIR, PhysicsTag.I_SCALE),
    "vScale_file": _resolve_physics_file(DEFAULT_DATA_DIR, PhysicsTag.V_SCALE),
    "f22PotMap_file": _resolve_physics_file(DEFAULT_DATA_DIR, PhysicsTag.F22_POT_MAP),
    "f22DepMap_file": _resolve_physics_file(DEFAULT_DATA_DIR, PhysicsTag.F22_DEP_MAP)
}

_DEFAULT_CONFIG = _RU_AZO_CONFIG.copy()

def default_device(data_dir=None):
    """
    Returns parameters and PWL paths for the default MolMemristor (mirrors Ru_azo).
    """
    if data_dir is None or data_dir == DEFAULT_DATA_DIR:
        return _DEFAULT_CONFIG.copy()
        
    return Ru_azo(data_dir)

# =============================================================================
# DEVICE REGISTRY MAP
# =============================================================================
def Ru_azo(data_dir=None):
    """
    Returns optimized parameters and PWL paths for the Ru_azo device (Clustered Multi-Sweep Fit).
    """
    if data_dir is None or data_dir == DEFAULT_DATA_DIR:
        return _RU_AZO_CONFIG.copy()
        
    return {
        "vSmooth": 0.103776,
        "nSmooth": 17.06,
        "kappa": 1.172865e+07,
        "alpha": 0.931756,
        "gScale": 6.227303,
        "vTh": 0.466409,
        "vRefPos": 0.900000,
        "vRefNeg": 0.750000,
        "nMax": 33040,
        "nInit": 0.000000,
        "kDischarge": 362.88,
        "rC": 0.008033,
        "E_a": 0.155923,
        "R_th": 52.14,
        "tau_th": 1.278e-07,
        "gamma": -0.012230,
        "beta_alpha": 0.125186,
        "beta_s": -4.382853,
        "rate_asymmetry": 0.985694,
        "rTopBlend": 0.050000,
        "fDischarge": 0.374848,
        "rBottomBlend": 0.714488,
        "rLatency": 0.878506,
        "k_overdrive": 7.296635,
        "safe_max_vwrite": 2.27,
        "f22StartVal": 0.12037256,
        "f22PeakVal": 0.87791974,
        "f22EndVal": 0.13955489,
        # Lookup table files / abstract bundle tags
        "iScale_file": _resolve_physics_file(data_dir, PhysicsTag.I_SCALE),
        "vScale_file": _resolve_physics_file(data_dir, PhysicsTag.V_SCALE),
        "f22PotMap_file": _resolve_physics_file(data_dir, PhysicsTag.F22_POT_MAP),
        "f22DepMap_file": _resolve_physics_file(data_dir, PhysicsTag.F22_DEP_MAP)
    }

_CONFIG_PRESETS = {
    "default": _DEFAULT_CONFIG,
    "ru_azo": _RU_AZO_CONFIG,
}

DEVICE_REGISTRY = {
    "default": default_device,
    "ru_azo": Ru_azo,
}

def get_device_config(device_type="default", data_dir=None):
    """
    Resolves the device type (string preset, dictionary, or builder function) 
    into a configuration dictionary containing device parameters and lookup paths.
    """
    if isinstance(device_type, dict):
        return device_type
    if isinstance(device_type, str):
        dev_lower = device_type.lower()
        if data_dir is None or data_dir == DEFAULT_DATA_DIR:
            preset = _CONFIG_PRESETS.get(dev_lower)
            if preset is not None:
                return preset.copy()
        getter = DEVICE_REGISTRY.get(dev_lower)
        if getter is not None:
            return getter(data_dir)
        raise ValueError(f"Unknown device type preset: {device_type}")
    elif callable(device_type):
        return device_type(data_dir)
    else:
        raise TypeError("device_type must be a string preset name, a callable returning a config, or a dictionary of parameters.")
