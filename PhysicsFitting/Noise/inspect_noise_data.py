import os
import sys
import scipy.io as sio
import numpy as np

def inspect_file(filepath):
    print(f"=== Inspecting: {os.path.basename(filepath)} ===")
    if not os.path.isfile(filepath):
        print("  File not found!")
        return
    try:
        vars_info = sio.whosmat(filepath)
        print(f"  Variables found ({len(vars_info)}):")
        for name, shape, dtype in vars_info:
            print(f"    Variable: '{name}' | Shape: {shape} | Dtype: {dtype}")
        
        # Load only variable keys and basic stats without overloading RAM
        data = sio.loadmat(filepath, variable_names=[v[0] for v in vars_info[:5]])
        for name, shape, dtype in vars_info[:5]:
            if name in data and isinstance(data[name], np.ndarray):
                arr = data[name]
                print(f"    Stats for '{name}': min={np.nanmin(arr):.4e}, max={np.nanmax(arr):.4e}, mean={np.nanmean(arr):.4e}, std={np.nanstd(arr):.4e}")
    except Exception as e:
        print(f"  Error loading with scipy.io: {e}")
        # Try h5py fallback if v7.3
        try:
            import h5py
            with h5py.File(filepath, 'r') as h5f:
                print(f"  HDF5 Keys found ({len(h5f.keys())}): {list(h5f.keys())}")
        except Exception as h5e:
            print(f"  HDF5 fallback also failed: {h5e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f1 = os.path.join(base_dir, "Write_Noise_Gd1.mat")
    f2 = os.path.join(base_dir, "Read_Noise_3000_Devices", "devices_3000_5x5.mat")
    d2 = sio.loadmat(f2)
    print("\n--- devices_3000_5x5 Details ---")
    print(f"  V_read: {d2['V_read'][0,0]} V | f_pulse: {d2['f_pulse'][0,0]} Hz | N_devices: {d2['N_devices'][0,0]}")
    print(f"  G_target levels (S): {d2['G_target'][0]}")
    
    # Calculate mean read noise relative to G across the 25 levels
    mean_sigma_G = np.nanmean(d2['sigma_G'], axis=0)
    g_targets = d2['G_target'][0]
    rel_read_noise = mean_sigma_G / g_targets
    print("\n  Target G (S) -> Mean Sigma_G (S) -> Relative Noise (Sigma/G):")
    for g, s, r in zip(g_targets[::3], mean_sigma_G[::3], rel_read_noise[::3]):
        print(f"    G = {g:.4e} S | Sigma_read = {s:.4e} S | Rel_Noise = {r*100:.2f}%")

    # Inspect D2D variation (actual programmed conductance vs target)
    g_actual = d2['G_actual']
    d2d_std_per_level = np.nanstd(g_actual, axis=0)
    rel_d2d_noise = d2d_std_per_level / g_targets
    print("\n  Target G (S) -> D2D Programming Std (S) -> Relative D2D Spread:")
    for g, s, r in zip(g_targets[::3], d2d_std_per_level[::3], rel_d2d_noise[::3]):
        print(f"    G = {g:.4e} S | D2D_Std = {s:.4e} S | Rel_D2D = {r*100:.2f}%")
        
    print("\n--- Write_Noise_Gd1 Details ---")
    # Inspect pulse-to-pulse delta G across the 2000 cycles
    # Load small subset of Gd1: first 500 pulses across all 2000 cycles
    d1 = sio.loadmat(f1)
    gd1 = d1['Gd1']
    print(f"  Gd1 shape: {gd1.shape} | Total pulses per cycle: {gd1.shape[0]} | Total cycles: {gd1.shape[1]}")
    # Compute delta G across pulses
    delta_g = np.diff(gd1, axis=0) # shape (16519, 2000)
    mean_delta_g = np.nanmean(delta_g, axis=1)
    std_delta_g = np.nanstd(delta_g, axis=1)
    print(f"  Mean delta_G per pulse (across pulses): min={np.nanmin(mean_delta_g):.4e} S, max={np.nanmax(mean_delta_g):.4e} S")
    print(f"  Std of delta_G across 2000 cycles: min={np.nanmin(std_delta_g):.4e} S, max={np.nanmax(std_delta_g):.4e} S")

