import os
import sys
import json
import numpy as np
import scipy.io as sio

def generate_pwl_tables():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f_write = os.path.join(base_dir, "Write_Noise_Gd1.mat")
    f_read = os.path.join(base_dir, "Read_Noise_3000_Devices", "devices_3000_5x5.mat")
    
    out_dir = os.path.join(base_dir, "generated_pwl")
    os.makedirs(out_dir, exist_ok=True)
    
    print("==========================================================")
    print("--- Generating Universal Noise PWL Tables from Empirical Data ---")
    print("==========================================================")
    
    # -------------------------------------------------------------
    # 1. READ NOISE PWL: sigma_read / G vs. g_norm
    # -------------------------------------------------------------
    print("\n[+] Processing Read Noise (3,000 Devices across 25 Target Levels)...")
    d_read = sio.loadmat(f_read)
    g_targets = d_read['G_target'][0]  # shape (25,)
    sigma_G = d_read['sigma_G']        # shape (3000, 25)
    g_actual = d_read['G_actual']      # shape (3000, 25)
    v_read = float(d_read['V_read'][0, 0])
    n_devices = int(d_read['N_devices'][0, 0])
    
    # Sort by target conductance
    sort_idx = np.argsort(g_targets)
    g_sorted = g_targets[sort_idx]
    sigma_sorted = np.nanmean(sigma_G, axis=0)[sort_idx]
    
    # Dimensionless normalized conductance: g_norm in [0.0, 1.0]
    g_min = g_sorted[0]
    g_max = g_sorted[-1]
    g_norm_read = (g_sorted - g_min) / (g_max - g_min)
    
    # Relative read noise: sigma_read / G
    rel_read_noise = sigma_sorted / np.maximum(g_sorted, 1e-12)
    
    # Ensure boundary points exist at exactly 0.0 and 1.0
    pwl_read_x = []
    pwl_read_y = []
    if g_norm_read[0] > 0.0:
        pwl_read_x.append(0.0)
        pwl_read_y.append(float(rel_read_noise[0]))
    for x, y in zip(g_norm_read, rel_read_noise):
        pwl_read_x.append(float(x))
        pwl_read_y.append(float(y))
    if pwl_read_x[-1] < 1.0:
        pwl_read_x.append(1.0)
        pwl_read_y.append(float(rel_read_noise[-1]))
        
    read_pwl_path = os.path.join(out_dir, "read_noise.pwl")
    with open(read_pwl_path, "w") as f:
        f.write("# Universal Read Noise PWL (Dimensionless)\n")
        f.write("# Col 1: Normalized Conductance g_norm in [0.0, 1.0]\n")
        f.write("# Col 2: Relative Read Current/Conductance Std Dev (sigma_read / G)\n")
        f.write(f"# Measured at V_read = {v_read} V across {n_devices} devices\n")
        for x, y in zip(pwl_read_x, pwl_read_y):
            f.write(f"{x:.8f} {y:.8e}\n")
    print(f"  [OK] Exported: {read_pwl_path} ({len(pwl_read_x)} points)")

    # -------------------------------------------------------------
    # 2. D2D VARIATION PWL: Empirical Inverse-CDF (Quantile Map)
    # -------------------------------------------------------------
    print("\n[+] Processing D2D Spatial Mismatch across 3,000 Crosspoints...")
    # Calculate relative deviation (G_actual - G_target) / G_target for each device
    # Pool across target levels to extract the full empirical quantile function
    rel_d2d_pool = []
    for j in range(len(g_targets)):
        target_val = g_targets[j]
        dev_vals = g_actual[:, j]
        valid = np.isfinite(dev_vals) & (dev_vals > 0)
        if np.any(valid):
            # Normalize to zero-mean, unit-variance empirical z-score
            diffs = dev_vals[valid] - target_val
            rel_diffs = diffs / target_val
            rel_d2d_pool.extend(rel_diffs)
            
    rel_d2d_pool = np.array(rel_d2d_pool)
    # Generate 101 quantiles from 0.001 to 0.999
    quantiles = np.linspace(0.001, 0.999, 101)
    d2d_quantile_vals = np.quantile(rel_d2d_pool, quantiles)
    
    # Boundary padding
    q_x = [0.0] + list(quantiles) + [1.0]
    q_y = [float(d2d_quantile_vals[0])] + [float(v) for v in d2d_quantile_vals] + [float(d2d_quantile_vals[-1])]
    
    d2d_pwl_path = os.path.join(out_dir, "d2d_mismatch_quantile.pwl")
    with open(d2d_pwl_path, "w") as f:
        f.write("# Universal D2D Spatial Variation Quantile PWL (Dimensionless Inverse-CDF)\n")
        f.write("# Col 1: Uniform Random Quantile u in [0.0, 1.0]\n")
        f.write("# Col 2: Relative Device Parameter Perturbation ((P - P_nom) / P_nom)\n")
        f.write(f"# Fitted across {n_devices} physical crosspoints (3000 x 25 = 75,000 measurements)\n")
        for x, y in zip(q_x, q_y):
            f.write(f"{x:.6f} {y:.8e}\n")
    print(f"  [OK] Exported: {d2d_pwl_path} ({len(q_x)} points)")

    # -------------------------------------------------------------
    # 3. WRITE NOISE PWL: rel_sigma_write vs. g_norm across 2,000 Cycles
    # -------------------------------------------------------------
    print("\n[+] Processing Write Noise across 2,000 Continuous Programming Cycles...")
    d_write = sio.loadmat(f_write)
    gd1 = d_write['Gd1'] # shape (16520, 2000)
    
    # Mean conductance profile across pulses
    mean_g_profile = np.nanmean(gd1, axis=1)
    g_write_min = np.nanmin(mean_g_profile)
    g_write_max = np.nanmax(mean_g_profile)
    
    # Pulse-to-pulse delta G across all 2000 cycles
    delta_g = np.diff(gd1, axis=0) # shape (16519, 2000)
    mean_delta_g = np.nanmean(delta_g, axis=1) # shape (16519,)
    std_delta_g = np.nanstd(delta_g, axis=1)   # shape (16519,)
    
    # Mean conductance at each step:
    g_mid = 0.5 * (mean_g_profile[:-1] + mean_g_profile[1:])
    g_norm_write_raw = (g_mid - g_write_min) / (g_write_max - g_write_min)
    
    # Bin into 50 conductance bins across [0.0, 1.0]
    n_bins = 50
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    pwl_write_x = []
    pwl_write_y = []
    
    # Global average relative write noise
    global_rel_write = float(np.nanmean(std_delta_g) / np.nanmean(np.abs(mean_delta_g)))
    
    for b in range(n_bins):
        b_min = bin_edges[b]
        b_max = bin_edges[b + 1]
        b_center = 0.5 * (b_min + b_max)
        in_bin = (g_norm_write_raw >= b_min) & (g_norm_write_raw < b_max)
        if np.sum(in_bin) > 5:
            bin_std = np.nanmean(std_delta_g[in_bin])
            bin_mean = np.nanmean(np.abs(mean_delta_g[in_bin]))
            rel_noise = bin_std / max(1e-12, bin_mean)
            pwl_write_x.append(float(b_center))
            pwl_write_y.append(float(rel_noise))
            
    if not pwl_write_x or pwl_write_x[0] > 0.0:
        pwl_write_x = [0.0] + pwl_write_x
        pwl_write_y = [pwl_write_y[0] if pwl_write_y else global_rel_write] + pwl_write_y
    if pwl_write_x[-1] < 1.0:
        pwl_write_x.append(1.0)
        pwl_write_y.append(pwl_write_y[-1] if pwl_write_y else global_rel_write)
        
    write_pwl_path = os.path.join(out_dir, "write_noise.pwl")
    with open(write_pwl_path, "w") as f:
        f.write("# Universal Write Noise PWL (Dimensionless)\n")
        f.write("# Col 1: Normalized Conductance g_norm in [0.0, 1.0]\n")
        f.write("# Col 2: Relative Write Noise Std Dev (sigma_delta_G / |mean_delta_G|)\n")
        f.write(f"# Extracted across {gd1.shape[1]} consecutive cycles ({gd1.shape[0]} pulses/cycle)\n")
        for x, y in zip(pwl_write_x, pwl_write_y):
            f.write(f"{x:.6f} {y:.8e}\n")
    print(f"  [OK] Exported: {write_pwl_path} ({len(pwl_write_x)} points)")

    # -------------------------------------------------------------
    # 4. EXPORT SUMMARY JSON
    # -------------------------------------------------------------
    summary_data = {
        "dataset_metadata": {
            "source_read_file": os.path.relpath(f_read, base_dir),
            "source_write_file": os.path.relpath(f_write, base_dir),
            "v_read_volts": v_read,
            "n_tested_devices": n_devices,
            "n_write_cycles": int(gd1.shape[1]),
            "pulses_per_write_cycle": int(gd1.shape[0])
        },
        "empirical_parameters": {
            "read_noise_floor_Siemens": float(sigma_sorted[0]),
            "read_noise_max_Siemens": float(sigma_sorted[-1]),
            "d2d_mean_std_Siemens": float(np.mean(np.nanstd(g_actual, axis=0))),
            "write_pulse_mean_step_Siemens": float(np.nanmean(mean_delta_g)),
            "write_pulse_mean_std_Siemens": float(np.nanmean(std_delta_g)),
            "global_relative_write_noise": global_rel_write
        },
        "generated_pwl_files": {
            "read_noise": os.path.relpath(read_pwl_path, base_dir),
            "write_noise": os.path.relpath(write_pwl_path, base_dir),
            "d2d_mismatch_quantile": os.path.relpath(d2d_pwl_path, base_dir)
        }
    }
    
    summary_path = os.path.join(out_dir, "noise_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  [OK] Exported Metadata Summary: {summary_path}")
    
    print("\n==========================================================")
    print("--- Noise PWL Generation Complete ---")
    print(f"Directory: {out_dir}")
    print("==========================================================")

if __name__ == "__main__":
    generate_pwl_tables()
