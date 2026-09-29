import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Add PythonSimulator to sys.path so we can import molmem_lib
fitting_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(fitting_dir)
sys.path.append(os.path.abspath(os.path.join(project_root, 'PythonSimulator')))

from molmem_lib import MolmemSimulator
from molmem_lib.sources import PulseSource

def get_default_ranges(data_dir=None):
    t_period = 160e-9
    vpot = PulseSource(node=None, amp=0.9, period=t_period, width=80e-9)
    vdep = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
    # 1. Non-saturation
    sim_ns = MolmemSimulator()
    sim_ns.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_ns = 33040.0 * t_period
    t_switch_ns = t_stop_ns / 2.0
    def v_pulse_train_ns(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
    sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=[t_switch_ns])
    sim_ns.addDcSource(col=1, voltage=0.0)
    sim_ns.run(tEnd=t_stop_ns, min_dt=1e-12, tol=1e-3)
    g_ns = sim_ns.getConductance(row=1, col=1) * 1e3
    
    # 2. Saturation
    sim_sat = MolmemSimulator()
    sim_sat.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_sat = 52000.0 * t_period
    t_switch_sat = t_stop_sat / 2.0
    def v_pulse_train_sat(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
    sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=[t_switch_sat])
    sim_sat.addDcSource(col=1, voltage=0.0)
    sim_sat.run(tEnd=t_stop_sat, min_dt=1e-12, tol=1e-3)
    g_sat = sim_sat.getConductance(row=1, col=1) * 1e3
    
    # 3. 1.22V Saturation
    sim_122 = MolmemSimulator()
    sim_122.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    t_stop_122 = 33040.0 * t_period
    t_switch_122 = t_stop_122 / 2.0
    vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
    def v_pulse_train_122(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
    sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep], edge_points=[t_switch_122])
    sim_122.addDcSource(col=1, voltage=0.0)
    sim_122.run(tEnd=t_stop_122, min_dt=1e-12, tol=1e-3)
    g_122 = sim_122.getConductance(row=1, col=1) * 1e3
    
    return (np.min(g_ns), np.max(g_ns)), (np.min(g_sat), np.max(g_sat)), (np.min(g_122), np.max(g_122))

def get_dynamic_ticks(max_p):
    # We want ticks at 0, max_p / 4, max_p / 2, 3 * max_p / 4, and max_p
    vals = [0, max_p / 4, max_p / 2, 3 * max_p / 4, max_p]
    vals = [int(round(v)) for v in vals]
    
    labels = []
    for v in vals:
        if v == 0:
            labels.append("0")
        elif v == vals[2]: # Switch point (e.g. 16.5k or 16520)
            if v >= 1000:
                labels.append(f"{v/1000:.1f}k")
            else:
                labels.append(str(v))
        else:
            if v >= 1000:
                labels.append(f"{int(round(v/1000))}k")
            else:
                labels.append(str(v))
    return vals, labels

def main():
    print("\n" + "="*64)
    print("  STAGE: Local Fitted Device Comparison & Plotting")
    print("="*64 + "\n")
    
    # 1. Scan for completed fit preset text files
    detected_presets = []
    for entry in os.scandir(fitting_dir):
        if entry.is_file() and "_preset_devices_py" in entry.name and entry.name.endswith(".txt"):
            # Extract device name from prefix, e.g. "Ru_azo" from "Ru_azo_preset_devices_py.txt" or "Ru_azo_preset_devices_py_subset_SAT.txt"
            device_name = entry.name.split("_preset_devices_py")[0]
            detected_presets.append({
                "device_name": device_name,
                "file_name": entry.name,
                "file_path": entry.path
            })
            
    detected_presets.sort(key=lambda d: d["file_name"])
    
    if not detected_presets:
        print("[Error] No valid device preset text files found.")
        print("Please run 'fit_device_subset.py' or 'fit_device_cluster.py' first to create a preset file.")
        sys.exit(1)
        
    print("Detected completed device fits:")
    for idx, preset in enumerate(detected_presets):
        print(f" [{idx + 1}] {preset['device_name']} ({preset['file_name']})")
        
    if len(detected_presets) == 1:
        print(f"\nAuto-selecting the only available device fit: {detected_presets[0]['device_name']} ({detected_presets[0]['file_name']})")
        sel_idx = 0
    else:
        try:
            selection = input(f"\nSelect a device to plot (1-{len(detected_presets)}): ").strip()
            sel_idx = int(selection) - 1
            if sel_idx < 0 or sel_idx >= len(detected_presets):
                raise ValueError
        except (ValueError, KeyboardInterrupt, EOFError):
            print("Invalid selection. Exiting.")
            sys.exit(1)
        
    selected_preset = detected_presets[sel_idx]
    device_name = selected_preset["device_name"]
    preset_path = selected_preset["file_path"]
    
    print(f"\nSelected: {device_name}")
    print(f"Loading parameter configuration from {preset_path}...")
    
    # Execute the local text file to load the builder function
    with open(preset_path, "r") as f:
        code = f.read()
        
    local_vars = {}
    try:
        exec(code, {"os": os}, local_vars)
    except Exception as e:
        print(f"[Error] Failed to parse preset file: {e}")
        sys.exit(1)
        
    callable_funcs = [v for k, v in local_vars.items() if callable(v)]
    if not callable_funcs:
        print("[Error] No builder function found in preset file.")
        sys.exit(1)
        
    device_builder = callable_funcs[0]
    
    # Resolve the data folder in python simulator
    data_dir = os.path.join(project_root, "PythonSimulator", "molmem_lib", "data")
    device_config = device_builder(data_dir=data_dir)
    
    # 2. Locate experimental CSV data in the device's subdirectory
    dev_sub_dir = ""
    for entry in os.scandir(fitting_dir):
        if entry.is_dir() and entry.name.lower() == device_name.lower():
            dev_sub_dir = entry.path
            break
            
    if not dev_sub_dir:
        print(f"[Error] Subdirectory '{device_name}' not found under {fitting_dir}.")
        sys.exit(1)
        
    csv_path_ns = ""
    csv_path_sat = ""
    csv_path_122 = ""
    for file in os.scandir(dev_sub_dir):
        if file.is_file() and file.name.lower().endswith(".csv"):
            if "nonsaturation" in file.name.lower():
                csv_path_ns = file.path
            elif "1.22vsaturation" in file.name.lower():
                csv_path_122 = file.path
            elif "saturation" in file.name.lower():
                csv_path_sat = file.path
                
    if not csv_path_ns or not csv_path_sat or not csv_path_122:
        print(f"[Error] Subdirectory '{dev_sub_dir}' must contain Non-Saturation, Saturation, and 1.22VSaturation CSV files.")
        print(f"Found non-sat: {csv_path_ns}, sat: {csv_path_sat}, 1.22v: {csv_path_122}")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # 3. LOAD & PROCESS EXPERIMENTAL DATA
    # -------------------------------------------------------------------------
    # A. Saturation experimental data loading and normalization
    print(f"\nLoading Saturation experimental data from {csv_path_sat}...")
    df_sat = pd.read_csv(csv_path_sat, comment='#')
    pulses_exp_sat_raw = df_sat["X_value"] * 520.0
    x_sat_min = pulses_exp_sat_raw.min()
    x_sat_max = pulses_exp_sat_raw.max()
    pulses_exp_sat = (pulses_exp_sat_raw - x_sat_min) / (x_sat_max - x_sat_min) * 52000.0
    cond_exp_sat_ms_raw = df_sat["Y_value"] / 0.5
    
    # Scale empirical data to match the default model ranges
    ns_range, sat_range, range_122 = get_default_ranges(data_dir=data_dir)
    
    c_sat_raw_min = cond_exp_sat_ms_raw.min()
    c_sat_raw_max = cond_exp_sat_ms_raw.max()
    cond_exp_sat_ms = sat_range[0] + ((cond_exp_sat_ms_raw - c_sat_raw_min) / (c_sat_raw_max - c_sat_raw_min)) * (sat_range[1] - sat_range[0])
    
    # Saturation PCHIP
    sort_sat = np.argsort(pulses_exp_sat)
    p_sat_sorted = pulses_exp_sat.iloc[sort_sat].values
    c_sat_sorted = cond_exp_sat_ms.iloc[sort_sat].values
    u_sat_idx = [0]
    for i in range(1, len(p_sat_sorted)):
        if p_sat_sorted[i] > p_sat_sorted[u_sat_idx[-1]] + 1e-5:
            u_sat_idx.append(i)
    p_sat_unique = p_sat_sorted[u_sat_idx]
    c_sat_unique = c_sat_sorted[u_sat_idx]
    pchip_sat = PchipInterpolator(p_sat_unique, c_sat_unique)
    p_sat_dense = np.arange(p_sat_unique[0], p_sat_unique[-1] + 1.0, 1.0)
    c_sat_dense = pchip_sat(p_sat_dense)
    
    # B. Non-Saturation experimental data loading and normalization
    print(f"Loading Non-Saturation experimental data from {csv_path_ns}...")
    df_ns = pd.read_csv(csv_path_ns, comment='#')
    pulses_exp_ns_raw = df_ns["X_value"] * 330.4
    x_ns_min = pulses_exp_ns_raw.min()
    x_ns_max = pulses_exp_ns_raw.max()
    pulses_exp_ns = (pulses_exp_ns_raw - x_ns_min) / (x_ns_max - x_ns_min) * 33040.0
    cond_exp_ns_ms_raw = df_ns["Y_value"] / 0.5
    c_ns_raw_min = cond_exp_ns_ms_raw.min()
    c_ns_raw_max = cond_exp_ns_ms_raw.max()
    cond_exp_ns_ms = ns_range[0] + ((cond_exp_ns_ms_raw - c_ns_raw_min) / (c_ns_raw_max - c_ns_raw_min)) * (ns_range[1] - ns_range[0])
    
    # Non-Saturation PCHIP
    sort_ns = np.argsort(pulses_exp_ns)
    p_ns_sorted = pulses_exp_ns.iloc[sort_ns].values
    c_ns_sorted = cond_exp_ns_ms.iloc[sort_ns].values
    u_ns_idx = [0]
    for i in range(1, len(p_ns_sorted)):
        if p_ns_sorted[i] > p_ns_sorted[u_ns_idx[-1]] + 1e-5:
            u_ns_idx.append(i)
    p_ns_unique = p_ns_sorted[u_ns_idx]
    c_ns_unique = c_ns_sorted[u_ns_idx]
    pchip_ns = PchipInterpolator(p_ns_unique, c_ns_unique)
    p_ns_dense = np.arange(p_ns_unique[0], p_ns_unique[-1] + 1.0, 1.0)
    c_ns_dense = pchip_ns(p_ns_dense)

    # C. 1.22V Saturation experimental data loading and normalization
    print(f"Loading 1.22V Saturation experimental data from {csv_path_122}...")
    df_122 = pd.read_csv(csv_path_122, comment='#')
    pulses_exp_122_raw = df_122["X_value"]
    x_122_min = pulses_exp_122_raw.min()
    x_122_max = pulses_exp_122_raw.max()
    pulses_exp_122 = (pulses_exp_122_raw - x_122_min) / (x_122_max - x_122_min) * 33040.0
    cond_exp_122_ms_raw = df_122["Y_value"] / 0.5
    c_122_raw_min = cond_exp_122_ms_raw.min()
    c_122_raw_max = cond_exp_122_ms_raw.max()
    cond_exp_122_ms = range_122[0] + ((cond_exp_122_ms_raw - c_122_raw_min) / (c_122_raw_max - c_122_raw_min)) * (range_122[1] - range_122[0])
    
    # 1.22V Saturation PCHIP
    sort_122 = np.argsort(pulses_exp_122)
    p_122_sorted = pulses_exp_122.iloc[sort_122].values
    c_122_sorted = cond_exp_122_ms.iloc[sort_122].values
    u_122_idx = [0]
    for i in range(1, len(p_122_sorted)):
        if p_122_sorted[i] > p_122_sorted[u_122_idx[-1]] + 1e-5:
            u_122_idx.append(i)
    p_122_unique = p_122_sorted[u_122_idx]
    c_122_unique = c_122_sorted[u_122_idx]
    pchip_122 = PchipInterpolator(p_122_unique, c_122_unique)
    p_122_dense = np.arange(p_122_unique[0], p_122_unique[-1] + 1.0, 1.0)
    c_122_dense = pchip_122(p_122_dense)

    # -------------------------------------------------------------------------
    # 4. RUN SIMULATIONS
    # -------------------------------------------------------------------------
    t_period = 160e-9
    from molmem_lib.sources import PulseSource
    vpot = PulseSource(node=None, amp=0.9, period=t_period, width=80e-9)
    vdep = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
    # A. Saturation Sweep (Dynamic pulse count)
    max_p_sat = np.max(pulses_exp_sat.values)
    t_switch_sat = (max_p_sat / 2.0) * t_period
    t_stop_sat = max_p_sat * t_period
    print(f"\nRunning transient physics simulation for Saturation case ({max_p_sat:.0f} pulses)...")
    sim_sat = MolmemSimulator()
    device_config_sat = device_config.copy()
    device_config_sat["nInit"] = 0.0
    sim_sat.addCrossbarMatrix(device_type=device_config_sat, multiThread=False, rows=1, cols=1)
    
    def v_pulse_train_sat(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
        
    pulses_dense_sat = np.linspace(1, max_p_sat, int(max_p_sat))
    edge_points_sat = np.concatenate(([t_switch_sat], (pulses_dense_sat - 0.05) * t_period))
    sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=edge_points_sat.tolist())
    sim_sat.addDcSource(col=1, voltage=0.0)
    sim_sat.run(tEnd=t_stop_sat, min_dt=1e-11, tol=3e-3)
    
    t_arr_sat = sim_sat.getTime()
    g_sim_sat = sim_sat.getConductance(row=1, col=1)
    n_sim_sat = t_arr_sat / t_period
    g_sim_sat_ms = g_sim_sat * 1e3
    
    # Print state statistics for Saturation
    n_physics_sat = sim_sat.getState(row=1, col=1)
    idx_max_sat = np.argmax(g_sim_sat_ms)
    print(f"  [State n Info - Saturation]")
    print(f"    Beginning: n = {n_physics_sat[0]:.2f} (G = {g_sim_sat_ms[0]:.3f} mS)")
    print(f"    Max Cond:  n = {n_physics_sat[idx_max_sat]:.2f} (G = {g_sim_sat_ms[idx_max_sat]:.3f} mS)")
    print(f"    End:       n = {n_physics_sat[-1]:.2f} (G = {g_sim_sat_ms[-1]:.3f} mS)\n")
    
    # B. Non-Saturation Sweep (Dynamic pulse count)
    max_p_ns = np.max(pulses_exp_ns.values)
    t_switch_ns = (max_p_ns / 2.0) * t_period
    t_stop_ns = max_p_ns * t_period
    print(f"Running transient physics simulation for Non-Saturation case ({max_p_ns:.0f} pulses)...")
    sim_ns = MolmemSimulator()
    device_config_ns = device_config.copy()
    device_config_ns["nInit"] = 0.0
    sim_ns.addCrossbarMatrix(device_type=device_config_ns, multiThread=False, rows=1, cols=1)
    
    def v_pulse_train_ns(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
        
    pulses_dense_ns = np.linspace(1, max_p_ns, int(max_p_ns))
    edge_points_ns = np.concatenate(([t_switch_ns], (pulses_dense_ns - 0.05) * t_period))
    sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=edge_points_ns.tolist())
    sim_ns.addDcSource(col=1, voltage=0.0)
    sim_ns.run(tEnd=t_stop_ns, min_dt=1e-11, tol=3e-3)
    
    t_arr_ns = sim_ns.getTime()
    g_sim_ns = sim_ns.getConductance(row=1, col=1)
    n_sim_ns = t_arr_ns / t_period
    g_sim_ns_ms = g_sim_ns * 1e3
    
    # Print state statistics for Non-Saturation
    n_physics_ns = sim_ns.getState(row=1, col=1)
    idx_max_ns = np.argmax(g_sim_ns_ms)
    print(f"  [State n Info - Non-Saturation]")
    print(f"    Beginning: n = {n_physics_ns[0]:.2f} (G = {g_sim_ns_ms[0]:.3f} mS)")
    print(f"    Max Cond:  n = {n_physics_ns[idx_max_ns]:.2f} (G = {g_sim_ns_ms[idx_max_ns]:.3f} mS)")
    print(f"    End:       n = {n_physics_ns[-1]:.2f} (G = {g_sim_ns_ms[-1]:.3f} mS)\n")

    # C. 1.22V Saturation Sweep (Dynamic pulse count)
    max_p_122 = np.max(pulses_exp_122.values)
    t_switch_122 = (max_p_122 / 2.0) * t_period
    t_stop_122 = max_p_122 * t_period
    print(f"Running transient physics simulation for 1.22V Saturation case ({max_p_122:.0f} pulses)...")
    sim_122 = MolmemSimulator()
    device_config_122 = device_config.copy()
    device_config_122["nInit"] = 0.0
    sim_122.addCrossbarMatrix(device_type=device_config_122, multiThread=False, rows=1, cols=1)
    
    vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
    vdep_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
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
        
    pulses_dense_122 = np.linspace(1, max_p_122, int(max_p_122))
    edge_points_122 = np.concatenate(([t_switch_122], (pulses_dense_122 - 0.05) * t_period))
    sim_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep_122], edge_points=edge_points_122.tolist())
    sim_122.addDcSource(col=1, voltage=0.0)
    sim_122.run(tEnd=t_stop_122, min_dt=1e-11, tol=3e-3)
    
    t_arr_122 = sim_122.getTime()
    g_sim_122 = sim_122.getConductance(row=1, col=1)
    n_sim_122 = t_arr_122 / t_period
    g_sim_122_ms = g_sim_122 * 1e3
    
    # Print state statistics for 1.22V Saturation
    n_physics_122 = sim_122.getState(row=1, col=1)
    idx_max_122 = np.argmax(g_sim_122_ms)
    print(f"  [State n Info - 1.22V Saturation]")
    print(f"    Beginning: n = {n_physics_122[0]:.2f} (G = {g_sim_122_ms[0]:.3f} mS)")
    print(f"    Max Cond:  n = {n_physics_122[idx_max_122]:.2f} (G = {g_sim_122_ms[idx_max_122]:.3f} mS)")
    print(f"    End:       n = {n_physics_122[-1]:.2f} (G = {g_sim_122_ms[-1]:.3f} mS)\n")
    
    # Debug print of neighborhood around 1.22V peak
    start_print = max(0, idx_max_122 - 3)
    end_print = min(len(g_sim_122_ms), idx_max_122 + 4)
    print("  [DEBUG - 1.22V Peak Neighborhood]")
    for idx in range(start_print, end_print):
        t_val = t_arr_122[idx]
        pulse_idx = t_val / t_period
        voltage = v_pulse_train_122(t_val, [vpot_122, vdep_122])
        print(f"    idx={idx:5d} | t={t_val:12.6e} (pulse={pulse_idx:10.4f}) | vApplied={voltage:6.3f} V | G={g_sim_122_ms[idx]:8.4f} mS | n={n_physics_122[idx]:8.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # 5. MATPLOTLIB INTERACTIVE PLOTTING (SIDE-BY-SIDE SUBPLOTS)
    # -------------------------------------------------------------------------
    print("\nDisplaying premium side-by-side comparison plot in GUI window...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # A. Subplot 1: Non-Saturating (Analog) Sweep
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.scatter(pulses_exp_ns, cond_exp_ns_ms, color="#d9b310", alpha=0.4, edgecolors="k", s=25, label="Empirical (Kundu 2025)", zorder=3)
    ax1.plot(p_ns_dense, c_ns_dense, color="#d9b310", linewidth=2.5, linestyle="--", label="Empirical Fit (PCHIP)", zorder=3)
    g_sim_ns_read = np.interp(pulses_dense_ns - 0.05, n_sim_ns, g_sim_ns_ms)
    ax1.plot(pulses_dense_ns, g_sim_ns_read, color="#328cc1", linewidth=3, label="Simulation Model (This Work)", zorder=2)
    
    ax1.set_title(f"Analog Programming (Non-Saturating Sweep) - {device_name}", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax1.set_ylabel("Conductance (mS)", fontsize=10)
    ax1.set_xlim(-0.03 * max_p_ns, 1.05 * max_p_ns)
    max_val_ns = max(np.max(g_sim_ns_read), np.max(cond_exp_ns_ms))
    ax1.set_ylim(0, np.ceil(max_val_ns + 1.0))
    vals_ns, labels_ns = get_dynamic_ticks(max_p_ns)
    ax1.set_xticks(vals_ns)
    ax1.set_xticklabels(labels_ns)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
    
    # B. Subplot 2: Saturating (Extended) Sweep
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.scatter(pulses_exp_sat, cond_exp_sat_ms, color="#ff5a5f", alpha=0.4, edgecolors="k", s=25, label="Empirical (Kundu 2025)", zorder=3)
    ax2.plot(p_sat_dense, c_sat_dense, color="#ff5a5f", linewidth=2.5, linestyle="--", label="Empirical Fit (PCHIP)", zorder=3)
    g_sim_sat_read = np.interp(pulses_dense_sat - 0.05, n_sim_sat, g_sim_sat_ms)
    ax2.plot(pulses_dense_sat, g_sim_sat_read, color="#0b3c5d", linewidth=3, label="Simulation Model (This Work)", zorder=2)
    
    ax2.set_title(f"Extended Programming (Saturating Sweep) - {device_name}", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax2.set_ylabel("Conductance (mS)", fontsize=10)
    ax2.set_xlim(-0.03 * max_p_sat, 1.05 * max_p_sat)
    max_val_sat = max(np.max(g_sim_sat_read), np.max(cond_exp_sat_ms))
    ax2.set_ylim(0, np.ceil(max_val_sat + 1.0))
    vals_sat, labels_sat = get_dynamic_ticks(max_p_sat)
    ax2.set_xticks(vals_sat)
    ax2.set_xticklabels(labels_sat)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
 
    # C. Subplot 3: 1.22V Saturating Sweep
    ax3.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax3.scatter(pulses_exp_122, cond_exp_122_ms, color="#10b981", alpha=0.4, edgecolors="k", s=25, label="Empirical (Kundu 2025)", zorder=3)
    ax3.plot(p_122_dense, c_122_dense, color="#10b981", linewidth=2.5, linestyle="--", label="Empirical Fit (PCHIP)", zorder=3)
    g_sim_122_read = np.interp(pulses_dense_122 - 0.05, n_sim_122, g_sim_122_ms)
    ax3.plot(pulses_dense_122, g_sim_122_read, color="#1e3a8a", linewidth=3, label="Simulation Model (This Work)", zorder=2)
    
    ax3.set_title(f"1.22V Extended Programming (Saturating Sweep) - {device_name}", fontsize=11, fontweight="bold", pad=12)
    ax3.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax3.set_ylabel("Conductance (mS)", fontsize=10)
    ax3.set_xlim(-0.03 * max_p_122, 1.05 * max_p_122)
    max_val_122 = max(np.max(g_sim_122_read), np.max(cond_exp_122_ms))
    ax3.set_ylim(0, np.ceil(max_val_122 + 1.0))
    vals_122, labels_122 = get_dynamic_ticks(max_p_122)
    ax3.set_xticks(vals_122)
    ax3.set_xticklabels(labels_122)
    ax3.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
    
    plt.suptitle(f"Fitted Device Model Verification vs. Empirical Data: {device_name}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save comparison plot to FittedFigures/{device_name}/
    os.makedirs(os.path.join(fitting_dir, "FittedFigures", device_name), exist_ok=True)
    out_plot_path = os.path.join(fitting_dir, "FittedFigures", device_name, f"{device_name}_comparison.png")
    plt.savefig(out_plot_path, dpi=300)
    print(f"Comparison plot successfully saved to {out_plot_path}")
    
    plt.show()
    print("Plot window closed.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
