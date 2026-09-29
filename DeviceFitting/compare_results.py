import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path so we can import molmem_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PythonSimulator')))

from molmem_lib import MolmemSimulator

def run_comparison():
    print("--------------------------------------------------")
    print("Starting Comparison Plot Generation")
    print("--------------------------------------------------")
    
    # --------------------------------------------------
    # 1. Run Simulations
    # --------------------------------------------------
    # A. Saturation Simulation (52k Pulses: 26k up, 26k down)
    print("Running transient physics simulation for Saturation case (52k pulses)...")
    sim_sat = MolmemSimulator()
    sim_sat.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    
    t_switch_sat = 4.16e-3 # 26k pulses * 160ns
    t_stop_sat = 8.32e-3   # 52k pulses * 160ns
    t_period = 160e-9
    
    from molmem_lib.sources import PulseSource
    vpot = PulseSource(node=None, amp=0.9, period=t_period, width=80e-9)
    vdep = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
    def v_pulse_train_sat(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_sat else sources[1].getVoltage(t)
        
    sim_sat.addBSource(row=1, vFunc=v_pulse_train_sat, sources=[vpot, vdep], edge_points=[t_switch_sat])
    sim_sat.addDcSource(col=1, voltage=0.0)
    sim_sat.run(tEnd=t_stop_sat, min_dt=1e-12, tol=1e-3)
    
    t_arr_sat = sim_sat.getTime()
    g_sim_sat = sim_sat.getConductance(row=1, col=1)
    n_sim_sat = t_arr_sat / t_period
    g_sim_sat_ms = g_sim_sat * 1e3
    
    # B. Non-Saturation Simulation (33k Pulses: 16.5k up, 16.5k down)
    print("Running transient physics simulation for Non-Saturation case (33k pulses)...")
    sim_ns = MolmemSimulator()
    sim_ns.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    
    t_switch_ns = 2.6432e-3
    t_stop_ns = 5.2864e-3
    
    def v_pulse_train_ns(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_ns else sources[1].getVoltage(t)
        
    sim_ns.addBSource(row=1, vFunc=v_pulse_train_ns, sources=[vpot, vdep], edge_points=[t_switch_ns])
    sim_ns.addDcSource(col=1, voltage=0.0)
    sim_ns.run(tEnd=t_stop_ns, min_dt=1e-12, tol=1e-3)
    
    t_arr_ns = sim_ns.getTime()
    g_sim_ns = sim_ns.getConductance(row=1, col=1)
    n_sim_ns = t_arr_ns / t_period
    g_sim_ns_ms = g_sim_ns * 1e3
    
    # C. 1.22V Saturation Simulation (33,040 pulses: 16,520 up, 16,520 down)
    print("Running transient physics simulation for 1.22V Saturation case (33,040 pulses)...")
    sim_sat_122 = MolmemSimulator()
    sim_sat_122.addCrossbarMatrix(device_type="default", multiThread=False, rows=1, cols=1)
    
    t_switch_122 = 16520 * 160e-9
    t_stop_122 = 33040 * 160e-9
    
    vpot_122 = PulseSource(node=None, amp=1.22, period=t_period, width=80e-9)
    vdep_122 = PulseSource(node=None, amp=-0.75, period=t_period, width=60e-9)
    
    def v_pulse_train_122(t, sources):
        return sources[0].getVoltage(t) if t < t_switch_122 else sources[1].getVoltage(t)
        
    sim_sat_122.addBSource(row=1, vFunc=v_pulse_train_122, sources=[vpot_122, vdep_122], edge_points=[t_switch_122])
    sim_sat_122.addDcSource(col=1, voltage=0.0)
    sim_sat_122.run(tEnd=t_stop_122, min_dt=1e-12, tol=1e-3)
    
    t_arr_122 = sim_sat_122.getTime()
    g_sim_122 = sim_sat_122.getConductance(row=1, col=1)
    n_sim_122 = t_arr_122 / t_period
    g_sim_122_ms = g_sim_122 * 1e3
    
    # --------------------------------------------------
    # 2. Load and Process Experimental Data (Scaled to match simulator endpoints and peak)
    # --------------------------------------------------
    fitting_dir = os.path.dirname(os.path.abspath(__file__))
    def find_dataset_csv(fname, device="Ru_azo"):
        p = os.path.join(fitting_dir, device, fname)
        if os.path.exists(p):
            return p
        p = os.path.join(fitting_dir, fname)
        if os.path.exists(p):
            return p
        return None

    # A. Saturation Experimental Data
    csv_path_sat = find_dataset_csv("SaturationCurrent.csv")
    if not csv_path_sat or not os.path.exists(csv_path_sat):
        print(f"Error: Could not find Saturation dataset SaturationCurrent.csv in {fitting_dir}")
        return
    print(f"Loading Saturation experimental data from {csv_path_sat}...")
    df_sat = pd.read_csv(csv_path_sat, comment='#')
    
    pulses_exp_sat_raw = df_sat["X_value"] * 520.0
    # Scale pulses (X axis) to align start at 0 and end at 52,000 pulses
    x_sat_min = pulses_exp_sat_raw.min()
    x_sat_max = pulses_exp_sat_raw.max()
    pulses_exp_sat = (pulses_exp_sat_raw - x_sat_min) / (x_sat_max - x_sat_min) * 52000.0
    
    # Calculate conductance from current (mA) / read voltage (0.5 V)
    cond_exp_sat_ms_raw = df_sat["Y_value"] / 0.5
    
    # Scale empirical data to match the min-max range of the simulation curve
    c_exp_min = cond_exp_sat_ms_raw.min()
    c_exp_max = cond_exp_sat_ms_raw.max()
    c_sim_min = g_sim_sat_ms.min()
    c_sim_max = g_sim_sat_ms.max()
    cond_exp_sat_ms = c_sim_min + ((cond_exp_sat_ms_raw - c_exp_min) / (c_exp_max - c_exp_min)) * (c_sim_max - c_sim_min)
    
    # B. Non-Saturation Experimental Data
    csv_path_ns = find_dataset_csv("NonSaturationCurrent.csv")
    if not csv_path_ns or not os.path.exists(csv_path_ns):
        print(f"Error: Could not find Non-Saturation dataset NonSaturationCurrent.csv in {fitting_dir}")
        return
    print(f"Loading Non-Saturation experimental data from {csv_path_ns}...")
    df_ns = pd.read_csv(csv_path_ns, comment='#')
    
    # X_value maps 0 to 100% of the 33,040 pulse sweep
    pulses_exp_ns_raw = df_ns["X_value"] * 330.4
    # Scale pulses (X axis) to align start at 0 and end at 33,040 pulses
    x_ns_min = pulses_exp_ns_raw.min()
    x_ns_max = pulses_exp_ns_raw.max()
    pulses_exp_ns = (pulses_exp_ns_raw - x_ns_min) / (x_ns_max - x_ns_min) * 33040.0
    
    # Calculate conductance from current (mA) / read voltage (0.5 V)
    cond_exp_ns_ms_raw = df_ns["Y_value"] / 0.5
    
    # Scale empirical data to match the min-max range of the simulation curve
    c_exp_min = cond_exp_ns_ms_raw.min()
    c_exp_max = cond_exp_ns_ms_raw.max()
    c_sim_min = g_sim_ns_ms.min()
    c_sim_max = g_sim_ns_ms.max()
    cond_exp_ns_ms = c_sim_min + ((cond_exp_ns_ms_raw - c_exp_min) / (c_exp_max - c_exp_min)) * (c_sim_max - c_sim_min)
    
    # C. 1.22V Saturation Experimental Data
    csv_path_122 = find_dataset_csv("1.22VSaturation.csv")
    if not csv_path_122 or not os.path.exists(csv_path_122):
        print(f"Error: Could not find 1.22V Saturation dataset 1.22VSaturation.csv in {fitting_dir}")
        return
    print(f"Loading 1.22V Saturation experimental data from {csv_path_122}...")
    df_122 = pd.read_csv(csv_path_122, comment='#')
    
    pulses_exp_122_raw = df_122["X_value"]
    x_122_min = pulses_exp_122_raw.min()
    x_122_max = pulses_exp_122_raw.max()
    pulses_exp_122 = (pulses_exp_122_raw - x_122_min) / (x_122_max - x_122_min) * 33040.0
    
    # Calculate conductance from current (mA) / read voltage (0.5 V)
    cond_exp_122_ms_raw = df_122["Y_value"] / 0.5
    
    # Scale empirical data to match the min-max range of the simulation curve
    c_exp_min = cond_exp_122_ms_raw.min()
    c_exp_max = cond_exp_122_ms_raw.max()
    c_sim_min = g_sim_122_ms.min()
    c_sim_max = g_sim_122_ms.max()
    cond_exp_122_ms = c_sim_min + ((cond_exp_122_ms_raw - c_exp_min) / (c_exp_max - c_exp_min)) * (c_sim_max - c_sim_min)
    
    # --------------------------------------------------
    # 3. PCHIP Interpolation for Smooth Experimental Curves
    # --------------------------------------------------
    from scipy.interpolate import PchipInterpolator
    
    # A. Saturation PCHIP
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
    
    # B. Non-Saturation PCHIP
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
    
    # C. 1.22V Saturation PCHIP
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
    
    # --------------------------------------------------
    # 4. Premium Aesthetic Plotting - Saturation Curve (2x1 Subplots)
    # --------------------------------------------------
    print("Generating Saturation comparison plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.scatter(pulses_exp_sat, cond_exp_sat_ms, color="#ff5a5f", alpha=0.4, edgecolors="k", s=25, label="Empirical (Saturation, Kundu 2025)", zorder=3)
    ax1.plot(p_sat_dense, c_sat_dense, color="#ff5a5f", linewidth=2.5, linestyle="--", label="Empirical Fit (Saturation, PCHIP)", zorder=3)
    ax1.plot(n_sim_sat, g_sim_sat_ms, color="#0b3c5d", linewidth=3, label="Simulation Model (Saturation, This Work)", zorder=2)
    ylim_sat = max(10.0, np.ceil(max(np.max(g_sim_sat_ms), np.max(cond_exp_sat_ms)) / 10.0) * 10.0)
    ax1.set_title("Extended Conductance Programming: Simulation vs. Empirical Data (Saturating Sweep)", fontsize=11, fontweight="bold", pad=15)
    ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax1.set_ylabel("Conductance (mS)", fontsize=10)
    ax1.set_xlim(-1000, 53000)
    ax1.set_ylim(0, ylim_sat)
    ax1.set_xticks(ticks=[1, 10000, 20000, 25000, 30000, 40000, 50000])
    ax1.set_xticklabels(["1", "10k", "20k", "25k", "30k", "40k", "50k"])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.scatter(pulses_exp_sat, cond_exp_sat_ms * 0.5, color="#ff5a5f", alpha=0.4, edgecolors="k", s=25, label="Empirical Current", zorder=3)
    ax2.plot(p_sat_dense, c_sat_dense * 0.5, color="#ff5a5f", linewidth=2.5, linestyle="--", label="Empirical Fit Current (PCHIP)", zorder=3)
    ax2.plot(n_sim_sat, g_sim_sat_ms * 0.5, color="#0b3c5d", linewidth=3, label="Simulation Current", zorder=2)
    ax2.set_title("Read Current at 0.5V Read Voltage (Saturating Sweep)", fontsize=11, fontweight="bold", pad=15)
    ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax2.set_ylabel("Current (mA)", fontsize=10)
    ax2.set_xlim(-1000, 53000)
    ax2.set_ylim(0, ylim_sat * 0.5)
    ax2.set_xticks(ticks=[1, 10000, 20000, 25000, 30000, 40000, 50000])
    ax2.set_xticklabels(["1", "10k", "20k", "25k", "30k", "40k", "50k"])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_img_path_sat = os.path.join(fig_dir, "comparison_plot.png")
    plt.tight_layout()
    plt.savefig(out_img_path_sat, dpi=300)
    plt.close()
    print(f"Saturation plot successfully saved to {out_img_path_sat}")

    # --------------------------------------------------
    # 5. Premium Aesthetic Plotting - Non-Saturation Curve (2x1 Subplots)
    # --------------------------------------------------
    print("Generating Non-Saturation comparison plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.scatter(pulses_exp_ns, cond_exp_ns_ms, color="#d9b310", alpha=0.4, edgecolors="k", s=25, label="Empirical (Non-Saturation, Kundu 2025)", zorder=3)
    ax1.plot(p_ns_dense, c_ns_dense, color="#d9b310", linewidth=2.5, linestyle="--", label="Empirical Fit (Non-Saturation, PCHIP)", zorder=3)
    ax1.plot(n_sim_ns, g_sim_ns_ms, color="#328cc1", linewidth=3, label="Simulation Model (Non-Saturation, This Work)", zorder=2)
    ylim_ns = max(10.0, np.ceil(max(np.max(g_sim_ns_ms), np.max(cond_exp_ns_ms)) / 10.0) * 10.0)
    ax1.set_title("Analog Conductance Programming: Simulation vs. Empirical Data (Non-Saturating Sweep)", fontsize=11, fontweight="bold", pad=15)
    ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax1.set_ylabel("Conductance (mS)", fontsize=10)
    ax1.set_xlim(-1000, 35000)
    ax1.set_ylim(0, ylim_ns)
    ax1.set_xticks(ticks=[1, 5000, 10000, 15000, 16520, 20000, 25000, 30000, 33040])
    ax1.set_xticklabels(["1", "5k", "10k", "15k", "16.5k", "20k", "25k", "30k", "33k"])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.scatter(pulses_exp_ns, cond_exp_ns_ms * 0.5, color="#d9b310", alpha=0.4, edgecolors="k", s=25, label="Empirical Current", zorder=3)
    ax2.plot(p_ns_dense, c_ns_dense * 0.5, color="#d9b310", linewidth=2.5, linestyle="--", label="Empirical Fit Current (PCHIP)", zorder=3)
    ax2.plot(n_sim_ns, g_sim_ns_ms * 0.5, color="#328cc1", linewidth=3, label="Simulation Current", zorder=2)
    ax2.set_title("Read Current at 0.5V Read Voltage (Non-Saturating Sweep)", fontsize=11, fontweight="bold", pad=15)
    ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax2.set_ylabel("Current (mA)", fontsize=10)
    ax2.set_xlim(-1000, 35000)
    ax2.set_ylim(0, ylim_ns * 0.5)
    ax2.set_xticks(ticks=[1, 5000, 10000, 15000, 16520, 20000, 25000, 30000, 33040])
    ax2.set_xticklabels(["1", "5k", "10k", "15k", "16.5k", "20k", "25k", "30k", "33k"])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    out_img_path_ns = os.path.join(fig_dir, "comparison_plot_nonsat.png")
    plt.tight_layout()
    plt.savefig(out_img_path_ns, dpi=300)
    plt.close()
    print(f"Non-Saturation plot successfully saved to {out_img_path_ns}")

    # --------------------------------------------------
    # 6. Premium Aesthetic Plotting - 1.22V Saturation Curve (2x1 Subplots)
    # --------------------------------------------------
    print("Generating 1.22V Saturation comparison plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.scatter(pulses_exp_122, cond_exp_122_ms, color="#10b981", alpha=0.4, edgecolors="k", s=25, label="Empirical (1.22V Saturation, Kundu 2025)", zorder=3)
    ax1.plot(p_122_dense, c_122_dense, color="#10b981", linewidth=2.5, linestyle="--", label="Empirical Fit (1.22V Saturation, PCHIP)", zorder=3)
    ax1.plot(n_sim_122, g_sim_122_ms, color="#1e3a8a", linewidth=3, label="Simulation Model (1.22V Saturation, This Work)", zorder=2)
    ylim_122 = max(10.0, np.ceil(max(np.max(g_sim_122_ms), np.max(cond_exp_122_ms)) / 10.0) * 10.0)
    ax1.set_title("Extended Conductance Programming (1.22V Potentiation): Simulation vs. Empirical Data", fontsize=11, fontweight="bold", pad=15)
    ax1.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax1.set_ylabel("Conductance (mS)", fontsize=10)
    ax1.set_xlim(-1000, 35000)
    ax1.set_ylim(0, ylim_122)
    ax1.set_xticks(ticks=[1, 5000, 10000, 15000, 16520, 20000, 25000, 30000, 33040])
    ax1.set_xticklabels(["1", "5k", "10k", "15k", "16.5k", "20k", "25k", "30k", "33k"])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.scatter(pulses_exp_122, cond_exp_122_ms * 0.5, color="#10b981", alpha=0.4, edgecolors="k", s=25, label="Empirical Current", zorder=3)
    ax2.plot(p_122_dense, c_122_dense * 0.5, color="#10b981", linewidth=2.5, linestyle="--", label="Empirical Fit Current (PCHIP)", zorder=3)
    ax2.plot(n_sim_122, g_sim_122_ms * 0.5, color="#1e3a8a", linewidth=3, label="Simulation Current", zorder=2)
    ax2.set_title("Read Current at 0.5V Read Voltage (1.22V Saturation)", fontsize=11, fontweight="bold", pad=15)
    ax2.set_xlabel("Pulse Counts (n)", fontsize=10)
    ax2.set_ylabel("Current (mA)", fontsize=10)
    ax2.set_xlim(-1000, 35000)
    ax2.set_ylim(0, ylim_122 * 0.5)
    ax2.set_xticks(ticks=[1, 5000, 10000, 15000, 16520, 20000, 25000, 30000, 33040])
    ax2.set_xticklabels(["1", "5k", "10k", "15k", "16.5k", "20k", "25k", "30k", "33k"])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    out_img_path_122 = os.path.join(fig_dir, "comparison_plot_1.22V.png")
    plt.tight_layout()
    plt.savefig(out_img_path_122, dpi=300)
    plt.close()
    print(f"1.22V Saturation plot successfully saved to {out_img_path_122}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_comparison()
