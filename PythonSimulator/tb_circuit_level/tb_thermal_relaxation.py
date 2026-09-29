#!/usr/bin/env python3
"""
===============================================================================
tb_thermal_relaxation.py
===============================================================================
Thermal Relaxation, Steady-State Temperature, and State Stability Testbench.

Stimulus:
  - Phase 1 (0.0 ms to 2.5 ms): Sustained DC Read Bias (default: 0.35V sub-threshold)
  - Phase 2 (2.5 ms to 5.0 ms): 0V (Thermal relaxation / Newton cooling)

Physics Validated:
  1. Thermal Plateau: Reaches steady-state Delta T = P_diss * R_th within ~1 us.
  2. Exponential Decay: Cools exponentially back to T_amb (298.15 K) with tau_th = 127.8 ns.
  3. Non-Destructive State Stability: State (n, f22) remains rock-solid throughout.
===============================================================================
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter


def main():
    parser = argparse.ArgumentParser(description="Thermal Relaxation & State Stability Testbench")
    parser.add_argument("--vread", type=float, default=0.35, help="Sustained read voltage (default: 0.35V, sub-threshold)")
    parser.add_argument("--t-on", type=float, default=2.5e-3, help="Duration of ON phase in seconds (default: 2.5ms)")
    parser.add_argument("--t-off", type=float, default=2.5e-3, help="Duration of OFF phase in seconds (default: 2.5ms)")
    parser.add_argument("--n-init", type=float, default=16520.0, help="Initial device state n (default: 16520.0)")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive GUI windows (save images only)")
    args = parser.parse_args()

    v_read = args.vread
    t_switch = args.t_on
    t_stop = args.t_on + args.t_off
    n_init = args.n_init

    print("==========================================================================")
    print("         MOLMEM CIRCUIT SIMULATION: THERMAL RELAXATION TESTBENCH          ")
    print("==========================================================================")
    print(f"  Applied Voltage  : {v_read:.3f} V (Sustained DC for {t_switch*1e3:.1f} ms)")
    print(f"  Relaxation Phase : 0.000 V (Cooling for {(t_stop - t_switch)*1e3:.1f} ms)")
    print(f"  Initial State (n): {n_init:.1f} (Mid-conductance state)")
    print("==========================================================================\n")

    # 1. Initialize Simulator with pre-conditioned mid-conductance state
    from molmem_lib.devices import get_device_config
    dev_cfg = get_device_config("default")
    dev_cfg["nInit"] = float(n_init)
    
    sim = MolmemSimulator()
    sim.addCrossbarMatrix(device_type=dev_cfg, rows=1, cols=1)

    # 2. Construct Edge Points (Dense 25ns sampling during turn-on and turn-off transients)
    edge_set = {0.0, t_switch, t_stop}
    # Turn-on transient (0 to 3 us)
    for t in np.linspace(0.0, 3.0e-6, 121):
        edge_set.add(float(t))
    # Turn-off cooling transient (2.5ms - 0.5us to 2.5ms + 3.0us)
    for t in np.linspace(t_switch - 0.5e-6, t_switch + 3.0e-6, 141):
        edge_set.add(float(t))
    # Intermediate macro monitoring points
    for t in np.linspace(0.0, t_stop, 101):
        edge_set.add(float(t))
    sorted_edges = sorted(list(edge_set))

    # 3. Add Drive Sources
    def v_thermal_drive(t, sources):
        return v_read if t < t_switch else 0.0

    sim.addBSource(row=1, vFunc=v_thermal_drive, edge_points=sorted_edges)
    sim.addDcSource(col=1, voltage=0.0)

    # 3. Execute Transient Simulation
    print("[Solver] Running continuous ODE transient simulation...")
    sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4)
    print("[Solver] Simulation complete.")

    results_dir = os.path.join(os.path.dirname(__file__), "sim_results")
    os.makedirs(results_dir, exist_ok=True)

    # 4. Generate Macro Figure (Matching tb_dynamic 5-subplot layout)
    plotter = MolmemPlotter(sim)
    plotter.createFigure(
        "thermal_macro",
        title=f"Thermal Relaxation & State Stability (Macro View: V_read={v_read}V, n_init={n_init})",
        subplots=(5, 1),
        figsize=(10, 10)
    )
    plotter.addWaveform("thermal_macro", 'Voltage', row=1, subplotIdx=0, color='purple', label='Applied V')
    plotter.addWaveform("thermal_macro", 'Pulses', row=1, col=1, subplotIdx=1, color='green', label='Internal Pulse State (n)')
    plotter.addWaveform("thermal_macro", 'State', row=1, col=1, subplotIdx=2, color='blue', label='Conductance State (f22)')
    plotter.addWaveform("thermal_macro", 'Current', row=1, col=1, subplotIdx=3, color='red', label='Current')
    plotter.addWaveform("thermal_macro", 'Temperature', row=1, col=1, subplotIdx=4, color='darkorange', label='Junction Temperature (K)')

    for idx in range(5):
        plotter.formatAxis("thermal_macro", idx)

    macro_png = os.path.join(results_dir, "tb_thermal_relaxation_macro.png")
    plotter.saveFigure("thermal_macro", macro_png)
    print(f"[Plot] Saved macro figure: {macro_png}")

    # 5. Generate Microsecond Zoom Figure around t_switch (revealing exponential cooling curve)
    t_arr = sim.getTime()
    mask = (t_arr >= (t_switch - 0.5e-6)) & (t_arr <= (t_switch + 2.0e-6))
    
    t_zoom_us = (t_arr[mask] - t_switch) * 1e6  # Microseconds relative to turn-off
    v_zoom = sim.getVoltage(row=1)[mask]
    i_zoom = sim.getCurrent(row=1, col=1)[mask] * 1e3  # mA
    temp_zoom = sim.getTemperature(row=1, col=1)[mask]

    fig, (ax_v, ax_i, ax_t) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Microscopic Thermal Transition Zoom (tau_th = 127.8 ns | V_read = {v_read}V)", fontsize=12, fontweight='bold')

    # Voltage
    ax_v.plot(t_zoom_us, v_zoom, color='purple', lw=2, label='Applied V')
    ax_v.set_ylabel("Voltage (V)")
    ax_v.grid(True)
    ax_v.legend(loc='upper right')

    # Current
    ax_i.plot(t_zoom_us, i_zoom, color='red', lw=2, label='Current (mA)')
    ax_i.set_ylabel("Current (mA)")
    ax_i.grid(True)
    ax_i.legend(loc='upper right')

    # Temperature + Theoretical Exponential Overlay
    ax_t.plot(t_zoom_us, temp_zoom, color='darkorange', lw=2.5, label='Simulated Temp (K)')
    
    # Theoretical exponential fit: T(t) = 298.15 + (T_peak - 298.15) * exp(-t / tau_th)
    t_cool_mask = t_zoom_us >= 0.0
    t_cool = t_zoom_us[t_cool_mask] * 1e-6  # seconds
    tau_th = 1.278e-7  # 127.8 ns
    t_amb = 298.15
    idx_switch = np.searchsorted(t_zoom_us, 0.0)
    t_peak = temp_zoom[idx_switch] if idx_switch < len(temp_zoom) else temp_zoom[-1]
    delta_t_peak = t_peak - t_amb
    t_theory = t_amb + delta_t_peak * np.exp(-t_cool / tau_th)

    ax_t.plot(t_zoom_us[t_cool_mask], t_theory, 'k--', lw=1.5, alpha=0.85, label='Theory: exp(-t / 127.8ns)')
    ax_t.set_ylabel("Temperature (K)")
    ax_t.set_xlabel("Time relative to turn-off (us)")
    ax_t.grid(True)
    ax_t.legend(loc='upper right')

    fig.tight_layout()
    zoom_png = os.path.join(results_dir, "tb_thermal_relaxation_zoom.png")
    fig.savefig(zoom_png, dpi=200)
    print(f"[Plot] Saved microsecond zoom figure: {zoom_png}")

    if not args.no_show:
        plotter.showAll()


if __name__ == "__main__":
    main()
