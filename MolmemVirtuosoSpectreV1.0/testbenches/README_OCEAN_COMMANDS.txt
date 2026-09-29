================================================================================
Cadence Spectre Testbenches & Ocean Waveform Plotting Instructions
================================================================================

Run commands from inside this directory (SpectreModelsV2/testbenches/):

--------------------------------------------------------------------------------
1. Dynamic I-V Pulse Train
--------------------------------------------------------------------------------
Simulation:
  spectre tb_dynamic.scs

Ocean Display:
  ocean -replay ../ocean/tb_dynamic.ocn

Description:
  Simulates +0.9V potentiation and -0.75V depotentiation pulse trains.
  Plots 6 stacked strips in ViVA:
    - Strip 1: Applied Voltage (vDrive, V)
    - Strip 2: Pulse State (pulseState, count n)
    - Strip 3: Conductance State (f22, 0.0 to 1.0)
    - Strip 4: Read Conductance (conductance, mS)
    - Strip 5: Conduction Current (Xdut:Anode, mA)
    - Strip 6: Junction Temperature (junctionTemp, K)


--------------------------------------------------------------------------------
2. Extended 50,000-Pulse Endurance Cycling
--------------------------------------------------------------------------------
Simulation:
  spectre tb_dynamic_extended.scs

Ocean Display:
  ocean -replay ../ocean/tb_dynamic_extended.ocn

Description:
  Simulates long-duration cycling (25,000 SET + 25,000 RESET pulses).
  Plots 6 stacked strips in ViVA tracking state saturation, non-linear
  relaxation, electrical conductance, current, and thermal equilibrium.


--------------------------------------------------------------------------------
3. Standard Calibrated Programming Cycle
--------------------------------------------------------------------------------
Simulation:
  spectre tb_standard_cycle.scs

Ocean Display:
  ocean -replay ../ocean/tb_standard_cycle.ocn

Description:
  Simulates the calibrated 16,520 SET + 16,520 RESET pulse cycle matching
  Python tb_standard_cycle.py.
  Plots 4 stacked strips in ViVA:
    - Strip 1: Applied Voltage (vDrive, V)
    - Strip 2: Conduction Current (iMem, mA)
    - Strip 3: Small-Signal Read Conductance (conductance, mS)
    - Strip 4: Conductance State (f22, 0.0 to 1.0)


--------------------------------------------------------------------------------
4. Thermal Self-Heating & Exponential Relaxation
--------------------------------------------------------------------------------
Simulation:
  spectre tb_thermal_relaxation.scs

Ocean Display:
  ocean -replay ../ocean/tb_thermal_relaxation.ocn

Description:
  Simulates sustained 0.1V DC heating followed by unpowered relaxation (0V).
  Plots 6 stacked strips in ViVA resolving exponential thermal cooling at
  tau_th = 127.8 ns and non-volatile state retention.


--------------------------------------------------------------------------------
5. Multi-Regime Electrical Overstress & Recovery
--------------------------------------------------------------------------------
Simulation:
  spectre tb_stress.scs

Ocean Display:
  ocean -replay ../ocean/tb_stress.ocn

Description:
  Simulates 6 operating regimes (+0.9V SET, -0.75V RESET, 0V idle, +1.2V
  overdrive, -1.6V high-field stress, and +1.2V recovery).
  Plots 6 stacked strips in ViVA verifying high-field kinetic overdrive boost
  and degradation recovery dynamics.


--------------------------------------------------------------------------------
6. Continuous Triangular I-V Sweep & Pinched Hysteresis
--------------------------------------------------------------------------------
Simulation:
  spectre tb_iv_sweep.scs

Ocean Display:
  ocean -replay ../ocean/tb_iv_sweep.ocn

Description:
  Simulates continuous bipolar triangular voltage sweeping (-1.0V to +1.2V).
  Plots 5 stacked strips in ViVA:
    - Strip 1: Sweep Voltage (vSweep, V)
    - Strip 2: Conduction Current (iMem, mA, showing pinched I-V loop)
    - Strip 3: Conductance State (f22, 0.0 to 1.0)
    - Strip 4: Small-Signal Read Conductance (conductance, mS)
    - Strip 5: Junction Temperature (junctionTemp, K)
================================================================================
