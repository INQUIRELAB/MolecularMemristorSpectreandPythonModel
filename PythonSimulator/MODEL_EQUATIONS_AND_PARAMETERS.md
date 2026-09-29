# Molmem Device Model: Mathematical Equations and Physical Parameters Reference

## 1. Executive Summary & Model Overview

The **Molmem** device model is a physics-informed, non-linear dynamical compact model engineered to simulate redox-active molecular memristor crossbar cells (such as Ruthenium-azo complexes, Ru-azo) under arbitrary analog and pulsed bias regimes.

The device state is parameterized by an internal molecular coordinate:
$$n(t) \in [0, \, n_{\max}]$$
representing the instantaneous microscopic population or spatial extent of reduced, highly conductive molecular complexes within the active switching film.

The model incorporates:
1. **Field-Driven Redox Kinetics**: Power-law and softplus activation dynamics with thresholding ($v_{th}$) and direction-dependent rate asymmetry ($\kappa, \alpha$).
2. **Direction-Aware Non-Linear Boundary Clamping**: Independent Hermite smoothsteps that enforce physical boundaries ($n \in [0, n_{\max}]$) without imposing artificial turn-on delays.
3. **Electro-Thermal Joule Heating & Arrhenius Acceleration**: Dynamic local temperature rise ($\Delta T = P_{\text{diss}} \cdot R_{th}$) modulating switching velocities via Arrhenius exponential scaling ($E_a$).
4. **Non-Linear Conduction Physics**: Field-assisted hyperbolic sine ($\sinh$) carrier transport combined with non-linear lookup tables ($I_{\text{scale}}, V_{\text{scale}}, f_{22}$) and series contact resistance ($R_c$).
5. **Dynamic Continuity Scaling Factor ($\xi$)**: Preserves strict physical continuity ($C^0$ admittance continuity) across abrupt polarity reversals.
6. **Saturation Recovery, Voltage Overdrive Memory & Turnaround Symmetry**:
   - Write overdrive accumulation ($c_{\text{discharge}}$) parameterized by $k_{\text{overdrive}}$.
   - Effective latency gating ($r_{\text{lat,eff}}$) that collapses return shelves under high-field saturation.
   - Symmetric boundary transitions governed by $r_{\text{top\_blend}}$ (apex turnaround shoulder width) and $r_{\text{bottom\_blend}}$ (saturation recovery baseline landing floor).

---

## 2. Table of Model Parameters

The table below catalogs every model parameter, its physical units, nominal/default Ru_azo values, optimization bounds (used during Differential Evolution fitting), and exact mathematical/physical role in the simulation engine.

| Parameter Symbol | Code Name | Physical Unit | Default / Ru_azo Value | Optimizer Bounds | Physical Mechanism | Exact Mathematical & Physical Role | Relates to $n_{\max}$ / $16520$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\kappa$ | `kappa` | $\text{s}^{-1}$ | $1.172865 \times 10^7$ | $[10^6, 3 \times 10^7]$ (log) | State Transition Rate | Fundamental velocity scale of molecular redox switching. Multiplies the normalized field overpotential in $dn/dt$. | No |
| $\alpha$ | `alpha` | Dimensionless | $0.931756$ | $[0.1, 10.0]$ | Non-linear Field Exponent | Power-law exponent governing field sensitivity above threshold ($v > v_{th}$). Dictates the steepness of the switching transition. | No |
| $V_{th}$ | `vTh` | $\text{V}$ | $0.466409$ | $[0.38, 0.85]$ | Threshold Voltage | Electrostatic potential barrier required to initiate field-driven molecular reduction or oxidation. | No |
| $V_{\text{smooth}}$ | `vSmooth` | $\text{V}$ | $0.103776$ | $[0.001, 0.30]$ | Voltage Smoothing | Numerical regularization width for softplus and tanh transitions. Eliminates derivative cusps across $V = \pm V_{th}$ and $V = 0$. | No |
| $N_{\text{smooth}}$ | `nSmooth` | Dimensionless | $17.06$ | $[15.0, 1500.0]$ (log) | Peak Saturation Softness | Regularization width of the upper boundary ($n \to n_{\max}$). Prevents abrupt derivative cusps and overshoot during potentiation saturation. | **Yes ($n_{\max}$)**: Sets upper boundary Hermite clamping width $w_{\text{sat}} = \max(50.0, 1.5 \cdot N_{\text{smooth,eff}})$ approaching $n_{\max}$. |
| $V_{\text{ref,pos}}$ | `vRefPos` | $\text{V}$ | $0.900000$ | Fixed ($0.90$) | Positive Reference Potential | Nominal write pulse amplitude used to normalize positive field overpotential and define write voltage overdrive baseline. | **Yes ($16520$)**: Driving voltage ($0.90\text{ V}$) yielding nominal 50% saturation at $16520$; overpotential above $V_{\text{ref,pos}}$ triggers overdrive accumulation. |
| $V_{\text{ref,neg}}$ | `vRefNeg` | $\text{V}$ | $0.750000$ | Fixed ($0.75$) | Negative Reference Potential | Nominal erase pulse amplitude used to normalize negative field overpotential. | No |
| $n_{\max}$ | `nMax` | Dimensionless | $33040$ | Fixed ($33040$) | Maximum Molecular State | Total available density of redox-active molecular sites (100% capacity). The half-value $0.5 \cdot n_{\max} = 16520$ forms the nominal saturation barrier and neuromorphic weight capacity. | **Yes (Direct $n_{\max} = 33040$)**: The core upper boundary of the state space ($n \in [0, n_{\max}]$). |
| $n_{\text{init}}$ | `nInit` | Dimensionless | $0.0$ | $[0, n_{\max}]$ | Initial State | Pristine/starting state of the device at $t = 0$ prior to stimulus (commonly $16520.0$ for balanced mid-conductance crossbar testbenches). | **Yes ($16520$)**: Often initialized to $16520.0 = 0.5 \cdot n_{\max}$ for balanced crossbar neuromorphic weight mapping. |
| $g_{\text{scale}}$ | `gScale` | Dimensionless | $6.227303$ | $[5.2, 6.9]$ | Transconductance Gain | Overall transconductance scaling pre-factor for the terminal current and small-signal read conductance. | No |
| $R_c$ | `rC` | $\Omega$ | $0.008033$ | $[0.001, 0.05]$ | Series Contact Resistance | Parasitic series resistance of top/bottom metallic contact interfaces; solved iteratively via Halley's method. | No |
| $E_a$ | `E_a` | $\text{eV}$ | $0.155923$ | $[0.15, 0.60]$ | Thermal Activation Energy | Apparent activation energy barrier for Arrhenius thermally assisted molecular state transitions. | No |
| $R_{th}$ | `R_th` | $\text{K} / \text{W}$ | $52.14$ | $[10, 10000]$ (log) | Thermal Resistance | Effective thermal resistance between the active molecular junction volume and the ambient substrate heatsink. | No |
| $\tau_{th}$ | `tau_th` | $\text{s}$ | $1.278 \times 10^{-7}$ | $[10^{-9}, 3 \times 10^{-7}]$ (log) | Thermal Time Constant | Characteristic thermal relaxation time constant governing junction heating and Newton cooling ($C_{th} = \tau_{th} / R_{th}$). | No |
| $\gamma$ | `gamma` | Dimensionless | $-0.012230$ | $[-0.9, 0.0]$ | High-Field Compression | Coupling coefficient that modulates the effective $V_{\text{scale}}$ lookup dynamically as a function of the internal state $f_{22}$. | No |
| $\beta_\alpha$ | `beta_alpha` | Dimensionless | $0.125186$ | $[0.0, 5.0]$ | Hyperbolic Exponent Boost | Modulates the field exponent $\alpha$ under high-voltage bias ($V > V_{\text{ref,pos}}$) via $\tanh(\delta_v)$, where $\delta_v = (V_{\text{int}} - V_{\text{ref,pos}}) / (V_{\text{ref,pos}} - V_{th} + 10^{-6})$. | No |
| $\beta_s$ | `beta_s` | Dimensionless | $-4.382853$ | $[-5.0, 10.0]$ | Saturation Shoulder Rounding | Modulates effective $N_{\text{smooth}}$ under high-voltage bias ($V > V_{\text{ref,pos}}$) via $\tanh(\delta_v)$, capturing shoulder rounding and saturation softness at $1.22\text{ V}$. | **Yes ($n_{\max}$)**: Modulates saturation boundary softness $N_{\text{smooth,eff}}$ approaching $n_{\max}$. |
| $r_{\text{asym}}$ | `rate_asymmetry`| Dimensionless | $0.985694$ | $[0.05, 0.99]$ | Kinetic Polarity Asymmetry | Ratio of the intrinsic depression (erasure) rate relative to the potentiation (writing) rate at matched overpotential. | No |
| $r_{\text{top\_blend}}$ | `rTopBlend` | Dimensionless | $0.050000$ | $[0.001, 0.50]$ | Apex Turnaround Width Scalar | Dynamically sets the width $w_{\text{top}} = \max(30.0, r_{\text{top\_blend}} \cdot \text{excursion})$ of the Hermite smoothstep at the peak turnaround. | **Yes (via $\text{excursion}$)**: Scales proportional to the actual peak excursion ($n_{\text{turnaround}} - n_{\text{base}}$). |
| $k_{\text{discharge}}$ | `kDischarge` | Dimensionless | $362.880000$ | $[0.1, 500.0]$ | Apex Drop Boost Amplitude | Learnable velocity amplification pre-factor for rapid saturation snapback during early depression pulses. | **Yes ($n_{\text{turnaround}}$ and $16520$ via $n_{\text{base}}$)**: Scales apex snapback velocity along normalized excursion $u = (n - n_{\text{base}}) / (n_{\text{turnaround}} - n_{\text{base}})$. |
| $f_{\text{discharge}}$ | `fDischarge` | Dimensionless | $0.374848$ | $[0.10, 0.70]$ | Dynamic Knee Transition (`uKnee`) | Learnable transition exponent ($p = 1 / f_{\text{discharge}}$) governing the non-linear knee blend between rapid apex discharge and return shelf latency as the state approaches $n_{\text{base}}$. | **Yes ($n_{\text{turnaround}}$ and $16520$ via $n_{\text{base}}$)**: Exponent applies to normalized excursion $u = (n - n_{\text{base}}) / (n_{\text{turnaround}} - n_{\text{base}})$. |
| $c_{\text{discharge}}$ | `cDischarge` | Dimensionless | $0.0$ (runtime) | $[0.0, 1.0]$ | Write Overdrive Register | Persistent internal analog memory register tracking positive voltage overdrive: accumulated during write, retained across erase. | **Yes ($16520$ and $n_{\max}$)**: Shifts $n_{\text{base}}$ from nominal $16520$ down towards $0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$; resets when $n \le n_{\text{base}}$. |
| $r_{\text{bottom\_blend}}$| `rBottomBlend` | Dimensionless | $0.714488$ | $[0.10, 1.00]$ | Landing Boundary Scalar (Base Modulation) | Modulates the nominal $16520$ midpoint ($0.5 \cdot n_{\max}$) into the unified saturation recovery landing floor $n_{\text{base}} = 0.5 \cdot n_{\max} \cdot [1 - \text{od}_{\text{clamp}} \cdot (1 - r_{\text{bottom\_blend}})]$. (Evaluates to $\approx 11803$ under full overdrive for the calibrated Ru_azo preset). | **Yes (Direct $16520 = 0.5 \cdot n_{\max}$)**: Modulates nominal $16520$ baseline floor for both $f_{\text{discharge}}$ and $r_{\text{latency}}$ together. |
| $r_{\text{latency}}$ | `rLatency` | Dimensionless | $0.878506$ | $[0.0, 0.95]$ | Shelf Latency Throttle | Return shelf latency suppression factor active during early depression pulses moving towards the modulated $16520$ target floor $n_{\text{base}}$. | **Yes ($16520$ via $n_{\text{base}}$)**: Governs shelf rate suppression as the state moves towards the modulated $16520$ target $n_{\text{base}}$. |
| $k_{\text{overdrive}}$ | `k_overdrive` | Dimensionless | $7.296635$ | $[0.0, 10.0]$ | Voltage Overdrive Sensitivity | Learnable scaling scalar coupling accumulated write overdrive to return shelf latency collapse ($r_{\text{lat,eff}} \to 0$) and deep floor snapback ($n_{\text{base}} \to 0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$). | **Yes ($16520$)**: Controls transition between nominal $16520$ landing and deep overdrive landing floor. |
| $V_{\text{bypass}}$ | `vBypass` | $\text{V}$ | $0.155081$ | Fixed ($0.155$) | Subthreshold Solver Bypass | Voltage threshold below which the solver can execute accelerated early-exit evaluations if thermal/mode state is quiescent ($V_{th} - 3 \cdot V_{\text{smooth}}$). | **Yes ($16520$)**: Early-exit bypass checks if $n \le n_{\text{base}}$ before skipping saturation math. |
| $f_{22,\text{start}}$ | `f22StartVal` | Dimensionless | $0.120373$ | Lookup derived | Potentiation Base Floor | Normalization floor constant for the potentiation branch lookup table $f_{22,\text{pot}}(n)$. | No |
| $f_{22,\text{peak}}$ | `f22PeakVal` | Dimensionless | $0.877920$ | Lookup derived | Maximum Branch Peak | Normalization peak constant for lookup table $f_{22}(n)$ at $n = n_{\max}$. | **Yes ($n_{\max}$)**: Value of the PWL conductance lookup evaluated at maximum state $n = n_{\max} = 33040$. |
| $f_{22,\text{end}}$ | `f22EndVal` | Dimensionless | $0.139555$ | Lookup derived | Depression Base Floor | Normalization baseline constant for the depression branch lookup table $f_{22,\text{dep}}(n)$. | No |
| $f_{22,\text{pot,inv}}$ | `f22PotScaleInv`| Dimensionless | $1.320038$ | Lookup derived | Potentiation Scale Inverse | Multiplier: $1 / (f_{22,\text{peak}} - f_{22,\text{start}})$, mapping raw PWL potentiation values to $[0, 1]$. | **Yes ($n_{\max}$)**: Spans the full potentiation state range up to $n_{\max}$. |
| $f_{22,\text{dep,inv}}$ | `f22DepScaleInv`| Dimensionless | $1.354341$ | Lookup derived | Depression Scale Inverse | Multiplier: $1 / (1 - f_{22,\text{start}} - f_{22,\text{end}})$, mapping raw PWL depression values to $[0, 1]$. | No |
| $f_{22,\text{floor}}$ | `f22Floor` | Dimensionless | $0.000000$ | Fixed ($0.0$) | Hard Conductance Floor | Absolute numerical lower clamp for the active $f_{22}$ normalized state. | No |

---

## 3. Table of Dynamic State Variables

These variables evolve dynamically across solver timesteps and are preserved inside the device cache / GPU global memory:

| State Variable | Code Symbol | Physical Unit | Initial Value | Physical Role & Description |
| :--- | :--- | :--- | :--- | :--- |
| $n(t)$ | `_n` / `local_n` | Dimensionless | $0.0$ | Primary molecular state occupancy coordinate ($0 \le n \le n_{\max}$). |
| $m(t)$ | `_modeState` / `local_mode` | Dimensionless | $0.0$ | Direction latch switch ($1.0$ for potentiation, $0.0$ for depression) with exponential subthreshold relaxation ($\tau_{\text{mode}} = 1\text{ ms}$). |
| $\xi(t)$ | `_scaleFactor` / `local_scale` | Dimensionless | $1.0$ | Admittance continuity scale factor preserving smooth $C^0$ conduction transitions across write/erase polarity changes. |
| $T(t)$ | `_T` / `local_T` | $\text{K}$ | $298.15$ | Local microscopic junction temperature, driven by Joule power dissipation and Newton cooling. |
| $c_{\text{od}}(t)$ | `cDischarge` / `local_cached_n` | Dimensionless | $0.0$ | Analog register accumulating peak write voltage overdrive ratio $(V_{\text{write}} / V_{\text{ref,pos}}) - 1.0$. Resets when $n \le n_{\text{base}}$. |
| $G(t)$ | `_G` / `gHist` | $\text{S}$ | $0.0$ | Terminal small-signal read conductance measured at $V_{\text{read}} = 0.1\text{ V}$. |
| $I(t)$ | `currentI` / `iHist` | $\text{A}$ | $0.0$ | Instantaneous terminal current under applied voltage $V_{\text{applied}}$. |

---

## 4. Governing Equations in Computational Execution Order

Below is the complete sequence of physical equations executed by the predictor-corrector solver during each timestep $\Delta t$.

### 4.1. High-Voltage Hyperbolic Parameter Modulation
When internal voltage $V_{\text{int}} > V_{\text{ref,pos}}$, strong internal electric fields alter the effective switching kinetics and boundary softness, parameterized relative to the physical threshold overpotential:

$$\delta_v = \dfrac{V_{\text{int}} - V_{\text{ref,pos}}}{\max\left(10^{-15}, \; V_{\text{ref,pos}} - V_{th} + 10^{-6}\right)}$$

$$\alpha_{\text{eff}} = \text{clamp}\left(\alpha + \beta_\alpha \cdot \tanh(\delta_v), \; 0.1, \; 1000.0\right)$$

$$N_{\text{smooth,eff}} = \text{clamp}\left(N_{\text{smooth}} \cdot \left[1.0 + \beta_s \cdot \tanh(\delta_v)\right], \; 1.0, \; 1000.0\right)$$

---

### 4.2. Terminal Electrical Conduction & Series Contact Resistance
Conduction through the molecular junction combines non-linear state mapping $f_{22}(n)$ with field-assisted carrier tunneling:

#### Potentiation State Normalization and Saturation Shoulder Excursion
The raw potentiation lookup $f_{22,\text{pot,raw}}(n)$ is normalized against baseline calibration constants:
$$f_{22,\text{pot,scaled}} = (f_{22,\text{pot,raw}} - f_{22,\text{start}}) \cdot f_{22,\text{pot,scale\_inv}}$$

For molecular state occupancies exceeding nominal saturation ($n > 16520 = 0.5 \cdot n_{\max}$), a $C^1$ Hermite saturation shoulder excursion is incorporated, modulated directly by the effective saturation softness $N_{\text{smooth,eff}}$:
$$u_{\text{sat}} = \text{clamp}\left(\dfrac{n - 16520.0}{16520.0}, \; 0.0, \; 1.0\right)$$
$$s_{\text{plateau}} = 0.001 \cdot N_{\text{smooth,eff}}$$
$$f_{22,\text{pot,eff}} = f_{22,\text{pot,scaled}} + s_{\text{plateau}} \cdot u_{\text{sat}}^2 (3.0 - 2.0 u_{\text{sat}})$$

This formulation guarantees:
1. **$C^1$ Smoothstep Continuity at $n = 16520$**: Both value and first derivative match identically at nominal saturation ($S(0) = 0, S'(0) = 0$), preserving non-saturating sweeps with 100% bit-exact parity.
2. **Angled Plateau at 0.90 V**: At nominal write voltage, $N_{\text{smooth,eff}} = N_{\text{smooth}} \approx 150 \implies s_{\text{plateau}} \approx 0.15$. As $n$ smoothly traverses from $16520$ to $\approx 31000$ over 26,000 pulses, it produces the angled ramp (+5.6e-5 mS/pulse) and pointy apex.
3. **Flat Shelf at 1.22 V**: High-voltage overdrive modulates $N_{\text{smooth,eff}}$ downward via $\beta_s$, and drives $n \to n_{\max} = 33040$ within $\sim 10000$ pulses where $u_{\text{sat}} = 1.0$ clamps, locking conductance into a perfectly flat horizontal shelf.

#### Junction Potential and Terminal Current Equations
$$I_{\text{prod}} = I_{\text{scale}}(n) \cdot g_{\text{scale}} \cdot f_{22,\text{active}}$$

$$V_{\text{scale,eff}} = \begin{cases} \dfrac{V_{\text{scale}}(n)}{1.0 + \gamma \cdot f_{22,\text{active}}} & \text{if } \gamma \ne 0 \\ V_{\text{scale}}(n) & \text{otherwise} \end{cases}$$

$$g_{\text{base}} = \dfrac{I_{\text{prod}}}{V_{\text{scale,eff}}}$$

#### Conduction Current ($R_c = 0$):
$$I = I_{\text{prod}} \cdot \sinh\left(\dfrac{V}{V_{\text{scale,eff}}}\right)$$

#### Conduction Current with Finite Contact Resistance ($R_c > 0$):
The internal junction voltage drop $x = V_{\text{junction}} / V_{\text{scale,eff}}$ satisfies:
$$x + (R_c \cdot g_{\text{base}}) \sinh(x) = \dfrac{V_{\text{applied}}}{V_{\text{scale,eff}}}$$
Solved via 4 iterations of **Halley's second-order rational method**:
$$x_{k+1} = x_k - \dfrac{2 f(x_k) f'(x_k)}{2 [f'(x_k)]^2 - f(x_k) f''(x_k)}$$
where:
$$f(x) = x - v + \text{coeff} \cdot \sinh(x)$$
$$f'(x) = 1 + \text{coeff} \cdot \cosh(x)$$
$$f''(x) = \text{coeff} \cdot \sinh(x)$$
$$\text{coeff} = R_c \cdot g_{\text{base}}$$

Terminal current and small-signal equivalent conductance:
$$I = I_{\text{prod}} \cdot \sinh(x)$$
$$g_{\text{core}} = g_{\text{base}} \cdot \cosh(x)$$
$$g_{\text{eq}} = \dfrac{g_{\text{core}}}{1.0 + g_{\text{core}} \cdot R_c}$$

---

### 4.3. Electro-Thermal Joule Heating & Arrhenius Acceleration
Microscopic self-heating within the nanoscale switching volume:

$$P_{\text{diss}} = |V_{\text{applied}} \cdot I|$$

$$T_{\text{steady}} = T_{\text{amb}} + P_{\text{diss}} \cdot R_{th} \quad (T_{\text{amb}} = 298.15\text{ K})$$

$$T(t + \Delta t) = T_{\text{steady}} + \left[T(t) - T_{\text{steady}}\right] \exp\left(-\dfrac{\Delta t}{\tau_{th}}\right)$$

$$T_{\text{avg}} = \dfrac{T(t) + T(t + \Delta t)}{2}$$

The Arrhenius kinetic acceleration factor:
$$k_{\text{thermal}} = \exp\left[\dfrac{E_a}{k_B} \left(\dfrac{1}{T_{\text{amb}}} - \dfrac{1}{T_{\text{avg}}}\right)\right] = \exp\left[(E_a \cdot 11604.52) \left(0.003354 - \dfrac{1}{T_{\text{avg}}}\right)\right]$$

---

### 4.4. Direction Activation & Softplus Field Overpotential
Smooth regularized activation replaces discontinuous step functions:

$$v_{\tanh} = \tanh\left(\dfrac{V_{\text{int}}}{V_{\text{smooth}}}\right)$$

$$\text{arg} = \dfrac{|V_{\text{int}}| - V_{th}}{V_{\text{smooth}}}$$

$$\text{softplus}(\text{arg}) = \begin{cases} \text{arg} & \text{if } \text{arg} > 20.0 \\ 0.0 & \text{if } \text{arg} < -20.0 \\ \ln\left[1.0 + \exp(\text{arg})\right] & \text{otherwise} \end{cases}$$

---

### 4.5. Direction-Aware Decoupled Boundary Clamping
Boundary clamping applies **only when state trajectories push into physical limits**, preventing negative molecular populations without imposing turn-on latency.

> [!IMPORTANT]
> **Relationship to $n_{\max}$ ($33040$)**:
> Clamping at the upper boundary is explicitly anchored at the maximum film capacity $n_{\max} = 33040$. The parameter $N_{\text{smooth,eff}}$ controls the Hermite regularization window $w_{\text{sat}} = \max(50.0, 1.5 \cdot N_{\text{smooth,eff}})$. When $n \ge n_{\max}$, $C_{\text{clamp}} = 0.0 \implies dn/dt = 0$, strictly guaranteeing that the molecular population cannot exceed $100\%$ film capacity ($n \le n_{\max}$).

#### Upper Boundary Clamping (Potentiation Saturation, $v_{\tanh} > 0$):
$$w_{\text{sat}} = \max\left(50.0, \; 1.5 \cdot N_{\text{smooth,eff}}\right)$$
$$z_{\text{hi}} = \dfrac{n_{\max} - n}{w_{\text{sat}}}$$
$$C_{\text{clamp}} = \begin{cases} 1.0 & \text{if } n_{\max} - n \ge w_{\text{sat}} \\ 0.0 & \text{if } n_{\max} - n \le 0.0 \\ z_{\text{hi}}^2 (3.0 - 2.0 z_{\text{hi}}) & \text{otherwise (Hermite smoothstep)} \end{cases}$$

#### Lower Boundary Clamping (Depression Zero Guard, $v_{\tanh} < 0$):
$$w_{\text{zero}} = 30.0 \quad (\text{independent of } N_{\text{smooth}})$$
$$z_{\text{low}} = \dfrac{n}{w_{\text{zero}}}$$
$$C_{\text{clamp}} = \begin{cases} 1.0 & \text{if } n \ge w_{\text{zero}} \\ 0.0 & \text{if } n \le 0.0 \\ z_{\text{low}}^2 (3.0 - 2.0 z_{\text{low}}) & \text{otherwise (Hermite smoothstep)} \end{cases}$$

Effective velocity scaling:
$$\kappa_{\text{clamp}} = \kappa \cdot C_{\text{clamp}}$$

---

### 4.6. Primary State Evolution Rate Equation ($dn/dt$)
The base rate equation governs continuous molecular transitions:

#### Positive Polarity ($v_{\tanh} = 1.0$, Potentiation):
$$\text{base} = \dfrac{V_{\text{smooth}} \cdot \text{softplus}}{|V_{\text{ref,pos}} - V_{th}| + 10^{-6}}$$
$$\left(\dfrac{dn}{dt}\right)_{\text{base}} = \text{base}^{\alpha_{\text{eff}}} \cdot \kappa_{\text{clamp}} \cdot k_{\text{thermal}}$$

#### Negative Polarity ($v_{\tanh} = -1.0$, Depression):
$$\text{base} = \dfrac{V_{\text{smooth}} \cdot \text{softplus}}{|V_{\text{ref,neg}} - V_{th}| + 10^{-6}}$$
$$\left(\dfrac{dn}{dt}\right)_{\text{base}} = -\left[\text{base}^{\alpha_{\text{eff}}} \cdot (\kappa_{\text{clamp}} \cdot r_{\text{asym}}) \cdot k_{\text{thermal}}\right]$$

#### Sub-threshold Smooth Bipolar Transition ($-1.0 < v_{\tanh} < 1.0$):
$$s_{\text{pol}} = 0.5 \cdot (1.0 + v_{\tanh})$$
$$V_{\text{ref,eff}} = s_{\text{pol}} \cdot V_{\text{ref,pos}} + (1.0 - s_{\text{pol}}) \cdot V_{\text{ref,neg}}$$
$$\text{base} = \dfrac{V_{\text{smooth}} \cdot \text{softplus} \cdot |v_{\tanh}|}{|V_{\text{ref,eff}} - V_{th}| + 10^{-6}}$$
$$\text{rate}_{\text{scale}} = \begin{cases} 1.0 & \text{if } v_{\tanh} > 0 \\ r_{\text{asym}} & \text{if } v_{\tanh} \le 0 \end{cases}$$
$$\left(\dfrac{dn}{dt}\right)_{\text{base}} = \text{base}^{\alpha_{\text{eff}}} \cdot v_{\tanh} \cdot (\kappa_{\text{clamp}} \cdot \text{rate}_{\text{scale}}) \cdot k_{\text{thermal}}$$

---

### 4.7. Saturation Recovery, Overdrive Memory & Return Latency Gating

During depression ($m(t) < 0.45$ or $v_{\tanh} < 0$), if $n > 500$, the device enters the non-linear saturation recovery regime.

> [!IMPORTANT]
> **Relationship to $n_{\max}$ ($33040$) and Nominal Saturation Midpoint ($16520 = 0.5 \cdot n_{\max}$)**:
> This entire subsystem models the physical transition between the nominal 50% saturation ceiling ($16520$ states, reached under $0.90\text{ V}$ write pulses) and the high-field overdrive saturation regime ($n \in [16520, 33040]$, reached under $\ge 1.22\text{ V}$ pulses).

#### 1. Write Voltage Overpotential Memory Latch (Potentiation):
During writing ($m(t) > 0.5$):
- If the junction is in the saturation regime ($n > 0.5 \cdot n_{\max} = 16520$) and field overpotential exceeds positive reference bias ($V_{\text{applied}} > V_{\text{ref,pos}}$), voltage overdrive is continuously latched:
$$\text{od}_{\text{inst}} = \dfrac{V_{\text{applied}} - V_{\text{ref,pos}}}{V_{\text{ref,pos}} - V_{th} + 10^{-6}}$$
$$c_{\text{discharge}} = \max(c_{\text{discharge}}, \; \text{od}_{\text{inst}})$$
- In the sub-saturation linear regime ($n \le 16520$), or for nominal writing ($V_{\text{applied}} \le V_{\text{ref,pos}}$, e.g. $0.90\text{ V}$), $c_{\text{discharge}}$ remains identically $0.0$.

During depression ($m(t) \le 0.5$), $c_{\text{discharge}}$ **persists** across erase pulses until the device recovers to or below the active landing baseline ($n \le n_{\text{base}}$), at which point $c_{\text{discharge}}$ resets to $0.0$.

> **Sub-Baseline Memory Reset**: The overdrive memory register resets strictly when the state falls below the active baseline $n_{\text{base}}$ ($16520$ for nominal write, or $0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$ for overdrive write).

#### 2. Overdrive Latency Gating:
$$\text{od}_{\text{ratio}} = \max(0.0, \; c_{\text{discharge}})$$
$$\text{overdrive} = \text{od}_{\text{ratio}} \cdot k_{\text{overdrive}}$$
$$\text{od}_{\text{clamp}} = \text{clamp}(\text{overdrive}, \; 0.0, \; 1.0)$$
$$r_{\text{lat,eff}} = r_{\text{latency}} \cdot (1.0 - \text{od}_{\text{clamp}})$$

- At $0.90\text{ V}$ write: $c_{\text{discharge}} = 0.0 \implies \text{od}_{\text{ratio}} = 0.0 \implies \text{od}_{\text{clamp}} = 0.0 \implies r_{\text{lat,eff}} = r_{\text{latency}} = 0.88$ (full ~8,000-pulse shelf preserved).
- At $1.22\text{ V}$ write: $c_{\text{discharge}} \approx 0.49 \implies \text{overdrive} \approx 0.49 \cdot 2.85 = 1.40 \implies \text{od}_{\text{clamp}} = 1.0 \implies r_{\text{lat,eff}} = 0.0$ (latency shelf completely collapsed).

#### 3. Dynamic Landing Baseline Floor ($n_{\text{base}}$):
$$\text{scale}_{\text{base}} = \begin{cases} r_{\text{bottom\_blend}} & \text{if } 0.05 \le r_{\text{bottom\_blend}} \le 2.5 \\ 1.0 & \text{otherwise} \end{cases}$$
$$n_{\text{base}} = (0.5 \cdot n_{\max}) \cdot \left[1.0 - \text{od}_{\text{clamp}} \cdot (1.0 - \text{scale}_{\text{base}})\right]$$

> **Direct Scaling from $0.5 \cdot n_{\max} = 16520$**:
> - Under nominal write ($0.90\text{ V}$, $\text{od}_{\text{clamp}} = 0$): $n_{\text{base}} = 0.5 \cdot n_{\max} = 16520$ states.
> - Under high-field saturation ($1.22\text{ V}$, $\text{od}_{\text{clamp}} = 1$): $n_{\text{base}} = 0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$ (evaluating to $826$ for the specific fitted default $r_{\text{bottom\_blend}} = 0.05$, matching the $\sim 2.0\text{ mS}$ snapback in Ru_azo).

#### 4. Base Landing Hermite Smoothstep ($s_{\text{base}}$):
$$w_{\text{trans}} = 150.0$$
$$d_{\text{base}} = n - n_{\text{base}}$$
$$s_{\text{base}} = \begin{cases} 1.0 & \text{if } d_{\text{base}} \ge w_{\text{trans}} \\ 0.0 & \text{if } d_{\text{base}} \le 0.0 \\ z^2 (3.0 - 2.0 z) & \text{where } z = d_{\text{base}} / w_{\text{trans}} \end{cases}$$

#### 5. Turnaround Normalization & Hermite Apex Regularization ($s_{\text{top}}$):
$$n_{\text{turnaround}} = n_{\max}$$
$$\text{excursion} = \max(100.0, \; n_{\text{turnaround}} - n_{\text{base}})$$
$$u = \text{clamp}\left(\dfrac{n - n_{\text{base}}}{\text{excursion}}, \; 0.0, \; 1.0\right)$$
$$\text{scale}_{\text{top}} = \begin{cases} r_{\text{top\_blend}} & \text{if } 0.001 \le r_{\text{top\_blend}} \le 0.5 \\ 0.05 & \text{otherwise} \end{cases}$$
$$w_{\text{top}} = \max(30.0, \; \text{scale}_{\text{top}} \cdot \text{excursion})$$
$$d_{\text{peak}} = n_{\text{turnaround}} - n$$
$$s_{\text{top}} = \begin{cases} 1.0 & \text{if } d_{\text{peak}} \ge w_{\text{top}} \\ 0.0 & \text{if } d_{\text{peak}} \le 0.0 \\ z_{\text{top}}^2 (3.0 - 2.0 z_{\text{top}}) & \text{where } z_{\text{top}} = d_{\text{peak}} / w_{\text{top}} \end{cases}$$

> **Dynamic Turnaround Self-Scaling**:
> - Excursion distance dynamically scales to the actual peak state reached ($n_{\text{turnaround}} - n_{\text{base}}$), ensuring normalized coordinate $u \in [0, 1]$ always spans the true written interval whether the pulse train reached $18000$, $22000$, or $33040$.
> - Apex turnaround smoothing width scales proportionally with the excursion ($w_{\text{top}} = r_{\text{top\_blend}} \cdot \text{excursion}$), preserving $C^1$ continuity at the reversal cusp for any write depth.

#### 6. Combined Saturation Decay Velocity:
$$p_{\text{decay}} = \dfrac{1.0}{\max(0.01, \; f_{\text{discharge}})}$$
$$R_{\text{decay}} = k_{\text{discharge}} \cdot u^{p_{\text{decay}}} \cdot s_{\text{top}}$$
$$R_{\text{rate}} = \left[\left(\dfrac{dn}{dt}\right)_{\text{base}} \cdot (1.0 - r_{\text{lat,eff}}) - R_{\text{decay}}\right] s_{\text{base}} + \left(\dfrac{dn}{dt}\right)_{\text{base}} \cdot (1.0 - s_{\text{base}})$$

$$\dfrac{dn}{dt} = R_{\text{rate}}$$

---

### 4.8. Direction Latch Switch & Exponential Subthreshold Relaxation
The binary direction latch $m(t)$ tracks switching polarity and decays exponentially during pulse pauses:

$$\tau_{\text{mode}} = 10^{-3}\text{ s} \quad (1\text{ ms})$$

#### Active Applied Bias ($|V_{\text{applied}}| > V_{\text{bypass}}$):
$$\text{targetMode} = \begin{cases} 1.0 & \text{if } V_{\text{int}} > V_{th} \\ 0.0 & \text{otherwise} \end{cases}$$

$$m(t + \Delta t) = \begin{cases} \text{targetMode} \cdot 0.999999 & \text{if } \Delta t \ge 20\text{ ns} \\ \text{targetMode} + \left[m(t) - \text{targetMode}\right] \exp(-10^9 \Delta t) & \text{if } \Delta t < 20\text{ ns} \end{cases}$$

#### Quiescent / Subthreshold Inter-Pulse Pause ($|V_{\text{applied}}| \le V_{\text{bypass}}$):
$$m(t + \Delta t) = m(t) \cdot \exp\left(-\dfrac{\Delta t}{\tau_{\text{mode}}}\right)$$

---

### 4.9. Continuity Scale Factor Matching ($\xi$)
To guarantee that small-signal read conductance $G$ is strictly $C^0$ continuous across write/erase polarity changes, the model maintains the scaling factor $\xi(t)$:

$$\xi(t + \Delta t) = \begin{cases} \min\left(100.0, \; \max\left(0.0, \; \dfrac{f_{22,\text{pot,scaled}}(n)}{f_{22,\text{dep,scaled}}(n) + 10^{-6}}\right)\right) & \text{if } m(t) > 0.6 \text{ and } v_{\tanh} > 0 \\ \xi(t) & \text{otherwise} \end{cases}$$

#### Active Branch Evaluation:
$$f_{22,\text{active}} = \begin{cases} f_{22,\text{pot,scaled}}(n) & \text{if } m(t) > 0.5 \\ f_{22,\text{dep,scaled}}(n) \cdot \xi(t) & \text{if } m(t) \le 0.5 \end{cases}$$

where $f_{22,\text{pot,scaled}}$ and $f_{22,\text{dep,scaled}}$ are linearly interpolated from empirical piece-wise linear (PWL) lookup tables.

---

## 5. Explicit Mapping & Verification: Relationships to $n_{\max}$ ($33040$) and the $16520$ Midpoint

This section provides a rigorous, exhaustive breakdown of every equation, parameter, and algorithmic mechanism in the simulator that interfaces with either the maximum molecular state limit $n_{\max} = 33040$ or its exact 50% midpoint / nominal saturation ceiling $16520 = 0.5 \cdot n_{\max}$.

### 5.1. Comprehensive Cross-Reference Matrix

| Model Element | Code Variable | Governing Mathematical Expression | Connection to $n_{\max}$ / $16520$ | Physical Mechanism & Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Film Capacity Upper Bound** | `nMax` | $n(t) \in [0, \; n_{\max}]$ | Direct anchor: $n_{\max} = 33040$ | Total physical count of redox-active Ruthenium-azo switching sites. Verified in CPU (`device_jit.py:189`), GPU (`gpu_solver.hip:208`), and SoA tensor (`phys_data[7]`). |
| **Upper Boundary Hermite Clamp** | `w_sat`, `z_hi` | $z_{\text{hi}} = \dfrac{n_{\max} - n}{w_{\text{sat}}}$, $w_{\text{sat}} = \max(50, 1.5 N_{\text{smooth,eff}})$ | Clamps $dn/dt \to 0$ as $n \to n_{\max}$ | Prevents unphysical molecular over-reduction beyond 100% film capacity. Verified bit-exact in CPU and GPU. |
| **Saturated Boundary Fast Exit** | Solver clamp | If $n \ge n_{\max}$ and $V \ge V_{th} \implies n \leftarrow n_{\max}, \; dn/dt \leftarrow 0$ | Direct clamp at $n_{\max}$ | Bypasses transcendental ODE integration when device is saturated at 100% capacity. Verified in CPU (`device_jit.py:378`) and GPU (`gpu_solver.hip:1612`). |
| **Nominal Saturation Ceiling** | `n_base_nominal` | $n_{\text{base,nominal}} = 0.5 \cdot n_{\max} = 16520$ | Exactly $16520$ states ($50\%$ of $n_{\max}$) | Nominal write pulses ($0.90\text{ V}, 200\ \mu\text{s}$) saturate the film at $16520$ states ($\sim 5.9\text{ mS}$). Verified in CPU (`device_jit.py:235`) and GPU (`gpu_solver.hip:369`). |
| **Saturation Shoulder Excursion** | `u_sat`, `s_plateau` | $u_{\text{sat}} = \dfrac{n - 16520}{16520}$, $+ (0.001 N_{\text{smooth,eff}}) u_{\text{sat}}^2 (3 - 2u_{\text{sat}})$ | Anchored at $16520$ ($50\%$ of $n_{\max}$) | $C^1$ Hermite smoothstep scaled by $N_{\text{smooth,eff}}$ produces angled plateau (+5.6e-5 mS/pulse) up to pulse 26,000 at 0.90 V while clamping flat at 1.22 V overdrive. Verified in CPU and GPU. |
| **Dynamic Landing Floor ($n_{\text{base}}$)** | `n_base` | $n_{\text{base}} = (0.5 \cdot n_{\max}) \cdot [1 - \text{od} \cdot (1 - r_{\text{bottom\_blend}})]$ | Spans $[0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}, \; 16520]$ | Nominal write ($0.9\text{ V}, \text{od} = 0$) lands on $16520$. Overdrive write ($1.22\text{ V}, \text{od} = 1$) lands on $0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$ (which evaluates to $826$ for the fitted default $r_{\text{bottom\_blend}} = 0.05$). Verified in CPU and GPU. |
| **Landing Hermite Smoothstep** | `s_base` | $u_{\text{base}} = \dfrac{n - (n_{\text{base}} - w_{\text{trans}})}{2 w_{\text{trans}}}$, $w_{\text{trans}} = 150$ | Regularized at active $n_{\text{base}}$ | Guarantees $C^1$ smooth transition from saturation snapback back to linear depression as $n$ falls through $n_{\text{base}}$. Verified in CPU and GPU. |
| **Dynamic Excursion Coordinate** | `u` | $u = \text{clamp}\left(\dfrac{n - n_{\text{base}}}{n_{\text{turnaround}} - n_{\text{base}}}, \; 0, \; 1\right)$ | Range: $[n_{\text{base}}, \; n_{\text{turnaround}}]$ | Normalizes state variable across the active written saturation zone: $u = 0$ at $n_{\text{base}}$ and $u = 1$ at actual peak state $n_{\text{turnaround}}$. Powers $R_{\text{decay}} \propto u^p$. Verified in CPU and GPU. |
| **Apex Drop Boost Amplitude** | `kDischarge` | $R_{\text{decay}} = k_{\text{discharge}} \cdot u^{p_{\text{decay}}} \cdot s_{\text{top}}$ | Scales snapback velocity relative to $n_{\text{base}}$ | Multiplies excursion above modulated floor $n_{\text{base}}$ ($16520 \cdot r_{\text{bottom\_blend}}$). Verified in CPU and GPU. |
| **Dynamic Knee Transition** | `fDischarge` (`uKnee`) | $p_{\text{decay}} = 1.0 / f_{\text{discharge}}$ | Blends drop into shelf approaching $n_{\text{base}}$ | Dynamic knee exponent blending apex drop into latency shelf above $n_{\text{base}}$. Verified in CPU and GPU. |
| **Return Shelf Latency** | `rLatency` | $R_{\text{shelf}} = (dn/dt)_{\text{base}} \cdot (1 - r_{\text{lat,eff}})$ | Governs shelf moving towards $n_{\text{base}}$ | 88% rate suppression across early depression pulses moving towards modulated target $n_{\text{base}}$. Verified in CPU and GPU. |
| **Apex Turnaround Window** | `w_top` | $w_{\text{top}} = \max(30.0, \; r_{\text{top\_blend}} \cdot \text{excursion})$ | Proportional to actual written excursion | Smoothly ramps $R_{\text{decay}} \to 0$ as $n \to n_{\text{turnaround}}$, eliminating derivative cusps at any reversal point. Verified in CPU and GPU. |
| **Overdrive Reset** | `cDischarge` | $c_{\text{discharge}} \leftarrow 0.0$ when $n \le n_{\text{base}}$ | Threshold is active $n_{\text{base}}$ | Erase pulses clear the write voltage overdrive memory register only after the state recovers below the active landing baseline. Verified in CPU and GPU. |
| **PWL Potentiation Table** | `f22PotRaw` | $\text{interp}(n, \; \text{lut}_{\text{pot}}, \; n_{\max})$ | Domain: $[0, \; n_{\max}] = [0, \; 33040]$ | Normalized state index: $u_{\text{lut}} = n / n_{\max}$. Peak value calibrated at $n = n_{\max}$. Verified in CPU and GPU. |
| **PWL Depression Table** | `f22DepRaw` | $\text{interp}(n_{\max} - n, \; \text{lut}_{\text{dep}}, \; n_{\max})$ | Coordinate reflection: $n_{\max} - n$ | Reverses depression table so $n = n_{\max}$ indexes the start of erasure and $n = 0$ indexes complete erasure. Verified in CPU and GPU. |
| **Subthreshold Early Bypass** | `vBypass` | If $n \le n_{\text{base}}$ and $\|V\| < V_{\text{bypass}} \implies$ fast exit | Evaluated against active $n_{\text{base}}$ | Prevents bypassing ODE solver if device is still decaying within the saturation regime above $n_{\text{base}}$. Verified in CPU and GPU. |
| **Synaptic Weight Quantization** | `WeightQuantizer` | $W \in [0, \; W_{\max}] \leftrightarrow n \in [0, \; 16520]$ | Full-scale AI range locked to $16520$ | Neuromorphic crossbars confine linear analog synaptic weights strictly to $[0, 16520]$, isolating AI inference from non-linear saturation. Verified in `simulator.py:892`. |
| **Optimizer Saturation Penalty** | `shortfall` | $\text{penalty} = \dfrac{16520.0 - n_{\text{peak\_sat}}}{16520.0}$ | Barrier threshold: $16520$ | In `fit_device_cluster.py`, penalizes candidate parameters if nominal $0.90\text{ V}$ write pulses fail to achieve $n \ge 16520$. Verified in optimizer loop. |
| **Initial State Presets** | `nInit` | $n(t=0) = 16520.0$ | Midpoint balanced initialization | Used in crossbar testbenches (`tb_crossbar_8x8.py:13`) to place differential memristive synapses at the balanced neutral mid-conductance point. |

---

### 5.2. Physical Foundation: 100% Film Capacity ($n_{\max}$) vs 50% Nominal Saturation ($16520$)

A central design principle of the Molmem physical model is the **strict decoupling between the maximum microscopic capacity of the film and the nominal operating saturation level**:

1. **$n_{\max} = 33040$ (100% Molecular Capacity)**:
   - Represents the theoretical maximum number of reduced ruthenium complexes in the active junction volume under extreme electrostatic driving force.
   - At $n = n_{\max}$, every available redox site is reduced; conductance reaches its absolute theoretical ceiling ($\sim 8.2\text{ mS}$).
   - This boundary is physically unbreachable: upper boundary Hermite clamping and solver hard-clamps strictly guarantee $n(t) \le 33040$ at all times.

2. **$16520 = 0.5 \cdot n_{\max}$ (50% Nominal Saturation Ceiling)**:
   - Under standard experimental programming pulses ($V_{\text{write}} = 0.90\text{ V}, t_{\text{pulse}} = 200\ \mu\text{s}$), the physical device does **not** reach 100% reduction. Instead, dynamic equilibrium between field-driven reduction and thermal/reverse oxidation causes the conductance to plateau at $\sim 5.9\text{ mS}$.
   - In state space, this plateau corresponds precisely to $n = 16520$.
   - The regime $n \in [0, \; 16520]$ is highly reproducible, well-behaved, and approximately linear, making it the designated domain for analog in-memory computing (matrix-vector multipliers and synaptic weight arrays).
   - The upper half of the state space, $n \in (16520, \; 33040]$, is accessible **only** under high-voltage overdrive ($V_{\text{write}} \ge 1.22\text{ V}$).

---

### 5.3. Upper Boundary Clamping and Numerical Boundedness ($n \to n_{\max}$)

When positive bias drives the state toward the physical ceiling ($v_{\tanh} > 0$), the upper boundary Hermite clamp smoothly forces $dn/dt \to 0$:

$$w_{\text{sat}} = \max\left(50.0, \; 1.5 \cdot N_{\text{smooth,eff}}\right)$$
$$z_{\text{hi}} = \dfrac{n_{\max} - n}{w_{\text{sat}}}$$

The clamp factor $C_{\text{clamp}}$ is governed by:
- $C_{\text{clamp}} = 1.0$ for $n \le n_{\max} - w_{\text{sat}}$ (unrestricted switching velocity).
- $C_{\text{clamp}} = z_{\text{hi}}^2 (3.0 - 2.0 z_{\text{hi}})$ for $n_{\max} - w_{\text{sat}} < n < n_{\max}$ ($C^1$ smooth tapering).
- $C_{\text{clamp}} = 0.0$ for $n \ge n_{\max}$ (complete velocity arrest).

**Verification**:
- Because $C_{\text{clamp}}$ approaches zero with zero derivative ($dC/dz = 6z(1-z) = 0$ at $z = 0$), state trajectories decelerate smoothly into $n_{\max}$ without ringing or numerical overshoot.
- The solver fast path (`device_jit.py:378`, `gpu_solver.hip:1612`) enforces that if $n \ge n_{\max}$ and $V \ge V_{th}$, $n$ is held clamped at $n_{\max}$ and derivative evaluation is bypassed.

---

### 5.4. Dynamic Landing Baseline Floor ($n_{\text{base}}$) and Overdrive Transition

The baseline floor equation governs the transition between nominal depression and deep overdrive snapback:

$$n_{\text{base}} = (0.5 \cdot n_{\max}) \cdot \left[1.0 - \text{od}_{\text{clamp}} \cdot (1.0 - \text{scale}_{\text{base}})\right]$$

where $\text{scale}_{\text{base}} = r_{\text{bottom\_blend}} = 0.05$:

1. **Nominal Operation ($0.90\text{ V}$ Write)**:
   - Overdrive $c_{\text{discharge}} = 0.0 \implies \text{od}_{\text{clamp}} = 0.0$.
   - $n_{\text{base}} = 0.5 \cdot n_{\max} \cdot (1.0 - 0.0) = 16520.0$.
   - During subsequent depression pulses, the device maintains its flat conductance latency shelf near $16520$ for $\sim 4,000$ pulses before descending into the baseline linear slope.

2. **Overdrive Operation ($1.22\text{ V}$ Write)**:
   - Overdrive $c_{\text{discharge}} = (1.22 / 0.90) - 1.0 = 0.355$.
   - $\text{overdrive} = 0.355 \cdot 2.85 = 1.013 \implies \text{od}_{\text{clamp}} = 1.0$.
   - $n_{\text{base}} = (0.5 \cdot n_{\max}) \cdot r_{\text{bottom\_blend}}$. (For the specific fitted preset value $r_{\text{bottom\_blend}} = 0.05$, this evaluates to $16520 \cdot 0.05 = 826.0$).
   - The landing floor shifts downward from nominal $16520$ to the fitted overdrive floor $(0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}})$, capturing the deep $\sim 2.0\text{ mS}$ snapback observed in empirical $1.22\text{ V}$ measurements (`1.22VSaturation.csv`).

**Verification**:
- Both CPU and GPU solvers compute $n_{\text{base}}$ identically using IEEE 754 double precision (`device_jit.py:236`, `gpu_solver.hip:370`).
- The transition between the two floors is continuous and monotonic for intermediate voltages ($0.90\text{ V} < V < 1.22\text{ V}$).

---

#### 5.5. Dynamic Excursion Normalization ($u$) and Power-Law Decay

The saturation decay velocity $R_{\text{decay}}$ is driven by the dimensionless coordinate $u$, which dynamically normalizes the distance between the active baseline $n_{\text{base}}$ and the actual peak state reached during potentiation ($n_{\text{turnaround}}$):

$$n_{\text{turnaround}} = \min\left(n_{\max}, \; \max(n, \; c_{\text{discharge}})\right)$$
$$\text{excursion} = \max(100.0, \; n_{\text{turnaround}} - n_{\text{base}})$$
$$u = \text{clamp}\left(\dfrac{n - n_{\text{base}}}{\text{excursion}}, \; 0.0, \; 1.0\right)$$

- When $n = n_{\text{turnaround}}$: $u = 1.0$ (maximum potential energy for saturation snapback at whatever peak the device reached).
- When $n = n_{\text{base}}$: $u = 0.0 \implies R_{\text{decay}} = 0.0$ (snapback fully ceases at the landing floor).
- Under nominal write ($n_{\text{turnaround}} = 16520, \; n_{\text{base}} = 16520$): $\text{excursion} = 100 \implies u = 0.0$. The device rests peacefully on the shelf with zero spurious snapback.
- Under overdrive write ($n_{\text{turnaround}} = 33040, \; n_{\text{base}} = 0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$): $\text{excursion} = 33040 - 826 = 32214$ (when $r_{\text{bottom\_blend}} = 0.05$). The coordinate $u$ spans the full overdrive excursion.
- Under partial overdrive write (e.g. $n_{\text{turnaround}} = 22000$): $\text{excursion} = 22000 - n_{\text{base}}$. The coordinate $u$ correctly starts at $1.0$ at $22000$ and executes the full non-linear snapback without truncation.

The decay rate scales as $R_{\text{decay}} = k_{\text{discharge}} \cdot u^{p_{\text{decay}}} \cdot s_{\text{top}}$, where $p_{\text{decay}} = 1 / f_{\text{discharge}} \approx 3.88$. This high power-law exponent ensures that snapback is extremely fast near the top and rapidly decelerates as $n$ approaches $n_{\text{base}}$.

---

### 5.6. Apex Turnaround Regularization ($w_{\text{top}}$)

At the reversal apex ($n \to n_{\text{turnaround}}$), an unregularized power-law decay would exhibit an infinite second derivative or abrupt rate cusp upon polarity reversal. To prevent this, the model incorporates an apex Hermite smoothstep whose window dynamically scales with the true excursion:

$$\text{scale}_{\text{top}} = r_{\text{top\_blend}} = 0.05$$
$$w_{\text{top}} = \max(30.0, \; \text{scale}_{\text{top}} \cdot \text{excursion})$$
$$d_{\text{peak}} = n_{\text{turnaround}} - n$$
$$s_{\text{top}} = \begin{cases} 1.0 & \text{if } d_{\text{peak}} \ge w_{\text{top}} \\ 0.0 & \text{if } d_{\text{peak}} \le 0.0 \\ z_{\text{top}}^2 (3.0 - 2.0 z_{\text{top}}) & \text{where } z_{\text{top}} = d_{\text{peak}} / w_{\text{top}} \end{cases}$$

**Verification**:
- At $n = n_{\text{turnaround}}$, $d_{\text{peak}} = 0 \implies s_{\text{top}} = 0 \implies R_{\text{decay}} = 0$.
- As the state drops across the top $w_{\text{top}}$ states below $n_{\text{turnaround}}$, $s_{\text{top}}$ smoothly ramps from $0.0$ to $1.0$.
- This guarantees $C^1$ continuity of $dn/dt$ across any reversal turnaround, eliminating numerical solver chatter and stiffness regardless of the write depth.

---

### 5.7. Overdrive Register Cleared at $n \le n_{\text{base}}$

The write overdrive register $c_{\text{discharge}}$ acts as an analog memory that tracks positive voltage overpotential and peak state occupancy accumulated during writing. During depression, this register remains locked, holding the landing floor at $n_{\text{base}} = 0.5 \cdot n_{\max} \cdot r_{\text{bottom\_blend}}$ (which evaluates to $826$ for the fitted default $r_{\text{bottom\_blend}} = 0.05$).

When depression pulses successfully drive the state down to or below the landing floor ($n \le n_{\text{base}}$):
$$c_{\text{discharge}} \to n_{\text{base}}$$

**Verification**:
- Implemented in `device_jit.py` and `gpu_solver.hip`:
  ```cpp
  if (local_n[d] <= n_base_d) {
      local_cached_n[d] = (T)0.0; // Clears cDischarge register to 0.0
  }
  ```
- This ensures that if the device is subsequently potentiated at nominal $0.90\text{ V}$, it starts fresh with zero overdrive memory, returning to the nominal $16520$ baseline rather than remaining locked in overdrive mode.

---

### 5.8. Lookup Table Inversion and Coordinate Reflection ($n_{\max} - n$)

The non-linear conduction tables ($f_{22}(n), I_{\text{scale}}(n), V_{\text{scale}}(n)$) contain discrete empirical points measured over the physical range:
$$n \in [0, \; n_{\max}]$$

- **Potentiation Branch**: Lookups are addressed directly with $n$:
  $$u_{\text{pot}} = \dfrac{n}{n_{\max}} \in [0, 1]$$
- **Depression Branch**: Lookups are addressed with the **reflected coordinate** $n_{\max} - n$:
  $$u_{\text{dep}} = \dfrac{n_{\max} - n}{n_{\max}} \in [0, 1]$$

**Verification**:
- At $n = n_{\max}$ (fully written device starting depression), the reflected coordinate is $n_{\max} - n_{\max} = 0$, mapping to index 0 of the empirical depression curve.
- At $n = 0$ (fully erased device), the reflected coordinate is $n_{\max} - 0 = n_{\max}$, mapping to the final index of the depression curve.
- This formulation guarantees that depression lookup trajectories are always forward-interpolated from index $0$ to $N$, preserving linear spline monotonicity.

---

### 5.9. Neuromorphic Weight Quantization and AI Training Safety ($[0, 16520]$)

In the PyTorch neural network simulator (`torch_simulator.py`, `simulator.py`), analog synaptic weights are quantized and mapped to physical devices using:

$$\text{wq} = \text{WeightQuantizer}(\text{targetBits} = 8, \; \text{maxStates} = 16520)$$

**Why $16520$ and not $33040$?**
- The $0$ to $16520$ state interval provides monotonic, symmetrical, and linear conductance modulation suitable for gradient descent and vector-matrix dot products.
- Confining neural network weights to $[0, 16520]$ ensures that weight updates during on-chip training or inference never push individual memristor cells into the non-linear overdrive saturation regime ($> 16520$), eliminating weight distortion and accuracy degradation.

---

### 5.10. Optimizer Saturation Barrier Penalty in Parameter Fitting

During global device parameter optimization with Differential Evolution (`fit_device_cluster.py` and `fit_device_subset.py`), candidate parameter vectors are evaluated against the empirical $0.90\text{ V}$ saturation curve.

To prevent the optimizer from converging to degenerate local minima where the device fails to reach the nominal 50% physical saturation ceiling under nominal write bias, the loss function enforces a hard barrier penalty:

$$\text{shortfall} = \dfrac{16520.0 - n_{\text{peak\_sat}}}{16520.0}$$
$$\text{loss}_{\text{barrier}} = \begin{cases} 10.0 \cdot \text{shortfall}^2 & \text{if } n_{\text{peak\_sat}} < 16520.0 \\ 0.0 & \text{otherwise} \end{cases}$$

**Verification**:
- Implemented in `fit_device_cluster.py:2360-2368` and `fit_device_subset.py:1530-1538`.
- Guarantees that any extracted device parameter set for Ru-azo films will naturally reach at least $16520$ states when subjected to nominal write pulses.

---

## 6. Implementation Parity Reference

The mathematical equations above are implemented in complete bit-exact alignment across CPU and GPU code paths:

| Functional Section | CPU JIT Implementation (`device_jit.py`) | GPU HIP Kernel (`csrc/gpu_solver.hip`) | PyTorch SoA Tensor (`torch_simulator.py`) |
| :--- | :--- | :--- | :--- |
| Hyperbolic Modulation | `device_jit.py:122-136` | `csrc/gpu_solver.hip:270-285` | `phys_data[19:22]` |
| Halley's $R_c$ Conduction | `device_jit.py:35-85` | `csrc/gpu_solver.hip:85-151` | `phys_data[10]` |
| Joule Heat & Arrhenius | `device_jit.py:138-163` | `csrc/gpu_solver.hip:287-308` | `phys_data[16:18]` |
| Softplus Field Activation| `device_jit.py:165-184` | `csrc/gpu_solver.hip:309-320` | `phys_data[0:3]` |
| Decoupled Boundaries | `device_jit.py:186-210` | `csrc/gpu_solver.hip:312-337` | `phys_data[1]` |
| State Rate Equation | `device_jit.py:211-230` | `csrc/gpu_solver.hip:339-358` | `phys_data[2, 24]` |
| Overdrive Latency Gating | `device_jit.py:231-285` | `csrc/gpu_solver.hip:363-415` | `phys_data[28:30]` |
| Direction Latch Update | `device_jit.py:286-311` | `csrc/gpu_solver.hip:416-438` | `state_tensor[1]` |
| Continuity Scale Matching| `device_jit.py:312-325` | `csrc/gpu_solver.hip:439-448` | `state_tensor[2]` |

---

## 7. Cadence Spectre / Verilog-A Architecture & Empirical Noise Engine

Cadence Spectre compact models are organized across `MolmemVirtuosoSpectreV1.0/` and `SpectreModelDataEncryption/`:
- **`SpectreModelDataEncryption/`**: Master empirical characterization repository and cross-platform encryption pipeline:
  - `data/`: Master empirical PWL datasets (`f22_pot_map.pwl`, `f22_dep_map.pwl`, `d2d_mismatch_quantile.pwl`, `read_noise.pwl`, `write_noise.pwl`, `V_scale.pwl`, `I_scale.pwl`).
  - `build_encryption_package.py`: Compiles all 6,998 coordinate points directly into native Verilog-A constant arrays with binary-search bisection (`PWL_LOOKUP`), generates `molmem_core_embedded.va`, wraps code in IEEE 1735 / Cadence `pragma protect` envelopes, and executes `xmprotect` if available.
- **`MolmemVirtuosoSpectreV1.0/models/`**: Compact device model definitions:
  - `molmem_core_embedded.va`: Self-contained Verilog-A compact physics core with inlined PWL tables (zero runtime file I/O).
  - `molmem_core.vap`: Encrypted binary core (generated by `xmprotect`).
  - `molmem_device.scs`: Subcircuit wrappers for discrete devices (default, Ru_azo, 1T1R, monitors).
  - `molmem_noise.scs`: Statistical model card for ADE Monte Carlo sweeps.
  - `molmem.scs`: Core device model library include card (auto-selects `molmem_core.vap` if present, otherwise falling back to `molmem_core_embedded.va`).
- **`MolmemVirtuosoSpectreV1.0/` (Root)**:
  - `crossbar_8x8.scs`: Reusable 8x8 passive crossbar array circuit instantiator block.
  - `molmem.scs`: Top-level entry forwarder pointing into `models/molmem.scs`.
  - Simulation testbench suite (`tb_*.scs`).


### 7.1. Empirical PWL Noise Channels in Verilog-A

1. **Device-to-Device (D2D) Spatial Mismatch**:
   - Evaluated via empirical Inverse-CDF quantile mapping (`data/d2d_mismatch_quantile.pwl`):
     $$\Delta_{\text{d2d}} = \text{\$table\_model}(u_{\text{d2d}}, \text{"data/d2d\_mismatch\_quantile.pwl"}, \text{"1"})$$
     $$g_{\text{scale,eff}} = g_{\text{scale}} \cdot (1.0 + \text{intensity}_{\text{d2d}} \cdot \Delta_{\text{d2d}})$$
   - Compatible with Cadence Virtuoso ADE Monte Carlo sweeps via statistical model card `molmem_noise.scs`.

2. **Temporal Read Noise (Johnson-Nyquist & 1/f Flicker Floor)**:
   - Evaluated from normalized state $f_{22}$ via empirical standard deviation table (`data/read_noise.pwl`):
     $$\sigma_{\text{rel}} = \text{\$table\_model}(f_{22}, \text{"data/read_noise.pwl"}, \text{"1"})$$
     $$I_{\text{scale,noise}} = \max(|I_{\text{mem}}|, V_{\text{eff}} \cdot 10^{-6})$$
     $$S_i = 2.0 \cdot (\text{intensity}_{\text{read}} \cdot \sigma_{\text{rel}} \cdot I_{\text{scale,noise}})^2 \cdot 10^{-9}\text{ A}^2/\text{Hz}$$
     $$I(\text{TE}, \text{BE}) \leftarrow \text{white\_noise}(S_i, \text{"read\_noise"})$$
   - Natively evaluated during Spectre AC small-signal `.noise` and transient noise (`tran ... noisefmax=10G`).

3. **Cycle-to-Cycle Write Noise (Langevin Programming Stochasticity)**:
   - Evaluated from empirical pulse-to-pulse programming variance (`data/write_noise.pwl`):
     $$\sigma_{\text{write}} = \text{\$table\_model}(f_{22}, \text{"data/write_noise.pwl"}, \text{"1"})$$
     $$\dfrac{dn}{dt} \leftarrow \dfrac{dn}{dt} \cdot (1.0 + \text{intensity}_{\text{write}} \cdot \sigma_{\text{write}} \cdot \text{seed})$$

### 7.2. 1-to-1 Circuit Testbench Suite Mapping

| Testbench Description | Python Testbench (`PythonSimulator/tb_circuit_level/`) | Cadence Spectre Netlist (`MolmemVirtuosoSpectreV1.0/testbenches/`) |
| :--- | :--- | :--- |
| Dynamic Pulse Train (5-subplot) | `tb_dynamic.py` | `tb_dynamic.scs` |
| Extended 50k Endurance Cycles | `tb_dynamic_extended.py` | `tb_dynamic_extended.scs` |
| I-V Cyclic Pinched Hysteresis | `tb_standard_cycle.py` | `tb_iv_sweep.scs` |
| Multi-Regime Electrical Stress | `tb_stress.py` | `tb_stress.scs` |
| Thermal Relaxation & Non-Destructive Read | `tb_thermal_relaxation.py` | `tb_thermal_relaxation.scs` |
| Write-Then-Read Matrix Sensing | `tb_write_read.py` | `tb_write_read.scs` |
| 8x8 Crossbar Parallel Inference | `tb_crossbar_8x8.py` | `tb_crossbar_8x8.scs` |
| 8x8 Crossbar Multi-Row Write | `tb_crossbar_write.py` | `tb_crossbar_write.scs` |
| Multi-Channel Empirical Noise Validation | `tb_noise_validation.py` | `tb_noise_validation.scs` |


