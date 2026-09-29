import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import PchipInterpolator

# --- Core Φ_mol Mathematical Functions ---
# Copied from MolMemristor_Figure7.py to make this script self-contained.

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)

def x_of_z_moving_average(z_values: np.ndarray, p: float, q: float, k: float, alpha: float, w: float) -> np.ndarray:
    """
    Moving-average smoothing with tapered blend to preserve endpoints.
    """
    def u_of_t(tt: np.ndarray) -> np.ndarray:
        s = (1.0 - (1.0 - tt) ** q) ** p
        return (1.0 + k) * s / (1.0 + k * (s ** 2))

    t = z_values / 30.0
    alpha_c = np.clip(alpha, 0.0, 1.0)
    w_c = np.clip(w, 0.0, 1.0)
    t_minus = np.clip(t - w_c, 0.0, 1.0)
    t_plus = np.clip(t + w_c, 0.0, 1.0)
    u_t = u_of_t(t)
    u_ma = 0.25 * u_of_t(t_minus) + 0.5 * u_t + 0.25 * u_of_t(t_plus)
    if w_c <= 0.0:
        a_eff = 0.0
    else:
        aL = _smoothstep01(np.clip(t / max(w_c, 1e-6), 0.0, 1.0))
        aR = _smoothstep01(np.clip((1.0 - t) / max(w_c, 1e-6), 0.0, 1.0))
        a_eff = alpha_c * (aL * aR)
    x = (1.0 - a_eff) * u_t + a_eff * u_ma
    return 0.7 * np.clip(x, 0.0, 1.0)

def logistic(x: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


class ConductanceModel:
    """
    A framework for developing a SPICE-friendly analytical model for the
    memristor's current-pulse behavior, based on the physics described in
    the research paper.
    
    This class follows the three-stage plan outlined in plan.txt and now
    includes the logic for calculating Φ_mol(z,n) directly.
    """
    def __init__(self, params_path: Path):
        self.param_names = ['p_i', 'q_i', 'k_i', 'a_i', 'w_i', 'p_f', 'q_f', 'k_f', 'a_f', 'w_f', 'z_mid', 'z_width']
        self._build_parameter_interpolators(params_path)
        self._initialize_constants()
        print("ConductanceModel initialized and ready.")

    def _build_parameter_interpolators(self, params_path: Path):
        """
        Reads the fitted_parameters.csv file and builds smooth interpolation
        functions for each of the 12 model parameters.
        """
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found at {params_path}. Please run MolMemristor_Figure7.py first.")

        df = pd.read_csv(params_path).sort_values(by="Pulse Counts").dropna()
        self.pulse_counts = df["Pulse Counts"].values

        self.param_interpolators = {}
        for param in self.param_names:
            param_values = df[param].values
            # Use smooth, monotonic PCHIP interpolation
            self.param_interpolators[param] = PchipInterpolator(self.pulse_counts, param_values, extrapolate=True)
        print("Parameter interpolators built successfully.")

    def get_phim_curve(self, n: float, num_points: int = 400) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Generates a Φ_mol(z,n) curve by evaluating the 12 parameter interpolators
        at pulse count 'n' and then applying the core mathematical model.
        """
        try:
            params_list = [self.param_interpolators[p](n) for p in self.param_names]
        except KeyError:
            print("Error: Parameter interpolators have not been built correctly.")
            return None

        params_initial = np.array(params_list[0:5])
        params_final = np.array(params_list[5:10])
        z_mid, z_width = params_list[10], params_list[11]
        
        z_grid = np.linspace(0.0, 30.0, num_points)
        
        z_norm = (z_grid - z_mid) / max(1e-9, z_width)
        w_final = np.array([logistic(val) for val in z_norm])
        w_initial = 1.0 - w_final

        p_i, q_i, k_i, a_i, w_i = params_initial
        p_f, q_f, k_f, a_f, w_f = params_final
        x_initial_base = x_of_z_moving_average(z_grid, p_i, q_i, k_i, a_i, w_i)
        x_final_base = x_of_z_moving_average(z_grid, p_f, q_f, k_f, a_f, w_f)

        phim_curve = w_initial * x_initial_base + w_final * x_final_base
        
        return z_grid, phim_curve

    def _initialize_constants(self):
        """Initialize physical and model constants from the paper."""
        self.kB = 8.617e-5  # Boltzmann constant (eV/K)
        self.T = 298.0      # Temperature (K)
        self.e = 1.6e-19    # Electron charge (C)
        self.G22 = 0.1      # Gibbs free energy for state '22' (eV)
        self.lam = 0.04     # Reorganization energy for state '22' (eV)
        self.A_ET = 100.0   # Pre-factor for k_ET calculation
        
        # --- Nucleation Model Parameters (from MATLAB BLOCK-A/B) ---
        self.k_ET_threshold = 0.3 # Switching threshold, from MATLAB
        # Potentiation (SET) parameters
        self.fp1_pot = -30.0
        self.fp2_pot = 15.0
        self.k_N_pot = 800.0
        
        # Depotentiation (RESET) parameters
        self.fp1_dep = -8.28
        self.fp2_dep = 16608.0
        self.k_N_dep = 500.0

        # --- Current Model Parameters (from MATLAB BLOCK-C) ---
        self.N_sites = 30          # Number of molecular sites
        self.delE = 0.07           # Energy level difference (eV)
        self.lamE = 0.05           # Reorganization energy for current (eV)
        self.AF = 1.0e5            # Current scaling factor
        self.cM31, self.cL31, self.cR31 = 8.0e7, 1.0e8, 1.0e8
        self.cM22, self.cL22, self.cR22 = 3.1e13, 8.0e12, 4.0e12

        print("Physical constants initialized.")

    # --- Stage 0: Automated Model Tuning ---
    def tune_A_ET_factor(self, target_p1: float, target_p2: float):
        """
        Action 0: Automatically tunes the A_ET pre-factor to match the p1/p2
        parameters from a reference model (e.g., MATLAB).
        """
        print("\n--- Stage 0: Tuning A_ET to Match Reference p1/p2 ---")

        def objective_function(A_ET_array):
            """
            This function is minimized by scipy.optimize. It calculates the
            error between the fitted p1/p2 and the target values for a given A_ET.
            """
            A_ET_val = A_ET_array[0]
            if A_ET_val <= 0: return 1e9 # Avoid non-physical values

            # --- Run the core fitting logic with the new A_ET ---
            # 1. Re-characterize and re-fit k_ET. We do this quietly to
            #    avoid flooding the console during optimization.
            phim_range = np.linspace(0.0, 0.7, 100)
            k_ET_physical = self.characterize_k_ET_physical(phim_range, A_ET_override=A_ET_val, verbose=False)
            self.fit_k_ET_analytical(phim_range, k_ET_physical, verbose=False)
            
            # 2. Re-calculate the n(Z22) dynamic fit
            n_fit_range = np.arange(1000, 16001, 1000)
            z22_values = []
            for n in n_fit_range:
                z_grid, phim_curve = self.get_phim_curve(n)
                k_ET_curve = self.k_ET_analytical(phim_curve)
                theta = (k_ET_curve > self.k_ET_threshold).astype(float)
                z_step = z_grid[1] - z_grid[0]
                z22 = np.sum(theta) * z_step
                z22_values.append(z22)

            z22_values = np.array(z22_values)
            valid_indices = z22_values > 1e-6
            if not np.any(valid_indices):
                return 1e9 # Return a large error if no switching occurs

            p1_fit, p2_fit = np.polyfit(z22_values[valid_indices], n_fit_range[valid_indices], 1)

            # 3. Calculate normalized squared error
            # We normalize by the target value to balance the scales of p1 and p2.
            # The squaring ensures the error is always positive.
            err_p1 = ((p1_fit - target_p1) / target_p1)**2
            err_p2 = ((p2_fit - target_p2) / target_p2)**2
            total_error = err_p1 + err_p2
            
            print(f"  Trying A_ET = {A_ET_val:7.2f} -> p1={p1_fit:7.2f}, p2={p2_fit:7.2f}, Error={total_error:.4f}")
            return total_error

        # --- Run the optimization ---
        initial_guess = [self.A_ET]
        result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'xatol': 1e-2, 'fatol': 1e-4})
        
        best_A_ET = result.x[0]
        print(f"  Optimization complete. Optimal A_ET = {best_A_ET:.4f}")
        
        # --- Finalize the model with the tuned value ---
        self.A_ET = best_A_ET
        print("\n--- Re-characterizing model with optimal A_ET ---")
        phim_range = np.linspace(0.0, 0.7, 100)
        k_ET_physical = self.characterize_k_ET_physical(phim_range)
        self.fit_k_ET_analytical(phim_range, k_ET_physical)


    # --- Stage 1: Model the Device's Physical State (f₂₂(n)) ---

    def characterize_k_ET_physical(self, phim_range: np.ndarray, A_ET_override: float = None, verbose: bool = True) -> np.ndarray:
        """
        Action 1.1 (Full Physics): Implements the integral in eq. (17) to
        calculate the "ground truth" electron transfer rate.
        """
        if verbose:
            print("\n--- Stage 1.1: Characterizing k_ET (Physical Model) ---")

        A_ET_to_use = A_ET_override if A_ET_override is not None else self.A_ET

        def integrand(x, phim):
            """The integrand from the MATLAB code (BLOCK-A, kET calculation)."""
            numerator = np.exp(-((self.lam + self.G22 + phim + x)**2) / (4 * self.lam * self.kB * self.T))
            denominator = np.sqrt(4 * np.pi * self.lam * self.kB * self.T)
            fermi_term = 1.0 - (1.0 / (1.0 + np.exp(x / (self.kB * self.T))))
            return (numerator / denominator) * fermi_term

        k_ET_values = []
        for phim in phim_range:
            # Integrate from -5 to 5, mirroring the MATLAB implementation's limits.
            result, _ = quad(integrand, -5, 5, args=(phim,))
            k_ET_values.append(A_ET_to_use * result)
        
        if verbose:
            print(f"Calculated {len(k_ET_values)} k_ET values.")
        return np.array(k_ET_values)

    def fit_k_ET_analytical(self, phim_values: np.ndarray, k_ET_values: np.ndarray, verbose: bool = True):
        """
        Action 1.1 (SPICE-Friendly Fit): Fits a simple analytical function
        to the characterized k_ET data. The data follows an exponential decay,
        so we fit to that instead of the originally planned tanh model.
        """
        if verbose:
            print("\n--- Stage 1.1: Fitting k_ET (Analytical Model) ---")

        def exp_model(phim, A, B):
            """Analytical model for k_ET based on an exponential decay."""
            return A * np.exp(-B * phim)

        # To get a good initial guess, fit a line to the log of the data
        log_k_ET = np.log(k_ET_values)
        valid_indices = np.isfinite(log_k_ET)
        
        # y = -B*x + log(A)
        fit_params = np.polyfit(phim_values[valid_indices], log_k_ET[valid_indices], 1)
        # initial_guess for [A, B]
        initial_guess = [np.exp(fit_params[1]), -fit_params[0]]
        
        params, _ = curve_fit(exp_model, phim_values, k_ET_values, p0=initial_guess)
        
        self.k_ET_params = {'A': params[0], 'B': params[1]}
        if verbose:
            print("Fitted k_ET analytical model parameters (Exponential):")
            print(f"  A = {self.k_ET_params['A']:.4e}")
            print(f"  B = {self.k_ET_params['B']:.4f}")
        
        return self.k_ET_params

    def k_ET_analytical(self, phim: np.ndarray) -> np.ndarray:
        """
        Calculates the electron transfer rate using the fitted analytical model.
        """
        if not hasattr(self, 'k_ET_params'):
            raise RuntimeError("The k_ET analytical model has not been fitted yet. Run fit_k_ET_analytical() first.")
        
        A = self.k_ET_params['A']
        B = self.k_ET_params['B']
        
        return A * np.exp(-B * phim)

    def calculate_f22(self, n_values: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Action 1.2: Implements the arctan-based nucleation model for the full
        potentiation (SET) and depotentiation (RESET) cycle, with nucleation
        parameters dynamically fitted from the phimol model.
        
        Returns:
            A tuple containing:
            - f22 (np.ndarray): The calculated f22 curve.
            - fit_details (dict): A dictionary with data from the n(Z22) fit.
        """
        print("\n--- Stage 1.2: Calculating f22 (Full SET/RESET Cycle from phimol) ---")
        
        # --- Step 1: Dynamically fit nucleation parameters from phimol ---
        print("  Generating Z22 vs. n data for linear fit...")
        n_fit_range = np.arange(1000, 16001, 1000)
        z22_values = []
        for n in n_fit_range:
            z_grid, phim_curve = self.get_phim_curve(n)
            # Calculate k_ET from the analytical exponential model
            k_ET_curve = self.k_ET_analytical(phim_curve)
            # Compare k_ET to the threshold to find the switched region
            theta = (k_ET_curve > self.k_ET_threshold).astype(float)
            z_step = z_grid[1] - z_grid[0]
            z22 = np.sum(theta) * z_step
            z22_values.append(z22)
        
        z22_values = np.array(z22_values)
        
        # Prepare a dictionary to hold the fitting results for plotting
        fit_details = {
            'z22_values': z22_values,
            'n_fit_range': n_fit_range,
            'p1_fit': None,
            'p2_fit': None,
            'valid_indices': np.array([])
        }

        if np.all(z22_values < 1e-6):
             print("  Warning: All Z22 values are near zero. Cannot perform fit.")
             return np.zeros_like(n_values), fit_details

        valid_indices = z22_values > 1e-6
        if not np.any(valid_indices):
            print("  Warning: No valid Z22 values > 0 to perform fit.")
            return np.zeros_like(n_values), fit_details
        
        p1_fit, p2_fit = np.polyfit(z22_values[valid_indices], n_fit_range[valid_indices], 1)
        print(f"  Dynamically fit n(Z22): p1 = {p1_fit:.4f}, p2 = {p2_fit:.4f}")

        # Store successful fit results for plotting
        fit_details['p1_fit'] = p1_fit
        fit_details['p2_fit'] = p2_fit
        fit_details['valid_indices'] = valid_indices
        
        # --- Step 2: Calculate nucleation centers for SET and RESET ---
        # Potentiation (SET) parameters
        a1_pot = p1_fit + self.fp1_pot
        a2_pot = p2_fit + self.fp2_pot
        z_centers = np.arange(1, 31)
        chi_N_pot = a1_pot * z_centers + a2_pot
        
        # Depotentiation (RESET) parameters
        a1_dep = p1_fit + self.fp1_dep
        a2_dep = p2_fit + self.fp2_dep
        chi_N_dep = a1_dep * z_centers + a2_dep
        
        # --- Step 3: Apply Calculations to Appropriate Pulse Ranges ---
        print(f"  Calculating f22 for {len(n_values)} pulse counts...")
        f22 = np.zeros_like(n_values, dtype=float)
        
        potentiation_mask = (n_values <= 16520)
        depotentiation_mask = (n_values > 16520)
        
        # Vectorized calculation for the SET phase
        n_pot = n_values[potentiation_mask][:, np.newaxis]
        chi_N_pot_row = chi_N_pot[np.newaxis, :]
        summand_pot = (1.0 / np.pi) * np.arctan((n_pot - chi_N_pot_row) / self.k_N_pot) + 0.5
        p_values_pot = np.sum(summand_pot, axis=1)
        f22[potentiation_mask] = p_values_pot / len(z_centers)

        # Vectorized calculation for the RESET phase
        n_dep = n_values[depotentiation_mask][:, np.newaxis]
        chi_N_dep_row = chi_N_dep[np.newaxis, :]
        summand_dep = (1.0 / np.pi) * np.arctan((n_dep - chi_N_dep_row) / self.k_N_dep) + 0.5
        p_values_dep = np.sum(summand_dep, axis=1)
        f22[depotentiation_mask] = 1.0 - (p_values_dep / len(z_centers))
        
        f22 = np.clip(f22, 0.0, 1.0)
        
        # --- Step 4: Apply Linear Rescaling for Boundary Conditions ---
        # The arctan model is asymptotic. We rescale each phase of the curve
        # to ensure it starts at 0, peaks at 1, and ends at 0.
        
        start_offset = f22[0]
        end_offset = f22[-1]
        
        # Find the actual peak value, which should be near the phase transition
        peak_idx = np.argmin(np.abs(n_values - 16520))
        peak_value = f22[peak_idx]
        
        # Rescale the potentiation (SET) phase to map [start, peak] -> [0, 1]
        if peak_value > start_offset:
            f22[potentiation_mask] = (f22[potentiation_mask] - start_offset) / (peak_value - start_offset)
        
        # Rescale the depotentiation (RESET) phase to map [end, peak] -> [0, 1]
        if peak_value > end_offset:
            f22[depotentiation_mask] = (f22[depotentiation_mask] - end_offset) / (peak_value - end_offset)
        
        # Clip again to ensure bounds are strictly met
        f22 = np.clip(f22, 0.0, 1.0)
        
        print("  f22 calculation complete.")
        return f22, fit_details

    # --- Stage 2: Calculate the Current (I(n, Vr)) ---

    def _calculate_marcus_integral(self, delta_g: float, is_reduction: bool) -> float:
        """
        Helper to calculate the Marcus theory integral for electron transfer rates
        at the contacts.
        
        Args:
            delta_g: The driving force for the reaction (e.g., -lamE+delE+aR).
            is_reduction: If true, uses the fermi term for reduction, otherwise oxidation.
        """
        def integrand(x):
            numerator = np.exp(-(delta_g + x)**2 / (4 * self.lamE * self.kB * self.T))
            denominator = np.sqrt(4 * np.pi * self.lamE * self.kB * self.T)
            if is_reduction:
                fermi_term = 1.0 / (1.0 + np.exp(x / (self.kB * self.T)))
            else: # Oxidation
                fermi_term = 1.0 - (1.0 / (1.0 + np.exp(x / (self.kB * self.T))))
            return (numerator / denominator) * fermi_term

        result, _ = quad(integrand, -5, 5)
        return result

    def characterize_current_physical(self, n_range: np.ndarray, vr_range: np.ndarray) -> np.ndarray:
        """
        Action 2.1 (Full Physics): Implements the full N-site hopping model
        to calculate the "ground truth" current.
        """
        print("\n--- Stage 2.1: Characterizing Current (Physical Model) ---")
        
        # Pre-calculate f22 for all n values to speed up the main loop
        print(f"  Pre-calculating f22 for {len(n_range)} pulse counts...")
        f22_n_values, _ = self.calculate_f22(n_range)

        # Output array for the I(n, Vr) surface
        I_surface = np.zeros((len(n_range), len(vr_range)))

        print(f"  Calculating current for {len(n_range)} x {len(vr_range)} grid points...")
        
        for i, n in enumerate(n_range):
            f22 = f22_n_values[i]
            
            # 1. Calculate state-dependent couplings (linear interpolation)
            cM = self.cM31 * (1.0 - f22) + self.cM22 * f22
            cL = self.cL31 * (1.0 - f22) + self.cL22 * f22
            cR = self.cR31 * (1.0 - f22) + self.cR22 * f22

            for j, vr in enumerate(vr_range):
                # 2. Calculate voltage drops and internal hopping rates
                aL = 0.072  # Voltage drop at Left Electrode
                aR = 0.108  # Voltage drop at Right Electrode
                aM = vr - (aL + aR)
                
                kf = cM * np.exp(aM / (self.N_sites * self.kB * self.T))
                kb = cM * np.exp(-aM / (self.N_sites * self.kB * self.T))

                # 3. Calculate contact injection/ejection rates using Marcus integrals
                # Matches the terms from MATLAB's BLOCK-C
                kTf = cR * self._calculate_marcus_integral(-self.lamE + self.delE + aR, is_reduction=False) # koxR
                kTb = cR * self._calculate_marcus_integral(-self.lamE - self.delE - aR, is_reduction=True)  # krdR
                kBf = cL * self._calculate_marcus_integral(-self.lamE - self.delE + aL, is_reduction=True)  # krdL
                kBb = cL * self._calculate_marcus_integral(-self.lamE + self.delE - aL, is_reduction=False) # koxL

                # 4. Construct and solve the master equation W * P = b
                W = np.zeros((self.N_sites + 1, self.N_sites + 1))
                b = np.zeros((self.N_sites + 1, 1))

                # Row 1
                W[0, 0] = -(kBb + kf)
                W[0, 1] = kb
                W[0, self.N_sites] = kBf

                # Rows 2 to N-1
                for row in range(1, self.N_sites - 1):
                    W[row, row-1] = kf
                    W[row, row] = -(kb + kf)
                    W[row, row+1] = kb
                
                # Row N
                W[self.N_sites-1, self.N_sites-2] = kf
                W[self.N_sites-1, self.N_sites-1] = -(kb + kTf)
                W[self.N_sites-1, self.N_sites] = kTb

                # Row N+1 (conservation of probability)
                W[self.N_sites, :] = 1.0
                b[self.N_sites] = 1.0

                try:
                    P = np.linalg.solve(W, b)
                except np.linalg.LinAlgError:
                    print(f"  Warning: Singular matrix at n={n}, Vr={vr}. Setting current to 0.")
                    P = np.zeros_like(b)

                # 5. Calculate the final current
                # Note: MATLAB P(N+1) is P[N] and P(1) is P[0] in 0-indexed Python
                I = self.AF * self.e * (kBf * P[self.N_sites] - kBb * P[0])
                I_surface[i, j] = I.item()

        print("Current characterization complete.")
        return I_surface

    def fit_current_analytical(self, n_values: np.ndarray, vr_values: np.ndarray, I_values: np.ndarray):
        """
        Action 2.1 (SPICE-Friendly Fit): Fits a single, analytical 2D function
        to the characterized current surface.
        
        Model: I(n, V) = (I_scale * f22(n)) * sinh(V / V_scale)
               where f22(n) is the nucleation fraction.
        """
        print("\n--- Stage 2.1: Fitting Current (Analytical Model) ---")
        
        # We need f22 values for the fit.
        # Since we don't have them passed in directly, we calculate them.
        f22_values, _ = self.calculate_f22(n_values)
        
        # Flatten arrays for curve_fit
        # I_values is (n_samples, vr_samples)
        # We need to tile n and vr to match
        
        n_grid, vr_grid = np.meshgrid(n_values, vr_values, indexing='ij')
        f22_grid, _ = np.meshgrid(f22_values, vr_values, indexing='ij')
        
        y_data = I_values.flatten()
        
        # Define the model function for curve_fit
        # x is (f22, V)
        def current_model(x, I_scale, V_scale):
            f22, V = x
            # We add a small epsilon to f22 to avoid zero current if f22 is 0 (though physically it should be 0)
            # But the model I = f22 * ... implies I=0 at f22=0.
            # Let's add a leakage term if needed, but for now keep it simple.
            return I_scale * f22 * np.sinh(V / V_scale)

        # Stack input data
        x_data = np.vstack((f22_grid.flatten(), vr_grid.flatten()))
        
        # Initial guess
        p0 = [1e-6, 0.5] 
        
        try:
            params, cov = curve_fit(current_model, x_data, y_data, p0=p0)
            self.I_params = {'I_scale': params[0], 'V_scale': params[1]}
            print(f"  Fitted Current Model: I = {params[0]:.4e} * f22 * sinh(V / {params[1]:.4f})")
        except RuntimeError as e:
            print(f"  Fit failed: {e}")
            self.I_params = {'I_scale': 1e-6, 'V_scale': 1.0} # Fallback

        return self.I_params

    # --- Stage 3: Generate the Spectre Behavioral Model ---

    def generate_spectre_expressions(self):
        """
        Action 3.1 & 3.2: Translates the final analytical Python functions
        into Spectre bsource expressions.
        """
        print("\n--- Stage 3: Generating Spectre Expressions ---")
        # This method is now primarily for verification or generating snippets.
        # The actual integration is done by exporting parameters.
        pass


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent
    params_path = base_path / "universal_model_lookup.csv"

    # Initialize the new ConductanceModel framework.
    # It now loads the parameter data itself.
    conductance_model = ConductanceModel(params_path)

    # --- Stage 0: Automatically tune the A_ET pre-factor ---
    # This will find the best A_ET and then re-run the final characterization.
    conductance_model.tune_A_ET_factor(target_p1=692.1, target_p2=-2227.0)

    # --- Stage 1.1: We no longer need to call characterization/fit here ---
    # The tuning function handles the final run. We just need to get the
    # data for the plot.
    phim_characterization_range = np.linspace(0.0, 0.7, 100)
    k_ET_physical = conductance_model.characterize_k_ET_physical(phim_characterization_range, verbose=False)
    k_ET_analytical_fit = conductance_model.k_ET_analytical(phim_characterization_range)

    # --- Plot k_ET Characterization Results ---
    fig_ket, ax_ket = plt.subplots(figsize=(8, 6))
    ax_ket.set_title("k_ET vs. Φ_mol: Physical Model vs. Analytical Fit")
    ax_ket.set_xlabel("Φ_mol (V)")
    ax_ket.set_ylabel("k_ET (s⁻¹)")
    ax_ket.grid(True, linestyle='--', alpha=0.6)
    
    # Plot the "ground truth" data from the physical model
    ax_ket.plot(phim_characterization_range, k_ET_physical, 'o', label='Physical Model (from integral)')
    
    # Plot the fitted analytical curve
    ax_ket.plot(phim_characterization_range, k_ET_analytical_fit, '-', lw=2, label='Analytical Fit (tanh model)')
    
    ax_ket.legend()
    ax_ket.set_yscale('log') # k_ET spans many orders of magnitude
    fig_ket.tight_layout()

    # --- Stage 1.2: Calculate and Plot f22(n) ---
    n_f22_range = np.linspace(1, 33040, 500)
    f22_values, fit_details = conductance_model.calculate_f22(n_f22_range)
    
    fig_f22, ax_f22 = plt.subplots(figsize=(8, 6))
    ax_f22.set_title("f₂₂(n): Fraction of Switched Molecules vs. Pulse Count")
    ax_f22.set_xlabel("Pulse Count (n)")
    ax_f22.set_ylabel("f₂₂(n)")
    ax_f22.grid(True, linestyle='--', alpha=0.6)
    
    ax_f22.plot(n_f22_range, f22_values, '-', lw=2, label='f₂₂(n) from Nucleation Model')
    
    ax_f22.legend()
    ax_f22.set_ylim(0, 1.05)
    fig_f22.tight_layout()

    # --- Plot n vs Z22 Fit ---
    p1_fit = fit_details.get('p1_fit')
    if p1_fit is not None:  # Check if fit was successful
        fig_fit, ax_fit = plt.subplots(figsize=(8, 6))
        ax_fit.set_title("n vs. Z₂₂: Dynamic Linear Fit")
        ax_fit.set_xlabel("Switched Region Width Z₂₂ (nm)")
        ax_fit.set_ylabel("Pulse Count (n)")
        ax_fit.grid(True, linestyle='--', alpha=0.6)

        # Plot the raw data points used for the fit
        valid_indices = fit_details['valid_indices']
        ax_fit.plot(
            fit_details['z22_values'][valid_indices],
            fit_details['n_fit_range'][valid_indices],
            'o',
            label='Calculated (Z₂₂, n) points'
        )

        # Plot the linear fit line
        p2_fit = fit_details['p2_fit']
        # Create a range for the fit line to span the plotted data
        fit_line_z22 = np.array([0, np.max(fit_details['z22_values'][valid_indices])])
        fit_line_n = p1_fit * fit_line_z22 + p2_fit
        ax_fit.plot(
            fit_line_z22,
            fit_line_n,
            '-',
            lw=2,
            label=f'Linear Fit (p1={p1_fit:.2f}, p2={p2_fit:.2f})'
        )

        ax_fit.legend()
        fig_fit.tight_layout()

    # --- Stage 2: Characterize the I(n, Vr) Surface ---
    # Use a small grid for faster testing
    n_current_range = np.array([5000, 10000, 15000, 20000, 25000])
    vr_current_range = np.linspace(-0.6, 0.6, 101)
    
    I_surface = conductance_model.characterize_current_physical(n_current_range, vr_current_range)
    
    # --- FIT THE CURRENT MODEL ---
    conductance_model.fit_current_analytical(n_current_range, vr_current_range, I_surface)

    # --- Plot I-V Curves from the Physical Model ---
    fig_iv, ax_iv = plt.subplots(figsize=(8, 6))
    ax_iv.set_title("I-V Curves from Physical Model for Different Pulse Counts")
    ax_iv.set_xlabel("Read Voltage (Vr)")
    ax_iv.set_ylabel("Current (A)")
    ax_iv.grid(True, linestyle='--', alpha=0.6)
    
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=n_current_range.min(), vmax=n_current_range.max())

    for i, n in enumerate(n_current_range):
        ax_iv.plot(vr_current_range, I_surface[i, :], color=cmap(norm(n)), label=f'n={n}')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_iv.colorbar(sm, ax=ax_iv)
    cbar.set_label("Pulse Counts (n)")
    
    ax_iv.legend()
    fig_iv.tight_layout()


    # --- Stage 2b: Characterize and Plot Current vs. Pulse Count ---
    print("\n--- Characterizing Current vs. Pulse Count at a fixed Vr ---")
    
    # Use a wider range of n for this plot, at a fixed read voltage
    n_I_vs_n_range = np.linspace(1, 33040, 400)
    vr_fixed = 0.5
    
    # Calculate the current for the n range at Vr=0.5V
    I_vs_n_surface = conductance_model.characterize_current_physical(n_I_vs_n_range, np.array([vr_fixed]))
    
    # Extract the single current column
    I_vs_n_values = I_vs_n_surface[:, 0]
    
    # --- Plot I vs n ---
    fig_ivn, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    fig_ivn.suptitle("Current vs. Pulse Count at Vr = 0.5V", fontsize=16)
    
    # Linear Plot
    ax_lin.set_title("Linear Scale")
    ax_lin.set_xlabel("Pulse Count (n)")
    ax_lin.set_ylabel("Current (A)")
    ax_lin.grid(True, linestyle='--', alpha=0.6)
    ax_lin.plot(n_I_vs_n_range, I_vs_n_values, '-', lw=2, color='r')
    
    # Semilog Plot
    ax_log.set_title("Logarithmic Scale")
    ax_log.set_xlabel("Pulse Count (n)")
    ax_log.set_ylabel("Current (A)")
    ax_log.grid(True, linestyle='--', alpha=0.6)
    ax_log.semilogy(n_I_vs_n_range, I_vs_n_values, '-', lw=2, color='r')
    
    fig_ivn.tight_layout(rect=[0, 0.03, 1, 0.95])


    # --- Initial Figure Setup ---
    # As a starting point, let's visualize the input to our model: the phim
    # curves for a range of pulses.
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Φ_mol(z,n) Curves - Input to Conductance Model")
    ax.set_xlabel("Φ_mol(z,n) (V)")
    ax.set_ylabel("z (Layer position)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Generate curves for n from 1000 to 16000 in steps of 1000
    n_pulse_range = np.arange(1000, 20001, 1000)
    
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=n_pulse_range.min(), vmax=n_pulse_range.max())

    print(f"\nGenerating initial Φ_mol curves for n = {n_pulse_range}...")
    for n in n_pulse_range:
        res = conductance_model.get_phim_curve(n)
        if res:
            z, phim = res
            ax.plot(phim, z, color=cmap(norm(n)), label=f'n={n}')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Pulse Counts (n)")
    
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 30.0)
    fig.tight_layout()

    plt.show()

    # --- Export Fitted Parameters ---
    print("\n--- Exporting Fitted Parameters ---")
    # Check if fit was successful before saving
    if fit_details.get('p1_fit') is not None:
        fitted_params = {
            'A_ET': [conductance_model.A_ET],
            'k_ET_A': [conductance_model.k_ET_params['A']],
            'k_ET_B': [conductance_model.k_ET_params['B']],
            'p1_fit': [fit_details['p1_fit']],
            'p2_fit': [fit_details['p2_fit']],
            'k_N_pot': [conductance_model.k_N_pot],
            'fp1_pot': [conductance_model.fp1_pot],
            'fp2_pot': [conductance_model.fp2_pot],
            'I_scale': [conductance_model.I_params['I_scale']],
            'V_scale': [conductance_model.I_params['V_scale']],
            # RESET Parameters
            'k_N_dep': [conductance_model.k_N_dep],
            'fp1_dep': [conductance_model.fp1_dep],
            'fp2_dep': [conductance_model.fp2_dep],
            # Rescaling Factors (calculated from f22 array)
            'f22_start': [f22_values[0]],
            'f22_peak': [f22_values[np.argmin(np.abs(n_f22_range - 16520))]],
            'f22_end': [f22_values[-1]]
        }
        
        df_params = pd.DataFrame(fitted_params)
        output_csv_path = base_path / "fitted_model_parameters.csv"
        df_params.to_csv(output_csv_path, index=False)
        print(f"Fitted parameters saved to {output_csv_path}")
    else:
        # --- Export Optimized Lookup Maps (Pre-calculated f22) ---
        print("\n--- Exporting Pre-Calculated f22 Maps for Optimization ---")
    
    # Define the full range for the lookup table
    # We go slightly beyond 33040 to be safe (0 to 34000)
    n_map_range = np.linspace(0, 34000, 3401)  # 10 points per step roughly (resolution 10)
    
    # We need to manually calculate the Pot and Dep curves using the *fitted* parameters
    # obtained in Stage 1.2.
    
    # Helper to calculate f22_raw for a given set of params (k_N, fp1, fp2)
    def calculate_raw_curve(n_arr, k_N, fp1, fp2, p1_fit, p2_fit):
        z_centers = np.arange(1, 31)
        # Calculate chi_N for each site
        # chi_N = (p1*z + p2) + (fp1*z + fp2)
        # Note: In the loop we did: (p1+fp1)*i + (p2+fp2)
        chi_N_slopes = (p1_fit + fp1) * z_centers + (p2_fit + fp2)
        
        # Calculate f22 for each n
        f22_arr = np.zeros_like(n_arr)
        
        # Vectorized over n and z
        # n: (M, 1), chi: (1, 30)
        N_mat = n_arr[:, np.newaxis]
        Chi_mat = chi_N_slopes[np.newaxis, :]
        
        # Sum of arctans
        sum_terms = (1.0 / np.pi) * np.arctan((N_mat - Chi_mat) / k_N) + 0.5
        f22_arr = np.mean(sum_terms, axis=1) # Average over 30 sites
        return f22_arr

    if fit_details.get('p1_fit') is not None:
        p1 = fit_details['p1_fit']
        p2 = fit_details['p2_fit']
        
        # 1. Potentiation Map
        f22_pot_raw = calculate_raw_curve(n_map_range, conductance_model.k_N_pot, conductance_model.fp1_pot, conductance_model.fp2_pot, p1, p2)
        
        # 2. Depotentiation Map
        # Note: Physics defines 'dep' as breaking connections.
        # The loop calculates 'state_dep'.
        # The core expects 'f22_dep_raw' (0 to 1).
        # And then the core usually inverts it (1-raw) if it represents "broken".
        # BUT wait: My Verilog-A loop outputs 'sum_dep / 30'. That is 'state_dep'.
        # In molmem_core (Pre-Fix/Post-Fix): "v=... (1.0 - V(f22_dep_raw))".
        # So the core DOES expect the raw state from the loop.
        # So we just export the raw calculation here.
        f22_dep_raw = calculate_raw_curve(n_map_range, conductance_model.k_N_dep, conductance_model.fp1_dep, conductance_model.fp2_dep, p1, p2)
        
        spectre_dir = base_path.parent / "SpectreModelDataEncryption" / "data"
        compiler_dir = base_path.parent / "DevicePhysicsCompiler"
        
        spectre_dir.mkdir(parents=True, exist_ok=True)
        compiler_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to save PWL to multiple target directories
        def save_pwl(filename, x, y, target_dirs):
            df = pd.DataFrame({'x': x, 'y': y})
            for d in target_dirs:
                df.to_csv(d / filename, sep=' ', header=False, index=False, float_format='%.6f')
                print(f"Saved {d / filename}")

        # 1. Save dynamic map tables to both Spectre and DevicePhysicsCompiler
        save_pwl("f22_pot_map.pwl", n_map_range, f22_pot_raw, [spectre_dir, compiler_dir])
        save_pwl("f22_dep_map.pwl", n_map_range, f22_dep_raw, [spectre_dir, compiler_dir])
        
        # 2. Save fitted scaling parameters as constant tables to both Spectre and DevicePhysicsCompiler
        x_const = [0.0, 100000.0]
        I_scale_val = conductance_model.I_params['I_scale']
        V_scale_val = conductance_model.I_params['V_scale']
        save_pwl("I_scale.pwl", x_const, [I_scale_val, I_scale_val], [spectre_dir, compiler_dir])
        save_pwl("V_scale.pwl", x_const, [V_scale_val, V_scale_val], [spectre_dir, compiler_dir])
        
        # 3. Save f22 boundary scaling factors to Spectre models
        f22_start_val = f22_values[0]
        f22_peak_val = f22_values[np.argmin(np.abs(n_f22_range - 16520))]
        f22_end_val = f22_values[-1]
        save_pwl("f22_start.pwl", x_const, [f22_start_val, f22_start_val], [spectre_dir])
        save_pwl("f22_peak.pwl", x_const, [f22_peak_val, f22_peak_val], [spectre_dir])
        save_pwl("f22_end.pwl", x_const, [f22_end_val, f22_end_val], [spectre_dir])
        
    else:
        print("Skipping map export due to failed fit.")
