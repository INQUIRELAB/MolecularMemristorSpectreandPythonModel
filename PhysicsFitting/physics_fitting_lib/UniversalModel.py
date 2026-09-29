import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
import itertools
import argparse
from scipy.interpolate import PchipInterpolator
try:
    from numba import njit
except ImportError:
    print("Numba not found. Running in pure Python mode. For a significant speedup, please install Numba (`pip install numba`).")
    # Create a dummy decorator if numba is not available
    def njit(func):
        return func

# To avoid code duplication, we import the core model equation directly from the fitting script.
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # We now need both the blended model and the underlying moving average model
    from MolMemristor_Figure7 import x_of_z_piecewise_blended, logistic, build_interpolated_behavior, x_of_z_moving_average
except ImportError:
    print("Error: Could not import from MolMemristor_Figure7.py.")
    print("Please ensure both UniversalModel.py and MolMemristor_Figure7.py are in the same directory.")
    exit()

# --- Numba-related code has been removed for correctness ---


class UniversalModel:
    """
    A predictive model that generates a Φ_mol(z) curve for any given pulse count 'n'.

    It operates on a lookup-based principle, using smooth interpolation between the
    individually fitted parameters from the Figure 7 script to generate new curves.
    """
    def __init__(self, params_path: Path, fit_method: str = 'interpolation'):
        self.param_names = ['p_i', 'q_i', 'k_i', 'a_i', 'w_i', 'p_f', 'q_f', 'k_f', 'a_f', 'w_f', 'z_mid', 'z_width']
        if fit_method not in ['interpolation', 'polynomial', 'linear']:
            raise ValueError("fit_method must be 'interpolation', 'polynomial', or 'linear'")
        self.fit_method = fit_method
        self._build_model_functions(params_path)
        print(f"Universal Model (method: {self.fit_method}) initialized and ready.")

    def _build_model_functions(self, params_path: Path):
        """
        Reads the fitted parameters and builds a set of 12 model functions
        based on the chosen fitting method.
        """
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found at {params_path}. Please run MolMemristor_Figure7.py first.")

        df = pd.read_csv(params_path).sort_values(by="Pulse Counts").dropna()
        self.pulse_counts = df["Pulse Counts"].values

        self.model_functions = {}
        for param in self.param_names:
            param_values = df[param].values
            if self.fit_method == 'interpolation':
                # Use smooth, monotonic interpolation
                self.model_functions[param] = PchipInterpolator(self.pulse_counts, param_values, extrapolate=True)
            elif self.fit_method == 'polynomial':
                # Use a 6th degree polynomial line of best fit
                poly_coeffs = np.polyfit(self.pulse_counts, param_values, 20)
                model_func = np.poly1d(poly_coeffs)
                self.model_functions[param] = model_func
            elif self.fit_method == 'linear':
                # Use a 1st degree polynomial (linear) line of best fit
                poly_coeffs = np.polyfit(self.pulse_counts, param_values, 1)
                self.model_functions[param] = np.poly1d(poly_coeffs)

        print(f"Parameter functions ({self.fit_method}) built successfully.")


    def get_curve(self, n: float, num_points: int = 400) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Generates a curve by evaluating the 12 model functions at 'n'.
        """
        # Look up the 12 parameters for the given pulse count 'n'
        try:
            params_list = [self.model_functions[p](n) for p in self.param_names]
        except KeyError:
            print("Error: Model functions have not been built correctly.")
            return None

        params_initial = np.array(params_list[0:5])
        params_final = np.array(params_list[5:10])
        z_mid, z_width = params_list[10], params_list[11]
        
        # Generate the curve using the core model equation
        z_grid = np.linspace(0.0, 30.0, num_points)
        
        # We need to call the imported logistic function, which works on scalars
        z_norm = (z_grid - z_mid) / max(1e-9, z_width)
        w_final = np.array([logistic(val) for val in z_norm])
        w_initial = 1.0 - w_final

        # The basis curves
        p_i, q_i, k_i, a_i, w_i = params_initial
        p_f, q_f, k_f, a_f, w_f = params_final
        x_initial_base = x_of_z_moving_average(z_grid, p_i, q_i, k_i, a_i, w_i)
        x_final_base = x_of_z_moving_average(z_grid, p_f, q_f, k_f, a_f, w_f)

        # The final blended model - note that there are no alpha scalers here
        x_curve = w_initial * x_initial_base + w_final * x_final_base
        
        return z_grid, x_curve

    def export_lookup_table(self, output_path: Path, start_pulse: int = 0, end_pulse: int = 20000, step: int = 100):
        """
        Generates a dense lookup table of interpolated parameter values by evaluating
        the model's functions at each step and saves it to a CSV file.
        """
        print(f"\nGenerating universal model lookup table...")
        pulse_counts_table = np.arange(start_pulse, end_pulse + 1, step)
        
        table_data = []
        for n in pulse_counts_table:
            row = {'Pulse Counts': n}
            # Evaluate each parameter's model function at the pulse count 'n'
            params = {p_name: func(n) for p_name, func in self.model_functions.items()}
            row.update(params)
            table_data.append(row)
            
        df_table = pd.DataFrame(table_data)
        
        # Ensure consistent column order
        column_order = ['Pulse Counts'] + self.param_names
        df_table = df_table[column_order]
        
        df_table.to_csv(output_path, index=False, float_format='%.8g')
        print(f"Universal model lookup table exported to {output_path}")


def plot_parameter_evolution(params_path: Path) -> None:
    """
    Reads the fitted parameters and plots their smooth evolution vs. pulse count,
    including a polynomial line of best fit for comparison.
    """
    if not params_path.exists():
        print(f"Parameter file not found at {params_path}, skipping evolution plot.")
        return

    df = pd.read_csv(params_path).sort_values(by="Pulse Counts").dropna()
    param_names = [col for col in df.columns if col != "Pulse Counts"]
    if not param_names: return
    pulse_counts = df["Pulse Counts"].values

    # We have 12 parameters, so we create a 4x3 grid of subplots.
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    axes = axes.flatten() # Convert the 2D grid of axes to a 1D array for easy iteration

    fig.suptitle("Evolution of Fitted Parameters vs. Pulse Count", fontsize=16)

    for i, param in enumerate(param_names):
        ax = axes[i]
        param_values = df[param].values

        # 1. Plot the original fitted data points
        ax.plot(pulse_counts, param_values, 'o', color='skyblue', label='Fitted Points')

        # 2. Plot the smooth PCHIP interpolation
        interp_func = PchipInterpolator(pulse_counts, param_values)
        fine_pulse_counts = np.linspace(pulse_counts.min(), pulse_counts.max(), 300)
        ax.plot(fine_pulse_counts, interp_func(fine_pulse_counts), '-', color='blue', label='PCHIP Interpolation')

        # 3. Plot the 6th degree polynomial line of best fit
        poly_coeffs_6th = np.polyfit(pulse_counts, param_values, 6)
        poly_fit_6th = np.poly1d(poly_coeffs_6th)
        ax.plot(fine_pulse_counts, poly_fit_6th(fine_pulse_counts), '--', color='red', label='6th Deg Poly Fit')

        # 4. Plot the 1st degree linear trend line
        poly_coeffs_1st = np.polyfit(pulse_counts, param_values, 1)
        poly_fit_1st = np.poly1d(poly_coeffs_1st)
        ax.plot(fine_pulse_counts, poly_fit_1st(fine_pulse_counts), '-.', color='green', label='Linear Trend')

        ax.set_title(param)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # Common X label
    for ax in axes[-3:]:
        ax.set_xlabel("Pulse Counts")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # NOTE: We do not call plt.show() here anymore.


def main() -> None:
    """Main function to demonstrate and verify the UniversalModel."""
    
    # --- Initialization ---
    base_path = Path(__file__).parent.parent
    params_path = base_path / "fitted_parameters_regularized.csv"
    ground_truth_path = base_path / "Phi_from_probability.csv"

    # Plot the raw parameter evolution first
    plot_parameter_evolution(params_path)

    # --- Model 1: Interpolation-based ---
    model_interp = UniversalModel(params_path, fit_method='interpolation')

    # --- EXPORT THE LOOKUP TABLE ---
    lookup_output_path = base_path / "universal_model_lookup.csv"
    model_interp.export_lookup_table(lookup_output_path)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_title("Universal Model Verification (Interpolation)")
    
    cmap = plt.get_cmap("viridis_r")
    df_truth_raw = pd.read_csv(ground_truth_path)
    # Ensure pulse_counts_truth is available for norm
    pulse_counts_truth = df_truth_raw["Pulse Counts"].values
    if len(pulse_counts_truth) == 0:
        print("No pulse counts found in ground truth data.")
        return

    norm = Normalize(vmin=np.min(pulse_counts_truth), vmax=np.max(pulse_counts_truth))
    
    print("\nGenerating model curves for verification plot (Interpolation)...")
    pulse_counts_smooth = np.linspace(df_truth_raw["Pulse Counts"].min(), df_truth_raw["Pulse Counts"].max(), 50)

    for n in pulse_counts_smooth:
        res = model_interp.get_curve(n)
        if res:
            z, x = res
            ax1.plot(x, z, color=cmap(norm(n)), linewidth=2.0)
    
    # Overlay the original "ground truth" data on the first plot
    print("Overlaying ground truth data...")
    z_truth_cols = [col for col in df_truth_raw.columns if col != "Pulse Counts"]
    z_layers = np.array([float(c) for c in z_truth_cols])
    for n in pulse_counts_truth:
        row = df_truth_raw[df_truth_raw["Pulse Counts"] == n]
        if row.empty: continue
        x_row = row[z_truth_cols].values.flatten()
        built = build_interpolated_behavior(x_row, z_layers)
        if built:
            (z_tail, x_tail), (z_csv, x_csv) = built
            ax1.plot(x_csv, z_csv, 'k--', linewidth=0.8, alpha=0.7)
            ax1.plot(x_tail, x_tail, 'k:', linewidth=0.8, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Pulse counts")
    ax1.set_xlabel("Φ_mol(z,n) (V)")
    ax1.set_ylabel("z (Layer position)")
    ax1.set_xlim(0.0, 0.7)
    ax1.set_ylim(0.0, 30.0)
    fig1.tight_layout()

    # --- Model 2: Polynomial-based ---
    model_poly = UniversalModel(params_path, fit_method='polynomial')
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title("Universal Model Verification (Polynomial Fit)")

    print("\nGenerating model curves for verification plot (Polynomial Fit)...")
    for n in pulse_counts_smooth:
        res = model_poly.get_curve(n)
        if res:
            z, x = res
            ax2.plot(x, z, color=cmap(norm(n)), linewidth=2.0)
            
    # Also overlay the ground truth on the second plot for comparison
    for n in pulse_counts_truth:
        row = df_truth_raw[df_truth_raw["Pulse Counts"] == n]
        if row.empty: continue
        x_row = row[z_truth_cols].values.flatten()
        built = build_interpolated_behavior(x_row, z_layers)
        if built:
            (z_tail, x_tail), (z_csv, x_csv) = built
            ax2.plot(x_csv, z_csv, 'k--', linewidth=0.8, alpha=0.7)
            ax2.plot(x_tail, x_tail, 'k:', linewidth=0.8, alpha=0.7)

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2)
    cbar2.set_label("Pulse counts")
    ax2.set_xlabel("Φ_mol(z,n) (V)")
    ax2.set_ylabel("z (Layer position)")
    ax2.set_xlim(0.0, 0.7)
    ax2.set_ylim(0.0, 30.0)
    fig2.tight_layout()
    
    # --- Model 3: Linear Trend-based ---
    model_linear = UniversalModel(params_path, fit_method='linear')
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.set_title("Universal Model Verification (Linear Trend)")

    print("\nGenerating model curves for verification plot (Linear Trend)...")
    for n in pulse_counts_smooth:
        res = model_linear.get_curve(n)
        if res:
            z, x = res
            ax3.plot(x, z, color=cmap(norm(n)), linewidth=2.0)
            
    # Also overlay the ground truth on the third plot for comparison
    for n in pulse_counts_truth:
        row = df_truth_raw[df_truth_raw["Pulse Counts"] == n]
        if row.empty: continue
        x_row = row[z_truth_cols].values.flatten()
        built = build_interpolated_behavior(x_row, z_layers)
        if built:
            (z_tail, x_tail), (z_csv, x_csv) = built
            ax3.plot(x_csv, z_csv, 'k--', linewidth=0.8, alpha=0.7)
            ax3.plot(x_tail, x_tail, 'k:', linewidth=0.8, alpha=0.7)

    sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax3)
    cbar3.set_label("Pulse counts")
    ax3.set_xlabel("Φ_mol(z,n) (V)")
    ax3.set_ylabel("z (Layer position)")
    ax3.set_xlim(0.0, 0.7)
    ax3.set_ylim(0.0, 30.0)
    fig3.tight_layout()

    # This single call will now render all four plot windows
    plt.show()


if __name__ == "__main__":
    main()
