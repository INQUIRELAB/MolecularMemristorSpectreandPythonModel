import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import PchipInterpolator


# --- Physical constants (in eV-based units) ---
KBT_EV = 0.0257  # k_B T at 298 K (eV)
E_LAMBDA = 0.04  # Reorganization energy (eV), from paper
DELTA_G0 = 0.1   # Gibbs free energy change (eV), from paper
E_CHARGE_EV_PER_V = 1.0  # 1 V -> 1 eV for an electron charge factor


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def probability_from_phi(phi_v: float, e_grid: np.ndarray) -> float:
    """
    Compute P = ∫ Normal(E; mu, sigma) * logistic(E/KBT_EV) dE from Marcus Theory.
    This implements the full physical equation for k_ET.
    """
    # Marcus theory parameters for the Gaussian distribution
    energy_shift = E_LAMBDA + DELTA_G0
    mu = E_CHARGE_EV_PER_V * phi_v - energy_shift
    gauss_denom = 4.0 * E_LAMBDA * KBT_EV
    gauss_norm = 1.0 / np.sqrt(np.pi * gauss_denom)

    gauss = gauss_norm * np.exp(-((e_grid - mu) ** 2) / gauss_denom)
    occ = logistic(e_grid / KBT_EV)
    integrand = gauss * occ
    return float(np.trapz(integrand, e_grid))


def invert_phi_for_probability(p_target: float, e_grid: np.ndarray, tol: float = 1e-6) -> float:
    """
    Monotone bisection on phi in [0, 0.7] to solve probability_from_phi(phi) = p_target.
    """
    lo, hi = 0.0, 0.7
    f_lo = probability_from_phi(lo, e_grid)
    f_hi = probability_from_phi(hi, e_grid)

    # Clamp targets outside the achievable range
    p_clamped = min(max(p_target, f_lo), f_hi)

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        f_mid = probability_from_phi(mid, e_grid)
        if abs(f_mid - p_clamped) < tol:
            return mid
        if f_mid < p_clamped:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    # Energy grid for integration (eV)
    e_grid = np.linspace(-0.6, 0.6, 1201)

    excel_path = Path(__file__).parent.parent / "Probability Matrix.xlsx"
    df = pd.read_excel(excel_path, header=1).dropna(how="all")

    # Columns: [Pulse Counts, 1, 2, ..., 30] → treat as z indices 0..29 (no scaling)
    pulse_counts = df.iloc[:, 0].to_numpy(dtype=float)
    # Align labels: reverse so row i in phi_matrix corresponds to pulse_counts_aligned[i]
    pulse_counts_aligned = pulse_counts[::-1]
    z_labels = [int(c) - 1 for c in df.columns[1:]]  # original integer indices 0..29
    prob_values = df.iloc[:, 1:].to_numpy(dtype=float)

    # Invert each probability to phi via simple bisection per entry
    phi_matrix = np.empty_like(prob_values, dtype=float)
    for i in range(prob_values.shape[0]):
        for j in range(prob_values.shape[1]):
            phi_matrix[i, j] = invert_phi_for_probability(prob_values[i, j], e_grid)

    # --- Stretch Correction based on 1000-pulse guideline ---
    # Rescale all phi values so that the initial slope of the 1000-pulse trace,
    # when extended as a straight line, passes through the top-right corner (0.7, 30).
    try:
        # Find the 1000-pulse trace to calculate the reference slope
        idx_1000 = int(np.argmin(np.abs(pulse_counts_aligned - 1000.0)))
        x_line_ref = phi_matrix[idx_1000, :]
        z_line_ref = np.asarray(z_labels, dtype=float)

        # Calculate the current slope 'm' from the initial segment (phi <= 0.1)
        mask = np.isfinite(x_line_ref) & (x_line_ref >= 0.0) & (x_line_ref <= 0.1)
        if np.count_nonzero(mask) >= 2:
            x_seg = x_line_ref[mask]
            z_seg = z_line_ref[mask]
            denom = float(np.dot(x_seg, x_seg))
            if denom > 0.0:
                m_current = float(np.dot(x_seg, z_seg) / denom)

                # The ideal slope m_ideal would make the line z = m*x pass through (0.7, 30).
                m_ideal = 30.0 / 0.7

                # The scaling factor 's' makes the new slope m_new = m_ideal.
                # m_new = z / (s*x) = m_current / s. So, s = m_current / m_ideal.
                if m_ideal > 0:
                    scale_factor = m_current / m_ideal
                    phi_matrix *= scale_factor
    except (ValueError, IndexError):
        pass

    # Build a dense, uniformly sampled version of each pulse line.
    # The cutoff is now determined by finding the "knee" of each curve.
    z_full = np.asarray(z_labels, dtype=float)
    z_max = float(z_full[-1]) if z_full.size else 29.0
    z_dense = np.linspace(0.0, z_max, 400)

    # --- Knee-Detection Logic for Global Cutoff ---
    knee_locations_phi = []
    for i in range(phi_matrix.shape[0]):
        row = phi_matrix[i, :]
        finite_mask = np.isfinite(row)
        if not np.any(finite_mask):
            continue

        z_raw = z_full[finite_mask].astype(float)
        x_raw = row[finite_mask].astype(float)

        if x_raw.size < 3:
            continue

        # Calculate slopes (dz/dx) between adjacent points
        dx = np.diff(x_raw)
        dz = np.diff(z_raw)
        # Avoid division by zero for vertical segments
        dx[np.abs(dx) < 1e-9] = 1e-9
        slopes = dz / dx

        if slopes.size < 2:
            continue
        
        # The "knee" is where the change in slope is largest
        slope_changes = np.abs(np.diff(slopes))
        # The knee is the point connecting the two segments with the largest slope change.
        knee_index = np.argmax(slope_changes) + 1
        knee_locations_phi.append(x_raw[knee_index])

    if knee_locations_phi:
        median_knee_phi = float(np.median(knee_locations_phi))
        # Place the cutoff slightly to the left of the median knee
        global_cut_save = median_knee_phi - 0.025
    else:
        # Fallback to the old statistical method if knee detection fails
        per_line_max = []
        for i in range(phi_matrix.shape[0]):
            row = phi_matrix[i, :]
            finite = np.isfinite(row)
            if np.any(finite):
                per_line_max.append(float(np.max(row[finite])))
        global_base = float(np.median(per_line_max)) - 0.01 if per_line_max else 0.7
        global_cut_save = 0.94 * float(max(0.0, min(0.7, global_base)))


    phi_dense = np.full((phi_matrix.shape[0], z_dense.size), np.nan, dtype=float)
    for i in range(phi_matrix.shape[0]):
        row = phi_matrix[i, :]
        finite_mask = np.isfinite(row)
        if not np.any(finite_mask):
            continue

        # Use the full raw data trace that is actually plotted in the convergence figure.
        z_raw = z_full[finite_mask].astype(float)
        x_raw = row[finite_mask].astype(float)

        if x_raw.size < 2:
            continue

        # --- Precise Truncation & Extrapolation on the RAW Plotted Line ---
        z_final = z_raw
        x_final = x_raw

        # Case 1: The raw line crosses the cutoff. Truncate at the intersection.
        if np.max(x_raw) > global_cut_save:
            cross_indices = np.where(x_raw > global_cut_save)[0]
            if cross_indices.size > 0:
                first_over_idx = cross_indices[0]
                if first_over_idx > 0:
                    z_before, x_before = z_raw[first_over_idx - 1], x_raw[first_over_idx - 1]
                    z_after, x_after = z_raw[first_over_idx], x_raw[first_over_idx]
                    z_intersect = np.interp(global_cut_save, [x_before, x_after], [z_before, z_after])
                    z_final = np.append(z_raw[:first_over_idx], z_intersect)
                    x_final = np.append(x_raw[:first_over_idx], global_cut_save)

        # Case 2: The entire raw line is below the cutoff. Extrapolate to the intersection.
        else:
            z_last_two = z_raw[-2:]
            x_last_two = x_raw[-2:]
            # Avoid extrapolation on a perfectly flat line to prevent infinite z
            if abs(x_last_two[1] - x_last_two[0]) > 1e-9:
                z_at_cutoff = np.interp(global_cut_save, x_last_two, z_last_two)
                z_final = np.append(z_raw, z_at_cutoff)
                x_final = np.append(x_raw, global_cut_save)

        # Use simple linear interpolation on the precisely truncated/extended data.
        phi_dense[i, :] = np.interp(z_dense, z_final, x_final, right=np.nan)

    # Save as CSV with dense z columns; values beyond each line's cutoff remain NaN
    out = pd.DataFrame(phi_dense, columns=[f"{z:.6f}" for z in z_dense])
    out.insert(0, "Pulse Counts", pulse_counts_aligned)
    out_path = Path(__file__).parent.parent / "Phi_from_probability.csv"
    out.to_csv(out_path, index=False)

    # Plot phi_mol(z, n) curves colored by pulse count
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=float(np.min(pulse_counts_aligned)), vmax=float(np.max(pulse_counts_aligned)))
    fig, ax = plt.subplots()
    # Plot with original z indices 0..29
    z_plot = np.asarray(z_labels, dtype=float)
    # Plot the full convergency traces (untrimmed), and show a SINGLE global vertical cutoff marker
    for i in range(phi_matrix.shape[0]):
        n = pulse_counts_aligned[i]
        color = cmap(norm(float(n)))
        ax.plot(phi_matrix[i, :], z_plot, color=color, linewidth=2.0)
    # Single global cutoff marker (6% back from the global base)
    ax.axvline(global_cut_save, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
    # Add dotted guideline: slope from first x<=0.1 segment of the ~1000 pulse line
    try:
        # pick index with pulse count closest to 1000
        idx_1000 = int(np.argmin(np.abs(pulse_counts_aligned - 1000.0)))
        x_line = phi_matrix[idx_1000, :]
        z_line = z_plot
        mask = np.isfinite(x_line) & (x_line >= 0.0) & (x_line <= 0.1)
        if np.count_nonzero(mask) >= 2:
            # fit z ≈ m*x over the initial segment (ignore intercept to anchor at origin)
            x_seg = x_line[mask]
            z_seg = z_line[mask]
            # least-squares slope anchored at origin: m = (x·z)/(x·x)
            denom = float(np.dot(x_seg, x_seg))
            if denom > 0.0:
                m = float(np.dot(x_seg, z_seg) / denom)
                x_guid = np.linspace(0.0, 0.7, 200)
                z_guid = m * x_guid
                ax.plot(x_guid, z_guid, linestyle=":", linewidth=1.5, color=cmap(norm(float(pulse_counts_aligned[idx_1000]))), alpha=0.9)
    except Exception:
        pass
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Pulse counts")
    ax.set_xlabel("Phi_mol(z,n) (V)")
    ax.set_ylabel("z (Layer position)")
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 30.0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


