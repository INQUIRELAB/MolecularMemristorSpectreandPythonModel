import os
# The MKL_NUM_THREADS and OMP_NUM_THREADS environment variables are set to 1
# to prevent conflicts between joblib's parallelism and numpy's internal parallelism,
# which can otherwise lead to significant slowdowns.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, curve_fit, least_squares, differential_evolution
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import time
import argparse
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

# --- Core Φ_mol Mathematical Functions ---
# Copied from MolMemristor_Figure7.py to make this script self-contained.

# ---- Numba JIT SSE (identical math, loop implementation) ----
if _HAVE_NUMBA:
    @njit(fastmath=False)
    def _sse_one_numba(p: float, q: float, k: float, a: float, w: float, t: np.ndarray, x: np.ndarray) -> float:
        sse = 0.0
        w_c = w
        for i in range(t.size):
            ti = t[i]
            s = (1.0 - (1.0 - ti) ** q) ** p
            u = (1.0 + k) * s / (1.0 + k * (s * s))
            # neighbors (clamped)
            t_m = ti - w_c
            if t_m < 0.0:
                t_m = 0.0
            if t_m > 1.0:
                t_m = 1.0
            t_p = ti + w_c
            if t_p < 0.0:
                t_p = 0.0
            if t_p > 1.0:
                t_p = 1.0
            s_m = (1.0 - (1.0 - t_m) ** q) ** p
            s_p = (1.0 - (1.0 - t_p) ** q) ** p
            u_m = (1.0 + k) * s_m / (1.0 + k * (s_m * s_m))
            u_p = (1.0 + k) * s_p / (1.0 + k * (s_p * s_p))
            u_ma = 0.25 * u_m + 0.5 * u + 0.25 * u_p
            # tapered alpha
            if w_c <= 0.0:
                a_eff = 0.0
            else:
                xL = ti / w_c
                if xL < 0.0:
                    xL = 0.0
                if xL > 1.0:
                    xL = 1.0
                xR = (1.0 - ti) / w_c
                if xR < 0.0:
                    xR = 0.0
                if xR > 1.0:
                    xR = 1.0
                aL = xL * xL * (3.0 - 2.0 * xL)
                aR = xR * xR * (3.0 - 2.0 * xR)
                a_eff = a * (aL * aR)
            x_model = 0.7 * ((1.0 - a_eff) * u + a_eff * u_ma)
            d = x_model - x[i]
            sse += d * d
        return sse

    @njit(parallel=False, fastmath=False)
    def _argmin_full_numba(P: np.ndarray, Q: np.ndarray, K: np.ndarray, A: np.ndarray, W: np.ndarray, t: np.ndarray, x: np.ndarray) -> tuple:
        best_val = 1.0e300
        bp = bq = bk = ba = bw = -1
        for iq in range(Q.size):
            qv = Q[iq]
            for ip in range(P.size):
                pv = P[ip]
                for ik in range(K.size):
                    kv = K[ik]
                    for ia in range(A.size):
                        av = A[ia]
                        for iw in range(W.size):
                            wv = W[iw]
                            val = _sse_one_numba(pv, qv, kv, av, wv, t, x)
                            if val < best_val:
                                best_val = val
                                bp, bq, bk, ba, bw = ip, iq, ik, ia, iw
        return bp, bq, bk, ba, bw, best_val

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)

def x_of_z_moving_average(z_values: np.ndarray, p: float, q: float, k: float, alpha: float, w: float) -> np.ndarray:
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
    return 1.0 / (1.0 + np.exp(-x))

def x_of_z_piecewise_blended(z_values: np.ndarray, params_initial: np.ndarray, params_final: np.ndarray,
                               blend_mid: float, blend_width: float) -> np.ndarray:
    """
    Combines two specialized versions of the moving-average model.
    - Model_Initial is active for low z, controlled by params_initial.
    - Model_Final is active for high z, controlled by params_final.
    A smooth logistic blend ensures SPICE compatibility.
    """
    # Unpack parameters for clarity
    p_i, q_i, k_i, a_i, w_i = params_initial
    p_f, q_f, k_f, a_f, w_f = params_final

    # Calculate the output of both models across all z values
    x_initial = x_of_z_moving_average(z_values, p_i, q_i, k_i, a_i, w_i)
    x_final = x_of_z_moving_average(z_values, p_f, q_f, k_f, a_f, w_f)

    # Blending weights based on z, not phi, for a well-defined spatial transition
    z_norm = (z_values - blend_mid) / max(1e-9, blend_width)
    w_final = logistic(z_norm)  # Starts at 0, goes to 1
    w_initial = 1.0 - w_final       # Starts at 1, goes to 0

    return w_initial * x_initial + w_final * x_final

def _residuals(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Helper function to calculate model residuals for optimization."""
    pi, qi, ki, ai, wi, pf, qf, kf, af, wf, z_mid, z_width = params
    param_i = np.array([pi, qi, ki, ai, wi])
    param_f = np.array([pf, qf, kf, af, wf])
    x_model = x_of_z_piecewise_blended(z_vals, param_i, param_f, z_mid, z_width)
    return x_model - x_vals

def _sse_cost(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """Sum of squared errors cost function for the global optimizer."""
    return np.sum(_residuals(params, z_vals, x_vals)**2)

def _residuals_5_param(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Helper for 5-parameter moving average model residuals."""
    p, q, k, a, w = params
    x_model = x_of_z_moving_average(z_vals, p, q, k, a, w)
    return x_model - x_vals

def _sse_cost_5_param(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """Sum of squared errors for the 5-parameter model."""
    return np.sum(_residuals_5_param(params, z_vals, x_vals)**2)

def solve_moving_average_params_hybrid(x: np.ndarray, z: np.ndarray,
                                       p0: float | None, q0: float | None, k0: float | None,
                                       bounds: tuple | None = None) -> tuple[float, float, float, float, float]:
    """
    Hybrid optimization from Figure7:
    1. Coarse, wide grid search to find a good starting region (robust).
    2. Fine-tuning with a fast local optimizer (scipy.optimize.least_squares).
    """
    if x.size < 3:
        return 0.8, 2.0, 0.5, 0.3, 0.05

    # --- Stage 1: Coarse grid search to find a good starting point ---
    if p0 is None:
        p_lo, p_hi = 0.2, 1.5
    else:
        p_lo, p_hi = max(0.05, p0 - 0.4), min(2.0, p0 + 0.4)
    if q0 is None:
        q_lo, q_hi = 0.8, 12.0
    else:
        q_lo, q_hi = max(0.2, q0 - 2.0), min(16.0, q0 + 2.0)
    if k0 is None:
        k_lo, k_hi = 0.0, 5.0
    else:
        k_lo, k_hi = max(0.0, k0 - 1.5), min(8.0, k0 + 1.5)
    alpha_lo, alpha_hi = 0.0, 1.0
    w_lo, w_hi = 0.0, 0.2

    best = (p0 or 0.8, q0 or 2.0, k0 or 0.5, 0.3, 0.05, np.inf)

    P = np.linspace(p_lo, p_hi, 13)
    Q = np.linspace(q_lo, q_hi, 13)
    K = np.linspace(k_lo, k_hi, 9)
    A = np.linspace(alpha_lo, alpha_hi, 9)
    W = np.linspace(w_lo, w_hi, 7)
    t = z / 30.0

    if _HAVE_NUMBA:
        bp, bq, bk, ba, bw, e_best = _argmin_full_numba(P, Q, K, A, W, t, x)
        if e_best < best[5]:
            best = (float(P[bp]), float(Q[bq]), float(K[bk]), float(A[ba]), float(W[bw]), float(e_best))
    else:
        # Fallback vectorized search
        TILE_A, TILE_W = 9, 7
        tB = t[None, None, None, None, None, :]
        one_minus_t = 1.0 - tB
        QB = Q[:, None, None, None, None, None]
        PB = P[None, :, None, None, None, None]
        KB = K[None, None, :, None, None, None]
        s = (1.0 - np.power(one_minus_t, QB)) ** PB
        u = (1.0 + KB) * s / (1.0 + KB * (s ** 2))
        for a_start in range(0, len(A), TILE_A):
            a_end = min(a_start + TILE_A, len(A))
            AB = A[a_start:a_end][None, None, None, :, None, None]
            for w_start in range(0, len(W), TILE_W):
                w_end = min(w_start + TILE_W, len(W))
                WB = W[w_start:w_end][None, None, None, None, :, None]
                t_minus, t_plus = np.clip(tB - WB, 0.0, 1.0), np.clip(tB + WB, 0.0, 1.0)
                s_m, s_p = (1.0 - np.power(1.0 - t_minus, QB)) ** PB, (1.0 - np.power(1.0 - t_plus, QB)) ** PB
                u_m, u_p = (1.0 + KB) * s_m / (1.0 + KB * (s_m ** 2)), (1.0 + KB) * s_p / (1.0 + KB * (s_p ** 2))
                u_ma = 0.25 * u_m + 0.5 * u + 0.25 * u_p
                epsW = 1e-6
                WL = np.maximum(WB, epsW)
                xL, xR = np.clip(tB / WL, 0.0, 1.0), np.clip((1.0 - tB) / WL, 0.0, 1.0)
                aL, aR = xL * xL * (3.0 - 2.0 * xL), xR * xR * (3.0 - 2.0 * xR)
                A_eff = AB * (aL * aR)
                x_model = 0.7 * ((1.0 - A_eff) * u + A_eff * u_ma)
                diff = x_model - x[None, None, None, None, None, :]
                sse_tile = np.sum(diff * diff, axis=-1)
                flat_idx = int(np.argmin(sse_tile))
                best_idx_tile = np.unravel_index(flat_idx, sse_tile.shape)
                q_i, p_i, k_i = int(best_idx_tile[0]), int(best_idx_tile[1]), int(best_idx_tile[2])
                a_i, w_i = a_start + int(best_idx_tile[3]), w_start + int(best_idx_tile[4])
                e_best = float(sse_tile[best_idx_tile])
                if e_best < best[5]:
                    best = (float(P[p_i]), float(Q[q_i]), float(K[k_i]), float(A[a_i]), float(W[w_i]), e_best)

    p_coarse, q_coarse, k_coarse, a_coarse, w_coarse, _ = best

    # --- Stage 2: Local optimization using the best point from the coarse search ---
    def residuals_5_param(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        p, q, k, a, w = params
        x_model = x_of_z_moving_average(z_vals, p, q, k, a, w)
        return x_model - x_vals

    x0 = [p_coarse, q_coarse, k_coarse, a_coarse, w_coarse]
    if bounds is None:
        bounds = ([0.05, 0.2, 0.0, 0.0, 0.0],
                  [np.inf, np.inf, np.inf, 1.0, 0.5])

    res = least_squares(residuals_5_param, x0, bounds=bounds, args=(z, x), method='trf', ftol=1e-6, xtol=1e-6)
    p_final, q_final, k_final, a_final, w_final = res.x

    return float(p_final), float(q_final), float(k_final), float(a_final), float(w_final)

def solve_full_piecewise_model(x_full_trace: np.ndarray, z_full_trace: np.ndarray, x_break: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fits the full piecewise blended model using the multi-stage logic from Figure7.
    1. Define a breakpoint and split the data.
    2. Fit an "initial" model to the first part of the curve.
    3. Fit a "final" model to the full trace, seeded by the initial fit.
    4. Fit the blend parameters to smoothly combine the two models.
    """
    # --- Stage 1 & 2: Define breakpoint and fit initial/final models ---
    # The breakpoint is now an x_value, passed in from the calling function.
    initial_indices = x_full_trace <= x_break
    x_initial_fit_data = x_full_trace[initial_indices]
    z_initial_fit_data = z_full_trace[initial_indices]

    # Anchor the initial fit with the final point for stability, as in Figure7.
    if x_initial_fit_data.size > 0:
        x_initial_fit_data = np.append(x_initial_fit_data, x_full_trace[-1])
        z_initial_fit_data = np.append(z_initial_fit_data, z_full_trace[-1])
    
    if x_initial_fit_data.size < 3: # Not enough data to fit
        # Fallback to a single-stage fit on the full curve
        print(f"Warning: Not enough data for multi-stage fit. Falling back to single-stage.")
        p_i, q_i, k_i, a_i, w_i = solve_moving_average_params_hybrid(x_full_trace, z_full_trace, None, None, None)
        params_initial_guess = np.array([p_i, q_i, k_i, a_i, w_i])
        params_final_guess = params_initial_guess.copy()
    else:
        # --- NEW: Global Seeding Stage for Initial Fit ---
        # Define bounds for the 5 initial parameters, matching Figure7.
        bounds_5_param_initial = ([0.05, 0.2, 0.0, 0.0, 0.0], [20.0, 20.0, 10.0, 1.0, 0.5])
        de_bounds_5_param = list(zip(bounds_5_param_initial[0], bounds_5_param_initial[1]))

        de_result_initial = differential_evolution(
            _sse_cost_5_param,
            bounds=de_bounds_5_param,
            args=(z_initial_fit_data, x_initial_fit_data),
            maxiter=50, # A quicker search is sufficient for seeding
            popsize=15,
            tol=0.01,
            workers=1 # Avoid nested parallelism
        )
        p_seed, q_seed, k_seed, _, _ = de_result_initial.x
        
        # Stage 1: Fit Initial Model (now seeded by the global search)
        p_i, q_i, k_i, a_i, w_i = solve_moving_average_params_hybrid(
            x_initial_fit_data, z_initial_fit_data, p_seed, q_seed, k_seed, bounds=bounds_5_param_initial
        )
        params_initial_guess = np.array([p_i, q_i, k_i, a_i, w_i])

        # Stage 2: Fit Final Model (seeded with Stage 1 p,q,k)
        # Use slightly different bounds for the final fit, as in Figure7.
        bounds_5_param_final = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
        p_f, q_f, k_f, a_f, w_f = solve_moving_average_params_hybrid(x_full_trace, z_full_trace, p_i, q_i, k_i, bounds=bounds_5_param_final)
        params_final_guess = np.array([p_f, q_f, k_f, a_f, w_f])

    # --- Stage 3: Fit Blending Parameters with least_squares ---
    # The initial guess is the combined results from the first two stages.
    # For the initial guess of z_mid, find the z-value corresponding to the x_break
    z_break_guess = np.interp(x_break, x_full_trace, z_full_trace)
    x0 = np.concatenate([params_initial_guess, params_final_guess, [z_break_guess, 2.0]])

    # Use the original, stricter bounds for this final refinement stage.
    bounds_i_final = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
    bounds_f_final = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
    bounds_blend_final = ([0.0, 0.1], [30.0, 10.0]) # z_mid, z_width
    bounds = (np.concatenate([bounds_i_final[0], bounds_f_final[0], bounds_blend_final[0]]),
              np.concatenate([bounds_i_final[1], bounds_f_final[1], bounds_blend_final[1]]))

    # Run the final local optimization. No need for a global search here,
    # as the staged process provides an excellent starting point.
    res = least_squares(_residuals, x0, bounds=bounds, args=(z_full_trace, x_full_trace),
                        method='trf', ftol=1e-5, xtol=1e-5, max_nfev=2000)

    p_i, q_i, k_i, a_i, w_i, p_f, q_f, k_f, a_f, w_f, z_mid, z_width = res.x
    params_i_final = np.array([p_i, q_i, k_i, a_i, w_i])
    params_f_final = np.array([p_f, q_f, k_f, a_f, w_f])

    return params_i_final, params_f_final, z_mid, z_width

def _find_breakpoint_from_raw_data(x_row: np.ndarray, z_vec: np.ndarray) -> float | None:
    """
    Finds the natural breakpoint from a raw data curve, mimicking Figure7 logic.
    The breakpoint is the x-value of the last usable data point before extension.
    """
    finite = np.isfinite(x_row) & np.isfinite(z_vec)
    if not np.any(finite):
        return None
    x_f = x_row[finite]
    z_f = z_vec[finite]
    # Determine per-curve cutoff consistent with other figures
    x_max = float(np.max(x_f))
    x_cut = float(max(0.0, min(0.7, x_max - 0.01)))
    use = x_f <= x_cut
    if not np.any(use) or np.sum(use) < 3:
        return None
    
    x_use = x_f[use]
    return float(x_use[-1])

# --- Data Preparation Functions ---

def _quadratic_tail(z0: float, x0: float, s0: float, z1: float, x1: float, num: int = 240) -> tuple[np.ndarray, np.ndarray]:
    dz = z1 - z0
    if abs(dz) < 1e-9: return np.array([z0]), np.array([x0])
    a = ((x1 - x0) / dz - s0) / dz
    b = s0 - 2 * a * z0
    c = x0 - a * z0**2 - b * z0
    z_eval = np.linspace(z0, z1, num)
    x_eval = a * (z_eval**2) + b * z_eval + c
    return z_eval, np.clip(x_eval, 0.0, 0.7)

def _cubic_hermite_tail_z_of_x(x0: float, z0: float, s0: float, x1: float, z1: float, num: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a cubic Hermite spline tail for z as a function of x.
    This provides a smooth curve that matches the position (z0, z1) and slope
    (s0, s1) at the endpoints (x0, x1).
    """
    dx = x1 - x0
    if abs(dx) < 1e-9:
        return np.array([]), np.array([])

    # Define the slope at the endpoint. A reasonable choice is the slope of the
    # secant line, which encourages the curve to smoothly approach the final point.
    s1 = (z1 - z0) / dx

    # Create the parameter 't' that goes from 0 to 1 over the interval
    x_eval = np.linspace(x0, x1, num)
    t = (x_eval - x0) / dx

    # Cubic Hermite basis functions
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    # Combine the basis functions with the endpoint conditions
    z_eval = (h00 * z0 +
              h10 * dx * s0 +
              h01 * z1 +
              h11 * dx * s1)

    # Exclude the start point because it's already in the original data array
    return x_eval[1:], np.clip(z_eval[1:], 0.0, 30.0)

def _pchip_tail_z_of_x(x_data: np.ndarray, z_data: np.ndarray, x_target: float, z_target: float, num: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates an extension tail using PCHIP by appending the target point.
    This ensures the interpolated curve passes through the target, and PCHIP's
    nature prevents overshooting for a smooth, monotonic extension.
    """
    if len(x_data) < 2:
        return np.array([]), np.array([])

    # Append the target to guide the interpolator
    x_interp_data = np.append(x_data, x_target)
    z_interp_data = np.append(z_data, z_target)

    # PCHIP requires monotonically increasing x values and no duplicates.
    unique_x, unique_idx = np.unique(x_interp_data, return_index=True)
    
    if len(unique_x) < 2:
        return np.array([]), np.array([]) # Not enough unique points to interpolate

    unique_z = z_interp_data[unique_idx]

    # Build the PCHIP interpolator
    pchip = PchipInterpolator(unique_x, unique_z)

    # Generate the tail from the last point of the original data to the target
    x_start_tail = x_data[-1]
    x_tail = np.linspace(x_start_tail, x_target, num)
    z_tail = pchip(x_tail)

    # Return the tail, excluding the first point which is already in the data
    return x_tail[1:], np.clip(z_tail[1:], 0.0, 30.0)

def build_interpolated_behavior(x_row: np.ndarray, z_vec: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """
    Build a smooth extension tail from the rightmost usable CSV point to (30, 0.7).
    This is the full version from MolMemristor_Figure7.py.
    """
    finite = np.isfinite(x_row) & np.isfinite(z_vec)
    if not np.any(finite):
        return None
    x_f = x_row[finite]
    z_f = z_vec[finite]
    # Determine per-curve cutoff consistent with other figures
    x_max = float(np.max(x_f))
    x_cut = float(max(0.0, min(0.7, x_max - 0.01)))
    use = x_f <= x_cut
    if not np.any(use) or np.sum(use) < 3:
        return None
    x_use = x_f[use]
    z_use = z_f[use]
    
    z0 = float(z_use[-1])
    x0 = float(x_use[-1])

    # --- Robust Slope Calculation ---
    n_pts = max(3, int(0.2 * x_use.size))
    x_tail_fit_data = x_use[-n_pts:]
    z_tail_fit_data = z_use[-n_pts:]

    try:
        p = np.polyfit(z_tail_fit_data, x_tail_fit_data, 2)
        s0 = 2.0 * p[0] * z0 + p[1]
    except np.linalg.LinAlgError:
        s0 = (float(x_use[-1]) - float(x_use[-2])) / max(1e-9, float(z_use[-1] - z_use[-2]))

    z1, x1 = 30.0, 0.7
    z_tail, x_tail = _quadratic_tail(z0, x0, s0, z1, x1, num=240)
    
    return (z_tail, x_tail), (z_use, x_use)


# --- Stage 1: Linear Baseline ---

def calculate_linear_baselines(df_params: pd.DataFrame) -> dict:
    """Calculates a linear trend line for each parameter."""
    baselines = {}
    pulse_counts = df_params["Pulse Counts"].values
    for param_name in df_params.columns:
        if param_name == "Pulse Counts": continue
        param_values = df_params[param_name].values
        # polyfit deg=1 is a linear fit (returns m, b)
        coeffs = np.polyfit(pulse_counts, param_values, 1)
        baselines[param_name] = np.poly1d(coeffs)
    print("Calculated linear baselines for all parameters.")
    return baselines

def double_logistic_function(x, L1, k1, x01, v1, L2, k2, x02, v2):
    """A sum of two general logistic functions for more flexible curve fitting."""
    term1 = L1 / np.power(1 + np.exp(-k1 * (x - x01)), v1)
    term2 = L2 / np.power(1 + np.exp(-k2 * (x - x02)), v2)
    return term1 + term2

def _check_crossing(n_check, phim_check, z_grid_check, existing_curves_dict, allowed_crossings):
    """
    Checks if the curve to check crosses any of the existing curves with a lower pulse count.
    existing_curves_dict is a dict of {n: (phim, z_grid)}
    Returns True if it crosses (should be filtered), False otherwise.
    """
    try:
        # Create an interpolator for the curve being checked
        unique_phim_check, idx_check = np.unique(phim_check, return_index=True)
        if len(unique_phim_check) < 2: return True # Cannot interpolate, filter it
        z_interp_check = PchipInterpolator(unique_phim_check, z_grid_check[idx_check], extrapolate=False)

        # Compare against all existing curves with a lower pulse count
        for m, (phim_m, z_grid_m) in existing_curves_dict.items():
            if m >= n_check:
                continue

            # Create an interpolator for the existing curve
            unique_phim_m, idx_m = np.unique(phim_m, return_index=True)
            if len(unique_phim_m) < 2: continue
            z_interp_m = PchipInterpolator(unique_phim_m, z_grid_m[idx_m], extrapolate=False)

            # Define a common comparison range, avoiding the noisy start
            phim_min = max(unique_phim_check.min(), unique_phim_m.min(), 0.0001)
            phim_max = min(unique_phim_check.max(), unique_phim_m.max())

            if phim_min >= phim_max: continue # No overlapping range to compare

            # Create a grid and get z-values from both interpolators
            phim_compare_grid = np.linspace(phim_min, phim_max, 50)
            z_values_check = z_interp_check(phim_compare_grid)
            z_values_m = z_interp_m(phim_compare_grid)

            # Check if the curve to check is below the existing curve.
            is_below = z_values_check < z_values_m
            # A crossing in either direction is a transition from False to True, or True to False.
            # We can count these by looking for non-zero changes in the integer representation.
            total_crossings = np.sum(np.diff(is_below.astype(int)) != 0)

            # If it crosses more than the allowed amount, it's a significant deviation.
            if total_crossings > allowed_crossings:
                return True # It crosses, so filter it

    except (ValueError, IndexError):
        # This can happen with unusual data that breaks Pchip
        return True # Interpolation failed, filter it

    return False # No crossing found

def generate_smooth_phim_baseline_plot(df_params: pd.DataFrame, smoothing_window: int, smoothing_order: int, target_curve_count: int, allowed_crossings: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict]:
    """
    Generates a plot comparing the direct PCHIP-interpolated phim behavior
    with a version that is either smoothed or snapped to the n=0 linear benchmark
    based on "spatial aboveness".
    """
    print("\nGenerating smooth (logistic fit + snapping) phim behavioral baseline plot...")
    param_names = [c for c in df_params.columns if c != "Pulse Counts"]
    interpolators = build_parameter_interpolators(df_params)
    
    n_range = np.arange(0, 20001, 2000)  # Changed from linspace to arange for 500 steps
    z_grid = np.linspace(0.0, 30.0, 400)
    
    # First, calculate the n=0 curve and its linear version to use as the benchmark
    params_n0 = np.array([interpolators[p_name](0) for p_name in param_names])
    
    # Correctly unpack the 12 parameters for the function call
    params_i_n0 = params_n0[0:5]
    params_f_n0 = params_n0[5:10]
    z_mid_n0, z_width_n0 = params_n0[10], params_n0[11]
    phim_n0_orig = x_of_z_piecewise_blended(z_grid, params_i_n0, params_f_n0, z_mid_n0, z_width_n0)
    
    phim_n0_linear_benchmark = np.linspace(phim_n0_orig[0], phim_n0_orig[-1], len(z_grid))
    
    # Create the plot figure and axes before the loop
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Φ_mol Baseline: Original vs. Smoothed/Snapped")
    ax.set_xlabel("Φ_mol(z,n) (V)")
    ax.set_ylabel("z (Layer position)")
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 30.0)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=20000)

    # --- Data Collection and Initial Baseline Generation ---
    # Start with the pulse counts from the original fitted parameters, which correspond to the raw CSV data.
    all_pulse_counts_orig = df_params["Pulse Counts"].values
    
    # Explicitly add the higher, extrapolated pulse counts we want to generate.
    extrapolated_counts = np.arange(np.max(all_pulse_counts_orig) + 2000, 20001, 2000)
    
    # Combine and get a unique, sorted list of all pulse counts to process.
    # We explicitly add [0] to ensure the baseline is calculated and saved.
    all_pulse_counts_to_process = np.unique(np.concatenate([[0], all_pulse_counts_orig, extrapolated_counts]))
    
    baseline_map = {} # Using a dictionary {pulse_count: (z_grid, phim_curve)}

    # Load the raw data needed for building extrapolated curves
    phi_csv_path = Path(__file__).parent.parent / "Phi_from_probability.csv"
    if not phi_csv_path.exists():
        raise FileNotFoundError("Phi_from_probability.csv not found, which is required for the in-between curves.")
    df_phi = pd.read_csv(phi_csv_path)
    z_layers = np.array([float(c) for c in df_phi.columns if c != "Pulse Counts"])
    pulse_to_idx = {pc: i for i, pc in enumerate(df_phi["Pulse Counts"])}

    # Initialize counters for the plot text and a dict for raw curves
    smoothed_count = 0
    raw_tailed_count = 0
    raw_curves_map = {}

    for n in all_pulse_counts_to_process:
        # 1. Always get the original PCHIP interpolated curve as the 'before' state
        params_orig = np.array([interpolators[p_name](n) for p_name in param_names])
        params_i_orig, params_f_orig = params_orig[0:5], params_orig[5:10]
        z_mid_orig, z_width_orig = params_orig[10], params_orig[11]
        phim_original_interp = x_of_z_piecewise_blended(z_grid, params_i_orig, params_f_orig, z_mid_orig, z_width_orig)

        # 2. Determine the 'after' state (the final baseline curve)
        if n in n_range: # This is a major interval that needs smoothing
            smoothed_count += 1
            # This logic is the same as before, for smoothing and snapping
            try:
                p0 = [
                    np.max(phim_original_interp) * 0.8, 1.0, np.median(z_grid), 1.0,
                    np.max(phim_original_interp) * 0.2, 1.0, np.median(z_grid) * 0.8, 1.0
                ]
                fit_params, _ = curve_fit(double_logistic_function, z_grid, phim_original_interp, p0=p0, maxfev=10000)
                phim_logistic_fit = double_logistic_function(z_grid, *fit_params)
            except RuntimeError:
                print(f"Warning: Double logistic fit failed for n={n}. Using original curve.")
                phim_logistic_fit = phim_original_interp

            phim_final_smoothed = savgol_filter(phim_logistic_fit, smoothing_window, smoothing_order)
            
            # --- Rule-based Snapping based on "Spatial Aboveness" ---
            snap_to_benchmark = False
            if n == 0:
                snap_to_benchmark = True
            else:
                try:
                    # To compare spatially, we need inverse functions: z(phim)
                    # Ensure phim is monotonic for interpolation
                    unique_phim_orig, idx_orig = np.unique(phim_original_interp, return_index=True)
                    z_interp_orig = PchipInterpolator(unique_phim_orig, z_grid[idx_orig])

                    unique_phim_bench, idx_bench = np.unique(phim_n0_linear_benchmark, return_index=True)
                    z_interp_bench = PchipInterpolator(unique_phim_bench, z_grid[idx_bench])

                    # Create a common grid in the phim-domain to compare
                    phim_grid = np.linspace(np.max([unique_phim_orig.min(), unique_phim_bench.min()]), 
                                            np.min([unique_phim_orig.max(), unique_phim_bench.max()]), 
                                            200)

                    z_values_orig = z_interp_orig(phim_grid)
                    z_values_bench = z_interp_bench(phim_grid)

                    delta_z = z_values_orig - z_values_bench
                    area_spatially_below = np.abs(np.sum(delta_z[delta_z < 0]))
                    area_spatially_above = np.sum(delta_z[delta_z > 0])
                    
                    if area_spatially_below > area_spatially_above:
                        snap_to_benchmark = True

                except ValueError:
                    # This can happen if a curve is perfectly flat and cannot be interpolated
                    snap_to_benchmark = True

            if snap_to_benchmark:
                # Snap this curve to the n=0 linear benchmark
                phim_final = phim_n0_linear_benchmark
            else:
                phim_final = phim_final_smoothed
        else: # This is an in-between pulse count
            raw_tailed_count += 1
            # Use the Figure 7 logic to get the raw, extrapolated curve
            if n in pulse_to_idx:
                idx = pulse_to_idx[n]
                x_row = df_phi.iloc[idx, 1:].values
                built = build_interpolated_behavior(x_row, z_layers)
                if built:
                    (z_tail, x_tail), (z_csv, x_csv) = built
                    z_full = np.concatenate((z_csv, z_tail))
                    x_full = np.concatenate((x_csv, x_tail))
                    sort_indices = np.argsort(z_full)
                    # We need to re-interpolate this onto our common z_grid for consistency
                    phim_final = np.interp(z_grid, z_full[sort_indices], x_full[sort_indices])
                    # Store the true raw curve for later plotting
                    raw_curves_map[n] = (z_grid, phim_final)
                else:
                    phim_final = phim_original_interp # Fallback
            else:
                phim_final = phim_original_interp # Fallback

        # --- NEW: Normalize all final curves to start at 0 ---
        phim_final = phim_final - phim_final[0]

        baseline_map[n] = {
            'original': phim_original_interp,
            'final': phim_final,
            'final_curve_data': (z_grid, phim_final)
        }

    # --- Plotting Loop for Initial Curves ---
    plotted_curves = {} # Keep track of curves that pass filters and are plotted
    for n in all_pulse_counts_to_process:
        color = cmap(norm(n))
        phim_orig = baseline_map[n]['original']
        z_grid_final, phim_final = baseline_map[n]['final_curve_data']

        # "Before" state is always the original PCHIP interpolation (shifted to 0)
        phim_orig_shifted = phim_orig - phim_orig[0]
        if not _check_crossing(n, phim_orig_shifted, z_grid_final, plotted_curves, allowed_crossings):
             ax.plot(phim_orig_shifted, z_grid_final, '--', color='gray', alpha=0.7, lw=1.0)
        
        # "After" state is the initial baseline (smoothed or raw)
        # --- PLOTTING FILTERS ---
        if not _check_crossing(n, phim_final, z_grid_final, plotted_curves, allowed_crossings):
            ax.plot(phim_final, z_grid, '-', color=color, lw=2.0)
            plotted_curves[n] = (phim_final, z_grid)

    # --- Step 1: Find the common truncation point (leftmost endpoint) ---
    right_endpoints = []
    # Use the curves that have already been plotted (i.e., passed initial filters)
    for n in sorted(plotted_curves.keys()):
        phim_final, _ = plotted_curves[n]
        if phim_final.size > 0:
            right_endpoints.append(np.max(phim_final))
    
    if not right_endpoints:
        print("Warning: No valid curves found to determine truncation point.")
        truncation_phim = 0.7
    else:
        truncation_phim = np.min(right_endpoints)
    print(f"Common truncation point set to Φ_mol = {truncation_phim:.4f}")

    # --- Step 2: Truncate and Extend all curves to a common endpoint (0.7) ---
    processed_baseline_map = {}
    # Iterate through the original, full set of curves to process them
    for n in all_pulse_counts_to_process:
        if n not in baseline_map:
            continue
            
        z_grid, phim_final = baseline_map[n]['final_curve_data']
        
        # Truncate the curve at the common endpoint
        valid_indices = np.where(phim_final <= truncation_phim)[0]
        
        if len(valid_indices) < 3: # Need at least a few points for slope calculation
            continue

        phim_trunc = phim_final[valid_indices]
        z_trunc = z_grid[valid_indices]
        
        # Extend the curve to the final target point (0.7, 30.0) using PCHIP
        x_tail, z_tail = _pchip_tail_z_of_x(phim_trunc, z_trunc, x_target=0.7, z_target=30.0)
        
        phim_new = np.concatenate((phim_trunc, x_tail))
        z_new = np.concatenate((z_trunc, z_tail))

        # Store the processed curve back into a new map
        baseline_map[n]['final_curve_data'] = (z_new, phim_new)


    # --- Re-plot the processed curves and apply filters again ---
    ax.clear() # Clear the old plot
    ax.set_title("Φ_mol Baseline: Truncated, Extended, and Filtered")
    ax.set_xlabel("Φ_mol(z,n) (V)")
    ax.set_ylabel("z (Layer position)")
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 30.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plotted_curves = {}
    for n in all_pulse_counts_to_process:
        if n not in baseline_map: continue
        color = cmap(norm(n))
        z_processed, phim_processed = baseline_map[n]['final_curve_data']

        if not _check_crossing(n, phim_processed, z_processed, plotted_curves, allowed_crossings):
            ax.plot(phim_processed, z_processed, '-', color=color, lw=2.0)
            plotted_curves[n] = (phim_processed, z_processed)

    # --- Iteratively add synthetic curves until TARGET_CURVE_COUNT is met ---
    while len(plotted_curves) < target_curve_count:
        current_pulse_counts = sorted(plotted_curves.keys())
        new_midpoints_to_add = []

        for i in range(len(current_pulse_counts) - 1):
            n1 = current_pulse_counts[i]
            n2 = current_pulse_counts[i+1]
            
            mid_n = (n1 + n2) / 2.0
            # Ensure we are not re-adding a curve that was filtered out
            if mid_n not in plotted_curves and mid_n not in baseline_map:
                new_midpoints_to_add.append(mid_n)
        
        if not new_midpoints_to_add: # No more midpoints can be added
            break

        for n_new in new_midpoints_to_add:
            # Find the curves to average between from the already plotted curves
            n_lower_idx = np.searchsorted(current_pulse_counts, n_new, side='right') - 1
            n_upper_idx = np.searchsorted(current_pulse_counts, n_new, side='left')
            
            if n_upper_idx >= len(current_pulse_counts): continue

            n_lower = current_pulse_counts[n_lower_idx]
            n_upper = current_pulse_counts[n_upper_idx]
            
            phim_lower, z_lower = plotted_curves[n_lower]
            phim_upper, z_upper = plotted_curves[n_upper]
            
            # Interpolate both curves onto a common phim grid for averaging z values
            phim_common = np.linspace(0, 0.7, 400)
            z_interp_lower = np.interp(phim_common, phim_lower, z_lower)
            z_interp_upper = np.interp(phim_common, phim_upper, z_upper)

            z_synthetic_target = (z_interp_lower + z_interp_upper) / 2.0
            
            # Store the new synthetic curve.
            # We add it to baseline_map so it can be found by the final return statement
            baseline_map[n_new] = {
                'original': z_synthetic_target, # Placeholder
                'final': z_synthetic_target,
                'final_curve_data': (z_synthetic_target, phim_common)
            }
            
            # Add to the plot for visualization, checking for crossings
            color = cmap(norm(n_new))
            if not _check_crossing(n_new, phim_common, z_synthetic_target, plotted_curves, allowed_crossings):
                ax.plot(phim_common, z_synthetic_target, '-', color=color, lw=1.0, alpha=0.7)
                plotted_curves[n_new] = (phim_common, z_synthetic_target)


    # --- Finalize, Filter, Sort, and Show ---
    # From all generated curves, only keep those that passed the crossing filter.
    # This ensures we only run the expensive fitting process on the "good" curves.
    all_pulse_counts_combined = sorted(plotted_curves.keys())
    # We now fetch the final data from the baseline_map, which contains the extended curves
    final_baseline_curves_combined = [baseline_map[n]['final_curve_data'] for n in all_pulse_counts_combined]
    
    # --- Add Informational Text to Plot ---
    count_pchip = len(all_pulse_counts_to_process)
    # Correctly count synthetic curves that were actually added and plotted
    synthetic_count = max(0, len(plotted_curves) - count_pchip)
    total_final_lines = len(all_pulse_counts_combined)
    
    info_text = (
        f"Line Counts:\n"
        f"- PCHIP Curves Processed: {count_pchip}\n"
        f"- Initial Baseline Curves: {count_pchip}\n"
        f"  - Raw w/ Tail: {raw_tailed_count}\n"
        f"  - Smoothed/Extrapolated: {smoothed_count}\n"
        f"- Synthetic Midpoints Added: {synthetic_count}\n"
        f"- Total Final Curves for Fitting: {total_final_lines}"
    )
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    from matplotlib.lines import Line2D
    legend_elements = [
                       Line2D([0], [0], color='gray', ls='-', lw=2.0, label='Processed Baseline'),
                       Line2D([0], [0], color='gray', ls='-', lw=1.0, alpha=0.7, label='Synthetic Midpoint')]
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return final_baseline_curves_combined, np.array(all_pulse_counts_combined), raw_curves_map


def build_parameter_interpolators(df_params: pd.DataFrame) -> dict[str, PchipInterpolator]:
    """Builds PchipInterpolators for each parameter column in the dataframe."""
    interpolators = {}
    param_names = [c for c in df_params.columns if c != "Pulse Counts"]
    pulse_counts = df_params["Pulse Counts"].values
    for param_name in param_names:
        param_values = df_params[param_name].values
        interpolators[param_name] = PchipInterpolator(pulse_counts, param_values, extrapolate=True)
    return interpolators

def _run_single_fit(pulse_count: float, curve_data: tuple[np.ndarray, np.ndarray], x_break: float):
    """
    A top-level helper function to fit a single curve, suitable for parallel execution.
    """
    z_smooth, x_smooth = curve_data
    
    # Filter out any potential NaNs from the smoothing process if necessary
    valid_indices = ~np.isnan(x_smooth) & ~np.isnan(z_smooth)
    z_target = z_smooth[valid_indices]
    x_target = x_smooth[valid_indices]
    
    if len(x_target) < 10: # Need enough points for a stable fit
        return None
        
    print(f"Fitting for n = {pulse_count} (x_break={x_break:.2f})...")
    params_i, params_f, z_mid, z_width = solve_full_piecewise_model(x_target, z_target, x_break)
    
    # Combine all 12 parameters into one list for returning
    all_params = np.concatenate([params_i, params_f, [z_mid, z_width]])
    return pulse_count, all_params

def _calculate_global_residuals(params_flat: np.ndarray, num_curves: int, num_params_per_curve: int,
                                z_targets_list: list[np.ndarray], x_targets_list: list[np.ndarray],
                                lambda_reg: float) -> np.ndarray:
    """
    Calculates residuals for a global optimization problem.
    This includes two components:
    1. Fitting Error: The standard deviation between the model and the target curves.
    2. Regularization Term: A penalty for large differences in parameters between
       adjacent curves, enforcing smoothness.
    """
    # Reshape the flat parameter array into a 2D array (num_curves x num_params)
    params_matrix = params_flat.reshape((num_curves, num_params_per_curve))
    
    # --- 1. Calculate Fitting Residuals ---
    fitting_residuals = []
    for i in range(num_curves):
        # Unpack the 12 parameters for the current curve
        pi, qi, ki, ai, wi, pf, qf, kf, af, wf, z_mid, z_width = params_matrix[i]
        params_i = np.array([pi, qi, ki, ai, wi])
        params_f = np.array([pf, qf, kf, af, wf])
        
        z_target = z_targets_list[i]
        x_target = x_targets_list[i]
        
        # Calculate the model output and the residual for this curve
        x_model = x_of_z_piecewise_blended(z_target, params_i, params_f, z_mid, z_width)
        fitting_residuals.extend(x_model - x_target)
        
    # --- 2. Calculate Regularization (Smoothness) Residuals ---
    # The penalty is applied to the difference between parameters of adjacent curves.
    # We scale this by sqrt(lambda) so the cost function penalty is lambda * diff^2.
    param_diffs = np.diff(params_matrix, axis=0)
    regularization_residuals = np.sqrt(lambda_reg) * param_diffs.flatten()
    
    # --- 3. Combine and Return ---
    # The optimizer will minimize the sum of squares of this combined vector.
    return np.concatenate([fitting_residuals, regularization_residuals])

# --- Main Application Logic ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run regularized fitting for MolMemristor parameters.")
    parser.add_argument('--load', nargs='?', const='fitted_parameters_regularized.csv', default=None,
                        help='Load existing regularized parameters from a file and proceed to visualization. '
                             'If no filename is given, defaults to "fitted_parameters_regularized.csv".')
    args = parser.parse_args()

    # --- Configuration ---
    LAMBDA_REG = 1e-4  # Regularization strength. Higher = smoother params.
    ALLOWED_CROSSINGS = 1      # Max allowed crossings in either direction. Higher value = less strict filtering.
    # --- Final Smoothing Tunable Parameters ---
    SMOOTHING_WINDOW = 21      # Must be odd. Larger = more smoothing.
    SMOOTHING_ORDER = 3        # Polynomial order. Smaller = more smoothing.
    TARGET_CURVE_COUNT = 50    # <<<--- NEW: Set the desired number of final curves to train on.
    N_JOBS = min(61, max(1, int((os.cpu_count() or 1) * 0.8)))  # Capped at 61 for Windows WaitForMultipleObjects limit
    print(f"Using {N_JOBS} parallel processes for cost function.")

    # --- Setup and Data Loading ---
    base_path = Path(__file__).parent.parent
    phi_csv_path = base_path / "Phi_from_probability.csv"
    params_csv_path = base_path / "fitted_parameters.csv"
    
    df_phi = pd.read_csv(phi_csv_path)
    z_layers = np.array([float(c) for c in df_phi.columns if c != "Pulse Counts"])
    df_params_orig = pd.read_csv(params_csv_path).sort_values(by="Pulse Counts").dropna()

    param_names = [c for c in df_params_orig.columns if c != "Pulse Counts"]
    
    # --- Stage 1: Generate the Ideal Smoothed Baseline Curves ---
    print("Generating ideal smoothed baseline curves...")
    ideal_baseline_curves, n_range_for_baseline, raw_curves_for_plot = generate_smooth_phim_baseline_plot(df_params_orig, SMOOTHING_WINDOW, SMOOTHING_ORDER, TARGET_CURVE_COUNT, ALLOWED_CROSSINGS)
    print(f"Generated a final baseline with {len(ideal_baseline_curves)} total curves for pulse counts.")
    
    if args.load:
        print(f"Loading existing regularized parameters from {args.load}...")
        load_path = base_path / args.load
        if not load_path.exists():
            print(f"Error: File not found at {load_path}")
            return
        df_params_new = pd.read_csv(load_path)

    else:
        # --- CONSTANT BREAKPOINT ---
        # To align with the original Figure7 logic, a single, constant breakpoint
        # is now used for all curves. The value 0.17 is chosen as a robust,
        # representative value from the original raw data breakpoints.
        X_BREAK_CONSTANT = 0.17
        final_breakpoints = np.full_like(n_range_for_baseline, X_BREAK_CONSTANT, dtype=float)
        print(f"Using constant breakpoint x_break = {X_BREAK_CONSTANT} for all curves.")

        output_path = base_path / "fitted_parameters_regularized.csv"
        
        # --- Stage 2: Fit the Piecewise Model to Each Smoothed Curve ---
        print("\nStarting parallel fitting process for each smoothed curve...")
        num_cores = min(61, max(1, int((os.cpu_count() or 1) * 0.8)))  # Capped at 61 for Windows WaitForMultipleObjects limit
        
        # Create a list of tasks to be executed in parallel
        tasks = zip(n_range_for_baseline, ideal_baseline_curves, final_breakpoints)

        results = Parallel(n_jobs=num_cores, prefer="processes")(
            delayed(_run_single_fit)(pulse_count, curve_data, x_break) for pulse_count, curve_data, x_break in tasks
        )
        print("\nParallel fitting finished.")

        # --- Stage 3: Process Initial Guess ---
        # Filter out None results from the parallel run and sort them by pulse count
        valid_results = sorted([r for r in results if r is not None], key=lambda item: item[0])
        
        if not valid_results:
            print("Error: No curves were successfully fitted in the initial stage.")
            return

        # Unpack the results from the initial fit to use as a starting guess
        pulse_counts_initial = np.array([r[0] for r in valid_results])
        params_initial_guess_matrix = np.array([r[1] for r in valid_results])
        
        # Also unpack the target data that corresponds to these valid results
        # by matching the pulse counts back to the original ideal curves.
        initial_pc_to_idx = {pc: i for i, pc in enumerate(n_range_for_baseline)}
        valid_indices = [initial_pc_to_idx[pc] for pc in pulse_counts_initial]
        
        z_targets = [ideal_baseline_curves[i][0] for i in valid_indices]
        x_targets = [ideal_baseline_curves[i][1] for i in valid_indices]
        num_curves = len(valid_results)
        num_params_per_curve = params_initial_guess_matrix.shape[1]

        # --- User Prompt for Stage 4 ---
        choice = input("Perform global regularized optimization (Stage 4)? [y/N]: ").lower().strip()

        if choice == 'y':
            # --- Stage 4: Global Regularized Optimization ---
            print("\nStarting global regularized optimization...")
            
            # Flatten the initial guess matrix into a 1D vector for the optimizer
            x0_global = params_initial_guess_matrix.flatten()
            
            # Create the bounds for the global problem by tiling the single-curve bounds
            bounds_i = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
            bounds_f = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
            bounds_blend = ([0.0, 0.1], [30.0, 10.0]) # z_mid, z_width
            single_curve_bounds_low = np.concatenate([bounds_i[0], bounds_f[0], bounds_blend[0]])
            single_curve_bounds_high = np.concatenate([bounds_i[1], bounds_f[1], bounds_blend[1]])
            
            bounds_low_global = np.tile(single_curve_bounds_low, num_curves)
            bounds_high_global = np.tile(single_curve_bounds_high, num_curves)
            
            # Run the global least_squares optimization
            res_global = least_squares(
                _calculate_global_residuals,
                x0_global,
                bounds=(bounds_low_global, bounds_high_global),
                args=(num_curves, num_params_per_curve, z_targets, x_targets, LAMBDA_REG),
                method='trf',
                verbose=2, # Print optimizer progress at each iteration
                ftol=1e-5,
                xtol=1e-5
            )
            print("Global optimization finished.")
            params_final_matrix = res_global.x.reshape((num_curves, num_params_per_curve))
            pulse_counts_final = pulse_counts_initial

        else:
            print("\nSkipping global regularized optimization. Using initial fit parameters from Stage 2.")
            params_final_matrix = params_initial_guess_matrix
            pulse_counts_final = pulse_counts_initial

        # --- Stage 5: Finalize and Save Results ---
        # Reshape the optimized flat vector back into a matrix (num_curves x num_params)
        param_names_for_df = ['p_i', 'q_i', 'k_i', 'a_i', 'w_i',
                       'p_f', 'q_f', 'k_f', 'a_f', 'w_f',
                       'z_mid', 'z_width']
        
        df_params_new = pd.DataFrame(params_final_matrix, columns=param_names_for_df)
        df_params_new.insert(0, "Pulse Counts", pulse_counts_final)
        df_params_new = df_params_new.sort_values("Pulse Counts").reset_index(drop=True)
        
        df_params_new.to_csv(output_path, index=False)
        print(f"Saved new fitted parameters to {output_path}")

    # --- Stage 6: Visualization ---
    # Load original and newly fitted parameters for comparison
    df_params_final = df_params_new # Use the new dataframe for plotting

    # --- Plot 1: Parameter Evolution Comparison ---
    fig1, axes1 = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    fig1.suptitle('Parameter Evolution: Original vs. Smoothed Target Fit', fontsize=16)
    axes1 = axes1.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes1[i]
        # Original jagged points
        ax.plot(df_params_orig["Pulse Counts"], df_params_orig[param_name], 'o--', color='skyblue', label='Original Fit', alpha=0.7)
        # New fit from smoothed target
        ax.plot(df_params_final["Pulse Counts"], df_params_final[param_name], 'o-', color='red', linewidth=2, label='Fit to Smoothed Target')
        ax.set_title(param_name)
        ax.grid(True, linestyle='--', alpha=0.6)

    for i in range(len(param_names), len(axes1)):
        axes1[i].set_visible(False) # Hide unused subplots
    
    # Create a single legend for the entire figure
    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))


    # --- Plot 2: Accuracy Verification ---
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    ax2.set_xlabel("Φ_mol(z,n) (V)")
    ax2.set_ylabel("z (Layer position)")
    ax2.set_xlim(0.0, 0.7)
    ax2.set_ylim(0.0, 30.0)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title("Φ_mol Curve Accuracy: New Fit vs. Original Raw Data")
    
    # Build PCHIP interpolators for the new parameters
    interpolators_new = build_parameter_interpolators(df_params_final)
    
    # Use a colormap with enough colors for all the new curves
    cmap = plt.cm.get_cmap("viridis", len(ideal_baseline_curves))

    # Plot the results for ALL baseline pulses
    z_grid = np.linspace(0, 30, 400)
    
    plotted_fits = {} # Keep track of curves that pass filters and are plotted
    for i, n in enumerate(n_range_for_baseline):
        color = cmap(i)

        # Plot Result (the new parameter fit)
        if n in interpolators_new['p_i'].x:
            params = [interpolators_new[p](n) for p in param_names]
            params_i = np.array(params[0:5])
            params_f = np.array(params[5:10])
            z_mid, z_width = params[10], params[11]
            
            phim_fit = x_of_z_piecewise_blended(z_grid, params_i, params_f, z_mid, z_width)
            # --- PLOTTING FILTERS ---
            # Shift the curve to start at 0 before checking/plotting
            phim_fit_shifted = phim_fit - phim_fit[0]
            if not _check_crossing(n, phim_fit_shifted, z_grid, plotted_fits, ALLOWED_CROSSINGS):
                ax2.plot(phim_fit_shifted, z_grid, '-', color=color, linewidth=2.0)
                plotted_fits[n] = (phim_fit_shifted, z_grid)

    # Plot the original raw curves on top as a reference
    plotted_raws = {} # Keep track of curves that pass filters and are plotted
    for n, (z_raw, phim_raw) in sorted(raw_curves_for_plot.items()):
        # --- PLOTTING FILTERS ---
        # Shift the curve to start at 0 before checking/plotting
        phim_raw_shifted = phim_raw - phim_raw[0]
        if not _check_crossing(n, phim_raw_shifted, z_raw, plotted_raws, ALLOWED_CROSSINGS):
            ax2.plot(phim_raw_shifted, z_raw, '--', color='black', linewidth=1.0, zorder=10)
            plotted_raws[n] = (phim_raw_shifted, z_raw)

    # Create a cleaner legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2.0, linestyle='-', label='New Parameter Fit (Colored by Pulse Count)'),
        Line2D([0], [0], color='black', lw=1.0, linestyle='--', label='Original Raw Curve')
    ]
    ax2.legend(handles=legend_elements, loc='best')

    plt.show()

if __name__ == "__main__":
    main()
