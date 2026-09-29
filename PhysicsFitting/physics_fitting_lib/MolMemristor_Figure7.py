import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
import argparse
import sys
from scipy.optimize import least_squares


try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_PCHIP = True
except Exception:
    _HAVE_PCHIP = False

try:
    from numba import njit, prange  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def x_of_z_combined(z_values: np.ndarray, p: float, q: float, k: float) -> np.ndarray:
    t = z_values / 30.0
    s = (1.0 - (1.0 - t) ** q) ** p
    x = (1.0 + k) * s / (1.0 + k * (s ** 2))
    return 0.7 * np.clip(x, 0.0, 1.0)


def x_of_z_moving_average(z_values: np.ndarray, p: float, q: float, k: float, alpha: float, w: float) -> np.ndarray:
    """
    Moving-average smoothing with tapered blend to preserve endpoints:
      u(t) from combined curve; 3-tap symmetric MA with clamped neighbors;
      a_eff(t) = alpha * smoothstep(t/w) * smoothstep((1-t)/w).
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


def _quadratic_tail(z0: float, x0: float, s0: float, z1: float, x1: float, num: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a quadratic (parabolic) tail from (z0,x0) to (z1,x1) that
    starts with the slope s0. A quadratic is used to avoid the inflection
    point that a cubic spline can create by not constraining the end slope.
    """
    dz = z1 - z0
    if abs(dz) < 1e-9:
        return np.array([z0]), np.array([x0])

    # Solve for a,b,c in x(z) = a*z^2 + b*z + c
    a = ((x1 - x0) / dz - s0) / dz
    b = s0 - 2 * a * z0
    c = x0 - a * z0**2 - b * z0

    z_eval = np.linspace(z0, z1, num)
    x_eval = a * (z_eval**2) + b * z_eval + c
    return z_eval, np.clip(x_eval, 0.0, 0.7)


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


def build_interpolated_behavior(x_row: np.ndarray, z_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build a smooth extension tail from the rightmost usable CSV point to (30, 0.7).
    The extension's starting slope is determined by fitting a quadratic to the
    last segment of the data, ensuring it follows the overall trend.
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
    if not np.any(use):
        return None
    x_use = x_f[use]
    z_use = z_f[use]
    if x_use.size < 3:  # Need at least 3 points for a stable fit
        return None
    # Start at the last point within cutoff
    z0 = float(z_use[-1])
    x0 = float(x_use[-1])

    # --- Robust Slope Calculation ---
    # Fit a quadratic to the last ~20% of the data, fitting x = f(z), which is
    # more stable as dx/dz approaches zero at the end of the trace.
    # This captures the local curvature for a more natural extension.
    n_pts = max(3, int(0.2 * x_use.size))
    x_tail = x_use[-n_pts:]
    z_tail = z_use[-n_pts:]

    try:
        # Fit a 2nd order polynomial: x = p[0]*z^2 + p[1]*z + p[2]
        p = np.polyfit(z_tail, x_tail, 2)
        # The derivative dx/dz is 2*p[0]*z + p[1]. Evaluate at the endpoint z0.
        s0 = 2.0 * p[0] * z0 + p[1]
    except np.linalg.LinAlgError:
        # Fallback to simple 2-point slope for ill-conditioned fits
        s0 = (float(x_use[-1]) - float(x_use[-2])) / max(1e-9, float(z_use[-1] - z_use[-2]))

    # Endpoint is fixed, but slope is unconstrained for a natural curve
    z1, x1 = 30.0, 0.7
    return _quadratic_tail(z0, x0, s0, z1, x1, num=240), (z_use, x_use)


def fit_combined_seed(x_data: np.ndarray, z_data: np.ndarray) -> tuple[float, float, float]:
    """
    Quick coarse fit for (p,q,k) on the combined curve to seed Figure 7 fits.
    """
    def sse(p: float, q: float, k: float) -> float:
        xp = x_of_z_combined(z_data, p, q, k)
        d = xp - x_data
        return float(np.dot(d, d))

    p_lo, p_hi = 0.2, 1.5
    q_lo, q_hi = 0.8, 12.0
    k_lo, k_hi = 0.0, 5.0
    best = (0.8, 2.0, 0.5, np.inf)
    for _ in range(2):
        P = np.linspace(p_lo, p_hi, 15)
        Q = np.linspace(q_lo, q_hi, 15)
        K = np.linspace(k_lo, k_hi, 11)
        for p_try in P:
            for q_try in Q:
                for k_try in K:
                    e = sse(float(p_try), float(q_try), float(k_try))
                    if e < best[3]:
                        best = (float(p_try), float(q_try), float(k_try), e)
        # refine around best
        p_c, q_c, k_c, _ = best
        dp = 0.15 * max(1e-3, p_c)
        dq = 0.15 * max(1e-3, q_c)
        dk = 0.15 * max(1e-3, k_c + 1e-3)
        p_lo, p_hi = max(0.05, p_c - dp), p_c + dp
        q_lo, q_hi = max(0.2, q_c - dq), q_c + dq
        k_lo, k_hi = max(0.0, k_c - dk), k_c + dk
    return float(best[0]), float(best[1]), float(best[2])


def solve_full_piecewise_model(x_csv: np.ndarray, z_csv: np.ndarray,
                                 x_tail: np.ndarray, z_tail: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fits the full piecewise blended model.
    1. Fits an "initial" model to the raw CSV data (anchored at the endpoint).
    2. Fits a "final" model to the full trace, seeded by the initial fit.
    3. Fits the blend parameters to smoothly combine the two models.
    """
    # --- Stage 1: Fit Initial Model ---
    # Data is the raw CSV import plus the final corner point for anchoring.
    x_initial_fit = np.append(x_csv, x_tail[-1])
    z_initial_fit = np.append(z_csv, z_tail[-1])
    p_i, q_i, k_i, a_i, w_i = solve_moving_average_params_hybrid(x_initial_fit, z_initial_fit, None, None, None)
    params_initial_guess = np.array([p_i, q_i, k_i, a_i, w_i])

    # --- Stage 2: Fit Final Model (seeded) ---
    # Data is the full trace (CSV + tail).
    x_full_trace = np.append(x_csv, x_tail)
    z_full_trace = np.append(z_csv, z_tail)
    p_f, q_f, k_f, a_f, w_f = solve_moving_average_params_hybrid(x_full_trace, z_full_trace, p_i, q_i, k_i)
    params_final_guess = np.array([p_f, q_f, k_f, a_f, w_f])

    # --- Stage 3: Fit Blending Parameters ---
    # The breakpoint is the z-value where the CSV data ends.
    z_break = z_csv[-1]

    def residuals(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        pi, qi, ki, ai, wi, pf, qf, kf, af, wf, z_mid, z_width = params
        param_i = np.array([pi, qi, ki, ai, wi])
        param_f = np.array([pf, qf, kf, af, wf])
        x_model = x_of_z_piecewise_blended(z_vals, param_i, param_f, z_mid, z_width)
        return x_model - x_vals

    # Initial guess for the full 12-parameter optimization
    x0 = np.concatenate([params_initial_guess, params_final_guess, [z_break, 2.0]])
    
    # Bounds for all 12 parameters
    bounds_i = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
    bounds_f = ([0.05, 0.2, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, 1.0, 0.5])
    bounds_blend = ([0.0, 0.1], [30.0, 10.0]) # z_mid, z_width
    bounds = (np.concatenate([bounds_i[0], bounds_f[0], bounds_blend[0]]),
              np.concatenate([bounds_i[1], bounds_f[1], bounds_blend[1]]))

    res = least_squares(residuals, x0, bounds=bounds, args=(z_full_trace, x_full_trace),
                        method='trf', ftol=1e-5, xtol=1e-5, max_nfev=1000)

    p_i, q_i, k_i, a_i, w_i, p_f, q_f, k_f, a_f, w_f, z_mid, z_width = res.x
    params_i_final = np.array([p_i, q_i, k_i, a_i, w_i])
    params_f_final = np.array([p_f, q_f, k_f, a_f, w_f])

    return params_i_final, params_f_final, z_mid, z_width


def solve_moving_average_params_hybrid(x: np.ndarray, z: np.ndarray,
                                       p0: float | None, q0: float | None, k0: float | None) -> tuple[float, float, float, float, float]:
    """
    Hybrid optimization:
    1. Coarse, wide grid search to find a good starting region (robust).
    2. Fine-tuning with a fast local optimizer (scipy.optimize.least_squares)
       to get the precise minimum (fast and accurate).
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

    # Use a much sparser grid than the original implementation
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
        # Fallback vectorized search for the coarse grid if numba is not available
        # This part remains the same as the original, just on a smaller grid
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
    def residuals(params: list[float], z_vals: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        p, q, k, a, w = params
        x_model = x_of_z_moving_average(z_vals, p, q, k, a, w)
        return x_model - x_vals

    x0 = [p_coarse, q_coarse, k_coarse, a_coarse, w_coarse]
    bounds = ([0.05, 0.2, 0.0, 0.0, 0.0],
              [np.inf, np.inf, np.inf, 1.0, 0.5])

    res = least_squares(residuals, x0, bounds=bounds, args=(z, x), method='trf', ftol=1e-6, xtol=1e-6)
    p_final, q_final, k_final, a_final, w_final = res.x

    return float(p_final), float(q_final), float(k_final), float(a_final), float(w_final)


def _resolve_n_jobs() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--low-mem", action="store_true", help="Use a single process for parallel sections")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override number of parallel processes")
    max_procs = min(61, max(1, int((os.cpu_count() or 1) * 0.8)))  # Capped at 61 for Windows WaitForMultipleObjects limit
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        return max_procs
    if args.low_mem:
        return 1
    if args.n_jobs is not None and args.n_jobs > 0:
        return max(1, min(int(args.n_jobs), max_procs))
    # Interactive numeric prompt (defaults to capped CPU count)
    try:
        if sys.stdin and sys.stdin.isatty():
            ans = input(f"Number of processes to use [default {max_procs}]: ").strip()
            if ans == "":
                return max_procs
            val = int(ans)
            return max(1, min(val, max_procs))
    except Exception:
        pass
    return max_procs


def main() -> None:
    N_JOBS = _resolve_n_jobs()
    # Load CSV
    phi_csv = Path(__file__).parent.parent / "Phi_from_probability.csv"
    if not phi_csv.exists():
        raise FileNotFoundError("Phi_from_probability.csv not found. Run phi_from_probability.py first.")
    df_phi = pd.read_csv(phi_csv)
    z_cols = [c for c in df_phi.columns if c != "Pulse Counts"]
    z_layers = np.array([float(c) for c in z_cols], dtype=float)
    pulse_counts = df_phi["Pulse Counts"].to_numpy(dtype=float)
    phi_vals = df_phi[z_cols].to_numpy(dtype=float)

    # Color map
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=float(np.nanmin(pulse_counts)), vmax=float(np.nanmax(pulse_counts)))

    # Per-curve: build interpolated behavior and fit moving-average params
    def _fit_one(i: int):
        x_row = phi_vals[i, :]
        built = build_interpolated_behavior(x_row, z_layers)
        if built is None:
            return None
        
        (z_tail, x_tail), (z_csv, x_csv) = built
        
        # This now returns all 12 parameters for the blended model
        params_i, params_f, z_mid, z_width = solve_full_piecewise_model(x_csv, z_csv, x_tail, z_tail)
        return i, params_i, params_f, z_mid, z_width, z_tail, x_tail, z_csv, x_csv

    results = Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(_fit_one)(i) for i in range(len(pulse_counts)))

    # Plot: overlay raw CSV dashed lines as in other figures (masked by x<=x_cut),
    # then smoothly extend each to (0.7, 30) from its own last masked point.
    fig, ax = plt.subplots()
    ax.set_title("Figure 7: Refit to PCHIP-extended behavioral lines (to 0.7,30)")
    order = np.argsort(pulse_counts)
    for i in order:
        match = next((r for r in results if r is not None and r[0] == i), None)
        if match is None:
            continue
        _, params_i, params_f, z_mid, z_width, z_tail, x_tail, z_csv, x_csv = match
        color = cmap(norm(float(pulse_counts[i])))
        z_grid = np.linspace(0.0, 30.0, 400)
        # model curve using the new blended model
        ax.plot(x_of_z_piecewise_blended(z_grid, params_i, params_f, z_mid, z_width), z_grid, color=color, linewidth=2.0)
        # raw CSV dashed segment, now in black and on top
        ax.plot(x_csv, z_csv, color='k', linewidth=0.8, linestyle="--", alpha=0.7)
        # extension tail, also in black and on top
        ax.plot(x_tail, z_tail, color='k', linewidth=0.8, linestyle=":", alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Pulse counts")
    ax.set_xlabel("Phi_mol(z,n) (V)")
    ax.set_ylabel("z (Layer position)")
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 30.0)
    plt.tight_layout()

    # Behavioral-only figure: raw CSV dashed segments (masked) + extension tails only
    fig2, ax2 = plt.subplots()
    ax2.set_title("Figure 8: Behavioral lines only (raw dashed + extension to 0.7,30)")
    for i in order:
        color = cmap(norm(float(pulse_counts[i])))
        # Data now comes directly from the results of the fitting process
        match = next((r for r in results if r is not None and r[0] == i), None)
        if match is None:
            continue
        
        _, _, _, _, _, z_tail, x_tail, z_csv, x_csv = match

        # raw CSV dashed segment from the delineation point
        ax2.plot(x_csv, z_csv, color=color, linewidth=1.2, linestyle="--", alpha=0.9)
        # extension tail
        ax2.plot(x_tail, z_tail, color=color, linewidth=1.2, linestyle=":", alpha=0.9)

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2)
    cbar2.set_label("Pulse counts")
    ax2.set_xlabel("Phi_mol(z,n) (V)")
    ax2.set_ylabel("z (Layer position)")
    ax2.set_xlim(0.0, 0.7)
    ax2.set_ylim(0.0, 30.0)
    plt.tight_layout()

    # Figure 9: Refit to RAW CSV (no interpolation or tail extension), same model as Figure 6
    def _build_raw_target(x_row: np.ndarray, z_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        finite = np.isfinite(x_row) & np.isfinite(z_vec)
        if not np.any(finite):
            return None
        x_f = x_row[finite]
        z_f = z_vec[finite]
        x_max = float(np.max(x_f))
        x_cut = float(max(0.0, min(0.7, x_max - 0.01)))
        use = x_f <= x_cut
        if not np.any(use):
            return None
        return z_f[use], x_f[use]

    def _fit_raw(i: int):
        x_row = phi_vals[i, :]
        built = _build_raw_target(x_row, z_layers)
        if built is None:
            return None
        z_raw, x_raw = built
        # This part will now use the original hybrid solver, not the piecewise one,
        # as it is fitting to a different target (raw data only).
        p_seed, q_seed, k_seed = fit_combined_seed(x_raw, z_raw)
        p, q, k, a, w = solve_moving_average_params_hybrid(x_raw, z_raw, p_seed, q_seed, k_seed)
        return i, p, q, k, a, w, z_raw, x_raw

    results_raw = Parallel(n_jobs=N_JOBS, prefer="processes")(delayed(_fit_raw)(i) for i in range(len(pulse_counts)))

    fig3, ax3 = plt.subplots()
    ax3.set_title("Figure 9: Refit to raw CSV (no tail), moving-average model")
    for i in order:
        match = next((r for r in results_raw if r is not None and r[0] == i), None)
        if match is None:
            continue
        _, p, q, k, a, w, z_raw, x_raw = match
        color = cmap(norm(float(pulse_counts[i])))
        z_grid = np.linspace(0.0, 30.0, 400)
        # model curve
        ax3.plot(x_of_z_moving_average(z_grid, p, q, k, a, w), z_grid, color=color, linewidth=2.0)
        # raw dashed overlay
        ax3.plot(x_raw, z_raw, color=color, linewidth=1.0, linestyle="--", alpha=0.8)

    sm3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax3)
    cbar3.set_label("Pulse counts")
    ax3.set_xlabel("Phi_mol(z,n) (V)")
    ax3.set_ylabel("z (Layer position)")
    ax3.set_xlim(0.0, 0.7)
    ax3.set_ylim(0.0, 30.0)
    plt.tight_layout()

    # --- Export final fitted parameters for the Universal Model ---
    param_names = ['p_i', 'q_i', 'k_i', 'a_i', 'w_i',
                   'p_f', 'q_f', 'k_f', 'a_f', 'w_f',
                   'z_mid', 'z_width']
    
    export_data = []
    # Use the ordered results to ensure data is sorted by pulse count
    for i in order:
        match = next((r for r in results if r is not None and r[0] == i), None)
        if match is None:
            continue
        
        # Unpack the results
        _, params_i, params_f, z_mid, z_width, _, _, _, _ = match
        
        # Create a dictionary for this row
        row_data = {'Pulse Counts': pulse_counts[i]}
        row_data.update(zip(param_names[:5], params_i))
        row_data.update(zip(param_names[5:10], params_f))
        row_data['z_mid'] = z_mid
        row_data['z_width'] = z_width
        export_data.append(row_data)

    if export_data:
        df_export = pd.DataFrame(export_data)
        export_path = Path(__file__).parent.parent / "fitted_parameters.csv"
        df_export.to_csv(export_path, index=False)
        print(f"Saved fitted parameters to {export_path}")

    plt.show()


if __name__ == "__main__":
    main()


