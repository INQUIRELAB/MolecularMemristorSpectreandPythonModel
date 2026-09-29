import numba as nb
import math

@nb.njit(fastmath=True, nogil=True, inline='always', cache=True)
def jit_interp_O1(x, yp, x_max=33040.0):
    """A highly optimized O(1) linear interpolator mapping [0, x_max] space n into uniformly spaced arrays."""
    if len(yp) <= 1: return yp[0]
    if math.isnan(x) or x <= 0.0: return yp[0]
    if x >= x_max: return yp[-1]
    
    n_steps = len(yp) - 1
    if n_steps == int(x_max) and x_max > 0.0:
        idx_float = x
    else:
        inv_max = 1.0 / x_max if x_max > 0.0 else 1.0
        idx_float = x * (n_steps * inv_max)
        
    idx = int(idx_float)
    if idx < 0: return yp[0]
    if idx >= n_steps: return yp[-1]
    y0 = yp[idx]
    return y0 + (idx_float - idx) * (yp[idx + 1] - y0)

@nb.njit(fastmath=True, nogil=True, inline='always', cache=True)
def jit_getIGeq(vApplied, currentIScale, currentVScale, currentF22, rC):
    vS = currentVScale + 1e-12
    rSeries = 2.0 * rC  # R_top + R_bot
    
    if currentF22 == 0.0:
        return 0.0, 0.0
        
    inv_vS = 1.0 / vS
    i_prod = currentIScale * currentF22
    g_base = i_prod * inv_vS
    
    if abs(vApplied) < 1e-12:
        gEq = g_base if rSeries == 0.0 else (g_base / (1.0 + g_base * rSeries))
        return 0.0, gEq
        
    v = vApplied * inv_vS
    coeff = rSeries * g_base
    
    # Analytical fast-path for ideal zero series contact resistance
    if coeff == 0.0:
        v_clip = -200.0 if v < -200.0 else (200.0 if v > 200.0 else v)
        sinh_val = math.sinh(v_clip)
        cosh_val = math.cosh(v_clip)
        currentI = i_prod * sinh_val
        gCore = g_base * cosh_val
        return currentI, gCore

    # Solve for V_core using Newton-Raphson method
    sgn = 1.0 if v >= 0.0 else -1.0
    abs_v = abs(v)
    inv_coeff = 1.0 / (coeff + 1e-20)
    y = abs_v * inv_coeff
    asinh_y = math.asinh(y)
    x = sgn * (abs_v if abs_v < asinh_y else asinh_y)
    
    sinh_x = 0.0
    cosh_x = 1.0
    for _ in range(8):
        x_clip = x
        if x_clip > 20.0: x_clip = 20.0
        elif x_clip < -20.0: x_clip = -20.0
        exp_x = math.exp(x_clip)
        inv_exp_x = 1.0 / exp_x
        sinh_x = 0.5 * (exp_x - inv_exp_x)
        cosh_x = 0.5 * (exp_x + inv_exp_x)
        f = x + coeff * sinh_x - v
        df = 1.0 + coeff * cosh_x
        dx = f / df
        x -= dx
        if abs(dx) < 1e-12:
            break
        
    if abs(x) <= 20.0:
        sinh_val = sinh_x
        cosh_val = cosh_x
    else:
        if x > 200.0: x = 200.0
        elif x < -200.0: x = -200.0
        sinh_val = math.sinh(x)
        cosh_val = math.cosh(x)
    currentI = i_prod * sinh_val
    gCore = g_base * cosh_val
    
    # Series resistance: G_total = G_core / (1 + G_core * R_series)
    gEq = gCore / (1.0 + gCore * rSeries)
    return currentI, gEq


@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_updateState(
    dt, vApplied, n, modeState, scaleFactor, T,
    vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax, kDischarge,
    f22DepScaled, f22PotScaled, currentIScale, currentVScale, currentF22, rC,
    E_a, R_th, tau_th,
    couplings, vScale,
    rateAsymmetry, rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
    k_overdrive=0.0
):
    beta_alpha, beta_s, gamma = couplings
    T_amb = 298.15
    inv_tau_th = 1.0 / tau_th if (E_a > 0.0 and R_th > 0.0 and tau_th != 0.0) else 0.0
    inv_vSmooth = 1.0 / vSmooth if vSmooth != 0.0 else 0.0
    
    # Fast path for negligible applied voltage (thermal cooling decay only)
    if abs(vApplied) < 1e-6:
        if modeState == 0.0:
            modeState_new = 0.0
        else:
            decay_factor = math.exp(-dt * 1000.0)
            modeState_new = modeState * decay_factor
            
        if E_a > 0.0 and R_th > 0.0 and abs(T - T_amb) >= 1e-4:
            T_new = T_amb + (T - T_amb) * math.exp(-dt * inv_tau_th)
        else:
            T_new = T_amb
        return n, modeState_new, scaleFactor, T_new

    vInt = vApplied
    if vInt > vRefPos and (beta_alpha != 0.0 or beta_s != 0.0):
        d_val_pos = vRefPos - vTh + 1e-6
        delta_v = (vInt - vRefPos) / (d_val_pos if d_val_pos > 1e-15 else 1e-15)
        tanh_delta_v = math.tanh(delta_v)
        alpha_eff = alpha + beta_alpha * tanh_delta_v
        if alpha_eff < 0.1:
            alpha_eff = 0.1
        n_smooth_eff = nSmooth * (1.0 + beta_s * tanh_delta_v)
        if n_smooth_eff < 1.0:
            n_smooth_eff = 1.0
    else:
        alpha_eff = alpha
        n_smooth_eff = nSmooth

    # Thermal analytical update and Arrhenius scaling factor calculation
    if E_a > 0.0 and R_th > 0.0:
        if gamma != 0.0:
            currentVScale_eff = vScale / (1.0 + gamma * currentF22)
        else:
            currentVScale_eff = vScale
        currentI, _ = jit_getIGeq(vApplied, currentIScale, currentVScale_eff, currentF22, rC)
        P_diss = abs(vApplied * currentI)
        
        if abs(T - T_amb) < 1e-4 and P_diss < 1e-9:
            T_new = T_amb
            thermal_factor = 1.0
        else:
            T_steady = T_amb + P_diss * R_th
            T_new = T_steady + (T - T_steady) * math.exp(-dt * inv_tau_th)
            if T_new < T_amb:
                T_new = T_amb
            elif T_new > 1000.0:
                T_new = 1000.0
                
            T_avg = 0.5 * (T + T_new)
            thermal_factor = math.exp((E_a * 11604.51812063) * (0.00335401643468 - 1.0 / T_avg))
    else:
        T_new = T_amb
        thermal_factor = 1.0
	
    # --- 1. Rate Equation (dn/dt) ---
    tanh_arg = vInt * inv_vSmooth
    if tanh_arg > 20.0:
        vTanh = 1.0
    elif tanh_arg < -20.0:
        vTanh = -1.0
    else:
        vTanh = math.tanh(tanh_arg)
    
    arg = (abs(vInt) - vTh) * inv_vSmooth
    if arg > 20.0:
        softplus = arg
    elif arg < -20.0:
        softplus = 0.0
    else:
        softplus = math.log1p(math.exp(arg))
        
    direction = vTanh
    
    if softplus == 0.0:
        dnDt = 0.0
    else:
        # Direction-aware boundary clamping:
        # Clamps ONLY when moving into a boundary to prevent premature turn-on delays and preserve peak saturation
        if vTanh > 0.0:
            # Potentiation pushing toward nMax (saturation region governed by n_smooth_eff)
            w_sat = max(50.0, 1.5 * n_smooth_eff)
            dist_hi = nMax - n
            if dist_hi >= w_sat:
                clampFactor = 1.0
            elif dist_hi <= 0.0:
                clampFactor = 0.0
            else:
                z_hi = dist_hi / w_sat
                clampFactor = z_hi * z_hi * (3.0 - 2.0 * z_hi)
        elif vTanh < 0.0:
            # Depression pushing toward 0 (minimal numerical guard independent of nSmooth)
            w_zero = 30.0
            if n >= w_zero:
                clampFactor = 1.0
            elif n <= 0.0:
                clampFactor = 0.0
            else:
                z_low = n / w_zero
                clampFactor = z_low * z_low * (3.0 - 2.0 * z_low)
        else:
            clampFactor = 1.0

        kappa_clamp = kappa * clampFactor
        if vTanh == 1.0:
            d_val = vRefPos - vTh + 1e-6
            base = (vSmooth * softplus) / (d_val if d_val > 1e-15 else 1e-15)
            dnDt = math.pow(base, alpha_eff) * kappa_clamp
        elif vTanh == -1.0:
            d_val = vRefNeg - vTh + 1e-6
            base = (vSmooth * softplus) / (d_val if d_val > 1e-15 else 1e-15)
            dnDt = -(math.pow(base, alpha_eff) * (kappa_clamp * rateAsymmetry))
        else:
            sPol = 0.5 * (1.0 + vTanh)
            vrefEff = sPol * vRefPos + (1.0 - sPol) * vRefNeg
            d_val = vrefEff - vTh + 1e-6
            base = (vSmooth * softplus * abs(vTanh)) / (d_val if d_val > 1e-15 else 1e-15)
            rateMag = math.pow(base, alpha_eff)
            if vTanh > 0.0:
                dnDt = (rateMag * vTanh) * kappa_clamp
            else:
                dnDt = (rateMag * vTanh) * (kappa_clamp * rateAsymmetry)
    
    if (modeState < 0.45 or vTanh < 0.0) and n > 500.0:
        n_base_nominal = 0.5 * nMax
        od_ratio = cDischarge if cDischarge > 0.0 else 0.0
        overdrive = (od_ratio * k_overdrive) if (od_ratio > 0.0 and k_overdrive > 0.0) else 0.0
        od_clamp = 1.0 if overdrive > 1.0 else (0.0 if overdrive < 0.0 else overdrive)

        scale_base = rBottomBlend if (rBottomBlend >= 0.05 and rBottomBlend <= 2.5) else 1.0
        n_base = n_base_nominal * (1.0 - od_clamp * (1.0 - scale_base))
        w_trans = 150.0

        # Base Landing: C1 Hermite smoothstep at n_base
        u_base = (n - (n_base - w_trans)) / (2.0 * w_trans)
        if u_base >= 1.0:
            s_base = 1.0
        elif u_base <= 0.0:
            s_base = 0.0
        else:
            s_base = u_base * u_base * (3.0 - 2.0 * u_base)

        if s_base > 0.0:
            # Turnaround Normalization:
            n_turnaround = nMax
            excursion = n_turnaround - n_base
            if excursion < 100.0:
                excursion = 100.0

            u = (n - n_base) / excursion
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)

            # Apex Turnaround Hermite Smoothstep:
            # Prevents derivative cusp by smoothly ramping the snapback boost across w_top states below peak
            scale_top = rTopBlend if (rTopBlend >= 0.001 and rTopBlend <= 0.5) else 0.05
            w_top = scale_top * excursion
            if w_top < 30.0:
                w_top = 30.0
            dist_from_peak = n_turnaround - n
            if dist_from_peak >= w_top:
                s_top = 1.0
            elif dist_from_peak <= 0.0:
                s_top = 0.0
            else:
                z_top = dist_from_peak / w_top
                s_top = z_top * z_top * (3.0 - 2.0 * z_top)

            # Continuous Monotonic Relaxation with Voltage Overdrive Latency Gating:
            r_lat = rLatency if (rLatency >= 0.0 and rLatency < 0.999) else 0.5
            r_lat_eff = r_lat * (1.0 - od_clamp) if (1.0 - od_clamp) > 0.0 else 0.0
            k_amp = kDischarge if kDischarge > 0.0 else 0.0
            p_decay = 1.0 / fDischarge if (fDischarge >= 0.05 and fDischarge <= 0.95) else 2.5

            u_decay = math.pow(u, p_decay) if u > 0.0 else 0.0
            eff_rate_factor = (1.0 - r_lat_eff) + (r_lat_eff + k_amp * s_top) * u_decay

            # Smooth landing at n_base into 100% baseline linear depression
            eff_mult = 1.0 - s_base * (1.0 - eff_rate_factor)
            dnDt = dnDt * eff_mult


    # Scale switching rate dynamically with thermal factor
    dnDt_scaled = dnDt * thermal_factor
    n_cand = n + dnDt_scaled * dt
    n_new = 0.0 if n_cand < 0.0 else (nMax if n_cand > nMax else n_cand)
    
    # --- 3. Direction Latch (Hysteresis with Voltage Overdrive Memory) ---
    if abs(vInt) <= vTh:
        if modeState == 0.0:
            modeState_new = 0.0
        else:
            decay_factor = math.exp(-dt * 1000.0)
            modeState_new = modeState * decay_factor
    else:
        if vInt > vTh:
            targetMode = 1.0
        else:
            targetMode = 0.0
        vEffTargetMode = targetMode * 0.9999990000009999
        if dt >= 20e-9:
            modeState_new = vEffTargetMode
        else:
            exp_arg_mode = -dt * 1.000001e9
            exp_factor_mode = 0.0 if exp_arg_mode < -20.0 else math.exp(exp_arg_mode)
            modeState_new = vEffTargetMode + (modeState - vEffTargetMode) * exp_factor_mode
    
    # --- 5. Continuity Scale Factor ---
    scaleFactor_new = scaleFactor
    if modeState_new > 0.6 and vTanh > 0.0:
        denomScale = f22DepScaled if f22DepScaled > 1e-6 else 1e-6
        raw_ratio = f22PotScaled / denomScale
        ratio = 0.0 if raw_ratio < 0.0 else (100.0 if raw_ratio > 100.0 else raw_ratio)
        if dt >= 2e-9:
            scaleFactor_new = ratio
        else:
            exp_arg_scale = -dt * 1e10
            exp_factor_scale = 0.0 if exp_arg_scale < -20.0 else math.exp(exp_arg_scale)
            scaleFactor_new = ratio + (scaleFactor - ratio) * exp_factor_scale
        
    if not math.isfinite(scaleFactor_new):
         scaleFactor_new = 1.0
         
    return n_new, modeState_new, scaleFactor_new, T_new

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_predict_correct(
    dt, vApplied, n, modeState, scaleFactor, T,
    vSmooth, nSmooth, kappa, alpha, vTh, vBypass, vRefPos, vRefNeg, nMax, kDischarge,
    f22DepScaled, f22PotScaled, currentIScale, currentVScale, currentF22, rC,
    E_a, R_th, tau_th,
    beta_alpha, beta_s, gamma, vScale,
    rateAsymmetry, rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
    k_overdrive=0.0
):
    T_amb = 298.15
    inv_tau_th = 1.0 / tau_th if (E_a > 0.0 and R_th > 0.0 and tau_th != 0.0) else 0.0
    
    # Fast path for negligible applied voltage
    if abs(vApplied) < 1e-6:
        if modeState == 0.0:
            modeState_new = 0.0
        else:
            decay_factor = math.exp(-dt * 1000.0)
            modeState_new = modeState * decay_factor
            
        if E_a > 0.0 and R_th > 0.0 and abs(T - T_amb) >= 1e-4:
            T_new = T_amb + (T - T_amb) * math.exp(-dt * inv_tau_th)
        else:
            T_new = T_amb
        return 0.0, n, modeState_new, scaleFactor, T_new

    # Behaviorally safe sub-threshold early exit
    if abs(vApplied) < vBypass:
        n_base_nominal = 0.5 * nMax
        od_ratio = cDischarge if cDischarge > 0.0 else 0.0
        overdrive = (od_ratio * k_overdrive) if (od_ratio > 0.0 and k_overdrive > 0.0) else 0.0
        od_clamp = 1.0 if overdrive > 1.0 else (0.0 if overdrive < 0.0 else overdrive)
        scale_base = rBottomBlend if (rBottomBlend >= 0.05 and rBottomBlend <= 2.5) else 1.0
        n_base = n_base_nominal * (1.0 - od_clamp * (1.0 - scale_base))
        if (modeState < 1e-9 and 
            abs(T - T_amb) < 1e-4 and 
            (n <= n_base or kDischarge == 0.0)):
            
            if modeState > 0.0:
                modeState_new = modeState * math.exp(-dt * 1000.0)
            else:
                modeState_new = 0.0
                
            if E_a > 0.0 and R_th > 0.0 and abs(T - T_amb) >= 1e-4:
                T_new = T_amb + (T - T_amb) * math.exp(-dt * inv_tau_th)
            else:
                T_new = T_amb
            return 0.0, n, modeState_new, scaleFactor, T_new

    # Fast path for saturated boundaries under driving voltages (avoids expensive transcendental math)
    if (n >= nMax and vApplied >= vTh) or (n <= 0.0 and vApplied <= -vTh):
        if E_a > 0.0 and R_th > 0.0:
            if gamma != 0.0:
                currentVScale_eff = vScale / (1.0 + gamma * currentF22)
            else:
                currentVScale_eff = vScale
            currentI, _ = jit_getIGeq(vApplied, currentIScale, currentVScale_eff, currentF22, rC)
            P_diss = abs(vApplied * currentI)
            T_steady = T_amb + P_diss * R_th
            T_new = T_steady + (T - T_steady) * math.exp(-dt * inv_tau_th)
            if T_new < T_amb: T_new = T_amb
            elif T_new > 1000.0: T_new = 1000.0
        else:
            T_new = T_amb
            
        if vApplied > 0.0:
            modeState_new = 1.0
            denomScale = f22DepScaled if f22DepScaled > 1e-6 else 1e-6
            ratio = f22PotScaled / denomScale
            scaleFactor_new = ratio
            n_new = nMax
        else:
            modeState_new = 0.0
            scaleFactor_new = scaleFactor
            n_new = 0.0
            
        return 0.0, n_new, modeState_new, scaleFactor_new, T_new

    # Take one massive step
    couplings = (beta_alpha, beta_s, gamma)
    n_single, mode_single, scale_single, T_single = jit_updateState(
        dt, vApplied, n, modeState, scaleFactor, T,
        vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax, kDischarge, f22DepScaled, f22PotScaled,
        currentIScale, currentVScale, currentF22, rC, E_a, R_th, tau_th,
        couplings, vScale,
        rateAsymmetry, rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
        k_overdrive
    )
    
    if n_single == n:
        return 0.0, n, mode_single, scale_single, T_single
        
    dt_half = 0.5 * dt
    # Take two half steps
    n_half, mode_half, scale_half, T_half = jit_updateState(
        dt_half, vApplied, n, modeState, scaleFactor, T,
        vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax, kDischarge, f22DepScaled, f22PotScaled,
        currentIScale, currentVScale, currentF22, rC, E_a, R_th, tau_th,
        couplings, vScale,
        rateAsymmetry, rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
        k_overdrive
    )
    if n_half == n:
        n_double = n
        mode_double = mode_half
        scale_double = scale_half
        T_double = T_half
    else:
        n_double, mode_double, scale_double, T_double = jit_updateState(
            dt_half, vApplied, n_half, mode_half, scale_half, T_half,
            vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax, kDischarge, f22DepScaled, f22PotScaled,
            currentIScale, currentVScale, currentF22, rC, E_a, R_th, tau_th,
            couplings, vScale,
            rateAsymmetry, rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
            k_overdrive
        )
    
    # Calculate Max Relative Error
    diff_n = abs(n_single - n_double)
    if diff_n < 1e-12:
        err_n = 0.0
    else:
        norm_floor = nMax * 1e-3
        norm_den = n_double if n_double > norm_floor else norm_floor
        err_n = diff_n / norm_den
    if E_a > 0.0 and R_th > 0.0:
        err_T = abs(T_single - T_double) / T_double
        err = err_n if err_n > err_T else err_T
    else:
        err = err_n
    
    return err, n_double, mode_double, scale_double, T_double


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def _jit_step_predictor_corrector_array_parallel(
    dt, vApplied_arr,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    f22DepScaled_arr, f22PotScaled_arr,
    currentIScale_arr, currentVScale_arr, currentF22_arr, rC_arr,
    E_a_arr, R_th_arr, tau_th_arr,
    next_states, err_arr, num_devices,
    beta_alpha_arr, beta_s_arr, gamma_arr, vScale_arr,
    rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr,
    k_overdrive_arr=None
):
    for i in nb.prange(num_devices):
        k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
        n_base_nom = 0.5 * nMax_arr[i]
        od_ratio = cDischarge_arr[i] if cDischarge_arr[i] > 0.0 else 0.0
        od_clamp = min(1.0, max(0.0, od_ratio * k_od))
        scale_base = rBottomBlend_arr[i] if (rBottomBlend_arr[i] >= 0.05 and rBottomBlend_arr[i] <= 2.5) else 1.0
        n_base = n_base_nom * (1.0 - od_clamp * (1.0 - scale_base))
        if modeState_arr[i] > 0.5 and n_arr[i] > n_base_nom:
            if vApplied_arr[i] > vRefPos_arr[i] and (vRefPos_arr[i] - vTh_arr[i]) > 1e-6:
                od_inst = (vApplied_arr[i] - vRefPos_arr[i]) / (vRefPos_arr[i] - vTh_arr[i])
                if od_inst > cDischarge_arr[i]:
                    cDischarge_arr[i] = od_inst
        elif n_arr[i] <= n_base:
            cDischarge_arr[i] = 0.0

        k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
        err, n_d, mode_d, scale_d, T_d = jit_predict_correct(
            dt, vApplied_arr[i], n_arr[i], modeState_arr[i], scaleFactor_arr[i], T_arr[i],
            vSmooth_arr[i], nSmooth_arr[i], kappa_arr[i], alpha_arr[i],
            vTh_arr[i], vBypass_arr[i], vRefPos_arr[i], vRefNeg_arr[i], nMax_arr[i], kDischarge_arr[i],
            f22DepScaled_arr[i], f22PotScaled_arr[i],
            currentIScale_arr[i], currentVScale_arr[i], currentF22_arr[i], rC_arr[i],
            E_a_arr[i], R_th_arr[i], tau_th_arr[i],
            beta_alpha_arr[i], beta_s_arr[i], gamma_arr[i], vScale_arr[i],
            rateAsymmetry_arr[i], rTopBlend_arr[i], fDischarge_arr[i], cDischarge_arr[i], rBottomBlend_arr[i], rLatency_arr[i],
            k_od
        )
        next_states[i, 0] = n_d
        next_states[i, 1] = mode_d
        next_states[i, 2] = scale_d
        next_states[i, 3] = T_d
        err_arr[i] = err

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_step_predictor_corrector_array(
    dt, vApplied_arr,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    f22DepScaled_arr, f22PotScaled_arr,
    currentIScale_arr, currentVScale_arr, currentF22_arr, rC_arr,
    E_a_arr, R_th_arr, tau_th_arr,
    next_states, err_arr, num_devices,
    beta_alpha_arr, beta_s_arr, gamma_arr, vScale_arr,
    rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr,
    k_overdrive_arr=None
):
    if num_devices < 64:
        for i in range(num_devices):
            k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
            n_base_nom = 0.5 * nMax_arr[i]
            od_ratio = cDischarge_arr[i] if cDischarge_arr[i] > 0.0 else 0.0
            od_clamp = min(1.0, max(0.0, od_ratio * k_od))
            scale_base = rBottomBlend_arr[i] if (rBottomBlend_arr[i] >= 0.05 and rBottomBlend_arr[i] <= 2.5) else 1.0
            n_base = n_base_nom * (1.0 - od_clamp * (1.0 - scale_base))
            if modeState_arr[i] > 0.5 and n_arr[i] > n_base_nom:
                if vApplied_arr[i] > vRefPos_arr[i] and (vRefPos_arr[i] - vTh_arr[i]) > 1e-6:
                    od_inst = (vApplied_arr[i] - vRefPos_arr[i]) / (vRefPos_arr[i] - vTh_arr[i])
                    if od_inst > cDischarge_arr[i]:
                        cDischarge_arr[i] = od_inst
            elif n_arr[i] <= n_base:
                cDischarge_arr[i] = 0.0

            k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
            err, n_d, mode_d, scale_d, T_d = jit_predict_correct(
                dt, vApplied_arr[i], n_arr[i], modeState_arr[i], scaleFactor_arr[i], T_arr[i],
                vSmooth_arr[i], nSmooth_arr[i], kappa_arr[i], alpha_arr[i],
                vTh_arr[i], vBypass_arr[i], vRefPos_arr[i], vRefNeg_arr[i], nMax_arr[i], kDischarge_arr[i],
                f22DepScaled_arr[i], f22PotScaled_arr[i],
                currentIScale_arr[i], currentVScale_arr[i], currentF22_arr[i], rC_arr[i],
                E_a_arr[i], R_th_arr[i], tau_th_arr[i],
                beta_alpha_arr[i], beta_s_arr[i], gamma_arr[i], vScale_arr[i],
                rateAsymmetry_arr[i], rTopBlend_arr[i], fDischarge_arr[i], cDischarge_arr[i], rBottomBlend_arr[i], rLatency_arr[i],
                k_od
            )
            next_states[i, 0] = n_d
            next_states[i, 1] = mode_d
            next_states[i, 2] = scale_d
            next_states[i, 3] = T_d
            err_arr[i] = err
    else:
        _jit_step_predictor_corrector_array_parallel(
            dt, vApplied_arr,
            n_arr, modeState_arr, scaleFactor_arr, T_arr,
            vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
            vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
            f22DepScaled_arr, f22PotScaled_arr,
            currentIScale_arr, currentVScale_arr, currentF22_arr, rC_arr,
            E_a_arr, R_th_arr, tau_th_arr,
            next_states, err_arr, num_devices,
            beta_alpha_arr, beta_s_arr, gamma_arr, vScale_arr,
            rateAsymmetry_arr,
            rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr,
            k_overdrive_arr
        )

@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def _jit_commit_state_array_parallel(
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    next_states, num_devices,
    _cached_n_int_arr, iScale_arr, vScale_arr, f22PotScaled_arr, f22DepScaled_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    y_i, y_v, y_fp, y_fd,
    gamma_arr, vApplied_arr,
    nSmooth_arr=None, beta_s_arr=None, vRefPos_arr=None, vTh_arr=None
):
    for i in nb.prange(num_devices):
        n_raw = next_states[i, 0]
        n_val = 0.0 if (math.isnan(n_raw) or n_raw < 0.0) else (33040.0 if n_raw > 33040.0 else n_raw)
        modeState_val = next_states[i, 1]
        scaleFactor_val = next_states[i, 2]
        T_val = next_states[i, 3]
        
        state_changed = abs(n_val - _cached_n_int_arr[i]) > 0.01
        mode_changed = (modeState_arr[i] > 0.5) != (modeState_val > 0.5)
        
        n_arr[i] = n_val
        modeState_arr[i] = modeState_val
        scaleFactor_arr[i] = scaleFactor_val
        T_arr[i] = T_val
        
        is_pot = (modeState_val > 0.5)
        if state_changed or mode_changed:
            _cached_n_int_arr[i] = int(n_val)
            iScale_arr[i] = jit_interp_O1(n_val, y_i)
            vScale_arr[i] = jit_interp_O1(n_val, y_v)
            
            f22DepRaw = jit_interp_O1(33040.0 - n_val, y_fd, 33040.0)
            dep_val = (1.0 - f22DepRaw - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
            f22DepScaled_arr[i] = 0.0 if dep_val < 0.0 else dep_val
            if is_pot:
                f22PotRaw = jit_interp_O1(n_val, y_fp)
                pot_val = (f22PotRaw - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
                if pot_val < 0.0:
                    pot_val = 0.0
                elif n_val > 16520.0:
                    u_sat = (n_val - 16520.0) / 16520.0
                    if u_sat > 1.0:
                        u_sat = 1.0
                    n_sm_eff = nSmooth_arr[i] if nSmooth_arr is not None else 150.0
                    if beta_s_arr is not None and vRefPos_arr is not None and vTh_arr is not None and vApplied_arr is not None:
                        v_app = abs(vApplied_arr[i])
                        v_ref = vRefPos_arr[i]
                        v_t = vTh_arr[i]
                        if v_app > v_ref and beta_s_arr[i] != 0.0:
                            d_v = v_ref - v_t + 1e-6
                            del_v = (v_app - v_ref) / (d_v if d_v > 1e-15 else 1e-15)
                            n_sm_eff = n_sm_eff * (1.0 + beta_s_arr[i] * math.tanh(del_v))
                            if n_sm_eff < 1.0: n_sm_eff = 1.0
                            elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                    pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                f22PotScaled_arr[i] = pot_val
            
            currentIScale_arr[i] = iScale_arr[i] * gScale_arr[i]
            
        if is_pot:
            f22 = f22PotScaled_arr[i]
        else:
            f22 = f22DepScaled_arr[i] * scaleFactor_arr[i]
            
        currentF22_arr[i] = 0.0 if f22 < 0.0 else f22
        if gamma_arr[i] != 0.0:
            currentVScale_arr[i] = vScale_arr[i] / (1.0 + gamma_arr[i] * currentF22_arr[i])
        else:
            currentVScale_arr[i] = vScale_arr[i]

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_commit_state_array(
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    next_states, num_devices,
    _cached_n_int_arr, iScale_arr, vScale_arr, f22PotScaled_arr, f22DepScaled_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    y_i, y_v, y_fp, y_fd,
    gamma_arr, vApplied_arr,
    nSmooth_arr=None, beta_s_arr=None, vRefPos_arr=None, vTh_arr=None
):
    if num_devices < 64:
        for i in range(num_devices):
            n_raw = next_states[i, 0]
            n_val = 0.0 if (math.isnan(n_raw) or n_raw < 0.0) else (33040.0 if n_raw > 33040.0 else n_raw)
            modeState_val = next_states[i, 1]
            scaleFactor_val = next_states[i, 2]
            T_val = next_states[i, 3]
            
            state_changed = abs(n_val - _cached_n_int_arr[i]) > 0.01
            mode_changed = (modeState_arr[i] > 0.5) != (modeState_val > 0.5)
            
            n_arr[i] = n_val
            modeState_arr[i] = modeState_val
            scaleFactor_arr[i] = scaleFactor_val
            T_arr[i] = T_val
            
            is_pot = (modeState_val > 0.5)
            if state_changed or mode_changed:
                _cached_n_int_arr[i] = int(n_val)
                iScale_arr[i] = jit_interp_O1(n_val, y_i)
                vScale_arr[i] = jit_interp_O1(n_val, y_v)
                
                f22DepRaw = jit_interp_O1(33040.0 - n_val, y_fd, 33040.0)
                dep_val = (1.0 - f22DepRaw - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
                f22DepScaled_arr[i] = 0.0 if dep_val < 0.0 else dep_val
                if is_pot:
                    f22PotRaw = jit_interp_O1(n_val, y_fp)
                    pot_val = (f22PotRaw - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
                    if pot_val < 0.0:
                        pot_val = 0.0
                    elif n_val > 16520.0:
                        u_sat = (n_val - 16520.0) / 16520.0
                        if u_sat > 1.0:
                            u_sat = 1.0
                        n_sm_eff = nSmooth_arr[i] if nSmooth_arr is not None else 150.0
                        if beta_s_arr is not None and vRefPos_arr is not None and vTh_arr is not None and vApplied_arr is not None:
                            v_app = abs(vApplied_arr[i])
                            v_ref = vRefPos_arr[i]
                            v_t = vTh_arr[i]
                            if v_app > v_ref and beta_s_arr[i] != 0.0:
                                d_v = v_ref - v_t + 1e-6
                                del_v = (v_app - v_ref) / (d_v if d_v > 1e-15 else 1e-15)
                                n_sm_eff = n_sm_eff * (1.0 + beta_s_arr[i] * math.tanh(del_v))
                                if n_sm_eff < 1.0: n_sm_eff = 1.0
                                elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                        pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                    f22PotScaled_arr[i] = pot_val
                
                currentIScale_arr[i] = iScale_arr[i] * gScale_arr[i]
                
            if is_pot:
                f22 = f22PotScaled_arr[i]
            else:
                f22 = f22DepScaled_arr[i] * scaleFactor_arr[i]
                
            currentF22_arr[i] = 0.0 if f22 < 0.0 else f22
            if gamma_arr[i] != 0.0:
                currentVScale_arr[i] = vScale_arr[i] / (1.0 + gamma_arr[i] * currentF22_arr[i])
            else:
                currentVScale_arr[i] = vScale_arr[i]
    else:
        _jit_commit_state_array_parallel(
            n_arr, modeState_arr, scaleFactor_arr, T_arr,
            next_states, num_devices,
            _cached_n_int_arr, iScale_arr, vScale_arr, f22PotScaled_arr, f22DepScaled_arr,
            currentF22_arr, currentIScale_arr, currentVScale_arr,
            gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
            y_i, y_v, y_fp, y_fd,
            gamma_arr, vApplied_arr,
            nSmooth_arr, beta_s_arr, vRefPos_arr, vTh_arr
        )



