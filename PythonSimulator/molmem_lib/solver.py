import numpy as np
import numba as nb
import math
import time
from .device_jit import (
    jit_step_predictor_corrector_array,
    jit_commit_state_array,
    jit_predict_correct
)


@nb.njit(fastmath=True, nogil=True, inline='always', cache=True)
def jit_reset_device_states(n_arr, modeState_arr, scaleFactor_arr, T_arr, currentF22_arr, _cached_n_int_arr, n_val, mode_val, scale_val, T_val):
    for i in range(len(n_arr)):
        n_arr[i] = n_val
        modeState_arr[i] = mode_val
        scaleFactor_arr[i] = scale_val
        T_arr[i] = T_val
        currentF22_arr[i] = 0.0
        _cached_n_int_arr[i] = int(n_val)

@nb.njit(fastmath=True, nogil=True, inline='always', cache=True)
def jit_eval_igeq(vCurrent_arr, dev_top_idx, dev_bot_idx, iScale_arr, vScale_arr, f22_arr, iDev_arr, gEq_arr, rC_arr, is_1t1r=False, dev_to_dyn_src=None, dynamic_G_arr=None, leak_g=1e-10):
    for d in range(len(iDev_arr)):
        if is_1t1r and dev_to_dyn_src is not None and dynamic_G_arr is not None:
            src_idx = dev_to_dyn_src[d]
            if src_idx >= 0 and dynamic_G_arr[src_idx] <= leak_g * 10.0:
                iDev_arr[d] = 0.0
                gEq_arr[d] = 1e-25
                continue

        top_i = dev_top_idx[d]
        bot_i = dev_bot_idx[d]
        v_top = vCurrent_arr[top_i] if 0 <= top_i < len(vCurrent_arr) else 0.0
        v_bot = vCurrent_arr[bot_i] if 0 <= bot_i < len(vCurrent_arr) else 0.0
        vDev = v_top - v_bot
        vS = vScale_arr[d] + 1e-12
        rSeries = 2.0 * rC_arr[d]
        
        if f22_arr[d] == 0.0:
            iDev_arr[d] = 0.0
            gEq_arr[d] = 0.0
        else:
            inv_vS = 1.0 / vS
            i_prod = iScale_arr[d] * f22_arr[d]
            g_base = i_prod * inv_vS
            
            if abs(vDev) < 1e-12:
                iDev_arr[d] = 0.0
                gEq_arr[d] = g_base if rSeries == 0.0 else (g_base / (1.0 + g_base * rSeries))
            else:
                v = vDev * inv_vS
                coeff = rSeries * g_base
                if coeff == 0.0:
                    v_clip = -200.0 if v < -200.0 else (200.0 if v > 200.0 else v)
                    sinh_val = math.sinh(v_clip)
                    cosh_val = math.cosh(v_clip)
                    iDev_arr[d] = i_prod * sinh_val
                    gEq_arr[d] = g_base * cosh_val
                else:
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
                    iDev_arr[d] = i_prod * sinh_val
                    gCore = g_base * cosh_val
                    gEq_arr[d] = gCore / (1.0 + gCore * rSeries)


@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_assemble_and_solve_dense(num_nodes, num_devices, dev_top_idx, dev_bot_idx, gEq_arr, iDev_arr, dyn_node_idx, dyn_G, dyn_I, vCurrent_arr, gMat, iRes):
    gMat.fill(0.0)
    iRes.fill(0.0)
    
    for d in range(num_devices):
        top = dev_top_idx[d]
        bot = dev_bot_idx[d]
        gEq = gEq_arr[d]
        iDev = iDev_arr[d]
        
        if top >= 0:
            iRes[top] += iDev
            gMat[top, top] += gEq
        if bot >= 0:
            iRes[bot] -= iDev
            gMat[bot, bot] += gEq
        if top >= 0 and bot >= 0:
            gMat[top, bot] -= gEq
            gMat[bot, top] -= gEq
            
    for n in range(num_nodes):
        gMat[n, n] += 1e-12
        
    for i in range(len(dyn_node_idx)):
        node = dyn_node_idx[i]
        if node >= 0:
            d_G = dyn_G[i]
            gMat[node, node] += d_G
            iRes[node] += vCurrent_arr[node] * d_G - dyn_I[i]
            
    # Native C-loop negation of current residuals avoiding new array allocations
    max_res = 0.0
    for n in range(num_nodes):
        val = abs(iRes[n])
        if val > max_res:
            max_res = val
            
    if max_res < 1e-12:
        return 0.0
        
    if num_nodes == 1:
        deltaV0 = -iRes[0] / gMat[0, 0] if abs(gMat[0, 0]) > 1e-20 else 0.0
        vCurrent_arr[0] += deltaV0
        return abs(deltaV0)
    elif num_nodes == 2:
        det = gMat[0, 0] * gMat[1, 1] - gMat[0, 1] * gMat[1, 0]
        inv_det = 1.0 / det if abs(det) > 1e-30 else 0.0
        deltaV0 = (iRes[1] * gMat[0, 1] - iRes[0] * gMat[1, 1]) * inv_det
        deltaV1 = (iRes[0] * gMat[1, 0] - iRes[1] * gMat[0, 0]) * inv_det
        vCurrent_arr[0] += deltaV0
        vCurrent_arr[1] += deltaV1
        abs_dV0 = abs(deltaV0)
        abs_dV1 = abs(deltaV1)
        return abs_dV0 if abs_dV0 > abs_dV1 else abs_dV1
    elif num_nodes == 3:
        # Analytic 3x3 Cramer's Rule Matrix Inversion bypassing LAPACK/BLAS overhead
        A = gMat[0, 0]; B = gMat[0, 1]; C = gMat[0, 2]
        D = gMat[1, 0]; E = gMat[1, 1]; F = gMat[1, 2]
        G = gMat[2, 0]; H = gMat[2, 1]; I = gMat[2, 2]
        
        c00 = E * I - F * H
        c01 = F * G - D * I
        c02 = D * H - E * G
        det = A * c00 + B * c01 + C * c02
        if abs(det) > 1e-40:
            inv_det = 1.0 / det
            c10 = C * H - B * I
            c11 = A * I - C * G
            c12 = B * G - A * H
            c20 = B * F - C * E
            c21 = C * D - A * F
            c22 = A * E - B * D
            
            r0 = -iRes[0]; r1 = -iRes[1]; r2 = -iRes[2]
            deltaV0 = (r0 * c00 + r1 * c10 + r2 * c20) * inv_det
            deltaV1 = (r0 * c01 + r1 * c11 + r2 * c21) * inv_det
            deltaV2 = (r0 * c02 + r1 * c12 + r2 * c22) * inv_det
            
            vCurrent_arr[0] += deltaV0
            vCurrent_arr[1] += deltaV1
            vCurrent_arr[2] += deltaV2
            
            abs_dV0 = abs(deltaV0)
            abs_dV1 = abs(deltaV1)
            abs_dV2 = abs(deltaV2)
            max_dev = abs_dV0 if abs_dV0 > abs_dV1 else abs_dV1
            return max_dev if max_dev > abs_dV2 else abs_dV2
        
    # In-place Gaussian elimination with partial pivoting (Zero LAPACK heap allocations / 100% thread-safe)
    for i in range(num_nodes):
        max_row = i
        max_val = abs(gMat[i, i])
        for k in range(i + 1, num_nodes):
            val = abs(gMat[k, i])
            if val > max_val:
                max_val = val
                max_row = k
        if max_row != i:
            for col in range(i, num_nodes):
                tmp = gMat[i, col]
                gMat[i, col] = gMat[max_row, col]
                gMat[max_row, col] = tmp
            tmp_b = iRes[i]
            iRes[i] = iRes[max_row]
            iRes[max_row] = tmp_b
            
        pivot = gMat[i, i]
        if abs(pivot) < 1e-30:
            pivot = 1e-30 if pivot >= 0.0 else -1e-30
        inv_pivot = 1.0 / pivot
        
        for k in range(i + 1, num_nodes):
            factor = gMat[k, i] * inv_pivot
            for col in range(i + 1, num_nodes):
                gMat[k, col] -= factor * gMat[i, col]
            iRes[k] -= factor * iRes[i]
            
    maxDev = 0.0
    for i in range(num_nodes - 1, -1, -1):
        sum_ax = 0.0
        for col in range(i + 1, num_nodes):
            sum_ax += gMat[i, col] * iRes[col]
        pivot = gMat[i, i]
        if abs(pivot) < 1e-30:
            pivot = 1e-30 if pivot >= 0.0 else -1e-30
        deltaV_i = (-iRes[i] - sum_ax) / pivot
        iRes[i] = deltaV_i
        val = abs(deltaV_i)
        if val > maxDev:
            maxDev = val
        vCurrent_arr[i] += deltaV_i
        
    return maxDev

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_freeze_1t1r_vdevs(num_devices, dev_to_dyn_src, dyn_g_arr, high_z_g, vDevs):
    threshold = high_z_g * 10.0
    for d in range(num_devices):
        src_idx = dev_to_dyn_src[d]
        if 0 <= src_idx < len(dyn_g_arr):
            if dyn_g_arr[src_idx] <= threshold:
                vDevs[d] = 0.0

@nb.njit(fastmath=True, nogil=True, inline='always', cache=True)
def jit_eval_sources(t, v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx, vCurrent):
    for i in range(len(v_types)):
        node = v_nodes[i]
        if node < 0 or node >= len(vCurrent):
            continue
        stype = v_types[i]
        if stype == 0: # DC
            pass # Already initialized once
        elif stype == 1: # Pulse
            delay = v_pulse_params[i, 3]
            if t < delay:
                vCurrent[node] = 0.0
            else:
                period = v_pulse_params[i, 1]
                max_pulses = v_pulse_params[i, 5]
                if max_pulses >= 0.0 and t >= delay + max_pulses * period - 1e-15:
                    vCurrent[node] = 0.0
                else:
                    tActive = t - delay
                    inv_period = v_pulse_params[i, 4]
                    cycles = tActive * inv_period
                    cycle_idx = int(cycles + 1e-11)
                    tCycle = tActive - cycle_idx * period
                    width = v_pulse_params[i, 2]
                    vCurrent[node] = v_pulse_params[i, 0] if tCycle < width - 1e-13 else 0.0
        elif stype == 2: # PWL
            if v_pwl_mask[i]:
                max_pts_minus_1 = v_pwl_t_matrix.shape[1] - 1
                idx_ptr = v_pwl_cache_idx[i]
                if idx_ptr < max_pts_minus_1:
                    next_t = v_pwl_t_matrix[i, idx_ptr + 1]
                    if next_t >= 0.0 and t >= next_t - 1e-13:
                        while idx_ptr < max_pts_minus_1 and v_pwl_t_matrix[i, idx_ptr + 1] >= 0.0 and t >= v_pwl_t_matrix[i, idx_ptr + 1] - 1e-13:
                            idx_ptr += 1
                        v_pwl_cache_idx[i] = idx_ptr
                v_val = 0.0
                if idx_ptr >= 0 and v_pwl_t_matrix[i, idx_ptr] <= t + 1e-13:
                    v_val = v_pwl_v_matrix[i, idx_ptr]
                vCurrent[node] = v_val

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_eval_dynamic_sources(t, dyn_types, dyn_dc_voltages, dyn_pulse_params, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask, dyn_pwl_cache_idx, dynamic_G, dynamic_I, leak_g):
    for i in range(len(dyn_types)):
        stype = dyn_types[i]
        v_val = 0.0
        if stype == 0: # DC
            v_val = dyn_dc_voltages[i]
        elif stype == 1: # Pulse
            delay = dyn_pulse_params[i, 3]
            if t < delay:
                v_val = 0.0
            else:
                period = dyn_pulse_params[i, 1]
                max_pulses = dyn_pulse_params[i, 5]
                if max_pulses >= 0.0 and t >= delay + max_pulses * period - 1e-15:
                    v_val = 0.0
                else:
                    tActive = t - delay
                    inv_period = dyn_pulse_params[i, 4]
                    cycles = tActive * inv_period
                    cycle_idx = int(cycles + 1e-11)
                    tCycle = tActive - cycle_idx * period
                    width = dyn_pulse_params[i, 2]
                    v_val = dyn_pulse_params[i, 0] if tCycle < width - 1e-13 else 0.0
        elif stype == 2: # PWL
            if dyn_pwl_mask[i]:
                max_pts_minus_1 = dyn_pwl_t_matrix.shape[1] - 1
                idx_ptr = dyn_pwl_cache_idx[i]
                if idx_ptr < max_pts_minus_1:
                    next_t = dyn_pwl_t_matrix[i, idx_ptr + 1]
                    if next_t >= 0.0 and t >= next_t - 1e-13:
                        while idx_ptr < max_pts_minus_1 and dyn_pwl_t_matrix[i, idx_ptr + 1] >= 0.0 and t >= dyn_pwl_t_matrix[i, idx_ptr + 1] - 1e-13:
                            idx_ptr += 1
                        dyn_pwl_cache_idx[i] = idx_ptr
                v_val = dyn_pwl_v_matrix[i, idx_ptr] if (idx_ptr >= 0 and dyn_pwl_t_matrix[i, idx_ptr] <= t + 1e-13) else 1e20
        
        # Set G and I
        is_open = v_val > 1e19
        dynamic_G[i] = leak_g if is_open else 1e5
        dynamic_I[i] = 0.0 if is_open else (v_val * 1e5)

@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_run_loop(
    tEnd, min_dt, tol, maxIter,
    numNodes, num_devices,
    dev_top_idx, dev_bot_idx, dev_top_full_idx, dev_bot_full_idx,
    v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx,
    dyn_types, dyn_dc_voltages, dyn_pulse_params, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask, dyn_pwl_cache_idx,
    dynamic_node_idx, dynamic_G_arr, dynamic_I_arr,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    f22DepScaled_arr, f22PotScaled_arr,
    _cached_n_int_arr, iScale_arr, vScale_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    E_a_arr, R_th_arr, tau_th_arr, gamma_arr, beta_alpha_arr, beta_s_arr, rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
    y_i, y_v, y_fp, y_fd,
    tArr_buffer, vHist_buffer, iHist_buffer, stateHist_buffer, f22Hist_buffer, gHist_buffer, tempHist_buffer, stepTimeHist_buffer, freq_val,
    vCurrent_arr, gEq_arr, iDev_arr, next_states, err_arr, rC_arr, vDevs_buffer,
    master_schedule, is_subsample, recordHistory, leak_g, is_1t1r, dev_to_dyn_src, t_start, hist_idx_start,
    recordStepTiming, enable_progress=True
):
    t = t_start
    dt = 100e-9
    schedule_idx = 0
    hist_idx = hist_idx_start
    last_print_t = t_start
    print_interval = max(tEnd * 0.05, 1e-12)
    
    t_start_step = 0.0
    gMat = np.zeros((numNodes, numNodes), dtype=np.float64)
    iRes = np.zeros(numNodes, dtype=np.float64)
    tol_100 = tol * 0.01
    tol_10 = tol * 0.1
    T_amb = 298.15
    step_count = 0
    time_span = tEnd - t_start
    est_physical_steps = int(time_span / max(1e-15, min_dt)) if time_span > 0.0 else 10000
    max_steps = max(10_000_000, max(est_physical_steps, int(len(master_schedule) * 1000)))
    
    # Fast forward scheduler queue to current t
    while schedule_idx < len(master_schedule) and master_schedule[schedule_idx] <= t + 1e-18:
        schedule_idx += 1
        
    prev_dn_arr = np.zeros(num_devices, dtype=np.float64)
    prev_v_arr = np.zeros(len(vCurrent_arr), dtype=np.float64)
    for d in range(num_devices):
        if cDischarge_arr[d] < 0.0:
            cDischarge_arr[d] = 0.0
    lin_tol_arr = tol * nMax_arr
    inv_tau_th_arr = np.zeros(num_devices, dtype=np.float64)
    inv_vSmooth_arr = np.zeros(num_devices, dtype=np.float64)
    for i in range(num_devices):
        inv_tau_th_arr[i] = 1.0 / tau_th_arr[i] if tau_th_arr[i] != 0.0 else 0.0
        inv_vSmooth_arr[i] = 1.0 / max(1e-6, vSmooth_arr[i])
    has_pwl = False
    for vt in v_types:
        if vt == 2:
            has_pwl = True
            break
            
    # Record initial step 0 (t = t_start) on initial chunk
    jit_eval_sources(t, v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx, vCurrent_arr)
    for node_i in range(len(vCurrent_arr)):
        prev_v_arr[node_i] = vCurrent_arr[node_i]
    if hist_idx_start == 0:
        if recordHistory:
            tArr_buffer[0] = t
            if recordHistory & 1:
                for node_i in range(len(vCurrent_arr)):
                    vHist_buffer[node_i, 0] = vCurrent_arr[node_i]
            if num_devices > 0:
                rec_state = (recordHistory & 4) != 0
                rec_f22 = (recordHistory & 8) != 0
                rec_g = (recordHistory & 16) != 0
                rec_temp = (recordHistory & 32) != 0
                rec_i = (recordHistory & 2) != 0
                if rec_i and numNodes == 0:
                    jit_eval_igeq(vCurrent_arr, dev_top_full_idx, dev_bot_full_idx, currentIScale_arr, currentVScale_arr, currentF22_arr, iDev_arr, gEq_arr, rC_arr, is_1t1r, dev_to_dyn_src, dynamic_G_arr, leak_g)
                if num_devices == 1:
                    if rec_state: stateHist_buffer[0, 0] = n_arr[0]
                    if rec_f22: f22Hist_buffer[0, 0] = currentF22_arr[0]
                    if rec_g: gHist_buffer[0, 0] = (currentIScale_arr[0] * currentF22_arr[0]) / (currentVScale_arr[0] + 1e-12) if currentF22_arr[0] > 0.0 else 0.0
                    if rec_temp: tempHist_buffer[0, 0] = T_arr[0]
                    if rec_i: iHist_buffer[0, 0] = iDev_arr[0]
                else:
                    for d in range(num_devices):
                        if rec_state: stateHist_buffer[d, 0] = n_arr[d]
                        if rec_f22: f22Hist_buffer[d, 0] = currentF22_arr[d]
                        if rec_g: gHist_buffer[d, 0] = (currentIScale_arr[d] * currentF22_arr[d]) / (currentVScale_arr[d] + 1e-12) if currentF22_arr[d] > 0.0 else 0.0
                        if rec_temp: tempHist_buffer[d, 0] = T_arr[d]
                        if rec_i: iHist_buffer[d, 0] = iDev_arr[d]
            hist_idx = 1
        else:
            tArr_buffer[0] = t
            hist_idx = 1
    else:
        hist_idx = 0
            
    while t < tEnd:
        if t >= tEnd - 1e-15:
            break
        step_count += 1
        if step_count > max_steps:
            return -9999999, t
            
        prev_sched_idx = schedule_idx
        while schedule_idx < len(master_schedule) and master_schedule[schedule_idx] <= t + 1e-18:
            schedule_idx += 1
            
        # 1. Update ideal voltage sources first to get current step voltage
        v_constant = True
        if step_count == 1:
            v_constant = False
            jit_eval_sources(t, v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx, vCurrent_arr)
        elif schedule_idx != prev_sched_idx or has_pwl:
            jit_eval_sources(t, v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx, vCurrent_arr)
            if len(vCurrent_arr) > 0:
                if len(vCurrent_arr) == 1:
                    if abs(vCurrent_arr[0] - prev_v_arr[0]) > 1e-6:
                        v_constant = False
                else:
                    for node_i in range(len(vCurrent_arr)):
                        if abs(vCurrent_arr[node_i] - prev_v_arr[node_i]) > 1e-6:
                            v_constant = False
                            break
        
        skipped_subsample = False
                    
        if v_constant and schedule_idx < len(master_schedule) and is_subsample[schedule_idx]:
            if num_devices > 0:
                is_linear = True
                if num_devices == 1:
                    curr_dn = next_states[0, 0] - n_arr[0]
                    if step_count > 1:
                        if abs(curr_dn - prev_dn_arr[0]) > lin_tol_arr[0]:
                            is_linear = False
                    else:
                        top_0 = dev_top_full_idx[0]
                        bot_0 = dev_bot_full_idx[0]
                        v_top_0 = vCurrent_arr[top_0] if 0 <= top_0 < len(vCurrent_arr) else 0.0
                        v_bot_0 = vCurrent_arr[bot_0] if 0 <= bot_0 < len(vCurrent_arr) else 0.0
                        v_app = abs(v_top_0 - v_bot_0)
                        v_th = vTh_arr[0]
                        softplus_val = (v_app - v_th) * inv_vSmooth_arr[0] if v_app > v_th else 0.0
                        k_eff = kappa_arr[0] * math.pow(softplus_val, alpha_arr[0]) if v_app > v_th else 1000.0
                        if k_eff * abs(curr_dn) * dt > lin_tol_arr[0]:
                            is_linear = False
                else:
                    for dev_i in range(num_devices):
                        curr_dn = next_states[dev_i, 0] - n_arr[dev_i]
                        if step_count > 1:
                            if abs(curr_dn - prev_dn_arr[dev_i]) > lin_tol_arr[dev_i]:
                                is_linear = False
                                break
                        else:
                            top_d = dev_top_full_idx[dev_i]
                            bot_d = dev_bot_full_idx[dev_i]
                            v_top_d = vCurrent_arr[top_d] if 0 <= top_d < len(vCurrent_arr) else 0.0
                            v_bot_d = vCurrent_arr[bot_d] if 0 <= bot_d < len(vCurrent_arr) else 0.0
                            v_app = abs(v_top_d - v_bot_d)
                            v_th = vTh_arr[dev_i]
                            softplus_val = (v_app - v_th) * inv_vSmooth_arr[dev_i] if v_app > v_th else 0.0
                            k_eff = kappa_arr[dev_i] * math.pow(softplus_val, alpha_arr[dev_i]) if v_app > v_th else 1000.0
                            if k_eff * abs(curr_dn) * dt > lin_tol_arr[dev_i]:
                                is_linear = False
                                break
                if is_linear:
                    while schedule_idx < len(master_schedule) - 1 and is_subsample[schedule_idx]:
                        schedule_idx += 1
                    skipped_subsample = True
            
        tNextEdge = master_schedule[schedule_idx] if schedule_idx < len(master_schedule) else tEnd
        edge_epsilon = 1e-16
        time_to_edge = tNextEdge - t
        
        if skipped_subsample:
            dt = time_to_edge if time_to_edge > min_dt else min_dt
        else:
            max_dt = time_to_edge if time_to_edge < 1e-3 else 1e-3
            if dt > max_dt:
                dt = max_dt
            
        if t + dt > tEnd + 1e-15:
            dt = tEnd - t
            
        # 2. Evaluate Dynamic Norton Sources
        if len(dyn_types) > 0:
            jit_eval_dynamic_sources(t, dyn_types, dyn_dc_voltages, dyn_pulse_params, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask, dyn_pwl_cache_idx, dynamic_G_arr, dynamic_I_arr, leak_g)
        
        # 3. Newton-Raphson Iteration to find unknown node voltages
        if numNodes > 0:
            for it in range(maxIter):
                jit_eval_igeq(vCurrent_arr, dev_top_full_idx, dev_bot_full_idx, currentIScale_arr, currentVScale_arr, currentF22_arr, iDev_arr, gEq_arr, rC_arr, is_1t1r, dev_to_dyn_src, dynamic_G_arr, leak_g)
                
                # Assemble dense solver and solve
                maxDev = jit_assemble_and_solve_dense(numNodes, num_devices, dev_top_idx, dev_bot_idx, gEq_arr, iDev_arr, dynamic_node_idx, dynamic_G_arr, dynamic_I_arr, vCurrent_arr, gMat, iRes)
                if maxDev < tol:
                    break
                    
        # 4. Device State Kinetics Subsystem
        err = 0.0
        if num_devices > 0:
            vDevs = vDevs_buffer
            if num_devices == 1:
                top_0 = dev_top_full_idx[0]
                bot_0 = dev_bot_full_idx[0]
                v_top_0 = vCurrent_arr[top_0] if 0 <= top_0 < len(vCurrent_arr) else 0.0
                v_bot_0 = vCurrent_arr[bot_0] if 0 <= bot_0 < len(vCurrent_arr) else 0.0
                v_diff = v_top_0 - v_bot_0
                vDevs[0] = v_diff
                max_vdev = abs(v_diff)
            else:
                max_vdev = 0.0
                for i in range(num_devices):
                    top_i = dev_top_full_idx[i]
                    bot_i = dev_bot_full_idx[i]
                    v_top_i = vCurrent_arr[top_i] if 0 <= top_i < len(vCurrent_arr) else 0.0
                    v_bot_i = vCurrent_arr[bot_i] if 0 <= bot_i < len(vCurrent_arr) else 0.0
                    v_diff = v_top_i - v_bot_i
                    vDevs[i] = v_diff
                    v_abs = abs(v_diff)
                    if v_abs > max_vdev:
                        max_vdev = v_abs
            if is_1t1r:
                jit_freeze_1t1r_vdevs(num_devices, dev_to_dyn_src, dynamic_G_arr, leak_g, vDevs)
                max_vdev = 0.0
                for i in range(num_devices):
                    v_abs = abs(vDevs[i])
                    if v_abs > max_vdev:
                        max_vdev = v_abs
                    
            if max_vdev < 1e-6:
                err = 0.0
                needs_decay = False
                for i in range(num_devices):
                    if modeState_arr[i] != 0.0 or abs(T_arr[i] - T_amb) > 1e-4:
                        needs_decay = True
                        break
                if needs_decay:
                    decay_factor = math.exp(-dt * 1000.0)
                    for i in range(num_devices):
                        if modeState_arr[i] != 0.0:
                            modeState_arr[i] *= decay_factor
                        is_pot = (modeState_arr[i] > 0.5)
                        if is_pot:
                            f22 = f22PotScaled_arr[i]
                        else:
                            f22 = f22DepScaled_arr[i] * scaleFactor_arr[i]
                        currentF22_arr[i] = 0.0 if f22 < 0.0 else f22
                        
                        if abs(T_arr[i] - T_amb) >= 1e-4 and E_a_arr[i] > 0.0 and R_th_arr[i] > 0.0:
                            T_arr[i] = T_amb + (T_arr[i] - T_amb) * math.exp(-dt * inv_tau_th_arr[i])
                        else:
                            T_arr[i] = T_amb
                        
                        if gamma_arr[i] != 0.0:
                            currentVScale_arr[i] = vScale_arr[i] / (1.0 + gamma_arr[i] * currentF22_arr[i])
                        else:
                            currentVScale_arr[i] = vScale_arr[i]
            else:
                if num_devices == 1:
                    k_od_0 = k_overdrive_arr[0] if k_overdrive_arr is not None else 0.0
                    od_val_0 = cDischarge_arr[0] * k_od_0
                    od_clamp_0 = 1.0 if od_val_0 > 1.0 else (0.0 if od_val_0 < 0.0 else od_val_0)
                    scale_base_0 = rBottomBlend_arr[0] if (rBottomBlend_arr[0] >= 0.05 and rBottomBlend_arr[0] <= 2.5) else 1.0
                    n_base_0 = 0.5 * nMax_arr[0] * (1.0 - od_clamp_0 * (1.0 - scale_base_0))
                    if modeState_arr[0] > 0.5 and n_arr[0] > 0.5 * nMax_arr[0]:
                        if vDevs[0] > vRefPos_arr[0] and (vRefPos_arr[0] - vTh_arr[0]) > 1e-6:
                            od_inst_0 = (vDevs[0] - vRefPos_arr[0]) / (vRefPos_arr[0] - vTh_arr[0])
                            if od_inst_0 > cDischarge_arr[0]:
                                cDischarge_arr[0] = od_inst_0
                    elif n_arr[0] <= n_base_0:
                        cDischarge_arr[0] = 0.0

                    err, n_d, mode_d, scale_d, T_d = jit_predict_correct(
                        dt, vDevs[0], n_arr[0], modeState_arr[0], scaleFactor_arr[0], T_arr[0],
                        vSmooth_arr[0], nSmooth_arr[0], kappa_arr[0], alpha_arr[0],
                        vTh_arr[0], vBypass_arr[0], vRefPos_arr[0], vRefNeg_arr[0], nMax_arr[0], kDischarge_arr[0],
                        f22DepScaled_arr[0], f22PotScaled_arr[0],
                        currentIScale_arr[0], currentVScale_arr[0], currentF22_arr[0], rC_arr[0],
                        E_a_arr[0], R_th_arr[0], tau_th_arr[0],
                        beta_alpha_arr[0], beta_s_arr[0], gamma_arr[0], vScale_arr[0],
                        rateAsymmetry_arr[0],
                        rTopBlend_arr[0], fDischarge_arr[0], cDischarge_arr[0], rBottomBlend_arr[0], rLatency_arr[0],
                        k_overdrive_arr[0]
                    )
                    next_states[0, 0] = n_d
                    next_states[0, 1] = mode_d
                    next_states[0, 2] = scale_d
                    next_states[0, 3] = T_d
                    err_arr[0] = err
                else:
                    jit_step_predictor_corrector_array(
                        dt, vDevs,
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
                if num_devices == 1:
                    err = err_arr[0]
                else:
                    err = 0.0
                    for dev_i in range(num_devices):
                        if err_arr[dev_i] > err:
                            err = err_arr[dev_i]

                
                # Step Acceptance check
                if err > tol and dt > min_dt:
                    half_dt = dt * 0.5
                    dt = half_dt if half_dt > min_dt else min_dt
                else:
                    if num_devices == 1:
                        prev_dn_arr[0] = next_states[0, 0] - n_arr[0]
                    else:
                        for dev_i in range(num_devices):
                            prev_dn_arr[dev_i] = next_states[dev_i, 0] - n_arr[dev_i]
                    # Commit state
                    jit_commit_state_array(
                        n_arr, modeState_arr, scaleFactor_arr, T_arr,
                        next_states, num_devices,
                        _cached_n_int_arr, iScale_arr, vScale_arr, f22PotScaled_arr, f22DepScaled_arr,
                        currentF22_arr, currentIScale_arr, currentVScale_arr,
                        gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
                        y_i, y_v, y_fp, y_fd,
                        gamma_arr, vDevs,
                        nSmooth_arr, beta_s_arr, vRefPos_arr, vTh_arr
                    )
                    if num_devices == 1:
                        k_od_0 = k_overdrive_arr[0] if k_overdrive_arr is not None else 0.0
                        od_val_0 = cDischarge_arr[0] * k_od_0
                        od_clamp_0 = 1.0 if od_val_0 > 1.0 else (0.0 if od_val_0 < 0.0 else od_val_0)
                        scale_base_0 = rBottomBlend_arr[0] if (rBottomBlend_arr[0] >= 0.05 and rBottomBlend_arr[0] <= 2.5) else 1.0
                        n_base_0 = 0.5 * nMax_arr[0] * (1.0 - od_clamp_0 * (1.0 - scale_base_0))
                        if modeState_arr[0] > 0.5 and n_arr[0] > 0.5 * nMax_arr[0]:
                            if vDevs[0] > vRefPos_arr[0] and (vRefPos_arr[0] - vTh_arr[0]) > 1e-6:
                                od_inst_0 = (vDevs[0] - vRefPos_arr[0]) / (vRefPos_arr[0] - vTh_arr[0])
                                if od_inst_0 > cDischarge_arr[0]:
                                    cDischarge_arr[0] = od_inst_0
                        elif n_arr[0] <= n_base_0:
                            cDischarge_arr[0] = 0.0
                    else:
                        for dev_i in range(num_devices):
                            k_od_i = k_overdrive_arr[dev_i] if k_overdrive_arr is not None else 0.0
                            od_val_i = cDischarge_arr[dev_i] * k_od_i
                            od_clamp_i = 1.0 if od_val_i > 1.0 else (0.0 if od_val_i < 0.0 else od_val_i)
                            scale_base_i = rBottomBlend_arr[dev_i] if (rBottomBlend_arr[dev_i] >= 0.05 and rBottomBlend_arr[dev_i] <= 2.5) else 1.0
                            n_base_i = 0.5 * nMax_arr[dev_i] * (1.0 - od_clamp_i * (1.0 - scale_base_i))
                            if modeState_arr[dev_i] > 0.5 and n_arr[dev_i] > 0.5 * nMax_arr[dev_i]:
                                if vDevs[dev_i] > vRefPos_arr[dev_i] and (vRefPos_arr[dev_i] - vTh_arr[dev_i]) > 1e-6:
                                    od_inst_i = (vDevs[dev_i] - vRefPos_arr[dev_i]) / (vRefPos_arr[dev_i] - vTh_arr[dev_i])
                                    if od_inst_i > cDischarge_arr[dev_i]:
                                        cDischarge_arr[dev_i] = od_inst_i
                            elif n_arr[dev_i] <= n_base_i:
                                cDischarge_arr[dev_i] = 0.0
                
        if not (err > tol and dt > min_dt):
            t += dt
            eps_snap = 2.220446049250313e-16
            diff_edge = tNextEdge - t
            snap_thresh = (t if t > 1.0 else 1.0) * eps_snap * 2.0
            if diff_edge >= 0.0 and diff_edge <= snap_thresh:
                t = tNextEdge
                
            jit_eval_sources(t, v_types, v_nodes, v_dc_voltages, v_pulse_params, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask, v_pwl_cache_idx, vCurrent_arr)
            for node_i in range(len(vCurrent_arr)):
                prev_v_arr[node_i] = vCurrent_arr[node_i]
                
            if num_devices > 0 and numNodes == 0:
                jit_eval_igeq(vCurrent_arr, dev_top_full_idx, dev_bot_full_idx, currentIScale_arr, currentVScale_arr, currentF22_arr, iDev_arr, gEq_arr, rC_arr, is_1t1r, dev_to_dyn_src, dynamic_G_arr, leak_g)
                
            if recordHistory:
                tArr_buffer[hist_idx] = t
                if recordHistory & 1:
                    for node_i in range(len(vCurrent_arr)):
                        vHist_buffer[node_i, hist_idx] = vCurrent_arr[node_i]
                if num_devices > 0:
                    rec_state = (recordHistory & 4) != 0
                    rec_f22 = (recordHistory & 8) != 0
                    rec_g = (recordHistory & 16) != 0
                    rec_temp = (recordHistory & 32) != 0
                    rec_i = (recordHistory & 2) != 0
                    
                    if num_devices == 1:
                        if rec_state:
                            stateHist_buffer[0, hist_idx] = n_arr[0]
                        if rec_f22:
                            f22Hist_buffer[0, hist_idx] = currentF22_arr[0]
                        if rec_g:
                            gHist_buffer[0, hist_idx] = (currentIScale_arr[0] * currentF22_arr[0]) / (currentVScale_arr[0] + 1e-12) if currentF22_arr[0] > 0.0 else 0.0
                        if rec_temp:
                            tempHist_buffer[0, hist_idx] = T_arr[0]
                        if rec_i:
                            iHist_buffer[0, hist_idx] = iDev_arr[0]
                    else:
                        for d in range(num_devices):
                            if rec_state:
                                stateHist_buffer[d, hist_idx] = n_arr[d]
                            if rec_f22:
                                f22Hist_buffer[d, hist_idx] = currentF22_arr[d]
                            if rec_g:
                                gHist_buffer[d, hist_idx] = (currentIScale_arr[d] * currentF22_arr[d]) / (currentVScale_arr[d] + 1e-12) if currentF22_arr[d] > 0.0 else 0.0
                            if rec_temp:
                                tempHist_buffer[d, hist_idx] = T_arr[d]
                            if rec_i:
                                iHist_buffer[d, hist_idx] = iDev_arr[d]
                hist_idx += 1
                
                # Check buffer capacity
                if hist_idx >= len(tArr_buffer):
                    # Return negative of completed steps to signal resize/chunk write in Python wrapper
                    return -hist_idx, t
            else:
                tArr_buffer[0] = t
                for node_i in range(len(vCurrent_arr)):
                    vHist_buffer[node_i, 0] = vCurrent_arr[node_i]
                if num_devices > 0:
                    rec_state = (recordHistory & 4) != 0
                    rec_f22 = (recordHistory & 8) != 0
                    rec_g = (recordHistory & 16) != 0
                    rec_temp = (recordHistory & 32) != 0
                    rec_i = (recordHistory & 2) != 0
                    for d in range(num_devices):
                        if rec_state:
                            stateHist_buffer[d, 0] = n_arr[d]
                        if rec_f22:
                            f22Hist_buffer[d, 0] = currentF22_arr[d]
                        if rec_g:
                            gHist_buffer[d, 0] = (currentIScale_arr[d] * currentF22_arr[d]) / (currentVScale_arr[d] + 1e-12)
                        if rec_temp:
                            tempHist_buffer[d, 0] = T_arr[d]
                        if rec_i:
                            iHist_buffer[d, 0] = iDev_arr[d]
                hist_idx = 1
            
            if err < tol_100:
                dt = 1e-3
            elif err < tol_10:
                dt = min(dt * 2.0, 1e-3)
                
        if recordStepTiming:
            elapsed = 0.0
            with nb.objmode(elapsed='float64'):
                elapsed = time.perf_counter() - t_start_step
            if recordHistory:
                if not (err > tol and dt > min_dt):
                    stepTimeHist_buffer[hist_idx - 1] += elapsed
                else:
                    if hist_idx < len(stepTimeHist_buffer):
                        stepTimeHist_buffer[hist_idx] += elapsed
                
    return hist_idx, t



@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_build_full_master_schedule(unique_edges, is_active, sub_dt_threshold, inv_sub_dt):
    N = len(unique_edges)
    if N <= 1:
        ms = unique_edges.copy()
        is_sub = np.zeros(len(ms), dtype=np.bool_)
        return ms, is_sub
        
    total_pts = N
    num_subs = np.zeros(N - 1, dtype=np.int32)
    for i in range(N - 1):
        dt_seg = unique_edges[i+1] - unique_edges[i]
        if is_active[i] and dt_seg > sub_dt_threshold:
            ns = int(math.ceil(dt_seg * inv_sub_dt))
            num_subs[i] = ns
            if ns > 1:
                total_pts += (ns - 1)
                
    master_schedule = np.empty(total_pts, dtype=np.float64)
    is_subsample = np.empty(total_pts, dtype=np.bool_)
    
    pos = 0
    for i in range(N - 1):
        master_schedule[pos] = unique_edges[i]
        is_subsample[pos] = False
        pos += 1
        
        ns = num_subs[i]
        if ns > 1:
            dt_seg = unique_edges[i+1] - unique_edges[i]
            step = dt_seg / ns
            t_s = unique_edges[i]
            for s in range(1, ns):
                master_schedule[pos] = t_s + s * step
                is_subsample[pos] = True
                pos += 1
                
    master_schedule[pos] = unique_edges[N - 1]
    is_subsample[pos] = False
    
    return master_schedule, is_subsample

@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def jit_cpu_batch_pass_input(
    pulse_widths_2d,
    currentIScale_arr,
    vScale_arr,
    currentF22_arr,
    rC_arr,
    vRead,
    rows,
    cols,
    adc_iMin,
    adc_iMax,
    max_adc_bits,
    maxPulseWidth,
    results
):
    batch_size = pulse_widths_2d.shape[0]
    inv_max_integral = float(rows) / (max_adc_bits * maxPulseWidth)
    adc_range = adc_iMax - adc_iMin if adc_iMax > adc_iMin else 1e-12
    inv_adc_range = 1.0 / adc_range

    # 1. Precompute steady-state cell currents under vRead
    i_cell = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            f22 = currentF22_arr[idx]
            if f22 == 0.0:
                i_cell[r, c] = 0.0
            else:
                vS = vScale_arr[idx] + 1e-12
                inv_vS = 1.0 / vS
                i_prod = currentIScale_arr[idx] * f22
                g_base = i_prod * inv_vS
                rSeries = 2.0 * rC_arr[idx]
                v = vRead * inv_vS
                coeff = rSeries * g_base
                if coeff == 0.0:
                    v_clip = -200.0 if v < -200.0 else (200.0 if v > 200.0 else v)
                    i_cell[r, c] = i_prod * math.sinh(v_clip)
                else:
                    sgn = 1.0 if v >= 0.0 else -1.0
                    abs_v = abs(v)
                    inv_coeff = 1.0 / (coeff + 1e-20)
                    y = abs_v * inv_coeff
                    asinh_y = math.asinh(y)
                    x = sgn * (abs_v if abs_v < asinh_y else asinh_y)
                    for _ in range(8):
                        sinh_x = math.sinh(x)
                        cosh_x = math.cosh(x)
                        fx = x + coeff * sinh_x - v
                        fpx = 1.0 + coeff * cosh_x
                        dx = fx / fpx
                        x -= dx
                        if abs(dx) < 1e-12:
                            break
                    i_cell[r, c] = i_prod * math.sinh(x)

    # 2. Parallel multi-core evaluation across batch vectors
    for b in nb.prange(batch_size):
        # Extract row pulse widths for batch item b
        w_local = np.empty(rows, dtype=np.float64)
        has_active = False
        for r in range(rows):
            pw = pulse_widths_2d[b, r]
            w_local[r] = pw
            if pw > 0.0:
                has_active = True

        if not has_active:
            for c in range(cols):
                results[b, c] = 0.0
            continue

        # In-place insertion sort to find unique sorted pulse edges
        sorted_w = np.empty(rows, dtype=np.float64)
        for r in range(rows):
            sorted_w[r] = w_local[r]

        for i in range(1, rows):
            key = sorted_w[i]
            j = i - 1
            while j >= 0 and sorted_w[j] > key:
                sorted_w[j + 1] = sorted_w[j]
                j -= 1
            sorted_w[j + 1] = key

        # Compact unique positive edges
        unique_edges = np.empty(rows + 1, dtype=np.float64)
        unique_edges[0] = 0.0
        n_edges = 1
        for i in range(rows):
            val = sorted_w[i]
            if val > unique_edges[n_edges - 1] + 1e-15:
                unique_edges[n_edges] = val
                n_edges += 1

        # Integrate digital charge per column across PWM time intervals
        for c in range(cols):
            q_sum = 0.0
            for k in range(1, n_edges):
                dt_k = unique_edges[k] - unique_edges[k - 1]
                t_mid = unique_edges[k]
                # Sum currents from all rows active during this interval
                i_col = 0.0
                for r in range(rows):
                    if w_local[r] >= t_mid - 1e-15:
                        i_col += i_cell[r, c]

                # ADC quantization to discrete digital bit levels
                i_clamped = adc_iMin if i_col < adc_iMin else (adc_iMax if i_col > adc_iMax else i_col)
                d_bits = math.floor((i_clamped - adc_iMin) * inv_adc_range * max_adc_bits + 0.5)
                q_sum += d_bits * dt_k

            results[b, c] = np.float32(q_sum * inv_max_integral)





