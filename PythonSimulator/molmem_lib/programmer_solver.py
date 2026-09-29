import threading
import math
import numpy as np
import numba as nb
from .device_jit import jit_predict_correct, jit_updateState, jit_interp_O1, jit_getIGeq

class DualEndedWorkQueue:
    """
    Lock-free converging task dispatcher for hybrid CPU + GPU crossbar computation
    with speculative racing fall-through.
    CPU claims forward chunks from the front (head -> tail) at native CPU multi-core speed.
    GPU claims backward slices from the back (tail -> head) in hardware-saturating batches.
    If the CPU reaches the tail while the GPU is still in-flight, the CPU speculatively
    steals and processes the remaining GPU slice so execution never stalls on slow GPU kernels.
    """
    def __init__(self, total_columns, rows=128, default_gpu_slice=64, default_cpu_chunk=32, cpu_allowance=None, min_horizon=None, pacing_ratio=None, force_mode=None):
        import os
        self.total_columns = int(total_columns)
        self.rows = int(rows)
        self.default_gpu_slice = max(1, int(default_gpu_slice))
        self.current_gpu_slice = self.default_gpu_slice
        self.default_cpu_chunk = max(1, int(default_cpu_chunk))
        self.current_cpu_chunk = self.default_cpu_chunk
        self.force_mode = force_mode or ('cpu' if os.environ.get('MOLMEM_HYBRID_FORCE_CPU') == '1' else ('gpu' if os.environ.get('MOLMEM_HYBRID_FORCE_GPU') == '1' else None))
        
        self._head = 0  # CPU sweeps forward: 0 -> N
        self._tail = int(total_columns)  # GPU sweeps backward: N -> 0
        self._lock = threading.Lock()
        
        # Speculative racing state
        self.gpu_in_flight = False
        self.gpu_in_flight_range = None
        self.gpu_completed_ranges = []
        self.all_completed_by_cpu = False
        
        # Telemetry counters
        self.gpu_columns_processed = 0
        self.cpu_columns_processed = 0
        self.gpu_chunks_count = 0
        self.cpu_chunks_count = 0
        
        # Continuous real-time empirical throughput tracking
        import time
        self.t_init = time.perf_counter()
        self.gpu_rate = None
        self.cpu_rate = None
        self.gpu_last_claim_time = self.t_init
        self.cpu_last_claim_time = self.t_init

    def claim_gpu_chunk(self):
        """
        Atomically claims a backward slice from the back (tail -> head).
        Continuously adjusts chunk size based on analytical real-time throughput ratios.
        """
        if self.force_mode == 'cpu':
            return None
            
        with self._lock:
            import time
            if self._head >= self._tail:
                return None
            remaining = self._tail - self._head
            if remaining <= 0:
                return None
            
            # Continuous Analytical Partitioning from live throughput measurements
            if self.gpu_rate is not None and self.cpu_rate is not None and (self.gpu_rate + self.cpu_rate) > 0:
                gpu_ratio = self.gpu_rate / (self.gpu_rate + self.cpu_rate)
                ideal_gpu_remainder = remaining * gpu_ratio
                if ideal_gpu_remainder < self.default_gpu_slice:
                    chunk = max(1, min(remaining, int(round(ideal_gpu_remainder))))
                else:
                    chunk = min(self.default_gpu_slice, remaining)
            else:
                chunk = min(self.default_gpu_slice, remaining)
                
            if chunk < 1:
                return None
                
            col_end = self._tail
            col_start = col_end - chunk
            self._tail = col_start
            self.gpu_in_flight = True
            self.gpu_in_flight_range = (col_start, col_end)
            self.gpu_columns_processed += chunk
            self.gpu_chunks_count += 1
            self.gpu_last_claim_time = time.perf_counter()
            return (col_start, col_end)

    claim_gpu_slice = claim_gpu_chunk

    def mark_gpu_completed(self, col_start, col_end):
        """Marks the in-flight GPU slice as completed and updates continuous GPU throughput."""
        with self._lock:
            import time
            elapsed = max(1e-5, time.perf_counter() - self.gpu_last_claim_time)
            cols = max(1, col_end - col_start)
            inst_rate = cols / elapsed
            self.gpu_rate = inst_rate if self.gpu_rate is None else (0.7 * self.gpu_rate + 0.3 * inst_rate)
            self.gpu_in_flight = False
            self.gpu_completed_ranges.append((col_start, col_end))

    def claim_cpu_chunk(self):
        """
        Atomically claims a forward chunk from the front (head -> tail).
        Continuously adjusts chunk size based on analytical real-time throughput ratios.
        """
        if self.force_mode == 'gpu':
            return None
            
        with self._lock:
            import time
            if self.cpu_chunks_count > 0:
                elapsed = max(1e-5, time.perf_counter() - self.cpu_last_claim_time)
                inst_rate = self.default_cpu_chunk / elapsed
                self.cpu_rate = inst_rate if self.cpu_rate is None else (0.7 * self.cpu_rate + 0.3 * inst_rate)
                
            if self._head >= self._tail:
                return None
            remaining = self._tail - self._head
            if remaining <= 0:
                return None
            
            # Continuous Analytical Partitioning from live throughput measurements
            if self.gpu_rate is not None and self.cpu_rate is not None and (self.gpu_rate + self.cpu_rate) > 0:
                cpu_ratio = self.cpu_rate / (self.gpu_rate + self.cpu_rate)
                ideal_cpu_remainder = remaining * cpu_ratio
                if ideal_cpu_remainder < self.default_cpu_chunk:
                    chunk = max(1, min(remaining, int(round(ideal_cpu_remainder))))
                else:
                    chunk = min(self.default_cpu_chunk, remaining)
            else:
                chunk = min(self.default_cpu_chunk, remaining)
                
            if chunk < 1:
                return None
                
            col_start = self._head
            col_end = col_start + chunk
            self._head = col_end
            self.cpu_columns_processed += chunk
            self.cpu_chunks_count += 1
            self.cpu_last_claim_time = time.perf_counter()
            return (col_start, col_end)

    def is_all_done_by_cpu(self):
        with self._lock:
            return self.gpu_columns_processed == 0 and self._head >= self.total_columns

    def get_progress(self):
        with self._lock:
            total_done = self.cpu_columns_processed + self.gpu_columns_processed
            return min(self.total_columns, total_done), self.total_columns, self.gpu_columns_processed, self.cpu_columns_processed

    def is_finished(self):
        with self._lock:
            return self._head >= self._tail or self.all_completed_by_cpu

HeterogeneousWorkQueue = DualEndedWorkQueue


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def jit_cpu_device_updates_slice(
    start_idx, end_idx, pulses, voltages, pwidths,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    f22DepScaled_arr, f22PotScaled_arr, iScale_arr, vScale_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    _cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
    y_i, y_v, y_fp, y_fd, G_arr, vRead=0.1
):
    for i in nb.prange(start_idx, end_idx):
        p_count = pulses[i]
        if p_count <= 0:
            if currentF22_arr[i] == 0.0:
                G_arr[i] = 0.0
            else:
                if gamma_arr[i] != 0.0:
                    vScale_eff = vScale_arr[i] / (1.0 + gamma_arr[i] * currentF22_arr[i])
                else:
                    vScale_eff = currentVScale_arr[i]
                gCore = currentIScale_arr[i] * currentF22_arr[i] / (vScale_eff + 1e-12)
                if abs(vRead) > 1e-5:
                    ratio = vRead / vScale_eff
                    ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                    gCore *= math.sinh(ratio_clipped) / ratio
                G_arr[i] = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))
            continue
            
        v_amp = voltages[i]
        pw = pwidths[i]

        for p in range(1, p_count + 1):
            n_next, mode_next, scale_next, T_next = jit_updateState(
                pw, v_amp, n_arr[i], modeState_arr[i], scaleFactor_arr[i], T_arr[i],
                vSmooth_arr[i], nSmooth_arr[i], kappa_arr[i], alpha_arr[i], vTh_arr[i], vRefPos_arr[i], vRefNeg_arr[i], nMax_arr[i], kDischarge_arr[i],
                f22DepScaled_arr[i], f22PotScaled_arr[i], currentIScale_arr[i], currentVScale_arr[i], currentF22_arr[i], rC_arr[i],
                E_a_arr[i], R_th_arr[i], tau_th_arr[i],
                (beta_alpha_arr[i], beta_s_arr[i], gamma_arr[i]), vScale_arr[i], rateAsymmetry_arr[i],
                rTopBlend_arr[i], fDischarge_arr[i], cDischarge_arr[i], rBottomBlend_arr[i], rLatency_arr[i],
                k_overdrive_arr[i]
            )
            n_arr[i] = n_next
            modeState_arr[i] = mode_next
            scaleFactor_arr[i] = scale_next
            T_arr[i] = T_next

            k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
            n_base_nom = 0.5 * nMax_arr[i]
            od_ratio = cDischarge_arr[i] if cDischarge_arr[i] > 0.0 else 0.0
            od_clamp = min(1.0, max(0.0, od_ratio * k_od))
            scale_base = rBottomBlend_arr[i] if (rBottomBlend_arr[i] >= 0.05 and rBottomBlend_arr[i] <= 2.5) else 1.0
            n_base = n_base_nom * (1.0 - od_clamp * (1.0 - scale_base))
            if modeState_arr[i] > 0.5 and n_arr[i] > n_base_nom:
                if v_amp > vRefPos_arr[i] and (vRefPos_arr[i] - vTh_arr[i]) > 1e-6:
                    od_inst = (v_amp - vRefPos_arr[i]) / (vRefPos_arr[i] - vTh_arr[i])
                    if od_inst > cDischarge_arr[i]:
                        cDischarge_arr[i] = od_inst
            elif n_arr[i] <= n_base:
                cDischarge_arr[i] = 0.0

            is_pot = (modeState_arr[i] > 0.5)
            state_changed = abs(n_next - _cached_n_int_arr[i]) > 0.01
            if state_changed:
                _cached_n_int_arr[i] = n_next
                iScale_arr[i] = jit_interp_O1(n_next, y_i, nMax_arr[i])
                vScale_arr[i] = jit_interp_O1(n_next, y_v, nMax_arr[i])
                f22DepRaw = jit_interp_O1(nMax_arr[i] - n_next, y_fd, nMax_arr[i])
                f22DepScaled_arr[i] = (1.0 - f22DepRaw - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
                f22PotRaw = jit_interp_O1(n_next, y_fp, nMax_arr[i])
                pot_val = (f22PotRaw - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
                if pot_val < 0.0:
                    pot_val = 0.0
                elif n_next > 16520.0:
                    u_sat = (n_next - 16520.0) / 16520.0
                    if u_sat > 1.0: u_sat = 1.0
                    n_sm_eff = nSmooth_arr[i]
                    v_amp_abs = abs(v_amp)
                    if v_amp_abs > vRefPos_arr[i] and beta_s_arr[i] != 0.0:
                        d_v = vRefPos_arr[i] - vTh_arr[i] + 1e-6
                        del_v = (v_amp_abs - vRefPos_arr[i]) / (d_v if d_v > 1e-15 else 1e-15)
                        n_sm_eff = n_sm_eff * (1.0 + beta_s_arr[i] * math.tanh(del_v))
                        if n_sm_eff < 1.0: n_sm_eff = 1.0
                        elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                    pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                f22PotScaled_arr[i] = pot_val
                currentIScale_arr[i] = iScale_arr[i] * gScale_arr[i]

            if modeState_arr[i] > 0.6:
                denom = f22DepScaled_arr[i] if f22DepScaled_arr[i] > 1e-6 else 1e-6
                ratio = f22PotScaled_arr[i] / denom
                scaleFactor_arr[i] = 0.0 if ratio < 0.0 else (100.0 if ratio > 100.0 else ratio)

            f22 = f22PotScaled_arr[i] if is_pot else (f22DepScaled_arr[i] * scaleFactor_arr[i])
            currentF22_arr[i] = 0.0 if f22 < 0.0 else f22

            # Relaxation phase between pulses
            if modeState_arr[i] != 0.0:
                modeState_arr[i] *= math.exp(-pw * 1000.0)
                is_pot_rel = (modeState_arr[i] > 0.5)
                f22_rel = f22PotScaled_arr[i] if is_pot_rel else (f22DepScaled_arr[i] * scaleFactor_arr[i])
                currentF22_arr[i] = 0.0 if f22_rel < 0.0 else f22_rel
            T_arr[i] = 298.15

        if currentF22_arr[i] == 0.0:
            G_arr[i] = 0.0
        else:
            if gamma_arr[i] != 0.0:
                vScale_eff = vScale_arr[i] / (1.0 + gamma_arr[i] * currentF22_arr[i])
            else:
                vScale_eff = vScale_arr[i]
            currentVScale_arr[i] = vScale_eff
            gCore = currentIScale_arr[i] * currentF22_arr[i] / (vScale_eff + 1e-12)
            if abs(vRead) > 1e-5:
                ratio = vRead / vScale_eff
                ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                gCore *= math.sinh(ratio_clipped) / ratio
            G_arr[i] = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))


@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_trace_pulse_history(
    t_start, num_devices, pulses, voltages, pwidths,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    f22DepScaled_arr, f22PotScaled_arr, iScale_arr, vScale_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    _cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
    y_i, y_v, y_fp, y_fd, vRead=0.1,
    rows=1, cols=1, is_col_major=False, is_column_sequential=False
):
    if is_column_sequential and cols > 1:
        col_max_p = np.zeros(cols, dtype=np.int32)
        col_max_pw = np.full(cols, 80e-9, dtype=np.float64)
        total_pulses_all_cols = 0
        for c in range(cols):
            m_p = 0
            m_pw = 80e-9
            for r in range(rows):
                idx = (c * rows + r) if is_col_major else (r * cols + c)
                if idx < num_devices:
                    if pulses[idx] > m_p:
                        m_p = pulses[idx]
                    if pwidths[idx] > m_pw:
                        m_pw = pwidths[idx]
            col_max_p[c] = m_p
            col_max_pw[c] = m_pw
            total_pulses_all_cols += m_p

        if total_pulses_all_cols <= 0:
            num_pts = 3
        else:
            num_pts = total_pulses_all_cols * 2 + 1
        period = 160e-9
    else:
        max_p = 0
        max_pw = 80e-9
        for i in range(num_devices):
            if pulses[i] > max_p:
                max_p = pulses[i]
            if pwidths[i] > max_pw:
                max_pw = pwidths[i]
                
        if max_p <= 0:
            max_p = 1
            
        num_pts = max_p * 2 + 1
        period = max_pw * 2.0

    t_out = np.zeros(num_pts, dtype=np.float64)
    g_out = np.zeros((num_devices, num_pts), dtype=np.float64)
    n_out = np.zeros((num_devices, num_pts), dtype=np.float64)
    f22_out = np.zeros((num_devices, num_pts), dtype=np.float64)
    temp_out = np.zeros((num_devices, num_pts), dtype=np.float64)
    
    t_out[0] = t_start
    
    # Clone states for tracing
    curr_n = n_arr.copy()
    curr_mode = modeState_arr.copy()
    curr_scale = scaleFactor_arr.copy()
    curr_T = T_arr.copy()
    curr_f22Dep = f22DepScaled_arr.copy()
    curr_f22Pot = f22PotScaled_arr.copy()
    curr_iScale = iScale_arr.copy()
    curr_vScale = vScale_arr.copy()
    curr_currentF22 = currentF22_arr.copy()
    curr_currentIScale = currentIScale_arr.copy()
    curr_currentVScale = currentVScale_arr.copy()
    curr_cached_n_int = _cached_n_int_arr.copy()
    curr_cDischarge = cDischarge_arr.copy()
    
    # Evaluate initial conductance for all devices
    for i in range(num_devices):
        if curr_n[i] > 0.0 and (curr_f22Pot[i] <= 0.0 or curr_f22Dep[i] <= 0.0):
            f22DepRaw_init = jit_interp_O1(nMax_arr[i] - curr_n[i], y_fd, nMax_arr[i])
            curr_f22Dep[i] = (1.0 - f22DepRaw_init - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
            f22PotRaw_init = jit_interp_O1(curr_n[i], y_fp, nMax_arr[i])
            pot_val_init = (f22PotRaw_init - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
            if pot_val_init < 0.0:
                pot_val_init = 0.0
            elif curr_n[i] > 16520.0:
                u_sat = (curr_n[i] - 16520.0) / 16520.0
                if u_sat > 1.0: u_sat = 1.0
                pot_val_init += 0.001 * nSmooth_arr[i] * u_sat * u_sat * (3.0 - 2.0 * u_sat)
            curr_f22Pot[i] = pot_val_init
            if curr_mode[i] > 0.6:
                denom = curr_f22Dep[i] if curr_f22Dep[i] > 1e-6 else 1e-6
                ratio = curr_f22Pot[i] / denom
                curr_scale[i] = 0.0 if ratio < 0.0 else (100.0 if ratio > 100.0 else ratio)
        if curr_currentF22[i] <= 0.0 and curr_n[i] > 0.0:
            is_pot_init = (curr_mode[i] > 0.5)
            f22_init = curr_f22Pot[i] if is_pot_init else (curr_f22Dep[i] * curr_scale[i])
            if f22_init > 0.0:
                curr_currentF22[i] = f22_init
        if curr_currentF22[i] == 0.0:
            g_out[i, 0] = 0.0
        else:
            if gamma_arr[i] != 0.0:
                vScale_eff = curr_vScale[i] / (1.0 + gamma_arr[i] * curr_currentF22[i])
            else:
                vScale_eff = curr_currentVScale[i]
            gCore = curr_currentIScale[i] * curr_currentF22[i] / (vScale_eff + 1e-12)
            if abs(vRead) > 1e-5:
                ratio = vRead / vScale_eff
                ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                gCore *= math.sinh(ratio_clipped) / ratio
            g_out[i, 0] = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))
        n_out[i, 0] = curr_n[i]
        f22_out[i, 0] = curr_currentF22[i]
        temp_out[i, 0] = curr_T[i]

    if is_column_sequential and cols > 1:
        if total_pulses_all_cols <= 0:
            t_out[1] = t_start + 10e-9
            t_out[2] = t_start + 20e-9
            for i in range(num_devices):
                g_out[i, 1] = g_out[i, 0]
                g_out[i, 2] = g_out[i, 0]
                n_out[i, 1] = n_out[i, 0]
                n_out[i, 2] = n_out[i, 0]
                f22_out[i, 1] = f22_out[i, 0]
                f22_out[i, 2] = f22_out[i, 0]
                temp_out[i, 1] = temp_out[i, 0]
                temp_out[i, 2] = temp_out[i, 0]
        else:
            curr_time = t_start
            pt_cursor = 0
            for c in range(cols):
                c_pulses = col_max_p[c]
                if c_pulses <= 0:
                    continue
                c_pw = col_max_pw[c]
                c_period = c_pw * 2.0
                for p in range(1, c_pulses + 1):
                    idx_pulse = pt_cursor + 1
                    idx_relax = pt_cursor + 2
                    pt_cursor += 2

                    t_pulse = curr_time + (p - 1) * c_period + c_pw
                    t_relax = curr_time + p * c_period
                    t_out[idx_pulse] = t_pulse
                    t_out[idx_relax] = t_relax

                    for i in range(num_devices):
                        dev_col = (i // rows) if is_col_major else (i % cols)
                        if dev_col == c:
                            p_total = pulses[i]
                            if p <= p_total:
                                v_amp = voltages[i]
                                pw = pwidths[i]

                                # Apply active pulse phase
                                n_next, mode_next, scale_next, T_next = jit_updateState(
                                    pw, v_amp, curr_n[i], curr_mode[i], curr_scale[i], curr_T[i],
                                    vSmooth_arr[i], nSmooth_arr[i], kappa_arr[i], alpha_arr[i], vTh_arr[i], vRefPos_arr[i], vRefNeg_arr[i], nMax_arr[i], kDischarge_arr[i],
                                    curr_f22Dep[i], curr_f22Pot[i], curr_currentIScale[i], curr_currentVScale[i], curr_currentF22[i], rC_arr[i],
                                    E_a_arr[i], R_th_arr[i], tau_th_arr[i],
                                    (beta_alpha_arr[i], beta_s_arr[i], gamma_arr[i]), curr_vScale[i], rateAsymmetry_arr[i],
                                    rTopBlend_arr[i], fDischarge_arr[i], curr_cDischarge[i], rBottomBlend_arr[i], rLatency_arr[i],
                                    k_overdrive_arr[i]
                                )
                                curr_n[i] = n_next
                                curr_mode[i] = mode_next
                                curr_scale[i] = scale_next
                                curr_T[i] = T_next

                                k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
                                n_base_nom = 0.5 * nMax_arr[i]
                                od_ratio = curr_cDischarge[i] if curr_cDischarge[i] > 0.0 else 0.0
                                od_clamp = min(1.0, max(0.0, od_ratio * k_od))
                                scale_base = rBottomBlend_arr[i] if (rBottomBlend_arr[i] >= 0.05 and rBottomBlend_arr[i] <= 2.5) else 1.0
                                n_base = n_base_nom * (1.0 - od_clamp * (1.0 - scale_base))
                                if curr_mode[i] > 0.5 and curr_n[i] > n_base_nom:
                                    if v_amp > vRefPos_arr[i] and (vRefPos_arr[i] - vTh_arr[i]) > 1e-6:
                                        od_inst = (v_amp - vRefPos_arr[i]) / (vRefPos_arr[i] - vTh_arr[i])
                                        if od_inst > curr_cDischarge[i]:
                                            curr_cDischarge[i] = od_inst
                                elif curr_n[i] <= n_base:
                                    curr_cDischarge[i] = 0.0

                                is_pot = (curr_mode[i] > 0.5)
                                state_changed = abs(n_next - curr_cached_n_int[i]) > 0.01
                                if state_changed:
                                    curr_cached_n_int[i] = n_next
                                    curr_iScale[i] = jit_interp_O1(n_next, y_i, nMax_arr[i])
                                    curr_vScale[i] = jit_interp_O1(n_next, y_v, nMax_arr[i])
                                    f22DepRaw = jit_interp_O1(nMax_arr[i] - n_next, y_fd, nMax_arr[i])
                                    curr_f22Dep[i] = (1.0 - f22DepRaw - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
                                    f22PotRaw = jit_interp_O1(n_next, y_fp, nMax_arr[i])
                                    pot_val = (f22PotRaw - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
                                    if pot_val < 0.0:
                                        pot_val = 0.0
                                    elif n_next > 16520.0:
                                        u_sat = (n_next - 16520.0) / 16520.0
                                        if u_sat > 1.0: u_sat = 1.0
                                        n_sm_eff = nSmooth_arr[i]
                                        v_amp_abs = abs(voltages[i])
                                        if v_amp_abs > vRefPos_arr[i] and beta_s_arr[i] != 0.0:
                                            d_v = vRefPos_arr[i] - vTh_arr[i] + 1e-6
                                            del_v = (v_amp_abs - vRefPos_arr[i]) / (d_v if d_v > 1e-15 else 1e-15)
                                            n_sm_eff = n_sm_eff * (1.0 + beta_s_arr[i] * math.tanh(del_v))
                                            if n_sm_eff < 1.0: n_sm_eff = 1.0
                                            elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                                        pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                                    curr_f22Pot[i] = pot_val
                                    curr_currentIScale[i] = curr_iScale[i] * gScale_arr[i]

                                if curr_mode[i] > 0.6:
                                    denom = curr_f22Dep[i] if curr_f22Dep[i] > 1e-6 else 1e-6
                                    ratio = curr_f22Pot[i] / denom
                                    curr_scale[i] = 0.0 if ratio < 0.0 else (100.0 if ratio > 100.0 else ratio)

                                f22 = curr_f22Pot[i] if is_pot else (curr_f22Dep[i] * curr_scale[i])
                                curr_currentF22[i] = 0.0 if f22 < 0.0 else f22

                                # Active pulse conductance
                                if curr_currentF22[i] == 0.0:
                                    g_val = 0.0
                                else:
                                    if gamma_arr[i] != 0.0:
                                        vScale_eff = curr_vScale[i] / (1.0 + gamma_arr[i] * curr_currentF22[i])
                                    else:
                                        vScale_eff = curr_vScale[i]
                                    curr_currentVScale[i] = vScale_eff
                                    gCore = curr_currentIScale[i] * curr_currentF22[i] / (vScale_eff + 1e-12)
                                    if abs(vRead) > 1e-5:
                                        ratio = vRead / vScale_eff
                                        ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                                        gCore *= math.sinh(ratio_clipped) / ratio
                                    g_val = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))

                                g_out[i, idx_pulse] = g_val
                                n_out[i, idx_pulse] = curr_n[i]
                                f22_out[i, idx_pulse] = curr_currentF22[i]
                                temp_out[i, idx_pulse] = curr_T[i]

                                # Relaxation phase
                                if curr_mode[i] != 0.0:
                                    curr_mode[i] *= math.exp(-pw * 1000.0)
                                    is_pot_rel = (curr_mode[i] > 0.5)
                                    f22_rel = curr_f22Pot[i] if is_pot_rel else (curr_f22Dep[i] * curr_scale[i])
                                    curr_currentF22[i] = 0.0 if f22_rel < 0.0 else f22_rel
                                curr_T[i] = 298.15

                                if curr_currentF22[i] == 0.0:
                                    g_rel = 0.0
                                else:
                                    if gamma_arr[i] != 0.0:
                                        vScale_eff = curr_vScale[i] / (1.0 + gamma_arr[i] * curr_currentF22[i])
                                    else:
                                        vScale_eff = curr_currentVScale[i]
                                    gCore = curr_currentIScale[i] * curr_currentF22[i] / (vScale_eff + 1e-12)
                                    if abs(vRead) > 1e-5:
                                        ratio = vRead / vScale_eff
                                        ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                                        gCore *= math.sinh(ratio_clipped) / ratio
                                    g_rel = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))

                                g_out[i, idx_relax] = g_rel
                                n_out[i, idx_relax] = curr_n[i]
                                f22_out[i, idx_relax] = curr_currentF22[i]
                                temp_out[i, idx_relax] = curr_T[i]
                            else:
                                g_out[i, idx_pulse] = g_out[i, idx_pulse - 1]
                                g_out[i, idx_relax] = g_out[i, idx_pulse - 1]
                                n_out[i, idx_pulse] = n_out[i, idx_pulse - 1]
                                n_out[i, idx_relax] = n_out[i, idx_pulse - 1]
                                f22_out[i, idx_pulse] = f22_out[i, idx_pulse - 1]
                                f22_out[i, idx_relax] = f22_out[i, idx_pulse - 1]
                                temp_out[i, idx_pulse] = temp_out[i, idx_pulse - 1]
                                temp_out[i, idx_relax] = temp_out[i, idx_pulse - 1]
                        else:
                            # Inactive column devices hold state
                            g_out[i, idx_pulse] = g_out[i, idx_pulse - 1]
                            g_out[i, idx_relax] = g_out[i, idx_pulse - 1]
                            n_out[i, idx_pulse] = n_out[i, idx_pulse - 1]
                            n_out[i, idx_relax] = n_out[i, idx_pulse - 1]
                            f22_out[i, idx_pulse] = f22_out[i, idx_pulse - 1]
                            f22_out[i, idx_relax] = f22_out[i, idx_pulse - 1]
                            temp_out[i, idx_pulse] = temp_out[i, idx_pulse - 1]
                            temp_out[i, idx_relax] = temp_out[i, idx_pulse - 1]
                curr_time += c_pulses * c_period
    else:
        for p in range(1, max_p + 1):
            idx_pulse = (p - 1) * 2 + 1
            idx_relax = (p - 1) * 2 + 2
            
            t_pulse = t_start + (p - 1) * period + max_pw
            t_relax = t_start + p * period
            t_out[idx_pulse] = t_pulse
            t_out[idx_relax] = t_relax
            
            for i in range(num_devices):
                p_total = pulses[i]
                if p <= p_total:
                    v_amp = voltages[i]
                    pw = pwidths[i]
                    
                    # Apply active pulse phase
                    n_next, mode_next, scale_next, T_next = jit_updateState(
                        pw, v_amp, curr_n[i], curr_mode[i], curr_scale[i], curr_T[i],
                        vSmooth_arr[i], nSmooth_arr[i], kappa_arr[i], alpha_arr[i], vTh_arr[i], vRefPos_arr[i], vRefNeg_arr[i], nMax_arr[i], kDischarge_arr[i],
                        curr_f22Dep[i], curr_f22Pot[i], curr_currentIScale[i], curr_currentVScale[i], curr_currentF22[i], rC_arr[i],
                        E_a_arr[i], R_th_arr[i], tau_th_arr[i],
                        (beta_alpha_arr[i], beta_s_arr[i], gamma_arr[i]), curr_vScale[i], rateAsymmetry_arr[i],
                        rTopBlend_arr[i], fDischarge_arr[i], curr_cDischarge[i], rBottomBlend_arr[i], rLatency_arr[i],
                        k_overdrive_arr[i]
                    )
                    curr_n[i] = n_next
                    curr_mode[i] = mode_next
                    curr_scale[i] = scale_next
                    curr_T[i] = T_next
                    
                    k_od = k_overdrive_arr[i] if k_overdrive_arr is not None else 0.0
                    n_base_nom = 0.5 * nMax_arr[i]
                    od_ratio = curr_cDischarge[i] if curr_cDischarge[i] > 0.0 else 0.0
                    od_clamp = min(1.0, max(0.0, od_ratio * k_od))
                    scale_base = rBottomBlend_arr[i] if (rBottomBlend_arr[i] >= 0.05 and rBottomBlend_arr[i] <= 2.5) else 1.0
                    n_base = n_base_nom * (1.0 - od_clamp * (1.0 - scale_base))
                    if curr_mode[i] > 0.5 and curr_n[i] > n_base_nom:
                        if v_amp > vRefPos_arr[i] and (vRefPos_arr[i] - vTh_arr[i]) > 1e-6:
                            od_inst = (v_amp - vRefPos_arr[i]) / (vRefPos_arr[i] - vTh_arr[i])
                            if od_inst > curr_cDischarge[i]:
                                curr_cDischarge[i] = od_inst
                    elif curr_n[i] <= n_base:
                        curr_cDischarge[i] = 0.0
                    
                    is_pot = (curr_mode[i] > 0.5)
                    state_changed = abs(n_next - curr_cached_n_int[i]) > 0.01
                    if state_changed:
                        curr_cached_n_int[i] = n_next
                        curr_iScale[i] = jit_interp_O1(n_next, y_i, nMax_arr[i])
                        curr_vScale[i] = jit_interp_O1(n_next, y_v, nMax_arr[i])
                        f22DepRaw = jit_interp_O1(nMax_arr[i] - n_next, y_fd, nMax_arr[i])
                        curr_f22Dep[i] = (1.0 - f22DepRaw - f22EndVal_arr[i]) * f22DepScaleInv_arr[i]
                        f22PotRaw = jit_interp_O1(n_next, y_fp, nMax_arr[i])
                        pot_val = (f22PotRaw - f22StartVal_arr[i]) * f22PotScaleInv_arr[i]
                        if pot_val < 0.0:
                            pot_val = 0.0
                        elif n_next > 16520.0:
                            u_sat = (n_next - 16520.0) / 16520.0
                            if u_sat > 1.0: u_sat = 1.0
                            n_sm_eff = nSmooth_arr[i]
                            v_amp_abs = abs(voltages[i])
                            if v_amp_abs > vRefPos_arr[i] and beta_s_arr[i] != 0.0:
                                d_v = vRefPos_arr[i] - vTh_arr[i] + 1e-6
                                del_v = (v_amp_abs - vRefPos_arr[i]) / (d_v if d_v > 1e-15 else 1e-15)
                                n_sm_eff = n_sm_eff * (1.0 + beta_s_arr[i] * math.tanh(del_v))
                                if n_sm_eff < 1.0: n_sm_eff = 1.0
                                elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                            pot_val += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                        curr_f22Pot[i] = pot_val
                        curr_currentIScale[i] = curr_iScale[i] * gScale_arr[i]
                        
                    if curr_mode[i] > 0.6:
                        denom = curr_f22Dep[i] if curr_f22Dep[i] > 1e-6 else 1e-6
                        ratio = curr_f22Pot[i] / denom
                        curr_scale[i] = 0.0 if ratio < 0.0 else (100.0 if ratio > 100.0 else ratio)

                    f22 = curr_f22Pot[i] if is_pot else (curr_f22Dep[i] * curr_scale[i])
                    curr_currentF22[i] = 0.0 if f22 < 0.0 else f22
                    
                    # Active pulse conductance
                    if curr_currentF22[i] == 0.0:
                        g_val = 0.0
                    else:
                        if gamma_arr[i] != 0.0:
                            vScale_eff = curr_vScale[i] / (1.0 + gamma_arr[i] * curr_currentF22[i])
                        else:
                            vScale_eff = curr_vScale[i]
                        curr_currentVScale[i] = vScale_eff
                        gCore = curr_currentIScale[i] * curr_currentF22[i] / (vScale_eff + 1e-12)
                        if abs(vRead) > 1e-5:
                            ratio = vRead / vScale_eff
                            ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                            gCore *= math.sinh(ratio_clipped) / ratio
                        g_val = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))
                        
                    g_out[i, idx_pulse] = g_val
                    n_out[i, idx_pulse] = curr_n[i]
                    f22_out[i, idx_pulse] = curr_currentF22[i]
                    temp_out[i, idx_pulse] = curr_T[i]
                    
                    # Relaxation phase
                    if curr_mode[i] != 0.0:
                        curr_mode[i] *= math.exp(-pw * 1000.0)
                        is_pot_rel = (curr_mode[i] > 0.5)
                        f22_rel = curr_f22Pot[i] if is_pot_rel else (curr_f22Dep[i] * curr_scale[i])
                        curr_currentF22[i] = 0.0 if f22_rel < 0.0 else f22_rel
                    curr_T[i] = 298.15
                    
                    if curr_currentF22[i] == 0.0:
                        g_rel = 0.0
                    else:
                        if gamma_arr[i] != 0.0:
                            vScale_eff = curr_vScale[i] / (1.0 + gamma_arr[i] * curr_currentF22[i])
                        else:
                            vScale_eff = curr_currentVScale[i]
                        gCore = curr_currentIScale[i] * curr_currentF22[i] / (vScale_eff + 1e-12)
                        if abs(vRead) > 1e-5:
                            ratio = vRead / vScale_eff
                            ratio_clipped = -50.0 if ratio < -50.0 else (50.0 if ratio > 50.0 else ratio)
                            gCore *= math.sinh(ratio_clipped) / ratio
                        g_rel = gCore if rC_arr[i] == 0.0 else (gCore / (1.0 + gCore * (2.0 * rC_arr[i])))
                        
                    g_out[i, idx_relax] = g_rel
                    n_out[i, idx_relax] = curr_n[i]
                    f22_out[i, idx_relax] = curr_currentF22[i]
                    temp_out[i, idx_relax] = curr_T[i]
                else:
                    g_out[i, idx_pulse] = g_out[i, idx_pulse - 1]
                    g_out[i, idx_relax] = g_out[i, idx_pulse - 1]
                    n_out[i, idx_pulse] = n_out[i, idx_pulse - 1]
                    n_out[i, idx_relax] = n_out[i, idx_pulse - 1]
                    f22_out[i, idx_pulse] = f22_out[i, idx_pulse - 1]
                    f22_out[i, idx_relax] = f22_out[i, idx_pulse - 1]
                    temp_out[i, idx_pulse] = temp_out[i, idx_pulse - 1]
                    temp_out[i, idx_relax] = temp_out[i, idx_pulse - 1]
                    
    return t_out, g_out, n_out, f22_out, temp_out


@nb.njit(fastmath=True, nogil=True, cache=True)
def jit_cpu_device_updates_parallel(
    num_devices, pulses, voltages, pwidths,
    n_arr, modeState_arr, scaleFactor_arr, T_arr,
    f22DepScaled_arr, f22PotScaled_arr, iScale_arr, vScale_arr,
    currentF22_arr, currentIScale_arr, currentVScale_arr,
    _cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
    vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
    vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
    gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
    beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
    rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
    y_i, y_v, y_fp, y_fd, G_arr, vRead=0.1
):
    jit_cpu_device_updates_slice(
        0, num_devices, pulses, voltages, pwidths,
        n_arr, modeState_arr, scaleFactor_arr, T_arr,
        f22DepScaled_arr, f22PotScaled_arr, iScale_arr, vScale_arr,
        currentF22_arr, currentIScale_arr, currentVScale_arr,
        _cached_n_int_arr, rC_arr, E_a_arr, R_th_arr, tau_th_arr,
        vSmooth_arr, nSmooth_arr, kappa_arr, alpha_arr,
        vTh_arr, vBypass_arr, vRefPos_arr, vRefNeg_arr, nMax_arr, kDischarge_arr,
        gScale_arr, f22StartVal_arr, f22PotScaleInv_arr, f22DepScaleInv_arr, f22EndVal_arr,
        beta_alpha_arr, beta_s_arr, gamma_arr, rateAsymmetry_arr,
        rTopBlend_arr, fDischarge_arr, cDischarge_arr, rBottomBlend_arr, rLatency_arr, k_overdrive_arr,
        y_i, y_v, y_fp, y_fd, G_arr, vRead
    )


@nb.njit(nogil=True, cache=True, parallel=True)
def jit_calibrate_grid_cpu(
    vWriteRange, pwRange, is_pot, maxPulses,
    targetMaxN, targetMinN, initial_n,
    vSmooth, nSmooth, kappa, alpha, vTh, vBypass, vRefPos, vRefNeg, nMax, kDischarge,
    beta_alpha, beta_s, gamma, vScale, rateAsymmetry,
    rTopBlend=0.050000, fDischarge=0.374848, cDischarge=0.0, rBottomBlend=0.714488, rLatency=0.878506, k_overdrive=7.296635,
    rC=0.008033, E_a=0.155923, R_th=52.14, tau_th=1.278e-07,
    y_i=None, y_v=None, y_fp=None, y_fd=None,
    gScale=1.0, f22StartVal=0.0, f22PotScaleInv=1.0, f22DepScaleInv=1.0, f22EndVal=0.0
):
    nV = len(vWriteRange)
    nPW = len(pwRange)
    max_pulses_f = float(maxPulses)
    optMap = np.full((nV, nPW), max_pulses_f, dtype=np.float64)
    
    pot_target = targetMaxN * 0.999
    dep_target = targetMinN + (targetMaxN * 0.001)
    
    inv_tau_th = 1.0 / tau_th if (E_a > 0.0 and R_th > 0.0 and tau_th != 0.0) else 0.0
    T_amb = 298.15
    
    for v_idx in nb.prange(nV):
        v_val = vWriteRange[v_idx]
        if v_val <= vTh:
            continue
        v_amp = v_val if is_pot else -v_val
        
        for pw_idx in range(nPW):
            if optMap[v_idx, pw_idx] < max_pulses_f:
                continue
                
            pw_val = pwRange[pw_idx]
            dt_pulse = pw_val
            
            n_curr = initial_n
            modeState = 1.0 if is_pot else 0.0
            scaleFactor = 1.0
            T_val = T_amb
            
            cached_n = -999.0
            f22DepScaled = 0.0
            f22PotScaled = 0.0
            currentIScale = 0.0
            currentVScale = vScale
            currentF22 = 0.0
            vScale_base = vScale
            
            pulses_needed = max_pulses_f
            
            for p in range(1, maxPulses + 1):
                if y_i is not None and y_v is not None and y_fp is not None and y_fd is not None:
                    if abs(n_curr - cached_n) > 0.01:
                        cached_n = n_curr
                        iScale_val = jit_interp_O1(n_curr, y_i, nMax)
                        vScale_lut = jit_interp_O1(n_curr, y_v, nMax)
                        f22DepRaw = jit_interp_O1(nMax - n_curr, y_fd, nMax)
                        dep_raw_scaled = (1.0 - f22DepRaw - f22EndVal) * f22DepScaleInv
                        f22DepScaled = 0.0 if dep_raw_scaled < 0.0 else dep_raw_scaled
                        f22PotRaw = jit_interp_O1(n_curr, y_fp, nMax)
                        pot_raw_scaled = (f22PotRaw - f22StartVal) * f22PotScaleInv
                        if pot_raw_scaled < 0.0:
                            pot_raw_scaled = 0.0
                        elif n_curr > 16520.0:
                            u_sat = (n_curr - 16520.0) / 16520.0
                            if u_sat > 1.0: u_sat = 1.0
                            n_sm_eff = nSmooth
                            v_amp_abs = abs(v_amp)
                            if v_amp_abs > vRefPos and beta_s != 0.0:
                                d_v = vRefPos - vTh + 1e-6
                                del_v = (v_amp_abs - vRefPos) / (d_v if d_v > 1e-15 else 1e-15)
                                n_sm_eff = n_sm_eff * (1.0 + beta_s * math.tanh(del_v))
                                if n_sm_eff < 1.0: n_sm_eff = 1.0
                                elif n_sm_eff > 1000.0: n_sm_eff = 1000.0
                            pot_raw_scaled += 0.001 * n_sm_eff * u_sat * u_sat * (3.0 - 2.0 * u_sat)
                        f22PotScaled = pot_raw_scaled
                        currentIScale = iScale_val * gScale
                        vScale_base = vScale_lut if vScale_lut > 1e-12 else vScale

                    if modeState > 0.6:
                        denom = f22DepScaled if f22DepScaled > 1e-6 else 1e-6
                        ratio = f22PotScaled / denom
                        scaleFactor = 0.0 if ratio < 0.0 else (100.0 if ratio > 100.0 else ratio)

                    f22_cand = f22PotScaled if (modeState > 0.5) else (f22DepScaled * scaleFactor)
                    currentF22 = 0.0 if f22_cand < 0.0 else f22_cand

                    if gamma != 0.0:
                        currentVScale = vScale_base / (1.0 + gamma * currentF22)
                    else:
                        currentVScale = vScale_base
                
                n_next, mode_next, scale_next, T_next = jit_updateState(
                    dt_pulse, v_amp, n_curr, modeState, scaleFactor, T_val,
                    vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax, kDischarge,
                    f22DepScaled, f22PotScaled, currentIScale, currentVScale, currentF22, rC,
                    E_a, R_th, tau_th,
                    (beta_alpha, beta_s, gamma), vScale, rateAsymmetry,
                    rTopBlend, fDischarge, cDischarge, rBottomBlend, rLatency,
                    k_overdrive
                )
                if n_next == n_curr:
                    pulses_needed = max_pulses_f
                    break
                n_curr = n_next
                modeState = mode_next
                scaleFactor = scale_next
                
                # Inter-pulse relaxation and thermal cooling decay
                if modeState != 0.0:
                    modeState *= math.exp(-dt_pulse * 1000.0)
                if E_a > 0.0 and R_th > 0.0 and inv_tau_th > 0.0:
                    T_val = T_amb + (T_next - T_amb) * math.exp(-dt_pulse * inv_tau_th)
                else:
                    T_val = T_next
                
                if is_pot:
                    hit = (n_curr >= pot_target)
                else:
                    hit = (n_curr <= dep_target)
                    
                if hit:
                    pulses_needed = float(p)
                    break
            
            if pulses_needed >= max_pulses_f and n_next == initial_n:
                optMap[v_idx, pw_idx:] = max_pulses_f
                break
                
            optMap[v_idx, pw_idx] = pulses_needed
            
            if pulses_needed == 1.0:
                optMap[v_idx, pw_idx + 1:] = 1.0
                break
                    
    return optMap


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def jit_compute_conductance_map(
    gMapN, v_read, nMax, gScale, gamma, rC,
    y_i, y_v, y_fp, f22StartVal, f22PotScaleInv,
    nSmooth=150.0
):
    N = len(gMapN)
    gMapG = np.empty(N, dtype=np.float64)
    inv_v_read = 1.0 / v_read if abs(v_read) > 1e-12 else 2.0
    for i in nb.prange(N):
        n_val = gMapN[i]
        iScale = jit_interp_O1(n_val, y_i, nMax)
        vScale = jit_interp_O1(n_val, y_v, nMax)
        f22PotRaw = jit_interp_O1(n_val, y_fp, nMax)
        pot_val = (f22PotRaw - f22StartVal) * f22PotScaleInv
        if pot_val < 0.0:
            pot_val = 0.0
        elif n_val > 16520.0:
            u_sat = (n_val - 16520.0) / 16520.0
            if u_sat > 1.0: u_sat = 1.0
            pot_val += 0.001 * nSmooth * u_sat * u_sat * (3.0 - 2.0 * u_sat)
        f22PotScaled = pot_val
        currentF22 = 0.0 if f22PotScaled < 0.0 else f22PotScaled
        currentIScale = iScale * gScale
        currentVScale = (vScale / (1.0 + gamma * currentF22)) if gamma != 0.0 else vScale
        currentI, _ = jit_getIGeq(v_read, currentIScale, currentVScale, currentF22, rC)
        gMapG[i] = currentI * inv_v_read
    return gMapG


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def jit_compute_min_pulses(optMap, shifts, coarse_tol=0.005, fine_tol=0.0001):
    n_shifts = len(shifts)
    min_coarse = np.empty(n_shifts, dtype=np.float64)
    min_fine = np.empty(n_shifts, dtype=np.float64)
    
    n_v = optMap.shape[0]
    n_pw = optMap.shape[1]
    
    for s_idx in nb.prange(n_shifts):
        s_val = shifts[s_idx]
        best_coarse = 1e30
        best_fine = 1e30
        
        for v_idx in range(n_v):
            for pw_idx in range(n_pw):
                opt_val = optMap[v_idx, pw_idx]
                if opt_val > 0.0:
                    p_tgt = round(opt_val * s_val)
                    if p_tgt > 0:
                        err = abs(p_tgt / opt_val - s_val)
                        if err < coarse_tol and p_tgt < best_coarse:
                            best_coarse = p_tgt
                        if err < fine_tol and p_tgt < best_fine:
                            best_fine = p_tgt
                            
        if best_coarse >= 1e29:
            min_coarse[s_idx] = math.ceil(s_val * n_shifts)
        else:
            min_coarse[s_idx] = best_coarse
            
        if best_fine >= 1e29:
            min_fine[s_idx] = math.ceil(s_val * n_shifts)
        else:
            min_fine[s_idx] = best_fine
            
    return min_coarse, min_fine


@nb.njit(fastmath=True, nogil=True, cache=True, parallel=True)
def jit_predict_pulses_matrix(
    targetStates, currentN, isCorrectionPass, tolerance,
    optMapPot, optMapDep,
    state_map_pot, reversed_dep,
    map_pot, map_dep,
    inv_full_shift_pot, inv_full_shift_dep,
    full_shift_pot, min_dep,
    scale_pot_matrix, scale_dep_matrix,
    v_write_range, v_erase_range, pw_range,
    fallback_pulses_pot, fallback_pulses_dep,
    v_write_def, v_erase_def, pw_pot_def, pw_dep_def,
    final_pulses_pot, final_v_pot, final_pw_pot,
    final_pulses_dep, final_v_dep, final_pw_dep
):
    rows = targetStates.shape[0]
    cols = targetStates.shape[1]
    
    V_len_write = optMapPot.shape[0]
    PW_len_write = optMapPot.shape[1]
    V_len_erase = optMapDep.shape[0]
    PW_len_erase = optMapDep.shape[1]
    
    len_pot_minus_1 = len(state_map_pot) - 1
    len_dep_minus_1 = len(reversed_dep) - 1
    full_shift_dep = 1.0 / inv_full_shift_dep if inv_full_shift_dep != 0.0 else 1.0
    
    for r in nb.prange(rows):
        for c in range(cols):
            # -------------------------------------------------------------
            # POTENTIATION
            # -------------------------------------------------------------
            t_pot = targetStates[r, c]
            n_curr = currentN[r, c]
            shift_pot_raw = (t_pot - n_curr) * inv_full_shift_pot
            if shift_pot_raw < 0.0:
                shift_pot = 0.0
            elif shift_pot_raw > 1.0:
                shift_pot = 1.0
            else:
                shift_pot = shift_pot_raw
                
            best_cost_pot = 1e30
            best_p_pot = 0
            best_v_idx_pot = 0
            best_pw_idx_pot = 0
            
            min_err_pot = 1e30
            min_err_p_pot = 0
            min_err_v_idx_pot = 0
            min_err_pw_idx_pot = 0
            
            for v_idx in range(V_len_write):
                for pw_idx in range(PW_len_write):
                    opt_val = optMapPot[v_idx, pw_idx]
                    if opt_val <= 0.0:
                        continue
                    if isCorrectionPass:
                        p_tgt = int(round(opt_val * shift_pot))
                        if p_tgt <= 0:
                            continue
                        raw_ach = n_curr + (float(p_tgt) / opt_val) * full_shift_pot
                        if raw_ach > full_shift_pot + 1e-6:
                            continue
                        if raw_ach > t_pot and (raw_ach - t_pot) > (t_pot - n_curr) + 1e-6:
                            continue
                        err = abs(float(p_tgt) / opt_val - shift_pot)
                        if err < min_err_pot and p_tgt > 0:
                            min_err_pot = err
                            min_err_p_pot = p_tgt
                            min_err_v_idx_pot = v_idx
                            min_err_pw_idx_pot = pw_idx
                        if err < tolerance:
                            cost = float(p_tgt)
                            if cost < best_cost_pot:
                                best_cost_pot = cost
                                best_p_pot = p_tgt
                                best_v_idx_pot = v_idx
                                best_pw_idx_pot = pw_idx
                    else:
                        scaled_shift = opt_val * shift_pot
                        p_fl = int(math.floor(scaled_shift))
                        p_ce = int(math.ceil(scaled_shift))
                        
                        # Floor candidate (potentiation)
                        cost_fl = 1e30
                        if p_fl > 0:
                            frac_fl = float(p_fl) / opt_val
                            delta_n_fl = frac_fl * full_shift_pot
                            raw_ach_fl = n_curr + delta_n_fl
                            if raw_ach_fl <= full_shift_pot + 1e-6:
                                ach_fl = raw_ach_fl
                                err_fl = abs(ach_fl - t_pot) * inv_full_shift_pot
                                if ach_fl <= t_pot:
                                    diff_pot = int(round((t_pot - ach_fl) * inv_full_shift_pot * len_pot_minus_1))
                                    if diff_pot < 0: diff_pot = 0
                                    elif diff_pot > len_pot_minus_1: diff_pot = len_pot_minus_1
                                    pred_p2 = map_pot[diff_pot]
                                else:
                                    diff_dep = int(round((ach_fl - t_pot) * inv_full_shift_dep * len_dep_minus_1))
                                    if diff_dep < 0: diff_dep = 0
                                    elif diff_dep > len_dep_minus_1: diff_dep = len_dep_minus_1
                                    pred_p2 = map_dep[diff_dep]
                                cost_fl = float(p_fl) + pred_p2 + err_fl * 5000.0
                                
                        # Ceil candidate (potentiation)
                        cost_ce = 1e30
                        if p_ce > 0:
                            frac_ce = float(p_ce) / opt_val
                            delta_n_ce = frac_ce * full_shift_pot
                            raw_ach_ce = n_curr + delta_n_ce
                            if raw_ach_ce <= full_shift_pot + 1e-6:
                                if raw_ach_ce > t_pot and (raw_ach_ce - t_pot) > (t_pot - n_curr) + 1e-6:
                                    pass
                                else:
                                    ach_ce = raw_ach_ce
                                    err_ce = abs(ach_ce - t_pot) * inv_full_shift_pot
                                    if ach_ce <= t_pot:
                                        diff_pot = int(round((t_pot - ach_ce) * inv_full_shift_pot * len_pot_minus_1))
                                        if diff_pot < 0: diff_pot = 0
                                        elif diff_pot > len_pot_minus_1: diff_pot = len_pot_minus_1
                                        pred_p2 = map_pot[diff_pot]
                                    else:
                                        diff_dep = int(round((ach_ce - t_pot) * inv_full_shift_dep * len_dep_minus_1))
                                        if diff_dep < 0: diff_dep = 0
                                        elif diff_dep > len_dep_minus_1: diff_dep = len_dep_minus_1
                                        pred_p2 = map_dep[diff_dep]
                                    cost_ce = float(p_ce) + pred_p2 + err_ce * 5000.0
                                
                        cand_p = p_fl if cost_fl <= cost_ce else p_ce
                        cand_cost = cost_fl if cost_fl <= cost_ce else cost_ce
                        if cand_cost < best_cost_pot:
                            best_cost_pot = cand_cost
                            best_p_pot = cand_p
                            best_v_idx_pot = v_idx
                            best_pw_idx_pot = pw_idx
                            
            if shift_pot <= tolerance:
                final_pulses_pot[r, c] = 0
                final_v_pot[r, c] = v_write_def
                final_pw_pot[r, c] = pw_pot_def
            elif best_p_pot > 0:
                final_pulses_pot[r, c] = best_p_pot
                final_v_pot[r, c] = v_write_range[best_v_idx_pot]
                final_pw_pot[r, c] = pw_range[best_pw_idx_pot]
            elif isCorrectionPass and min_err_p_pot > 0 and min_err_pot < shift_pot:
                final_pulses_pot[r, c] = min_err_p_pot
                final_v_pot[r, c] = v_write_range[min_err_v_idx_pot]
                final_pw_pot[r, c] = pw_range[min_err_pw_idx_pot]
            else:
                final_pulses_pot[r, c] = fallback_pulses_pot[r, c]
                final_v_pot[r, c] = v_write_def
                final_pw_pot[r, c] = pw_pot_def
                
            # -------------------------------------------------------------
            # DEPRESSION
            # -------------------------------------------------------------
            shift_dep_raw = (n_curr - t_pot) * inv_full_shift_dep
            if shift_dep_raw < 0.0:
                shift_dep = 0.0
            elif shift_dep_raw > 2.05:
                shift_dep = 2.05
            else:
                shift_dep = shift_dep_raw
                
            best_cost_dep = 1e30
            best_p_dep = 0
            best_v_idx_dep = 0
            best_pw_idx_dep = 0
            
            min_err_dep = 1e30
            min_err_p_dep = 0
            min_err_v_idx_dep = 0
            min_err_pw_idx_dep = 0
            
            for v_idx in range(V_len_erase):
                for pw_idx in range(PW_len_erase):
                    opt_val = optMapDep[v_idx, pw_idx]
                    if opt_val <= 0.0:
                        continue
                    if isCorrectionPass:
                        p_tgt = int(round(opt_val * shift_dep))
                        if p_tgt <= 0:
                            continue
                        raw_ach = n_curr - (float(p_tgt) / opt_val) * full_shift_dep
                        if raw_ach < min_dep - 1e-6:
                            continue
                        if raw_ach < t_pot and (t_pot - raw_ach) > (n_curr - t_pot) + 1e-6:
                            continue
                        err = abs(float(p_tgt) / opt_val - shift_dep)
                        if err < min_err_dep and p_tgt > 0:
                            min_err_dep = err
                            min_err_p_dep = p_tgt
                            min_err_v_idx_dep = v_idx
                            min_err_pw_idx_dep = pw_idx
                        if err < tolerance:
                            cost = float(p_tgt)
                            if cost < best_cost_dep:
                                best_cost_dep = cost
                                best_p_dep = p_tgt
                                best_v_idx_dep = v_idx
                                best_pw_idx_dep = pw_idx
                    else:
                        scaled_shift = opt_val * shift_dep
                        p_fl = int(math.floor(scaled_shift))
                        p_ce = int(math.ceil(scaled_shift))
                        
                        # Floor candidate (depression)
                        cost_fl = 1e30
                        if p_fl > 0:
                            frac_fl = float(p_fl) / opt_val
                            delta_n_fl = frac_fl * full_shift_dep
                            raw_ach_fl = n_curr - delta_n_fl
                            if raw_ach_fl >= min_dep - 1e-6:
                                ach_fl = raw_ach_fl
                                err_fl = abs(ach_fl - t_pot) * inv_full_shift_dep
                                if ach_fl >= t_pot:
                                    diff_dep = int(round((ach_fl - t_pot) * inv_full_shift_dep * len_dep_minus_1))
                                    if diff_dep < 0: diff_dep = 0
                                    elif diff_dep > len_dep_minus_1: diff_dep = len_dep_minus_1
                                    pred_p2_dep = map_dep[diff_dep]
                                else:
                                    diff_pot = int(round((t_pot - ach_fl) * inv_full_shift_pot * len_pot_minus_1))
                                    if diff_pot < 0: diff_pot = 0
                                    elif diff_pot > len_pot_minus_1: diff_pot = len_pot_minus_1
                                    pred_p2_dep = map_pot[diff_pot]
                                cost_fl = float(p_fl) + pred_p2_dep + err_fl * 5000.0
                                
                        # Ceil candidate (depression)
                        cost_ce = 1e30
                        if p_ce > 0:
                            frac_ce = float(p_ce) / opt_val
                            delta_n_ce = frac_ce * full_shift_dep
                            raw_ach_ce = n_curr - delta_n_ce
                            if raw_ach_ce >= min_dep - 1e-6:
                                if raw_ach_ce < t_pot and (t_pot - raw_ach_ce) > (n_curr - t_pot) + 1e-6:
                                    pass
                                else:
                                    ach_ce = raw_ach_ce
                                    err_ce = abs(ach_ce - t_pot) * inv_full_shift_dep
                                    if ach_ce >= t_pot:
                                        diff_dep = int(round((ach_ce - t_pot) * inv_full_shift_dep * len_dep_minus_1))
                                        if diff_dep < 0: diff_dep = 0
                                        elif diff_dep > len_dep_minus_1: diff_dep = len_dep_minus_1
                                        pred_p2_dep = map_dep[diff_dep]
                                    else:
                                        diff_pot = int(round((t_pot - ach_ce) * inv_full_shift_pot * len_pot_minus_1))
                                        if diff_pot < 0: diff_pot = 0
                                        elif diff_pot > len_pot_minus_1: diff_pot = len_pot_minus_1
                                        pred_p2_dep = map_pot[diff_pot]
                                    cost_ce = float(p_ce) + pred_p2_dep + err_ce * 5000.0
                                
                        cand_p = p_fl if cost_fl <= cost_ce else p_ce
                        cand_cost = cost_fl if cost_fl <= cost_ce else cost_ce
                        if cand_cost < best_cost_dep:
                            best_cost_dep = cand_cost
                            best_p_dep = cand_p
                            best_v_idx_dep = v_idx
                            best_pw_idx_dep = pw_idx
                            
            if shift_dep <= tolerance:
                final_pulses_dep[r, c] = 0
                final_v_dep[r, c] = v_erase_def
                final_pw_dep[r, c] = pw_dep_def
            elif best_p_dep > 0:
                final_pulses_dep[r, c] = best_p_dep
                final_v_dep[r, c] = v_erase_range[best_v_idx_dep]
                final_pw_dep[r, c] = pw_range[best_pw_idx_dep]
            elif isCorrectionPass and min_err_p_dep > 0 and min_err_dep < shift_dep:
                final_pulses_dep[r, c] = min_err_p_dep
                final_v_dep[r, c] = v_erase_range[min_err_v_idx_dep]
                final_pw_dep[r, c] = pw_range[min_err_pw_idx_dep]
            else:
                final_pulses_dep[r, c] = fallback_pulses_dep[r, c]
                final_v_dep[r, c] = v_erase_def
                final_pw_dep[r, c] = pw_dep_def
