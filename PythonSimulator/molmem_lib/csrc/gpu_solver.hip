#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <corecrt_search.h>
#include <xtr1common>
#endif
#include <type_traits>
#include <utility>
#include <limits>

#ifdef __HIP_PLATFORM_AMD__
#pragma push_macro("__clang__")
#undef __clang__
#pragma push_macro("_M_CEE")
#define _M_CEE 1
#include <cmath>
#include <math.h>
#pragma pop_macro("_M_CEE")
#pragma pop_macro("__clang__")

#include <__clang_hip_runtime_wrapper.h>
#include <hip/hip_runtime.h>
#else
#include <cmath>
#include <math.h>
#include <hip/hip_runtime.h>

#ifndef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernel, grid, block, shared, stream, ...) \
   hipLaunchKernelGGL(( kernel), dim3((grid)), dim3((block)), (shared), (stream), __VA_ARGS__)
#endif
typedef hipStream_t hipStream_t;
#endif

// Helper math functions for CUDA/HIP device kernel execution
template <typename T>
__device__ inline T clamp(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

template <typename T>
__device__ inline T interpolate_1d(T x, const T* __restrict__ yp, int num_points, T scale, T x_max=(T)33040.0) {
    if (num_points <= 1) {
        return yp[0];
    }
    T x_clipped = clamp(x, (T)0.0, x_max);
    T idx_float = (scale == (T)1.0) ? x_clipped : (x_clipped * scale);
    int idx = (int)idx_float;
    if (idx < 0) idx = 0;
    if (idx >= num_points - 1) idx = num_points - 2;
    T y0 = yp[idx];
    T y1 = yp[idx + 1];
    return fma(idx_float - (T)idx, y1 - y0, y0);
}

template <typename T>
__device__ inline T fast_pow_alpha(T base, T alpha) {
    if (base <= (T)0.0) return (T)0.0;
    if (alpha == (T)1.0) return base;
    if (alpha == (T)2.0) return base * base;
    if (alpha == (T)3.0) return (base * base) * base;
    if (alpha == (T)4.0) { T b2 = base * base; return b2 * b2; }
    if (alpha == (T)0.5) return sqrt(base);
    if (alpha == (T)0.0) return (T)1.0;
    return pow(base, alpha);
}

// 1. Memristor non-linear current and conductance solver (sinh/asinh NR)
template <typename T>
__device__ inline void compute_IGeq(
    T vApplied, T currentIScale, T currentVScale_eff, T currentF22, T rC,
    T& currentI, T& gEq
) {
    if (currentF22 == (T)0.0) {
        currentI = (T)0.0;
        gEq = (T)0.0;
        return;
    }
    
    T vS = currentVScale_eff + (T)1e-12;
    T rSeries = (T)2.0 * rC;
    T inv_vS = (T)1.0 / vS;
    T i_prod = currentIScale * currentF22;
    T g_base = i_prod * inv_vS;
    
    T vApplied_abs = (vApplied >= (T)0.0 ? vApplied : -vApplied);
    if (vApplied_abs < (T)1e-12) {
        currentI = (T)0.0;
        gEq = (rSeries == (T)0.0 ? g_base : (g_base / ((T)1.0 + g_base * rSeries)));
        return;
    }
    
    T v = vApplied * inv_vS;
    T coeff = rSeries * g_base;
    
    if (coeff <= (T)1e-18) {
        T v_clip = clamp(v, (T)-200.0, (T)200.0);
        T exp_v = exp(v_clip);
        T inv_exp_v = (T)1.0 / exp_v;
        T sinh_val = (T)0.5 * (exp_v - inv_exp_v);
        T cosh_val = (T)0.5 * (exp_v + inv_exp_v);
        currentI = i_prod * sinh_val;
        gEq = (rSeries == (T)0.0 ? (g_base * cosh_val) : ((g_base * cosh_val) / ((T)1.0 + (g_base * cosh_val) * rSeries)));
        return;
    }
    
    T sgn = (v >= (T)0.0 ? (T)1.0 : (T)-1.0);
    T abs_v = (v >= (T)0.0 ? v : -v);
    T inv_coeff = (T)1.0 / (coeff + (T)1e-20);
    T y = abs_v * inv_coeff;
    T asinh_y = asinh(y);
    T x = sgn * (abs_v < asinh_y ? abs_v : asinh_y);
    
    T sinh_x = (T)0.0;
    T cosh_x = (T)1.0;
    for (int iter = 0; iter < 4; ++iter) {
        T x_clip = clamp(x, (T)-20.0, (T)20.0);
        T exp_x = exp(x_clip);
        T inv_exp_x = (T)1.0 / exp_x;
        sinh_x = (T)0.5 * (exp_x - inv_exp_x);
        cosh_x = (T)0.5 * (exp_x + inv_exp_x);
        T f2 = coeff * sinh_x;
        T f = f2 + (x - v);
        T df = fma(coeff, cosh_x, (T)1.0);
        T denom = (T)2.0 * df * df - f * f2;
        T dx = (denom > (T)1e-12) ? ((T)2.0 * f * df / denom) : (f / df);
        x -= dx;
        T dx_abs = (dx >= (T)0.0 ? dx : -dx);
        if (dx_abs < (T)1e-12) {
            break;
        }
    }
    
    T abs_x = (x >= (T)0.0 ? x : -x);
    T sinh_val, cosh_val;
    if (abs_x <= (T)20.0) {
        sinh_val = sinh_x;
        cosh_val = cosh_x;
    } else {
        T x_clip = clamp(x, (T)-200.0, (T)200.0);
        T exp_x = exp(x_clip);
        T inv_exp_x = (T)1.0 / exp_x;
        sinh_val = (T)0.5 * (exp_x - inv_exp_x);
        cosh_val = (T)0.5 * (exp_x + inv_exp_x);
    }
    currentI = i_prod * sinh_val;
    T gCore = g_base * cosh_val;
    gEq = gCore / ((T)1.0 + gCore * rSeries);
}

template <typename T>
struct DeviceParams {
    T vSmooth, nSmooth, kappa, alpha, vTh, vRefPos, vRefNeg, nMax;
    T kDischarge, gScale, rC, f22StartVal, f22EndVal, f22PotScaleInv;
    T f22DepScaleInv, E_a, R_th, tau_th, gamma, beta_alpha, beta_s;
    T vBypass;
    T rate_asymmetry;
    T rTopBlend;
    T fDischarge;
    T cDischarge;
    T rBottomBlend;
    T rLatency;
    T k_overdrive;
    T inv_vSmooth;
    T inv_tau_th;
    T inv_vRefPos_diff;
    T inv_vRefNeg_diff;
};

struct GpuProfilerTelemetry {
    unsigned long long cyc_lut;
    unsigned long long cyc_thermal;
    unsigned long long cyc_rate;
    unsigned long long cyc_boundary;
    unsigned long long cyc_mode;
};

// 2. Physical state update equations
template <typename T>
__device__ inline void update_state_vars(
    T dt, T vApplied, T n, T modeState, T scaleFactor, T T_val,
    const DeviceParams<T>& params,
    const T* y_i, const T* y_v, const T* y_fp, const T* y_fd, int lookup_size, T lookup_scale,
    T& n_new, T& modeState_new, T& scaleFactor_new, T& T_new,
    int b, int d, T t,
    GpuProfilerTelemetry* prof = nullptr
) {
    T vApplied_abs = (vApplied >= (T)0.0 ? vApplied : -vApplied);
    if (vApplied_abs < (T)1e-6) {
        n_new = n;
        modeState_new = (modeState != (T)0.0) ? (modeState * exp(-dt * (T)1000.0)) : (T)0.0;
        scaleFactor_new = scaleFactor;
        T T_amb = (T)298.15;
        T T_diff_cool = (T_val - T_amb >= (T)0.0 ? T_val - T_amb : T_amb - T_val);
        T_new = (params.E_a > (T)0.0 && params.R_th > (T)0.0) ? ((T_diff_cool >= (T)1e-4) ? (T_amb + (T_val - T_amb) * exp(-dt * params.inv_tau_th)) : T_amb) : T_amb;
        return;
    }

    T vSmooth = params.vSmooth;
    T nSmooth = params.nSmooth;
    T kappa = params.kappa;
    T alpha = params.alpha;
    T vTh = params.vTh;
    T vRefPos = params.vRefPos;
    T vRefNeg = params.vRefNeg;
    T nMax = params.nMax;
    T kDischarge = params.kDischarge;
    T gScale = params.gScale;
    T rC = params.rC;
    T f22StartVal = params.f22StartVal;
    T f22EndVal = params.f22EndVal;
    T f22PotScaleInv = params.f22PotScaleInv;
    T f22DepScaleInv = params.f22DepScaleInv;
    T E_a = params.E_a;
    T R_th = params.R_th;
    T tau_th = params.tau_th;
    T gamma = params.gamma;
    T beta_alpha = params.beta_alpha;
    T beta_s = params.beta_s;
    T rate_asymmetry = params.rate_asymmetry;

    unsigned long long t_lut_start = 0;
    if (prof) t_lut_start = clock64();
    T iScale = interpolate_1d(n, y_i, lookup_size, lookup_scale, nMax);
    T currentIScale = iScale * gScale;
    T vScale_val = interpolate_1d(n, y_v, lookup_size, lookup_scale, nMax);

    T T_amb = (T)298.15;
    T vInt = vApplied;
    T alpha_eff = alpha;
    T n_smooth_eff = nSmooth;
    T gamma_eff = gamma;
    if (vInt > vRefPos && (beta_alpha != (T)0.0 || beta_s != (T)0.0)) {
        T d_val_pos = vRefPos - vTh + (T)1e-6;
        T delta_v = (vInt - vRefPos) / ((d_val_pos > (T)1e-15) ? d_val_pos : (T)1e-15);
        T tanh_delta_v = tanh(delta_v);
        alpha_eff = clamp(alpha + beta_alpha * tanh_delta_v, (T)0.1, (T)1000.0);
        n_smooth_eff = clamp(nSmooth * ((T)1.0 + beta_s * tanh_delta_v), (T)1.0, (T)1000.0);
    }

    // Compute currentF22 based on n, modeState, scaleFactor
    T f22DepRaw_val = interpolate_1d(nMax - n, y_fd, lookup_size, lookup_scale, nMax);
    T f22DepScaled_val = ((T)1.0 - f22DepRaw_val - f22EndVal) * f22DepScaleInv;
    T f22PotRaw_val = interpolate_1d(n, y_fp, lookup_size, lookup_scale, nMax);
    T f22PotScaled_val = (f22PotRaw_val - f22StartVal) * f22PotScaleInv;
    if (f22PotScaled_val > (T)0.0 && n > (T)16520.0) {
        T u_sat = (n - (T)16520.0) / (T)16520.0;
        if (u_sat > (T)1.0) u_sat = (T)1.0;
        f22PotScaled_val += (T)0.001 * n_smooth_eff * u_sat * u_sat * ((T)3.0 - (T)2.0 * u_sat);
    }
    T sDir_val = clamp(modeState, (T)0.0, (T)1.0);
    T f22_val = (sDir_val > (T)0.5) ? f22PotScaled_val : (f22DepScaled_val * scaleFactor);
    T currentF22 = (f22_val > (T)0.0 ? f22_val : (T)0.0);
    if (prof) prof->cyc_lut += (clock64() - t_lut_start);

    unsigned long long t_th_start = 0;
    if (prof) t_th_start = clock64();
    T T_new_heat = T_amb;
    T thermal_factor = (T)1.0;
    if (E_a > (T)0.0 && R_th > (T)0.0) {
        T f22_clamped = (currentF22 > (T)0.0 ? currentF22 : (T)0.0);
        T currentVScale_eff = (gamma_eff != (T)0.0) ? (vScale_val / ((T)1.0 + gamma_eff * f22_clamped)) : vScale_val;
        
        T currentI = (T)0.0;
        T gDummy = (T)0.0;
        compute_IGeq(vApplied, currentIScale, currentVScale_eff, f22_clamped, rC, currentI, gDummy);
        
        T P_diss = vApplied * currentI;
        if (P_diss < (T)0.0) P_diss = -P_diss;
        
        T T_steady = T_amb + P_diss * R_th;
        T T_diff = T_val - T_amb;
        if (T_diff < (T)0.0) T_diff = -T_diff;
        bool relaxed_mask = (T_diff < (T)1e-4) && (P_diss < (T)1e-9);
        
        T T_decay = relaxed_mask ? T_amb : (T_steady + (T_val - T_steady) * exp(-dt * params.inv_tau_th));
        T_new_heat = clamp(T_decay, T_amb, (T)1000.0);
        
        if (!relaxed_mask) {
            T T_avg = (T)0.5 * (T_val + T_new_heat);
            thermal_factor = exp((E_a * (T)11604.51812063) * ((T)0.00335401643468 - (T)1.0 / T_avg));
        }
    }
    if (prof) prof->cyc_thermal += (clock64() - t_th_start);
    
    unsigned long long t_rt_start = 0;
    if (prof) t_rt_start = clock64();
    T tanh_arg = vInt * params.inv_vSmooth;
    T vTanh;
    if (tanh_arg >= (T)19.0) {
        vTanh = (T)1.0;
    } else if (tanh_arg <= (T)-19.0) {
        vTanh = (T)-1.0;
    } else {
        vTanh = tanh(tanh_arg);
    }
    
    T vInt_abs = (vInt >= (T)0.0 ? vInt : -vInt);
    T arg = (vInt_abs - vTh) * params.inv_vSmooth;
    T softplus;
    if (arg <= (T)-36.0) {
        softplus = (T)0.0;
    } else if (arg >= (T)36.0) {
        softplus = arg;
    } else if (arg > (T)0.0) {
        softplus = arg + log1p(exp(-arg));
    } else {
        softplus = log1p(exp(arg));
    }
    
    T direction = vTanh;
    T dnDt = (T)0.0;
    
    if (softplus > (T)0.0) {
        T clampFactor = (T)1.0;
        // Direction-aware boundary clamping:
        // Clamps ONLY when moving into a boundary to prevent premature turn-on delays and preserve peak saturation
        if (vTanh > (T)0.0) {
            // Potentiation pushing toward nMax (saturation region governed by n_smooth_eff)
            T w_sat = ((T)1.5 * n_smooth_eff > (T)50.0) ? ((T)1.5 * n_smooth_eff) : (T)50.0;
            T dist_hi = nMax - n;
            if (dist_hi >= w_sat) {
                clampFactor = (T)1.0;
            } else if (dist_hi <= (T)0.0) {
                clampFactor = (T)0.0;
            } else {
                T z_hi = dist_hi / w_sat;
                clampFactor = z_hi * z_hi * ((T)3.0 - (T)2.0 * z_hi);
            }
        } else if (vTanh < (T)0.0) {
            // Depression pushing toward 0 (minimal numerical guard independent of nSmooth)
            T w_zero = (T)30.0;
            if (n >= w_zero) {
                clampFactor = (T)1.0;
            } else if (n <= (T)0.0) {
                clampFactor = (T)0.0;
            } else {
                T z_low = n / w_zero;
                clampFactor = z_low * z_low * ((T)3.0 - (T)2.0 * z_low);
            }
        } else {
            clampFactor = (T)1.0;
        }
        
        T kappa_clamp = kappa * clampFactor;
        if (vTanh == (T)1.0) {
            T base = (vSmooth * softplus) * params.inv_vRefPos_diff;
            T rateMag = fast_pow_alpha(base, alpha_eff);
            dnDt = rateMag * kappa_clamp;
        } else if (vTanh == (T)-1.0) {
            T base = (vSmooth * softplus) * params.inv_vRefNeg_diff;
            T rateMag = fast_pow_alpha(base, alpha_eff);
            dnDt = -rateMag * (kappa_clamp * rate_asymmetry);
        } else {
            T sPol = (T)0.5 * ((T)1.0 + vTanh);
            T vrefEff = sPol * vRefPos + ((T)1.0 - sPol) * vRefNeg;
            T denom = vrefEff - vTh + (T)1e-6;
            T vTanh_abs = (vTanh >= (T)0.0 ? vTanh : -vTanh);
            T base = (vSmooth * softplus * vTanh_abs) / (denom > (T)1e-15 ? denom : (T)1e-15);
            T rateMag = fast_pow_alpha(base, alpha_eff);
            T rate_scale = (direction > (T)0.0) ? (T)1.0 : rate_asymmetry;
            dnDt = rateMag * direction * (kappa_clamp * rate_scale);
        }
    }
    if (prof) prof->cyc_rate += (clock64() - t_rt_start);
    
    unsigned long long t_bd_start = 0;
    if (prof) t_bd_start = clock64();
    if ((modeState < (T)0.45 || vTanh < (T)0.0) && (n > (T)500.0)) {
        T n_base_nominal = (T)0.5 * nMax;
        T od_ratio = (params.cDischarge > (T)0.0) ? params.cDischarge : (T)0.0;
        T overdrive = (od_ratio > (T)0.0 && params.k_overdrive > (T)0.0) ? (od_ratio * params.k_overdrive) : (T)0.0;
        T od_clamp = (overdrive > (T)1.0) ? (T)1.0 : ((overdrive < (T)0.0) ? (T)0.0 : overdrive);

        T scale_base = ((params.rBottomBlend >= (T)0.05) && (params.rBottomBlend <= (T)2.5)) ? params.rBottomBlend : (T)1.0;
        T n_base = n_base_nominal * ((T)1.0 - od_clamp * ((T)1.0 - scale_base));
        T w_trans = (T)150.0;

        T u_base = (n - (n_base - w_trans)) / ((T)2.0 * w_trans);
        T s_base;
        if (u_base >= (T)1.0) {
            s_base = (T)1.0;
        } else if (u_base <= (T)0.0) {
            s_base = (T)0.0;
        } else {
            s_base = u_base * u_base * ((T)3.0 - (T)2.0 * u_base);
        }

        if (s_base > (T)0.0) {
            T n_turnaround = nMax;
            T excursion = n_turnaround - n_base;
            if (excursion < (T)100.0) excursion = (T)100.0;

            T u = (n - n_base) / excursion;
            u = (u < (T)0.0) ? (T)0.0 : ((u > (T)1.0) ? (T)1.0 : u);

            T scale_top = ((params.rTopBlend >= (T)0.001) && (params.rTopBlend <= (T)0.5)) ? params.rTopBlend : (T)0.05;
            T w_top = scale_top * excursion;
            if (w_top < (T)30.0) w_top = (T)30.0;
            T dist_from_peak = n_turnaround - n;
            T s_top = (T)1.0;
            if (dist_from_peak <= (T)0.0) {
                s_top = (T)0.0;
            } else if (dist_from_peak < w_top) {
                T z_top = dist_from_peak / w_top;
                s_top = z_top * z_top * ((T)3.0 - (T)2.0 * z_top);
            }

            // Continuous Monotonic Relaxation with Voltage Overdrive Latency Gating:
            T r_lat = ((params.rLatency >= (T)0.0) && (params.rLatency < (T)0.999)) ? params.rLatency : (T)0.5;
            T r_lat_eff = ((T)1.0 - od_clamp > (T)0.0) ? (r_lat * ((T)1.0 - od_clamp)) : (T)0.0;
            T k_amp = (kDischarge > (T)0.0) ? kDischarge : (T)0.0;
            T p_decay = ((params.fDischarge >= (T)0.05) && (params.fDischarge <= (T)0.95)) ? ((T)1.0 / params.fDischarge) : (T)2.5;
            T u_decay = (u > (T)0.0) ? pow(u, p_decay) : (T)0.0;
            T eff_rate_factor = ((T)1.0 - r_lat_eff) + (r_lat_eff + k_amp * s_top) * u_decay;

            T eff_mult = (T)1.0 - s_base * ((T)1.0 - eff_rate_factor);
            dnDt = dnDt * eff_mult;
        }
    }
    
    T dnDt_scaled = dnDt * thermal_factor;
    T n_new_heat = clamp(n + dnDt_scaled * dt, (T)0.0, nMax);
    if (prof) prof->cyc_boundary += (clock64() - t_bd_start);
    
    unsigned long long t_md_start = 0;
    if (prof) t_md_start = clock64();
    T modeState_new_heat;
    if (vInt_abs <= vTh) {
        modeState_new_heat = (modeState != (T)0.0) ? (modeState * exp(-dt * (T)1000.0)) : (T)0.0;
    } else {
        T targetMode;
        if (vInt > vTh) {
            targetMode = (T)1.0;
        } else {
            targetMode = (T)0.0;
        }
        T vEffTargetMode = targetMode * (T)0.9999990000009999;
        if (dt >= (T)20e-9) {
            modeState_new_heat = vEffTargetMode;
        } else {
            T exp_arg_mode = -dt * (T)1.000001e9;
            T exp_factor_mode = (exp_arg_mode < (T)-20.0) ? (T)0.0 : exp(exp_arg_mode);
            modeState_new_heat = vEffTargetMode + (modeState - vEffTargetMode) * exp_factor_mode;
        }
    }
    
    T denomScale = (f22DepScaled_val > (T)1e-6 ? f22DepScaled_val : (T)1e-6);
    T ratio = clamp(f22PotScaled_val / denomScale, (T)0.0, (T)100.0);
    T exp_arg_scale = -dt * (T)1e10;
    T exp_factor_scale = (dt >= (T)2e-9) ? (T)0.0 : ((exp_arg_scale < (T)-20.0) ? (T)0.0 : exp(exp_arg_scale));
    
    T scaleFactor_new_heat = (modeState_new_heat > (T)0.6 && vTanh > (T)0.0) ? (ratio + (scaleFactor - ratio) * exp_factor_scale) : scaleFactor;
    if (!isfinite(scaleFactor_new_heat)) {
        scaleFactor_new_heat = (T)1.0;
    }
    if (prof) prof->cyc_mode += (clock64() - t_md_start);
    

    
    n_new = n_new_heat;
    modeState_new = modeState_new_heat;
    scaleFactor_new = scaleFactor_new_heat;
    T_new = T_new_heat;
}

// 3. Piecewise Linear PWL signal evaluator on GPU with monotonic O(1) cache prefetching
template <typename T>
__device__ inline T evaluate_pwl_value(
    T t, const T* __restrict__ t_matrix, const T* __restrict__ v_matrix, const int* __restrict__ mask, int source_idx, int max_points, int b, int pad_sources, int& cache_idx
) {
    if (mask[source_idx] == 0) return (T)0.0;
    
    int base_offset = (b * pad_sources + source_idx) * max_points;
    const T* __restrict__ t_src = t_matrix + base_offset;
    const T* __restrict__ v_src = v_matrix + base_offset;
    
    int idx = cache_idx;
    if (idx >= max_points) idx = max_points - 1;
    if (idx < 0) idx = 0;
    
    if (t_src[idx] <= t) {
        while (idx + 1 < max_points && t_src[idx + 1] <= t) {
            idx++;
        }
    } else {
        int low = 0;
        int high = idx;
        idx = 0;
        while (low <= high) {
            int mid = low + ((high - low) >> 1);
            if (t_src[mid] <= t) {
                idx = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    cache_idx = idx;
    T t0 = t_src[idx];
    T t1 = (idx + 1 < max_points) ? t_src[idx + 1] : t0;
    T dt_seg = t1 - t0;
    T v0 = v_src[idx];
    T v1 = (idx + 1 < max_points) ? v_src[idx + 1] : v0;
    if (dt_seg < (T)1e-15 || t >= t1) return v1;
    return v0 + (t - t0) * (v1 - v0) / dt_seg;
}

template <typename T, int MAX_DEVICES_VAL>
struct DeviceStateArray {
    T* ptr;
    int batch_size;
    int b;

    __device__ __forceinline__ DeviceStateArray() : ptr(nullptr), batch_size(0), b(0) {}
    __device__ __forceinline__ DeviceStateArray(T* p, int bs, int b_idx) : ptr(p), batch_size(bs), b(b_idx) {}

    __device__ __forceinline__ void init(T* p, int bs, int b_idx) {
        ptr = p;
        batch_size = bs;
        b = b_idx;
    }

    __device__ __forceinline__ T& operator[](int d) {
        if (MAX_DEVICES_VAL > 0) {
            return ptr[d];
        } else {
            return ptr[d * batch_size + b];
        }
    }

    __device__ __forceinline__ const T& operator[](int d) const {
        if (MAX_DEVICES_VAL > 0) {
            return ptr[d];
        } else {
            return ptr[d * batch_size + b];
        }
    }
};

// 4. Custom C++/HIP general simulation kernel template
template <typename T, typename T_cg, int MAX_NODES, int MAX_DEVICES>
__global__ void run_general_transient_kernel(
    T t_start, T t_end, T min_dt, T tol, int maxIter, int max_steps,
    const T* __restrict__ master_schedule, const int* __restrict__ is_subsample, int num_edges,
    T* __restrict__ device_states_soa,
    const T* __restrict__ device_params_soa,
    int numNodes, int num_devices,
    const int* __restrict__ dev_top_full_idx, const int* __restrict__ dev_bot_full_idx,
    const int* __restrict__ dev_top_idx, const int* __restrict__ dev_bot_idx,
    const int* __restrict__ gMat_row_indices, const int* __restrict__ gMat_col_indices, const int* __restrict__ iRes_indices,
    const int* __restrict__ top_indices, const int* __restrict__ bot_indices, const int* __restrict__ both_indices, const int* __restrict__ dev_both_idx,
    int L_top, int L_bot, int L_both,
    const int* __restrict__ d_nodes, const int* __restrict__ dyn_indices, const int* __restrict__ dev_to_dyn_src,
    const int* __restrict__ v_nodes_gpu, const T* __restrict__ v_pwl_t_matrix, const T* __restrict__ v_pwl_v_matrix, const int* __restrict__ v_pwl_mask_gpu,
    const int* __restrict__ dyn_nodes_gpu, const T* __restrict__ dyn_pwl_t_matrix, const T* __restrict__ dyn_pwl_v_matrix, const int* __restrict__ dyn_pwl_mask_gpu,
    bool is_1t1r, bool recordHistory, T leakage_g,
    int num_v_sources, int num_dyn_sources, int num_all_nodes,
    // History buffers
    T* __restrict__ tArr_buffer, T* __restrict__ vHist_buffer, T* __restrict__ iHist_buffer,
    T* __restrict__ stateHist_buffer, T* __restrict__ f22Hist_buffer, T* __restrict__ gHist_buffer,
    T* __restrict__ tempHist_buffer,
    // System dimensions
    int num_gMat_vals,
    int max_pwl_points,
    int batch_size,
    // Lookup tables
    const T* __restrict__ y_i,
    const T* __restrict__ y_v,
    const T* __restrict__ y_fp,
    const T* __restrict__ y_fd,
    int lookup_size,
    T* __restrict__ d_out,
    T_cg* __restrict__ global_workspace_cg = nullptr,
    T* __restrict__ global_workspace_t = nullptr
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    int pad_sources = num_v_sources > num_dyn_sources ? num_v_sources : num_dyn_sources;
    if (pad_sources < 2) pad_sources = 2;
    
    // Allocate local thread variables for device state parameters using unified stride indexing wrapper
    DeviceStateArray<T, MAX_DEVICES> local_n;
    DeviceStateArray<T, MAX_DEVICES> local_mode;
    DeviceStateArray<T, MAX_DEVICES> local_scale;
    DeviceStateArray<T, MAX_DEVICES> local_f22Dep;
    DeviceStateArray<T, MAX_DEVICES> local_f22Pot;
    DeviceStateArray<T, MAX_DEVICES> local_cached_n;
    DeviceStateArray<T, MAX_DEVICES> local_iScale;
    DeviceStateArray<T, MAX_DEVICES> local_vScale;
    DeviceStateArray<T, MAX_DEVICES> local_currentF22;
    DeviceStateArray<T, MAX_DEVICES> local_currentIScale;
    DeviceStateArray<T, MAX_DEVICES> local_currentVScale;
    DeviceStateArray<T, MAX_DEVICES> local_T;

    DeviceStateArray<T, MAX_DEVICES> next_n;
    DeviceStateArray<T, MAX_DEVICES> next_mode;
    DeviceStateArray<T, MAX_DEVICES> next_scale;
    DeviceStateArray<T, MAX_DEVICES> next_T;

    // Declare pointers for solver arrays
    T* gMat_val;
    T* diag_g_orig;
    T* diag_g_reg;
    T* iRes;
    T_cg* deltaV;
    T_cg* solver_r;
    T_cg* solver_z;
    T_cg* solver_p;
    T_cg* solver_Ap;
    T_cg* inv_diag_g_reg;
    T* vCurrent;
    DeviceStateArray<T, MAX_DEVICES> local_iDev;
    DeviceStateArray<T, MAX_DEVICES> local_gEq;

    // Stack arrays for compile-time sized models (MAX_DEVICES > 0). Dynamic size mode sets to size 1 to avoid stack footprint.
    T stack_local_n[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_mode[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_scale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_f22Dep[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_f22Pot[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_cached_n[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_iScale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_vScale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_currentF22[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_currentIScale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_currentVScale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_local_T[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    
    T stack_next_n[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_next_mode[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_next_scale[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_next_T[MAX_DEVICES > 0 ? MAX_DEVICES : 1];

    T local_gMat_val[MAX_DEVICES > 0 ? (MAX_DEVICES << 2) : 1];
    T local_diag_g_orig[MAX_NODES > 0 ? MAX_NODES : 1];
    T local_diag_g_reg[MAX_NODES > 0 ? MAX_NODES : 1];
    T local_iRes[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_deltaV[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_solver_r[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_solver_z[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_solver_p[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_solver_Ap[MAX_NODES > 0 ? MAX_NODES : 1];
    T_cg local_inv_diag_g_reg[MAX_NODES > 0 ? MAX_NODES : 1];
    T local_vCurrent[MAX_NODES > 0 ? MAX_NODES + 128 : 1];
    T stack_iDev[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    T stack_gEq[MAX_DEVICES > 0 ? MAX_DEVICES : 1];

    // Assign pointers based on template bounds
    if (MAX_NODES > 0) {
        diag_g_orig = local_diag_g_orig;
        diag_g_reg = local_diag_g_reg;
        iRes = local_iRes;
        deltaV = local_deltaV;
        solver_r = local_solver_r;
        solver_z = local_solver_z;
        solver_p = local_solver_p;
        solver_Ap = local_solver_Ap;
        inv_diag_g_reg = local_inv_diag_g_reg;
        vCurrent = local_vCurrent;
    } else {
        deltaV = global_workspace_cg + (0 * numNodes) * batch_size + b * numNodes;
        solver_r = global_workspace_cg + (1 * numNodes) * batch_size + b * numNodes;
        solver_z = global_workspace_cg + (2 * numNodes) * batch_size + b * numNodes;
        solver_p = global_workspace_cg + (3 * numNodes) * batch_size + b * numNodes;
        solver_Ap = global_workspace_cg + (4 * numNodes) * batch_size + b * numNodes;
        inv_diag_g_reg = global_workspace_cg + (5 * numNodes) * batch_size + b * numNodes;
        
        diag_g_orig = global_workspace_t + num_gMat_vals * batch_size + b * numNodes;
        diag_g_reg = global_workspace_t + (num_gMat_vals + 1 * numNodes) * batch_size + b * numNodes;
        iRes = global_workspace_t + (num_gMat_vals + 2 * numNodes) * batch_size + b * numNodes;
        vCurrent = global_workspace_t + (num_gMat_vals + 3 * numNodes) * batch_size + b * (numNodes + num_v_sources + pad_sources);
    }
    
    if (MAX_DEVICES > 0) {
        gMat_val = local_gMat_val;
        local_iDev.init(stack_iDev, 0, 0);
        local_gEq.init(stack_gEq, 0, 0);
        
        local_n.init(stack_local_n, 0, 0);
        local_mode.init(stack_local_mode, 0, 0);
        local_scale.init(stack_local_scale, 0, 0);
        local_f22Dep.init(stack_local_f22Dep, 0, 0);
        local_f22Pot.init(stack_local_f22Pot, 0, 0);
        local_cached_n.init(stack_local_cached_n, 0, 0);
        local_iScale.init(stack_local_iScale, 0, 0);
        local_vScale.init(stack_local_vScale, 0, 0);
        local_currentF22.init(stack_local_currentF22, 0, 0);
        local_currentIScale.init(stack_local_currentIScale, 0, 0);
        local_currentVScale.init(stack_local_currentVScale, 0, 0);
        local_T.init(stack_local_T, 0, 0);
        
        next_n.init(stack_next_n, 0, 0);
        next_mode.init(stack_next_mode, 0, 0);
        next_scale.init(stack_next_scale, 0, 0);
        next_T.init(stack_next_T, 0, 0);
    } else {
        gMat_val = global_workspace_t + (0 * num_gMat_vals) * batch_size + b * num_gMat_vals;
        
        T* base_dev = global_workspace_t + (num_gMat_vals + 4 * numNodes + num_v_sources + pad_sources) * batch_size;
        local_iDev.init(base_dev + 0 * num_devices * batch_size, batch_size, b);
        local_gEq.init(base_dev + 1 * num_devices * batch_size, batch_size, b);
        
        local_n.init(base_dev + 2 * num_devices * batch_size, batch_size, b);
        local_mode.init(base_dev + 3 * num_devices * batch_size, batch_size, b);
        local_scale.init(base_dev + 4 * num_devices * batch_size, batch_size, b);
        local_f22Dep.init(base_dev + 5 * num_devices * batch_size, batch_size, b);
        local_f22Pot.init(base_dev + 6 * num_devices * batch_size, batch_size, b);
        local_cached_n.init(base_dev + 7 * num_devices * batch_size, batch_size, b);
        local_iScale.init(base_dev + 8 * num_devices * batch_size, batch_size, b);
        local_vScale.init(base_dev + 9 * num_devices * batch_size, batch_size, b);
        local_currentF22.init(base_dev + 10 * num_devices * batch_size, batch_size, b);
        local_currentIScale.init(base_dev + 11 * num_devices * batch_size, batch_size, b);
        local_currentVScale.init(base_dev + 12 * num_devices * batch_size, batch_size, b);
        local_T.init(base_dev + 13 * num_devices * batch_size, batch_size, b);
        
        next_n.init(base_dev + 14 * num_devices * batch_size, batch_size, b);
        next_mode.init(base_dev + 15 * num_devices * batch_size, batch_size, b);
        next_scale.init(base_dev + 16 * num_devices * batch_size, batch_size, b);
        next_T.init(base_dev + 17 * num_devices * batch_size, batch_size, b);
    }

    __shared__ DeviceParams<T> local_params[MAX_DEVICES > 0 ? MAX_DEVICES : 1];
    
    if (MAX_DEVICES > 0) {
        for (int d = threadIdx.x; d < num_devices; d += blockDim.x) {
            local_params[d].vSmooth = device_params_soa[0 * num_devices + d];
            local_params[d].nSmooth = device_params_soa[1 * num_devices + d];
            local_params[d].kappa = device_params_soa[2 * num_devices + d];
            local_params[d].alpha = device_params_soa[3 * num_devices + d];
            local_params[d].vTh = device_params_soa[4 * num_devices + d];
            local_params[d].vRefPos = device_params_soa[5 * num_devices + d];
            local_params[d].vRefNeg = device_params_soa[6 * num_devices + d];
            local_params[d].nMax = device_params_soa[7 * num_devices + d];
            local_params[d].kDischarge = device_params_soa[8 * num_devices + d];
            local_params[d].gScale = device_params_soa[9 * num_devices + d];
            local_params[d].rC = device_params_soa[10 * num_devices + d];
            local_params[d].f22StartVal = device_params_soa[11 * num_devices + d];
            local_params[d].f22EndVal = device_params_soa[13 * num_devices + d];
            local_params[d].f22PotScaleInv = device_params_soa[14 * num_devices + d];
            local_params[d].f22DepScaleInv = device_params_soa[15 * num_devices + d];
            local_params[d].E_a = device_params_soa[16 * num_devices + d];
            local_params[d].R_th = device_params_soa[17 * num_devices + d];
            local_params[d].tau_th = device_params_soa[18 * num_devices + d];
            local_params[d].gamma = device_params_soa[19 * num_devices + d];
            local_params[d].beta_alpha = device_params_soa[20 * num_devices + d];
            local_params[d].beta_s = device_params_soa[21 * num_devices + d];
            local_params[d].vBypass = device_params_soa[22 * num_devices + d];
            local_params[d].rate_asymmetry = device_params_soa[23 * num_devices + d];
            local_params[d].rTopBlend = device_params_soa[24 * num_devices + d];
            local_params[d].fDischarge = device_params_soa[25 * num_devices + d];
            local_params[d].cDischarge = device_params_soa[26 * num_devices + d];
            local_params[d].rBottomBlend = device_params_soa[27 * num_devices + d];
            local_params[d].rLatency = device_params_soa[28 * num_devices + d];
            local_params[d].k_overdrive = device_params_soa[29 * num_devices + d];
            local_params[d].inv_vSmooth = (local_params[d].vSmooth != (T)0.0) ? (T)1.0 / local_params[d].vSmooth : (T)0.0;
            local_params[d].inv_tau_th = (local_params[d].tau_th != (T)0.0) ? (T)1.0 / local_params[d].tau_th : (T)0.0;
            local_params[d].inv_vRefPos_diff = (T)1.0 / (local_params[d].vRefPos - local_params[d].vTh + (T)1e-6);
            local_params[d].inv_vRefNeg_diff = (T)1.0 / (local_params[d].vRefNeg - local_params[d].vTh + (T)1e-6);
        }
        __syncthreads();
    }
    
    DeviceParams<T> dev0_params;
    if (num_devices == 1) {
        dev0_params.vSmooth = device_params_soa[0];
        dev0_params.nSmooth = device_params_soa[1];
        dev0_params.kappa = device_params_soa[2];
        dev0_params.alpha = device_params_soa[3];
        dev0_params.vTh = device_params_soa[4];
        dev0_params.vRefPos = device_params_soa[5];
        dev0_params.vRefNeg = device_params_soa[6];
        dev0_params.nMax = device_params_soa[7];
        dev0_params.kDischarge = device_params_soa[8];
        dev0_params.gScale = device_params_soa[9];
        dev0_params.rC = device_params_soa[10];
        dev0_params.f22StartVal = device_params_soa[11];
        dev0_params.f22EndVal = device_params_soa[13];
        dev0_params.f22PotScaleInv = device_params_soa[14];
        dev0_params.f22DepScaleInv = device_params_soa[15];
        dev0_params.E_a = device_params_soa[16];
        dev0_params.R_th = device_params_soa[17];
        dev0_params.tau_th = device_params_soa[18];
        dev0_params.gamma = device_params_soa[19];
        dev0_params.beta_alpha = device_params_soa[20];
        dev0_params.beta_s = device_params_soa[21];
        dev0_params.vBypass = device_params_soa[22];
        dev0_params.rate_asymmetry = device_params_soa[23];
        dev0_params.rTopBlend = device_params_soa[24];
        dev0_params.fDischarge = device_params_soa[25];
        dev0_params.cDischarge = device_params_soa[26];
        dev0_params.rBottomBlend = device_params_soa[27];
        dev0_params.rLatency = device_params_soa[28];
        dev0_params.k_overdrive = device_params_soa[29];
        dev0_params.inv_vSmooth = (dev0_params.vSmooth != (T)0.0) ? (T)1.0 / dev0_params.vSmooth : (T)0.0;
        dev0_params.inv_tau_th = (dev0_params.tau_th != (T)0.0) ? (T)1.0 / dev0_params.tau_th : (T)0.0;
        dev0_params.inv_vRefPos_diff = (T)1.0 / (dev0_params.vRefPos - dev0_params.vTh + (T)1e-6);
        dev0_params.inv_vRefNeg_diff = (T)1.0 / (dev0_params.vRefNeg - dev0_params.vTh + (T)1e-6);
    }
    
    T nMax_ref = (MAX_DEVICES > 0) ? local_params[0].nMax : (num_devices == 1 ? dev0_params.nMax : (num_devices > 0 ? device_params_soa[7 * num_devices] : (T)33040.0));
    T lookup_scale = (lookup_size == (int)(nMax_ref + (T)1.0)) ? (T)1.0 : ((T)(lookup_size - 1) / (nMax_ref > (T)0.0 ? nMax_ref : (T)33040.0));

    if (b >= batch_size) return;

    // Populate initial device states directly into the resolved pointers
    for (int d = 0; d < num_devices; ++d) {
        int offset_d = d * batch_size + b;
        local_n[d] = device_states_soa[0 * num_devices * batch_size + offset_d];
        local_mode[d] = device_states_soa[1 * num_devices * batch_size + offset_d];
        local_scale[d] = device_states_soa[2 * num_devices * batch_size + offset_d];
        local_f22Dep[d] = device_states_soa[3 * num_devices * batch_size + offset_d];
        local_f22Pot[d] = device_states_soa[4 * num_devices * batch_size + offset_d];
        local_cached_n[d] = device_states_soa[5 * num_devices * batch_size + offset_d];
        if (local_cached_n[d] < (T)0.0) {
            local_cached_n[d] = (T)0.0;
        }
        local_iScale[d] = device_states_soa[6 * num_devices * batch_size + offset_d];
        local_vScale[d] = device_states_soa[7 * num_devices * batch_size + offset_d];
        local_currentF22[d] = device_states_soa[8 * num_devices * batch_size + offset_d];
        local_currentIScale[d] = device_states_soa[9 * num_devices * batch_size + offset_d];
        local_currentVScale[d] = device_states_soa[10 * num_devices * batch_size + offset_d];
        local_T[d] = device_states_soa[11 * num_devices * batch_size + offset_d];
        
        T n_init = local_n[d];
        T f22StartVal_init = (MAX_DEVICES > 0) ? local_params[d].f22StartVal : device_params_soa[11 * num_devices + d];
        T f22EndVal_init = (MAX_DEVICES > 0) ? local_params[d].f22EndVal : device_params_soa[13 * num_devices + d];
        T f22PotScaleInv_init = (MAX_DEVICES > 0) ? local_params[d].f22PotScaleInv : device_params_soa[14 * num_devices + d];
        T f22DepScaleInv_init = (MAX_DEVICES > 0) ? local_params[d].f22DepScaleInv : device_params_soa[15 * num_devices + d];
        T gScale_init = (MAX_DEVICES > 0) ? local_params[d].gScale : device_params_soa[9 * num_devices + d];
        
        if (local_f22Dep[d] <= (T)0.0) {
            T f22DepRaw = interpolate_1d((T)33040.0 - n_init, y_fd, lookup_size, lookup_scale);
            local_f22Dep[d] = ((T)1.0 - f22DepRaw - f22EndVal_init) * f22DepScaleInv_init;
        }
        if (local_f22Pot[d] <= (T)0.0) {
            T f22PotRaw = interpolate_1d(n_init, y_fp, lookup_size, lookup_scale);
            T pot_val = (f22PotRaw - f22StartVal_init) * f22PotScaleInv_init;
            T nSmooth_init = (MAX_DEVICES > 0) ? local_params[d].nSmooth : device_params_soa[1 * num_devices + d];
            if (pot_val > (T)0.0 && n_init > (T)16520.0) {
                T u_sat = (n_init - (T)16520.0) / (T)16520.0;
                if (u_sat > (T)1.0) u_sat = (T)1.0;
                pot_val += (T)0.001 * nSmooth_init * u_sat * u_sat * ((T)3.0 - (T)2.0 * u_sat);
            }
            local_f22Pot[d] = pot_val;
        }
        if (local_scale[d] <= (T)0.0) {
            T denom_s = (local_f22Dep[d] > (T)1e-6) ? local_f22Dep[d] : (T)1e-6;
            local_scale[d] = local_f22Pot[d] / denom_s;
        }
        if (local_iScale[d] <= (T)0.0) {
            local_iScale[d] = interpolate_1d(n_init, y_i, lookup_size, lookup_scale);
            local_currentIScale[d] = local_iScale[d] * gScale_init;
        }
        if (local_vScale[d] <= (T)0.0) {
            local_vScale[d] = interpolate_1d(n_init, y_v, lookup_size, lookup_scale);
            local_currentVScale[d] = local_vScale[d];
        }
        if (local_currentF22[d] <= (T)0.0) {
            bool is_pot_init = (local_mode[d] > (T)0.5);
            T f22_comm = is_pot_init ? local_f22Pot[d] : (local_f22Dep[d] * local_scale[d]);
            local_currentF22[d] = (f22_comm > (T)0.0 ? f22_comm : (T)0.0);
        }
    }
    
    int v_pwl_cache[128];
    int dyn_pwl_cache[128];
    for (int i = 0; i < 128; ++i) {
        v_pwl_cache[i] = 0;
        dyn_pwl_cache[i] = 0;
    }
    
    for (int n = 0; n < numNodes + num_v_sources + pad_sources; ++n) {
        vCurrent[n] = (T)0.0;
    }
    
    T* prev_dn = global_workspace_t + (num_gMat_vals + 4 * numNodes + num_v_sources + pad_sources + 20 * num_devices) * batch_size + b * num_devices;
    T* prev_prev_dn = global_workspace_t + (num_gMat_vals + 4 * numNodes + num_v_sources + pad_sources + 21 * num_devices) * batch_size + b * num_devices;
    T* prev_v = global_workspace_t + (num_gMat_vals + 4 * numNodes + num_v_sources + pad_sources + 22 * num_devices) * batch_size + b * (numNodes + num_v_sources + pad_sources);
    T* dynamic_G = global_workspace_t + (num_gMat_vals + 5 * numNodes + 2 * (num_v_sources + pad_sources) + 22 * num_devices) * batch_size + b * pad_sources;
    T* dynamic_I = global_workspace_t + (num_gMat_vals + 5 * numNodes + 3 * (num_v_sources + pad_sources) + 22 * num_devices) * batch_size + b * pad_sources;
    for (int s = 0; s < pad_sources; ++s) {
        dynamic_G[s] = (T)0.0;
        dynamic_I[s] = (T)0.0;
    }
    int step_count = 0;
    for (int d = 0; d < num_devices; ++d) {
        prev_dn[d] = (T)0.0;
        prev_prev_dn[d] = (T)0.0;
    }
    for (int n = 0; n < numNodes + num_v_sources + pad_sources; ++n) {
        prev_v[n] = (T)0.0;
    }

    T t = t_start;
    T dt = (T)1e-7;
    int hist_idx = (int)d_out[0];
    T max_dt = (sizeof(T) == 8 ? (T)1e-3 : (T)1e-5);
    int schedule_ptr = 0;
    bool step_rejected = false;
    // Record initial step 0 (t = t_start)
    for (int s = 0; s < num_v_sources; ++s) {
        int dummy_cache = 0;
        int& c_idx = (s < 128) ? v_pwl_cache[s] : dummy_cache;
        T val = evaluate_pwl_value(t, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask_gpu, s, max_pwl_points, b, pad_sources, c_idx);
        vCurrent[v_nodes_gpu[s]] = val;
    }
    for (int node_i = 0; node_i < numNodes + num_v_sources + pad_sources; ++node_i) {
        prev_v[node_i] = vCurrent[node_i];
    }
    if (hist_idx == 0) {
        if (numNodes == 0) {
            for (int d = 0; d < num_devices; ++d) {
                int top_idx = dev_top_full_idx[d];
                int bot_idx = dev_bot_full_idx[d];
                T v_app = vCurrent[top_idx] - vCurrent[bot_idx];
                T currentF22 = local_currentF22[d];
                T currentIScale = local_currentIScale[d];
                T currentVScale = local_currentVScale[d];
                T rC = (MAX_DEVICES > 0) ? local_params[d].rC : (num_devices == 1 ? dev0_params.rC : device_params_soa[10 * num_devices + d]);
                T currentI = (T)0.0;
                T gEq = (T)0.0;
                T v_app_abs = (v_app >= (T)0.0 ? v_app : -v_app);
                if (v_app_abs >= (T)1e-12 && currentF22 > (T)0.0) {
                    compute_IGeq(v_app, currentIScale, currentVScale, currentF22, rC, currentI, gEq);
                }
                if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                    int src_idx = dev_to_dyn_src[d];
                    if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                        currentI = (T)0.0;
                        gEq = (T)1e-25;
                    }
                }
                local_iDev[d] = currentI;
                local_gEq[d] = gEq;
            }
        }
        if (recordHistory) {
            int time_offset = b * max_steps + 0;
            tArr_buffer[time_offset] = t;
            for (int n = 0; n < num_all_nodes; ++n) {
                int offset = (b * num_all_nodes + n) * max_steps + 0;
                vHist_buffer[offset] = (n < numNodes + num_v_sources + pad_sources) ? vCurrent[n] : (T)0.0;
            }
            if (num_devices == 1) {
                int offset = b * max_steps + 0;
                stateHist_buffer[offset] = local_n[0];
                f22Hist_buffer[offset] = local_currentF22[0];
                gHist_buffer[offset] = (local_currentF22[0] > (T)0.0) ? (local_currentIScale[0] * local_currentF22[0]) / (local_currentVScale[0] + (T)1e-12) : (T)0.0;
                tempHist_buffer[offset] = local_T[0];
                iHist_buffer[offset] = local_iDev[0];
            } else {
                for (int d = 0; d < num_devices; ++d) {
                    int offset = (b * num_devices + d) * max_steps + 0;
                    stateHist_buffer[offset] = local_n[d];
                    f22Hist_buffer[offset] = local_currentF22[d];
                    gHist_buffer[offset] = (local_currentF22[d] > (T)0.0) ? (local_currentIScale[d] * local_currentF22[d]) / (local_currentVScale[d] + (T)1e-12) : (T)0.0;
                    tempHist_buffer[offset] = local_T[d];
                    iHist_buffer[offset] = local_iDev[d];
                }
            }
            hist_idx = 1;
        } else {
            int time_offset = b * max_steps + 0;
            tArr_buffer[time_offset] = t;
            hist_idx = 1;
        }
    }
    bool underflow_warned = false;
    unsigned long long prof_t0 = 0;
    unsigned long long prof_cyc_pwl = 0;
    unsigned long long prof_cyc_assembly = 0;
    unsigned long long prof_cyc_cg = 0;
    unsigned long long prof_cyc_ode = 0;
    unsigned long long prof_cyc_hist = 0;
    unsigned long long prof_cg_iters = 0;
    unsigned long long prof_steps = 0;
    GpuProfilerTelemetry prof_ode = {0, 0, 0, 0, 0};
    unsigned long long prof_cyc_pred_full = 0;
    unsigned long long prof_cyc_pred_half = 0;
    unsigned long long prof_cyc_lte_err = 0;
    unsigned long long prof_cyc_state_commit = 0;
    if (b == 0) {
        prof_t0 = clock64();
    }
    while (t < t_end) {
        if (t >= t_end - (T)1e-14) {
            t = t_end;
            break;
        }
        // 0. Align dt with next schedule edge to hit transitions exactly
        int prev_schedule_ptr = schedule_ptr;
        while (schedule_ptr < num_edges && master_schedule[schedule_ptr] <= t + (T)1e-18) {
            schedule_ptr++;
        }

        // 1. Evaluate voltage and dynamic sources at current step time t
        unsigned long long t_pwl_start = (b == 0) ? clock64() : 0;
        for (int s = 0; s < num_v_sources; ++s) {
            int dummy_cache = 0;
            int& c_idx = (s < 128) ? v_pwl_cache[s] : dummy_cache;
            T val = evaluate_pwl_value(t, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask_gpu, s, max_pwl_points, b, pad_sources, c_idx);
            vCurrent[v_nodes_gpu[s]] = val;
        }
        
        for (int s = 0; s < num_dyn_sources; ++s) {
            int dummy_cache = 0;
            int& c_idx = (s < 128) ? dyn_pwl_cache[s] : dummy_cache;
            T val = evaluate_pwl_value(t, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask_gpu, s, max_pwl_points, b, pad_sources, c_idx);
            if (val > (T)1e19) {
                dynamic_G[s] = leakage_g;
                dynamic_I[s] = (T)0.0;
            } else {
                dynamic_G[s] = (T)1e5;
                dynamic_I[s] = val * (T)1e5;
            }
        }
        if (b == 0) {
            prof_cyc_pwl += (clock64() - t_pwl_start);
        }

        // Evaluate source voltages at current step to check v_constant
        bool v_constant = (step_count != 0);
        if (step_count == 0 || schedule_ptr != prev_schedule_ptr) {
            int total_v_nodes = numNodes + num_v_sources + pad_sources;
            if (total_v_nodes == 1) {
                T diff_v = vCurrent[0] - prev_v[0];
                if (diff_v < (T)0.0) diff_v = -diff_v;
                if (diff_v > (T)1e-6) v_constant = false;
            } else {
                for (int node_i = 0; node_i < total_v_nodes; ++node_i) {
                    T diff_v = vCurrent[node_i] - prev_v[node_i];
                    if (diff_v < (T)0.0) diff_v = -diff_v;
                    if (diff_v > (T)1e-6) {
                        v_constant = false;
                        break;
                    }
                }
            }
        }

        int target_schedule_ptr = schedule_ptr;
        bool skipped_subsample = false;
        if (v_constant && schedule_ptr < num_edges && is_subsample[schedule_ptr] != 0 && !step_rejected) {
            if (num_devices > 0) {
                bool is_linear = true;
                for (int dev_i = 0; dev_i < num_devices; ++dev_i) {
                    T curr_dn = prev_dn[dev_i];
                    if (curr_dn < (T)0.0) curr_dn = -curr_dn;
                    T nMax = (MAX_DEVICES > 0) ? local_params[dev_i].nMax : device_params_soa[7 * num_devices + dev_i];
                    T lin_tol = tol * nMax;
                    if (step_count > 1) {
                        T diff_dn = prev_dn[dev_i] - prev_prev_dn[dev_i];
                        if (diff_dn < (T)0.0) diff_dn = -diff_dn;
                        if (diff_dn > lin_tol) {
                            is_linear = false;
                            break;
                        }
                    } else {
                        int top_idx = dev_top_full_idx[dev_i];
                        int bot_idx = dev_bot_full_idx[dev_i];
                        T v_app = vCurrent[top_idx] - vCurrent[bot_idx];
                        if (v_app < (T)0.0) v_app = -v_app;
                        T v_th = (MAX_DEVICES > 0) ? local_params[dev_i].vTh : device_params_soa[4 * num_devices + dev_i];
                        T inv_vSmooth = (MAX_DEVICES > 0) ? local_params[dev_i].inv_vSmooth : ((device_params_soa[0 * num_devices + dev_i] != (T)0.0) ? (T)1.0 / device_params_soa[0 * num_devices + dev_i] : (T)0.0);
                        T softplus_val = (v_app > v_th) ? ((v_app - v_th) * inv_vSmooth) : (T)0.0;
                        T kappa = (MAX_DEVICES > 0) ? local_params[dev_i].kappa : device_params_soa[2 * num_devices + dev_i];
                        T alpha = (MAX_DEVICES > 0) ? local_params[dev_i].alpha : device_params_soa[3 * num_devices + dev_i];
                        T k_eff = (v_app > v_th) ? (kappa * fast_pow_alpha(softplus_val, alpha)) : (T)1000.0;
                        if (k_eff * curr_dn * dt > lin_tol || (v_app > v_th && curr_dn == (T)0.0)) {
                            is_linear = false;
                            break;
                        }
                    }
                }
                if (is_linear) {
                    while (target_schedule_ptr < num_edges - 1 && is_subsample[target_schedule_ptr] != 0) {
                        target_schedule_ptr++;
                    }
                    skipped_subsample = true;
                }
            }
        }

        T tNextEdge = t_end;
        if (target_schedule_ptr < num_edges && master_schedule[target_schedule_ptr] < (T)1e19) {
            tNextEdge = master_schedule[target_schedule_ptr];
        }
        
        T time_to_edge = tNextEdge - t;
        if (time_to_edge <= (T)1e-15) {
            while (schedule_ptr < num_edges && master_schedule[schedule_ptr] <= t + (T)1e-15) {
                schedule_ptr++;
            }
            if (schedule_ptr < num_edges && master_schedule[schedule_ptr] < (T)1e19) {
                tNextEdge = master_schedule[schedule_ptr];
            } else {
                tNextEdge = t_end;
            }
            time_to_edge = tNextEdge - t;
            if (time_to_edge <= (T)1e-15) {
                time_to_edge = min_dt;
            }
        }
        
        if (skipped_subsample && !step_rejected) {
            dt = (time_to_edge > min_dt ? time_to_edge : min_dt);
        } else {
            T max_dt_val = (time_to_edge < max_dt ? time_to_edge : max_dt);
            if (dt > max_dt_val) {
                dt = max_dt_val;
            }
        }
        if (dt < min_dt) {
            dt = min_dt;
        }
        
        if (t + dt >= t_end - (T)1e-14) {
            dt = t_end - t;
        }
        
        // 3. Newton-Raphson Solver Loop
        if (numNodes > 0) {
            for (int n = 0; n < numNodes; ++n) {
                deltaV[n] = (T)0.0;
            }
            
            bool converged = false;
            for (int it = 0; it < maxIter; ++it) {
                unsigned long long t_ass_start = (b == 0) ? clock64() : 0;
                // Initialize equations
                for (int n = 0; n < numNodes; ++n) {
                    iRes[n] = (T)0.0;
                    diag_g_orig[n] = leakage_g;
                    diag_g_reg[n] = leakage_g + (T)1e-12;
                }
                for (int m = 0; m < num_gMat_vals; ++m) {
                    gMat_val[m] = (T)0.0;
                }
                
                // Assemble devices into equations
                for (int d = 0; d < num_devices; ++d) {
                    int top_n = dev_top_full_idx[d];
                    int bot_n = dev_bot_full_idx[d];
                    T vApplied = vCurrent[top_n] - vCurrent[bot_n];
                    
                    T currentIScale = local_currentIScale[d];
                    T currentVScale = local_currentVScale[d];
                    T currentF22 = local_currentF22[d];
                    T rC = (MAX_DEVICES > 0) ? local_params[d].rC : (num_devices == 1 ? dev0_params.rC : device_params_soa[10 * num_devices + d]);
                    
                    T iDev_val = (T)0.0;
                    T gEq_val = (T)0.0;
                    compute_IGeq(vApplied, currentIScale, currentVScale, currentF22, rC, iDev_val, gEq_val);
                    if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                        int src_idx = dev_to_dyn_src[d];
                        if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                            iDev_val = (T)0.0;
                            gEq_val = (T)1e-25;
                        }
                    }
                    
                    local_iDev[d] = iDev_val;
                    local_gEq[d] = gEq_val;
                    
                    int top_unk = dev_top_idx[d];
                    int bot_unk = dev_bot_idx[d];
                    
                    if (top_unk >= 0) {
                        iRes[top_unk] += iDev_val;
                        diag_g_orig[top_unk] += gEq_val;
                        diag_g_reg[top_unk] += gEq_val;
                    }
                    if (bot_unk >= 0) {
                        iRes[bot_unk] -= iDev_val;
                        diag_g_orig[bot_unk] += gEq_val;
                        diag_g_reg[bot_unk] += gEq_val;
                    }
                    
                    int k = dev_both_idx[d];
                    if (k >= 0) {
                        gMat_val[L_top + L_bot + k] = -gEq_val;
                        gMat_val[L_top + L_bot + L_both + k] = -gEq_val;
                    }
                }
                
                // Assemble dynamic source connections
                for (int s = 0; s < num_dyn_sources; ++s) {
                    int node = d_nodes[s];
                    int idx = dyn_indices[s];
                    T val_dyn = vCurrent[node] * dynamic_G[s] - dynamic_I[s];
                    if (node >= 0) {
                        iRes[node] += val_dyn;
                        diag_g_orig[node] += dynamic_G[s];
                        diag_g_reg[node] += dynamic_G[s];
                    }
                }
                
                T_cg max_res = (T_cg)0.0;
                for (int n = 0; n < numNodes; ++n) {
                    T_cg res_val = (T_cg)iRes[n];
                    T_cg res_abs = (res_val >= (T_cg)0.0 ? res_val : -res_val);
                    if (res_abs > max_res) max_res = res_abs;
                }
                if (b == 0) {
                    prof_cyc_assembly += (clock64() - t_ass_start);
                }
                if (max_res < (T_cg)1e-12) {
                    converged = true;
                    break;
                }
                
                unsigned long long t_cg_start = (b == 0) ? clock64() : 0;
                if (numNodes == 1) {
                    T_cg d0 = (T_cg)diag_g_reg[0];
                    deltaV[0] = -(T_cg)iRes[0] / (d0 >= (T_cg)1e-18 ? d0 : (T_cg)1e-18);
                } else if (numNodes == 2) {
                    T_cg g00 = (T_cg)diag_g_reg[0];
                    T_cg g11 = (T_cg)diag_g_reg[1];
                    T_cg g01 = (T_cg)0.0;
                    T_cg g10 = (T_cg)0.0;
                    for (int d = 0; d < num_gMat_vals; ++d) {
                        int row = gMat_row_indices[d];
                        int col = gMat_col_indices[d];
                        if (row == 0 && col == 1) g01 += (T_cg)gMat_val[d];
                        if (row == 1 && col == 0) g10 += (T_cg)gMat_val[d];
                    }
                    T_cg det = g00 * g11 - g01 * g10;
                    T_cg det_abs = (det >= (T_cg)0.0 ? det : -det);
                    T_cg inv_det = (det_abs > (T_cg)1e-30) ? ((T_cg)1.0 / det) : (T_cg)0.0;
                    deltaV[0] = ((T_cg)iRes[1] * g01 - (T_cg)iRes[0] * g11) * inv_det;
                    deltaV[1] = ((T_cg)iRes[0] * g10 - (T_cg)iRes[1] * g00) * inv_det;
                } else if (numNodes == 3) {
                    T_cg A = (T_cg)diag_g_reg[0];
                    T_cg E = (T_cg)diag_g_reg[1];
                    T_cg I = (T_cg)diag_g_reg[2];
                    T_cg B = (T_cg)0.0, C = (T_cg)0.0, D = (T_cg)0.0, F = (T_cg)0.0, G = (T_cg)0.0, H = (T_cg)0.0;
                    for (int d = 0; d < num_gMat_vals; ++d) {
                        int row = gMat_row_indices[d];
                        int col = gMat_col_indices[d];
                        if (row == 0 && col == 1) B += (T_cg)gMat_val[d];
                        else if (row == 0 && col == 2) C += (T_cg)gMat_val[d];
                        else if (row == 1 && col == 0) D += (T_cg)gMat_val[d];
                        else if (row == 1 && col == 2) F += (T_cg)gMat_val[d];
                        else if (row == 2 && col == 0) G += (T_cg)gMat_val[d];
                        else if (row == 2 && col == 1) H += (T_cg)gMat_val[d];
                    }
                    T_cg c00 = E * I - F * H;
                    T_cg c01 = F * G - D * I;
                    T_cg c02 = D * H - E * G;
                    T_cg det = A * c00 + B * c01 + C * c02;
                    T_cg det_abs = (det >= (T_cg)0.0 ? det : -det);
                    if (det_abs > (T_cg)1e-30) {
                        T_cg inv_det = (T_cg)1.0 / det;
                        T_cg c10 = C * H - B * I;
                        T_cg c11 = A * I - C * G;
                        T_cg c12 = B * G - A * H;
                        T_cg c20 = B * F - C * E;
                        T_cg c21 = C * D - A * F;
                        T_cg c22 = A * E - B * D;
                        
                        T_cg r0 = -(T_cg)iRes[0], r1 = -(T_cg)iRes[1], r2 = -(T_cg)iRes[2];
                        deltaV[0] = (r0 * c00 + r1 * c10 + r2 * c20) * inv_det;
                        deltaV[1] = (r0 * c01 + r1 * c11 + r2 * c21) * inv_det;
                        deltaV[2] = (r0 * c02 + r1 * c12 + r2 * c22) * inv_det;
                    } else {
                        deltaV[0] = (T_cg)0.0;
                        deltaV[1] = (T_cg)0.0;
                        deltaV[2] = (T_cg)0.0;
                    }
                } else {
                    // Solve: diag_g_reg * deltaV + gMat_val * deltaV = -iRes using Conjugate Gradient (CG)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_r[n] = -(T_cg)iRes[n];
                        deltaV[n] = (T_cg)0.0;
                        inv_diag_g_reg[n] = (T_cg)1.0 / (diag_g_reg[n] >= (T)1e-18 ? (T_cg)diag_g_reg[n] : (T_cg)1e-18);
                    }
                    
                    T_cg max_r0 = (T_cg)0.0;
                    for (int n = 0; n < numNodes; ++n) {
                        T_cg r_abs = solver_r[n] >= (T_cg)0.0 ? solver_r[n] : -solver_r[n];
                        if (r_abs > max_r0) max_r0 = r_abs;
                    }
                    T tol_cg_val = (tol > (T)1e-3 ? tol : (T)1e-3);
                    T_cg tol_cg = (T_cg)tol_cg_val;
                    T_cg cg_limit = tol_cg * max_r0;
                    T_cg min_cg_limit = (sizeof(T) == 8) ? (T_cg)1e-28 : (T_cg)1e-13;
                    if (cg_limit < min_cg_limit) cg_limit = min_cg_limit;
                    
                    // Polynomial preconditioner step:
                    // z0 = inv_diag * r (store in solver_z)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_z[n] = solver_r[n] * inv_diag_g_reg[n];
                    }
                    
                    // v = A * z0 (store in solver_Ap)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_Ap[n] = (T_cg)diag_g_reg[n] * solver_z[n];
                    }
                    for (int d = 0; d < num_gMat_vals; ++d) {
                        int row = gMat_row_indices[d];
                        int col = gMat_col_indices[d];
                        if (row >= 0 && col >= 0 && row != col) {
                            solver_Ap[row] += (T_cg)gMat_val[d] * solver_z[col];
                        }
                    }
                    
                    // u = -9.0909 * z0 + 3.3670 * inv_diag * v (store in solver_z)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_z[n] = -(T_cg)9.0909 * solver_z[n] + (T_cg)3.3670 * inv_diag_g_reg[n] * solver_Ap[n];
                    }
                    
                    // w = A * u (store in solver_Ap)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_Ap[n] = (T_cg)diag_g_reg[n] * solver_z[n];
                    }
                    for (int d = 0; d < num_gMat_vals; ++d) {
                        int row = gMat_row_indices[d];
                        int col = gMat_col_indices[d];
                        if (row >= 0 && col >= 0 && row != col) {
                            solver_Ap[row] += (T_cg)gMat_val[d] * solver_z[col];
                        }
                    }
                    
                    // z_new = inv_diag * (6.5657 * r + w) (store in solver_z)
                    for (int n = 0; n < numNodes; ++n) {
                        solver_z[n] = inv_diag_g_reg[n] * ((T_cg)6.5657 * solver_r[n] + solver_Ap[n]);
                    }
                    
                    // p = z
                    for (int n = 0; n < numNodes; ++n) {
                        solver_p[n] = solver_z[n];
                    }
                    
                    T_cg rz = (T_cg)0.0;
                    for (int n = 0; n < numNodes; ++n) {
                        rz += solver_r[n] * solver_z[n];
                    }
                    
                    for (int cg_it = 0; cg_it < 40; ++cg_it) {
                        T_cg max_res = (T_cg)0.0;
                        for (int n = 0; n < numNodes; ++n) {
                            T_cg res_abs = solver_r[n] >= (T_cg)0.0 ? solver_r[n] : -solver_r[n];
                            if (res_abs > max_res) max_res = res_abs;
                        }
                        if (max_res < cg_limit) break;
                        
                        // Ap = A * p
                        for (int n = 0; n < numNodes; ++n) {
                            solver_Ap[n] = (T_cg)diag_g_reg[n] * solver_p[n];
                        }
                        for (int d = 0; d < num_gMat_vals; ++d) {
                            int row = gMat_row_indices[d];
                            int col = gMat_col_indices[d];
                            if (row >= 0 && col >= 0 && row != col) {
                                solver_Ap[row] += (T_cg)gMat_val[d] * solver_p[col];
                            }
                        }
                        
                        T_cg pAp = (T_cg)0.0;
                        for (int n = 0; n < numNodes; ++n) {
                            pAp += solver_p[n] * solver_Ap[n];
                        }
                        if (pAp >= (T_cg)0.0) {
                            if (pAp < (T_cg)1e-20) pAp = (T_cg)1e-20;
                        } else {
                            if (pAp > (T_cg)-1e-20) pAp = (T_cg)-1e-20;
                        }
                        
                        T_cg alpha_cg = rz / pAp;
                        for (int n = 0; n < numNodes; ++n) {
                            deltaV[n] += alpha_cg * solver_p[n];
                            solver_r[n] -= alpha_cg * solver_Ap[n];
                        }
                        
                        // z0_new = inv_diag * r (store in solver_z)
                        for (int n = 0; n < numNodes; ++n) {
                            solver_z[n] = solver_r[n] * inv_diag_g_reg[n];
                        }
                        
                        // v = A * z0_new (store in solver_Ap)
                        for (int n = 0; n < numNodes; ++n) {
                            solver_Ap[n] = (T_cg)diag_g_reg[n] * solver_z[n];
                        }
                        for (int d = 0; d < num_gMat_vals; ++d) {
                            int row = gMat_row_indices[d];
                            int col = gMat_col_indices[d];
                            if (row >= 0 && col >= 0 && row != col) {
                                solver_Ap[row] += (T_cg)gMat_val[d] * solver_z[col];
                            }
                        }
                        
                        // u = -9.0909 * z0_new + 3.3670 * inv_diag * v (store in solver_z)
                        for (int n = 0; n < numNodes; ++n) {
                            solver_z[n] = -(T_cg)9.0909 * solver_z[n] + (T_cg)3.3670 * inv_diag_g_reg[n] * solver_Ap[n];
                        }
                        
                        // w = A * u (store in solver_Ap)
                        for (int n = 0; n < numNodes; ++n) {
                            solver_Ap[n] = (T_cg)diag_g_reg[n] * solver_z[n];
                        }
                        for (int d = 0; d < num_gMat_vals; ++d) {
                            int row = gMat_row_indices[d];
                            int col = gMat_col_indices[d];
                            if (row >= 0 && col >= 0 && row != col) {
                                solver_Ap[row] += (T_cg)gMat_val[d] * solver_z[col];
                            }
                        }
                        
                        // z_new = inv_diag * (6.5657 * r + w) (store in solver_z)
                        for (int n = 0; n < numNodes; ++n) {
                            solver_z[n] = inv_diag_g_reg[n] * ((T_cg)6.5657 * solver_r[n] + solver_Ap[n]);
                        }
                        
                        T_cg rz_new = (T_cg)0.0;
                        for (int n = 0; n < numNodes; ++n) {
                            rz_new += solver_r[n] * solver_z[n];
                        }
                        
                        T_cg rz_clamped = rz >= (T_cg)0.0 ? (rz < (T_cg)1e-20 ? (T_cg)1e-20 : rz) : (rz > (T_cg)-1e-20 ? (T_cg)-1e-20 : rz);
                        T_cg beta_cg = rz_new / rz_clamped;
                        for (int n = 0; n < numNodes; ++n) {
                            solver_p[n] = solver_z[n] + beta_cg * solver_p[n];
                        }
                        rz = rz_new;
                    }
                }
                if (b == 0) {
                    prof_cyc_cg += (clock64() - t_cg_start);
                }
                
                T max_dv = (T)0.0;
                if (numNodes == 1) {
                    T dv_val = (T)deltaV[0];
                    vCurrent[0] += dv_val;
                    max_dv = (dv_val >= (T)0.0 ? dv_val : -dv_val);
                } else if (numNodes == 2) {
                    T dv0 = (T)deltaV[0];
                    T dv1 = (T)deltaV[1];
                    vCurrent[0] += dv0;
                    vCurrent[1] += dv1;
                    T abs_dv0 = (dv0 >= (T)0.0 ? dv0 : -dv0);
                    T abs_dv1 = (dv1 >= (T)0.0 ? dv1 : -dv1);
                    max_dv = (abs_dv0 > abs_dv1 ? abs_dv0 : abs_dv1);
                } else if (numNodes == 3) {
                    T dv0 = (T)deltaV[0];
                    T dv1 = (T)deltaV[1];
                    T dv2 = (T)deltaV[2];
                    vCurrent[0] += dv0;
                    vCurrent[1] += dv1;
                    vCurrent[2] += dv2;
                    T abs_dv0 = (dv0 >= (T)0.0 ? dv0 : -dv0);
                    T abs_dv1 = (dv1 >= (T)0.0 ? dv1 : -dv1);
                    T abs_dv2 = (dv2 >= (T)0.0 ? dv2 : -dv2);
                    T max01 = (abs_dv0 > abs_dv1 ? abs_dv0 : abs_dv1);
                    max_dv = (max01 > abs_dv2 ? max01 : abs_dv2);
                } else {
                    for (int n = 0; n < numNodes; ++n) {
                        T dv_val = (T)deltaV[n];
                        vCurrent[n] += dv_val;
                        T deltaV_abs = (dv_val >= (T)0.0 ? dv_val : -dv_val);
                        if (deltaV_abs > max_dv) {
                            max_dv = deltaV_abs;
                        }
                    }
                }
                
                if (max_dv < tol) {
                    converged = true;
                    break;
                }
            }
            
            if (!converged && dt > min_dt) {
                dt = (dt * (T)0.5 > min_dt ? dt * (T)0.5 : min_dt);
                step_rejected = true;
                continue;
            }
        } else {
            for (int d = 0; d < num_devices; ++d) {
                int top_n = dev_top_full_idx[d];
                int bot_n = dev_bot_full_idx[d];
                T vApplied = vCurrent[top_n] - vCurrent[bot_n];
                
                T currentIScale = local_currentIScale[d];
                T currentVScale = local_currentVScale[d];
                T currentF22 = local_currentF22[d];
                T rC = (MAX_DEVICES > 0) ? local_params[d].rC : device_params_soa[10 * num_devices + d];
                
                T iDev_val = (T)0.0;
                T gEq_val = (T)0.0;
                compute_IGeq(vApplied, currentIScale, currentVScale, currentF22, rC, iDev_val, gEq_val);
                if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                    int src_idx = dev_to_dyn_src[d];
                    if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                        iDev_val = (T)0.0;
                        gEq_val = (T)1e-25;
                    }
                }
                local_iDev[d] = iDev_val;
                local_gEq[d] = gEq_val;
            }
        }
        
        // 4. State Integration and Predictor-Corrector timestepping
        unsigned long long t_ode_start = (b == 0) ? clock64() : 0;
        bool step_accepted = true;
        T max_err = (T)0.0;
        
        T max_vdev = (T)0.0;
        for (int d = 0; d < num_devices; ++d) {
            int top_n = dev_top_full_idx[d];
            int bot_n = dev_bot_full_idx[d];
            T vApplied = vCurrent[top_n] - vCurrent[bot_n];
            if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                int src_idx = dev_to_dyn_src[d];
                if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                    vApplied = (T)0.0;
                }
            }
            T vApplied_abs = (vApplied >= (T)0.0 ? vApplied : -vApplied);
            if (vApplied_abs > max_vdev) {
                max_vdev = vApplied_abs;
            }
        }
        
        for (int d = 0; d < num_devices; ++d) {
            int top_n = dev_top_full_idx[d];
            int bot_n = dev_bot_full_idx[d];
            T vApplied = vCurrent[top_n] - vCurrent[bot_n];
            
            DeviceParams<T> params_d;
            if (MAX_DEVICES > 0) {
                params_d = local_params[d];
            } else if (num_devices == 1) {
                params_d = dev0_params;
            } else {
                params_d.vSmooth = device_params_soa[0 * num_devices + d];
                params_d.nSmooth = device_params_soa[1 * num_devices + d];
                params_d.kappa = device_params_soa[2 * num_devices + d];
                params_d.alpha = device_params_soa[3 * num_devices + d];
                params_d.vTh = device_params_soa[4 * num_devices + d];
                params_d.vRefPos = device_params_soa[5 * num_devices + d];
                params_d.vRefNeg = device_params_soa[6 * num_devices + d];
                params_d.nMax = device_params_soa[7 * num_devices + d];
                params_d.kDischarge = device_params_soa[8 * num_devices + d];
                params_d.gScale = device_params_soa[9 * num_devices + d];
                params_d.rC = device_params_soa[10 * num_devices + d];
                params_d.f22StartVal = device_params_soa[11 * num_devices + d];
                params_d.f22EndVal = device_params_soa[13 * num_devices + d];
                params_d.f22PotScaleInv = device_params_soa[14 * num_devices + d];
                params_d.f22DepScaleInv = device_params_soa[15 * num_devices + d];
                params_d.E_a = device_params_soa[16 * num_devices + d];
                params_d.R_th = device_params_soa[17 * num_devices + d];
                params_d.tau_th = device_params_soa[18 * num_devices + d];
                params_d.gamma = device_params_soa[19 * num_devices + d];
                params_d.beta_alpha = device_params_soa[20 * num_devices + d];
                params_d.beta_s = device_params_soa[21 * num_devices + d];
                params_d.vBypass = device_params_soa[22 * num_devices + d];
                params_d.rate_asymmetry = device_params_soa[23 * num_devices + d];
                params_d.rTopBlend = device_params_soa[24 * num_devices + d];
                params_d.fDischarge = device_params_soa[25 * num_devices + d];
                params_d.cDischarge = device_params_soa[26 * num_devices + d];
                params_d.rBottomBlend = device_params_soa[27 * num_devices + d];
                params_d.rLatency = device_params_soa[28 * num_devices + d];
                params_d.k_overdrive = device_params_soa[29 * num_devices + d];
                params_d.inv_vSmooth = (params_d.vSmooth != (T)0.0) ? (T)1.0 / params_d.vSmooth : (T)0.0;
                params_d.inv_tau_th = (params_d.tau_th != (T)0.0) ? (T)1.0 / params_d.tau_th : (T)0.0;
                params_d.inv_vRefPos_diff = (T)1.0 / (params_d.vRefPos - params_d.vTh + (T)1e-6);
                params_d.inv_vRefNeg_diff = (T)1.0 / (params_d.vRefNeg - params_d.vTh + (T)1e-6);
            }



            if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                int src_idx = dev_to_dyn_src[d];
                if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                    vApplied = (T)0.0;
                }
            }
            
            T vApplied_abs = (vApplied >= (T)0.0 ? vApplied : -vApplied);
            T vBypass = params_d.vBypass;
            T nMax = params_d.nMax;
            T vTh = params_d.vTh;
            T E_a = params_d.E_a;
            T R_th = params_d.R_th;
            T tau_th = params_d.tau_th;
            T kDischarge = params_d.kDischarge;
            T T_amb = (T)298.15;
            
            // 1. Cool path early exit (applied voltage < 1e-6)
            if (vApplied_abs < (T)1e-6) {
                next_n[d] = local_n[d];
                T latch_cool = local_mode[d] * exp(-dt * (T)1000.0);
                next_mode[d] = (local_mode[d] == (T)0.0) ? (T)0.0 : latch_cool;
                next_scale[d] = local_scale[d];
                next_T[d] = (E_a > (T)0.0 && R_th > (T)0.0) ? clamp(T_amb + (local_T[d] - T_amb) * exp(-dt * params_d.inv_tau_th), T_amb, (T)1000.0) : T_amb;
                continue;
            }
            
            // 2. Sub-threshold early exit (applied voltage < vBypass)
            T T_diff = local_T[d] - T_amb;
            if (T_diff < (T)0.0) T_diff = -T_diff;
            if (vApplied_abs < vBypass) {
                T od_ratio = (params_d.cDischarge > (T)0.0) ? params_d.cDischarge : (T)0.0;
                T od_val = od_ratio * params_d.k_overdrive;
                T od_clamp = (od_val > (T)1.0) ? (T)1.0 : ((od_val < (T)0.0) ? (T)0.0 : od_val);
                T rBottom = ((params_d.rBottomBlend >= (T)0.05) && (params_d.rBottomBlend <= (T)2.5)) ? params_d.rBottomBlend : (T)1.0;
                T n_base_d = (T)0.5 * params_d.nMax * ((T)1.0 - od_clamp * ((T)1.0 - rBottom));
                if (local_mode[d] < (T)1e-9 && T_diff < (T)1e-4 && (local_n[d] <= n_base_d || kDischarge == (T)0.0)) {
                    next_n[d] = local_n[d];
                    next_mode[d] = (local_mode[d] > (T)0.0) ? (local_mode[d] * exp(-dt * (T)1000.0)) : (T)0.0;
                    next_scale[d] = local_scale[d];
                    next_T[d] = (E_a > (T)0.0 && R_th > (T)0.0) ? ((T_diff >= (T)1e-4) ? clamp(T_amb + (local_T[d] - T_amb) * exp(-dt * params_d.inv_tau_th), T_amb, (T)1000.0) : T_amb) : T_amb;
                    continue;
                }
            }
            
            // 3. Saturated boundary early exit
            if (((local_n[d] >= nMax) && (vApplied >= vTh)) || ((local_n[d] <= (T)0.0) && (vApplied <= -vTh))) {
                next_n[d] = (vApplied > (T)0.0) ? nMax : (T)0.0;
                T targetMode = (vApplied > vTh) ? (T)1.0 : (T)0.0;
                next_mode[d] = targetMode * (T)0.9999990000009999;
                
                if (vApplied > (T)0.0) {
                    T f22StartVal = params_d.f22StartVal;
                    T f22EndVal = params_d.f22EndVal;
                    T f22PotScaleInv = params_d.f22PotScaleInv;
                    T f22DepScaleInv = params_d.f22DepScaleInv;
                    T f22DepRaw_val = interpolate_1d(params_d.nMax - local_n[d], y_fd, lookup_size, lookup_scale, params_d.nMax);
                    T f22DepScaled_val = ((T)1.0 - f22DepRaw_val - f22EndVal) * f22DepScaleInv;
                    T f22PotRaw_val = interpolate_1d(local_n[d], y_fp, lookup_size, lookup_scale, params_d.nMax);
                    T f22PotScaled_val = (f22PotRaw_val - f22StartVal) * f22PotScaleInv;
                    T n_sm_eff = params_d.nSmooth;
                    if (vApplied > params_d.vRefPos && params_d.beta_s != (T)0.0) {
                        T d_val_pos = params_d.vRefPos - params_d.vTh + (T)1e-6;
                        T delta_v = (vApplied - params_d.vRefPos) / ((d_val_pos > (T)1e-15) ? d_val_pos : (T)1e-15);
                        n_sm_eff = clamp(params_d.nSmooth * ((T)1.0 + params_d.beta_s * tanh(delta_v)), (T)1.0, (T)1000.0);
                    }
                    if (f22PotScaled_val > (T)0.0 && local_n[d] > (T)16520.0) {
                        T u_sat = (local_n[d] - (T)16520.0) / (T)16520.0;
                        if (u_sat > (T)1.0) u_sat = (T)1.0;
                        f22PotScaled_val += (T)0.001 * n_sm_eff * u_sat * u_sat * ((T)3.0 - (T)2.0 * u_sat);
                    }
                    T denom_sat = (f22DepScaled_val > (T)1e-6 ? f22DepScaled_val : (T)1e-6);
                    T ratio_sat = f22PotScaled_val / denom_sat;
                    next_scale[d] = ratio_sat;
                } else {
                    next_scale[d] = local_scale[d];
                }
                
                if (E_a > (T)0.0 && R_th > (T)0.0) {
                    T rC = params_d.rC;
                    T currentI_sat = (T)0.0;
                    T gDummy_sat = (T)0.0;
                    compute_IGeq(vApplied, local_currentIScale[d], local_currentVScale[d], local_currentF22[d], rC, currentI_sat, gDummy_sat);
                    T P_diss_sat = vApplied * currentI_sat;
                    if (P_diss_sat < (T)0.0) P_diss_sat = -P_diss_sat;
                    T T_steady_sat = T_amb + P_diss_sat * R_th;
                    next_T[d] = clamp(T_steady_sat + (local_T[d] - T_steady_sat) * exp(-dt * params_d.inv_tau_th), T_amb, (T)1000.0);
                } else {
                    next_T[d] = T_amb;
                }
                continue;
            }
            
            T n_s, mode_s, scale_s, T_s;
            unsigned long long t_pf = (b == 0) ? clock64() : 0;
            params_d.cDischarge = local_cached_n[d];
            update_state_vars(
                dt, vApplied, local_n[d], local_mode[d], local_scale[d], local_T[d],
                params_d,
                y_i, y_v, y_fp, y_fd, lookup_size, lookup_scale,
                n_s, mode_s, scale_s, T_s,
                b, d, t,
                (b == 0 ? &prof_ode : nullptr)
            );
            if (b == 0) prof_cyc_pred_full += (clock64() - t_pf);
            
            // 4. Sub-step error calculation shortcut
            if (n_s == local_n[d]) {
                next_n[d] = n_s;
                next_mode[d] = mode_s;
                next_scale[d] = scale_s;
                next_T[d] = T_s;
                continue;
            }
            
            T n_h = n_s, mode_h = mode_s, scale_h = scale_s, T_h = T_s;
            T n_d = n_s, mode_d = mode_s, scale_d = scale_s, T_d = T_s;
            T err_n = (T)0.0;

            if (max_vdev >= (T)1e-6) {
                unsigned long long t_ph = (b == 0) ? clock64() : 0;
                T dt_half = dt * (T)0.5;
                update_state_vars(
                    dt_half, vApplied, local_n[d], local_mode[d], local_scale[d], local_T[d],
                    params_d,
                    y_i, y_v, y_fp, y_fd, lookup_size, lookup_scale,
                    n_h, mode_h, scale_h, T_h,
                    b, d, t,
                    (b == 0 ? &prof_ode : nullptr)
                );
                
                if (n_h == local_n[d]) {
                    n_d = n_h;
                    mode_d = mode_h;
                    scale_d = scale_h;
                    T_d = T_h;
                } else {
                    update_state_vars(
                        dt_half, vApplied, n_h, mode_h, scale_h, T_h,
                        params_d,
                        y_i, y_v, y_fp, y_fd, lookup_size, lookup_scale,
                        n_d, mode_d, scale_d, T_d,
                        b, d, t + dt_half,
                        (b == 0 ? &prof_ode : nullptr)
                    );
                }
                if (b == 0) prof_cyc_pred_half += (clock64() - t_ph);
               
                unsigned long long t_err = (b == 0) ? clock64() : 0;
                T diff_n = n_s - n_d;
                if (diff_n < (T)0.0) diff_n = -diff_n;
                if (diff_n < (T)1e-12) {
                    err_n = (T)0.0;
                } else {
                    T n_d_abs = (n_d >= (T)0.0 ? n_d : -n_d);
                    T norm_floor = params_d.nMax * (T)1e-3;
                    T norm_den = (n_d_abs > norm_floor ? n_d_abs : norm_floor);
                    err_n = diff_n / norm_den;
                }
                if (b == 0) prof_cyc_lte_err += (clock64() - t_err);
            }
            
            if (E_a > (T)0.0 && R_th > (T)0.0) {
                T diff_T = T_s - T_d;
                if (diff_T < (T)0.0) diff_T = -diff_T;
                T err_T = diff_T / T_d;
                if (err_T > err_n) err_n = err_T;
            }
            
            if (err_n > max_err) {
                max_err = err_n;
            }
            
            if (err_n > tol && dt > min_dt) {
                step_accepted = false;
                break;
            }
            
            next_n[d] = n_d;
            next_mode[d] = mode_d;
            next_scale[d] = scale_d;
            next_T[d] = T_d;
        }
        
        if (!step_accepted) {
            T eps = (sizeof(T) == 8 ? (T)2.220446049250313e-16 : (T)1.1920929e-7);
            T limit = t * eps * (T)2.0;
            T resolved_min_dt = (min_dt > limit ? min_dt : limit);
            if (dt <= resolved_min_dt) {
#if __HIP_DEVICE_COMPILE__
                underflow_warned = true;
#endif
                step_accepted = true;
                step_rejected = false;
            } else {
                dt = (dt * (T)0.5 > resolved_min_dt ? dt * (T)0.5 : resolved_min_dt);
                step_rejected = true;
                continue;
            }
        } else {
            step_rejected = false;
        }
        
        // 5. Commit accepted step states and update lookups correctly
        unsigned long long t_sc = (b == 0) ? clock64() : 0;
        for (int d = 0; d < num_devices; ++d) {
            T prev_n_d = local_n[d];
            T prev_mode_d = local_mode[d];
            prev_prev_dn[d] = prev_dn[d];
            prev_dn[d] = next_n[d] - local_n[d];
            local_n[d] = next_n[d];
            local_mode[d] = next_mode[d];
            local_scale[d] = next_scale[d];
            local_T[d] = next_T[d];

            int top_n_d = dev_top_full_idx[d];
            int bot_n_d = dev_bot_full_idx[d];
            T vApplied_d = vCurrent[top_n_d] - vCurrent[bot_n_d];
            if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                int src_idx = dev_to_dyn_src[d];
                if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                    vApplied_d = (T)0.0;
                }
            }

            T rBottom = (MAX_DEVICES > 0) ? local_params[d].rBottomBlend : (num_devices == 1 ? dev0_params.rBottomBlend : device_params_soa[27 * num_devices + d]);
            T nMax_d = (MAX_DEVICES > 0) ? local_params[d].nMax : (num_devices == 1 ? dev0_params.nMax : device_params_soa[7 * num_devices + d]);
            T vRefPos_d = (MAX_DEVICES > 0) ? local_params[d].vRefPos : (num_devices == 1 ? dev0_params.vRefPos : device_params_soa[5 * num_devices + d]);
            T k_od_d = (MAX_DEVICES > 0) ? local_params[d].k_overdrive : (num_devices == 1 ? dev0_params.k_overdrive : device_params_soa[29 * num_devices + d]);
            T n_base_nom_d = (T)0.5 * nMax_d;
            T od_ratio_d = (local_cached_n[d] > (T)0.0) ? local_cached_n[d] : (T)0.0;
            T od_val_d = od_ratio_d * k_od_d;
            T od_clamp_d = (od_val_d > (T)1.0) ? (T)1.0 : ((od_val_d < (T)0.0) ? (T)0.0 : od_val_d);
            T scale_base_d = ((rBottom >= (T)0.05) && (rBottom <= (T)2.5)) ? rBottom : (T)1.0;
            T n_base_d = n_base_nom_d * ((T)1.0 - od_clamp_d * ((T)1.0 - scale_base_d));
            if (local_mode[d] > (T)0.5 && local_n[d] > n_base_nom_d) {
                T vTh_d = (MAX_DEVICES > 0) ? local_params[d].vTh : (num_devices == 1 ? dev0_params.vTh : device_params_soa[4 * num_devices + d]);
                if (vApplied_d > vRefPos_d && (vRefPos_d - vTh_d) > (T)1e-6) {
                    T od_inst = (vApplied_d - vRefPos_d) / (vRefPos_d - vTh_d);
                    if (od_inst > local_cached_n[d]) {
                        local_cached_n[d] = od_inst;
                    }
                }
            } else if (local_n[d] <= n_base_d) {
                local_cached_n[d] = (T)0.0;
            }
            
            T n_int = local_n[d];
            T f22StartVal = (MAX_DEVICES > 0) ? local_params[d].f22StartVal : device_params_soa[11 * num_devices + d];
            T f22EndVal = (MAX_DEVICES > 0) ? local_params[d].f22EndVal : device_params_soa[13 * num_devices + d];
            T f22PotScaleInv = (MAX_DEVICES > 0) ? local_params[d].f22PotScaleInv : device_params_soa[14 * num_devices + d];
            T f22DepScaleInv = (MAX_DEVICES > 0) ? local_params[d].f22DepScaleInv : device_params_soa[15 * num_devices + d];
            T gScale = (MAX_DEVICES > 0) ? local_params[d].gScale : device_params_soa[9 * num_devices + d];
            
            bool is_pot_prev = (prev_mode_d > (T)0.5);
            bool is_pot_curr = (local_mode[d] > (T)0.5);
            bool mode_changed = (is_pot_prev != is_pot_curr);
            
            if (n_int != prev_n_d || mode_changed) {
                T iScale = interpolate_1d(n_int, y_i, lookup_size, lookup_scale);
                T vScale = interpolate_1d(n_int, y_v, lookup_size, lookup_scale);
                
                local_iScale[d] = iScale;
                local_vScale[d] = vScale;
                
                // Calculate and commit f22DepScaled and f22PotScaled
                T f22DepRaw = interpolate_1d((T)33040.0 - n_int, y_fd, lookup_size, lookup_scale);
                local_f22Dep[d] = ((T)1.0 - f22DepRaw - f22EndVal) * f22DepScaleInv;
                
                T f22PotRaw = interpolate_1d(n_int, y_fp, lookup_size, lookup_scale);
                T pot_val = (f22PotRaw - f22StartVal) * f22PotScaleInv;
                T nSmooth_d = (MAX_DEVICES > 0) ? local_params[d].nSmooth : device_params_soa[1 * num_devices + d];
                T beta_s_d = (MAX_DEVICES > 0) ? local_params[d].beta_s : device_params_soa[21 * num_devices + d];
                T vRefPos_d_param = (MAX_DEVICES > 0) ? local_params[d].vRefPos : device_params_soa[5 * num_devices + d];
                T vTh_d_param = (MAX_DEVICES > 0) ? local_params[d].vTh : device_params_soa[4 * num_devices + d];
                T n_sm_eff = nSmooth_d;
                if (vApplied_d > vRefPos_d_param && beta_s_d != (T)0.0) {
                    T d_val_pos = vRefPos_d_param - vTh_d_param + (T)1e-6;
                    T delta_v = (vApplied_d - vRefPos_d_param) / ((d_val_pos > (T)1e-15) ? d_val_pos : (T)1e-15);
                    n_sm_eff = clamp(nSmooth_d * ((T)1.0 + beta_s_d * tanh(delta_v)), (T)1.0, (T)1000.0);
                }
                if (pot_val > (T)0.0 && n_int > (T)16520.0) {
                    T u_sat = (n_int - (T)16520.0) / (T)16520.0;
                    if (u_sat > (T)1.0) u_sat = (T)1.0;
                    pot_val += (T)0.001 * n_sm_eff * u_sat * u_sat * ((T)3.0 - (T)2.0 * u_sat);
                }
                local_f22Pot[d] = pot_val;
                
                local_currentIScale[d] = iScale * gScale;
            }
            
            bool is_pot = is_pot_curr;
            T f22_committed = is_pot ? local_f22Pot[d] : (local_f22Dep[d] * local_scale[d]);
            T f22_clamped = (f22_committed > (T)0.0 ? f22_committed : (T)0.0);
            local_currentF22[d] = f22_clamped;
            
            T currentI = (T)0.0;
            T gEq = (T)0.0;
            int top_n = dev_top_full_idx[d];
            int bot_n = dev_bot_full_idx[d];
            T vApplied = vCurrent[top_n] - vCurrent[bot_n];
            T gamma = (MAX_DEVICES > 0) ? local_params[d].gamma : device_params_soa[19 * num_devices + d];
            T rC = (MAX_DEVICES > 0) ? local_params[d].rC : device_params_soa[10 * num_devices + d];
            if (gamma != (T)0.0) {
                local_currentVScale[d] = (local_vScale[d] / ((T)1.0 + gamma * f22_clamped));
            } else {
                local_currentVScale[d] = local_vScale[d];
            }
            if (numNodes > 0) {
                compute_IGeq(vApplied, local_currentIScale[d], local_currentVScale[d], f22_clamped, rC, currentI, gEq);
                if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                    int src_idx = dev_to_dyn_src[d];
                    if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                        currentI = (T)0.0;
                        gEq = (T)1e-25;
                    }
                }
                local_iDev[d] = currentI;
            }
        }
        if (b == 0) prof_cyc_state_commit += (clock64() - t_sc);
        
        T t_prev = t;
        t += dt;
        if (t <= t_prev || t >= t_end - (T)1e-14) {
            t = t_end;
        } else {
            T eps_snap = (sizeof(T) == 8 ? (T)2.220446049250313e-16 : (T)1.1920929e-7);
            T snap_thresh = dt * eps_snap * (T)2.0;
            if (snap_thresh < (T)1e-14) snap_thresh = (T)1e-14;
            if (snap_thresh > (T)1e-11) snap_thresh = (T)1e-11;
            T diff_edge = tNextEdge - t;
            if (diff_edge >= (T)0.0 && diff_edge <= snap_thresh) {
                t = tNextEdge;
            }
        }
        
        for (int s = 0; s < num_v_sources; ++s) {
            int dummy_cache = 0;
            int& c_idx = (s < 128) ? v_pwl_cache[s] : dummy_cache;
            T val = evaluate_pwl_value(t, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask_gpu, s, max_pwl_points, b, pad_sources, c_idx);
            vCurrent[v_nodes_gpu[s]] = val;
        }
        for (int node_i = 0; node_i < numNodes + num_v_sources; ++node_i) {
            prev_v[node_i] = vCurrent[node_i];
        }
        if (numNodes == 0) {
            for (int d = 0; d < num_devices; ++d) {
                int top_idx = dev_top_full_idx[d];
                int bot_idx = dev_bot_full_idx[d];
                T v_app = vCurrent[top_idx] - vCurrent[bot_idx];
                T currentF22 = local_currentF22[d];
                T currentIScale = local_currentIScale[d];
                T currentVScale = local_currentVScale[d];
                T rC = (MAX_DEVICES > 0) ? local_params[d].rC : (num_devices == 1 ? dev0_params.rC : device_params_soa[10 * num_devices + d]);
                T currentI = (T)0.0;
                T gEq = (T)0.0;
                T v_app_abs = (v_app >= (T)0.0 ? v_app : -v_app);
                if (v_app_abs >= (T)1e-12 && currentF22 > (T)0.0) {
                    compute_IGeq(v_app, currentIScale, currentVScale, currentF22, rC, currentI, gEq);
                }
                if (is_1t1r && num_dyn_sources > 0 && dev_to_dyn_src[d] >= 0) {
                    int src_idx = dev_to_dyn_src[d];
                    if (dynamic_G[src_idx] <= leakage_g * (T)10.0) {
                        currentI = (T)0.0;
                        gEq = (T)1e-25;
                    }
                }
                local_iDev[d] = currentI;
                local_gEq[d] = gEq;
            }
        }
        if (b == 0) {
            prof_cyc_ode += (clock64() - t_ode_start);
        }
        
        unsigned long long t_hist_start = (b == 0) ? clock64() : 0;
        if (recordHistory) {
            if (hist_idx < max_steps) {
                int time_offset = b * max_steps + hist_idx;
                tArr_buffer[time_offset] = t;
                
                for (int n = 0; n < num_all_nodes; ++n) {
                    int offset = (b * num_all_nodes + n) * max_steps + hist_idx;
                    vHist_buffer[offset] = (n < numNodes + num_v_sources + pad_sources) ? vCurrent[n] : (T)0.0;
                }
                
                if (num_devices == 1) {
                    int offset = b * max_steps + hist_idx;
                    stateHist_buffer[offset] = local_n[0];
                    f22Hist_buffer[offset] = local_currentF22[0];
                    gHist_buffer[offset] = (local_currentF22[0] > (T)0.0) ? (local_currentIScale[0] * local_currentF22[0]) / (local_currentVScale[0] + (T)1e-12) : (T)0.0;
                    tempHist_buffer[offset] = local_T[0];
                    iHist_buffer[offset] = local_iDev[0];
                } else {
                    for (int d = 0; d < num_devices; ++d) {
                        int offset = (b * num_devices + d) * max_steps + hist_idx;
                        stateHist_buffer[offset] = local_n[d];
                        f22Hist_buffer[offset] = local_currentF22[d];
                        gHist_buffer[offset] = (local_currentF22[d] > (T)0.0) ? (local_currentIScale[d] * local_currentF22[d]) / (local_currentVScale[d] + (T)1e-12) : (T)0.0;
                        tempHist_buffer[offset] = local_T[d];
                        iHist_buffer[offset] = local_iDev[d];
                    }
                }
                hist_idx++;
            }
        } else {
            int time_offset = b * max_steps + 0;
            tArr_buffer[time_offset] = t;
            hist_idx = 1;
        }
        if (b == 0) {
            prof_cyc_hist += (clock64() - t_hist_start);
            prof_steps++;
        }
        
        // Predictor-Corrector recovery matching CPU logic:
        if (max_vdev < (T)1e-6 || max_err < tol * (T)0.01) {
            dt = clamp(max_dt, min_dt, max_dt);
        } else if (max_err < tol * (T)0.1) {
            dt = clamp(dt * (T)2.0, min_dt, max_dt);
        } else {
            dt = clamp(dt, min_dt, max_dt);
        }
        step_count++;
        for (int node_i = 0; node_i < numNodes + num_v_sources + pad_sources; ++node_i) {
            prev_v[node_i] = vCurrent[node_i];
        }
    }
    
    if (recordHistory && hist_idx < max_steps) {
        int time_offset = b * max_steps + hist_idx;
        tArr_buffer[time_offset] = t;
        
        for (int n = 0; n < num_all_nodes; ++n) {
            int offset = (b * num_all_nodes + n) * max_steps + hist_idx;
            vHist_buffer[offset] = (n < numNodes + num_v_sources + pad_sources) ? vCurrent[n] : (T)0.0;
        }
        
        if (num_devices == 1) {
            int offset = b * max_steps + hist_idx;
            stateHist_buffer[offset] = local_n[0];
            f22Hist_buffer[offset] = local_currentF22[0];
            gHist_buffer[offset] = (local_currentF22[0] > (T)0.0) ? (local_currentIScale[0] * local_currentF22[0]) / (local_currentVScale[0] + (T)1e-12) : (T)0.0;
            tempHist_buffer[offset] = local_T[0];
            iHist_buffer[offset] = local_iDev[0];
        } else {
            for (int d = 0; d < num_devices; ++d) {
                int offset = (b * num_devices + d) * max_steps + hist_idx;
                stateHist_buffer[offset] = local_n[d];
                f22Hist_buffer[offset] = local_currentF22[d];
                gHist_buffer[offset] = (local_currentF22[d] > (T)0.0) ? (local_currentIScale[d] * local_currentF22[d]) / (local_currentVScale[d] + (T)1e-12) : (T)0.0;
                tempHist_buffer[offset] = local_T[d];
                iHist_buffer[offset] = local_iDev[d];
            }
        }
        hist_idx++;
    }
    
    if (b < batch_size) {
        for (int d = 0; d < num_devices; ++d) {
            int offset_d = d * batch_size + b;
            device_states_soa[0 * num_devices * batch_size + offset_d] = local_n[d];
            device_states_soa[1 * num_devices * batch_size + offset_d] = local_mode[d];
            device_states_soa[2 * num_devices * batch_size + offset_d] = local_scale[d];
            device_states_soa[3 * num_devices * batch_size + offset_d] = local_f22Dep[d];
            device_states_soa[4 * num_devices * batch_size + offset_d] = local_f22Pot[d];
            device_states_soa[5 * num_devices * batch_size + offset_d] = local_cached_n[d];
            device_states_soa[6 * num_devices * batch_size + offset_d] = local_iScale[d];
            device_states_soa[7 * num_devices * batch_size + offset_d] = local_vScale[d];
            device_states_soa[8 * num_devices * batch_size + offset_d] = local_currentF22[d];
            device_states_soa[9 * num_devices * batch_size + offset_d] = local_currentIScale[d];
            device_states_soa[10 * num_devices * batch_size + offset_d] = local_currentVScale[d];
            device_states_soa[11 * num_devices * batch_size + offset_d] = local_T[d];
        }
        // Write back final transient node solver current and conductance to global workspace
        T* w_base_dev = global_workspace_t + (num_gMat_vals + 4 * numNodes + num_v_sources + pad_sources) * batch_size;
        T* dest_iDev = w_base_dev + (18 * num_devices) * batch_size + b * num_devices;
        T* dest_gEq = w_base_dev + (19 * num_devices) * batch_size + b * num_devices;
        for (int d = 0; d < num_devices; ++d) {
            dest_iDev[d] = local_iDev[d];
            dest_gEq[d] = local_gEq[d];
        }
    }
    
    if (d_out != nullptr) {
        if (b == 0) {
            d_out[0] = (T)hist_idx;
            d_out[1] = t;
            d_out[2] = (underflow_warned ? (T)1.0 : (T)0.0);
            d_out[3] = (T)(clock64() - prof_t0);
        d_out[4] = (T)prof_cyc_pwl;
        d_out[5] = (T)prof_cyc_assembly;
        d_out[6] = (T)prof_cyc_cg;
        d_out[7] = (T)prof_cyc_ode;
        d_out[8] = (T)prof_cyc_hist;
        d_out[9] = (T)prof_ode.cyc_lut;
        d_out[10] = (T)prof_ode.cyc_thermal;
        d_out[11] = (T)prof_ode.cyc_rate;
        d_out[12] = (T)prof_ode.cyc_boundary;
        d_out[13] = (T)prof_ode.cyc_mode;
        d_out[14] = (T)prof_cyc_pred_full;
        d_out[15] = (T)prof_cyc_pred_half;
        d_out[16] = local_currentF22[0];
        d_out[17] = local_currentIScale[0];
        d_out[18] = local_currentVScale[0];
        d_out[19] = local_iDev[0];
        d_out[20] = local_n[0];
        d_out[21] = local_mode[0];
        d_out[22] = local_scale[0];
        d_out[23] = (dev_top_full_idx != nullptr) ? vCurrent[dev_top_full_idx[0]] : (T)0.0;
        d_out[24] = (dev_bot_full_idx != nullptr) ? vCurrent[dev_bot_full_idx[0]] : (T)0.0;
        d_out[25] = (dev_top_full_idx != nullptr) ? (T)dev_top_full_idx[0] : (T)0.0;
        d_out[26] = (dev_bot_full_idx != nullptr) ? (T)dev_bot_full_idx[0] : (T)0.0;
        }
        d_out[32 + b] = (T)hist_idx;
    }
}

extern "C" void launch_general_transient_kernel_double(
    int dev_idx,
    double t_start, double t_end, double min_dt, double tol, int maxIter, int max_steps,
    const double* master_schedule, const int* is_subsample, int num_edges,
    double* device_states_soa,
    const double* device_params_soa,
    int numNodes, int num_devices,
    const int* dev_top_full_idx, const int* dev_bot_full_idx,
    const int* dev_top_idx, const int* dev_bot_idx,
    const int* gMat_row_indices, const int* gMat_col_indices, const int* iRes_indices,
    const int* top_indices, const int* bot_indices, const int* both_indices, const int* dev_both_idx,
    int L_top, int L_bot, int L_both,
    const int* d_nodes, const int* dyn_indices, const int* dev_to_dyn_src,
    const int* v_nodes_gpu, const double* v_pwl_t_matrix, const double* v_pwl_v_matrix, const int* v_pwl_mask_gpu,
    const int* dyn_nodes_gpu, const double* dyn_pwl_t_matrix, const double* dyn_pwl_v_matrix, const int* dyn_pwl_mask_gpu,
    bool is_1t1r, bool recordHistory, double leakage_g,
    int num_v_sources, int num_dyn_sources, int num_all_nodes,
    // History buffers
    double* tArr_buffer, double* vHist_buffer, double* iHist_buffer,
    double* stateHist_buffer, double* f22Hist_buffer, double* gHist_buffer,
    double* tempHist_buffer,
    // System dimensions
    int num_gMat_vals,
    int max_pwl_points,
    int batch_size,
    // Lookup tables
    const double* y_i,
    const double* y_v,
    const double* y_fp,
    const double* y_fd,
    int lookup_size,
    double* global_workspace_cg,
    double* global_workspace_t,
    double* d_out,
    void* stream_ptr
) {
#ifdef __HIP_PLATFORM_AMD__
    if (dev_idx >= 0) {
        hipSetDevice(dev_idx);
    }
#else
    if (dev_idx >= 0) {
        hipSetDevice(dev_idx);
    }
#endif

    int threadsPerBlock = 64;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    #define LAUNCH_DOUBLE_KERN(MAX_NODES, MAX_DEVICES) \
        hipLaunchKernelGGL( \
            (run_general_transient_kernel<double, double, MAX_NODES, MAX_DEVICES>), \
            dim3(blocksPerGrid), \
            dim3(threadsPerBlock), \
            0, (hipStream_t)stream_ptr, \
            t_start, t_end, min_dt, tol, maxIter, max_steps, \
            master_schedule, is_subsample, num_edges, \
            device_states_soa, \
            device_params_soa, \
            numNodes, num_devices, \
            dev_top_full_idx, dev_bot_full_idx, \
            dev_top_idx, dev_bot_idx, \
            gMat_row_indices, gMat_col_indices, iRes_indices, \
            top_indices, bot_indices, both_indices, dev_both_idx, \
            L_top, L_bot, L_both, \
            d_nodes, dyn_indices, dev_to_dyn_src, \
            v_nodes_gpu, v_pwl_t_matrix, v_pwl_v_matrix, v_pwl_mask_gpu, \
            dyn_nodes_gpu, dyn_pwl_t_matrix, dyn_pwl_v_matrix, dyn_pwl_mask_gpu, \
            is_1t1r, recordHistory, leakage_g, \
            num_v_sources, num_dyn_sources, num_all_nodes, \
            tArr_buffer, vHist_buffer, iHist_buffer, \
            stateHist_buffer, f22Hist_buffer, gHist_buffer, \
            tempHist_buffer, \
            num_gMat_vals, \
            max_pwl_points, \
            batch_size, \
            y_i, \
            y_v, \
            y_fp, \
            y_fd, \
            lookup_size, \
            d_out, \
            global_workspace_cg, \
            global_workspace_t \
        )

    LAUNCH_DOUBLE_KERN(0, 0);
    #undef LAUNCH_DOUBLE_KERN

#ifdef __HIP_PLATFORM_AMD__
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        throw std::runtime_error(std::string("MOLMEM C++ Extension - HIP double launch failure: ") + hipGetErrorString(launch_err));
    }
#else
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        throw std::runtime_error(std::string("MOLMEM C++ Extension - CUDA double launch failure: ") + hipGetErrorString(launch_err));
    }
#endif
}

extern "C" int reset_gpu_device() {
#ifdef __HIP_PLATFORM_AMD__
    hipError_t err = hipDeviceReset();
    return (int)err;
#else
    hipError_t err = hipDeviceReset();
    return (int)err;
#endif
}

