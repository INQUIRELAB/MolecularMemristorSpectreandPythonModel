#!/usr/bin/env python3
"""
fit_safe_vwrite.py - Parallel Crossbar Inference Search for Optimal Safe Maximum Write Voltage (safe_max_vwrite)

Evaluates multi-layer parallel crossbar programming accuracy using concurrent 16x16 arrays
(Layer1_pos, Layer1_neg, Layer2_pos, Layer2_neg - 1,024 analog memristors total).
Directly tests end-to-end matrix-vector inference fidelity (HW Y Out vs Math Y Out = X @ W)
and identifies the largest write voltage candidate where the hardware output matches
the mathematical ground truth within 0.05.
"""

import os
import sys
import argparse
import json
import time
import numpy as np

# Ensure molmem_lib is accessible
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "PythonSimulator")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from molmem_lib import MolmemSimulator, MolmemSilence, updateCrossbarsParallel

# Deterministic evaluation setup: 4 x (16x16) crossbars (1,024 analog memristors)
EVAL_SEED = 42
_eval_rng = np.random.RandomState(EVAL_SEED)
EVAL_W1_POS = _eval_rng.uniform(0.0, 1.0, (16, 16))
EVAL_W1_NEG = _eval_rng.uniform(0.0, 1.0, (16, 16))
EVAL_W2_POS = _eval_rng.uniform(0.0, 1.0, (16, 16))
EVAL_W2_NEG = _eval_rng.uniform(0.0, 1.0, (16, 16))
EVAL_WEIGHTS = [EVAL_W1_POS, EVAL_W1_NEG, EVAL_W2_POS, EVAL_W2_NEG]
EVAL_X_INPUT = _eval_rng.uniform(0.0, 1.0, 16)


def evaluate_vwrite_candidate(v_cand, device_type="Ru_azo", max_pw=320e-9):
    """
    Evaluates candidate max_vWrite across 4 concurrent 16x16 crossbar arrays (1,024 memristors).
    Directly measures physical matrix-vector inference accuracy (HW Y Out vs Math Y Out = X @ W).
    """
    t0 = time.perf_counter()
    with MolmemSilence():
        sim1_pos = MolmemSimulator(instanceName="Layer1_pos_16x16")
        sim1_pos.addCrossbarMatrix(
            rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14,
            detailedPrint=False, architecture='1T1R', max_vWrite=v_cand, max_pw=max_pw,
            UpdatesPer="Column", device_type=device_type, ignore_safety_cap=True
        )
        sim1_neg = MolmemSimulator(instanceName="Layer1_neg_16x16")
        sim1_neg.addCrossbarMatrix(
            rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14,
            detailedPrint=False, architecture='1T1R', max_vWrite=v_cand, max_pw=max_pw,
            UpdatesPer="Column", device_type=device_type, ignore_safety_cap=True
        )
        sim2_pos = MolmemSimulator(instanceName="Layer2_pos_16x16")
        sim2_pos.addCrossbarMatrix(
            rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14,
            detailedPrint=False, architecture='1T1R', max_vWrite=v_cand, max_pw=max_pw,
            UpdatesPer="Column", device_type=device_type, ignore_safety_cap=True
        )
        sim2_neg = MolmemSimulator(instanceName="Layer2_neg_16x16")
        sim2_neg.addCrossbarMatrix(
            rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14,
            detailedPrint=False, architecture='1T1R', max_vWrite=v_cand, max_pw=max_pw,
            UpdatesPer="Column", device_type=device_type, ignore_safety_cap=True
        )

        simulators = [sim1_pos, sim1_neg, sim2_pos, sim2_neg]
        updateCrossbarsParallel(simulators, EVAL_WEIGHTS, detailedPrint=False, prefix="FitParallel", fineTune=False, recordHistory=False)

        all_diffs = []
        all_w_errs = []
        layer_maes = []
        max_conductances = []
        sample_hw = None
        sample_math = None

        for idx, (sim, w_target) in enumerate(zip(simulators, EVAL_WEIGHTS)):
            # Hardware inference pass
            y_hw = sim.passInput(EVAL_X_INPUT, detailedPrint=False)
            y_math = np.dot(EVAL_X_INPUT, w_target)
            diff = np.abs(y_hw - y_math)
            all_diffs.extend(diff)
            layer_maes.append(float(np.mean(diff)))

            if idx == 0:
                sample_hw = [round(float(val), 4) for val in y_hw[:3]]
                sample_math = [round(float(val), 4) for val in y_math[:3]]

            # Read back physical conductances and weights
            gMatrix, _ = sim.readCrossbarMatrix(detailedPrint=False, purpose=None)
            max_conductances.append(float(np.max(gMatrix)))
            w_hw = sim._conductanceToWeights(gMatrix)
            all_w_errs.extend(np.abs(w_target - w_hw).flatten())

    t_elapsed = time.perf_counter() - t0
    inf_mae = float(np.mean(all_diffs))
    inf_max = float(np.max(all_diffs))
    weight_mae = float(np.mean(all_w_errs))
    weight_max = float(np.max(all_w_errs))
    max_g = max(max_conductances)
    is_saturated = bool((max_g >= 8.5e-3) or np.isnan(inf_mae))

    return {
        "v_cand": round(float(v_cand), 4),
        "inf_mae": float(inf_mae),
        "inf_max": float(inf_max),
        "weight_mae": float(weight_mae),
        "weight_max": float(weight_max),
        "sample_hw": sample_hw,
        "sample_math": sample_math,
        "layer_maes": [round(float(m), 4) for m in layer_maes],
        "max_g_mS": round(float(max_g * 1e3), 3),
        "is_saturated": bool(is_saturated),
        "eval_time_s": round(float(t_elapsed), 3)
    }


def pick_largest_safe_voltage(results_list, target_error=0.05):
    """
    Selects the largest voltage candidate whose hardware inference output
    matches the mathematical ground truth within target_error (default: 0.05).
    """
    passing = [r for r in results_list if r["inf_mae"] <= target_error and not r["is_saturated"]]

    if passing:
        best = max(passing, key=lambda r: r["v_cand"])
        print(f"  >> Selected Largest Voltage within {target_error:.2f}: {best['v_cand']:.2f} V (Inf MAE: {best['inf_mae']:.4f})")
        return best, best["inf_mae"]
    else:
        valid = [r for r in results_list if not np.isnan(r["inf_mae"])]
        fallback = min(valid, key=lambda r: r["inf_mae"]) if valid else results_list[0]
        print(f"  >> [Notice] No candidate had Inf MAE <= {target_error:.2f}. Fallback to lowest error: {fallback['v_cand']:.2f} V (MAE: {fallback['inf_mae']:.4f})")
        return fallback, fallback["inf_mae"]


def update_devices_py_safecap(safe_vwrite, device_type="Ru_azo"):
    """
    Updates the safe_max_vwrite parameter in PythonSimulator/molmem_lib/devices.py.
    Supports legacy configurations by injecting if not already present.
    """
    devices_path = os.path.join(_SIM_DIR, "molmem_lib", "devices.py")
    if not os.path.exists(devices_path):
        print(f"[ERROR] Cannot locate devices.py at {devices_path}")
        return False

    with open(devices_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    if '"safe_max_vwrite":' in content:
        updated = re.sub(
            r'("safe_max_vwrite":\s*)[0-9.]+',
            f'\\g<1>{safe_vwrite:.2f}',
            content
        )
    else:
        # Legacy fallback: inject right after k_overdrive
        updated = re.sub(
            r'("k_overdrive":\s*[0-9.]+,\n)',
            f'\\g<1>    "safe_max_vwrite": {safe_vwrite:.2f},\n',
            content
        )

    with open(devices_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"[APPLY] Updated safe_max_vwrite = {safe_vwrite:.2f} V in {devices_path}")
    return True


def discover_presets(device_name=None):
    """
    Scans DeviceFitting/ for completed fit preset text files (*_preset_devices_py*.txt).
    Returns a list of dictionaries with metadata for each preset.
    """
    import re
    fitting_dir = _CURRENT_DIR
    detected = []
    for entry in os.scandir(fitting_dir):
        if entry.is_file() and "_preset_devices_py" in entry.name and entry.name.endswith(".txt"):
            dev_prefix = entry.name.split("_preset_devices_py")[0]
            safe_cap = None
            has_param = False
            try:
                with open(entry.path, "r", encoding="utf-8") as f:
                    content = f.read()
                m = re.search(r'"safe_max_vwrite":\s*([0-9.]+)', content)
                if m:
                    safe_cap = float(m.group(1))
                    has_param = True
            except Exception:
                pass
            detected.append({
                "device_name": dev_prefix,
                "file_name": entry.name,
                "file_path": entry.path,
                "safe_cap": safe_cap,
                "has_param": has_param
            })
    detected.sort(key=lambda d: d["file_name"])
    return detected


def load_preset_config(preset_path):
    """
    Loads and executes a preset .txt file to retrieve its device parameter dictionary.
    Supports legacy parameter configurations lacking safe_max_vwrite with a 0.0 placeholder fallback.
    """
    with open(preset_path, "r", encoding="utf-8") as f:
        code = f.read()

    local_namespace = {"os": os}
    exec(code, local_namespace)

    builder_func = None
    for k, v in local_namespace.items():
        if callable(v) and not k.startswith("_"):
            builder_func = v
            break

    if builder_func is None:
        raise ValueError(f"No device builder function found inside {preset_path}")

    device_data_dir = os.path.join(_SIM_DIR, "molmem_lib", "data")
    config = builder_func(data_dir=device_data_dir)

    if "safe_max_vwrite" not in config or config["safe_max_vwrite"] is None:
        config["safe_max_vwrite"] = 0.0

    return config


def update_preset_txt_safecap(safe_vwrite, preset_path):
    """
    Updates the safe_max_vwrite value in an existing preset .txt file.
    Injects safe_max_vwrite into legacy parameter lists if missing.
    """
    if not os.path.exists(preset_path):
        print(f"[ERROR] Preset path does not exist: {preset_path}")
        return False

    with open(preset_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    if '"safe_max_vwrite":' in content:
        updated = re.sub(
            r'("safe_max_vwrite":\s*)[0-9.]+',
            f'\\g<1>{safe_vwrite:.2f}',
            content
        )
    else:
        # Legacy fallback: inject right after k_overdrive
        updated = re.sub(
            r'("k_overdrive":\s*[0-9.]+,\n)',
            f'\\g<1>        "safe_max_vwrite": {safe_vwrite:.2f},\n',
            content
        )

    if updated != content:
        with open(preset_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"[APPLY] Updated safe_max_vwrite = {safe_vwrite:.2f} V in preset: {os.path.basename(preset_path)}")
        return True
    else:
        print(f"[NOTICE] Preset file {os.path.basename(preset_path)} already has safe_max_vwrite = {safe_vwrite:.2f} V")
        return True


def main():
    parser = argparse.ArgumentParser(description="Parallel Crossbar Inference Search for Safe Maximum Write Voltage (safe_max_vwrite)")
    parser.add_argument("--device", type=str, default="Ru_azo", help="Target device preset (default: Ru_azo)")
    parser.add_argument("--min-vwrite", type=float, default=1.00, help="Minimum write voltage (default: 1.00 V)")
    parser.add_argument("--max-vwrite", type=float, default=5.00, help="Maximum write voltage (default: 5.00 V)")
    parser.add_argument("--coarse-step", type=float, default=0.25, help="Coarse sweep step size (default: 0.25 V)")
    parser.add_argument("--target-error", type=float, default=0.05, help="Target threshold for HW vs Math output error (default: 0.05)")
    parser.add_argument("--preset-file", type=str, default=None, help="Specific preset .txt file to update")
    parser.add_argument("--apply", action="store_true", help="Automatically update devices.py in addition to the selected preset .txt file")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON results")
    args = parser.parse_args()

    # Discover and select preset configuration
    all_presets = discover_presets(device_name=args.device)
    matching_presets = [p for p in all_presets if p["device_name"].lower() == args.device.lower()] if args.device else all_presets

    selected_preset = None
    if args.preset_file:
        p_path = args.preset_file if os.path.isabs(args.preset_file) else os.path.join(_CURRENT_DIR, args.preset_file)
        if os.path.exists(p_path):
            dev_name = os.path.basename(p_path).split("_preset_devices_py")[0]
            selected_preset = {"device_name": dev_name, "file_name": os.path.basename(p_path), "file_path": p_path}
        else:
            print(f"[ERROR] Specified preset file does not exist: {p_path}")
            return 1
    elif matching_presets:
        if len(matching_presets) == 1:
            selected_preset = matching_presets[0]
            print(f"[Preset Selection] Auto-selected device preset: {selected_preset['file_name']}")
        else:
            print(f"Detected {len(matching_presets)} available preset parameter lists for device '{args.device}':")
            for idx, p in enumerate(matching_presets):
                cap_str = f"{p['safe_cap']:.2f} V" if p['safe_cap'] is not None else "0.00 V (placeholder)"
                print(f"  [{idx + 1}] {p['device_name']} ({p['file_name']}) | current safe_max_vwrite: {cap_str}")

            if sys.stdin.isatty():
                while True:
                    try:
                        ans = input(f"\nSelect a preset file to update (1-{len(matching_presets)}) [1]: ").strip()
                        if not ans:
                            sel_idx = 0
                        else:
                            sel_idx = int(ans) - 1
                        if 0 <= sel_idx < len(matching_presets):
                            selected_preset = matching_presets[sel_idx]
                            break
                    except (ValueError, KeyboardInterrupt, EOFError):
                        pass
                    print("Invalid selection. Please enter a valid number.")
            else:
                selected_preset = matching_presets[0]
                print(f"[Non-Interactive] Auto-selecting preset [1]: {selected_preset['file_name']}")

    # Load configuration
    if selected_preset:
        print(f"\nLoading configuration from: {selected_preset['file_path']}")
        device_config = load_preset_config(selected_preset["file_path"])
        device_eval_target = device_config
        target_device_name = selected_preset["device_name"]
        print(f"Loaded params list (kappa: {device_config.get('kappa', 0):.2e}, alpha: {device_config.get('alpha', 0):.2f}, "
              f"vTh: {device_config.get('vTh', 0):.2f} V, current safe_max_vwrite: {device_config.get('safe_max_vwrite', 0.0):.2f} V)")
    else:
        print(f"\nUsing built-in configuration for device: {args.device}")
        device_eval_target = args.device
        target_device_name = args.device

    print("================================================================================")
    print("  MOLMEM PARALLEL CROSSBAR INFERENCE SAFE MAX WRITE VOLTAGE FITTER")
    print("================================================================================")
    print(f"  Target Device       : {target_device_name}")
    if selected_preset:
        print(f"  Target Preset File  : {selected_preset['file_name']}")
    print(f"  Benchmark Config    : 4 concurrent 16x16 arrays (1,024 memristors)")
    print(f"  Voltage Range       : [{args.min_vwrite:.2f} V, {args.max_vwrite:.2f} V]")
    print(f"  Coarse Step         : {args.coarse_step * 1e3:.0f} mV")
    print(f"  Acceptance Criterion: HW vs Math output error <= {args.target_error:.2f}")
    print("================================================================================\n")

    eval_cache = {}

    def get_eval(v):
        v_key = round(float(v), 4)
        if v_key not in eval_cache:
            res = evaluate_vwrite_candidate(v_key, device_type=device_eval_target)
            eval_cache[v_key] = res
        return eval_cache[v_key]

    # -------------------------------------------------------------------------
    # STAGE 1: COARSE SWEEP (250 mV Grid)
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("  STAGE 1: COARSE SWEEP (250 mV Step Grid)")
    print("--------------------------------------------------------------------------------")
    coarse_grid = np.arange(args.min_vwrite, args.max_vwrite + 1e-6, args.coarse_step)
    coarse_results = []

    for idx, v in enumerate(coarse_grid):
        res = get_eval(v)
        coarse_results.append(res)
        status = "PASS (<= 0.05)" if res["inf_mae"] <= args.target_error and not res["is_saturated"] else "FAIL (> 0.05)"
        sat_str = " [SAT]" if res["is_saturated"] else ""
        print(f"  [COARSE {idx+1:02d}/{len(coarse_grid):02d}] V = {res['v_cand']:.2f} V | "
              f"HW Y: {res['sample_hw']} vs Math: {res['sample_math']} | "
              f"MAE: {res['inf_mae']:.4f} | {status}{sat_str} | {res['eval_time_s']}s")
        if res["inf_mae"] > args.target_error or res["is_saturated"]:
            print(f"  >> Tolerance threshold ({args.target_error:.2f}) exceeded at {res['v_cand']:.2f} V. Breaking coarse sweep.")
            break

    best_coarse, _ = pick_largest_safe_voltage(coarse_results, target_error=args.target_error)
    print(f"\n  >> Stage 1 Complete: Selected Coarse Candidate = {best_coarse['v_cand']:.2f} V "
          f"(HW Y: {best_coarse['sample_hw']} vs Math: {best_coarse['sample_math']}, MAE: {best_coarse['inf_mae']:.4f})\n")

    # -------------------------------------------------------------------------
    # STAGE 2: MEDIUM REFINEMENT (50 mV Grid)
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("  STAGE 2: MEDIUM REFINEMENT (50 mV Step Grid)")
    print("--------------------------------------------------------------------------------")
    b_low = max(args.min_vwrite, round(best_coarse["v_cand"] - args.coarse_step, 4))
    b_high = min(args.max_vwrite, round(best_coarse["v_cand"] + args.coarse_step, 4))
    med_grid = np.arange(b_low, b_high + 1e-6, 0.05)
    med_results = []

    for idx, v in enumerate(med_grid):
        res = get_eval(v)
        med_results.append(res)
        status = "PASS (<= 0.05)" if res["inf_mae"] <= args.target_error and not res["is_saturated"] else "FAIL (> 0.05)"
        sat_str = " [SAT]" if res["is_saturated"] else ""
        print(f"  [MED {idx+1:02d}/{len(med_grid):02d}]    V = {res['v_cand']:.2f} V | "
              f"HW Y: {res['sample_hw']} vs Math: {res['sample_math']} | "
              f"MAE: {res['inf_mae']:.4f} | {status}{sat_str} | {res['eval_time_s']}s")
        if res["inf_mae"] > args.target_error or res["is_saturated"]:
            print(f"  >> Tolerance threshold ({args.target_error:.2f}) exceeded at {res['v_cand']:.2f} V. Breaking medium refinement sweep.")
            break

    best_med, _ = pick_largest_safe_voltage(med_results, target_error=args.target_error)
    print(f"\n  >> Stage 2 Complete: Selected Medium Candidate = {best_med['v_cand']:.2f} V "
          f"(HW Y: {best_med['sample_hw']} vs Math: {best_med['sample_math']}, MAE: {best_med['inf_mae']:.4f})\n")

    # -------------------------------------------------------------------------
    # STAGE 3: FINE REFINEMENT (10 mV Grid)
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("  STAGE 3: FINE REFINEMENT (10 mV Step Grid)")
    print("--------------------------------------------------------------------------------")
    f_low = max(args.min_vwrite, round(best_med["v_cand"] - 0.05, 4))
    f_high = min(args.max_vwrite, round(best_med["v_cand"] + 0.05, 4))
    fine_grid = np.arange(f_low, f_high + 1e-6, 0.01)
    fine_results = []

    for idx, v in enumerate(fine_grid):
        res = get_eval(v)
        fine_results.append(res)
        status = "PASS (<= 0.05)" if res["inf_mae"] <= args.target_error and not res["is_saturated"] else "FAIL (> 0.05)"
        sat_str = " [SAT]" if res["is_saturated"] else ""
        print(f"  [FINE {idx+1:02d}/{len(fine_grid):02d}]   V = {res['v_cand']:.2f} V | "
              f"HW Y: {res['sample_hw']} vs Math: {res['sample_math']} | "
              f"MAE: {res['inf_mae']:.4f} | {status}{sat_str} | {res['eval_time_s']}s")
        if res["inf_mae"] > args.target_error or res["is_saturated"]:
            print(f"  >> Tolerance threshold ({args.target_error:.2f}) exceeded at {res['v_cand']:.2f} V. Breaking fine refinement sweep.")
            break

    best_final, _ = pick_largest_safe_voltage(fine_results, target_error=args.target_error)

    print("\n================================================================================")
    print("  CONVERGENCE SUMMARY: OPTIMAL SAFE MAXIMUM WRITE VOLTAGE")
    print("================================================================================")
    print(f"  Target Device Preset    : {target_device_name}")
    if selected_preset:
        print(f"  Target Preset File      : {selected_preset['file_name']}")
    status_summary = "PASS (<= 0.05)" if best_final['inf_mae'] <= args.target_error else "FAIL (> 0.05)"
    print(f"  Selected safe_max_vwrite: {best_final['v_cand']:.2f} V (Largest voltage within {args.target_error:.2f})")
    print(f"  Hardware Output (Sample): {best_final['sample_hw']}")
    print(f"  Math Output (Sample)    : {best_final['sample_math']}")
    print(f"  Overall Inference MAE   : {best_final['inf_mae']:.4f} -> {status_summary}")
    print(f"  Max Output Difference   : {best_final['inf_max']:.4f}")
    print(f"  Absolute Weight MAE     : {best_final['weight_mae']:.4e}")
    print(f"  Layer MAEs (4 crossbars): {best_final['layer_maes']}")
    print(f"  Peak Conductance Seen   : {best_final['max_g_mS']} mS")
    print("================================================================================\n")

    # Save JSON summary
    out_dir = os.path.join(_CURRENT_DIR, args.device)
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(out_dir, "safe_vwrite_fit_results.json")

    summary_data = {
        "device": target_device_name,
        "preset_file": selected_preset["file_name"] if selected_preset else None,
        "benchmark": "4x(16x16) Parallel Crossbars (1,024 Memristors)",
        "safe_max_vwrite": float(best_final["v_cand"]),
        "sample_hw": best_final["sample_hw"],
        "sample_math": best_final["sample_math"],
        "inf_mae": float(best_final["inf_mae"]),
        "inf_max": float(best_final["inf_max"]),
        "weight_mae": float(best_final["weight_mae"]),
        "weight_max": float(best_final["weight_max"]),
        "layer_maes": best_final["layer_maes"],
        "max_g_mS": float(best_final["max_g_mS"]),
        "all_evaluations": sorted(list(eval_cache.values()), key=lambda x: x["v_cand"])
    }

    def _json_serial(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return str(obj)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=_json_serial)
    print(f"[SAVE] Full evaluation history saved to {out_json}")

    # Update selected preset .txt file
    if selected_preset:
        update_preset_txt_safecap(best_final["v_cand"], selected_preset["file_path"])

    # Optionally update PythonSimulator/molmem_lib/devices.py
    if args.apply:
        update_devices_py_safecap(best_final["v_cand"], device_type=target_device_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
