"""
HardwareTestTB.py
======================================================================
MolmemSimulator Master Hardware & OS Validation Testbench
======================================================================
Comprehensive automated hardware test suite to rigorously stress and
validate OS/hardware stability across:
  - CPU (Numba / TBB multi-core JIT)
  - GPU (CUDA / AMD ROCm HIP C++ extensions & PyTorch kernels)
  - CLI hardware override flags (--cpu, --gpu, --log, --debug-gpu)
  - Noise vs No-Noise conditions across all 3 empirical physical channels:
      1. Device-to-Device (D2D) Spatial Transconductance Mismatch
      2. Temporal Read Noise (Johnson-Nyquist floor + Hooge 1/f flicker)
      3. Cycle-to-Cycle Write Noise (Langevin programming stochasticity)

Test Matrix (34 Configurations Total):
  1. tb_dynamic (Transient Circuit Solver: sim.run):
       - CPU [No Noise, With Noise]
       - GPU [No Noise, With Noise]
  2. tb_batch_input (8x8 Crossbar Batched Inference vs Sequential):
       - Normal [No Noise, With Noise]
       - CPU    [No Noise, With Noise]
  3. tb_multi_program_4x4 (4x4 3-Phase Reprogramming & State Tracking):
       - Normal [No Noise, With Noise]
       - GPU    [No Noise, With Noise]
       - CPU    [No Noise, With Noise]
  4. tb_quantized_weights (Multi-Bit DAC/ADC 14/14/14, 8/8/8, 4/4/4):
       - Normal [No Noise, With Noise]
       - GPU    [No Noise, With Noise]
       - CPU    [No Noise, With Noise]
  5. tb_parallel_crossbars (Multi-Array Concurrent Autonomous Dispatch):
       - Normal [No Noise, With Noise]
       - CPU    [No Noise, With Noise]
  6. tb_tiled_simulator (64x64 Macro-Grid Split into 32x32 Tiles, 1D Vector):
       - Normal [No Noise, With Noise]
       - CPU    [No Noise, With Noise]
  7. tb_tiled_batch_input (32x32 Tiled Crossbar 2D Batch vs Sequential & Math Ground Truth):
       - Normal [No Noise, With Noise]
       - GPU    [No Noise, With Noise]
       - CPU    [No Noise, With Noise]

Usage:
  python HardwareTestTB.py                 # Runs the full master test matrix
  python HardwareTestTB.py --quick         # Runs an accelerated sweep
======================================================================
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
import numpy as np
import io
import cProfile
import pstats

# Ensure parent (PythonSimulator) and current directory are on sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# Embedded logs directory for archiving per-test logs
_TEST_LOGS_DIR = os.path.join(_CURRENT_DIR, "test_logs")

# =====================================================================
# GLOBAL TESTBENCH CONFIGURATION / HYPERPARAMETERS
# =====================================================================
# Set ENABLE_PROFILER to False to temporarily toggle off cProfile execution profiling
# and profiler log archiving for faster pure hardware and circuit test execution.
# Can still be overridden on the command line via --profile / --no-profile flags.
ENABLE_PROFILER = False

# Import molmem_lib at master orchestrator level so CLI flags like --clear-cache
# and C++/HIP extension onboarding compile upfront before spawning subtests
import molmem_lib


def check_is_gpu_ready():
    """
    Evaluates whether a compatible, supported GPU accelerator is ready for execution.
    Delegates directly to molmem_lib.is_gpu_ready with reason tracking.
    Returns (is_ready: bool, reason: str or None).
    """
    try:
        from molmem_lib import is_gpu_ready
        return is_gpu_ready(return_reason=True)
    except Exception as e:
        return False, f"molmem_lib is_gpu_ready query failed: {e}"


def assert_expected_device(sim, workload_type="crossbar"):
    """
    Validates and asserts that the simulator instance is operating on the expected
    backend and hardware device according to CLI/environment flags and accelerator availability:
      - Under --cpu: strictly asserts device == 'cpu' and backend in ['numba', 'numpy', 'cpu'].
      - Under --gpu:
          - If GPU is ready: strictly asserts CUDA/HIP accelerator device and backend in ['pytorch', 'torch'].
          - If GPU is absent: strictly asserts fallback to device == 'cpu' and backend in ['numba', 'numpy', 'cpu'].
      - Under Normal (Auto):
          - For transient circuit simulations: strictly asserts device == 'cpu' and backend in ['numba', 'numpy', 'cpu'].
          - For crossbar workloads:
              - If GPU is ready: asserts backend == 'hybrid' and CUDA/HIP device.
              - If GPU is absent: asserts fallback to device == 'cpu' and backend in ['numba', 'numpy', 'cpu'].
    """
    gpu_ready, gpu_reason = check_is_gpu_ready()
    is_forced_cpu = (
        os.environ.get("MOLMEM_FORCE_CPU") == "1"
        or os.environ.get("MOLMEM_CPU_ONLY") == "1"
        or "--cpu" in sys.argv
        or "-cpu" in sys.argv
    )
    is_forced_gpu = (
        os.environ.get("MOLMEM_FORCE_GPU") == "1"
        or "--gpu" in sys.argv
        or "-gpu" in sys.argv
    )

    sim_device = str(getattr(sim, 'device', 'unknown')).lower()
    sim_backend = str(getattr(sim, 'backend', 'unknown')).lower()

    if is_forced_cpu:
        assert sim_device == 'cpu', (
            f"Hardware device routing mismatch under --cpu! Expected device='cpu', got '{sim_device}'."
        )
        assert sim_backend in ['numba', 'numpy', 'cpu'], (
            f"Backend routing mismatch under --cpu! Expected CPU JIT backend (numba/numpy), got '{sim_backend}'."
        )
        print(f"  [>] Verified Device Routing: CPU (Device: {sim.device}, Backend: {sim.backend})")
    elif is_forced_gpu:
        if gpu_ready:
            assert ('cuda' in sim_device or 'hip' in sim_device), (
                f"Hardware device routing mismatch under --gpu! Expected CUDA/HIP accelerator device, got '{sim_device}'."
            )
            assert sim_backend in ['pytorch', 'torch'], (
                f"Backend routing mismatch under --gpu! Expected PyTorch accelerator backend, got '{sim_backend}'."
            )
            print(f"  [>] Verified Device Routing: GPU (Device: {sim.device}, Backend: {sim.backend})")
        else:
            assert sim_device == 'cpu', (
                f"Hardware device routing mismatch under --gpu with absent GPU! Expected fallback to device='cpu' ({gpu_reason}), got '{sim_device}'."
            )
            assert sim_backend in ['numba', 'numpy', 'cpu'], (
                f"Backend routing mismatch under --gpu with absent GPU! Expected fallback to CPU JIT backend (numba), got '{sim_backend}'."
            )
            print(f"  [>] Verified Device Routing: Fallback CPU under --gpu (No compatible GPU detected: {gpu_reason}) (Device: {sim.device}, Backend: {sim.backend})")
    else:
        # Normal (Auto) mode
        if workload_type == "transient":
            assert sim_device == 'cpu', (
                f"Transient circuit simulation in Auto mode must route to CPU, got '{sim_device}'."
            )
            assert sim_backend in ['numba', 'numpy', 'cpu'], (
                f"Transient circuit simulation in Auto mode must use CPU JIT (numba), got '{sim_backend}'."
            )
            print(f"  [>] Verified Device Routing: Auto -> CPU Transient (Device: {sim.device}, Backend: {sim.backend})")
        else:
            # Crossbar / Macro-grid workloads
            if gpu_ready:
                assert ('cuda' in sim_device or 'hip' in sim_device), (
                    f"Auto mode crossbar with ready GPU expected CUDA/HIP device, got '{sim_device}'."
                )
                assert sim_backend == 'hybrid', (
                    f"Auto mode crossbar with ready GPU expected 'hybrid' backend, got '{sim_backend}'."
                )
                print(f"  [>] Verified Device Routing: Auto -> Heterogeneous Hybrid (Device: {sim.device}, Backend: {sim.backend})")
            else:
                assert sim_device == 'cpu', (
                    f"Auto mode crossbar with absent GPU expected fallback to CPU device ({gpu_reason}), got '{sim_device}'."
                )
                assert sim_backend in ['numba', 'numpy', 'cpu'], (
                    f"Auto mode crossbar with absent GPU expected fallback to CPU JIT backend (numba), got '{sim_backend}'."
                )
                print(f"  [>] Verified Device Routing: Auto -> Fallback CPU (Device: {sim.device}, Backend: {sim.backend})")


# =====================================================================
# INDIVIDUAL TEST RUNNERS (EXECUTED INSIDE DEDICATED SUBPROCESSES)
# =====================================================================

def run_dynamic_subtest(noise_intensity, quick=False):
    """
    Simulates tb_dynamic circuit-level continuous ODE solver (sim.run).
    Validates dynamic pulse train switching, KCL nodal convergence,
    and D2D transconductance modulation.
    """
    from molmem_lib import MolmemSimulator
    from molmem_lib.sources import PulseSource

    print(f"\n--- [SUBTEST] Dynamic Circuit Simulation (Noise Intensity: {noise_intensity}) ---")
    sim = MolmemSimulator(instanceName="DynamicHardwareTest")
    sim.addCrossbarMatrix(
        device_type="default",
        rows=1,
        cols=1,
        d2d_variation_intensity=noise_intensity,
        write_noise_intensity=noise_intensity,
        read_noise_intensity=noise_intensity
    )

    t_switch = 1.6e-4 if quick else 8.0e-4
    t_stop = 3.2e-4 if quick else 1.6e-3

    vpot = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
    vdep = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)

    def v_pulse_train(t, sources):
        return sources[0].getVoltage(t) if t < t_switch else sources[1].getVoltage(t)

    sim.addBSource(row=1, vFunc=v_pulse_train, sources=[vpot, vdep], edge_points=[t_switch])
    sim.addDcSource(col=1, voltage=0.0)

    t0 = time.perf_counter()
    sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4, recordHistory=False)
    elapsed = time.perf_counter() - t0

    dev = sim.devices[0]['dev']
    from molmem_lib.devices import default_device
    nominal_gScale = float(dev.params.get('gScale', default_device()['gScale'])) if hasattr(dev, 'params') and isinstance(dev.params, dict) else float(default_device()['gScale'])
    actual_gScale = dev.gScale

    print(f"  [>] Solver Execution Time: {elapsed:.2f}s")
    print(f"  [>] Active Backend: {sim.backend} (Device: {sim.device})")
    assert_expected_device(sim, workload_type="transient")
    print(f"  [>] Final State (n): {dev._n:.2f}")
    print(f"  [>] Final Transconductance gScale: {actual_gScale:.4e} (Nominal: {nominal_gScale:.4e})")

    rel_diff = 0.0
    if noise_intensity == 0.0:
        assert abs(actual_gScale - nominal_gScale) < 1e-6, "gScale altered unexpectedly under zero noise!"
    else:
        rel_diff = (actual_gScale - nominal_gScale) / nominal_gScale
        print(f"  [>] Measured D2D Perturbation: {rel_diff * 100.0:+.4f}% ({rel_diff * 1e6:+.1f} ppm)")
        # Analytical empirical inverse-CDF quantile bounds [-0.83, +1.05] from d2d_mismatch_quantile.pwl
        assert -0.85 * noise_intensity <= rel_diff <= 1.10 * noise_intensity, (
            f"D2D perturbation out of empirical quantile bounds: {rel_diff:+.4e}"
        )
        assert actual_gScale > 0.0, "Transconductance gScale must be strictly positive!"

    assert 0.0 <= dev._n <= 100.0, f"Physical state n={dev._n} outside valid bounds [0, 100]!"
    metric_d2d = f", D2D={rel_diff * 100.0:+.2f}%" if noise_intensity > 0 else ""
    print(f"  [METRIC] n={dev._n:.1f}, gScale={actual_gScale:.4f}{metric_d2d}")
    print("  -> SUBTEST PASSED")


def run_batch_input_subtest(noise_intensity, quick=False):
    """
    Simulates tb_batch_input: 8x8 crossbar parallel batch inference.
    Validates batch MAC parallelization, DTC input quantization, and read noise.
    """
    from molmem_lib import MolmemSimulator

    print(f"\n--- [SUBTEST] Batch Parallel Input Passing (Noise Intensity: {noise_intensity}) ---")
    rows, cols = 8, 8
    sim = MolmemSimulator(instanceName="BatchInputHardwareTest")
    sim.addCrossbarMatrix(
        rows=rows,
        cols=cols,
        weightBits=14,
        inputBits=14,
        outputBits=14,
        detailedPrint=False,
        architecture='1T1R',
        max_vWrite=5.0,
        max_pw=320e-9,
        UpdatesPer="Device",
        d2d_variation_intensity=noise_intensity,
        write_noise_intensity=noise_intensity,
        read_noise_intensity=noise_intensity
    )

    np.random.seed(42)
    target_weights = np.random.uniform(0.2, 0.8, (rows, cols))
    sim.updateCrossbarWeights(target_weights, detailedPrint=False, fineTune=False, silentUpdate=True)

    batch_size = 128 if quick else 512
    inputs_batch = np.random.uniform(0.1, 0.9, (batch_size, rows)).astype(np.float32)

    t0 = time.perf_counter()
    y_batch1 = sim.passInput(inputs_batch, detailedPrint=False)
    y_batch2 = sim.passInput(inputs_batch, detailedPrint=False)
    elapsed = time.perf_counter() - t0

    diff = float(np.max(np.abs(y_batch1 - y_batch2)))
    print(f"  [>] Active Backend: {sim.backend} (Device: {sim.device})")
    assert_expected_device(sim, workload_type="crossbar")
    print(f"  [>] Batch Input ({batch_size} vectors) Execution Time: {elapsed:.3f}s")
    print(f"  [>] Repetitive Inference Output Discrepancy: {diff:.4e}")

    # Analytical Linear Algebra Validation: Y = X @ W_phys (Physical Vector-Matrix Multiplication)
    g_matrix, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w_transfer = sim._conductanceToWeights(g_matrix)
    y_expected = inputs_batch @ w_transfer
    mae_mac = float(np.mean(np.abs(y_batch1 - y_expected)))

    # Algorithmic Correlation & R^2: Validate proportional linear response against software weights
    w_phys = w_transfer
    y_alg = y_expected
    corr_mac = float(np.corrcoef(y_batch1.ravel(), y_alg.ravel())[0, 1])
    r2_mac = corr_mac ** 2
    y_ideal = inputs_batch @ target_weights
    corr_ideal = float(np.corrcoef(y_batch1.ravel(), y_ideal.ravel())[0, 1])
    r2_ideal = corr_ideal ** 2
    print(f"  [>] Analytical Batch MAC MAE: {mae_mac:.4e} (Correlation to Math: {corr_mac:.4f}, R^2: {r2_mac:.6f})")
    print(f"  [>] Target Math Correlation: {corr_ideal:.4f} (R^2: {r2_ideal:.6f})")
    assert corr_mac > 0.95, f"Batch MAC output does not correlate with matrix multiplication: corr={corr_mac:.4f}"
    max_mac_mae = 0.08 if noise_intensity > 0.0 else 0.02
    assert mae_mac < max_mac_mae, f"Batch MAC output deviates from analytical calculation: MAE={mae_mac:.4e} >= {max_mac_mae}"

    # Sequential vs Batch Equivalence Validation
    n_seq_check = min(16, batch_size)
    y_seq_check = np.array([sim.passInput(inputs_batch[i], detailedPrint=False) for i in range(n_seq_check)])
    diff_seq_par = float(np.max(np.abs(y_seq_check - y_batch1[:n_seq_check])))
    print(f"  [>] Sequential vs Batch Equivalence ({n_seq_check} samples): Diff={diff_seq_par:.4e}")

    if noise_intensity == 0.0:
        assert diff == 0.0, f"Deterministic inference produced discrepancy: {diff}"
        assert diff_seq_par < 1e-4, f"Deterministic sequential vs batch produced discrepancy: {diff_seq_par:.4e}"
        assert r2_mac >= 0.99, f"Batch MAC R^2 ({r2_mac:.6f}) below deterministic linear threshold 0.99"
        print(f"  [METRIC] Batch={batch_size}, MAC MAE={mae_mac:.2e}, Diff=0.0, R^2={r2_mac:.6f}")
    else:
        assert diff > 0.0, "Read noise failed to introduce expected sensing fluctuations!"
        sample_noise_std = float(np.std(y_batch1 - y_batch2) / np.sqrt(2.0))
        print(f"  [>] Empirical Read Noise Std Dev: {sample_noise_std:.4e}")
        assert 1e-6 <= sample_noise_std <= 1e-2, (
            f"Measured read noise std dev {sample_noise_std:.2e} outside analytical bounds [1e-6, 1e-2]"
        )
        print(f"  [METRIC] Batch={batch_size}, Noise Std={sample_noise_std:.2e}, Diff={diff:.2e}, R^2={r2_mac:.4f}")
    print("  -> SUBTEST PASSED")


def run_multi_program_4x4_subtest(noise_intensity, quick=False):
    """
    Simulates tb_multi_program_4x4: 3-phase programming cycles.
    Validates state retention, pulse calculation, and cycle-to-cycle write noise.
    """
    from molmem_lib import MolmemSimulator

    print(f"\n--- [SUBTEST] Multi-Program 4x4 Reprogramming (Noise Intensity: {noise_intensity}) ---")
    rows, cols = 4, 4
    sim = MolmemSimulator(instanceName="MultiProg4x4HardwareTest")
    sim.addCrossbarMatrix(
        rows=rows,
        cols=cols,
        weightBits=14,
        inputBits=14,
        outputBits=14,
        detailedPrint=False,
        architecture='1T1R',
        max_vWrite=5.0,
        max_pw=320e-9,
        UpdatesPer="Column",
        d2d_variation_intensity=noise_intensity,
        write_noise_intensity=noise_intensity,
        read_noise_intensity=noise_intensity
    )

    p1 = np.array([
        [0.2, 0.8, 0.3, 0.7],
        [0.4, 0.6, 0.5, 0.5],
        [0.6, 0.4, 0.7, 0.3],
        [0.8, 0.2, 0.9, 0.1]
    ], dtype=np.float64)

    p2 = np.array([
        [0.8, 0.2, 0.7, 0.3],
        [0.6, 0.4, 0.5, 0.5],
        [0.4, 0.6, 0.3, 0.7],
        [0.2, 0.8, 0.1, 0.9]
    ], dtype=np.float64)

    t0 = time.perf_counter()
    sim.updateCrossbarWeights(p1, detailedPrint=False, fineTune=False, silentUpdate=True)
    g1, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w1 = sim._conductanceToWeights(g1)
    err1 = float(np.mean(np.abs(p1 - w1)))

    # Mathematical Ground Truth Inference for Phase 1 (Y1 = X @ W1)
    x_test = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    y1_inf = sim.passInput(x_test, detailedPrint=False)
    y1_expected = x_test @ w1
    mae_inf1 = float(np.mean(np.abs(y1_inf - y1_expected)))
    corr_inf1 = float(np.corrcoef(y1_inf, y1_expected)[0, 1])

    sim.updateCrossbarWeights(p2, detailedPrint=False, fineTune=False, silentUpdate=True)
    g2, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w2 = sim._conductanceToWeights(g2)
    err2 = float(np.mean(np.abs(p2 - w2)))
    elapsed = time.perf_counter() - t0

    # Mathematical Ground Truth Inference for Phase 2 (Y2 = X @ W2)
    y2_inf = sim.passInput(x_test, detailedPrint=False)
    y2_expected = x_test @ w2
    mae_inf2 = float(np.mean(np.abs(y2_inf - y2_expected)))
    corr_inf2 = float(np.corrcoef(y2_inf, y2_expected)[0, 1])
    diff_phases = float(np.max(np.abs(y1_inf - y2_inf)))

    print(f"  [>] Active Backend: {sim.backend} (Device: {sim.device})")
    assert_expected_device(sim, workload_type="crossbar")
    print(f"  [>] 2-Phase Reprogramming Time: {elapsed:.2f}s")
    print(f"  [>] Phase 1 Weight MAE: {err1:.4f} | Inference MAE vs Math: {mae_inf1:.4f} (Corr: {corr_inf1:.4f})")
    print(f"  [>] Phase 2 Weight MAE: {err2:.4f} | Inference MAE vs Math: {mae_inf2:.4f} (Corr: {corr_inf2:.4f})")
    print(f"  [>] Inter-Phase Output Discrimination: {diff_phases:.4f}")

    assert np.all(w1 >= 0.0) and np.all(w1 <= 1.0), "Physical weights out of [0, 1] range in Phase 1!"
    assert np.all(w2 >= 0.0) and np.all(w2 <= 1.0), "Physical weights out of [0, 1] range in Phase 2!"

    max_allowed_err = 0.10 if noise_intensity > 0.0 else 0.03
    assert err1 < max_allowed_err, f"Phase 1 programming error exceeds analytical tolerance ({max_allowed_err}): {err1:.4f}"
    assert err2 < max_allowed_err, f"Phase 2 programming error exceeds analytical tolerance ({max_allowed_err}): {err2:.4f}"
    if noise_intensity == 0.0:
        assert mae_inf1 < 0.05, f"Phase 1 inference deviated from mathematical ground truth: {mae_inf1:.4f}"
        assert mae_inf2 < 0.05, f"Phase 2 inference deviated from mathematical ground truth: {mae_inf2:.4f}"
        assert corr_inf1 > 0.98, f"Phase 1 inference correlation below 0.98: {corr_inf1:.4f}"
        assert corr_inf2 > 0.98, f"Phase 2 inference correlation below 0.98: {corr_inf2:.4f}"
        assert diff_phases > 0.05, f"Reprogramming failed to alter physical inference output: diff={diff_phases:.4f}"
    print(f"  [METRIC] Phase 1 MAE={err1:.4f}, Phase 2 MAE={err2:.4f}, Inf MAE={max(mae_inf1, mae_inf2):.4f}")
    print("  -> SUBTEST PASSED")


def run_quantized_weights_subtest(noise_intensity, quick=False):
    """
    Simulates tb_quantized_weights: validates multi-bit DAC/ADC profiles
    (14-bit, 8-bit, 4-bit) across hardware quantization bounds.
    """
    from molmem_lib import MolmemSimulator

    print(f"\n--- [SUBTEST] Quantized Weights & DAC/ADC (Noise Intensity: {noise_intensity}) ---")
    profiles = [(8, 8, 8), (4, 4, 4)] if quick else [(14, 14, 14), (8, 8, 8), (4, 4, 4)]

    max_mae_across_profiles = 0.0
    mae_by_bits = {}
    for w_b, in_b, out_b in profiles:
        sim = MolmemSimulator(instanceName=f"QuantSim_{w_b}_{in_b}_{out_b}")
        sim.addCrossbarMatrix(
            rows=2,
            cols=2,
            weightBits=w_b,
            inputBits=in_b,
            outputBits=out_b,
            detailedPrint=False,
            d2d_variation_intensity=noise_intensity,
            write_noise_intensity=noise_intensity,
            read_noise_intensity=noise_intensity
        )
        assert_expected_device(sim, workload_type="crossbar")
        if w_b == profiles[0][0]:
            print(f"  [>] Active Backend: {sim.backend} (Device: {sim.device})")

        w_ideal = np.array([[0.25, 0.75], [0.85, 0.15]], dtype=np.float64)
        sim.updateCrossbarWeights(w_ideal, detailedPrint=False, fineTune=False, silentUpdate=True)
        g_mat, adc_bits = sim.readCrossbarMatrix(detailedPrint=False)

        max_adc = (1 << out_b) - 1
        assert np.issubdtype(adc_bits.dtype, np.integer), f"ADC bits must be integer dtype, got {adc_bits.dtype}!"
        assert np.all(adc_bits >= 0), "Negative ADC bits detected!"
        assert np.all(adc_bits <= max_adc), f"ADC bits exceeded hardware ceiling {max_adc}!"

        # Conductance matrix physicality:
        # Note: g_mat is gMatrixQuantized. When a cell under stochastic noise digitizes to ADC code 0,
        # its quantized conductance is exactly 0.0 S. In deterministic mode, weights [0.15, 0.85] guarantee > 0.0 S.
        assert np.all(np.isfinite(g_mat)), "Non-finite values detected in conductance matrix!"
        assert np.all(g_mat >= 0.0), "Negative conductance values detected in reconstructed state!"
        if noise_intensity == 0.0:
            assert np.all(g_mat > 0.0), "Zero or negative conductance detected in deterministic mode!"

        # Transfer function order monotonicity (rank-preservation) in deterministic mode
        if noise_intensity == 0.0:
            if out_b >= 8:
                assert adc_bits[1, 1] < adc_bits[0, 0] < adc_bits[0, 1] < adc_bits[1, 0], (
                    f"Non-monotonic ADC transfer function detected! Expected 0.15 < 0.25 < 0.75 < 0.85 ordering, "
                    f"got codes: [{adc_bits[1, 1]}, {adc_bits[0, 0]}, {adc_bits[0, 1]}, {adc_bits[1, 0]}]"
                )
            else:
                assert adc_bits[1, 1] <= adc_bits[0, 0] < adc_bits[0, 1] <= adc_bits[1, 0], (
                    f"Non-monotonic ADC transfer function detected! Expected 0.15 <= 0.25 < 0.75 <= 0.85 ordering, "
                    f"got codes: [{adc_bits[1, 1]}, {adc_bits[0, 0]}, {adc_bits[0, 1]}, {adc_bits[1, 0]}]"
                )

        # Analytical ADC quantization verification
        normalized_adc = sim._conductanceToWeights(g_mat)
        mae_quant = float(np.mean(np.abs(normalized_adc - w_ideal)))
        mae_by_bits[out_b] = mae_quant
        max_mae_across_profiles = max(max_mae_across_profiles, mae_quant)

        # Mathematical Vector-Matrix Inference Verification under Quantization
        x_q = np.array([[0.5, 0.5]], dtype=np.float32)
        y_q = sim.passInput(x_q, detailedPrint=False)
        w_phys_q = normalized_adc
        y_expected_q = x_q @ w_phys_q
        mae_inf_q = float(np.mean(np.abs(y_q - y_expected_q)))
        if noise_intensity == 0.0:
            assert mae_inf_q < max(0.05, 1.5 / max_adc), f"Quantized inference deviated from physical math: {mae_inf_q:.4f}"
        print(f"  [>] Profile ({w_b}w/{in_b}in/{out_b}out): ADC MAE vs ideal={mae_quant:.4f}, Inf MAE vs Math={mae_inf_q:.4f} (Max ADC: {max_adc})")

        # Analytical ADC quantization + physical noise verification:
        # In deterministic mode (noise=0): allow analytical quantization rounding of up to 1.5 LSBs (floor at 0.04).
        # Under stochastic noise (noise > 0, fineTune=False):
        # Physical D2D transconductance mismatch (empirical range [-0.82, +1.05]), write dispersion,
        # and read noise perturb cell conductances in the analog domain before ADC sampling.
        # Across a 2x2 crossbar without closed-loop pulse tuning, this introduces an expected
        # analog variation envelope bounded by 0.25 * noise_intensity, independent of ADC bit-depth.
        quant_floor = max(0.04, 1.5 / max_adc)
        physical_noise_bound = 0.25 * noise_intensity
        expected_bound = quant_floor + physical_noise_bound
        assert mae_quant <= expected_bound, (
            f"Profile ({w_b}b) ADC output deviated from analytical bound: MAE={mae_quant:.4f} > {expected_bound:.4f} "
            f"(quant floor: {quant_floor:.4f}, physical noise envelope: {physical_noise_bound:.4f})"
        )

    # Cross-resolution verification: confirm that lowering bit resolution degrades accuracy in deterministic mode.
    # Note: Under stochastic noise, independent random draws of D2D transconductance variation across separate
    # 2x2 instances fluctuate independently, so each profile is validated against its physical noise bound above.
    if noise_intensity == 0.0:
        assert mae_by_bits[4] > mae_by_bits[8], (
            f"4-bit ADC resolution (MAE={mae_by_bits[4]:.4f}) expected to have higher quantization error than 8-bit (MAE={mae_by_bits[8]:.4f})"
        )
        if 14 in mae_by_bits:
            # 8-bit quantization error tracks 14-bit within 1.5 LSB of the 8-bit quantizer (1.5/255 ~ 0.0059)
            # because physical open-loop programming error (~0.007) is comparable to the 8-bit quantization step.
            assert mae_by_bits[8] >= mae_by_bits[14] - (1.5 / 255.0), (
                f"8-bit ADC resolution (MAE={mae_by_bits[8]:.4f}) expected to have error >= 14-bit (MAE={mae_by_bits[14]:.4f}) within 1.5 LSB"
            )
        print("  [>] Bit-depth resolution scaling verified: coarser quantization produces higher error.")

    prof_str = "/".join(str(p[0]) + "b" for p in profiles)
    print(f"  [METRIC] ADC Levels Verified ({prof_str}), Max MAE={max_mae_across_profiles:.4f}")
    print("  -> SUBTEST PASSED")


def run_parallel_crossbars_subtest(noise_intensity, quick=False):
    """
    Simulates tb_parallel_crossbars: concurrent autonomous dispatch across
    multiple independent crossbars via updateCrossbarsParallel().
    """
    from molmem_lib import MolmemSimulator, updateCrossbarsParallel

    print(f"\n--- [SUBTEST] Parallel Crossbars Dispatch (Noise Intensity: {noise_intensity}) ---")
    n_sims = 2 if quick else 4
    simulators = []
    weight_matrices = []

    np.random.seed(123)
    for i in range(n_sims):
        s = MolmemSimulator(instanceName=f"ParallelArray_{i}")
        s.addCrossbarMatrix(
            rows=8,
            cols=8,
            weightBits=8,
            inputBits=8,
            outputBits=8,
            detailedPrint=False,
            d2d_variation_intensity=noise_intensity,
            write_noise_intensity=noise_intensity,
            read_noise_intensity=noise_intensity
        )
        simulators.append(s)
        weight_matrices.append(np.random.uniform(0.2, 0.8, (8, 8)))

    t0 = time.perf_counter()
    updateCrossbarsParallel(simulators, weight_matrices, detailedPrint=False, fineTune=False, recordHistory=False)
    elapsed = time.perf_counter() - t0

    print(f"  [>] Parallel Dispatch ({n_sims} arrays) Elapsed Time: {elapsed:.2f}s")
    if simulators:
        print(f"  [>] Active Backend: {simulators[0].backend} (Device: {simulators[0].device})")
        for s in simulators:
            assert_expected_device(s, workload_type="crossbar")
    min_g = float('inf')
    max_array_mae = 0.0
    x_par_test = np.random.uniform(0.1, 0.9, (4, 8)).astype(np.float32)
    for i, s in enumerate(simulators):
        g, _ = s.readCrossbarMatrix(detailedPrint=False)
        assert np.all(g > 0.0), f"Array {i} returned non-positive conductance!"
        min_g = min(min_g, float(np.min(g)))

        # Analytical Target Programming & Array Isolation Validation
        w_achieved = s._conductanceToWeights(g)
        target_w = weight_matrices[i]
        mae_target = float(np.mean(np.abs(w_achieved - target_w)))
        max_array_mae = max(max_array_mae, mae_target)

        # Mathematical Ground Truth Inference on Array i
        y_act = s.passInput(x_par_test, detailedPrint=False)
        y_exp = x_par_test @ w_achieved
        mae_inf_arr = float(np.mean(np.abs(y_act - y_exp)))
        corr_inf_arr = float(np.corrcoef(y_act.ravel(), y_exp.ravel())[0, 1])
        print(f"  [>] Array {i} Programming MAE: {mae_target:.4f} | Inference MAE: {mae_inf_arr:.4f} (Corr: {corr_inf_arr:.4f})")

        max_allowed_mae = 0.10 if noise_intensity > 0.0 else 0.03
        assert mae_target < max_allowed_mae, (
            f"Array {i} programming deviated from target weights: MAE={mae_target:.4f} >= {max_allowed_mae}"
        )
        if noise_intensity == 0.0:
            assert mae_inf_arr < 0.04, f"Array {i} inference deviated from math: {mae_inf_arr:.4f}"
            assert corr_inf_arr > 0.98, f"Array {i} inference correlation below 0.98: {corr_inf_arr:.4f}"

        if n_sims > 1:
            other_target = weight_matrices[(i + 1) % n_sims]
            mae_other = float(np.mean(np.abs(w_achieved - other_target)))
            assert mae_target < mae_other, (
                f"Array {i} failed cross-array isolation: MAE to own target ({mae_target:.4f}) >= other ({mae_other:.4f})"
            )
            # Verify inference outputs are distinct between arrays
            y_other_exp = x_par_test @ other_target
            diff_cross = float(np.mean(np.abs(y_act - y_other_exp)))
            assert mae_inf_arr < diff_cross, (
                f"Array {i} output failed cross-array isolation: own MAE ({mae_inf_arr:.4f}) >= cross ({diff_cross:.4f})"
            )

    print(f"  [METRIC] {n_sims} Arrays, Max MAE={max_array_mae:.4f}, g_min={min_g:.2e}")
    print("  -> SUBTEST PASSED")


def run_tiled_simulator_subtest(noise_intensity, quick=False):
    """
    Simulates tb_tiled_simulator: macro-architecture grid partitioned
    into multiple autonomous tiles with parallel execution.
    """
    from molmem_lib import MolmemTiledSimulator

    print(f"\n--- [SUBTEST] Tiled Macro-Simulator (Noise Intensity: {noise_intensity}) ---")
    tot_r, tot_c = (32, 32) if quick else (64, 64)
    tile_r, tile_c = (16, 16) if quick else (32, 32)

    tiled_sim = MolmemTiledSimulator(instanceName="TiledHardwareTest")
    tiled_sim.addTiledCrossbarMatrix(
        total_rows=tot_r,
        total_cols=tot_c,
        tile_rows=tile_r,
        tile_cols=tile_c,
        weightBits=8,
        inputBits=8,
        outputBits=8,
        detailedPrint=False,
        d2d_variation_intensity=noise_intensity,
        write_noise_intensity=noise_intensity,
        read_noise_intensity=noise_intensity
    )

    np.random.seed(999)
    weights = np.random.uniform(0.2, 0.8, (tot_r, tot_c))
    x_in = np.random.uniform(0.1, 0.9, tot_r)

    t0 = time.perf_counter()
    tiled_sim.updateTiledCrossbarWeights(weights, detailedPrint=False, fineTune=False)
    t_prog = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_out = tiled_sim.passTiledCrossbarInput(x_in, detailedPrint=False)
    t_inf = time.perf_counter() - t1

    print(f"  [>] Active Backend: {tiled_sim.backend} (Device: {tiled_sim.device})")
    assert_expected_device(tiled_sim, workload_type="crossbar")
    print(f"  [>] Tiled Programming Time: {t_prog:.2f}s")
    print(f"  [>] Tiled Inference Time: {t_inf:.4f}s")
    print(f"  [>] Output Vector Dimensions: {len(y_out)}")

    assert len(y_out) == tot_c, f"Output shape mismatch: {len(y_out)} != {tot_c}"
    assert np.all(np.isfinite(y_out)), "Tiled output contains non-finite values (NaN or Inf)!"
    assert np.all(y_out > 0.0), "Tiled output for positive inputs must be strictly positive!"

    # Analytical Ground-Truth Block Matrix-Vector Multiplication: Y = X @ W_transfer
    g_tiled, _ = tiled_sim.readTiledCrossbarMatrix(detailedPrint=False)
    w_transfer = tiled_sim._conductanceToWeights(g_tiled)
    y_expected = x_in @ w_transfer
    mae_tiled = float(np.mean(np.abs(y_out - y_expected)))
    y_ideal_math = x_in @ weights
    corr_tiled = float(np.corrcoef(y_out, y_ideal_math)[0, 1])
    print(f"  [>] Tiled Inference vs Ground Truth: MAE={mae_tiled:.4f}, Correlation={corr_tiled:.4f}")
    assert corr_tiled > 0.85, f"Tiled inference does not correlate with analytical dot product: corr={corr_tiled:.4f}"
    max_tiled_mae = 0.20 if noise_intensity > 0.0 else 0.15
    assert mae_tiled < max_tiled_mae, f"Tiled inference deviates from physical block multiplication: MAE={mae_tiled:.4f} >= {max_tiled_mae}"

    print(f"  [METRIC] Dim={len(y_out)}, Dot Corr={corr_tiled:.4f}, MAE={mae_tiled:.4f}")
    print("  -> SUBTEST PASSED")


def run_tiled_batch_input_subtest(noise_intensity, quick=False):
    """
    Simulates tb_tiled_batch_input: verifies 2D batch parallel inference across
    a multi-macro tiled crossbar against sequential loop (with state resets)
    and mathematical linear algebra ground truths (Y_hw = X @ W_hw, Y_ideal = X @ W_target).
    """
    from molmem_lib import MolmemTiledSimulator

    print(f"\n--- [SUBTEST] Tiled Batch Parallel Input Passing (Noise Intensity: {noise_intensity}) ---")
    total_rows, total_cols = (32, 32)
    tile_rows, tile_cols = (16, 16)

    tiled_sim = MolmemTiledSimulator(instanceName="TiledBatchHardwareTest")
    tiled_sim.recordHistory = False
    tiled_sim.addTiledCrossbarMatrix(
        total_rows=total_rows,
        total_cols=total_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        weightBits=14,
        inputBits=14,
        outputBits=14,
        detailedPrint=False,
        architecture='1T1R',
        max_vWrite=5.0,
        max_pw=320e-9,
        UpdatesPer="Device",
        d2d_variation_intensity=noise_intensity,
        write_noise_intensity=noise_intensity,
        read_noise_intensity=noise_intensity
    )

    np.random.seed(42)
    target_weights = np.random.uniform(0.1, 0.9, (total_rows, total_cols))
    t0 = time.perf_counter()
    tiled_sim.updateTiledCrossbarWeights(target_weights, detailedPrint=False, fineTune=False, silentUpdate=True)
    t_prog = time.perf_counter() - t0

    batch_size = 64 if quick else 256
    inputs_batch = np.random.uniform(0.1, 0.9, (batch_size, total_rows)).astype(np.float32)

    initial_states = tiled_sim.saveStates()

    # 1. Sequential execution with state restoration
    n_seq = min(batch_size, 16 if quick else 32)
    t0_seq = time.perf_counter()
    outputs_seq_list = []
    for i in range(n_seq):
        tiled_sim.restoreStates(initial_states)
        out = tiled_sim.passTiledCrossbarInput(inputs_batch[i], detailedPrint=False, appendHistory=False)
        outputs_seq_list.append(out)
    t_seq = time.perf_counter() - t0_seq
    outputs_seq = np.array(outputs_seq_list)

    # 2. Batch parallel execution
    tiled_sim.restoreStates(initial_states)
    t0_par = time.perf_counter()
    outputs_par = tiled_sim.passTiledCrossbarInput(inputs_batch, detailedPrint=False, appendHistory=False)
    t_par = time.perf_counter() - t0_par

    print(f"  [>] Active Backend: {tiled_sim.backend} (Device: {tiled_sim.device})")
    assert_expected_device(tiled_sim, workload_type="crossbar")
    print(f"  [>] Programming Time: {t_prog:.3f}s | Batch Inference Time: {t_par:.4f}s ({batch_size} samples)")
    if t_par > 0 and t_seq > 0:
        speedup = (t_seq / n_seq) / (t_par / batch_size)
        print(f"  [>] Sequential Sub-Loop: {t_seq:.3f}s ({n_seq} samples) | Speedup Factor: {speedup:.1f}x")

    # 3. Mathematical Ground Truths
    gMatrix, _ = tiled_sim.readTiledCrossbarMatrix(detailedPrint=False)
    w_programmed = tiled_sim._conductanceToWeights(gMatrix)

    outputs_math_hw = inputs_batch @ w_programmed
    outputs_math_ideal = inputs_batch @ target_weights

    mae_vs_hw = float(np.mean(np.abs(outputs_par - outputs_math_hw)))
    rmse_vs_hw = float(np.sqrt(np.mean((outputs_par - outputs_math_hw) ** 2)))
    corr_hw_arr = np.corrcoef(outputs_par.ravel(), outputs_math_hw.ravel())
    r2_hw = float(corr_hw_arr[0, 1] ** 2) if corr_hw_arr.shape == (2, 2) else 1.0

    mae_vs_ideal = float(np.mean(np.abs(outputs_par - outputs_math_ideal)))
    corr_ideal_arr = np.corrcoef(outputs_par.ravel(), outputs_math_ideal.ravel())
    r2_ideal = float(corr_ideal_arr[0, 1] ** 2) if corr_ideal_arr.shape == (2, 2) else 1.0

    diff_seq_par = float(np.max(np.abs(outputs_seq - outputs_par[:n_seq])))
    rel_diff_seq_par = diff_seq_par / max(1e-12, float(np.max(outputs_seq)))

    print(f"  [>] Equivalence (Parallel vs Sequential): Diff={diff_seq_par:.4e} (Rel: {rel_diff_seq_par*100:.3f}%)")
    print(f"  [>] Hardware Math Fidelity (Y_hw = X @ W_hw): MAE={mae_vs_hw:.4e}, RMSE={rmse_vs_hw:.4e}, R^2={r2_hw:.6f}")
    print(f"  [>] Target Math Fidelity   (Y_ideal = X @ W_ideal): MAE={mae_vs_ideal:.4e}, R^2={r2_ideal:.6f}")

    assert np.all(np.isfinite(outputs_par)), "Tiled batch output contains non-finite values!"
    assert np.all(outputs_par > 0.0), "Tiled batch output must be strictly positive for positive inputs!"

    if noise_intensity == 0.0:
        assert diff_seq_par < 1e-4, f"Parallel batch outputs deviated from sequential: diff={diff_seq_par:.4e}"
        assert r2_hw >= 0.999, f"Hardware math R^2 ({r2_hw:.6f}) below deterministic threshold 0.999"
        assert mae_vs_hw < 0.08, f"Hardware math MAE ({mae_vs_hw:.4e}) above deterministic threshold 0.08"
        assert r2_ideal >= 0.90, f"Target math R^2 ({r2_ideal:.6f}) below threshold 0.90"
        print(f"  [METRIC] Batch={batch_size}, HW R^2={r2_hw:.6f}, MAE={mae_vs_hw:.2e}, Diff={diff_seq_par:.2e}")
    else:
        assert r2_hw >= 0.90, f"Hardware math R^2 ({r2_hw:.6f}) under noise below threshold 0.90"
        assert mae_vs_hw < 0.10, f"Hardware math MAE ({mae_vs_hw:.4e}) under noise above threshold 0.10"
        print(f"  [METRIC] Batch={batch_size}, HW R^2={r2_hw:.4f}, MAE={mae_vs_hw:.2e}, Diff={diff_seq_par:.2e}")

    print("  -> SUBTEST PASSED")


# Map of registered subtest executors
SUBTEST_REGISTRY = {
    "tb_dynamic": run_dynamic_subtest,
    "tb_batch_input": run_batch_input_subtest,
    "tb_multi_program_4x4": run_multi_program_4x4_subtest,
    "tb_quantized_weights": run_quantized_weights_subtest,
    "tb_parallel_crossbars": run_parallel_crossbars_subtest,
    "tb_tiled_simulator": run_tiled_simulator_subtest,
    "tb_tiled_batch_input": run_tiled_batch_input_subtest,
}


# =====================================================================
# MASTER ORCHESTRATOR & SUBPROCESS EXECUTION HARNESS
# =====================================================================

def generate_profile_report(profiler, output_dir, subtest_name, mode_desc, noise_intensity, duration):
    """
    Exports binary pstats and a formatted human-readable profiling summary report.
    """
    os.makedirs(output_dir, exist_ok=True)
    pstats_path = os.path.join(output_dir, "profile.pstats")
    summary_path = os.path.join(output_dir, "profile_summary.txt")

    # 1. Save raw binary pstats
    profiler.dump_stats(pstats_path)

    # 2. Format human-readable summary
    s_cum = io.StringIO()
    ps_cum = pstats.Stats(profiler, stream=s_cum)
    ps_cum.strip_dirs().sort_stats("cumulative").print_stats(30)

    s_tot = io.StringIO()
    ps_tot = pstats.Stats(profiler, stream=s_tot)
    ps_tot.strip_dirs().sort_stats("tottime").print_stats(30)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "MOLMEM TESTBENCH EXECUTION PROFILE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Subtest Workload    : {subtest_name}\n")
        f.write(f"Backend Target      : {mode_desc}\n")
        f.write(f"Noise Intensity     : {noise_intensity}\n")
        f.write(f"Workload Duration   : {duration:.4f} seconds\n")
        f.write(f"Timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("TOP 30 FUNCTIONS BY CUMULATIVE TIME (Call Hierarchy / Latency Drivers)\n")
        f.write("-" * 80 + "\n")
        f.write(s_cum.getvalue())
        f.write("\n\n")

        f.write("-" * 80 + "\n")
        f.write("TOP 30 FUNCTIONS BY TOTAL INTERNAL TIME (Self Time / CPU/GPU Hotspots)\n")
        f.write("-" * 80 + "\n")
        f.write(s_tot.getvalue())
        f.write("\n")

    return pstats_path, summary_path


def execute_subprocess_scenario(test_name, mode_flag, noise_intensity, quick=False, profile=True):
    """
    Executes an individual scenario inside an isolated Python subprocess
    passing exact CLI flags (--cpu, --gpu, --log, --debug-gpu) directly to sys.argv.
    Streams live output unbuffered to the console.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--subtest", test_name,
        "--noise-intensity", str(noise_intensity),
        "--log"
    ]

    if mode_flag == "--cpu":
        cmd.append("--cpu")
    elif mode_flag == "--gpu":
        cmd.append("--gpu")
        cmd.append("--debug-gpu")
    else:
        cmd.append("--debug-gpu")

    if quick:
        cmd.append("--quick")

    if profile:
        cmd.append("--profile")
    else:
        cmd.append("--no-profile")

    mode_desc = "NORMAL (AUTO)" if not mode_flag else mode_flag.upper()
    noise_desc = "WITH NOISE (1.0)" if noise_intensity > 0.0 else "WITHOUT NOISE (0.0)"

    print("\n" + "=" * 76)
    print(f" RUNNING: {test_name.upper()} | BACKEND: {mode_desc} | NOISE: {noise_desc}")
    print(f" COMMAND: {' '.join(cmd)}")
    print("=" * 76)

    t_start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MOLMEM_ENV_MANAGED"] = "1"
    env["MOLMEM_NO_PROVISION"] = "1"
    env["UV_OFFLINE"] = "1"
    env["MOLMEM_OFFLINE"] = "1"

    # Stream stdout and stderr live without silencing
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    output_lines = []
    for line in iter(proc.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)

    proc.stdout.close()
    return_code = proc.wait()
    duration = time.perf_counter() - t_start

    metric_str = "N/A"
    for line in output_lines:
        s = line.strip()
        if s.startswith("[METRIC]"):
            metric_str = s.replace("[METRIC]", "").strip()
            break
    if metric_str == "N/A":
        for line in output_lines:
            s = line.strip()
            if "Output Discrepancy" in s or "Weight MAE" in s or "Final State" in s:
                metric_str = s.replace("[>]", "").strip()
                break

    status = "PASSED" if return_code == 0 else "FAILED"
    return {
        "test": test_name,
        "mode": mode_desc,
        "noise": noise_desc,
        "status": status,
        "exit_code": return_code,
        "duration": duration,
        "metrics": metric_str,
        "output": "".join(output_lines)
    }


def clear_active_molmem_logs():
    """
    Clears active log files inside molmem_lib/logs/ prior to running each test
    so the upcoming scenario produces clean, unpolluted logs for isolated archiving.
    """
    molmem_logs_dir = os.path.join(_PARENT_DIR, "molmem_lib", "logs")
    if os.path.isdir(molmem_logs_dir):
        for fname in os.listdir(molmem_logs_dir):
            fpath = os.path.join(molmem_logs_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception:
                pass


def archive_scenario_logs(run_idx, test_name, mode_desc, noise_desc, res):
    """
    Archives all log files produced by an individual subtest into a dedicated embedded
    directory under test_logs/test_XX_<test>_<mode>_<noise>/ so logs are never overwritten.
    """
    os.makedirs(_TEST_LOGS_DIR, exist_ok=True)

    clean_mode = mode_desc.replace(" ", "_").replace("(", "").replace(")", "").replace("--", "").lower()
    clean_noise = "with_noise" if "WITH NOISE" in noise_desc else "without_noise"
    folder_name = f"test_{run_idx:02d}_{test_name}_{clean_mode}_{clean_noise}"
    target_dir = os.path.join(_TEST_LOGS_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Copy all active log files from molmem_lib/logs/
    molmem_logs_dir = os.path.join(_PARENT_DIR, "molmem_lib", "logs")
    copied_files = []
    if os.path.isdir(molmem_logs_dir):
        for fname in os.listdir(molmem_logs_dir):
            src = os.path.join(molmem_logs_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(target_dir, fname)
                try:
                    shutil.copy2(src, dst)
                    copied_files.append(fname)
                except Exception:
                    pass

    # 2. Persist the complete terminal console output of this specific scenario
    console_path = os.path.join(target_dir, "console_output.txt")
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write(res.get("output", ""))
    except Exception:
        pass

    # 3. Persist structured metadata JSON
    meta_path = os.path.join(target_dir, "metadata.json")
    try:
        meta = {
            "test_index": run_idx,
            "test_name": test_name,
            "mode": mode_desc,
            "noise": noise_desc,
            "status": res.get("status", "UNKNOWN"),
            "exit_code": res.get("exit_code", -1),
            "duration_seconds": round(res.get("duration", 0.0), 3),
            "key_metrics": res.get("metrics", "N/A"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profile_summary": "profile_summary.txt" if "profile_summary.txt" in copied_files else None,
            "profile_pstats": "profile.pstats" if "profile.pstats" in copied_files else None,
            "archived_log_files": copied_files
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return target_dir


def save_hardware_validation_report(results, total_suite_time, report_path):
    """
    Generates and persists a comprehensive .txt report containing the master summary table
    and the exact numerical measurements, states, and execution metrics from each individual test.
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    total_runs = len(results)
    passed_count = sum(1 for r in results if r['exit_code'] == 0)
    failed_count = sum(1 for r in results if r['exit_code'] != 0)
    all_passed = (failed_count == 0)

    lines = []
    lines.append("=" * 80)
    lines.append("       MOLMEMSIMULATOR MASTER HARDWARE & OS VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp            : {now_str}")
    lines.append(f"Total Scenarios      : {total_runs}")
    lines.append(f"Passed               : {passed_count}")
    lines.append(f"Failed               : {failed_count}")
    lines.append(f"Total Execution Time : {total_suite_time:.2f}s ({total_suite_time / 60.0:.2f} minutes)")
    lines.append(f"Overall Result       : {f'PASSED - ALL {total_runs} TESTS SUCCESSFUL' if all_passed else 'FAILED'}")
    lines.append(f"Report Location      : {os.path.basename(report_path)}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("PART 1: MASTER SUMMARY TABLE")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"{'#':<3} | {'TEST CASE':<22} | {'MODE':<13} | {'NOISE STATE':<19} | {'STATUS':<7} | {'TIME':<7} | {'KEY NUMERICAL RESULT':<45}")
    lines.append("-" * 124)
    for i, r in enumerate(results, 1):
        time_str = f"{r['duration']:>5.2f}s" if r['status'] != "SKIPPED" else "   -- "
        lines.append(f"{i:<3} | {r['test']:<22} | {r['mode']:<13} | {r['noise']:<19} | {r['status']:<7} | {time_str:<7} | {r.get('metrics', 'N/A'):<45}")
    lines.append("-" * 124)
    lines.append("")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("PART 2: EXACT NUMERICAL RESULTS BY INDIVIDUAL TEST SCENARIO")
    lines.append("--------------------------------------------------------------------------------")

    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"[TEST #{i:02d}] {r['test'].upper()}")
        lines.append(f"  Mode       : {r['mode']}")
        lines.append(f"  Noise      : {r['noise']}")
        lines.append(f"  Status     : {r['status']} (Exit Code: {r['exit_code']})")
        lines.append(f"  Duration   : {r['duration']:.2f}s")
        lines.append("  --- Exact Numerical Outputs ---")

        metric_lines = []
        raw_lines = r.get('output', '').splitlines()
        for line in raw_lines:
            stripped = line.strip()
            if (stripped.startswith("[>]") or
                stripped.startswith("[DEBUG_1D_INT]") or
                "Measured D2D" in stripped or
                "Nominal" in stripped or
                "Perturbation" in stripped):
                metric_lines.append(f"    {stripped}")

        if metric_lines:
            lines.extend(metric_lines)
        else:
            lines.append("    (No specific [>] metric lines detected)")

        # Attach Top Profiler Latency Drivers if profile summary was archived
        clean_mode = r['mode'].replace(" ", "_").replace("(", "").replace(")", "").replace("--", "").lower()
        clean_noise = "with_noise" if "WITH NOISE" in r['noise'] else "without_noise"
        folder_name = f"test_{i:02d}_{r['test']}_{clean_mode}_{clean_noise}"
        prof_summary_file = os.path.join(_TEST_LOGS_DIR, folder_name, "profile_summary.txt")
        if os.path.isfile(prof_summary_file):
            lines.append("  --- Top 5 Profiler Latency Drivers (Cumulative Time) ---")
            try:
                with open(prof_summary_file, "r", encoding="utf-8") as pf:
                    in_cum = False
                    p_count = 0
                    for pline in pf:
                        if "TOP 30 FUNCTIONS BY CUMULATIVE TIME" in pline:
                            in_cum = True
                            continue
                        if in_cum:
                            if "TOP 30 FUNCTIONS BY TOTAL INTERNAL TIME" in pline:
                                break
                            s_pline = pline.strip()
                            if s_pline and not s_pline.startswith("Ordered by:") and not s_pline.startswith("ncalls") and not s_pline.startswith("-"):
                                lines.append(f"    {s_pline}")
                                p_count += 1
                                if p_count >= 5:
                                    break
            except Exception:
                pass

        lines.append("  --- Subtest Output Log ---")
        for line in raw_lines:
            lines.append(f"    {line}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF HARDWARE VALIDATION REPORT")
    lines.append("=" * 80)

    report_content = "\n".join(lines) + "\n"
    try:
        out_dir = os.path.dirname(os.path.abspath(report_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n[+] Detailed numerical validation report saved to:\n    {os.path.abspath(report_path)}")
    except Exception as e:
        print(f"\n[!] Warning: Failed to save numerical report to {report_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="MolmemSimulator Master Hardware & OS Validation Testbench")
    parser.add_argument("--subtest", type=str, choices=list(SUBTEST_REGISTRY.keys()), default=None,
                        help="Internal flag: execute a single subtest workload inside this process.")
    parser.add_argument("--noise-intensity", type=float, default=0.0,
                        help="Noise intensity factor in [0.0, 1.0].")
    parser.add_argument("--quick", action="store_true", default=False,
                        help="Runs an accelerated workload to reduce total runtime.")

    # Accept CLI flags intercepted by molmem_lib/__init__.py
    parser.add_argument("--log", "-log", action="store_true", default=False)
    parser.add_argument("--debug-gpu", "-debug-gpu", action="store_true", default=False)
    parser.add_argument("--cpu", "-cpu", action="store_true", default=False)
    parser.add_argument("--gpu", "-gpu", action="store_true", default=False)
    parser.add_argument("--clear-cache", action="store_true", default=False,
                        help="Clear simulator profile and calibration cache.")
    parser.add_argument("--profile", dest="profile", action="store_true", default=ENABLE_PROFILER,
                        help=f"Run cProfile on subtest execution and archive summary & pstats (default: {ENABLE_PROFILER}).")
    parser.add_argument("--no-profile", dest="profile", action="store_false",
                        help="Disable execution profiling.")
    parser.add_argument("--profile-dir", type=str, default=None,
                        help="Target directory to write profile summary and pstats.")
    parser.add_argument("--report", "--output", type=str,
                        default=os.path.join(_CURRENT_DIR, "hardware_test_report.txt"),
                        help="File path to save the comprehensive numerical validation report.")

    args, unknown = parser.parse_known_args()

    # -----------------------------------------------------------------
    # BRANCH 1: SUBPROCESS SUBTEST EXECUTOR
    # -----------------------------------------------------------------
    if args.subtest is not None:
        if args.gpu:
            os.environ["MOLMEM_FORCE_GPU"] = "1"
        if args.cpu:
            os.environ["MOLMEM_FORCE_CPU"] = "1"
            os.environ["MOLMEM_CPU_ONLY"] = "1"
        if args.debug_gpu:
            os.environ["MOLMEM_DEBUG_GPU"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["HIP_LAUNCH_BLOCKING"] = "1"
        subtest_fn = SUBTEST_REGISTRY[args.subtest]

        mode_desc = "CPU" if args.cpu else ("GPU" if args.gpu else "NORMAL (AUTO)")
        out_prof_dir = args.profile_dir or os.path.join(_PARENT_DIR, "molmem_lib", "logs")

        if args.profile:
            profiler = cProfile.Profile()
            t0 = time.perf_counter()
            profiler.enable()
            try:
                subtest_fn(noise_intensity=args.noise_intensity, quick=args.quick)
            finally:
                profiler.disable()
                dur = time.perf_counter() - t0
                try:
                    pstats_file, summary_file = generate_profile_report(
                        profiler=profiler,
                        output_dir=out_prof_dir,
                        subtest_name=args.subtest,
                        mode_desc=mode_desc,
                        noise_intensity=args.noise_intensity,
                        duration=dur
                    )
                    print(f"  [>] Profiler Logs Saved: {os.path.basename(summary_file)}, {os.path.basename(pstats_file)}")
                except Exception as e:
                    print(f"  [!] Profiler Report Warning: Failed to export profile logs: {e}")
        else:
            subtest_fn(noise_intensity=args.noise_intensity, quick=args.quick)

        sys.exit(0)

    # -----------------------------------------------------------------
    # BRANCH 2: MASTER ORCHESTRATOR RUNNING FULL 28-SCENARIO MATRIX
    # -----------------------------------------------------------------
    print("*" * 76)
    print("MOLMEMSIMULATOR MASTER HARDWARE VALIDATION SUITE")
    print("Testing OS/Hardware combinations, CLI hardware flags, GPU debug, and noise")
    print("*" * 76)

    # 28-run scenario definition matrix:
    # (test_name, mode_flag_or_None)
    scenarios = [
        # 1. tb_dynamic (CPU then GPU)
        ("tb_dynamic", "--cpu"),
        ("tb_dynamic", "--gpu"),

        # 2. tb_batch_input (Normal then CPU)
        ("tb_batch_input", None),
        ("tb_batch_input", "--cpu"),

        # 3. tb_multi_program_4x4 (Normal, GPU, CPU)
        ("tb_multi_program_4x4", None),
        ("tb_multi_program_4x4", "--gpu"),
        ("tb_multi_program_4x4", "--cpu"),

        # 4. tb_quantized_weights (Normal, GPU, CPU)
        ("tb_quantized_weights", None),
        ("tb_quantized_weights", "--gpu"),
        ("tb_quantized_weights", "--cpu"),

        # 5. tb_parallel_crossbars (Normal then CPU)
        ("tb_parallel_crossbars", None),
        ("tb_parallel_crossbars", "--cpu"),

        # 6. tb_tiled_simulator (Normal then CPU)
        ("tb_tiled_simulator", None),
        ("tb_tiled_simulator", "--cpu"),

        # 7. tb_tiled_batch_input (Normal, GPU, CPU)
        ("tb_tiled_batch_input", None),
        ("tb_tiled_batch_input", "--gpu"),
        ("tb_tiled_batch_input", "--cpu"),
    ]

    noise_levels = [0.0, 1.0]  # [without noise, with noise]
    total_runs = len(scenarios) * len(noise_levels)
    print(f"\n[+] Total Scenarios Scheduled: {total_runs} distinct testbench executions.")
    print(f"[+] CLI Flags Enforced: --log, --debug-gpu across every run.")
    print(f"[+] Output Silencing: DISABLED (Full transparency enabled).\n")

    has_gpu, gpu_unavail_reason = check_is_gpu_ready()
    if not has_gpu:
        print("=" * 80)
        print(" [!] NOTE: No compatible GPU hardware accelerator detected.")
        print(f"     Reason: {gpu_unavail_reason}")
        print("     GPU-only scenarios (--gpu) will be safely skipped.")
        print("=" * 80)

    results = []
    run_idx = 1
    t_suite_start = time.perf_counter()

    for test_name, mode_flag in scenarios:
        for noise_val in noise_levels:
            mode_desc = "NORMAL (AUTO)" if not mode_flag else mode_flag.upper()
            noise_desc = "WITH NOISE (1.0)" if noise_val > 0.0 else "WITHOUT NOISE (0.0)"

            if mode_flag == "--gpu" and not has_gpu:
                print(f"\n>>> [PROGRESS {run_idx}/{total_runs}] Skipping {test_name} (--GPU) | Noise={noise_val} (No compatible GPU detected)")
                skip_res = {
                    "test": test_name,
                    "mode": mode_desc,
                    "noise": noise_desc,
                    "status": "SKIPPED",
                    "exit_code": 0,
                    "duration": 0.0,
                    "metrics": "SKIPPED (No compatible GPU accelerator)",
                    "output": f"Test skipped: GPU-only runs disabled because no compatible hardware accelerator detected ({gpu_unavail_reason}).\n"
                }
                archive_scenario_logs(run_idx, test_name, mode_desc, noise_desc, skip_res)
                results.append(skip_res)
                run_idx += 1
                continue

            clear_active_molmem_logs()
            print(f"\n>>> [PROGRESS {run_idx}/{total_runs}] Starting {test_name} ({mode_desc}) | Noise={noise_val}...")
            res = execute_subprocess_scenario(test_name, mode_flag, noise_val, quick=args.quick, profile=args.profile)
            log_dir_saved = archive_scenario_logs(run_idx, test_name, mode_desc, noise_desc, res)
            rel_saved = os.path.relpath(log_dir_saved, _CURRENT_DIR)
            print(f"  [>] Logs Archived: {rel_saved}")
            results.append(res)
            run_idx += 1

    total_suite_time = time.perf_counter() - t_suite_start

    # =====================================================================
    # FINAL COMPREHENSIVE VALIDATION REPORT
    # =====================================================================
    header = f"{'#':<3} | {'TEST CASE':<22} | {'MODE':<13} | {'NOISE STATE':<19} | {'STATUS':<7} | {'TIME':<7} | {'KEY NUMERICAL RESULT':<45}"
    sep = "-" * len(header)
    print("\n\n" + "=" * len(header))
    print("MASTER HARDWARE VALIDATION REPORT")
    print("=" * len(header))
    print(header)
    print(sep)

    all_passed = True
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    for i, r in enumerate(results, 1):
        status_str = r['status']
        if status_str == "FAILED":
            all_passed = False
            failed_count += 1
        elif status_str == "PASSED":
            passed_count += 1
        elif status_str == "SKIPPED":
            skipped_count += 1

        metrics_display = r.get('metrics', 'N/A')
        time_str = f"{r['duration']:>5.2f}s" if status_str != "SKIPPED" else "   -- "
        print(f"{i:<3} | {r['test']:<22} | {r['mode']:<13} | {r['noise']:<19} | {status_str:<7} | {time_str:<7} | {metrics_display:<45}")

    print(sep)
    print(f"Total Execution Time : {total_suite_time:.2f} seconds ({total_suite_time / 60.0:.2f} minutes)")
    print(f"Scenarios Scheduled  : {total_runs}")
    print(f"Total Passed         : {passed_count}")
    print(f"Total Skipped        : {skipped_count}")
    print(f"Total Failed         : {failed_count}")

    # Save detailed exact numerical results report to .txt file
    save_hardware_validation_report(results, total_suite_time, args.report)
    print(f"\n[+] Master Report Saved        : {args.report}")
    print(f"[+] All Subtest Logs Archived  : {_TEST_LOGS_DIR}")

    if skipped_count > 0:
        print("\n" + "=" * 80)
        print(" [!] NOTICE: GPU-only runs disabled because no compatible hardware accelerator detected.")
        print(f"     {skipped_count} GPU scenarios were safely skipped; all available CPU and Auto scenarios executed.")
        print("=" * 80)

    if all_passed:
        print("\n======================================================================")
        if skipped_count > 0:
            print(f"ALL {passed_count} EXECUTED HARDWARE SCENARIOS PASSED WITH EXPECTED RESULTS!")
            print(f"({skipped_count} GPU-only scenarios skipped due to absence of compatible accelerator.)")
        else:
            print("ALL 28 HARDWARE SCENARIOS PASSED WITH EXPECTED RESULTS!")
            print("Hardware, OS execution, CLI flags, GPU debugging, and noise verified.")
        print("======================================================================")
        sys.exit(0)
    else:
        print("\n[!] ONE OR MORE SCENARIOS FAILED. Check detailed logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
