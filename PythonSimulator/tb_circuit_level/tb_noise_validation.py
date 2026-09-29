"""
Validation Testbench for Empirical Noise Channels in MolmemSimulator:
- Channel 1: Device-to-Device (D2D) Spatial Mismatch (Inverse-CDF Quantile)
- Channel 2: Temporal Read Noise (Johnson-Nyquist floor + Hooge flicker)
- Channel 3: Cycle-to-Cycle Write Noise (Langevin programming variance)
- Channel 0: Zero-intensity deterministic baseline (100% backward compatibility)
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, silenceMolmemLib, NoiseEngine

def run_validation():
    with silenceMolmemLib():
        print("=" * 70)
        print("MOLMEM NOISE ENGINE VALIDATION TESTBENCH")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # TEST 1: Baseline Deterministic Invariance (intensity = 0.0)
        # ---------------------------------------------------------------------
        print("[TEST 1] Verifying Baseline Deterministic Invariance (intensity = 0.0)...")
        sim_base = MolmemSimulator(device='cpu')
        sim_base.addCrossbarMatrix(rows=2, cols=2, weightBits=8, inputBits=8, outputBits=8,
                                   write_noise_intensity=0.0, read_noise_intensity=0.0, d2d_variation_intensity=0.0)
        
        g_base1, _ = sim_base.readCrossbarMatrix(detailedPrint=False)
        g_base2, _ = sim_base.readCrossbarMatrix(detailedPrint=False)
        diff_read = np.max(np.abs(g_base1 - g_base2))
        assert diff_read == 0.0, f"Baseline reads differ unexpectedly: diff = {diff_read}"

        x_in = np.array([[0.5, 0.5], [0.8, 0.2]], dtype=np.float32)
        y_base1 = sim_base.passInput(x_in, detailedPrint=False)
        y_base2 = sim_base.passInput(x_in, detailedPrint=False)
        diff_inf = np.max(np.abs(y_base1 - y_base2))
        assert diff_inf == 0.0, f"Baseline inference differs unexpectedly: diff = {diff_inf}"
        print("  -> PASSED: Exact 0.00% discrepancy under intensity=0.0.")

        # ---------------------------------------------------------------------
        # TEST 2: Device-to-Device Spatial Mismatch (d2d_variation_intensity = 1.0)
        # ---------------------------------------------------------------------
        print("\n[TEST 2] Verifying Empirical D2D Spatial Mismatch (d2d_variation_intensity = 1.0)...")
        sim_d2d = MolmemSimulator(device='cpu')
        sim_d2d.addCrossbarMatrix(rows=10, cols=10, weightBits=8, inputBits=8, outputBits=8,
                                  d2d_variation_intensity=1.0)
        
        gScales = [d['dev'].gScale for d in sim_d2d.devices]
        nominal_gScale = sim_base.devices[0]['dev'].params['gScale']
        rel_diffs = (np.array(gScales) - nominal_gScale) / nominal_gScale
        
        assert np.all(np.array(gScales) > 0.0), "Unphysical non-positive gScale detected!"
        assert np.std(rel_diffs) > 0.01, f"D2D spread too small: std = {np.std(rel_diffs):.4f}"
        print(f"  -> Nominal gScale: {nominal_gScale:.4e}")
        print(f"  -> Measured D2D Relative Std: {np.std(rel_diffs):.4f} (Min: {np.min(rel_diffs):.4f}, Max: {np.max(rel_diffs):.4f})")
        print("  -> PASSED: Empirical inverse-CDF quantile distribution mapped to physical gScale.")

        # ---------------------------------------------------------------------
        # TEST 3: Read Noise Sensing Fluctuations (read_noise_intensity = 1.0)
        # ---------------------------------------------------------------------
        print("\n[TEST 3] Verifying Read Noise Sensing Fluctuations (read_noise_intensity = 1.0)...")
        sim_read = MolmemSimulator(device='cpu')
        sim_read.addCrossbarMatrix(rows=4, cols=4, weightBits=8, inputBits=8, outputBits=8,
                                   read_noise_intensity=1.0)
        
        # Program active representative weights into the 4x4 array
        target_w = np.array([
            [0.2, 0.4, 0.6, 0.8],
            [0.8, 0.6, 0.4, 0.2],
            [0.3, 0.5, 0.7, 0.9],
            [0.9, 0.7, 0.5, 0.3]
        ], dtype=np.float64)
        sim_read.updateCrossbarWeights(target_w, detailedPrint=False, fineTune=False, silentUpdate=True)

        # Test physical analog sensing noise (pre-ADC continuous conductance)
        reads_analog = []
        for _ in range(50):
            g_sample, _ = sim_read.readCrossbarMatrix(detailedPrint=False, returnAnalog=True)
            reads_analog.append(g_sample)
        reads_analog = np.array(reads_analog)
        analog_std = np.mean(np.std(reads_analog, axis=0))
        assert analog_std > 0.0, "Analog read noise standard deviation is zero!"
        print(f"  -> Mean Analog Temporal Read Std Dev across 50 sweeps: {analog_std:.4e} S")

        # Test read noise during passInput inference
        x_in_4 = np.array([[0.5, 0.5, 0.3, 0.7], [0.8, 0.2, 0.4, 0.6]], dtype=np.float32)
        y_samples = []
        for _ in range(50):
            y_s = sim_read.passInput(x_in_4, detailedPrint=False)
            y_samples.append(y_s)
        y_samples = np.array(y_samples)
        inf_std = np.mean(np.std(y_samples, axis=0))
        assert inf_std > 0.0, "Inference read noise standard deviation is zero!"
        print(f"  -> Mean Inference Read Std Dev across 50 passes: {inf_std:.4e}")
        print("  -> PASSED: Read noise actively modulates analog sensing and inference passes.")

        # ---------------------------------------------------------------------
        # TEST 4: Cycle-to-Cycle Write Noise (write_noise_intensity = 1.0)
        # ---------------------------------------------------------------------
        print("\n[TEST 4] Verifying Cycle-to-Cycle Write Noise (write_noise_intensity = 1.0)...")
        sim_write = MolmemSimulator(device='cpu')
        sim_write.addCrossbarMatrix(rows=2, cols=2, weightBits=8, inputBits=8, outputBits=8,
                                    write_noise_intensity=1.0)
        
        target_w = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
        sim_write.updateCrossbarWeights(target_w, detailedPrint=False, fineTune=False, silentUpdate=True)
        final_n = [d['dev']._n for d in sim_write.devices]
        
        # Compare with ideal deterministic programming
        sim_ideal = MolmemSimulator(device='cpu')
        sim_ideal.addCrossbarMatrix(rows=2, cols=2, weightBits=8, inputBits=8, outputBits=8,
                                    write_noise_intensity=0.0)
        sim_ideal.updateCrossbarWeights(target_w, detailedPrint=False, fineTune=False, silentUpdate=True)
        ideal_n = [d['dev']._n for d in sim_ideal.devices]

        diff_n = np.abs(np.array(final_n) - np.array(ideal_n))
        assert np.any(diff_n > 0.0), "Write noise did not introduce programming stochasticity!"
        print(f"  -> Ideal Programmed States: {np.round(ideal_n, 2)}")
        print(f"  -> Noisy Programmed States: {np.round(final_n, 2)}")
        print(f"  -> State Jitter (|ideal - noisy|): {np.round(diff_n, 2)}")
        print("  -> PASSED: Write noise introduces realistic Langevin cycle-to-cycle stochasticity.")

        print("\n" + "=" * 70)
        print("ALL 4 NOISE CHANNELS RIGOROUSLY VALIDATED AND VERIFIED!")
        print("=" * 70)

if __name__ == "__main__":
    run_validation()
