import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator
import numpy as np

def test_quantization_config(w_bits, in_bits, out_bits, rows=2, cols=2):
    """
    Tests a specific (weightBits, inputBits, outputBits) quantization profile.
    Validates:
      1. Digital-to-Analog Converter (DAC) weight quantization discretization.
      2. Hardware ADC integer bit-depth bounds [0, 2^out_bits - 1].
      3. Digital-to-Time Converter (DTC) input PWM pulse-width resolution.
      4. Hardware-in-the-loop VMM inference accuracy vs ideal quantized math.
    """
    print(f"\n" + "=" * 64)
    print(f" Testing Quantization Profile: {w_bits}-bit Weights | {in_bits}-bit Inputs | {out_bits}-bit Outputs")
    print("=" * 64)
    
    sim = MolmemSimulator(instanceName=f"QuantSim_{w_bits}w_{in_bits}in_{out_bits}out")
    sim.addCrossbarMatrix(
        rows=rows, 
        cols=cols, 
        weightBits=w_bits, 
        inputBits=in_bits, 
        outputBits=out_bits, 
        detailedPrint=False, 
        architecture='1T1R', 
        max_vWrite=5.0, 
        max_pw=320e-9, 
        UpdatesPer="Column"
    )
    
    # 1. Target Continuous Weights
    weights_ideal = np.array([
        [0.10, 0.95],
        [0.60, 0.35]
    ], dtype=np.float64)
    
    # 2. Mathematical Expected Weight Quantization
    w_levels = (1 << w_bits) - 1
    expected_wq = np.round(weights_ideal * w_levels) / w_levels
    quantizer_wq = sim.wq.quantizeWeights(weights_ideal)
    
    np.testing.assert_allclose(
        quantizer_wq, expected_wq, atol=1e-12,
        err_msg=f"WeightQuantizer failed for {w_bits}-bit"
    )
    print(f"  [>] Weight Discretization Levels : {1 << w_bits} levels (LSB = {1.0/w_levels:.6f})")
    print(f"      Ideal Continuous Weights     :\n{weights_ideal}")
    print(f"      Quantized Target Weights     :\n{np.round(expected_wq, 4)}")
    
    # 3. Hardware Closed-Loop Weight Programming
    sim.updateCrossbarWeights(weights_ideal, detailedPrint=False, fineTune=True, recordHistory=False)
    
    # 4. Hardware ADC Matrix Read
    gMatrix, gAdcBits = sim.readCrossbarMatrix(detailedPrint=False, purpose=None)
    w_achieved = sim._conductanceToWeights(gMatrix)
    
    max_adc_val = (1 << out_bits) - 1
    adc_in_bounds = bool(np.all(gAdcBits >= 0) and np.all(gAdcBits <= max_adc_val))
    weight_mae = float(np.mean(np.abs(w_achieved - expected_wq)))
    
    print(f"  [>] Hardware Achieved Weights    :\n{np.round(w_achieved, 4)}")
    print(f"  [>] Static ADC Output Bits       :\n{gAdcBits}")
    print(f"      ADC Range [0, {max_adc_val}] : {'PASSED' if adc_in_bounds else 'FAILED'}")
    print(f"      Weight Tracking MAE          : {weight_mae:.4e}")
    
    # 5. Input PWM Pulse Width DTC Quantization
    xA = np.array([1.0, 0.5], dtype=np.float64)
    input_pw = sim.inQ.quantizeInputs(xA)
    in_levels = (1 << in_bits) - 1
    expected_pw = np.round(xA * in_levels) * 1e-9
    
    np.testing.assert_allclose(
        input_pw, expected_pw, atol=1e-15,
        err_msg=f"InputQuantizer failed for {in_bits}-bit"
    )
    print(f"  [>] Input PWM Pulse Widths (ns)  : {input_pw * 1e9} ns")
    
    # 6. Physical Hardware VMM Inference
    y_hw = sim.passInput(xA, detailedPrint=False)
    
    # Mathematical Reference
    x_quant = np.round(xA * in_levels) / in_levels
    y_ideal_quant = np.dot(x_quant, expected_wq)
    inf_mae = float(np.mean(np.abs(y_hw - y_ideal_quant)))
    
    print(f"  [>] HW Output Vector (Y = X * W) : {np.round(y_hw, 4)}")
    print(f"      Ideal Quantized Math Output  : {np.round(y_ideal_quant, 4)}")
    print(f"      Inference MAE                : {inf_mae:.4e}")
    
    # Validation assertion checks
    assert adc_in_bounds, f"ADC bit reading exceeded {out_bits}-bit bounds: {gAdcBits}"
    assert weight_mae < 0.05, f"Hardware weight MAE too high: {weight_mae}"
    assert inf_mae < 0.08, f"Hardware inference MAE too high: {inf_mae}"
    
    return {
        "w_bits": w_bits,
        "in_bits": in_bits,
        "out_bits": out_bits,
        "w_levels": 1 << w_bits,
        "weight_mae": weight_mae,
        "max_adc_bits": int(np.max(gAdcBits)),
        "max_adc_limit": max_adc_val,
        "inf_mae": inf_mae,
        "status": "PASSED"
    }

def main():
    configs = [
        (4, 4, 6),    # Low-precision edge profile
        (6, 6, 8),    # Medium-precision profile
        (8, 8, 10),   # Standard INT8 neuromorphic profile
        (10, 10, 12), # High-precision profile
        (14, 14, 14), # Ultra-precision physical limit profile
    ]
    
    results = []
    for w_b, in_b, out_b in configs:
        res = test_quantization_config(w_b, in_b, out_b)
        results.append(res)
        
    print("\n" + "=" * 78)
    print("                 QUANTIZATION BENCHMARK & VALIDATION SUMMARY")
    print("=" * 78)
    print(f" {'Config (W/In/Out)':<20} | {'W Levels':<10} | {'Weight MAE':<12} | {'ADC Peak / Max':<16} | {'Inf MAE':<10} | {'Status'}")
    print("-" * 78)
    for r in results:
        cfg_str = f"{r['w_bits']}b / {r['in_bits']}b / {r['out_bits']}b"
        adc_str = f"{r['max_adc_bits']} / {r['max_adc_limit']}"
        print(f" {cfg_str:<20} | {r['w_levels']:<10} | {r['weight_mae']:<12.4e} | {adc_str:<16} | {r['inf_mae']:<10.4e} | {r['status']}")
    print("=" * 78)
    print("[+] All quantization levels validated successfully against physical hardware models.\n")

if __name__ == "__main__":
    main()
