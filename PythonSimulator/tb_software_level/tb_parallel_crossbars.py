import sys
import os
# Append the parent directory to sys.path so it can find molmem_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, updateCrossbarsParallel

import time
import numpy as np

def main():
    print("==========================================================")
    print("--- Parallel Programming on Multiple Flat Crossbars    ---")
    print("==========================================================")
    print("This testbench demonstrates the dynamic thread-resource pooling algorithm")
    print("for dispatching concurrent hardware-level pulse programming across multiple")
    print("independent, isolated crossbar arrays (e.g., discrete layers in a neural network).\n")

    np.random.seed(42)

    # 1. Initialize Hardware Simulators (differential pair: pos/neg for each layer)
    print("[+] Provisioning Hardware Simulators...")
    
    sim1_pos = MolmemSimulator(instanceName="Layer1_pos_16x16")
    sim1_pos.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")
    
    sim1_neg = MolmemSimulator(instanceName="Layer1_neg_16x16")
    sim1_neg.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")

    sim2_pos = MolmemSimulator(instanceName="Layer2_pos_16x16")
    sim2_pos.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")
    
    sim2_neg = MolmemSimulator(instanceName="Layer2_neg_16x16")
    sim2_neg.addCrossbarMatrix(rows=16, cols=16, weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column")

    # 2. Package simulators corresponding target weight matrices
    simulators = [sim1_pos, sim1_neg, sim2_pos, sim2_neg]
    
    # Generate random target matrices for each flat simulator
    W1_pos = np.random.uniform(0.0, 1.0, (16, 16))
    W1_neg = np.random.uniform(0.0, 1.0, (16, 16))
    W2_pos = np.random.uniform(0.0, 1.0, (16, 16))
    W2_neg = np.random.uniform(0.0, 1.0, (16, 16))
    
    weight_matrices = [W1_pos, W1_neg, W2_pos, W2_neg]

    print(f"\n[Test] Hardware initializations complete. Dispatching {len(simulators)} concurrent autonomous hardware tracking loops...")
    
    # 3. Parallel Dispatch
    t0 = time.perf_counter()
    updateCrossbarsParallel(simulators, weight_matrices, detailedPrint=False, prefix="ParallelUpdate", fineTune=False, recordHistory=False)
    t1 = time.perf_counter()
    
    # 4. Verification read & Inference
    print("\n[Test] Verifying hardware conductance matching vs Mathematical Ideals...")
    
    # Random normalized vector for all tests
    x_input = np.random.uniform(0.0, 1.0, 16)
    
    for i, sim in enumerate(simulators):
        W_target = weight_matrices[i]
        
        # Read exact physical weights
        gMatrix, _ = sim.readCrossbarMatrix(detailedPrint=False, purpose=None)
        wFinalNorm = sim._conductanceToWeights(gMatrix)
        
        mae = np.mean(np.abs(W_target - wFinalNorm))
        max_err = np.max(np.abs(W_target - wFinalNorm))
        
        # Hardware vs Software Inference
        y_hw = sim.passInput(x_input, detailedPrint=False)
        y_math = np.dot(x_input, W_target)
        inf_diff = np.max(np.abs(y_hw - y_math))
        
        backend_info = getattr(sim, 'backend', 'unknown')
        print(f"\n       -> {sim.instanceName} [Backend: {backend_info}]:")
        print(f"          Absolute Weight MAE : {mae:.4e}  |  Max Error: {max_err:.4e}")
        print(f"          Max Inference Diff  : {inf_diff:.4e}")
        print(f"          HW Y Out (first 3)  : {np.round(y_hw[:3], 4)}")
        print(f"          Math Y Out (first 3): {np.round(y_math[:3], 4)}")
        
    print(f"\n[Test] Successfully programmed the entire multi-layer stack in {t1-t0:.2f}s!")

if __name__ == "__main__":
    main()
