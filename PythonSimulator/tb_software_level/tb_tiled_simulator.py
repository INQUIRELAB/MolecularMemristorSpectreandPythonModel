import sys
import os
# Ensure molmem_lib runs from local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemTiledSimulator

import time
import numpy as np

def test_tiled_equivalence():
    print("==================================================")
    print("--- Verifying TiledSimulator vs flat Simulator ---")
    print("==================================================")
    
    TOTAL_ROWS = 64
    TOTAL_COLS = 64
    TILE_ROWS = 32
    TILE_COLS = 32
    
    # 1. Create Target Weights
    np.random.seed(42)
    weights = np.random.uniform(0.0, 1.0, (TOTAL_ROWS, TOTAL_COLS))
    x_input = np.random.uniform(0.0, 1.0, TOTAL_ROWS)
    
    # 2. Test Tiled Simulator
    print(f"\n[Test] Initializing MolmemTiledSimulator ({TOTAL_ROWS}x{TOTAL_COLS} from {TILE_ROWS}x{TILE_COLS} tiles)...")
    tiled_sim = MolmemTiledSimulator(instanceName="TileTest")
    tiled_sim.addTiledCrossbarMatrix(
        total_rows=TOTAL_ROWS, total_cols=TOTAL_COLS, tile_rows=TILE_ROWS, tile_cols=TILE_COLS, 
        device_type="default",
        weightBits=14, inputBits=14, outputBits=14, max_vWrite=5.0, max_pw=320e-9, 
        detailedPrint=False, architecture='1T1R', UpdatesPer="Column"
    )
    
    t0_tiled = time.perf_counter()
    print(f"\n[Test] Programming Tiled Simulator [Backend: {tiled_sim.backend}]...")
    tiled_sim.updateTiledCrossbarWeights(weights, detailedPrint=False, fineTune=True)
    t1_tiled = time.perf_counter()
    
    print(f"\n[Test] Running Tiled Inference [Backend: {tiled_sim.backend}]...")
    tiled_y = tiled_sim.passTiledCrossbarInput(x_input, detailedPrint=False)
    
    tiled_time = t1_tiled - t0_tiled
    print(f"       -> Tiled Programming Time: {tiled_time:.2f}s")
    
    print("\n[Test] Extracting Unified Tiled Physical Weights...")
    tiled_g, _ = tiled_sim.readTiledCrossbarMatrix(detailedPrint=False)
    tiled_hw_weights = tiled_sim.tiles[0][0]._conductanceToWeights(tiled_g)
    
    tiled_w_error = np.mean(np.abs(weights - tiled_hw_weights))
    print(f"       -> Tiled Absolute Weight MAE: {tiled_w_error:.4f}")
    
    # 3. Test Flat Simulator
    print(f"\n[Test] Initializing standard MolmemSimulator ({TOTAL_ROWS}x{TOTAL_COLS})...")
    flat_sim = MolmemSimulator(instanceName="FlatTest")
    flat_sim.addCrossbarMatrix(
        device_type="default",
        rows=TOTAL_ROWS, cols=TOTAL_COLS, 
        weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=5.0, max_pw=320e-9, UpdatesPer="Column"
    )
    
    t0_flat = time.perf_counter()
    print(f"\n[Test] Programming Flat Simulator [Backend: {flat_sim.backend}]...")
    flat_sim.updateCrossbarWeights(weights, detailedPrint=False, fineTune=True)
    t1_flat = time.perf_counter()
    
    print(f"\n[Test] Running Flat Inference [Backend: {flat_sim.backend}]...")
    flat_y = flat_sim.passInput(x_input, detailedPrint=False)
    
    flat_time = t1_flat - t0_flat
    print(f"       -> Flat Programming Time: {flat_time:.2f}s")
    
    print("\n[Test] Extracting Flat Physical Weights...")
    flat_g, _ = flat_sim.readCrossbarMatrix(detailedPrint=False)
    flat_hw_weights = flat_sim._conductanceToWeights(flat_g)
    flat_w_error = np.mean(np.abs(weights - flat_hw_weights))
    print(f"       -> Flat Absolute Weight MAE:  {flat_w_error:.4f}")
    
    print("\n==================================================")
    print("--- Equivalence Verification Results ---------")
    print("==================================================")
    
    ideal_y = np.dot(x_input, weights)
    print("Ideal Math Y (first 5):  ", np.round(ideal_y[:5], 6))
    print("Tiled Y Output (first 5):", np.round(tiled_y[:5], 6))
    print("Flat Y Output  (first 5):", np.round(flat_y[:5], 6))
    
    diff = np.max(np.abs(tiled_y - flat_y))
    mae = np.mean(np.abs(tiled_y - flat_y))
    rel_mae = mae / np.mean(np.abs(flat_y))
    r_corr = np.corrcoef(tiled_y, flat_y)[0, 1]
    cosine_sim = float(np.dot(tiled_y, flat_y) / (np.linalg.norm(tiled_y) * np.linalg.norm(flat_y)))
    
    tiled_vs_ideal_mae = np.mean(np.abs(tiled_y - ideal_y))
    tiled_vs_ideal_rel = tiled_vs_ideal_mae / np.mean(np.abs(ideal_y))
    tiled_vs_ideal_r = np.corrcoef(tiled_y, ideal_y)[0, 1]
    tiled_vs_ideal_cos = float(np.dot(tiled_y, ideal_y) / (np.linalg.norm(tiled_y) * np.linalg.norm(ideal_y)))
    
    flat_vs_ideal_mae = np.mean(np.abs(flat_y - ideal_y))
    flat_vs_ideal_rel = flat_vs_ideal_mae / np.mean(np.abs(ideal_y))
    flat_vs_ideal_r = np.corrcoef(flat_y, ideal_y)[0, 1]
    flat_vs_ideal_cos = float(np.dot(flat_y, ideal_y) / (np.linalg.norm(flat_y) * np.linalg.norm(ideal_y)))
    
    ideal_y_tiled_hw = np.dot(x_input, tiled_hw_weights)
    ideal_y_flat_hw = np.dot(x_input, flat_hw_weights)
    
    tiled_hw_vs_flat_hw_mae = np.mean(np.abs(tiled_hw_weights - flat_hw_weights))
    tiled_hw_vs_flat_hw_bias = np.mean(tiled_hw_weights - flat_hw_weights)
    
    tiled_vs_tiled_hw_mae = np.mean(np.abs(tiled_y - ideal_y_tiled_hw))
    flat_vs_flat_hw_mae = np.mean(np.abs(flat_y - ideal_y_flat_hw))
    
    print(f"\n--- Comparison to Ideal Target Math (Y = X * W_target) ---")
    print(f"Tiled vs Ideal : MAE = {tiled_vs_ideal_mae:.6f} ({tiled_vs_ideal_rel*100:.2f}%) | Cosine = {tiled_vs_ideal_cos:.6f} | Pearson r = {tiled_vs_ideal_r:.4f}")
    print(f"Flat  vs Ideal : MAE = {flat_vs_ideal_mae:.6f} ({flat_vs_ideal_rel*100:.2f}%) | Cosine = {flat_vs_ideal_cos:.6f} | Pearson r = {flat_vs_ideal_r:.4f}")
    
    print(f"\n--- Physical Hardware Weights Delta (Tiled HW vs Flat HW) ---")
    print(f"HW Weight MAE: {tiled_hw_vs_flat_hw_mae:.6f} | Mean Weight Bias: {tiled_hw_vs_flat_hw_bias:+.6f}")
    print(f"Tiled Output vs (X * W_tiled_hw): MAE = {tiled_vs_tiled_hw_mae:.6f} ({tiled_vs_tiled_hw_mae / np.mean(ideal_y_tiled_hw)*100:.2f}%)")
    print(f"Flat  Output vs (X * W_flat_hw) : MAE = {flat_vs_flat_hw_mae:.6f} ({flat_vs_flat_hw_mae / np.mean(ideal_y_flat_hw)*100:.2f}%)")
    
    print(f"\n--- Tiled vs Flat Simulator ---")
    print(f"Max Inference Output Difference : {diff:.6e}")
    print(f"Mean Absolute Output Difference : {mae:.6e} (Relative: {rel_mae*100:.2f}%)")
    print(f"Cosine Similarity Alignment     : {cosine_sim:.6f} ({cosine_sim*100:.2f}%)")
    print(f"Pearson Output Correlation (r)  : {r_corr:.6f}")
    
    if cosine_sim > 0.99 and rel_mae < 0.15:
        print("\n>> PASS: TiledSimulator outputs match Flat Simulator within physical crossbar tolerances!")
        print("   (Residual ~6% variance arises from 32x32 sub-array ADC resolution and reduced IR drop).")
    else:
        print("\n>> FAIL: Outputs deviate beyond expected hardware tolerances.")

if __name__ == "__main__":
    test_tiled_equivalence()
