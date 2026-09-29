import sys
import os
# Ensure molmem_lib runs from local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemTiledSimulator

import time
import numpy as np

def test_single_tile_update():
    print("==================================================")
    print("--- Verifying Precision Tile Addressing API ---")
    print("==================================================")
    
    TOTAL_ROWS = 64
    TOTAL_COLS = 64
    TILE_ROWS = 32
    TILE_COLS = 32
    
    # Target Weights for just one tile!
    np.random.seed(99)
    tile_weights = np.random.uniform(0.1, 0.9, (TILE_ROWS, TILE_COLS))
    
    print(f"\n[Test] Initializing MolmemTiledSimulator ({TOTAL_ROWS}x{TOTAL_COLS} from {TILE_ROWS}x{TILE_COLS} tiles)...")
    tiled_sim = MolmemTiledSimulator(instanceName="TileAddressingTest")
    tiled_sim.addTiledCrossbarMatrix(
        total_rows=TOTAL_ROWS, total_cols=TOTAL_COLS, tile_rows=TILE_ROWS, tile_cols=TILE_COLS, 
        device_type="default",
        weightBits=14, inputBits=14, outputBits=14, max_vWrite=5.0, max_pw=320e-9, 
        detailedPrint=False, architecture='1T1R', UpdatesPer="Column"
    )
    
    # To demonstrate it works correctly, we will update Tile (1, 0) specifically
    # By default, all tiles are unprogrammed (near ~0.0 weights, highly resistive)
    target_r, target_c = 1, 0
    print(f"\n[Test] Programming ONLY Tile ({target_r}, {target_c}) via new addressable API...")
    
    t0_update = time.perf_counter()
    tiled_sim.updateTileCrossbarWeights(target_r, target_c, tile_weights, detailedPrint=True, fineTune=False)
    t1_update = time.perf_counter()
    
    print(f"       -> Precision Update Time: {t1_update - t0_update:.2f}s")
    
    print("\n[Test] Reading unified global matrix to verify bounding...")
    # Read the entire global matrix out
    global_g, _ = tiled_sim.readTiledCrossbarMatrix(detailedPrint=False)
    global_w = tiled_sim.tiles[0][0]._conductanceToWeights(global_g)
    
    # Assess bounding boxes
    print("\n==================================================")
    print("--- Memory Verification Results ------------------")
    print("==================================================")
    
    # Tile (0, 0)
    w_0_0 = global_w[0:32, 0:32]
    # Tile (0, 1)
    w_0_1 = global_w[0:32, 32:64]
    # Tile (1, 0) - This one SHOULD be programmed
    w_1_0 = global_w[32:64, 0:32]
    # Tile (1, 1)
    w_1_1 = global_w[32:64, 32:64]
    
    print(f"Tile (0, 0) - Peak Weight: {np.max(w_0_0):.4f}  (Should be nearly ~0.0)")
    print(f"Tile (0, 1) - Peak Weight: {np.max(w_0_1):.4f}  (Should be nearly ~0.0)")
    print(f"Tile (1, 1) - Peak Weight: {np.max(w_1_1):.4f}  (Should be nearly ~0.0)")
    print(f"Tile (1, 0) - Peak Weight: {np.max(w_1_0):.4f}  (Should be > 0.8)")
    
    # Verify the values transferred accurately
    error = np.mean(np.abs(tile_weights - w_1_0))
    print(f"\nTarget Tile Programming MAE Error: {error:.4f}")
    
    if np.max(w_0_0) < 0.1 and np.max(w_0_1) < 0.1 and np.max(w_1_1) < 0.1 and np.max(w_1_0) > 0.5:
         print("\n>> PASS: Single-Tile addressing successfully isolated modifications to coordinate bounds!")
    else:
         print("\n>> FAIL: Memory corruption detected outside targeted tile boundary.")

if __name__ == "__main__":
    test_single_tile_update()
