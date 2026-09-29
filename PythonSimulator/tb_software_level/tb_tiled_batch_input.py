import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemTiledSimulator
import numpy as np
import time

def main():
    print("==========================================================")
    print("Benchmark: Tiled Sequential Loop vs Tiled Batch Parallel Input")
    print("==========================================================")

    total_rows, total_cols = 32, 32
    tile_rows, tile_cols = 16, 16
    sim = MolmemTiledSimulator(instanceName="TiledBatchInputTest")
    sim.recordHistory = False
    sim.addTiledCrossbarMatrix(
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
        UpdatesPer="Device"
    )

    # Program random weights into the tiled crossbar
    np.random.seed(42)
    target_weights = np.random.uniform(0.1, 0.9, (total_rows, total_cols))
    print(f"[+] Programming {sim.num_row_tiles}x{sim.num_col_tiles} tiled crossbar ({total_rows}x{total_cols} total, {tile_rows}x{tile_cols} tiles) with random weights...")
    sim.updateTiledCrossbarWeights(target_weights, detailedPrint=False, fineTune=True)

    # Generate a batch of random inputs (e.g. 512 vectors of size total_rows)
    batch_size = 512
    inputs_batch = np.random.uniform(0.1, 0.9, (batch_size, total_rows))

    initial_states = sim.saveStates()
    print(f"\n[1] Running {batch_size} inputs sequentially in a loop (with tile state resets)...")
    outputs_seq = []
    t0_seq = time.perf_counter()
    for i in range(batch_size):
        sim.restoreStates(initial_states)
        out = sim.passTiledCrossbarInput(inputs_batch[i], detailedPrint=False, appendHistory=False)
        outputs_seq.append(out)
    t1_seq = time.perf_counter()
    outputs_seq = np.array(outputs_seq)
    time_seq = t1_seq - t0_seq
    print(f"Sequential processing complete in {time_seq:.4f} seconds.")

    # 2. Tiled Batch parallel execution
    sim.restoreStates(initial_states)
    print(f"\n[2] Running {batch_size} inputs concurrently in tiled batch parallel mode...")
    t0_par = time.perf_counter()
    outputs_par = sim.passTiledCrossbarInput(inputs_batch, detailedPrint=False, appendHistory=False)
    t1_par = time.perf_counter()
    time_par = t1_par - t0_par
    print(f"Batch parallel processing complete in {time_par:.4f} seconds.")

    # 3. Equivalence and Mathematical Ground-Truth Validation
    outputs_seq_np = outputs_seq.detach().cpu().numpy() if hasattr(outputs_seq, 'detach') else np.asarray(outputs_seq)
    outputs_par_np = outputs_par.detach().cpu().numpy() if hasattr(outputs_par, 'detach') else np.asarray(outputs_par)

    # Physical programmed weights derived from direct tiled crossbar read
    gMatrix, _ = sim.readTiledCrossbarMatrix(detailedPrint=False)
    w_programmed = sim._conductanceToWeights(gMatrix)

    # Theoretical ideal (Y_ideal = X @ W_target) and programmed hardware math (Y_hw = X @ W_hw)
    outputs_math_ideal = np.matmul(inputs_batch, target_weights)
    outputs_math_prog = np.matmul(inputs_batch, w_programmed)

    diff_seq_par = np.max(np.abs(outputs_seq_np - outputs_par_np))
    rel_diff_seq_par = diff_seq_par / max(1e-12, float(np.max(outputs_seq_np)))

    mae_vs_ideal = np.mean(np.abs(outputs_par_np - outputs_math_ideal))
    rmse_vs_ideal = np.sqrt(np.mean((outputs_par_np - outputs_math_ideal) ** 2))

    mae_vs_prog = np.mean(np.abs(outputs_par_np - outputs_math_prog))
    rmse_vs_prog = np.sqrt(np.mean((outputs_par_np - outputs_math_prog) ** 2))

    corr_prog = np.corrcoef(outputs_par_np.ravel(), outputs_math_prog.ravel())
    r2_prog = (corr_prog[0, 1] ** 2) if corr_prog.shape == (2, 2) else 1.0

    corr_ideal = np.corrcoef(outputs_par_np.ravel(), outputs_math_ideal.ravel())
    r2_ideal = (corr_ideal[0, 1] ** 2) if corr_ideal.shape == (2, 2) else 1.0

    print(f"outputs_seq shape: {outputs_seq_np.shape}, min: {outputs_seq_np.min():.4e}, max: {outputs_seq_np.max():.4e}")
    print(f"outputs_par shape: {outputs_par_np.shape}, min: {outputs_par_np.min():.4e}, max: {outputs_par_np.max():.4e}")
    print(f"outputs_math_ideal shape: {outputs_math_ideal.shape}, min: {outputs_math_ideal.min():.4e}, max: {outputs_math_ideal.max():.4e}")
    print(f"outputs_seq[0]         : {outputs_seq_np[0]}")
    print(f"outputs_par[0]         : {outputs_par_np[0]}")
    print(f"outputs_math_prog[0]   : {outputs_math_prog[0]}")
    print(f"outputs_math_ideal[0]  : {outputs_math_ideal[0]}")

    print("\n==========================================================")
    print("                       RESULTS                            ")
    print("==========================================================")
    print(f"Parallel vs Sequential Difference  : {diff_seq_par:.6e} (rel: {rel_diff_seq_par * 100:.4f}%)")
    if diff_seq_par < 0.01:
        print(">> PASS: Parallel batch outputs match sequential outputs within physical ODE tolerance!")
    else:
        print(">> FAIL: Outputs do not match!")

    print(f"\n--- Physical vs Mathematical Reference Fidelity ---")
    print(f"Hardware Math MAE  (vs Y_hw)       : {mae_vs_prog:.6e} (RMSE: {rmse_vs_prog:.6e}, R^2: {r2_prog:.6f})")
    print(f"Ideal Model MAE    (vs Y_ideal)    : {mae_vs_ideal:.6e} (RMSE: {rmse_vs_ideal:.6e}, R^2: {r2_ideal:.6f})")
    if r2_prog > 0.99:
        print(">> PASS: Tiled crossbar inference accurately performs physical matrix-vector multiplication (Y = X @ W)!")
    else:
        print(">> WARNING: Deviation observed between physical ODE output and mathematical reference.")

    print(f"\n--- Execution Performance ---")
    print(f"Backend                : {getattr(sim, 'backend', 'auto')} (Device: {getattr(sim, 'device', 'auto')})")
    print(f"Sequential loop time   : {time_seq:.4f}s")
    print(f"Batch parallel time    : {time_par:.4f}s")
    speedup = time_seq / time_par if time_par > 0 else 0
    print(f"Speedup factor         : {speedup:.2f}x")
    print("==========================================================")

if __name__ == "__main__":
    main()
