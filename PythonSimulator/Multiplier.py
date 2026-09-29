from molmem_lib import MolmemSimulator
import numpy as np
import time

def run_multiplier_example():
    print("==========================================================")
    print("      Memristor Crossbar Vector-Matrix Multiplier         ")
    print("==========================================================")
    
    rows, cols = 3, 3
    
    # 1. Initialize Simulator using the clean high-level API
    sim = MolmemSimulator(instanceName="MultiplierExample")
    sim.addCrossbarMatrix(
        device_type="default", 
        rows=rows, 
        cols=cols, 
        weightBits=14, 
        inputBits=14, 
        outputBits=14, 
        detailedPrint=False,
        UpdatesPer="Device"
    )
    
    # 2. Define a synthetic Weight Matrix (Values between 0.0 and 1.0)
    target_weights = np.array([
        [1.0, 0.2, 0.0],
        [0.0, 1.0, 0.5],
        [0.8, 0.0, 1.0]
    ])
    
    print("\n[Phase 1] Programming the Weight Matrix into Hardware...")
    start_time = time.time()
    
    # Program the physical crossbar
    sim.updateCrossbarWeights(target_weights, detailedPrint=False, fineTune=True)
    print(f"Programming complete in {time.time() - start_time:.2f} seconds.")
    
    # Read back the actual analog programming achieved by the hardware
    gMatrix, _ = sim.readCrossbarMatrix(detailedPrint=False, purpose=None)
    wPhysical = sim._conductanceToWeights(gMatrix)
    
    print("\n   Target Weights:")
    print(np.round(target_weights, 3))
    print("\n   Achieved Physical Weights:")
    print(np.round(wPhysical, 3))
    
    # 3. Perform Vector-Matrix Computations natively via physics
    print("\n[Phase 2] Passing Input Vectors (Y = X * W)")
    
    test_vectors = [
        np.array([1.0, 0.0, 0.0]),  # Selects Row 0
        np.array([0.0, 1.0, 0.0]),  # Selects Row 1
        np.array([1.0, 0.5, 0.2]),  # Mixed vector
        np.array([1.0, 1.0, 1.0])   # Full load
    ]
    
    for i, X in enumerate(test_vectors):
        print(f"\n--- Test Vector {i+1} ---")
        print(f"Input X: {X}")
        
        # Ground Truth Software Math
        y_expected = np.dot(X, target_weights)
        
        # Hardware Physics Computation
        y_hardware = sim.passInput(X, detailedPrint=False)
        
        print(f"  Expected Math (Y):  {np.round(y_expected, 3)}")
        print(f"  Hardware Yield (Y): {np.round(y_hardware, 3)}")
        
        error = np.abs(y_expected - y_hardware)
        print(f"  Absolute Error:     {np.round(error, 3)}")

if __name__ == "__main__":
    run_multiplier_example()
