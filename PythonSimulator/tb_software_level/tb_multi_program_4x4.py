import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
import numpy as np

def main():
    rows, cols = 4, 4
    
    # Initialize the high-level simulator object
    sim = MolmemSimulator(instanceName="ReprogrammingMatrix4x4")
    sim.addCrossbarMatrix(rows=rows, cols=cols, device_type="Ru_azo", weightBits=14, inputBits=14, outputBits=14, detailedPrint=False, architecture='1T1R', max_vWrite=3.0, max_pw=320e-9, UpdatesPer="Column")
    
    # Define three different weight programming phases for 4x4
    weights_phase1 = np.array([
        [0.10, 0.90, 0.20, 0.80],
        [0.20, 0.80, 0.30, 0.70],
        [0.30, 0.70, 0.40, 0.60],
        [0.40, 0.60, 0.50, 0.50]
    ])
    
    weights_phase2 = np.array([
        [0.90, 0.10, 0.80, 0.20],
        [0.80, 0.20, 0.70, 0.30],
        [0.70, 0.30, 0.60, 0.40],
        [0.60, 0.40, 0.50, 0.50]
    ])
    
    weights_phase3 = np.full((rows, cols), 0.50)
    
    # Execute the programming cycles once and track SE
    se_matrices = {}
    
    print("--- Starting Phase 1 Programming ---")
    sim.updateCrossbarWeights(weights_phase1, detailedPrint=True, fineTune=True, recordHistory=True)
    g, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w = sim._conductanceToWeights(g)
    se_matrices["Phase 1"] = (weights_phase1 - w) ** 2
    
    print("\n--- Starting Phase 2 Programming ---")
    sim.updateCrossbarWeights(weights_phase2, detailedPrint=True, fineTune=True, recordHistory=True)
    g, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w = sim._conductanceToWeights(g)
    se_matrices["Phase 2"] = (weights_phase2 - w) ** 2
    
    print("\n--- Starting Phase 3 Programming ---")
    sim.updateCrossbarWeights(weights_phase3, detailedPrint=True, fineTune=True, recordHistory=True)
    g, _ = sim.readCrossbarMatrix(detailedPrint=False)
    w = sim._conductanceToWeights(g)
    se_matrices["Phase 3"] = (weights_phase3 - w) ** 2
    
    # Plot the full transient conductance history
    plotter = MolmemPlotter(sim)
    plotter.createFigure("multi_program", title="4x4 Crossbar Multi-Phase Reprogramming", figsize=(18, 12), subplots=(rows, cols))
    
    # Plot formatting for the 4x4 grid dynamically
    for r in range(rows):
        for c in range(cols):
            plotter.addWaveform("multi_program", 'Conductance', row=r+1, col=c+1, subplotIdx=(r,c), label=f'Device ({r},{c})')
            
            y_label = None if c == 0 else ""
            x_label = "Auto" if r == rows - 1 else ""
            plotter.formatAxis("multi_program", (r,c), xlabel=x_label, ylabel=y_label, title=f"Weight {r},{c}")
    
    plotter.saveFigure("multi_program", "sim_results/tb_multi_program_4x4.png")
    print(f"\nSaved rendering to sim_results/tb_multi_program_4x4.png")
    
    print("\n=======================================================")
    print("SQUARED ERROR (SE) MATRIX PER SLOT (4x4)")
    print("=======================================================")
    for name, se in se_matrices.items():
        print(f"\n{name} Squared Error Matrix:")
        print(np.array2string(se, precision=6, separator=', '))
    print("=======================================================\n")
    
    if not os.environ.get("MOLMEM_SKIP_SHOW"):
        plotter.showAll()

if __name__ == "__main__":
    main()
