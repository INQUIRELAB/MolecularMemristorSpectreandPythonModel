import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=8, cols=8)

# Initialize diagonal devices to half-programmed state (n_init = 16520) 
# to represent the diagonal Identity weights, matching crossbar_8x8.scs
for i in range(1, 9):
    dev = sim.devices[sim.crossbarDevices[(i, i)]]['dev']
    dev.n = 16520.0
    dev._forceUpdateConductanceParams()

t_pts = [
    0.0, 
    1e-6, 1e-6 + 1e-9, 
    2e-6, 2e-6 + 1e-9, 
    4e-6, 4e-6 + 1e-9, 
    6e-6, 6e-6 + 1e-9, 
    8e-6, 8e-6 + 1e-9, 
    10e-6
]

for r in range(1, 9):
    v0 = 0.0
    v1 = 0.2 if r == 1 else 0.0
    v2 = 0.1
    v3 = 0.2 if (r-1) % 2 == 0 else 0.0
    v4 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40][r-1]
    v5 = 0.0
    
    v_segments = [
        v0, 
        v0, v1, 
        v1, v2, 
        v2, v3, 
        v3, v4, 
        v4, v5, 
        v5
    ]
    sim.addPwlSource(row=r, t_points=t_pts, v_points=v_segments)

for c in range(1, 9):
    sim.addDcSource(col=c, voltage=0.0)

sim.run(tEnd=10e-6, min_dt=1e-12, tol=1e-3)

plotter = MolmemPlotter(sim)
plotter.createFigure("crossbar_8x8", title="8x8 Crossbar Matrix Multiplication", subplots=(2, 1))

for r in range(1, 9):
    plotter.addWaveform("crossbar_8x8", 'Voltage', row=r, subplotIdx=0, label=f'Vin {r-1}')

plotter.addWaveform("crossbar_8x8", 'Current', row=1, col=1, subplotIdx=1, color='red', label='Driven Iout(0)')
plotter.addWaveform("crossbar_8x8", 'Current', row=8, col=8, subplotIdx=1, color='green', label='Sneak Iout(7)')

plotter.formatAxis("crossbar_8x8", 0, ylabel="Input Row Voltages (V)", legend=False)
plotter.formatAxis("crossbar_8x8", 1, xlabel="Time (s)", ylabel="Output Sneak Currents (uA)")
plotter.saveFigure("crossbar_8x8", "sim_results/tb_crossbar_8x8.png")
plotter.showAll()
