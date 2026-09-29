import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=1, cols=1)

vpot = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
vdep = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)
vpot_hard = PulseSource(node=None, amp=1.2, period=160e-9, width=80e-9)
vdep_hard = PulseSource(node=None, amp=-1.6, period=240e-9, width=120e-9)

edges = [1.3e-3, 2.0e-3, 3.0e-3, 5.0e-3, 5.5e-3]

def stress_func(t, sources):
    vp, vd, vph, vdh = sources
    if t < 1.3e-3:
        return vp.getVoltage(t)
    elif t < 2.0e-3:
        return vd.getVoltage(t)
    elif t < 3.0e-3:
        return 0.0
    elif t < 5.0e-3:
        return vph.getVoltage(t)
    elif t < 5.5e-3:
        return vdh.getVoltage(t)
    else:
        return vph.getVoltage(t)

sim.addBSource(row=1, vFunc=stress_func, sources=[vpot, vdep, vpot_hard, vdep_hard], edge_points=edges)
sim.addDcSource(col=1, voltage=0.0)

t_stop = 7.0e-3
sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-3)

plotter = MolmemPlotter(sim)
plotter.createFigure("stress", title="MolMemristor Stress Test", subplots=(3, 1), figsize=(10, 8))
plotter.addWaveform("stress", 'Voltage', row=1, subplotIdx=0, color='purple')
plotter.addWaveform("stress", 'State', row=1, col=1, subplotIdx=1, color='green', label='Internal Pulse State (n)')
plotter.addWaveform("stress", 'Current', row=1, col=1, subplotIdx=2, color='red')
plotter.formatAxis("stress", 0)
plotter.formatAxis("stress", 1)
plotter.formatAxis("stress", 2)
plotter.saveFigure("stress", "sim_results/tb_stress.png")
plotter.showAll()
