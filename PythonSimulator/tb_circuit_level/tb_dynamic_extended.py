import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=1, cols=1)

t_switch = 4.0e-3
t_stop = 8.0e-3

vpot = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
vdep = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)

def v_pulse_train(t, sources):
    return sources[0].getVoltage(t) if t < t_switch else sources[1].getVoltage(t)

sim.addBSource(row=1, vFunc=v_pulse_train, sources=[vpot, vdep], edge_points=[t_switch])
sim.addDcSource(col=1, voltage=0.0)

sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-3)

plotter = MolmemPlotter(sim)
plotter.createFigure("dynamic_ext", title="Full 50k Pulse Train", figsize=(12, 12), subplots=(5, 1))
plotter.addWaveform("dynamic_ext", 'Voltage', row=1, subplotIdx=0, color='purple')
plotter.addWaveform("dynamic_ext", 'Current', row=1, col=1, subplotIdx=1, color='red')
plotter.addWaveform("dynamic_ext", 'State', row=1, col=1, subplotIdx=2, color='green', label='Pulse Count (n)')
plotter.addWaveform("dynamic_ext", 'Conductance', row=1, col=1, subplotIdx=3, color='orange', label='Read Conductance')
plotter.addWaveform("dynamic_ext", 'Temperature', row=1, col=1, subplotIdx=4, color='darkorange', label='Junction Temperature (K)')
plotter.formatAxis("dynamic_ext", 0)
plotter.formatAxis("dynamic_ext", 1)
plotter.formatAxis("dynamic_ext", 2)
plotter.formatAxis("dynamic_ext", 3)
plotter.formatAxis("dynamic_ext", 4)
plotter.saveFigure("dynamic_ext", "sim_results/tb_dynamic_extended.png")
plotter.showAll()
