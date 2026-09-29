import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=1, cols=1)

t_switch = 2.6432e-3
t_stop = 5.3e-3

vpot = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
vdep = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)

def v_pulse_train(t, sources):
    return sources[0].getVoltage(t) if t < t_switch else sources[1].getVoltage(t)

sim.addBSource(row=1, vFunc=v_pulse_train, sources=[vpot, vdep], edge_points=[t_switch])
sim.addDcSource(col=1, voltage=0.0)

sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4)

plotter = MolmemPlotter(sim)
plotter.createFigure("dynamic", title="Single Device Dynamic I-V", subplots=(5, 1), figsize=(10, 10))
plotter.addWaveform("dynamic", 'Voltage', row=1, subplotIdx=0, color='purple', label='Applied V')
plotter.addWaveform("dynamic", 'Pulses', row=1, col=1, subplotIdx=1, color='green', label='Internal Pulse State (n)')
plotter.addWaveform("dynamic", 'State', row=1, col=1, subplotIdx=2, color='blue', label='Conductance State (f22)')
plotter.addWaveform("dynamic", 'Current', row=1, col=1, subplotIdx=3, color='red', label='Current')
plotter.addWaveform("dynamic", 'Temperature', row=1, col=1, subplotIdx=4, color='darkorange', label='Junction Temperature (K)')
plotter.formatAxis("dynamic", 0)
plotter.formatAxis("dynamic", 1)
plotter.formatAxis("dynamic", 2)
plotter.formatAxis("dynamic", 3)
plotter.formatAxis("dynamic", 4)
plotter.saveFigure("dynamic", "sim_results/tb_dynamic.png")
plotter.showAll()
