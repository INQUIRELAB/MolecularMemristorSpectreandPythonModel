import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=1, cols=1)

n_pulses = 16520
t_width = 80e-9
t_period = 160e-9
vref_pos = 0.9
vref_neg = 0.75

t_half = n_pulses * t_period
t_stop = 2.0 * t_half

vclk_pot = PulseSource(node=None, amp=1.0, period=t_period, width=80e-9)
vclk_dep = PulseSource(node=None, amp=1.0, period=t_period, width=60e-9)

def e_drive(t, sources):
    if t < t_half:
        return sources[0].getVoltage(t) * vref_pos
    else:
        return sources[1].getVoltage(t) * -vref_neg

sim.addBSource(row=1, vFunc=e_drive, sources=[vclk_pot, vclk_dep], edge_points=[t_half])
sim.addDcSource(col=1, voltage=0.0)

sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4)

plotter = MolmemPlotter(sim)
plotter.createFigure("standard_cycle", title="Standard Programming Cycle", subplots=(3, 1))
plotter.addWaveform("standard_cycle", 'Voltage', row=1, subplotIdx=0, color='purple')
plotter.addWaveform("standard_cycle", 'Current', row=1, col=1, subplotIdx=1, color='red')
plotter.addWaveform("standard_cycle", 'Conductance', row=1, col=1, subplotIdx=2, color='green')
plotter.formatAxis("standard_cycle", 0)
plotter.formatAxis("standard_cycle", 1)
plotter.formatAxis("standard_cycle", 2)
plotter.saveFigure("standard_cycle", "sim_results/tb_standard_cycle.png")
plotter.showAll()
